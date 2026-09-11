"""Static AST checks for the formal optimizer/runtime trust boundary.

The checker never imports candidate code. It follows direct same-module calls
and imports from the declared runtime module, then applies the contract only to
reachable code. It is intentionally conservative: unresolved dynamic dispatch
is a violation, not a reason to assume compliance.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping


FORBIDDEN_GEOMETRY_FRAGMENTS = (
    "central_difference",
    "central_tangent",
    "averaged_chord",
    "equal_arclength",
    "polyline",
    "cubic",
    "saddle",
    "chord_coherent",
    "tangent_chord",
)

FORBIDDEN_OPTIMIZER_DATA_FRAGMENTS = (
    "article",
    "opponent",
    "checkpoint",
    "reference_barrier",
    "linear500",
    "deltae_ref",
    "offline_metric",
)

DIRECT_POTENTIAL_CALLS = {
    "energy_gradient",
    "energy_gradient_batch",
    "muller_energy",
    "muller_energy_gradient",
    "potential",
}


@dataclass(frozen=True)
class ContractConfiguration:
    optimizer_entrypoint: str
    runtime_module: str
    reparameterization_entrypoint: str
    tangent_entrypoint: str


@dataclass(frozen=True)
class ContractReport:
    optimizer_reachable: tuple[str, ...]
    runtime_reachable: tuple[str, ...]
    reparameterization_calls: tuple[str, ...]
    tangent_calls: tuple[str, ...]
    violations: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_jsonable(self) -> dict[str, object]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


class StaticContractError(RuntimeError):
    pass


def _literal_string_assignment(tree: ast.Module, name: str) -> str:
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
                raise StaticContractError(f"{name} must be a literal string")
    raise StaticContractError(f"missing literal contract declaration {name}")


def configuration_from_optimizer(tree: ast.Module) -> ContractConfiguration:
    return ContractConfiguration(
        optimizer_entrypoint=_literal_string_assignment(tree, "FORMAL_OPTIMIZER_ENTRYPOINT"),
        runtime_module=_literal_string_assignment(tree, "FORMAL_RUNTIME_MODULE"),
        reparameterization_entrypoint=_literal_string_assignment(
            tree, "FORMAL_REPARAMETERIZATION_ENTRYPOINT"
        ),
        tangent_entrypoint=_literal_string_assignment(tree, "FORMAL_TANGENT_ENTRYPOINT"),
    )


def _definitions(tree: ast.Module) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _runtime_imports(tree: ast.Module, runtime_module: str) -> tuple[set[str], dict[str, str]]:
    module_aliases: set[str] = set()
    symbol_aliases: dict[str, str] = {}
    runtime_tail = runtime_module.rsplit(".", 1)[-1]
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.rsplit(".", 1)[-1] == runtime_tail:
                    module_aliases.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").rsplit(".", 1)[-1]
            if module == runtime_tail:
                for alias in node.names:
                    if alias.name == "*":
                        raise StaticContractError("wildcard runtime import is forbidden")
                    symbol_aliases[alias.asname or alias.name] = alias.name
            elif not node.module:
                # ``from . import irivm_runtime`` is represented with a null
                # module and a named alias.
                for alias in node.names:
                    if alias.name.rsplit(".", 1)[-1] == runtime_tail:
                        module_aliases.add(alias.asname or alias.name)
    return module_aliases, symbol_aliases


def _call_target(call: ast.Call) -> str | None:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name):
        return f"{function.value.id}.{function.attr}"
    return None


def _direct_calls(node: ast.AST) -> list[str]:
    return [target for item in ast.walk(node) if isinstance(item, ast.Call) if (target := _call_target(item))]


def _reachable_local(
    definitions: Mapping[str, ast.AST], seeds: Iterable[str]
) -> tuple[set[str], list[str]]:
    pending = list(seeds)
    reached: set[str] = set()
    unresolved_dynamic: list[str] = []
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        node = definitions.get(name)
        if node is None:
            continue
        reached.add(name)
        for item in ast.walk(node):
            if not isinstance(item, ast.Call):
                continue
            target = _call_target(item)
            if target is None:
                unresolved_dynamic.append(f"{name}:{getattr(item, 'lineno', '?')}")
            elif "." not in target and target in definitions:
                pending.append(target)
    return reached, unresolved_dynamic


def _identifiers_and_strings(nodes: Iterable[ast.AST]) -> list[tuple[str, int]]:
    values: list[tuple[str, int]] = []
    for root in nodes:
        docstring_nodes: set[int] = set()
        for owner in ast.walk(root):
            body = getattr(owner, "body", None)
            if (
                isinstance(body, list)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))
        for node in ast.walk(root):
            if isinstance(node, ast.Name):
                values.append((node.id.lower(), getattr(node, "lineno", -1)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                values.append((node.name.lower(), getattr(node, "lineno", -1)))
            elif isinstance(node, ast.arg):
                values.append((node.arg.lower(), getattr(node, "lineno", -1)))
            elif isinstance(node, ast.alias):
                values.append((node.name.lower(), getattr(node, "lineno", -1)))
                if node.asname:
                    values.append((node.asname.lower(), getattr(node, "lineno", -1)))
            elif isinstance(node, ast.Attribute):
                values.append((node.attr.lower(), getattr(node, "lineno", -1)))
            elif (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstring_nodes
            ):
                values.append((node.value.lower(), getattr(node, "lineno", -1)))
    return values


def _comparison_names_and_numbers(node: ast.AST) -> tuple[set[str], set[int]]:
    names = {
        item.id.lower()
        for item in ast.walk(node)
        if isinstance(item, ast.Name)
    }
    names.update(
        item.attr.lower()
        for item in ast.walk(node)
        if isinstance(item, ast.Attribute)
    )
    numbers = {
        int(item.value)
        for item in ast.walk(node)
        if isinstance(item, ast.Constant)
        and isinstance(item.value, int)
        and not isinstance(item.value, bool)
    }
    return names, numbers


def analyze_source_texts(
    optimizer_source: str,
    runtime_source: str,
    *,
    configuration: ContractConfiguration | None = None,
) -> ContractReport:
    optimizer_tree = ast.parse(optimizer_source, filename="<optimizer>")
    runtime_tree = ast.parse(runtime_source, filename="<runtime>")
    config = configuration or configuration_from_optimizer(optimizer_tree)
    optimizer_defs = _definitions(optimizer_tree)
    runtime_defs = _definitions(runtime_tree)
    violations: list[str] = []
    warnings: list[str] = []

    if config.optimizer_entrypoint not in optimizer_defs:
        violations.append(f"missing optimizer entrypoint {config.optimizer_entrypoint}")
    if config.reparameterization_entrypoint not in runtime_defs:
        violations.append(
            f"missing sole spring reparameterization entrypoint "
            f"{config.reparameterization_entrypoint}"
        )
    if config.tangent_entrypoint not in runtime_defs:
        violations.append(f"missing HJ tangent entrypoint {config.tangent_entrypoint}")

    optimizer_reachable, dynamic_optimizer = _reachable_local(
        optimizer_defs, [config.optimizer_entrypoint]
    )
    if dynamic_optimizer:
        violations.extend(
            f"unresolved dynamic optimizer call at {location}"
            for location in dynamic_optimizer
        )

    module_aliases, symbol_aliases = _runtime_imports(optimizer_tree, config.runtime_module)
    runtime_seeds = {
        config.reparameterization_entrypoint,
        config.tangent_entrypoint,
    }
    optimizer_calls: list[str] = []
    for name in sorted(optimizer_reachable):
        optimizer_calls.extend(_direct_calls(optimizer_defs[name]))
    for target in optimizer_calls:
        if "." in target:
            prefix, attribute = target.split(".", 1)
            if prefix in module_aliases:
                runtime_seeds.add(attribute)
        elif target in symbol_aliases:
            runtime_seeds.add(symbol_aliases[target])

    runtime_reachable, dynamic_runtime = _reachable_local(runtime_defs, runtime_seeds)
    if dynamic_runtime:
        violations.extend(
            f"unresolved dynamic runtime call at {location}"
            for location in dynamic_runtime
        )

    optimizer_nodes = [optimizer_defs[name] for name in sorted(optimizer_reachable)]
    runtime_nodes = [runtime_defs[name] for name in sorted(runtime_reachable)]
    # The formal optimizer/slim runtime must not retain dormant forbidden
    # surfaces, so these token checks are global rather than reachability-only.
    for value, line in _identifiers_and_strings([optimizer_tree, runtime_tree]):
        for forbidden in FORBIDDEN_GEOMETRY_FRAGMENTS:
            if forbidden in value:
                violations.append(f"reachable forbidden geometry {forbidden!r} at line {line}")
        if any(fragment in value for fragment in FORBIDDEN_OPTIMIZER_DATA_FRAGMENTS):
            # Runtime docstrings may truthfully say that Article data are absent;
            # only executable identifiers and non-docstring literals should fail.
            violations.append(f"reachable optimizer/offline data token {value!r} at line {line}")

    for name in sorted(optimizer_reachable):
        node = optimizer_defs[name]
        for item in ast.walk(node):
            if isinstance(item, (ast.If, ast.IfExp)):
                test = item.test
                names, numbers = _comparison_names_and_numbers(test)
                if "n_internal" in names and numbers.intersection({10, 20}):
                    violations.append(
                        f"resolution-specific branch in optimizer {name} at line {item.lineno}"
                    )
                if any("event" in identifier or "log" in identifier for identifier in names):
                    warnings.append(
                        f"review possible decision read from audit logging in {name} at line {item.lineno}"
                    )
            elif isinstance(item, ast.Subscript):
                names, numbers = _comparison_names_and_numbers(item)
                if "n_internal" in names and numbers.issuperset({10, 20}):
                    violations.append(
                        f"resolution-specific table dispatch in optimizer {name} at line {item.lineno}"
                    )
            elif hasattr(ast, "Match") and isinstance(item, ast.Match):
                names, numbers = _comparison_names_and_numbers(item)
                if "n_internal" in names and numbers.intersection({10, 20}):
                    violations.append(
                        f"resolution-specific match dispatch in optimizer {name} at line {item.lineno}"
                    )

    runtime_tail = config.runtime_module.rsplit(".", 1)[-1]
    normalized_reparam = {
        config.reparameterization_entrypoint,
        f"{runtime_tail}.{config.reparameterization_entrypoint}",
    }
    normalized_tangent = {
        config.tangent_entrypoint,
        f"{runtime_tail}.{config.tangent_entrypoint}",
    }
    reparam_calls: list[str] = []
    tangent_calls: list[str] = []
    all_calls = optimizer_calls[:]
    for name in sorted(runtime_reachable):
        all_calls.extend(_direct_calls(runtime_defs[name]))
    for target in all_calls:
        tail = target.rsplit(".", 1)[-1]
        if "reparam" in tail.lower():
            reparam_calls.append(target)
            if tail != config.reparameterization_entrypoint:
                violations.append(f"alternate reachable reparameterization call: {target}")
        if "tangent" in tail.lower():
            tangent_calls.append(target)
            if any(fragment in tail.lower() for fragment in FORBIDDEN_GEOMETRY_FRAGMENTS):
                violations.append(f"forbidden reachable tangent call: {target}")

    if not any(target.rsplit(".", 1)[-1] == config.reparameterization_entrypoint for target in reparam_calls):
        violations.append("reachable optimizer never calls the registered spring reparameterization")
    if not any(target.rsplit(".", 1)[-1] == config.tangent_entrypoint for target in tangent_calls):
        violations.append("reachable spring/runtime never calls the registered HJ tangent")

    for target in optimizer_calls:
        if target.rsplit(".", 1)[-1].lower() in DIRECT_POTENTIAL_CALLS:
            violations.append(f"optimizer directly calls potential operation {target}")

    # Deduplicate while preserving diagnostic order.
    violations = list(dict.fromkeys(violations))
    warnings = list(dict.fromkeys(warnings))
    return ContractReport(
        tuple(sorted(optimizer_reachable)),
        tuple(sorted(runtime_reachable)),
        tuple(reparam_calls),
        tuple(tangent_calls),
        tuple(violations),
        tuple(warnings),
    )


def analyze_files(optimizer_path: Path, runtime_path: Path) -> ContractReport:
    return analyze_source_texts(
        optimizer_path.read_text(encoding="utf-8"),
        runtime_path.read_text(encoding="utf-8"),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("optimizer", type=Path)
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    report = analyze_files(arguments.optimizer, arguments.runtime)
    if arguments.json:
        print(json.dumps(report.to_jsonable(), sort_keys=True, indent=2))
    else:
        print("PASS" if report.passed else "FAIL")
        for violation in report.violations:
            print(f"violation: {violation}")
        for warning in report.warnings:
            print(f"warning: {warning}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
