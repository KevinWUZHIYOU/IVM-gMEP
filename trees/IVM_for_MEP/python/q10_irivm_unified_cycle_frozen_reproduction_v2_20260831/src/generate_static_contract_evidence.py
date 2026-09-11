#!/usr/bin/env python3
"""Generate the pre-run static reachability/interception evidence.

This script never imports or executes the optimizer.  It analyzes the frozen
optimizer, its broad historical core, and the registered observation adapter
as source text.  The broad core is intentionally not claimed to be free of
legacy definitions; instead, the report proves that those definitions are not
reachable from the frozen ``run_clean`` entry point.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


THIS_FILE = Path(__file__).resolve()
CAMPAIGN_ROOT = THIS_FILE.parents[1]
REPOSITORY_ROOT = THIS_FILE.parents[3]
OPTIMIZER = CAMPAIGN_ROOT / "src/q10_unified_cycle_optimizer.py"
ADAPTER = CAMPAIGN_ROOT / "src/registered_adapter.py"
BROAD_CORE = REPOSITORY_ROOT / (
    "python/q10_irivm_low_r_coherence_secant_v2_20260830/src/"
    "q10_irivm_atomic_tangent_secant_core.py"
)
BACKEND = REPOSITORY_ROOT / (
    "python/q10_irivm_low_r_coherence_secant_v2_20260830/src/"
    "q10_solver_backend.py"
)
OUTPUT = CAMPAIGN_ROOT / "evidence/static_contract_report.json"

SPRING_NAME = "hj_nearest_neighbor_spring_reparameterize"
EXPECTED_SPRING_CALLERS = {
    "recovery_proposal_spring": "recovery_proposal_spring",
    "synchronized_spring_recovery": "synchronized_spring_recovery",
    "build_severe_rejection_hj_spring_retry": (
        "severe_rejection_hj_spring_retry"
    ),
    "prepare_normal_stream_exit_hj_spring": (
        "normal_stream_exit_hj_spring"
    ),
    "run_clean": "post_main_direct_hj_spring",
}
DORMANT_LEGACY = (
    "improved_spring_reparameterize",
    "polyline_equal_arclength",
    "chord_coherent_recovery_seed",
    "tangent_chord_constraint_project",
    "bicriteria_merge",
    "single_synchronized_recovery",
    "run",
    "run_one",
)
FORBIDDEN_REACHABLE = (
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


class StaticEvidenceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dotted_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        if prefix is not None:
            return prefix + "." + node.attr
    return None


def top_level_definitions(tree: ast.Module) -> Dict[str, ast.AST]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def calls_in(node: ast.AST) -> List[Tuple[str, int]]:
    result = []
    for item in ast.walk(node):
        if isinstance(item, ast.Call):
            name = dotted_name(item.func)
            if name is not None:
                result.append((name, int(getattr(item, "lineno", -1))))
    return result


def reachable(
    definitions: Mapping[str, ast.AST], seeds: Iterable[str]
) -> Set[str]:
    pending = list(seeds)
    found = set()
    while pending:
        name = pending.pop()
        if name in found or name not in definitions:
            continue
        found.add(name)
        for target, _line in calls_in(definitions[name]):
            leaf = target.rsplit(".", 1)[-1]
            if "." not in target and leaf in definitions and leaf not in found:
                pending.append(leaf)
    return found


def parent_map(tree: ast.AST) -> Dict[ast.AST, ast.AST]:
    result = {}
    for owner in ast.walk(tree):
        for child in ast.iter_child_nodes(owner):
            result[child] = owner
    return result


def enclosing_top_level_function(
    node: ast.AST, parents: Mapping[ast.AST, ast.AST]
) -> Optional[str]:
    current = node
    candidate = None
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            candidate = current.name
        if isinstance(current, ast.Module):
            break
    return candidate


def literal_mapping(tree: ast.Module, name: str) -> Dict[str, str]:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            continue
        if not isinstance(node.value, ast.Dict):
            raise StaticEvidenceError(name + " is not a literal dictionary")
        result = {}
        for key, value in zip(node.value.keys, node.value.values):
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                raise StaticEvidenceError(name + " has a nonliteral entry")
            result[str(key.value)] = str(value.value)
        return result
    raise StaticEvidenceError("missing literal dictionary " + name)


def source_segment(source: str, node: ast.AST) -> str:
    lines = source.splitlines()
    start = int(getattr(node, "lineno", 1)) - 1
    end = int(getattr(node, "end_lineno", start + 1))
    return "\n".join(lines[start:end])


def canonical_write(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(str(temporary), str(path))


def build_report() -> Dict[str, Any]:
    optimizer_source = OPTIMIZER.read_text(encoding="utf-8")
    adapter_source = ADAPTER.read_text(encoding="utf-8")
    core_source = BROAD_CORE.read_text(encoding="utf-8")
    optimizer_tree = ast.parse(optimizer_source, filename=str(OPTIMIZER))
    adapter_tree = ast.parse(adapter_source, filename=str(ADAPTER))
    core_tree = ast.parse(core_source, filename=str(BROAD_CORE))
    optimizer_defs = top_level_definitions(optimizer_tree)
    core_defs = top_level_definitions(core_tree)

    optimizer_reachable = reachable(optimizer_defs, ("run_clean",))
    core_seeds = set()
    optimizer_calls = []
    for name in sorted(optimizer_reachable):
        for target, line in calls_in(optimizer_defs[name]):
            optimizer_calls.append((name, target, line))
            if target.startswith("alg."):
                leaf = target.split(".", 1)[1]
                if leaf in core_defs:
                    core_seeds.add(leaf)
    core_reachable = reachable(core_defs, core_seeds)
    core_calls = []
    for name in sorted(core_reachable):
        for target, line in calls_in(core_defs[name]):
            core_calls.append((name, target, line))

    parents = parent_map(optimizer_tree)
    spring_calls = []
    for item in ast.walk(optimizer_tree):
        if not isinstance(item, ast.Call) or dotted_name(item.func) != SPRING_NAME:
            continue
        spring_calls.append(
            {
                "enclosing_function": enclosing_top_level_function(item, parents),
                "line": int(item.lineno),
                "target": SPRING_NAME,
            }
        )
    spring_callers = {
        str(item["enclosing_function"]): EXPECTED_SPRING_CALLERS.get(
            str(item["enclosing_function"])
        )
        for item in spring_calls
    }

    spring_definition = optimizer_defs.get(SPRING_NAME)
    if not isinstance(spring_definition, ast.FunctionDef):
        raise StaticEvidenceError("missing sole HJ spring definition")
    spring_targets = [target for target, _line in calls_in(spring_definition)]
    tangent_in_spring = [
        target for target in spring_targets if target == "core.improved_tangents"
    ]
    spring_text = source_segment(optimizer_source, spring_definition)
    spring_formula_checks = {
        "fixed_entry_energies": (
            "fixed_energies = np.asarray(energies, dtype=np.float64)"
            in spring_text
        ),
        "adjacent_length_difference": (
            "length_differences = lengths[1:] - lengths[:-1]" in spring_text
        ),
        "force_is_length_difference_times_tangent": (
            "spring_force = length_differences[:, None] * tangents" in spring_text
        ),
        "update_uses_frozen_dtau": (
            "result[1:-1] += alg.SPRING_DTAU * spring_force" in spring_text
        ),
        "endpoint_restore": (
            "result[0] = path[0]" in spring_text
            and "result[-1] = path[-1]" in spring_text
        ),
        "one_hj_tangent_call_site": len(tangent_in_spring) == 1,
    }

    reachable_calls = optimizer_calls + core_calls
    forbidden_counts = {}
    for fragment in FORBIDDEN_REACHABLE:
        forbidden_counts[fragment] = sum(
            1 for _owner, target, _line in reachable_calls
            if fragment in target.lower()
        )

    tangent_calls = [
        {
            "source": "optimizer" if item in optimizer_calls else "broad_core",
            "enclosing_function": owner,
            "target": target,
            "line": line,
        }
        for item in reachable_calls
        for owner, target, line in (item,)
        if target == "core.improved_tangents"
    ]
    direct_global_tangents = [
        {"enclosing_function": owner, "target": target, "line": line}
        for owner, target, line in reachable_calls
        if target == "improved_tangents"
    ]

    dormant = []
    for name in DORMANT_LEGACY:
        node = core_defs.get(name)
        dormant.append(
            {
                "name": name,
                "defined": node is not None,
                "defined_line": (
                    int(getattr(node, "lineno", -1)) if node is not None else None
                ),
                "reachable_from_run_clean": name in core_reachable,
            }
        )

    adapter_mapping = literal_mapping(adapter_tree, "SPRING_CALLERS")
    adapter_defs = top_level_definitions(adapter_tree)
    observed_text = source_segment(
        adapter_source, adapter_defs["observed_optimizer"]
    )
    registered_text = source_segment(
        adapter_source, adapter_defs["run_registered_member"]
    )
    proxy_class = adapter_defs.get("AuditedCoreProxy")
    proxy_text = source_segment(adapter_source, proxy_class) if proxy_class else ""
    adapter_checks = {
        "spring_caller_mapping_exact": adapter_mapping == EXPECTED_SPRING_CALLERS,
        "patches_backend_loader": (
            "algorithm_runtime._load_backend = load_backend_observed"
            in observed_text
        ),
        "backend_loader_returns_core_proxy": (
            "return AuditedCoreProxy(raw_core, observer), potential"
            in observed_text
        ),
        "patches_sole_spring_global": (
            "optimizer.hj_nearest_neighbor_spring_reparameterize = spring_observed"
            in observed_text
        ),
        "spring_wrapper_rejects_unknown_caller": (
            "if raw_caller not in SPRING_CALLERS" in observed_text
        ),
        "spring_wrapper_returns_original_result": "return result" in observed_text,
        "patches_cached_evaluator_query": (
            "algorithm_runtime.Evaluator.query = query_observed" in observed_text
        ),
        "restores_all_hooks": all(
            text in observed_text
            for text in (
                "optimizer.hj_nearest_neighbor_spring_reparameterize = original_spring",
                "algorithm_runtime.Evaluator.query = original_query",
                "algorithm_runtime._load_backend = original_load_backend",
            )
        ),
        "registered_entry_executes_under_observer": (
            "with observed_optimizer(optimizer, observer):" in registered_text
            and "result = optimizer.run_clean(" in registered_text
        ),
        "core_proxy_records_hj_and_returns_same_result": (
            "result = self._raw_improved_tangents" in proxy_text
            and "self._observer.record_tangent" in proxy_text
            and "return result" in proxy_text
        ),
    }

    spring_enclosers = [str(item["enclosing_function"]) for item in spring_calls]
    unique_reparam = (
        len([name for name in optimizer_defs if "reparam" in name.lower()]) == 1
        and len(spring_calls) == len(EXPECTED_SPRING_CALLERS)
        and set(spring_enclosers) == set(EXPECTED_SPRING_CALLERS)
        and all(item["enclosing_function"] in optimizer_reachable for item in spring_calls)
    )
    all_runtime_calls_intercepted = bool(
        unique_reparam
        and all(adapter_checks.values())
        and tangent_calls
        and not direct_global_tangents
        and not any(forbidden_counts.values())
        and not any(item["reachable_from_run_clean"] for item in dormant)
    )
    passed = bool(
        all_runtime_calls_intercepted
        and all(spring_formula_checks.values())
        and len(tangent_in_spring) == 1
    )

    return {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "scope": (
            "preformal static source reachability and registered-adapter "
            "interception; no optimizer execution"
        ),
        "generated_before_registered_run": True,
        "sources": {
            "optimizer": {
                "path": "src/q10_unified_cycle_optimizer.py",
                "sha256": sha256_file(OPTIMIZER),
            },
            "adapter": {
                "path": "src/registered_adapter.py",
                "sha256": sha256_file(ADAPTER),
            },
            "broad_core": {
                "path": (
                    "repo:python/q10_irivm_low_r_coherence_secant_v2_20260830/"
                    "src/q10_irivm_atomic_tangent_secant_core.py"
                ),
                "sha256": sha256_file(BROAD_CORE),
            },
            "backend": {
                "path": (
                    "repo:python/q10_irivm_low_r_coherence_secant_v2_20260830/"
                    "src/q10_solver_backend.py"
                ),
                "sha256": sha256_file(BACKEND),
            },
            "generator": {
                "path": "src/generate_static_contract_evidence.py",
                "sha256": sha256_file(THIS_FILE),
            },
        },
        "optimizer_entrypoint": "q10_unified_cycle_optimizer.run_clean",
        "registered_entrypoint": "registered_adapter.run_registered_member",
        "optimizer_reachable": sorted(optimizer_reachable),
        "broad_core_seeds": sorted(core_seeds),
        "broad_core_reachable": sorted(core_reachable),
        "reparameterization_target": SPRING_NAME,
        "reparam_definition_count": int(SPRING_NAME in optimizer_defs),
        "reparam_calls": spring_calls,
        "spring_callers_expected": EXPECTED_SPRING_CALLERS,
        "spring_callers_observed": spring_callers,
        "spring_formula_checks": spring_formula_checks,
        "tangent_target": "core.improved_tangents (Henkelman-Jonsson improved)",
        "tangent_calls": tangent_calls,
        "direct_global_tangent_calls": direct_global_tangents,
        "adapter_interception_checks": adapter_checks,
        "all_runtime_calls_intercepted": all_runtime_calls_intercepted,
        "dormant_legacy_paths": dormant,
        "forbidden_reachable_counts": forbidden_counts,
        "honest_boundary": (
            "The broad historical core retains dormant legacy definitions. "
            "This report claims only that they are unreachable from run_clean. "
            "At runtime _load_backend is replaced by a wrapper returning "
            "AuditedCoreProxy, so every reachable core.improved_tangents call "
            "is observed; the sole reparameterization global is separately "
            "wrapped and restricted to the five statically enumerated callers."
        ),
    }


def main() -> int:
    report = build_report()
    canonical_write(OUTPUT, report)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

