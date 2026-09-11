#!/usr/bin/env python3
"""Freeze the R2 analysis manifest after ODE and surface audits are complete.

This is a binding-only, zero-physical operation.  It refuses mutable inputs,
overwrites, partial ODE caches, and surface adapters that do not attest the
call515/NF14983 uncommitted diagnostic semantics.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "bb_discrete32_analysis_r2_finalizer", HERE / "bb_discrete32_analysis_r2.py"
)
R2 = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError("cannot load R2 analysis module")
SPEC.loader.exec_module(R2)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def regular(path: Path, role: str) -> Path:
    """Resolve a regular nonsymlink file; its bytes are bound in the output.

    Production scientific inputs use :func:`readonly` below.  Keeping this
    helper permission-neutral also lets synthetic tests construct temporary
    manifest templates without weakening any production-input check.
    """

    resolved = path.expanduser().resolve(strict=True)
    status = resolved.lstat()
    require(
        stat.S_ISREG(status.st_mode) and not resolved.is_symlink(),
        f"{role} is not a regular nonsymlink file",
    )
    return resolved


def readonly(path: Path, role: str) -> Path:
    resolved = regular(path, role)
    status = resolved.lstat()
    require(
        stat.S_IMODE(status.st_mode) & 0o222 == 0,
        f"{role} is not a read-only regular nonsymlink file",
    )
    return resolved


def binding(path: Path, role: str) -> dict[str, str]:
    resolved = readonly(path, role)
    return {"path": str(resolved), "sha256": R2.BASE_MODULE.sha256_file(resolved)}


def load_json(path: Path, role: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{role} is not a JSON object")
    return value


def write_new(path: Path, value: dict[str, Any]) -> None:
    target = path.expanduser().resolve()
    require(not target.exists(), f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    os.chmod(target, 0o444)
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def validate_surface_adapter(
        adapter: dict[str, Any], expected_center: dict[str, Any],
        surface_path: Path, surface_audit_path: Path) -> None:
    """Validate the adapter's sized bindings without weakening manifest bindings."""

    adapter_surface_result = R2._resolve_sized_binding(
        adapter.get("surface_result"), "adapter surface result"
    )
    adapter_surface_audit = R2._resolve_sized_binding(
        adapter.get("surface_audit"), "adapter surface audit"
    )
    require(
        adapter.get("schema") == "bb-irivm-call515-surface-center-adapter-audit-v1"
        and adapter.get("status") == "PASS_ZERO_PHYSICAL_SURFACE_CENTER_BINDING"
        and adapter.get("physical_force_calls") == 0
        and adapter.get("openmm_contexts_constructed") == 0
        and adapter.get("center_state") == expected_center
        and adapter_surface_result == surface_path
        and adapter_surface_audit == surface_audit_path,
        "surface adapter does not bind the required call515 diagnostic center",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--terminal-audit", type=Path, required=True)
    parser.add_argument("--incremental-cache", type=Path, required=True)
    parser.add_argument("--surface-result", type=Path, required=True)
    parser.add_argument("--surface-audit", type=Path, required=True)
    parser.add_argument("--surface-center-adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    template_path = regular(args.template, "R2 analysis template")
    document = load_json(template_path, "R2 analysis template")
    require(
        document.get("schema") == R2.MANIFEST_SCHEMA
        and document.get("status") == "WAITING_FOR_ODE_NF30000_AND_SURFACE",
        "R2 template identity/status differs",
    )
    run_dir = args.run_dir.expanduser().resolve(strict=True)
    require(run_dir.is_dir() and not run_dir.is_symlink(), "ODE run directory is unsafe")
    terminal_path = R2.resolve_replica_terminal_audit_file(args.terminal_audit)
    terminal_binding = {
        "path": str(terminal_path),
        "sha256": R2.BASE_MODULE.sha256_file(terminal_path),
    }
    cache_binding = binding(args.incremental_cache, "ODE final incremental cache")
    surface_binding = binding(args.surface_result, "call515-centered surface result")
    surface_audit_binding = binding(args.surface_audit, "call515-centered surface audit")
    adapter_binding = binding(
        args.surface_center_adapter, "call515 surface-center adapter audit"
    )

    terminal = load_json(Path(terminal_binding["path"]), "ODE terminal audit")
    candidate = terminal.get("candidate", {})
    require(
        terminal.get("status") == "PASS"
        and terminal.get("terminal_eligible") is True
        and terminal.get("formal_figure_eligible") is True
        and candidate.get("phase") == "formal"
        and float(candidate.get("spring_k_eV_per_A2", -1)) == 0.1
        and int(terminal.get("paid_batches", -1)) == R2.FORMAL_BATCHES
        and int(terminal.get("paid_optimization_N_F", -1)) == R2.FORMAL_NF,
        "ODE terminal audit is not the selected NF30000 PASS",
    )
    cache = load_json(Path(cache_binding["path"]), "ODE final incremental cache")
    require(
        cache.get("status") == "FINAL_BOUND_TO_CANONICAL_TERMINAL_AUDIT"
        and cache.get("partial_is_final") is False
        and int(cache.get("target_batches", -1)) == R2.FORMAL_BATCHES
        and int(cache.get("target_optimization_N_F", -1)) == R2.FORMAL_NF
        and int(cache.get("processed_batches", -1)) == R2.FORMAL_BATCHES
        and int(cache.get("processed_optimization_N_F", -1)) == R2.FORMAL_NF
        and cache.get("canonical_terminal_audit") == terminal_binding
        and cache.get("physical_calls_started") == 0
        and cache.get("openmm_contexts_constructed") == 0
        and cache.get("bb_locks_acquired") == 0
        and cache.get("local_mutagen_replica_transport")
        == R2.expected_cache_replica_transport(),
        "ODE cache is not the final zero-physical NF30000 cache",
    )
    adapter = load_json(Path(adapter_binding["path"]), "surface-center adapter audit")
    validate_surface_adapter(
        adapter,
        document["inputs"]["surface_center_state"],
        Path(surface_binding["path"]),
        Path(surface_audit_binding["path"]),
    )

    document["inputs"]["formal_ode_neb"] = {
        "run_dir": str(run_dir),
        "terminal_audit": terminal_binding,
        "incremental_cache": cache_binding,
    }
    document["inputs"]["ir_centered_surface_bundle"] = surface_binding
    document["inputs"]["ir_centered_surface_audit"] = surface_audit_binding
    document["inputs"]["surface_center_state_audit"] = adapter_binding
    document["status"] = "FINAL_INPUTS_FROZEN"
    document["finalization"] = {
        "mode": "O_EXCL_readonly_zero_physical",
        "physical_force_calls": 0,
        "openmm_contexts_constructed": 0,
        "template": {
            "path": str(template_path),
            "sha256": R2.BASE_MODULE.sha256_file(template_path),
        },
    }
    require(not R2.readiness_r2(document), "final R2 manifest is still incomplete")
    require("openmm" not in sys.modules and "simtk.openmm" not in sys.modules,
            "R2 manifest finalizer imported OpenMM")
    write_new(args.output, document)
    print(json.dumps({
        "status": "FINAL_INPUTS_FROZEN",
        "output": str(args.output.expanduser().resolve()),
        "sha256": R2.BASE_MODULE.sha256_file(args.output.expanduser().resolve()),
        "physical_force_calls": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
