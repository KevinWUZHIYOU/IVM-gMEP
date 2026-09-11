#!/usr/bin/env python3
"""Render an explicit NOT FINAL R2 preview from the sealed batch-500 cache."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "bb_discrete32_analysis_r2_preview", HERE / "bb_discrete32_analysis_r2.py"
)
R2 = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError("cannot load R2 analysis module")
SPEC.loader.exec_module(R2)

CACHE_SPEC = importlib.util.spec_from_file_location(
    "bb_nf30000_local_mutagen_cache_adapter_preview",
    R2.CACHE_REPLICA_WRAPPER,
)
CACHE_ADAPTER = importlib.util.module_from_spec(CACHE_SPEC)
if CACHE_SPEC.loader is None:
    raise RuntimeError("cannot load local Mutagen cache adapter")
CACHE_SPEC.loader.exec_module(CACHE_ADAPTER)

CACHE = (
    R2.CACHE_REPLICA_RUN.parent.parent / "analysis" /
    "formal_discrete32_cache_r2_audited_v2_local_mutagen" /
    "formal_discrete32_cache_through_batch_0501.json"
)
ADAPTER = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
    "runs/diagnostics/bb_irivm_call515_nf14983_terminal_uncommitted_tail_"
    "0k_section_surface_center_adapter_20260903_r2b.audit.json"
)
OUTPUT_DIR = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf")
PREFIX = "barnase_barstar_discrete32_ode_in_progress_nf15000_r2_v2_20260903"
OUTPUTS = {
    "cost": OUTPUT_DIR / f"{PREFIX}_cost_to_precision_NOT_FINAL.pdf",
    "energy": OUTPUT_DIR / f"{PREFIX}_energy_vs_arc_length_NOT_FINAL.pdf",
    "evolution": OUTPUT_DIR / f"{PREFIX}_evolution_NOT_FINAL.pdf",
    "receipt": OUTPUT_DIR / f"{PREFIX}_preview_receipt_NOT_FINAL.json",
}
PREVIEW_BATCHES = 500
PREVIEW_NF = 15000
EVOLUTION_BUDGETS = (2040, 6000, 15000)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readonly(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    status = resolved.lstat()
    require(stat.S_ISREG(status.st_mode) and not resolved.is_symlink()
            and stat.S_IMODE(status.st_mode) & 0o222 == 0,
            f"{role} is not a read-only regular nonsymlink file")
    return resolved


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def load_partial_ode(cache_path: Path, requested_budgets: Sequence[int]):
    cache_path = readonly(cache_path, "batch-500 adapted cache")
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    require(
        cache.get("schema") == "bb-ode-formal-discrete32-incremental-cache-v1"
        and cache.get("status") == "PARTIAL_NOT_FINAL"
        and cache.get("partial_is_final") is False
        and cache.get("run_dir") == str(R2.CACHE_REPLICA_RUN)
        and cache.get("target_batches") == 1000
        and cache.get("target_optimization_N_F") == 30000
        and cache.get("processed_batches") == 501
        and cache.get("processed_optimization_N_F") == 15030
        and cache.get("canonical_terminal_audit") is None
        and cache.get("physical_calls_started") == 0
        and cache.get("openmm_contexts_constructed") == 0
        and cache.get("bb_locks_acquired") == 0
        and cache.get("local_mutagen_replica_transport")
        == R2.expected_cache_replica_transport(),
        "batch-500 cache is not the exact adapted partial source",
    )
    rows = cache.get("rows")
    require(isinstance(rows, list) and len(rows) == 501,
            "sealed source cache row count differs")
    source_rows = []
    output = []
    for batch_id, row in enumerate(rows, 1):
        unsigned = dict(row); row_hash = unsigned.pop("row_sha256", "")
        expected_hash = hashlib.sha256(json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode()).hexdigest()
        require(row_hash == expected_hash and row.get("batch_id") == batch_id
                and row.get("optimization_N_F") == 30 * batch_id
                and math.isfinite(float(row.get("i_perp", math.nan)))
                and math.isfinite(float(row.get("delta_e32_eV", math.nan))),
                f"batch-500 row {batch_id} differs")
        source_rows.append({key: row[key] for key in (
            "batch_id", "optimization_N_F", "source_artifact_sha256",
            "full_state_wal_event_sha256", "ledger_commit_event_sha256", "row_sha256",
        )})
        if batch_id <= PREVIEW_BATCHES:
            output.append({
                "method": "ODE-NEB", "optimization_N_F": int(row["optimization_N_F"]),
                "i_perp": float(row["i_perp"]),
                "delta_e32_eV": float(row["delta_e32_eV"]),
                "delta_e32_kcal_per_mol": float(row["delta_e32_kcal_per_mol"]),
                "path_sha256": row["current_path_sha256"],
                "energy_sha256": row["current_energy_sha256"],
                "force_sha256": row["current_force_sha256"],
                "state_role": "current accepted path after paid batch; ODE still in progress",
                "source": {"incremental_cache_sha256": sha256_file(cache_path),
                           "batch_id": batch_id, "formal_figure_eligible": False},
            })
    closure = hashlib.sha256(json.dumps(
        source_rows, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode()).hexdigest()
    require(closure == cache.get("rows_source_closure_sha256"),
            "batch-500 rows-source closure differs")
    states = cache.get("selected_state_bindings")
    require(isinstance(states, Mapping) and set(states) == {"240", "2040", "15000"},
            "batch-500 retained state set differs")
    for text, record in states.items():
        nf = int(text)
        path = R2.BASE_MODULE.resolve_binding(record, f"preview ODE NF{nf} state")
        with np.load(path, allow_pickle=False) as saved:
            path32 = np.asarray(saved["path32_A"], dtype=np.float64)
            energy32 = np.asarray(saved["energy32_eV"], dtype=np.float64)
            force32 = np.asarray(saved["full_force32_eV_per_A"], dtype=np.float64)
            require(R2.BASE_MODULE.scalar(saved, "schema")
                    == "bb-ode-formal-discrete32-selected-state-v1"
                    and int(R2.BASE_MODULE.scalar(saved, "optimization_N_F")) == nf
                    and R2.BASE_MODULE.sha256_array(path32)
                    == R2.BASE_MODULE.scalar(saved, "path_sha256")
                    and R2.BASE_MODULE.sha256_array(energy32)
                    == R2.BASE_MODULE.scalar(saved, "energy_sha256")
                    and R2.BASE_MODULE.sha256_array(force32)
                    == R2.BASE_MODULE.scalar(saved, "force_sha256"),
                    f"preview ODE NF{nf} state binding differs")
        row = output[nf // 30 - 1]
        measured_i, measured_e = R2.BASE_MODULE.metric_pair(path32, energy32, force32)
        require(measured_i == row["i_perp"] and measured_e == row["delta_e32_eV"],
                f"preview ODE NF{nf} metric differs")
        row.update(path32=np.ascontiguousarray(path32),
                   energy32_eV=np.ascontiguousarray(energy32),
                   full_force32_eV_per_A=np.ascontiguousarray(force32))
    missing_budgets = set(requested_budgets) - {int(value) for value in states}
    require(missing_budgets == {6000},
            "preview reconstruction request differs from the registered NF6000 state")
    recorder, _selection = CACHE_ADAPTER.import_mutagen_transport_source(
        CACHE_ADAPTER.FROZEN_SOURCE_ROOT
    )
    for nf in sorted(missing_budgets):
        row = output[nf // 30 - 1]
        source_batch = int(rows[nf // 30 - 1]["current_source_batch_id"])
        reconstructed = recorder.reconstruct_batch(
            CACHE_ADAPTER.MUTAGEN_ARCHIVE, source_batch
        )
        path32 = np.ascontiguousarray(reconstructed[0], dtype=np.float64)
        energy32 = np.ascontiguousarray(reconstructed[1], dtype=np.float64)
        force32 = np.ascontiguousarray(reconstructed[2], dtype=np.float64)
        current_path32 = np.ascontiguousarray(reconstructed[3], dtype=np.float64)
        require(
            np.array_equal(path32, current_path32)
            and R2.BASE_MODULE.sha256_array(path32) == row["path_sha256"]
            and R2.BASE_MODULE.sha256_array(energy32) == row["energy_sha256"]
            and R2.BASE_MODULE.sha256_array(force32) == row["force_sha256"],
            f"reconstructed preview ODE NF{nf} state differs from its cache row",
        )
        measured_i, measured_e = R2.BASE_MODULE.metric_pair(
            path32, energy32, force32
        )
        require(measured_i == row["i_perp"] and measured_e == row["delta_e32_eV"],
                f"reconstructed preview ODE NF{nf} metric differs")
        row.update(
            path32=path32,
            energy32_eV=energy32,
            full_force32_eV_per_A=force32,
            state_role=(
                "current accepted path reconstructed from its immutable paid "
                f"source batch {source_batch}; ODE still in progress"
            ),
        )
    return output, cache_path, cache


def manifest_with_surface() -> dict[str, Any]:
    manifest = json.loads((HERE / "ANALYSIS_INPUTS_R2.template.json").read_text())
    adapter_path = readonly(ADAPTER, "R2B surface center adapter")
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
    manifest["inputs"]["ir_centered_surface_bundle"] = {
        key: adapter["surface_result"][key] for key in ("path", "sha256")
    }
    manifest["inputs"]["ir_centered_surface_audit"] = {
        key: adapter["surface_audit"][key] for key in ("path", "sha256")
    }
    manifest["inputs"]["surface_center_state_audit"] = {
        "path": str(adapter_path), "sha256": sha256_file(adapter_path),
    }
    return manifest


def publish_figures(figures: Mapping[str, Any]) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    require(all(not OUTPUTS[key].exists() for key in figures),
            "refusing to overwrite a versioned partial preview")
    staged = []; published = []
    try:
        for key, figure in figures.items():
            target = OUTPUTS[key]
            temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}.pdf")
            figure.savefig(temporary, format="pdf", bbox_inches="tight")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o444)
            staged.append((temporary, target))
        for temporary, target in staged:
            os.link(temporary, target); published.append(target)
        directory = os.open(OUTPUT_DIR, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        for target in reversed(published): target.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _target in staged: temporary.unlink(missing_ok=True)
    return {key: binding(OUTPUTS[key]) for key in figures}


def main() -> None:
    readonly(Path(__file__), "partial preview source")
    manifest = manifest_with_surface()
    source = Path(manifest["inputs"]["ode_source_root"])
    if str(source) not in sys.path: sys.path.insert(0, str(source))
    authorization = importlib.import_module("src.kgrid_execution_authorization")
    selection, _evidence = R2.load_adopted_selection(manifest, authorization)
    require(float(selection["selected_spring_k_eV_per_A2"]) == 0.1,
            "preview adopted k differs")
    ode, cache_path, cache = load_partial_ode(CACHE, EVOLUTION_BUDGETS)
    ivm, diagnostic = R2.load_ir_ivm_r2(manifest, EVOLUTION_BUDGETS)
    methods = {
        "IVM": ivm,
        "Previous IVM": R2._BASE_LOAD_OLD_IVM(manifest, EVOLUTION_BUDGETS),
        "ODE-NEB": ode,
        R2.IR_DIAGNOSTIC_KEY: diagnostic,
    }
    surface = R2.load_surface_r2(manifest)
    figures = {
        "cost": R2.plot_cost_r2(methods, live=True),
        "energy": R2.plot_energy_profiles_r2(methods, PREVIEW_NF),
        "evolution": R2.plot_evolution_r2(methods, surface, EVOLUTION_BUDGETS),
    }
    try:
        for figure in figures.values():
            figure.text(
                0.5, 0.50, "ODE IN PROGRESS THROUGH NF15000 — NOT FINAL",
                ha="center", va="center", fontsize=13, color="#A00000",
                alpha=0.24, rotation=12, weight="bold",
            )
        output_bindings = publish_figures(figures)
    finally:
        for figure in figures.values(): R2.BASE_MODULE.plt.close(figure)
    ode_state = next(row for row in ode if row["optimization_N_F"] == PREVIEW_NF)
    projected = R2.BASE_MODULE.project_path(
        ode_state["path32"], surface["center_path59_A"], surface["center_s"],
        surface["normal_directions"],
    )
    grid = surface["normal_coordinate_rms_A"]
    require(projected[:, 1].min() >= grid.min() - 2e-8
            and projected[:, 1].max() <= grid.max() + 2e-8,
            "NF15000 ODE path leaves the R2B diagnostic surface envelope")
    receipt = {
        "schema": "bb-discrete32-ode-nf30000-r2-partial-preview-v1",
        "status": "ODE_IN_PROGRESS_THROUGH_NF15000_NOT_FINAL",
        "formal_figure_eligible": False,
        "cache": binding(cache_path),
        "cache_processed_batches": cache["processed_batches"],
        "cache_processed_optimization_N_F": cache["processed_optimization_N_F"],
        "plotted_ode_cutoff_batches": PREVIEW_BATCHES,
        "plotted_ode_cutoff_optimization_N_F": PREVIEW_NF,
        "evolution_optimization_N_F": list(EVOLUTION_BUDGETS),
        "outputs": output_bindings,
        "ode_nf15000_surface_projection": {
            "normal_min_A": float(projected[:, 1].min()),
            "normal_max_A": float(projected[:, 1].max()),
            "grid_min_A": float(grid.min()), "grid_max_A": float(grid.max()),
            "strict_envelope_pass": True,
        },
        "physical_calls": 0, "openmm_contexts_constructed": 0,
        "bb_locks_acquired": 0,
    }
    require(not OUTPUTS["receipt"].exists(), "refusing to overwrite preview receipt")
    descriptor = os.open(OUTPUTS["receipt"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(json.dumps(receipt, sort_keys=True, separators=(",", ":"),
                                allow_nan=False).encode() + b"\n")
        stream.flush(); os.fsync(stream.fileno())
    require("openmm" not in sys.modules and "simtk.openmm" not in sys.modules,
            "partial preview imported OpenMM")
    print(json.dumps({"status": receipt["status"], "outputs": output_bindings,
                      "physical_calls": 0}, indent=2))


if __name__ == "__main__":
    main()
