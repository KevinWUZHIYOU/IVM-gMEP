#!/usr/bin/env python3
"""Render the established BB evolution style with the audited R2 surface.

This is a thin, zero-physics publication adapter.  It keeps the frozen IR-IVM
and ODE-NEB path snapshots and the established plotting routine, while replacing
only the old partially relaxed surface with the complete R2 merged 15 x 10
surface.  It refuses to run until the R2 merge and postprocess audits are sealed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
LEGACY = HERE / "plot_bb_relaxed15x9_diagnostic_evolution.py"
LEGACY_SHA256 = "0e396771c15c11d1cb0eea0133f1b6bd0f325d0f68d3c236a4877a25352ba756"
CONTINUATION = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "bb_relaxed_surface_continuation_20260903_r1"
)
ADAPTER = CONTINUATION / "postprocess_r2_merged_surface.py"
ADAPTER_SHA256 = "329bf9504fa9372005027611ce6b0c52f7803622f63014f56a5d0ecfa96b2c67"
SURFACE_ROOT = CONTINUATION / "postprocess/r2_merged"
SURFACE_GRID = SURFACE_ROOT / "surface_grid.npz"
SURFACE_SOURCES = SURFACE_ROOT / "surface_sources.json"
FINAL_TRANSFER_TOOL = (
    CONTINUATION.parent / "bb_storage_archive_20260903_r1/r2_final_aggregate_transfer.py"
)
FINAL_TRANSFER_TOOL_SHA256 = "540e09db14d28403f70b638130914257eff3448654852a4554f6e26133083aee"
FINAL_TRANSFER_ROOT = CONTINUATION.parent / "bb_storage_archive_20260903_r1/r2_final_aggregate"
FINAL_TRANSFER_MANIFEST = FINAL_TRANSFER_ROOT / "transfer.manifest.json"
FINAL_TRANSFER_RECEIPT = FINAL_TRANSFER_ROOT / "local_install.receipt.json"
DEFAULT_OUTPUT = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/"
    "barnase_barstar_irivm_constrained_relaxed_evolution.pdf"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, object]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage.", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_validated_inputs():
    require(sha256(LEGACY) == LEGACY_SHA256, "established plot implementation changed")
    require(sha256(ADAPTER) == ADAPTER_SHA256, "R2 postprocess adapter changed")
    require(sha256(FINAL_TRANSFER_TOOL) == FINAL_TRANSFER_TOOL_SHA256,
            "R2 final aggregate transfer tool changed")
    require(SURFACE_GRID.is_file() and SURFACE_SOURCES.is_file(),
            "run the R2 postprocess adapter after PASS_MERGED_150_POINTS first")
    transfer_receipt = json.loads(FINAL_TRANSFER_RECEIPT.read_text(encoding="utf-8"))
    require(
        transfer_receipt.get("schema") == "bb-r2-final-aggregate-transfer-v1-local-install-receipt"
        and transfer_receipt.get("status") == "PASS_LOCAL_INSTALLED_AND_REVALIDATED_R2_AGGREGATE"
        and transfer_receipt.get("source_hostname") == "Mac-mini.local"
        and transfer_receipt.get("target_hostname") == "wuzhiyoudeMacBook-Air-206.local"
        and transfer_receipt.get("new_physical_force_calls") == 0
        and transfer_receipt.get("processes_controlled") is False
        and transfer_receipt.get("data_deleted") is False
        and transfer_receipt.get("manifest") == binding(FINAL_TRANSFER_MANIFEST),
        "R2 final aggregate was not revalidated after the registered Mutagen transfer",
    )
    if str(CONTINUATION) not in sys.path:
        sys.path.insert(0, str(CONTINUATION))
    adapter = load_file_module("bb_r2_merged_surface_adapter", ADAPTER)
    legacy = load_file_module("bb_established_relaxed_evolution", LEGACY)
    analysis = legacy.load_module()
    inputs, report = adapter.load_for_plot(legacy, analysis)
    published = json.loads(SURFACE_SOURCES.read_text(encoding="utf-8"))
    expected = {**report, "surface_grid": adapter.r1.bind(SURFACE_GRID)}
    require(published == expected, "published R2 surface provenance differs from revalidation")
    return adapter, legacy, analysis, inputs, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--color-curvature", type=float, default=2.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--allow-bound-reference-clipping", action="store_true",
        help=("retain the established max(E-E_bound, 0) display when the audited "
              "surface contains points below the bound endpoint"),
    )
    args = parser.parse_args()
    _adapter, legacy, analysis, inputs, report = load_validated_inputs()
    energy_audit = report["energy_display_audit"]
    if args.preflight_only:
        print(json.dumps({
            "status": "PASS_R2_PLOT_PREFLIGHT",
            "surface_sources": binding(SURFACE_SOURCES),
            "fresh_converged": report["fresh_converged"],
            "fresh_maxstep": report["fresh_maxstep"],
            "energy_display_audit": energy_audit,
            "new_physical_force_calls": 0,
        }, indent=2, sort_keys=True, allow_nan=False))
        return
    require(
        not energy_audit["legacy_plot_would_clip_negative_values"]
        or args.allow_bound_reference_clipping,
        "audited surface has E < E_bound; inspect preflight and explicitly allow established clipping",
    )

    output = args.output.expanduser().resolve()
    figure, display = legacy.plot(analysis, *inputs, args.color_curvature)
    try:
        legacy.atomic_save_pdf(figure, output)
    finally:
        plt.close(figure)

    fully_relaxed = report["fresh_maxstep"] == 0
    provenance = {
        "schema": "bb-irivm-constrained-relaxed-diagnostic-evolution-r2-merged-v1",
        "status": "DIAGNOSTIC_NOT_FORMALLY_REPORTABLE",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scientific_interpretation": {
            "surface": "audited R2 merged 15 x 10 fixed-subspace constrained-relaxation terminal surface",
            "grid_shape_s_by_n": [15, 10],
            "grid_points": 150,
            "fresh_grid_points": 148,
            "fresh_converged": report["fresh_converged"],
            "maximum_update_steps": report["fresh_maxstep"],
            "cached_fixed_endpoints": 2,
            "fully_converged_in_fixed_subspace": fully_relaxed,
            "fully_unconstrained_or_global_minimum_claimed": False,
            "diagnostic_status": (
                "CONVERGED_FIXED_SUBSPACE_DIAGNOSTIC" if fully_relaxed
                else "PARTIAL_FIXED_SUBSPACE_DIAGNOSTIC"
            ),
        },
        "accounting": {
            "postprocessing_physical_calls": 0,
            "plotting_physical_calls": 0,
            "optimization_N_F_contribution": 0,
            "stored_coordinates_energy_raw_and_projected_force_recomputed": False,
        },
        "inputs": {
            "r2_surface_grid": binding(SURFACE_GRID),
            "r2_surface_sources": binding(SURFACE_SOURCES),
            "r2_merged_result": report["merged_result"],
            "r2_merged_audit": report["merged_audit"],
            "r2_assignment": report["assignment"],
            "r2_checkpoint": report["checkpoint"],
            "r2_dispatcher": report["dispatcher"],
            "r2_batch_transfer_tool": report["batch_transfer_tool"],
            "r2_native_batches": report["native_batches"],
            "r2_final_transfer_tool": binding(FINAL_TRANSFER_TOOL),
            "r2_final_transfer_manifest": binding(FINAL_TRANSFER_MANIFEST),
            "r2_final_transfer_receipt": binding(FINAL_TRANSFER_RECEIPT),
            "legacy_plot_code": binding(LEGACY),
            "r2_surface_adapter_code": binding(ADAPTER),
            "plot_wrapper_code": binding(Path(__file__)),
        },
        "energy_display_audit": energy_audit,
        "established_bound_reference_clipping_explicitly_allowed": bool(
            args.allow_bound_reference_clipping
        ),
        "display": display,
        "output": binding(output),
        "formally_reportable": False,
    }
    provenance_path = output.with_suffix(".provenance.json")
    atomic_json(provenance_path, provenance)
    print(json.dumps({
        "status": provenance["status"],
        "pdf": str(output),
        "provenance": str(provenance_path),
        "fresh_converged": report["fresh_converged"],
        "fresh_maxstep": report["fresh_maxstep"],
        "plotting_physical_calls": 0,
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
