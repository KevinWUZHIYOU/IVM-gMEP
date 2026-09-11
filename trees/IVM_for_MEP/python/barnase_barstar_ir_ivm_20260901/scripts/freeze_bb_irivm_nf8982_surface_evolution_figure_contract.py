#!/usr/bin/env python3
"""Freeze the final IR-centered BB evolution figure contract after surface PASS.

This is read-only with respect to physical data and cannot create an OpenMM
context.  It refuses to freeze until the complete R4 surface and exact-path
result/WAL/audit closures all validate.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP")
PLOTTER = ROOT / "scripts/plot_bb_irivm_nf8982_surface_evolution.py"
PATH_LOADER = ROOT / "scripts/plot_bb_irivm_evolution_hashbound.py"
PATH_SOURCE_CONTRACT = ROOT / "registrations/BB_IRIVM_EVOLUTION_FIGURE_CONTRACT_20260902_R1.json"
SURFACE_EVALUATOR = ROOT / "scripts/evaluate_bb_irivm_nf8982_0k_section_energy_r4.py"
SURFACE_EXECUTION_CONTRACT = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_CONTRACT_R4_20260902.json"
SURFACE_AUTHORIZATION = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_0K_SECTION_ENERGY_EXECUTION_AUTHORIZATION_R4_20260902.json"
SURFACE_RESULT = ROOT / "runs/diagnostics/bb_irivm_nf8982_0k_section_energy_59x28_20260902_r4.npz"
SURFACE_AUDIT = SURFACE_RESULT.with_suffix(".audit.json")
SURFACE_WAL = SURFACE_RESULT.with_suffix(".energy.wal.jsonl")
PATH_RESULT = ROOT / "runs/diagnostics/bb_irivm_nf8982_original_path32_energy_20260902_r4.npz"
PATH_AUDIT = PATH_RESULT.with_suffix(".audit.json")
PATH_WAL = PATH_RESULT.with_suffix(".energy.wal.jsonl")
COORDINATE_MANIFEST = ROOT / "runs/diagnostics/bb_irivm_nf8982_0k_section_coordinates_59x28_20260902_r2.npz"
STYLE_REFERENCE = DOWNLOADS / "Figures/barnase_barstar/barnase_branch_local_true0k_evolution.pdf"
CONTRACT = ROOT / "registrations/diagnostics/BB_IRIVM_NF8982_SURFACE_EVOLUTION_FIGURE_CONTRACT_R1_20260902.json"
OUTPUT = DOWNLOADS / "Figures/barnase_barstar/active_candidates/barnase_barstar_irivm_nf8982_surface_evolution.pdf"


def load_plotter():
    spec = importlib.util.spec_from_file_location("bb_ir_surface_figure_freezer", PLOTTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=CONTRACT)
    parser.add_argument("--figure-output", type=Path, default=OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    plotter = load_plotter()
    destination = args.output.expanduser().resolve()
    figure_output = args.figure_output.expanduser().resolve()
    plotter.require(destination == CONTRACT.resolve(), "noncanonical figure contract destination")
    plotter.require(figure_output == OUTPUT.resolve(), "noncanonical active-candidate output")
    plotter.require(not figure_output.exists(), "prospective final figure already exists")
    plotter.require(not figure_output.with_suffix(".json").exists(), "prospective provenance already exists")
    required = (
        PLOTTER,
        PATH_LOADER,
        PATH_SOURCE_CONTRACT,
        SURFACE_EVALUATOR,
        SURFACE_EXECUTION_CONTRACT,
        SURFACE_AUTHORIZATION,
        SURFACE_RESULT,
        SURFACE_AUDIT,
        SURFACE_WAL,
        PATH_RESULT,
        PATH_AUDIT,
        PATH_WAL,
        COORDINATE_MANIFEST,
        STYLE_REFERENCE,
    )
    missing = [str(path) for path in required if not path.exists()]
    plotter.require(not missing, f"final figure inputs are not complete: {missing}")
    payload = {
        "schema": plotter.FIGURE_SCHEMA,
        "status": plotter.FIGURE_STATUS,
        "scientific_identity": "Barnase--Barstar path evolution on the new IR-IVM-NF8982-centered unrelaxed Cartesian 0 K section",
        "legacy_reference_role": "visual style only; neither the legacy surface nor legacy calibrated CV supplies plotted data",
        "plotter": plotter.file_binding(PLOTTER),
        "contract_freezer": plotter.file_binding(Path(__file__).resolve()),
        "path_loader": plotter.file_binding(PATH_LOADER),
        "path_source_contract": plotter.file_binding(PATH_SOURCE_CONTRACT),
        "surface_evaluator": plotter.file_binding(SURFACE_EVALUATOR),
        "surface_execution_contract": plotter.file_binding(SURFACE_EXECUTION_CONTRACT),
        "surface_authorization": plotter.file_binding(SURFACE_AUTHORIZATION),
        "surface_result": plotter.file_binding(SURFACE_RESULT),
        "surface_audit": plotter.file_binding(SURFACE_AUDIT),
        "surface_wal": plotter.file_binding(SURFACE_WAL),
        "original_path32_result": plotter.file_binding(PATH_RESULT),
        "original_path32_audit": plotter.file_binding(PATH_AUDIT),
        "original_path32_wal": plotter.file_binding(PATH_WAL),
        "coordinate_manifest": plotter.file_binding(COORDINATE_MANIFEST),
        "style_reference_pdf": plotter.file_binding(STYLE_REFERENCE),
        "panels": [
            {"ivm_optimization_N_F": ivm, "ode_neb_optimization_N_F": ode}
            for ivm, ode in plotter.EXPECTED_PANELS
        ],
        "projection": {
            "path_resampling": "independent uniform Cartesian arc-length resampling to frozen center_s (59 points)",
            "normal_coordinate": "dot(resampled_path-center59, frozen_normal)/sqrt(3132)",
            "fixed_endpoint_n_A": 0.0,
        },
        "surface": {
            "shape": [59, 28],
            "s_range": [0.0, 1.0],
            "n_range_A": [-0.12, 0.69],
            "stored_energy_values": 1652,
            "fresh_energy_only_calls": 1650,
            "inherited_fixed_endpoint_values": 2,
            "force_calls": 0,
            "optimization_N_F_contribution": 0,
        },
        "style": {
            "layout": "1x4",
            "figure_inches": [7.2, 2.38],
            "colormap": "cividis",
            "normalization": "log1p(scale=40) followed by 1-(1-t)^2.5",
            "filled_levels": 41,
            "line_levels": 20,
            "ivm_color": plotter.IVM_COLOR,
            "ode_neb_color": plotter.ODE_COLOR,
            "initial_color": plotter.INITIAL_COLOR,
        },
        "output_pdf": str(figure_output),
        "output_provenance": str(figure_output.with_suffix(".json")),
        "no_physical_calls_permitted_by_this_contract": True,
    }
    # Validate the entire real closure before freezing.  A temporary contract
    # is used only as an in-memory/path context and is removed immediately.
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=destination.parent, delete=False) as stream:
        json.dump(payload, stream)
        temporary = Path(stream.name)
    try:
        surface = plotter._load_surface(temporary, payload)
        initial, panels, _ = plotter._load_paths(temporary, payload, surface)
        plotter.require(initial.shape == (59, 2) and len(panels) == 3, "final plot inventory mismatch")
    finally:
        temporary.unlink(missing_ok=True)
    report = {"status": "PASS_ZERO_PHYSICAL_FINAL_FIGURE_PREFLIGHT", "surface_result": plotter.file_binding(SURFACE_RESULT), "panels": list(plotter.EXPECTED_PANELS), "physical_calls": 0}
    if args.validate_only:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    atomic_json(destination, payload)
    report["contract"] = plotter.file_binding(destination)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
