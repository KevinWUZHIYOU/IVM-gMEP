#!/usr/bin/env python3
"""Plot saved BB IR-IVM paths nearest 20k and 30k on the audited R2 surface.

This is a zero-physics diagnostic renderer.  It reads the two already captured
native 32-image path/energy/force shards, verifies them against the primary
NF30000 history audit, and projects the real images directly onto the saved
59-point section coordinate basis.  It does not resample paths, call the
potential, or add any optimization N_F.

The 20k state (N_F=19,993) is an accepted/formally reportable path.  The 30k
state (N_F=29,978) is a saved terminal uncommitted charged trial and is labelled
as such in the figure and provenance.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
REPOSITORY = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP")
PROJECT = Path("/Users/wuzhiyou/IVM_for_MEP")
SWEEP_SCRIPT = HERE / "plot_bb_r2_square_all_images_curvature_sweep.py"
SMOOTH_NORM_SCRIPT = HERE / "plot_bb_r2_smooth_c2_gamma15_to_linear_preview.py"
REFERENCE_PDF = (
    REPOSITORY / "output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution.pdf"
)
OUTPUT = (
    REPOSITORY / "output/pdf/barnase_barstar_irivm_20k_30k_diagnostic_evolution.pdf"
)
PROVENANCE = OUTPUT.with_suffix(".provenance.json")
TRACEABILITY = OUTPUT.with_suffix(".traceability.md")
PRIMARY_AUDIT = (
    PROJECT / "python/barnase_barstar_ir_ivm_20260901/reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF30000_20260904_R6B_PRIMARY.json"
)
INPUT_DIRECTORY = (
    PROJECT / "python/barnase_barstar_ir_ivm_20260901/reports/plot_inputs/"
    "bb_irivm_nf20k_nf30k_20260904_r1"
)
SHARDS = {
    693: INPUT_DIRECTORY / "call_0693_stable_last.npz",
    1053: INPUT_DIRECTORY / "call_1053_stable_last.npz",
}
TARGETS = (
    {"call": 693, "target_N_F": 20000, "title": "IR-IVM 20k"},
    {"call": 1053, "target_N_F": 30000, "title": "IR-IVM 30k trial*"},
)
EV_TO_KCAL = 23.06054783061903
N_IMAGES = 32
N_ATOMS = 3132
N_DOF = 9396


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load_module("bb_irivm_20k_30k_sweep", SWEEP_SCRIPT)
smooth = load_module("bb_irivm_20k_30k_smooth_norm", SMOOTH_NORM_SCRIPT)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage.", suffix=path.suffix, dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_and_validate_states() -> tuple[list[dict[str, object]], dict[str, object]]:
    audit = json.loads(PRIMARY_AUDIT.read_text(encoding="utf-8"))
    require(
        audit.get("status")
        == "PASS_R6B_NF30000_EXACT_PREFIX318_AND_FULL_STATE_CAPTURE",
        "primary NF30000 history audit does not have the required PASS status",
    )
    inventory = {int(row["call_per_image"]): row for row in audit["path_inventory"]}
    states = []
    for target in TARGETS:
        call = int(target["call"])
        expected = inventory[call]
        shard_path = SHARDS[call]
        require(sweep.sha256(shard_path) == expected["shard_sha256"],
                f"call {call} shard hash differs from the primary audit")
        with np.load(shard_path, allow_pickle=False) as saved:
            path = np.asarray(saved["path32_A"], dtype=np.float64)
            energy = np.asarray(saved["energy_eV"], dtype=np.float64)
            force = np.asarray(saved["force_eV_per_A"], dtype=np.float64)
            optimization_nf = int(saved["optimization_N_F"])
            call_per_image = int(saved["call_per_image"])
            discrete_iperp = float(saved["discrete_I_perp"])
            stored_hashes = {
                "path_sha256": str(saved["path_sha256"].item()),
                "energy_sha256": str(saved["energy_sha256"].item()),
                "force_sha256": str(saved["force_sha256"].item()),
            }
            fully_evaluated = bool(saved["fully_evaluated_current_path"])
            added_calls = int(saved["additional_physical_calls"])
        require(path.shape == (N_IMAGES, N_DOF),
                f"call {call} path is not the native 32-image path")
        require(energy.shape == (N_IMAGES,) and force.shape == (N_IMAGES, N_DOF),
                f"call {call} energy/force dimensions differ")
        require(call_per_image == call and optimization_nf == expected["optimization_N_F"],
                f"call {call} counters differ from the primary audit")
        computed_hashes = {
            "path_sha256": sha256_array(path),
            "energy_sha256": sha256_array(energy),
            "force_sha256": sha256_array(force),
        }
        require(stored_hashes == computed_hashes,
                f"call {call} embedded array hashes do not match array bytes")
        require(all(computed_hashes[key] == expected[key] for key in computed_hashes),
                f"call {call} array hashes differ from the primary audit")
        require(fully_evaluated and added_calls == 0,
                f"call {call} is not a complete zero-call captured state")
        states.append({
            **target,
            "optimization_N_F": optimization_nf,
            "path32": path.reshape(N_IMAGES, N_ATOMS, 3),
            "energy_eV": energy,
            "discrete_I_perp": discrete_iperp,
            "Delta_E_32_kcal_per_mol": float((np.max(energy) - energy[0]) * EV_TO_KCAL),
            "formally_reportable": bool(expected["formally_reportable"]),
            "terminal_uncommitted_charged_tail": bool(
                expected["terminal_uncommitted_charged_tail"]
            ),
            "array_hashes": computed_hashes,
            "shard": sweep.binding(shard_path),
            "remote_authoritative_shard": expected["shard"],
        })
    require(states[0]["formally_reportable"] is True,
            "20k state must be accepted/formally reportable")
    require(states[1]["formally_reportable"] is False
            and states[1]["terminal_uncommitted_charged_tail"] is True,
            "30k state must remain explicitly diagnostic-only")
    return states, audit


def render(states: list[dict[str, object]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    require(sweep.sha256(sweep.BASE_WRAPPER) == sweep.BASE_WRAPPER_SHA256,
            "base R2 plotting wrapper changed")
    wrapper = sweep.load_file_module("bb_irivm_20k_30k_wrapper", sweep.BASE_WRAPPER)
    _adapter, legacy, analysis, inputs, report = wrapper.load_validated_inputs()
    _ir, _ode, pilot, s, n, _center, _normals, _initial = inputs
    base = analysis.BASE_MODULE
    base.configure_style()

    energy_surface = np.asarray(pilot["final_energy_eV"], dtype=np.float64)
    zero_index = int(np.argmin(np.abs(n)))
    reference_eV = float(energy_surface[0, zero_index])
    relative = np.maximum((energy_surface - reference_eV) * EV_TO_KCAL, 0.0)
    maximum = float(np.max(relative))
    norm = smooth.SmoothGaussianCDFLinearFitNorm(
        0.0, maximum, curvature=smooth.REFERENCE_CURVATURE
    )
    norm_verification = smooth.norm_audit(norm)
    filled_bands = sweep.REQUESTED_FILLED_CONTOUR_BANDS
    filled_levels = norm.inverse(np.linspace(0.0, 1.0, filled_bands + 1))
    sm, nm = np.meshgrid(s, n, indexing="ij")

    basis = sweep.load_projection_basis()
    projection_rows = []
    for state in states:
        projected = sweep.project_native_path32(state["path32"], basis)
        state["projected"] = projected
        require(projected[:, 1].min() >= -0.12 - 2.0e-8
                and projected[:, 1].max() <= 0.72 + 2.0e-8,
                f"call {state['call']} leaves the audited section envelope")
        projection_rows.append({
            "call_per_image": int(state["call"]),
            "native_points": int(projected.shape[0]),
            "resampled_or_densified": False,
            "s_range": [float(projected[:, 0].min()), float(projected[:, 0].max())],
            "n_range_rms_A": [float(projected[:, 1].min()), float(projected[:, 1].max())],
        })

    figure, axes = plt.subplots(1, 2, figsize=(4.25, 2.62), sharex=True, sharey=True)
    contour = None
    for panel_index, (axis, state) in enumerate(zip(axes, states)):
        contour = axis.contourf(
            sm, nm, relative, levels=filled_levels, cmap="cividis",
            norm=norm, antialiased=False,
        )
        if hasattr(contour, "set_rasterized"):
            contour.set_rasterized(True)
        else:
            for collection in contour.collections:
                collection.set_rasterized(True)
        axis.contour(
            sm, nm, relative, levels=contour.levels[1:-1], colors="white",
            linewidths=0.25, linestyles="solid", alpha=0.28,
        )
        projected = state["projected"]
        axis.plot(
            projected[:, 0], projected[:, 1],
            color=sweep.REFERENCE_PATH_STYLES["IVM"]["color"],
            linestyle="solid", linewidth=1.75, marker="s", markersize=2.15,
            markevery=1, markeredgecolor="black", markeredgewidth=0.35,
            zorder=6,
        )
        axis.set_title(str(state["title"]), fontsize=6.8, pad=3.0)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-0.12, 0.72)
        axis.set_xticks((0.0, 0.5, 1.0))
        axis.tick_params(direction="out", length=2.8)
        axis.set_box_aspect(1.0)
        axis.text(
            0.035, 0.965, f"({chr(97 + panel_index)})", transform=axis.transAxes,
            va="top", ha="left", fontsize=8.5, fontweight="bold",
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    axes[0].get_xticklabels()[-1].set_visible(False)
    axes[0].set_ylabel(r"RMS path-normal $n$ ($\AA$)")
    figure.supxlabel(r"path progress $s$", x=0.47, y=0.105)
    figure.text(
        0.08, 0.026,
        r"* Saved charged trial at $N_F=29{,}978$; not an accepted path state.",
        ha="left", va="bottom", fontsize=5.0,
    )
    color_axis = figure.add_axes([0.895, 0.255, 0.018, 0.62])
    colorbar = figure.colorbar(contour, cax=color_axis)
    colorbar.set_label(r"$E_{\mathrm{0K}}-E_{\mathrm{bound}}$ (kcal mol$^{-1}$)")
    colorbar_tick_audit = smooth.configure_colorbar(figure, norm)
    figure.subplots_adjust(left=0.12, right=0.86, bottom=0.23, top=0.90, wspace=0.10)
    legacy.atomic_save_pdf(figure, OUTPUT)
    plt.close(figure)

    display = {
        "reference_energy_eV": reference_eV,
        "display_maximum_kcal_per_mol": maximum,
        "surface_shape": list(relative.shape),
        "surface_grid": sweep.binding(wrapper.SURFACE_GRID),
        "surface_sources": sweep.binding(wrapper.SURFACE_SOURCES),
        "fresh_converged": report["fresh_converged"],
        "fresh_maxstep": report["fresh_maxstep"],
        "filled_contour_bands": filled_bands,
        "internal_white_contours": filled_bands - 1,
        "white_contours_match_internal_fill_boundaries": True,
        "white_contour_style": {
            "color": "white", "linewidth_pt": 0.25,
            "alpha": 0.28, "linestyle": "solid",
        },
        "filled_contour_rasterized": True,
        "filled_contour_raster_dpi": 600,
        "normalization": norm_verification,
        "colorbar_ticks": colorbar_tick_audit,
        "panel_geometry": "two square axes",
        "path_style": sweep.REFERENCE_PATH_STYLES["IVM"],
    }
    projection_rows.append({"coordinate_basis": basis["source"]})
    return display, projection_rows


def main() -> int:
    states, audit = load_and_validate_states()
    display, projection = render(states)
    panel_records = []
    for state in states:
        panel_records.append({
            "title": state["title"],
            "target_N_F": int(state["target_N_F"]),
            "optimization_N_F": int(state["optimization_N_F"]),
            "call_per_image": int(state["call"]),
            "native_image_count": N_IMAGES,
            "discrete_I_perp": float(state["discrete_I_perp"]),
            "Delta_E_32_kcal_per_mol": float(state["Delta_E_32_kcal_per_mol"]),
            "formally_reportable": bool(state["formally_reportable"]),
            "terminal_uncommitted_charged_tail": bool(
                state["terminal_uncommitted_charged_tail"]
            ),
            "local_shard": state["shard"],
            "remote_authoritative_shard": state["remote_authoritative_shard"],
            "array_hashes": state["array_hashes"],
        })
    payload = {
        "schema": "bb-irivm-20k-30k-diagnostic-evolution-v1",
        "status": "PASS_20K_ACCEPTED_AND_30K_UNCOMMITTED_TRIAL_ZERO_PHYSICS_RENDER",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output": sweep.binding(OUTPUT),
        "code": sweep.binding(Path(__file__)),
        "reference_style_figure": sweep.binding(REFERENCE_PDF),
        "primary_history_audit": sweep.binding(PRIMARY_AUDIT),
        "primary_history_status": audit["status"],
        "last_formally_reportable_call": int(audit["last_complete_call"]),
        "terminal_optimization_N_F": int(audit["terminal_optimization_N_F"]),
        "panels": panel_records,
        "projection": projection,
        "display": display,
        "claim_policy": {
            "20k": "accepted/formally reportable path nearest 20k",
            "30k": "diagnostic-only saved terminal uncommitted charged trial nearest 30k",
            "30k_must_not_be_called_accepted_or_final": True,
        },
        "physical_calls": 0,
        "optimization_N_F_contribution": 0,
        "paths_resampled_or_densified": False,
        "stored_energy_or_force_values_changed": False,
    }
    sweep.atomic_json(PROVENANCE, payload)

    trace = f"""# Barnase--Barstar IR-IVM 20k/30k diagnostic evolution traceability

## Output

- PDF: `{OUTPUT}`
- SHA-256: `{payload['output']['sha256']}`
- Builder: `{Path(__file__).resolve()}`
- Provenance JSON: `{PROVENANCE}`

## What is shown

- Panel (a): call 693, accepted IR-IVM path at actual optimization $N_F=19,993$.
- Panel (b): call 1053, saved terminal trial at actual optimization $N_F=29,978$.
  It is part of the charged but uncommitted tail and is therefore diagnostic-only,
  not an accepted/final IR-IVM path.
- Both panels use the native 32 images. No path was interpolated or densified.

## Data lineage and validation

- Primary history audit: `{PRIMARY_AUDIT}`
- Primary audit SHA-256: `{payload['primary_history_audit']['sha256']}`
- 20k shard SHA-256: `{panel_records[0]['local_shard']['sha256']}`
- 30k shard SHA-256: `{panel_records[1]['local_shard']['sha256']}`
- Each shard's path, energy, and force array hashes were recomputed and matched
  both the embedded hashes and the primary history inventory.

## Rendering protocol

- Background: the same audited R2 fully converged fixed-subspace surface used by
  `barnase_barstar_irivm_constrained_relaxed_evolution.pdf`.
- Coordinates: direct projection of every native image onto the saved 59-point
  section coordinate basis.
- Styling: formal BB IR-IVM blue solid path with 32 square markers; 17 filled
  contour bands; 16 white solid contours on the actual internal fill boundaries;
  identical global smooth Gaussian-CDF/linear color normalization.
- New potential/force calls: 0. Optimization $N_F$ added: 0.

## Panel metrics retained for identification

| panel | actual $N_F$ | discrete $I_\\perp$ | $\\Delta E_{{32}}$ (kcal/mol) | status |
|---|---:|---:|---:|---|
| 20k | {panel_records[0]['optimization_N_F']} | {panel_records[0]['discrete_I_perp']:.12g} | {panel_records[0]['Delta_E_32_kcal_per_mol']:.12g} | accepted |
| 30k trial | {panel_records[1]['optimization_N_F']} | {panel_records[1]['discrete_I_perp']:.12g} | {panel_records[1]['Delta_E_32_kcal_per_mol']:.12g} | uncommitted charged trial |
"""
    atomic_text(TRACEABILITY, trace)
    print(json.dumps({
        "status": payload["status"],
        "pdf": payload["output"],
        "provenance": sweep.binding(PROVENANCE),
        "traceability": sweep.binding(TRACEABILITY),
        "physical_calls": 0,
        "optimization_N_F_contribution": 0,
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
