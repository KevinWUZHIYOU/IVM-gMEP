#!/usr/bin/env python3
"""Render one zero-physics hybrid-normalization preview.

At and below 200 kcal/mol the color mapping is exactly the native32 sweep's
curvature-1.5 mapping.  Above 200 kcal/mol it is a continuous linear map from
the curvature-1.5 value at 200 to color coordinate 1 at the measured maximum.
All surface data, native 32-image path projections, artists, contour counts,
panel geometry, and axis limits are inherited without modification.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
SWEEP_SCRIPT = HERE / "plot_bb_r2_square_all_images_curvature_sweep.py"
OUTPUT = (
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf")
    / "barnase_barstar_irivm_constrained_relaxed_evolution_"
      "hybrid_c1p5_linear_above_200_preview.pdf"
)
SWITCH_ENERGY = 200.0
REFERENCE_CURVATURE = 1.5


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load_module("bb_native32_sweep_for_hybrid", SWEEP_SCRIPT)


class HybridCurvature15LinearAbove200Norm(Normalize):
    """Curvature-1.5 mapping through 200, then a continuous linear tail."""

    def __init__(self, vmin, vmax, scale=40.0, curvature=REFERENCE_CURVATURE):
        super().__init__(vmin=float(vmin), vmax=float(vmax), clip=True)
        sweep.require(self.vmin == 0.0, "hybrid norm requires zero reference")
        sweep.require(self.vmax > SWITCH_ENERGY,
                      "hybrid norm requires vmax above 200 kcal/mol")
        sweep.require(math.isclose(float(curvature), REFERENCE_CURVATURE,
                                   rel_tol=0.0, abs_tol=0.0),
                      "hybrid preview must use curvature 1.5")
        self.curvature = REFERENCE_CURVATURE
        self.reference = sweep.SegmentedMidrangeNorm(
            self.vmin, self.vmax, scale=scale, curvature=self.curvature
        )
        self.switch_color = float(self.reference(SWITCH_ENERGY))
        sweep.require(0.0 < self.switch_color < 1.0,
                      "hybrid switch color is outside (0,1)")

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        energy = np.clip(result.data, self.vmin, self.vmax)
        mapped = np.asarray(self.reference(energy), dtype=np.float64).copy()
        high = energy > SWITCH_ENERGY
        mapped[high] = self.switch_color + (1.0 - self.switch_color) * (
            (energy[high] - SWITCH_ENERGY) / (self.vmax - SWITCH_ENERGY)
        )
        output = np.ma.array(mapped, mask=result.mask, copy=False)
        return output[0] if is_scalar else output

    def inverse(self, value):
        mapped = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
        energy = np.asarray(self.reference.inverse(mapped), dtype=np.float64)
        high = mapped > self.switch_color
        energy[high] = SWITCH_ENERGY + (self.vmax - SWITCH_ENERGY) * (
            (mapped[high] - self.switch_color) / (1.0 - self.switch_color)
        )
        return energy


def norm_audit(norm: HybridCurvature15LinearAbove200Norm) -> dict[str, object]:
    low_grid = np.unique(np.r_[np.linspace(0.0, SWITCH_ENERGY, 2001),
                               50.0, 200.0])
    reference = sweep.SegmentedMidrangeNorm(
        norm.vmin, norm.vmax, curvature=REFERENCE_CURVATURE
    )
    actual_low = np.asarray(norm(low_grid), dtype=np.float64)
    reference_low = np.asarray(reference(low_grid), dtype=np.float64)
    sweep.require(np.array_equal(actual_low, reference_low),
                  "hybrid mapping differs from curvature 1.5 at E <= 200")

    energy = np.unique(np.r_[np.linspace(norm.vmin, norm.vmax, 10001),
                             50.0, SWITCH_ENERGY, 240.0, norm.vmax])
    mapped = np.asarray(norm(energy), dtype=np.float64)
    sweep.require(np.all(np.diff(mapped) > 0.0),
                  "hybrid mapping is not strictly monotone")
    restored = np.asarray(norm.inverse(mapped), dtype=np.float64)
    inverse_error = float(np.max(np.abs(restored - energy)))
    sweep.require(inverse_error <= 2.0e-12,
                  "hybrid inverse round-trip exceeds tolerance")

    epsilon = 1.0e-9
    around = [SWITCH_ENERGY - epsilon, SWITCH_ENERGY,
              SWITCH_ENERGY + epsilon]
    colors = [float(norm(value)) for value in around]
    sweep.require(colors[0] <= colors[1] <= colors[2],
                  "hybrid mapping is discontinuous at 200")
    return {
        "reference_mapping_exact_for_E_le_200": True,
        "comparison_grid_points": int(low_grid.size),
        "maximum_absolute_difference_for_E_le_200": 0.0,
        "strictly_monotone_on_dense_grid": True,
        "dense_grid_points": int(energy.size),
        "maximum_inverse_round_trip_error_kcal_per_mol": inverse_error,
        "switch_energy_kcal_per_mol": SWITCH_ENERGY,
        "switch_color_coordinate": norm.switch_color,
        "continuity_probe_energy_kcal_per_mol": around,
        "continuity_probe_color_coordinate": colors,
        "linear_tail_endpoint": [float(norm.vmax), 1.0],
    }


def configure_colorbar(figure, norm) -> dict[str, object]:
    """Expose the hybrid split while avoiding stacked high-energy labels."""
    axis = figure.axes[-1]
    ticks = [50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0,
             600.0, float(norm.vmax)]
    labels = ["50", "100", "150", "200", "", "400", "", "600", ""]
    axis.set_yticks(ticks)
    axis.set_yticklabels(labels)
    axis.tick_params(axis="y", labelsize=4.8, length=2.0, pad=1.0)
    for value in (300.0, 500.0):
        axis.text(-0.32, float(norm(value)), f"{int(value)}",
                  transform=axis.transAxes, ha="right", va="center",
                  fontsize=4.8, clip_on=False)
    exponent = int(math.floor(math.log10(norm.vmax)))
    mantissa = norm.vmax / (10.0 ** exponent)
    top_label = rf"${mantissa:.1f}\times10^{{{exponent}}}$"
    axis.text(0.5, 1.025, top_label, transform=axis.transAxes,
              ha="center", va="bottom", fontsize=4.8, clip_on=False)
    return {
        "tick_energies_kcal_per_mol": ticks,
        "right_label_energies_kcal_per_mol": [50.0, 100.0, 150.0,
                                               200.0, 400.0, 600.0],
        "left_label_energies_kcal_per_mol": [300.0, 500.0],
        "top_exact_vmax_label": top_label,
        "alternating_high_energy_label_sides_avoid_overlap": True,
    }


def main() -> int:
    sweep.require(sweep.sha256(sweep.BASE_WRAPPER)
                  == sweep.BASE_WRAPPER_SHA256,
                  "base plotting wrapper changed")
    wrapper = sweep.load_file_module("bb_hybrid_preview_wrapper",
                                     sweep.BASE_WRAPPER)
    _adapter, legacy, analysis, inputs, report = wrapper.load_validated_inputs()
    sweep.require(legacy.FILLED_CONTOUR_BANDS
                  == sweep.PREVIOUS_FILLED_CONTOUR_BANDS,
                  "inherited contour count differs")
    legacy.FILLED_CONTOUR_BANDS = sweep.REQUESTED_FILLED_CONTOUR_BANDS
    analysis.BASE_MODULE.LogQuadraticNorm = HybridCurvature15LinearAbove200Norm

    snapshots = sweep.path_snapshot_records(inputs)
    projections, projection_audit = sweep.build_native_projections(inputs)
    figure, inherited_display = legacy.plot(
        analysis, *inputs, color_curvature=REFERENCE_CURVATURE
    )
    try:
        style_audit = sweep.adapt_figure(figure, projections)
        inherited_display["plot_ylim_rms_A"] = style_audit[
            "final_plot_ylim_rms_A"
        ]
        vmax = float(inherited_display["display_maximum_kcal_per_mol"])
        norm = HybridCurvature15LinearAbove200Norm(
            0.0, vmax, curvature=REFERENCE_CURVATURE
        )
        verification = norm_audit(norm)
        colorbar_tick_audit = configure_colorbar(figure, norm)
        inherited_display["color_norm"] = {
            "name": "HybridCurvature15LinearAbove200Norm",
            "curvature_1p5_mapping_exact_at_and_below_200": True,
            "switch_energy_kcal_per_mol": SWITCH_ENERGY,
            "switch_color_coordinate": norm.switch_color,
            "linear_above_200_to_true_vmax": True,
            "true_vmax_kcal_per_mol": vmax,
        }
        legacy.atomic_save_pdf(figure, OUTPUT)
    finally:
        plt.close(figure)

    provenance = {
        "schema": "bb-irivm-r2-hybrid-c1p5-linear-above200-preview-v1",
        "status": "PASS_ZERO_PHYSICS_HYBRID_PREVIEW",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output": sweep.binding(OUTPUT),
        "code": sweep.binding(Path(__file__)),
        "native32_sweep_code": sweep.binding(SWEEP_SCRIPT),
        "base_plot_wrapper": sweep.binding(sweep.BASE_WRAPPER),
        "inherited_display": inherited_display,
        "requested_style_audit": style_audit,
        "native_projection_audit": projection_audit,
        "unchanged_scientific_inputs": {
            "surface_grid": sweep.binding(wrapper.SURFACE_GRID),
            "surface_sources": sweep.binding(wrapper.SURFACE_SOURCES),
            "path_snapshots": snapshots,
            "fresh_converged": report["fresh_converged"],
            "fresh_maxstep": report["fresh_maxstep"],
        },
        "normalization_verification": verification,
        "colorbar_tick_audit": colorbar_tick_audit,
        "physical_calls": 0,
        "optimization_N_F_contribution": 0,
        "formally_reportable": False,
    }
    provenance_path = OUTPUT.with_suffix(".provenance.json")
    sweep.atomic_json(provenance_path, provenance)
    print(json.dumps({
        "status": provenance["status"],
        "pdf": provenance["output"],
        "provenance": sweep.binding(provenance_path),
        "normalization_verification": verification,
        "colorbar_tick_audit": colorbar_tick_audit,
        "native_projected_points_per_path": 32,
        "path_resampled_or_densified": False,
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
