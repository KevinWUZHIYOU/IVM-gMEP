#!/usr/bin/env python3
"""Render a zero-physics globally smooth color-normalization canonical.

The color coordinate is one C-infinity function over the full energy range: a
positive mixture of a linear map and a normalized Gaussian CDF.  Its three
parameters are fitted to the user's requested layout: approximately the old
curvature-1.5 tick positions through 200 kcal/mol and an approximately uniform
tail above 200.  There are no energy-domain joins or piecewise derivatives.
All scientific inputs, native 32-image paths, artists, contours, and geometry
are inherited unchanged.
"""

from __future__ import annotations

import datetime as dt
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import erf  # noqa: E402


HERE = Path(__file__).resolve().parent
SWEEP_SCRIPT = HERE / "plot_bb_r2_square_all_images_curvature_sweep.py"
PREVIEW_OUTPUT = (
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf")
    / "barnase_barstar_irivm_constrained_relaxed_evolution_"
      "smooth_global_gaussian_cdf_fit_preview.pdf"
)
CANONICAL_OUTPUT = (
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf")
    / "barnase_barstar_irivm_constrained_relaxed_evolution.pdf"
)
CANONICAL_MANIFEST = CANONICAL_OUTPUT.with_suffix(
    ".smooth_global_fit.manifest.json"
)
IR20K_SHARD = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
    "reports/plot_inputs/bb_irivm_nf20k_nf30k_20260904_r1/"
    "call_0693_stable_last.npz"
)
IR20K_PRIMARY_AUDIT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
    "reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF30000_20260904_R6B_PRIMARY.json"
)
EXPLICIT_GAMMA2_OUTPUT = (
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf")
    / "barnase_barstar_irivm_constrained_relaxed_evolution_"
      "square_all_points_curvature_2.pdf"
)
REFERENCE_CURVATURE = 1.5
FIT_SWITCH_ENERGY = 200.0
FIT_SWITCH_COLOR = 0.72
FIT_ANCHOR_ENERGIES = np.array(
    [0.0, 50.0, 100.0, 150.0, 200.0, 240.0, 300.0, 400.0, 500.0, 600.0],
    dtype=np.float64,
)
FIT_LOW_WEIGHT = 5.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sweep = load_module("bb_native32_sweep_for_smooth_c2", SWEEP_SCRIPT)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.stage.", suffix=destination.suffix,
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def preserve_gamma2_sweep_member() -> dict[str, object]:
    """Move the former canonical address into an explicit sweep address."""
    explicit_provenance_path = EXPLICIT_GAMMA2_OUTPUT.with_suffix(
        ".provenance.json"
    )
    if (EXPLICIT_GAMMA2_OUTPUT.is_file()
            and explicit_provenance_path.is_file()
            and sweep.MANIFEST.is_file()):
        return {
            "pdf": sweep.binding(EXPLICIT_GAMMA2_OUTPUT),
            "provenance": sweep.binding(explicit_provenance_path),
            "seven_variant_manifest": sweep.binding(sweep.MANIFEST),
        }
    old_provenance_path = CANONICAL_OUTPUT.with_suffix(".provenance.json")
    sweep.require(CANONICAL_OUTPUT.is_file() and old_provenance_path.is_file(),
                  "former curvature-2 canonical and provenance are required")
    old_provenance = json.loads(old_provenance_path.read_text())
    sweep.require(
        old_provenance.get("schema")
        == "bb-irivm-r2-midband-native32-curvature-variant-v3"
        and float(old_provenance.get("color_curvature")) == 2.0,
        "current canonical is not the audited curvature-2 sweep member",
    )
    atomic_copy(CANONICAL_OUTPUT, EXPLICIT_GAMMA2_OUTPUT)
    sweep.require(sweep.sha256(EXPLICIT_GAMMA2_OUTPUT)
                  == sweep.sha256(CANONICAL_OUTPUT),
                  "explicit curvature-2 copy differs from former canonical")
    explicit_provenance = dict(old_provenance)
    explicit_provenance["output"] = sweep.binding(EXPLICIT_GAMMA2_OUTPUT)
    explicit_provenance["address_change"] = {
        "reason": "preserve the curvature-2 member before smooth-C2 canonical promotion",
        "former_path": str(CANONICAL_OUTPUT),
        "pdf_bytes_changed": False,
        "scientific_or_display_values_changed": False,
    }
    sweep.atomic_json(explicit_provenance_path, explicit_provenance)

    manifest = json.loads(sweep.MANIFEST.read_text())
    sweep.require(
        manifest.get("schema")
        == "bb-irivm-r2-midband-native32-curvature-sweep-v3",
        "seven-variant manifest schema differs",
    )
    matches = [row for row in manifest["outputs"]
               if float(row["curvature"]) == 2.0]
    sweep.require(len(matches) == 1,
                  "seven-variant manifest must have one curvature-2 member")
    matches[0]["pdf"] = sweep.binding(EXPLICIT_GAMMA2_OUTPUT)
    matches[0]["provenance"] = sweep.binding(explicit_provenance_path)
    manifest["created_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest["curvature_2_address_change"] = {
        "former_canonical_path": str(CANONICAL_OUTPUT),
        "explicit_preserved_path": str(EXPLICIT_GAMMA2_OUTPUT),
        "pdf_bytes_changed": False,
        "reason": "the smooth-C2 rendering now occupies the canonical address",
    }
    sweep.atomic_json(sweep.MANIFEST, manifest)
    return {
        "pdf": sweep.binding(EXPLICIT_GAMMA2_OUTPUT),
        "provenance": sweep.binding(explicit_provenance_path),
        "seven_variant_manifest": sweep.binding(sweep.MANIFEST),
    }


class SmoothGaussianCDFLinearFitNorm(Normalize):
    """A globally smooth, strictly monotone Gaussian-CDF/linear mixture."""

    def __init__(self, vmin, vmax, scale=40.0, curvature=REFERENCE_CURVATURE):
        super().__init__(vmin=float(vmin), vmax=float(vmax), clip=True)
        sweep.require(self.vmin == 0.0, "smooth norm requires zero reference")
        sweep.require(self.vmax > FIT_ANCHOR_ENERGIES[-1],
                      "smooth norm requires vmax above 600 kcal/mol")
        sweep.require(math.isclose(float(curvature), REFERENCE_CURVATURE,
                                   rel_tol=0.0, abs_tol=0.0),
                      "smooth preview must use curvature 1.5")
        self.curvature = REFERENCE_CURVATURE
        self.reference = sweep.SegmentedMidrangeNorm(
            self.vmin, self.vmax, scale=scale, curvature=self.curvature
        )
        self.fit_energies = np.unique(np.r_[FIT_ANCHOR_ENERGIES, self.vmax])
        self.fit_targets = self._target_layout(self.fit_energies)
        self.fit_weights = np.where(
            self.fit_energies <= FIT_SWITCH_ENERGY, FIT_LOW_WEIGHT, 1.0
        )
        self.center, self.width, self.gaussian_fraction = self._fit_parameters()
        self.linear_fraction = 1.0 - self.gaussian_fraction
        self._cdf_zero = float(erf(-self.center / self.width))
        self._cdf_span = float(
            erf((self.vmax - self.center) / self.width) - self._cdf_zero
        )
        sweep.require(self._cdf_span > 0.0,
                      "Gaussian-CDF normalization span is not positive")

    def _target_layout(self, energy):
        value = np.asarray(energy, dtype=np.float64)
        target = np.asarray(self.reference(value), dtype=np.float64)
        reference_switch_color = float(self.reference(FIT_SWITCH_ENERGY))
        low = value <= FIT_SWITCH_ENERGY
        target[low] *= FIT_SWITCH_COLOR / reference_switch_color
        tail = value > FIT_SWITCH_ENERGY
        target[tail] = FIT_SWITCH_COLOR + (1.0 - FIT_SWITCH_COLOR) * (
            (value[tail] - FIT_SWITCH_ENERGY)
            / (self.vmax - FIT_SWITCH_ENERGY)
        )
        return target

    @staticmethod
    def _normalized_cdf(energy, center, width, vmax):
        cdf_zero = erf(-center / width)
        cdf_span = erf((vmax - center) / width) - cdf_zero
        return (erf((np.asarray(energy) - center) / width)
                - cdf_zero) / cdf_span

    def _fit_parameters(self):
        # Two deterministic searches make the fit reproducible without an
        # optimizer-dependent stopping condition.  For each (center, width),
        # the non-negative mixture fraction is solved analytically.
        best = None
        searches = [
            (np.linspace(80.0, 140.0, 121), np.linspace(45.0, 100.0, 111)),
            (None, None),
        ]
        for search_index, (centers, widths) in enumerate(searches):
            if search_index == 1:
                centers = np.linspace(best[1] - 1.0, best[1] + 1.0, 201)
                widths = np.linspace(best[2] - 1.0, best[2] + 1.0, 201)
            for center in centers:
                gaussian = self._normalized_cdf(
                    self.fit_energies, center, widths[:, None], self.vmax
                )
                linear = self.fit_energies / self.vmax
                direction = gaussian - linear
                residual = self.fit_targets - linear
                numerator = np.sum(
                    self.fit_weights * direction * residual, axis=1
                )
                denominator = np.sum(
                    self.fit_weights * direction * direction, axis=1
                )
                fractions = np.clip(numerator / denominator, 0.0, 1.0)
                predictions = linear + fractions[:, None] * direction
                errors = np.sum(
                    self.fit_weights * (predictions - self.fit_targets) ** 2,
                    axis=1,
                )
                index = int(np.argmin(errors))
                candidate = (
                    float(errors[index]), float(center), float(widths[index]),
                    float(fractions[index]),
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
        sweep.require(best is not None and 0.0 < best[3] < 1.0,
                      "smooth color fit failed")
        return best[1], best[2], best[3]

    def _gaussian_map(self, energy):
        return (erf((np.asarray(energy, dtype=np.float64) - self.center)
                    / self.width) - self._cdf_zero) / self._cdf_span

    def derivative(self, energy):
        value = np.asarray(energy, dtype=np.float64)
        gaussian_derivative = (
            2.0 / (math.sqrt(math.pi) * self.width * self._cdf_span)
            * np.exp(-((value - self.center) / self.width) ** 2)
        )
        return (self.linear_fraction / self.vmax
                + self.gaussian_fraction * gaussian_derivative)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        energy = np.clip(result.data, self.vmin, self.vmax)
        mapped = (self.linear_fraction * energy / self.vmax
                  + self.gaussian_fraction * self._gaussian_map(energy))
        output = np.ma.array(mapped, mask=result.mask, copy=False)
        return output[0] if is_scalar else output

    def inverse(self, value):
        original = np.asarray(value, dtype=np.float64)
        mapped = np.clip(original.reshape(-1), 0.0, 1.0)
        lower = np.full(mapped.shape, self.vmin)
        upper = np.full(mapped.shape, self.vmax)
        for _ in range(64):
            midpoint = 0.5 * (lower + upper)
            below = np.asarray(self(midpoint), dtype=np.float64) < mapped
            lower[below] = midpoint[below]
            upper[~below] = midpoint[~below]
        energy = 0.5 * (lower + upper)
        shaped = energy.reshape(original.shape)
        return float(shaped) if original.ndim == 0 else shaped


def configure_colorbar(figure, norm) -> dict[str, object]:
    axis = figure.axes[-1]
    ticks = [50.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0,
             600.0, float(norm.vmax)]
    labels = ["50", "100", "150", "200", "300", "400", "500", "600", ""]
    axis.set_yticks(ticks)
    axis.set_yticklabels(labels)
    axis.tick_params(axis="y", labelsize=4.8, length=2.0, pad=1.0)
    exponent = int(math.floor(math.log10(norm.vmax)))
    mantissa = norm.vmax / (10.0 ** exponent)
    top_label = rf"${mantissa:.1f}\times10^{{{exponent}}}$"
    axis.text(0.5, 1.025, top_label, transform=axis.transAxes,
              ha="center", va="bottom", fontsize=4.8, clip_on=False)
    return {
        "tick_energies_kcal_per_mol": ticks,
        "left_label_energies_kcal_per_mol": [],
        "right_label_energies_kcal_per_mol": [50.0, 100.0, 150.0,
                                               200.0, 300.0, 400.0,
                                               500.0, 600.0],
        "top_exact_vmax_label": top_label,
    }


def norm_audit(norm: SmoothGaussianCDFLinearFitNorm) -> dict[str, object]:
    energy = np.unique(np.r_[np.linspace(norm.vmin, norm.vmax, 100001),
                             norm.fit_energies])
    mapped = np.asarray(norm(energy), dtype=np.float64)
    slopes = np.asarray(norm.derivative(energy), dtype=np.float64)
    sweep.require(np.all(np.diff(mapped) > 0.0) and np.all(slopes > 0.0),
                  "smooth mapping is not strictly monotone")
    restored = np.asarray(norm.inverse(mapped), dtype=np.float64)
    inverse_error = float(np.max(np.abs(restored - energy)))
    sweep.require(inverse_error <= 2.0e-12,
                  "smooth inverse round-trip exceeds tolerance")

    fitted = np.asarray(norm(norm.fit_energies), dtype=np.float64)
    errors = fitted - norm.fit_targets
    below = norm.fit_energies <= FIT_SWITCH_ENERGY
    above = norm.fit_energies > FIT_SWITCH_ENERGY
    return {
        "selected_scheme": "global Gaussian-CDF plus linear weighted fit",
        "global_continuity": "C-infinity; no joins",
        "fit_target": (
            "curvature-1.5 relative spacing through 200 kcal/mol, rescaled "
            "to 72% of the bar, and a linear target from 200 kcal/mol to the "
            "measured maximum"
        ),
        "target_color_coordinate_at_200": FIT_SWITCH_COLOR,
        "fit_anchor_energies_kcal_per_mol": norm.fit_energies.tolist(),
        "fit_target_color_coordinates": norm.fit_targets.tolist(),
        "fitted_color_coordinates": fitted.tolist(),
        "fit_residuals": errors.tolist(),
        "weighted_sum_squared_error": float(np.sum(
            norm.fit_weights * errors ** 2
        )),
        "maximum_absolute_fit_error_below_or_at_200": float(
            np.max(np.abs(errors[below]))
        ),
        "maximum_absolute_fit_error_above_200": float(
            np.max(np.abs(errors[above]))
        ),
        "fitted_parameters": {
            "gaussian_center_kcal_per_mol": norm.center,
            "gaussian_width_kcal_per_mol": norm.width,
            "gaussian_fraction": norm.gaussian_fraction,
            "linear_fraction": norm.linear_fraction,
        },
        "strictly_monotone_on_dense_grid": True,
        "dense_grid_points": int(energy.size),
        "minimum_derivative_per_kcal_per_mol": float(np.min(slopes)),
        "maximum_inverse_round_trip_error_kcal_per_mol": inverse_error,
        "exact_endpoints": [[0.0, float(norm(0.0))],
                            [float(norm.vmax), float(norm(norm.vmax))]],
        "inverse": "64-step monotone bisection",
    }


def load_validated_ir20k_projection():
    """Load the accepted path nearest 20k without any physics evaluation."""
    audit = json.loads(IR20K_PRIMARY_AUDIT.read_text(encoding="utf-8"))
    sweep.require(
        audit.get("status")
        == "PASS_R6B_NF30000_EXACT_PREFIX318_AND_FULL_STATE_CAPTURE",
        "primary NF30000 history audit does not have the required PASS status",
    )
    matches = [
        row for row in audit["path_inventory"]
        if int(row["call_per_image"]) == 693
    ]
    sweep.require(len(matches) == 1, "primary audit lacks unique IR-IVM call 693")
    expected = matches[0]
    sweep.require(
        expected["optimization_N_F"] == 19993
        and expected["formally_reportable"] is True
        and expected["terminal_uncommitted_charged_tail"] is False,
        "IR-IVM call 693 is not the accepted path nearest 20k",
    )
    source = sweep.binding(IR20K_SHARD)
    sweep.require(source["sha256"] == expected["shard_sha256"],
                  "local IR-IVM 20k shard differs from primary audit")
    with np.load(IR20K_SHARD, allow_pickle=False) as saved:
        path_raw = np.asarray(saved["path32_A"], dtype=np.float64)
        optimization_nf = int(saved["optimization_N_F"])
        call_per_image = int(saved["call_per_image"])
        path_sha = str(saved["path_sha256"].item())
    sweep.require(
        path_raw.shape == (sweep.N_IMAGES, sweep.N_DOF)
        and optimization_nf == 19993
        and call_per_image == 693
        and sweep.array_sha256(path_raw) == path_sha == expected["path_sha256"],
        "IR-IVM 20k path data or counters differ from primary audit",
    )
    basis = sweep.load_projection_basis()
    projected = sweep.project_native_path32(
        path_raw.reshape(sweep.N_IMAGES, sweep.N_ATOMS, 3), basis
    )
    return projected, {
        "target_N_F": 20000,
        "optimization_N_F": optimization_nf,
        "call_per_image": call_per_image,
        "formally_reportable": True,
        "terminal_uncommitted_charged_tail": False,
        "source": source,
        "native_path": {
            "array_key": "path32_A",
            "native_stored_shape": list(path_raw.shape),
            "native_array_sha256": path_sha,
        },
        "primary_audit": sweep.binding(IR20K_PRIMARY_AUDIT),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--promote-canonical", action="store_true",
        help="render the globally smooth fitted normalization as canonical",
    )
    parser.add_argument(
        "--final-ir20k", action="store_true",
        help="replace only the final-panel accepted IR-IVM 15k path with 20k",
    )
    args = parser.parse_args(argv)
    output = CANONICAL_OUTPUT if args.promote_canonical else PREVIEW_OUTPUT
    preserved_gamma2 = (
        preserve_gamma2_sweep_member() if args.promote_canonical else None
    )
    sweep.require(sweep.sha256(sweep.BASE_WRAPPER)
                  == sweep.BASE_WRAPPER_SHA256,
                  "base plotting wrapper changed")
    wrapper = sweep.load_file_module("bb_smooth_global_fit_wrapper",
                                     sweep.BASE_WRAPPER)
    _adapter, legacy, analysis, inputs, report = wrapper.load_validated_inputs()
    sweep.require(legacy.FILLED_CONTOUR_BANDS
                  == sweep.PREVIOUS_FILLED_CONTOUR_BANDS,
                  "inherited contour count differs")
    legacy.FILLED_CONTOUR_BANDS = sweep.REQUESTED_FILLED_CONTOUR_BANDS
    analysis.BASE_MODULE.LogQuadraticNorm = SmoothGaussianCDFLinearFitNorm

    snapshots = sweep.path_snapshot_records(inputs)
    projections, projection_audit = sweep.build_native_projections(inputs)
    ir20k_record = None
    if args.final_ir20k:
        ir20k_projection, ir20k_record = load_validated_ir20k_projection()
        projections[3]["IR-IVM"] = ir20k_projection
        snapshots["IR-IVM"].pop("15000")
        snapshots["IR-IVM"]["20000"] = ir20k_record
    figure, inherited_display = legacy.plot(
        analysis, *inputs, color_curvature=REFERENCE_CURVATURE
    )
    try:
        style_audit = sweep.adapt_figure(figure, projections)
        if args.final_ir20k:
            figure.axes[3].set_title(
                "IR-IVM 20k | ODE-NEB 30k", fontsize=6.8, pad=3.0
            )
            inherited_display["panel_budget_labels"][-1]["IR-IVM"] = 20000
            inherited_display["actual_ir_ivm_N_F"][-1] = 19993
        inherited_display["plot_ylim_rms_A"] = style_audit[
            "final_plot_ylim_rms_A"
        ]
        vmax = float(inherited_display["display_maximum_kcal_per_mol"])
        norm = SmoothGaussianCDFLinearFitNorm(
            0.0, vmax, curvature=REFERENCE_CURVATURE
        )
        verification = norm_audit(norm)
        colorbar_tick_audit = configure_colorbar(figure, norm)
        inherited_display["color_norm"] = {
            "name": "SmoothGaussianCDFLinearFitNorm",
            "global_continuity": "C-infinity; no joins",
            "fit_target_below_or_at_200": (
                "curvature 1.5 relative tick spacing rescaled to 72%"
            ),
            "fit_target_above_200": "uniform tick-position progression",
            "target_color_coordinate_at_200": FIT_SWITCH_COLOR,
            "true_vmax_kcal_per_mol": vmax,
            "gaussian_center_kcal_per_mol": norm.center,
            "gaussian_width_kcal_per_mol": norm.width,
            "gaussian_fraction": norm.gaussian_fraction,
            "linear_fraction": norm.linear_fraction,
        }
        legacy.atomic_save_pdf(figure, output)
    finally:
        plt.close(figure)

    provenance = {
        "schema": (
            "bb-irivm-r2-smooth-global-gaussian-cdf-fit-canonical-v1"
            if args.promote_canonical
            else "bb-irivm-r2-smooth-global-gaussian-cdf-fit-preview-v1"
        ),
        "status": (
            "PASS_ZERO_PHYSICS_SMOOTH_GLOBAL_FIT_CANONICAL"
            if args.promote_canonical
            else "PASS_ZERO_PHYSICS_SMOOTH_GLOBAL_FIT_PREVIEW"
        ),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output": sweep.binding(output),
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
        "formally_reportable": bool(args.promote_canonical),
        "canonical_promotion": bool(args.promote_canonical),
        "final_panel_ir_ivm_replacement": (
            {
                "old_target_N_F": 15000,
                "new_target_N_F": 20000,
                "new_actual_optimization_N_F": 19993,
                "only_final_ir_ivm_path_and_title_changed": True,
                "record": ir20k_record,
            }
            if args.final_ir20k else None
        ),
    }
    if preserved_gamma2 is not None:
        provenance["preserved_curvature_2_sweep_member"] = preserved_gamma2
    provenance_path = output.with_suffix(".provenance.json")
    sweep.atomic_json(provenance_path, provenance)
    result = {
        "status": provenance["status"],
        "pdf": provenance["output"],
        "provenance": sweep.binding(provenance_path),
        "normalization_verification": verification,
        "native_projected_points_per_path": 32,
        "path_resampled_or_densified": False,
    }
    if args.promote_canonical:
        canonical_manifest = {
            "schema": "bb-irivm-r2-smooth-global-fit-canonical-manifest-v1",
            "status": "PASS_SMOOTH_GLOBAL_FIT_CANONICAL_AND_GAMMA2_PRESERVATION",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "canonical_pdf": sweep.binding(output),
            "canonical_provenance": sweep.binding(provenance_path),
            "builder": sweep.binding(Path(__file__)),
            "source_preview": sweep.binding(PREVIEW_OUTPUT),
            "source_preview_provenance": sweep.binding(
                PREVIEW_OUTPUT.with_suffix(".provenance.json")
            ),
            "preserved_curvature_2_sweep_member": preserved_gamma2,
            "normalization_verification": verification,
            "physical_calls": 0,
            "optimization_N_F_contribution": 0,
            "final_panel_ir_ivm_target_N_F": (
                20000 if args.final_ir20k else 15000
            ),
        }
        sweep.atomic_json(CANONICAL_MANIFEST, canonical_manifest)
        result["canonical_manifest"] = sweep.binding(CANONICAL_MANIFEST)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
