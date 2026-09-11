#!/usr/bin/env python3
"""Plot two BB previews from hash-bound, already-computed data only.

Outputs:
1. Near-NF9990 0 K energy versus Cartesian arc length for selected IR-IVM,
   ODE--NEB, and the historical IVM baseline.
2. Selected IR-IVM/ODE--NEB evolution drawn on the *existing* manuscript
   branch-local 0 K section.  No OpenMM object is imported or constructed.

The legacy section is not reinterpreted as a section centered on IR-IVM.
Coordinates outside its stored normal range are shown over a hatched region.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


EV_TO_KCAL = 23.0605
N_ATOMS = 3132
IR_COLOR = "#0072B2"
ODE_COLOR = "#D55E00"
OLD_IVM_COLOR = "#6A6A6A"
INITIAL_COLOR = "#7B3294"

CANONICAL = Path("/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901")
DOWNLOADS = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP")
IR_ENERGY = CANONICAL / "runs/diagnostics/bb_irivm_selected_9990_arc500_20260902_r1.npz"
ODE_ENERGY = CANONICAL / "runs/diagnostics/bb_ode_neb_arc500_preview_20260901_r1.npz"
OLD_IVM_ENERGY = CANONICAL / "runs/diagnostics/bb_original_ivm_arc500_20260901_r1.npz"
IR_RESULT = CANONICAL / (
    "runs/production/ivm_bb_irivm_a1e0_r01_m05_inf9990_r1_"
    "central_state_ivm_cfge0354bb7178e_budget9990_storagecommitted.npz"
)
ODE_CONTRACT = CANONICAL / "registrations/diagnostics/BB_ODE_NEB_ARC500_PREVIEW_20260901_R1.json"
LEGACY_SURFACE = DOWNLOADS / (
    "code_on_server/barnase_barstar_chigstyle/figures/manuscript_candidate_20260727/"
    "manuscript_branch_local_true0k_pes.npz"
)
LEGACY_REFERENCE_PDF = DOWNLOADS / "Figures/barnase_barstar/barnase_branch_local_true0k_evolution.pdf"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def binding(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


class LogQuadraticNorm(Normalize):
    """The transform used by the accepted legacy BB evolution figure."""

    def __init__(self, vmin: float, vmax: float, scale: float = 40.0, curvature: float = 2.5):
        super().__init__(vmin=vmin, vmax=vmax, clip=True)
        self.scale = float(scale)
        self.curvature = float(curvature)
        self.log_span = np.log1p((self.vmax - self.vmin) / self.scale)

    def __call__(self, value, clip=None):
        result, scalar = self.process_value(value)
        raw = np.clip((result.data - self.vmin) / self.scale, 0.0, (self.vmax - self.vmin) / self.scale)
        logarithmic = np.log1p(raw) / self.log_span
        mapped = 1.0 - np.power(1.0 - logarithmic, self.curvature)
        output = np.ma.array(mapped, mask=result.mask, copy=False)
        return output[0] if scalar else output

    def inverse(self, value):
        mapped = np.asarray(value, dtype=float)
        logarithmic = 1.0 - np.power(np.clip(1.0 - mapped, 0.0, 1.0), 1.0 / self.curvature)
        return self.vmin + self.scale * np.expm1(logarithmic * self.log_span)


def save_atomic(fig, output: Path, metadata: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + f".tmp.{os.getpid()}" + output.suffix)
    fig.savefig(temporary, bbox_inches="tight", metadata=metadata)
    os.replace(temporary, output)


def load_energy_row(path: Path, target_nf: int, method: str) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        match = np.flatnonzero(nf == int(target_nf))
        require(match.size == 1, f"{method}: expected one NF{target_nf} row")
        index = int(match[0])
        require(str(np.asarray(saved["sampling"]).item()) == "uniform_cartesian_arclength", "sampling drift")
        require(str(np.asarray(saved["interpolation"]).item()) == "piecewise_linear", "interpolation drift")
        require(int(np.asarray(saved["number_of_points"]).item()) == 500, "point-count drift")
        energy = np.asarray(saved["energy_eV"][index], dtype=np.float64)
        require(energy.shape == (500,) and np.all(np.isfinite(energy)), "bad energy row")
        embedded_energy_hash = str(np.asarray(saved["energy_sha256"][index]).item())
        require(sha256_array(energy) == embedded_energy_hash, f"{method}: energy hash mismatch")
        length = float(np.asarray(saved["total_arclength_A"][index]).item())
        delta = float(np.asarray(saved["densified_Delta_E_kcal_per_mol"][index]).item())
        relative = (energy - energy[0]) * EV_TO_KCAL
        require(np.isclose(float(np.max(relative)), delta, rtol=0.0, atol=1e-10), f"{method}: Delta E mismatch")
        return {
            "method": method,
            "optimization_N_F": int(target_nf),
            "energy_eV": energy,
            "relative_kcal": relative,
            "arc_length_A": np.linspace(0.0, length, 500),
            "total_arclength_A": length,
            "Delta_E_kcal_per_mol": delta,
            "energy_sha256": embedded_energy_hash,
            "path_sha256": str(np.asarray(saved["selected_path_sha256"][index]).item()),
            "source_row_index": index,
        }


def plot_energy_landscape(output: Path) -> list[dict]:
    curves = [
        load_energy_row(IR_ENERGY, 8982, "IR-IVM"),
        load_energy_row(ODE_ENERGY, 9990, "ODE-NEB"),
        load_energy_row(OLD_IVM_ENERGY, 9990, "Original IVM"),
    ]
    style = {
        "IR-IVM": (IR_COLOR, "-", "s"),
        "ODE-NEB": (ODE_COLOR, "-", "o"),
        "Original IVM": (OLD_IVM_COLOR, "--", "^"),
    }
    fig, axis = plt.subplots(figsize=(4.15, 2.85))
    for curve in curves:
        color, line, marker = style[curve["method"]]
        axis.plot(
            curve["arc_length_A"], curve["relative_kcal"],
            color=color, linestyle=line, linewidth=1.65,
            marker=marker, markersize=2.5, markevery=40,
            label=(
                f"{curve['method']}  "
                rf"($N_F={curve['optimization_N_F']}$, "
                rf"$\Delta E={curve['Delta_E_kcal_per_mol']:.0f}$)"
            ),
        )
        peak = int(np.argmax(curve["relative_kcal"]))
        axis.scatter(
            [curve["arc_length_A"][peak]], [curve["relative_kcal"][peak]],
            s=15, facecolor="white", edgecolor=color, linewidth=0.9, zorder=5,
        )
    axis.set_xlabel(r"Cartesian arc length $s$ ($\AA$)")
    axis.set_ylabel(r"$E(s)-E(0)$ (kcal mol$^{-1}$)")
    axis.set_title(r"Near-$N_F=9990$ energy landscapes", pad=5)
    axis.set_xlim(0.0, max(curve["total_arclength_A"] for curve in curves))
    axis.set_ylim(bottom=0.0)
    axis.grid(True, which="major", color="#D8D8D8", linewidth=0.45, alpha=0.75)
    axis.legend(frameon=False, loc="upper right", handlelength=2.3)
    axis.tick_params(direction="out", length=3)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.90)
    save_atomic(fig, output, {
        "Title": "Barnase--Barstar near-NF9990 energy landscapes",
        "Subject": "Existing arc-length-500 energy data; no new physical calls",
    })
    plt.close(fig)
    return curves


def load_ode_paths() -> dict[int, np.ndarray]:
    contract = json.loads(ODE_CONTRACT.read_text(encoding="utf-8"))
    rows = {}
    for item in contract["milestones"]:
        nf = int(item["optimization_N_F"])
        if nf not in (240, 480, 9990):
            continue
        source_record = item["coordinate_archive"]
        source = (ODE_CONTRACT.parent / source_record["path"]).resolve(strict=True)
        require(sha256_file(source) == str(source_record["sha256"]), f"ODE NF{nf} source drift")
        with np.load(source, allow_pickle=False) as saved:
            path3 = np.concatenate((
                np.asarray(saved["endpoint0_A"], dtype=np.float64)[None],
                np.asarray(saved["coordinates30_A"], dtype=np.float64),
                np.asarray(saved["endpoint1_A"], dtype=np.float64)[None],
            ), axis=0)
        path = np.ascontiguousarray(path3.reshape(32, -1))
        require(path.shape == (32, N_ATOMS * 3), "ODE path shape drift")
        rows[nf] = path
    require(set(rows) == {240, 480, 9990}, "missing ODE evolution paths")
    return rows


def load_ir_paths() -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    with np.load(IR_RESULT, allow_pickle=False) as saved:
        calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        paths = np.asarray(saved["stored_paths"], dtype=np.float64)
        history_calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        history_nf = np.asarray(saved["total_internal_joint_evaluations"], dtype=np.int64)
        last_complete_call = int(np.asarray(saved["total_internal_nf_last_complete_state_index"]).item()) + 1
        require(last_complete_call == 318, "IR-IVM last complete call drift")
        selected = {}
        call_to_nf = {}
        for call in (7, 20, 318):
            match = np.flatnonzero(calls == call)
            require(match.size == 1, f"IR-IVM call {call} is not uniquely stored")
            selected[call] = np.ascontiguousarray(paths[int(match[0])])
            hmatch = np.flatnonzero(history_calls == call)
            require(hmatch.size == 1, f"IR-IVM call {call} history mismatch")
            call_to_nf[call] = int(history_nf[int(hmatch[0])])
        initial = np.ascontiguousarray(saved["initial_path"], dtype=np.float64)
    require(call_to_nf == {7: 210, 20: 429, 318: 8982}, "IR-IVM panel N_F drift")
    return initial, selected, call_to_nf


def resample_to_progress(path: np.ndarray, targets: np.ndarray) -> np.ndarray:
    band = np.asarray(path, dtype=np.float64)
    segment = np.linalg.norm(np.diff(band, axis=0), axis=1)
    require(np.all(segment > 0.0), "degenerate path segment")
    source = np.concatenate(([0.0], np.cumsum(segment)))
    source /= source[-1]
    index = np.clip(np.searchsorted(source, targets, side="right") - 1, 0, len(segment) - 1)
    fraction = (targets - source[index]) / (source[index + 1] - source[index])
    result = (1.0 - fraction[:, None]) * band[index] + fraction[:, None] * band[index + 1]
    result[0] = band[0]
    result[-1] = band[-1]
    return np.ascontiguousarray(result)


def project_on_legacy_section(path: np.ndarray, center_s, center_path, normal) -> np.ndarray:
    matched = resample_to_progress(path, center_s)
    signed = np.sum((matched - center_path) * normal, axis=1) / np.sqrt(float(N_ATOMS))
    signed[0] = 0.0
    signed[-1] = 0.0
    return np.column_stack((center_s, signed))


def plot_legacy_surface_evolution(output: Path) -> dict:
    initial, ir_by_call, call_to_nf = load_ir_paths()
    ode_by_nf = load_ode_paths()
    with np.load(LEGACY_SURFACE, allow_pickle=False) as saved:
        center_path = np.asarray(saved["center_path"], dtype=np.float64)
        center_s = np.asarray(saved["center_s"], dtype=np.float64)
        normal = np.asarray(saved["normal_directions"], dtype=np.float64)
        normal_grid = np.asarray(saved["normal_coordinate_rms_A"], dtype=np.float64)
        energy_eV = np.asarray(saved["energy_eV"], dtype=np.float64)
        bound = float(np.asarray(saved["bound_energy_eV"]).item())
        embedded_center_sha = str(np.asarray(saved["center_path_sha256"]).item())
        surface_kind = str(np.asarray(saved["surface_kind"]).item())
    require(center_path.shape == normal.shape == (59, N_ATOMS * 3), "legacy basis shape drift")
    require(center_s.shape == (59,) and normal_grid.shape == (28,), "legacy grid shape drift")
    require(energy_eV.shape == (59, 28), "legacy energy-grid shape drift")
    require(sha256_array(center_path) == embedded_center_sha, "legacy center-path hash mismatch")
    relative = np.maximum((energy_eV - bound) * EV_TO_KCAL, 0.0)
    capped = np.minimum(relative, 1.0e6)
    norm = LogQuadraticNorm(0.0, 1.0e6, scale=40.0, curvature=2.5)
    filled_levels = norm.inverse(np.linspace(0.0, 1.0, 41))
    line_levels = norm.inverse(np.linspace(0.0, 1.0, 20))
    smesh, nmesh = np.meshgrid(center_s, normal_grid, indexing="ij")
    panels = [
        ("Initial path", initial, None, None, None),
        (None, ir_by_call[7], ode_by_nf[240], 210, 240),
        (None, ir_by_call[20], ode_by_nf[480], 429, 480),
        (None, ir_by_call[318], ode_by_nf[9990], 8982, 9990),
    ]
    projections = []
    ymax = float(normal_grid[-1])
    for _, ir_path, ode_path, _, _ in panels:
        ir_projected = project_on_legacy_section(ir_path, center_s, center_path, normal)
        ode_projected = None if ode_path is None else project_on_legacy_section(ode_path, center_s, center_path, normal)
        projections.append((ir_projected, ode_projected))
        ymax = max(ymax, float(np.max(ir_projected[:, 1])))
        if ode_projected is not None:
            ymax = max(ymax, float(np.max(ode_projected[:, 1])))
    ylim_top = max(float(normal_grid[-1]) + 0.02, np.ceil((ymax + 0.015) / 0.05) * 0.05)
    fig, axes = plt.subplots(1, 4, figsize=(7.25, 2.55), sharex=True, sharey=True)
    contour = None
    for panel_index, (axis, panel, projected) in enumerate(zip(axes, panels, projections)):
        title, _, ode_path, ir_nf, ode_nf = panel
        ir_projected, ode_projected = projected
        axis.set_facecolor("#F4F4F4")
        contour = axis.contourf(
            smesh, nmesh, capped, levels=filled_levels, cmap="cividis", norm=norm,
            antialiased=True, extend="max",
        )
        axis.contour(
            smesh, nmesh, capped, levels=line_levels, colors="#F3F3F3",
            linewidths=0.32, alpha=0.58,
        )
        if ylim_top > float(normal_grid[-1]):
            axis.axhspan(
                float(normal_grid[-1]), ylim_top, facecolor="#F4F4F4",
                edgecolor="#C6C6C6", hatch="////", linewidth=0.0, zorder=1,
            )
        if ode_path is None:
            axis.plot(
                ir_projected[:, 0], ir_projected[:, 1], color=INITIAL_COLOR,
                linewidth=1.65, marker="o", markerfacecolor="white",
                markeredgewidth=0.75, markersize=2.4, markevery=4,
                label="Initial path", zorder=5,
            )
            axis.set_title(title, pad=5)
        else:
            axis.plot(
                ode_projected[:, 0], ode_projected[:, 1], color=ODE_COLOR,
                linewidth=1.45, marker="o", markersize=2.2, markevery=4,
                label="ODE-NEB", zorder=5,
            )
            axis.plot(
                ir_projected[:, 0], ir_projected[:, 1], color=IR_COLOR,
                linewidth=1.75, marker="s", markersize=2.2, markevery=4,
                label="IR-IVM", zorder=6,
            )
            axis.set_title(
                rf"IR-IVM $N_F={ir_nf}$" + "\n" + rf"ODE-NEB $N_F={ode_nf}$",
                pad=4,
            )
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(float(normal_grid[0]), ylim_top)
        axis.set_xticks((0.0, 0.5, 1.0))
        axis.tick_params(direction="out", length=2.8)
        axis.text(
            0.035, 0.965, chr(97 + panel_index), transform=axis.transAxes,
            va="top", ha="left", fontsize=8.5, fontweight="bold",
        )
    axes[0].set_ylabel(r"Signed path-normal $n$ ($\AA$)")
    axes[-1].legend(
        frameon=True, facecolor="white", edgecolor="#B3B3B3", framealpha=0.92,
        loc="upper right", handlelength=2.0,
    )
    if ylim_top > float(normal_grid[-1]):
        axes[0].text(
            0.98, float(normal_grid[-1]) + 0.008, "outside legacy section",
            transform=axes[0].get_yaxis_transform(), ha="right", va="bottom",
            fontsize=6.3, color="#666666",
        )
    fig.supxlabel(r"Path progress $s$", x=0.48, y=0.04)
    color_axis = fig.add_axes([0.906, 0.235, 0.014, 0.645])
    colorbar = fig.colorbar(contour, cax=color_axis, extend="max")
    colorbar.set_ticks((0.0, 40.0, 240.0, 1.0e3, 1.0e4, 1.0e6))
    colorbar.set_ticklabels(("0", "40", "240", r"$10^3$", r"$10^4$", r"$10^6$"))
    colorbar.set_label(r"$E_{\mathrm{0K}}-E_{\mathrm{bound}}$ (kcal mol$^{-1}$)")
    fig.subplots_adjust(left=0.075, right=0.885, bottom=0.23, top=0.88, wspace=0.08)
    save_atomic(fig, output, {
        "Title": "Barnase--Barstar IR-IVM evolution on the legacy 0 K section",
        "Subject": "Existing legacy surface reused without new OpenMM evaluations",
    })
    plt.close(fig)
    panel_records = []
    for panel, projected in zip(panels, projections):
        title, ir_path, ode_path, ir_nf, ode_nf = panel
        ir_projected, ode_projected = projected
        panel_records.append({
            "title": title,
            "IR_IVM_optimization_N_F": ir_nf,
            "ODE_NEB_optimization_N_F": ode_nf,
            "IR_IVM_path_sha256": sha256_array(ir_path),
            "ODE_NEB_path_sha256": None if ode_path is None else sha256_array(ode_path),
            "IR_IVM_projected_n_range_A": [float(np.min(ir_projected[:, 1])), float(np.max(ir_projected[:, 1]))],
            "ODE_NEB_projected_n_range_A": None if ode_projected is None else [float(np.min(ode_projected[:, 1])), float(np.max(ode_projected[:, 1]))],
        })
    return {
        "surface_kind": surface_kind,
        "surface_shape": [59, 28],
        "surface_s_range": [float(center_s[0]), float(center_s[-1])],
        "surface_n_range_A": [float(normal_grid[0]), float(normal_grid[-1])],
        "display_n_range_A": [float(normal_grid[0]), float(ylim_top)],
        "surface_energy_display_cap_kcal_per_mol": 1.0e6,
        "projection": "piecewise-linear arc-length match to the legacy 59-node center, followed by the stored signed normal projection divided by sqrt(3132)",
        "surface_recomputed": False,
        "panels": panel_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-output", type=Path, required=True)
    parser.add_argument("--evolution-output", type=Path, required=True)
    args = parser.parse_args()
    energy_output = args.energy_output.expanduser().resolve()
    evolution_output = args.evolution_output.expanduser().resolve()
    require(energy_output.suffix == evolution_output.suffix == ".pdf", "both outputs must be PDFs")
    require(not energy_output.exists() and not evolution_output.exists(), "refusing to overwrite an existing PDF")
    configure_style()
    energy_curves = plot_energy_landscape(energy_output)
    evolution = plot_legacy_surface_evolution(evolution_output)
    provenance = evolution_output.with_name("barnase_barstar_near_nf9990_preview_provenance.json")
    require(not provenance.exists(), "refusing to overwrite provenance")
    payload = {
        "schema": "bb-near-nf9990-energy-and-legacy-surface-evolution-preview-v1",
        "status": "PASS_EXISTING_DATA_ONLY_PREVIEW",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_scope": {
            "new_OpenMM_contexts": 0,
            "new_energy_or_force_calls": 0,
            "optimization_N_F_contribution": 0,
            "energy_landscape": "three existing uniformly arc-length-sampled 500-point energy rows",
            "evolution_surface": "exact reuse of the accepted legacy 59x28 unrelaxed Cartesian 0 K section",
            "legacy_surface_is_not_claimed_to_be_centered_on_new_IR_IVM": True,
            "strict_new_dense_EF_storage_protocol_satisfied_by_this_preview": False,
            "note": "This immediate preview predates the replacement joint-E/F dense datasets; it does not replace them.",
        },
        "bindings": {
            "plot_script": binding(Path(__file__)),
            "IR_IVM_arc500_energy": binding(IR_ENERGY),
            "ODE_NEB_arc500_energy": binding(ODE_ENERGY),
            "original_IVM_arc500_energy": binding(OLD_IVM_ENERGY),
            "IR_IVM_result": binding(IR_RESULT),
            "ODE_NEB_contract": binding(ODE_CONTRACT),
            "legacy_surface": binding(LEGACY_SURFACE),
            "legacy_reference_pdf": binding(LEGACY_REFERENCE_PDF),
        },
        "energy_curves": [
            {key: value for key, value in curve.items() if key not in {"energy_eV", "relative_kcal", "arc_length_A"}}
            for curve in energy_curves
        ],
        "evolution": evolution,
        "outputs": {
            "energy_landscape_pdf": binding(energy_output),
            "legacy_surface_evolution_pdf": binding(evolution_output),
        },
    }
    temporary = provenance.with_name(provenance.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, provenance)
    print(json.dumps({
        "status": payload["status"],
        "energy_output": str(energy_output),
        "evolution_output": str(evolution_output),
        "provenance": str(provenance),
    }, indent=2))


if __name__ == "__main__":
    main()
