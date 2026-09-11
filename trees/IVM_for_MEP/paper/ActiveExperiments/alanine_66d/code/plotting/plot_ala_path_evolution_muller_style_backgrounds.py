#!/usr/bin/env python3
"""Render background candidates for the official Ala path-evolution figure.

The scientific inputs and four checkpoint definitions are inherited from the
frozen Stage-IV v4 Pareto figure.  Only publication styling and the potential-
energy background are varied.  Outputs are candidates and never overwrite the
stable files under paper/Figures/ala2_66d/official_20260802/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np


REPO = Path(__file__).resolve().parents[5]
STUDY = REPO / "python/alanine_ivm_clean_four_level_20260801"
LEGACY = REPO / "python/alanine_v10_chignolin_v9_fixed_alpha_20260730"
REFERENCE_STYLE_PDF = (
    REPO
    / "paper/ActiveExperiments/muller_q10_10d/figures/active_candidates"
    / "muller_q10_article_method_comparison_n10_n20_2x3_linear3000.pdf"
)
OFFICIAL_SOURCE_PDF = (
    REPO
    / "paper/Figures/ala2_66d/official_20260802"
    / "alanine_all_methods_1x4_path_evolution.pdf"
)
DEFAULT_OUTPUT_DIR = (
    REPO
    / "paper/ActiveExperiments/alanine_66d/figures/active_candidates"
)
STEM = "alanine66d_stage4_v4_1x4_path_evolution_muller_style"
LIGHT_CONTOUR_CMAP = LinearSegmentedColormap.from_list(
    "ala_light_contours",
    ["#D2D4D3", "#E2E3E1", "#EEEEEB", "#F8F8F4", "#FFFFFF"],
)

BACKGROUNDS = {
    "gray": {
        "label": "Muller gray",
        "cmap": "Greys_r",
        "fill_alpha": 0.92,
        "line_color": "0.25",
        "line_alpha": 0.78,
    },
    "light": {
        "label": "Light contour",
        "cmap": LIGHT_CONTOUR_CMAP,
        "fill_alpha": 1.0,
        "line_color": "0.38",
        "line_alpha": 0.70,
    },
    "pink": {
        "label": "Ala warm pink",
        "cmap": "pink",
        "fill_alpha": 0.96,
        "line_color": "0.28",
        "line_alpha": 0.62,
    },
    "cividis": {
        "label": "Colorblind-safe cividis",
        "cmap": "cividis",
        "fill_alpha": 0.88,
        "line_color": "0.22",
        "line_alpha": 0.65,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--backgrounds",
        nargs="+",
        choices=tuple(BACKGROUNDS),
        default=tuple(BACKGROUNDS),
    )
    return parser.parse_args()


def mm_to_inch(value: float) -> float:
    return value / 25.4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, source: Path):
    specification = importlib.util.spec_from_file_location(name, source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load module from {source}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def configure_muller_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "it",
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "legend.frameon": False,
            "legend.handlelength": 2.2,
            "legend.handletextpad": 0.5,
            "legend.borderaxespad": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.dpi": 220,
            "savefig.dpi": 600,
            "savefig.bbox": None,
            "savefig.transparent": False,
        }
    )


def load_frozen_inputs():
    formal = load_module(
        "ala_stage4_v4_formal_plotter_for_style_candidates",
        STUDY / "src/plot_stage4_v4_pareto_all_methods.py",
    )
    legacy_module = formal.load_legacy_module(REPO)
    paths = {
        "combined": STUDY
        / "runs/stage4_inexact_optimized_resample"
        / "stage4_uniform_v3_a20_phaseA34_phaseB220.npz",
        "archive": STUDY
        / "runs/pareto_archive/pareto_clean_stage4_v4_a20_phaseA34_phaseB220.npz",
        "dense_prefix": STUDY
        / "runs/diagnostics/dense_barrier_clean_stage4_v4_a20_phaseA34_preactivation.npz",
        "dense_screen": STUDY
        / "runs/diagnostics/dense_barrier_clean_stage4_pareto_screen_a20_phaseA34_phaseB220.npz",
        "common_initial": STUDY / "inputs/common_initial_path_equidistant.npz",
        "ode": LEGACY
        / "formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
        "ode_metrics": LEGACY
        / "formal_figure_inputs/metrics_ala_0to1_N22_k50_alpha0.0001_ode_results.npz",
        "dense_ode": LEGACY
        / "runs/diagnostics/dense_barrier_ala_ode_neb_k50_logsample_dense.npz",
        "rama": LEGACY / "formal_figure_inputs/ala2_scan_cache.pkl",
    }
    for path in (*paths.values(), REFERENCE_STYLE_PDF, OFFICIAL_SOURCE_PDF):
        if not path.exists():
            raise FileNotFoundError(path)

    common_initial = np.asarray(
        np.load(paths["common_initial"], allow_pickle=False)["initial_path"],
        dtype=float,
    )
    ivm, _ = formal.load_pareto_ivm(
        legacy_module,
        paths["combined"],
        paths["archive"],
        paths["dense_prefix"],
        paths["dense_screen"],
        common_initial,
    )
    ode = legacy_module.load_ode(
        paths["ode"], paths["ode_metrics"], paths["dense_ode"], common_initial
    )
    if not np.allclose(ivm.initial_path, ode.initial_path, atol=5e-7, rtol=0.0):
        raise RuntimeError("IVM and ODE-NEB do not share the frozen IDPP path")
    with paths["rama"].open("rb") as handle:
        rama = legacy_module._ContourCacheUnpickler(handle).load()
    return legacy_module, ivm, ode, rama, paths


def path_at_or_before(method, target: int | None) -> tuple[np.ndarray, int]:
    if target is None:
        return np.asarray(method.selected_path, dtype=float), int(method.selected_force)
    forces = np.asarray(method.path_force, dtype=int)
    eligible = np.flatnonzero(forces <= int(target))
    if not eligible.size:
        return np.asarray(method.initial_path, dtype=float), 0
    index = int(eligible[-1])
    return np.asarray(method.paths[index], dtype=float), int(forces[index])


def build_panels(module, ivm, ode):
    definitions = (
        ("Common IDPP initialization", 0),
        (r"$N_F\leq10^3$", 1000),
        (r"$N_F\leq10^4$", 10000),
        ("Selected checkpoints", None),
    )
    panels = []
    for title, target in definitions:
        entries = []
        if target == 0:
            entries.append(
                {
                    "method": "Initial (IDPP)",
                    "force": 0,
                    "path": np.asarray(ivm.initial_path, dtype=float),
                }
            )
        else:
            for method in (ivm, ode):
                path, force = path_at_or_before(method, target)
                entries.append(
                    {"method": method.short_label, "force": force, "path": path}
                )
        panels.append({"title": title, "target": target, "entries": entries})
    return panels


def path_style(method: str) -> dict:
    if method == "IVM":
        return {
            "color": "#0072B2",
            "linestyle": "-",
            "linewidth": 1.85,
            "marker": "o",
            "markersize": 3.1,
            "zorder": 25,
        }
    if method == "ODE--NEB":
        return {
            "color": "#D55E00",
            "linestyle": "--",
            "linewidth": 1.65,
            "marker": "s",
            "markersize": 2.9,
            "zorder": 24,
        }
    return {
        "color": "#777777",
        "linestyle": ":",
        "linewidth": 1.55,
        "marker": "D",
        "markersize": 2.7,
        "zorder": 23,
    }


def draw_candidate(
    module,
    rama,
    panels,
    background_key: str,
    output: Path,
) -> None:
    background = BACKGROUNDS[background_key]
    phi_grid, psi_grid = np.meshgrid(
        np.asarray(rama["phi_g"], dtype=float),
        np.asarray(rama["psi_g"], dtype=float),
        indexing="ij",
    )
    energy_grid = (
        np.asarray(rama["PES"], dtype=float)
        - float(np.nanmin(np.asarray(rama["PES"], dtype=float)))
    ) * float(module.EV_TO_KCAL)
    fill_levels = np.linspace(0.0, 15.0, 31)
    line_levels = np.linspace(0.0, 15.0, 9)

    figure = plt.figure(figsize=(mm_to_inch(180), mm_to_inch(58)), dpi=220)
    grid = GridSpec(
        1,
        5,
        figure=figure,
        width_ratios=(1.0, 1.0, 1.0, 1.0, 0.045),
        left=0.055,
        right=0.925,
        bottom=0.19,
        top=0.77,
        wspace=0.24,
    )
    axes = [figure.add_subplot(grid[0, column]) for column in range(4)]
    colorbar_axis = figure.add_subplot(grid[0, 4])
    contour = None

    for panel_index, (axis, panel) in enumerate(zip(axes, panels)):
        contour = axis.contourf(
            phi_grid,
            psi_grid,
            energy_grid,
            levels=fill_levels,
            cmap=background["cmap"],
            alpha=background["fill_alpha"],
            extend="max",
            zorder=-10,
        )
        axis.contour(
            phi_grid,
            psi_grid,
            energy_grid,
            levels=line_levels,
            colors=background["line_color"],
            linewidths=0.35,
            alpha=background["line_alpha"],
            zorder=-5,
        )
        for entry in reversed(panel["entries"]):
            phi, psi = module.path_to_dihedrals(entry["path"])
            style = path_style(entry["method"])
            axis.plot(
                phi,
                psi,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor="white",
                markeredgewidth=0.45,
                markevery=1,
                zorder=style["zorder"],
            )

        initial_phi, initial_psi = module.path_to_dihedrals(
            panels[0]["entries"][0]["path"]
        )
        axis.scatter(
            initial_phi[0],
            initial_psi[0],
            s=30,
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=80,
        )
        axis.scatter(
            initial_phi[-1],
            initial_psi[-1],
            s=58,
            marker="*",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=85,
        )
        axis.set_xlim(-180, 0)
        axis.set_ylim(0, 180)
        axis.set_xticks((-180, -120, -60, 0))
        axis.set_yticks((0, 60, 120, 180))
        axis.set_xlabel(r"$\phi$ (deg)")
        if panel_index == 0:
            axis.set_ylabel(r"$\psi$ (deg)")
        else:
            axis.set_ylabel("")
        axis.set_title(panel["title"], pad=5.0)
        axis.set_box_aspect(1.0)

    if contour is None:
        raise RuntimeError("No contour artist was created")
    colorbar = figure.colorbar(contour, cax=colorbar_axis, extend="max")
    colorbar.set_ticks((0, 3, 6, 9, 12, 15))
    colorbar.ax.tick_params(labelsize=6.2, width=0.65, length=2.6)
    colorbar.outline.set_linewidth(0.75)
    colorbar.set_label(
        r"$\Delta E$ (kcal mol$^{-1}$)", fontsize=6.8, labelpad=2.5
    )

    figure.canvas.draw()
    for axis, letter in zip(axes, "abcd"):
        position = axis.get_position()
        figure.text(
            position.x0 - 0.026,
            position.y1 + 0.012,
            letter,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )

    handles = []
    for method in ("Initial (IDPP)", "IVM", "ODE--NEB"):
        style = path_style(method)
        label = method if method != "IVM" else r"IVM ($\alpha=20$)"
        if method == "ODE--NEB":
            label = r"ODE--NEB ($k=50$)"
        handles.append(
            Line2D(
                [0],
                [0],
                label=label,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor="white",
                markeredgewidth=0.45,
            )
        )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.49, 0.985),
        ncol=3,
        columnspacing=1.6,
        handlelength=2.5,
        handletextpad=0.55,
        frameon=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="pdf",
        dpi=600,
        metadata={
            "Title": f"Alanine 66D path evolution - {background['label']}",
            "Subject": "Muller-style visual candidate; frozen Stage-IV v4 data",
        },
    )
    plt.close(figure)


def source_rows(module, panels, paths: dict[str, Path]):
    rows = []
    for panel_index, panel in enumerate(panels):
        for entry in panel["entries"]:
            phi, psi = module.path_to_dihedrals(entry["path"])
            source = paths["common_initial"]
            if entry["method"] == "IVM":
                source = paths["archive"]
            elif entry["method"] == "ODE--NEB":
                source = paths["ode"]
            for image_index, (phi_value, psi_value) in enumerate(zip(phi, psi)):
                rows.append(
                    {
                        "panel": chr(ord("a") + panel_index),
                        "panel_title": panel["title"],
                        "target_NF": "selected" if panel["target"] is None else panel["target"],
                        "method": entry["method"],
                        "actual_NF": entry["force"],
                        "image_index": image_index,
                        "phi_deg": float(phi_value),
                        "psi_deg": float(psi_value),
                        "source_file": str(source),
                    }
                )
    return rows


def write_source_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    configure_muller_style()
    module, ivm, ode, rama, paths = load_frozen_inputs()
    panels = build_panels(module, ivm, ode)
    rows = source_rows(module, panels, paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_hashes = {
        key: {"path": str(path), "sha256": sha256(path)}
        for key, path in paths.items()
    }
    input_hashes["reference_style_pdf"] = {
        "path": str(REFERENCE_STYLE_PDF),
        "sha256": sha256(REFERENCE_STYLE_PDF),
    }
    input_hashes["official_source_pdf"] = {
        "path": str(OFFICIAL_SOURCE_PDF),
        "sha256": sha256(OFFICIAL_SOURCE_PDF),
    }

    outputs = []
    for background_key in args.backgrounds:
        stem = f"{STEM}_bg_{background_key}"
        pdf_path = args.output_dir / f"{stem}.pdf"
        csv_path = args.output_dir / f"{stem}_source.csv"
        manifest_path = args.output_dir / f"{stem}_manifest.json"
        draw_candidate(module, rama, panels, background_key, pdf_path)
        write_source_csv(csv_path, rows)
        background_manifest = {
            key: value
            for key, value in BACKGROUNDS[background_key].items()
            if key != "cmap"
        }
        background_manifest.update(
            {
                "key": background_key,
                "cmap": (
                    BACKGROUNDS[background_key]["cmap"]
                    if isinstance(BACKGROUNDS[background_key]["cmap"], str)
                    else BACKGROUNDS[background_key]["cmap"].name
                ),
            }
        )
        manifest = {
            "status": "active candidate; not cited by paper/main.tex",
            "scientific_lineage": "Alanine 66D clean Stage-IV v4 Pareto, integer alpha=20",
            "change_scope": "style and potential-energy background only",
            "layout": "1x4 at 180 mm publication width",
            "style_reference": str(REFERENCE_STYLE_PDF),
            "background": background_manifest,
            "energy_background": {
                "range_kcal_mol": [0.0, 15.0],
                "filled_levels": 31,
                "line_levels": 9,
                "normalization": "PES minus its finite minimum, then eV to kcal/mol",
            },
            "method_style": {
                "IVM": path_style("IVM"),
                "ODE--NEB": path_style("ODE--NEB"),
                "Initial (IDPP)": path_style("Initial (IDPP)"),
            },
            "panels": [
                {
                    "letter": chr(ord("a") + index),
                    "title": panel["title"],
                    "target_NF": panel["target"],
                    "actual_NF": {
                        entry["method"]: entry["force"] for entry in panel["entries"]
                    },
                }
                for index, panel in enumerate(panels)
            ],
            "inputs": input_hashes,
            "source_csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
            "output_pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
            "plotter": str(Path(__file__).resolve()),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
        outputs.append(str(pdf_path))
    print(json.dumps({"outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
