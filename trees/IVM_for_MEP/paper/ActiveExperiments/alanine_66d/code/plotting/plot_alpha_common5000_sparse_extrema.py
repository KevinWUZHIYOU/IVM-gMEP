#!/usr/bin/env python3
"""Thin the official raw alpha trajectories without removing their oscillations.

The scientific input and charged-NF construction are identical to the official
v10 alpha figure.  Only the connected vertices are thinned for display: within
each logarithmic NF bin, the first point, last point, local minimum I_perp, and
local maximum I_perp are retained.  No smoothing or interpolation is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


REPO = Path(__file__).resolve().parents[5]
STUDY = REPO / "python/alanine_ivm_clean_four_level_20260801"
SOURCE_DIR = STUDY / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import plot_alpha_common5000_decades_retitle_v10 as official  # noqa: E402
from plot_alpha_common5000_decades_v9 import (  # noqa: E402
    ARCHIVE_PREFIX,
    COMMON_FORCE_CUTOFF,
    DECADE_LABELS,
    filter_decades,
)
from plot_alpha_common5000_v8 import (  # noqa: E402
    truncate_stage1,
    truncate_stage4,
    validate_common_cutoff,
)
from plot_alpha_full_budget_v6 import (  # noqa: E402
    TARGET_IPERP,
    alpha_legend,
    alpha_style,
    alpha_tag,
    configure_style,
    grid,
    load_stage1_minimum_force,
    load_stage4_minimum_force,
    marker_for,
    panel_label,
)


OUTPUT_DIR = REPO / "paper/ActiveExperiments/alanine_66d/figures/active_candidates"
STEM = "alanine66d_alpha_common5000_raw_trajectory_sparse_extrema"
OFFICIAL_PDF = (
    REPO
    / "paper/Figures/ala2_66d/official_20260802"
    / "alanine_alpha_cost_to_precision_exact_vs_inexact.pdf"
)
COMMON_INITIAL = STUDY / "inputs/common_initial_path_equidistant.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--log-nf-bins",
        type=int,
        default=24,
        help="Number of logarithmic NF bins used for extrema-preserving thinning.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records():
    stage1_all = truncate_stage1(
        load_stage1_minimum_force(
            STUDY,
            minimum_terminal_force=COMMON_FORCE_CUTOFF,
            minimum_macro_budget=1000,
        )
    )
    stage4_all = truncate_stage4(
        load_stage4_minimum_force(
            STUDY,
            minimum_terminal_force=COMMON_FORCE_CUTOFF,
            minimum_phase_b=600,
            archive_prefix=ARCHIVE_PREFIX,
        )
    )
    validate_common_cutoff(stage1_all, stage4_all)
    stage1 = filter_decades(stage1_all)
    stage4 = filter_decades(stage4_all)
    protocol_validation = official.validate_title_protocol(stage1, stage4)
    return stage1, stage4, protocol_validation


def sparse_extrema_indices(
    force: np.ndarray,
    iperp: np.ndarray,
    log_nf_bins: int,
) -> tuple[np.ndarray, dict[int, list[str]]]:
    """Keep endpoints and the local first/last/min/max in each log-NF bin."""
    force = np.asarray(force, dtype=float)
    iperp = np.asarray(iperp, dtype=float)
    if log_nf_bins < 1:
        raise ValueError("log_nf_bins must be positive")
    if (
        force.ndim != 1
        or iperp.ndim != 1
        or len(force) != len(iperp)
        or len(force) == 0
        or np.any(~np.isfinite(force))
        or np.any(~np.isfinite(iperp))
        or np.any(force <= 0)
        or np.any(iperp <= 0)
        or np.any(np.diff(force) < 0)
    ):
        raise RuntimeError("Invalid charged-NF/raw-I_perp trajectory")

    log_force = np.log10(force)
    edges = np.linspace(log_force[0], log_force[-1], log_nf_bins + 1)
    bin_index = np.searchsorted(edges, log_force, side="right") - 1
    bin_index = np.clip(bin_index, 0, log_nf_bins - 1)
    reasons: dict[int, set[str]] = {0: {"trajectory_first"}, len(force) - 1: {"trajectory_last"}}

    for current_bin in range(log_nf_bins):
        members = np.flatnonzero(bin_index == current_bin)
        if members.size == 0:
            continue
        selected = {
            "bin_first": int(members[0]),
            "bin_last": int(members[-1]),
            "bin_min_i_perp": int(members[np.argmin(iperp[members])]),
            "bin_max_i_perp": int(members[np.argmax(iperp[members])]),
        }
        for reason, index in selected.items():
            reasons.setdefault(index, set()).add(reason)

    indices = np.asarray(sorted(reasons), dtype=int)
    serialized_reasons = {
        index: sorted(reasons[index]) for index in indices.tolist()
    }
    if indices[0] != 0 or indices[-1] != len(force) - 1:
        raise RuntimeError("Trajectory endpoints were not retained")
    return indices, serialized_reasons


def plot_sparse_trajectory(axis, records, show_ylabel: bool, log_nf_bins: int):
    audit = {}
    csv_rows = []
    for item in records:
        force, iperp = official.fully_charged_raw_trajectory(item)
        indices, reasons = sparse_extrema_indices(force, iperp, log_nf_bins)
        sparse_force = force[indices]
        sparse_iperp = iperp[indices]
        style = alpha_style(item.alpha)
        marker = marker_for(item.alpha)
        axis.plot(
            sparse_iperp,
            sparse_force,
            marker=marker,
            markersize=3.0,
            markevery=max(1, len(indices) // 10),
            markeredgewidth=0.25,
            **style,
        )
        axis.scatter(
            sparse_iperp[-1],
            sparse_force[-1],
            marker="X",
            s=40,
            facecolor=style["color"],
            edgecolor="#202020",
            linewidth=0.55,
            zorder=style["zorder"] + 3,
        )

        selected = set(indices.tolist())
        carried_endpoint = bool(
            item.cutoff_carried_forward
            and int(force[-1]) == COMMON_FORCE_CUTOFF
        )
        for index, (nf, value) in enumerate(zip(force, iperp)):
            row_reasons = list(reasons.get(index, []))
            is_carried = carried_endpoint and index == len(force) - 1
            if is_carried:
                row_reasons.append("cutoff_carry_forward")
            csv_rows.append(
                {
                    "family": item.family,
                    "alpha": float(item.alpha),
                    "raw_index": index,
                    "charged_NF": int(nf),
                    "raw_i_perp": float(value),
                    "selected_for_plot": index in selected,
                    "selection_reason": ";".join(sorted(row_reasons)),
                    "is_cutoff_carry_forward": is_carried,
                    "raw_source": str(item.raw_source),
                    "archive_source": (
                        str(item.archive_source) if item.archive_source else ""
                    ),
                }
            )

        audit[f"{item.alpha:g}"] = {
            "complete_states_and_cutoff_endpoint": int(len(force)),
            "connected_vertices": int(len(indices)),
            "retained_fraction": float(len(indices) / len(force)),
            "raw_i_perp_increases": int(np.count_nonzero(np.diff(iperp) > 0)),
            "connected_i_perp_increases": int(
                np.count_nonzero(np.diff(sparse_iperp) > 0)
            ),
            "global_minimum_retained": bool(
                int(np.argmin(iperp)) in selected
            ),
            "global_maximum_retained": bool(
                int(np.argmax(iperp)) in selected
            ),
            "terminal_force": int(force[-1]),
            "terminal_raw_i_perp": float(iperp[-1]),
        }

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(1.2e3, 4.0e-6)
    axis.set_ylim(16, 6500)
    axis.set_xlabel(r"Raw $I_\perp$ (smaller is better)")
    axis.set_ylabel("Number of Forces" if show_ylabel else "")
    grid(axis)
    return audit, csv_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def file_manifest(stage1, stage4) -> dict:
    paths: dict[str, Path] = {
        "official_source_pdf": OFFICIAL_PDF,
        "common_initial": COMMON_INITIAL,
    }
    for family, records in (("stage1", stage1), ("stage4", stage4)):
        for item in records:
            tag = alpha_tag(item.alpha)
            paths[f"{family}_alpha_{tag}_raw"] = Path(item.raw_source)
            if item.archive_source is not None:
                paths[f"{family}_alpha_{tag}_archive"] = Path(item.archive_source)
            if family == "stage4":
                with np.load(item.raw_source, allow_pickle=False) as combined:
                    paths[f"{family}_alpha_{tag}_phase_a"] = Path(
                        str(official.scalar(combined, "phase_a_result"))
                    )
                    paths[f"{family}_alpha_{tag}_phase_b"] = Path(
                        str(official.scalar(combined, "phase_b_result"))
                    )
    return {
        key: {"path": str(path), "sha256": sha256(path)}
        for key, path in paths.items()
    }


def code_manifest() -> dict:
    paths = {
        "candidate_plotter": Path(__file__).resolve(),
        "official_v10_plotter": SOURCE_DIR / "plot_alpha_common5000_decades_retitle_v10.py",
        "decade_filter": SOURCE_DIR / "plot_alpha_common5000_decades_v9.py",
        "common_cutoff_loader": SOURCE_DIR / "plot_alpha_common5000_v8.py",
        "alpha_run_loader_and_style": SOURCE_DIR / "plot_alpha_full_budget_v6.py",
    }
    return {
        key: {"path": str(path), "sha256": sha256(path)}
        for key, path in paths.items()
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    stage1, stage4, protocol_validation = load_records()

    figure, axes = plt.subplots(1, 2, figsize=(9.3, 3.55), sharey=True)
    stage1_audit, stage1_rows = plot_sparse_trajectory(
        axes[0], stage1, True, args.log_nf_bins
    )
    stage4_audit, stage4_rows = plot_sparse_trajectory(
        axes[1], stage4, False, args.log_nf_bins
    )
    axes[0].set_title(r"Exact-like subminimization ($r=0.1$), no resampling")
    axes[1].set_title(
        r"Optimized inexact subminimization + resampling "
        r"($r=0.96\,\rightarrow\,0.8$)"
    )
    axes[1].axvline(TARGET_IPERP, color="#6a6a6a", ls=":", lw=0.9, zorder=1)
    axes[1].text(
        0.68,
        0.055,
        r"joint target $1.9\times10^{-4}$",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=6.9,
        color="#505050",
    )
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                ls="",
                marker="X",
                ms=5.2,
                markerfacecolor="#b0b0b0",
                markeredgecolor="#202020",
                markeredgewidth=0.5,
                label=r"common charged $N_F=5000$ cutoff",
            )
        ],
        frameon=False,
        loc="upper left",
        fontsize=6.7,
    )
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    alpha_legend(figure, 0.995, stage1, title=None)
    figure.subplots_adjust(
        left=0.075, right=0.99, bottom=0.17, top=0.76, wspace=0.12
    )

    pdf_path = args.output_dir / f"{STEM}.pdf"
    csv_path = args.output_dir / f"{STEM}_source.csv"
    manifest_path = args.output_dir / f"{STEM}_manifest.json"
    figure.savefig(
        pdf_path,
        bbox_inches="tight",
        metadata={
            "Title": "Alanine alpha raw cost trajectories with extrema-preserving thinning",
            "Subject": "Same raw trajectories as official v10; visual thinning only",
        },
    )
    plt.close(figure)
    rows = stage1_rows + stage4_rows
    write_csv(csv_path, rows)

    manifest = {
        "status": "active candidate; not cited by paper/main.tex",
        "official_source_preserved": str(OFFICIAL_PDF),
        "scientific_change": False,
        "visual_change": (
            "Connected vertices are thinned in logarithmic charged-NF bins; "
            "raw oscillations remain visible and no smoothing/interpolation is used."
        ),
        "thinning_rule": {
            "log_nf_bins": args.log_nf_bins,
            "retained_per_nonempty_bin": [
                "first",
                "last",
                "minimum_raw_i_perp",
                "maximum_raw_i_perp",
            ],
            "trajectory_endpoints_always_retained": True,
            "all_retained_vertices_are_observed_or_explicit_cutoff_carry_forward": True,
        },
        "common_charged_NF_cutoff": COMMON_FORCE_CUTOFF,
        "alpha_grid": list(DECADE_LABELS),
        "protocol_validation": protocol_validation,
        "trajectory_audit": {
            "stage1": stage1_audit,
            "stage4": stage4_audit,
        },
        "inputs": file_manifest(stage1, stage4),
        "plotting_code": code_manifest(),
        "source_csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
        "output_pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pdf": str(pdf_path), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
