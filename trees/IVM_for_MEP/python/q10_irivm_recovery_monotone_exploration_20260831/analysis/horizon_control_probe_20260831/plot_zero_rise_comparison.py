#!/usr/bin/env python3
"""Plot the audited N20 recovery-rise removal against Article IVM.

The plot is intentionally generated from the independently scored public
trajectory archives.  It does not rerun either optimizer and it refuses to
draw unless the registered checkpoint and rise-count claims are present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from python.q10_irivm_unified_cycle_frozen_reproduction_v2_20260831.src.final_audit_core import (
    REFERENCE_BARRIER,
    load_and_audit_article,
)


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUN_ROOT = (
    REPOSITORY_ROOT
    / "python/q10_irivm_recovery_monotone_exploration_20260831/runs"
)
DATA_ROOT = REPOSITORY_ROOT / "paper/ActiveExperiments/muller_q10_10d/data"

DEFAULT_OLD = RUN_ROOT / "exact_envelope_n20_a0p001_r0p3_m8/trajectory.npz"
DEFAULT_OLD_SUMMARY = (
    RUN_ROOT / "exact_envelope_n20_a0p001_r0p3_m8/summary.json"
)
DEFAULT_NEW = RUN_ROOT / (
    "horizon_control_probe_n20_a0p001_r0p3_m8_b3000_root_audit/trajectory.npz"
)
DEFAULT_NEW_SUMMARY = RUN_ROOT / (
    "horizon_control_probe_n20_a0p001_r0p3_m8_b3000_root_audit/summary.json"
)
DEFAULT_PDF = REPOSITORY_ROOT / "output/pdf/q10_irivm_zero_rise_n20_comparison.pdf"
DEFAULT_PNG = REPOSITORY_ROOT / "output/pdf/q10_irivm_zero_rise_n20_comparison.png"
DEFAULT_CSV = HERE / "q10_irivm_zero_rise_n20_source.csv"
DEFAULT_MANIFEST = HERE / "q10_irivm_zero_rise_n20_figure_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def verify_inputs(
    old: dict[str, np.ndarray],
    new: dict[str, np.ndarray],
    old_summary: dict,
    new_summary: dict,
) -> None:
    if old_summary.get("i_perp_rise_count") != 1:
        raise RuntimeError("the previous IR-IVM trajectory must contain one rise")
    rise_events = old_summary.get("rise_events", [])
    if len(rise_events) != 1 or int(rise_events[0]["nof"]) != 80:
        raise RuntimeError("the registered previous rise is not the NoF 80 event")
    expected_new = {
        "i_perp_rise_count": 0,
        "strict_i_perp_count": 98,
        "strict_barrier_count": 98,
        "strict_joint_count": 98,
        "checkpoint_count": 98,
        "terminal_nof": 2363,
        "nof_per_image_sum": 2363,
    }
    for key, expected in expected_new.items():
        if new_summary.get(key) != expected:
            raise RuntimeError(
                f"new trajectory summary gate failed for {key}: "
                f"{new_summary.get(key)!r} != {expected!r}"
            )
    if not np.array_equal(old["public_nf"], new["public_nf"]):
        raise RuntimeError("old and new public NoF arrays differ")
    after_100 = old["public_nf"] >= 100
    for field in ("public_path", "public_i_perp", "public_delta_e"):
        if not np.array_equal(old[field][after_100], new[field][after_100]):
            raise RuntimeError(f"old and new {field} differ from NoF 100 onward")
    if np.any(np.diff(new["public_i_perp"]) > 0.0):
        raise RuntimeError("new public I_perp trajectory is not monotone")


def write_source_csv(
    path: Path,
    article_nof: np.ndarray,
    article_i: np.ndarray,
    article_delta: np.ndarray,
    old: dict[str, np.ndarray],
    new: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "method",
        "nof",
        "phase",
        "i_perp",
        "densified_delta_e",
        "densified_barrier_error",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, nof, phase, i_perp, delta_e in (
            (
                "Article IVM",
                article_nof,
                np.full(article_nof.shape, "article_public_state", dtype="U32"),
                article_i,
                article_delta,
            ),
            (
                "IR-IVM (previous recovery)",
                old["public_nf"],
                old["public_phase"],
                old["public_i_perp"],
                old["public_delta_e"],
            ),
            (
                "IR-IVM (zero-rise recovery)",
                new["public_nf"],
                new["public_phase"],
                new["public_i_perp"],
                new["public_delta_e"],
            ),
        ):
            for index in range(nof.size):
                writer.writerow(
                    {
                        "method": method,
                        "nof": int(nof[index]),
                        "phase": str(phase[index]),
                        "i_perp": f"{float(i_perp[index]):.17g}",
                        "densified_delta_e": f"{float(delta_e[index]):.17g}",
                        "densified_barrier_error": (
                            f"{abs(float(delta_e[index]) - REFERENCE_BARRIER):.17g}"
                        ),
                    }
                )


def draw(
    output_pdf: Path,
    output_png: Path,
    article_nof: np.ndarray,
    article_i: np.ndarray,
    article_delta: np.ndarray,
    old: dict[str, np.ndarray],
    new: dict[str, np.ndarray],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.0,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    colors = {
        "Article IVM": "#444444",
        "IR-IVM (previous recovery)": "#4C72B0",
        "IR-IVM (zero-rise recovery)": "#C44E52",
    }
    styles = {
        "Article IVM": "-",
        "IR-IVM (previous recovery)": "--",
        "IR-IVM (zero-rise recovery)": "-",
    }
    widths = {
        "Article IVM": 1.6,
        "IR-IVM (previous recovery)": 1.65,
        "IR-IVM (zero-rise recovery)": 2.0,
    }
    display_labels = {
        "Article IVM": "Article IVM",
        "IR-IVM (previous recovery)": "Previous IR-IVM",
        "IR-IVM (zero-rise recovery)": "IR-IVM",
    }
    series = {
        "Article IVM": (
            article_nof,
            article_i,
            np.abs(article_delta - REFERENCE_BARRIER),
        ),
        "IR-IVM (previous recovery)": (
            old["public_nf"],
            old["public_i_perp"],
            np.abs(old["public_delta_e"] - REFERENCE_BARRIER),
        ),
        "IR-IVM (zero-rise recovery)": (
            new["public_nf"],
            new["public_i_perp"],
            np.abs(new["public_delta_e"] - REFERENCE_BARRIER),
        ),
    }
    terminal = int(article_nof[-1])
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    metrics = (1, 2)
    ylabels = (
        r"$I_\perp$",
        r"$|\Delta E_{\mathrm{linear500}}-\Delta E_{\mathrm{ref}}|$",
    )
    panel_titles = (
        r"(a) Path-integrated perpendicular force",
        r"(b) Densified energy-barrier error",
    )

    for axis, metric, ylabel, panel_title in zip(
        axes, metrics, ylabels, panel_titles
    ):
        for method, (nof, i_perp, barrier_error) in series.items():
            values = i_perp if metric == 1 else barrier_error
            if np.any(values <= 0.0) or not np.all(np.isfinite(values)):
                raise RuntimeError(f"{method} has a non-positive log-scale metric")
            axis.step(
                nof,
                values,
                where="post",
                color=colors[method],
                linestyle=styles[method],
                linewidth=widths[method],
                label=display_labels[method],
                zorder=3 if "zero-rise" in method else 2,
            )
            if int(nof[-1]) < terminal:
                axis.plot(
                    [int(nof[-1]), terminal],
                    [float(values[-1]), float(values[-1])],
                    color=colors[method],
                    linestyle=":",
                    linewidth=1.4,
                    zorder=2,
                )
        axis.set_yscale("log")
        axis.set_xlim(0, terminal)
        axis.set_xlabel("Number of Forces")
        axis.set_ylabel(ylabel)
        axis.set_title(panel_title, loc="left", pad=7)
        axis.grid(True, which="both", color="#B0B0B0", alpha=0.25, linewidth=0.55)
        axis.axvline(80, color="#999999", linewidth=0.8, alpha=0.65, zorder=1)

        inset = axis.inset_axes([0.43, 0.53, 0.53, 0.42])
        for method, (nof, i_perp, barrier_error) in series.items():
            values = i_perp if metric == 1 else barrier_error
            early = nof <= 140
            inset.step(
                nof[early],
                values[early],
                where="post",
                color=colors[method],
                linestyle=styles[method],
                linewidth=1.35 if "zero-rise" not in method else 1.65,
                zorder=3 if "zero-rise" in method else 2,
            )
            point = np.flatnonzero(nof == 80)
            if point.size:
                inset.scatter(
                    [80],
                    [float(values[int(point[0])])],
                    s=25,
                    facecolor="white",
                    edgecolor=colors[method],
                    linewidth=1.2,
                    zorder=5,
                )
        inset.set_xlim(20, 140)
        if metric == 1:
            inset.set_ylim(0, 30000)
            old_80 = float(
                old["public_i_perp"][np.flatnonzero(old["public_nf"] == 80)[0]]
            )
            new_80 = float(
                new["public_i_perp"][np.flatnonzero(new["public_nf"] == 80)[0]]
            )
            inset.annotate(
                "",
                xy=(84, new_80),
                xytext=(84, old_80),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": colors["IR-IVM (zero-rise recovery)"],
                    "linewidth": 1.1,
                },
                zorder=6,
            )
            inset.text(
                0.04,
                0.06,
                r"NoF 80: 25,360 $\rightarrow$ 10,529",
                transform=inset.transAxes,
                fontsize=8.0,
                color=colors["IR-IVM (zero-rise recovery)"],
            )
        else:
            inset.set_ylim(0, 55)
            inset.text(
                0.04,
                0.06,
                r"NoF 80: 33.97 $\rightarrow$ 18.76",
                transform=inset.transAxes,
                fontsize=8.0,
                color=colors["IR-IVM (zero-rise recovery)"],
            )
        inset.axvline(80, color="#999999", linewidth=0.7, alpha=0.7)
        inset.grid(True, alpha=0.22, linewidth=0.45)
        inset.tick_params(labelsize=7.5)
        inset.set_title("Early recovery", fontsize=8.4, pad=2)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.015),
    )
    figure.suptitle(
        r"QM10, $N_{\mathrm{int}}=20$: removing the recovery-induced $I_\perp$ rise",
        y=0.955,
        fontsize=11.3,
    )
    figure.text(
        0.5,
        0.012,
        "From NoF 100 onward, the numerical states are bitwise identical. "
        "Horizon-first IR-IVM has zero rises and remains strictly better than Article IVM at all 98 checkpoints.",
        ha="center",
        va="bottom",
        fontsize=8.6,
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.955), w_pad=2.2)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, format="pdf", bbox_inches="tight")
    figure.savefig(output_png, format="png", dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, default=DEFAULT_OLD)
    parser.add_argument("--old-summary", type=Path, default=DEFAULT_OLD_SUMMARY)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--new-summary", type=Path, default=DEFAULT_NEW_SUMMARY)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    arguments = parser.parse_args()

    old = load_trajectory(arguments.old)
    new = load_trajectory(arguments.new)
    old_summary = load_summary(arguments.old_summary)
    new_summary = load_summary(arguments.new_summary)
    verify_inputs(old, new, old_summary, new_summary)

    with np.load(DATA_ROOT / "inputs/Q_data.npz", allow_pickle=False) as archive:
        q = np.asarray(archive["Q"], dtype=np.float64)
    article_path = DATA_ROOT / "frozen_runs/tuned_ivm_alpha1e3_selected.npz"
    article = load_and_audit_article(article_path, q, 20)
    draw(
        arguments.pdf,
        arguments.png,
        article.nof,
        article.i_perp,
        article.delta_e,
        old,
        new,
    )
    write_source_csv(
        arguments.csv,
        article.nof,
        article.i_perp,
        article.delta_e,
        old,
        new,
    )
    manifest = {
        "schema_version": 1,
        "status": "awaiting_visual_qa",
        "question": "Eliminate the public I_perp rise without weakening the registered checkpoints",
        "n_internal": 20,
        "controls": {"alpha": "1e-3", "r": 0.3, "m_max": 8},
        "reference_barrier": REFERENCE_BARRIER,
        "inputs": {
            str(arguments.old): sha256_file(arguments.old),
            str(arguments.old_summary): sha256_file(arguments.old_summary),
            str(arguments.new): sha256_file(arguments.new),
            str(arguments.new_summary): sha256_file(arguments.new_summary),
            str(article_path): sha256_file(article_path),
        },
        "outputs": {
            str(arguments.pdf): sha256_file(arguments.pdf),
            str(arguments.png): sha256_file(arguments.png),
            str(arguments.csv): sha256_file(arguments.csv),
        },
        "plot_source_sha256": sha256_file(Path(__file__)),
        "verified_claims": {
            "previous_i_perp_rises": 1,
            "zero_rise_i_perp_rises": 0,
            "zero_rise_joint_strict_checkpoints": "98/98",
            "bitwise_equal_from_nof": 100,
        },
    }
    arguments.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
