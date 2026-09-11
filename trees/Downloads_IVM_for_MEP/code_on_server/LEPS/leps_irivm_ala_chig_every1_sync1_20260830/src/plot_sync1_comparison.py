#!/usr/bin/env python3
"""Compare LEPS article IVM and the three recovery schedules at equal NoF."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


THIS_FILE = Path(__file__).resolve()
CAMPAIGN_ROOT = THIS_FILE.parents[1]
LEPS_ROOT = THIS_FILE.parents[2]
METRIC_SOURCE = LEPS_ROOT / "leps_irivm_ala_chig_20260829/src/audit_and_select.py"
spec = importlib.util.spec_from_file_location("_sync1_comparison_metrics", METRIC_SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(METRIC_SOURCE)
metric = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = metric
spec.loader.exec_module(metric)

ORIGINAL_RUN_ID = "756bf4584a55e959"  # alpha=100, r=1e-2, m_max=4
SYNC1_BEST_RUN_ID = "356307bb5a618535"  # alpha=100, r=1e-3, m_max=5
PERIOD5 = (
    LEPS_ROOT
    / "leps_irivm_ala_chig_lepsnative_20260829/runs"
    / f"{ORIGINAL_RUN_ID}.npz"
)
PERIOD1_REPEATED = (
    LEPS_ROOT
    / "leps_irivm_ala_chig_every1_20260830/runs"
    / f"{ORIGINAL_RUN_ID}.npz"
)
PERIOD1_SYNC = CAMPAIGN_ROOT / "runs" / f"{SYNC1_BEST_RUN_ID}.npz"
OUTDIR = CAMPAIGN_ROOT / "reports/figure"


def candidate_trajectory(path: Path):
    with np.load(path, allow_pickle=False) as saved:
        nof = np.asarray(saved["article_nf"], dtype=np.int64)
        paths = np.asarray(saved["path_history"], dtype=float)
    return nof, paths


def evaluated(nof, paths, reference, end=3000):
    i_perp = np.asarray([metric.i_perp(path) for path in paths])
    delta_e = np.asarray(
        [abs(metric.dense_barrier(path) - reference) for path in paths]
    )
    if int(nof[-1]) < int(end):
        nof = np.append(nof, int(end))
        i_perp = np.append(i_perp, i_perp[-1])
        delta_e = np.append(delta_e, delta_e[-1])
    return nof, i_perp, delta_e


def main():
    reference = float(metric.reference_saddle()["forward_barrier"])
    article_nof, article_paths = metric.distinct_article_trajectory()
    p5_nof, p5_paths = candidate_trajectory(PERIOD5)
    repeated_nof, repeated_paths = candidate_trajectory(PERIOD1_REPEATED)
    sync_nof, sync_paths = candidate_trajectory(PERIOD1_SYNC)
    series = [
        (
            "Article IVM",
            *evaluated(article_nof, article_paths, reference),
            "#222222",
            "o",
            "-",
        ),
        (
            "IR-IVM: recovery every 5 outer iterations (21/21)",
            *evaluated(p5_nof, p5_paths, reference),
            "#1f77b4",
            "s",
            "-",
        ),
        (
            "IR-IVM: repeated recovery every outer iteration (1/21)",
            *evaluated(repeated_nof, repeated_paths, reference),
            "#d62728",
            "^",
            "--",
        ),
        (
            "IR-IVM: single synchronized recovery every outer iteration (19/21)",
            *evaluated(sync_nof, sync_paths, reference),
            "#2ca02c",
            "D",
            "-",
        ),
    ]

    plt.rcParams.update(
        {
            "font.size": 9.6,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 7.8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.1, 4.2), constrained_layout=True)
    for label, nof, i_perp, delta_e, color, marker, linestyle in series:
        axes[0].step(
            nof, i_perp, where="post", color=color, lw=1.8,
            ls=linestyle, label=label,
        )
        axes[0].plot(nof[:-1], i_perp[:-1], marker, color=color, ms=3.5, ls="none")
        axes[1].step(
            nof, delta_e, where="post", color=color, lw=1.8,
            ls=linestyle, label=label,
        )
        axes[1].plot(nof[:-1], delta_e[:-1], marker, color=color, ms=3.5, ls="none")

    # These are the only two equal-NoF checks missed by the best synchronized run.
    for failed_nf in (150, 278):
        axes[1].axvline(failed_nf, color="#b26b00", lw=0.8, ls=":", alpha=0.8)
    axes[1].text(
        305,
        1.65,
        "Only failed checks:\nNoF 150 and 278\n(densified barrier)",
        color="#8a5200",
        fontsize=8.2,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#d5a253", "alpha": 0.9},
    )

    axes[0].set_title("Path convergence")
    axes[0].set_ylabel(r"$I_\perp$")
    axes[1].set_title("Densified barrier accuracy")
    axes[1].set_ylabel(r"$|\Delta E_{500}-\Delta E_{\rm ref}|$ (eV)")
    for axis in axes:
        axis.set_xlabel("Number of Forces")
        axis.set_xlim(0, 3000)
        axis.set_yscale("log")
        axis.grid(True, which="both", color="#d9d9d9", lw=0.6, alpha=0.75)
    axes[0].legend(loc="lower left", frameon=True)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTDIR / "leps_single_synchronized_recovery_comparison.pdf"
    png = OUTDIR / "leps_single_synchronized_recovery_comparison.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()
