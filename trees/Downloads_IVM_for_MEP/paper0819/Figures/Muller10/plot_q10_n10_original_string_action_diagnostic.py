#!/usr/bin/env python3
"""Plot only the N_int=10 Original String geometric-action trajectory.

The left panel reproduces the complete 0-3000 NoF trace used in panel c of
the formal comparison.  The right panel magnifies the final 300 NoF so the
single alternating polyline can be inspected checkpoint by checkpoint.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
REPOSITORY = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP")
SOURCE_CSV = (
    HERE
    / "muller_q10_article_method_comparison_n10_n20_2x4_geometric_action_linear3000_source.csv"
)
SOURCE_NPZ = Path(
    "/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/"
    "muller_q10_10d/image12/data/frozen_runs/original_string_selected.npz"
)
OUTPUT_PDF = (
    REPOSITORY
    / "output/pdf/q10_n10_original_string_geometric_action_diagnostic.pdf"
)
QA_PNG = REPOSITORY / "tmp/pdfs/q10_n10_original_string_geometric_action_diagnostic.png"
ACTION_COLUMN = "Geometric_Action_discrete_trapezoidal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_trace() -> tuple[np.ndarray, np.ndarray, dict]:
    rows = []
    with SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["case"] == "10 interior images"
                and row["method"] == "Original String"
                and float(row["number_of_forces"]) <= 3000.0
            ):
                rows.append(row)

    rows.sort(key=lambda row: float(row["number_of_forces"]))
    nof = np.asarray([float(row["number_of_forces"]) for row in rows])
    action = np.asarray([float(row[ACTION_COLUMN]) for row in rows])
    expected_nof = np.arange(10.0, 3000.0 + 10.0, 10.0)
    np.testing.assert_array_equal(nof, expected_nof)
    if len(np.unique(nof)) != len(nof):
        raise RuntimeError("duplicate Number of Forces values in Original String trace")
    if not np.all(np.isfinite(action)):
        raise RuntimeError("non-finite geometric action value")

    source_files = {row["source_file"] for row in rows}
    source_hashes = {row["source_sha256"] for row in rows}
    if source_files != {str(SOURCE_NPZ)}:
        raise RuntimeError(f"unexpected source files: {source_files}")
    if source_hashes != {sha256(SOURCE_NPZ)}:
        raise RuntimeError("source NPZ hash does not match registered CSV")

    with np.load(SOURCE_NPZ, allow_pickle=True) as archive:
        saved_nof = np.asarray(archive["total_force_evaluations"], dtype=float)
        np.testing.assert_array_equal(saved_nof[: len(nof)], nof)
        if np.asarray(archive["path_history"]).shape[:2] != (1000, 12):
            raise RuntimeError("unexpected Original String path_history shape")
        config = json.loads(str(archive["config_json"].item()))
    if config != {"dt": 0.0005, "reparam_interval": 2}:
        raise RuntimeError(f"unexpected Original String configuration: {config}")

    audit = {
        "checkpoint_count": int(len(nof)),
        "unique_nof_count": int(len(np.unique(nof))),
        "first_nof": float(nof[0]),
        "last_nof": float(nof[-1]),
        "source_npz": str(SOURCE_NPZ),
        "source_npz_sha256": sha256(SOURCE_NPZ),
        "configuration": config,
        "late_values": {
            str(int(x)): float(y)
            for x, y in zip(nof[-6:], action[-6:])
        },
    }
    return nof, action, audit


def style_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3.5, width=0.8)


def main() -> int:
    nof, action, audit = load_trace()
    gray = "#777777"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), dpi=220)

    axes[0].plot(
        nof,
        action,
        color=gray,
        linewidth=0.95,
        marker="D",
        markersize=3.0,
        markerfacecolor="white",
        markeredgecolor=gray,
        markeredgewidth=0.8,
        markevery=25,
    )
    axes[0].set_xlim(0.0, 3000.0)
    axes[0].set_ylim(160.0, 390.0)
    axes[0].set_title("Complete trace")
    axes[0].set_xlabel("Number of Forces")
    axes[0].set_ylabel(r"Discrete geometric action $\widehat{S}_h$")
    style_axis(axes[0])

    zoom = nof >= 2700.0
    axes[1].plot(
        nof[zoom],
        action[zoom],
        color=gray,
        linewidth=1.0,
        marker="D",
        markersize=3.4,
        markerfacecolor="white",
        markeredgecolor=gray,
        markeredgewidth=0.8,
    )
    axes[1].set_xlim(2695.0, 3005.0)
    axes[1].set_ylim(168.5, 181.5)
    axes[1].set_title("Final 300 NoF: one alternating polyline")
    axes[1].set_xlabel("Number of Forces")
    axes[1].text(
        2710.0,
        180.25,
        "post-reparameterization checkpoints",
        ha="left",
        va="center",
        fontsize=7.4,
        color="#333333",
    )
    axes[1].text(
        2710.0,
        169.75,
        "intermediate checkpoints",
        ha="left",
        va="center",
        fontsize=7.4,
        color="#333333",
    )
    style_axis(axes[1])

    figure.suptitle(
        r"$N_{\mathrm{int}}=10$: Original String only",
        y=0.995,
        fontsize=10.0,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.012,
        "Each checkpoint has one unique NoF value; adjacent checkpoints alternate between two update phases.",
        ha="center",
        va="bottom",
        fontsize=7.1,
        color="#444444",
    )
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.22, top=0.82, wspace=0.30)

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    QA_PNG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PDF,
        format="pdf",
        bbox_inches="tight",
        metadata={
            "Title": "Q-transformed Muller N_int=10 Original String action diagnostic",
            "Subject": "Standalone view of the single alternating Original String trace",
        },
    )
    figure.savefig(QA_PNG, format="png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    print(json.dumps({"output_pdf": str(OUTPUT_PDF), **audit}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
