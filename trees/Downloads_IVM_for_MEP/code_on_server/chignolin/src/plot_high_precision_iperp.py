#!/usr/bin/env python3
"""High-precision I_perp comparison for one complete IVM and fixed NEB run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_production import (
    IVM_COLOR,
    NEB_COLOR,
    clean_axis,
    common_thresholds,
    configure_style,
    panel_label,
    running_first_passage,
    save_figure,
)


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]) for name in saved.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--ivm", required=True)
    parser.add_argument("--neb", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    ivm = load(Path(args.ivm))
    neb = load(Path(args.neb))
    output_dir = root / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    ivm_calls = np.asarray(ivm["calls_per_image"], dtype=float)
    neb_calls = np.asarray(neb["calls_per_image"], dtype=float)
    ivm_i = np.asarray(ivm["i_perp"], dtype=float)
    neb_i = np.asarray(neb["i_perp"], dtype=float)
    thresholds = common_thresholds(ivm_i, neb_i)
    ivm_cost = running_first_passage(ivm_calls, ivm_i, thresholds)
    neb_cost = running_first_passage(neb_calls, neb_i, thresholds)
    common = np.isfinite(ivm_cost) & np.isfinite(neb_cost)
    thresholds = thresholds[common]
    ivm_cost = ivm_cost[common]
    neb_cost = neb_cost[common]

    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    axes[0].plot(ivm_calls, ivm_i, color=IVM_COLOR, linewidth=2.0, label="IVM")
    neb_mask = neb_calls <= 2500
    axes[0].plot(
        neb_calls[neb_mask],
        neb_i[neb_mask],
        color=NEB_COLOR,
        linewidth=1.65,
        label="NEB",
    )
    axes[0].set_yscale("log")
    axes[0].set_xlim(0, 2500)
    axes[0].set_xlabel("Max evaluations per internal image")
    axes[0].set_ylabel(r"Unnormalized $I_{\perp}$")
    axes[0].legend(loc="upper right")
    clean_axis(axes[0])

    axes[1].plot(
        thresholds, ivm_cost, color=IVM_COLOR, marker="s", markersize=3.0, label="IVM"
    )
    axes[1].plot(
        thresholds, neb_cost, color=NEB_COLOR, marker="o", markersize=3.0, label="NEB"
    )
    axes[1].set_xscale("log")
    axes[1].invert_xaxis()
    axes[1].set_xlabel(r"Target $I_{\perp}$")
    axes[1].set_ylabel("Minimum evaluations per internal image")
    clean_axis(axes[1])
    finest = -1
    ratio = float(neb_cost[finest] / ivm_cost[finest])
    threshold = float(thresholds[finest])
    axes[1].annotate(
        "",
        xy=(threshold, ivm_cost[finest]),
        xytext=(threshold, neb_cost[finest]),
        arrowprops=dict(arrowstyle="<->", color="0.2", linewidth=0.8),
    )
    axes[1].annotate(
        f"{ratio:.2f}×",
        xy=(threshold, 0.5 * (ivm_cost[finest] + neb_cost[finest])),
        xytext=(5, 0),
        textcoords="offset points",
        ha="left",
        va="center",
    )
    panel_label(axes[0], "a")
    panel_label(axes[1], "b")
    figure.subplots_adjust(
        left=0.10, right=0.985, bottom=0.19, top=0.95, wspace=0.31
    )
    save_figure(figure, output_dir / "chignolin_high_precision_iperp")

    summary = {
        "ivm": str(Path(args.ivm).resolve()),
        "neb": str(Path(args.neb).resolve()),
        "finest_common_i_perp": threshold,
        "ivm_first_passage_calls": float(ivm_cost[finest]),
        "neb_first_passage_calls": float(neb_cost[finest]),
        "speedup": ratio,
        "ivm_color": IVM_COLOR,
        "neb_color": NEB_COLOR,
    }
    (output_dir / "chignolin_high_precision_iperp.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
