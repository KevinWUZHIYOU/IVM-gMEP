#!/usr/bin/env python3
"""Rank the fixed NEB scan, build its pointwise envelope, and plot it."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from chignolin_core import TANGENT_DEFINITION


NEB_COLOR = "#D55E00"
EXPECTED_K = (0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 80.0, 100.0, 500.0, 1000.0, 2000.0)
FIXED_INTERFACE_ALPHA = 1e-4


def forward_fill(calls, values, grid):
    calls = np.asarray(calls, dtype=int)
    values = np.asarray(values, dtype=float)
    order = np.argsort(calls)
    calls, values = calls[order], values[order]
    indices = np.searchsorted(calls, grid, side="right") - 1
    indices = np.clip(indices, 0, len(calls) - 1)
    return values[indices]


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.7,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="0.88", linewidth=0.45)


def save_figure(figure, stem):
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    candidate_paths = sorted(
        (root / "runs" / "neb").glob("neb_central_*budget500.npz")
    )

    grid = np.arange(1, 501, dtype=int)
    runs = []
    initial_hashes = set()
    for path in candidate_paths:
        with np.load(path, allow_pickle=False) as saved:
            spring_k = float(saved["spring_k"])
            alpha = float(saved["optimizer_alpha"])
            optimizer = str(saved["optimizer"]).lower()
            if (
                optimizer != "ode"
                or not np.isclose(alpha, FIXED_INTERFACE_ALPHA)
                or not any(np.isclose(spring_k, expected) for expected in EXPECTED_K)
            ):
                continue
            if str(saved["tangent_definition"]) != TANGENT_DEFINITION:
                raise RuntimeError(f"Non-central tangent result: {path}")
            initial_hashes.add(str(saved["initial_path_sha256"]))
            calls = np.asarray(saved["calls_per_image"], dtype=int)
            iperp = forward_fill(calls, saved["i_perp"], grid)
            barrier = forward_fill(calls, saved["barrier_kcal"], grid)
            fmax = forward_fill(calls, saved["physical_fmax"], grid)
            runs.append(
                {
                    "path": str(path),
                    "spring_k": spring_k,
                    "alpha": alpha,
                    "optimizer": optimizer,
                    "iperp": iperp,
                    "barrier": barrier,
                    "fmax": fmax,
                }
            )
    found_k = sorted(run["spring_k"] for run in runs)
    if len(runs) != len(EXPECTED_K) or not np.allclose(found_k, EXPECTED_K):
        raise RuntimeError(
            "Expected exactly one pure-ODE result for each requested k; "
            f"expected={list(EXPECTED_K)}, found={found_k}"
        )
    if len(initial_hashes) != 1:
        raise RuntimeError(f"NEB scan used multiple initial paths: {initial_hashes}")

    iperp_stack = np.asarray([run["iperp"] for run in runs])
    barrier_stack = np.asarray([run["barrier"] for run in runs])
    envelope_owner_paths = np.asarray([run["path"] for run in runs])
    envelope_i = np.min(iperp_stack, axis=0)
    envelope_b = np.min(barrier_stack, axis=0)
    envelope_i_owner = np.argmin(iperp_stack, axis=0)
    envelope_b_owner = np.argmin(barrier_stack, axis=0)
    for index, run in enumerate(runs):
        i_ratio = np.maximum(run["iperp"] / envelope_i, 1.0)
        b_ratio = np.maximum(run["barrier"] / envelope_b, 1.0)
        run["score"] = float(
            np.mean(np.log10(i_ratio[1:]))
            + np.mean(np.log10(b_ratio[1:]))
            + 0.5 * np.log10(i_ratio[-1])
            + 0.5 * np.log10(b_ratio[-1])
        )
    runs.sort(key=lambda item: item["score"])

    report_rows = []
    for rank, run in enumerate(runs, start=1):
        report_rows.append(
            {
                "rank": rank,
                "path": run["path"],
                "spring_k": run["spring_k"],
                "alpha": run["alpha"],
                "optimizer": run["optimizer"],
                "score": run["score"],
                "i_perp_at_500": float(run["iperp"][-1]),
                "barrier_at_500_kcal": float(run["barrier"][-1]),
                "physical_fmax_at_500": float(run["fmax"][-1]),
            }
        )
    report = {
        "initial_path_sha256": next(iter(initial_hashes)),
        "n_configurations": len(runs),
        "scan_variable": "spring_k_only",
        "spring_constants": list(EXPECTED_K),
        "optimizer": "NEBOptimizer(method='ODE')",
        "fixed_interface_alpha": FIXED_INTERFACE_ALPHA,
        "alpha_used_by_ode": False,
        "score_definition": (
            "log-area ratios to the pointwise I_perp and raw-barrier envelopes, "
            "plus half-weight final ratios"
        ),
        "top_three": report_rows[:3],
        "ranking": report_rows,
    }
    report_path = root / "reports" / "neb_scan_analysis.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    with (root / "reports" / "neb_scan_ranking.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)
    np.savez_compressed(
        root / "runs" / "neb" / "neb_pointwise_envelope_budget500.npz",
        calls=grid,
        i_perp=envelope_i,
        barrier_kcal=envelope_b,
        i_perp_owner_index=envelope_i_owner,
        barrier_owner_index=envelope_b_owner,
        owner_paths=envelope_owner_paths,
    )

    configure_style()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    group = sorted(runs, key=lambda item: item["spring_k"])
    spring_k = [run["spring_k"] for run in group]
    iperp_final = [run["iperp"][-1] for run in group]
    barrier_final = [run["barrier"][-1] for run in group]
    axes[0].plot(
        spring_k,
        iperp_final,
        marker="o",
        markersize=3.4,
        color=NEB_COLOR,
        label="NEBOptimizer (ODE)",
    )
    axes[1].plot(
        spring_k,
        barrier_final,
        marker="o",
        markersize=3.4,
        color=NEB_COLOR,
    )
    for axis in axes:
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel(r"Spring constant $k$")
        clean_axis(axis)
    axes[0].set_ylabel(r"$I_{\perp}$ at 500 evaluations")
    axes[1].set_ylabel(r"Raw barrier at 500 (kcal mol$^{-1}$)")
    handles, labels = axes[0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique[label] = handle
    axes[0].legend(unique.values(), unique.keys(), loc="best", ncol=2)
    axes[0].text(
        -0.15, 1.03, "a", transform=axes[0].transAxes, fontsize=11, fontweight="bold"
    )
    axes[1].text(
        -0.15, 1.03, "b", transform=axes[1].transAxes, fontsize=11, fontweight="bold"
    )
    figure.subplots_adjust(left=0.10, right=0.985, bottom=0.18, top=0.95, wspace=0.31)
    save_figure(figure, root / "figures" / "chignolin_neb_scan")


if __name__ == "__main__":
    main()
