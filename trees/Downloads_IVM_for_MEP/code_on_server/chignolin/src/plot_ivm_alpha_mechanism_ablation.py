"""Paper-grade mechanism figure for Chignolin fixed-alpha IVM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#0072B2"
ORANGE = "#D55E00"
GRAY = "#666666"


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.3,
            "ytick.labelsize": 7.3,
            "legend.fontsize": 6.9,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
        }
    )


def clean_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(
        axis="both",
        which="major",
        color="#D9D9D9",
        linewidth=0.45,
        alpha=0.65,
        zorder=0,
    )


def load_run(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {
            "path": path,
            "calls": np.asarray(saved["calls_per_image"], dtype=int),
            "forces": np.asarray(saved["counter_vectors"], dtype=np.int64).sum(
                axis=1
            ),
            "i_perp": np.asarray(saved["i_perp"], dtype=float),
            "fmax": np.asarray(saved["physical_fmax"], dtype=float),
            "accepted_fraction": np.asarray(
                saved["accepted_fraction"], dtype=float
            ),
            "backtracks": np.asarray(
                saved["subproblem_line_search_backtracks"], dtype=int
            ),
            "relative_kkt": np.asarray(
                saved["subproblem_relative_kkt_max"], dtype=float
            ),
            "stop_reason": str(scalar(saved, "stop_reason")),
        }


def value_at_call(run: dict, call: int, key: str) -> float:
    eligible = np.flatnonzero(run["calls"] <= call)
    if not eligible.size:
        raise RuntimeError(f"No record at or before call {call}")
    return float(run[key][eligible[-1]])


def save_figure(figure, output: Path, stem: str) -> list[str]:
    files = []
    for suffix, kwargs in (
        (".pdf", {}),
        (".svg", {}),
        (".png", {"dpi": 600}),
    ):
        path = output / f"{stem}{suffix}"
        figure.savefig(path, **kwargs)
        files.append(str(path))
    return files


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--output-directory",
        default="figures/alpha_mechanism_20260728a",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = root / args.output_directory
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    baseline_manifest = json.loads(
        (
            root
            / "reports"
            / "ivm_fixed_alpha_extended_20260728a"
            / "manifest.json"
        ).read_text()
    )
    ablation_manifest = json.loads(
        (
            root
            / "reports"
            / "ivm_alpha_mechanism_ablations_20260728a"
            / "manifest.json"
        ).read_text()
    )
    dense_manifest = json.loads(
        (
            root
            / "reports"
            / "ivm_alpha_mechanism_dense_20260728a"
            / "manifest.json"
        ).read_text()
    )
    baseline_records = {
        float(row["alpha"]): row for row in baseline_manifest["records"]
    }
    ablation_records = {
        (row["variant"], float(row["alpha"])): row
        for row in ablation_manifest["records"]
    }
    dense_records = {
        (row["variant"], float(row["alpha"])): row
        for row in dense_manifest["records"]
    }
    baseline = {
        alpha: load_run(Path(baseline_records[alpha]["result_file"]))
        for alpha in (1.0, 1000.0)
    }
    proximal_only = {
        alpha: load_run(
            Path(ablation_records[("proximal_only", alpha)]["result_file"])
        )
        for alpha in (1.0, 1000.0)
    }

    setup_style()
    figure, axes = plt.subplots(2, 2, figsize=(7.25, 5.05))
    axes = axes.ravel()

    for alpha, axis, color in (
        (1.0, axes[0], BLUE),
        (1000.0, axes[1], ORANGE),
    ):
        protected = baseline[alpha]
        unguarded = proximal_only[alpha]
        protected_mask = (protected["calls"] >= 5) & (
            protected["calls"] <= 150
        )
        unguarded_mask = (unguarded["calls"] >= 5) & (
            unguarded["calls"] <= 150
        )
        axis.plot(
            protected["forces"][protected_mask],
            protected["i_perp"][protected_mask],
            color=color,
            linewidth=1.8,
            label="Production guard",
            zorder=3,
        )
        axis.plot(
            unguarded["forces"][unguarded_mask],
            unguarded["i_perp"][unguarded_mask],
            color=GRAY,
            linestyle="--",
            linewidth=1.35,
            label=r"Proximal-only acceptance",
            zorder=2,
        )
        axis.set_yscale("log")
        axis.set_xlim(100, 3050)
        axis.set_xlabel("Number of Force Evaluations")
        axis.set_ylabel(r"Unnormalized $I_\perp$")
        clean_axis(axis)
        axis.text(
            0.04,
            0.07,
            rf"$\alpha={alpha:g}$",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            color=color,
            fontsize=8.2,
        )
    axes[0].set_ylim(7e-2, 3e3)
    axes[1].set_ylim(7e-2, 3e3)
    axes[0].text(
        0.97,
        0.79,
        "curves coincide",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=6.8,
    )
    unguarded_high = proximal_only[1000.0]
    minimum_index = int(np.argmin(unguarded_high["i_perp"]))
    minimum_value = float(unguarded_high["i_perp"][minimum_index])
    final_value = value_at_call(unguarded_high, 150, "i_perp")
    axes[1].annotate(
        f"{final_value / minimum_value:.1f}x rebound",
        xy=(
            unguarded_high["forces"][-1],
            unguarded_high["i_perp"][-1],
        ),
        xytext=(2220, 24),
        arrowprops={
            "arrowstyle": "->",
            "color": GRAY,
            "linewidth": 0.8,
        },
        ha="center",
        va="bottom",
        fontsize=6.8,
    )
    axes[0].legend(frameon=False, loc="upper right")
    axes[1].legend(frameon=False, loc="upper right")

    alpha_values = []
    accepted_yield = []
    backtrack_fraction = []
    mechanism_rows = []
    for record in baseline_manifest["records"]:
        alpha = float(record["alpha"])
        run = load_run(Path(record["result_file"]))
        common = run["calls"] <= 300
        alpha_values.append(alpha)
        accepted = float(np.mean(run["accepted_fraction"][common] > 0.0))
        backtracked = float(np.mean(run["backtracks"][common] > 0))
        accepted_yield.append(accepted)
        backtrack_fraction.append(backtracked)
        kkt = run["relative_kkt"][common]
        mechanism_rows.append(
            {
                "alpha": alpha,
                "maximum_call": 300,
                "accepted_state_yield": accepted,
                "line_search_backtracking_fraction": backtracked,
                "median_relative_kkt_max": float(np.nanmedian(kkt)),
            }
        )
    alpha_values = np.asarray(alpha_values)
    axes[2].plot(
        alpha_values,
        accepted_yield,
        color=BLUE,
        marker="o",
        markersize=3.7,
        linewidth=1.45,
        label="Accepted-state yield",
    )
    axes[2].plot(
        alpha_values,
        backtrack_fraction,
        color=ORANGE,
        marker="s",
        markersize=3.5,
        linewidth=1.35,
        label="Backtracking fraction",
    )
    axes[2].axvline(
        1.0, color=BLUE, linestyle=":", linewidth=1.0, alpha=0.95
    )
    axes[2].set_xscale("log")
    axes[2].set_ylim(-0.03, 1.05)
    axes[2].set_xlabel(r"Fixed proximal parameter $\alpha$")
    axes[2].set_ylabel("Fraction of call labels")
    axes[2].legend(frameon=False, loc="center left")
    clean_axis(axes[2])

    variants = (
        ("production", "Production"),
        ("relaxed_trust", "Relaxed\ntrust"),
        ("proximal_only", "Proximal\nonly"),
        ("both_relaxed", "Both\nrelaxed"),
    )
    x = np.arange(len(variants), dtype=float)
    endpoint_rows = []
    alpha1_values = []
    alpha1000_values = []
    for key, label in variants:
        if key == "production":
            values = {
                alpha: value_at_call(baseline[alpha], 150, "i_perp")
                for alpha in (1.0, 1000.0)
            }
            dense_values = {
                1.0: 22.727157308713963,
                1000.0: 25.440961585139974,
            }
        else:
            values = {
                alpha: float(
                    ablation_records[(key, alpha)]["final_i_perp"]
                )
                for alpha in (1.0, 1000.0)
            }
            dense_values = {
                alpha: float(
                    dense_records[(key, alpha)]["dense_barrier_kcal"]
                )
                for alpha in (1.0, 1000.0)
            }
        alpha1_values.append(values[1.0])
        alpha1000_values.append(values[1000.0])
        for alpha in (1.0, 1000.0):
            endpoint_rows.append(
                {
                    "variant": key,
                    "variant_label": label.replace("\n", " "),
                    "alpha": alpha,
                    "number_of_force_evaluations": 3002,
                    "i_perp": values[alpha],
                    "dense_barrier_kcal": dense_values[alpha],
                    "ratio_to_same_variant_alpha1": (
                        values[alpha] / values[1.0]
                    ),
                }
            )
    for index in range(len(x)):
        axes[3].plot(
            [x[index], x[index]],
            [alpha1_values[index], alpha1000_values[index]],
            color="#B8B8B8",
            linewidth=1.0,
            zorder=1,
        )
    axes[3].scatter(
        x - 0.05,
        alpha1_values,
        color=BLUE,
        marker="o",
        s=25,
        label=r"$\alpha=1$",
        zorder=3,
    )
    axes[3].scatter(
        x + 0.05,
        alpha1000_values,
        color=ORANGE,
        marker="s",
        s=24,
        label=r"$\alpha=1000$",
        zorder=3,
    )
    for index, (low, high) in enumerate(
        zip(alpha1_values, alpha1000_values)
    ):
        axes[3].text(
            x[index] + 0.08,
            high * 1.13,
            f"{high / low:.1f}x",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    axes[3].set_yscale("log")
    axes[3].set_ylim(7e-2, 20)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels([label for _, label in variants])
    axes[3].set_ylabel(r"$I_\perp$ at $N_F=3{,}002$")
    axes[3].legend(frameon=False, loc="upper left")
    clean_axis(axes[3])

    for index, axis in enumerate(axes):
        axis.text(
            -0.15,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=10,
            ha="left",
            va="bottom",
        )
    figure.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.13,
        top=0.97,
        wspace=0.36,
        hspace=0.42,
    )
    files = save_figure(
        figure, output, "chignolin_ivm_alpha_mechanism_ablation"
    )
    plt.close(figure)

    write_csv(data_dir / "protected_solver_mechanism.csv", mechanism_rows)
    write_csv(data_dir / "matched_ablation_endpoints.csv", endpoint_rows)
    summary = {
        "figure_files": files,
        "production_results_untouched": True,
        "common_ablation_budget_per_internal_image": 150,
        "primary_cost_axis": "counter_vectors.sum(axis=1)",
        "primary_cost_label": "Number of Force Evaluations",
        "key_findings": {
            "alpha1_proximal_only_equals_production_at_call150": bool(
                np.isclose(
                    alpha1_values[0],
                    alpha1_values[2],
                    rtol=0.0,
                    atol=0.0,
                )
            ),
            "alpha1000_proximal_only_rebound": (
                final_value / minimum_value
            ),
            "alpha1000_proximal_only_final_to_alpha1_ratio": (
                alpha1000_values[2] / alpha1_values[2]
            ),
            "relaxed_trust_alpha1000_relative_change_vs_production": (
                alpha1000_values[1] / alpha1000_values[0] - 1.0
            ),
        },
        "source_manifests": [
            str(
                root
                / "reports"
                / "ivm_fixed_alpha_extended_20260728a"
                / "manifest.json"
            ),
            str(
                root
                / "reports"
                / "ivm_alpha_mechanism_ablations_20260728a"
                / "manifest.json"
            ),
            str(
                root
                / "reports"
                / "ivm_alpha_mechanism_dense_20260728a"
                / "manifest.json"
            ),
        ],
    }
    summary_path = output / "chignolin_ivm_alpha_mechanism_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(summary_path, flush=True)


if __name__ == "__main__":
    main()
