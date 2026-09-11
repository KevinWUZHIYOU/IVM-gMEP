#!/usr/bin/env python3
"""Render the formal QM10 comparison after removing geometric action."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_muller_q10_article_method_comparison_n10_n20_2x4_geometric_action as legacy  # noqa: E402


DEFAULT_OUTPUT = (
    HERE
    / "muller_q10_article_method_comparison_n10_n20_2x4_geometric_action_linear3000.pdf"
)
PROVENANCE = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.stem + "_PROVENANCE.md")
PAPER_ROOT = HERE.parents[1]
MANUSCRIPT = PAPER_ROOT / "main.tex"
REFERENCE_BARRIER = 67.50188060819488
IRIVM_EXPERIMENT_ROOT = (
    legacy.MAIN_REPOSITORY
    / "python/q10_irivm_recovery_monotone_exploration_20260831"
)
IRIVM_ALGORITHM_SOURCE = (
    IRIVM_EXPERIMENT_ROOT / "src/q10_horizon_control_probe_optimizer.py"
)
IRIVM_RUNNER_SOURCE = IRIVM_EXPERIMENT_ROOT / "src/run_and_score.py"
IRIVM_CORE_SOURCE = (
    legacy.MAIN_REPOSITORY
    / "python/q10_irivm_low_r_coherence_secant_v2_20260830/src/"
    "q10_irivm_atomic_tangent_secant_core.py"
)
IRIVM_AUDIT_SOURCE = (
    legacy.MAIN_REPOSITORY
    / "python/q10_irivm_unified_cycle_frozen_reproduction_v2_20260831/src/"
    "final_audit_core.py"
)
REGISTERED_REFERENCE_CSV = legacy.REGISTERED_SOURCE_CSV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def registered_file(path: Path, role: str) -> dict:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing registered {role}: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "role": role,
    }


def qm10_manuscript_registration(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    start_marker = r'\subsubsection{\textbf{10D M\"uller Potential}}'
    end_marker = r'\subsubsection{\textbf{66D Alanine Dipeptide}}'
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    section = text[start:end]
    required_reference = (
        "Figures/Muller10/"
        "muller_q10_article_method_comparison_n10_n20_2x4_"
        "geometric_action_linear3000.pdf"
    )
    if required_reference not in section:
        raise RuntimeError("QM10 manuscript section does not reference this figure")
    return {
        "path": str(path.resolve()),
        "section_label": "subsubsec:Muller10",
        "figure_label": "fig:M10_density",
        "includegraphics_path": required_reference,
        "qm10_section_sha256": hashlib.sha256(
            section.encode("utf-8")
        ).hexdigest(),
    }


def draw_row(namespace: dict, axes: list, runs: list[dict]) -> dict:
    namespace["plot_metric_on_axis"](
        ax=axes[0],
        runs=runs,
        metric_name="loss_perp",
        ylabel=namespace["METRIC_INFO"]["loss_perp"]["ylabel"],
        use_compressed_y=True,
        panel_letter="placeholder",
    )
    legacy.registered.base.apply_force_axis(axes[0])
    namespace["plot_metric_on_axis"](
        ax=axes[1],
        runs=runs,
        metric_name="Delta_Energy",
        ylabel=namespace["METRIC_INFO"]["Delta_Energy"]["ylabel"],
        use_compressed_y=False,
        panel_letter="placeholder",
    )
    legacy.registered.base.apply_force_axis(axes[1])

    contour_runs, checkpoints = legacy.registered.base.contour_runs_at_budget(runs)
    namespace["draw_contour_with_paths"](
        ax=axes[2],
        path_specs=namespace["build_path_specs_from_runs"](contour_runs),
        V_func=namespace["V_muller10_np"],
        Q_np=namespace["Q_np"],
        n_grid=namespace["CONTOUR_N_GRID"],
        xlim=namespace["CONTOUR_XLIM"],
        ylim=namespace["CONTOUR_YLIM"],
        potential_level_min=namespace["POTENTIAL_LEVEL_MIN"],
        potential_level_max=namespace["POTENTIAL_LEVEL_MAX"],
        potential_n_levels=namespace["POTENTIAL_N_LEVELS"],
        contour_labels=namespace["CONTOUR_LABELS"],
        show_colorbar=namespace["SHOW_CONTOUR_COLORBAR"],
        colorbar_inset=False,
        show_start_end=namespace["SHOW_START_END"],
        panel_letter="placeholder",
    )
    for axis in axes:
        for artist in tuple(axis.texts):
            if artist.get_text().strip().lower() in {
                "start",
                "end",
                "placeholder",
            }:
                artist.remove()
        axis.set_box_aspect(1.0)
    return checkpoints


def draw_figure(
    namespace: dict,
    n10_runs: list[dict],
    n20_runs: list[dict],
) -> tuple[plt.Figure, dict, dict]:
    figure = plt.figure(
        figsize=(namespace["mm_to_inch"](180), namespace["mm_to_inch"](126)),
        dpi=220,
    )
    grid = GridSpec(
        2,
        3,
        figure=figure,
        width_ratios=(1.0, 1.0, 1.0),
        left=0.068,
        right=0.94,
        bottom=0.08,
        top=0.90,
        wspace=0.42,
        hspace=0.12,
    )
    axes = [
        [figure.add_subplot(grid[row, column]) for column in range(3)]
        for row in range(2)
    ]
    n10_checkpoints = draw_row(namespace, axes[0], n10_runs)
    n20_checkpoints = draw_row(namespace, axes[1], n20_runs)
    for axis in axes[0]:
        axis.set_xlabel("")

    figure.canvas.draw()
    row_titles = (
        r"$\mathbf{N}_{\mathrm{int}}=\mathbf{10}$",
        r"$\mathbf{N}_{\mathrm{int}}=\mathbf{20}$",
    )
    letter_rows = (("a", "b", "c"), ("d", "e", "f"))
    for row, (letters, row_title) in enumerate(zip(letter_rows, row_titles)):
        row_left = axes[row][0].get_position().x0
        row_right = axes[row][2].get_position().x1
        row_top = max(axis.get_position().y1 for axis in axes[row])
        header_y = row_top + 0.033
        header_center = 0.5 * (row_left + row_right)
        title_half_gap = 0.060
        for line_start, line_end in (
            (row_left, header_center - title_half_gap),
            (header_center + title_half_gap, row_right),
        ):
            figure.add_artist(
                Line2D(
                    [line_start, line_end],
                    [header_y, header_y],
                    transform=figure.transFigure,
                    color="#C2C2BE",
                    linewidth=0.6,
                    solid_capstyle="butt",
                    clip_on=False,
                    zorder=2,
                )
            )
        figure.text(
            header_center,
            header_y,
            row_title,
            ha="center",
            va="center",
            color="#272727",
            fontsize=8.4,
            fontweight="normal",
            zorder=3,
        )
        for axis, letter in zip(axes[row], letters):
            position = axis.get_position()
            figure.text(
                position.x0 - 0.04,
                row_top + 0.001,
                letter,
                ha="left",
                va="bottom",
                fontweight="bold",
                zorder=3,
                fontsize=namespace["plt"].rcParams["font.size"] + 1,
            )

    handles, labels = axes[0][0].get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        unique.setdefault(label, handle)
    legend_order = (
        "IVM",
        "Original String",
        "ODE-NEB",
        "OBSM-FIRE",
        "Static NEB",
        "OBSM-LBFGS",
    )
    figure.legend(
        [unique[label] for label in legend_order],
        list(legend_order),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        frameon=False,
        columnspacing=1.05,
        handlelength=2.2,
        handletextpad=0.45,
        fontsize=6.5,
    )
    return figure, n10_checkpoints, n20_checkpoints


def write_source_csv(path: Path, rows_by_case: list[tuple[str, list[dict]]]) -> None:
    fieldnames = (
        "case",
        "method",
        "number_of_forces",
        "source_record_index",
        "I_perp",
        "Delta_E_dense_linear500",
        "source_file",
        "source_sha256",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case_name, runs in rows_by_case:
            for run in runs:
                metrics = run["metrics"]
                source_file = Path(run["file"])
                source_hash = sha256(source_file)
                for index, nof in enumerate(metrics["grad_count_seq"]):
                    writer.writerow(
                        {
                            "case": case_name,
                            "method": run["label"],
                            "number_of_forces": f"{float(nof):.17g}",
                            "source_record_index": index,
                            "I_perp": f"{float(metrics['loss_perp'][index]):.17g}",
                            "Delta_E_dense_linear500": (
                                f"{float(metrics['Delta_Energy'][index]):.17g}"
                            ),
                            "source_file": str(source_file),
                            "source_sha256": source_hash,
                        }
                    )


def main() -> int:
    arguments = parse_args()
    namespace = legacy.registered.base.load_formal_namespace(
        legacy.registered.base.N10_ROOT
    )
    n10_runs, n20_runs, formal_summaries = legacy.load_runs(namespace)
    formal_ivm_validation = legacy.validate_formal_ivm(n10_runs, n20_runs)

    figure, n10_checkpoints, n20_checkpoints = draw_figure(
        namespace, n10_runs, n20_runs
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    staged_pdf = arguments.output.with_name(arguments.output.stem + ".stage.pdf")
    output_png = arguments.output.with_suffix(".png")
    staged_png = output_png.with_name(output_png.stem + ".stage.png")
    figure.savefig(staged_pdf, format="pdf", dpi=600)
    figure.savefig(staged_png, format="png", dpi=300)
    plt.close(figure)
    staged_pdf.replace(arguments.output)
    staged_png.replace(output_png)

    source_csv = arguments.output.with_name(arguments.output.stem + "_source.csv")
    staged_csv = source_csv.with_name(source_csv.stem + ".stage.csv")
    write_source_csv(
        staged_csv,
        [
            ("10 interior images", n10_runs),
            ("20 interior images", n20_runs),
        ],
    )
    staged_csv.replace(source_csv)
    registered_validation = legacy.validate_registered_rows(source_csv)

    q_path = (
        legacy.MAIN_REPOSITORY
        / "paper/ActiveExperiments/muller_q10_10d/code/inputs/Q_data.npz"
    )
    orthogonality_error = float(
        max(
            np.linalg.norm(legacy.Q.T @ legacy.Q - np.eye(10), ord=np.inf),
            np.linalg.norm(legacy.Q @ legacy.Q.T - np.eye(10), ord=np.inf),
        )
    )
    q_array_sha256 = hashlib.sha256(
        np.ascontiguousarray(legacy.Q).tobytes()
    ).hexdigest()
    if q_array_sha256 != legacy.EXPECTED_Q_ARRAY_SHA256:
        raise RuntimeError(f"unexpected Q array SHA-256: {q_array_sha256}")
    if orthogonality_error > 1.0e-12:
        raise RuntimeError(
            f"Q orthogonality gate failed: {orthogonality_error:.3e}"
        )

    input_data = []
    for case_name, runs in (
        ("10 interior images", n10_runs),
        ("20 interior images", n20_runs),
    ):
        for run in runs:
            entry = registered_file(
                Path(run["file"]),
                f"{case_name}; {run['label']} plotted trajectory",
            )
            entry.update({"case": case_name, "method": run["label"]})
            input_data.append(entry)

    algorithm_sha256 = sha256(IRIVM_ALGORITHM_SOURCE)
    formal_ivm_inputs = {}
    for n_internal in (10, 20):
        trajectory_path, summary_path = legacy.registered.IR_INPUTS[n_internal][
            legacy.registered.IRIVM_LABEL
        ]
        summary = formal_summaries[f"N{n_internal}"]
        if summary["algorithm_sha256"] != algorithm_sha256:
            raise RuntimeError(
                f"N{n_internal} summary does not authenticate the registered "
                "IR-IVM algorithm source"
            )
        formal_ivm_inputs[f"N{n_internal}"] = {
            "trajectory": registered_file(
                trajectory_path, f"N{n_internal} formal IVM trajectory"
            ),
            "summary": registered_file(
                summary_path, f"N{n_internal} formal IVM run summary"
            ),
            "algorithm_sha256_recorded_by_summary": summary[
                "algorithm_sha256"
            ],
            "variant": summary["variant"],
        }

    plotting_sources = {
        "registered_no_action_plot": Path(__file__).resolve(),
        "legacy_layout_and_validation_helper": Path(legacy.__file__).resolve(),
        "registered_irivm_overlay_loader": Path(
            legacy.registered.__file__
        ).resolve(),
        "base_qm10_comparison_loader": Path(
            legacy.registered.base.__file__
        ).resolve(),
        "n10_formal_plotting_adapter": (
            legacy.registered.base.N10_ROOT
            / "code/plotting/plot_q10_article_muller12.py"
        ),
        "formal_plotting_notebook_source": (
            legacy.registered.base.N10_ROOT
            / "code/plotting/reference/figure_M10_N20_v2.ipynb"
        ),
        "qm10_potential": legacy.POTENTIAL_SOURCE,
        "qm10_improved_tangent": legacy.TANGENT_SOURCE,
    }
    plotting_code = {
        name: registered_file(path, name)
        for name, path in plotting_sources.items()
    }
    experiment_code = {
        "formal_irivm_optimizer": registered_file(
            IRIVM_ALGORITHM_SOURCE, "formal IR-IVM optimizer"
        ),
        "formal_irivm_core": registered_file(
            IRIVM_CORE_SOURCE, "IR-IVM tangent/secant core"
        ),
        "formal_irivm_runner_and_scorer": registered_file(
            IRIVM_RUNNER_SOURCE, "IR-IVM run and metric entry point"
        ),
        "independent_metric_auditor": registered_file(
            IRIVM_AUDIT_SOURCE, "independent QM10 metric auditor"
        ),
    }

    manifest = {
        "schema_version": 3,
        "status": "awaiting_visual_qa",
        "case": "fixed-Q unperturbed 10D Muller-Brown",
        "layout": "2x3; 10 interior images above 20 interior images",
        "panels": ["I_perp", "densified Delta E", "path contour"],
        "geometric_action_included": False,
        "legacy_filename_note": (
            "the historical filename contains 2x4_geometric_action, but this "
            "registered artifact is the formal 2x3 comparison without a "
            "geometric-action panel"
        ),
        "xscale": "linear",
        "xlim": [0, 3000],
        "i_perp_y_axis": "compressed logarithmic scale",
        "energy_barrier_y_axis": "standard logarithmic scale",
        "energy_barrier_metric": (
            "500-point coordinate-wise linear densification; raw Delta E, "
            "not barrier error"
        ),
        "reference_barrier": REFERENCE_BARRIER,
        "methods": [run["label"] for run in n10_runs],
        "ivm_series_policy": (
            "the new unified IR-IVM trajectory is the sole series labeled IVM; "
            "the article IVM and Previous IR-IVM are excluded"
        ),
        "cost_axis": (
            "direct sum of uncached force calls over all interior images"
        ),
        "formal_ivm_validation": formal_ivm_validation,
        "formal_ivm_parameters": {
            key: {
                "alpha": summary["alpha"],
                "r": summary["r"],
                "m_max": summary["m_max"],
                "terminal_nof": summary["terminal_nof"],
            }
            for key, summary in formal_summaries.items()
        },
        "formal_ivm_inputs": formal_ivm_inputs,
        "plotted_input_data": input_data,
        "experiment_code": experiment_code,
        "plotting_code_dependencies": plotting_code,
        "registered_curve_validation": registered_validation,
        "validation_reference": registered_file(
            REGISTERED_REFERENCE_CSV,
            "registered pre-removal 2x3 curve table used for rowwise audit",
        ),
        "q_matrix": {
            "path": str(q_path),
            "sha256": sha256(q_path),
            "array_bytes_sha256": q_array_sha256,
            "maximum_orthogonality_error": orthogonality_error,
            "coordinate_convention": "z=Q@x; x=Q.T@z",
        },
        "contour_methods": sorted(legacy.registered.base.CONTOUR_LABELS),
        "contour_checkpoint_policy": (
            "last actually paid checkpoint with total forces <= 3000"
        ),
        "contour_checkpoint_forces": {
            "10_interior": n10_checkpoints,
            "20_interior": n20_checkpoints,
        },
        "source_csv": {
            "path": str(source_csv),
            "sha256": sha256(source_csv),
            "source_record_index_origin": 0,
        },
        "plot_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "provenance": {
            "path": str(PROVENANCE),
            "sha256": sha256(PROVENANCE),
        },
        "manuscript_reference": qm10_manuscript_registration(MANUSCRIPT),
        "reproduction": {
            "working_directory": str(PAPER_ROOT.resolve()),
            "command": (
                "/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python "
                "Figures/Muller10/"
                "plot_muller_q10_article_method_comparison_n10_n20_2x3_"
                "no_action.py"
            ),
            "python_executable": sys.executable,
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "matplotlib_version": matplotlib.__version__,
        },
        "outputs": {
            str(arguments.output): sha256(arguments.output),
            str(output_png): sha256(output_png),
        },
    }
    manifest_path = arguments.output.with_name(
        arguments.output.stem + "_manifest.json"
    )
    staged_manifest = manifest_path.with_name(manifest_path.stem + ".stage.json")
    staged_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    staged_manifest.replace(manifest_path)
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
