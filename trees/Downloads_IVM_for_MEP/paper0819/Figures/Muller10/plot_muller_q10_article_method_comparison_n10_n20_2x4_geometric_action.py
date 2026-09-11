#!/usr/bin/env python3
"""Render the formal QM10 comparison with discrete geometric action.

The new unified IR-IVM trajectory is the sole formal ``IVM`` series.  The
article-era IVM and the exact-envelope predecessor are excluded.  The third
column adds

    S_hat,h(phi) = sum_i 0.5 * (|grad V_i| + |grad V_{i+1}|)
                              * |phi_{i+1} - phi_i|.

For panel c, every stored 12-point chain is linearly interpolated in the
normalized image-index parameter to 502 total points (500 interior points)
before the quadrature is applied.  Panel g retains its 22 native stored points
(20 interior points).  The line integral itself is tangent-independent.  For
every checkpoint, however, the script audits the original solver path with
the exact Henkelman-Jonsson improved-tangent routine used by QM10; no
central-difference tangent is introduced.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
MAIN_REPOSITORY = Path("/Users/wuzhiyou/IVM_for_MEP")
for source_directory in (
    MAIN_REPOSITORY / "paper/ActiveExperiments/muller_q10_10d/code/ivm",
    MAIN_REPOSITORY / "paper/ActiveExperiments/muller_q10_10d/code/baselines",
    MAIN_REPOSITORY / "paper/ActiveExperiments/muller_q10_10d/code/common",
):
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import plot_muller_q10_article_method_comparison_n10_n20_2x3_with_irivm as registered  # noqa: E402
from q10_ivm_core import improved_tangents_arbitrary  # noqa: E402
from q10_potential import Q, energy_gradient  # noqa: E402


ACTION_KEY = "Geometric_Action"
ACTION_COLUMN = "Geometric_Action_discrete_trapezoidal"
FORMAL_IVM_LABEL = "IVM"
PANEL_C_ACTION_INTERIOR_IMAGES = 500
PANEL_G_ACTION_INTERIOR_IMAGES = 20
EXPECTED_Q_ARRAY_SHA256 = (
    "372962a47504e6df28a20d8b022482a6835680765434b4385a6dba0e826974bc"
)
DEFAULT_OUTPUT = (
    HERE
    / "muller_q10_article_method_comparison_n10_n20_2x4_geometric_action_linear3000_with_action_legacy.pdf"
)
REGISTERED_SOURCE_CSV = registered.DEFAULT_OUTPUT.with_name(
    registered.DEFAULT_OUTPUT.stem + "_source.csv"
)
POTENTIAL_SOURCE = (
    MAIN_REPOSITORY
    / "paper/ActiveExperiments/muller_q10_10d/code/common/q10_potential.py"
)
TANGENT_SOURCE = (
    MAIN_REPOSITORY
    / "paper/ActiveExperiments/muller_q10_10d/code/ivm/q10_ivm_core.py"
)


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


def linearly_interpolate_path(
    path: np.ndarray,
    target_interior_images: int = PANEL_C_ACTION_INTERIOR_IMAGES,
) -> np.ndarray:
    """Interpolate a path in its normalized image-index parameter."""
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 10:
        raise ValueError(f"invalid QM10 path shape: {points.shape}")
    if not np.all(np.isfinite(points)):
        raise FloatingPointError("non-finite QM10 path")
    target_total_images = int(target_interior_images) + 2
    if target_total_images < 3:
        raise ValueError("target path must contain at least one interior image")
    if points.shape[0] == target_total_images:
        return points.copy()
    if points.shape[0] > target_total_images:
        raise ValueError(
            "linear action interpolation must not downsample a stored path: "
            f"{points.shape[0]} > {target_total_images}"
        )

    old_parameter = np.linspace(0.0, 1.0, points.shape[0])
    new_parameter = np.linspace(0.0, 1.0, target_total_images)
    interpolated = np.column_stack(
        [
            np.interp(new_parameter, old_parameter, points[:, coordinate])
            for coordinate in range(points.shape[1])
        ]
    )
    interpolated[0] = points[0]
    interpolated[-1] = points[-1]
    if not np.all(np.isfinite(interpolated)):
        raise FloatingPointError("non-finite linearly interpolated QM10 path")
    np.testing.assert_array_equal(interpolated[[0, -1]], points[[0, -1]])
    return interpolated


def discrete_geometric_action(
    path: np.ndarray,
    action_path: np.ndarray | None = None,
) -> tuple[float, float, float, float, int]:
    """Return action and audits, keeping the solver-path tangent unchanged."""
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] != 10:
        raise ValueError(f"invalid QM10 path shape: {points.shape}")
    if not np.all(np.isfinite(points)):
        raise FloatingPointError("non-finite QM10 path")

    evaluated = [energy_gradient(point) for point in points]
    energies = np.asarray([item[0] for item in evaluated], dtype=float)
    gradients = np.asarray([item[1] for item in evaluated], dtype=float)

    # Tangent audit: this is the exact arbitrary-length routine used by QM10
    # IR-IVM.  S_hat itself depends only on d ell and therefore does not use a
    # finite-difference tangent direction.
    tangents = improved_tangents_arbitrary(points, energies)
    tangent_norms = np.linalg.norm(tangents, axis=1)
    if (
        tangents.shape != (points.shape[0] - 2, points.shape[1])
        or not np.all(np.isfinite(tangents))
        or np.any(tangent_norms <= 0.0)
    ):
        raise FloatingPointError("invalid Henkelman-Jonsson improved tangent")
    tangent_error = float(np.max(np.abs(tangent_norms - 1.0)))
    if tangent_error > 1.0e-12:
        raise FloatingPointError(
            f"improved tangent is not normalized: error={tangent_error:.3e}"
        )

    # This reproduces the solver's tangent-dependent I_perp diagnostic.  The
    # direction is the improved tangent above.  np.gradient supplies only the
    # scalar speed used by the historical article quadrature; it is never used
    # as a tangent direction.
    internal_forces = -gradients[1:-1]
    normal_forces = internal_forces - np.sum(
        internal_forces * tangents, axis=1
    )[:, None] * tangents
    parameter = np.linspace(0.0, 1.0, points.shape[0])
    speed = np.linalg.norm(
        np.gradient(points, parameter, axis=0, edge_order=2), axis=1
    )
    audited_i_perp = float(
        np.sum(np.sum(normal_forces**2, axis=1) * speed[1:-1])
        / (points.shape[0] - 1)
    )

    if action_path is None:
        action_points = points
        action_gradients = gradients
        action_evaluation_count = int(points.shape[0])
    else:
        action_points = np.asarray(action_path, dtype=float)
        if (
            action_points.ndim != 2
            or action_points.shape[0] < 3
            or action_points.shape[1] != points.shape[1]
            or not np.all(np.isfinite(action_points))
        ):
            raise ValueError(f"invalid action path shape: {action_points.shape}")
        action_gradients = np.asarray(
            [energy_gradient(point)[1] for point in action_points], dtype=float
        )
        action_evaluation_count = int(points.shape[0] + action_points.shape[0])

    gradient_norms = np.linalg.norm(action_gradients, axis=1)
    segment_lengths = np.linalg.norm(np.diff(action_points, axis=0), axis=1)
    action = float(
        np.dot(
            0.5 * (gradient_norms[:-1] + gradient_norms[1:]),
            segment_lengths,
        )
    )
    node_weights = np.empty(action_points.shape[0], dtype=float)
    node_weights[0] = 0.5 * segment_lengths[0]
    node_weights[-1] = 0.5 * segment_lengths[-1]
    node_weights[1:-1] = 0.5 * (
        segment_lengths[:-1] + segment_lengths[1:]
    )
    node_weight_action = float(np.dot(gradient_norms, node_weights))
    np.testing.assert_allclose(
        action, node_weight_action, rtol=5.0e-15, atol=1.0e-13
    )
    quadrature_identity_error = abs(action - node_weight_action)
    if not np.isfinite(action) or action <= 0.0:
        raise FloatingPointError(f"invalid discrete geometric action: {action}")
    return (
        action,
        tangent_error,
        audited_i_perp,
        quadrature_identity_error,
        action_evaluation_count,
    )


def attach_geometric_action(
    runs: list[dict],
    interpolate_to_interior_images: int | None = None,
) -> dict:
    audit = {}
    for run in runs:
        paths = np.asarray(run["images_seq"], dtype=float)
        count = np.asarray(run["metrics"]["grad_count_seq"], dtype=float)
        if paths.shape[0] != count.size:
            raise RuntimeError(
                f"path/checkpoint mismatch for {run['label']}: "
                f"{paths.shape[0]} != {count.size}"
            )
        values = []
        tangent_errors = []
        audited_i_perp = []
        quadrature_identity_errors = []
        offline_evaluation_counts = []
        for path in paths:
            action_path = (
                linearly_interpolate_path(
                    path,
                    target_interior_images=interpolate_to_interior_images,
                )
                if interpolate_to_interior_images is not None
                else None
            )
            (
                value,
                tangent_error,
                i_perp_value,
                quadrature_identity_error,
                offline_evaluation_count,
            ) = discrete_geometric_action(path, action_path=action_path)
            values.append(value)
            tangent_errors.append(tangent_error)
            audited_i_perp.append(i_perp_value)
            quadrature_identity_errors.append(quadrature_identity_error)
            offline_evaluation_counts.append(offline_evaluation_count)
        values_array = np.asarray(values, dtype=float)
        audited_i_perp_array = np.asarray(audited_i_perp, dtype=float)
        recorded_i_perp = np.asarray(run["metrics"]["loss_perp"], dtype=float)
        absolute_i_perp_error = np.abs(audited_i_perp_array - recorded_i_perp)
        relative_i_perp_error = absolute_i_perp_error / np.maximum(
            np.abs(recorded_i_perp), 1.0e-300
        )
        if float(np.max(relative_i_perp_error)) > 1.0e-10:
            raise RuntimeError(
                f"improved-tangent I_perp audit failed for {run['label']}"
            )
        run["metrics"][ACTION_KEY] = values_array
        visible = count <= 3000.0
        if not np.any(visible):
            raise RuntimeError(f"no checkpoint within plot range for {run['label']}")
        visible_indices = np.flatnonzero(visible)
        last_visible_index = int(visible_indices[-1])
        audit[run["label"]] = {
            "checkpoint_count": int(values_array.size),
            "minimum": float(np.min(values_array)),
            "maximum": float(np.max(values_array)),
            "initial": float(values_array[0]),
            "terminal": float(values_array[-1]),
            "visible_checkpoint_count": int(visible_indices.size),
            "visible_minimum": float(np.min(values_array[visible])),
            "visible_maximum": float(np.max(values_array[visible])),
            "last_visible_nof": float(count[last_visible_index]),
            "last_visible_value": float(values_array[last_visible_index]),
            "maximum_improved_tangent_norm_error": float(
                np.max(tangent_errors)
            ),
            "maximum_i_perp_recalculation_absolute_error": float(
                np.max(absolute_i_perp_error)
            ),
            "maximum_i_perp_recalculation_relative_error": float(
                np.max(relative_i_perp_error)
            ),
            "maximum_edge_vs_node_weight_action_difference": float(
                np.max(quadrature_identity_errors)
            ),
            "original_total_images": int(paths.shape[1]),
            "action_total_images": int(
                interpolate_to_interior_images + 2
                if interpolate_to_interior_images is not None
                else paths.shape[1]
            ),
            "linear_interpolation_parameter": (
                "normalized image index"
                if interpolate_to_interior_images is not None
                else "none; stored path used directly"
            ),
            "offline_gradient_evaluations": int(
                np.sum(offline_evaluation_counts)
            ),
            "visible_offline_gradient_evaluations": int(
                np.sum(np.asarray(offline_evaluation_counts)[visible])
            ),
        }
    return audit


def plot_action_on_axis(namespace: dict, axis, runs: list[dict]) -> None:
    visible_values = []
    for run in runs:
        x_values = namespace["get_x_values_for_run"](run)
        y_values = np.asarray(run["metrics"][ACTION_KEY], dtype=float)
        style = namespace["get_plot_style"](run)
        axis.plot(
            x_values,
            y_values,
            label=run["label"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            color=style["color"],
            marker=style["marker"],
            markersize=style["markersize"],
            markerfacecolor="white",
            markeredgewidth=0.7,
            markevery=namespace["get_markevery"](x_values, target_markers=8),
            zorder=style["zorder"],
        )
        visible_values.extend(y_values[np.asarray(x_values) <= 3000.0])
    visible_values = np.asarray(visible_values, dtype=float)
    lower = float(np.min(visible_values))
    upper = float(np.max(visible_values))
    padding = 0.055 * (upper - lower)
    axis.set_ylim(lower - padding, upper + padding)
    axis.set_ylabel(r"Discrete geometric action $\widehat{S}_h$")
    axis.set_xlabel("Number of Forces")
    namespace["style_common_axes"](axis)


def load_runs(namespace: dict) -> tuple[list[dict], list[dict], dict]:
    registered.base.extend_styles(namespace)
    n10_runs, _ = registered.base.load_n10_runs(namespace)
    n20_runs, _ = registered.base.load_n20_runs(namespace)
    summaries = {}
    for n_internal, runs in ((10, n10_runs), (20, n20_runs)):
        formal_run, summary = registered.make_irivm_run(
            *registered.IR_INPUTS[n_internal][registered.IRIVM_LABEL],
            FORMAL_IVM_LABEL,
            n_internal,
        )
        expected_checkpoints = 79 if n_internal == 10 else 98
        if int(summary["i_perp_rise_count"]) != 0:
            raise RuntimeError(f"formal N={n_internal} IVM is not zero-rise")
        for field in (
            "strict_i_perp_count",
            "strict_barrier_count",
            "strict_joint_count",
            "checkpoint_count",
        ):
            if int(summary[field]) != expected_checkpoints:
                raise RuntimeError(
                    f"formal N={n_internal} IVM fails {field}: "
                    f"{summary[field]} != {expected_checkpoints}"
                )
        summaries[f"N{n_internal}"] = summary
        retained_baselines = [
            run for run in runs if run["label"] != FORMAL_IVM_LABEL
        ]
        runs[:] = [formal_run] + retained_baselines
        labels = [run["label"] for run in runs]
        if len(labels) != 6 or len(set(labels)) != 6:
            raise RuntimeError(
                f"expected one formal IVM plus five baselines, got {labels}"
            )
    return n10_runs, n20_runs, summaries


def validate_formal_ivm(n10_runs: list[dict], n20_runs: list[dict]) -> dict:
    result = {}
    for n_internal, runs in ((10, n10_runs), (20, n20_runs)):
        matches = [run for run in runs if run["label"] == FORMAL_IVM_LABEL]
        if len(matches) != 1:
            raise RuntimeError(
                f"N={n_internal} must contain exactly one formal IVM"
            )
        run = matches[0]
        expected_source = registered.IR_INPUTS[n_internal][
            registered.IRIVM_LABEL
        ][0]
        if Path(run["file"]) != expected_source:
            raise RuntimeError(
                f"N={n_internal} formal IVM does not use the new IR-IVM source"
            )
        result[f"N{n_internal}"] = {
            "figure_label": FORMAL_IVM_LABEL,
            "source_role": "new unified IR-IVM",
            "trajectory": str(expected_source),
            "trajectory_sha256": sha256(expected_source),
            "checkpoint_count": int(run["images_seq"].shape[0]),
            "terminal_nof": float(run["metrics"]["grad_count_seq"][-1]),
        }
    return {
        "series_count_per_row": 1,
        "article_ivm_included": False,
        "previous_irivm_included": False,
        "new_irivm_included_as_ivm": True,
        "by_resolution": result,
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
    registered.base.apply_force_axis(axes[0])
    namespace["plot_metric_on_axis"](
        ax=axes[1],
        runs=runs,
        metric_name="Delta_Energy",
        ylabel=namespace["METRIC_INFO"]["Delta_Energy"]["ylabel"],
        use_compressed_y=False,
        panel_letter="placeholder",
    )
    registered.base.apply_force_axis(axes[1])
    plot_action_on_axis(namespace, axes[2], runs)
    registered.base.apply_force_axis(axes[2])

    contour_runs, checkpoints = registered.base.contour_runs_at_budget(runs)
    namespace["draw_contour_with_paths"](
        ax=axes[3],
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
        figsize=(namespace["mm_to_inch"](240), namespace["mm_to_inch"](126)),
        dpi=220,
    )
    grid = GridSpec(
        2,
        4,
        figure=figure,
        width_ratios=(1.0, 1.0, 1.0, 1.0),
        left=0.052,
        right=0.952,
        bottom=0.08,
        top=0.90,
        wspace=0.43,
        hspace=0.12,
    )
    axes = [
        [figure.add_subplot(grid[row, column]) for column in range(4)]
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
    letter_rows = (("a", "b", "c", "d"), ("e", "f", "g", "h"))
    for row, (letters, row_title) in enumerate(zip(letter_rows, row_titles)):
        row_left = axes[row][0].get_position().x0
        row_right = axes[row][3].get_position().x1
        row_top = max(axis.get_position().y1 for axis in axes[row])
        header_y = row_top + 0.033
        header_center = 0.5 * (row_left + row_right)
        title_half_gap = 0.047
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
                position.x0 - 0.030,
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
        ACTION_COLUMN,
        "action_evaluation_interior_images",
        "action_evaluation_total_images",
        "action_path_sampling",
        "source_file",
        "source_sha256",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case_name, runs in rows_by_case:
            if case_name == "10 interior images":
                action_interior_images = PANEL_C_ACTION_INTERIOR_IMAGES
                action_path_sampling = "linear_interp_normalized_image_index"
            elif case_name == "20 interior images":
                action_interior_images = PANEL_G_ACTION_INTERIOR_IMAGES
                action_path_sampling = "native_stored_path"
            else:
                raise RuntimeError(f"unexpected action case: {case_name}")
            for run in runs:
                metrics = run["metrics"]
                file_path = Path(run["file"])
                file_hash = sha256(file_path)
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
                            ACTION_COLUMN: (
                                f"{float(metrics[ACTION_KEY][index]):.17g}"
                            ),
                            "action_evaluation_interior_images": (
                                action_interior_images
                            ),
                            "action_evaluation_total_images": (
                                action_interior_images + 2
                            ),
                            "action_path_sampling": action_path_sampling,
                            "source_file": str(file_path),
                            "source_sha256": file_hash,
                        }
                    )


def validate_registered_rows(path: Path) -> dict:
    with REGISTERED_SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        registered_rows = list(csv.DictReader(handle))
    with path.open(newline="", encoding="utf-8") as handle:
        new_rows = list(csv.DictReader(handle))
    expected_rows = []
    excluded_article = 0
    excluded_previous = 0
    relabeled_formal = 0
    for row in registered_rows:
        if row["method"] == FORMAL_IVM_LABEL:
            excluded_article += 1
            continue
        if row["method"] == registered.PREVIOUS_LABEL:
            excluded_previous += 1
            continue
        expected = dict(row)
        if row["method"] == registered.IRIVM_LABEL:
            expected["method"] = FORMAL_IVM_LABEL
            relabeled_formal += 1
        expected_rows.append(expected)
    if len(expected_rows) != len(new_rows):
        raise RuntimeError(
            "filtered registered and formal source row counts differ: "
            f"{len(expected_rows)} != {len(new_rows)}"
        )
    identity_fields = (
        "case",
        "method",
        "number_of_forces",
        "source_file",
        "source_sha256",
    )
    i_differences = []
    barrier_differences = []
    for index, (old, new) in enumerate(zip(expected_rows, new_rows)):
        for field in identity_fields:
            if old[field] != new[field]:
                raise RuntimeError(
                    f"registered row {index} differs in {field}: "
                    f"{old[field]!r} != {new[field]!r}"
                )
        if int(new["source_record_index"]) < 0:
            raise RuntimeError(f"negative source record index at row {index}")
        i_differences.append(abs(float(old["I_perp"]) - float(new["I_perp"])))
        barrier_differences.append(
            abs(
                float(old["Delta_E_dense_linear500"])
                - float(new["Delta_E_dense_linear500"])
            )
        )
    identity_counts = Counter(
        (row["case"], row["method"], row["number_of_forces"])
        for row in expected_rows
    )
    return {
        "registered_source_rows": len(registered_rows),
        "formal_figure_rows": len(new_rows),
        "excluded_article_ivm_rows": excluded_article,
        "excluded_previous_irivm_rows": excluded_previous,
        "new_irivm_rows_relabeled_ivm": relabeled_formal,
        "retained_baseline_rows": len(new_rows) - relabeled_formal,
        "duplicate_case_method_nof_rows_preserved": int(
            sum(count - 1 for count in identity_counts.values() if count > 1)
        ),
        "comparison_policy": (
            "ordered row-by-row after excluding article IVM and Previous "
            "IR-IVM and relabeling new IR-IVM as IVM; duplicates retained"
        ),
        "maximum_i_perp_difference": float(max(i_differences, default=0.0)),
        "maximum_delta_e_difference": float(
            max(barrier_differences, default=0.0)
        ),
    }


def main() -> int:
    arguments = parse_args()
    namespace = registered.base.load_formal_namespace(registered.base.N10_ROOT)
    n10_runs, n20_runs, formal_summaries = load_runs(namespace)
    action_audit = {
        "10_interior": attach_geometric_action(
            n10_runs,
            interpolate_to_interior_images=PANEL_C_ACTION_INTERIOR_IMAGES,
        ),
        "20_interior": attach_geometric_action(n20_runs),
    }
    formal_ivm_validation = validate_formal_ivm(n10_runs, n20_runs)
    offline_gradient_evaluations = int(
        sum(
            method_audit["offline_gradient_evaluations"]
            for case_audit in action_audit.values()
            for method_audit in case_audit.values()
        )
    )
    visible_offline_gradient_evaluations = int(
        sum(
            method_audit["visible_offline_gradient_evaluations"]
            for case_audit in action_audit.values()
            for method_audit in case_audit.values()
        )
    )
    action_gradient_evaluations = int(
        sum(
            method_audit["checkpoint_count"]
            * method_audit["action_total_images"]
            for case_audit in action_audit.values()
            for method_audit in case_audit.values()
        )
    )
    visible_action_gradient_evaluations = int(
        sum(
            method_audit["visible_checkpoint_count"]
            * method_audit["action_total_images"]
            for case_audit in action_audit.values()
            for method_audit in case_audit.values()
        )
    )
    tangent_audit_gradient_evaluations = int(
        sum(
            method_audit["checkpoint_count"]
            * method_audit["original_total_images"]
            for case_audit in action_audit.values()
            for method_audit in case_audit.values()
        )
    )
    visible_tangent_audit_gradient_evaluations = int(
        sum(
            method_audit["visible_checkpoint_count"]
            * method_audit["original_total_images"]
            for case_audit in action_audit.values()
            for method_audit in case_audit.values()
        )
    )

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
    registered_validation = validate_registered_rows(source_csv)

    q_path = (
        MAIN_REPOSITORY
        / "paper/ActiveExperiments/muller_q10_10d/code/inputs/Q_data.npz"
    )
    orthogonality_error = float(
        max(
            np.linalg.norm(Q.T @ Q - np.eye(10), ord=np.inf),
            np.linalg.norm(Q @ Q.T - np.eye(10), ord=np.inf),
        )
    )
    q_array_sha256 = hashlib.sha256(
        np.ascontiguousarray(Q).tobytes()
    ).hexdigest()
    if q_array_sha256 != EXPECTED_Q_ARRAY_SHA256:
        raise RuntimeError(
            f"unexpected Q array SHA-256: {q_array_sha256}"
        )
    if orthogonality_error > 1.0e-12:
        raise RuntimeError(
            f"Q orthogonality gate failed: {orthogonality_error:.3e}"
        )
    manifest = {
        "schema_version": 1,
        "status": "awaiting_visual_qa",
        "case": "fixed-Q unperturbed 10D Muller-Brown",
        "layout": "2x4; 10 interior images above 20 interior images",
        "xscale": "linear",
        "xlim": [0, 3000],
        "methods": [run["label"] for run in n10_runs],
        "ivm_series_policy": (
            "the new unified IR-IVM trajectory is the sole series labeled IVM; "
            "the article IVM and Previous IR-IVM are excluded"
        ),
        "cost_axis": (
            "direct sum of uncached force calls over all interior images; "
            "postprocessing-only metric evaluation is excluded"
        ),
        "geometric_action": {
            "continuous_definition": "S_hat(phi)=integral_phi |grad V| d ell",
            "discrete_definition": (
                "sum_i 0.5*(|grad V(phi_i)|+|grad V(phi_(i+1))|)"
                "*|phi_(i+1)-phi_i|"
            ),
            "quadrature": "segmentwise trapezoidal",
            "y_axis_scale": "linear",
            "evaluation_interior_images": {
                "panel_c_N_int_10": PANEL_C_ACTION_INTERIOR_IMAGES,
                "panel_g_N_int_20": PANEL_G_ACTION_INTERIOR_IMAGES,
            },
            "evaluation_total_images": {
                "panel_c_N_int_10": PANEL_C_ACTION_INTERIOR_IMAGES + 2,
                "panel_g_N_int_20": PANEL_G_ACTION_INTERIOR_IMAGES + 2,
            },
            "path_sampling": {
                "panel_c_N_int_10": (
                    "all six stored 12-point paths are coordinate-wise "
                    "linearly interpolated to 502 total points in the "
                    "normalized image-index parameter s in [0,1]"
                ),
                "panel_g_N_int_20": (
                    "the native stored 22-point paths are used unchanged"
                ),
                "endpoint_policy": "both stored endpoints are preserved exactly",
            },
            "coordinate_space": "10D transformed x space",
            "gradient": "analytic full grad_x V",
            "metric_evaluations_added_to_nof": 0,
            "actual_offline_gradient_function_calls_all_saved_checkpoints": (
                offline_gradient_evaluations
            ),
            "actual_offline_gradient_function_calls_with_nof_le_3000": (
                visible_offline_gradient_evaluations
            ),
            "logical_action_gradient_evaluations_all_saved_checkpoints": (
                action_gradient_evaluations
            ),
            "logical_action_gradient_evaluations_with_nof_le_3000": (
                visible_action_gradient_evaluations
            ),
            "logical_tangent_audit_gradient_evaluations_all_saved_checkpoints": (
                tangent_audit_gradient_evaluations
            ),
            "logical_tangent_audit_gradient_evaluations_with_nof_le_3000": (
                visible_tangent_audit_gradient_evaluations
            ),
            "tangent_role": (
                "the scalar line integral is tangent-independent; every path "
                "was separately audited by exactly recalculating I_perp with "
                "the solver's improved tangent"
            ),
            "tangent_definition": (
                "ASE Henkelman-Jonsson energy-weighted improved tangent"
            ),
            "central_difference_tangent_used": False,
            "interpretation_warning": (
                "Panel c uses 500 reporting-only interpolated interior action "
                "points; these add no optimization information. Panel g uses "
                "its native 20-interior-image path unchanged. Absolute action "
                "values across the two panels therefore use different reporting "
                "quadrature densities."
            ),
        },
        "potential_source": {
            "path": str(POTENTIAL_SOURCE),
            "sha256": sha256(POTENTIAL_SOURCE),
        },
        "tangent_source": {
            "path": str(TANGENT_SOURCE),
            "function": "improved_tangents_arbitrary",
            "sha256": sha256(TANGENT_SOURCE),
        },
        "q_matrix": {
            "path": str(q_path),
            "sha256": sha256(q_path),
            "array_bytes_sha256": q_array_sha256,
            "expected_array_bytes_sha256": EXPECTED_Q_ARRAY_SHA256,
            "maximum_orthogonality_error": orthogonality_error,
            "maximum_allowed_orthogonality_error": 1.0e-12,
            "coordinate_convention": "z=Q@x; x=Q.T@z",
        },
        "action_audit": action_audit,
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
        "registered_curve_validation": registered_validation,
        "registered_2x3_figure": {
            "path": str(registered.DEFAULT_OUTPUT),
            "sha256": sha256(registered.DEFAULT_OUTPUT),
            "source_csv": str(REGISTERED_SOURCE_CSV),
            "source_csv_sha256": sha256(REGISTERED_SOURCE_CSV),
        },
        "contour_methods": sorted(registered.base.CONTOUR_LABELS),
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
            "action_metadata_columns": {
                "value": ACTION_COLUMN,
                "interior_count": "action_evaluation_interior_images",
                "total_count": "action_evaluation_total_images",
                "sampling": "action_path_sampling",
            },
        },
        "plot_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
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
