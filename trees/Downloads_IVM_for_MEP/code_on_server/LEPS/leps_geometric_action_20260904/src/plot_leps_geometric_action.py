#!/usr/bin/env python3
"""Postprocess the six registered LEPS trajectories; never run an optimizer."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import sys

import ase
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]
LEPS = HERE.parents[2]
PROJECT = LEPS.parents[1]
REGISTERED = LEPS / "leps_irivm_sync1_refined_grid_20260830"
REGISTRY = REGISTERED / "reports/figure/leps_all_methods_metrics_provenance.json"
FORMAL = PROJECT / "Figures/LEPS/leps_all_methods_metrics.pdf"
EXPECTED_FORMAL_SHA = "081aad16011737c67d8d9eaded57559655f9b45f6ab18817940ca019c17f66b4"
REPORTS = ROOT / "reports"
OUTPUT = PROJECT / "output/pdf/leps_all_methods_geometric_action.pdf"
FIGURE = PROJECT / "Figures/LEPS/leps_all_methods_geometric_action.pdf"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(path):
    return {"path": str(path.relative_to(PROJECT)), "sha256": sha256(path)}


def quadrature(path, gradient):
    """Trapezoidal full-gradient integral on the un-densified polygon."""
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    norms = np.linalg.norm(gradient, axis=1)
    terms = 0.5 * (norms[:-1] + norms[1:]) * lengths
    weights = np.zeros(len(path))
    weights[:-1] += 0.5 * lengths
    weights[1:] += 0.5 * lengths
    np.testing.assert_allclose(terms.sum(), weights @ norms, rtol=1e-14, atol=1e-14)
    return float(terms.sum()), lengths, norms


def quadrature_tests():
    # Constant |g|=5 over a two-segment polygon of length 5: action=25.
    path = np.array([[0., 0.], [3., 0.], [3., 2.]])
    gradient = np.tile([3., 4.], (3, 1))
    assert quadrature(path, gradient)[0] == 25.0
    assert quadrature(path[::-1], gradient[::-1])[0] == 25.0
    assert quadrature(path, np.zeros_like(path))[0] == 0.0
    # Nonconstant node norms test endpoint half-weights explicitly.
    gradient = np.array([[1., 0.], [3., 0.], [5., 0.]])
    assert quadrature(path, gradient)[0] == 14.0


def main():
    quadrature_tests()
    registration = json.loads(REGISTRY.read_text())
    assert sha256(FORMAL) == EXPECTED_FORMAL_SHA == registration["figure_sha256"]
    original_csv = LEPS / registration["source_csv"]
    assert sha256(original_csv) == registration["source_csv_sha256"]
    selected_plot = LEPS / registration["plot_script"]
    assert sha256(selected_plot) == registration["plot_script_sha256"]
    frozen_plot = LEPS / registration["frozen_plot_primitives"]["path"]
    assert sha256(frozen_plot) == registration["frozen_plot_primitives"]["sha256"]
    spec = importlib.util.spec_from_file_location("registered_leps_plot", selected_plot)
    selected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selected)
    plot = selected.plot  # Import only; do not call main(), which replaces the old PDF.
    with original_csv.open(newline="") as handle:
        original_rows = list(csv.DictReader(handle))
    assert set(row["method"] for row in original_rows) == set(plot.METHOD_ORDER)

    rows, details, summary, plotted = [], {}, {}, {}
    source_paths = {"IVM": plot.WINNER, **plot.OTHER_METHODS}
    protected = {p: sha256(p) for p in [FORMAL, REGISTRY, original_csv, *source_paths.values()]}
    common_initial = None
    max_i_error = 0.0
    max_tangent_error = 0.0
    for name in plot.METHOD_ORDER:
        result = source_paths[name]
        registered_result = registration["plotted_results"][name]
        assert result == LEPS / registered_result["path"]
        result_sha = sha256(result)
        assert result_sha == registered_result["sha256"]
        nof, paths = (plot.ir_ivm_trace if name == "IVM" else plot.baseline_trace)(result)
        keep = nof <= plot.FORCE_MAX
        nof, paths = nof[keep], paths[keep]
        assert paths.shape == (len(nof), 22, 6)
        assert np.all(np.isfinite(paths)) and np.all(np.diff(nof) >= 0)
        np.testing.assert_array_equal(paths[:, :, [1, 2, 4, 5]], 0.0)
        if common_initial is None:
            common_initial = paths[0].copy()
        np.testing.assert_array_equal(paths[0], common_initial)
        previous = [row for row in original_rows if row["method"] == name]
        assert len(previous) == len(paths)
        action_values, lengths_values, norms_values, tangent_values, i_values = [], [], [], [], []
        for index, (cost, path, old) in enumerate(zip(nof, paths, previous)):
            assert index == int(old["plotted_index"])
            assert int(cost) == int(old["number_of_forces"])
            assert old["production_result_sha256"] == result_sha
            physical = plot.flat_to_physical(path)
            energies, gradient2 = plot.leps_energy_gradient(physical[:, 0], physical[:, 1])
            gradient = np.zeros_like(path)
            gradient[:, 0], gradient[:, 3] = gradient2[:, 0], gradient2[:, 1]
            tangents = plot.improved_tangents(path, energies)
            norm_t = np.linalg.norm(tangents, axis=1)
            np.testing.assert_allclose(norm_t, 1.0, rtol=0, atol=1e-12)
            max_tangent_error = max(max_tangent_error, float(np.max(np.abs(norm_t - 1))))
            parallel = np.sum(gradient[1:-1] * tangents, axis=1)
            normal = gradient[1:-1] - parallel[:, None] * tangents
            normal_squared = np.sum(normal * normal, axis=1)
            np.testing.assert_allclose(
                normal_squared + parallel**2, np.sum(gradient[1:-1]**2, axis=1),
                rtol=1e-12, atol=1e-14,
            )
            action, lengths, norms = quadrature(path, gradient)
            weights = 0.5 * (lengths[:-1] + lengths[1:])
            i_perp = float(weights @ normal_squared)
            old_i = float(old["i_perp"])
            np.testing.assert_allclose(i_perp, old_i, rtol=1e-11, atol=1e-14)
            max_i_error = max(max_i_error, abs(i_perp - old_i))
            rows.append({
                "method": name, "plotted_index": index, "number_of_forces": int(cost),
                "geometric_action_eV": action, "i_perp_alignment_check": i_perp,
                "path_length_angstrom": float(lengths.sum()), "node_count": len(path),
                "production_result": registered_result["path"],
                "production_result_sha256": registered_result["sha256"],
            })
            action_values.append(action)
            lengths_values.append(lengths)
            norms_values.append(norms)
            tangent_values.append(tangents)
            i_values.append(i_perp)
        action_values = np.asarray(action_values)
        key = name.lower().replace("-", "_").replace(" ", "_")
        details.update({
            key + "_nof": nof, key + "_action_eV": action_values,
            key + "_segment_lengths": np.asarray(lengths_values),
            key + "_gradient_norms": np.asarray(norms_values),
            key + "_improved_tangents": np.asarray(tangent_values),
            key + "_i_perp": np.asarray(i_values),
        })
        best = int(np.argmin(action_values))
        summary[name] = {
            "number_of_paths": len(paths), "first_nof": int(nof[0]),
            "last_plotted_nof": int(nof[-1]), "initial_action_eV": float(action_values[0]),
            "last_action_eV": float(action_values[-1]),
            "minimum_observed_action_eV": float(action_values[best]),
            "nof_at_minimum_observed_action": int(nof[best]),
        }
        plotted[name] = (nof, action_values)

    assert len(rows) == len(original_rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS / "leps_all_methods_geometric_action_source.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    arrays_path = REPORTS / "quadrature_and_tangent_audit.npz"
    np.savez_compressed(arrays_path, **details)
    summary_path = REPORTS / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8, "legend.fontsize": 6.6, "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.75, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(130 / 25.4, 82 / 25.4))
    for name in plot.METHOD_ORDER:
        nof, action = plotted[name]
        marker, size = plot.MARKERS[name]
        ax.plot(nof, action, label=name, color=plot.COLORS[name], linestyle=plot.STYLES[name],
                linewidth=2.05 if name == "IVM" else 1.5, marker=marker, markersize=size,
                markerfacecolor="white", markeredgewidth=0.7,
                markevery=1 if name == "IVM" else max(len(nof) // 8, 1),
                zorder=30 if name == "IVM" else 5)
    ax.set_xlim(0, plot.FORCE_MAX)
    ax.set_xticks(plot.FORCE_TICKS)
    ax.set_ylim(0, max(float(values.max()) for _, values in plotted.values()) * 1.06)
    ax.set_xlabel("Number of Forces")
    ax.set_ylabel(r"Discrete geometric action $\widehat{S}$ (eV)")
    ax.grid(False)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.55, 0.99),
               ncol=3, columnspacing=1.3, handlelength=2.35, handletextpad=0.5)
    fig.subplots_adjust(left=0.115, right=0.965, bottom=0.16, top=0.81)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, metadata={"Title": "LEPS discrete geometric action versus Number of Forces",
                                 "CreationDate": None, "ModDate": None})
    plt.close(fig)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUTPUT, FIGURE)
    assert sha256(OUTPUT) == sha256(FIGURE)
    for path, original_hash in protected.items():
        assert sha256(path) == original_hash, f"input unexpectedly changed: {path}"

    manifest = {
        "date": "2026-09-04", "purpose": "postprocessing only; no optimization or reselection",
        "figure": identity(FIGURE), "output_copy": identity(OUTPUT),
        "script": identity(HERE), "original_figure_unchanged": identity(FORMAL),
        "original_registry": identity(REGISTRY), "original_plotted_points": identity(original_csv),
        "original_plot_script": identity(selected_plot), "frozen_plot_primitives": identity(frozen_plot),
        "leps_energy_gradient_and_tangent_code": identity(LEPS / "src/leps_core.py"),
        "raw_results": {name: identity(path) for name, path in source_paths.items()},
        "source_csv": identity(csv_path), "quadrature_and_tangent_audit": identity(arrays_path),
        "summary": identity(summary_path),
        "ivm_parameters": {"alpha": 100, "r": 0.003, "m_max": 5},
        "metric_contract": {
            "definition": "integral of full Euclidean gradient norm over path arc length",
            "quadrature": "sum_i 0.5*(norm(g_i)+norm(g_{i+1}))*norm(x_{i+1}-x_i)",
            "nodes": "original 22 nodes, including both endpoints; no densification",
            "coordinates": "same flat 6D coordinates as LEPS; active coordinates 0 and 3",
            "tangent": "leps_core.improved_tangents / ASE Henkelman-Jonsson ImprovedTangentMethod",
            "tangent_role": "original I_perp alignment audit only; full-gradient action is tangent-independent",
            "nof": "exactly the original figure's optimization ledger; postprocessing excluded",
            "curves": "all original plotted states <=20000 NoF, raw values, no best-so-far envelope or extrapolation",
            "ivm_last_path_nof": 1528, "ivm_terminal_optimization_nof": 1558,
            "caution": "coarse trapezoidal action is not an exact continuum integral or convergence certificate",
        },
        "checks": {
            "all_raw_result_hashes_match_registration": True,
            "all_inputs_preserved": True, "all_nof_points_match_original_csv": True,
            "all_methods_share_initial_path": True, "quadrature_unit_tests_pass": True,
            "node_weight_and_segment_sum_agree": True,
            "max_i_perp_absolute_disagreement": max_i_error,
            "max_improved_tangent_unit_norm_error": max_tangent_error,
            "total_paths": len(rows), "diagnostic_gradient_node_evaluations": len(rows) * 22,
        },
        "runtime": {"python": sys.version, "executable": sys.executable,
                    "platform": platform.platform(), "numpy": np.__version__,
                    "matplotlib": matplotlib.__version__, "ase": ase.__version__},
    }
    (REPORTS / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"figure": str(FIGURE), "checks": manifest["checks"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
