#!/usr/bin/env python3
"""Audit the final Chignolin V9 data, figures, and accounting bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def audit_optimization(path: Path) -> dict:
    audit = load(path)
    per_image = [int(value) for value in audit["per_image_joint_evaluations"]]
    backend = int(audit["backend_raw_joint_evaluations"])
    summed = int(audit["sum_image_joint_evaluations"])
    assert audit["consistent"]
    assert len(per_image) == 22
    assert backend == summed == sum(per_image)
    assert all(audit["per_image_request_identity_ok"])
    assert all(audit["per_image_query_outcome_identity_ok"])
    return {
        "audit": str(path),
        "sha256": sha256(path),
        "backend_joint_evaluations": backend,
        "sum_per_image_joint_evaluations": sum(per_image),
        "n_image_counters": len(per_image),
    }


def audit_dense(path: Path) -> dict:
    audit = load(path)
    backend = int(audit["diagnostic_backend_raw_joint_evaluations"])
    misses = int(audit["diagnostic_evaluator_cache_misses"])
    assert backend == misses
    assert audit["counts_excluded_from_optimization_axis"]
    assert int(audit["initial_subdivisions"]) == 10
    assert int(audit["comparison_subdivisions"]) == 20
    assert float(audit["adaptive_threshold_kcal"]) == 0.05
    assert int(audit["maximum_subdivisions"]) == 80
    return {
        "audit": str(path),
        "sha256": sha256(path),
        "diagnostic_backend_joint_evaluations": backend,
        "diagnostic_cache_misses": misses,
        "excluded_from_optimization_cost": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--article-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    article = args.article_root.expanduser().resolve()

    grid_path = (
        root
        / "reports/v9_kkt0p7_cap30_final_grid_20260730"
        / "v9_kkt0p7_cap30_grid_summary.json"
    )
    dense_path = (
        root
        / "reports/v9_kkt0p7_cap30_final_dense_20260730"
        / "dense_manifest.json"
    )
    score_path = (
        root
        / "reports/v9_kkt0p7_cap30_final_selection_20260730"
        / "V9_ALPHA_FINAL_WINDOW_SCORES.json"
    )
    method_path = (
        root
        / "figures/article_update_20260730_v9_final"
        / "chignolin_method_comparison_summary.json"
    )
    preserve_path = (
        root
        / "reports/preserved_cap_assets_20260729"
        / "PREEXISTING_CAP_ASSETS_SHA256.json"
    )
    grid = load(grid_path)
    dense = load(dense_path)
    score = load(score_path)
    methods = load(method_path)

    assert grid["alpha_grid"] == [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    assert grid["tangent_definition"].startswith(
        "normalized central difference"
    )
    assert not grid["projection_uses_spline"]
    assert not grid["sparse_barrier_used_for_scoring"]
    assert "final 30%" in grid["primary_alpha_sensitivity_statistic"]
    assert Path(dense["source_grid_summary"]).name == grid_path.name
    assert Path(score["source_grid_summary"]).name == grid_path.name
    assert sha256(grid_path) == dense["source_grid_summary_sha256"]
    assert sha256(grid_path) == score["source_grid_summary_sha256"]
    assert not dense["sparse_22_image_barrier_used"]
    assert dense["diagnostic_calls_excluded_from_optimization_axis"]
    assert "10/20" in dense["barrier_protocol"]
    assert "40/80" in dense["barrier_protocol"]
    assert float(score["selected_alpha"]) == 1.0
    assert not score["selection_rejected"]
    assert score["i_perp_unique_interior_well_at_alpha_1_all_budgets"]
    assert score["dense_barrier_unique_moderate_interior_well_all_budgets"]
    assert not methods["sparse_22_image_barrier_used_in_figures_or_scores"]
    assert methods["cost_includes_all_22_images"]
    assert methods["dense_diagnostic_calls_excluded_from_optimization_axis"]
    assert (
        methods["selected_ivm"]["selection_provenance"]["sha256"]
        == sha256(score_path)
    )

    optimization_audits = [
        audit_optimization(Path(record["audit"]))
        for record in grid["records"]
    ]
    selected_method_audits = [
        audit_optimization(Path(record["run_audit"]))
        for record in methods["methods"].values()
    ]
    dense_audits = [
        audit_dense(Path(record["dense_audit"]))
        for record in dense["records"]
    ]
    selected_method_dense_audits = [
        audit_dense(Path(record["dense_audit"]))
        for record in methods["methods"].values()
    ]
    assert sum(
        row["diagnostic_backend_joint_evaluations"] for row in dense_audits
    ) == int(dense["total_diagnostic_joint_evaluations"])

    preserve = load(preserve_path)
    for record in preserve["records"]:
        path = root / record["relative_path"]
        assert path.is_file()
        assert path.stat().st_size == int(record["size_bytes"])
        assert sha256(path) == record["sha256"]

    figures = {}
    figure_names = (
        "chignolin_main_metrics_total_nf",
        "chignolin_cost_to_precision_total_nf",
        "chignolin_ivm_fixed_alpha_sensitivity_total_nf",
    )
    figure_root = root / "figures/article_update_20260730_v9_final"
    article_figure_root = article / "Figures/chignolin"
    for name in figure_names:
        source = figure_root / f"{name}.pdf"
        installed = article_figure_root / f"{name}_v9_20260730.pdf"
        assert source.is_file() and installed.is_file()
        assert sha256(source) == sha256(installed)
        figures[name] = {
            "source": str(source),
            "source_sha256": sha256(source),
            "article_copy": str(installed),
            "article_copy_sha256": sha256(installed),
        }

    main_tex = article / "main.tex"
    main_pdf = article / "main.pdf"
    assert main_tex.is_file() and main_pdf.is_file()
    assert main_pdf.stat().st_size > 1_000_000
    report = {
        "verified": True,
        "cost_axis": "all 22 image joint energy-force evaluations",
        "initial_path_sha256": grid["initial_path_sha256"],
        "tangent": grid["tangent_definition"],
        "selected_alpha": score["selected_alpha"],
        "sparse_barrier_used": False,
        "dense_barrier_protocol": dense["barrier_protocol"],
        "dense_diagnostic_calls_excluded_from_optimization_axis": True,
        "optimization_alpha_grid_audits": optimization_audits,
        "selected_method_optimization_audits": selected_method_audits,
        "alpha_grid_dense_audits": dense_audits,
        "selected_method_dense_audits": selected_method_dense_audits,
        "alpha_grid_dense_diagnostic_total": int(
            dense["total_diagnostic_joint_evaluations"]
        ),
        "preserved_cap_assets": {
            "verified": True,
            "file_count": len(preserve["records"]),
            "manifest": str(preserve_path),
            "manifest_sha256": sha256(preserve_path),
        },
        "figures": figures,
        "article": {
            "main_tex": str(main_tex),
            "main_tex_sha256": sha256(main_tex),
            "main_pdf": str(main_pdf),
            "main_pdf_sha256": sha256(main_pdf),
        },
        "source_reports": {
            "grid": str(grid_path),
            "grid_sha256": sha256(grid_path),
            "dense": str(dense_path),
            "dense_sha256": sha256(dense_path),
            "selection": str(score_path),
            "selection_sha256": sha256(score_path),
            "methods": str(method_path),
            "methods_sha256": sha256(method_path),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
