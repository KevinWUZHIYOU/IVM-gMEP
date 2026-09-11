#!/usr/bin/env python3
"""Reproducibly finalize the audited Chignolin transactional-alpha study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


BASELINE_KEYS = {
    "ODE-NEB": "confirmed1000_ode_k50",
    "Static-NEB": "confirmed1000_static_k40_a0p012575",
    "Original String": "confirmed1000_original_dt0p012575",
    "OBSM-FIRE": "confirmed1000_obsm_l0p9_n5_dt0p12",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_checked(
    command: list[str],
    cwd: Path,
    log: Path,
    environment: dict[str, str],
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            f"Command failed with status {completed.returncode}; see {log}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--namespace",
        default="transactional_alpha_final_20260729",
    )
    # Each OpenMM CPU context creates its own worker pool.  Running several
    # diagnostic contexts concurrently severely oversubscribes the eight-core
    # workstation and is slower than one context at a time.  This changes only
    # wall-clock scheduling: every diagnostic coordinate and counter remains
    # identical.
    parser.add_argument("--outer-dense-workers", type=int, default=1)
    parser.add_argument("--standard-dense-workers", type=int, default=1)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    src = root / "src"
    report = root / "reports" / args.namespace
    figures = root / "figures" / args.namespace
    logs = root / "logs" / args.namespace
    report.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(src)
    for variable in (
        "OPENMM_CPU_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        environment[variable] = "1"

    preserved_cap_manifest = (
        root
        / "reports"
        / "preserved_cap_assets_20260729"
        / "PREEXISTING_CAP_ASSETS_SHA256.json"
    )
    cap_verify_command = [
        str(python),
        str(src / "preserve_cap_assets_20260729.py"),
        "--root",
        str(root),
        "--manifest",
        str(preserved_cap_manifest),
        "--mode",
        "verify",
    ]
    run_checked(
        cap_verify_command,
        root,
        logs / "00_preserved_cap_assets_before.log",
        environment,
    )

    old_manifest = (
        root
        / "reports"
        / "escape_relock_alpha_20260729a"
        / "alpha_branch_reuse_classification.json"
    )
    alpha_manifest_dir = report / "alpha_grid"
    audit_command = [
        str(python),
        str(src / "audit_transactional_alpha_grid_20260729.py"),
        "--root",
        str(root),
        "--old-manifest",
        str(old_manifest),
        "--output-dir",
        str(alpha_manifest_dir),
    ]
    run_checked(audit_command, root, logs / "01_alpha_audit.log", environment)
    alpha_manifest = alpha_manifest_dir / "manifest.json"

    trailing_dense_dir = report / "trailing_dense"
    dense_command = [
        str(python),
        str(src / "run_v2_trailing_dense_grid_20260729.py"),
        "--root",
        str(root),
        "--manifest",
        str(alpha_manifest),
        "--report-dir",
        str(trailing_dense_dir),
        "--python",
        str(python),
        "--outer-workers",
        str(max(1, int(args.outer_dense_workers))),
    ]
    run_checked(
        dense_command, root, logs / "02_trailing_dense.log", environment
    )
    trailing_dense_manifest = trailing_dense_dir / "dense_manifest.json"

    baseline_summary = (
        root
        / "figures"
        / "article_update_20260728e"
        / "chignolin_method_comparison_summary.json"
    )
    score_command = [
        str(python),
        str(src / "score_v2_alpha_grid_20260729.py"),
        "--manifest",
        str(alpha_manifest),
        "--dense-manifest",
        str(trailing_dense_manifest),
        "--method-summary",
        str(baseline_summary),
        "--output-dir",
        str(report),
    ]
    run_checked(score_command, root, logs / "03_score.log", environment)
    score_file = report / "V2_ALPHA_SCORES.json"
    scores = json.loads(score_file.read_text())
    if scores["selection_rejected"] or scores["selected_record"] is None:
        raise RuntimeError("No production-eligible transactional alpha")
    if not scores["i_perp_strict_single_well_all_budgets"]:
        raise RuntimeError(
            "The raw connected I_perp curves do not form one strict "
            "interior well at every preregistered force budget"
        )
    selected = scores["selected_record"]
    selected_result = Path(selected["optimization_result"]).resolve()
    if sha256(selected_result) != selected["optimization_result_sha256"]:
        raise RuntimeError("Selected IVM result changed after scoring")

    alpha_plot_command = [
        str(python),
        str(src / "plot_v2_alpha_well_20260729.py"),
        "--scores",
        str(score_file),
        "--output-dir",
        str(figures),
    ]
    run_checked(
        alpha_plot_command, root, logs / "04_alpha_plot.log", environment
    )

    production_manifest_path = (
        root
        / "reports"
        / "confirmed1000_productions_20260728c"
        / "manifest.json"
    )
    production_manifest = json.loads(production_manifest_path.read_text())
    production_by_name = {
        record["name"]: Path(record["result_file"]).resolve()
        for record in production_manifest["records"]
    }
    standardized_namespace = f"{args.namespace}_dense48"
    standard_dense_command = [
        str(python),
        str(src / "run_standardized_dense48_20260729.py"),
        "--root",
        str(root),
        "--namespace",
        standardized_namespace,
        "--python",
        str(python),
        "--workers",
        str(max(1, int(args.standard_dense_workers))),
        "--record",
        f"IVM={selected_result}",
    ]
    for method, key in BASELINE_KEYS.items():
        standard_dense_command.extend(
            ["--record", f"{method}={production_by_name[key]}"]
        )
    run_checked(
        standard_dense_command,
        root,
        logs / "05_standardized_dense48.log",
        environment,
    )
    standardized_manifest = (
        root
        / "reports"
        / standardized_namespace
        / "dense48_manifest.json"
    )

    method_plot_command = [
        str(python),
        str(src / "plot_paper_method_comparison_v2_20260729.py"),
        "--root",
        str(root),
        "--output-directory",
        str(figures),
        "--ivm-result",
        str(selected_result),
        "--ivm-selection-report",
        str(score_file),
        "--standardized-dense-manifest",
        str(standardized_manifest),
    ]
    run_checked(
        method_plot_command, root, logs / "06_method_plots.log", environment
    )
    run_checked(
        cap_verify_command,
        root,
        logs / "07_preserved_cap_assets_after.log",
        environment,
    )

    required = [
        figures / "chignolin_main_metrics_total_nf.pdf",
        figures / "chignolin_cost_to_precision_total_nf.pdf",
        figures
        / "chignolin_ivm_fixed_alpha_sensitivity_total_nf.pdf",
    ]
    if any(not path.exists() for path in required):
        raise RuntimeError(f"Missing required figure: {required}")
    required_audit_artifacts = [
        figures / "chignolin_force_call_audit_total_nf.pdf",
        figures
        / "data"
        / "chignolin_force_call_audit_total_nf.csv",
    ]
    if any(not path.exists() for path in required_audit_artifacts):
        raise RuntimeError(
            f"Missing required force-call audit: {required_audit_artifacts}"
        )
    method_summary = json.loads(
        (
            figures / "chignolin_method_comparison_summary.json"
        ).read_text()
    )
    if (
        method_summary["primary_cost_axis"]
        != "counter_vectors.sum(axis=1)"
        or method_summary["sparse_22_image_barrier_used_in_figures_or_scores"]
        or not method_summary[
            "dense_diagnostic_calls_excluded_from_optimization_axis"
        ]
    ):
        raise RuntimeError("Final method summary violates cost/barrier policy")

    completion = {
        "namespace": args.namespace,
        "selected_alpha": selected["alpha"],
        "selected_result": str(selected_result),
        "selected_result_sha256": sha256(selected_result),
        "alpha_manifest": str(alpha_manifest),
        "alpha_manifest_sha256": sha256(alpha_manifest),
        "trailing_dense_manifest": str(trailing_dense_manifest),
        "trailing_dense_manifest_sha256": sha256(
            trailing_dense_manifest
        ),
        "standardized_dense_manifest": str(standardized_manifest),
        "standardized_dense_manifest_sha256": sha256(
            standardized_manifest
        ),
        "score_file": str(score_file),
        "score_file_sha256": sha256(score_file),
        "required_figures": [
            {"path": str(path), "sha256": sha256(path)}
            for path in required
        ],
        "required_force_call_audits": [
            {"path": str(path), "sha256": sha256(path)}
            for path in required_audit_artifacts
        ],
        "barrier_policy": (
            "Real OpenMM CPU-double energy on 10/20 Cartesian-linear "
            "subdivisions per segment, adaptively 40/80 when successive "
            "barriers differ by >0.05 kcal/mol; diagnostic calls excluded "
            "from Number of Forces."
        ),
        "sparse_22_image_barrier_used": False,
        "preserved_cap_assets": {
            "manifest": str(preserved_cap_manifest),
            "manifest_sha256": sha256(preserved_cap_manifest),
            "verified_before_and_after_pipeline": True,
        },
        "source_files": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (
                src / "chignolin_core.py",
                src / "run_ivm_early_block_relock_v2_20260729.py",
                src / "run_ivm_early_block_relock_v3_20260729.py",
                src / "audit_transactional_alpha_grid_20260729.py",
                src / "postprocess_trailing_dense_barrier_v2_20260729.py",
                src / "run_v2_trailing_dense_grid_20260729.py",
                src / "score_v2_alpha_grid_20260729.py",
                src / "plot_v2_alpha_well_20260729.py",
                src / "postprocess_dense_barrier.py",
                src / "run_standardized_dense48_20260729.py",
                src / "plot_paper_method_comparison_v2_20260729.py",
                src / "preserve_cap_assets_20260729.py",
                Path(__file__).resolve(),
                root
                / "logs"
                / "fixed_alpha_v2_universal_20260729"
                / "launch_v2_fixed_alpha.zsh",
            )
        ],
        "commands": {
            "alpha_audit": audit_command,
            "trailing_dense": dense_command,
            "score": score_command,
            "alpha_plot": alpha_plot_command,
            "standardized_dense48": standard_dense_command,
            "method_plots": method_plot_command,
            "preserved_cap_assets_verify": cap_verify_command,
        },
    }
    output = report / "FINAL_PIPELINE_MANIFEST.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(completion, indent=2) + "\n")
    os.replace(temporary, output)
    print(output)


if __name__ == "__main__":
    main()
