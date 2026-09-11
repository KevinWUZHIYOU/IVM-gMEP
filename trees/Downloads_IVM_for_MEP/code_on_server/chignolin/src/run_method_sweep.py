#!/usr/bin/env python3
"""Restart-safe Chignolin screening for Static-NEB, String, and OBSM.

Screening ranks complete fixed-parameter trajectories using the audited total
number of force evaluations, ``counter_vectors.sum(axis=1)``.  A screening
trajectory is never promoted directly into the paper comparison: the selected
configuration must be rerun independently as a production trajectory.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


INITIAL_PATH_SHA256 = (
    "57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log_auc(cost: np.ndarray, values: np.ndarray) -> float:
    cost = np.asarray(cost, dtype=float)
    values = np.maximum(np.asarray(values, dtype=float), 1e-300)
    if cost.size < 2 or cost[-1] <= cost[0]:
        return float("inf")
    return float(
        np.trapz(np.log10(values), cost) / float(cost[-1] - cost[0])
    )


def broad_design(family: str) -> list[dict]:
    if family == "static_neb_ultrafine":
        spring_constants = (34.0, 36.0, 38.0, 40.0, 42.0, 44.0)
        alphas = np.arange(0.01245, 0.01270 + 0.0000125, 0.000025)
        return [
            {
                "method": "static_neb",
                "spring_k": k,
                "alpha": float(alpha),
            }
            for k in spring_constants
            for alpha in alphas
        ]
    if family == "original_string_ultrafine":
        dts = np.arange(0.01235, 0.01275 + 0.0000125, 0.000025)
        return [
            {
                "method": "original_string",
                "dt": float(dt),
                "reparam_interval": 1,
            }
            for dt in dts
        ]
    if family == "static_neb_boundary":
        # The previous refinement stopped at alpha=0.0125 although k<=35
        # remained stable there.  Resolve the actual stability boundary with
        # a 1e-4 grid, while retaining enough k values to separate the normal
        # relaxation rate from spring-controlled parameterization quality.
        spring_constants = (10.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0)
        alphas = np.arange(0.0124, 0.0140 + 0.00005, 0.0001)
        return [
            {
                "method": "static_neb",
                "spring_k": k,
                "alpha": float(alpha),
            }
            for k in spring_constants
            for alpha in alphas
        ]
    if family == "original_string_boundary":
        # The previous grid had no point between the stable dt=0.0125 and
        # unstable dt=0.0138237 runs.
        dts = np.arange(0.0124, 0.0138 + 0.00005, 0.0001)
        return [
            {
                "method": "original_string",
                "dt": float(dt),
                "reparam_interval": 1,
            }
            for dt in dts
        ]
    if family == "static_neb_broad":
        spring_constants = (
            0.1,
            1.0,
            5.0,
            10.0,
            20.0,
            35.0,
            50.0,
            80.0,
            100.0,
            200.0,
            500.0,
            1000.0,
            2000.0,
        )
        alphas = (
            1e-5,
            3e-5,
            1e-4,
            3e-4,
            6e-4,
            1e-3,
            1.5e-3,
            2e-3,
            3e-3,
            5e-3,
            7.5e-3,
            1e-2,
        )
        return [
            {"method": "static_neb", "spring_k": k, "alpha": alpha}
            for k in spring_constants
            for alpha in alphas
        ]
    if family == "original_string_broad":
        dts = sorted(
            {
                *np.geomspace(1e-5, 3e-2, 32).tolist(),
                1e-2,
            }
        )
        return [
            {
                "method": "original_string",
                "dt": float(dt),
                "reparam_interval": 1,
            }
            for dt in dts
        ]
    if family == "obsm_fire_broad":
        return [
            {
                "method": "obsm_fire",
                "lambda_mix": lambda_mix,
                "inner_steps": inner_steps,
                "fire_dt": fire_dt,
                "fire_beta": 1.0,
                "fire_mass": 1.0,
                "reparam_interval": 1,
            }
            for lambda_mix in (0.1, 0.25, 0.5, 0.75, 0.9)
            for inner_steps in (1, 2, 5, 10, 20, 50)
            for fire_dt in (0.0025, 0.005, 0.01, 0.02, 0.03, 0.05)
        ]
    if family == "obsm_fire_dt_extension":
        return [
            {
                "method": "obsm_fire",
                "lambda_mix": lambda_mix,
                "inner_steps": inner_steps,
                "fire_dt": fire_dt,
                "fire_beta": 1.0,
                "fire_mass": 1.0,
                "reparam_interval": 1,
            }
            for lambda_mix in (0.75, 0.9, 0.95)
            for inner_steps in (3, 5, 7, 10, 15, 20, 30)
            for fire_dt in (0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12)
        ]
    if family == "obsm_fire_dt_high_extension":
        return [
            {
                "method": "obsm_fire",
                "lambda_mix": lambda_mix,
                "inner_steps": inner_steps,
                "fire_dt": fire_dt,
                "fire_beta": 1.0,
                "fire_mass": 1.0,
                "reparam_interval": 1,
            }
            for lambda_mix in (0.9, 0.95)
            for inner_steps in (5, 7, 10, 15, 20)
            for fire_dt in (0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20)
        ]
    raise ValueError(f"Unknown broad family: {family}")


def parent_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return [
        row
        for row in payload["records"]
        if row.get("stable") and row.get("selection_rank") is not None
    ]


def refined_design(family: str, parent: Path) -> list[dict]:
    rows = parent_rows(parent)
    configurations = []
    if family == "static_neb_fine":
        # Directly extend the best distinct fixed configurations.  Do not
        # perturb parameters again: the parent boundary grid is already much
        # finer than the numerical resolution needed for the production
        # comparison.
        for row in rows[:24]:
            configurations.append(dict(row["configuration"]))
    elif family == "original_string_fine":
        for row in rows[:8]:
            configurations.append(dict(row["configuration"]))
    elif family == "static_neb_refine":
        for row in rows[:16]:
            base = row["configuration"]
            for scale in (0.75, 0.875, 1.0, 1.125, 1.25):
                configurations.append(
                    {
                        "method": "static_neb",
                        "spring_k": float(base["spring_k"]),
                        "alpha": float(base["alpha"]) * scale,
                    }
                )
    elif family == "original_string_refine":
        for row in rows[:3]:
            base = row["configuration"]
            for scale in np.linspace(0.70, 1.30, 13):
                configurations.append(
                    {
                        "method": "original_string",
                        "dt": float(base["dt"]) * float(scale),
                        "reparam_interval": 1,
                    }
                )
    elif family == "obsm_fire_refine":
        for row in rows[:12]:
            base = row["configuration"]
            for beta in (0.1, 0.3, 0.5, 1.0, 2.0):
                for reparam_interval in (1, 2, 5):
                    configurations.append(
                        {
                            "method": "obsm_fire",
                            "lambda_mix": float(base["lambda_mix"]),
                            "inner_steps": int(base["inner_steps"]),
                            "fire_dt": float(base["fire_dt"]),
                            "fire_beta": beta,
                            "fire_mass": 1.0,
                            "reparam_interval": reparam_interval,
                        }
                    )
    else:
        raise ValueError(f"Unknown refined family: {family}")
    unique = {}
    for configuration in configurations:
        key = json.dumps(configuration, sort_keys=True)
        unique[key] = configuration
    return [unique[key] for key in sorted(unique)]


def tag_number(value: float) -> str:
    return f"{float(value):.12g}".replace(".", "p").replace("-", "m")


def expected_glob(root: Path, namespace: str, label: str, config: dict) -> str:
    if config["method"] == "static_neb":
        return str(root / "runs" / "neb" / f"neb_{label}_central_*.npz")
    return str(root / "runs" / namespace / f"{namespace}_{label}_*.npz")


def command_for(
    python: Path,
    root: Path,
    namespace: str,
    label: str,
    budget: int,
    config: dict,
) -> list[str]:
    common = [
        str(python),
    ]
    if config["method"] == "static_neb":
        return common + [
            str(root / "src" / "run_neb.py"),
            "--root",
            str(root),
            "--initial-path-file",
            "inputs/common_initial_path_equidistant.npz",
            "--run-label",
            label,
            "--spring-k",
            f"{float(config['spring_k']):.12g}",
            "--alpha",
            f"{float(config['alpha']):.12g}",
            "--optimizer",
            "static",
            "--budget",
            str(budget),
            "--target-fmax",
            "0",
        ]
    if config["method"] == "original_string":
        return common + [
            str(root / "src" / "run_string_family.py"),
            "--root",
            str(root),
            "--initial-path-file",
            "inputs/common_initial_path_equidistant.npz",
            "--method",
            "original",
            "--run-label",
            label,
            "--output-namespace",
            namespace,
            "--dt",
            f"{float(config['dt']):.12g}",
            "--reparam-interval",
            str(int(config["reparam_interval"])),
            "--string-length-tol",
            "0",
            "--budget",
            str(budget),
        ]
    if config["method"] == "obsm_fire":
        return common + [
            str(root / "src" / "run_obsm.py"),
            "--root",
            str(root),
            "--initial-path-file",
            "inputs/common_initial_path_equidistant.npz",
            "--solver",
            "fire",
            "--run-label",
            label,
            "--output-namespace",
            namespace,
            "--lambda-mix",
            f"{float(config['lambda_mix']):.12g}",
            "--inner-mode",
            "few_steps",
            "--inner-steps",
            str(int(config["inner_steps"])),
            "--reparam-interval",
            str(int(config["reparam_interval"])),
            "--fire-dt",
            f"{float(config['fire_dt']):.12g}",
            "--fire-beta",
            f"{float(config['fire_beta']):.12g}",
            "--fire-mass",
            f"{float(config['fire_mass']):.12g}",
            "--budget",
            str(budget),
        ]
    raise ValueError(f"Unknown method: {config['method']}")


def run_one(
    *,
    python: Path,
    root: Path,
    namespace: str,
    stage: str,
    index: int,
    budget: int,
    config: dict,
) -> dict:
    label = f"{stage}_{index:04d}"
    output_glob = expected_glob(root, namespace, label, config)
    existing = sorted(Path(root).glob(str(Path(output_glob).relative_to(root))))
    log_dir = root / "logs" / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}.log"
    command = command_for(python, root, namespace, label, budget, config)
    if existing:
        return {
            "index": index,
            "configuration": config,
            "status": "reused",
            "result_file": str(existing[-1].resolve()),
            "log_file": str(log_path.resolve()),
            "command": command,
        }
    environment = os.environ.copy()
    environment["OPENMM_CPU_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["PYTHONPATH"] = str(root / "src")
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    outputs = sorted(Path(root).glob(str(Path(output_glob).relative_to(root))))
    return {
        "index": index,
        "configuration": config,
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "result_file": str(outputs[-1].resolve()) if outputs else None,
        "log_file": str(log_path.resolve()),
        "command": command,
    }


def summarize(record: dict) -> dict:
    row = dict(record)
    result_text = row.get("result_file")
    if not result_text:
        row.update({"stable": False, "failure": "no result file"})
        return row
    result = Path(result_text)
    audit_path = result.with_suffix(".audit.json")
    if not audit_path.exists():
        row.update({"stable": False, "failure": "no audit file"})
        return row
    try:
        audit = json.loads(audit_path.read_text())
        with np.load(result, allow_pickle=False) as saved:
            counters = np.asarray(saved["counter_vectors"], dtype=int)
            total_force = counters.sum(axis=1)
            i_perp = np.asarray(saved["i_perp"], dtype=float)
            barrier = np.asarray(saved["barrier_kcal"], dtype=float)
            fmax = np.asarray(saved["physical_fmax"], dtype=float)
            equidistance = np.asarray(
                saved["equidistance_integral"], dtype=float
            )
            segment_cv = np.asarray(saved["segment_length_cv"], dtype=float)
            calls = np.asarray(saved["calls_per_image"], dtype=int)
            stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
            final_path = np.asarray(saved["final_path"], dtype=float)
            stored_paths = np.asarray(saved["stored_paths"], dtype=float)
            tangent = str(saved["tangent_definition"])
            initial_hash = str(saved["initial_path_sha256"])
            stop_reason = str(saved["stop_reason"])
        finite = bool(
            np.all(np.isfinite(i_perp))
            and np.all(np.isfinite(barrier))
            and np.all(np.isfinite(fmax))
            and np.all(np.isfinite(equidistance))
            and np.all(np.isfinite(segment_cv))
        )
        counter_consistent = bool(
            audit.get("consistent", False)
            and audit.get("backend_raw_joint_evaluations")
            == audit.get("sum_image_joint_evaluations")
        )
        final_is_evaluated = bool(
            stored_calls.size
            and int(stored_calls[-1]) == int(calls[-1])
            and np.array_equal(final_path, stored_paths[-1])
        )
        improved = bool(
            i_perp[-1] < i_perp[0] and barrier[-1] < barrier[0]
        )
        stable = bool(
            finite
            and counter_consistent
            and final_is_evaluated
            and initial_hash == INITIAL_PATH_SHA256
            and "central difference" in tangent.lower()
            and improved
            and i_perp[-1] <= 1.5 * np.min(i_perp)
            and barrier[-1] <= 1.2 * np.min(barrier)
        )
        row.update(
            {
                "stable": stable,
                "failure": None if stable else "failed stability criteria",
                "audit_file": str(audit_path.resolve()),
                "counter_audit_consistent": counter_consistent,
                "final_path_is_last_evaluated_path": final_is_evaluated,
                "initial_path_sha256": initial_hash,
                "tangent_definition": tangent,
                "stop_reason": stop_reason,
                "n_records": int(total_force.size),
                "final_number_of_force_evaluations": int(total_force[-1]),
                "final_max_calls_per_internal_image": int(calls[-1]),
                "final_i_perp": float(i_perp[-1]),
                "final_barrier_kcal": float(barrier[-1]),
                "final_physical_fmax": float(fmax[-1]),
                "final_equidistance_integral": float(equidistance[-1]),
                "final_segment_length_cv": float(segment_cv[-1]),
                "i_perp_log_auc": log_auc(total_force, i_perp),
                "barrier_log_auc": log_auc(total_force, barrier),
                "equidistance_log_auc": log_auc(
                    total_force, 1.0 + equidistance
                ),
                "result_sha256": file_sha256(result),
                "audit_sha256": file_sha256(audit_path),
            }
        )
    except Exception as exc:
        row.update(
            {
                "stable": False,
                "failure": f"{type(exc).__name__}: {exc}",
            }
        )
    return row


def assign_ranks(rows: list[dict]) -> list[dict]:
    stable = [row for row in rows if row.get("stable")]
    fields = (
        "i_perp_log_auc",
        "barrier_log_auc",
        "final_i_perp",
        "final_barrier_kcal",
        "final_equidistance_integral",
    )
    weights = (0.30, 0.25, 0.20, 0.15, 0.10)
    if stable:
        for field in fields:
            values = np.asarray([row[field] for row in stable], dtype=float)
            order = np.argsort(
                np.argsort(values, kind="stable"), kind="stable"
            )
            denominator = max(len(values) - 1, 1)
            for row, rank in zip(stable, order / denominator):
                row[f"{field}_rank_fraction"] = float(rank)
        for row in stable:
            row["selection_score"] = float(
                sum(
                    weight * row[f"{field}_rank_fraction"]
                    for field, weight in zip(fields, weights)
                )
            )
        stable.sort(
            key=lambda row: (
                row["selection_score"],
                row["final_i_perp"],
                row["final_barrier_kcal"],
            )
        )
        for rank, row in enumerate(stable, start=1):
            row["selection_rank"] = rank
    for row in rows:
        if not row.get("stable"):
            row["selection_score"] = None
            row["selection_rank"] = None
    return sorted(
        rows,
        key=lambda row: (
            row.get("selection_rank") is None,
            row.get("selection_rank") or 10**9,
            row["index"],
        ),
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = []
    for row in rows:
        for key in row:
            if key == "command":
                continue
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output.pop("command", None)
            output["configuration"] = json.dumps(
                output["configuration"], sort_keys=True
            )
            writer.writerow(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument(
        "--family",
        choices=(
            "static_neb_ultrafine",
            "original_string_ultrafine",
            "static_neb_boundary",
            "original_string_boundary",
            "static_neb_broad",
            "original_string_broad",
            "obsm_fire_broad",
            "obsm_fire_dt_extension",
            "obsm_fire_dt_high_extension",
            "static_neb_refine",
            "original_string_refine",
            "static_neb_fine",
            "original_string_fine",
            "obsm_fire_refine",
        ),
        required=True,
    )
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    parent = (
        None
        if args.parent_manifest is None
        else args.parent_manifest.expanduser().resolve()
    )
    if (
        args.family.endswith("_broad")
        or args.family
        in {
            "static_neb_ultrafine",
            "original_string_ultrafine",
            "static_neb_boundary",
            "original_string_boundary",
            "obsm_fire_dt_extension",
            "obsm_fire_dt_high_extension",
        }
    ):
        configurations = broad_design(args.family)
    else:
        if parent is None:
            raise ValueError("--parent-manifest is required for refinement")
        configurations = refined_design(args.family, parent)
    namespace = f"screen_{args.stage}"
    tasks = [
        {
            "python": args.python.expanduser().resolve(),
            "root": root,
            "namespace": namespace,
            "stage": args.stage,
            "index": index,
            "budget": int(args.budget),
            "config": configuration,
        }
        for index, configuration in enumerate(configurations)
    ]
    records = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.workers))
    ) as executor:
        futures = {
            executor.submit(run_one, **task): task for task in tasks
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records.append(record)
            print(
                f"[{completed:4d}/{len(futures):4d}] "
                f"index={record['index']:04d} status={record['status']}",
                flush=True,
            )
    rows = assign_ranks([summarize(record) for record in records])
    report_dir = root / "reports" / args.stage
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "ranking.csv"
    manifest_path = report_dir / "manifest.json"
    write_csv(csv_path, rows)
    stable_count = sum(bool(row.get("stable")) for row in rows)
    manifest = {
        "stage": args.stage,
        "family": args.family,
        "parent_manifest": None if parent is None else str(parent),
        "n_configurations": len(configurations),
        "n_stable": stable_count,
        "budget_per_internal_image": int(args.budget),
        "primary_cost_axis": "counter_vectors.sum(axis=1)",
        "primary_cost_label": "Number of Force Evaluations",
        "cost_includes_all_22_images": True,
        "dense_diagnostic_calls_excluded": True,
        "screening_trajectory_is_not_publication_curve": True,
        "common_initial_path_sha256": INITIAL_PATH_SHA256,
        "selection_score": {
            "i_perp_log_auc_rank_fraction": 0.30,
            "raw_barrier_log_auc_rank_fraction": 0.25,
            "final_i_perp_rank_fraction": 0.20,
            "final_raw_barrier_rank_fraction": 0.15,
            "final_equidistance_integral_rank_fraction": 0.10,
        },
        "records": rows,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(csv_path, flush=True)
    print(manifest_path, flush=True)


if __name__ == "__main__":
    main()
