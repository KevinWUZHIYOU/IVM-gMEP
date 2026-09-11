#!/usr/bin/env python3
"""Independent 500-call production reruns after all Chignolin screening."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


PRODUCTIONS = {
    "static_neb_balanced": {
        "method": "static_neb",
        "spring_k": 35.0,
        "alpha": 0.0125,
        "selection_role": "best static-NEB whole-curve score",
    },
    "original_string_balanced": {
        "method": "original_string",
        "dt": 0.0125,
        "reparam_interval": 1,
        "selection_role": "best Original String whole-curve score",
    },
    "obsm_balanced": {
        "method": "obsm_fire",
        "lambda_mix": 0.95,
        "inner_steps": 15,
        "fire_dt": 0.12,
        "fire_beta": 1.0,
        "fire_mass": 1.0,
        "reparam_interval": 1,
        "selection_role": "best OBSM whole-curve score",
    },
    "obsm_barrier": {
        "method": "obsm_fire",
        "lambda_mix": 0.95,
        "inner_steps": 20,
        "fire_dt": 0.12,
        "fire_beta": 1.0,
        "fire_mass": 1.0,
        "reparam_interval": 1,
        "selection_role": "best low-barrier OBSM candidate with r=1",
    },
    "obsm_low_iperp": {
        "method": "obsm_fire",
        "lambda_mix": 0.9,
        "inner_steps": 10,
        "fire_dt": 0.12,
        "fire_beta": 1.0,
        "fire_mass": 1.0,
        "reparam_interval": 1,
        "selection_role": "best low-I_perp OBSM candidate with r=1",
    },
    "obsm_low_iperp_r2": {
        "method": "obsm_fire",
        "lambda_mix": 0.9,
        "inner_steps": 10,
        "fire_dt": 0.12,
        "fire_beta": 1.0,
        "fire_mass": 1.0,
        "reparam_interval": 2,
        "selection_role": "low-I_perp OBSM reparameterization sensitivity",
    },
    "obsm_barrier_lam09": {
        "method": "obsm_fire",
        "lambda_mix": 0.9,
        "inner_steps": 20,
        "fire_dt": 0.12,
        "fire_beta": 1.0,
        "fire_mass": 1.0,
        "reparam_interval": 1,
        "selection_role": "low-barrier lambda=0.9 OBSM candidate",
    },
    "obsm_spacing": {
        "method": "obsm_fire",
        "lambda_mix": 0.9,
        "inner_steps": 5,
        "fire_dt": 0.12,
        "fire_beta": 1.0,
        "fire_mass": 1.0,
        "reparam_interval": 1,
        "selection_role": "best-spaced competitive OBSM candidate",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_for(
    python: Path,
    root: Path,
    namespace: str,
    name: str,
    config: dict,
    budget: int,
) -> list[str]:
    if config["method"] == "static_neb":
        return [
            str(python),
            str(root / "src" / "run_neb.py"),
            "--root",
            str(root),
            "--initial-path-file",
            "inputs/common_initial_path_equidistant.npz",
            "--run-label",
            f"paper_production_{name}",
            "--spring-k",
            str(config["spring_k"]),
            "--alpha",
            str(config["alpha"]),
            "--optimizer",
            "static",
            "--budget",
            str(budget),
            "--target-fmax",
            "0",
            "--save-all-paths",
        ]
    if config["method"] == "original_string":
        return [
            str(python),
            str(root / "src" / "run_string_family.py"),
            "--root",
            str(root),
            "--initial-path-file",
            "inputs/common_initial_path_equidistant.npz",
            "--method",
            "original",
            "--run-label",
            f"paper_production_{name}",
            "--output-namespace",
            namespace,
            "--dt",
            str(config["dt"]),
            "--reparam-interval",
            str(config["reparam_interval"]),
            "--string-length-tol",
            "0",
            "--budget",
            str(budget),
            "--save-all-paths",
        ]
    return [
        str(python),
        str(root / "src" / "run_obsm.py"),
        "--root",
        str(root),
        "--initial-path-file",
        "inputs/common_initial_path_equidistant.npz",
        "--solver",
        "fire",
        "--run-label",
        f"paper_production_{name}",
        "--output-namespace",
        namespace,
        "--lambda-mix",
        str(config["lambda_mix"]),
        "--inner-mode",
        "few_steps",
        "--inner-steps",
        str(config["inner_steps"]),
        "--reparam-interval",
        str(config["reparam_interval"]),
        "--fire-dt",
        str(config["fire_dt"]),
        "--fire-beta",
        str(config["fire_beta"]),
        "--fire-mass",
        str(config["fire_mass"]),
        "--budget",
        str(budget),
        "--save-all-paths",
    ]


def output_glob(root: Path, namespace: str, name: str, method: str) -> str:
    if method == "static_neb":
        return f"runs/neb/neb_paper_production_{name}_central_*.npz"
    return f"runs/{namespace}/{namespace}_paper_production_{name}_*.npz"


def run_one(
    python: Path,
    root: Path,
    namespace: str,
    report_namespace: str,
    name: str,
    config: dict,
    budget: int,
) -> dict:
    pattern = output_glob(root, namespace, name, config["method"])
    existing = sorted(root.glob(pattern))
    command = command_for(python, root, namespace, name, config, budget)
    log_dir = root / "logs" / report_namespace
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    if existing:
        result = existing[-1]
        status = "reused"
        returncode = 0
    else:
        environment = os.environ.copy()
        for variable in (
            "OPENMM_CPU_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OMP_NUM_THREADS",
        ):
            environment[variable] = "1"
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
        outputs = sorted(root.glob(pattern))
        result = outputs[-1] if outputs else None
        status = "completed" if completed.returncode == 0 else "failed"
        returncode = completed.returncode
    return {
        "name": name,
        "configuration": config,
        "status": status,
        "returncode": returncode,
        "result_file": None if result is None else str(result.resolve()),
        "log_file": str(log_path.resolve()),
        "command": command,
    }


def audit_record(record: dict) -> dict:
    result_text = record.get("result_file")
    if not result_text:
        return {**record, "valid": False, "failure": "no result"}
    result = Path(result_text)
    audit_path = result.with_suffix(".audit.json")
    audit = json.loads(audit_path.read_text())
    with np.load(result, allow_pickle=False) as saved:
        counters = np.asarray(saved["counter_vectors"], dtype=int)
        stored_paths = np.asarray(saved["stored_paths"], dtype=float)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
        calls = np.asarray(saved["calls_per_image"], dtype=int)
        initial_hash = str(saved["initial_path_sha256"])
        final_path = np.asarray(saved["final_path"], dtype=float)
        metrics = {
            key: np.asarray(saved[key], dtype=float)
            for key in (
                "i_perp",
                "barrier_kcal",
                "physical_fmax",
                "equidistance_integral",
                "segment_length_cv",
            )
        }
    total_force = counters.sum(axis=1)
    valid = bool(
        audit.get("consistent", False)
        and audit.get("backend_raw_joint_evaluations")
        == audit.get("sum_image_joint_evaluations")
        and initial_hash
        == "57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996"
        and stored_paths.shape[0] == len(calls)
        and np.array_equal(stored_calls, calls)
        and np.array_equal(stored_paths[-1], final_path)
        and all(np.all(np.isfinite(values)) for values in metrics.values())
    )
    return {
        **record,
        "valid": valid,
        "audit_file": str(audit_path.resolve()),
        "result_sha256": sha256(result),
        "audit_sha256": sha256(audit_path),
        "initial_path_sha256": initial_hash,
        "n_records": int(len(calls)),
        "final_max_calls_per_internal_image": int(calls[-1]),
        "final_number_of_force_evaluations": int(total_force[-1]),
        "final_i_perp": float(metrics["i_perp"][-1]),
        "final_barrier_kcal": float(metrics["barrier_kcal"][-1]),
        "final_physical_fmax": float(metrics["physical_fmax"][-1]),
        "final_equidistance_integral": float(
            metrics["equidistance_integral"][-1]
        ),
        "final_segment_length_cv": float(
            metrics["segment_length_cv"][-1]
        ),
        "all_recorded_paths_saved": bool(stored_paths.shape[0] == len(calls)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--namespace", default="paper_method_productions_20260727a"
    )
    parser.add_argument(
        "--report-namespace", default="paper_method_productions_20260727a"
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    jobs = [
        (
            args.python.expanduser().resolve(),
            root,
            args.namespace,
            args.report_namespace,
            name,
            config,
            int(args.budget),
        )
        for name, config in PRODUCTIONS.items()
    ]
    records = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.workers))
    ) as executor:
        futures = {
            executor.submit(run_one, *job): job[4] for job in jobs
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records.append(record)
            print(
                f"[{completed}/{len(futures)}] {record['name']} "
                f"{record['status']}",
                flush=True,
            )
    audited = [audit_record(record) for record in records]
    if not all(record["valid"] for record in audited):
        failed = [
            record["name"] for record in audited if not record["valid"]
        ]
        raise RuntimeError(f"Invalid production runs: {failed}")
    report_dir = root / "reports" / args.report_namespace
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "budget_per_internal_image": int(args.budget),
        "primary_cost_axis": "counter_vectors.sum(axis=1)",
        "primary_cost_label": "Number of Force Evaluations",
        "cost_includes_all_22_images": True,
        "dense_diagnostic_calls_excluded": True,
        "screening_trajectories_reused": False,
        "records": sorted(audited, key=lambda row: row["name"]),
    }
    output = report_dir / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
