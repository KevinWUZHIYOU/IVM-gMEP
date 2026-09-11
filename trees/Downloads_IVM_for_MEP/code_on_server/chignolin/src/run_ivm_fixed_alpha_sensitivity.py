#!/usr/bin/env python3
"""Independent fixed-alpha Chignolin IVM sensitivity productions."""

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


ALPHAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 20.0)
SELECTED_ALPHA1 = (
    "runs/ivm/"
    "ivm_ivm_equal_idpp_v52_fixed_alpha1_production_central_state_ivm_"
    "cfg31b81924ce51_budget500.npz"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpha_tag(alpha: float) -> str:
    return f"{float(alpha):g}".replace(".", "p")


def command_for(
    python: Path, root: Path, alpha: float, budget: int
) -> list[str]:
    return [
        str(python),
        str(root / "src" / "run_ivm.py"),
        "--root",
        str(root),
        "--initial-path-file",
        "inputs/common_initial_path_equidistant.npz",
        "--run-label",
        f"paper_fixed_alpha_sensitivity_a{alpha_tag(alpha)}",
        "--solver",
        "state_ivm",
        "--alpha",
        str(alpha),
        "--alpha-mode",
        "fixed",
        "--alpha-min",
        "0.03",
        "--alpha-max",
        "20",
        "--max-step",
        "1",
        "--state-trust-fraction",
        "2",
        "--state-polish-trust-fraction",
        "2",
        "--reparam-mode",
        "state_curve_projected",
        "--state-curve-direct-solve",
        "--state-curve-direct-trial-calls",
        "15",
        "--reparam-ready-relative-iperp",
        "2",
        "--reparam-cv-trigger",
        "-1",
        "--reparam-recovery-calls",
        "30",
        "--inner-calls",
        "15",
        "--plbfgs-kkt-rtol",
        "0.8",
        "--budget",
        str(budget),
        "--save-all-paths",
    ]


def run_one(
    python: Path,
    root: Path,
    alpha: float,
    budget: int,
    report_namespace: str,
) -> dict:
    command = command_for(python, root, alpha, budget)
    if alpha == 1.0:
        result = root / SELECTED_ALPHA1
        status = "reused_selected_production"
        returncode = 0
    else:
        pattern = (
            "runs/ivm/"
            f"ivm_paper_fixed_alpha_sensitivity_a{alpha_tag(alpha)}_"
            f"central_state_ivm_*_budget{budget}.npz"
        )
        existing = sorted(root.glob(pattern))
        log_dir = root / "logs" / report_namespace
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"alpha_{alpha_tag(alpha)}.log"
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
            status = (
                "completed" if completed.returncode == 0 else "failed"
            )
            returncode = completed.returncode
    return {
        "alpha": alpha,
        "status": status,
        "returncode": returncode,
        "result_file": None if result is None else str(result.resolve()),
        "command": command,
    }


def audit(record: dict) -> dict:
    result_text = record["result_file"]
    if result_text is None:
        return {**record, "valid": False, "failure": "no result"}
    result = Path(result_text)
    audit_path = result.with_suffix(".audit.json")
    source_audit = json.loads(audit_path.read_text())
    with np.load(result, allow_pickle=False) as saved:
        counters = np.asarray(saved["counter_vectors"], dtype=int)
        active = np.asarray(saved["active_alpha"], dtype=float)
        initial_hash = str(saved["initial_path_sha256"])
        tangent = str(saved["tangent_definition"])
        alpha_mode = str(saved["alpha_mode"])
        calls = np.asarray(saved["calls_per_image"], dtype=int)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
        stored_paths = np.asarray(saved["stored_paths"], dtype=float)
        final_path = np.asarray(saved["final_path"], dtype=float)
        i_perp = np.asarray(saved["i_perp"], dtype=float)
        barrier = np.asarray(saved["barrier_kcal"], dtype=float)
        equidistance = np.asarray(
            saved["equidistance_integral"], dtype=float
        )
    total_force = counters.sum(axis=1)
    valid = bool(
        source_audit.get("consistent", False)
        and source_audit.get("backend_raw_joint_evaluations")
        == source_audit.get("sum_image_joint_evaluations")
        and alpha_mode == "fixed"
        and np.array_equal(
            np.unique(active), np.asarray([record["alpha"]], dtype=float)
        )
        and initial_hash
        == "57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996"
        and "central difference" in tangent.lower()
        and stored_calls[-1] == calls[-1]
        and np.array_equal(stored_paths[-1], final_path)
        and np.all(np.isfinite(i_perp))
        and np.all(np.isfinite(barrier))
        and np.all(np.isfinite(equidistance))
    )
    return {
        **record,
        "valid": valid,
        "audit_file": str(audit_path.resolve()),
        "result_sha256": sha256(result),
        "audit_sha256": sha256(audit_path),
        "initial_path_sha256": initial_hash,
        "tangent_definition": tangent,
        "n_records": int(len(calls)),
        "final_number_of_force_evaluations": int(total_force[-1]),
        "final_max_calls_per_internal_image": int(calls[-1]),
        "final_i_perp": float(i_perp[-1]),
        "final_sparse_barrier_kcal": float(barrier[-1]),
        "final_equidistance_integral": float(equidistance[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--report-namespace",
        default="ivm_fixed_alpha_sensitivity_total_nf_20260727a",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    jobs = [
        (
            args.python.expanduser().resolve(),
            root,
            alpha,
            int(args.budget),
            args.report_namespace,
        )
        for alpha in ALPHAS
    ]
    records = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.workers))
    ) as executor:
        futures = {
            executor.submit(run_one, *job): job[2] for job in jobs
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records.append(record)
            print(
                f"[{completed}/{len(futures)}] alpha={record['alpha']:g} "
                f"{record['status']}",
                flush=True,
            )
    audited = [audit(record) for record in records]
    if not all(record["valid"] for record in audited):
        failed = [
            record["alpha"] for record in audited if not record["valid"]
        ]
        raise RuntimeError(f"Invalid alpha productions: {failed}")
    report_dir = root / "reports" / args.report_namespace
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "manifest.json"
    output.write_text(
        json.dumps(
            {
                "fixed_alphas": list(ALPHAS),
                "selected_alpha": 1.0,
                "budget_per_internal_image": int(args.budget),
                "primary_cost_axis": "counter_vectors.sum(axis=1)",
                "primary_cost_label": "Number of Force Evaluations",
                "cost_includes_all_22_images": True,
                "records": sorted(audited, key=lambda row: row["alpha"]),
            },
            indent=2,
        )
        + "\n"
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
