#!/usr/bin/env python3
"""Fixed-alpha Chignolin IVM sweep without the outer I_perp guard."""

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


ALPHAS = (
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    20.0,
    30.0,
    100.0,
    300.0,
    1000.0,
)
SELECTED_ALPHA1 = (
    "runs/ivm/"
    "ivm_alpha_mechanism_proximal_only_a1_full_central_state_ivm_"
    "cfg85ee8e3b477d_budget500.npz"
)
INITIAL_PATH_SHA256 = (
    "57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpha_tag(alpha: float) -> str:
    return f"{float(alpha):g}".replace(".", "p")


def parse_alphas(specification: str) -> tuple[float, ...]:
    values = tuple(
        float(item.strip())
        for item in specification.split(",")
        if item.strip()
    )
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("--alphas must contain positive comma-separated values")
    if len(set(values)) != len(values):
        raise ValueError("--alphas must not contain duplicates")
    return values


def command_for(
    python: Path,
    root: Path,
    alpha: float,
    budget: int,
    run_label_prefix: str,
    max_step: float,
    state_trust_fraction: float,
    state_polish_trust_fraction: float,
    direct_trial_calls: int,
    inner_calls: int,
    kkt_rtol: float,
) -> list[str]:
    return [
        str(python),
        str(root / "src" / "run_ivm_alpha_ablation.py"),
        "--root",
        str(root),
        "--initial-path-file",
        "inputs/common_initial_path_equidistant.npz",
        "--run-label",
        f"{run_label_prefix}_a{alpha_tag(alpha)}",
        "--solver",
        "state_ivm",
        "--alpha",
        str(alpha),
        "--alpha-mode",
        "fixed",
        "--alpha-min",
        "0.03",
        "--alpha-max",
        str(max(20.0, float(alpha))),
        "--max-step",
        str(max_step),
        "--state-trust-fraction",
        str(state_trust_fraction),
        "--state-polish-trust-fraction",
        str(state_polish_trust_fraction),
        "--state-globalization",
        "proximal_only",
        "--reparam-mode",
        "state_curve_projected",
        "--state-curve-direct-solve",
        "--state-curve-direct-trial-calls",
        str(direct_trial_calls),
        "--reparam-ready-relative-iperp",
        "2",
        "--reparam-cv-trigger",
        "-1",
        "--reparam-recovery-calls",
        "30",
        "--inner-calls",
        str(inner_calls),
        "--plbfgs-kkt-rtol",
        str(kkt_rtol),
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
    run_label_prefix: str,
    max_step: float,
    state_trust_fraction: float,
    state_polish_trust_fraction: float,
    direct_trial_calls: int,
    inner_calls: int,
    kkt_rtol: float,
    reuse_selected_alpha1: bool,
) -> dict:
    command = command_for(
        python,
        root,
        alpha,
        budget,
        run_label_prefix,
        max_step,
        state_trust_fraction,
        state_polish_trust_fraction,
        direct_trial_calls,
        inner_calls,
        kkt_rtol,
    )
    log_dir = root / "logs" / report_namespace
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"alpha_{alpha_tag(alpha)}.log"
    if alpha == 1.0 and reuse_selected_alpha1:
        result = root / SELECTED_ALPHA1
        status = "reused_selected_proximal_only_production"
        returncode = 0
    else:
        pattern = (
            "runs/ivm/"
            f"ivm_{run_label_prefix}_a{alpha_tag(alpha)}_"
            f"central_state_ivm_*_budget{budget}.npz"
        )
        existing = sorted(root.glob(pattern))
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
                "completed"
                if completed.returncode == 0 and result is not None
                else "failed"
            )
            returncode = completed.returncode
    return {
        "alpha": float(alpha),
        "status": status,
        "returncode": int(returncode),
        "result_file": None if result is None else str(result.resolve()),
        "log_file": str(log_path.resolve()),
        "command": command,
    }


def audit(record: dict, minimum_call: int) -> dict:
    if record["result_file"] is None:
        return {**record, "valid": False, "failure": "no result"}
    result = Path(record["result_file"])
    audit_path = result.with_suffix(".audit.json")
    source_audit = json.loads(audit_path.read_text())
    with np.load(result, allow_pickle=False) as saved:
        counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        active = np.asarray(saved["active_alpha"], dtype=float)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        stored_paths = np.asarray(saved["stored_paths"], dtype=float)
        final_path = np.asarray(saved["final_path"], dtype=float)
        initial_hash = str(saved["initial_path_sha256"])
        tangent = str(saved["tangent_definition"])
        alpha_mode = str(saved["alpha_mode"])
        globalization = str(saved["state_globalization"])
        stop_reason = str(saved["stop_reason"])
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
        and globalization == "proximal_only"
        and np.array_equal(
            np.unique(active), np.asarray([record["alpha"]], dtype=float)
        )
        and initial_hash == INITIAL_PATH_SHA256
        and "central difference" in tangent.lower()
        and (
            calls[-1] >= minimum_call
            or stop_reason == "lbfgsb_no_progress"
        )
        and stored_calls[-1] == calls[-1]
        and np.array_equal(stored_paths[-1], final_path)
        and np.all(np.diff(counters, axis=0) >= 0)
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
        "state_globalization": globalization,
        "n_records": int(len(calls)),
        "final_number_of_force_evaluations": int(total_force[-1]),
        "final_max_calls_per_internal_image": int(calls[-1]),
        "stop_reason": stop_reason,
        "final_i_perp": float(i_perp[-1]),
        "minimum_i_perp": float(np.min(i_perp)),
        "final_sparse_barrier_kcal": float(barrier[-1]),
        "final_equidistance_integral": float(equidistance[-1]),
        "backend_raw_joint_evaluations": int(
            source_audit["backend_raw_joint_evaluations"]
        ),
        "sum_image_joint_evaluations": int(
            source_audit["sum_image_joint_evaluations"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--alphas",
        default=",".join(f"{alpha:g}" for alpha in ALPHAS),
        help="Positive comma-separated fixed alpha values.",
    )
    parser.add_argument(
        "--run-label-prefix",
        default="paper_fixed_alpha_proximal_only",
    )
    parser.add_argument("--max-step", type=float, default=1.0)
    parser.add_argument("--state-trust-fraction", type=float, default=2.0)
    parser.add_argument(
        "--state-polish-trust-fraction", type=float, default=2.0
    )
    parser.add_argument("--direct-trial-calls", type=int, default=15)
    parser.add_argument("--inner-calls", type=int, default=15)
    parser.add_argument("--kkt-rtol", type=float, default=0.8)
    reuse_group = parser.add_mutually_exclusive_group()
    reuse_group.add_argument(
        "--reuse-selected-alpha1",
        dest="reuse_selected_alpha1",
        action="store_true",
    )
    reuse_group.add_argument(
        "--no-reuse-selected-alpha1",
        dest="reuse_selected_alpha1",
        action="store_false",
    )
    parser.set_defaults(reuse_selected_alpha1=True)
    parser.add_argument(
        "--report-namespace",
        default="ivm_fixed_alpha_proximal_only_20260728a",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    alphas = parse_alphas(args.alphas)
    jobs = [
        (
            args.python.expanduser().resolve(),
            root,
            alpha,
            int(args.budget),
            args.report_namespace,
            args.run_label_prefix,
            float(args.max_step),
            float(args.state_trust_fraction),
            float(args.state_polish_trust_fraction),
            int(args.direct_trial_calls),
            int(args.inner_calls),
            float(args.kkt_rtol),
            bool(args.reuse_selected_alpha1),
        )
        for alpha in alphas
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
                f"[{completed}/{len(futures)}] "
                f"alpha={record['alpha']:g} {record['status']}",
                flush=True,
            )
    audited = [
        audit(record, minimum_call=int(args.budget))
        for record in records
    ]
    if not all(record["valid"] for record in audited):
        failed = [
            {
                "alpha": record["alpha"],
                "status": record["status"],
                "returncode": record["returncode"],
                "final_call": record.get("final_max_calls_per_internal_image"),
            }
            for record in audited
            if not record["valid"]
        ]
        raise RuntimeError(f"Invalid proximal-only alpha runs: {failed}")
    report_dir = root / "reports" / args.report_namespace
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "manifest.json"
    output.write_text(
        json.dumps(
            {
                "fixed_alphas": list(alphas),
                "selected_alpha": 1.0,
                "state_globalization": "proximal_only",
                "outer_i_perp_acceptance_guard": False,
                "proximal_objective_line_search": True,
                "run_label_prefix": args.run_label_prefix,
                "solver_configuration": {
                    "max_step": float(args.max_step),
                    "state_trust_fraction": float(
                        args.state_trust_fraction
                    ),
                    "state_polish_trust_fraction": float(
                        args.state_polish_trust_fraction
                    ),
                    "direct_trial_calls": int(args.direct_trial_calls),
                    "inner_calls": int(args.inner_calls),
                    "kkt_rtol": float(args.kkt_rtol),
                    "reuse_selected_alpha1": bool(
                        args.reuse_selected_alpha1
                    ),
                },
                "minimum_call_per_internal_image": int(args.budget),
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
