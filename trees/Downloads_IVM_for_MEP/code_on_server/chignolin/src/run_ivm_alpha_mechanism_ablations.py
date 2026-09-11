"""Run matched Chignolin alpha/globalization mechanism ablations."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np


ALPHAS = (1.0, 1000.0)
VARIANTS = {
    "relaxed_trust": {
        "state_globalization": "iperp_monotone",
        "max_step": 10.0,
        "trust_fraction": 20.0,
    },
    "proximal_only": {
        "state_globalization": "proximal_only",
        "max_step": 1.0,
        "trust_fraction": 2.0,
    },
    "both_relaxed": {
        "state_globalization": "proximal_only",
        "max_step": 10.0,
        "trust_fraction": 20.0,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def command_for(
    python: Path,
    script: Path,
    root: Path,
    alpha: float,
    variant: str,
    budget: int,
) -> list[str]:
    settings = VARIANTS[variant]
    return [
        str(python),
        str(script),
        "--root",
        str(root),
        "--initial-path-file",
        "inputs/common_initial_path_equidistant.npz",
        "--run-label",
        f"alpha_mechanism_{variant}_a{tag(alpha)}",
        "--solver",
        "state_ivm",
        "--alpha",
        str(alpha),
        "--alpha-mode",
        "fixed",
        "--alpha-min",
        "0.03",
        "--alpha-max",
        str(max(20.0, alpha)),
        "--max-step",
        str(settings["max_step"]),
        "--state-trust-fraction",
        str(settings["trust_fraction"]),
        "--state-polish-trust-fraction",
        str(settings["trust_fraction"]),
        "--state-globalization",
        str(settings["state_globalization"]),
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
    script: Path,
    root: Path,
    alpha: float,
    variant: str,
    budget: int,
    namespace: str,
) -> dict:
    command = command_for(python, script, root, alpha, variant, budget)
    result_pattern = (
        root
        / "runs"
        / "ivm"
        / (
            f"ivm_alpha_mechanism_{variant}_a{tag(alpha)}_"
            f"central_state_ivm_*_budget{budget}.npz"
        )
    )
    existing = sorted(result_pattern.parent.glob(result_pattern.name))
    log_dir = root / "logs" / namespace
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{variant}_a{tag(alpha)}.log"
    if existing:
        result = existing[-1]
        status = "reused"
        returncode = 0
    else:
        environment = os.environ.copy()
        environment.update(
            {
                "OPENMM_CPU_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            }
        )
        with log_path.open("w") as log:
            process = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        returncode = int(process.returncode)
        completed = sorted(result_pattern.parent.glob(result_pattern.name))
        result = completed[-1] if completed else None
        status = "completed" if returncode == 0 and result else "failed"

    record = {
        "variant": variant,
        "alpha": alpha,
        "settings": VARIANTS[variant],
        "status": status,
        "returncode": returncode,
        "result_file": None if result is None else str(result),
        "log_file": str(log_path),
        "command": command,
    }
    if result is None:
        return record
    with np.load(result, allow_pickle=False) as saved:
        active_alpha = np.unique(
            np.asarray(saved["active_alpha"], dtype=float)
        )
        record.update(
            {
                "result_sha256": sha256(result),
                "initial_path_sha256": str(saved["initial_path_sha256"]),
                "state_globalization": str(saved["state_globalization"]),
                "active_alpha_unique": active_alpha.tolist(),
                "final_call": int(saved["calls_per_image"][-1]),
                "final_total_force_evaluations": int(
                    np.asarray(saved["counter_vectors"], dtype=np.int64)[
                        -1
                    ].sum()
                ),
                "final_i_perp": float(saved["i_perp"][-1]),
                "final_barrier_kcal": float(saved["barrier_kcal"][-1]),
                "final_fmax": float(saved["physical_fmax"][-1]),
                "stop_reason": str(saved["stop_reason"]),
            }
        )
    audit_path = result.with_suffix(".audit.json")
    record["audit_file"] = str(audit_path)
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        record["counter_audit_consistent"] = bool(audit["consistent"])
        record["backend_raw_joint_evaluations"] = int(
            audit["backend_raw_joint_evaluations"]
        )
        record["sum_image_joint_evaluations"] = int(
            audit["sum_image_joint_evaluations"]
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--ivm-script", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=150)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--namespace", default="ivm_alpha_mechanism_ablations_20260728a"
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    script = args.ivm_script.expanduser().resolve()
    configurations = [
        (alpha, variant)
        for variant in VARIANTS
        for alpha in ALPHAS
    ]
    records = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(
                run_one,
                python,
                script,
                root,
                alpha,
                variant,
                args.budget,
                args.namespace,
            ): (alpha, variant)
            for alpha, variant in configurations
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            record = future.result()
            records.append(record)
            print(
                f"[{completed}/{len(futures)}] "
                f"{record['variant']} alpha={record['alpha']:g} "
                f"{record['status']}",
                flush=True,
            )
    records.sort(key=lambda row: (row["variant"], row["alpha"]))
    if any(
        row["status"] not in {"completed", "reused"}
        or not row.get("counter_audit_consistent", False)
        or row.get("active_alpha_unique") != [row["alpha"]]
        for row in records
    ):
        raise RuntimeError("One or more alpha mechanism ablations failed audit")
    report_dir = root / "reports" / args.namespace
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "manifest.json"
    output.write_text(
        json.dumps(
            {
                "budget_per_internal_image": args.budget,
                "same_initial_path_required": True,
                "production_results_untouched": True,
                "ablation_script": str(script),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(output, flush=True)


if __name__ == "__main__":
    main()
