#!/usr/bin/env python3
"""Run reproducible within-method Ala competitor screens.

Each candidate starts from the same 22-image IDPP path.  The screen never
uses IVM or another competitor for selection.  Every backend cache miss,
including rejected trials, remains on the stored whole-path force-call axis.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ORIGINAL_DT = (
    0.0005,
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.006,
    0.008,
    0.010,
    0.0125,
    0.015,
    0.020,
    0.030,
    0.040,
)
ORIGINAL_FINE_DT = (
    0.0105,
    0.011,
    0.0115,
    0.012,
    0.01225,
    0.0125,
    0.01275,
    0.013,
    0.0135,
    0.014,
)
ORIGINAL_FINALIST_DT = (0.013, 0.0135, 0.014)
STATIC_K = (0.1, 1.0, 5.0, 10.0, 20.0, 40.0, 50.0, 80.0, 100.0)
STATIC_DT = (0.001, 0.002, 0.004, 0.006, 0.008, 0.010, 0.0125, 0.016, 0.020)
STATIC_FINE_K = (20.0, 30.0, 40.0, 50.0, 60.0)
STATIC_FINE_DT = (0.012, 0.01225, 0.0125, 0.01275, 0.013)
STATIC_FINALISTS = (
    (20.0, 0.013),
    (30.0, 0.0125),
    (30.0, 0.01275),
    (30.0, 0.013),
    (40.0, 0.0125),
    (50.0, 0.0125),
)
OBSM_LAMBDA = (0.10, 0.25, 0.50, 0.75, 0.90)
OBSM_INNER = (1, 3, 5, 8)
OBSM_REPARAM = (1, 3, 5, 10)
FIRE_DT = (0.01, 0.02, 0.05, 0.10)
BFGS_STAGE = tuple(
    (lambda_mix, inner_steps, interval)
    for lambda_mix in (0.25, 0.50, 0.75, 0.90)
    for inner_steps in (1, 3, 5)
    for interval in (1, 5)
)
BFGS_CURVATURE_STAGE = tuple(
    (lambda_mix, inner_steps, interval)
    for lambda_mix in (0.02, 0.05, 0.10, 0.20)
    for inner_steps in (1, 3)
    for interval in (1, 5)
)
BFGS_IDENTITY_SMALL_STAGE = tuple(
    (lambda_mix, inner_steps, interval)
    for lambda_mix in (0.002, 0.005, 0.01, 0.02)
    for inner_steps in (2, 3)
    for interval in (1, 5)
)
BFGS_IDENTITY_MID_STAGE = tuple(
    (lambda_mix, inner_steps)
    for lambda_mix in (0.05, 0.10, 0.15, 0.20)
    for inner_steps in (1, 2, 3)
)
BFGS_CARRY_MID_STAGE = (0.10, 0.20, 0.30)
BFGS_FINALISTS = (0.20, 0.25, 0.30)
FIRE_STAGE = tuple(
    (lambda_mix, inner_steps, interval, fire_dt)
    for lambda_mix in (0.25, 0.50, 0.75, 0.90)
    for inner_steps in (1, 3, 5)
    for interval in (1, 5)
    for fire_dt in (0.02, 0.05, 0.10)
)
FIRE_REFINE = tuple(
    (lambda_mix, inner_steps, fire_dt)
    for lambda_mix in (0.70, 0.75, 0.80, 0.85)
    for inner_steps in (2, 3, 4)
    for fire_dt in (0.08, 0.10, 0.12, 0.15)
)
FIRE_FINALISTS = (
    (0.75, 3, 0.10),
    (0.80, 3, 0.10),
    (0.85, 3, 0.10),
    (0.75, 2, 0.12),
    (0.80, 2, 0.12),
    (0.85, 2, 0.12),
)


def commands(root: Path, method: str, budget: int) -> list[list[str]]:
    python = sys.executable
    source = root / "src"
    common = ["--root", str(root), "--budget", str(int(budget))]
    result: list[list[str]] = []
    if method == "original":
        for dt in ORIGINAL_DT:
            result.append(
                [
                    python,
                    str(source / "run_string_family.py"),
                    *common,
                    "--method",
                    "original",
                    "--run-label",
                    "screen20260731",
                    "--output-namespace",
                    "competitor_screen_original",
                    "--dt",
                    str(dt),
                    "--reparam-interval",
                    "1",
                    "--string-length-tol",
                    "0",
                ]
            )
    elif method == "original_fine":
        for dt in ORIGINAL_FINE_DT:
            result.append(
                [
                    python,
                    str(source / "run_string_family.py"),
                    *common,
                    "--method",
                    "original",
                    "--run-label",
                    "fine20260731",
                    "--output-namespace",
                    "competitor_screen_original_fine",
                    "--dt",
                    str(dt),
                    "--reparam-interval",
                    "1",
                    "--string-length-tol",
                    "0",
                ]
            )
    elif method == "original_finalists":
        for dt in ORIGINAL_FINALIST_DT:
            result.append(
                [
                    python,
                    str(source / "run_string_family.py"),
                    *common,
                    "--method",
                    "original",
                    "--run-label",
                    "finalists20260731",
                    "--output-namespace",
                    "competitor_screen_original_finalists",
                    "--dt",
                    str(dt),
                    "--reparam-interval",
                    "1",
                    "--string-length-tol",
                    "0",
                    "--save-all-until",
                    "100",
                    "--path-interval",
                    "25",
                ]
            )
    elif method == "original_rank1_long":
        result.append(
            [
                python,
                str(source / "run_string_family.py"),
                *common,
                "--method",
                "original",
                "--run-label",
                "rank1_long20260731",
                "--output-namespace",
                "competitor_screen_original_rank1_long",
                "--dt",
                "0.0125",
                "--reparam-interval",
                "1",
                "--string-length-tol",
                "0",
                "--save-all-until",
                "100",
                "--path-interval",
                "25",
            ]
        )
    elif method == "static":
        for spring_k in STATIC_K:
            for dt in STATIC_DT:
                result.append(
                    [
                        python,
                        str(source / "run_neb_competitor.py"),
                        *common,
                        "--spring-k",
                        str(spring_k),
                        "--alpha",
                        str(dt),
                        "--optimizer",
                        "static",
                        "--run-label",
                        "screen20260731",
                    ]
                )
    elif method == "static_fine":
        for spring_k in STATIC_FINE_K:
            for dt in STATIC_FINE_DT:
                result.append(
                    [
                        python,
                        str(source / "run_neb_competitor.py"),
                        *common,
                        "--spring-k",
                        str(spring_k),
                        "--alpha",
                        str(dt),
                        "--optimizer",
                        "static",
                        "--run-label",
                        "fine20260731",
                    ]
                )
    elif method == "static_finalists":
        for spring_k, dt in STATIC_FINALISTS:
            result.append(
                [
                    python,
                    str(source / "run_neb_competitor.py"),
                    *common,
                    "--spring-k",
                    str(spring_k),
                    "--alpha",
                    str(dt),
                    "--optimizer",
                    "static",
                    "--run-label",
                    "finalists20260731",
                    "--save-all-until",
                    "100",
                    "--path-interval",
                    "25",
                ]
            )
    elif method == "bfgs_stage":
        for lambda_mix, inner_steps, interval in BFGS_STAGE:
            result.append(
                [
                    python,
                    str(source / "run_obsm.py"),
                    *common,
                    "--solver",
                    "bfgs",
                    "--run-label",
                    "stage20260731",
                    "--output-namespace",
                    "competitor_screen_bfgs_stage",
                    "--lambda-mix",
                    str(lambda_mix),
                    "--inner-mode",
                    "few_steps",
                    "--inner-steps",
                    str(inner_steps),
                    "--reparam-interval",
                    str(interval),
                    "--inner-stop-norm",
                    "l2",
                    "--bfgs-h0",
                    "identity",
                    "--bfgs-line-search-direction",
                    "paper_p",
                    "--bfgs-state-scope",
                    "paper_local",
                ]
            )
    elif method == "bfgs_curvature_stage":
        for lambda_mix, inner_steps, interval in BFGS_CURVATURE_STAGE:
            result.append(
                [
                    python,
                    str(source / "run_obsm.py"),
                    *common,
                    "--solver",
                    "bfgs",
                    "--run-label",
                    "curvature_stage20260731",
                    "--output-namespace",
                    "competitor_screen_bfgs_curvature_stage",
                    "--lambda-mix",
                    str(lambda_mix),
                    "--inner-mode",
                    "few_steps",
                    "--inner-steps",
                    str(inner_steps),
                    "--reparam-interval",
                    str(interval),
                    "--inner-stop-norm",
                    "l2",
                    "--bfgs-h0",
                    "bonded_forcefield",
                    "--bfgs-line-search-direction",
                    "paper_p",
                    "--bfgs-state-scope",
                    "paper_local",
                ]
            )
    elif method == "bfgs_carry_stage":
        for lambda_mix, inner_steps, interval in BFGS_CURVATURE_STAGE:
            result.append(
                [
                    python,
                    str(source / "run_obsm.py"),
                    *common,
                    "--solver",
                    "bfgs",
                    "--run-label",
                    "carry_stage20260731",
                    "--output-namespace",
                    "competitor_screen_bfgs_carry_stage",
                    "--lambda-mix",
                    str(lambda_mix),
                    "--inner-mode",
                    "few_steps",
                    "--inner-steps",
                    str(inner_steps),
                    "--reparam-interval",
                    str(interval),
                    "--inner-stop-norm",
                    "l2",
                    "--bfgs-h0",
                    "bonded_forcefield",
                    "--bfgs-bonded-h0-cache",
                    "inputs/obsm_bfgs_common_initial_bonded_h0.npz",
                    "--bfgs-line-search-direction",
                    "paper_p",
                    "--bfgs-state-scope",
                    "curvature_carry",
                ]
            )
    elif method == "bfgs_identity_small_stage":
        for lambda_mix, inner_steps, interval in BFGS_IDENTITY_SMALL_STAGE:
            result.append(
                [
                    python,
                    str(source / "run_obsm.py"),
                    *common,
                    "--solver",
                    "bfgs",
                    "--run-label",
                    "identity_small_stage20260731",
                    "--output-namespace",
                    "competitor_screen_bfgs_identity_small_stage",
                    "--lambda-mix",
                    str(lambda_mix),
                    "--inner-mode",
                    "few_steps",
                    "--inner-steps",
                    str(inner_steps),
                    "--reparam-interval",
                    str(interval),
                    "--inner-stop-norm",
                    "l2",
                    "--bfgs-h0",
                    "identity",
                    "--bfgs-line-search-direction",
                    "paper_p",
                    "--bfgs-state-scope",
                    "paper_local",
                ]
            )
    elif method == "bfgs_identity_mid_stage":
        for lambda_mix, inner_steps in BFGS_IDENTITY_MID_STAGE:
            result.append(
                [
                    python,
                    str(source / "run_obsm.py"),
                    *common,
                    "--solver",
                    "bfgs",
                    "--run-label",
                    "identity_mid_stage20260731",
                    "--output-namespace",
                    "competitor_screen_bfgs_identity_mid_stage",
                    "--lambda-mix",
                    str(lambda_mix),
                    "--inner-mode",
                    "few_steps",
                    "--inner-steps",
                    str(inner_steps),
                    "--reparam-interval",
                    "1",
                    "--inner-stop-norm",
                    "l2",
                    "--bfgs-h0",
                    "identity",
                    "--bfgs-line-search-direction",
                    "paper_p",
                    "--bfgs-state-scope",
                    "paper_local",
                ]
            )
    elif method == "bfgs_carry_mid_stage":
        for lambda_mix in BFGS_CARRY_MID_STAGE:
            result.append(
                [
                    python,
                    str(source / "run_obsm.py"),
                    *common,
                    "--solver",
                    "bfgs",
                    "--run-label",
                    "carry_mid_stage20260731",
                    "--output-namespace",
                    "competitor_screen_bfgs_carry_mid_stage",
                    "--lambda-mix",
                    str(lambda_mix),
                    "--inner-mode",
                    "few_steps",
                    "--inner-steps",
                    "1",
                    "--reparam-interval",
                    "1",
                    "--inner-stop-norm",
                    "l2",
                    "--bfgs-h0",
                    "bonded_forcefield",
                    "--bfgs-bonded-h0-cache",
                    "inputs/obsm_bfgs_common_initial_bonded_h0.npz",
                    "--bfgs-line-search-direction",
                    "paper_p",
                    "--bfgs-state-scope",
                    "curvature_carry",
                ]
            )
    elif method == "bfgs_finalists":
        for lambda_mix in BFGS_FINALISTS:
            result.append(
                [
                    python,
                    str(source / "run_obsm.py"),
                    *common,
                    "--solver",
                    "bfgs",
                    "--run-label",
                    "finalists20260731",
                    "--output-namespace",
                    "competitor_screen_bfgs_finalists",
                    "--lambda-mix",
                    str(lambda_mix),
                    "--inner-mode",
                    "few_steps",
                    "--inner-steps",
                    "1",
                    "--reparam-interval",
                    "1",
                    "--inner-stop-norm",
                    "l2",
                    "--bfgs-h0",
                    "bonded_forcefield",
                    "--bfgs-bonded-h0-cache",
                    "inputs/obsm_bfgs_common_initial_bonded_h0.npz",
                    "--bfgs-line-search-direction",
                    "paper_p",
                    "--bfgs-state-scope",
                    "curvature_carry",
                    "--save-all-until",
                    "100",
                    "--path-interval",
                    "25",
                ]
            )
    elif method == "bfgs_finalists_r5":
        for lambda_mix in BFGS_FINALISTS:
            result.append(
                [
                    python,
                    str(source / "run_obsm.py"),
                    *common,
                    "--solver",
                    "bfgs",
                    "--run-label",
                    "finalists_r5_20260731",
                    "--output-namespace",
                    "competitor_screen_bfgs_finalists_r5",
                    "--lambda-mix",
                    str(lambda_mix),
                    "--inner-mode",
                    "few_steps",
                    "--inner-steps",
                    "1",
                    "--reparam-interval",
                    "5",
                    "--inner-stop-norm",
                    "l2",
                    "--bfgs-h0",
                    "bonded_forcefield",
                    "--bfgs-bonded-h0-cache",
                    "inputs/obsm_bfgs_common_initial_bonded_h0.npz",
                    "--bfgs-line-search-direction",
                    "paper_p",
                    "--bfgs-state-scope",
                    "curvature_carry",
                    "--save-all-until",
                    "100",
                    "--path-interval",
                    "25",
                ]
            )
    elif method == "bfgs_diagnostic_finalists":
        for lambda_mix in (0.20, 0.30):
            result.append(
                [
                    python,
                    str(source / "run_obsm.py"),
                    *common,
                    "--solver",
                    "bfgs",
                    "--run-label",
                    "diagnostic_finalists20260731",
                    "--output-namespace",
                    "competitor_screen_bfgs_diagnostic_finalists",
                    "--lambda-mix",
                    str(lambda_mix),
                    "--inner-mode",
                    "few_steps",
                    "--inner-steps",
                    "1",
                    "--reparam-interval",
                    "1",
                    "--inner-stop-norm",
                    "l2",
                    "--bfgs-h0",
                    "bonded_forcefield",
                    "--bfgs-bonded-h0-cache",
                    "inputs/obsm_bfgs_common_initial_bonded_h0.npz",
                    "--bfgs-line-search-direction",
                    "paper_p",
                    "--bfgs-state-scope",
                    "curvature_carry",
                    "--save-all-paths",
                ]
            )
    elif method == "fire_stage":
        for lambda_mix, inner_steps, interval, fire_dt in FIRE_STAGE:
            result.append(
                [
                    python,
                    str(source / "run_obsm.py"),
                    *common,
                    "--solver",
                    "fire",
                    "--run-label",
                    "stage20260731",
                    "--output-namespace",
                    "competitor_screen_fire_stage",
                    "--lambda-mix",
                    str(lambda_mix),
                    "--inner-mode",
                    "few_steps",
                    "--inner-steps",
                    str(inner_steps),
                    "--reparam-interval",
                    str(interval),
                    "--inner-stop-norm",
                    "l2",
                    "--fire-mass",
                    "1",
                    "--fire-beta",
                    "0.1",
                    "--fire-dt",
                    str(fire_dt),
                ]
            )
    elif method == "fire_refine":
        for lambda_mix, inner_steps, fire_dt in FIRE_REFINE:
            result.append(
                [
                    python,
                    str(source / "run_obsm.py"),
                    *common,
                    "--solver",
                    "fire",
                    "--run-label",
                    "refine20260731",
                    "--output-namespace",
                    "competitor_screen_fire_refine",
                    "--lambda-mix",
                    str(lambda_mix),
                    "--inner-mode",
                    "few_steps",
                    "--inner-steps",
                    str(inner_steps),
                    "--reparam-interval",
                    "1",
                    "--inner-stop-norm",
                    "l2",
                    "--fire-mass",
                    "1",
                    "--fire-beta",
                    "0.1",
                    "--fire-dt",
                    str(fire_dt),
                ]
            )
    elif method == "fire_finalists":
        for lambda_mix, inner_steps, fire_dt in FIRE_FINALISTS:
            result.append(
                [
                    python,
                    str(source / "run_obsm.py"),
                    *common,
                    "--solver",
                    "fire",
                    "--run-label",
                    "finalists20260731",
                    "--output-namespace",
                    "competitor_screen_fire_finalists",
                    "--lambda-mix",
                    str(lambda_mix),
                    "--inner-mode",
                    "few_steps",
                    "--inner-steps",
                    str(inner_steps),
                    "--reparam-interval",
                    "1",
                    "--inner-stop-norm",
                    "l2",
                    "--fire-mass",
                    "1",
                    "--fire-beta",
                    "0.1",
                    "--fire-dt",
                    str(fire_dt),
                    "--save-all-until",
                    "100",
                    "--path-interval",
                    "25",
                ]
            )
    elif method in {"bfgs", "fire"}:
        fire_steps = FIRE_DT if method == "fire" else (0.02,)
        for lambda_mix in OBSM_LAMBDA:
            for inner_steps in OBSM_INNER:
                for interval in OBSM_REPARAM:
                    for fire_dt in fire_steps:
                        command = [
                            python,
                            str(source / "run_obsm.py"),
                            *common,
                            "--solver",
                            method,
                            "--run-label",
                            "screen20260731",
                            "--output-namespace",
                            f"competitor_screen_{method}",
                            "--lambda-mix",
                            str(lambda_mix),
                            "--inner-mode",
                            "few_steps",
                            "--inner-steps",
                            str(inner_steps),
                            "--reparam-interval",
                            str(interval),
                            "--inner-stop-norm",
                            "l2",
                        ]
                        if method == "bfgs":
                            command.extend(
                                [
                                    "--bfgs-h0",
                                    "identity",
                                    "--bfgs-line-search-direction",
                                    "paper_p",
                                    "--bfgs-state-scope",
                                    "paper_local",
                                ]
                            )
                        else:
                            command.extend(
                                [
                                    "--fire-mass",
                                    "1",
                                    "--fire-beta",
                                    "0.1",
                                    "--fire-dt",
                                    str(fire_dt),
                                ]
                            )
                        result.append(command)
    else:
        raise ValueError(method)
    return result


def run_one(
    index: int,
    total: int,
    command: list[str],
    root: Path,
    log_dir: Path,
) -> dict:
    start = time.time()
    environment = {
        **os.environ,
        "OPENMM_CPU_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(root / "src"),
    }
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log = log_dir / f"{index:04d}.log"
    log.write_text(completed.stdout)
    output = None
    for line in reversed(completed.stdout.splitlines()):
        candidate = Path(line.strip())
        if line.strip().endswith(".npz") and candidate.exists():
            output = str(candidate.resolve())
            break
        if line.startswith("Saved "):
            candidate = Path(line[len("Saved ") :].strip())
            if candidate.exists():
                output = str(candidate.resolve())
                break
    return {
        "index": index,
        "total": total,
        "returncode": completed.returncode,
        "seconds": time.time() - start,
        "command": command,
        "log": str(log.resolve()),
        "result": output,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--method",
        choices=(
            "original",
            "original_fine",
            "original_finalists",
            "original_rank1_long",
            "static",
            "static_fine",
            "static_finalists",
            "bfgs",
            "bfgs_stage",
            "bfgs_curvature_stage",
            "bfgs_carry_stage",
            "bfgs_identity_small_stage",
            "bfgs_identity_mid_stage",
            "bfgs_carry_mid_stage",
            "bfgs_finalists",
            "bfgs_finalists_r5",
            "bfgs_diagnostic_finalists",
            "fire_stage",
            "fire_refine",
            "fire_finalists",
            "fire",
        ),
        required=True,
    )
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    candidate_commands = commands(root, args.method, args.budget)
    log_dir = root / "logs" / f"competitor_screen_{args.method}_budget{args.budget}"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.workers))
    ) as executor:
        futures = [
            executor.submit(
                run_one,
                index,
                len(candidate_commands),
                command,
                root,
                log_dir,
            )
            for index, command in enumerate(candidate_commands, start=1)
        ]
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            row = future.result()
            rows.append(row)
            print(
                f"[{args.method}] {completed}/{len(futures)} "
                f"candidate={row['index']} rc={row['returncode']} "
                f"seconds={row['seconds']:.2f}",
                flush=True,
            )
    rows.sort(key=lambda row: row["index"])
    payload = {
        "method": args.method,
        "budget": int(args.budget),
        "candidate_count": len(candidate_commands),
        "workers": int(args.workers),
        "selection_uses_other_methods": False,
        "initial_path": str((root / "inputs" / "common_initial_path_equidistant.npz").resolve()),
        "rows": rows,
    }
    report = root / "reports" / f"competitor_screen_{args.method}_budget{args.budget}_launch.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, report)
    failures = [row for row in rows if row["returncode"] != 0 or not row["result"]]
    print(report, flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} candidate runs failed; see {report}")


if __name__ == "__main__":
    main()
