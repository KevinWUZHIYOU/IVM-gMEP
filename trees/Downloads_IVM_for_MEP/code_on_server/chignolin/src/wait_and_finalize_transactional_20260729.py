#!/usr/bin/env python3
"""Wait for the live alpha grid, repair half-cycles, then finalize outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional


V2_WAIT = {
    0.2: "ivm_production_v2_universal_a0p2_central_state_ivm_*_budget500.npz",
    0.3: "ivm_production_v2_universal_a0p3_central_state_ivm_*_budget500.npz",
    0.5: "ivm_production_v2_universal_a0p5_central_state_ivm_*_budget500.npz",
    1.0: (
        "ivm_production_a1_early_block_relock_ls4_v2_"
        "central_state_ivm_*_budget500.npz"
    ),
}
V3_WAIT = {
    alpha: (
        "ivm_production_v3_transactional_"
        f"a{str(alpha).replace('.', 'p')}_central_state_ivm_*_budget500.npz"
    )
    for alpha in (5, 10, 15, 20, 30, 50, 70, 100, 300, 1000)
}
V2_REUSE_CHECK = {
    0.2: V2_WAIT[0.2],
    0.3: V2_WAIT[0.3],
    0.5: V2_WAIT[0.5],
    1.0: V2_WAIT[1.0],
    2.0: "ivm_production_v2_universal_a2_central_state_ivm_*_budget500.npz",
    3.0: "ivm_production_v2_universal_a3_central_state_ivm_*_budget500.npz",
}


def one_complete(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous result pattern {pattern}: {matches}")
    if not matches:
        return None
    result = matches[0].resolve()
    return result if result.with_suffix(".audit.json").exists() else None


def write_status(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps({"updated_unix": time.time(), **payload}, indent=2)
        + "\n"
    )
    os.replace(temporary, path)


def load_audit_module(path: Path):
    spec = importlib.util.spec_from_file_location(
        "transactional_alpha_audit", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--namespace",
        default="transactional_alpha_final_20260729",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    runs = root / "runs" / "ivm"
    src = root / "src"
    report = root / "reports" / args.namespace
    logs = root / "logs" / args.namespace
    status = report / "LIVE_FINALIZER_STATUS.json"
    logs.mkdir(parents=True, exist_ok=True)

    wait_patterns = {**V2_WAIT, **V3_WAIT}
    while True:
        complete = {}
        missing = []
        for alpha, pattern in wait_patterns.items():
            result = one_complete(runs, pattern)
            if result is None:
                missing.append(alpha)
            else:
                complete[str(alpha)] = str(result)
        write_status(
            status,
            phase="waiting_for_optimization_results",
            complete=complete,
            missing=missing,
        )
        if not missing:
            break
        time.sleep(max(5.0, float(args.poll_seconds)))

    audit_module = load_audit_module(
        src / "audit_transactional_alpha_grid_20260729.py"
    )
    replacements = []
    v2_records = {}
    for alpha, pattern in V2_REUSE_CHECK.items():
        result = one_complete(runs, pattern)
        if result is None:
            raise RuntimeError(f"V2 audit target disappeared: alpha={alpha}")
        try:
            record = audit_module.audit_result(
                result,
                "overnight exact-reuse classification",
                alpha,
                "v2_exact_reuse",
            )
            v2_records[str(alpha)] = {
                "result": str(result),
                "accepted_for_exact_reuse": True,
                "final_segment_length_cv": record[
                    "final_segment_length_cv"
                ],
                "final_i_perp": record["final_i_perp"],
            }
        except RuntimeError as error:
            if "must be rerun transactionally" not in str(error):
                raise
            replacements.append(alpha)
            v2_records[str(alpha)] = {
                "result": str(result),
                "accepted_for_exact_reuse": False,
                "reason": str(error),
            }
    write_status(
        status,
        phase="classified_v2_terminal_states",
        v2_classification=v2_records,
        transactional_replacements=replacements,
    )

    launcher = (
        root
        / "logs"
        / "fixed_alpha_v2_universal_20260729"
        / "launch_v2_fixed_alpha.zsh"
    )
    for alpha in replacements:
        log = logs / f"transactional_replacement_alpha_{alpha:g}.log"
        # Every OpenMM CPU context owns a worker pool.  Replacement runs are
        # therefore deliberately serialized: concurrent contexts oversubscribe
        # this eight-core workstation without changing the audited algorithm.
        with log.open("w") as stream:
            completed = subprocess.run(
                ["/bin/zsh", str(launcher), f"{alpha:g}", "500"],
                cwd=root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        status_code = completed.returncode
        if status_code:
            raise RuntimeError(
                f"Transactional replacement alpha={alpha:g} failed; see {log}"
            )

    write_status(
        status,
        phase="running_final_pipeline",
        v2_classification=v2_records,
        transactional_replacements=replacements,
    )
    pipeline_log = logs / "08_final_pipeline_driver.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(src)
    command = [
        str(python),
        str(src / "finalize_transactional_alpha_pipeline_20260729.py"),
        "--root",
        str(root),
        "--python",
        str(python),
        "--namespace",
        args.namespace,
    ]
    with pipeline_log.open("w") as stream:
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        write_status(
            status,
            phase="final_pipeline_failed",
            pipeline_log=str(pipeline_log),
            return_code=completed.returncode,
            v2_classification=v2_records,
            transactional_replacements=replacements,
        )
        raise RuntimeError(f"Final pipeline failed; see {pipeline_log}")
    final_manifest = report / "FINAL_PIPELINE_MANIFEST.json"
    write_status(
        status,
        phase="complete",
        final_manifest=str(final_manifest),
        v2_classification=v2_records,
        transactional_replacements=replacements,
    )
    print(final_manifest)


if __name__ == "__main__":
    main()
