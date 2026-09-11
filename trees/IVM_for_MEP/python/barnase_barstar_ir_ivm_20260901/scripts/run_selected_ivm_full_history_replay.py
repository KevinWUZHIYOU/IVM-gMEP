#!/usr/bin/env python3
"""Freeze, authorize, or execute the selected 9990-N_F full-path replay.

This is an output-only replay of the already selected IR-IVM member.  Relative
to the selected registration, the only runtime differences are run label,
output directory, ``path_storage_mode=legacy``, and ``--save-all-paths``.
The optimization budget and every numerical/algorithmic option stay identical.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

import numpy as np


SCHEMA = "bb-irivm-selected-full-history-output-only-replay-v1"
AUTH_SCHEMA = "bb-irivm-selected-full-history-output-only-replay-authorization-v1"
STATUS = "FROZEN_BEFORE_FIRST_FULL_HISTORY_REPLAY_PHYSICAL_CALL"
LABEL = "bb_irivm_a1e0_r01_m05_inf9990_fullhistory_r1"
REFERENCE_LABEL = "bb_irivm_a1e0_r01_m05_inf9990_r1"
BUDGET = 9990
OUTPUT_SUBDIR = "runs/full_history_replay"
EXPECTED_HISTORY_ROWS = 352
EXPECTED_LAST_COMPLETE_CALL = 318
EXPECTED_LAST_COMPLETE_NF = 8982
EXPECTED_TERMINAL_CALL = 352
EXPECTED_TERMINAL_NF = 9985
EXPECTED_TERMINAL_TAIL_NF = 1003
MINIMUM_FREE_BYTES = 12 * 1024**3
THREAD_CAPS = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENMM_CPU_THREADS",
)
LOCKS = (
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"),
    Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"),
)
ALGORITHM_SOURCES = (
    "bb_core.py", "bb_ffprecon.py", "storage_policy.py",
    "run_ivm_v9_spring_projected_production_20260730.py",
    "frozen_discrete_b040.py",
    "run_ivm_v9_discrete_ffrbk_takeover_selector_shim.py",
    "run_ivm_v9_discrete_selected_eta_shim.py",
    "run_ivm_v9_discrete_ffrbk_takeover_grid_shim.py",
    "run_ivm_v9_total_internal_nf_budget.py",
    "run_barnase_barstar_ir_ivm_member.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def bind(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def write_exclusive(path: Path, payload: dict) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def option_value(argv: list[str], flag: str) -> str:
    require(argv.count(flag) == 1, f"command must contain exactly one {flag}")
    index = argv.index(flag)
    require(index + 1 < len(argv), f"command lacks a value for {flag}")
    return argv[index + 1]


def replace_option(argv: list[str], flag: str, value: str) -> None:
    argv[argv.index(flag) + 1] = value


def replay_argv(reference: dict) -> list[str]:
    argv = [str(value) for value in reference["command"]]
    require(option_value(argv, "--run-label") == REFERENCE_LABEL, "reference label drift")
    require(option_value(argv, "--output-dir") == "runs/production", "reference output directory drift")
    require(option_value(argv, "--budget") == str(BUDGET), "reference safety budget drift")
    require(option_value(argv, "--total-internal-nf-budget") == str(BUDGET), "reference total budget drift")
    require(option_value(argv, "--path-storage-mode") == "committed", "reference storage mode drift")
    require("--save-all-paths" not in argv, "reference unexpectedly saves all paths")
    replace_option(argv, "--run-label", LABEL)
    replace_option(argv, "--output-dir", OUTPUT_SUBDIR)
    replace_option(argv, "--path-storage-mode", "legacy")
    argv.append("--save-all-paths")
    return argv


def validate_runtime_delta(reference: list[str], replay: list[str]) -> None:
    require(reference.count("--save-all-paths") == 0, "reference save-all flag drift")
    require(replay.count("--save-all-paths") == 1, "replay requires one save-all flag")
    reduced = replay.copy()
    reduced.remove("--save-all-paths")
    require(len(reference) == len(reduced), "unexpected runtime command length drift")
    allowed_flags = {"--run-label", "--output-dir", "--path-storage-mode"}
    allowed_value_indices = {reference.index(flag) + 1 for flag in allowed_flags}
    differences = {index for index, pair in enumerate(zip(reference, reduced)) if pair[0] != pair[1]}
    require(differences == allowed_value_indices, f"unexpected runtime delta indices: {differences}")
    require(option_value(replay, "--run-label") == LABEL, "replay label mismatch")
    require(option_value(replay, "--output-dir") == OUTPUT_SUBDIR, "replay output mismatch")
    require(option_value(replay, "--path-storage-mode") == "legacy", "replay storage mismatch")
    require(option_value(replay, "--budget") == str(BUDGET), "safety budget changed")
    require(option_value(replay, "--total-internal-nf-budget") == str(BUDGET), "total budget changed")
    for flag, expected in (
        ("--alpha", "1"), ("--discrete-residual-ratio", "0.1"),
        ("--discrete-inner-max-steps", "5"),
        ("--discrete-step-multiplier-eta", "1.1"), ("--max-step", "1"),
        ("--reparam-mode", "state_curve"), ("--reparam-every", "1"),
    ):
        require(option_value(replay, flag) == expected, f"algorithm control changed: {flag}")


def reference_paths(root: Path) -> dict[str, Path]:
    return {
        "selected_member_registration": root / "registrations/production/bb_irivm_a1e0_r01_m05_inf9990_r1.json",
        "selected_terminal_result": root / "runs/production/ivm_bb_irivm_a1e0_r01_m05_inf9990_r1_central_state_ivm_cfge0354bb7178e_budget9990_storagecommitted.npz",
        "selected_terminal_audit": root / "runs/production/ivm_bb_irivm_a1e0_r01_m05_inf9990_r1_central_state_ivm_cfge0354bb7178e_budget9990_storagecommitted.audit.json",
        "selected_terminal_verification": root / "reports/verifications/BB_IRIVM_A1E0_R01_M05_INF9990_R1_VERIFICATION_20260902.json",
        "migration_contract": root / "contracts/MIGRATION_CONTRACT.json",
        "metric_policy": root / "registrations/USER_METRIC_POLICY_AMENDMENT_20260901_R1.json",
        "common_initial_path": root / "inputs/common_initial_path32.npz",
        "endpoint_cache": root / "inputs/endpoint_joint_ef_cache.npz",
        "topology": root / "inputs/complex_fixed.pdb",
    }


def sealed_history_facts(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        nf = np.asarray(saved["total_internal_joint_evaluations"], dtype=np.int64)
        require(calls.shape == (EXPECTED_HISTORY_ROWS,), "sealed history row count drift")
        require(np.array_equal(calls, np.arange(1, EXPECTED_HISTORY_ROWS + 1)), "sealed calls are not 1..352")
        require(int(nf[-1]) == EXPECTED_TERMINAL_NF, "sealed terminal N_F drift")
        require(int(np.asarray(saved["total_internal_nf_last_complete_state_nf"]).item()) == EXPECTED_LAST_COMPLETE_NF, "sealed last complete N_F drift")
        require(int(np.asarray(saved["total_internal_nf_last_complete_state_index"]).item()) + 1 == EXPECTED_LAST_COMPLETE_CALL, "sealed last complete call drift")
        require(bool(np.asarray(saved["total_internal_nf_terminal_partial_tail_quarantined"]).item()), "sealed tail is not quarantined")
        require(int(np.asarray(saved["total_internal_nf_terminal_uncommitted_charged_tail_nf"]).item()) == EXPECTED_TERMINAL_TAIL_NF, "sealed tail N_F drift")
        return {
            "external_iteration_count": int(calls.size),
            "stored_calls_required": [int(value) for value in calls],
            "formally_reportable_call_range_after_exact_verification": [1, EXPECTED_LAST_COMPLETE_CALL],
            "excluded_terminal_uncommitted_tail_call_range": [EXPECTED_LAST_COMPLETE_CALL + 1, EXPECTED_TERMINAL_CALL],
            "last_complete_optimization_N_F": EXPECTED_LAST_COMPLETE_NF,
            "terminal_consumed_optimization_N_F": EXPECTED_TERMINAL_NF,
            "terminal_uncommitted_charged_tail_N_F": EXPECTED_TERMINAL_TAIL_NF,
        }


def freeze(root: Path, output: Path) -> None:
    paths = reference_paths(root)
    for role, path in paths.items():
        require(path.is_file(), f"missing {role}: {path}")
    reference = read_json(paths["selected_member_registration"])
    original = [str(value) for value in reference["command"]]
    runtime = replay_argv(reference)
    validate_runtime_delta(original, runtime)
    verifier = root / "scripts/verify_selected_ivm_full_history_replay.py"
    require(verifier.is_file(), f"missing post-run verifier: {verifier}")
    sources = {name: root / "src" / name for name in ALGORITHM_SOURCES}
    sources[Path(__file__).name] = Path(__file__).resolve(strict=True)
    sources[verifier.name] = verifier
    history = sealed_history_facts(paths["selected_terminal_result"])
    raw_history_bytes = EXPECTED_HISTORY_ROWS * 32 * 9396 * 8
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_utc": utc_now(),
        "execution_authorized_by_this_file": False,
        "purpose": "exact selected 9990-N_F replay retaining every external-iteration coordinate path for future offline diagnostics",
        "label": LABEL,
        "reference_label": REFERENCE_LABEL,
        "runtime_argv": runtime,
        "only_permitted_runtime_differences_from_selected_9990": {
            "--run-label": LABEL,
            "--output-dir": OUTPUT_SUBDIR,
            "--path-storage-mode": "legacy",
            "--save-all-paths": True,
        },
        "algorithm_identity": {
            "all_numerical_controls_unchanged": True,
            "optimization_N_F_budget": BUDGET,
            "alpha": 1.0, "r": 0.1, "m_max": 5, "eta": 1.1,
            "reparameterization": "state_curve every outer iteration with unchanged direct proposal and recovery",
            "fresh_from_common_initial_path": True,
            "warm_start": False,
        },
        "path_retention": {
            "path_storage_mode": "legacy", "save_all_paths": True,
            "path_storage_affects_online_algorithm": False,
            "expected_path_shape": [32, 9396],
            **history,
            "postrun_exact_prefix_verification_required": True,
            "terminal_tail_must_be_stored_but_never_reported": True,
        },
        "resource_estimate": {
            "reference_M4_wall_seconds": 11286.506678819656,
            "expected_M4_wall_time": "3.2--3.5 hours including final serialization",
            "raw_stored_path_bytes": raw_history_bytes,
            "raw_stored_path_GiB": raw_history_bytes / 1024**3,
            "expected_result_disk_GiB": "0.8--1.1",
            "estimated_incremental_peak_RAM_GiB_from_full_history": "1.6--2.4",
            "minimum_free_disk_bytes_before_child": MINIMUM_FREE_BYTES,
        },
        "concurrency": {
            "one_BB_task_on_host": True, "jobs": 1, "all_thread_caps": 1,
            "two_host_local_fcntl_locks_required_before_child": True,
            "minimum_free_disk_bytes_before_child": MINIMUM_FREE_BYTES,
            "dedicated_Mutagen_session_must_be_connected_and_flushed": "bb-irivm-20260901",
        },
        "bindings": {role: bind(path) for role, path in paths.items()},
        "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
    }
    write_exclusive(output, payload)
    print(output.expanduser().resolve())
    print(sha256_file(output.expanduser().resolve(strict=True)))


def validate_registration(root: Path, registration_path: Path) -> dict:
    registration = read_json(registration_path)
    require(registration.get("schema") == SCHEMA and registration.get("status") == STATUS, "registration identity/status mismatch")
    require(registration.get("label") == LABEL, "registration label mismatch")
    for role, record in registration["bindings"].items():
        require(bind(Path(record["path"])) == record, f"registration binding drift: {role}")
    for name, expected in registration["source_sha256"].items():
        path = root / "scripts" / name if name.startswith(("run_selected_", "verify_selected_")) else root / "src" / name
        require(sha256_file(path.resolve(strict=True)) == expected, f"source drift: {name}")
    reference = read_json(Path(registration["bindings"]["selected_member_registration"]["path"]))
    validate_runtime_delta([str(value) for value in reference["command"]], [str(value) for value in registration["runtime_argv"]])
    return registration


def authorize(root: Path, registration_path: Path, output: Path) -> None:
    registration_path = registration_path.expanduser().resolve(strict=True)
    registration = validate_registration(root, registration_path)
    payload = {
        "schema": AUTH_SCHEMA, "enabled": True, "issued_utc": utc_now(),
        "authorization_scope": "one exact fresh selected 9990-N_F output-only full-history replay",
        "label": LABEL, "registration": bind(registration_path),
        "runtime_argv_sha256": hashlib.sha256(json.dumps(registration["runtime_argv"], separators=(",", ":")).encode()).hexdigest(),
        "required_thread_caps": {name: "1" for name in THREAD_CAPS},
        "required_lock_paths": [str(path) for path in LOCKS],
        "minimum_free_disk_bytes_before_child": MINIMUM_FREE_BYTES,
        "required_Mutagen_session": "bb-irivm-20260901",
    }
    write_exclusive(output, payload)
    print(output.expanduser().resolve())
    print(sha256_file(output.expanduser().resolve(strict=True)))


def require_parent_lock_holder() -> None:
    parent = os.getppid()
    for path in LOCKS:
        result = subprocess.run(["/usr/sbin/lsof", "-t", str(path)], check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(result.returncode in (0, 1), f"lsof failed for {path}")
        require(parent in {int(line) for line in result.stdout.splitlines() if line.strip()}, f"parent does not hold {path}")


def execute(root: Path, registration_path: Path, authorization_path: Path) -> None:
    registration_path = registration_path.expanduser().resolve(strict=True)
    authorization_path = authorization_path.expanduser().resolve(strict=True)
    registration = validate_registration(root, registration_path)
    authorization = read_json(authorization_path)
    require(authorization.get("schema") == AUTH_SCHEMA and authorization.get("enabled") is True, "authorization disabled/invalid")
    require(authorization.get("label") == LABEL and authorization.get("registration") == bind(registration_path), "authorization binding mismatch")
    require(not {name: os.environ.get(name) for name in THREAD_CAPS if os.environ.get(name) != "1"}, "all six thread caps must equal 1")
    require_parent_lock_holder()
    free = shutil.disk_usage(root).free
    require(free >= int(authorization["minimum_free_disk_bytes_before_child"]), f"disk gate failed: {free} < {authorization['minimum_free_disk_bytes_before_child']}")
    output_dir = root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    require(not list(output_dir.glob(f"ivm_{LABEL}_*.npz")), "refusing to overwrite a full-history result")
    require(not list(output_dir.glob(f"ivm_{LABEL}_*.audit.json")), "refusing to overwrite a full-history audit")
    sys.path.insert(0, str(root / "src"))
    import run_ivm_v9_discrete_ffrbk_takeover_grid_shim as runtime  # pylint: disable=import-outside-toplevel
    runtime.main([str(value) for value in registration["runtime_argv"]])
    results = list(output_dir.glob(f"ivm_{LABEL}_*.npz"))
    audits = list(output_dir.glob(f"ivm_{LABEL}_*.audit.json"))
    require(len(results) == len(audits) == 1, "full-history replay did not produce one result/audit pair")
    print(results[0])
    print(audits[0])


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    modes = parser.add_subparsers(dest="mode", required=True)
    freeze_mode = modes.add_parser("freeze")
    freeze_mode.add_argument("--output", type=Path, required=True)
    auth_mode = modes.add_parser("authorize")
    auth_mode.add_argument("--registration", type=Path, required=True)
    auth_mode.add_argument("--output", type=Path, required=True)
    run_mode = modes.add_parser("execute")
    run_mode.add_argument("--registration", type=Path, required=True)
    run_mode.add_argument("--authorization", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root = args.root.expanduser().resolve(strict=True)
    if args.mode == "freeze":
        freeze(root, args.output)
    elif args.mode == "authorize":
        authorize(root, args.registration, args.output)
    else:
        execute(root, args.registration, args.authorization)


if __name__ == "__main__":
    main()
