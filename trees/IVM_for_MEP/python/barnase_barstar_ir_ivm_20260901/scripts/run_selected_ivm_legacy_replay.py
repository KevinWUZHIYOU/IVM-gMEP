#!/usr/bin/env python3
"""Prospectively register and run an output-only IR-IVM path-history replay.

The selected Barnase--Barstar IR-IVM production result retained only the
initial and reparameterization-committed paths.  A cost-to-precision curve
needs a few additional *fully evaluated current paths*.  This entry point
replays the exact selected algorithm from the common initial path while
changing only:

* the hard stop from 9990 to 1500 internal force evaluations; and
* the output-only path-history policy from ``committed`` to ``legacy``.

The legacy policy stores the pre-existing milestone calls
1, 2, 5, 10, 20, and 50 before the 1500-N_F stop.  It is explicitly not
used by any online numerical decision.  The terminal partial tail is never an
eligible reporting state.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable


SCHEMA = "bb-irivm-selected-output-only-legacy-replay-v1"
AUTH_SCHEMA = "bb-irivm-selected-output-only-legacy-replay-authorization-v1"
STATUS = "FROZEN_BEFORE_FIRST_REPLAY_PHYSICAL_CALL"
LABEL = "bb_irivm_a1e0_r01_m05_inf1500_legacy_replay_r1"
REFERENCE_LABEL = "bb_irivm_a1e0_r01_m05_inf9990_r1"
REPLAY_BUDGET = 1500
OUTPUT_SUBDIR = "runs/dense_replay"
EXPECTED_LEGACY_CALLS = (1, 2, 5, 10, 20, 50)
THREAD_CAPS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENMM_CPU_THREADS",
)
LOCKS = (
    Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"),
    Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"),
)
ALGORITHM_SOURCES = (
    "bb_core.py",
    "bb_ffprecon.py",
    "storage_policy.py",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_exclusive(path: Path, payload: dict) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def file_binding(path: Path) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def option_value(argv: list[str], flag: str) -> str:
    require(argv.count(flag) == 1, f"runtime command must contain one {flag}")
    index = argv.index(flag)
    require(index + 1 < len(argv), f"runtime command lacks a value for {flag}")
    return argv[index + 1]


def replace_option(argv: list[str], flag: str, value: str) -> None:
    index = argv.index(flag)
    argv[index + 1] = value


def replay_argv(reference: dict) -> list[str]:
    argv = [str(value) for value in reference["command"]]
    require(option_value(argv, "--run-label") == REFERENCE_LABEL, "reference run label drift")
    require(option_value(argv, "--budget") == "9990", "reference safety budget drift")
    require(
        option_value(argv, "--total-internal-nf-budget") == "9990",
        "reference total budget drift",
    )
    require(option_value(argv, "--path-storage-mode") == "committed", "reference storage drift")
    replace_option(argv, "--run-label", LABEL)
    replace_option(argv, "--output-dir", OUTPUT_SUBDIR)
    replace_option(argv, "--budget", str(REPLAY_BUDGET))
    replace_option(argv, "--total-internal-nf-budget", str(REPLAY_BUDGET))
    replace_option(argv, "--path-storage-mode", "legacy")
    return argv


def validate_runtime_delta(reference: list[str], replay: list[str]) -> None:
    require(len(reference) == len(replay), "runtime command length drift")
    allowed_flags = {
        "--run-label",
        "--output-dir",
        "--budget",
        "--total-internal-nf-budget",
        "--path-storage-mode",
    }
    allowed_value_indices = {reference.index(flag) + 1 for flag in allowed_flags}
    differences = {index for index, pair in enumerate(zip(reference, replay)) if pair[0] != pair[1]}
    require(differences == allowed_value_indices, f"unexpected runtime delta indices: {differences}")
    require(option_value(replay, "--alpha") == "1", "alpha changed")
    require(option_value(replay, "--discrete-residual-ratio") == "0.1", "r changed")
    require(option_value(replay, "--discrete-inner-max-steps") == "5", "m_max changed")
    require(option_value(replay, "--discrete-step-multiplier-eta") == "1.1", "eta changed")
    require(option_value(replay, "--max-step") == "1", "maximum step changed")
    require(option_value(replay, "--reparam-mode") == "state_curve", "reparameterization changed")
    require(option_value(replay, "--reparam-every") == "1", "reparameterization cadence changed")


def reference_paths(root: Path) -> dict[str, Path]:
    return {
        "selected_member_registration": root / "registrations/production/bb_irivm_a1e0_r01_m05_inf9990_r1.json",
        "selected_terminal_result": root / "runs/production/ivm_bb_irivm_a1e0_r01_m05_inf9990_r1_central_state_ivm_cfge0354bb7178e_budget9990_storagecommitted.npz",
        "selected_terminal_audit": root / "runs/production/ivm_bb_irivm_a1e0_r01_m05_inf9990_r1_central_state_ivm_cfge0354bb7178e_budget9990_storagecommitted.audit.json",
        "selected_terminal_verification": root / "reports/verifications/BB_IRIVM_A1E0_R01_M05_INF9990_R1_VERIFICATION_20260902.json",
        "selected_exact_prefix_verification": root / "reports/verifications/BB_IRIVM_SELECTED_9990_EXACT_4080_PREFIX_20260902_R1.json",
        "migration_contract": root / "contracts/MIGRATION_CONTRACT.json",
        "metric_policy": root / "registrations/USER_METRIC_POLICY_AMENDMENT_20260901_R1.json",
        "common_initial_path": root / "inputs/common_initial_path32.npz",
        "endpoint_cache": root / "inputs/endpoint_joint_ef_cache.npz",
        "topology": root / "inputs/complex_fixed.pdb",
    }


def freeze(root: Path, output: Path) -> None:
    paths = reference_paths(root)
    for role, path in paths.items():
        require(path.is_file(), f"missing {role}: {path}")
    reference = read_json(paths["selected_member_registration"])
    require(reference.get("label") == REFERENCE_LABEL, "selected registration label mismatch")
    original_argv = [str(value) for value in reference["command"]]
    runtime_argv = replay_argv(reference)
    validate_runtime_delta(original_argv, runtime_argv)
    source_paths = {name: root / "src" / name for name in ALGORITHM_SOURCES}
    source_paths[Path(__file__).name] = Path(__file__).resolve(strict=True)
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_utc": utc_now(),
        "purpose": (
            "Replay the exact selected IR-IVM prefix solely to retain a prospectively "
            "fixed set of fully evaluated current paths for an offline two-metric curve."
        ),
        "label": LABEL,
        "reference_label": REFERENCE_LABEL,
        "algorithm_controls": {
            "alpha": 1.0,
            "r": 0.1,
            "m_max": 5,
            "eta": 1.1,
            "history": 7,
            "physical_max_step_A": 1.0,
            "reparameterization": "state_curve every outer iteration with the unchanged direct proposal and recovery transaction",
        },
        "replay": {
            "fresh_from_common_initial_path": True,
            "warm_start": False,
            "total_internal_optimization_N_F_budget": REPLAY_BUDGET,
            "fixed_endpoint_optimization_N_F": 0,
            "path_storage_mode": "legacy",
            "path_storage_affects_online_algorithm": False,
            "prospective_reporting_calls": list(EXPECTED_LEGACY_CALLS),
            "eligible_state_definition": (
                "fully evaluated current path at a prospectively registered legacy call; "
                "terminal partial/uncommitted tail excluded"
            ),
            "output_subdir": OUTPUT_SUBDIR,
        },
        "runtime_argv": runtime_argv,
        "only_permitted_runtime_differences_from_selected_9990": {
            "--run-label": LABEL,
            "--output-dir": OUTPUT_SUBDIR,
            "--budget": str(REPLAY_BUDGET),
            "--total-internal-nf-budget": str(REPLAY_BUDGET),
            "--path-storage-mode": "legacy",
        },
        "concurrency": {
            "one_BB_task_on_host": True,
            "jobs": 1,
            "all_thread_caps": 1,
            "two_host_local_fcntl_locks_required_before_child": True,
            "minimum_free_disk_bytes_before_child": 8 * 1024**3,
            "dedicated_Mutagen_session_must_be_connected": "bb-irivm-20260901",
        },
        "diagnostic_feedback_into_optimization": "forbidden",
        "bindings": {role: file_binding(path) for role, path in paths.items()},
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
    }
    write_exclusive(output, payload)
    print(output)
    print(sha256_file(output))


def validate_registration(root: Path, registration_path: Path) -> dict:
    registration = read_json(registration_path)
    require(registration.get("schema") == SCHEMA, "registration schema mismatch")
    require(registration.get("status") == STATUS, "registration is not prospectively frozen")
    require(registration.get("label") == LABEL, "registration label mismatch")
    for role, record in registration["bindings"].items():
        path = Path(record["path"]).expanduser().resolve(strict=True)
        require(sha256_file(path) == record["sha256"], f"registration binding drift: {role}")
    for name, expected in registration["source_sha256"].items():
        path = Path(__file__).resolve() if name == Path(__file__).name else root / "src" / name
        require(sha256_file(path) == expected, f"source drift: {name}")
    reference = read_json(Path(registration["bindings"]["selected_member_registration"]["path"]))
    reference_argv = [str(value) for value in reference["command"]]
    runtime_argv = [str(value) for value in registration["runtime_argv"]]
    validate_runtime_delta(reference_argv, runtime_argv)
    require(int(option_value(runtime_argv, "--budget")) == REPLAY_BUDGET, "replay budget drift")
    return registration


def authorize(root: Path, registration_path: Path, output: Path) -> None:
    registration_path = registration_path.expanduser().resolve(strict=True)
    registration = validate_registration(root, registration_path)
    payload = {
        "schema": AUTH_SCHEMA,
        "enabled": True,
        "issued_utc": utc_now(),
        "authorization_scope": "one exact fresh selected-prefix output-only replay",
        "label": LABEL,
        "registration": file_binding(registration_path),
        "runtime_argv_sha256": hashlib.sha256(
            json.dumps(registration["runtime_argv"], separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "required_thread_caps": {name: "1" for name in THREAD_CAPS},
        "required_lock_paths": [str(path) for path in LOCKS],
        "minimum_free_disk_bytes_before_child": int(
            registration["concurrency"]["minimum_free_disk_bytes_before_child"]
        ),
        "diagnostic_feedback_into_optimization": "forbidden",
    }
    write_exclusive(output, payload)
    print(output)
    print(sha256_file(output))


def require_parent_lock_holder() -> None:
    parent = os.getppid()
    for path in LOCKS:
        result = subprocess.run(
            ["/usr/sbin/lsof", "-t", str(path)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        require(result.returncode in (0, 1), f"lsof failed for {path}")
        holders = {int(line) for line in result.stdout.splitlines() if line.strip()}
        require(parent in holders, f"parent process does not hold {path}")


def execute(root: Path, registration_path: Path, authorization_path: Path) -> None:
    registration_path = registration_path.expanduser().resolve(strict=True)
    authorization_path = authorization_path.expanduser().resolve(strict=True)
    registration = validate_registration(root, registration_path)
    authorization = read_json(authorization_path)
    require(authorization.get("schema") == AUTH_SCHEMA, "authorization schema mismatch")
    require(authorization.get("enabled") is True, "replay is not authorized")
    require(authorization.get("label") == LABEL, "authorization label mismatch")
    require(
        authorization["registration"] == file_binding(registration_path),
        "authorization/registration binding mismatch",
    )
    invalid_caps = {name: os.environ.get(name) for name in THREAD_CAPS if os.environ.get(name) != "1"}
    require(not invalid_caps, f"all thread caps must equal 1 before Python starts: {invalid_caps}")
    require_parent_lock_holder()
    output_dir = root / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob(f"ivm_{LABEL}_*.npz"))
    existing += sorted(output_dir.glob(f"ivm_{LABEL}_*.audit.json"))
    require(not existing, f"refusing to overwrite an existing replay artifact: {existing}")
    src = root / "src"
    sys.path.insert(0, str(src))
    import run_ivm_v9_discrete_ffrbk_takeover_grid_shim as runtime  # pylint: disable=import-outside-toplevel

    runtime.main([str(value) for value in registration["runtime_argv"]])
    results = sorted(output_dir.glob(f"ivm_{LABEL}_*.npz"))
    audits = sorted(output_dir.glob(f"ivm_{LABEL}_*.audit.json"))
    require(len(results) == 1 and len(audits) == 1, "replay did not produce one result/audit pair")
    print(results[0])
    print(audits[0])


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    subparsers = parser.add_subparsers(dest="mode", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--output", type=Path, required=True)
    auth_parser = subparsers.add_parser("authorize")
    auth_parser.add_argument("--registration", type=Path, required=True)
    auth_parser.add_argument("--output", type=Path, required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--registration", type=Path, required=True)
    execute_parser.add_argument("--authorization", type=Path, required=True)
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
