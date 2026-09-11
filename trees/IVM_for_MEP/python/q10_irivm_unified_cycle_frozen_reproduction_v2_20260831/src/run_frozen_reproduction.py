"""Fail-closed runner for the two sealed QM10 frozen reproductions.

The optimizer is imported only after seal, source, dependency, environment,
lock, member, and empty-output gates pass.  Formal execution additionally
requires the explicit ``--execute-formal`` command-line switch.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Any, Dict, List, Mapping, Tuple

from .formal_io import (
    FormalIOError,
    HostLocalFcntlLock,
    atomic_savez_compressed,
    atomic_write_json,
    derive_campaign_paths,
    resolve_registered_path,
    sha256_file,
    verify_registration_seal,
)
from .optimizer_interface import FrozenControls, controls_from_registration


FORMAL_EXECUTION_ENABLED = True
OPENBLAS_THREADS = "8"
UNSET_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
EXPECTED_Q_SHA256 = (
    "372962a47504e6df28a20d8b022482a6835680765434b4385a6dba0e826974bc"
)


class RunnerGateError(RuntimeError):
    pass


def _threadpool_identity() -> Tuple[str, List[Dict[str, Any]]]:
    # Import NumPy only after the environment variables have been checked by
    # the caller.  A tiny array operation ensures the BLAS runtime is visible.
    import numpy as np
    from threadpoolctl import threadpool_info

    np.dot(np.ones(1, dtype=np.float64), np.ones(1, dtype=np.float64))
    version = importlib.metadata.version("threadpoolctl")
    normalized = []
    for pool in threadpool_info():
        normalized.append(
            {
                "user_api": str(pool.get("user_api", "")),
                "internal_api": str(pool.get("internal_api", "")),
                "prefix": str(pool.get("prefix", "")),
                "version": str(pool.get("version", "")),
                "num_threads": int(pool.get("num_threads", -1)),
                "threading_layer": str(pool.get("threading_layer", "")),
                "architecture": str(pool.get("architecture", "")),
                "library_basename": Path(str(pool.get("filepath", ""))).name,
            }
        )
    normalized.sort(
        key=lambda item: (
            item["user_api"], item["internal_api"], item["prefix"],
            item["library_basename"],
        )
    )
    return version, normalized


def runtime_identity() -> Dict[str, Any]:
    threadpoolctl_version, pools = _threadpool_identity()
    identity = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": importlib.metadata.version("numpy"),
        "threadpoolctl_version": threadpoolctl_version,
        "threadpool_info": pools,
    }
    try:
        identity["scipy_version"] = importlib.metadata.version("scipy")
    except importlib.metadata.PackageNotFoundError:
        identity["scipy_version"] = "not_installed"
    return identity


def _verify_pools_are_eight(identity: Mapping[str, Any], label: str) -> None:
    pools = identity["threadpool_info"]
    if not pools:
        raise RunnerGateError(label + " threadpool_info found no numerical runtime")
    if not any(pool["internal_api"] == "openblas" for pool in pools):
        raise RunnerGateError(label + " OpenBLAS runtime is not loaded")
    if any(int(pool["num_threads"]) != 8 for pool in pools):
        raise RunnerGateError(label + " numerical pools must all report 8 threads")


def _verify_environment(runtime_contract: Mapping[str, Any]) -> Dict[str, Any]:
    if int(runtime_contract.get("jobs", -1)) != 1:
        raise RunnerGateError("registration must freeze jobs=1")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RunnerGateError("PYTHONHASHSEED must equal 0 before interpreter start")
    expected_environment = {"OPENBLAS_NUM_THREADS": OPENBLAS_THREADS}
    for name in UNSET_THREAD_VARIABLES:
        expected_environment[name] = None
    if runtime_contract.get("thread_environment") != expected_environment:
        raise RunnerGateError("registered thread environment drifted")
    if os.environ.get("OPENBLAS_NUM_THREADS") != OPENBLAS_THREADS:
        raise RunnerGateError("OPENBLAS_NUM_THREADS must equal 8")
    for name in UNSET_THREAD_VARIABLES:
        if name in os.environ:
            raise RunnerGateError(name + " must be unset")

    actual = runtime_identity()
    _verify_pools_are_eight(actual, "preflight")
    if runtime_contract.get("preflight_runtime_identity") != actual:
        raise RunnerGateError("registered preflight runtime identity drifted")
    return actual


def _verify_post_optimizer_environment(
    runtime_contract: Mapping[str, Any]
) -> Dict[str, Any]:
    actual = runtime_identity()
    _verify_pools_are_eight(actual, "post-optimizer")
    if runtime_contract.get("post_optimizer_runtime_identity") != actual:
        raise RunnerGateError("registered post-optimizer runtime identity drifted")
    return actual


def _verify_exact_members(registration: Mapping[str, Any]) -> List[FrozenControls]:
    members = registration.get("registered_members")
    if not isinstance(members, list) or len(members) != 2:
        raise RunnerGateError("registration must contain exactly two members")
    budget = int(registration.get("number_of_forces_budget_per_run", -1))
    controls = [controls_from_registration(member, budget) for member in members]
    by_resolution = {item.n_internal: item for item in controls}
    if set(by_resolution) != {10, 20}:
        raise RunnerGateError("registered members must be N=10 and N=20")
    expected = {
        10: (0.003, "3e-3", 0.1, 7),
        20: (0.001, "1e-3", 0.3, 8),
    }
    for resolution, values in expected.items():
        item = by_resolution[resolution]
        if (item.alpha, item.alpha_label, item.r, item.m_max) != values:
            raise RunnerGateError("registered controls drifted for N=%d" % resolution)
    return [by_resolution[20], by_resolution[10]]


def _verify_hashes(
    campaign_root: Path,
    repository_root: Path,
    registration: Mapping[str, Any],
) -> None:
    source_manifest = registration.get("source_sha256")
    dependency_manifest = registration.get("dependency_sha256")
    if not isinstance(source_manifest, dict) or not source_manifest:
        raise RunnerGateError("source_sha256 manifest is missing")
    if not isinstance(dependency_manifest, dict) or not dependency_manifest:
        raise RunnerGateError("dependency_sha256 manifest is missing")
    for relative, expected in source_manifest.items():
        path = resolve_registered_path(campaign_root, relative)
        if sha256_file(path) != expected:
            raise RunnerGateError("campaign source hash mismatch: " + relative)
    for tagged, expected in dependency_manifest.items():
        if not str(tagged).startswith("repo:"):
            raise RunnerGateError("dependency lacks repo: tag: " + str(tagged))
        relative = str(tagged)[len("repo:"):]
        path = resolve_registered_path(repository_root, relative)
        if sha256_file(path) != expected:
            raise RunnerGateError("repository dependency hash mismatch: " + relative)


def _verify_result_identity(registration: Mapping[str, Any]) -> Dict[str, str]:
    identity = registration.get("result_identity")
    paths = registration.get("result_identity_paths")
    if not isinstance(identity, dict) or not isinstance(paths, dict):
        raise RunnerGateError("result_identity and result_identity_paths are required")
    digest_keys = {
        "algorithm_source_sha256", "optimizer_sha256", "adapter_sha256",
        "runtime_sha256", "backend_sha256", "runner_sha256",
        "exploratory_source_sha256", "preformal_equivalence_sha256",
        "instrumentation_equivalence_sha256",
    }
    if set(paths) != digest_keys:
        raise RunnerGateError("result_identity_paths key set drifted")
    source_manifest = registration["source_sha256"]
    dependency_manifest = registration["dependency_sha256"]
    for key in digest_keys:
        tagged = str(paths[key])
        if tagged.startswith("campaign:"):
            relative = tagged[len("campaign:"):]
            expected = source_manifest.get(relative)
        elif tagged.startswith("repo:"):
            expected = dependency_manifest.get(tagged)
        else:
            raise RunnerGateError("identity path lacks campaign:/repo: tag: " + key)
        if expected is None or identity.get(key) != expected:
            raise RunnerGateError("result identity does not match manifest: " + key)
    if identity.get("optimizer_sha256") != identity.get("algorithm_source_sha256"):
        raise RunnerGateError("optimizer_sha256 must alias algorithm source")
    if identity.get("q_sha256") != EXPECTED_Q_SHA256:
        raise RunnerGateError("registered Q matrix identity drifted")
    required = digest_keys | {"q_sha256"}
    if set(identity) != required:
        raise RunnerGateError("result_identity key set drifted")
    return {str(key): str(value) for key, value in identity.items()}


def _require_empty_formal_outputs(campaign_root: Path) -> None:
    patterns = (
        "runs/n10/*.npz", "runs/n20/*.npz",
        "reports/receipts/n10/*.json", "reports/receipts/n20/*.json",
    )
    existing = [
        str(path)
        for pattern in patterns
        for path in campaign_root.glob(pattern)
    ]
    if existing:
        raise RunnerGateError("formal output boundary is not empty: %r" % existing)


def preflight(registration_path: Path) -> Dict[str, Any]:
    paths = derive_campaign_paths(__file__)
    seal_path = registration_path.with_suffix(".sha256")
    registration, registration_digest = verify_registration_seal(
        registration_path, seal_path
    )
    if registration.get("registered_before_first_formal_reproduction") is not True:
        raise RunnerGateError("registration does not assert pre-run sealing")
    if registration.get("formal_artifacts_present_at_registration") != []:
        raise RunnerGateError("registration records pre-existing formal artifacts")
    controls = _verify_exact_members(registration)
    _verify_hashes(paths.campaign_root, paths.repository_root, registration)
    result_identity = _verify_result_identity(registration)
    actual_runtime = _verify_environment(registration.get("runtime_contract", {}))
    _require_empty_formal_outputs(paths.campaign_root)
    return {
        "paths": paths,
        "registration": registration,
        "registration_path": registration_path.resolve(strict=True),
        "registration_sha256": registration_digest,
        "registration_seal_sha256": sha256_file(seal_path),
        "controls": controls,
        "result_identity": result_identity,
        "preflight_runtime_identity": actual_runtime,
    }


def _public_preflight_report(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "campaign_id": bundle["registration"].get("campaign_id"),
        "registration_sha256": bundle["registration_sha256"],
        "registration_seal_sha256": bundle["registration_seal_sha256"],
        "member_order": [item.n_internal for item in bundle["controls"]],
        "preflight_runtime_identity": bundle["preflight_runtime_identity"],
        "preflight_passed": True,
    }


def _load_after_all_gates() -> Tuple[Any, Any, Any]:
    package = __package__
    optimizer = importlib.import_module(package + ".q10_unified_cycle_optimizer")
    adapter = importlib.import_module(package + ".registered_adapter")
    schema = importlib.import_module(package + ".result_schema")
    if not callable(getattr(optimizer, "run_clean", None)):
        raise RunnerGateError("materialized optimizer lacks run_clean")
    return optimizer, adapter, schema


def _member_identity(
    bundle: Mapping[str, Any], post_optimizer_runtime_identity: Mapping[str, Any]
) -> Dict[str, Any]:
    registration = bundle["registration"]
    return {
        "campaign_id": registration["campaign_id"],
        "registration_sha256": bundle["registration_sha256"],
        "registration_seal_sha256": bundle["registration_seal_sha256"],
        "result_identity": bundle["result_identity"],
        "dependency_manifest": registration["dependency_sha256"],
        "runtime_identity": post_optimizer_runtime_identity,
    }


def _receipt(
    bundle: Mapping[str, Any],
    controls: FrozenControls,
    result_relative_path: str,
    result_sha256: str,
    result: Mapping[str, Any],
    post_optimizer_runtime_identity: Mapping[str, Any],
) -> Dict[str, Any]:
    identity = bundle["result_identity"]
    registration = bundle["registration"]
    return {
        "schema_version": 2,
        "campaign_id": registration["campaign_id"],
        "run_id": controls.run_id,
        "n_internal": int(controls.n_internal),
        "result_relative_path": result_relative_path,
        "result_sha256": result_sha256,
        "registration_relative_path": "registrations/CAMPAIGN_PREREGISTRATION.json",
        "registration_sha256": bundle["registration_sha256"],
        "registration_seal_sha256": bundle["registration_seal_sha256"],
        "algorithm_source_sha256": identity["algorithm_source_sha256"],
        "optimizer_sha256": identity["optimizer_sha256"],
        "adapter_sha256": identity["adapter_sha256"],
        "runtime_sha256": identity["runtime_sha256"],
        "backend_sha256": identity["backend_sha256"],
        "runner_sha256": identity["runner_sha256"],
        "exploratory_source_sha256": identity["exploratory_source_sha256"],
        "preformal_equivalence_sha256": identity["preformal_equivalence_sha256"],
        "instrumentation_equivalence_sha256": identity["instrumentation_equivalence_sha256"],
        "q_sha256": identity["q_sha256"],
        "source_sha256": registration["source_sha256"],
        "dependency_sha256": registration["dependency_sha256"],
        "terminal_nof": int(result["terminal_nf"]),
        "stop_reason": str(result["stop_reason"]),
        "runtime_identity": post_optimizer_runtime_identity,
    }


def execute_formal(bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if not FORMAL_EXECUTION_ENABLED:
        raise RunnerGateError("formal execution is disabled in this sealed source")
    optimizer, adapter, schema = _load_after_all_gates()
    campaign_root = bundle["paths"].campaign_root
    receipts = []
    for controls in bundle["controls"]:
        result, observer = adapter.run_registered_member(optimizer, controls)
        post_runtime = _verify_post_optimizer_environment(
            bundle["registration"]["runtime_contract"]
        )
        arrays = adapter.result_npz_fields(
            result, observer, controls, _member_identity(bundle, post_runtime)
        )
        schema.validate_result_mapping(arrays)
        result_relative = "runs/n%d/%s.npz" % (
            controls.n_internal, controls.run_id
        )
        result_path = campaign_root / result_relative
        atomic_savez_compressed(
            result_path,
            arrays,
            validator=lambda path: schema.validate_result_archive(path),
            refuse_overwrite=True,
        )
        # Reopen the installed path, not only the same-directory temporary.
        schema.validate_result_archive(result_path)
        result_digest = sha256_file(result_path)
        receipt = _receipt(
            bundle, controls, result_relative, result_digest, result, post_runtime
        )
        receipt_path = campaign_root / (
            "reports/receipts/n%d/%s.receipt.json"
            % (controls.n_internal, controls.run_id)
        )
        atomic_write_json(receipt_path, receipt, refuse_overwrite=True)
        receipts.append(receipt)
    return receipts


def main() -> int:
    paths = derive_campaign_paths(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--registration",
        type=Path,
        default=paths.campaign_root / "registrations/CAMPAIGN_PREREGISTRATION.json",
    )
    parser.add_argument("--execute-formal", action="store_true")
    arguments = parser.parse_args()
    lock_path = (
        paths.campaign_root / "locks/host-local-q10-unified-cycle-frozen.lock"
    )
    try:
        with HostLocalFcntlLock(lock_path):
            bundle = preflight(arguments.registration)
            print(json.dumps(_public_preflight_report(bundle), sort_keys=True, indent=2))
            if not arguments.execute_formal:
                return 0
            receipts = execute_formal(bundle)
            print(json.dumps(receipts, sort_keys=True, indent=2))
            return 0
    except (FormalIOError, RunnerGateError, OSError, ValueError, KeyError) as exc:
        print("RUNNER REFUSED: " + str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
