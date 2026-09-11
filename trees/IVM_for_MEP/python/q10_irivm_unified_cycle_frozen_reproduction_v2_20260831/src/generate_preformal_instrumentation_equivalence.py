#!/usr/bin/env python3
"""Fresh-process equivalence gate for registered write-only instrumentation.

This is preformal evidence only.  It directly invokes the frozen numerical
``run_clean`` entry point, never the sealed formal runner, and writes only
``evidence/preformal_instrumentation_equivalence.json``.  Four fresh worker
processes are used: uninstrumented and instrumented at N=20, then the same pair
at N=10.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


THIS_FILE = Path(__file__).resolve()
CAMPAIGN_ROOT = THIS_FILE.parents[1]
REPOSITORY_ROOT = THIS_FILE.parents[3]
OUTPUT = CAMPAIGN_ROOT / "evidence/preformal_instrumentation_equivalence.json"
OPTIMIZER = CAMPAIGN_ROOT / "src/q10_unified_cycle_optimizer.py"
ADAPTER = CAMPAIGN_ROOT / "src/registered_adapter.py"
STATIC_EVIDENCE = CAMPAIGN_ROOT / "evidence/static_contract_report.json"
BROAD_CORE = REPOSITORY_ROOT / (
    "python/q10_irivm_low_r_coherence_secant_v2_20260830/src/"
    "q10_irivm_atomic_tangent_secant_core.py"
)
BACKEND = REPOSITORY_ROOT / (
    "python/q10_irivm_low_r_coherence_secant_v2_20260830/src/"
    "q10_solver_backend.py"
)

THREAD_VARIABLES_UNSET = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
CONFIGURATIONS = {
    20: {
        "alpha": 1.0e-3,
        "alpha_label": "1e-3",
        "r": 0.3,
        "m_max": 8,
        "budget": 3000,
        "expected_terminal_nf": 2981,
        "expected_stop_reason": "budget",
        "expected_record_count": 171,
        "expected_numeric_trajectory_sha256": (
            "22d4969de10ec5d3918e92b4b4c548b739244e1a28f1cf44fac2659f27753c60"
        ),
        "expected_reparam_count": 188,
        "expected_nof_breakdown": {
            "initial_band": 20,
            "recovery_band_trial": 2825,
            "normal_subproblem": 56,
            "direct_spring_check": 40,
            "synchronized_recovery_trial": 40,
        },
    },
    10: {
        "alpha": 3.0e-3,
        "alpha_label": "3e-3",
        "r": 0.1,
        "m_max": 7,
        "budget": 3000,
        "expected_terminal_nf": 782,
        "expected_stop_reason": "stationary",
        "expected_record_count": 90,
        "expected_numeric_trajectory_sha256": (
            "ccd4f916cd9c3017540065eae26ff329a2f3284449754d1a447ccef710107eee"
        ),
        "expected_reparam_count": 134,
        "expected_nof_breakdown": {
            "initial_band": 10,
            "recovery_band_trial": 679,
            "normal_subproblem": 63,
            "direct_spring_check": 10,
            "synchronized_recovery_trial": 20,
        },
    },
}
ALLOWED_REPARAM_CALLERS = {
    "recovery_proposal_spring",
    "synchronized_spring_recovery",
    "severe_rejection_hj_spring_retry",
    "normal_stream_exit_hj_spring",
    "post_main_direct_hj_spring",
}


class EquivalenceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tag(digest: Any, name: bytes, payload: bytes = b"") -> None:
    digest.update(name)
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def canonical_update(digest: Any, value: Any) -> None:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        _tag(digest, b"none")
    elif isinstance(value, bool):
        _tag(digest, b"bool", b"1" if value else b"0")
    elif isinstance(value, int):
        _tag(digest, b"int", str(value).encode("ascii"))
    elif isinstance(value, float):
        _tag(digest, b"float64", struct.pack(">d", value))
    elif isinstance(value, str):
        _tag(digest, b"str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        _tag(digest, b"bytes", value)
    elif isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise EquivalenceError("object array in numerical transcript")
        array = np.ascontiguousarray(value)
        _tag(digest, b"ndarray-dtype", array.dtype.str.encode("ascii"))
        canonical_update(digest, tuple(int(item) for item in array.shape))
        _tag(digest, b"ndarray-bytes", array.tobytes(order="C"))
    elif hasattr(value, "as_dict") and callable(value.as_dict):
        canonical_update(digest, value.as_dict())
    elif is_dataclass(value):
        canonical_update(digest, asdict(value))
    elif isinstance(value, Mapping):
        keys = sorted(value)
        _tag(digest, b"mapping-count", struct.pack(">Q", len(keys)))
        for key in keys:
            if not isinstance(key, str):
                raise EquivalenceError("non-string mapping key in transcript")
            canonical_update(digest, key)
            canonical_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        _tag(digest, b"sequence-count", struct.pack(">Q", len(value)))
        for item in value:
            canonical_update(digest, item)
    else:
        raise EquivalenceError(
            "unsupported numerical transcript type: " + type(value).__name__
        )


def canonical_sha256(value: Any) -> str:
    digest = hashlib.sha256(b"q10-instrumentation-equivalence-v1\x00")
    canonical_update(digest, value)
    return digest.hexdigest()


def numeric_trajectory_sha256(result: Mapping[str, Any]) -> str:
    """Match the trajectory identity already frozen in preformal_equivalence."""

    digest = hashlib.sha256()
    for record in result["records"]:
        digest.update(np.asarray([record["nf"]], dtype=np.int64).tobytes())
        for key in ("path", "energies", "gradients"):
            array = np.ascontiguousarray(record[key], dtype=np.float64)
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(path.name + ".tmp-%d" % os.getpid())
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(str(temporary), str(path))


def environment_contract() -> Dict[str, Any]:
    observed = {
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }
    for name in THREAD_VARIABLES_UNSET:
        observed[name] = os.environ.get(name)
    passed = bool(
        observed["OPENBLAS_NUM_THREADS"] == "8"
        and observed["PYTHONHASHSEED"] == "0"
        and all(observed[name] is None for name in THREAD_VARIABLES_UNSET)
    )
    return {
        "required": {
            "OPENBLAS_NUM_THREADS": "8",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
            "VECLIB_MAXIMUM_THREADS": None,
            "NUMEXPR_NUM_THREADS": None,
        },
        "observed": observed,
        "passed": passed,
    }


def runtime_identity() -> Dict[str, Any]:
    from threadpoolctl import threadpool_info

    pools = []
    for item in threadpool_info():
        pools.append(
            {
                "user_api": item.get("user_api"),
                "internal_api": item.get("internal_api"),
                "prefix": item.get("prefix"),
                "version": item.get("version"),
                "num_threads": item.get("num_threads"),
                "threading_layer": item.get("threading_layer"),
                "architecture": item.get("architecture"),
                "library_basename": Path(str(item.get("filepath", ""))).name,
            }
        )
    pools.sort(
        key=lambda item: (
            str(item.get("internal_api")),
            str(item.get("prefix")),
            str(item.get("version")),
        )
    )
    openblas_ok = bool(
        any(item.get("internal_api") == "openblas" for item in pools)
        and all(
            int(item.get("num_threads", -1)) == 8
            for item in pools
            if item.get("internal_api") == "openblas"
        )
    )
    return {
        "python_implementation": sys.implementation.name,
        "python_version": ".".join(str(item) for item in sys.version_info[:3]),
        "numpy_version": importlib.metadata.version("numpy"),
        "scipy_version": importlib.metadata.version("scipy"),
        "threadpoolctl_version": importlib.metadata.version("threadpoolctl"),
        "threadpool_info": pools,
        "all_openblas_pools_have_8_threads": openblas_ok,
    }


def observer_summary(observer: Any) -> Dict[str, Any]:
    evaluation_digest = canonical_sha256(
        [
            {
                "sequence": item.sequence,
                "nof": item.nof,
                "outer": item.outer,
                "phase": item.phase,
                "image_index": item.image_index,
                "coordinate": item.coordinate,
                "energy": item.energy,
                "gradient": item.gradient,
            }
            for item in observer.evaluations
        ]
    )
    reparam_payload = [
        {
            "event_id": item.event_id,
            "caller": item.caller,
            "outer": item.outer,
            "nof_at_entry": item.nof_at_entry,
            "input_path": item.input_path,
            "fixed_energy": item.fixed_energy,
            "output_path": item.output_path,
            "iterations": item.iterations,
            "final_residual": item.final_residual,
            "tangent_call_start": item.tangent_call_start,
            "tangent_call_count": item.tangent_call_count,
        }
        for item in observer.reparameterizations
    ]
    tangent_payload = [
        {
            "event_id": item.event_id,
            "context": item.context,
            "parent_reparam_event_id": item.parent_reparam_event_id,
            "path_sha256": item.path_sha256,
            "energy_sha256": item.energy_sha256,
            "output_sha256": item.output_sha256,
        }
        for item in observer.tangents
    ]
    caller_counts = {}
    for item in observer.reparameterizations:
        caller_counts[item.caller] = caller_counts.get(item.caller, 0) + 1
    linked = sum(
        int(item.tangent_call_count) for item in observer.reparameterizations
    )
    positive_parents = sum(
        int(item.parent_reparam_event_id > 0) for item in observer.tangents
    )
    return {
        "evaluation_count": len(observer.evaluations),
        "evaluation_transcript_sha256": evaluation_digest,
        "reparam_count": len(observer.reparameterizations),
        "reparam_caller_counts": caller_counts,
        "reparam_transcript_sha256": canonical_sha256(reparam_payload),
        "tangent_count": len(observer.tangents),
        "tangent_mismatch_count": int(observer.tangent_mismatch_count),
        "tangent_transcript_sha256": canonical_sha256(tangent_payload),
        "spring_linked_tangent_count": linked,
        "positive_parent_tangent_count": positive_parents,
        "all_reparam_callers_registered": (
            set(caller_counts).issubset(ALLOWED_REPARAM_CALLERS)
        ),
        "all_spring_tangents_linked_once": linked == positive_parents,
    }


def worker(n_internal: int, mode: str) -> Dict[str, Any]:
    contract = environment_contract()
    if not contract["passed"]:
        raise EquivalenceError("worker environment contract failed")
    from src import q10_unified_cycle_optimizer as optimizer

    config = CONFIGURATIONS[int(n_internal)]
    observer = None
    if mode == "instrumented":
        from src.optimizer_interface import FrozenControls
        from src.registered_adapter import run_registered_member

        controls = FrozenControls(
            int(n_internal),
            float(config["alpha"]),
            str(config["alpha_label"]),
            float(config["r"]),
            int(config["m_max"]),
            int(config["budget"]),
        )
        result, observer = run_registered_member(optimizer, controls)
    elif mode == "uninstrumented":
        result = optimizer.run_clean(
            int(n_internal),
            float(config["alpha"]),
            float(config["r"]),
            int(config["m_max"]),
            budget=int(config["budget"]),
        )
    else:
        raise EquivalenceError("unknown worker mode")

    component = {
        key: canonical_sha256(result[key]) for key in sorted(result)
    }
    summary = {
        "record_count": len(result["records"]),
        "evaluation_trace_length": len(result["evaluation_miss_trace"]),
        "terminal_nf": int(result["terminal_nf"]),
        "stop_reason": str(result["stop_reason"]),
        "per_image_joint_evaluations": np.asarray(
            result["per_image_joint_evaluations"], dtype=np.int64
        ).tolist(),
        "nof_breakdown": {
            str(key): int(value) for key, value in result["nof_breakdown"].items()
        },
        "working_equals_public_at_exit": bool(
            result["working_equals_public_at_exit"]
        ),
        "numeric_trajectory_sha256": numeric_trajectory_sha256(result),
        "aggregate_result_sha256": canonical_sha256(result),
        "component_sha256": component,
    }
    expected_matched = bool(
        summary["record_count"] == int(config["expected_record_count"])
        and summary["terminal_nf"] == int(config["expected_terminal_nf"])
        and summary["stop_reason"] == str(config["expected_stop_reason"])
        and summary["numeric_trajectory_sha256"]
        == str(config["expected_numeric_trajectory_sha256"])
        and summary["nof_breakdown"] == config["expected_nof_breakdown"]
    )
    identity = runtime_identity()
    return {
        "pid": os.getpid(),
        "mode": mode,
        "n_internal": int(n_internal),
        "controls": {
            "alpha": float(config["alpha"]),
            "alpha_label": str(config["alpha_label"]),
            "r": float(config["r"]),
            "m_max": int(config["m_max"]),
            "budget": int(config["budget"]),
        },
        "environment": contract,
        "runtime_identity_after_optimizer": identity,
        "openblas_runtime_contract_passed": bool(
            identity["all_openblas_pools_have_8_threads"]
        ),
        "expected_frozen_trajectory_matched": expected_matched,
        "result": summary,
        "observer": observer_summary(observer) if observer is not None else None,
    }


def worker_environment() -> Dict[str, str]:
    environment = dict(os.environ)
    for name in THREAD_VARIABLES_UNSET:
        environment.pop(name, None)
    environment["OPENBLAS_NUM_THREADS"] = "8"
    environment["PYTHONHASHSEED"] = "0"
    return environment


def run_fresh_worker(n_internal: int, mode: str) -> Dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "src.generate_preformal_instrumentation_equivalence",
        "--worker",
        "--n",
        str(int(n_internal)),
        "--mode",
        mode,
    ]
    completed = subprocess.run(
        command,
        cwd=str(CAMPAIGN_ROOT),
        env=worker_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if completed.returncode != 0:
        raise EquivalenceError(
            "fresh worker failed for N=%d %s\nstdout:\n%s\nstderr:\n%s"
            % (n_internal, mode, completed.stdout, completed.stderr)
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise EquivalenceError(
            "worker emitted non-JSON stdout: " + completed.stdout
        ) from error


def build_controller_report() -> Dict[str, Any]:
    comparisons = []
    all_passed = True
    pids = []
    for n_internal in (20, 10):
        print(
            "preformal equivalence: starting fresh N=%d uninstrumented"
            % n_internal,
            flush=True,
        )
        plain = run_fresh_worker(n_internal, "uninstrumented")
        pids.append(int(plain["pid"]))
        print(
            "preformal equivalence: finished N=%d uninstrumented at NoF=%d"
            % (n_internal, plain["result"]["terminal_nf"]),
            flush=True,
        )
        print(
            "preformal equivalence: starting fresh N=%d instrumented"
            % n_internal,
            flush=True,
        )
        instrumented = run_fresh_worker(n_internal, "instrumented")
        pids.append(int(instrumented["pid"]))
        print(
            "preformal equivalence: finished N=%d instrumented at NoF=%d"
            % (n_internal, instrumented["result"]["terminal_nf"]),
            flush=True,
        )

        plain_components = plain["result"]["component_sha256"]
        instrumented_components = instrumented["result"]["component_sha256"]
        all_keys = sorted(set(plain_components) | set(instrumented_components))
        differing = [
            key for key in all_keys
            if plain_components.get(key) != instrumented_components.get(key)
        ]
        observer = instrumented["observer"]
        expected_reparam = int(CONFIGURATIONS[n_internal]["expected_reparam_count"])
        passed = bool(
            not differing
            and plain["result"]["aggregate_result_sha256"]
            == instrumented["result"]["aggregate_result_sha256"]
            and plain["expected_frozen_trajectory_matched"]
            and instrumented["expected_frozen_trajectory_matched"]
            and plain["environment"]["passed"]
            and instrumented["environment"]["passed"]
            and plain["openblas_runtime_contract_passed"]
            and instrumented["openblas_runtime_contract_passed"]
            and observer["evaluation_count"]
            == instrumented["result"]["terminal_nf"]
            and observer["reparam_count"] == expected_reparam
            and observer["tangent_mismatch_count"] == 0
            and observer["all_reparam_callers_registered"]
            and observer["all_spring_tangents_linked_once"]
        )
        all_passed = all_passed and passed
        comparisons.append(
            {
                "n_internal": n_internal,
                "passed": passed,
                "fresh_process_pair": int(plain["pid"]) != int(instrumented["pid"]),
                "all_result_components_bitwise_equal": not differing,
                "differing_result_components": differing,
                "aggregate_result_sha256_equal": (
                    plain["result"]["aggregate_result_sha256"]
                    == instrumented["result"]["aggregate_result_sha256"]
                ),
                "expected_reparam_count": expected_reparam,
                "uninstrumented": plain,
                "instrumented": instrumented,
            }
        )
    distinct_workers = len(set(pids)) == 4
    all_passed = all_passed and distinct_workers
    return {
        "schema_version": 1,
        "status": "PASS" if all_passed else "FAIL",
        "passed": all_passed,
        "all_passed": all_passed,
        "scope": (
            "preformal fresh-process uninstrumented versus registered-adapter "
            "instrumented complete optimizer runs; not formal campaign runs"
        ),
        "generated_before_registered_run": True,
        "formal_runner_invoked": False,
        "formal_outputs_written": False,
        "worker_order": [
            "N20 uninstrumented",
            "N20 instrumented",
            "N10 uninstrumented",
            "N10 instrumented",
        ],
        "four_distinct_worker_pids": distinct_workers,
        "required_environment": {
            "OPENBLAS_NUM_THREADS": "8",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": None,
            "MKL_NUM_THREADS": None,
            "VECLIB_MAXIMUM_THREADS": None,
            "NUMEXPR_NUM_THREADS": None,
        },
        "sources": {
            "optimizer": {
                "path": "src/q10_unified_cycle_optimizer.py",
                "sha256": sha256_file(OPTIMIZER),
            },
            "adapter": {
                "path": "src/registered_adapter.py",
                "sha256": sha256_file(ADAPTER),
            },
            "broad_core": {
                "path": (
                    "repo:python/q10_irivm_low_r_coherence_secant_v2_20260830/"
                    "src/q10_irivm_atomic_tangent_secant_core.py"
                ),
                "sha256": sha256_file(BROAD_CORE),
            },
            "backend": {
                "path": (
                    "repo:python/q10_irivm_low_r_coherence_secant_v2_20260830/"
                    "src/q10_solver_backend.py"
                ),
                "sha256": sha256_file(BACKEND),
            },
            "static_contract_report": {
                "path": "evidence/static_contract_report.json",
                "sha256": sha256_file(STATIC_EVIDENCE),
            },
            "generator": {
                "path": "src/generate_preformal_instrumentation_equivalence.py",
                "sha256": sha256_file(THIS_FILE),
            },
        },
        "comparison_method": {
            "canonical_result": (
                "recursive type-tagged SHA-256 over every optimizer return key; "
                "float scalars use exact IEEE-754 bits and NumPy arrays include "
                "dtype, shape, and all C-order bytes"
            ),
            "component_gate": (
                "every top-level result component digest must be identical"
            ),
            "known_trajectory_gate": (
                "public numeric trajectory SHA-256, terminal NoF, stop reason, "
                "record count, and NoF categories must match preformal_equivalence.json"
            ),
            "instrumentation_closure": (
                "full evaluation count closes to terminal NoF; every HJ tangent "
                "matches an independent formula; every spring event and parent "
                "tangent relationship closes"
            ),
        },
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--n", type=int, choices=(10, 20))
    parser.add_argument(
        "--mode", choices=("uninstrumented", "instrumented")
    )
    arguments = parser.parse_args()
    if arguments.worker:
        if arguments.n is None or arguments.mode is None:
            parser.error("worker mode requires --n and --mode")
        report = worker(arguments.n, arguments.mode)
        print(
            json.dumps(
                report,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0
    report = build_controller_report()
    canonical_json_write(OUTPUT, report)
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

