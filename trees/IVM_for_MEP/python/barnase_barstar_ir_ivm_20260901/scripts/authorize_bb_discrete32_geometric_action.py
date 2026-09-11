#!/usr/bin/env python3
"""Authorize the BB discrete-32 geometric-action diagnostic after preflight."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = ROOT / "scripts/evaluate_bb_discrete32_geometric_action.py"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("bb_discrete32_action_authorizer", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def atomic_json_no_overwrite(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary file: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv=None) -> None:
    evaluator = load_evaluator()
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=evaluator.CONTRACT)
    parser.add_argument("--preflight", type=Path, default=evaluator.PREAUTH)
    parser.add_argument("--output", type=Path, default=evaluator.AUTHORIZATION)
    args = parser.parse_args(argv)

    evaluator.check_thread_caps()
    evaluator.check_imported_bb_core()
    contract_path = args.contract.expanduser().resolve(strict=True)
    preflight_path = args.preflight.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    evaluator.require(contract_path == evaluator.CONTRACT.resolve(), "noncanonical geometric-action contract")
    evaluator.require(preflight_path == evaluator.PREAUTH.resolve(), "noncanonical geometric-action preflight")
    evaluator.require(output == evaluator.AUTHORIZATION.resolve(), "noncanonical geometric-action authorization")
    evaluator.require(not output.exists(), "authorization already exists")

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    evaluator.require(contract.get("schema") == evaluator.CONTRACT_SCHEMA, "contract schema mismatch")
    evaluator.require(contract.get("status") == evaluator.CONTRACT_STATUS, "contract status mismatch")
    evaluator.require(contract.get("jobs") == 1 and contract.get("host_role") == "local", "contract host/jobs mismatch")
    evaluator.require(contract.get("evaluator") == evaluator.file_binding(EVALUATOR), "contract evaluator binding mismatch")
    evaluator.require(contract.get("execution_authorizer") == evaluator.file_binding(Path(__file__).resolve()), "contract authorizer binding mismatch")
    code = contract.get("code_and_topology", {})
    expected_code = {
        "evaluator": EVALUATOR,
        "contract_freezer": evaluator.FREEZER,
        "execution_authorizer": Path(__file__).resolve(),
        "plotter": evaluator.PLOTTER,
        "tests": evaluator.TEST_FILE,
        "bb_core": evaluator.BB_CORE,
        "topology": evaluator.TOPOLOGY,
    }
    for role, expected in expected_code.items():
        evaluator.require(
            evaluator.resolve_binding(contract_path, code.get(role, {}), role)
            == Path(expected).resolve(strict=True),
            f"contract {role} binding mismatch",
        )
    source_artifacts = contract.get("source_artifacts", {})
    expected_sources = {
        "IVM_exact_path_and_energy": evaluator.IR_ENERGY_RESULT,
        "IVM_exact_path_and_energy_audit": evaluator.IR_ENERGY_AUDIT,
        "fixed_endpoint_joint_EF_cache": evaluator.ENDPOINT_CACHE,
        "ODE_NEB_internal_joint_EF": evaluator.ODE_SOURCE,
    }
    for role, expected in expected_sources.items():
        evaluator.require(
            evaluator.resolve_binding(contract_path, source_artifacts.get(role, {}), role)
            == Path(expected).resolve(strict=True),
            f"contract {role} binding mismatch",
        )
    evaluator.require(
        evaluator.resolve_binding(contract_path, contract.get("manuscript", {}).get("source", {}), "manuscript")
        == evaluator.MANUSCRIPT.resolve(strict=True),
        "contract manuscript binding mismatch",
    )
    evaluator.require(contract.get("result_output") == str(evaluator.RESULT.resolve()), "contract result mismatch")
    evaluator.require(
        contract.get("result_publication_output") == str(evaluator.RESULT_PUBLICATION.resolve()),
        "contract result-publication mismatch",
    )
    evaluator.require(contract.get("zero_openmm_preflight_output") == str(preflight_path), "contract preflight mismatch")
    evaluator.require(contract.get("execution_authorization_output") == str(output), "contract authorization mismatch")
    evaluator.require(contract.get("formula", {}).get("id") == evaluator.FORMULA_ID, "contract formula ID mismatch")
    evaluator.require(contract.get("formula", {}).get("text") == evaluator.FORMULA_TEXT, "contract formula text mismatch")
    evaluator.require(
        contract.get("sampling")
        == {
            "images": 32,
            "segments": 31,
            "coordinate_metric": "unweighted Cartesian Euclidean",
            "densification": "none",
            "interpolation": "none",
            "smoothing": "none",
        },
        "contract sampling mismatch",
    )
    call_policy = contract.get("physical_call_policy", {})
    evaluator.require(int(call_policy.get("IVM_internal_fresh_joint_EF_attempt_maximum", -1)) == evaluator.N_INTERNAL, "contract internal-call bound mismatch")
    evaluator.require(int(call_policy.get("IVM_fixed_endpoint_fresh_joint_EF_attempt_maximum", -1)) == 0, "contract endpoint-call bound mismatch")
    evaluator.require(int(call_policy.get("ODE_NEB_fresh_joint_EF_attempt_maximum", -1)) == 0, "contract ODE-call bound mismatch")
    evaluator.require(int(call_policy.get("optimization_N_F_contribution", -1)) == 0, "contract optimization accounting mismatch")
    evaluator.require(
        float(call_policy.get("maximum_joint_vs_energy_only_abs_drift_eV", -1.0))
        == evaluator.MAX_ENERGY_RECHECK_DRIFT_EV,
        "contract energy-recheck tolerance mismatch",
    )

    evaluator.require(preflight.get("schema") == evaluator.PREAUTH_SCHEMA, "preflight schema mismatch")
    evaluator.require(preflight.get("status") == evaluator.PREAUTH_STATUS, "preflight status mismatch")
    evaluator.require(preflight.get("contract") == evaluator.file_binding(contract_path), "preflight contract binding mismatch")
    evaluator.require(preflight.get("evaluator") == evaluator.file_binding(EVALUATOR), "preflight evaluator binding mismatch")
    evaluator.require(preflight.get("result_output") == str(evaluator.RESULT.resolve()), "preflight result mismatch")
    evaluator.require(int(preflight.get("physical_calls_started", -1)) == 0, "preflight physical-call count is not zero")
    evaluator.require(int(preflight.get("openmm_contexts_constructed", -1)) == 0, "preflight OpenMM-context count is not zero")
    evaluator.require(int(preflight.get("maximum_fresh_internal_joint_ef_attempts", -1)) == evaluator.N_INTERNAL, "preflight internal-call bound mismatch")
    evaluator.require(int(preflight.get("maximum_fresh_endpoint_joint_ef_attempts", -1)) == 0, "preflight endpoint-call bound mismatch")
    evaluator.require(int(preflight.get("optimization_N_F_contribution", -1)) == 0, "preflight optimization accounting mismatch")
    source_validation = preflight.get("source_validation", {})
    evaluator.require(source_validation.get("IVM_path_sha256") == evaluator.EXPECTED_IR_PATH_SHA256, "preflight IR source mismatch")
    evaluator.require(source_validation.get("ODE_NEB_path_sha256") == evaluator.EXPECTED_ODE_PATH_SHA256, "preflight ODE source mismatch")
    evaluator.require(source_validation.get("ODE_NEB_force_sha256") == evaluator.EXPECTED_ODE_FORCE_SHA256, "preflight ODE force mismatch")

    result = evaluator.RESULT.resolve()
    for artifact in (
        result,
        evaluator.RESULT_PUBLICATION.resolve(),
        result.with_suffix(".joint_ef.wal.jsonl"),
        result.with_suffix(".audit.json"),
        evaluator.FIGURE.resolve(),
        evaluator.PROVENANCE.resolve(),
        evaluator.FIGURE_PUBLICATION.resolve(),
    ):
        evaluator.require(not artifact.exists(), f"artifact predates execution authorization: {artifact}")

    locks = [Path(value).expanduser().resolve() for value in contract.get("host_local_lock_paths", [])]
    evaluator.require(set(locks) == evaluator.CANONICAL_LOCKS and len(set(locks)) == 2, "contract lock set mismatch")
    expected_locks = {str(path) for path in evaluator.CANONICAL_LOCKS}
    probe = preflight.get("canonical_two_lock_nonblocking_probe", {})
    evaluator.require(probe.get("status") == "PASS" and set(probe.get("locks", [])) == expected_locks, "preflight lock probe mismatch")
    evaluator.require(preflight.get("thread_policy") == {name: "1" for name in evaluator.THREAD_VARS}, "preflight thread policy mismatch")
    evaluator.load_sources()
    with evaluator.acquire_two_locks(locks) as held:
        second_probe = list(held)

    payload = {
        "schema": evaluator.AUTH_SCHEMA,
        "status": evaluator.AUTH_STATUS,
        "enabled": True,
        "authorized_utc": evaluator.utc_now(),
        "host_role": "local",
        "jobs": 1,
        "contract": evaluator.file_binding(contract_path),
        "evaluator": evaluator.file_binding(EVALUATOR),
        "authorizer": evaluator.file_binding(Path(__file__).resolve()),
        "zero_openmm_preflight": evaluator.file_binding(preflight_path),
        "result_output": str(result),
        "result_publication_output": str(evaluator.RESULT_PUBLICATION.resolve()),
        "joint_EF_WAL_output": str(result.with_suffix(".joint_ef.wal.jsonl")),
        "audit_output": str(result.with_suffix(".audit.json")),
        "maximum_fresh_internal_joint_ef_attempts": evaluator.N_INTERNAL,
        "maximum_fresh_endpoint_joint_ef_attempts": 0,
        "maximum_fresh_ODE_NEB_joint_ef_attempts": 0,
        "optimization_N_F_contribution": 0,
        "thread_policy": {name: "1" for name in evaluator.THREAD_VARS},
        "canonical_two_lock_second_probe": {"status": "PASS", "locks": second_probe},
        "physical_calls_started_before_authorization": 0,
        "openmm_contexts_constructed_before_authorization": 0,
        "successful_prefix_resume_policy": "reuse only complete coordinate-hash-bound WAL successes",
        "failed_or_open_attempt_policy": "fail closed; require superseding authorization and new output",
    }
    atomic_json_no_overwrite(output, payload)
    print(
        json.dumps(
            {
                "status": evaluator.AUTH_STATUS,
                "authorization": str(output),
                "sha256": evaluator.sha256_file(output),
                "physical_calls_started_before_authorization": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
