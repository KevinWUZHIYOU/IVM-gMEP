#!/usr/bin/env python3
"""Freeze the BB discrete-32 geometric-action contract and zero-call preflight.

This script only validates immutable inputs, output names, thread/lock/disk gates,
and writes JSON metadata.  It never constructs an OpenMM Context.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = ROOT / "scripts/evaluate_bb_discrete32_geometric_action.py"
AUTHORIZER = ROOT / "scripts/authorize_bb_discrete32_geometric_action.py"
PLOTTER = ROOT / "scripts/plot_bb_discrete32_geometric_action.py"
TEST_FILE = ROOT / "tests/test_bb_discrete32_geometric_action.py"
BB_CORE = ROOT / "src/bb_core.py"
TOPOLOGY = ROOT / "inputs/complex_fixed.pdb"
MANUSCRIPT = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/main.tex")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("bb_discrete32_action_freezer", EVALUATOR)
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
    parser.add_argument("--output", type=Path, default=evaluator.CONTRACT)
    parser.add_argument("--preflight-output", type=Path, default=evaluator.PREAUTH)
    parser.add_argument("--authorization-output", type=Path, default=evaluator.AUTHORIZATION)
    parser.add_argument("--result-output", type=Path, default=evaluator.RESULT)
    parser.add_argument("--figure-output", type=Path, default=evaluator.FIGURE)
    parser.add_argument("--provenance-output", type=Path, default=evaluator.PROVENANCE)
    parser.add_argument("--host-role", choices=("local",), default="local")
    parser.add_argument("--global-lock-path", type=Path, required=True)
    parser.add_argument("--peer-global-lock-path", type=Path, required=True)
    args = parser.parse_args(argv)

    evaluator.check_thread_caps()
    evaluator.check_imported_bb_core()
    destinations = {
        "contract": args.output.expanduser().resolve(),
        "preflight": args.preflight_output.expanduser().resolve(),
        "authorization": args.authorization_output.expanduser().resolve(),
        "result": args.result_output.expanduser().resolve(),
        "figure": args.figure_output.expanduser().resolve(),
        "provenance": args.provenance_output.expanduser().resolve(),
        "publication": evaluator.FIGURE_PUBLICATION.resolve(),
    }
    expected_destinations = {
        "contract": evaluator.CONTRACT.resolve(),
        "preflight": evaluator.PREAUTH.resolve(),
        "authorization": evaluator.AUTHORIZATION.resolve(),
        "result": evaluator.RESULT.resolve(),
        "figure": evaluator.FIGURE.resolve(),
        "provenance": evaluator.PROVENANCE.resolve(),
        "publication": evaluator.FIGURE_PUBLICATION.resolve(),
    }
    evaluator.require(destinations == expected_destinations, "noncanonical discrete32-action destination")
    locks = [args.global_lock_path.expanduser().resolve(), args.peer_global_lock_path.expanduser().resolve()]
    evaluator.require(len(set(locks)) == 2 and set(locks) == evaluator.CANONICAL_LOCKS, "canonical two-lock set required")

    for required in (
        EVALUATOR,
        AUTHORIZER,
        PLOTTER,
        TEST_FILE,
        BB_CORE,
        TOPOLOGY,
        MANUSCRIPT,
        evaluator.IR_ENERGY_RESULT,
        evaluator.IR_ENERGY_AUDIT,
        evaluator.ENDPOINT_CACHE,
        evaluator.ODE_SOURCE,
    ):
        required.resolve(strict=True)
    sources = evaluator.load_sources()
    evaluator.require(sources["ir"]["path"].shape == (evaluator.N_IMAGES, evaluator.DIMENSION), "IR source shape drift")
    evaluator.require(sources["ode"]["path"].shape == (evaluator.N_IMAGES, evaluator.DIMENSION), "ODE source shape drift")

    physical_artifacts = (
        destinations["result"],
        evaluator.RESULT_PUBLICATION.resolve(),
        destinations["result"].with_suffix(".joint_ef.wal.jsonl"),
        destinations["result"].with_suffix(".audit.json"),
        destinations["figure"],
        destinations["provenance"],
        destinations["publication"],
    )
    for artifact in physical_artifacts:
        evaluator.require(not artifact.exists(), f"prospective artifact already exists: {artifact}")
    evaluator.require(not destinations["authorization"].exists(), "authorization predates contract freeze")

    destinations["result"].parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destinations["result"].parent).free
    required_free = evaluator.MINIMUM_FREE_BYTES + evaluator.ESTIMATED_FINAL_BYTES
    evaluator.require(free >= required_free, f"insufficient free disk: {free} < {required_free}")
    with evaluator.acquire_two_locks(locks) as held:
        lock_probe = list(held)

    contract = {
        "schema": evaluator.CONTRACT_SCHEMA,
        "status": evaluator.CONTRACT_STATUS,
        "label": "bb_ivm_ode_neb_discrete32_geometric_action_20260902_r1",
        "host_role": args.host_role,
        "jobs": 1,
        "scientific_quantity": "fixed-endpoint path-dependent part of the geometric action",
        "manuscript": {
            "source": evaluator.file_binding(MANUSCRIPT),
            "definition_evidence": "main.tex lines 197--206; \\hat S=\\int_\\varphi |\\nabla V| d\\ell",
            "full_oriented_action_evidence": "main.tex lines 823--836",
            "bb_metric_evidence": "main.tex lines 1959--1971; G=I_9396",
            "excluded_distinct_metric": "I_arc in main.tex lines 1068--1070 is not used",
        },
        "formula": {
            "id": evaluator.FORMULA_ID,
            "text": evaluator.FORMULA_TEXT,
            "quadrature": "segmentwise trapezoidal rule on force norms",
            "force_identity": "OpenMM F=-grad(V), hence ||F||_2=||grad(V)||_2",
            "x_unit": "angstrom",
            "force_unit": "eV/angstrom",
            "action_unit": "eV (also stored after fixed conversion to kcal/mol)",
            "mass_weighting": False,
            "energy_reference_term": False,
        },
        "sampling": {
            "images": evaluator.N_IMAGES,
            "segments": evaluator.N_IMAGES - 1,
            "coordinate_metric": "unweighted Cartesian Euclidean",
            "densification": "none",
            "interpolation": "none",
            "smoothing": "none",
        },
        "source_paths": {
            "IVM": {
                "optimization_N_F": evaluator.IR_OPTIMIZATION_NF,
                "call_per_image": evaluator.IR_CALL_PER_IMAGE,
                "path_sha256": evaluator.EXPECTED_IR_PATH_SHA256,
            },
            "ODE-NEB": {
                "optimization_N_F": evaluator.ODE_OPTIMIZATION_NF,
                "path_sha256": evaluator.EXPECTED_ODE_PATH_SHA256,
            },
        },
        "source_artifacts": {
            "IVM_exact_path_and_energy": evaluator.file_binding(evaluator.IR_ENERGY_RESULT),
            "IVM_exact_path_and_energy_audit": evaluator.file_binding(evaluator.IR_ENERGY_AUDIT),
            "fixed_endpoint_joint_EF_cache": evaluator.file_binding(evaluator.ENDPOINT_CACHE),
            "ODE_NEB_internal_joint_EF": evaluator.file_binding(evaluator.ODE_SOURCE),
        },
        "code_and_topology": {
            "evaluator": evaluator.file_binding(EVALUATOR),
            "contract_freezer": evaluator.file_binding(Path(__file__).resolve()),
            "execution_authorizer": evaluator.file_binding(AUTHORIZER),
            "plotter": evaluator.file_binding(PLOTTER),
            "tests": evaluator.file_binding(TEST_FILE),
            "bb_core": evaluator.file_binding(BB_CORE),
            "topology": evaluator.file_binding(TOPOLOGY),
        },
        "evaluator": evaluator.file_binding(EVALUATOR),
        "execution_authorizer": evaluator.file_binding(AUTHORIZER),
        "physical_call_policy": {
            "IVM_internal_fresh_joint_EF_attempt_maximum": evaluator.N_INTERNAL,
            "IVM_fixed_endpoint_fresh_joint_EF_attempt_maximum": 0,
            "ODE_NEB_fresh_joint_EF_attempt_maximum": 0,
            "joint_getState_getEnergy_and_getForces": True,
            "diagnostic_calls_only": True,
            "optimization_N_F_contribution": 0,
            "maximum_joint_vs_energy_only_abs_drift_eV": evaluator.MAX_ENERGY_RECHECK_DRIFT_EV,
        },
        "storage_and_durability_policy": {
            "all_32_coordinates_energies_full_forces_for_both_methods": True,
            "append_only_fsync_WAL_with_base64_little_endian_full_force": True,
            "coordinate_energy_force_array_hashes": True,
            "atomic_no_overwrite_result_audit_figure_provenance": True,
            "failed_or_open_attempt_requires_superseding_authorization_and_new_output": True,
            "successful_prefix_resume_requires_exact_coordinate_hash": True,
            "minimum_free_bytes": evaluator.MINIMUM_FREE_BYTES,
            "estimated_final_bytes": evaluator.ESTIMATED_FINAL_BYTES,
        },
        "thread_policy": {name: "1" for name in evaluator.THREAD_VARS},
        "host_local_lock_paths": [str(path) for path in sorted(locks, key=str)],
        "result_output": str(destinations["result"]),
        "result_publication_output": str(evaluator.RESULT_PUBLICATION.resolve()),
        "joint_EF_WAL_output": str(destinations["result"].with_suffix(".joint_ef.wal.jsonl")),
        "audit_output": str(destinations["result"].with_suffix(".audit.json")),
        "figure_output": str(destinations["figure"]),
        "figure_provenance_output": str(destinations["provenance"]),
        "figure_publication_output": str(destinations["publication"]),
        "zero_openmm_preflight_output": str(destinations["preflight"]),
        "execution_authorization_output": str(destinations["authorization"]),
        "physical_execution_requires_separate_authorization": True,
    }
    atomic_json_no_overwrite(destinations["contract"], contract)

    preauth = {
        "schema": evaluator.PREAUTH_SCHEMA,
        "status": evaluator.PREAUTH_STATUS,
        "completed_utc": evaluator.utc_now(),
        "contract": evaluator.file_binding(destinations["contract"]),
        "evaluator": evaluator.file_binding(EVALUATOR),
        "source_validation": {
            "IVM_path_shape": list(sources["ir"]["path"].shape),
            "IVM_path_sha256": evaluator.sha256_array(sources["ir"]["path"]),
            "IVM_energy_sha256": evaluator.sha256_array(sources["ir"]["energy"]),
            "ODE_NEB_path_shape": list(sources["ode"]["path"].shape),
            "ODE_NEB_path_sha256": evaluator.sha256_array(sources["ode"]["path"]),
            "ODE_NEB_energy_sha256": evaluator.sha256_array(sources["ode"]["energy"]),
            "ODE_NEB_force_sha256": evaluator.sha256_array(sources["ode"]["force"]),
            "endpoint_coordinate_sha256": evaluator.sha256_array(sources["endpoint"]["coordinates"]),
            "endpoint_energy_sha256": evaluator.sha256_array(sources["endpoint"]["energy"]),
            "endpoint_force_sha256": evaluator.sha256_array(sources["endpoint"]["force"]),
        },
        "result_output": str(destinations["result"]),
        "figure_output": str(destinations["figure"]),
        "figure_provenance_output": str(destinations["provenance"]),
        "figure_publication_output": str(destinations["publication"]),
        "canonical_two_lock_nonblocking_probe": {"status": "PASS", "locks": lock_probe},
        "thread_policy": {name: "1" for name in evaluator.THREAD_VARS},
        "observed_free_bytes": free,
        "required_free_bytes": required_free,
        "maximum_fresh_internal_joint_ef_attempts": evaluator.N_INTERNAL,
        "maximum_fresh_endpoint_joint_ef_attempts": 0,
        "optimization_N_F_contribution": 0,
        "physical_calls_started": 0,
        "openmm_contexts_constructed": 0,
    }
    atomic_json_no_overwrite(destinations["preflight"], preauth)
    print(
        json.dumps(
            {
                "status": "FROZEN_ZERO_OPENMM",
                "contract": evaluator.file_binding(destinations["contract"]),
                "preflight": evaluator.file_binding(destinations["preflight"]),
                "physical_calls_started": 0,
                "openmm_contexts_constructed": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
