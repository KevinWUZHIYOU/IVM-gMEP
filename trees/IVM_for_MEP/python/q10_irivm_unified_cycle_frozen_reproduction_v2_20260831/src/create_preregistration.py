"""Create and seal the QM10 frozen-reproduction preregistration.

This command is deliberately one-shot.  It refuses to run unless both formal
result directories and both receipt directories are empty, the preformal
evidence is complete, and the numerical runtime has the exact registered
identity before and after importing the optimizer.  The registration and its
one-line SHA-256 seal are installed atomically and are never overwritten.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from .formal_io import (
    FormalIOError,
    atomic_write_bytes,
    atomic_write_json,
    derive_campaign_paths,
    load_canonical_json,
    sha256_file,
)
from . import run_frozen_reproduction as runner


CAMPAIGN_ID = "q10-irivm-unified-cycle-frozen-reproduction-v2-20260831"
V1_CAMPAIGN_ID = "q10-irivm-unified-cycle-frozen-reproduction-20260831"
V1_REGISTRATION_SHA256 = (
    "eeee1a336a30899143e4118404b5bc8f99287b009330f0c44c83335c18ab0b12"
)
V1_RESULT_IDENTITIES = {
    "n10": {
        "result_sha256": (
            "e2035c0cab5f69c1dff39178a84b83558bfd872cafe8861f581aabcfcecc1134"
        ),
        "run_id": "0fa57295d005c597",
        "terminal_nof": 782,
    },
    "n20": {
        "result_sha256": (
            "7f5470841ba95f47acda471171f483d24e617a6fe62e83a9d03de4501d764c5d"
        ),
        "run_id": "8cb8b3a582ac6f8e",
        "terminal_nof": 2981,
    },
}
V1_CAMPAIGN_RELATIVE = (
    "python/q10_irivm_unified_cycle_frozen_reproduction_20260831"
)
Q_ARRAY_SHA256 = (
    "372962a47504e6df28a20d8b022482a6835680765434b4385a6dba0e826974bc"
)

REGISTERED_INPUTS = {
    "metric_source": "python/unified_ir_ivm_20260827/src/q10_article_metrics.py",
    "article_n10": (
        "paper/ActiveExperiments/muller_q10_10d/image12/data/frozen_runs/"
        "tuned_ivm_alpha2e3_selected.npz"
    ),
    "article_n20": (
        "paper/ActiveExperiments/muller_q10_10d/data/frozen_runs/"
        "tuned_ivm_alpha1e3_selected.npz"
    ),
    "q_n10": (
        "paper/ActiveExperiments/muller_q10_10d/image12/data/inputs/Q_data.npz"
    ),
    "q_n20": "paper/ActiveExperiments/muller_q10_10d/data/inputs/Q_data.npz",
}

RUNTIME_DEPENDENCIES = (
    "python/q10_irivm_low_r_coherence_secant_v2_20260830/src/"
    "q10_irivm_atomic_tangent_secant_core.py",
    "python/q10_irivm_low_r_coherence_secant_v2_20260830/src/"
    "q10_solver_backend.py",
    "paper/ActiveExperiments/muller_q10_10d/image12/code/experiments/"
    "q10_ivm_core.py",
    "paper/ActiveExperiments/muller_q10_10d/image12/code/experiments/"
    "run_q10_baselines.py",
    "paper/ActiveExperiments/muller_q10_10d/image12/code/experiments/"
    "q10_potential.py",
    "paper/ActiveExperiments/muller_q10_10d/code/ivm/q10_ivm_core.py",
    "paper/ActiveExperiments/muller_q10_10d/code/baselines/"
    "run_q10_baselines.py",
    "paper/ActiveExperiments/muller_q10_10d/code/common/q10_potential.py",
)


class PreregistrationError(RuntimeError):
    pass


def _campaign_source_paths(campaign_root: Path) -> Iterable[Path]:
    fixed = (
        campaign_root / "PROTOCOL.md",
        campaign_root / "RESULT_NPZ_SCHEMA.md",
        campaign_root / "registrations/README.md",
    )
    yielded = set()
    for path in fixed:
        if not path.is_file():
            raise PreregistrationError("missing campaign document: " + str(path))
        yielded.add(path)
        yield path
    for pattern in ("src/*.py", "tests/*.py", "evidence/*"):
        for path in sorted(campaign_root.glob(pattern)):
            if path.is_file() and path not in yielded:
                yielded.add(path)
                yield path


def _load_pass(
    path: Path,
    *,
    accepted_statuses: tuple[str, ...],
    require_canonical: bool = True,
) -> Dict[str, Any]:
    if require_canonical:
        payload = load_canonical_json(path)
    else:
        # The two preserved exploratory reports predate this formal campaign
        # and were intentionally copied byte-for-byte.  Their exact bytes are
        # hash-registered below; reformatting them merely to satisfy a newer
        # serialization convention would weaken, not strengthen, provenance.
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PreregistrationError("invalid preserved JSON: " + str(path)) from exc
    if not isinstance(payload, dict):
        raise PreregistrationError("evidence root is not an object: " + str(path))
    status = str(payload.get("status", payload.get("audit_status", ""))).upper()
    if status not in accepted_statuses:
        raise PreregistrationError("evidence is not PASS: " + str(path))
    return payload


def _verify_preformal_evidence(campaign_root: Path) -> Dict[str, Any]:
    portability = _load_pass(
        campaign_root / "evidence/preformal_equivalence.json",
        accepted_statuses=("PASS",),
        require_canonical=False,
    )
    refactor = _load_pass(
        campaign_root / "evidence/exploratory_refactor_audit.json",
        accepted_statuses=("PASS",),
        require_canonical=False,
    )
    static = _load_pass(
        campaign_root / "evidence/static_contract_report.json",
        accepted_statuses=("PASS",),
    )
    instrumentation = _load_pass(
        campaign_root / "evidence/preformal_instrumentation_equivalence.json",
        accepted_statuses=("PASS",),
    )
    v1_diagnosis = _load_pass(
        campaign_root / "evidence/v1_auditor_failure_diagnosis.json",
        accepted_statuses=("SUPERSEDED_AUDITOR_FAILURE",),
    )
    if static.get("passed") is not True or static.get(
        "all_runtime_calls_intercepted"
    ) is not True:
        raise PreregistrationError("static reachable-call/interception gate failed")
    forbidden = static.get("forbidden_reachable_counts")
    if not isinstance(forbidden, dict) or not forbidden or any(
        int(value) != 0 for value in forbidden.values()
    ):
        raise PreregistrationError("static report contains a reachable forbidden path")
    if not (
        instrumentation.get("passed") is True
        or instrumentation.get("all_passed") is True
    ):
        raise PreregistrationError("instrumentation equivalence gate failed")

    expected = {
        10: (782, "ccd4f916cd9c3017540065eae26ff329a2f3284449754d1a447ccef710107eee"),
        20: (2981, "22d4969de10ec5d3918e92b4b4c548b739244e1a28f1cf44fac2659f27753c60"),
    }
    portability_runs = {
        int(item["n_internal"]): item for item in portability.get("runs", [])
    }
    if set(portability_runs) != set(expected):
        raise PreregistrationError("portability evidence is not exactly N=10/N=20")
    for resolution, (terminal, trajectory_hash) in expected.items():
        item = portability_runs[resolution]
        if (
            item.get("all_required_fields_bitwise_or_exact_equal") is not True
            or int(item.get("terminal_nf", -1)) != terminal
            or str(item.get("numeric_trajectory_sha256", "")) != trajectory_hash
        ):
            raise PreregistrationError(
                "frozen exploratory trajectory identity failed for N=%d" % resolution
            )
    optimizer_hash = sha256_file(campaign_root / "src/q10_unified_cycle_optimizer.py")
    if portability.get("source", {}).get("formal_candidate_sha256") != optimizer_hash:
        raise PreregistrationError("portable optimizer differs from equivalence evidence")
    if refactor.get("reparameterization_static_audit", {}).get(
        "geometric_resampling_or_alternative_reparameterization_present"
    ) is not False:
        raise PreregistrationError("refactor audit did not exclude alternate reparam")
    if v1_diagnosis.get("old_campaign_id") != V1_CAMPAIGN_ID:
        raise PreregistrationError("v1 auditor diagnosis has the wrong old campaign ID")
    if v1_diagnosis.get("old_registration_sha256") != V1_REGISTRATION_SHA256:
        raise PreregistrationError(
            "v1 auditor diagnosis has the wrong old registration identity"
        )
    if v1_diagnosis.get("old_results") != V1_RESULT_IDENTITIES:
        raise PreregistrationError(
            "v1 auditor diagnosis has the wrong old formal-result identities"
        )
    if v1_diagnosis.get("algorithm_changed_in_v2") is not False:
        raise PreregistrationError(
            "v1 auditor diagnosis must record that the v2 algorithm is unchanged"
        )
    if v1_diagnosis.get("v1_artifacts_preserved") is not True:
        raise PreregistrationError("v1 auditor diagnosis does not preserve v1 artifacts")
    source_reconstruction = v1_diagnosis.get("exhaustive_source_reconstruction")
    if not isinstance(source_reconstruction, dict) or set(source_reconstruction) != {
        "n10", "n20"
    }:
        raise PreregistrationError(
            "v1 auditor diagnosis must cover exactly N=10 and N=20"
        )
    expected_source_kind_counts = {
        "n10": {
            "complete_direct_spring_check_band": 8,
            "public_state": 126,
        },
        "n20": {
            "complete_direct_spring_check_band": 2,
            "public_state": 184,
            "recovery_candidate_state": 2,
        },
    }
    for resolution, expected_kind_counts in expected_source_kind_counts.items():
        item = source_reconstruction[resolution]
        if not isinstance(item, dict):
            raise PreregistrationError(
                "v1 source reconstruction is malformed for " + resolution
            )
        if item.get("all_internal_source_coordinates_found_in_prior_paid_trace") is not True:
            raise PreregistrationError(
                "v1 source reconstruction lacks prior-paid-source closure for "
                + resolution
            )
        if float(item.get("fixed_energy_maximum_absolute_error", float("inf"))) != 0.0:
            raise PreregistrationError(
                "v1 source reconstruction does not exactly reproduce energies for "
                + resolution
            )
        event_count = int(item.get("event_count", -1))
        kind_counts = item.get("source_kind_counts")
        if (
            not isinstance(kind_counts, dict)
            or kind_counts != expected_kind_counts
            or event_count != sum(expected_kind_counts.values())
        ):
            raise PreregistrationError(
                "v1 source reconstruction event closure failed for " + resolution
            )
    return {
        "portability_sha256": sha256_file(
            campaign_root / "evidence/preformal_equivalence.json"
        ),
        "refactor_audit_sha256": sha256_file(
            campaign_root / "evidence/exploratory_refactor_audit.json"
        ),
        "static_contract_sha256": sha256_file(
            campaign_root / "evidence/static_contract_report.json"
        ),
        "instrumentation_equivalence_sha256": sha256_file(
            campaign_root / "evidence/preformal_instrumentation_equivalence.json"
        ),
        "v1_auditor_failure_diagnosis_sha256": sha256_file(
            campaign_root / "evidence/v1_auditor_failure_diagnosis.json"
        ),
    }


def _verify_superseded_v1_artifacts(
    campaign_root: Path, repository_root: Path
) -> Dict[str, Any]:
    registration_relative = (
        V1_CAMPAIGN_RELATIVE + "/registrations/CAMPAIGN_PREREGISTRATION.json"
    )
    seal_relative = (
        V1_CAMPAIGN_RELATIVE + "/registrations/CAMPAIGN_PREREGISTRATION.sha256"
    )
    result_relatives = {
        "n10": V1_CAMPAIGN_RELATIVE + "/runs/n10/0fa57295d005c597.npz",
        "n20": V1_CAMPAIGN_RELATIVE + "/runs/n20/8cb8b3a582ac6f8e.npz",
    }
    required = [registration_relative, seal_relative] + list(
        result_relatives.values()
    )
    paths: Dict[str, Path] = {}
    for relative in required:
        path = repository_root / relative
        if not path.is_file():
            raise PreregistrationError(
                "superseded v1 formal artifact is missing: repo:" + relative
            )
        paths[relative] = path

    registration_path = paths[registration_relative]
    if sha256_file(registration_path) != V1_REGISTRATION_SHA256:
        raise PreregistrationError("superseded v1 registration bytes drifted")
    old_registration = load_canonical_json(registration_path)
    if old_registration.get("campaign_id") != V1_CAMPAIGN_ID:
        raise PreregistrationError("superseded v1 registration campaign ID drifted")
    expected_old_outputs = {
        "n10_result": "runs/n10/0fa57295d005c597.npz",
        "n20_result": "runs/n20/8cb8b3a582ac6f8e.npz",
        "n10_receipt": "reports/receipts/n10/0fa57295d005c597.receipt.json",
        "n20_receipt": "reports/receipts/n20/8cb8b3a582ac6f8e.receipt.json",
    }
    if old_registration.get("expected_formal_outputs") != expected_old_outputs:
        raise PreregistrationError("superseded v1 expected formal paths drifted")
    expected_old_members = [
        {
            "n_internal": 10,
            "alpha": 0.003,
            "alpha_label": "3e-3",
            "r": 0.1,
            "m_max": 7,
            "run_id": "0fa57295d005c597",
        },
        {
            "n_internal": 20,
            "alpha": 0.001,
            "alpha_label": "1e-3",
            "r": 0.3,
            "m_max": 8,
            "run_id": "8cb8b3a582ac6f8e",
        },
    ]
    if old_registration.get("registered_members") != expected_old_members:
        raise PreregistrationError("superseded v1 registered controls drifted")

    old_identity = old_registration.get("result_identity")
    if not isinstance(old_identity, dict):
        raise PreregistrationError("superseded v1 result identity is missing")
    current_execution_files = {
        "algorithm_source_sha256": campaign_root
        / "src/q10_unified_cycle_optimizer.py",
        "optimizer_sha256": campaign_root / "src/q10_unified_cycle_optimizer.py",
        "adapter_sha256": campaign_root / "src/registered_adapter.py",
        "runtime_sha256": repository_root / RUNTIME_DEPENDENCIES[0],
        "backend_sha256": repository_root / RUNTIME_DEPENDENCIES[1],
    }
    unchanged_execution_identities: Dict[str, str] = {}
    for field, path in current_execution_files.items():
        if not path.is_file():
            raise PreregistrationError(
                "v2 unchanged-execution identity file is missing: " + str(path)
            )
        current_hash = sha256_file(path)
        if current_hash != old_identity.get(field):
            raise PreregistrationError(
                "v2 changed an execution identity relative to v1: " + field
            )
        unchanged_execution_identities[field] = current_hash

    seal_path = paths[seal_relative]
    if seal_path.read_bytes() != (V1_REGISTRATION_SHA256 + "\n").encode("ascii"):
        raise PreregistrationError("superseded v1 registration seal does not agree")

    result_provenance: Dict[str, Any] = {}
    for resolution, relative in result_relatives.items():
        expected = V1_RESULT_IDENTITIES[resolution]
        actual = sha256_file(paths[relative])
        if actual != expected["result_sha256"]:
            raise PreregistrationError(
                "superseded v1 formal result bytes drifted for " + resolution
            )
        result_provenance[resolution] = {
            "path": "repo:" + relative,
            **expected,
        }

    return {
        "campaign_id": V1_CAMPAIGN_ID,
        "campaign_root": "repo:" + V1_CAMPAIGN_RELATIVE,
        "registration": {
            "path": "repo:" + registration_relative,
            "sha256": V1_REGISTRATION_SHA256,
        },
        "registration_seal": {
            "path": "repo:" + seal_relative,
            "sha256": sha256_file(seal_path),
            "sealed_registration_sha256": V1_REGISTRATION_SHA256,
        },
        "results": result_provenance,
        "artifacts_exist_and_match": True,
        "algorithm_changed_in_v2": False,
        "unchanged_execution_identities": unchanged_execution_identities,
    }


def _require_empty_formal_boundary(campaign_root: Path) -> None:
    patterns = (
        "runs/n10/*.npz",
        "runs/n20/*.npz",
        "reports/receipts/n10/*.json",
        "reports/receipts/n20/*.json",
    )
    found = [str(path) for pattern in patterns for path in campaign_root.glob(pattern)]
    if found:
        raise PreregistrationError("formal boundary is not empty: %r" % found)


def _runtime_contract() -> Dict[str, Any]:
    expected_environment = {
        "OPENBLAS_NUM_THREADS": "8",
        "OMP_NUM_THREADS": None,
        "MKL_NUM_THREADS": None,
        "VECLIB_MAXIMUM_THREADS": None,
        "NUMEXPR_NUM_THREADS": None,
    }
    # This applies the same environment/pool checks as the later runner, before
    # any optimizer import can load its additional numerical libraries.
    provisional = {
        "jobs": 1,
        "thread_environment": expected_environment,
    }
    preflight = runner._verify_environment(
        {
            **provisional,
            "preflight_runtime_identity": runner.runtime_identity(),
        }
    )
    package = __package__
    optimizer = importlib.import_module(package + ".q10_unified_cycle_optimizer")
    importlib.import_module(package + ".registered_adapter")
    importlib.import_module(package + ".result_schema")
    # The runner records its post-optimizer identity after the member has
    # loaded the article QM10 backend.  Reproduce that import boundary without
    # executing an optimization or evaluating a force.  Both resolution
    # loaders must settle on the same numerical-library identity.
    optimizer.alg._load_backend(20)
    post = runner.runtime_identity()
    runner._verify_pools_are_eight(post, "post-optimizer preregistration")
    optimizer.alg._load_backend(10)
    post_n10 = runner.runtime_identity()
    runner._verify_pools_are_eight(post_n10, "post-N10-backend preregistration")
    if post_n10 != post:
        raise PreregistrationError(
            "N=10 and N=20 backend imports have different runtime identities"
        )
    return {
        **provisional,
        "pythonhashseed": "0",
        "preflight_runtime_identity": preflight,
        "post_optimizer_runtime_identity": post,
        # The result archive stores the post-import identity.  This explicit
        # alias lets the independent auditor compare it without importing the
        # optimizer.
        "runtime_identity": post,
    }


def build_registration() -> Dict[str, Any]:
    paths = derive_campaign_paths(__file__)
    campaign_root = paths.campaign_root
    repository_root = paths.repository_root
    _require_empty_formal_boundary(campaign_root)
    evidence = _verify_preformal_evidence(campaign_root)
    superseded_provenance = _verify_superseded_v1_artifacts(
        campaign_root, repository_root
    )
    superseded_provenance["diagnosis_evidence_sha256"] = evidence[
        "v1_auditor_failure_diagnosis_sha256"
    ]
    runtime_contract = _runtime_contract()

    source_manifest = {
        path.relative_to(campaign_root).as_posix(): sha256_file(path)
        for path in _campaign_source_paths(campaign_root)
    }
    dependency_relatives = tuple(REGISTERED_INPUTS.values()) + RUNTIME_DEPENDENCIES
    dependency_manifest = {}
    for relative in dependency_relatives:
        path = repository_root / relative
        if not path.is_file():
            raise PreregistrationError("missing repository dependency: " + relative)
        dependency_manifest["repo:" + relative] = sha256_file(path)

    identity_paths = {
        "algorithm_source_sha256": "campaign:src/q10_unified_cycle_optimizer.py",
        "optimizer_sha256": "campaign:src/q10_unified_cycle_optimizer.py",
        "adapter_sha256": "campaign:src/registered_adapter.py",
        "runtime_sha256": (
            "repo:python/q10_irivm_low_r_coherence_secant_v2_20260830/src/"
            "q10_irivm_atomic_tangent_secant_core.py"
        ),
        "backend_sha256": (
            "repo:python/q10_irivm_low_r_coherence_secant_v2_20260830/src/"
            "q10_solver_backend.py"
        ),
        "runner_sha256": "campaign:src/run_frozen_reproduction.py",
        "exploratory_source_sha256": "campaign:evidence/exploratory_candidate.py",
        "preformal_equivalence_sha256": (
            "campaign:evidence/preformal_equivalence.json"
        ),
        "instrumentation_equivalence_sha256": (
            "campaign:evidence/preformal_instrumentation_equivalence.json"
        ),
    }
    result_identity = {"q_sha256": Q_ARRAY_SHA256}
    for field, tagged in identity_paths.items():
        if tagged.startswith("campaign:"):
            result_identity[field] = source_manifest[tagged[len("campaign:"):]]
        else:
            result_identity[field] = dependency_manifest[tagged]

    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "methodological_status": (
            "post-selection frozen reproduction on the same QM10 opponent, inputs, "
            "checkpoints, and metrics; not a holdout, transfer, or generalization test"
        ),
        "supersedes_campaign_id": V1_CAMPAIGN_ID,
        "supersession_reason": (
            "v1 formal runs were preserved but its auditor compared frozen anchor/direct-"
            "state spring energies against the unevaluated raw input path; v2 changes only "
            "the independent auditor's energy-source provenance reconstruction, not the "
            "algorithm, adapter, controls, runtime, or expected trajectories"
        ),
        "superseded_campaign_provenance": superseded_provenance,
        "registered_before_first_formal_reproduction": True,
        "formal_artifacts_present_at_registration": [],
        "number_of_forces_budget_per_run": 3000,
        "execution_order_n_internal": [20, 10],
        "registered_members": [
            {
                "n_internal": 10,
                "alpha": 0.003,
                "alpha_label": "3e-3",
                "r": 0.1,
                "m_max": 7,
                "run_id": "0fa57295d005c597",
            },
            {
                "n_internal": 20,
                "alpha": 0.001,
                "alpha_label": "1e-3",
                "r": 0.3,
                "m_max": 8,
                "run_id": "8cb8b3a582ac6f8e",
            },
        ],
        "algorithm_contract": {
            "same_optimizer_source_for_both_resolutions": True,
            "tangent": "Henkelman-Jonsson improved tangent everywhere",
            "sole_reparameterization": (
                "nearest-neighbour spring flow F_i=(L_{i+1}-L_i)*tau_i"
            ),
            "reparameterization_constants": {
                "dtau": 0.05,
                "maximum_iterations": 500,
                "residual_tolerance": 1e-5,
                "energies_frozen_during_zero_nof_flow": True,
                "endpoints_fixed": True,
            },
            "forbidden_reparameterizations": [
                "central-difference",
                "averaged-chord",
                "polyline/equal-arclength",
                "cubic",
                "saddle",
            ],
            "recovery_is_not_reparameterization": True,
            "tuned_controls_only": ["alpha", "r", "m_max"],
            "no_nof_or_training_stage_parameter_schedule": True,
            "optimizer_cannot_read_article_or_offline_metrics": True,
            "all_remaining_constants_and_literals_frozen_by_source_sha256": True,
        },
        "fixed_constants": {
            "history_size": 15,
            "armijo": 0.01,
            "backtrack": 0.5,
            "kkt_absolute_tolerance": 1e-14,
            "rounding_factor": 64.0,
            "trust_radius_fraction": 2.0,
            "trust_radius_maximum": 1.0,
            "recovery_equidistance_contraction": 0.2,
            "recovery_secant_curvature": 1e-6,
            "optimizer_outer_safety_limit": 4096,
            "optimizer_zero_nof_safety_limit": 64,
        },
        "nof_contract": {
            "unit": "one uncached joint energy-gradient evaluation at one internal image",
            "endpoints_charged": False,
            "exact_coordinate_cache_hits_charged": False,
            "rejected_and_rolled_back_uncached_trials_charged": True,
            "directions_tangents_spring_and_offline_metrics_charged": False,
            "closure": (
                "terminal NoF = ordered miss-trace length = per-image sum = category sum"
            ),
        },
        "acceptance_rule": {
            "shared_initial_state_verified_and_excluded": True,
            "article_checkpoint_count": {"10": 79, "20": 98},
            "locf": "last IR-IVM public state at or before each exact Article NoF",
            "reference_barrier": 67.50188060819488,
            "densified_barrier": "coordinate-wise linear interpolation with 500 samples",
            "strict_at_every_checkpoint": [
                "I_perp(IR-IVM) < I_perp(Article IVM)",
                "abs(DeltaE(IR-IVM)-reference) < abs(DeltaE(Article IVM)-reference)",
            ],
            "equality_missing_nonfinite_or_integrity_failure": "fail",
            "secondary_diagnostic": (
                "public I_perp rise count and maximum rise; not part of primary acceptance"
            ),
        },
        "registered_inputs": dict(REGISTERED_INPUTS),
        "source_sha256": source_manifest,
        "dependency_sha256": dependency_manifest,
        "result_identity_paths": identity_paths,
        "result_identity": result_identity,
        "runtime_contract": runtime_contract,
        "preformal_evidence": evidence,
        "expected_formal_outputs": {
            "n10_result": "runs/n10/0fa57295d005c597.npz",
            "n20_result": "runs/n20/8cb8b3a582ac6f8e.npz",
            "n10_receipt": (
                "reports/receipts/n10/0fa57295d005c597.receipt.json"
            ),
            "n20_receipt": (
                "reports/receipts/n20/8cb8b3a582ac6f8e.receipt.json"
            ),
        },
    }


def main() -> int:
    paths = derive_campaign_paths(__file__)
    registration_path = (
        paths.campaign_root / "registrations/CAMPAIGN_PREREGISTRATION.json"
    )
    seal_path = registration_path.with_suffix(".sha256")
    try:
        if registration_path.exists() or seal_path.exists():
            raise PreregistrationError("registration or seal already exists")
        registration = build_registration()
        atomic_write_json(registration_path, registration, refuse_overwrite=True)
        digest = sha256_file(registration_path)
        atomic_write_bytes(
            seal_path, (digest + "\n").encode("ascii"), refuse_overwrite=True
        )
        # Reopen both exact bytes before reporting success.
        reopened = load_canonical_json(registration_path)
        if reopened != registration or sha256_file(registration_path) != digest:
            raise PreregistrationError("installed registration failed reopen check")
        if seal_path.read_bytes() != (digest + "\n").encode("ascii"):
            raise PreregistrationError("installed seal failed reopen check")
    except (FormalIOError, PreregistrationError, OSError, KeyError, ValueError) as exc:
        print("PREREGISTRATION REFUSED: " + str(exc))
        return 1
    print(
        json.dumps(
            {
                "campaign_id": CAMPAIGN_ID,
                "registration": str(registration_path),
                "registration_sha256": digest,
                "sealed_before_formal_execution": True,
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
