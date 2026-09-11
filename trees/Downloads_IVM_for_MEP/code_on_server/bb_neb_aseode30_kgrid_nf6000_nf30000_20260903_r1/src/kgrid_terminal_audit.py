"""Independent zero-Force terminal audit for one BB ASE-ODE k-grid run."""

from __future__ import annotations

from pathlib import Path
import stat
from typing import Any

import numpy as np

from .accounting import array_sha256, audit_wal
from .full_batch_recorder import audit_full_state_archive
from .integrity import (
    canonical_json_sha256,
    load_json_object,
    parse_json_object,
    sha256_file,
    tree_file_manifest,
    verify_source_identity,
)
from .kgrid_execution_authorization import verify_execution_authorization
from .kgrid_formal_spec import (
    load_and_verify_protocol,
    verify_inputs_and_endpoint_cache,
    verify_physical_contract,
)
from .kgrid_runner import (
    COMPLETE_SCHEMA,
    PRESTART_SCHEMA,
    TERMINAL_SEAL_SCHEMA,
)
from .kgrid_spec import ENDPOINT_CACHE_PATH, PROTOCOL_PATH, build_run


AUDIT_SCHEMA = "bb-neb-aseode-kgrid-external-terminal-audit-v1"


class KGridTerminalAuditError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KGridTerminalAuditError(message)


def _readonly_regular(path: Path, label: str) -> None:
    status = path.lstat()
    _require(
        stat.S_ISREG(status.st_mode) and not path.is_symlink(),
        f"{label} is not a regular nonsymlink file",
    )
    _require(stat.S_IMODE(status.st_mode) & 0o222 == 0, f"{label} is not read-only")


def _strict_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        _require(bool(line), f"empty optimization WAL line {line_number}")
        records.append(
            parse_json_object(line, source=f"{path}:line {line_number}")
        )
    return records


def audit_terminal(
    *, run_dir: Path | str, authorization_path: Path | str
) -> dict[str, Any]:
    """Audit a completed run without importing or constructing OpenMM."""

    source = verify_source_identity()
    protocol = load_and_verify_protocol(require_frozen=True)
    inputs = verify_inputs_and_endpoint_cache(protocol)
    physical = verify_physical_contract(protocol)
    authorization_file = Path(authorization_path).resolve(strict=True)
    authorization = verify_execution_authorization(
        authorization_file, require_fresh_run=False
    )
    candidate_record = authorization["candidate"]
    candidate = build_run(
        spring_k=candidate_record["spring_k_eV_per_A2"],
        phase=candidate_record["phase"],
        run_label=candidate_record["run_label"],
    )
    _require(candidate_record == candidate.as_dict(), "authorization candidate changed")
    run = Path(run_dir).resolve(strict=True)
    _require(
        run == Path(authorization["paths"]["run_directory"]).resolve()
        and run.name == candidate.run_label
        and run.is_dir()
        and not run.is_symlink(),
        "run directory differs from its authorization",
    )
    _require(
        not (run / "PERMANENT_FAIL_STOP.json").exists(),
        "permanent-fail-stop run is terminal-ineligible",
    )

    prestart_path = run / "PRESTART.json"
    configuration_path = run / "ASE_CONFIGURATION.json"
    completion_path = run / "OPTIMIZATION_COMPLETE.json"
    seal_path = run / "TERMINAL_SEAL.json"
    optimization_wal_path = run / "optimization_force_wal.jsonl"
    for path, label in (
        (prestart_path, "PRESTART"),
        (configuration_path, "ASE configuration"),
        (completion_path, "completion"),
        (seal_path, "terminal seal"),
    ):
        _readonly_regular(path, label)

    seal = load_json_object(seal_path, require_canonical=True)
    actual_tree = tree_file_manifest(
        run, excluded_relative_paths=("TERMINAL_SEAL.json",)
    )
    _require(
        seal.get("schema") == TERMINAL_SEAL_SCHEMA
        and seal.get("status") == "SEALED"
        and seal.get("candidate") == candidate.as_dict()
        and seal.get("paid_optimization_N_F") == candidate.terminal_nf
        and seal.get("paid_batches") == candidate.batches
        and seal.get("endpoint_optimization_N_F") == 0
        and seal.get("dense_calls") == 0
        and seal.get("full_state_shards") == candidate.batches
        and seal.get("external_terminal_audit_required") is True
        and seal.get("metrics_read") is False
        and seal.get("run_files_excluding_seal") == actual_tree
        and seal.get("run_file_count_excluding_seal") == len(actual_tree)
        and seal.get("run_files_closure_sha256")
        == canonical_json_sha256(actual_tree),
        "terminal tree seal contract failed",
    )

    prestart = load_json_object(prestart_path, require_canonical=True)
    _require(
        prestart.get("schema") == PRESTART_SCHEMA
        and prestart.get("status") == "claimed-before-OpenMM-context"
        and prestart.get("candidate") == candidate.as_dict()
        and prestart.get("authorization")
        == {"path": str(authorization_file), "sha256": sha256_file(authorization_file)}
        and prestart.get("source_identity") == source.as_dict()
        and prestart.get("protocol_sha256") == sha256_file(PROTOCOL_PATH)
        and prestart.get("physical_authority_file_sha256")
        == physical["authority_file_sha256"]
        and prestart.get("physical_identity_sha256") == physical["identity_sha256"]
        and prestart.get("initial_path_file_sha256")
        == inputs["initial_path_file_sha256"]
        and prestart.get("initial_path_array_sha256")
        == inputs["initial_path_array_sha256"]
        and prestart.get("endpoint_array_sha256") == inputs["endpoint_array_sha256"]
        and prestart.get("endpoint_cache_sha256") == inputs["endpoint_cache_sha256"]
        and prestart.get("pdb_sha256") == inputs["pdb_sha256"]
        and prestart.get("dual_host_local_BB_locks_held")
        == authorization["paths"]["host_local_BB_lock_paths"]
        and prestart.get("jobs") == 1
        and prestart.get("physical_force_calls") == 0
        and prestart.get("endpoint_force_calls") == 0
        and prestart.get("dense_calls") == 0
        and prestart.get("context_construction_started") is False
        and prestart.get("resume") is False
        and prestart.get("label_reusable") is False,
        "PRESTART/source/input/zero-Force binding failed",
    )

    configuration = load_json_object(configuration_path, require_canonical=True)
    _require(
        configuration.get("schema")
        == "bb-neb-aseode-kgrid-ase-configuration-v1"
        and configuration.get("candidate") == candidate.as_dict()
        and configuration.get("spring_k_eV_per_A2") == candidate.spring_k
        and configuration.get("optimizer_method") == "ODE"
        and configuration.get("ode_integrator") == "ase.optimize.ode.ode12r"
        and configuration.get("ode_single_invocation_required") is True
        and configuration.get("ode_trial_steps") == candidate.trial_steps
        and configuration.get("images") == 32
        and configuration.get("interior_paid_per_get_forces") == 30
        and configuration.get("endpoint_paid_per_get_forces") == 0
        and configuration.get("tangent")
        == "tau_i = normalize(R_(i+1) - R_(i-1))"
        and configuration.get(
            "central_tangent_rechecked_before_every_paid_optimizer_force_call"
        )
        is True
        and configuration.get("prestart_sha256") == sha256_file(prestart_path)
        and configuration.get("authorization_sha256")
        == sha256_file(authorization_file)
        and configuration.get("source_identity") == source.as_dict()
        and configuration.get("protocol_sha256") == sha256_file(PROTOCOL_PATH),
        "ASE/ODE/central-tangent configuration binding failed",
    )

    records = _strict_records(optimization_wal_path)
    # accounting.audit_wal independently recomputes the hash chain, paid
    # coordinate packages, requests, outcomes, commits, and snapshots.
    wal_audit = audit_wal(
        optimization_wal_path,
        require_terminal=True,
        expected_batches=candidate.batches,
        expected_milestones=candidate.milestones_nf,
    )
    reserves = [record for record in records if record.get("event") == "batch_reserve"]
    outcomes = [
        record
        for record in records
        if record.get("event") in {"force_response", "force_failure"}
    ]
    commits = [record for record in records if record.get("event") == "batch_commit"]
    snapshots = {
        int(record["nf"]): record
        for record in records
        if record.get("event") == "snapshot"
    }
    _require(
        len(reserves) == len(commits) == candidate.batches
        and [int(record["batch_id"]) for record in reserves]
        == list(range(1, candidate.batches + 1))
        and [int(record["batch_id"]) for record in commits]
        == list(range(1, candidate.batches + 1))
        and [str(record["state_role"]) for record in reserves]
        == ["current_initial"] + ["trial"] * candidate.trial_steps
        and str(commits[0]["disposition"]) == "current_initial"
        and all(
            str(record["disposition"]) in {"accepted", "rejected"}
            for record in commits[1:]
        )
        and len(outcomes) == candidate.terminal_nf
        and all(record.get("event") == "force_response" for record in outcomes)
        and [int(record["oracle_attempt_index"]) for record in outcomes]
        == list(range(1, candidate.terminal_nf + 1))
        and set(snapshots) == set(candidate.milestones_nf),
        "paid role/disposition/attempt/snapshot topology failed",
    )

    full_state = audit_full_state_archive(
        run / "paid_batch_full_state",
        spring_k=candidate.spring_k,
        terminal_batches=candidate.batches,
        run_label=candidate.run_label,
        endpoint_cache=ENDPOINT_CACHE_PATH,
        authorization_sha256=sha256_file(authorization_file),
        source_closure_sha256=source.whole_source_closure_sha256,
        optimizer_commits=commits,
    )
    completion = load_json_object(completion_path, require_canonical=True)
    accepted = sum(str(record["disposition"]) == "accepted" for record in commits)
    rejected = sum(str(record["disposition"]) == "rejected" for record in commits)
    _require(
        completion.get("schema") == COMPLETE_SCHEMA
        and completion.get("status") == "OPTIMIZATION_COMPLETE"
        and completion.get("candidate") == candidate.as_dict()
        and completion.get("method")
        == "ase.mep.neb.NEB+NEBOptimizer(method=ODE)"
        and completion.get("spring_k_eV_per_A2") == candidate.spring_k
        and completion.get("paid_batches") == candidate.batches
        and completion.get("paid_nf") == candidate.terminal_nf
        and completion.get("paid_force_evaluations") == candidate.terminal_nf
        and completion.get("oracle_attempts") == candidate.terminal_nf
        and completion.get("oracle_successes") == candidate.terminal_nf
        and completion.get("endpoint_physical_calls") == 0
        and completion.get("dense_calls") == 0
        and completion.get("full_state_shards") == candidate.batches
        and completion.get("full_state_completion_sha256")
        == full_state["completion_sha256"]
        and completion.get("full_state_wal_sha256") == full_state["wal_sha256"]
        and completion.get("optimization_wal_sha256")
        == sha256_file(optimization_wal_path)
        and completion.get("ode_invocations") == 1
        and completion.get("official_ode12r_calls") == 1
        and completion.get("get_forces_calls") == candidate.batches
        and completion.get("current_initial_batches") == 1
        and completion.get("trial_batches") == candidate.trial_steps
        and completion.get("current_restart_batches") == 0
        and completion.get("accepted_ode_steps") == accepted
        and completion.get("rejected_ode_trials") == rejected
        and accepted >= 1
        and accepted + rejected == candidate.trial_steps
        and completion.get("prestart_sha256") == sha256_file(prestart_path)
        and completion.get("authorization_sha256")
        == sha256_file(authorization_file)
        and completion.get("source_identity") == source.as_dict()
        and completion.get("protocol_sha256") == sha256_file(PROTOCOL_PATH)
        and completion.get("source_reverified_after_final_force") is True
        and completion.get("resume") is False
        and completion.get("label_reusable") is False
        and completion.get("external_terminal_audit_required") is True,
        "completion/WAL/Force-count/full-state contract failed",
    )
    _require(
        seal.get("completion_sha256") == sha256_file(completion_path),
        "terminal seal completion hash failed",
    )

    snapshot_dir = run / "snapshots"
    expected_names = {"nf00000.npz"} | {
        f"nf{nf:05d}.npz" for nf in candidate.milestones_nf
    }
    _require(
        {path.name for path in snapshot_dir.iterdir()} == expected_names,
        "snapshot file set is not exact",
    )
    with np.load(snapshot_dir / "nf00000.npz", allow_pickle=False) as archive:
        nf0 = np.asarray(archive["nodes"], dtype=np.float64)
        _require(
            int(np.asarray(archive["optimization_N_F"]).reshape(())) == 0
            and int(np.asarray(archive["evaluated_current"]).reshape(())) == 0
            and float(np.asarray(archive["spring_k_eV_per_A2"]).reshape(()))
            == candidate.spring_k
            and np.array_equal(nf0, inputs["nodes"]),
            "NF0 snapshot differs from the unevaluated common initial path",
        )
    declared_snapshots = completion.get("snapshot_sha256")
    _require(
        isinstance(declared_snapshots, dict)
        and set(declared_snapshots) == {str(nf) for nf in candidate.milestones_nf},
        "completion snapshot map is not exact",
    )
    for nf in candidate.milestones_nf:
        path = snapshot_dir / f"nf{nf:05d}.npz"
        event = snapshots[nf]
        with np.load(path, allow_pickle=False) as archive:
            nodes = np.asarray(archive["nodes"], dtype=np.float64)
            observed = {
                "optimization_N_F": int(
                    np.asarray(archive["optimization_N_F"]).reshape(())
                ),
                "batch_id": int(np.asarray(archive["batch_id"]).reshape(())),
                "evaluated_current": int(
                    np.asarray(archive["evaluated_current"]).reshape(())
                ),
                "state_sha256": str(
                    np.asarray(archive["state_sha256"]).reshape(())
                ),
                "spring_k": float(
                    np.asarray(archive["spring_k_eV_per_A2"]).reshape(())
                ),
                "tangent": str(np.asarray(archive["tangent"]).reshape(())),
            }
        _require(
            observed["optimization_N_F"] == nf
            and observed["batch_id"] * 30 == nf
            and observed["evaluated_current"] == 1
            and observed["spring_k"] == candidate.spring_k
            and observed["tangent"]
            == "tau_i = normalize(R_(i+1) - R_(i-1))"
            and observed["state_sha256"] == str(event["state_sha256"])
            and array_sha256(nodes.reshape((32, -1, 3)))
            == str(event["path32_sha256"])
            and declared_snapshots[str(nf)] == sha256_file(path),
            f"snapshot/current/WAL binding failed at NF{nf}",
        )
        _require(
            np.array_equal(nodes[[0, -1]], inputs["nodes"][[0, -1]]),
            f"fixed endpoint changed at NF{nf}",
        )
    _require(
        completion.get("terminal_snapshot_sha256")
        == declared_snapshots[str(candidate.terminal_nf)],
        "terminal snapshot binding failed",
    )

    return {
        "schema": AUDIT_SCHEMA,
        "status": "PASS",
        "terminal_eligible": True,
        "selection_eligible": candidate.phase == "screening",
        "formal_figure_eligible": candidate.phase == "formal",
        "candidate": candidate.as_dict(),
        "run_dir": str(run),
        "authorization": {
            "path": str(authorization_file),
            "sha256": sha256_file(authorization_file),
        },
        "source_identity": source.as_dict(),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "prestart_sha256": sha256_file(prestart_path),
        "configuration_sha256": sha256_file(configuration_path),
        "optimization_wal_sha256": sha256_file(optimization_wal_path),
        "full_state_wal_sha256": full_state["wal_sha256"],
        "full_state_completion_sha256": full_state["completion_sha256"],
        "completion_sha256": sha256_file(completion_path),
        "terminal_seal_sha256": sha256_file(seal_path),
        "terminal_tree_closure_sha256": seal["run_files_closure_sha256"],
        "paid_batches": candidate.batches,
        "paid_optimization_N_F": candidate.terminal_nf,
        "oracle_attempts": candidate.terminal_nf,
        "oracle_successes": candidate.terminal_nf,
        "accepted_trials": accepted,
        "rejected_trials": rejected,
        "endpoint_optimization_N_F": 0,
        "dense_calls": 0,
        "full_state_shards": candidate.batches,
        "wal_audit": wal_audit,
        "full_state_audit": full_state,
        "physical_force_calls_by_external_audit": 0,
        "OpenMM_contexts_constructed_by_external_audit": 0,
        "metrics_computed_by_external_audit": False,
    }
