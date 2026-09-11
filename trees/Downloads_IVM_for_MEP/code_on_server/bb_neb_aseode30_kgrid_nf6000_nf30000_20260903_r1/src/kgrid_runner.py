"""Authorized fresh-only producer for BB ASE ``NEBOptimizer(method='ODE')``."""

from __future__ import annotations

import json
from pathlib import Path
import time
import traceback
from typing import Any

from ase.io import read as ase_read
import numpy as np

from .accounting import PaidInteriorLedger
from .ase_neb import (
    ODE_MAXTOL_EV_PER_A,
    PaidBatchCoordinator,
    PaidODENEBOptimizer,
    assert_formal_ase_configuration,
    build_ase_neb,
)
from .full_batch_recorder import FullBatchRecorder
from .integrity import (
    SOURCE_ROOT,
    canonical_json_sha256,
    sha256_file,
    tree_file_manifest,
    verify_source_identity,
    write_json_new,
    write_npz_new,
)
from .kgrid_execution_authorization import verify_execution_authorization
from .kgrid_formal_spec import (
    load_and_verify_protocol,
    verify_inputs_and_endpoint_cache,
    verify_physical_contract,
)
from .kgrid_preflight import preflight_authorized_run
from .kgrid_spec import ENDPOINT_CACHE_PATH, PDB_PATH, PROTOCOL_PATH, build_run
from .oracle import OpenMMOracle
from .resource_guard import ResourceGuard


PRESTART_SCHEMA = "bb-neb-aseode-kgrid-prestart-v1"
COMPLETE_SCHEMA = "bb-neb-aseode-kgrid-optimization-complete-v1"
TERMINAL_SEAL_SCHEMA = "bb-neb-aseode-kgrid-terminal-artifact-seal-v1"
FAIL_STOP_SCHEMA = "bb-neb-aseode-kgrid-permanent-fail-stop-v1"


class KGridRunnerError(RuntimeError):
    pass


class DeferredOracle:
    """Keep physical evaluation unreachable until PRESTART is durable."""

    def __init__(self) -> None:
        self._target: Any | None = None

    @property
    def attempt_count(self) -> int:
        return 0 if self._target is None else int(self._target.attempt_count)

    @property
    def success_count(self) -> int:
        return 0 if self._target is None else int(self._target.success_count)

    def bind(self, target: Any) -> None:
        if self._target is not None:
            raise KGridRunnerError("physical oracle gate may bind exactly once")
        if int(getattr(target, "attempt_count", -1)) != 0 \
                or int(getattr(target, "success_count", -1)) != 0:
            raise KGridRunnerError("oracle made a Force call before PRESTART binding")
        self._target = target

    def evaluate(self, coordinates_A: Any) -> Any:
        if self._target is None:
            raise KGridRunnerError("physical oracle remains unreachable before PRESTART")
        return self._target.evaluate(coordinates_A)

    def metadata(self) -> dict[str, Any]:
        if self._target is None:
            raise KGridRunnerError("oracle metadata requested before PRESTART binding")
        return dict(self._target.metadata())


def _verify_oracle_zero_force(gate: DeferredOracle, physical: dict[str, Any]) -> dict[str, Any]:
    metadata = gate.metadata()
    expected = {
        "openmm_version": physical["openmm_version"],
        "atom_count": physical["atom_count"],
        "system_xml_bytes": physical["system_xml_bytes"],
        "system_xml_sha256": physical["system_xml_sha256"],
        "force_class_sequence": physical["force_class_sequence"],
        "cache": False,
        "joint_energy_force": True,
        "platform": "CPU",
        "precision": "double",
    }
    mismatch = {key: {"actual": metadata.get(key), "expected": value}
                for key, value in expected.items() if metadata.get(key) != value}
    if mismatch:
        raise KGridRunnerError(f"zero-Force OpenMM runtime contract changed: {mismatch}")
    if gate.attempt_count != 0 or gate.success_count != 0:
        raise KGridRunnerError("OpenMM Context construction consumed a physical Force call")
    return {**metadata, "authority_file_sha256": physical["authority_file_sha256"],
            "identity_sha256": physical["identity_sha256"],
            "attempt_count_before_first_paid_reserve": 0,
            "success_count_before_first_paid_reserve": 0}


def _save_snapshot(path: Path, *, spring_k: float, path32_A: Any, nf: int,
                   batch_id: int, state_sha256: str, disposition: str,
                   ase_residual: float, optimizer_metadata: dict[str, Any], **_: Any) -> None:
    nodes = np.asarray(path32_A, dtype=np.float64).reshape((32, -1))
    write_npz_new(
        path,
        nodes=nodes,
        optimization_N_F=np.asarray(nf, dtype=np.int64),
        batch_id=np.asarray(batch_id, dtype=np.int64),
        evaluated_current=np.asarray(1, dtype=np.int8),
        method=np.asarray("ase.mep.neb.NEB+NEBOptimizer(method=ODE)"),
        tangent=np.asarray("tau_i = normalize(R_(i+1) - R_(i-1))"),
        state_sha256=np.asarray(state_sha256),
        disposition=np.asarray(disposition),
        ase_residual=np.asarray(ase_residual, dtype=np.float64),
        spring_k_eV_per_A2=np.asarray(spring_k, dtype=np.float64),
        ode_maxtol_eV_per_A=np.asarray(ODE_MAXTOL_EV_PER_A, dtype=np.float64),
        full_state_shard_sha256=np.asarray(
            optimizer_metadata["full_state_archive"]["file_sha256"]
        ),
    )


def _write_fail_stop(
    run_dir: Path, error: BaseException, *, candidate: Any, reserved_nf: int
) -> None:
    path = run_dir / "PERMANENT_FAIL_STOP.json"
    if not run_dir.exists() or path.exists():
        return
    write_json_new(
        path,
        {
            "schema": FAIL_STOP_SCHEMA,
            "status": "permanent-fail-stop",
            "candidate": candidate.as_dict(),
            "reserved_optimization_N_F": int(reserved_nf),
            "exception_class": type(error).__module__ + "." + type(error).__name__,
            "exception": str(error)[:4000],
            "traceback": traceback.format_exc(),
            "resume": False,
            "label_reusable": False,
        },
        readonly=True,
    )


def _terminal_seal_document(
    run_dir: Path, *, candidate: Any, completion_sha256: str
) -> dict[str, Any]:
    entries = tree_file_manifest(run_dir, excluded_relative_paths=("TERMINAL_SEAL.json",))
    return {
        "schema": TERMINAL_SEAL_SCHEMA,
        "status": "SEALED",
        "candidate": candidate.as_dict(),
        "paid_optimization_N_F": candidate.terminal_nf,
        "paid_batches": candidate.batches,
        "endpoint_optimization_N_F": 0,
        "dense_calls": 0,
        "full_state_shards": candidate.batches,
        "completion_sha256": completion_sha256,
        "run_file_count_excluding_seal": len(entries),
        "run_files_excluding_seal": entries,
        "run_files_closure_sha256": canonical_json_sha256(entries),
        "external_terminal_audit_required": True,
        "metrics_read": False,
    }


def run_from_authorization(authorization_path: Path | str) -> int:
    """Run the sole candidate encoded by an immutable authorization file."""

    authorization_file = Path(authorization_path).resolve(strict=True)
    preflight = preflight_authorized_run(authorization_file, probe_locks=False)
    authorization = verify_execution_authorization(authorization_file)
    candidate_record = authorization["candidate"]
    candidate = build_run(
        spring_k=candidate_record["spring_k_eV_per_A2"],
        phase=candidate_record["phase"],
        run_label=candidate_record["run_label"],
    )
    protocol = load_and_verify_protocol(require_frozen=True)
    source_start = verify_source_identity()
    inputs = verify_inputs_and_endpoint_cache(protocol)
    physical = verify_physical_contract(protocol)
    nodes_flat = np.asarray(inputs["nodes"], dtype=np.float64)
    nodes = nodes_flat.reshape((32, -1, 3))
    run_dir = Path(authorization["paths"]["run_directory"]).resolve()
    if run_dir.exists():
        raise FileExistsError(f"fresh output already exists: {run_dir}")

    guard: Any | None = None
    ledger: PaidInteriorLedger | None = None
    physical_oracle: Any | None = None
    recorder: FullBatchRecorder | None = None
    gate = DeferredOracle()
    reserved_nf = 0
    terminal_success = False
    try:
        guard = ResourceGuard(
            run_dir,
            machine=str(authorization["target_machine"]["machine_id"]),
            lock_paths=authorization["paths"]["host_local_BB_lock_paths"],
        )
        # Resource disk gates must follow the actual campaign volume.  Remote
        # production stores raw histories on an external volume while both
        # host-local BB lock files remain under the two canonical checkouts.
        guard.project_root = run_dir
        guard.preflight()
        authorization_sha = sha256_file(authorization_file)
        # The paid ledger owns fresh run-directory creation.  Construct it
        # before the recorder so its strict fresh-label gate cannot mistake
        # this run's new ``paid_batch_full_state`` directory for prior state.
        ledger = PaidInteriorLedger(
            run_dir,
            candidate.accounting(),
            gate,
            method="neb_ase_ode",
            label=candidate.run_label,
            frozen_endpoints_A=nodes[[0, -1]],
            protocol_sha256=sha256_file(PROTOCOL_PATH),
            source_closure_sha256=source_start.whole_source_closure_sha256,
            before_slot=lambda batch_id, slot_id: guard.assert_safe(batch_id, slot_id),
            max_batches=candidate.batches,
        )
        recorder = FullBatchRecorder(
            run_dir,
            ENDPOINT_CACHE_PATH,
            spring_k=candidate.spring_k,
            terminal_batches=candidate.batches,
            run_label=candidate.run_label,
            authorization_sha256=authorization_sha,
            source_closure_sha256=source_start.whole_source_closure_sha256,
        )
        recorder_disk = recorder.disk_preflight()
        prestart = {
            "schema": PRESTART_SCHEMA,
            "status": "claimed-before-OpenMM-context",
            "candidate": candidate.as_dict(),
            "authorization": {"path": str(authorization_file),
                              "sha256": authorization_sha},
            "source_identity": source_start.as_dict(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "preflight": preflight,
            "recorder_disk_preflight": recorder_disk,
            "physical_authority_file_sha256": physical["authority_file_sha256"],
            "physical_identity_sha256": physical["identity_sha256"],
            "initial_path_file_sha256": inputs["initial_path_file_sha256"],
            "initial_path_array_sha256": inputs["initial_path_array_sha256"],
            "endpoint_array_sha256": inputs["endpoint_array_sha256"],
            "endpoint_cache_sha256": inputs["endpoint_cache_sha256"],
            "pdb_sha256": inputs["pdb_sha256"],
            "dual_host_local_BB_locks_held": authorization["paths"][
                "host_local_BB_lock_paths"
            ],
            "jobs": 1,
            "physical_force_calls": 0,
            "endpoint_force_calls": 0,
            "dense_calls": 0,
            "context_construction_started": False,
            "resume": False,
            "label_reusable": False,
        }
        prestart_sha = write_json_new(run_dir / "PRESTART.json", prestart, readonly=True)

        # The only OpenMM Context construction is deliberately below PRESTART.
        physical_oracle = OpenMMOracle(PDB_PATH)
        gate.bind(physical_oracle)
        oracle_zero_force = _verify_oracle_zero_force(gate, physical)
        guard.rebaseline(reason="post_openmm_context_zero_force")

        snapshot_dir = run_dir / "snapshots"
        snapshot_dir.mkdir(mode=0o700)
        write_npz_new(
            snapshot_dir / "nf00000.npz",
            nodes=nodes_flat,
            optimization_N_F=np.asarray(0, dtype=np.int64),
            evaluated_current=np.asarray(0, dtype=np.int8),
            spring_k_eV_per_A2=np.asarray(candidate.spring_k, dtype=np.float64),
        )
        last_report = 0.0

        def batch_hook(**payload: Any) -> None:
            nonlocal last_report, reserved_nf
            batch_id = int(payload["batch_id"])
            reserved_nf = int(payload["nf"])
            resource = guard.checkpoint(batch_id)
            now = time.time()
            if now - last_report >= 1200.0 or reserved_nf == candidate.terminal_nf:
                print(json.dumps({
                    "candidate_id": candidate.candidate_id,
                    "phase": candidate.phase,
                    "spring_k_eV_per_A2": candidate.spring_k,
                    "paid_optimization_N_F": reserved_nf,
                    "batch": batch_id,
                    "full_state_shards": recorder.last_batch,
                    "oracle_attempts": gate.attempt_count,
                    "oracle_successes": gate.success_count,
                    "endpoint_force_calls": 0,
                    "dense_calls": 0,
                    "memory_free_percent": resource["memory_free_percent"],
                    "disk_free_bytes": resource["disk_free_bytes"],
                }, sort_keys=True), flush=True)
                last_report = now

        def snapshot_hook(**payload: Any) -> None:
            nf = int(payload["nf"])
            if nf not in candidate.milestones_nf:
                raise KGridRunnerError("optimizer attempted an unregistered milestone")
            _save_snapshot(
                snapshot_dir / f"nf{nf:05d}.npz",
                spring_k=candidate.spring_k,
                **payload,
            )

        coordinator = PaidBatchCoordinator(
            ledger,
            spring_k=candidate.spring_k,
            snapshot_hook=snapshot_hook,
            batch_hook=batch_hook,
            full_state_hook=recorder.record,
            max_batches=candidate.batches,
        )
        template = ase_read(str(PDB_PATH), index=0)
        band = build_ase_neb(nodes, template, coordinator, spring_k=candidate.spring_k)
        optimizer = PaidODENEBOptimizer(band, coordinator)
        ase_configuration = assert_formal_ase_configuration(
            band, optimizer, candidate.spring_k
        )
        ase_configuration.update({
            "schema": "bb-neb-aseode-kgrid-ase-configuration-v1",
            "candidate": candidate.as_dict(),
            "central_tangent_rechecked_before_every_paid_optimizer_force_call": True,
            "ase_runtime_identity": preflight["ase_runtime"],
            "oracle_system_identity": oracle_zero_force,
            "prestart_sha256": prestart_sha,
            "authorization_sha256": authorization_sha,
            "source_identity": source_start.as_dict(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
        })
        write_json_new(run_dir / "ASE_CONFIGURATION.json", ase_configuration, readonly=True)
        if gate.attempt_count != 0 or gate.success_count != 0:
            raise KGridRunnerError("pre-optimizer setup consumed a physical Force call")

        report = optimizer.run_to_exact_budget()
        reserved_nf = int(report["paid_nf"])
        if recorder.last_batch != candidate.batches:
            raise KGridRunnerError("full-state archive count differs from paid optimizer batches")
        source_terminal = verify_source_identity()
        if source_terminal != source_start:
            raise KGridRunnerError("source identity changed during physical optimization")
        if gate.attempt_count != candidate.terminal_nf \
                or gate.success_count != candidate.terminal_nf:
            raise KGridRunnerError(
                f"terminal OpenMM counts differ: attempts={gate.attempt_count}, "
                f"successes={gate.success_count}, expected={candidate.terminal_nf}"
            )
        full_state_complete = recorder.seal()
        ledger.seal()
        ledger.close()
        ledger = None
        physical_oracle.close()
        physical_oracle = None
        guard.close()
        guard = None

        snapshot_sha = {
            str(nf): sha256_file(snapshot_dir / f"nf{nf:05d}.npz")
            for nf in candidate.milestones_nf
        }
        completion = {
            "schema": COMPLETE_SCHEMA,
            "status": "OPTIMIZATION_COMPLETE",
            "candidate": candidate.as_dict(),
            "method": "ase.mep.neb.NEB+NEBOptimizer(method=ODE)",
            "spring_k_eV_per_A2": candidate.spring_k,
            **report,
            "paid_force_evaluations": candidate.terminal_nf,
            "oracle_attempts": gate.attempt_count,
            "oracle_successes": gate.success_count,
            "endpoint_physical_calls": 0,
            "dense_calls": 0,
            "full_state_shards": candidate.batches,
            "full_state_completion_sha256": sha256_file(recorder.complete_path),
            "full_state_wal_sha256": full_state_complete["wal_sha256"],
            "optimization_wal_sha256": sha256_file(
                run_dir / "optimization_force_wal.jsonl"
            ),
            "snapshot_sha256": snapshot_sha,
            "terminal_snapshot_sha256": snapshot_sha[str(candidate.terminal_nf)],
            "prestart_sha256": prestart_sha,
            "authorization_sha256": authorization_sha,
            "source_identity": source_terminal.as_dict(),
            "protocol_sha256": sha256_file(PROTOCOL_PATH),
            "source_reverified_after_final_force": True,
            "resume": False,
            "label_reusable": False,
            "external_terminal_audit_required": True,
        }
        completion_sha = write_json_new(
            run_dir / "OPTIMIZATION_COMPLETE.json", completion, readonly=True
        )
        write_json_new(
            run_dir / "TERMINAL_SEAL.json",
            _terminal_seal_document(
                run_dir, candidate=candidate, completion_sha256=completion_sha
            ),
            readonly=True,
        )
        terminal_success = True
        return 0
    except BaseException as error:
        if ledger is not None and not ledger.sealed and not ledger.failed:
            try:
                ledger.fail_stop_active(
                    "kgrid_runner_exception",
                    metadata={"exception_class": type(error).__module__ + "."
                              + type(error).__name__,
                              "exception": str(error)[:2000]},
                )
                reserved_nf = int(ledger.reserved_nf)
            except Exception:
                pass
        if run_dir.exists() and not terminal_success:
            try:
                _write_fail_stop(
                    run_dir, error, candidate=candidate, reserved_nf=reserved_nf
                )
            except Exception:
                pass
        raise
    finally:
        if ledger is not None:
            try:
                ledger.close()
            except Exception:
                pass
        if physical_oracle is not None:
            try:
                physical_oracle.close()
            except Exception:
                pass
        if guard is not None:
            try:
                guard.close()
            except Exception:
                pass
