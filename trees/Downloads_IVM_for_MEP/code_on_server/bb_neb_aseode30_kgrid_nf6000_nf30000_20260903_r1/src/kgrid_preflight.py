"""Zero-physical preflight for one authorized k-grid run."""

from __future__ import annotations

import fcntl
import inspect
from pathlib import Path
import shutil
from typing import Any

import ase
import ase.mep.neb as ase_neb_module
import ase.optimize.ode as ase_ode_module

from .integrity import sha256_file, verify_source_identity
from .kgrid_execution_authorization import (
    require_thread_environment,
    verify_execution_authorization,
)
from .kgrid_formal_spec import (
    load_and_verify_protocol,
    verify_inputs_and_endpoint_cache,
    verify_physical_contract,
)
from .kgrid_spec import ASE_NEB_SHA256, ASE_ODE_SHA256, ASE_VERSION, build_run


PREFLIGHT_SCHEMA = "bb-neb-aseode-kgrid-zero-physical-preflight-v1"


class KGridPreflightError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KGridPreflightError(message)


def verify_ase_runtime() -> dict[str, Any]:
    neb_path = Path(inspect.getsourcefile(ase_neb_module) or "").resolve(strict=True)
    ode_path = Path(inspect.getsourcefile(ase_ode_module) or "").resolve(strict=True)
    result = {
        "ase_version": str(ase.__version__),
        "ase_neb_runtime_provenance": "pre-existing locally modified ASE 3.23.0 neb.py",
        "ase_mep_neb_py_path": str(neb_path),
        "ase_mep_neb_py_sha256": sha256_file(neb_path),
        "ase_optimize_ode_py_path": str(ode_path),
        "ase_optimize_ode_py_sha256": sha256_file(ode_path),
        "ode12r_module": ase_ode_module.ode12r.__module__,
        "ode12r_name": ase_ode_module.ode12r.__name__,
    }
    _require(result["ase_version"] == ASE_VERSION
             and result["ase_mep_neb_py_sha256"] == ASE_NEB_SHA256
             and result["ase_optimize_ode_py_sha256"] == ASE_ODE_SHA256
             and result["ode12r_module"] == "ase.optimize.ode"
             and result["ode12r_name"] == "ode12r",
             f"ASE 3.23 NEB/ODE runtime identity changed: {result}")
    return {"pass": True, **result}


def _probe_locks(paths: list[str]) -> None:
    streams = []
    try:
        for raw in paths:
            path = Path(raw).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except Exception:
                stream.close()
                raise
            streams.append(stream)
    finally:
        for stream in reversed(streams):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def preflight_authorized_run(
    authorization_path: Path | str, *, probe_locks: bool
) -> dict[str, Any]:
    """Verify every non-OpenMM start gate; no Context is constructed."""

    thread_environment = require_thread_environment()
    authorization_file = Path(authorization_path).resolve(strict=True)
    authorization = verify_execution_authorization(authorization_file)
    protocol = load_and_verify_protocol(require_frozen=True)
    source = verify_source_identity()
    inputs = verify_inputs_and_endpoint_cache(protocol)
    physical = verify_physical_contract(protocol)
    runtime = verify_ase_runtime()
    candidate_data = authorization["candidate"]
    candidate = build_run(
        spring_k=candidate_data["spring_k_eV_per_A2"],
        phase=candidate_data["phase"],
        run_label=candidate_data["run_label"],
    )
    output_parent = Path(authorization["paths"]["output_parent"])
    disk_probe = output_parent.parent
    _require(disk_probe.exists(), "campaign parent must exist before preflight")
    free = shutil.disk_usage(disk_probe).free
    history_required = candidate.batches * 10_000_000 + (1 << 30)
    required = max(history_required, 5 * 1024**3)
    _require(free >= required,
             f"disk gate failed without compression credit: {free} < {required}")
    lock_paths = list(authorization["paths"]["host_local_BB_lock_paths"])
    _require(len(lock_paths) == len(set(lock_paths)) == 2,
             "authorization does not bind two distinct host-local BB locks")
    if probe_locks:
        try:
            _probe_locks(lock_paths)
        except BlockingIOError as error:
            raise KGridPreflightError("one host-local BB lock is held") from error
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "PASS",
        "authorization": {"path": str(authorization_file),
                          "sha256": sha256_file(authorization_file)},
        "candidate": candidate.as_dict(),
        "source_identity": source.as_dict(),
        "protocol_sha256": sha256_file(Path(authorization["protocol"]["path"])),
        "input_bindings": {
            "initial_path_file_sha256": inputs["initial_path_file_sha256"],
            "initial_path_array_sha256": inputs["initial_path_array_sha256"],
            "endpoint_array_sha256": inputs["endpoint_array_sha256"],
            "endpoint_cache_sha256": inputs["endpoint_cache_sha256"],
            "pdb_sha256": inputs["pdb_sha256"],
        },
        "physical_contract": physical,
        "ase_runtime": runtime,
        "thread_environment": thread_environment,
        "jobs": 1,
        "dual_host_local_BB_lock_paths": lock_paths,
        "locks_probed_available": bool(probe_locks),
        "disk_free_bytes": int(free),
        "disk_required_bytes": int(required),
        "disk_compression_credit": 0,
        "OpenMM_contexts_constructed": 0,
        "physical_force_calls": 0,
        "dense_calls": 0,
    }
