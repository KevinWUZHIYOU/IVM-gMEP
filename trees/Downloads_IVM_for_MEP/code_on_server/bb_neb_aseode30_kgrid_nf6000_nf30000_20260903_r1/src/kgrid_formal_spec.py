"""Fail-closed protocol, input, endpoint-cache, and OpenMM verification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .accounting import array_sha256
from .full_batch_recorder import _load_endpoint_cache
from .integrity import (
    IntegrityError,
    canonical_json_sha256,
    load_json_object,
    sha256_file,
)
from .kgrid_spec import (
    ASE_NEB_SHA256,
    ASE_ODE_SHA256,
    ASE_VERSION,
    ATOM_COORDINATES,
    CONTRACT_PATH,
    ENDPOINT_ARRAY_SHA256,
    ENDPOINT_CACHE_PATH,
    ENDPOINT_CACHE_SHA256,
    FORMAL_MILESTONES,
    INITIAL_ARRAY_SHA256,
    INITIAL_FILE_SHA256,
    INITIAL_PATH,
    K_VALUES,
    PDB_PATH,
    PDB_SHA256,
    PHYSICAL_CONTRACT_SHA256,
    PHYSICAL_IDENTITY_SHA256,
    PREREGISTRATION_PATH,
    PROTOCOL_PATH,
    SCREENING_MILESTONES,
    SELECTION_PROTOCOL_PATH,
    SYSTEM_XML_SHA256,
)


PROTOCOL_SCHEMA = "bb-neb-aseode30-kgrid-nf6000-nf30000-protocol-v1"
FROZEN_STATUS = "SOURCE_FROZEN_ZERO_PHYSICAL_READY_FOR_INDEPENDENT_AUDIT"
BUILD_STATUS = "ZERO_PHYSICAL_BUILD_PENDING_SOURCE_FREEZE"


class KGridFormalSpecError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KGridFormalSpecError(message)


def _raw_array_sha256(value: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def load_and_verify_protocol(*, require_frozen: bool = True) -> dict[str, Any]:
    try:
        protocol = load_json_object(PROTOCOL_PATH)
    except (OSError, IntegrityError) as error:
        raise KGridFormalSpecError(str(error)) from error
    _require(protocol.get("schema") == PROTOCOL_SCHEMA, "k-grid protocol schema changed")
    _require(set(protocol) == {
        "schema", "status", "physical_force_calls_authorized_by_this_file",
        "spring_k_eV_per_A2", "phase_budgets", "accounting", "optimizer",
        "full_state_archive", "selection", "security", "input_bindings",
        "runtime", "scientific_core", "preregistration_sha256",
        "selection_protocol_sha256",
    }, "k-grid protocol top-level fields changed")
    allowed = {FROZEN_STATUS} if require_frozen else {BUILD_STATUS, FROZEN_STATUS}
    _require(protocol.get("status") in allowed, "k-grid protocol status is not eligible")
    _require(protocol.get("physical_force_calls_authorized_by_this_file") == 0,
             "protocol cannot authorize a physical Force call")
    _require(protocol.get("spring_k_eV_per_A2") == list(K_VALUES), "spring grid changed")
    _require(protocol.get("phase_budgets") == {
        "screening": {"terminal_optimization_N_F": 6000, "paid_batches": 200},
        "formal": {"terminal_optimization_N_F": 30000, "paid_batches": 1000},
    }, "phase budgets changed")
    _require(protocol.get("accounting") == {
        "images": 32,
        "paid_interiors_per_batch": 30,
        "fixed_endpoints": 2,
        "endpoint_optimization_N_F": 0,
        "new_OpenMM_calls_per_paid_batch": 30,
        "failed_and_rejected_calls_are_paid": True,
        "fresh_only": True,
        "resume_allowed": False,
        "screening_milestones_N_F": list(SCREENING_MILESTONES),
        "formal_milestones_N_F": list(FORMAL_MILESTONES),
    }, "k-grid accounting contract changed")
    _require(protocol.get("optimizer") == {
        "ase_version": ASE_VERSION,
        "neb_class": "ase.mep.neb.NEB",
        "neb_method_argument": "aseneb",
        "optimizer_class": "ase.mep.neb.NEBOptimizer",
        "optimizer_method_argument": "ODE",
        "ode_integrator": "ase.optimize.ode.ode12r",
        "single_ode_invocation": True,
        "central_tangent": "tau_i = normalize(R_(i+1) - R_(i-1))",
        "central_tangent_rechecked_before_every_paid_optimizer_force_call": True,
        "climb": False,
        "parallel": False,
        "remove_rotation_and_translation": False,
        "fmax_eV_per_A": 0.0,
        "maxtol_eV_per_A": 1000000.0,
        "rtol": 0.1,
        "C1": 0.01,
        "C2": 2.0,
    }, "ASE ODE/central-tangent contract changed")
    _require(protocol.get("full_state_archive") == {
        "every_paid_call": True,
        "arrays": ["paid_path32", "energy32", "full_force32", "current_path32"],
        "endpoint_E_F_from_precomputed_cache": True,
        "endpoint_additional_OpenMM_calls": 0,
        "published_before_optimizer_ledger_commit": True,
        "accepted_rejected_current_mapping": True,
        "encoding": "25-batch float64 checkpoint plus bit-exact uint64 XOR deltas",
        "hash_chained_WAL": True,
        "terminal_seal": True,
    }, "full-state archive contract changed")
    _require(protocol.get("selection") == {
        "protocol_path": "KGRID_SELECTION_PROTOCOL.json",
        "screening_requires_all_four_complete_traces": True,
        "formal_authorization_requires_frozen_selection_receipt": True,
        "formal_starts_fresh_N_F0": True,
        "screening_state_not_resumed_or_spliced": True,
        "densification": False,
    }, "selection/fresh-formal contract changed")
    _require(protocol.get("security") == {
        "source_manifest_required": True,
        "independent_source_audit_required": True,
        "mutagen_gate_required": True,
        "preregistration_hash_bound": True,
        "authorization_O_EXCL_fsync_readonly": True,
        "prestart_before_OpenMM_context": True,
        "exclusive_output": True,
        "dual_host_local_BB_fcntl_locks": True,
        "jobs": 1,
        "all_six_thread_caps": 1,
        "disk_preflight_without_compression_credit": True,
        "permanent_fail_stop": True,
        "external_terminal_audit_required": True,
    }, "security/gate contract changed")
    core = protocol.get("scientific_core", {})
    expected_names = (
        "src/accounting.py", "src/ase_neb.py", "src/full_batch_recorder.py",
        "src/integrity.py", "src/kgrid_execution_authorization.py",
        "src/kgrid_compact_trace.py", "src/kgrid_formal_spec.py",
        "src/kgrid_metric_evidence.py",
        "src/kgrid_preflight.py", "src/kgrid_runner.py",
        "src/kgrid_selection.py", "src/kgrid_spec.py",
        "src/kgrid_terminal_audit.py", "src/oracle.py", "src/resource_guard.py",
    )
    _require(set(core) == set(expected_names), "scientific-core file set changed")
    for relative in expected_names:
        _require(core[relative] == sha256_file(PROTOCOL_PATH.parent / relative),
                 f"scientific-core file hash changed: {relative}")
    runtime = protocol.get("runtime", {})
    _require(runtime == {
        "ase_version": ASE_VERSION,
        "ase_neb_runtime_provenance": "pre-existing locally modified ASE 3.23.0 neb.py",
        "ase_mep_neb_py_sha256": ASE_NEB_SHA256,
        "ase_optimize_ode_py_sha256": ASE_ODE_SHA256,
        "openmm_version": "8.1.2",
        "platform": "CPU",
        "precision": "double",
    }, "runtime identity contract changed")
    _require(protocol.get("preregistration_sha256")
             == sha256_file(PREREGISTRATION_PATH),
             "preregistration hash binding changed")
    _require(protocol.get("selection_protocol_sha256")
             == sha256_file(SELECTION_PROTOCOL_PATH),
             "selection protocol hash binding changed")
    return protocol


def verify_inputs_and_endpoint_cache(protocol: dict[str, Any]) -> dict[str, Any]:
    _require(sha256_file(INITIAL_PATH) == INITIAL_FILE_SHA256, "initial path file changed")
    _require(sha256_file(PDB_PATH) == PDB_SHA256, "fixed PDB changed")
    _require(sha256_file(ENDPOINT_CACHE_PATH) == ENDPOINT_CACHE_SHA256,
             "endpoint joint E/F cache changed")
    with np.load(INITIAL_PATH, allow_pickle=False) as archive:
        _require(set(archive.files) == {"initial_path"}, "initial path NPZ keys changed")
        nodes = np.ascontiguousarray(archive["initial_path"], dtype="<f8")
    _require(nodes.shape == (32, ATOM_COORDINATES) and np.isfinite(nodes).all(),
             "initial path shape/finiteness changed")
    _require(_raw_array_sha256(nodes) == INITIAL_ARRAY_SHA256,
             "initial path raw-array SHA changed")
    _require(_raw_array_sha256(nodes[[0, -1]]) == ENDPOINT_ARRAY_SHA256,
             "initial endpoint raw-array SHA changed")
    endpoint_coordinates, endpoint_energies, endpoint_forces = _load_endpoint_cache(
        ENDPOINT_CACHE_PATH
    )
    _require(np.array_equal(endpoint_coordinates, nodes[[0, -1]]),
             "endpoint cache coordinates differ from the common initial endpoints")
    bindings = protocol.get("input_bindings", {})
    expected = {
        "initial_path_file_sha256": INITIAL_FILE_SHA256,
        "initial_path_raw_array_sha256": INITIAL_ARRAY_SHA256,
        "endpoint_raw_array_sha256": ENDPOINT_ARRAY_SHA256,
        "complex_fixed_pdb_sha256": PDB_SHA256,
        "endpoint_joint_ef_cache_sha256": ENDPOINT_CACHE_SHA256,
        "canonical_openmm_contract_sha256": PHYSICAL_CONTRACT_SHA256,
    }
    _require(bindings == expected, "protocol input bindings changed")
    return {
        "nodes": nodes,
        "initial_path_file_sha256": INITIAL_FILE_SHA256,
        "initial_path_array_sha256": array_sha256(nodes),
        "endpoint_array_sha256": array_sha256(nodes[[0, -1]]),
        "pdb_sha256": PDB_SHA256,
        "endpoint_cache_sha256": ENDPOINT_CACHE_SHA256,
        "endpoint_coordinates": endpoint_coordinates,
        "endpoint_energies": endpoint_energies,
        "endpoint_forces": endpoint_forces,
    }


def verify_physical_contract(protocol: dict[str, Any]) -> dict[str, Any]:
    del protocol
    _require(sha256_file(CONTRACT_PATH) == PHYSICAL_CONTRACT_SHA256,
             "canonical OpenMM authority file changed")
    try:
        contract = load_json_object(CONTRACT_PATH)
    except IntegrityError as error:
        raise KGridFormalSpecError(str(error)) from error
    identity = contract.get("identity", {})
    _require(canonical_json_sha256(identity) == PHYSICAL_IDENTITY_SHA256,
             "canonical OpenMM physical identity changed")
    system = identity.get("system", {})
    _require(identity.get("openmm_version") == "8.1.2"
             and identity.get("pdb", {}).get("file_sha256") == PDB_SHA256
             and identity.get("pdb", {}).get("topology", {}).get("atoms") == 3132
             and system.get("serialized_xml_raw_sha256") == SYSTEM_XML_SHA256
             and system.get("serialized_xml_raw_bytes") == 3918711
             and identity.get("joint_oracle_semantics", {}).get("cache") is False
             and identity.get("joint_oracle_semantics", {}).get(
                 "joint_getState_calls_per_attempt") == 1,
             "canonical physical oracle is not the frozen BB joint E/full-force system")
    return {
        "authority_file_sha256": PHYSICAL_CONTRACT_SHA256,
        "identity_sha256": PHYSICAL_IDENTITY_SHA256,
        "system_xml_sha256": SYSTEM_XML_SHA256,
        "system_xml_bytes": 3918711,
        "force_class_sequence": [record["class"] for record in system["forces"]],
        "openmm_version": "8.1.2",
        "atom_count": 3132,
    }
