"""Per-run O_EXCL authorization for the fresh BB ASE-ODE k grid."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import socket
import stat
import subprocess
import sys
from typing import Any, Mapping

from .integrity import (
    SOURCE_ROOT,
    SourceIdentity,
    is_within,
    load_json_object,
    sha256_file,
    verify_source_identity,
    write_json_new,
)
from .kgrid_formal_spec import (
    load_and_verify_protocol,
    verify_inputs_and_endpoint_cache,
    verify_physical_contract,
)
from .kgrid_spec import (
    CANONICAL_LOCK_PATHS,
    ENDPOINT_CACHE_PATH,
    INITIAL_PATH,
    K_VALUES,
    PDB_PATH,
    PREREGISTRATION_PATH,
    PROTOCOL_PATH,
    SELECTION_PROTOCOL_PATH,
    build_run,
)


AUTHORIZATION_SCHEMA = "bb-neb-aseode-kgrid-execution-authorization-v1"
SOURCE_AUDIT_SCHEMA = "bb-neb-aseode-kgrid-independent-source-audit-v1"
MUTAGEN_GATE_SCHEMA = "bb-neb-aseode-kgrid-mutagen-gate-v1"
SELECTION_SCHEMA = "bb-ase-ode-kgrid-selection-v1"
SELECTION_STATUS = "SELECTION_COMPLETE"
# The four NF6000 screening runs were completed and independently audited
# before the formal budget was extended from NF15000 to NF30000.  Repeating
# them would change no selection input and would spend 24000 redundant force
# evaluations.  Formal authorization may therefore adopt only this exact
# earlier receipt, after rerunning its complete portable-trace verification
# with the exact frozen source that created those trace audits.
PRIOR_SELECTION_SOURCE_ROOT = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
    "bb_neb_aseode30_kgrid_nf6000_nf15000_20260903_r1"
)
PRIOR_SELECTION_RECEIPT_PATH = (
    PRIOR_SELECTION_SOURCE_ROOT.parent /
    "bb_neb_aseode_kgrid_campaign_nf6000_nf15000_20260903_r1" /
    "selection" / "BB_ASEODE_KGRID_SELECTION_NF6000_20260903_R1.json"
)
PRIOR_SELECTION_SOURCE_IDENTITY = {
    "source_manifest_sha256":
        "dfd404f08d39ac283e63a1f9c9b5d8214b4d99ad2a10fc32719248998121942b",
    "whole_source_closure_sha256":
        "52f7981bbdd7a6c7ec06bb5d472356f3a17448b56ac4529999fda9d2ad8bd8b2",
    "scientific_core_closure_sha256":
        "8ec4172bbaeb12cd860506514c06e2f7bf02598c77aebd74597bd80051025a07",
    "file_count": 44,
}
PRIOR_SELECTION_RECEIPT_SHA256 = (
    "4da93e5f277acfa3e9f0dc3a170968abd081dba9689bac4cb60eb1edd60551fc"
)
PRIOR_SELECTION_PROTOCOL_SHA256 = (
    "bd77d6320ab40b6be443605cf28f2be8ef55935c08589ca9e6cba87563468cd1"
)
PROJECT_ROOT = SOURCE_ROOT.parents[1]
FORMAL_ENTRYPOINT = SOURCE_ROOT / "scripts" / "run_kgrid_openmm_production.py"
THREAD_ENVIRONMENT = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OPENMM_CPU_THREADS",
)


class KGridAuthorizationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KGridAuthorizationError(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _readonly_regular(path: Path, label: str) -> None:
    status = path.lstat()
    _require(stat.S_ISREG(status.st_mode) and not path.is_symlink(),
             f"{label} is not a regular nonsymlink file")
    _require(stat.S_IMODE(status.st_mode) & 0o222 == 0, f"{label} must be read-only")


def _external_receipt(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    receipt_path = Path(path).resolve(strict=True)
    _require(not is_within(receipt_path, SOURCE_ROOT), f"{label} cannot be inside source")
    _readonly_regular(receipt_path, label)
    return receipt_path, load_json_object(receipt_path, require_canonical=True)


def _verify_source_audit(path: Path | str, source: SourceIdentity) -> dict[str, Any]:
    receipt_path, receipt = _external_receipt(path, label="independent source audit")
    _require(receipt.get("schema") == SOURCE_AUDIT_SCHEMA
             and receipt.get("status") == "verified-go"
             and receipt.get("physical_force_calls") == 0
             and Path(str(receipt.get("source_root", ""))).resolve() == SOURCE_ROOT
             and receipt.get("source_identity") == source.as_dict(),
             "independent source audit does not authorize this exact source")
    return {"path": str(receipt_path), "sha256": sha256_file(receipt_path),
            "schema": SOURCE_AUDIT_SCHEMA, "status": "verified-go"}


def _verify_mutagen_gate(
    path: Path | str, *, source: SourceIdentity, machine_id: str
) -> dict[str, Any]:
    receipt_path, receipt = _external_receipt(path, label="Mutagen gate receipt")
    _require(receipt.get("schema") == MUTAGEN_GATE_SCHEMA
             and receipt.get("status") == "synchronized-connected"
             and receipt.get("mutagen_status") == "Watching,true,true"
             and receipt.get("physical_force_calls") == 0
             and receipt.get("source_identity") == source.as_dict()
             and receipt.get("target_machine_id") == machine_id,
             "Mutagen receipt does not bind synchronized source and target machine")
    return {"path": str(receipt_path), "sha256": sha256_file(receipt_path),
            "schema": MUTAGEN_GATE_SCHEMA, "status": "synchronized-connected",
            "mutagen_status": "Watching,true,true"}


def _recompute_prior_selection(receipt: dict[str, Any]) -> dict[str, Any]:
    """Reverify the prior four traces inside their own frozen source identity."""

    prior_manifest = PRIOR_SELECTION_SOURCE_ROOT / "SOURCE_MANIFEST.json"
    prior_protocol = PRIOR_SELECTION_SOURCE_ROOT / "KGRID_SELECTION_PROTOCOL.json"
    _readonly_regular(prior_manifest, "prior selection source manifest")
    _require(sha256_file(prior_manifest)
             == PRIOR_SELECTION_SOURCE_IDENTITY["source_manifest_sha256"],
             "prior selection source manifest hash changed")
    prior_protocol_status = prior_protocol.lstat()
    _require(stat.S_ISREG(prior_protocol_status.st_mode)
             and not prior_protocol.is_symlink(),
             "prior selection protocol is not a regular nonsymlink file")
    _require(sha256_file(prior_protocol) == PRIOR_SELECTION_PROTOCOL_SHA256,
             "prior selection protocol hash changed")
    _require(verify_source_identity(PRIOR_SELECTION_SOURCE_ROOT).as_dict()
             == PRIOR_SELECTION_SOURCE_IDENTITY,
             "prior selection source identity changed before recomputation")
    helper = (
        "import json,sys; "
        "sys.dont_write_bytecode=True; sys.path.insert(0,sys.argv[1]); "
        "from src.integrity import verify_source_identity; "
        "from src.kgrid_compact_trace import select_from_compact_traces; "
        "receipt=json.load(open(sys.argv[2],encoding='utf-8')); "
        "identity=verify_source_identity().as_dict(); "
        "bindings=[dict(x['trace'],compact_audit=x['compact_audit']) "
        "for x in receipt['screening_traces']]; "
        "result=select_from_compact_traces(bindings,"
        "selection_protocol_path=sys.argv[3]); "
        "print(json.dumps({'identity':identity,'selection':result},"
        "sort_keys=True,separators=(',',':'),allow_nan=False))"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", helper,
         str(PRIOR_SELECTION_SOURCE_ROOT),
         str(PRIOR_SELECTION_RECEIPT_PATH),
         str(prior_protocol)],
        cwd=str(PRIOR_SELECTION_SOURCE_ROOT),
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    result = json.loads(completed.stdout)
    _require(result.get("identity") == PRIOR_SELECTION_SOURCE_IDENTITY,
             "prior selection source identity changed")
    _require(verify_source_identity(PRIOR_SELECTION_SOURCE_ROOT).as_dict()
             == PRIOR_SELECTION_SOURCE_IDENTITY,
             "prior selection source identity changed after recomputation")
    return result["selection"]


def _verify_selection_receipt(path: Path | str, *, selected_k: float) -> dict[str, Any]:
    receipt_path, receipt = _external_receipt(path, label="k-grid selection receipt")
    receipt_sha256 = sha256_file(receipt_path)
    _require(receipt.get("schema") == SELECTION_SCHEMA
             and receipt.get("status") == SELECTION_STATUS
             and receipt.get("screening_terminal_optimization_N_F") == 6000
             and float(receipt.get("selected_spring_k_eV_per_A2", -1)) == selected_k,
             "selection receipt schema/status/budget/selected k changed")
    current_protocol = {
        "path": str(SELECTION_PROTOCOL_PATH),
        "sha256": sha256_file(SELECTION_PROTOCOL_PATH),
    }
    prior_protocol = {
        "path": str(PRIOR_SELECTION_SOURCE_ROOT / "KGRID_SELECTION_PROTOCOL.json"),
        "sha256": PRIOR_SELECTION_PROTOCOL_SHA256,
    }
    is_prior_selection = (
        receipt_sha256 == PRIOR_SELECTION_RECEIPT_SHA256
        and receipt_path == PRIOR_SELECTION_RECEIPT_PATH
        and receipt.get("selection_protocol") == prior_protocol
        and current_protocol["sha256"] == PRIOR_SELECTION_PROTOCOL_SHA256
    )
    _require(is_prior_selection,
             "formal selection is not the exact adopted prior receipt")
    traces = receipt.get("screening_traces")
    _require(isinstance(traces, list) and len(traces) == 4,
             "selection receipt must bind exactly four screening traces")
    observed_k = set()
    normalized = []
    for record in traces:
        _require(isinstance(record, dict) and set(record) == {
            "spring_k_eV_per_A2", "candidate", "authorization",
            "terminal_audit", "trace", "compact_audit", "origin_machine_id",
            "terminal_tree_closure_sha256",
        }, "selection trace binding fields are not exact")
        spring_k = float(record.get("spring_k_eV_per_A2", -1))
        observed_k.add(spring_k)
        normalized_record = dict(record)
        for key in ("trace", "compact_audit"):
            binding = record.get(key)
            _require(isinstance(binding, dict) and set(binding) == {"path", "sha256"},
                     f"selection {key} binding is incomplete")
            bound_path = Path(str(binding["path"])).resolve(strict=True)
            _readonly_regular(bound_path, f"selection {key}")
            _require(sha256_file(bound_path) == binding["sha256"],
                     f"selection {key} hash changed")
            normalized_record[key] = {"path": str(bound_path), "sha256": binding["sha256"]}
        terminal_binding = record.get("terminal_audit")
        _require(isinstance(terminal_binding, dict)
                 and set(terminal_binding) == {"path", "sha256"}
                 and isinstance(terminal_binding["path"], str)
                 and len(terminal_binding["path"]) > 0
                 and Path(terminal_binding["path"]).is_absolute()
                 and isinstance(terminal_binding["sha256"], str)
                 and len(terminal_binding["sha256"]) == 64,
                 "selection origin terminal-audit binding is malformed")
        authorization_binding = record.get("authorization")
        _require(isinstance(authorization_binding, dict)
                 and set(authorization_binding) == {"path", "sha256"}
                 and isinstance(authorization_binding["path"], str)
                 and bool(authorization_binding["path"])
                 and Path(authorization_binding["path"]).is_absolute()
                 and isinstance(authorization_binding["sha256"], str)
                 and len(authorization_binding["sha256"]) == 64,
                 "selection origin authorization binding is malformed")
        candidate_binding = record.get("candidate")
        _require(isinstance(candidate_binding, Mapping),
                 "selection origin candidate binding is absent")
        origin_candidate = build_run(
            spring_k=float(candidate_binding.get("spring_k_eV_per_A2", -1)),
            phase=str(candidate_binding.get("phase", "")),
            run_label=str(candidate_binding.get("run_label", "")),
        )
        _require(candidate_binding == origin_candidate.as_dict()
                 and origin_candidate.phase == "screening"
                 and origin_candidate.spring_k == spring_k
                 and origin_candidate.batches == 200
                 and origin_candidate.terminal_nf == 6000,
                 "selection origin candidate is not an exact screening run")
        _require(isinstance(record.get("origin_machine_id"), str)
                 and bool(record["origin_machine_id"])
                 and isinstance(record.get("terminal_tree_closure_sha256"), str)
                 and len(record["terminal_tree_closure_sha256"]) == 64,
                 "selection origin machine/tree binding is malformed")
        normalized.append(normalized_record)
    _require(observed_k == set(K_VALUES), "selection receipt does not bind the exact four-k grid")
    _require(receipt.get("screening_traces") == normalized,
             "selection receipt trace paths are not canonical")
    # Reverify every origin-attested portable trace and rerun the sole frozen
    # four-way decision helper.  A self-consistent receipt alone is not enough.
    recomputed = _recompute_prior_selection(receipt)
    _require(receipt == recomputed,
             "selection receipt differs from full-history compact-trace recomputation")
    return {"path": str(receipt_path), "sha256": receipt_sha256,
            "schema": SELECTION_SCHEMA, "status": SELECTION_STATUS,
            "selected_spring_k_eV_per_A2": selected_k,
            "verification_mode": "exact-prior-source-full-portable-recomputation",
            "selection_source_identity": PRIOR_SELECTION_SOURCE_IDENTITY}


def _label_absent_everywhere(label: str) -> None:
    code_root = PROJECT_ROOT / "code_on_server"
    _require(code_root.is_dir(), "fixed code_on_server root is absent")
    for directory, directory_names, _ in os.walk(code_root):
        if label in directory_names:
            raise KGridAuthorizationError(
                f"fresh label was already used and is permanently nonreusable: "
                f"{Path(directory) / label}"
            )


def build_authorization_document(
    *,
    campaign_root: Path | str,
    machine_id: str,
    spring_k: float,
    phase: str,
    run_label: str,
    independent_source_audit: Path | str,
    mutagen_gate_receipt: Path | str,
    selection_receipt: Path | str | None = None,
) -> dict[str, Any]:
    candidate = build_run(spring_k=spring_k, phase=phase, run_label=run_label)
    _require(isinstance(machine_id, str) and bool(machine_id), "machine_id is empty")
    protocol = load_and_verify_protocol(require_frozen=True)
    source = verify_source_identity()
    inputs = verify_inputs_and_endpoint_cache(protocol)
    physical = verify_physical_contract(protocol)
    source_audit = _verify_source_audit(independent_source_audit, source)
    mutagen = _verify_mutagen_gate(
        mutagen_gate_receipt, source=source, machine_id=machine_id
    )
    if candidate.phase == "screening":
        _require(selection_receipt is None,
                 "screening authorization cannot consume outcome-dependent selection")
        selection = None
    else:
        _require(selection_receipt is not None,
                 "formal NF30000 authorization requires frozen four-trace selection")
        selection = _verify_selection_receipt(
            selection_receipt, selected_k=candidate.spring_k
        )

    campaign = Path(campaign_root).resolve()
    _require(campaign != SOURCE_ROOT and not is_within(campaign, SOURCE_ROOT),
             "campaign root must be separate from immutable source")
    _require(campaign.parent == PROJECT_ROOT / "code_on_server",
             "campaign root must be top-level under code_on_server")
    _label_absent_everywhere(candidate.run_label)
    run_directory = campaign / "runs" / candidate.run_label
    _require(not run_directory.exists(), "fresh output directory already exists")
    authorization_path = (
        campaign / "authorizations" / f"{candidate.run_label}.execution-authorization.json"
    )
    _require(not authorization_path.exists(), "execution authorization already exists")
    return {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "authorized",
        "execution_authorized": True,
        "created_utc": _utc_now(),
        "candidate": candidate.as_dict(),
        "source_root": str(SOURCE_ROOT),
        "source_identity": source.as_dict(),
        "independent_source_audit": source_audit,
        "mutagen_gate": mutagen,
        "preregistration": {
            "path": str(PREREGISTRATION_PATH),
            "sha256": sha256_file(PREREGISTRATION_PATH),
        },
        "selection_protocol": {
            "path": str(SELECTION_PROTOCOL_PATH),
            "sha256": sha256_file(SELECTION_PROTOCOL_PATH),
        },
        "selection_receipt": selection,
        "protocol": {"path": str(PROTOCOL_PATH), "sha256": sha256_file(PROTOCOL_PATH)},
        "inputs": {
            "initial_path": {"path": str(INITIAL_PATH),
                             "file_sha256": inputs["initial_path_file_sha256"],
                             "array_sha256": inputs["initial_path_array_sha256"],
                             "endpoint_array_sha256": inputs["endpoint_array_sha256"]},
            "pdb": {"path": str(PDB_PATH), "sha256": inputs["pdb_sha256"]},
            "endpoint_joint_ef_cache": {
                "path": str(ENDPOINT_CACHE_PATH),
                "sha256": inputs["endpoint_cache_sha256"],
                "endpoint_optimization_N_F": 0,
                "additional_OpenMM_calls": 0,
            },
        },
        "physical_oracle": {"authority_path": str(PROTOCOL_PATH.parent / "contracts" /
                                                   "CANONICAL_OPENMM_CONTRACT.json"),
                            **physical},
        "paths": {
            "campaign_root": str(campaign),
            "output_parent": str(campaign / "runs"),
            "run_directory": str(run_directory),
            "authorization_file": str(authorization_path),
            "host_local_BB_lock_paths": [str(path) for path in CANONICAL_LOCK_PATHS],
        },
        "target_machine": {
            "machine_id": machine_id,
            "hostname": socket.gethostname(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "python_executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
        },
        "execution_policy": {
            "mode": "fresh-nonresume",
            "phase": candidate.phase,
            "spring_k_eV_per_A2": candidate.spring_k,
            "paid_optimization_force_evaluation_cap": candidate.terminal_nf,
            "paid_batches": candidate.batches,
            "paid_interior_calls_per_batch": 30,
            "endpoint_evaluation_budget": 0,
            "dense_evaluation_budget": 0,
            "single_official_ode12r_invocation": True,
            "all_paid_calls_full_state_archived": True,
            "jobs": 1,
        },
        "thread_environment": {name: "1" for name in THREAD_ENVIRONMENT},
        "formal_entrypoint": {"path": str(FORMAL_ENTRYPOINT),
                              "sha256": sha256_file(FORMAL_ENTRYPOINT),
                              "api": "src.kgrid_runner.run_from_authorization"},
        "authorization_file": {"path": str(authorization_path),
                               "creation": "O_EXCL+fsync", "mode": "0444"},
        "preauthorization_physical_force_calls": 0,
        "preauthorization_dense_calls": 0,
    }


def write_execution_authorization_new(**kwargs: Any) -> dict[str, Any]:
    document = build_authorization_document(**kwargs)
    path = Path(document["authorization_file"]["path"])
    digest = write_json_new(path, document, readonly=True)
    return {"path": str(path), "sha256": digest, "document": document}


def verify_execution_authorization(
    path: Path | str, *, require_fresh_run: bool = True
) -> dict[str, Any]:
    authorization_path = Path(path).resolve(strict=True)
    _readonly_regular(authorization_path, "execution authorization")
    observed = load_json_object(authorization_path, require_canonical=True)
    _require(observed.get("schema") == AUTHORIZATION_SCHEMA
             and observed.get("status") == "authorized"
             and observed.get("execution_authorized") is True,
             "execution authorization is not enabled")
    candidate_record = observed.get("candidate", {})
    candidate = build_run(
        spring_k=candidate_record.get("spring_k_eV_per_A2"),
        phase=str(candidate_record.get("phase", "")),
        run_label=str(candidate_record.get("run_label", "")),
    )
    _require(candidate_record == candidate.as_dict(), "authorization candidate changed")
    protocol = load_and_verify_protocol(require_frozen=True)
    source = verify_source_identity()
    inputs = verify_inputs_and_endpoint_cache(protocol)
    physical = verify_physical_contract(protocol)
    _require(observed.get("source_root") == str(SOURCE_ROOT)
             and observed.get("source_identity") == source.as_dict(),
             "authorization source identity changed")
    expected_source_audit = _verify_source_audit(
        observed.get("independent_source_audit", {}).get("path", ""), source
    )
    _require(observed.get("independent_source_audit") == expected_source_audit,
             "authorization source-audit binding changed")
    machine_id = str(observed.get("target_machine", {}).get("machine_id", ""))
    expected_mutagen = _verify_mutagen_gate(
        observed.get("mutagen_gate", {}).get("path", ""),
        source=source,
        machine_id=machine_id,
    )
    _require(observed.get("mutagen_gate") == expected_mutagen,
             "authorization Mutagen binding changed")
    expected_selection = None
    if candidate.phase == "formal":
        expected_selection = _verify_selection_receipt(
            observed.get("selection_receipt", {}).get("path", ""),
            selected_k=candidate.spring_k,
        )
    _require(observed.get("selection_receipt") == expected_selection,
             "authorization selection binding changed")
    _require(observed.get("preregistration") == {
        "path": str(PREREGISTRATION_PATH), "sha256": sha256_file(PREREGISTRATION_PATH)},
        "authorization preregistration binding changed")
    _require(observed.get("selection_protocol") == {
        "path": str(SELECTION_PROTOCOL_PATH), "sha256": sha256_file(SELECTION_PROTOCOL_PATH)},
        "authorization selection-protocol binding changed")
    _require(observed.get("protocol") == {
        "path": str(PROTOCOL_PATH), "sha256": sha256_file(PROTOCOL_PATH)},
        "authorization protocol binding changed")
    expected_inputs = {
        "initial_path": {"path": str(INITIAL_PATH),
                         "file_sha256": inputs["initial_path_file_sha256"],
                         "array_sha256": inputs["initial_path_array_sha256"],
                         "endpoint_array_sha256": inputs["endpoint_array_sha256"]},
        "pdb": {"path": str(PDB_PATH), "sha256": inputs["pdb_sha256"]},
        "endpoint_joint_ef_cache": {"path": str(ENDPOINT_CACHE_PATH),
                                    "sha256": inputs["endpoint_cache_sha256"],
                                    "endpoint_optimization_N_F": 0,
                                    "additional_OpenMM_calls": 0},
    }
    _require(observed.get("inputs") == expected_inputs, "authorization input binding changed")
    expected_physical = {"authority_path": str(PROTOCOL_PATH.parent / "contracts" /
                                               "CANONICAL_OPENMM_CONTRACT.json"), **physical}
    _require(observed.get("physical_oracle") == expected_physical,
             "authorization physical-oracle binding changed")
    paths = observed.get("paths", {})
    campaign = Path(str(paths.get("campaign_root", ""))).resolve()
    run = Path(str(paths.get("run_directory", ""))).resolve()
    _require(campaign.parent == PROJECT_ROOT / "code_on_server"
             and campaign != SOURCE_ROOT and not is_within(campaign, SOURCE_ROOT)
             and Path(str(paths.get("output_parent", ""))).resolve() == campaign / "runs"
             and run == campaign / "runs" / candidate.run_label
             and Path(str(paths.get("authorization_file", ""))).resolve()
             == authorization_path
             and paths.get("host_local_BB_lock_paths")
             == [str(item) for item in CANONICAL_LOCK_PATHS],
             "authorization output/dual-lock paths changed")
    if require_fresh_run:
        _require(not run.exists(), "fresh run directory already exists")
    else:
        _require(run.is_dir() and not run.is_symlink(), "authorized run directory is absent")
    target = observed.get("target_machine", {})
    _require(target.get("hostname") == socket.gethostname()
             and target.get("platform_system") == platform.system()
             and target.get("platform_machine") == platform.machine()
             and Path(str(target.get("python_executable", ""))).resolve()
             == Path(sys.executable).resolve()
             and target.get("python_version") == platform.python_version(),
             "authorization was not built for this exact runtime host")
    expected_policy = {
        "mode": "fresh-nonresume", "phase": candidate.phase,
        "spring_k_eV_per_A2": candidate.spring_k,
        "paid_optimization_force_evaluation_cap": candidate.terminal_nf,
        "paid_batches": candidate.batches, "paid_interior_calls_per_batch": 30,
        "endpoint_evaluation_budget": 0, "dense_evaluation_budget": 0,
        "single_official_ode12r_invocation": True,
        "all_paid_calls_full_state_archived": True, "jobs": 1,
    }
    _require(observed.get("execution_policy") == expected_policy,
             "authorization execution policy changed")
    _require(observed.get("thread_environment") == {name: "1" for name in THREAD_ENVIRONMENT},
             "authorization thread caps changed")
    _require(observed.get("formal_entrypoint") == {
        "path": str(FORMAL_ENTRYPOINT), "sha256": sha256_file(FORMAL_ENTRYPOINT),
        "api": "src.kgrid_runner.run_from_authorization"},
        "authorization production entrypoint changed")
    _require(observed.get("authorization_file") == {
        "path": str(authorization_path), "creation": "O_EXCL+fsync", "mode": "0444"},
        "authorization self-binding changed")
    _require(observed.get("preauthorization_physical_force_calls") == 0
             and observed.get("preauthorization_dense_calls") == 0,
             "authorization contains preauthorization physical work")
    return observed


def require_thread_environment() -> dict[str, str]:
    observed = {name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT}
    invalid = {key: value for key, value in observed.items() if value != "1"}
    _require(not invalid, f"all six thread caps must equal 1: {invalid}")
    return observed
