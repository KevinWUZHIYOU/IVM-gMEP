#!/usr/bin/env python3
"""Zero-physical R2 analysis for the BB ODE-NF30000 campaign.

This module hash-pins and reuses the completed NF15000 plotting layer.  The
R2 overrides are intentionally narrow:

* consume 1000 published ODE batches through optimization ``N_F=30000``;
* verify the adopted NF6000 selection through the new authorizer;
* keep IR-IVM call 318 / ``N_F=8982`` as the last formal IVM state; and
* expose call 515 / ``N_F=14983`` only as an explicitly uncommitted diagnostic.

No function in this file imports OpenMM, evaluates an energy, evaluates a
Force, acquires a BB compute lock, or densifies a path.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


sys.dont_write_bytecode = True

BASE = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
    "bb_neb_aseode_kgrid_campaign_nf6000_nf15000_20260903_r1/analysis/"
    "bb_discrete32_analysis.py"
)
BASE_SHA256 = "496573af9c086fcf5d875001260d1d71c8c64625779575499e64f4a5d05fc656"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if BASE.resolve(strict=True) != BASE or _file_sha256(BASE) != BASE_SHA256:
    raise RuntimeError("hash-pinned NF15000 analysis implementation changed")

SPEC = importlib.util.spec_from_file_location("bb_discrete32_nf15000_base", BASE)
BASE_MODULE = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError("cannot load hash-pinned NF15000 analysis implementation")
SPEC.loader.exec_module(BASE_MODULE)

MANIFEST_SCHEMA = "bb-discrete32-ode-nf30000-r2-analysis-manifest-v1"
PACK_SCHEMA = "bb-discrete32-ode-nf30000-r2-comparison-pack-v1"
FORMAL_BATCHES = 1000
FORMAL_NF = 30000
RETAINED_ODE_STATE_NF = (240, 2040, 15000, 30000)
IR_DIAGNOSTIC_KEY = "IVM returned terminal (uncommitted)"
CACHE_REPLICA_WRAPPER = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
    "bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/"
    "cache_formal_discrete32_incrementally_r2.py"
)
CACHE_REPLICA_WRAPPER_SHA256 = (
    "e77c92187a25dc2c11e776d4405b6edd0563e00e7cc1291369ae4f6400142464"
)
CACHE_REPLICA_WRAPPER_BYTES = 14754
CACHE_REPLICA_RECEIPT = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
    "bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/"
    "LOCAL_MUTAGEN_REPLICA_TRANSPORT_RECEIPT_R2.json"
)
CACHE_REPLICA_RECEIPT_SHA256 = (
    "7164b8b2593ae1d6a592e11da8b8d99b94578ab631b8527f9f752ec301a41213"
)
CACHE_REPLICA_RECEIPT_BYTES = 4863
CACHE_REPLICA_SESSION_ID = "sync_wluhxwR03mFsMVwykAr1wJZEhU2OBSLb9h1LhC0Dus0"
CACHE_REPLICA_RUN = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
    "bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/runs/"
    "fresh-nonresume-bb-neb-aseode-k0p1-formal-nf30000-remote-20260903-r1"
)
CACHE_REPLICA_PRESTART_SHA256 = (
    "1c5aa970ab806eccb785e41cb273c52bf5d38b087c834d5dda5fc09555522422"
)
CACHE_REPLICA_PRESTART_BYTES = 5648
SURFACE_RESULT_SCHEMA = "bb-irivm-call515-terminal-tail-section-energy-r2-v1-result"
SURFACE_AUDIT_SCHEMA = "bb-irivm-call515-terminal-tail-section-energy-r2-v1-audit"
SURFACE_WAL_SCHEMA = "bb-irivm-call515-terminal-tail-section-energy-r2-v1-wal"
SURFACE_STATE_ROLE = (
    "returned terminal state; terminal uncommitted charged tail; not committed IVM"
)
SURFACE_POINTS = 59
SURFACE_NORMAL_POINTS = 28
SURFACE_ZERO_INDEX = 4
SURFACE_STORED_VALUES = SURFACE_POINTS * SURFACE_NORMAL_POINTS
SURFACE_FRESH_VALUES = SURFACE_STORED_VALUES - 2
EVOLUTION_DISPLAY_CAP_KCAL_PER_MOL = 1.0e6
EVOLUTION_CLIP_REFERENCE = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
    "scripts/plot_bb_near_nf9990_energy_and_legacy_surface_evolution.py"
)
EVOLUTION_CLIP_REFERENCE_SHA256 = (
    "3483a56933f7283fb54fb852bf5d1f9c99e156ee8f0fa688ac0ce015b61ada61"
)
SURFACE_RESULT_FIELDS = frozenset({
    "schema", "status", "stage", "method",
    "center_source_optimization_N_F", "center_source_call_per_image",
    "coordinate_manifest", "coordinate_manifest_sha256",
    "energy_call_policy", "energy_call_policy_sha256",
    "endpoint_cache", "endpoint_cache_sha256",
    "endpoint_coordinate_pair_sha256", "endpoint_energy_pair_sha256",
    "contract", "contract_sha256", "authorization", "authorization_sha256",
    "preauthorization_preflight", "preauthorization_preflight_sha256",
    "evaluator_sha256", "bb_core_sha256", "topology_sha256", "energy_wal",
    "held_host_locks", "fresh_fixed_endpoint_energy_calls", "force_calls",
    "optimization_N_F_contribution", "returned_terminal_state",
    "formally_reportable", "terminal_uncommitted_charged_tail",
    "coordinate_audit", "coordinate_audit_sha256", "surface_coordinates_A",
    "surface_coordinate_sha256", "surface_energy_eV", "surface_energy_sha256",
    "center_path59_A", "center_s", "normal_directions",
    "common_initial_path59_A", "normal_coordinate_rms_A", "zero_normal_index",
    "fresh_evaluated_mask", "inherited_endpoint_mask", "energy_value_lineage",
    "stored_energy_values", "fresh_energy_only_attempts",
    "inherited_endpoint_energy_values",
})
IR_INTERPRETATION = {
    "last_formally_reportable_call": 318,
    "last_formally_reportable_optimization_N_F": 8982,
    "returned_terminal_call": 515,
    "returned_terminal_optimization_N_F": 14983,
    "returned_terminal_formally_reportable": False,
    "returned_terminal_state_role": "terminal_uncommitted_charged_tail",
}
FIGURE_CONTRACT_R2 = {
    "cost_to_precision": {
        "formal_methods": ["IVM", "ODE-NEB"],
        "axes": [
            {"metric": "I_perp", "xscale": "log", "xlim": [1.0e8, 1.0e1],
             "yscale": "linear", "ylim": [0.0, 30000.0]},
            {"metric": "Delta E_32", "xscale": "linear", "xlim": [5000.0, 0.0],
             "yscale": "linear", "ylim": [0.0, 30000.0]},
        ],
        "ylabel": "Number of Forces",
    },
    "energy_vs_arc_length": {
        "budget_N_F": 30000,
        "line_labels": [
            "IVM (NF 8982)",
            "ODE-NEB (NF 30000)",
            "Previous IVM (NF 9990)",
            "IVM trial (NF 14983; uncommitted)",
        ],
        "points_per_line": 32,
        "normalized_arc_length_endpoints": [0.0, 1.0],
        "normalized_arc_length_strictly_increasing": True,
    },
}

_BASE_BUILD_PACK = BASE_MODULE.build_pack
_BASE_PUBLISH_ARTIFACT_SET_NEW = BASE_MODULE.publish_artifact_set_new
_BASE_LOAD_IR = BASE_MODULE.load_ir_ivm
_BASE_LOAD_OLD_IVM = BASE_MODULE.load_old_ivm
_BASE_LOAD_SURFACE = BASE_MODULE.load_surface
_BASE_PLOT_COST = BASE_MODULE.plot_cost
_BASE_PLOT_ENERGY = BASE_MODULE.plot_energy_profiles
_BASE_PLOT_EVOLUTION = BASE_MODULE.plot_evolution
_BASE_READINESS = BASE_MODULE.readiness


def expected_cache_replica_transport() -> dict[str, Any]:
    """Return and locally revalidate the exact sealed cache-transport binding."""

    wrapper = CACHE_REPLICA_WRAPPER.resolve(strict=True)
    receipt = CACHE_REPLICA_RECEIPT.resolve(strict=True)
    BASE_MODULE.require(
        _file_sha256(wrapper) == CACHE_REPLICA_WRAPPER_SHA256
        and wrapper.stat().st_size == CACHE_REPLICA_WRAPPER_BYTES
        and wrapper.stat().st_mode & 0o222 == 0
        and _file_sha256(receipt) == CACHE_REPLICA_RECEIPT_SHA256
        and receipt.stat().st_size == CACHE_REPLICA_RECEIPT_BYTES
        and receipt.stat().st_mode & 0o222 == 0,
        "sealed local-replica cache wrapper/receipt binding differs",
    )
    return {
        "schema": "bb-nf30000-local-mutagen-replica-cache-binding-v1",
        "status": "PASS_EXACT_REPLICA_ADAPTER_BOUND",
        "wrapper": {
            "path": str(wrapper),
            "sha256": CACHE_REPLICA_WRAPPER_SHA256,
            "bytes": CACHE_REPLICA_WRAPPER_BYTES,
        },
        "transport_receipt": {
            "path": str(receipt),
            "sha256": CACHE_REPLICA_RECEIPT_SHA256,
            "bytes": CACHE_REPLICA_RECEIPT_BYTES,
        },
        "mutagen_session_identifier": CACHE_REPLICA_SESSION_ID,
        "exact_run_dir": str(CACHE_REPLICA_RUN),
        "prestart": {
            "path": str(CACHE_REPLICA_RUN / "PRESTART.json"),
            "sha256": CACHE_REPLICA_PRESTART_SHA256,
            "bytes": CACHE_REPLICA_PRESTART_BYTES,
            "local_mode_octal": "0600",
            "remote_receipt_mode_octal": "0444",
        },
        "preaudit_diagnostic_cache_reused": False,
        "physical_calls": 0,
    }


def resolve_replica_terminal_audit_file(
        value: Any, expected_sha256: str | None = None) -> Path:
    """Resolve only the terminal audit allowed by the sealed replica policy."""

    expected = (
        CACHE_REPLICA_RUN.parent.parent / "terminal_audits" /
        f"{CACHE_REPLICA_RUN.name}.terminal-audit.json"
    )
    candidate = Path(str(value)).expanduser()
    BASE_MODULE.require(
        candidate.is_absolute() and candidate == expected,
        "terminal audit is outside the exact Mutagen replica target",
    )
    path = candidate.resolve(strict=True)
    expected_cache_replica_transport()
    spec = importlib.util.spec_from_file_location(
        "bb_nf30000_terminal_replica_policy", CACHE_REPLICA_WRAPPER
    )
    adapter = importlib.util.module_from_spec(spec)
    BASE_MODULE.require(spec.loader is not None, "cannot load replica policy")
    spec.loader.exec_module(adapter)
    adapter.mutagen_transport_readonly_regular(path, "canonical terminal audit")
    if expected_sha256 is not None:
        BASE_MODULE.require(
            _file_sha256(path) == expected_sha256,
            "canonical terminal-audit SHA-256 differs",
        )
    BASE_MODULE.require(
        "openmm" not in sys.modules and "simtk.openmm" not in sys.modules,
        "terminal-audit replica resolver imported OpenMM",
    )
    return path


def load_adopted_selection(
    manifest: dict[str, Any], authorization_module: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Recompute the exact prior selection through the NF30000 gate."""

    path = BASE_MODULE.resolve_binding(
        manifest["inputs"].get("selection_receipt"), "adopted k selection receipt"
    )
    receipt = BASE_MODULE.load_json(path)
    selected_k = float(receipt.get("selected_spring_k_eV_per_A2", -1.0))
    BASE_MODULE.require(selected_k in BASE_MODULE.K_VALUES, "selected k is outside the grid")
    evidence = authorization_module._verify_selection_receipt(
        path, selected_k=selected_k
    )
    BASE_MODULE.require(
        evidence.get("verification_mode")
        == "exact-prior-source-full-portable-recomputation"
        and evidence.get("selection_source_identity")
        == authorization_module.PRIOR_SELECTION_SOURCE_IDENTITY,
        "selection was not recomputed from the exact frozen prior source",
    )
    configured = manifest["inputs"].get("screening_traces")
    BASE_MODULE.require(
        isinstance(configured, list) and len(configured) == 4,
        "analysis manifest lacks the four adopted screening traces",
    )
    configured_by_k = {float(row["spring_k_eV_per_A2"]): row for row in configured}
    receipt_by_k = {
        float(row["spring_k_eV_per_A2"]): row
        for row in receipt.get("screening_traces", [])
    }
    BASE_MODULE.require(
        set(configured_by_k) == set(BASE_MODULE.K_VALUES)
        and set(receipt_by_k) == set(BASE_MODULE.K_VALUES),
        "adopted screening grid differs",
    )
    for spring_k in BASE_MODULE.K_VALUES:
        configured_row = configured_by_k[spring_k]
        receipt_row = receipt_by_k[spring_k]
        BASE_MODULE.require(
            configured_row.get("trace") == receipt_row.get("trace")
            and configured_row.get("compact_audit") == receipt_row.get("compact_audit"),
            f"configured adopted trace differs for k={spring_k:g}",
        )
    return receipt, evidence


def _load_uncommitted_ir_terminal(
    verification_path: Path,
    verification: Mapping[str, Any],
    item: Mapping[str, Any],
) -> dict[str, Any]:
    inventory = verification.get("path_inventory")
    BASE_MODULE.require(isinstance(inventory, list) and inventory, "IR inventory is empty")
    reportable = [row for row in inventory if row.get("formally_reportable") is True]
    BASE_MODULE.require(reportable, "IR inventory has no reportable state")
    last_reportable = reportable[-1]
    state_binding_path = BASE_MODULE.resolve_binding(
        item.get("state_binding"), "IR call318/call515 state binding"
    )
    state_binding = BASE_MODULE.load_json(state_binding_path)
    BASE_MODULE.require(
        state_binding.get("schema")
        == "bb-irivm-r2-formal-and-diagnostic-state-binding-v1"
        and state_binding.get("status") == "PASS_ZERO_PHYSICAL_INPUT_BINDING"
        and state_binding.get("physical_force_calls") == 0
        and state_binding.get("verification")
        == {
            "path": str(verification_path),
            "sha256": BASE_MODULE.sha256_file(verification_path),
        },
        "IR call318/call515 state binding is not valid",
    )
    expected = item.get("state_interpretation")
    BASE_MODULE.require(expected == IR_INTERPRETATION, "IR state interpretation differs")
    BASE_MODULE.require(
        int(verification.get("last_complete_call", -1))
        == IR_INTERPRETATION["last_formally_reportable_call"]
        and int(verification.get("actual_canonical_call_count", -1))
        == IR_INTERPRETATION["returned_terminal_call"]
        and int(verification.get("terminal_optimization_N_F", -1))
        == IR_INTERPRETATION["returned_terminal_optimization_N_F"]
        and int(last_reportable.get("call_per_image", -1))
        == IR_INTERPRETATION["last_formally_reportable_call"]
        and int(last_reportable.get("optimization_N_F", -1))
        == IR_INTERPRETATION["last_formally_reportable_optimization_N_F"],
        "IR verification does not preserve call318/NF8982 as the formal terminal",
    )
    formal_binding = state_binding.get("formal_terminal", {})
    BASE_MODULE.require(
        formal_binding.get("call_per_image")
        == IR_INTERPRETATION["last_formally_reportable_call"]
        and formal_binding.get("optimization_N_F")
        == IR_INTERPRETATION["last_formally_reportable_optimization_N_F"]
        and formal_binding.get("formally_reportable") is True
        and formal_binding.get("path_sha256") == last_reportable.get("path_sha256")
        and formal_binding.get("energy_sha256") == last_reportable.get("energy_sha256")
        and formal_binding.get("force_sha256") == last_reportable.get("force_sha256")
        and formal_binding.get("shard")
        == {
            "path": last_reportable.get("shard"),
            "sha256": last_reportable.get("shard_sha256"),
        },
        "formal call318 binding differs from the verified inventory",
    )
    reportable_index = inventory.index(last_reportable)
    tail = inventory[reportable_index + 1 :]
    BASE_MODULE.require(
        tail
        and all(
            row.get("formally_reportable") is False
            and row.get("terminal_uncommitted_charged_tail") is True
            for row in tail
        ),
        "IR post-call318 history is not an entirely uncommitted charged tail",
    )
    terminal = tail[-1]
    BASE_MODULE.require(
        int(terminal.get("call_per_image", -1))
        == IR_INTERPRETATION["returned_terminal_call"]
        and int(terminal.get("optimization_N_F", -1))
        == IR_INTERPRETATION["returned_terminal_optimization_N_F"],
        "IR returned-terminal row is not call515/NF14983",
    )
    diagnostic_binding = state_binding.get("diagnostic_returned_terminal", {})
    BASE_MODULE.require(
        diagnostic_binding.get("call_per_image")
        == IR_INTERPRETATION["returned_terminal_call"]
        and diagnostic_binding.get("optimization_N_F")
        == IR_INTERPRETATION["returned_terminal_optimization_N_F"]
        and diagnostic_binding.get("formally_reportable") is False
        and diagnostic_binding.get("state_role")
        == IR_INTERPRETATION["returned_terminal_state_role"]
        and diagnostic_binding.get("path_sha256") == terminal.get("path_sha256")
        and diagnostic_binding.get("energy_sha256") == terminal.get("energy_sha256")
        and diagnostic_binding.get("force_sha256") == terminal.get("force_sha256")
        and diagnostic_binding.get("shard")
        == {"path": terminal.get("shard"), "sha256": terminal.get("shard_sha256")},
        "diagnostic call515 binding differs from the verified inventory",
    )
    override_value = item.get("shard_root_override")
    override = None if override_value in {None, ""} else BASE_MODULE.resolve_directory(
        override_value, "IR-IVM portable shard root"
    )
    declared = Path(str(terminal["shard"])).expanduser()
    shard = (override / declared.name) if override is not None else declared
    shard = shard.resolve(strict=True)
    BASE_MODULE.require(
        BASE_MODULE.sha256_file(shard) == terminal.get("shard_sha256"),
        "IR returned-terminal shard hash differs",
    )
    with np.load(shard, allow_pickle=False) as saved:
        path = np.asarray(saved["path32_A"], dtype=np.float64)
        energy = np.asarray(saved["energy_eV"], dtype=np.float64)
        force = np.asarray(saved["force_eV_per_A"], dtype=np.float64)
        call = int(BASE_MODULE.scalar(saved, "call_per_image"))
        nf = int(BASE_MODULE.scalar(saved, "optimization_N_F"))
        stored_i = float(BASE_MODULE.scalar(saved, "discrete_I_perp"))
        fully_evaluated = bool(BASE_MODULE.scalar(saved, "fully_evaluated_current_path"))
        pending = bool(BASE_MODULE.scalar(
            saved, "formal_reportability_pending_final_result"
        ))
        additional_calls = int(BASE_MODULE.scalar(saved, "additional_physical_calls"))
    BASE_MODULE.require(
        call == IR_INTERPRETATION["returned_terminal_call"]
        and nf == IR_INTERPRETATION["returned_terminal_optimization_N_F"]
        and fully_evaluated and pending and additional_calls == 0
        and BASE_MODULE.sha256_array(path) == terminal.get("path_sha256")
        and BASE_MODULE.sha256_array(energy) == terminal.get("energy_sha256")
        and BASE_MODULE.sha256_array(force) == terminal.get("force_sha256"),
        "IR returned-terminal arrays/semantics differ from the verified inventory",
    )
    result = BASE_MODULE.record(
        IR_DIAGNOSTIC_KEY,
        nf,
        path,
        energy,
        force,
        {
            "verification_sha256": BASE_MODULE.sha256_file(verification_path),
            "shard": str(shard),
            "shard_sha256": terminal["shard_sha256"],
            "call_per_image": call,
            "formally_reportable": False,
            "terminal_uncommitted_charged_tail": True,
        },
        state_role=(
            "diagnostic returned terminal; charged but uncommitted and excluded "
            "from the formal IVM history"
        ),
        retain_arrays=True,
    )
    BASE_MODULE.require(
        np.asarray(result["i_perp"]).tobytes() == np.asarray(stored_i).tobytes(),
        "IR returned-terminal stored/recomputed I_perp differs",
    )
    BASE_MODULE.require(
        result["i_perp"] == diagnostic_binding.get("i_perp")
        and result["delta_e32_eV"] == diagnostic_binding.get("delta_e32_eV")
        and result["delta_e32_kcal_per_mol"]
        == diagnostic_binding.get("delta_e32_kcal_per_mol"),
        "diagnostic call515 binding metrics differ",
    )
    return result


def load_ir_ivm_r2(
    manifest: dict[str, Any], requested_budgets: Sequence[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    item = manifest["inputs"].get("ir_ivm_full_state")
    BASE_MODULE.require(isinstance(item, Mapping), "IR-IVM full-state input is absent")
    verification_path = BASE_MODULE.resolve_binding(
        item.get("verification"), "IR-IVM R5B verification"
    )
    verification = BASE_MODULE.load_json(verification_path)
    official = _BASE_LOAD_IR(manifest, requested_budgets)
    BASE_MODULE.require(
        official[-1]["optimization_N_F"]
        == IR_INTERPRETATION["last_formally_reportable_optimization_N_F"],
        "formal IVM history does not end at committed NF8982",
    )
    diagnostic = _load_uncommitted_ir_terminal(
        verification_path, verification, item
    )
    return official, [diagnostic]


def load_formal_ode_cache_r2(
    cache_path: Path,
    terminal_path: Path,
    run_dir: Path,
    selected_k: float,
    requested_budgets: Sequence[int],
) -> list[dict[str, Any]]:
    cache = BASE_MODULE.load_json(cache_path)
    BASE_MODULE.require(
        cache.get("schema") == "bb-ode-formal-discrete32-incremental-cache-v1"
        and cache.get("status") == "FINAL_BOUND_TO_CANONICAL_TERMINAL_AUDIT"
        and cache.get("partial_is_final") is False
        and cache.get("run_dir") == str(run_dir)
        and float(cache.get("spring_k_eV_per_A2", -1)) == selected_k
        and int(cache.get("target_batches", -1)) == FORMAL_BATCHES
        and int(cache.get("target_optimization_N_F", -1)) == FORMAL_NF
        and int(cache.get("processed_batches", -1)) == FORMAL_BATCHES
        and int(cache.get("processed_optimization_N_F", -1)) == FORMAL_NF
        and cache.get("physical_calls_started") == 0
        and cache.get("openmm_contexts_constructed") == 0
        and cache.get("bb_locks_acquired") == 0
        and cache.get("local_mutagen_replica_transport")
        == expected_cache_replica_transport(),
        "formal ODE-NEB NF30000 cache is not final/compatible",
    )
    BASE_MODULE.require(
        cache.get("canonical_terminal_audit")
        == {"path": str(terminal_path), "sha256": BASE_MODULE.sha256_file(terminal_path)},
        "formal NF30000 cache terminal-audit binding differs",
    )
    terminal = BASE_MODULE.load_json(terminal_path)
    candidate = terminal.get("candidate", {})
    BASE_MODULE.require(
        terminal.get("schema") == "bb-neb-aseode-kgrid-external-terminal-audit-v1"
        and terminal.get("status") == "PASS"
        and terminal.get("terminal_eligible") is True
        and terminal.get("formal_figure_eligible") is True
        and candidate.get("phase") == "formal"
        and float(candidate.get("spring_k_eV_per_A2", -1)) == selected_k
        and int(terminal.get("paid_batches", -1)) == FORMAL_BATCHES
        and int(terminal.get("paid_optimization_N_F", -1)) == FORMAL_NF,
        "formal ODE-NEB terminal audit is not the selected NF30000 PASS",
    )
    rows = cache.get("rows")
    BASE_MODULE.require(
        isinstance(rows, list)
        and len(rows) == FORMAL_BATCHES
        and [int(row.get("optimization_N_F", -1)) for row in rows]
        == list(range(30, FORMAL_NF + 1, 30)),
        "formal NF30000 cache budget axis differs",
    )
    source_rows = []
    output = []
    for batch_id, row in enumerate(rows, 1):
        unsigned = dict(row)
        row_hash = str(unsigned.pop("row_sha256", ""))
        expected_hash = hashlib.sha256(json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")).hexdigest()
        BASE_MODULE.require(
            row_hash == expected_hash
            and int(row.get("batch_id", -1)) == batch_id
            and math.isfinite(float(row.get("i_perp", math.nan)))
            and math.isfinite(float(row.get("delta_e32_eV", math.nan))),
            f"formal NF30000 cache row differs at batch {batch_id}",
        )
        source_rows.append({key: row[key] for key in (
            "batch_id",
            "optimization_N_F",
            "source_artifact_sha256",
            "full_state_wal_event_sha256",
            "ledger_commit_event_sha256",
            "row_sha256",
        )})
        output.append({
            "method": "ODE-NEB",
            "optimization_N_F": int(row["optimization_N_F"]),
            "i_perp": float(row["i_perp"]),
            "delta_e32_eV": float(row["delta_e32_eV"]),
            "delta_e32_kcal_per_mol": float(row["delta_e32_kcal_per_mol"]),
            "path_sha256": str(row["current_path_sha256"]),
            "energy_sha256": str(row["current_energy_sha256"]),
            "force_sha256": str(row["current_force_sha256"]),
            "state_role": "current accepted path after paid batch",
            "source": {
                "run_dir": str(run_dir),
                "batch_id": batch_id,
                "terminal_audit_sha256": BASE_MODULE.sha256_file(terminal_path),
                "incremental_cache_sha256": BASE_MODULE.sha256_file(cache_path),
            },
        })
    source_digest = hashlib.sha256(json.dumps(
        source_rows,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    BASE_MODULE.require(
        source_digest == cache.get("rows_source_closure_sha256"),
        "formal NF30000 cache rows-source closure differs",
    )
    states = cache.get("selected_state_bindings")
    BASE_MODULE.require(
        isinstance(states, Mapping)
        and set(states) == {str(value) for value in RETAINED_ODE_STATE_NF},
        "formal NF30000 cache lacks the exact retained state packs",
    )
    required_nfs = BASE_MODULE.retained_state_nfs(
        [row["optimization_N_F"] for row in output], requested_budgets
    )
    BASE_MODULE.require(
        required_nfs <= {int(value) for value in states},
        "formal NF30000 cache lacks a plotted state pack",
    )
    for nf_text, state_binding in states.items():
        nf = int(nf_text)
        state_path = BASE_MODULE.resolve_binding(
            state_binding, f"formal ODE-NEB NF{nf} state pack"
        )
        with np.load(state_path, allow_pickle=False) as saved:
            BASE_MODULE.require(
                BASE_MODULE.scalar(saved, "schema")
                == "bb-ode-formal-discrete32-selected-state-v1"
                and int(BASE_MODULE.scalar(saved, "optimization_N_F")) == nf
                and int(BASE_MODULE.scalar(saved, "endpoint_optimization_N_F")) == 0
                and int(BASE_MODULE.scalar(saved, "additional_physical_calls")) == 0,
                f"formal ODE-NEB NF{nf} retained-state identity differs",
            )
            path32 = np.asarray(saved["path32_A"], dtype=np.float64)
            energy32 = np.asarray(saved["energy32_eV"], dtype=np.float64)
            force32 = np.asarray(saved["full_force32_eV_per_A"], dtype=np.float64)
            BASE_MODULE.require(
                BASE_MODULE.sha256_array(path32) == BASE_MODULE.scalar(saved, "path_sha256")
                and BASE_MODULE.sha256_array(energy32)
                == BASE_MODULE.scalar(saved, "energy_sha256")
                and BASE_MODULE.sha256_array(force32)
                == BASE_MODULE.scalar(saved, "force_sha256"),
                f"formal ODE-NEB NF{nf} retained-state array hash differs",
            )
        row = output[nf // 30 - 1]
        measured_i, measured_e = BASE_MODULE.metric_pair(path32, energy32, force32)
        BASE_MODULE.require(
            measured_i == row["i_perp"] and measured_e == row["delta_e32_eV"],
            f"formal ODE-NEB NF{nf} cached metric/state differs",
        )
        row["path32"] = np.ascontiguousarray(path32)
        row["energy32_eV"] = np.ascontiguousarray(energy32)
        row["full_force32_eV_per_A"] = np.ascontiguousarray(force32)
    return output


def load_formal_ode_r2(
    manifest: dict[str, Any], selected_k: float, requested_budgets: Sequence[int]
) -> list[dict[str, Any]]:
    item = manifest["inputs"].get("formal_ode_neb")
    BASE_MODULE.require(isinstance(item, Mapping), "formal ODE-NEB input is absent")
    run_dir = BASE_MODULE.resolve_directory(item.get("run_dir"), "formal ODE-NEB run")
    terminal_record = item.get("terminal_audit")
    BASE_MODULE.require(
        isinstance(terminal_record, Mapping)
        and set(terminal_record) == {"path", "sha256"},
        "formal ODE-NEB terminal-audit binding is absent or not exact",
    )
    terminal_path = resolve_replica_terminal_audit_file(
        terminal_record["path"], str(terminal_record["sha256"])
    )
    cache_path = BASE_MODULE.resolve_binding(
        item.get("incremental_cache"), "formal ODE-NEB NF30000 final cache"
    )
    return load_formal_ode_cache_r2(
        cache_path, terminal_path, run_dir, selected_k, requested_budgets
    )


def validate_evolution_registration(manifest: Mapping[str, Any]):
    comparison = manifest.get("comparison", {})
    registration = comparison.get("evolution_surface_envelope_registration", {})
    legacy = comparison.get("legacy_evolution_layout_reference", {})
    projection = registration.get("excluded_budget_projection", {})
    alternative = registration.get(
        "alternative_geometry_valid_budget_projection", {}
    )
    ivm = projection.get("IVM", {})
    ode = projection.get("ODE-NEB", {})
    BASE_MODULE.require(
        comparison.get("evolution_same_N_F_budgets") == [2040, 15000, 30000]
        and set(registration) == {
            "alternative_geometry_valid_budget_N_F",
            "alternative_geometry_valid_budget_projection",
            "alternative_not_selected_reason",
            "excluded_budget_N_F", "excluded_budget_projection",
            "grid_normal_max_A", "grid_normal_min_A",
            "path_clipping_permitted", "pre_final_protocol_amendment",
            "requested_budgets_N_F", "selection_not_based_on_performance",
            "selection_reason",
            "terminal_budget_requires_fresh_strict_envelope_pass",
        }
        and registration.get("requested_budgets_N_F") == [2040, 15000, 30000]
        and registration.get("alternative_geometry_valid_budget_N_F") == 990
        and alternative == {
            "IVM": {
                "actual_optimization_N_F": 969,
                "normal_max_A": 0.4513690503490503,
                "normal_min_A": 0.0,
                "strict_envelope_pass": True,
            },
            "ODE-NEB": {
                "actual_optimization_N_F": 990,
                "normal_max_A": 0.640372983587181,
                "normal_min_A": 0.0,
                "strict_envelope_pass": True,
            },
        }
        and "neither choice used performance" in str(
            registration.get("alternative_not_selected_reason")
        )
        and registration.get("excluded_budget_N_F") == 240
        and registration.get("grid_normal_min_A") == -0.12
        and registration.get("grid_normal_max_A") == 0.69
        and registration.get("path_clipping_permitted") is False
        and registration.get("pre_final_protocol_amendment") is True
        and registration.get("selection_not_based_on_performance") is True
        and registration.get(
            "terminal_budget_requires_fresh_strict_envelope_pass"
        ) is True
        and set(projection) == {"IVM", "ODE-NEB"}
        and ivm == {
            "actual_optimization_N_F": 240,
            "normal_max_A": 0.6275520851946432,
            "normal_min_A": 0.0,
            "strict_envelope_pass": True,
        }
        and ode == {
            "actual_optimization_N_F": 240,
            "normal_max_A": 0.7038122887583026,
            "normal_min_A": 0.0,
            "strict_envelope_pass": False,
        }
        and "retained-state, geometry-valid milestone" in str(
            registration.get("selection_reason")
        )
        and "not selected by performance" in str(
            registration.get("selection_reason")
        ),
        "evolution common-budget surface-envelope registration differs",
    )
    BASE_MODULE.require(
        set(legacy) == {
            "layout", "panel_checkpoint_labels_not_number_of_forces",
            "panel_number_of_forces", "pdf", "plot_source",
        }
        and legacy.get("panel_checkpoint_labels_not_number_of_forces")
        == [0, 50, 200, 500]
        and legacy.get("panel_number_of_forces") == {
            "50": {"IVM": 2792, "ODE-NEB": 2852},
            "200": {"IVM": 11309, "ODE-NEB": 11402},
            "500": {"IVM": 28073, "ODE-NEB": 28502},
        },
        "legacy evolution layout/budget provenance differs",
    )
    BASE_MODULE.resolve_binding(legacy.get("pdf"), "legacy evolution layout PDF")
    BASE_MODULE.resolve_binding(
        legacy.get("plot_source"), "legacy evolution plotting source"
    )
    return dict(registration), dict(legacy)


def load_complete_data_r2(
    manifest: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    evolution_registration, legacy_layout = validate_evolution_registration(manifest)
    source_root = BASE_MODULE.resolve_directory(
        manifest["inputs"].get("ode_source_root"), "frozen NF30000 ODE source"
    )
    compact_module, recorder_module, terminal_module = BASE_MODULE.import_ode_source(
        source_root
    )
    authorization_module = importlib.import_module("src.kgrid_execution_authorization")
    imported = {
        "portable trace verifier": (
            compact_module,
            source_root / "src" / "kgrid_compact_trace.py",
        ),
        "full-state recorder": (
            recorder_module,
            source_root / "src" / "full_batch_recorder.py",
        ),
        "terminal auditor": (
            terminal_module,
            source_root / "src" / "kgrid_terminal_audit.py",
        ),
        "execution authorizer": (
            authorization_module,
            source_root / "src" / "kgrid_execution_authorization.py",
        ),
    }
    for role, (module, expected_path) in imported.items():
        BASE_MODULE.require(
            Path(str(module.__file__)).resolve(strict=True)
            == expected_path.resolve(strict=True),
            f"{role} was resolved outside the frozen NF30000 source root",
        )
    source_identity = authorization_module.verify_source_identity().as_dict()
    selection, selection_evidence = load_adopted_selection(
        manifest, authorization_module
    )
    selected_k = float(selection["selected_spring_k_eV_per_A2"])
    plot_budgets = BASE_MODULE.requested_plot_budgets(manifest)
    official_ivm, diagnostic = load_ir_ivm_r2(manifest, plot_budgets)
    methods = {
        "IVM": official_ivm,
        "Previous IVM": _BASE_LOAD_OLD_IVM(manifest, plot_budgets),
        "ODE-NEB": load_formal_ode_r2(manifest, selected_k, plot_budgets),
        IR_DIAGNOSTIC_KEY: diagnostic,
    }
    initial_hashes = {
        methods[name][0]["path_sha256"]
        for name in ("IVM", "Previous IVM", "ODE-NEB")
    }
    BASE_MODULE.require(
        len(initial_hashes) == 1,
        "the three formal method histories do not share the exact initial path",
    )
    return methods, {
        "selected_spring_k_eV_per_A2": selected_k,
        "common_initial_path_sha256": next(iter(initial_hashes)),
        "selection_verification": selection_evidence,
        "ode_source_identity": source_identity,
        "ir_ivm_state_interpretation": dict(IR_INTERPRETATION),
        "evolution_surface_envelope_registration": evolution_registration,
        "legacy_evolution_layout_reference": legacy_layout,
    }


def _formal_methods(methods: Mapping[str, Sequence[dict[str, Any]]]):
    return {name: methods[name] for name in ("IVM", "Previous IVM", "ODE-NEB")}


def build_pack_r2(manifest_path, manifest, methods, meta):
    pack = _BASE_BUILD_PACK(
        manifest_path, manifest, _formal_methods(methods), meta
    )
    pack["schema"] = PACK_SCHEMA
    pack["status"] = "COMPLETE_ZERO_FORCE_DISCRETE32_NF30000_R2_ANALYSIS"
    pack["ode_source_identity"] = meta["ode_source_identity"]
    pack["adopted_selection_verification"] = meta["selection_verification"]
    pack["ir_ivm_state_interpretation"] = meta["ir_ivm_state_interpretation"]
    pack["evolution_surface_envelope_registration"] = meta[
        "evolution_surface_envelope_registration"
    ]
    pack["legacy_evolution_layout_reference"] = meta[
        "legacy_evolution_layout_reference"
    ]
    pack["diagnostic_uncommitted_states"] = [
        BASE_MODULE.serializable_row(row) for row in methods[IR_DIAGNOSTIC_KEY]
    ]
    pack["diagnostic_exclusion"] = {
        "excluded_from_formal_IVM_history": True,
        "excluded_from_cost_to_precision": True,
        "excluded_from_same_N_F_envelopes": True,
        "permitted_only_in_energy_profile_and_evolution_as_labeled_diagnostic": True,
    }
    pack["evolution_surface_display_policy"] = {
        "source_surface": "unrelaxed Cartesian 0 K R2B surface; values unchanged",
        "rendering_only_high_energy_clip": True,
        "display_cap_kcal_per_mol": EVOLUTION_DISPLAY_CAP_KCAL_PER_MOL,
        "normalization": "log1p(scale=40) followed by 1-(1-t)^2.5",
        "reference_implementation": {
            "path": str(EVOLUTION_CLIP_REFERENCE),
            "sha256": EVOLUTION_CLIP_REFERENCE_SHA256,
        },
    }
    return pack


def plot_cost_r2(methods, *, live):
    figure = _BASE_PLOT_COST(_formal_methods(methods), live=live)
    for axis in figure.axes:
        axis.set_ylim(0, FORMAL_NF)
    contract = FIGURE_CONTRACT_R2["cost_to_precision"]
    BASE_MODULE.require(len(figure.axes) == 2, "R2 cost figure is not exactly two panels")
    for axis, expected in zip(figure.axes, contract["axes"]):
        labels = [line.get_label() for line in axis.lines]
        BASE_MODULE.require(
            labels == contract["formal_methods"]
            and axis.get_xscale() == expected["xscale"]
            and axis.get_yscale() == expected["yscale"]
            and np.allclose(axis.get_xlim(), expected["xlim"], rtol=0.0, atol=1.0e-12)
            and np.allclose(axis.get_ylim(), expected["ylim"], rtol=0.0, atol=1.0e-12),
            f"R2 cost panel contract differs for {expected['metric']}",
        )
    BASE_MODULE.require(
        figure.axes[0].get_ylabel() == contract["ylabel"],
        "R2 cost y-axis label differs",
    )
    return figure


def plot_energy_profiles_r2(methods, budget):
    figure = _BASE_PLOT_ENERGY(_formal_methods(methods), budget)
    diagnostic = methods[IR_DIAGNOSTIC_KEY][-1]
    if int(diagnostic["optimization_N_F"]) <= int(budget):
        BASE_MODULE.require_stored_arrays(diagnostic, "uncommitted IVM energy profile")
        coordinate = BASE_MODULE.arc_coordinate(diagnostic["path32"])
        relative = (
            diagnostic["energy32_eV"] - diagnostic["energy32_eV"][0]
        ) * BASE_MODULE.EV_TO_KCAL
        axis = figure.axes[0]
        axis.plot(
            coordinate,
            relative,
            color="#CC79A7",
            marker="x",
            linestyle="--",
            linewidth=1.2,
            markersize=3.0,
            label=(
                f"IVM trial (NF {diagnostic['optimization_N_F']}; uncommitted)"
            ),
        )
        axis.legend(frameon=False, fontsize=6.8)
        axis.text(
            0.02,
            0.02,
            "Uncommitted trial is diagnostic only",
            transform=axis.transAxes,
            fontsize=6.5,
            color="#8A2D68",
            va="bottom",
        )
    if int(budget) == FORMAL_NF:
        contract = FIGURE_CONTRACT_R2["energy_vs_arc_length"]
        BASE_MODULE.require(len(figure.axes) == 1, "R2 energy figure is not one panel")
        lines = figure.axes[0].lines
        BASE_MODULE.require(
            [line.get_label() for line in lines] == contract["line_labels"],
            "R2 energy-profile line identities/order differ",
        )
        for line in lines:
            x = np.asarray(line.get_xdata(), dtype=np.float64)
            y = np.asarray(line.get_ydata(), dtype=np.float64)
            BASE_MODULE.require(
                x.shape == y.shape == (contract["points_per_line"],)
                and np.isfinite(x).all() and np.isfinite(y).all()
                and np.all(np.diff(x) > 0.0)
                and np.allclose(
                    x[[0, -1]], contract["normalized_arc_length_endpoints"],
                    rtol=0.0, atol=1.0e-12,
                ),
                f"R2 energy-profile 32-point contract differs for {line.get_label()}",
            )
    return figure


def _resolve_sized_binding(record: Any, role: str) -> Path:
    BASE_MODULE.require(
        isinstance(record, Mapping) and set(record) == {"path", "sha256", "bytes"},
        f"{role} is not an exact path/SHA/byte binding",
    )
    path = Path(str(record["path"])).expanduser().resolve(strict=True)
    BASE_MODULE.require(
        path.is_file() and not path.is_symlink()
        and int(record["bytes"]) == path.stat().st_size
        and str(record["sha256"]) == BASE_MODULE.sha256_file(path),
        f"{role} binding differs",
    )
    return path


def _surface_scalar(saved: Mapping[str, Any], key: str) -> Any:
    return BASE_MODULE.scalar(saved, key)


def _validate_surface_wal(
    wal_path: Path,
    result_path: Path,
    coordinate_hashes: np.ndarray,
    surface_energy: np.ndarray,
    audit: Mapping[str, Any],
) -> None:
    lines = wal_path.read_text(encoding="utf-8").splitlines()
    BASE_MODULE.require(lines, "call515 surface energy WAL is empty")
    header = json.loads(lines[0])
    BASE_MODULE.require(
        isinstance(header, Mapping)
        and header.get("schema") == SURFACE_WAL_SCHEMA
        and header.get("status") == "OPEN_APPEND_ONLY_FSYNC"
        and header.get("stage") == "surface"
        and int(header.get("fresh_energy_only_attempts_planned", -1))
        == SURFACE_FRESH_VALUES
        and int(header.get("maximum_attempt_started_records", -1))
        == SURFACE_FRESH_VALUES
        and int(header.get("force_calls_planned", -1)) == 0
        and int(header.get("optimization_N_F_contribution", -1)) == 0,
        "call515 surface energy WAL header differs",
    )
    starts: dict[str, Mapping[str, Any]] = {}
    terminals: dict[str, Mapping[str, Any]] = {}
    publication: Mapping[str, Any] | None = None
    for number, line in enumerate(lines[1:], 2):
        event = json.loads(line)
        BASE_MODULE.require(isinstance(event, Mapping), f"surface WAL line {number} differs")
        kind = event.get("event")
        BASE_MODULE.require(
            kind in {"attempt_started", "attempt_succeeded", "attempt_failed", "result_published"},
            f"surface WAL line {number} has an unknown event",
        )
        if kind == "result_published":
            BASE_MODULE.require(publication is None, "surface WAL has duplicate publications")
            publication = event
            continue
        BASE_MODULE.require(publication is None, "surface WAL attempt appears after publication")
        attempt_id = str(event.get("attempt_id", ""))
        BASE_MODULE.require(attempt_id, f"surface WAL line {number} lacks attempt id")
        if kind == "attempt_started":
            BASE_MODULE.require(attempt_id not in starts, "surface WAL repeats an attempt id")
            starts[attempt_id] = event
            continue
        BASE_MODULE.require(
            attempt_id in starts and attempt_id not in terminals,
            "surface WAL has an unmatched/duplicate terminal event",
        )
        start = starts[attempt_id]
        for key in ("stage", "logical_index", "center_index", "normal_index", "coordinate_sha256"):
            BASE_MODULE.require(event.get(key) == start.get(key),
                                f"surface WAL terminal/start {key} differs")
        terminals[attempt_id] = event

    expected_indices = set(range(SURFACE_STORED_VALUES)) - {
        SURFACE_ZERO_INDEX,
        (SURFACE_POINTS - 1) * SURFACE_NORMAL_POINTS + SURFACE_ZERO_INDEX,
    }
    successes: dict[int, Mapping[str, Any]] = {}
    for attempt_id, start in starts.items():
        terminal = terminals.get(attempt_id)
        BASE_MODULE.require(
            terminal is not None and terminal.get("event") == "attempt_succeeded",
            "surface WAL has a failed or open physical energy attempt",
        )
        logical_index = int(start.get("logical_index", -1))
        center_index, normal_index = divmod(logical_index, SURFACE_NORMAL_POINTS)
        BASE_MODULE.require(
            logical_index in expected_indices and logical_index not in successes
            and start.get("stage") == "surface"
            and int(start.get("center_index", -1)) == center_index
            and int(start.get("normal_index", -1)) == normal_index
            and str(start.get("coordinate_sha256", ""))
            == str(coordinate_hashes[center_index, normal_index]),
            "surface WAL logical-coordinate identity differs",
        )
        energy = float(terminal.get("energy_eV", math.nan))
        BASE_MODULE.require(
            math.isfinite(energy)
            and BASE_MODULE.sha256_array(np.asarray(energy, dtype=np.float64))
            == terminal.get("energy_scalar_sha256")
            and np.asarray(energy).tobytes()
            == np.asarray(surface_energy[center_index, normal_index]).tobytes(),
            "surface WAL/result energy differs",
        )
        successes[logical_index] = terminal
    BASE_MODULE.require(
        set(successes) == expected_indices
        and len(starts) == len(terminals) == SURFACE_FRESH_VALUES,
        "surface WAL is not the exact complete 1650-call inventory",
    )
    BASE_MODULE.require(
        publication is not None
        and publication.get("stage") == "surface"
        and publication.get("result") == str(result_path)
        and publication.get("result_sha256") == BASE_MODULE.sha256_file(result_path)
        and int(publication.get("result_bytes", -1)) == result_path.stat().st_size
        and publication.get("result_schema") == SURFACE_RESULT_SCHEMA,
        "surface WAL publication footer differs",
    )
    lifetime = {
        "attempt_started": SURFACE_FRESH_VALUES,
        "attempt_succeeded": SURFACE_FRESH_VALUES,
        "attempt_failed": 0,
        "attempt_open": 0,
        "accepted_unique_successes": SURFACE_FRESH_VALUES,
        "retry_attempts": 0,
    }
    BASE_MODULE.require(
        audit.get("energy_WAL_lifetime") == lifetime
        and int(audit.get("authorization_lifetime_attempt_started_across_both_stages", -1))
        == SURFACE_FRESH_VALUES,
        "surface audit/WAL lifetime accounting differs",
    )


def _load_call515_surface_result(
    surface_path: Path, audit_path: Path
) -> dict[str, np.ndarray]:
    audit = BASE_MODULE.load_json(audit_path)
    BASE_MODULE.require(
        audit.get("schema") == SURFACE_AUDIT_SCHEMA
        and audit.get("status") == "PASS"
        and audit.get("stage") == "surface"
        and int(audit.get("center_source_call_per_image", -1)) == 515
        and int(audit.get("center_source_optimization_N_F", -1)) == 14983
        and audit.get("state_role") == SURFACE_STATE_ROLE
        and int(audit.get("stored_energy_values", -1)) == SURFACE_STORED_VALUES
        and int(audit.get("fresh_energy_only_attempts_lineage", -1))
        == SURFACE_FRESH_VALUES
        and int(audit.get("inherited_endpoint_energy_values", -1)) == 2
        and int(audit.get("fresh_fixed_endpoint_energy_calls", -1)) == 0
        and int(audit.get("force_calls", -1)) == 0
        and int(audit.get("optimization_N_F_contribution", -1)) == 0,
        "call515 physical surface audit identity/accounting differs",
    )
    result_binding_path = _resolve_sized_binding(audit.get("result"), "surface result")
    BASE_MODULE.require(result_binding_path == surface_path,
                        "surface audit binds a different result")
    bound_paths = {
        role: _resolve_sized_binding(audit.get(role), f"surface {role}")
        for role in (
            "contract", "authorization", "preauthorization_preflight",
            "coordinate_manifest", "coordinate_audit", "energy_call_policy",
            "endpoint_cache", "energy_wal",
        )
    }
    coordinate_audit = BASE_MODULE.load_json(bound_paths["coordinate_audit"])
    BASE_MODULE.require(
        coordinate_audit.get("schema")
        == "bb-irivm-call515-terminal-tail-section-coordinate-audit-r2-v1"
        and coordinate_audit.get("status")
        == "PASS_ZERO_OPENMM_RETURNED_TERMINAL_COORDINATE_AUDIT"
        and int(coordinate_audit.get("center_source_call_per_image", -1)) == 515
        and int(coordinate_audit.get("center_source_optimization_N_F", -1)) == 14983
        and coordinate_audit.get("state_role") == SURFACE_STATE_ROLE
        and coordinate_audit.get("surface_shape") == [SURFACE_POINTS, SURFACE_NORMAL_POINTS]
        and int(coordinate_audit.get("stored_coordinate_values", -1))
        == SURFACE_STORED_VALUES
        and int(coordinate_audit.get("physical_calls_started", -1)) == 0
        and int(coordinate_audit.get("force_calls", -1)) == 0
        and int(coordinate_audit.get("optimization_N_F_contribution", -1)) == 0
        and coordinate_audit.get("coordinate_manifest")
        == audit.get("coordinate_manifest"),
        "call515 coordinate audit differs",
    )

    with np.load(surface_path, allow_pickle=False) as saved:
        BASE_MODULE.require(set(saved.files) == SURFACE_RESULT_FIELDS,
                            "call515 surface result field inventory differs")
        scalar_expectations = {
            "schema": SURFACE_RESULT_SCHEMA,
            "status": "PASS",
            "stage": "surface",
            "method": "IR-IVM returned terminal diagnostic",
            "center_source_call_per_image": 515,
            "center_source_optimization_N_F": 14983,
            "fresh_fixed_endpoint_energy_calls": 0,
            "force_calls": 0,
            "optimization_N_F_contribution": 0,
            "returned_terminal_state": True,
            "formally_reportable": False,
            "terminal_uncommitted_charged_tail": True,
            "zero_normal_index": SURFACE_ZERO_INDEX,
            "stored_energy_values": SURFACE_STORED_VALUES,
            "fresh_energy_only_attempts": SURFACE_FRESH_VALUES,
            "inherited_endpoint_energy_values": 2,
        }
        for key, expected_value in scalar_expectations.items():
            BASE_MODULE.require(_surface_scalar(saved, key) == expected_value,
                                f"call515 surface result {key} differs")
        for key, role in (
            ("coordinate_manifest", "coordinate_manifest"),
            ("coordinate_audit", "coordinate_audit"),
            ("energy_call_policy", "energy_call_policy"),
            ("endpoint_cache", "endpoint_cache"),
            ("contract", "contract"),
            ("authorization", "authorization"),
            ("preauthorization_preflight", "preauthorization_preflight"),
        ):
            BASE_MODULE.require(
                str(_surface_scalar(saved, key)) == str(bound_paths[role])
                and str(_surface_scalar(saved, key + "_sha256"))
                == BASE_MODULE.sha256_file(bound_paths[role]),
                f"call515 result/audit {key} binding differs",
            )
        BASE_MODULE.require(
            str(_surface_scalar(saved, "energy_wal")) == str(bound_paths["energy_wal"]),
            "call515 result energy-WAL path differs",
        )

        coordinates = np.ascontiguousarray(saved["surface_coordinates_A"], dtype=np.float64)
        coordinate_hashes = np.asarray(saved["surface_coordinate_sha256"])
        energy = np.ascontiguousarray(saved["surface_energy_eV"], dtype=np.float64)
        center = np.ascontiguousarray(saved["center_path59_A"], dtype=np.float64)
        center_s = np.ascontiguousarray(saved["center_s"], dtype=np.float64)
        normals = np.ascontiguousarray(saved["normal_directions"], dtype=np.float64)
        initial = np.ascontiguousarray(saved["common_initial_path59_A"], dtype=np.float64)
        normal_grid = np.ascontiguousarray(saved["normal_coordinate_rms_A"], dtype=np.float64)
        fresh_mask = np.asarray(saved["fresh_evaluated_mask"], dtype=np.bool_)
        inherited_mask = np.asarray(saved["inherited_endpoint_mask"], dtype=np.bool_)
        lineage = np.asarray(saved["energy_value_lineage"])
        BASE_MODULE.require(
            coordinates.shape == (SURFACE_POINTS, SURFACE_NORMAL_POINTS, 3132, 3)
            and coordinate_hashes.shape == (SURFACE_POINTS, SURFACE_NORMAL_POINTS)
            and energy.shape == (SURFACE_POINTS, SURFACE_NORMAL_POINTS)
            and center.shape == normals.shape == initial.shape == (SURFACE_POINTS, 3132, 3)
            and center_s.shape == (SURFACE_POINTS,)
            and normal_grid.shape == (SURFACE_NORMAL_POINTS,)
            and fresh_mask.shape == inherited_mask.shape == lineage.shape
            == (SURFACE_POINTS, SURFACE_NORMAL_POINTS)
            and np.all(np.isfinite(coordinates)) and np.all(np.isfinite(energy))
            and np.all(np.isfinite(center)) and np.all(np.isfinite(normals))
            and np.all(np.isfinite(initial)) and np.all(np.isfinite(center_s))
            and np.all(np.isfinite(normal_grid)),
            "call515 surface result dimensions/finiteness differ",
        )
        expected_grid = np.arange(-0.12, 0.6901, 0.03, dtype=np.float64)
        expected_grid[SURFACE_ZERO_INDEX] = 0.0
        expected_fresh = np.ones((SURFACE_POINTS, SURFACE_NORMAL_POINTS), dtype=np.bool_)
        expected_fresh[0, SURFACE_ZERO_INDEX] = False
        expected_fresh[-1, SURFACE_ZERO_INDEX] = False
        expected_lineage = np.full(
            (SURFACE_POINTS, SURFACE_NORMAL_POINTS),
            "fresh_unrelaxed_cartesian_surface",
            dtype="<U48",
        )
        expected_lineage[~expected_fresh] = "inherited_canonical_endpoint_cache"
        BASE_MODULE.require(
            np.array_equal(normal_grid, expected_grid)
            and np.array_equal(coordinates[:, SURFACE_ZERO_INDEX], center)
            and np.array_equal(fresh_mask, expected_fresh)
            and np.array_equal(inherited_mask, ~expected_fresh)
            and np.array_equal(lineage, expected_lineage)
            and np.all(np.diff(center_s) > 0.0)
            and center_s[0] == 0.0 and center_s[-1] == 1.0
            and BASE_MODULE.sha256_array(energy)
            == str(_surface_scalar(saved, "surface_energy_sha256")),
            "call515 surface coordinates/energy lineage differs",
        )
        for center_index in range(SURFACE_POINTS):
            for normal_index in range(SURFACE_NORMAL_POINTS):
                BASE_MODULE.require(
                    BASE_MODULE.sha256_array(coordinates[center_index, normal_index])
                    == str(coordinate_hashes[center_index, normal_index]),
                    "call515 surface coordinate cell hash differs",
                )
        with np.load(bound_paths["coordinate_manifest"], allow_pickle=False) as coordinate:
            comparisons = (
                ("coordinates_A", coordinates),
                ("coordinate_sha256", coordinate_hashes),
                ("center_path59_A", center),
                ("center_s", center_s),
                ("normal_directions", normals),
                ("common_initial_path59_A", initial),
                ("normal_coordinate_rms_A", normal_grid),
            )
            BASE_MODULE.require(
                all(name in coordinate and np.array_equal(np.asarray(coordinate[name]), value)
                    for name, value in comparisons),
                "call515 energy result differs from its frozen coordinate manifest",
            )
        with np.load(bound_paths["endpoint_cache"], allow_pickle=False) as endpoint:
            # The canonical endpoint cache stores its two Cartesian vectors
            # flattened as (2, 9396).  Reshape is byte-preserving, matches the
            # evaluator/BB-core interpretation, and permits exact comparison
            # with the 59x28 surface cells shaped as (3132, 3).
            endpoint_coordinate = np.asarray(
                endpoint["coordinates_A"], dtype=np.float64
            ).reshape(2, 3132, 3)
            endpoint_energy = np.asarray(endpoint["energies_eV"], dtype=np.float64)
        BASE_MODULE.require(
            endpoint_coordinate.shape == (2, 3132, 3)
            and endpoint_energy.shape == (2,)
            and BASE_MODULE.sha256_array(endpoint_coordinate)
            == str(_surface_scalar(saved, "endpoint_coordinate_pair_sha256"))
            and BASE_MODULE.sha256_array(endpoint_energy)
            == str(_surface_scalar(saved, "endpoint_energy_pair_sha256"))
            and np.array_equal(coordinates[[0, -1], SURFACE_ZERO_INDEX], endpoint_coordinate)
            and np.array_equal(energy[[0, -1], SURFACE_ZERO_INDEX], endpoint_energy),
            "call515 surface inherited endpoint lineage differs",
        )
        locks = [str(value) for value in np.asarray(saved["held_host_locks"]).tolist()]
        BASE_MODULE.require(
            locks == list(audit.get("held_host_locks", [])) and len(set(locks)) == 2,
            "call515 surface held-lock record differs",
        )
        output = {
            "center_s": center_s,
            "normal_coordinate_rms_A": normal_grid,
            "surface_energy_eV": energy,
            "center_path59_A": center,
            "normal_directions": normals,
            "common_initial_path59_A": initial,
        }
    _validate_surface_wal(
        bound_paths["energy_wal"], surface_path, coordinate_hashes, energy, audit
    )
    return output


def load_surface_r2(manifest):
    surface_path = BASE_MODULE.resolve_binding(
        manifest["inputs"].get("ir_centered_surface_bundle"),
        "R2 call515-centered surface result",
    )
    audit_path = BASE_MODULE.resolve_binding(
        manifest["inputs"].get("ir_centered_surface_audit"),
        "R2 call515-centered physical surface audit",
    )
    adapter_path = BASE_MODULE.resolve_binding(
        manifest["inputs"].get("surface_center_state_audit"),
        "R2 call515 surface-center adapter audit",
    )
    adapter = BASE_MODULE.load_json(adapter_path)
    expected = manifest["inputs"].get("surface_center_state")
    adapter_result = adapter.get("surface_result")
    adapter_audit = adapter.get("surface_audit")
    BASE_MODULE.require(
        expected
        == {
            "method": "IVM",
            "call_per_image": 515,
            "optimization_N_F": 14983,
            "path_sha256": "5a4ba3c5dc8f7063c1963f0bfd951f82356e60d0ad9c4fc17c0bc8b0d36d8569",
            "state_role": "terminal_uncommitted_charged_tail",
            "formally_reportable": False,
        }
        and adapter.get("schema")
        == "bb-irivm-call515-surface-center-adapter-audit-v1"
        and adapter.get("status") == "PASS_ZERO_PHYSICAL_SURFACE_CENTER_BINDING"
        and adapter.get("physical_force_calls") == 0
        and adapter.get("openmm_contexts_constructed") == 0
        and adapter.get("center_state") == expected
        and isinstance(adapter_result, Mapping)
        and set(adapter_result) == {"path", "sha256", "bytes"}
        and adapter_result
        == {"path": str(surface_path),
            "sha256": BASE_MODULE.sha256_file(surface_path),
            "bytes": surface_path.stat().st_size}
        and isinstance(adapter_audit, Mapping)
        and set(adapter_audit) == {"path", "sha256", "bytes"}
        and adapter_audit
        == {"path": str(audit_path),
            "sha256": BASE_MODULE.sha256_file(audit_path),
            "bytes": audit_path.stat().st_size},
        "R2 surface is not explicitly bound to the call515 uncommitted diagnostic",
    )
    return _load_call515_surface_result(surface_path, audit_path)


def _surface_for_evolution_display(surface):
    BASE_MODULE.require(
        EVOLUTION_CLIP_REFERENCE.resolve(strict=True) == EVOLUTION_CLIP_REFERENCE
        and BASE_MODULE.sha256_file(EVOLUTION_CLIP_REFERENCE)
        == EVOLUTION_CLIP_REFERENCE_SHA256,
        "fixed high-energy clipping reference changed",
    )
    energy = np.asarray(surface["surface_energy_eV"], dtype=np.float64)
    normal_grid = np.asarray(surface["normal_coordinate_rms_A"], dtype=np.float64)
    BASE_MODULE.require(
        energy.shape == (SURFACE_POINTS, SURFACE_NORMAL_POINTS)
        and normal_grid.shape == (SURFACE_NORMAL_POINTS,),
        "surface display arrays differ",
    )
    zero_index = int(np.argmin(np.abs(normal_grid)))
    BASE_MODULE.require(zero_index == SURFACE_ZERO_INDEX,
                        "surface display zero-normal index differs")
    reference_energy_eV = float(energy[0, zero_index])
    relative_kcal = np.maximum(
        (energy - reference_energy_eV) * BASE_MODULE.EV_TO_KCAL, 0.0
    )
    clipped_mask = relative_kcal > EVOLUTION_DISPLAY_CAP_KCAL_PER_MOL
    display = dict(surface)
    display["surface_energy_eV"] = np.ascontiguousarray(
        reference_energy_eV
        + np.minimum(relative_kcal, EVOLUTION_DISPLAY_CAP_KCAL_PER_MOL)
        / BASE_MODULE.EV_TO_KCAL,
        dtype=np.float64,
    )
    audit = {
        "schema": "bb-r2-evolution-surface-display-policy-v1",
        "source_surface_values_unchanged": True,
        "rendering_copy_clipped_only": True,
        "display_cap_kcal_per_mol": EVOLUTION_DISPLAY_CAP_KCAL_PER_MOL,
        "unclipped_maximum_kcal_per_mol": float(np.max(relative_kcal)),
        "clipped_grid_values": int(np.count_nonzero(clipped_mask)),
        "stored_grid_values": int(relative_kcal.size),
        "reference_implementation": {
            "path": str(EVOLUTION_CLIP_REFERENCE),
            "sha256": EVOLUTION_CLIP_REFERENCE_SHA256,
        },
        "physical_calls": 0,
    }
    return display, audit


def plot_evolution_r2(methods, surface, budgets):
    display_surface, display_audit = _surface_for_evolution_display(surface)
    figure = _BASE_PLOT_EVOLUTION(
        _formal_methods(methods), display_surface, budgets
    )
    figure.axes[-1].set_title(r"$>10^6$ clipped", fontsize=6.0, pad=2.0)
    figure._bb_surface_display_audit = display_audit
    diagnostic = methods[IR_DIAGNOSTIC_KEY][-1]
    plotted = False
    for index, budget in enumerate(budgets, 1):
        axis = figure.axes[index]
        ivm = BASE_MODULE.latest_at_or_before(methods["IVM"], budget)
        ode = BASE_MODULE.latest_at_or_before(methods["ODE-NEB"], budget)
        title = (
            f"IVM NF{ivm['optimization_N_F']}\n"
            f"ODE-NEB NF{ode['optimization_N_F']}"
        )
        if int(budget) >= int(diagnostic["optimization_N_F"]):
            projected = BASE_MODULE.project_path(
                diagnostic["path32"],
                surface["center_path59_A"],
                surface["center_s"],
                surface["normal_directions"],
            )
            normal_grid = surface["normal_coordinate_rms_A"]
            BASE_MODULE.require(
                projected[:, 1].min() >= normal_grid.min() - 2.0e-8
                and projected[:, 1].max() <= normal_grid.max() + 2.0e-8,
                "uncommitted IVM returned terminal leaves its diagnostic surface",
            )
            axis.plot(
                projected[:, 0],
                projected[:, 1],
                color="#CC79A7",
                marker="x",
                markevery=4,
                markersize=2.3,
                linestyle="--",
                linewidth=1.15,
                label="IVM trial NF14983 (uncommitted)",
                zorder=7,
            )
            title += "\ntrial NF14983 (uncommitted)"
            plotted = True
        axis.set_title(
            title,
            fontsize=6.6,
            pad=2.0,
            linespacing=1.0,
            multialignment="center",
        )
    if plotted:
        old_legend = figure.axes[3].get_legend()
        if old_legend is not None:
            old_legend.remove()
        figure.axes[3].legend(
            frameon=True,
            facecolor="white",
            edgecolor="#B3B3B3",
            framealpha=0.90,
            loc="upper right",
            fontsize=6.1,
            handlelength=1.8,
        )
    return figure


def publish_artifact_set_new_r2(artifacts):
    """Reuse the atomic publisher while freezing every staged output first."""

    frozen_artifacts = []
    for target, writer in artifacts:
        def write_and_freeze(path, delegate=writer):
            delegate(path)
            path.chmod(0o444)

        frozen_artifacts.append((target, write_and_freeze))
    _BASE_PUBLISH_ARTIFACT_SET_NEW(frozen_artifacts)
    for target, _writer in artifacts:
        published = Path(target).expanduser().resolve(strict=True)
        BASE_MODULE.require(
            published.is_file()
            and not published.is_symlink()
            and published.stat().st_mode & 0o222 == 0,
            f"R2 published output is not immutable: {published}",
        )


def readiness_r2(manifest):
    missing = _BASE_READINESS(manifest)
    validate_evolution_registration(manifest)
    inputs = manifest.get("inputs", {})
    formal = inputs.get("formal_ode_neb")
    if isinstance(formal, Mapping) and formal.get("incremental_cache") is None:
        missing.append("final zero-physical NF30000 incremental cache")
    ir = inputs.get("ir_ivm_full_state")
    if not isinstance(ir, Mapping) or ir.get("state_interpretation") != IR_INTERPRETATION:
        missing.append("exact IR call318/call515 state interpretation")
    if inputs.get("surface_center_state_audit") is None:
        missing.append("PASS zero-physical call515 surface-center adapter audit")
    return list(dict.fromkeys(missing))


# The reused entry point resolves these names from its own module globals.
BASE_MODULE.MANIFEST_SCHEMA = MANIFEST_SCHEMA
BASE_MODULE.PACK_SCHEMA = PACK_SCHEMA
BASE_MODULE.load_complete_data = load_complete_data_r2
BASE_MODULE.build_pack = build_pack_r2
BASE_MODULE.plot_cost = plot_cost_r2
BASE_MODULE.plot_energy_profiles = plot_energy_profiles_r2
BASE_MODULE.load_surface = load_surface_r2
BASE_MODULE.plot_evolution = plot_evolution_r2
BASE_MODULE.publish_artifact_set_new = publish_artifact_set_new_r2
BASE_MODULE.readiness = readiness_r2


def main(argv: Sequence[str] | None = None) -> None:
    BASE_MODULE.main(argv)


if __name__ == "__main__":
    main()
