#!/usr/bin/env python3
"""Freeze the R5B-bound dual-host late-six arc500 execution closure.

``prepare`` is the only mode intended to run before R5B finishes.  It reads
the already frozen cost-only R3 preregistration and split template, but it
does not read the future selection, the replay result, or any metric value.

After the two R5B verifiers pass, the remaining modes enforce this order:
selection -> input Mutagen gate -> two disjoint host contracts plus bundle ->
publication Mutagen gate.  No mode imports or constructs an OpenMM backend.
The host runner separately requires a canonical zero-physical preflight and
authorization before it can delegate any physical work.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
from typing import Iterable, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


SELECTOR_PATH = HERE / "select_bb_irivm_cost_only_late_six_r3.py"
DELEGATE_PATH = HERE / "evaluate_bb_irivm_late_six_joint_ef.py"
RUNNER_PATH = HERE / "run_bb_irivm_cost_late_six_arc500_r4.py"
AUTHORIZER_PATH = HERE / "authorize_bb_irivm_cost_late_six_arc500_r4.py"
TEST_PATH = ROOT / "tests/test_bb_irivm_cost_late_six_arc500_r4.py"
SELECTOR = _load("bb_cost_late_six_selector_r3_for_r4", SELECTOR_PATH)
DELEGATE = _load("bb_cost_late_six_delegate_r1_for_r4", DELEGATE_PATH)

PREPARATION_SCHEMA = "bb-irivm-cost-late-six-arc500-preparation-v4"
PREPARATION_STATUS = "FROZEN_ZERO_PHYSICAL_BEFORE_R5B_FINAL_SELECTION"
CONTRACT_SCHEMA = "bb-irivm-cost-late-host-arc500-contract-v4"
CONTRACT_STATUS = "FROZEN_AFTER_R5B_DUAL_PASS_AND_COST_ONLY_SELECTION"
BUNDLE_SCHEMA = "bb-irivm-cost-late-six-dual-host-bundle-v4"
BUNDLE_STATUS = "FROZEN_DISJOINT_REMOTE4_LOCAL2_CONTRACTS"
GATE_SCHEMA = "bb-irivm-cost-late-six-mutagen-gate-v4"
GATE_STATUS = "PASS_CONNECTED_FLUSHED"
PREFLIGHT_SCHEMA = "bb-irivm-cost-late-host-arc500-preflight-v4"
PREFLIGHT_STATUS = "PASS_ZERO_PHYSICAL_NONBLOCKING_HOST_PREFLIGHT"
AUTH_SCHEMA = "bb-irivm-cost-late-host-arc500-authorization-v4"
AUTH_STATUS = "AUTHORIZED_AFTER_HOST_ZERO_PHYSICAL_PREFLIGHT"

N_IMAGES = 32
DIMENSION = 9396
N_DENSE = 500
FRESH_PER_PATH = 498
REMOTE_COUNT = 4
LOCAL_COUNT = 2
MINIMUM_FREE_BYTES_RESERVE = 2 * 1024**3
ESTIMATED_SHARD_BYTES = (
    (2 * N_DENSE * DIMENSION + N_IMAGES * DIMENSION) * 8 + 8 * 1024**2
)

PREPARATION = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_COST_LATE_SIX_ARC500_PREPARATION_NF15000_20260902_R4.json"
).resolve()
INPUT_GATE = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_COST_LATE_SIX_ARC500_INPUT_MUTAGEN_GATE_20260902_R4.json"
).resolve()
PUBLICATION_GATE = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_COST_LATE_SIX_ARC500_PUBLICATION_MUTAGEN_GATE_20260902_R4.json"
).resolve()
AUTHORIZATION_GATE = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_COST_LATE_SIX_ARC500_AUTHORIZATION_MUTAGEN_GATE_20260902_R4.json"
).resolve()
BUNDLE = (
    ROOT
    / "registrations/diagnostics/"
    "BB_IRIVM_COST_LATE_SIX_ARC500_DUAL_HOST_BUNDLE_20260902_R4.json"
).resolve()
REMOTE_PREFLIGHT = (
    ROOT
    / "reports/preflights/"
    "BB_IRIVM_COST_LATE_SIX_REMOTE4_ARC500_20260902_R4.json"
).resolve()
LOCAL_PREFLIGHT = (
    ROOT
    / "reports/preflights/"
    "BB_IRIVM_COST_LATE_SIX_LOCAL2_ARC500_20260902_R4.json"
).resolve()

MUTAGEN = Path("/Users/wuzhiyou/.local/bin/mutagen")
MUTAGEN_LIST_TEMPLATE = (
    "{{ range . }}{{ .Name }}|{{ .Identifier }}|{{ .Status }}|"
    "{{ .Alpha.Connected }}|{{ .Beta.Connected }}\n{{ end }}"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bind(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": SELECTOR.sha256_file(resolved)}


def read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"{path} is not a JSON object")
    return payload


def write_exclusive(path: Path, payload: dict) -> None:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _template_paths(split: dict) -> dict:
    remote = split["remote_four"]
    local = split["local_two"]
    return {
        "remote_contract": Path(remote["future_contract"]).resolve(),
        "local_contract": Path(local["future_contract"]).resolve(),
        "remote_authorization": Path(remote["future_authorization"]).resolve(),
        "local_authorization": Path(local["future_authorization"]).resolve(),
        "remote_result": Path(remote["future_result"]).resolve(),
        "local_result": Path(local["future_result"]).resolve(),
    }


def _host_plan(split: dict, role: str) -> dict:
    source = split["remote_four" if role == "remote" else "local_two"]
    paths = _template_paths(split)
    count = REMOTE_COUNT if role == "remote" else LOCAL_COUNT
    return {
        "host_role": role,
        "host_identity": source["identity"],
        "selection_roles": source["selection_roles"],
        "path_count": count,
        "fresh_joint_EF_calls": count * FRESH_PER_PATH,
        "stored_arc500_joint_EF_values": count * N_DENSE,
        "contract_output": str(paths[f"{role}_contract"]),
        "authorization_output": str(paths[f"{role}_authorization"]),
        "result_output": str(paths[f"{role}_result"]),
        "preflight_output": str(
            REMOTE_PREFLIGHT if role == "remote" else LOCAL_PREFLIGHT
        ),
    }


def _command_templates(split: dict) -> dict:
    paths = _template_paths(split)
    python = "/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python"
    common = {
        "working_directory": str(ROOT),
        "python": python,
        "environment": dict(SELECTOR.THREAD_CAPS),
    }
    return {
        **common,
        "remote_ssh_argv_prefix": [
            "ssh",
            "-i",
            "/Users/wuzhiyou/.ssh/ivm_mac_mini_ed25519",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "UserKnownHostsFile=/Users/wuzhiyou/.ssh/known_hosts",
            "-o",
            "StrictHostKeyChecking=yes",
            "wuzhiyou@100.74.85.3",
        ],
        "remote_working_directory": str(ROOT),
        "selection": [
            python,
            str(SELECTOR_PATH),
            "select",
            "--preregistration",
            str(SELECTOR.CANONICAL_PREREGISTRATION),
            "--split-template",
            str(SELECTOR.CANONICAL_SPLIT_TEMPLATE),
            "--primary-verification",
            str(SELECTOR.CANONICAL_PRIMARY),
            "--supplemental-verification",
            str(SELECTOR.CANONICAL_SUPPLEMENTAL),
            "--manifest",
            str(SELECTOR.CANONICAL_MANIFEST),
            "--canonical-index",
            str(SELECTOR.CANONICAL_INDEX),
            "--output",
            str(SELECTOR.CANONICAL_SELECTION),
        ],
        "input_mutagen_gate": [
            python,
            str(Path(__file__).resolve()),
            "mutagen-input-gate",
            "--preparation",
            str(PREPARATION),
            "--selection",
            str(SELECTOR.CANONICAL_SELECTION),
            "--output",
            str(INPUT_GATE),
        ],
        "freeze_contracts": [
            python,
            str(Path(__file__).resolve()),
            "freeze-contracts",
            "--preparation",
            str(PREPARATION),
            "--selection",
            str(SELECTOR.CANONICAL_SELECTION),
            "--input-gate",
            str(INPUT_GATE),
            "--remote-contract",
            str(paths["remote_contract"]),
            "--local-contract",
            str(paths["local_contract"]),
            "--bundle",
            str(BUNDLE),
        ],
        "publication_mutagen_gate": [
            python,
            str(Path(__file__).resolve()),
            "mutagen-publication-gate",
            "--preparation",
            str(PREPARATION),
            "--bundle",
            str(BUNDLE),
            "--remote-contract",
            str(paths["remote_contract"]),
            "--local-contract",
            str(paths["local_contract"]),
            "--output",
            str(PUBLICATION_GATE),
        ],
        "remote_preflight": [
            python,
            str(RUNNER_PATH),
            "--contract",
            str(paths["remote_contract"]),
            "--bundle",
            str(BUNDLE),
            "--publication-gate",
            str(PUBLICATION_GATE),
            "--host-role",
            "remote",
            "--output",
            str(paths["remote_result"]),
            "--preflight-receipt",
            str(REMOTE_PREFLIGHT),
            "--preflight-only",
        ],
        "local_preflight": [
            python,
            str(RUNNER_PATH),
            "--contract",
            str(paths["local_contract"]),
            "--bundle",
            str(BUNDLE),
            "--publication-gate",
            str(PUBLICATION_GATE),
            "--host-role",
            "local",
            "--output",
            str(paths["local_result"]),
            "--preflight-receipt",
            str(LOCAL_PREFLIGHT),
            "--preflight-only",
        ],
        "remote_authorize": [
            python,
            str(AUTHORIZER_PATH),
            "--preparation",
            str(PREPARATION),
            "--bundle",
            str(BUNDLE),
            "--publication-gate",
            str(PUBLICATION_GATE),
            "--host-role",
            "remote",
            "--preflight",
            str(REMOTE_PREFLIGHT),
            "--output",
            str(paths["remote_authorization"]),
        ],
        "local_authorize": [
            python,
            str(AUTHORIZER_PATH),
            "--preparation",
            str(PREPARATION),
            "--bundle",
            str(BUNDLE),
            "--publication-gate",
            str(PUBLICATION_GATE),
            "--host-role",
            "local",
            "--preflight",
            str(LOCAL_PREFLIGHT),
            "--output",
            str(paths["local_authorization"]),
        ],
        "authorization_mutagen_gate": [
            python,
            str(Path(__file__).resolve()),
            "mutagen-authorization-gate",
            "--preparation",
            str(PREPARATION),
            "--bundle",
            str(BUNDLE),
            "--remote-authorization",
            str(paths["remote_authorization"]),
            "--local-authorization",
            str(paths["local_authorization"]),
            "--output",
            str(AUTHORIZATION_GATE),
        ],
        "remote_run": [
            python,
            str(RUNNER_PATH),
            "--contract",
            str(paths["remote_contract"]),
            "--bundle",
            str(BUNDLE),
            "--publication-gate",
            str(PUBLICATION_GATE),
            "--authorization",
            str(paths["remote_authorization"]),
            "--authorization-gate",
            str(AUTHORIZATION_GATE),
            "--host-role",
            "remote",
            "--output",
            str(paths["remote_result"]),
        ],
        "local_run": [
            python,
            str(RUNNER_PATH),
            "--contract",
            str(paths["local_contract"]),
            "--bundle",
            str(BUNDLE),
            "--publication-gate",
            str(PUBLICATION_GATE),
            "--authorization",
            str(paths["local_authorization"]),
            "--authorization-gate",
            str(AUTHORIZATION_GATE),
            "--host-role",
            "local",
            "--output",
            str(paths["local_result"]),
        ],
    }


def build_preparation(*, require_preselection_absence: bool = True) -> dict:
    prereg = SELECTOR.validate_preregistration(SELECTOR.CANONICAL_PREREGISTRATION)
    split = SELECTOR.validate_split_template(
        SELECTOR.CANONICAL_SPLIT_TEMPLATE,
        SELECTOR.CANONICAL_PREREGISTRATION,
    )
    if require_preselection_absence:
        require(
            not SELECTOR.CANONICAL_SELECTION.exists(),
            "preparation must be frozen before the future selection exists",
        )
    return {
        "schema": PREPARATION_SCHEMA,
        "status": PREPARATION_STATUS,
        "frozen": True,
        "created_utc": utc_now(),
        "execution_authorized_by_this_file": False,
        "R5B_final_artifacts_read": False,
        "selection_read": False,
        "metric_values_read": False,
        "OpenMM_imported_or_called_by_freezer": False,
        "scientific_policy": {
            "coordinate_space": "unweighted_3N_cartesian",
            "dense_grid": "500 uniform arc-length points including endpoints",
            "joint_energy_full_force_stored_per_path": N_DENSE,
            "original32_energy_full_force_source": "selected R5B stable-last shard",
            "original30_new_physical_calls": 0,
            "endpoint_joint_EF_source": "sealed shared endpoint cache",
            "endpoint_new_physical_calls_per_path": 0,
            "fresh_dense_interior_joint_EF_calls_per_path": FRESH_PER_PATH,
            "optimization_N_F_contribution": 0,
        },
        "host_plans": {
            "remote": _host_plan(split, "remote"),
            "local": _host_plan(split, "local"),
        },
        "resource_policy": {
            "one_BB_process_per_host": True,
            "jobs": 1,
            "thread_caps": dict(SELECTOR.THREAD_CAPS),
            "dual_host_local_fcntl_locks": list(SELECTOR.LOCKS),
            "nonblocking_lock_probe_required_before_authorization": True,
            "minimum_free_bytes_reserve": MINIMUM_FREE_BYTES_RESERVE,
            "estimated_uncompressed_shard_bytes_per_path": ESTIMATED_SHARD_BYTES,
        },
        "fail_closed_order": [
            "R5B primary PASS and supplemental PASS",
            "freeze metric-blind R3 selection",
            "dedicated Mutagen input flush gate",
            "freeze disjoint remote4/local2 contracts and bundle",
            "dedicated Mutagen publication flush gate",
            "run nonblocking zero-physical preflight on each assigned host",
            "issue separate host authorization bound to its preflight receipt",
            "dedicated Mutagen authorization publication flush gate",
            "launch both authorized host commands concurrently",
        ],
        "future_paths": {
            **{key: str(value) for key, value in _template_paths(split).items()},
            "input_Mutagen_gate": str(INPUT_GATE),
            "publication_Mutagen_gate": str(PUBLICATION_GATE),
            "authorization_Mutagen_gate": str(AUTHORIZATION_GATE),
            "bundle": str(BUNDLE),
            "remote_preflight": str(REMOTE_PREFLIGHT),
            "local_preflight": str(LOCAL_PREFLIGHT),
        },
        "command_templates": _command_templates(split),
        "bindings": {
            "cost_only_preregistration": bind(SELECTOR.CANONICAL_PREREGISTRATION),
            "dual_host_split_template": bind(SELECTOR.CANONICAL_SPLIT_TEMPLATE),
            "cost_only_selector": bind(SELECTOR_PATH),
            "audited_dense_delegate": bind(DELEGATE_PATH),
            "contract_freezer": bind(Path(__file__).resolve()),
            "authorized_runner": bind(RUNNER_PATH),
            "execution_authorizer": bind(AUTHORIZER_PATH),
            "zero_physical_tests": bind(TEST_PATH),
            "endpoint_joint_EF_cache": prereg["bindings"][
                "endpoint_joint_EF_cache"
            ],
            "topology": prereg["bindings"]["topology"],
            "joint_EF_engine": prereg["bindings"]["joint_EF_engine"],
            "bb_core": bind(ROOT / "src/bb_core.py"),
        },
    }


def validate_preparation(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    require(resolved == PREPARATION, "noncanonical R4 preparation path")
    payload = read_json(resolved)
    require(
        payload.get("schema") == PREPARATION_SCHEMA
        and payload.get("status") == PREPARATION_STATUS
        and payload.get("frozen") is True
        and payload.get("execution_authorized_by_this_file") is False
        and payload.get("R5B_final_artifacts_read") is False
        and payload.get("selection_read") is False
        and payload.get("metric_values_read") is False,
        "invalid R4 preparation",
    )
    split = SELECTOR.validate_split_template(
        SELECTOR.CANONICAL_SPLIT_TEMPLATE,
        SELECTOR.CANONICAL_PREREGISTRATION,
    )
    expected = build_preparation(require_preselection_absence=False)
    for field in (
        "scientific_policy",
        "host_plans",
        "resource_policy",
        "fail_closed_order",
        "future_paths",
        "command_templates",
        "bindings",
    ):
        require(payload.get(field) == expected[field], f"preparation drift: {field}")
    return payload


def _scalar(saved, key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"source shard {key} is not scalar")
    return value.item()


def _selection_closure(selection_path: Path) -> tuple[dict, list[dict]]:
    resolved = Path(selection_path).expanduser().resolve(strict=True)
    require(resolved == SELECTOR.CANONICAL_SELECTION, "noncanonical R3 selection")
    selection = read_json(resolved)
    require(
        selection.get("schema") == SELECTOR.SELECTION_SCHEMA
        and selection.get("status") == SELECTOR.SELECTION_STATUS
        and selection.get("selection_metric_values_consulted") is False
        and selection.get("terminal_partial_tail_selected") is False,
        "R3 selection is not metric-blind PASS",
    )
    closure = SELECTOR.validate_final_closure(
        SELECTOR.CANONICAL_PREREGISTRATION,
        SELECTOR.CANONICAL_SPLIT_TEMPLATE,
        SELECTOR.CANONICAL_PRIMARY,
        SELECTOR.CANONICAL_SUPPLEMENTAL,
        SELECTOR.CANONICAL_MANIFEST,
        SELECTOR.CANONICAL_INDEX,
    )
    expected_rows = SELECTOR.select_rows(closure[7])
    require(
        selection.get("selected_paths_in_preregistered_role_order")
        == expected_rows,
        "frozen selection differs from deterministic cost-only projection",
    )
    expected_bindings = {
        "preregistration": bind(SELECTOR.CANONICAL_PREREGISTRATION),
        "dual_host_template": bind(SELECTOR.CANONICAL_SPLIT_TEMPLATE),
        "primary_R5B_verification": bind(SELECTOR.CANONICAL_PRIMARY),
        "supplemental_R5B_verification": bind(SELECTOR.CANONICAL_SUPPLEMENTAL),
        "R5B_capture_manifest": bind(SELECTOR.CANONICAL_MANIFEST),
        "R5B_canonical_index": bind(SELECTOR.CANONICAL_INDEX),
    }
    for role, record in expected_bindings.items():
        require(
            selection.get("bindings", {}).get(role) == record,
            f"selection binding drift: {role}",
        )
    return selection, expected_rows


def _source_row(selected: Mapping, row_index: int, host_role: str) -> tuple[dict, dict]:
    shard_path = Path(str(selected["source_full_state_shard"])).resolve(strict=True)
    shard_binding = bind(shard_path)
    require(
        shard_binding["sha256"] == selected["source_full_state_shard_sha256"],
        "selected full-state shard file hash drift",
    )
    with np.load(shard_path, allow_pickle=False) as saved:
        required = {
            "call_per_image",
            "optimization_N_F",
            "path32_A",
            "energy_eV",
            "force_eV_per_A",
            "path_sha256",
            "energy_sha256",
            "force_sha256",
            "joint_evaluation_vector",
            "force_request_vector",
            "discrete_I_perp",
            "fully_evaluated_current_path",
            "additional_physical_calls",
        }
        require(required.issubset(saved.files), "selected R5B shard lacks full state")
        path = np.ascontiguousarray(saved["path32_A"], dtype=np.float64)
        energy = np.ascontiguousarray(saved["energy_eV"], dtype=np.float64)
        force = np.ascontiguousarray(saved["force_eV_per_A"], dtype=np.float64)
        counters = np.ascontiguousarray(
            saved["joint_evaluation_vector"], dtype=np.int64
        )
        force_requests = np.ascontiguousarray(
            saved["force_request_vector"], dtype=np.int64
        )
        embedded = {
            "call": int(_scalar(saved, "call_per_image")),
            "nf": int(_scalar(saved, "optimization_N_F")),
            "path_sha256": str(_scalar(saved, "path_sha256")),
            "energy_sha256": str(_scalar(saved, "energy_sha256")),
            "force_sha256": str(_scalar(saved, "force_sha256")),
            "discrete_I_perp": float(_scalar(saved, "discrete_I_perp")),
            "fully_evaluated": bool(_scalar(saved, "fully_evaluated_current_path")),
            "additional_physical_calls": int(
                _scalar(saved, "additional_physical_calls")
            ),
        }
    require(
        path.shape == force.shape == (N_IMAGES, DIMENSION)
        and energy.shape == (N_IMAGES,)
        and counters.shape == force_requests.shape == (N_IMAGES,),
        "selected R5B full-state shapes drift",
    )
    require(
        all(np.all(np.isfinite(value)) for value in (path, energy, force)),
        "selected R5B full state is non-finite",
    )
    require(
        embedded["call"] == int(selected["call_per_image"])
        and embedded["nf"] == int(selected["optimization_N_F"])
        and embedded["path_sha256"] == selected["selected_path_sha256"]
        and embedded["fully_evaluated"] is True
        and embedded["additional_physical_calls"] == 0,
        "selected R5B shard identity/evaluation drift",
    )
    require(
        DELEGATE.verified_array_sha256(path) == embedded["path_sha256"]
        and DELEGATE.ENGINE.sha256_array(energy) == embedded["energy_sha256"]
        and DELEGATE.ENGINE.sha256_array(force) == embedded["force_sha256"],
        "selected R5B array hash drift",
    )
    recomputed_i = float(
        DELEGATE.ENGINE.path_diagnostics(
            path, np.zeros(N_IMAGES, dtype=np.float64), force[1:-1]
        )["i_perp"]
    )
    require(
        DELEGATE.same_float(recomputed_i, embedded["discrete_I_perp"]),
        "selected R5B discrete I_perp cannot be reproduced from full force",
    )
    require(
        np.all(force_requests[1:-1] > 0)
        and int(counters[1:-1].sum(dtype=np.int64)) == embedded["nf"],
        "selected R5B original32 force/N_F lineage drift",
    )
    endpoint = DELEGATE._load_endpoint_cache()
    require(
        np.array_equal(path[[0, -1]], endpoint["coordinates"])
        and np.array_equal(energy[[0, -1]], endpoint["energies"])
        and np.array_equal(force[[0, -1]], endpoint["forces"]),
        "selected R5B endpoints differ from sealed cache",
    )
    frozen = {
        "row_index": row_index,
        "label": f"IVM_call{embedded['call']}_NF{embedded['nf']}",
        "method": "IVM",
        "host_role": host_role,
        "selection_role": str(selected["selection_role"]),
        "optimization_N_F": embedded["nf"],
        "path_identifier": {
            "history_index": int(selected["history_index"]),
            "calls_per_image": embedded["call"],
            "optimization_N_F": embedded["nf"],
            "fully_evaluated_current_path": True,
            "reparameterization_committed": True,
            "stable_last": True,
            "formally_reportable": True,
            "terminal_uncommitted_charged_tail": False,
        },
        "path_sha256": embedded["path_sha256"],
        "source_full_state_shard": shard_binding,
        "source_energy_sha256": embedded["energy_sha256"],
        "source_force_sha256": embedded["force_sha256"],
        "discrete_I_perp": embedded["discrete_I_perp"],
        "discrete_I_perp_source": (
            "recomputed_from_R5B_source_full_state_force_eV_per_A_internal30"
        ),
        "source_counter_vector_sha256": DELEGATE.ENGINE.sha256_array(counters),
        "source_force_request_vector_sha256": DELEGATE.ENGINE.sha256_array(
            force_requests
        ),
        "original32_new_physical_calls": 0,
        "dense_policy": (
            "uniform_arc500_store_coordinates_energy_full_force_with_two_"
            "inherited_fixed_endpoints_and_498_fresh_joint_calls"
        ),
        "fresh_dense_joint_EF_calls": FRESH_PER_PATH,
        "optimization_N_F_contribution": 0,
    }
    runtime = {
        **frozen,
        "path": path,
        "source_counter_vector": counters,
        "source_force_request_vector": force_requests,
    }
    return frozen, runtime


def load_verified_selection(selection_path: Path) -> tuple[dict, dict[str, list]]:
    selection, selected = _selection_closure(selection_path)
    by_role = {str(row["selection_role"]): row for row in selected}
    split = SELECTOR.validate_split_template(
        SELECTOR.CANONICAL_SPLIT_TEMPLATE,
        SELECTOR.CANONICAL_PREREGISTRATION,
    )
    result: dict[str, list] = {}
    for host_role, key in (("remote", "remote_four"), ("local", "local_two")):
        roles = split[key]["selection_roles"]
        frozen_rows, runtime_rows = [], []
        for row_index, role in enumerate(roles):
            require(role in by_role, f"selection lacks assigned role {role}")
            frozen, runtime = _source_row(by_role[role], row_index, host_role)
            runtime.update(
                {
                    "result_path": Path(
                        selection["bindings"]["R5B_replay_result"]["path"]
                    ).resolve(strict=True),
                    "verification_path": Path(
                        selection["bindings"]["primary_R5B_verification"]["path"]
                    ).resolve(strict=True),
                    "audit_path": Path(
                        selection["bindings"]["R5B_capture_audit"]["path"]
                    ).resolve(strict=True),
                }
            )
            frozen_rows.append(frozen)
            runtime_rows.append(runtime)
        result[host_role] = [frozen_rows, runtime_rows]
    require(
        len(result["remote"][0]) == REMOTE_COUNT
        and len(result["local"][0]) == LOCAL_COUNT,
        "remote4/local2 selected row count drift",
    )
    return selection, result


def _mutagen_snapshot() -> dict:
    executable = MUTAGEN.expanduser().resolve(strict=True)
    flush = subprocess.run(
        [str(executable), "sync", "flush", SELECTOR.MUTAGEN["name"]],
        check=False,
        text=True,
        capture_output=True,
    )
    require(flush.returncode == 0, "dedicated Mutagen flush failed")
    listing = subprocess.run(
        [str(executable), "sync", "list", "--template", MUTAGEN_LIST_TEMPLATE],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [line.split("|") for line in listing.stdout.splitlines()]
    matches = [row for row in rows if row and row[0] == SELECTOR.MUTAGEN["name"]]
    require(len(matches) == 1 and len(matches[0]) == 5, "Mutagen session not unique")
    name, identifier, status, alpha, beta = matches[0]
    details = subprocess.run(
        [str(executable), "sync", "list", SELECTOR.MUTAGEN["name"]],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.lower()
    snapshot = {
        "session_name": name,
        "session_identifier": identifier,
        "status": status,
        "alpha_connected": alpha.lower() == "true",
        "beta_connected": beta.lower() == "true",
        "flush_exit_code": int(flush.returncode),
        "transition_problems": int(
            "problems:" in details or "conflicts:" in details or "error:" in details
        ),
        "mutagen_executable": bind(executable),
    }
    require(
        snapshot["session_name"] == SELECTOR.MUTAGEN["name"]
        and snapshot["session_identifier"] == SELECTOR.MUTAGEN["identifier"]
        and snapshot["status"] in {"Watching", "Watching for changes"}
        and snapshot["alpha_connected"] is True
        and snapshot["beta_connected"] is True
        and snapshot["transition_problems"] == 0,
        "Mutagen session gate failed",
    )
    return snapshot


def _gate_payload(stage: str, preparation_path: Path, **artifacts: Path) -> dict:
    preparation = validate_preparation(preparation_path)
    require(
        stage in {"input", "publication", "authorization"},
        "invalid Mutagen gate stage",
    )
    return {
        "schema": GATE_SCHEMA,
        "status": GATE_STATUS,
        "stage": stage,
        "created_utc": utc_now(),
        "OpenMM_imported_or_called": False,
        "snapshot": _mutagen_snapshot(),
        "artifacts": {name: bind(path) for name, path in artifacts.items()},
        "bindings": {
            "preparation": bind(preparation_path),
            "contract_freezer": preparation["bindings"]["contract_freezer"],
            "authorized_runner": preparation["bindings"]["authorized_runner"],
        },
    }


def validate_gate(path: Path, stage: str, preparation_path: Path, **artifacts: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    canonical = {
        "input": INPUT_GATE,
        "publication": PUBLICATION_GATE,
        "authorization": AUTHORIZATION_GATE,
    }[stage]
    require(resolved == canonical, f"noncanonical {stage} Mutagen gate")
    payload = read_json(resolved)
    require(
        payload.get("schema") == GATE_SCHEMA
        and payload.get("status") == GATE_STATUS
        and payload.get("stage") == stage
        and payload.get("OpenMM_imported_or_called") is False,
        f"invalid {stage} Mutagen gate",
    )
    expected_artifacts = {name: bind(value) for name, value in artifacts.items()}
    require(payload.get("artifacts") == expected_artifacts, f"{stage} gate artifact drift")
    preparation = validate_preparation(preparation_path)
    require(
        payload.get("bindings", {}).get("preparation") == bind(preparation_path)
        and payload["bindings"]["contract_freezer"]
        == preparation["bindings"]["contract_freezer"]
        and payload["bindings"]["authorized_runner"]
        == preparation["bindings"]["authorized_runner"],
        f"{stage} gate binding drift",
    )
    snapshot = payload.get("snapshot", {})
    require(
        snapshot.get("session_name") == SELECTOR.MUTAGEN["name"]
        and snapshot.get("session_identifier") == SELECTOR.MUTAGEN["identifier"]
        and snapshot.get("status") in {"Watching", "Watching for changes"}
        and snapshot.get("flush_exit_code") == 0
        and snapshot.get("alpha_connected") is True
        and snapshot.get("beta_connected") is True
        and snapshot.get("transition_problems") == 0,
        f"{stage} Mutagen snapshot drift",
    )
    return payload


def _contract_payload(
    preparation_path: Path,
    selection_path: Path,
    input_gate_path: Path,
    preparation: dict,
    selection: dict,
    host_role: str,
    frozen_rows: list,
) -> dict:
    plan = preparation["host_plans"][host_role]
    endpoint = DELEGATE._load_endpoint_cache()
    return {
        "schema": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "frozen": True,
        "created_utc": utc_now(),
        "host_role": host_role,
        "remote_identity": plan["host_identity"],
        "jobs": 1,
        "thread_caps": dict(SELECTOR.THREAD_CAPS),
        "result_output": plan["result_output"],
        "execution_authorization_output": plan["authorization_output"],
        "preflight_receipt_output": plan["preflight_output"],
        "bundle_output": str(BUNDLE),
        "publication_gate_output": str(PUBLICATION_GATE),
        "authorization_gate_output": str(AUTHORIZATION_GATE),
        "mutagen_gate_at_freeze": read_json(input_gate_path)["snapshot"],
        "host_local_lock_paths": list(SELECTOR.LOCKS),
        "selection_roles": plan["selection_roles"],
        "paths": frozen_rows,
        "endpoint_cache_arrays": {
            "coordinate_sha256": DELEGATE.ENGINE.sha256_array(
                endpoint["coordinates"]
            ),
            "energy_sha256": DELEGATE.ENGINE.sha256_array(endpoint["energies"]),
            "force_sha256": DELEGATE.ENGINE.sha256_array(endpoint["forces"]),
        },
        "physical_call_policy": {
            "selected_paths": len(frozen_rows),
            "original32_joint_EF_reused_per_path": N_IMAGES,
            "original32_new_physical_calls": 0,
            "stored_arc500_joint_EF_values_per_path": N_DENSE,
            "inherited_endpoint_joint_EF_values_per_path": 2,
            "fresh_dense_interior_joint_EF_calls_per_path": FRESH_PER_PATH,
            "maximum_fresh_attempt_started_records": len(frozen_rows)
            * FRESH_PER_PATH,
            "optimization_N_F_contribution": 0,
        },
        "storage_policy": {
            "authoritative_unit": "complete_atomic_path_shard",
            "arc500_coordinates_energy_full_force_stored": True,
            "source_original32_full_state_bound_per_path": True,
            "aggregate_is_index_and_metrics_only": True,
        },
        "durability_policy": {
            "append_only_fsync_WAL": True,
            "atomic_complete_path_shard": True,
            "complete_validated_shard_only_resume": True,
            "individual_incomplete_success_reuse": False,
            "open_failed_retried_attempt_requires_superseding_contract": True,
            "no_overwrite": True,
        },
        "selection_and_replay_evidence": {
            "selection": bind(selection_path),
            "all_paths_result": selection["bindings"]["R5B_replay_result"],
            "all_paths_audit": selection["bindings"]["R5B_capture_audit"],
            "exact_all_paths_verification": selection["bindings"][
                "primary_R5B_verification"
            ],
            "supplemental_verification": selection["bindings"][
                "supplemental_R5B_verification"
            ],
            "capture_manifest": selection["bindings"]["R5B_capture_manifest"],
            "canonical_index": selection["bindings"]["R5B_canonical_index"],
            "capture_WAL": selection["bindings"]["R5B_capture_WAL"],
        },
        "bindings": {
            **preparation["bindings"],
            "preparation": bind(preparation_path),
            "selection": bind(selection_path),
            "input_Mutagen_gate": bind(input_gate_path),
        },
        "optimization_N_F_contribution": 0,
    }


def freeze_contracts(
    preparation_path: Path,
    selection_path: Path,
    input_gate_path: Path,
    remote_contract: Path,
    local_contract: Path,
    bundle_path: Path,
) -> dict:
    preparation = validate_preparation(preparation_path)
    require(
        Path(remote_contract).resolve() == Path(
            preparation["host_plans"]["remote"]["contract_output"]
        ).resolve()
        and Path(local_contract).resolve()
        == Path(preparation["host_plans"]["local"]["contract_output"]).resolve()
        and Path(bundle_path).resolve() == BUNDLE,
        "noncanonical contract/bundle output",
    )
    selection, rows = load_verified_selection(selection_path)
    validate_gate(
        input_gate_path,
        "input",
        preparation_path,
        selection=selection_path,
    )
    outputs = [Path(remote_contract), Path(local_contract), Path(bundle_path)]
    require(not any(path.exists() for path in outputs), "contract output already exists")
    remote_payload = _contract_payload(
        preparation_path,
        selection_path,
        input_gate_path,
        preparation,
        selection,
        "remote",
        rows["remote"][0],
    )
    local_payload = _contract_payload(
        preparation_path,
        selection_path,
        input_gate_path,
        preparation,
        selection,
        "local",
        rows["local"][0],
    )
    write_exclusive(remote_contract, remote_payload)
    write_exclusive(local_contract, local_payload)
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "status": BUNDLE_STATUS,
        "frozen": True,
        "created_utc": utc_now(),
        "remote_contract": bind(remote_contract),
        "local_contract": bind(local_contract),
        "selection": bind(selection_path),
        "preparation": bind(preparation_path),
        "input_Mutagen_gate": bind(input_gate_path),
        "publication_Mutagen_gate_required": str(PUBLICATION_GATE),
        "role_partition": {
            "remote": preparation["host_plans"]["remote"]["selection_roles"],
            "local": preparation["host_plans"]["local"]["selection_roles"],
        },
        "fresh_joint_EF_calls": {
            "remote": REMOTE_COUNT * FRESH_PER_PATH,
            "local": LOCAL_COUNT * FRESH_PER_PATH,
            "total": (REMOTE_COUNT + LOCAL_COUNT) * FRESH_PER_PATH,
        },
        "optimization_N_F_contribution": 0,
    }
    write_exclusive(bundle_path, bundle)
    return bundle


def validate_bundle(
    path: Path,
    preparation_path: Path,
    remote_contract: Path | None = None,
    local_contract: Path | None = None,
) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    require(resolved == BUNDLE, "noncanonical R4 bundle")
    preparation = validate_preparation(preparation_path)
    remote = Path(
        remote_contract
        or preparation["host_plans"]["remote"]["contract_output"]
    ).resolve(strict=True)
    local = Path(
        local_contract or preparation["host_plans"]["local"]["contract_output"]
    ).resolve(strict=True)
    payload = read_json(resolved)
    require(
        payload.get("schema") == BUNDLE_SCHEMA
        and payload.get("status") == BUNDLE_STATUS
        and payload.get("frozen") is True
        and payload.get("remote_contract") == bind(remote)
        and payload.get("local_contract") == bind(local)
        and payload.get("preparation") == bind(preparation_path)
        and payload.get("selection") == bind(SELECTOR.CANONICAL_SELECTION)
        and payload.get("input_Mutagen_gate") == bind(INPUT_GATE)
        and payload.get("publication_Mutagen_gate_required")
        == str(PUBLICATION_GATE)
        and payload.get("optimization_N_F_contribution") == 0,
        "invalid R4 contract bundle",
    )
    require(
        payload.get("role_partition")
        == {
            "remote": preparation["host_plans"]["remote"]["selection_roles"],
            "local": preparation["host_plans"]["local"]["selection_roles"],
        }
        and payload.get("fresh_joint_EF_calls")
        == {"remote": 1992, "local": 996, "total": 2988},
        "bundle role/call partition drift",
    )
    return payload


def load_host_state(
    contract_path: Path,
    bundle_path: Path,
    publication_gate_path: Path,
    host_role: str,
) -> dict:
    preparation = validate_preparation(PREPARATION)
    bundle = validate_bundle(bundle_path, PREPARATION)
    expected_contract = Path(
        preparation["host_plans"][host_role]["contract_output"]
    ).resolve(strict=True)
    contract_path = Path(contract_path).expanduser().resolve(strict=True)
    require(contract_path == expected_contract, "noncanonical host contract")
    require(
        bundle[f"{host_role}_contract"] == bind(contract_path),
        "bundle/host contract binding drift",
    )
    validate_gate(
        publication_gate_path,
        "publication",
        PREPARATION,
        bundle=bundle_path,
        remote_contract=Path(
            preparation["host_plans"]["remote"]["contract_output"]
        ),
        local_contract=Path(
            preparation["host_plans"]["local"]["contract_output"]
        ),
    )
    contract = read_json(contract_path)
    require(
        contract.get("schema") == CONTRACT_SCHEMA
        and contract.get("status") == CONTRACT_STATUS
        and contract.get("frozen") is True
        and contract.get("host_role") == host_role
        and contract.get("jobs") == 1
        and contract.get("thread_caps") == SELECTOR.THREAD_CAPS,
        "invalid host contract",
    )
    selection, rows = load_verified_selection(SELECTOR.CANONICAL_SELECTION)
    frozen, runtime = rows[host_role]
    require(contract.get("paths") == frozen, "host contract selected rows drift")
    plan = preparation["host_plans"][host_role]
    require(
        contract.get("selection_roles") == plan["selection_roles"]
        and contract.get("remote_identity") == plan["host_identity"]
        and contract.get("result_output") == plan["result_output"]
        and contract.get("execution_authorization_output")
        == plan["authorization_output"]
        and contract.get("preflight_receipt_output") == plan["preflight_output"]
        and contract.get("bundle_output") == str(BUNDLE)
        and contract.get("publication_gate_output") == str(PUBLICATION_GATE)
        and contract.get("authorization_gate_output") == str(AUTHORIZATION_GATE)
        and contract.get("host_local_lock_paths") == SELECTOR.LOCKS,
        "host contract path/resource drift",
    )
    physical = contract.get("physical_call_policy", {})
    require(
        physical.get("selected_paths") == len(frozen)
        and physical.get("original32_new_physical_calls") == 0
        and physical.get("fresh_dense_interior_joint_EF_calls_per_path")
        == FRESH_PER_PATH
        and physical.get("maximum_fresh_attempt_started_records")
        == len(frozen) * FRESH_PER_PATH
        and physical.get("optimization_N_F_contribution") == 0,
        "host contract call accounting drift",
    )
    endpoint = DELEGATE._load_endpoint_cache()
    expected_endpoint = {
        "coordinate_sha256": DELEGATE.ENGINE.sha256_array(endpoint["coordinates"]),
        "energy_sha256": DELEGATE.ENGINE.sha256_array(endpoint["energies"]),
        "force_sha256": DELEGATE.ENGINE.sha256_array(endpoint["forces"]),
    }
    require(
        contract.get("endpoint_cache_arrays") == expected_endpoint,
        "host contract endpoint cache drift",
    )
    evidence = contract["selection_and_replay_evidence"]
    expected_evidence = {
        "selection": bind(SELECTOR.CANONICAL_SELECTION),
        "all_paths_result": selection["bindings"]["R5B_replay_result"],
        "all_paths_audit": selection["bindings"]["R5B_capture_audit"],
        "exact_all_paths_verification": selection["bindings"][
            "primary_R5B_verification"
        ],
        "supplemental_verification": selection["bindings"][
            "supplemental_R5B_verification"
        ],
        "capture_manifest": selection["bindings"]["R5B_capture_manifest"],
        "canonical_index": selection["bindings"]["R5B_canonical_index"],
        "capture_WAL": selection["bindings"]["R5B_capture_WAL"],
    }
    require(
        evidence == expected_evidence,
        "host contract R5B evidence drift",
    )
    expected_bindings = {
        **preparation["bindings"],
        "preparation": bind(PREPARATION),
        "selection": bind(SELECTOR.CANONICAL_SELECTION),
        "input_Mutagen_gate": bind(INPUT_GATE),
    }
    require(
        contract.get("bindings") == expected_bindings,
        "host contract source binding drift",
    )
    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_sha256": SELECTOR.sha256_file(contract_path),
        "selection_path": SELECTOR.CANONICAL_SELECTION,
        "rows": runtime,
        "endpoint": endpoint,
        "topology": Path(contract["bindings"]["topology"]["path"]).resolve(
            strict=True
        ),
        "locks": [Path(value).resolve() for value in contract["host_local_lock_paths"]],
        "result_output": Path(contract["result_output"]).resolve(),
        "host_role": host_role,
        "bundle": bundle,
    }


def check_host(expected: dict) -> dict:
    observed = {
        "hostname": socket.gethostname(),
        "platform": platform.system(),
        "machine": platform.machine(),
    }
    require(
        observed["hostname"] == expected["hostname"],
        f"wrong execution host: {observed['hostname']}",
    )
    require(
        observed["platform"] == expected["platform"],
        f"wrong execution platform: {observed['platform']}",
    )
    require(
        observed["machine"] == expected["machine"],
        f"wrong execution machine: {observed['machine']}",
    )
    return observed


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    input_gate = modes.add_parser("mutagen-input-gate")
    input_gate.add_argument("--preparation", type=Path, required=True)
    input_gate.add_argument("--selection", type=Path, required=True)
    input_gate.add_argument("--output", type=Path, required=True)
    freeze = modes.add_parser("freeze-contracts")
    freeze.add_argument("--preparation", type=Path, required=True)
    freeze.add_argument("--selection", type=Path, required=True)
    freeze.add_argument("--input-gate", type=Path, required=True)
    freeze.add_argument("--remote-contract", type=Path, required=True)
    freeze.add_argument("--local-contract", type=Path, required=True)
    freeze.add_argument("--bundle", type=Path, required=True)
    publication = modes.add_parser("mutagen-publication-gate")
    publication.add_argument("--preparation", type=Path, required=True)
    publication.add_argument("--bundle", type=Path, required=True)
    publication.add_argument("--remote-contract", type=Path, required=True)
    publication.add_argument("--local-contract", type=Path, required=True)
    publication.add_argument("--output", type=Path, required=True)
    authorization = modes.add_parser("mutagen-authorization-gate")
    authorization.add_argument("--preparation", type=Path, required=True)
    authorization.add_argument("--bundle", type=Path, required=True)
    authorization.add_argument("--remote-authorization", type=Path, required=True)
    authorization.add_argument("--local-authorization", type=Path, required=True)
    authorization.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.mode == "prepare":
        require(args.output.resolve() == PREPARATION, "noncanonical preparation output")
        write_exclusive(args.output, build_preparation())
        print(args.output.resolve(strict=True))
        print(SELECTOR.sha256_file(args.output.resolve(strict=True)))
    elif args.mode == "mutagen-input-gate":
        require(args.output.resolve() == INPUT_GATE, "noncanonical input gate output")
        _selection_closure(args.selection)
        write_exclusive(
            args.output,
            _gate_payload(
                "input", args.preparation, selection=args.selection
            ),
        )
        print(args.output.resolve(strict=True))
        print(SELECTOR.sha256_file(args.output.resolve(strict=True)))
    elif args.mode == "freeze-contracts":
        payload = freeze_contracts(
            args.preparation,
            args.selection,
            args.input_gate,
            args.remote_contract,
            args.local_contract,
            args.bundle,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.mode == "mutagen-publication-gate":
        require(
            args.output.resolve() == PUBLICATION_GATE,
            "noncanonical publication gate output",
        )
        validate_bundle(
            args.bundle,
            args.preparation,
            args.remote_contract,
            args.local_contract,
        )
        write_exclusive(
            args.output,
            _gate_payload(
                "publication",
                args.preparation,
                bundle=args.bundle,
                remote_contract=args.remote_contract,
                local_contract=args.local_contract,
            ),
        )
        print(args.output.resolve(strict=True))
        print(SELECTOR.sha256_file(args.output.resolve(strict=True)))
    else:
        require(
            args.output.resolve() == AUTHORIZATION_GATE,
            "noncanonical authorization gate output",
        )
        validate_bundle(args.bundle, args.preparation)
        split = SELECTOR.validate_split_template(
            SELECTOR.CANONICAL_SPLIT_TEMPLATE,
            SELECTOR.CANONICAL_PREREGISTRATION,
        )
        paths = _template_paths(split)
        require(
            args.remote_authorization.resolve(strict=True)
            == paths["remote_authorization"]
            and args.local_authorization.resolve(strict=True)
            == paths["local_authorization"],
            "noncanonical host authorizations",
        )
        write_exclusive(
            args.output,
            _gate_payload(
                "authorization",
                args.preparation,
                bundle=args.bundle,
                remote_authorization=args.remote_authorization,
                local_authorization=args.local_authorization,
            ),
        )
        print(args.output.resolve(strict=True))
        print(SELECTOR.sha256_file(args.output.resolve(strict=True)))


if __name__ == "__main__":
    main()
