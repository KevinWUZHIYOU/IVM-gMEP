#!/usr/bin/env python3
"""Build the formal BB IR-IVM20k/ODE-NEB30k discrete cost figure.

This is a zero-physical-call post-processing program.  It recomputes all
three quantities from the already stored 32-image Cartesian path, energy,
and full-Force histories:

* central-difference/Voronoi ``I_perp``;
* ``Delta E = max_i E_i - E_0``; and
* ``S_hat = integral_path |grad V| d ell``, evaluated by the trapezoidal
  rule on the 31 native path segments.

The IR-IVM history ends at the latest accepted state at or below 20k
optimization Forces (call 693, N_F=19993).  The ODE-NEB history contains all
1000 paid 30-image batches through N_F=30000.  Fixed endpoints are inherited
and contribute zero optimization Forces.  No path is densified or smoothed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


sys.dont_write_bytecode = True

WORKSPACE = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP")
DATA_ROOT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901"
)
ANALYSIS_ROOT = WORKSPACE / (
    "code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis"
)
ODE_SOURCE_ROOT = WORKSPACE / (
    "code_on_server/bb_neb_aseode30_kgrid_nf6000_nf30000_20260903_r1"
)
ODE_RUN = WORKSPACE / (
    "code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/"
    "runs/fresh-nonresume-bb-neb-aseode-k0p1-formal-nf30000-remote-20260903-r1"
)
ODE_FULL_STATE = ODE_RUN / "paid_batch_full_state"
ODE_CACHE = ANALYSIS_ROOT / (
    "formal_discrete32_cache_r2_audited_v2_local_mutagen/"
    "formal_discrete32_cache_FINAL.json"
)

R5_PRIMARY = DATA_ROOT / (
    "reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B_PRIMARY.json"
)
R5_SUPPLEMENTAL = DATA_ROOT / (
    "reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B_SUPPLEMENTAL.json"
)
R6_PRIMARY = DATA_ROOT / (
    "reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF30000_20260904_R6B_PRIMARY.json"
)
R6_SUPPLEMENTAL = DATA_ROOT / (
    "reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF30000_20260904_R6B_SUPPLEMENTAL.json"
)
R5_SHARDS = DATA_ROOT / (
    "runs/full_state_history_replay_r5b/"
    "bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b_observer/shards"
)
R6_SUFFIX_METRICS = DATA_ROOT / (
    "reports/plot_inputs/bb_irivm_r6b_calls0516_0693_20260905_r1/"
    "r6b_calls0516_0693_native32_metrics.json"
)
R6_SUFFIX_GENERATOR = ANALYSIS_ROOT / "generate_bb_irivm_r6b_20k_suffix_metrics_remote.py"

OUTPUT_STEM = WORKSPACE / "output/pdf/barnase_barstar_irivm20k_ode30k_cost_to_precision_1x3"
PDF = OUTPUT_STEM.with_suffix(".pdf")
DATA_JSON = OUTPUT_STEM.with_name(OUTPUT_STEM.name + ".data.json")
PROVENANCE_JSON = OUTPUT_STEM.with_name(OUTPUT_STEM.name + ".provenance.json")
TRACEABILITY_MD = OUTPUT_STEM.with_name(OUTPUT_STEM.name + ".traceability.md")

N_IMAGES = 32
N_DOF = 9396
IR_TARGET_NF = 20000
IR_FINAL_CALL = 693
IR_FINAL_NF = 19993
ODE_FINAL_BATCH = 1000
ODE_FINAL_NF = 30000
EV_TO_KCAL_PER_MOL = 23.0605
IVM_COLOR = "#0072B2"
ODE_COLOR = "#D55E00"

R5_PRIMARY_SCHEMA = "bb-irivm-selected-full-state-history-verification-v5b"
R5_PRIMARY_STATUS = "PASS_R5B_EXACT_PREFIX318_AND_DYNAMIC_STABLE_LAST_FULL_STATE_REPLAY"
R5_SUPPLEMENTAL_SCHEMA = "bb-irivm-full-state-r5b-supplemental-verification-v1"
R5_SUPPLEMENTAL_STATUS = "PASS_R5B_EXACT_COUNTER_SCHEMA_AND_ARTIFACT_CLOSURE"
R6_PRIMARY_SCHEMA = "bb-irivm-selected-full-state-history-verification-v6b"
R6_PRIMARY_STATUS = "PASS_R6B_NF30000_EXACT_PREFIX318_AND_FULL_STATE_CAPTURE"
R6_SUPPLEMENTAL_SCHEMA = "bb-irivm-full-state-r6b-supplemental-verification-v1"
R6_SUPPLEMENTAL_STATUS = "PASS_R6B_NF30000_COUNTER_BUDGET_LOCK_AND_ARTIFACT_CLOSURE"

DATA_SCHEMA = "bb-irivm20k-ode30k-discrete-cost-data-v1"
PROVENANCE_SCHEMA = "bb-irivm20k-ode30k-discrete-cost-provenance-v1"
FORMULA_ID = "native32-cartesian-trapezoid-norm-force-action-v1"


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def scalar(archive: Mapping[str, Any], key: str) -> Any:
    require(key in archive, f"archive lacks {key}")
    value = np.asarray(archive[key])
    require(value.shape in {(), (1,)}, f"{key} is not scalar-like")
    return value.reshape(()).item()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
    )


def metric_triplet(path32: Any, energy32: Any, force32: Any) -> tuple[float, float, float, dict[str, float]]:
    """Evaluate the three metrics on exactly the same native 32-image state."""

    path = np.ascontiguousarray(np.asarray(path32, dtype=np.float64)).reshape(N_IMAGES, N_DOF)
    energy = np.ascontiguousarray(np.asarray(energy32, dtype=np.float64)).reshape(N_IMAGES)
    force = np.ascontiguousarray(np.asarray(force32, dtype=np.float64)).reshape(N_IMAGES, N_DOF)
    require(np.isfinite(path).all() and np.isfinite(energy).all() and np.isfinite(force).all(),
            "metric state contains non-finite values")

    tangent = path[2:] - path[:-2]
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    require(np.all(tangent_norm > 1.0e-30), "central-difference tangent is degenerate")
    tangent /= tangent_norm
    internal_force = force[1:-1]
    perpendicular = internal_force - np.sum(
        internal_force * tangent, axis=1, keepdims=True
    ) * tangent
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    require(segment.shape == (31,) and np.all(segment > 0.0), "native segment is degenerate")
    weights = 0.5 * (segment[:-1] + segment[1:])
    i_perp = float(np.sum(weights * np.linalg.norm(perpendicular, axis=1) ** 2))

    delta_e_eV = float(np.max(energy) - energy[0])
    force_norm = np.linalg.norm(force, axis=1)
    action_eV = float(np.sum(0.5 * (force_norm[:-1] + force_norm[1:]) * segment))
    require(
        math.isfinite(i_perp) and i_perp >= 0.0
        and math.isfinite(delta_e_eV) and delta_e_eV >= 0.0
        and math.isfinite(action_eV) and action_eV > 0.0,
        "computed metric is invalid",
    )
    diagnostics = {
        "arc_length_A": float(np.sum(segment)),
        "minimum_segment_length_A": float(np.min(segment)),
        "maximum_segment_length_A": float(np.max(segment)),
        "minimum_force_norm_eV_per_A": float(np.min(force_norm)),
        "maximum_force_norm_eV_per_A": float(np.max(force_norm)),
    }
    return i_perp, delta_e_eV, action_eV, diagnostics


def validate_binding_record(record: Any, role: str) -> None:
    require(isinstance(record, Mapping), f"{role} binding is absent")
    path = Path(str(record.get("path", ""))).resolve(strict=True)
    require(sha256_file(path) == str(record.get("sha256", "")), f"{role} binding drift")


def validate_ir_verifiers() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    r5 = load_json(R5_PRIMARY)
    r5s = load_json(R5_SUPPLEMENTAL)
    r6 = load_json(R6_PRIMARY)
    r6s = load_json(R6_SUPPLEMENTAL)
    require(
        r5.get("schema") == R5_PRIMARY_SCHEMA and r5.get("status") == R5_PRIMARY_STATUS,
        "R5B primary verifier is not the selected PASS artifact",
    )
    require(
        r5s.get("schema") == R5_SUPPLEMENTAL_SCHEMA
        and r5s.get("status") == R5_SUPPLEMENTAL_STATUS,
        "R5B supplemental verifier is not PASS",
    )
    require(
        r6.get("schema") == R6_PRIMARY_SCHEMA and r6.get("status") == R6_PRIMARY_STATUS,
        "R6B primary verifier is not the selected PASS artifact",
    )
    require(
        r6s.get("schema") == R6_SUPPLEMENTAL_SCHEMA
        and r6s.get("status") == R6_SUPPLEMENTAL_STATUS,
        "R6B supplemental verifier is not PASS",
    )
    require(
        int(r5.get("actual_canonical_call_count", -1)) == 515
        and len(r5.get("path_inventory", [])) == 515,
        "R5B inventory length differs",
    )
    require(
        int(r6.get("actual_canonical_call_count", -1)) == 1053
        and int(r6.get("last_complete_call", -1)) == 818
        and int(r6.get("terminal_optimization_N_F", -1)) == 29978
        and len(r6.get("path_inventory", [])) == 1053,
        "R6B terminal partition differs",
    )
    for role, value in (("R5B", r5), ("R6B", r6)):
        require(
            value.get("additional_physical_calls_from_capture") == 0
            and value.get("optimization_N_F_contribution_from_capture") == 0,
            f"{role} capture added physical calls",
        )
    for role, value in (("R5B supplemental", r5s), ("R6B supplemental", r6s)):
        validate_binding_record(value.get("bindings", {}).get("primary_verification"), role)
    return r5, r5s, r6, r6s


def load_ir_history() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    r5, _r5s, r6, _r6s = validate_ir_verifiers()
    r5_inventory = r5["path_inventory"]
    r6_inventory = r6["path_inventory"]
    rows = [
        row for row in r6_inventory
        if row.get("formally_reportable") is True
        and int(row["optimization_N_F"]) <= IR_TARGET_NF
    ]
    require(
        len(rows) == IR_FINAL_CALL
        and int(rows[-1]["call_per_image"]) == IR_FINAL_CALL
        and int(rows[-1]["optimization_N_F"]) == IR_FINAL_NF,
        "IR-IVM 20k accepted prefix does not end at call693/NF19993",
    )
    require(
        [int(row["call_per_image"]) for row in rows] == list(range(1, IR_FINAL_CALL + 1)),
        "IR-IVM 20k call axis is not continuous",
    )

    suffix = load_json(R6_SUFFIX_METRICS)
    require(
        suffix.get("schema") == "bb-irivm-r6b-native32-metric-suffix-v1"
        and suffix.get("status") == "PASS_ZERO_OPENMM_ZERO_NEW_NOF"
        and int(suffix.get("start_call", -1)) == 516
        and int(suffix.get("end_call", -1)) == 693
        and suffix.get("physical_calls") == 0
        and suffix.get("optimization_N_F_contribution") == 0,
        "R6B compact metric suffix is not the zero-call calls516--693 artifact",
    )
    suffix_generator = suffix.get("generator", {})
    require(
        str(suffix_generator.get("canonical_path")) == str(R6_SUFFIX_GENERATOR)
        and str(suffix_generator.get("sha256")) == sha256_file(R6_SUFFIX_GENERATOR),
        "R6B compact metric generator binding differs",
    )
    suffix_primary = suffix.get("primary_verification", {})
    require(
        str(suffix_primary.get("path")) == str(R6_PRIMARY)
        and str(suffix_primary.get("sha256")) == sha256_file(R6_PRIMARY),
        "R6B compact suffix primary-verification binding differs",
    )
    suffix_rows = suffix.get("rows")
    require(
        isinstance(suffix_rows, list) and len(suffix_rows) == 178
        and [int(row.get("call_per_image", -1)) for row in suffix_rows]
        == list(range(516, 694)),
        "R6B compact suffix row axis differs",
    )
    suffix_by_call = {int(row["call_per_image"]): row for row in suffix_rows}

    output: list[dict[str, Any]] = []
    previous_nf = -1
    r5_transport_count = 0
    r6_transport_count = 0
    for row in rows:
        call = int(row["call_per_image"])
        if call <= 515:
            r5_row = r5_inventory[call - 1]
            require(
                all(
                    str(r5_row[key]) == str(row[key])
                    for key in ("optimization_N_F", "path_sha256", "energy_sha256", "force_sha256")
                ),
                f"R5B transport arrays differ from the R6B accepted prefix at call {call}",
            )
            shard = R5_SHARDS / f"call_{call:04d}_stable_last.npz"
            expected_shard_sha256 = str(r5_row["shard_sha256"])
            transport = "bit-identical R5B local full-state shard"
            r5_transport_count += 1
            shard = shard.resolve(strict=True)
            require(sha256_file(shard) == expected_shard_sha256, f"IR shard drift at call {call}")
            with np.load(shard, allow_pickle=False) as saved:
                path = np.asarray(saved["path32_A"], dtype=np.float64)
                energy = np.asarray(saved["energy_eV"], dtype=np.float64)
                force = np.asarray(saved["force_eV_per_A"], dtype=np.float64)
                nf = int(scalar(saved, "optimization_N_F"))
                saved_call = int(scalar(saved, "call_per_image"))
                stored_i = float(scalar(saved, "discrete_I_perp"))
                additional_calls = int(scalar(saved, "additional_physical_calls"))
            require(saved_call == call and nf == int(row["optimization_N_F"]),
                    f"IR call/NF identity drift at call {call}")
            require(additional_calls == 0, f"IR capture reports extra calls at call {call}")
            require(
                sha256_array(path) == str(row["path_sha256"])
                and sha256_array(energy) == str(row["energy_sha256"])
                and sha256_array(force) == str(row["force_sha256"]),
                f"IR path/E/Force identity drift at call {call}",
            )
            i_perp, delta_e_eV, action_eV, diagnostics = metric_triplet(path, energy, force)
            require(
                np.asarray(i_perp).tobytes() == np.asarray(stored_i).tobytes(),
                f"IR stored and recomputed I_perp differ at call {call}",
            )
            compact_row = {
                "method": "IVM",
                "algorithm": "IR-IVM (inexact + equal-arclength reparameterization)",
                "optimization_N_F": nf,
                "call_per_image": call,
                "i_perp": i_perp,
                "delta_e_eV": delta_e_eV,
                "delta_e_kcal_per_mol": delta_e_eV * EV_TO_KCAL_PER_MOL,
                "geometric_action_eV": action_eV,
                "geometric_action_kcal_per_mol": action_eV * EV_TO_KCAL_PER_MOL,
                "path_sha256": str(row["path_sha256"]),
                "energy_sha256": str(row["energy_sha256"]),
                "force_sha256": str(row["force_sha256"]),
                "source_shard": str(shard),
                "source_shard_sha256": expected_shard_sha256,
                "source_transport": transport,
                "formally_reportable_under_R6B": True,
                **diagnostics,
            }
        else:
            compact_row = dict(suffix_by_call[call])
            require(
                int(compact_row["optimization_N_F"]) == int(row["optimization_N_F"])
                and compact_row.get("formally_reportable_under_R6B") is True
                and str(compact_row["source_shard_sha256"]) == str(row["shard_sha256"])
                and str(compact_row["path_sha256"]) == str(row["path_sha256"])
                and str(compact_row["energy_sha256"]) == str(row["energy_sha256"])
                and str(compact_row["force_sha256"]) == str(row["force_sha256"]),
                f"R6B compact row/source identity drift at call {call}",
            )
            for key in (
                "i_perp", "delta_e_eV", "delta_e_kcal_per_mol",
                "geometric_action_eV", "geometric_action_kcal_per_mol",
            ):
                require(math.isfinite(float(compact_row[key])), f"non-finite compact metric {key}")
            require(
                np.asarray(float(compact_row["delta_e_kcal_per_mol"])).tobytes()
                == np.asarray(float(compact_row["delta_e_eV"]) * EV_TO_KCAL_PER_MOL).tobytes()
                and np.asarray(float(compact_row["geometric_action_kcal_per_mol"])).tobytes()
                == np.asarray(float(compact_row["geometric_action_eV"]) * EV_TO_KCAL_PER_MOL).tobytes(),
                f"R6B compact unit conversion drift at call {call}",
            )
            nf = int(compact_row["optimization_N_F"])
            r6_transport_count += 1
        require(nf >= previous_nf, f"IR Number-of-Forces axis decreased at call {call}")
        output.append(compact_row)
        previous_nf = nf
    return output, {
        "selected_call_count": len(output),
        "selected_last_call": int(output[-1]["call_per_image"]),
        "selected_last_N_F": int(output[-1]["optimization_N_F"]),
        "R5B_local_transport_shards": r5_transport_count,
        "R6B_remote_compact_metric_rows": r6_transport_count,
        "R6B_compact_metric_ledger": binding(R6_SUFFIX_METRICS),
        "R6B_formal_last_complete_call": int(r6["last_complete_call"]),
        "R6B_formal_last_complete_N_F": int(r6_inventory[int(r6["last_complete_call"]) - 1]["optimization_N_F"]),
        "R6B_terminal_uncommitted_tail_excluded": True,
    }


def import_recorder():
    source = str(ODE_SOURCE_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    recorder = importlib.import_module("src.full_batch_recorder")
    require(
        Path(recorder.__file__).resolve(strict=True)
        == (ODE_SOURCE_ROOT / "src/full_batch_recorder.py").resolve(strict=True),
        "noncanonical ODE full-state decoder imported",
    )
    return recorder


def decode_ode_shard(path: Path, previous: tuple[np.ndarray, ...] | None,
                     *, checkpoint: bool, recorder):
    """Decode the exact Mutagen transport replica with bit-hash validation."""

    require(path.is_file() and not path.is_symlink(), f"unsafe ODE shard: {path}")
    array_keys = {"paid_path32", "energy32", "full_force32", "current_path32"}
    expected_keys = {
        "schema", "batch_id", "optimization_N_F", "spring_k_eV_per_A2",
        "role", "disposition", "encoding", "block_size", "previous_batch_id",
        "current_source_batch_id", *array_keys, "paid_path_sha256",
        "energy32_sha256", "full_force32_sha256", "current_path_sha256",
        "declared_current_state_sha256", "endpoint_cache_sha256",
        "endpoint_values_inherited", "endpoint_optimization_N_F",
        "additional_OpenMM_calls",
    }
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == expected_keys, f"ODE shard schema fields changed: {path.name}")
        metadata = {
            key: np.asarray(archive[key]).reshape(()).item()
            for key in expected_keys - array_keys
        }
        encoded = tuple(
            np.asarray(archive[key])
            for key in ("paid_path32", "energy32", "full_force32", "current_path32")
        )
    require(str(metadata["schema"]) == str(recorder.SHARD_SCHEMA), "ODE shard schema changed")
    require(int(metadata["block_size"]) == int(recorder.BLOCK_SIZE), "ODE block size changed")
    if checkpoint:
        require(previous is None and str(metadata["encoding"]) == "full_float64",
                "ODE checkpoint encoding changed")
        arrays = tuple(np.ascontiguousarray(value, dtype="<f8") for value in encoded)
    else:
        require(previous is not None and str(metadata["encoding"]) == "xor_uint64",
                "ODE delta encoding changed")
        arrays = tuple(
            np.ascontiguousarray(
                np.bitwise_xor(np.asarray(value, dtype=np.uint64), prior.view(np.uint64)).view("<f8")
            )
            for value, prior in zip(encoded, previous)
        )
    for array, key in zip(
        arrays,
        ("paid_path_sha256", "energy32_sha256", "full_force32_sha256", "current_path_sha256"),
    ):
        require(recorder.array_sha256(array) == str(metadata[key]),
                f"ODE decoded array hash differs for {key}")
    return arrays, metadata


def canonical_row_hash(row: Mapping[str, Any]) -> str:
    unsigned = dict(row)
    value = str(unsigned.pop("row_sha256", ""))
    measured = hashlib.sha256(json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False,
    ).encode("utf-8")).hexdigest()
    require(measured == value, f"ODE cache row hash drift at batch {row.get('batch_id')}")
    return value


def load_ode_history() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = load_json(ODE_CACHE)
    require(
        cache.get("schema") == "bb-ode-formal-discrete32-incremental-cache-v1"
        and cache.get("status") == "FINAL_BOUND_TO_CANONICAL_TERMINAL_AUDIT"
        and cache.get("partial_is_final") is False
        and int(cache.get("processed_batches", -1)) == ODE_FINAL_BATCH
        and int(cache.get("processed_optimization_N_F", -1)) == ODE_FINAL_NF
        and float(cache.get("spring_k_eV_per_A2", -1.0)) == 0.1
        and cache.get("physical_calls_started") == 0
        and cache.get("openmm_contexts_constructed") == 0
        and cache.get("bb_locks_acquired") == 0,
        "ODE final discrete cache is not the zero-call NF30000 PASS artifact",
    )
    require(Path(str(cache.get("run_dir", ""))).resolve() == ODE_RUN.resolve(),
            "ODE cache run directory differs")
    terminal = cache.get("canonical_terminal_audit")
    validate_binding_record(terminal, "ODE canonical terminal audit")
    rows = cache.get("rows")
    require(
        isinstance(rows, list) and len(rows) == ODE_FINAL_BATCH
        and [int(row.get("optimization_N_F", -1)) for row in rows]
        == list(range(30, ODE_FINAL_NF + 1, 30)),
        "ODE cache budget axis differs",
    )

    recorder = import_recorder()
    previous_arrays = None
    current = None
    current_source_batch_id = 0
    current_metrics = None
    result: list[dict[str, Any]] = []
    accepted = 0
    rejected = 0
    for batch_id, row in enumerate(rows, 1):
        canonical_row_hash(row)
        require(int(row.get("batch_id", -1)) == batch_id, f"ODE batch identity drift at {batch_id}")
        checkpoint = (batch_id - 1) % int(recorder.BLOCK_SIZE) == 0
        if checkpoint:
            previous_arrays = None
        shard = ODE_FULL_STATE / f"batch_{batch_id:04d}_nf{30 * batch_id:05d}.npz"
        shard = shard.resolve(strict=True)
        require(sha256_file(shard) == str(row["source_artifact_sha256"]),
                f"ODE shard file hash drift at batch {batch_id}")
        arrays, metadata = decode_ode_shard(
            shard, previous_arrays, checkpoint=checkpoint, recorder=recorder
        )
        previous_arrays = arrays
        paid_path, energy, force, current_path = arrays
        require(
            int(metadata["batch_id"]) == batch_id
            and int(metadata["optimization_N_F"]) == 30 * batch_id
            and int(metadata["additional_OpenMM_calls"]) == 0,
            f"ODE decoded batch/NF/call identity drift at batch {batch_id}",
        )
        disposition = str(metadata["disposition"])
        require(disposition == str(row["disposition"]), f"ODE disposition drift at batch {batch_id}")
        if disposition in {"current_initial", "accepted"}:
            current = (paid_path, energy, force)
            current_metrics = metric_triplet(*current)
            current_source_batch_id = batch_id
            accepted += 1
        else:
            require(disposition == "rejected", f"unknown ODE disposition at batch {batch_id}")
            rejected += 1
        require(current is not None and current_metrics is not None, "ODE has no current state")
        require(
            np.array_equal(current_path, current[0])
            and int(metadata["current_source_batch_id"]) == current_source_batch_id
            and int(row["current_source_batch_id"]) == current_source_batch_id,
            f"ODE rejected/accepted current-state mapping drift at batch {batch_id}",
        )
        path_hash = sha256_array(current[0])
        energy_hash = sha256_array(current[1])
        force_hash = sha256_array(current[2])
        require(
            path_hash == str(row["current_path_sha256"])
            and energy_hash == str(row["current_energy_sha256"])
            and force_hash == str(row["current_force_sha256"]),
            f"ODE current path/E/Force identity drift at batch {batch_id}",
        )
        i_perp, delta_e_eV, action_eV, diagnostics = current_metrics
        require(
            np.asarray(i_perp).tobytes() == np.asarray(float(row["i_perp"])).tobytes()
            and np.asarray(delta_e_eV).tobytes()
            == np.asarray(float(row["delta_e32_eV"])).tobytes(),
            f"ODE cached and recomputed I_perp/DeltaE differ at batch {batch_id}",
        )
        source_shard = ODE_FULL_STATE / (
            f"batch_{current_source_batch_id:04d}_nf{30 * current_source_batch_id:05d}.npz"
        )
        result.append({
            "method": "ODE-NEB",
            "optimization_N_F": 30 * batch_id,
            "batch_id": batch_id,
            "disposition": disposition,
            "current_source_batch_id": current_source_batch_id,
            "i_perp": i_perp,
            "delta_e_eV": delta_e_eV,
            "delta_e_kcal_per_mol": delta_e_eV * EV_TO_KCAL_PER_MOL,
            "geometric_action_eV": action_eV,
            "geometric_action_kcal_per_mol": action_eV * EV_TO_KCAL_PER_MOL,
            "path_sha256": path_hash,
            "energy_sha256": energy_hash,
            "force_sha256": force_hash,
            "paid_source_shard": str(shard),
            "paid_source_shard_sha256": str(row["source_artifact_sha256"]),
            "current_source_shard": str(source_shard),
            "cache_row_sha256": str(row["row_sha256"]),
            **diagnostics,
        })
    require(result[-1]["optimization_N_F"] == ODE_FINAL_NF, "ODE history does not end at 30k")
    return result, {
        "batches": len(result),
        "last_N_F": int(result[-1]["optimization_N_F"]),
        "accepted_or_initial_batches": accepted,
        "rejected_paid_batches": rejected,
        "spring_k_eV_per_A2": float(cache["spring_k_eV_per_A2"]),
        "terminal_audit": terminal,
        "transport": cache.get("local_mutagen_replica_transport"),
    }


def lower_frontier(rows: Sequence[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    best = math.inf
    output: list[dict[str, Any]] = []
    for row in rows:
        value = float(row[metric])
        tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(best) if math.isfinite(best) else 1.0)
        if value < best - tolerance:
            output.append(dict(row))
            best = value
    require(output, f"empty lower frontier for {metric}")
    return output


def first_passage(rows: Sequence[Mapping[str, Any]], metric: str, threshold: float) -> int:
    for row in rows:
        if float(row[metric]) <= threshold:
            return int(row["optimization_N_F"])
    raise AnalysisError(f"no first passage for {metric} at {threshold}")


def build_data() -> dict[str, Any]:
    ir, ir_summary = load_ir_history()
    ode, ode_summary = load_ode_history()
    methods = {"IVM": ir, "ODE-NEB": ode}
    metric_keys = ("i_perp", "delta_e_kcal_per_mol", "geometric_action_kcal_per_mol")
    frontiers = {
        method: {metric: lower_frontier(rows, metric) for metric in metric_keys}
        for method, rows in methods.items()
    }
    comparisons: dict[str, Any] = {}
    for metric in metric_keys:
        threshold = max(min(float(row[metric]) for row in ir),
                        min(float(row[metric]) for row in ode))
        ivm_nf = first_passage(ir, metric, threshold)
        ode_nf = first_passage(ode, metric, threshold)
        comparisons[metric] = {
            "strictest_common_threshold": threshold,
            "IVM_first_passage_N_F": ivm_nf,
            "ODE_NEB_first_passage_N_F": ode_nf,
            "Number_of_Forces_ratio_ODE_over_IVM": ode_nf / ivm_nf,
        }
    return {
        "schema": DATA_SCHEMA,
        "status": "PASS_ZERO_PHYSICAL_CALLS",
        "created_utc": utc_now(),
        "methods": ["IVM", "ODE-NEB"],
        "method_identity": {
            "IVM": "selected BB IR-IVM: alpha=1, r=0.1, m_max=5, equal-arclength reparameterization",
            "ODE-NEB": "ASE NEBOptimizer(method='ODE'), k=0.1 eV A^-2",
        },
        "metric_protocol": {
            "images": 32,
            "segments": 31,
            "densification": False,
            "interpolation": False,
            "smoothing": False,
            "mass_weighting": False,
            "i_perp": "central-difference tangent; normal full Force; local Cartesian Voronoi weights",
            "delta_e": "max(E_0,...,E_31)-E_0 on the same stored 32-image path",
            "geometric_action": {
                "name": "discrete action",
                "latex": r"\widehat S",
                "continuous_parent": r"\widehat{S}(\varphi)=\int_\varphi |\nabla V|\,\mathrm{d}\ell",
                "formula_id": FORMULA_ID,
                "discrete_formula": "sum_{i=0}^{30} 0.5*(||F_i||_2+||F_{i+1}||_2)*||x_{i+1}-x_i||_2",
                "force_identity": "stored OpenMM Force=-grad V, so Euclidean norms agree",
            },
            "energy_and_action_conversion": {
                "from": "eV", "to": "kcal mol^-1", "factor": EV_TO_KCAL_PER_MOL
            },
        },
        "cost_protocol": {
            "quantity": "optimization Number of Forces",
            "fixed_endpoint_optimization_N_F": 0,
            "additional_physical_calls": 0,
            "metric_optimization_N_F_contribution": 0,
            "curve": "strict cumulative-best first-passage frontier; no metric or N_F interpolation",
            "IR_IVM_prefix": "all formally reportable R6B states at or below N_F=20000",
            "ODE_NEB_history": "current accepted path after each paid batch; rejected paid trials retain the preceding current path",
        },
        "source_bindings": {
            "R5B_primary": binding(R5_PRIMARY),
            "R5B_supplemental": binding(R5_SUPPLEMENTAL),
            "R6B_primary": binding(R6_PRIMARY),
            "R6B_supplemental": binding(R6_SUPPLEMENTAL),
            "R6B_compact_metric_suffix": binding(R6_SUFFIX_METRICS),
            "R6B_compact_metric_generator": binding(R6_SUFFIX_GENERATOR),
            "ODE_final_cache": binding(ODE_CACHE),
            "ODE_decoder": binding(ODE_SOURCE_ROOT / "src/full_batch_recorder.py"),
        },
        "history_summary": {"IVM": ir_summary, "ODE-NEB": ode_summary},
        "histories": methods,
        "strict_lower_frontiers": frontiers,
        "strictest_common_thresholds": comparisons,
        "summary": {
            method: {
                "states": len(rows),
                "last_N_F": int(rows[-1]["optimization_N_F"]),
                **{
                    f"best_{metric}": min(float(row[metric]) for row in rows)
                    for metric in metric_keys
                },
            }
            for method, rows in methods.items()
        },
    }


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.4,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.4,
        "xtick.labelsize": 7.1,
        "ytick.labelsize": 7.1,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def force_formatter(value: float, _position: int) -> str:
    return f"{value / 1000:g}k" if abs(value) >= 1000 else f"{value:g}"


def plot_data(data: Mapping[str, Any]) -> plt.Figure:
    require(data.get("schema") == DATA_SCHEMA and data.get("status") == "PASS_ZERO_PHYSICAL_CALLS",
            "data pack is not a completed zero-call result")
    configure_style()
    # Match the physical width of the formal alanine/chignolin figures so the
    # paper does not down-scale the annotation text after inclusion.
    figure, axes = plt.subplots(1, 3, figsize=(7.085, 3.23), sharey=True)
    styles = {
        "IVM": dict(
            color=IVM_COLOR, marker="o", linestyle="-", linewidth=1.85,
            markerfacecolor="white", markeredgewidth=0.70,
        ),
        "ODE-NEB": dict(
            color=ODE_COLOR, marker="s", linestyle="--", linewidth=1.65,
            markerfacecolor="white", markeredgewidth=0.70,
        ),
    }
    panels = (
        (
            "i_perp", r"$I_\perp$", (1.0e4, 5.0),
            (1.0e4, 1.0e3, 1.0e2, 1.0e1),
            (r"$10^4$", r"$10^3$", r"$10^2$", r"$10^1$"),
        ),
        (
            "delta_e_kcal_per_mol", r"$\Delta E$ (kcal mol$^{-1}$)",
            (1.0e3, 80.0), (1.0e3, 5.0e2, 2.0e2, 1.0e2),
            ("1000", "500", "200", "100"),
        ),
        (
            "geometric_action_kcal_per_mol",
            r"$\widehat S$ (kcal mol$^{-1}$)",
            (1.0e5, 3.0e3), (1.0e5, 3.0e4, 1.0e4, 3.0e3),
            (r"$10^5$", r"$3\!\times\!10^4$", r"$10^4$", r"$3\!\times\!10^3$"),
        ),
    )
    panel_labels = ("a", "b", "c")
    for axis, panel_label, (metric, xlabel, requested_xlim, tick_values, tick_labels) in zip(
        axes, panel_labels, panels
    ):
        for method in ("IVM", "ODE-NEB"):
            frontier = data["strict_lower_frontiers"][method][metric]
            x = [float(row[metric]) for row in frontier]
            y = [int(row["optimization_N_F"]) for row in frontier]
            axis.plot(
                x, y, label=method, markersize=2.8,
                markevery=max(1, len(frontier) // 11),
                **styles[method],
            )
        comparison = data["strictest_common_thresholds"][metric]
        threshold = float(comparison["strictest_common_threshold"])
        ivm_nf = int(comparison["IVM_first_passage_N_F"])
        ode_nf = int(comparison["ODE_NEB_first_passage_N_F"])
        if ivm_nf != ode_nf:
            low, high = sorted((ivm_nf, ode_nf))
            axis.axvline(
                threshold, color="#606060", linewidth=0.60,
                linestyle=(0, (1.0, 2.2)), zorder=0,
            )
            axis.annotate(
                "", xy=(threshold, high), xytext=(threshold, low),
                arrowprops={
                    "arrowstyle": "<->", "color": "#202020", "lw": 1.10,
                    "mutation_scale": 9.0,
                },
            )
            axis.annotate(
                f"{float(comparison['Number_of_Forces_ratio_ODE_over_IVM']):.1f}\N{MULTIPLICATION SIGN}",
                xy=(threshold, 0.5 * (low + high)), xytext=(5, 0),
                textcoords="offset points", va="center", ha="left",
                fontsize=8.8, fontweight="bold", color="#202020", clip_on=False,
                bbox={
                    "boxstyle": "round,pad=0.14", "facecolor": "white",
                    "edgecolor": "none", "alpha": 0.94,
                },
            )
        axis.set_xscale("log")
        axis.set_xlim(*requested_xlim)
        axis.xaxis.set_major_locator(mpl.ticker.FixedLocator(tick_values))
        axis.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(tick_labels))
        axis.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        if metric == "geometric_action_kcal_per_mol":
            axis.get_xticklabels()[-1].set_horizontalalignment("right")
        axis.set_ylim(0.0, 1.05 * ODE_FINAL_NF)
        axis.set_xlabel(xlabel)
        axis.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(force_formatter))
        axis.grid(color="#D0D0D0", linewidth=0.45, alpha=0.55)
        axis.set_box_aspect(1.0)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(-0.16, 1.035, panel_label, transform=axis.transAxes,
                  ha="left", va="bottom", fontsize=8.5, fontweight="bold")
    axes[0].set_ylabel("Number of Forces")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, frameon=False, loc="upper center", ncol=2,
                  bbox_to_anchor=(0.5, 0.995))
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.20, top=0.83, wspace=0.27)
    return figure


def build_plot(data: Mapping[str, Any]) -> None:
    figure = plot_data(data)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{PDF.name}.", suffix=".pdf", dir=PDF.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        figure.savefig(
            temporary, format="pdf",
            metadata={
                "Title": "Barnase--Barstar IVM20k versus ODE-NEB30k discrete cost to precision",
                "Subject": "I_perp, Delta E, and native-32 geometric action",
                "Author": "IVM for MEP analysis",
                "Creator": "Matplotlib",
            },
        )
        os.replace(temporary, PDF)
    finally:
        temporary.unlink(missing_ok=True)
        plt.close(figure)
    require(PDF.is_file() and PDF.stat().st_size > 10000, "PDF is absent or implausibly small")


def build_provenance(data: Mapping[str, Any]) -> None:
    script = Path(__file__).resolve(strict=True)
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "status": "PASS_PENDING_VISUAL_QA",
        "created_utc": utc_now(),
        "figure": binding(PDF),
        "data": binding(DATA_JSON),
        "plot_and_analysis_code": binding(script),
        "source_bindings": data["source_bindings"],
        "methods": data["method_identity"],
        "history_summary": data["history_summary"],
        "metric_protocol": data["metric_protocol"],
        "cost_protocol": data["cost_protocol"],
        "strictest_common_thresholds": data["strictest_common_thresholds"],
        "figure_contract": {
            "layout": "1x3",
            "each_axes_box_aspect": 1.0,
            "panels": ["I_perp", "Delta E", "discrete geometric action"],
            "legend_labels": ["IVM", "ODE-NEB"],
            "IR_IVM_display_budget": "20k (actual N_F=19993)",
            "ODE_NEB_display_budget": "30k (actual N_F=30000)",
            "all_metrics_from_same_native_state": True,
            "x_axis_scale": "reversed logarithmic for all three panels",
            "display_windows": {
                "I_perp": [1.0e4, 5.0],
                "Delta_E_kcal_per_mol": [1.0e3, 80.0],
                "discrete_action_kcal_per_mol": [1.0e5, 3.0e3],
            },
        },
        "physical_calls": 0,
        "optimization_N_F_contribution": 0,
    }
    atomic_json(PROVENANCE_JSON, provenance)

    thresholds = data["strictest_common_thresholds"]
    summary = data["summary"]
    traceability = f"""# Barnase--Barstar IVM20k / ODE-NEB30k discrete cost-to-precision

Status: `PASS_PENDING_VISUAL_QA`

## Output

- PDF: `{PDF}`
- PDF SHA-256: `{sha256_file(PDF)}`
- Data: `{DATA_JSON}`
- Data SHA-256: `{sha256_file(DATA_JSON)}`
- Analysis and plotting code: `{script}`
- Code SHA-256: `{sha256_file(script)}`
- Provenance JSON: `{PROVENANCE_JSON}`

## Formal method histories

- `IVM` is the selected BB IR-IVM algorithm (inexact update plus equal-arclength
  reparameterization; alpha=1, r=0.1, m_max=5). The figure includes every R6B
  formally reportable state with `N_F <= 20000`: calls 1--693, ending at the
  actual `N_F=19993`. Calls 1--515 use locally stored R5B shards whose path,
  energy, and Force hashes are independently checked against the R6B inventory;
  calls 516--693 use a compact 178-row ledger computed directly beside the
  remote external-disk R6B shards and synchronized through the existing
  Mutagen project session. Every compact row carries its source shard and
  path/E/Force hashes. No state after 20k and no terminal-uncommitted tail is
  included.
- `ODE-NEB` is ASE `NEBOptimizer(method="ODE")` with the selected
  `k=0.1 eV A^-2`. All 1000 paid interior-image batches are used through the
  exact `N_F=30000`. After a rejected trial, the already accepted current path
  remains the reported state while the 30 paid Forces remain counted.
- Fixed endpoints were inherited and add zero optimization Forces.

## Metrics

All three metrics use the same saved 32-image state, with no densification,
interpolation, smoothing, or mass weighting.

- `I_perp`: centered path tangent, normal component of the full stored Force,
  and local Cartesian Voronoi weights.
- `Delta E`: `max(E_0,...,E_31)-E_0`, converted with
  `1 eV = {EV_TO_KCAL_PER_MOL} kcal mol^-1`. The plot label is simply
  `Delta E`, as requested.
- Discrete action `S_hat`:
  `S_hat=sum_i 0.5*(||F_i||+||F_(i+1)||)*||x_(i+1)-x_i||` over the 31 native
  Cartesian segments. Because OpenMM stores `F=-grad V`, `||F||=||grad V||`.

The curves are strict cumulative-best first-passage frontiers. Neither the
metric nor `N_F` is interpolated. This offline calculation performs zero new
energy/Force evaluations and adds zero `N_F`.

## Display geometry

All three metric axes use the same reversed logarithmic convention.  The
display windows are `I_perp: 1e4 to 5`, `Delta E: 1000 to 80 kcal mol^-1`, and
`S_hat: 1e5 to 3e3 kcal mol^-1`.  These windows focus the shared
convergence regime; the strict first-passage thresholds and ratios are still
computed from each method's complete saved frontier before display clipping.

## Numerical closure

- IVM: {summary['IVM']['states']} formally reportable stored checkpoints, final `N_F={summary['IVM']['last_N_F']}`.
- ODE-NEB: {summary['ODE-NEB']['states']} paid-batch states, final `N_F={summary['ODE-NEB']['last_N_F']}`.
- Strictest common `I_perp`: `{thresholds['i_perp']['strictest_common_threshold']:.12g}`;
  first passage IVM/ODE-NEB = `{thresholds['i_perp']['IVM_first_passage_N_F']}` /
  `{thresholds['i_perp']['ODE_NEB_first_passage_N_F']}` Forces.
- Strictest common `Delta E`: `{thresholds['delta_e_kcal_per_mol']['strictest_common_threshold']:.12g}`
  kcal mol^-1; first passage IVM/ODE-NEB =
  `{thresholds['delta_e_kcal_per_mol']['IVM_first_passage_N_F']}` /
  `{thresholds['delta_e_kcal_per_mol']['ODE_NEB_first_passage_N_F']}` Forces.
- Strictest common action: `{thresholds['geometric_action_kcal_per_mol']['strictest_common_threshold']:.12g}`
  kcal mol^-1; first passage IVM/ODE-NEB =
  `{thresholds['geometric_action_kcal_per_mol']['IVM_first_passage_N_F']}` /
  `{thresholds['geometric_action_kcal_per_mol']['ODE_NEB_first_passage_N_F']}` Forces.

Every plotted row records its path, energy, and Force SHA-256 in the data JSON.
Every IR source shard and every ODE paid transport shard is hash-checked before
its values are admitted. The ODE decoder additionally reconstructs every XOR
delta block bit-exactly and verifies all declared logical-array hashes.
"""
    atomic_text(TRACEABILITY_MD, traceability)


def run_compute() -> dict[str, Any]:
    data = build_data()
    atomic_json(DATA_JSON, data)
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("compute", "plot", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode in {"compute", "all"}:
        data = run_compute()
    else:
        data = load_json(DATA_JSON)
    if args.mode in {"plot", "all"}:
        build_plot(data)
        build_provenance(data)
    print(json.dumps({
        "pdf": str(PDF) if PDF.exists() else None,
        "data": str(DATA_JSON) if DATA_JSON.exists() else None,
        "provenance": str(PROVENANCE_JSON) if PROVENANCE_JSON.exists() else None,
        "traceability": str(TRACEABILITY_MD) if TRACEABILITY_MD.exists() else None,
        "summary": data.get("summary"),
        "strictest_common_thresholds": data.get("strictest_common_thresholds"),
        "additional_physical_calls": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
