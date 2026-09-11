#!/usr/bin/env python3
"""Render the variable-density BB IVM/ODE-NEB cost-to-precision figure.

Every accepted observation is backed by one immutable joint energy/full-force
arc500 shard.  The plotting curve is the observed lower envelope; faint open
markers retain every measured observation.  No line segment is interpreted as
an unmeasured first-passage result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FIGURE_SCHEMA = "bb-joint-ef-cost-to-precision-figure-contract-v1"
FIGURE_STATUS = "FROZEN_COMPLETE_JOINT_EF_INPUTS"
PACK_SCHEMA = "bb-joint-ef-cost-to-precision-pack-spec-v1"
PACK_STATUS = "COMPLETE_ZERO_OPENMM_VALIDATED_PACK"
PROVENANCE_SCHEMA = "bb-joint-ef-cost-to-precision-provenance-v1"
N_IMAGES = 32
N_INTERNAL = 30
DIMENSION = 9396
N_DENSE = 500
EV_TO_KCAL = 23.0605
IVM_COLOR = "#0072B2"
ODE_COLOR = "#D55E00"
METHODS = ("IVM", "ODE-NEB")
STYLE_REFERENCE = (
    Path("/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/alanine_66d")
    / "code/plotting/plot_ala_discrete_single_cost_to_precision_muller_style.py"
)
FREEZER_PATH = HERE / "freeze_bb_cost_to_precision_joint_ef_dense_figure_contract.py"
PACK_FREEZER_PATH = HERE / "freeze_bb_joint_ef_cost_pack_spec.py"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def same_float(left: object, right: object) -> bool:
    return np.asarray(float(left), dtype=np.float64).tobytes() == np.asarray(
        float(right), dtype=np.float64
    ).tobytes()


def scalar(saved, key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} must be scalar")
    return value.item()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def resolve_from(base: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base.parent / path
    return path.resolve(strict=True)


def validate_binding(record: dict, base: Path, role: str) -> dict:
    require(
        isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and isinstance(record.get("sha256"), str),
        f"{role} must contain path and SHA-256",
    )
    path = resolve_from(base, record["path"])
    observed = sha256_file(path)
    require(observed == record["sha256"], f"{role} SHA-256 mismatch")
    return {"path": str(path), "sha256": observed}


def file_binding(path: Path) -> dict:
    path = Path(path).expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": sha256_file(path)}


def collect_file_bindings(value, base: Path) -> set[tuple[str, str]]:
    """Collect and validate explicit path/SHA pairs, including legacy siblings."""

    found: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            record = validate_binding(value, base, "recursive binding")
            found.add((record["path"], record["sha256"]))
        for key, child in value.items():
            sibling = value.get(f"{key}_sha256")
            if key != "path" and isinstance(child, str) and isinstance(sibling, str):
                try:
                    path = resolve_from(base, child)
                except (FileNotFoundError, OSError):
                    path = None
                if path is not None:
                    require(sha256_file(path) == sibling, f"legacy binding drift: {key}")
                    found.add((str(path), sibling))
            found.update(collect_file_bindings(child, base))
    elif isinstance(value, list):
        for child in value:
            found.update(collect_file_bindings(child, base))
    return found


def all_values_for_key(value, key: str) -> list:
    found = []
    if isinstance(value, dict):
        for name, child in value.items():
            if name == key:
                found.append(child)
            found.extend(all_values_for_key(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(all_values_for_key(child, key))
    return found


def uniform_arc500(path: np.ndarray) -> np.ndarray:
    curve = np.asarray(path, dtype=np.float64)
    require(curve.shape == (N_IMAGES, DIMENSION), "selected path shape mismatch")
    require(np.all(np.isfinite(curve)), "selected path has non-finite coordinates")
    lengths = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    require(np.all(np.isfinite(lengths)) and np.all(lengths > 0.0), "invalid path segment")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, float(cumulative[-1]), N_DENSE)
    index = np.clip(
        np.searchsorted(cumulative, targets, side="right") - 1,
        0,
        len(lengths) - 1,
    )
    fraction = (targets - cumulative[index]) / lengths[index]
    dense = (1.0 - fraction[:, None]) * curve[index] + fraction[:, None] * curve[index + 1]
    dense[0] = curve[0]
    dense[-1] = curve[-1]
    return np.ascontiguousarray(dense)


def discrete_iperp(path: np.ndarray, internal_force: np.ndarray) -> float:
    curve = np.asarray(path, dtype=np.float64)
    force = np.asarray(internal_force, dtype=np.float64)
    require(curve.shape == (N_IMAGES, DIMENSION), "I_perp path shape mismatch")
    require(force.shape == (N_INTERNAL, DIMENSION), "I_perp force shape mismatch")
    require(np.all(np.isfinite(force)), "I_perp force contains non-finite values")
    tangent = curve[2:] - curve[:-2]
    norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    require(np.all(np.isfinite(norm)) and np.all(norm > 0.0), "invalid central tangent")
    tangent = tangent / norm
    normal_force = force - np.sum(force * tangent, axis=1, keepdims=True) * tangent
    segment = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    weights = 0.5 * (segment[:-1] + segment[1:])
    per_image_l2 = np.linalg.norm(normal_force, axis=1)
    return float(np.sum(weights * per_image_l2**2))


def _validate_ivm_online_proof(proof: dict, point: dict, selected: np.ndarray, spec_path: Path) -> dict:
    require(proof.get("kind") == "sealed_online_history", "IVM requires sealed online I_perp proof")
    replay_record = validate_binding(proof.get("replay_result"), spec_path, "IVM replay result")
    verification_record = validate_binding(
        proof.get("verification_report"), spec_path, "IVM replay verification"
    )
    expected_schema = str(proof.get("verification_schema", ""))
    expected_status = str(proof.get("verification_status", ""))
    require(expected_schema and expected_status, "IVM proof must freeze verification schema/status")
    verification = read_json(Path(verification_record["path"]))
    require(
        verification.get("schema") == expected_schema
        and verification.get("status") == expected_status,
        "IVM replay verification identity mismatch",
    )
    call = int(proof.get("calls_per_image", -1))
    history_index = int(proof.get("history_index", -1))
    nf = int(point["optimization_N_F"])
    path_sha = str(point["selected_path_sha256"])
    metric = float(point["discrete_I_perp"])
    rows = verification.get("new_reporting_rows")
    if not isinstance(rows, list):
        rows = verification.get("reporting_rows")
    require(isinstance(rows, list), "IVM verification lacks reporting rows")
    matches = [
        row
        for row in rows
        if int(row.get("call_per_image", row.get("calls_per_image", -1))) == call
        and int(row.get("optimization_N_F", -1)) == nf
        and str(row.get("selected_path_sha256")) == path_sha
    ]
    require(len(matches) == 1, "IVM verification row is absent or non-unique")
    row = matches[0]
    require(
        int(row.get("history_index", -1)) == history_index
        and row.get("fully_evaluated_current_path") is True
        and row.get("terminal_tail") is False
        and same_float(row.get("discrete_I_perp_eV2_per_A"), metric),
        "IVM verification row metric/completeness mismatch",
    )
    with np.load(replay_record["path"], allow_pickle=False) as saved:
        required = {
            "stored_path_calls",
            "stored_paths",
            "calls_per_image",
            "total_internal_joint_evaluations",
            "i_perp",
        }
        require(required.issubset(saved.files), "IVM replay result fields missing")
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        stored_paths = np.asarray(saved["stored_paths"], dtype=np.float64)
        online_calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        online_nf = np.asarray(saved["total_internal_joint_evaluations"], dtype=np.int64)
        online_i = np.asarray(saved["i_perp"], dtype=np.float64)
    stored_index = np.flatnonzero(stored_calls == call)
    require(len(stored_index) == 1, "IVM replay stored call is absent or non-unique")
    require(
        stored_paths.shape[0] == len(stored_calls)
        and stored_paths.shape[1:] == (N_IMAGES, DIMENSION),
        "IVM replay stored path shape mismatch",
    )
    require(
        np.array_equal(stored_paths[int(stored_index[0])], selected),
        "IVM replay path differs from authoritative shard",
    )
    require(
        0 <= history_index < len(online_calls)
        and online_calls[history_index] == call
        and online_nf[history_index] == nf
        and same_float(online_i[history_index], metric),
        "IVM online-history I_perp/N_F closure mismatch",
    )
    return {
        "kind": "sealed_online_history",
        "calls_per_image": call,
        "history_index": history_index,
        "replay_result": replay_record,
        "verification_report": verification_record,
        "verification_schema": expected_schema,
        "verification_status": expected_status,
        "new_physical_calls": 0,
    }


def _find_contract_row(contract: dict, nf: int, path_sha: str) -> dict:
    rows = contract.get("paths")
    require(isinstance(rows, list), "pack contract lacks path rows")
    matches = [
        row
        for row in rows
        if int(row.get("optimization_N_F", -1)) == nf
        and str(row.get("path_sha256")) == path_sha
    ]
    require(len(matches) == 1, "pack contract row is absent or non-unique")
    return matches[0]


def _validate_shard(
    point: dict,
    method: str,
    shard_path: Path,
    spec_path: Path,
) -> dict:
    with np.load(shard_path, allow_pickle=False) as saved:
        required = {
            "status",
            "method",
            "optimization_N_F",
            "selected_path",
            "selected_path_sha256",
            "discrete_I_perp",
            "dense_coordinates_A",
            "dense_energy_eV",
            "dense_force_eV_per_A",
            "dense_coordinate_sha256",
            "dense_energy_sha256",
            "dense_force_sha256",
            "densified_Delta_E_kcal_per_mol",
            "number_of_points",
            "optimization_N_F_contribution",
        }
        require(required.issubset(saved.files), "authoritative shard fields missing")
        require(str(scalar(saved, "status")) == "PASS", "authoritative shard is not PASS")
        require(str(scalar(saved, "method")) == method, "authoritative shard method mismatch")
        nf = int(scalar(saved, "optimization_N_F"))
        require(nf == int(point["optimization_N_F"]), "authoritative shard N_F mismatch")
        require(int(scalar(saved, "number_of_points")) == N_DENSE, "shard is not arc500")
        require(
            int(scalar(saved, "optimization_N_F_contribution")) == 0,
            "diagnostic joint E/F contaminated optimization N_F",
        )
        selected = np.asarray(saved["selected_path"])
        dense = np.asarray(saved["dense_coordinates_A"])
        energy = np.asarray(saved["dense_energy_eV"])
        force = np.asarray(saved["dense_force_eV_per_A"])
        require(
            selected.dtype == dense.dtype == energy.dtype == force.dtype == np.dtype("float64"),
            "authoritative scientific arrays must be float64",
        )
        require(selected.shape == (N_IMAGES, DIMENSION), "selected path shape mismatch")
        require(
            dense.shape == force.shape == (N_DENSE, DIMENSION)
            and energy.shape == (N_DENSE,),
            "arc500 coordinate/energy/full-force shape mismatch",
        )
        require(
            np.all(np.isfinite(selected))
            and np.all(np.isfinite(dense))
            and np.all(np.isfinite(energy))
            and np.all(np.isfinite(force)),
            "authoritative shard contains non-finite values",
        )
        path_sha = sha256_array(selected)
        dense_sha = sha256_array(dense)
        energy_sha = sha256_array(energy)
        force_sha = sha256_array(force)
        require(
            path_sha
            == str(scalar(saved, "selected_path_sha256"))
            == str(point["selected_path_sha256"]),
            "selected path SHA mismatch",
        )
        require(
            dense_sha == str(scalar(saved, "dense_coordinate_sha256")),
            "dense coordinate SHA mismatch",
        )
        require(
            energy_sha == str(scalar(saved, "dense_energy_sha256")),
            "dense energy SHA mismatch",
        )
        require(
            force_sha == str(scalar(saved, "dense_force_sha256")),
            "dense full-force SHA mismatch",
        )
        expected_dense = uniform_arc500(selected)
        require(
            np.array_equal(dense, expected_dense),
            "dense coordinates differ from deterministic Cartesian arc500",
        )
        metric = float(scalar(saved, "discrete_I_perp"))
        delta_e = float(scalar(saved, "densified_Delta_E_kcal_per_mol"))
        require(np.isfinite(metric) and metric > 0.0, "discrete I_perp must be finite and positive")
        require(np.isfinite(delta_e) and delta_e >= 0.0, "densified Delta E must be finite and nonnegative")
        require(same_float(metric, point["discrete_I_perp"]), "point/shard I_perp mismatch")
        recomputed_delta = float((np.max(energy) - energy[0]) * EV_TO_KCAL)
        require(
            same_float(delta_e, recomputed_delta)
            and same_float(delta_e, point["densified_Delta_E_kcal_per_mol"]),
            "densified Delta E reconstruction mismatch",
        )
        evidence = point.get("discrete_I_perp_evidence")
        require(isinstance(evidence, dict), "discrete I_perp evidence missing")
        if method == "ODE-NEB":
            require(
                evidence.get("kind") == "recomputed_original_internal_force",
                "ODE-NEB requires original-path force reconstruction",
            )
            force_key = str(evidence.get("force_key", ""))
            require(
                force_key in {"internal_diagnostic_force_eV_per_A", "paid_internal_force_eV_per_A"}
                and force_key in saved.files,
                "ODE-NEB original internal-force field missing",
            )
            original_force = np.asarray(saved[force_key])
            require(original_force.dtype == np.dtype("float64"), "original force must be float64")
            measured = discrete_iperp(selected, original_force.reshape(N_INTERNAL, DIMENSION))
            require(same_float(measured, metric), "discrete I_perp reconstruction mismatch")
            evidence_report = {
                "kind": "recomputed_original_internal_force",
                "force_key": force_key,
                "force_sha256": sha256_array(original_force),
                "new_physical_calls": 0,
            }
        else:
            evidence_report = _validate_ivm_online_proof(evidence, point, selected, spec_path)
    return {
        "method": method,
        "label": str(point["label"]),
        "optimization_N_F": nf,
        "selected_path_sha256": path_sha,
        "dense_coordinate_sha256": dense_sha,
        "dense_energy_sha256": energy_sha,
        "dense_force_sha256": force_sha,
        "discrete_I_perp": metric,
        "densified_Delta_E_kcal_per_mol": delta_e,
        "discrete_I_perp_evidence": evidence_report,
        "arc500_full_force_vectors_verified": N_DENSE,
        "optimization_N_F_contribution": 0,
    }


def _require_pair_in(bindings: set[tuple[str, str]], record: dict, role: str) -> None:
    require((record["path"], record["sha256"]) in bindings, f"{role} is not bound by source closure")


def validate_pack_spec(raw_path: Path) -> tuple[dict, list[dict]]:
    spec_path = Path(raw_path).expanduser().resolve(strict=True)
    spec = read_json(spec_path)
    require(spec.get("schema") == PACK_SCHEMA, "unsupported pack-spec schema")
    require(spec.get("status") == PACK_STATUS, "pack spec is not complete/frozen")
    require(spec.get("include_all_pack_points") is True, "pack spec permits metric-based point omission")
    builder = validate_binding(spec.get("pack_spec_freezer"), spec_path, "pack-spec freezer")
    require(builder["path"] == str(PACK_FREEZER_PATH.resolve()), "pack-spec freezer path mismatch")
    method = str(spec.get("method"))
    require(method in METHODS, "pack spec method mismatch")
    closure = spec.get("closure")
    require(isinstance(closure, dict), "pack closure missing")
    required_roles = {"result", "audit", "contract", "manifest", "evaluator", "topology", "bb_core", "WALs"}
    require(required_roles.issubset(closure), "pack closure roles missing")
    resolved = {
        role: validate_binding(closure[role], spec_path, f"pack {role}")
        for role in ("result", "audit", "contract", "manifest", "evaluator", "topology", "bb_core")
    }
    require(isinstance(closure["WALs"], list) and closure["WALs"], "pack WAL list is empty")
    resolved["WALs"] = [
        validate_binding(record, spec_path, f"pack WAL {index}")
        for index, record in enumerate(closure["WALs"])
    ]
    audit = read_json(Path(resolved["audit"]["path"]))
    contract = read_json(Path(resolved["contract"]["path"]))
    manifest = read_json(Path(resolved["manifest"]["path"]))
    require(audit.get("status") == "PASS", "pack audit is not PASS")
    require(manifest.get("status") == "PASS", "pack manifest is not PASS")
    if "method" in audit:
        require(audit["method"] == method, "pack audit method mismatch")
    if "method" in contract:
        require(contract["method"] == method, "pack contract method mismatch")
    require(contract.get("frozen") is True, "pack contract is not frozen")
    contribution_values = (
        all_values_for_key(audit, "optimization_N_F_contribution")
        + all_values_for_key(contract, "optimization_N_F_contribution")
        + all_values_for_key(manifest, "optimization_N_F_contribution")
    )
    require(contribution_values, "pack closure lacks optimization N_F accounting")
    require(
        all(int(value) == 0 for value in contribution_values),
        "pack closure charges diagnostics to optimization N_F",
    )
    audit_bindings = collect_file_bindings(audit, Path(resolved["audit"]["path"]))
    manifest_bindings = collect_file_bindings(manifest, Path(resolved["manifest"]["path"]))
    contract_bindings = collect_file_bindings(contract, Path(resolved["contract"]["path"]))
    for role in ("result", "contract", "manifest"):
        if role != "audit":
            _require_pair_in(audit_bindings, resolved[role], f"audit/{role}")
    _require_pair_in(manifest_bindings, resolved["result"], "manifest/result")
    _require_pair_in(manifest_bindings, resolved["contract"], "manifest/contract")
    for role in ("evaluator", "topology", "bb_core"):
        _require_pair_in(contract_bindings, resolved[role], f"contract/{role}")
    for record in resolved["WALs"]:
        require(
            (record["path"], record["sha256"]) in audit_bindings.union(manifest_bindings),
            "pack WAL is not bound by audit/manifest",
        )

    with np.load(resolved["result"]["path"], allow_pickle=False) as saved:
        required = {
            "status",
            "method",
            "optimization_N_F",
            "selected_path_sha256",
            "discrete_I_perp",
            "densified_Delta_E_kcal_per_mol",
            "shard_path",
            "shard_sha256",
            "optimization_N_F_contribution",
        }
        require(required.issubset(saved.files), "pack aggregate fields missing")
        require(str(scalar(saved, "status")) == "PASS", "pack aggregate is not PASS")
        require(str(scalar(saved, "method")) == method, "pack aggregate method mismatch")
        require(int(scalar(saved, "optimization_N_F_contribution")) == 0, "pack aggregate N_F contamination")
        aggregate_nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        aggregate_path_sha = np.asarray(saved["selected_path_sha256"]).astype(str)
        aggregate_i = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        aggregate_delta = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
        aggregate_shard_path = np.asarray(saved["shard_path"]).astype(str)
        aggregate_shard_sha = np.asarray(saved["shard_sha256"]).astype(str)
    count = len(aggregate_nf)
    require(count > 0, "pack aggregate has no points")
    require(
        all(array.shape == (count,) for array in (aggregate_path_sha, aggregate_i, aggregate_delta, aggregate_shard_path, aggregate_shard_sha)),
        "pack aggregate vector shapes mismatch",
    )
    require(len(set(int(value) for value in aggregate_nf)) == count, "pack aggregate repeats N_F")
    points = spec.get("points")
    require(isinstance(points, list) and len(points) == count, "pack spec must include every aggregate point")
    manifest_rows = manifest.get("shards")
    require(isinstance(manifest_rows, list) and len(manifest_rows) == count, "pack manifest shard count mismatch")
    reports = []
    seen = set()
    for point in points:
        require(isinstance(point, dict), "pack point must be an object")
        nf = int(point.get("optimization_N_F", -1))
        require(nf > 0 and nf not in seen, "pack point has invalid/repeated N_F")
        seen.add(nf)
        index_values = np.flatnonzero(aggregate_nf == nf)
        require(len(index_values) == 1, "pack point N_F not unique in aggregate")
        index = int(index_values[0])
        shard_record = validate_binding(point.get("authoritative_shard"), spec_path, f"NF{nf} shard")
        aggregate_path = Path(aggregate_shard_path[index]).expanduser().resolve(strict=True)
        require(
            shard_record["path"] == str(aggregate_path)
            and shard_record["sha256"] == aggregate_shard_sha[index]
            and sha256_file(aggregate_path) == aggregate_shard_sha[index],
            "pack aggregate/shard binding mismatch",
        )
        require(
            str(point.get("selected_path_sha256")) == aggregate_path_sha[index]
            and same_float(point.get("discrete_I_perp"), aggregate_i[index])
            and same_float(point.get("densified_Delta_E_kcal_per_mol"), aggregate_delta[index]),
            "pack point/aggregate metric mismatch",
        )
        contract_row = _find_contract_row(contract, nf, aggregate_path_sha[index])
        if "online_discrete_I_perp" in contract_row:
            require(
                same_float(contract_row["online_discrete_I_perp"], aggregate_i[index]),
                "contract online I_perp mismatch",
            )
        manifest_matches = []
        for row in manifest_rows:
            row_path_raw = row.get("path")
            row_sha = row.get("sha256")
            if isinstance(row_path_raw, str) and isinstance(row_sha, str):
                row_path = resolve_from(Path(resolved["manifest"]["path"]), row_path_raw)
                if int(row.get("optimization_N_F", -1)) == nf and str(row_path) == shard_record["path"]:
                    manifest_matches.append(row)
        require(len(manifest_matches) == 1, "manifest shard row absent or non-unique")
        require(manifest_matches[0]["sha256"] == shard_record["sha256"], "manifest shard SHA mismatch")
        require(
            (shard_record["path"], shard_record["sha256"]) in manifest_bindings,
            "authoritative shard is not hash-bound by manifest",
        )
        report = _validate_shard(point, method, Path(shard_record["path"]), spec_path)
        report["authoritative_shard"] = shard_record
        report["pack_id"] = str(spec.get("pack_id"))
        reports.append(report)
    reports.sort(key=lambda row: row["optimization_N_F"])
    return {
        "pack_spec": file_binding(spec_path),
        "pack_spec_freezer": builder,
        "pack_id": str(spec.get("pack_id")),
        "method": method,
        "closure": resolved,
        "point_count": len(reports),
    }, reports


def observed_lower_envelope(nf: np.ndarray, metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(nf, kind="stable")
    costs = np.asarray(nf, dtype=np.int64)[order]
    values = np.asarray(metric, dtype=np.float64)[order]
    out_metric, out_cost = [], []
    best = np.inf
    for cost, value in zip(costs, values):
        if float(value) < best:
            best = float(value)
            out_metric.append(best)
            out_cost.append(int(cost))
    return np.asarray(out_metric), np.asarray(out_cost, dtype=np.int64)


def earliest_observed_passage(nf: np.ndarray, metric: np.ndarray, threshold: float) -> int:
    order = np.argsort(nf, kind="stable")
    best = np.inf
    for index in order:
        best = min(best, float(metric[index]))
        if best <= threshold:
            return int(nf[index])
    raise ValueError("method never reaches the common observed threshold")


def analyze_observed(points: dict[str, list[dict]]) -> tuple[dict, dict]:
    arrays = {}
    for method in METHODS:
        rows = sorted(points[method], key=lambda row: row["optimization_N_F"])
        require(len(rows) >= 2, f"{method} needs at least two observations")
        nf = np.asarray([row["optimization_N_F"] for row in rows], dtype=np.int64)
        i_perp = np.asarray([row["discrete_I_perp"] for row in rows], dtype=np.float64)
        delta_e = np.asarray([row["densified_Delta_E_kcal_per_mol"] for row in rows], dtype=np.float64)
        require(np.all(np.diff(nf) > 0), f"{method} N_F values are not unique/increasing")
        require(np.all(np.isfinite(i_perp)) and np.all(i_perp > 0.0), f"{method} I_perp invalid")
        require(np.all(np.isfinite(delta_e)) and np.all(delta_e >= 0.0), f"{method} Delta E invalid")
        arrays[method] = {"number_of_forces": nf, "i_perp": i_perp, "delta_e": delta_e}
    report = {
        "claim": "observed lower envelope only",
        "raw_observations_are_all_shown": True,
        "interpolation": "none",
        "unmeasured_N_F_first_passage_claim": "forbidden",
        "thresholds": {},
        "earliest_observed_passage_N_F": {},
        "ODE_NEB_over_IVM_observed_ratio": {},
        "lower_envelopes": {},
    }
    for key in ("i_perp", "delta_e"):
        frontiers = {
            method: observed_lower_envelope(
                arrays[method]["number_of_forces"], arrays[method][key]
            )
            for method in METHODS
        }
        threshold = max(float(np.min(frontiers[method][0])) for method in METHODS)
        costs = {
            method: earliest_observed_passage(
                arrays[method]["number_of_forces"], arrays[method][key], threshold
            )
            for method in METHODS
        }
        report["thresholds"][key] = threshold
        report["earliest_observed_passage_N_F"][key] = costs
        report["ODE_NEB_over_IVM_observed_ratio"][key] = float(costs["ODE-NEB"]) / float(costs["IVM"])
        report["lower_envelopes"][key] = {
            method: [
                {"metric": float(metric), "optimization_N_F": int(cost)}
                for metric, cost in zip(*frontiers[method])
            ]
            for method in METHODS
        }
    return arrays, report


def validate_figure_contract(path: Path, output: Path) -> tuple[dict, dict[str, list[dict]], list[dict]]:
    contract_path = Path(path).expanduser().resolve(strict=True)
    contract = read_json(contract_path)
    require(contract.get("schema") == FIGURE_SCHEMA, "figure contract schema mismatch")
    require(contract.get("status") == FIGURE_STATUS, "figure contract is not frozen")
    require(
        Path(contract["figure_output"]).expanduser().resolve() == output,
        "figure output differs from frozen contract",
    )
    policy = contract.get("claim_policy", {})
    require(
        policy
        == {
            "main_curve": "observed lower envelope",
            "show_all_raw_points_faintly": True,
            "interpolation": "forbidden",
            "first_passage_claim": "earliest observed first passage only",
            "visible_arrow_text": "numeric multiplier only",
            "arrow_text_position": "right of vertical arrow at arithmetic line midpoint",
        },
        "figure claim policy mismatch",
    )
    protocol = contract.get("metric_protocol", {})
    require(
        protocol.get("discrete_metric") == "I_perp_on_original_32_image_path"
        and protocol.get("densified_metric") == "Delta_E_on_same_path_arc500"
        and protocol.get("coordinate_space") == "unweighted_3N_cartesian"
        and protocol.get("sampling") == "uniform_cartesian_arclength"
        and int(protocol.get("points", -1)) == N_DENSE
        and protocol.get("full_force_storage") is True
        and int(protocol.get("optimization_N_F_contribution", -1)) == 0,
        "figure metric protocol mismatch",
    )
    dependencies = contract.get("dependencies", {})
    for role, expected in (
        ("plotter", Path(__file__).resolve()),
        ("freezer", FREEZER_PATH.resolve()),
        ("muller_alanine_style_reference", STYLE_REFERENCE.resolve()),
    ):
        observed = validate_binding(dependencies.get(role), contract_path, role)
        require(observed["path"] == str(expected), f"figure dependency path mismatch: {role}")
    specs = contract.get("pack_specs")
    require(isinstance(specs, list) and specs, "figure contract pack specs missing")
    points = {method: [] for method in METHODS}
    packs = []
    for index, record in enumerate(specs):
        binding = validate_binding(record, contract_path, f"pack spec {index}")
        pack, rows = validate_pack_spec(Path(binding["path"]))
        require(pack["pack_spec"] == binding, "pack spec identity mismatch")
        packs.append(pack)
        points[pack["method"]].extend(rows)
    minimum = int(contract["point_count_policy"]["minimum_per_method"])
    maximum = int(contract["point_count_policy"]["maximum_per_method"])
    frozen_actual = contract["point_count_policy"].get("actual")
    require(isinstance(frozen_actual, dict), "figure contract lacks frozen point counts")
    for method in METHODS:
        require(minimum <= len(points[method]) <= maximum, f"{method} point count outside frozen policy")
        require(int(frozen_actual.get(method, -1)) == len(points[method]), f"{method} frozen point count mismatch")
        nf = [row["optimization_N_F"] for row in points[method]]
        require(len(nf) == len(set(nf)), f"{method} duplicate N_F across packs")
    topology = {pack["closure"]["topology"]["sha256"] for pack in packs}
    core = {pack["closure"]["bb_core"]["sha256"] for pack in packs}
    require(len(topology) == len(core) == 1, "methods/packs do not share topology and bb_core")
    return {
        "contract": file_binding(contract_path),
        "point_count_policy": contract["point_count_policy"],
        "claim_policy": policy,
        "metric_protocol": protocol,
    }, points, packs


def _setup_style(mpl) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "mathtext.fontset": "dejavusans",
            "mathtext.default": "it",
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _force_formatter(value: float, _position: int) -> str:
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def _annotate_ratio(axis, threshold: float, ivm_cost: int, ode_cost: int, ratio: float) -> None:
    axis.axvline(threshold, color="#767676", linewidth=0.55, linestyle=":", zorder=1)
    midpoint = 0.5 * (ivm_cost + ode_cost)
    axis.annotate(
        "",
        xy=(threshold, ivm_cost),
        xytext=(threshold, ode_cost),
        arrowprops={"arrowstyle": "<->", "color": "#202020", "lw": 0.8},
        zorder=25,
    )
    axis.annotate(
        rf"{ratio:.1f}$\times$",
        xy=(threshold, midpoint),
        xytext=(6, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color="#202020",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.8},
        zorder=26,
    )


def render(arrays: dict, analysis: dict, output: Path, preview_note: str | None = None) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter

    _setup_style(mpl)
    figure = plt.figure(figsize=(180 / 25.4, 72 / 25.4), dpi=220)
    grid = GridSpec(
        1, 2, figure=figure, left=0.072, right=0.935, bottom=0.18, top=0.78, wspace=0.25
    )
    axes = [figure.add_subplot(grid[0, index]) for index in range(2)]
    styles = {
        "IVM": {"color": IVM_COLOR, "marker": "o", "linestyle": "-", "linewidth": 1.85, "zorder": 20},
        "ODE-NEB": {"color": ODE_COLOR, "marker": "s", "linestyle": "--", "linewidth": 1.65, "zorder": 6},
    }
    for axis, key, xlabel, xscale in (
        (axes[0], "i_perp", r"$I_\perp$", "log"),
        (axes[1], "delta_e", r"$\Delta E$ (kcal mol$^{-1}$)", "linear"),
    ):
        for method in METHODS:
            data = arrays[method]
            style = styles[method]
            axis.scatter(
                data[key],
                data["number_of_forces"],
                s=10,
                marker=style["marker"],
                facecolors="none",
                edgecolors=style["color"],
                linewidths=0.45,
                alpha=0.27,
                zorder=style["zorder"] - 1,
            )
            frontier_x, frontier_y = observed_lower_envelope(
                data["number_of_forces"], data[key]
            )
            axis.plot(
                frontier_x,
                frontier_y,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=3.3 if method == "IVM" else 3.1,
                markerfacecolor="white",
                markeredgewidth=0.6,
                zorder=style["zorder"],
            )
        passage = analysis["earliest_observed_passage_N_F"][key]
        _annotate_ratio(
            axis,
            analysis["thresholds"][key],
            passage["IVM"],
            passage["ODE-NEB"],
            analysis["ODE_NEB_over_IVM_observed_ratio"][key],
        )
        all_x = np.concatenate([arrays[method][key] for method in METHODS])
        if xscale == "log":
            axis.set_xscale("log")
            axis.set_xlim(float(np.max(all_x)) * 1.45, float(np.min(all_x)) / 1.45)
        else:
            span = max(float(np.ptp(all_x)), 1.0)
            axis.set_xlim(
                float(np.max(all_x)) + 0.06 * span,
                max(0.0, float(np.min(all_x)) - 0.06 * span),
            )
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Number of Forces")
        axis.grid(True, which="major", color="#D9D9D6", linewidth=0.4, alpha=0.62)
        axis.yaxis.set_major_formatter(FuncFormatter(_force_formatter))
    ymax = max(
        11000.0,
        1.08
        * max(float(np.max(arrays[method]["number_of_forces"])) for method in METHODS),
    )
    for axis in axes:
        axis.set_ylim(0.0, ymax)
    figure.canvas.draw()
    for axis, letter in zip(axes, "ab"):
        position = axis.get_position()
        figure.text(
            position.x0 - 0.03,
            position.y1 + 0.012,
            letter,
            ha="left",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    handles = [
        Line2D(
            [0], [0], label=method, color=styles[method]["color"],
            linestyle=styles[method]["linestyle"], linewidth=styles[method]["linewidth"],
            marker=styles[method]["marker"], markersize=3.3 if method == "IVM" else 3.1,
            markerfacecolor="white", markeredgewidth=0.6,
        )
        for method in METHODS
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
        columnspacing=1.5,
        handlelength=2.35,
        handletextpad=0.5,
        frameon=False,
    )
    if preview_note:
        figure.text(
            0.975, 0.965, preview_note, ha="right", va="top", fontsize=6.4, color="#666666"
        )
    figure.savefig(
        output,
        format="pdf",
        dpi=600,
        metadata={
            "Title": "Barnase-Barstar IVM versus ODE-NEB cost to precision",
            "Subject": "Discrete I_perp and same-path Cartesian arc500 Delta E; observed lower envelopes",
        },
    )
    plt.close(figure)


def synthetic_points() -> dict[str, list[dict]]:
    nf_ivm = np.asarray([60, 150, 300, 720, 1329, 2550, 5100, 8982])
    nf_ode = np.asarray([60, 150, 300, 1320, 2040, 4080, 8160, 9990])
    i_ivm = np.asarray([6.8e7, 2.9e7, 8.9e5, 1.1e6, 952.0, 1.2e3, 11.0, 2.5])
    i_ode = np.asarray([6.8e7, 4.5e7, 1.2e7, 9.0e5, 1.4e6, 1.4e3, 3.0e3, 18.0])
    d_ivm = np.asarray([8.2, 7.4, 7.65, 5.5, 5.0, 5.2, 4.62, 4.55])
    d_ode = np.asarray([8.2, 8.0, 8.15, 6.9, 7.1, 5.8, 6.0, 4.72])
    return {
        method: [
            {
                "method": method,
                "label": f"synthetic_{method}_{int(nf)}",
                "optimization_N_F": int(nf),
                "selected_path_sha256": "synthetic",
                "dense_coordinate_sha256": "synthetic",
                "dense_energy_sha256": "synthetic",
                "dense_force_sha256": "synthetic",
                "discrete_I_perp": float(i_value),
                "densified_Delta_E_kcal_per_mol": float(d_value),
                "optimization_N_F_contribution": 0,
            }
            for nf, i_value, d_value in zip(
                nf_ivm if method == "IVM" else nf_ode,
                i_ivm if method == "IVM" else i_ode,
                d_ivm if method == "IVM" else d_ode,
            )
        ]
        for method in METHODS
    }


def atomic_publish(output: Path, provenance: dict, arrays: dict, analysis: dict, preview_note: str | None) -> dict:
    provenance_path = output.with_suffix(".json")
    require(not output.exists() and not provenance_path.exists(), "refusing to overwrite figure/provenance")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(output.parent), prefix=".bb-cost-joint-ef-") as temporary:
        temp_pdf = Path(temporary) / output.name
        temp_json = Path(temporary) / provenance_path.name
        render(arrays, analysis, temp_pdf, preview_note=preview_note)
        provenance["figure_sha256"] = sha256_file(temp_pdf)
        with temp_json.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_pdf, output)
        os.link(temp_json, provenance_path)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return provenance


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--synthetic-preview", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    output = args.output.expanduser().resolve()
    require(output.suffix.lower() == ".pdf", "output must be a PDF")
    require(not (args.synthetic_preview and args.contract), "synthetic preview cannot use a formal contract")
    require(args.synthetic_preview or args.contract is not None, "formal rendering requires --contract")
    if args.synthetic_preview:
        require(not args.validate_only, "synthetic preview must render for visual QA")
        points = synthetic_points()
        contract_report = None
        packs = []
        status = "SYNTHETIC_STYLE_PREVIEW_NOT_SCIENTIFIC"
        preview_note = "synthetic style preview - not scientific data"
    else:
        contract_report, points, packs = validate_figure_contract(args.contract, output)
        status = "PASS_AUDITED_JOINT_EF_OBSERVED_ONLY"
        preview_note = None
    arrays, analysis = analyze_observed(points)
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "status": status,
        "created_utc": utc_now(),
        "figure": str(output),
        "figure_contract": contract_report,
        "source_packs": packs,
        "point_counts": {method: len(points[method]) for method in METHODS},
        "raw_points": {
            method: [
                {
                    "optimization_N_F": int(row["optimization_N_F"]),
                    "discrete_I_perp": float(row["discrete_I_perp"]),
                    "densified_Delta_E_kcal_per_mol": float(row["densified_Delta_E_kcal_per_mol"]),
                    "selected_path_sha256": str(row["selected_path_sha256"]),
                    "dense_coordinate_sha256": str(row["dense_coordinate_sha256"]),
                    "dense_energy_sha256": str(row["dense_energy_sha256"]),
                    "dense_force_sha256": str(row["dense_force_sha256"]),
                    "optimization_N_F_contribution": 0,
                }
                for row in sorted(points[method], key=lambda item: item["optimization_N_F"])
            ]
            for method in METHODS
        },
        "analysis": analysis,
        "legend_labels": list(METHODS),
        "vertical_axis": "Number of Forces",
        "visible_arrow_text": "numeric multiplier only",
        "visible_arrow_position": "right of vertical arrow at arithmetic line midpoint",
        "ratio_interpretation": "observed-passage cost ratio only",
        "unmeasured_first_passage_claim": False,
        "same_original_32_image_path_for_both_metrics": True,
        "arc_length_points": N_DENSE,
        "dense_full_force_vectors_verified_per_point": N_DENSE,
        "diagnostic_joint_EF_optimization_N_F_contribution": 0,
        "plotter": file_binding(Path(__file__).resolve()),
        "muller_alanine_style_reference": file_binding(STYLE_REFERENCE),
    }
    if args.validate_only:
        print(json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False))
        return
    result = atomic_publish(output, provenance, arrays, analysis, preview_note)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
