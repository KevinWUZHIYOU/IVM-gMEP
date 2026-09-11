#!/usr/bin/env python3
"""Freeze a minimal, evidence-bound ODE--NEB arc500 follow-up.

This selector is deliberately zero-OpenMM.  It reconstructs the retained
32-image ODE--NEB state at every paid force batch from the sealed optimization
WAL, inventories exact discrete ``I_perp`` values and completed arc-length-500
barriers, and freezes only the next path(s) needed by a declared claim mode.

``observed`` mode is the economical diagnostic requested for the final figure:
among unmeasured paths carrying an exact paid-force ``I_perp``, it selects the
path closest to the final IR--IVM ``I_perp`` threshold.  The resulting barrier
claim remains explicitly an *observed first passage*.

``exact`` mode is a chronological certification scan.  It selects the earliest
unmeasured retained path(s), stopping only after every earlier distinct path is
known.  This is the only generally valid way to certify an exact first passage
for a non-monotone densified barrier; no binary-search or monotonicity
assumption is made.

The output is a preregistration, not an execution authorization.  This module
does not construct an OpenMM context and does not evaluate an energy or force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from bb_core import path_diagnostics


SCHEMA = "bb-ode-neb-targeted-arc500-selection-v1"
EXPECTED_LABEL = "bbcr30_neb_ode1e6_k0p1_prod_20260814_r1"
EXPECTED_BATCHES = 333
EXPECTED_INTERNAL = 30
EXPECTED_IMAGES = 32
EXPECTED_ATOMS = 3132
N_DENSE_POINTS = 500
COORDINATE_SPACE = "unweighted_3N_cartesian"
INTERPOLATION = "piecewise_linear"
SAMPLING = "uniform_cartesian_arclength"
GRID_FORMULA = "s_j=j*L/499,j=0,...,499"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def official_array_sha256(value: np.ndarray) -> str:
    """Hash an array exactly as the sealed ODE--NEB ledger does."""

    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    require(bool(np.all(np.isfinite(array))), "official array is non-finite")
    metadata = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(metadata)
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} does not contain a JSON object")
    return value


def _load_coordinate_archive(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as saved:
        coordinates = np.asarray(saved["coordinates30_A"], dtype=np.float64)
        endpoint0 = np.asarray(saved["endpoint0_A"], dtype=np.float64)
        endpoint1 = np.asarray(saved["endpoint1_A"], dtype=np.float64)
        embedded = str(np.asarray(saved["path32_sha256"]).item())
    require(
        coordinates.shape == (EXPECTED_INTERNAL, EXPECTED_ATOMS, 3),
        f"coordinate archive has wrong internal shape: {path}",
    )
    require(
        endpoint0.shape == endpoint1.shape == (EXPECTED_ATOMS, 3),
        f"coordinate archive has wrong endpoint shape: {path}",
    )
    path32 = np.concatenate((endpoint0[None], coordinates, endpoint1[None]), axis=0)
    official = official_array_sha256(path32)
    require(official == embedded, f"embedded official path hash mismatch: {path}")
    return path32, coordinates, official


def load_official_paid_ledger(source_root: Path) -> tuple[list[dict], dict]:
    """Return all 333 retained paid states, not rejected trial proposals."""

    source = Path(source_root).expanduser().resolve(strict=True)
    complete_path = source / "OPTIMIZATION_COMPLETE.json"
    audit_path = source / "OPTIMIZATION_AUDIT.json"
    wal_path = source / "optimization_force_wal.jsonl"
    endpoints_path = source / "frozen_endpoints_A.npy"
    for path in (complete_path, audit_path, wal_path, endpoints_path):
        require(path.is_file(), f"missing sealed ODE--NEB input: {path}")

    complete = _read_json(complete_path)
    audit = _read_json(audit_path)
    expected = {
        "label": EXPECTED_LABEL,
        "paid_batches": EXPECTED_BATCHES,
        "paid_nf": EXPECTED_BATCHES * EXPECTED_INTERNAL,
        "images": EXPECTED_IMAGES,
        "interior_images": EXPECTED_INTERNAL,
        "endpoint_physical_calls": 0,
    }
    for key, value in expected.items():
        require(complete.get(key) == value, f"terminal metadata mismatch: {key}")
    require(
        complete.get("optimization_wal_sha256") == sha256_file(wal_path),
        "optimization WAL hash mismatch",
    )
    require(
        audit.get("pass") is True
        or audit.get("consistent") is True
        or audit.get("status") == "PASS",
        "optimization audit is not passing",
    )

    reserves: dict[int, dict] = {}
    commits: dict[int, dict] = {}
    previous_sequence = -1
    with wal_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            sequence = int(event["sequence"])
            require(sequence == previous_sequence + 1, "non-contiguous WAL sequence")
            previous_sequence = sequence
            kind = event.get("event")
            if kind not in {"batch_reserve", "batch_commit"}:
                continue
            batch_id = int(event["batch_id"])
            target = reserves if kind == "batch_reserve" else commits
            require(batch_id not in target, f"duplicate {kind} for batch {batch_id}")
            target[batch_id] = event
    expected_ids = list(range(1, EXPECTED_BATCHES + 1))
    require(sorted(reserves) == expected_ids, "WAL lacks a complete reserve ledger")
    require(sorted(commits) == expected_ids, "WAL lacks a complete commit ledger")

    frozen_endpoints = np.asarray(np.load(endpoints_path, allow_pickle=False), dtype=np.float64)
    require(
        frozen_endpoints.shape == (2, EXPECTED_ATOMS, 3),
        "frozen endpoint array has wrong shape",
    )

    archives_by_hash: dict[str, dict] = {}
    for batch_id in expected_ids:
        reserve = reserves[batch_id]
        archive = (source / str(reserve["batch_coordinate_file"])).resolve(strict=True)
        expected_name = f"batch_{batch_id:04d}_paid_coordinates30_A.npz"
        require(archive.name == expected_name, f"batch {batch_id} archive routing mismatch")
        archive_sha = sha256_file(archive)
        require(
            archive_sha == str(reserve["batch_coordinate_file_sha256"]),
            f"batch {batch_id} archive SHA-256 mismatch",
        )
        path32, _, official = _load_coordinate_archive(archive)
        require(
            np.array_equal(path32[0], frozen_endpoints[0])
            and np.array_equal(path32[-1], frozen_endpoints[1]),
            f"batch {batch_id} fixed endpoints mismatch",
        )
        require(
            official == str(reserve["path32_sha256"]),
            f"batch {batch_id} reserve path identity mismatch",
        )
        archives_by_hash[official] = {
            "source_batch_id": batch_id,
            "coordinate_archive": str(archive),
            "coordinate_archive_sha256": archive_sha,
            "official_path32_sha256": official,
        }

    rows: list[dict] = []
    previous_current: str | None = None
    for paid_batch_id in expected_ids:
        commit = commits[paid_batch_id]
        nf = int(commit["nf_after"])
        require(nf == paid_batch_id * EXPECTED_INTERNAL, "paid N_F sequence mismatch")
        disposition = str(commit["disposition"])
        current_hash = str(commit["current_path32_sha256"])
        if disposition == "rejected":
            require(
                current_hash == previous_current,
                f"rejected batch {paid_batch_id} did not retain its predecessor",
            )
        else:
            require(
                disposition in {"current_initial", "accepted"},
                f"unsupported batch disposition: {disposition}",
            )
            require(
                current_hash == str(commit["paid_path32_sha256"]),
                f"batch {paid_batch_id} accepted/current identity mismatch",
            )
        require(current_hash in archives_by_hash, "retained path has no coordinate archive")
        residual = commit.get("optimizer_metadata", {}).get("current_ase_residual")
        require(
            isinstance(residual, (int, float)) and math.isfinite(float(residual)),
            f"batch {paid_batch_id} lacks finite ASE residual",
        )
        rows.append(
            {
                "paid_batch_id": paid_batch_id,
                "optimization_N_F": nf,
                "disposition": disposition,
                "ase_current_residual_eV_per_A": float(residual),
                **archives_by_hash[current_hash],
            }
        )
        previous_current = current_hash

    unique_hashes = {row["official_path32_sha256"] for row in rows}
    dispositions = [row["disposition"] for row in rows]
    require(len(unique_hashes) == 284, "expected exactly 284 distinct retained paths")
    require(dispositions.count("rejected") == 49, "expected exactly 49 rejected trials")
    require(dispositions.count("accepted") == 283, "expected exactly 283 accepted trials")
    bindings = {
        "optimization_complete": {"path": str(complete_path), "sha256": sha256_file(complete_path)},
        "optimization_audit": {"path": str(audit_path), "sha256": sha256_file(audit_path)},
        "optimization_wal": {"path": str(wal_path), "sha256": sha256_file(wal_path)},
        "frozen_endpoints": {"path": str(endpoints_path), "sha256": sha256_file(endpoints_path)},
    }
    return rows, bindings


def load_exact_force_receipts(
    receipt_paths: Sequence[Path], paid_rows: Sequence[dict]
) -> tuple[dict[str, float], list[dict], list[dict]]:
    """Reconstruct exact article-metric I_perp from already-paid force arrays."""

    by_nf = {int(row["optimization_N_F"]): row for row in paid_rows}
    values: dict[str, float] = {}
    bindings: list[dict] = []
    excluded: list[dict] = []
    for raw_path in receipt_paths:
        receipt = Path(raw_path).expanduser().resolve(strict=True)
        with np.load(receipt, allow_pickle=False) as saved:
            coordinates = np.asarray(saved["coordinates30_A"], dtype=np.float64)
            forces = np.asarray(saved["forces_eV_per_A"], dtype=np.float64)
            nf = int(np.asarray(saved["nf_after"]).item())
            batch_id = int(np.asarray(saved["batch_id"]).item())
        require(nf in by_nf, f"force receipt N_F is outside official ledger: {receipt}")
        row = by_nf[nf]
        require(batch_id == int(row["paid_batch_id"]), "force receipt batch/N_F mismatch")
        path32, retained_coordinates, official = _load_coordinate_archive(
            Path(row["coordinate_archive"])
        )
        receipt_is_retained = (
            coordinates.shape == retained_coordinates.shape
            and np.array_equal(coordinates, retained_coordinates)
        )
        if not receipt_is_retained:
            require(
                row["disposition"] == "rejected",
                f"accepted force receipt is not the retained path at N_F={nf}",
            )
            # A rejected ODE trial is paid and has a valid force receipt, but
            # the scientific output state at the same N_F is its predecessor.
            # Verify the receipt against the sealed trial archive, then exclude
            # it instead of silently assigning the trial metric to the current
            # path.
            trial_archive = (
                Path(row["coordinate_archive"]).parent
                / f"batch_{batch_id:04d}_paid_coordinates30_A.npz"
            ).resolve(strict=True)
            _, trial_coordinates, trial_hash = _load_coordinate_archive(trial_archive)
            require(
                coordinates.shape == trial_coordinates.shape
                and np.array_equal(coordinates, trial_coordinates),
                f"rejected force receipt does not match its paid trial at N_F={nf}",
            )
            excluded.append(
                {
                    "path": str(receipt),
                    "sha256": sha256_file(receipt),
                    "optimization_N_F": nf,
                    "paid_batch_id": batch_id,
                    "paid_trial_path32_sha256": trial_hash,
                    "retained_path32_sha256": str(row["official_path32_sha256"]),
                    "reason": "paid ODE trial was rejected; its force cannot define current-path I_perp",
                }
            )
            continue
        require(forces.shape == coordinates.shape, "force receipt shape mismatch")
        require(official == row["official_path32_sha256"], "retained path hash mismatch")
        diagnostic = path_diagnostics(
            path32.reshape(EXPECTED_IMAGES, -1),
            np.zeros(EXPECTED_IMAGES, dtype=np.float64),
            forces.reshape(EXPECTED_INTERNAL, -1),
        )
        value = float(diagnostic["i_perp"])
        require(math.isfinite(value) and value >= 0.0, "invalid reconstructed I_perp")
        previous = values.get(official)
        require(previous is None or previous == value, "conflicting exact I_perp values")
        values[official] = value
        bindings.append(
            {
                "path": str(receipt),
                "sha256": sha256_file(receipt),
                "optimization_N_F": nf,
                "paid_batch_id": batch_id,
                "official_path32_sha256": official,
                "discrete_I_perp_eV2_per_A": value,
            }
        )
    return (
        values,
        sorted(bindings, key=lambda row: int(row["optimization_N_F"])),
        sorted(excluded, key=lambda row: int(row["optimization_N_F"])),
    )


def _npz_scalar(saved: np.lib.npyio.NpzFile, name: str):
    require(name in saved.files, f"diagnostic lacks {name}")
    return np.asarray(saved[name]).item()


def load_arc500_pairs(
    pairs: Sequence[tuple[Path, Path]], official_hash_by_nf: Mapping[int, str]
) -> tuple[dict[str, float], dict[str, float], list[dict]]:
    """Load only complete, zero-force, hash-bound arc500 diagnostics."""

    delta: dict[str, float] = {}
    exact_i: dict[str, float] = {}
    bindings: list[dict] = []
    for result_raw, audit_raw in pairs:
        result = Path(result_raw).expanduser().resolve(strict=True)
        audit_path = Path(audit_raw).expanduser().resolve(strict=True)
        audit = _read_json(audit_path)
        require(audit.get("status") == "PASS", f"arc500 audit is not PASS: {audit_path}")
        require(audit.get("output_sha256") == sha256_file(result), "arc500 output hash mismatch")
        require(int(audit.get("diagnostic_force_requests", 0)) == 0, "arc500 requested a force")
        require(int(audit.get("optimization_N_F_contribution", 0)) == 0, "arc500 changed N_F")
        with np.load(result, allow_pickle=False) as saved:
            hashes = np.asarray(saved["official_path32_sha256"]).astype(str)
            nfs = np.asarray(saved["optimization_N_F"], dtype=np.int64)
            i_values = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
            delta_values = np.asarray(
                saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64
            )
            attempts = np.asarray(
                saved["diagnostic_energy_attempts_per_path"], dtype=np.int64
            )
            forces = np.asarray(saved["diagnostic_force_requests_per_path"], dtype=np.int64)
            require(int(_npz_scalar(saved, "number_of_points")) == N_DENSE_POINTS, "not arc500")
            require(str(_npz_scalar(saved, "coordinate_space")) == COORDINATE_SPACE, "coordinate space mismatch")
            require(str(_npz_scalar(saved, "interpolation")) == INTERPOLATION, "interpolation mismatch")
            require(str(_npz_scalar(saved, "sampling")) == SAMPLING, "sampling mismatch")
            require(str(_npz_scalar(saved, "grid_formula")) == GRID_FORMULA, "dense grid mismatch")
            require(int(_npz_scalar(saved, "optimization_N_F_contribution")) == 0, "diagnostic changed N_F")
        count = len(hashes)
        require(
            nfs.shape == i_values.shape == delta_values.shape == attempts.shape == forces.shape == (count,),
            "arc500 row-array shape mismatch",
        )
        require(bool(np.all(attempts == N_DENSE_POINTS)), "arc500 row lacks 500 energies")
        require(bool(np.all(forces == 0)), "arc500 row contains a force request")
        for index, path_hash in enumerate(hashes):
            nf = int(nfs[index])
            require(nf in official_hash_by_nf, "arc500 N_F is outside official paid ledger")
            require(
                path_hash == official_hash_by_nf[nf],
                f"arc500 path is not the official retained path at N_F={nf}",
            )
            value = float(delta_values[index])
            i_value = float(i_values[index])
            require(math.isfinite(value) and math.isfinite(i_value), "non-finite arc500 metric")
            require(path_hash not in delta or delta[path_hash] == value, "conflicting Delta E")
            require(path_hash not in exact_i or exact_i[path_hash] == i_value, "conflicting I_perp")
            delta[path_hash] = value
            exact_i[path_hash] = i_value
        bindings.append(
            {
                "result": str(result),
                "result_sha256": sha256_file(result),
                "audit": str(audit_path),
                "audit_sha256": sha256_file(audit_path),
                "optimization_N_F": [int(value) for value in nfs],
                "official_path32_sha256": [str(value) for value in hashes],
            }
        )
    return delta, exact_i, bindings


def load_selected_thresholds(result_raw: Path, audit_raw: Path) -> tuple[dict, dict]:
    """Read a completed selected IR--IVM arc500 result and its PASS audit."""

    result = Path(result_raw).expanduser().resolve(strict=True)
    audit_path = Path(audit_raw).expanduser().resolve(strict=True)
    audit = _read_json(audit_path)
    require(audit.get("status") == "PASS", "selected IR--IVM arc500 audit is not PASS")
    require(audit.get("output_sha256") == sha256_file(result), "selected arc500 hash mismatch")
    require(int(audit.get("diagnostic_force_requests", 0)) == 0, "selected diagnostic requested force")
    require(int(audit.get("optimization_N_F_contribution", 0)) == 0, "selected diagnostic changed N_F")
    require(
        audit.get("every_path_is_bound_to_exactly_500_successful_energy_evaluations") is True,
        "selected audit does not certify 500 successful energies per path",
    )
    with np.load(result, allow_pickle=False) as saved:
        nfs = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        i_values = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        delta_values = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
        path_hashes = np.asarray(saved["selected_path_sha256"]).astype(str)
        attempts = np.asarray(saved["diagnostic_energy_attempts_per_path"], dtype=np.int64)
        forces = np.asarray(saved["diagnostic_force_requests_per_path"], dtype=np.int64)
        require(int(_npz_scalar(saved, "number_of_points")) == N_DENSE_POINTS, "selected result is not arc500")
        require(str(_npz_scalar(saved, "coordinate_space")) == COORDINATE_SPACE, "selected coordinate space mismatch")
        require(str(_npz_scalar(saved, "interpolation")) == INTERPOLATION, "selected interpolation mismatch")
        require(str(_npz_scalar(saved, "sampling")) == SAMPLING, "selected sampling mismatch")
        require(str(_npz_scalar(saved, "grid_formula")) == GRID_FORMULA, "selected grid mismatch")
    require(
        nfs.shape == i_values.shape == delta_values.shape == path_hashes.shape == attempts.shape == forces.shape,
        "selected metric arrays have inconsistent shapes",
    )
    require(len(nfs) > 0, "selected arc500 result contains no path")
    require(bool(np.all(attempts == N_DENSE_POINTS)), "selected path lacks 500 energies")
    require(bool(np.all(forces == 0)), "selected path contains diagnostic force request")
    require(bool(np.all(np.isfinite(i_values))) and bool(np.all(np.isfinite(delta_values))), "selected metrics are non-finite")

    i_index = int(np.argmin(i_values))
    delta_index = int(np.argmin(delta_values))
    thresholds = {
        "discrete_I_perp_eV2_per_A": float(i_values[i_index]),
        "discrete_I_perp_source": {
            "optimization_N_F": int(nfs[i_index]),
            "selected_path_sha256": str(path_hashes[i_index]),
        },
        "densified_Delta_E_kcal_per_mol": float(delta_values[delta_index]),
        "densified_Delta_E_source": {
            "optimization_N_F": int(nfs[delta_index]),
            "selected_path_sha256": str(path_hashes[delta_index]),
        },
    }
    binding = {
        "result": str(result),
        "result_sha256": sha256_file(result),
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
    }
    return thresholds, binding


def classify_first_passage(
    paid_rows: Sequence[Mapping[str, object]],
    values_by_hash: Mapping[str, float],
    threshold: float,
) -> dict:
    """Classify exact versus observed passage without assuming monotonicity."""

    require(math.isfinite(threshold), "first-passage threshold is non-finite")
    observed: list[tuple[int, float, str]] = []
    missing_first_nf: dict[str, int] = {}
    for row in paid_rows:
        path_hash = str(row["official_path32_sha256"])
        nf = int(row["optimization_N_F"])
        if path_hash in values_by_hash:
            value = float(values_by_hash[path_hash])
            require(math.isfinite(value), "metric ledger contains a non-finite value")
            if value <= threshold:
                observed.append((nf, value, path_hash))
        else:
            missing_first_nf.setdefault(path_hash, nf)

    first = min(observed, default=None, key=lambda item: item[0])
    if first is not None:
        nf, value, path_hash = first
        earlier_missing = sorted(
            first_nf for first_nf in missing_first_nf.values() if first_nf < nf
        )
        exact = not earlier_missing
        return {
            "claim": "exact_first_passage" if exact else "earliest_observed_first_passage",
            "observed_first_passage_N_F": nf,
            "observed_value": value,
            "official_path32_sha256": path_hash,
            "possible_exact_first_passage_N_F_interval": (
                [nf, nf] if exact else [earlier_missing[0], nf]
            ),
            "unmeasured_distinct_paths_before_observed_passage": len(earlier_missing),
        }

    if not missing_first_nf:
        return {
            "claim": "exact_no_first_passage_through_N_F_9990",
            "observed_first_passage_N_F": None,
            "possible_exact_first_passage_N_F_interval": None,
            "unmeasured_distinct_paths": 0,
        }
    return {
        "claim": "no_observed_first_passage_exact_status_undetermined",
        "observed_first_passage_N_F": None,
        "possible_exact_first_passage_N_F_interval": [
            min(missing_first_nf.values()),
            int(paid_rows[-1]["optimization_N_F"]),
        ],
        "unmeasured_distinct_paths": len(missing_first_nf),
    }


def unique_first_rows(paid_rows: Sequence[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for row in paid_rows:
        path_hash = str(row["official_path32_sha256"])
        if path_hash not in seen:
            result.append(dict(row))
            seen.add(path_hash)
    return result


def choose_paths_to_densify(
    paid_rows: Sequence[dict],
    delta_by_hash: Mapping[str, float],
    exact_i_by_hash: Mapping[str, float],
    i_threshold: float,
    delta_threshold: float,
    claim_mode: str,
    max_new_paths: int,
) -> tuple[list[dict], dict]:
    """Choose a minimal next batch without making a false exact claim."""

    require(claim_mode in {"observed", "exact"}, "invalid claim mode")
    require(max_new_paths > 0, "max_new_paths must be positive")
    unique_rows = unique_first_rows(paid_rows)
    unmeasured = [
        row for row in unique_rows if row["official_path32_sha256"] not in delta_by_hash
    ]
    passage = classify_first_passage(paid_rows, delta_by_hash, delta_threshold)

    if claim_mode == "exact":
        observed_nf = passage.get("observed_first_passage_N_F")
        candidates = [
            row
            for row in unmeasured
            if observed_nf is None or int(row["optimization_N_F"]) < int(observed_nf)
        ]
        selected = candidates[:max_new_paths]
        rationale = {
            "policy": "chronological_exact_certification_scan",
            "mathematical_reason": (
                "Delta E along the ODE--NEB iteration sequence is not assumed monotone; "
                "therefore every earlier distinct retained path must be measured before "
                "an exact first-passage statement is valid."
            ),
            "remaining_unmeasured_paths_in_required_prefix_after_this_freeze": max(
                0, len(candidates) - len(selected)
            ),
        }
        return selected, rationale

    candidates = [
        row
        for row in unmeasured
        if str(row["official_path32_sha256"]) in exact_i_by_hash
    ]
    if not candidates:
        return [], {
            "policy": "targeted_observed_search",
            "status": "NO_UNMEASURED_PATH_WITH_EXACT_PAID_FORCE_I_PERP",
            "mathematical_reason": (
                "ASE residual is not substituted for the article-defined discrete I_perp; "
                "no evidence-bound target can be ranked without another sealed paid-force receipt."
            ),
        }

    def rank(row: Mapping[str, object]) -> tuple[int, float, int]:
        i_value = float(exact_i_by_hash[str(row["official_path32_sha256"])])
        joint_side = 0 if i_value <= i_threshold else 1
        scale_distance = abs(
            math.log(
                max(i_value, np.finfo(float).tiny)
                / max(i_threshold, np.finfo(float).tiny)
            )
        )
        return joint_side, scale_distance, int(row["optimization_N_F"])

    selected = sorted(candidates, key=rank)[:max_new_paths]
    return selected, {
        "policy": "targeted_observed_search",
        "ranking": (
            "prefer exact-paid-force ODE paths satisfying the final IR--IVM I_perp "
            "threshold, then minimize absolute logarithmic I_perp distance, then N_F"
        ),
        "claim_limit": (
            "This ranking is a diagnostic heuristic only; it cannot certify an exact "
            "Delta E first passage because no monotonic relation between I_perp and "
            "densified Delta E is assumed."
        ),
    }


def _inventory_summary(
    paid_rows: Sequence[dict], exact_i: Mapping[str, float], delta: Mapping[str, float]
) -> dict:
    unique = unique_first_rows(paid_rows)
    exact_i_nfs = [
        int(row["optimization_N_F"])
        for row in unique
        if row["official_path32_sha256"] in exact_i
    ]
    delta_nfs = [
        int(row["optimization_N_F"])
        for row in unique
        if row["official_path32_sha256"] in delta
    ]
    return {
        "paid_N_F_state_count": len(paid_rows),
        "distinct_retained_path_count": len(unique),
        "accepted_trial_count": sum(row["disposition"] == "accepted" for row in paid_rows),
        "rejected_trial_count": sum(row["disposition"] == "rejected" for row in paid_rows),
        "exact_discrete_I_perp_distinct_path_count": len(exact_i),
        "exact_discrete_I_perp_first_occurrence_N_F": exact_i_nfs,
        "arc500_distinct_path_count": len(delta),
        "arc500_first_occurrence_N_F": delta_nfs,
        "ASE_residual_is_discrete_I_perp": False,
    }


def _parse_pairs(raw_pairs: Sequence[Sequence[str]]) -> list[tuple[Path, Path]]:
    return [(Path(pair[0]), Path(pair[1])) for pair in raw_pairs]


def _write_exclusive(path: Path, payload: dict) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--force-receipt", type=Path, action="append", default=[],
        help="already-paid 30-image force receipt; repeat as needed",
    )
    parser.add_argument(
        "--arc500-pair", nargs=2, action="append", default=[],
        metavar=("RESULT", "AUDIT"),
        help="completed ODE--NEB arc500 result and PASS audit; repeat as needed",
    )
    parser.add_argument("--selected-result", type=Path)
    parser.add_argument("--selected-audit", type=Path)
    parser.add_argument("--claim-mode", choices=("observed", "exact"), default="observed")
    parser.add_argument("--max-new-paths", type=int, default=1)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--include-paid-state-inventory", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    paid_rows, official_bindings = load_official_paid_ledger(args.source_root)
    official_hash_by_nf = {
        int(row["optimization_N_F"]): str(row["official_path32_sha256"])
        for row in paid_rows
    }
    exact_i, force_bindings, excluded_force_bindings = load_exact_force_receipts(
        args.force_receipt, paid_rows
    )
    delta, diagnostic_i, arc_bindings = load_arc500_pairs(
        _parse_pairs(args.arc500_pair), official_hash_by_nf
    )
    for path_hash, value in diagnostic_i.items():
        require(
            path_hash not in exact_i or exact_i[path_hash] == value,
            "force receipt and arc500 diagnostic disagree on discrete I_perp",
        )
        exact_i[path_hash] = value
    summary = _inventory_summary(paid_rows, exact_i, delta)

    if args.inventory_only:
        require(args.output is None, "inventory-only mode does not write an output")
        payload = {"status": "PASS_ZERO_OPENMM_INVENTORY", "summary": summary}
        if args.include_paid_state_inventory:
            payload["paid_states"] = [
                {
                    **row,
                    "discrete_I_perp_eV2_per_A": exact_i.get(
                        str(row["official_path32_sha256"])
                    ),
                    "densified_Delta_E_kcal_per_mol": delta.get(
                        str(row["official_path32_sha256"])
                    ),
                }
                for row in paid_rows
            ]
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return

    require(args.output is not None, "selection mode requires --output")
    require(args.selected_result is not None, "selection mode requires --selected-result")
    require(args.selected_audit is not None, "selection mode requires --selected-audit")
    thresholds, selected_binding = load_selected_thresholds(
        args.selected_result, args.selected_audit
    )
    i_threshold = float(thresholds["discrete_I_perp_eV2_per_A"])
    delta_threshold = float(thresholds["densified_Delta_E_kcal_per_mol"])
    i_passage = classify_first_passage(paid_rows, exact_i, i_threshold)
    delta_passage = classify_first_passage(paid_rows, delta, delta_threshold)
    selected_rows, rationale = choose_paths_to_densify(
        paid_rows,
        delta,
        exact_i,
        i_threshold,
        delta_threshold,
        args.claim_mode,
        args.max_new_paths,
    )

    payload = {
        "schema": SCHEMA,
        "status": "FROZEN_ZERO_OPENMM_TARGET_SELECTION",
        "claim_mode": args.claim_mode,
        "purpose": (
            "Select the smallest next ODE--NEB arc-length-500 diagnostic set supported "
            "by the completed selected IR--IVM thresholds and the sealed paid-batch ledger."
        ),
        "physical_calls_made_by_selector": {"energy": 0, "force": 0},
        "optimization_N_F_contribution": 0,
        "execution_authorized_by_this_file": False,
        "thresholds_from_selected_IR_IVM": thresholds,
        "official_ODE_NEB_inventory": summary,
        "discrete_I_perp_first_passage": i_passage,
        "densified_Delta_E_first_passage_before_followup": delta_passage,
        "selection_rationale": rationale,
        "diagnostic_protocol_for_each_selected_path": {
            "coordinate_space": COORDINATE_SPACE,
            "interpolation": INTERPOLATION,
            "sampling": SAMPLING,
            "number_of_points": N_DENSE_POINTS,
            "include_both_endpoints": True,
            "grid_formula": GRID_FORMULA,
            "energy_evaluations_per_distinct_path": N_DENSE_POINTS,
            "force_evaluations_per_path": 0,
            "optimization_N_F_contribution": 0,
            "exact_hash_reuse_only": True,
        },
        "paths_to_densify": [
            {
                "optimization_N_F_first_occurrence": int(row["optimization_N_F"]),
                "paid_batch_id_first_occurrence": int(row["paid_batch_id"]),
                "source_batch_id": int(row["source_batch_id"]),
                "official_path32_sha256": str(row["official_path32_sha256"]),
                "coordinate_archive": str(row["coordinate_archive"]),
                "coordinate_archive_sha256": str(row["coordinate_archive_sha256"]),
                "exact_discrete_I_perp_eV2_per_A": exact_i.get(
                    str(row["official_path32_sha256"])
                ),
            }
            for row in selected_rows
        ],
        "bindings": {
            **official_bindings,
            "selected_IR_IVM_arc500": selected_binding,
            "paid_force_receipts": force_bindings,
            "excluded_paid_force_receipts": excluded_force_bindings,
            "completed_ODE_NEB_arc500": arc_bindings,
            "selector": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
        "claim_guard": {
            "ASE_residual_is_not_discrete_I_perp": True,
            "Delta_E_monotonicity_assumed": False,
            "unmeasured_earlier_paths_forbid_exact_first_passage_claim": True,
            "allowed_nonexact_term": "earliest observed first passage",
        },
    }
    if args.include_paid_state_inventory:
        payload["official_paid_states"] = [
            {
                **row,
                "discrete_I_perp_eV2_per_A": exact_i.get(
                    str(row["official_path32_sha256"])
                ),
                "densified_Delta_E_kcal_per_mol": delta.get(
                    str(row["official_path32_sha256"])
                ),
            }
            for row in paid_rows
        ]
    _write_exclusive(args.output, payload)
    print(args.output.expanduser().resolve())
    print(sha256_file(args.output.expanduser().resolve()))


if __name__ == "__main__":
    main()
