#!/usr/bin/env python3
"""Zero-OpenMM exact-prefix verifier for the selected BB IR-IVM run.

The comparison stops at the last reparameterization-committed state of the
reference 4080-upper-bound trajectory.  Wall time and the quarantined,
uncommitted terminal tail are deliberately outside the comparison domain.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


SCHEMA = "bb-irivm-selected-9990-exact-prefix-verification-v1"
EXPECTED_CRITERION_SCHEMA = "bb-irivm-long-prefix-reproducibility-criterion-v1"
EXCLUDED_ONLINE_HISTORY_KEYS = frozenset({"wall_seconds"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_float64_bytes(value: np.ndarray) -> bytes:
    array = np.asarray(value, dtype=np.dtype("<f8"), order="C")
    return np.ascontiguousarray(array).tobytes(order="C")


def canonical_float64_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(canonical_float64_bytes(value)).hexdigest()


def exact_array_bytes(first: np.ndarray, second: np.ndarray) -> bool:
    left = np.asarray(first)
    right = np.asarray(second)
    return (
        left.shape == right.shape
        and left.dtype.str == right.dtype.str
        and np.ascontiguousarray(left).tobytes(order="C")
        == np.ascontiguousarray(right).tobytes(order="C")
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def scalar(saved: np.lib.npyio.NpzFile, key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def write_exclusive_json(path: Path, payload: Dict[str, object]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", closefd=False) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _binding_path(verification: Dict[str, object], role: str) -> Path:
    bindings = verification.get("bindings", {})
    require(role in bindings, f"verification lacks {role} binding")
    record = bindings[role]
    path = Path(str(record["path"])).expanduser().resolve(strict=True)
    require(sha256_file(path) == str(record["sha256"]), f"{role} binding drift")
    return path


def _verify_terminal_receipt(
    verification_path: Path,
    expected_result: Path,
    expected_registration: Path,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    require(verification.get("status") == "PASS", "long-member verification did not pass")
    require(
        verification.get("schema") == "bb-irivm-long-budget-member-verification-v2",
        "unsupported long-member verification schema",
    )
    result_path = _binding_path(verification, "result")
    registration_path = _binding_path(verification, "registration")
    require(result_path == expected_result, "verification/result path mismatch")
    require(
        registration_path == expected_registration,
        "verification/registration path mismatch",
    )
    require(int(verification.get("endpoint_optimization_N_F", -1)) == 0, "endpoint NoF is not zero")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    require(verification.get("label") == registration.get("label"), "verification label mismatch")
    return verification, registration


def _committed_states(
    saved: np.lib.npyio.NpzFile,
) -> List[Dict[str, object]]:
    calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
    committed = np.asarray(saved["reparam_committed"], dtype=bool)
    counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
    i_perp = np.asarray(saved["i_perp"])
    stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
    stored_paths = np.asarray(saved["stored_paths"])
    require(committed.shape == calls.shape, "commit history shape mismatch")
    require(counters.shape == (calls.size, 32), "counter history shape mismatch")
    require(i_perp.shape == calls.shape, "I_perp history shape mismatch")
    require(stored_paths.shape[0] == stored_calls.size, "stored path shape mismatch")
    states: List[Dict[str, object]] = []
    for stored_index, call_value in enumerate(stored_calls):
        call = int(call_value)
        matching = np.flatnonzero(calls == call)
        require(matching.size == 1, f"call {call} is not unique")
        history_index = int(matching[0])
        if call == 1 or not bool(committed[history_index]):
            continue
        path = np.asarray(stored_paths[stored_index])
        states.append(
            {
                "call_per_image": call,
                "history_index": history_index,
                "optimization_N_F": int(counters[history_index, 1:-1].sum(dtype=np.int64)),
                "discrete_I_perp_bytes": np.ascontiguousarray(i_perp[history_index]).tobytes(),
                "path": path,
                "path_sha256_canonical_le_float64": canonical_float64_sha256(path),
            }
        )
    require(states, "trajectory has no post-initial committed stored state")
    return states


def _history_keys(
    reference: np.lib.npyio.NpzFile,
    selected: np.lib.npyio.NpzFile,
    reference_rows: int,
    selected_rows: int,
) -> List[str]:
    keys = []
    for key in reference.files:
        if key in EXCLUDED_ONLINE_HISTORY_KEYS or key not in selected.files:
            continue
        left = np.asarray(reference[key])
        right = np.asarray(selected[key])
        if (
            left.ndim >= 1
            and right.ndim >= 1
            and left.shape[0] == reference_rows
            and right.shape[0] == selected_rows
            and left.shape[1:] == right.shape[1:]
            and left.dtype.str == right.dtype.str
        ):
            keys.append(key)
    return sorted(keys)


def compare_exact_prefix(
    reference: np.lib.npyio.NpzFile,
    selected: np.lib.npyio.NpzFile,
) -> Dict[str, object]:
    reference_states = _committed_states(reference)
    selected_states = _committed_states(selected)
    require(
        len(selected_states) >= len(reference_states),
        "9990 trajectory has fewer eligible committed states than the 4080 reference",
    )
    reference_prefix = [
        (int(row["call_per_image"]), int(row["optimization_N_F"]))
        for row in reference_states
    ]
    selected_prefix = [
        (int(row["call_per_image"]), int(row["optimization_N_F"]))
        for row in selected_states[: len(reference_states)]
    ]
    require(selected_prefix == reference_prefix, "eligible call/N_F sequence is not an exact prefix")

    state_records = []
    for reference_row, selected_row in zip(
        reference_states, selected_states[: len(reference_states)]
    ):
        require(
            canonical_float64_bytes(reference_row["path"])
            == canonical_float64_bytes(selected_row["path"]),
            f"stored path mismatch at N_F={reference_row['optimization_N_F']}",
        )
        require(
            reference_row["discrete_I_perp_bytes"]
            == selected_row["discrete_I_perp_bytes"],
            f"discrete I_perp mismatch at N_F={reference_row['optimization_N_F']}",
        )
        state_records.append(
            {
                "call_per_image": int(reference_row["call_per_image"]),
                "optimization_N_F": int(reference_row["optimization_N_F"]),
                "path_sha256_canonical_le_float64": str(
                    reference_row["path_sha256_canonical_le_float64"]
                ),
            }
        )

    cutoff = int(reference_states[-1]["history_index"])
    reference_rows = int(np.asarray(reference["calls_per_image"]).size)
    selected_rows = int(np.asarray(selected["calls_per_image"]).size)
    require(cutoff < selected_rows, "9990 history is shorter than the reference committed cutoff")
    history_keys = _history_keys(reference, selected, reference_rows, selected_rows)
    require("i_perp" in history_keys, "I_perp is absent from online history comparison")
    require("counter_vectors" in history_keys, "counter history is absent from comparison")
    require("reparam_committed" in history_keys, "commit history is absent from comparison")
    mismatches = []
    for key in history_keys:
        left = np.asarray(reference[key])[: cutoff + 1]
        right = np.asarray(selected[key])[: cutoff + 1]
        if not exact_array_bytes(left, right):
            mismatches.append(key)
    require(not mismatches, "online history prefix mismatch: " + ", ".join(mismatches))

    audit_event_keys: List[str] = []
    if "discrete_audit_outer" in reference.files:
        outer = np.asarray(reference["discrete_audit_outer"], dtype=np.int64)
        event_count = int(np.searchsorted(outer, cutoff + 1, side="left"))
        require(
            np.all(outer[:event_count] <= cutoff),
            "reference discrete-audit event selection failed",
        )
        for key in reference.files:
            if key not in selected.files:
                continue
            left = np.asarray(reference[key])
            right = np.asarray(selected[key])
            if (
                key.startswith("discrete_")
                and left.ndim >= 1
                and right.ndim >= 1
                and left.shape[0] == outer.size
                and right.shape[0] >= event_count
                and left.shape[1:] == right.shape[1:]
                and left.dtype.str == right.dtype.str
            ):
                require(
                    exact_array_bytes(left[:event_count], right[:event_count]),
                    f"discrete audit prefix mismatch: {key}",
                )
                audit_event_keys.append(key)
    else:
        event_count = 0

    return {
        "reference_eligible_prefix": state_records,
        "reference_last_committed_history_index": cutoff,
        "compared_online_history_record_count": cutoff + 1,
        "compared_online_history_keys": history_keys,
        "excluded_online_history_keys": sorted(EXCLUDED_ONLINE_HISTORY_KEYS),
        "compared_discrete_audit_event_count": event_count,
        "compared_discrete_audit_keys": sorted(audit_event_keys),
        "selected_total_eligible_committed_state_count": len(selected_states),
    }


def main(argv: Sequence[str] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--criterion", type=Path, required=True)
    parser.add_argument("--reference-registration", type=Path, required=True)
    parser.add_argument("--reference-result", type=Path, required=True)
    parser.add_argument("--reference-verification", type=Path, required=True)
    parser.add_argument("--selected-registration", type=Path, required=True)
    parser.add_argument("--selected-result", type=Path, required=True)
    parser.add_argument("--selected-verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = {
        name: path.expanduser().resolve(strict=True)
        for name, path in vars(args).items()
        if name != "output"
    }
    criterion = json.loads(inputs["criterion"].read_text(encoding="utf-8"))
    require(criterion.get("schema") == EXPECTED_CRITERION_SCHEMA, "criterion schema mismatch")
    require(criterion.get("tolerance_fallback") == "forbidden", "criterion permits a tolerance fallback")

    reference_verification, reference_registration = _verify_terminal_receipt(
        inputs["reference_verification"],
        inputs["reference_result"],
        inputs["reference_registration"],
    )
    selected_verification, selected_registration = _verify_terminal_receipt(
        inputs["selected_verification"],
        inputs["selected_result"],
        inputs["selected_registration"],
    )
    require(int(reference_registration["internal_nf_budget"]) == 4080, "reference budget is not 4080")
    require(int(selected_registration["internal_nf_budget"]) == 9990, "selected budget is not 9990")
    for key in (
        "alpha",
        "residual_ratio_r",
        "inner_max_steps_m_max",
        "physical_max_step_angstrom",
        "eta",
        "history",
        "reparameterization",
        "openmm_cpu_threads",
        "input_sha256",
        "source_sha256",
    ):
        require(reference_registration[key] == selected_registration[key], f"control/source mismatch: {key}")

    reference_lock = json.loads(_binding_path(reference_verification, "lock_receipt").read_text())
    selected_lock = json.loads(_binding_path(selected_verification, "lock_receipt").read_text())
    require(reference_lock.get("host") == selected_lock.get("host"), "reference-host mismatch")
    frozen_host = criterion.get("reference_host", {})
    for key in ("hostname", "architecture", "processor"):
        require(
            reference_lock.get("host", {}).get(key) == frozen_host.get(key),
            f"criterion/reference-host mismatch: {key}",
        )

    with np.load(inputs["reference_result"], allow_pickle=False) as reference, np.load(
        inputs["selected_result"], allow_pickle=False
    ) as selected:
        comparison = compare_exact_prefix(reference, selected)

    payload = {
        "schema": SCHEMA,
        "status": "PASS_EXACT_PREFIX",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scientific_scope": (
            "the selected 9990-N_F trajectory exactly reproduces every eligible "
            "reparameterization-committed state of the reference-host 4080-upper-bound trajectory"
        ),
        "endpoint_optimization_N_F_reference": int(reference_verification["endpoint_optimization_N_F"]),
        "endpoint_optimization_N_F_selected": int(selected_verification["endpoint_optimization_N_F"]),
        "tolerance_fallback_used": False,
        **comparison,
        "bindings": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in inputs.items()
        },
    }
    write_exclusive_json(args.output, payload)
    print(f"Saved {args.output.expanduser().resolve()}")
    print(f"sha256={sha256_file(args.output.expanduser().resolve())}")


if __name__ == "__main__":
    main()
