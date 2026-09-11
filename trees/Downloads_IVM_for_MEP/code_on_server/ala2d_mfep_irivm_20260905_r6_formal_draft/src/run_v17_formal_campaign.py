#!/usr/bin/env python3
"""Candidate-agnostic runner and audit scaffold for Ala2D V17 formal MD.

``validate-draft`` is the only command that can succeed with the checked-in
registration.  Every command capable of constructing a restrained-MD oracle
first requires a different, immutable, fully sealed registration with explicit
authorization and no placeholders.  The gate runs before a result directory,
lock file, execution manifest, or oracle is created.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ala2d_irivm_v17 import (
    IMPLEMENTATION_ID,
    IRIVMControls,
    N_INTERNAL,
    load_public_trajectory,
    run_irivm,
)
from ala2d_oracles import RestrainedMDOracle
from formal_contract import (
    ATTEMPT_AUDIT_SCHEMA,
    COMPLETE_BAND_NOF,
    EXECUTABLE_REGISTRATION_SCHEMA,
    FORMAL_INCREMENTAL_NOF_BUDGET,
    HOST_GLOBAL_LOCK_RELATIVE_TO_WORKSPACE,
    PUBLIC_LEDGER_SCHEMA,
    SEALED_STATUS,
    FormalContractError,
    assert_registration_is_not_executable,
    load_json_object,
    validate_draft_registration,
    validate_public_trajectory,
)


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parents[1]
DRAFT_REGISTRATION = ROOT / "registrations/V17_FORMAL_CAMPAIGN_DRAFT.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TERMINAL_ATTEMPT_STATUSES = frozenset(
    {
        "failed_before_mean_force",
        "succeeded",
        "recovered_success",
        "mean_force_returned_but_ineligible",
    }
)
SUCCESS_ATTEMPT_STATUSES = frozenset({"succeeded", "recovered_success"})


class FormalCampaignError(RuntimeError):
    """Fail-closed registration, execution, persistence, or ledger error."""


@dataclass(frozen=True)
class SealedCandidate:
    parameter_id: str
    controls: IRIVMControls


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def atomic_write_once_json(path: Path, payload: Mapping[str, object]) -> None:
    """Publish immutable canonical JSON or require byte-identical replay."""

    expected = canonical_json_bytes(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise FormalCampaignError(f"unsafe write-once JSON path: {path}")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if path.read_bytes() != expected:
            raise FormalCampaignError(f"write-once JSON differs: {path}")
        return
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(expected)
        stream.flush()
        os.fsync(stream.fileno())


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise FormalCampaignError(f"registration.{key} must be an object")
    return value


def _integer_seed_list(parent: Mapping[str, object], key: str, count: int) -> Tuple[int, ...]:
    value = parent.get(key)
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(type(item) is not int for item in value)
    ):
        raise FormalCampaignError(f"registration.seeds.{key} must contain {count} integers")
    result = tuple(int(item) for item in value)
    if len(set(result)) != len(result):
        raise FormalCampaignError(f"registration.seeds.{key} contains duplicates")
    if any(value <= 0 or value >= 2**31 for value in result):
        raise FormalCampaignError(f"registration.seeds.{key} contains an invalid seed")
    return result


def _sealed_sha256(parent: Mapping[str, object], key: str, label: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FormalCampaignError(f"{label}.{key} is not a sealed SHA-256 digest")
    return value


def _nonempty_text(parent: Mapping[str, object], key: str, label: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise FormalCampaignError(f"{label}.{key} must be nonempty text")
    return value


def _validate_hash_table_shape(
    parent: Mapping[str, object], key: str, label: str
) -> None:
    table = parent.get(key)
    if not isinstance(table, Mapping) or not table:
        raise FormalCampaignError(f"{label}.{key} must be a nonempty hash table")
    for raw_path, raw_digest in table.items():
        if not isinstance(raw_path, str) or not raw_path:
            raise FormalCampaignError(f"{label}.{key} contains an invalid path")
        if not isinstance(raw_digest, str) or SHA256_RE.fullmatch(raw_digest) is None:
            raise FormalCampaignError(
                f"{label}.{key}[{raw_path!r}] is not a sealed SHA-256 digest"
            )


def load_draft(path: Path = DRAFT_REGISTRATION) -> Mapping[str, object]:
    registration = load_json_object(path)
    validate_draft_registration(registration)
    return registration


def validate_executable_registration(
    registration: Mapping[str, object],
) -> Tuple[SealedCandidate, Tuple[int, ...], Tuple[int, ...]]:
    """Validate a future sealed registration; never upgrades the r6 draft."""

    try:
        assert_registration_is_not_executable(registration)
    except FormalContractError as error:
        raise FormalCampaignError(str(error)) from error
    if registration.get("schema_version") != EXECUTABLE_REGISTRATION_SCHEMA:
        raise FormalCampaignError("wrong executable registration schema")
    if registration.get("status") != SEALED_STATUS:
        raise FormalCampaignError("formal registration is not sealed")

    candidate = _mapping(registration, "candidate")
    selected = candidate.get("selected_parameter")
    if not isinstance(selected, Mapping):
        raise FormalCampaignError("sealed candidate must be an explicit parameter object")
    parameter_id = selected.get("parameter_id")
    if not isinstance(parameter_id, str) or not parameter_id:
        raise FormalCampaignError("sealed candidate lacks parameter_id")
    try:
        controls = IRIVMControls(
            alpha=float(selected["alpha"]),
            r=float(selected["r"]),
            m_max=int(selected["m_max"]),
        ).validate()
    except (KeyError, TypeError, ValueError) as error:
        raise FormalCampaignError("sealed V17 controls are invalid") from error
    if candidate.get("implementation_id") != IMPLEMENTATION_ID:
        raise FormalCampaignError("sealed candidate has the wrong V17 identity")
    implementation_digest = _sealed_sha256(
        candidate, "implementation_sha256", "registration.candidate"
    )
    if implementation_digest != sha256_file(ROOT / "src/ala2d_irivm_v17.py"):
        raise FormalCampaignError("sealed V17 implementation bytes changed")
    if candidate.get("development_exact_trajectory_count_required") != 288:
        raise FormalCampaignError("candidate selection must bind all 288 development trajectories")
    for key in (
        "development_plan_path",
        "development_execution_manifest_path",
        "development_analysis_path",
        "selection_receipt_path",
    ):
        _nonempty_text(candidate, key, "registration.candidate")
    for key in (
        "development_plan_sha256",
        "development_execution_manifest_sha256",
        "development_analysis_sha256",
        "selection_receipt_sha256",
    ):
        _sealed_sha256(candidate, key, "registration.candidate")

    seeds = _mapping(registration, "seeds")
    optimization = _integer_seed_list(seeds, "optimization", 3)
    diagnostic = _integer_seed_list(seeds, "terminal_diagnostic", 10)
    bootstrap = seeds.get("bootstrap")
    contour = seeds.get("contour")
    if (
        type(bootstrap) is not int
        or type(contour) is not int
        or int(bootstrap) <= 0
        or int(bootstrap) >= 2**31
    ):
        raise FormalCampaignError("bootstrap and contour seeds must be explicit integers")
    if int(contour) not in optimization:
        raise FormalCampaignError("contour seed must be prospectively chosen from optimization seeds")
    all_seeds = optimization + diagnostic + (int(bootstrap),)
    if len(set(all_seeds)) != len(all_seeds):
        raise FormalCampaignError("optimization, diagnostic, and bootstrap seeds overlap")
    forbidden_groups = seeds.get("forbidden_prior_seed_sets")
    if (
        not isinstance(forbidden_groups, list)
        or not forbidden_groups
        or any(
            not isinstance(group, list)
            or not group
            or any(type(item) is not int for item in group)
            for group in forbidden_groups
        )
    ):
        raise FormalCampaignError("forbidden prior seed sets are malformed")
    forbidden = {int(item) for group in forbidden_groups for item in group}
    if forbidden.intersection(all_seeds):
        raise FormalCampaignError("sealed formal seeds reuse a forbidden prior seed")
    _nonempty_text(seeds, "novelty_attestation_path", "registration.seeds")
    _sealed_sha256(
        seeds, "novelty_attestation_sha256", "registration.seeds"
    )

    execution = _mapping(registration, "execution")
    if execution.get("jobs") != 1 or type(execution.get("jobs")) is not int:
        raise FormalCampaignError("formal Ala execution requires jobs=1")
    if execution.get("lock_kind") != "host-global fcntl":
        raise FormalCampaignError("formal Ala execution requires the host-global fcntl lock")
    if execution.get("lock_path_relative_to_workspace") != HOST_GLOBAL_LOCK_RELATIVE_TO_WORKSPACE:
        raise FormalCampaignError("formal Ala execution lock path changed")
    if execution.get("run_root_must_be_absent") is not True:
        raise FormalCampaignError("formal run root must be absent before activation")
    if execution.get("cross_host_execution_authorized") is not False:
        raise FormalCampaignError("cross-host formal execution is not registered")
    if execution.get("maximum_attempts_per_logical_evaluation") != 3:
        raise FormalCampaignError("formal retry cap must be exactly three attempts")
    if execution.get("mutagen_preflight_required") is not True:
        raise FormalCampaignError("formal execution may not bypass Mutagen preflight")
    _nonempty_text(
        execution, "registered_formal_md_hostname", "registration.execution"
    )
    mutagen = _mapping(execution, "mutagen")
    for key in (
        "executable",
        "version",
        "session_identifier",
        "session_name",
        "local_root",
        "remote_protocol",
        "remote_user",
        "remote_host",
        "remote_root",
        "mode",
        "hash",
        "required_status",
    ):
        _nonempty_text(mutagen, key, "registration.execution.mutagen")
    _sealed_sha256(
        mutagen, "executable_sha256", "registration.execution.mutagen"
    )
    if (
        mutagen.get("endpoints_connected_and_scanned_required") is not True
        or mutagen.get("paused_must_be_false") is not True
    ):
        raise FormalCampaignError("registered Mutagen readiness gates changed")

    hashes = _mapping(registration, "hashes")
    if hashes.get("algorithm") != "sha256":
        raise FormalCampaignError("formal hash algorithm must be sha256")
    for key in ("source", "input", "runtime"):
        _validate_hash_table_shape(hashes, key, "registration.hashes")

    nof = _mapping(registration, "nof")
    required_nof = {
        "mean_force_vector_dimension": 2,
        "complete_band": N_INTERNAL,
        "incremental_budget_per_optimization_run": FORMAL_INCREMENTAL_NOF_BUDGET,
        "common_initial_incremental_cost": 0,
        "repeated_center_is_charged_again": True,
        "hold_and_validation_bands_are_charged": True,
        "zero_nof_refresh_creates_public_state": False,
        "hidden_candidate_or_recovery_evaluations": False,
    }
    for key, expected in required_nof.items():
        if nof.get(key) != expected or type(nof.get(key)) is not type(expected):
            raise FormalCampaignError(f"registration.nof.{key} changed")
    return SealedCandidate(parameter_id, controls), optimization, diagnostic


def host_global_lock_path(workspace: Path = WORKSPACE) -> Path:
    return Path(workspace).resolve() / HOST_GLOBAL_LOCK_RELATIVE_TO_WORKSPACE


@contextmanager
def host_global_ala_lock(
    *, workspace: Path = WORKSPACE, purpose: str
) -> Iterator[Path]:
    """Acquire the one writer lock shared by every Ala campaign on this host."""

    lock_path = host_global_lock_path(workspace)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FormalCampaignError(
                f"another Ala task holds the host-global lock {lock_path}"
            ) from error
        os.ftruncate(descriptor, 0)
        os.write(
            descriptor,
            (json.dumps({"pid": os.getpid(), "purpose": purpose}, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.fsync(descriptor)
        yield lock_path
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _attempt_identity(record: Mapping[str, object]) -> Tuple[int, int]:
    band = record.get("band_id")
    image = record.get("image_index")
    if type(band) is not int or band <= 0:
        raise FormalCampaignError("MD attempt has an invalid positive band_id")
    if type(image) is not int or not 0 <= image < N_INTERNAL:
        raise FormalCampaignError("MD attempt has an invalid image_index")
    return int(band), int(image)


def audit_md_attempt_records(
    records: Sequence[Mapping[str, object]],
    trajectory: Mapping[str, object],
    *,
    parameter_id: str,
    master_seed: int,
    expected_budget: int = FORMAL_INCREMENTAL_NOF_BUDGET,
    maximum_attempts_per_logical_evaluation: int = 3,
) -> Mapping[str, object]:
    """Prove that exactly the public paid bands—and no hidden paths—were queried."""

    terminal = validate_public_trajectory(
        trajectory, expected_budget=int(expected_budget)
    )
    if (
        type(maximum_attempts_per_logical_evaluation) is not int
        or maximum_attempts_per_logical_evaluation < 1
    ):
        raise FormalCampaignError("maximum attempt count must be a positive integer")
    paths = np.asarray(trajectory["full_path_unwrapped"], dtype=np.float64)
    by_attempt: Dict[str, List[Mapping[str, object]]] = {}
    seen_event = set()
    for record in records:
        attempt_id = record.get("attempt_id")
        status = record.get("status")
        if not isinstance(attempt_id, str) or not attempt_id:
            raise FormalCampaignError("MD attempt lacks a unique attempt_id")
        if re.fullmatch(r"b\d{6}-i\d{2}-t\d+-p\d+", attempt_id) is None:
            raise FormalCampaignError("MD attempt has a malformed attempt_id")
        if status != "started" and status not in TERMINAL_ATTEMPT_STATUSES:
            raise FormalCampaignError(f"unknown MD attempt status {status!r}")
        event_key = (attempt_id, status)
        if event_key in seen_event:
            raise FormalCampaignError("duplicate atomic attempt event")
        seen_event.add(event_key)
        if record.get("parameter_id") != parameter_id:
            raise FormalCampaignError("MD attempt belongs to another parameter")
        if record.get("master_seed") != master_seed:
            raise FormalCampaignError("MD attempt belongs to another master seed")
        for digest_key in (
            "registration_sha256",
            "registration_binding_sha256",
            "seeded_mdp_sha256",
            "gmx_sha256",
            "plumed_kernel_sha256",
        ):
            digest = record.get(digest_key)
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                raise FormalCampaignError(
                    f"MD attempt lacks sealed identity field {digest_key}"
                )
        if type(record.get("seed")) is not int or type(record.get("thermostat_seed")) is not int:
            raise FormalCampaignError("MD attempt lacks exact velocity/thermostat seeds")
        input_hashes = record.get("input_sha256")
        if not isinstance(input_hashes, Mapping) or not input_hashes:
            raise FormalCampaignError("MD attempt lacks its sealed input hash table")
        if any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            for name, digest in input_hashes.items()
        ):
            raise FormalCampaignError("MD attempt contains an invalid input hash binding")
        band_id, image_index = _attempt_identity(record)
        if band_id > terminal.index:
            raise FormalCampaignError("hidden or out-of-prefix MD evaluation is present")
        center = np.asarray(record.get("center"), dtype=np.float64)
        if center.shape != (2,) or not np.array_equal(center, paths[band_id, image_index + 1]):
            raise FormalCampaignError("MD attempt center differs from its public paid path")
        expected_nof = 0 if status in {"started", "failed_before_mean_force"} else 1
        if status != "started" and record.get("nof_returned") != expected_nof:
            raise FormalCampaignError("MD attempt status has inconsistent realized NoF")
        by_attempt.setdefault(attempt_id, []).append(record)

    successful_by_logical: Dict[Tuple[int, int], Mapping[str, object]] = {}
    attempts_by_logical: Dict[Tuple[int, int], int] = {}
    identity_by_logical: Dict[Tuple[int, int], Tuple[object, ...]] = {}
    realized_nof = 0
    failed_attempts = 0
    for attempt_id, events in by_attempt.items():
        started = [row for row in events if row.get("status") == "started"]
        finished = [row for row in events if row.get("status") in TERMINAL_ATTEMPT_STATUSES]
        if len(started) != 1 or len(finished) != 1:
            raise FormalCampaignError(f"attempt {attempt_id} lacks one start and one terminal")
        if _attempt_identity(started[0]) != _attempt_identity(finished[0]):
            raise FormalCampaignError("attempt identity changed between start and terminal")
        logical = _attempt_identity(started[0])
        immutable_fields = (
            "parameter_id",
            "registration_sha256",
            "registration_binding_sha256",
            "band_id",
            "image_index",
            "master_seed",
            "seed",
            "thermostat_seed",
            "center",
            "seeded_mdp_sha256",
            "gmx_sha256",
            "plumed_kernel_sha256",
            "input_sha256",
        )
        started_identity = tuple(started[0].get(key) for key in immutable_fields)
        finished_identity = tuple(finished[0].get(key) for key in immutable_fields)
        if started_identity != finished_identity:
            raise FormalCampaignError("attempt immutable identity changed at terminal record")
        prior_identity = identity_by_logical.setdefault(logical, started_identity)
        if prior_identity != started_identity:
            raise FormalCampaignError("retry changed the registered logical evaluation identity")
        attempts_by_logical[logical] = attempts_by_logical.get(logical, 0) + 1
        if attempts_by_logical[logical] > maximum_attempts_per_logical_evaluation:
            raise FormalCampaignError("logical evaluation exceeds the registered retry cap")
        status = str(finished[0]["status"])
        if status == "failed_before_mean_force":
            failed_attempts += 1
            continue
        realized_nof += 1
        if status == "mean_force_returned_but_ineligible":
            raise FormalCampaignError("an ineligible returned mean force cannot support formal evidence")
        for digest_key in (
            "result_npz_sha256",
            "processed_topology_sha256",
            "restrained_tpr_sha256",
        ):
            digest = finished[0].get(digest_key)
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                raise FormalCampaignError(
                    f"successful MD attempt lacks artifact field {digest_key}"
                )
        if logical in successful_by_logical:
            raise FormalCampaignError("more than one successful artifact supplies one logical evaluation")
        successful_by_logical[logical] = finished[0]

    expected_logical = {
        (band, image)
        for band in range(1, terminal.index + 1)
        for image in range(N_INTERNAL)
    }
    observed_logical = {
        _attempt_identity(row)
        for events in by_attempt.values()
        for row in events
    }
    if observed_logical - expected_logical:
        raise FormalCampaignError("hidden or out-of-prefix MD evaluation is present")
    if set(successful_by_logical) != expected_logical:
        raise FormalCampaignError("public paid bands lack one unique successful artifact per image")
    expected_nof = terminal.nof
    if realized_nof != expected_nof:
        raise FormalCampaignError(
            f"realized mean-force NoF {realized_nof} differs from public cost {expected_nof}"
        )
    return {
        "schema_version": ATTEMPT_AUDIT_SCHEMA,
        "implementation_id": IMPLEMENTATION_ID,
        "parameter_id": parameter_id,
        "master_seed": int(master_seed),
        "paid_band_count": terminal.index,
        "unique_logical_evaluation_count": len(successful_by_logical),
        "realized_mean_force_nof": realized_nof,
        "failed_before_mean_force_attempt_count": failed_attempts,
        "maximum_attempts_per_logical_evaluation": maximum_attempts_per_logical_evaluation,
        "hidden_logical_evaluation_count": 0,
        "pass": True,
    }


def load_atomic_attempt_records(run_directory: Path) -> Tuple[Mapping[str, object], ...]:
    directory = Path(run_directory) / "md_attempt_records"
    if not directory.is_dir() or directory.is_symlink():
        raise FormalCampaignError("atomic MD attempt-record directory is missing or unsafe")
    rows: List[Mapping[str, object]] = []
    for path in sorted(directory.glob("*.json")):
        record = load_json_object(path)
        expected_name = f"{record.get('attempt_id')}.{record.get('status')}.json"
        if path.name != expected_name:
            raise FormalCampaignError("atomic attempt filename/content mismatch")
        rows.append(record)
    ledger_path = Path(run_directory) / "md_attempts.jsonl"
    if not ledger_path.is_file() or ledger_path.is_symlink():
        raise FormalCampaignError("append-only MD attempt ledger is missing or unsafe")
    ledger_rows: List[Mapping[str, object]] = []
    try:
        for line_number, line in enumerate(
            ledger_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                raise FormalCampaignError(
                    f"blank line in MD attempt ledger at line {line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise FormalCampaignError(
                    f"non-object MD attempt ledger row at line {line_number}"
                )
            ledger_rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FormalCampaignError("cannot parse append-only MD attempt ledger") from error
    canonical_atomic = [canonical_json_bytes(row) for row in rows]
    canonical_ledger = [canonical_json_bytes(row) for row in ledger_rows]
    if len(set(canonical_ledger)) != len(canonical_ledger):
        raise FormalCampaignError("duplicate event is present in the append-only MD ledger")
    if sorted(canonical_atomic) != sorted(canonical_ledger):
        raise FormalCampaignError(
            "atomic attempt records and append-only MD ledger are not one-to-one"
        )
    return tuple(rows)


def build_public_state_ledger(run_directory: Path) -> Mapping[str, object]:
    """Hash every exact public state after deterministic V17 replay validation."""

    run_directory = Path(run_directory).resolve()
    trajectory = load_public_trajectory(run_directory)
    terminal = validate_public_trajectory(trajectory)
    state_paths = sorted((run_directory / "states").glob("state_*.npz"))
    if len(state_paths) != terminal.index + 1:
        raise FormalCampaignError("public-state file count differs from replayed trajectory")
    rows = []
    for index, path in enumerate(state_paths):
        if path.name != f"state_{index:06d}.npz" or path.is_symlink():
            raise FormalCampaignError("public-state filenames are not exact and contiguous")
        rows.append(
            {
                "state_index": index,
                "incremental_nof": index * N_INTERNAL,
                "state_kind": str(np.asarray(trajectory["state_kind"])[index]),
                "relative_file": f"states/{path.name}",
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema_version": PUBLIC_LEDGER_SCHEMA,
        "implementation_id": IMPLEMENTATION_ID,
        "state_count_including_common_initial": len(rows),
        "terminal_incremental_nof": terminal.nof,
        "states": rows,
    }


def audit_completed_optimization_run(
    run_directory: Path,
    *,
    parameter_id: str,
    master_seed: int,
) -> Mapping[str, object]:
    run_directory = Path(run_directory).resolve()
    trajectory = load_public_trajectory(run_directory)
    completion = load_json_object(run_directory / "completion.json")
    terminal = validate_public_trajectory(trajectory, completion=completion)
    attempts = load_atomic_attempt_records(run_directory)
    attempt_audit = audit_md_attempt_records(
        attempts,
        trajectory,
        parameter_id=parameter_id,
        master_seed=master_seed,
    )
    public_ledger = build_public_state_ledger(run_directory)
    return {
        "implementation_id": IMPLEMENTATION_ID,
        "parameter_id": parameter_id,
        "master_seed": int(master_seed),
        "terminal_state_kind": terminal.state_kind,
        "terminal_pending_next_action": terminal.pending_next_action,
        "terminal_path_sha256": hashlib.sha256(
            np.ascontiguousarray(terminal.full_path_unwrapped).view(np.uint8)
        ).hexdigest(),
        "attempt_audit": attempt_audit,
        "public_state_ledger": public_ledger,
        "pass": True,
    }


def _future_execution_disabled(registration: Mapping[str, object]) -> None:
    """Central gate called before any future run/diagnostic/plot mutation."""

    validate_executable_registration(registration)
    raise FormalCampaignError(
        "sealed execution adapter is intentionally not published in this candidate-agnostic draft"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, default=DRAFT_REGISTRATION)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-draft")
    for name in ("run-optimization", "run-terminal-diagnostics", "accept", "plot-formal"):
        subparsers.add_parser(name)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    registration = load_json_object(arguments.registration)
    if arguments.command == "validate-draft":
        validate_draft_registration(registration)
        print(
            json.dumps(
                {
                    "status": "valid non-executable draft",
                    "formal_evidence": False,
                    "new_restrained_md_authorized": False,
                    "implementation_id": IMPLEMENTATION_ID,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    _future_execution_disabled(registration)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FormalCampaignError, FormalContractError, ValueError) as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        raise SystemExit(2)
