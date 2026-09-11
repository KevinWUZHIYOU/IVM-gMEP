"""Fresh-only paid accounting for 30 interior configurations per batch.

Endpoints are frozen state, are hash-checked on every query, and never enter
the physical oracle.  Every reserved interior slot is nevertheless charged,
including current, trial, rejected, failed, or otherwise rolled-back states.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
from scipy.interpolate import BSpline


INTERIOR_PER_BATCH = 30
NF_PER_BATCH = 30
MAX_BATCHES = 333
TERMINAL_NF = 9990
MATCHED_MILESTONES = (30, 240, 480, 990, 2040, 4080, 6150, 8160, 9990)
_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class AccountingInvariantError(RuntimeError):
    pass


class PermanentFailStop(RuntimeError):
    pass


class ResourceFailStop(RuntimeError):
    """Marker accepted from the resource guard before a physical call."""


class JointOracle(Protocol):
    attempt_count: int

    def evaluate(self, coordinates_A: np.ndarray) -> Any: ...


@dataclass(frozen=True)
class PaidInteriorResult:
    batch_id: int
    nf_after: int
    energies: np.ndarray
    forces: np.ndarray
    success: bool
    failed_slots: tuple[int, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_array(value: Any) -> np.ndarray:
    result = np.asarray(value, dtype="<f8")
    if not np.isfinite(result).all():
        raise AccountingInvariantError("non-finite array cannot enter the paid ledger")
    return np.ascontiguousarray(result)


def coordinate_array(value: Any) -> np.ndarray:
    result = canonical_array(value)
    if result.ndim == 1:
        if result.size % 3:
            raise AccountingInvariantError("flattened coordinate must contain 3N values")
        result = result.reshape((-1, 3))
    if result.ndim != 2 or result.shape[1] != 3:
        raise AccountingInvariantError("coordinate must have shape (N,3) or (3N,)")
    return np.ascontiguousarray(result)


def array_sha256(value: Any) -> str:
    array = canonical_array(value)
    digest = hashlib.sha256()
    digest.update(_json_bytes({"dtype": array.dtype.str, "shape": list(array.shape)}))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _deeplr_current_continuous_state_sha256(
    *, batch_id: int, arm: str, alpha: float, theta: Any, knots: Any, base: Any
) -> str:
    payload = {
        "schema": "bb-aseode30-deeplr-current-continuous-state-v1",
        "batch": int(batch_id),
        "evaluated": True,
        "arm": str(arm),
        "alpha_A2_per_eV": float(alpha),
        "representation": "clamped-cubic-C0-bspline",
        "current_theta_sha256": array_sha256(theta),
        "spline_knots_sha256": array_sha256(knots),
        "spline_base_coefficients_sha256": array_sha256(base),
    }
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_npy(path: Path, value: np.ndarray) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _exclusive_npz(path: Path, **values: Any) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            np.savez_compressed(stream, **values)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _unpack_oracle_result(value: Any) -> tuple[float, Any]:
    if hasattr(value, "energy_eV") and hasattr(value, "force_eV_per_A"):
        return float(value.energy_eV), value.force_eV_per_A
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]), value[1]
    if isinstance(value, Mapping):
        return float(value["energy_eV"]), value["force_eV_per_A"]
    raise AccountingInvariantError("oracle result must be one energy and one full Force")


class PaidInteriorLedger:
    """One non-resumable WAL for exact 30-interior-call batches."""

    def __init__(
        self,
        run_dir: Path | str,
        accounting: Mapping[str, Any],
        oracle: JointOracle,
        *,
        method: str,
        label: str,
        frozen_endpoints_A: Any,
        protocol_sha256: str,
        source_closure_sha256: str,
        before_slot: Callable[[int, int], None] | None = None,
        max_batches: int,
    ) -> None:
        if not _LABEL_RE.fullmatch(label):
            raise AccountingInvariantError("unsafe or empty fresh label")
        if method not in {"deeplr", "ivm", "neb_ase_ode"}:
            raise AccountingInvariantError(f"unsupported method {method!r}")
        for name, digest in {
            "protocol_sha256": protocol_sha256,
            "source_closure_sha256": source_closure_sha256,
        }.items():
            if not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise AccountingInvariantError(f"{name} must be lowercase SHA-256")
        max_batches = int(max_batches)
        if max_batches not in (200, 1000):
            raise AccountingInvariantError("max_batches must be 200 or 1000")
        terminal_nf = 30 * max_batches
        expected = {
            "internal_per_batch": 30,
            "endpoints_per_batch": 0,
            "nf_per_batch": 30,
            "batches": max_batches,
            "terminal_nf": terminal_nf,
            "fresh_only": True,
            "resume_allowed": False,
        }
        for key, value in expected.items():
            if accounting.get(key) != value:
                raise AccountingInvariantError(
                    f"accounting[{key!r}]={accounting.get(key)!r}; frozen value is {value!r}"
                )
        milestones = tuple(int(value) for value in accounting.get("milestones_nf", ()))
        if not milestones or milestones[-1] != terminal_nf or any(
            value <= 0 or value % 30 or value > terminal_nf for value in milestones
        ):
            raise AccountingInvariantError("milestones must be positive paid-batch NF ending at budget")

        endpoints = canonical_array(frozen_endpoints_A)
        if endpoints.ndim == 2 and endpoints.shape[0] == 2 and endpoints.shape[1] % 3 == 0:
            endpoints = endpoints.reshape((2, -1, 3))
        if endpoints.ndim != 3 or endpoints.shape[0] != 2 or endpoints.shape[2] != 3:
            raise AccountingInvariantError("frozen endpoints must have shape (2,N,3) or (2,3N)")

        self.run_dir = Path(run_dir).resolve()
        if self.run_dir.exists():
            # ResourceGuard may have exclusively created the fresh label just
            # before the ledger.  No prior Force/accounting artifact is ever
            # accepted, which keeps resume and label reuse impossible.
            forbidden = {
                "optimization_force_wal.jsonl",
                "frozen_endpoints_A.npy",
                "requests",
                "receipts",
            }
            if any((self.run_dir / name).exists() for name in forbidden):
                raise AccountingInvariantError("existing paid artifact/label cannot be reused")
            unexpected = {
                path.name for path in self.run_dir.iterdir()
            } - {"resource_monitor.jsonl"}
            if unexpected:
                raise AccountingInvariantError(
                    f"fresh guard directory contains unexpected artifacts: {sorted(unexpected)}"
                )
        else:
            self.run_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.batch_coordinate_dir = self.run_dir / "paid_batch_coordinates"
        self.milestone_receipt_dir = self.run_dir / "milestone_force_receipts"
        self.batch_coordinate_dir.mkdir(mode=0o700)
        self.milestone_receipt_dir.mkdir(mode=0o700)
        _fsync_directory(self.run_dir.parent)
        _fsync_directory(self.run_dir)
        self.method = method
        self.label = label
        self.oracle = oracle
        self.before_slot = before_slot
        self.milestones = frozenset(milestones)
        self.frozen_endpoints = np.ascontiguousarray(endpoints, dtype="<f8")
        self.endpoint_hash = array_sha256(self.frozen_endpoints)
        self.protocol_sha256 = protocol_sha256
        self.source_closure_sha256 = source_closure_sha256
        self.batch_count = 0
        self.max_batches = max_batches
        self.terminal_nf = terminal_nf
        self.reserved_nf = 0
        self.failed = False
        self.sealed = False
        self._active: dict[str, Any] | None = None
        self._sequence = -1
        self._previous_event_hash: str | None = None

        _exclusive_npy(self.run_dir / "frozen_endpoints_A.npy", self.frozen_endpoints)
        wal = self.run_dir / "optimization_force_wal.jsonl"
        descriptor = os.open(
            wal,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
            0o600,
        )
        self._stream = os.fdopen(descriptor, "ab", buffering=0)
        _fsync_directory(self.run_dir)
        self._append(
            "run_open",
            protocol_sha256=protocol_sha256,
            source_closure_sha256=source_closure_sha256,
            endpoint_array_sha256=self.endpoint_hash,
            endpoint_physical_calls_per_batch=0,
            interior_physical_calls_per_batch=30,
            terminal_nf=self.terminal_nf,
            cache_allowed=False,
            resume_allowed=False,
        )

    @property
    def has_active_batch(self) -> bool:
        """Read-only public state for runner-side permanent fail-stop handling."""

        return self._active is not None

    def _append(self, event: str, **fields: Any) -> None:
        self._sequence += 1
        record = {
            "schema": "bb-aseode30-paid-wal-v1",
            "sequence": self._sequence,
            "utc": _utc_now(),
            "event": event,
            "run_label": self.label,
            "method": self.method,
            "previous_event_sha256": self._previous_event_hash,
            **fields,
        }
        digest = hashlib.sha256(_json_bytes(record)).hexdigest()
        record["event_sha256"] = digest
        self._stream.write(_json_bytes(record) + b"\n")
        os.fsync(self._stream.fileno())
        self._previous_event_hash = digest

    def _validate_path_and_interior(self, path32_A: Any, coordinates30_A: Any) -> tuple[np.ndarray, np.ndarray]:
        path = canonical_array(path32_A)
        coordinates = canonical_array(coordinates30_A)
        if path.ndim == 2:
            if path.shape[0] != 32 or path.shape[1] % 3:
                raise AccountingInvariantError("path32 must have 32 flattened configurations")
            path = path.reshape((32, -1, 3))
        if coordinates.ndim == 2:
            if coordinates.shape[0] != 30 or coordinates.shape[1] % 3:
                raise AccountingInvariantError("coordinates30 must have 30 flattened configurations")
            coordinates = coordinates.reshape((30, -1, 3))
        if path.ndim != 3 or path.shape[0] != 32 or path.shape[2] != 3:
            raise AccountingInvariantError("path32 must have shape (32,N,3) or (32,3N)")
        if coordinates.ndim != 3 or coordinates.shape[0] != 30 or coordinates.shape[2] != 3:
            raise AccountingInvariantError("coordinates30 must have shape (30,N,3) or (30,3N)")
        if path.shape[1:] != self.frozen_endpoints.shape[1:] or coordinates.shape[1:] != path.shape[1:]:
            raise AccountingInvariantError("path/query atom count differs from frozen endpoints")
        if array_sha256(path[[0, -1]]) != self.endpoint_hash:
            raise AccountingInvariantError("query path changed a frozen endpoint")
        if not np.array_equal(coordinates, path[1:-1]):
            raise AccountingInvariantError(
                "paid coordinates must be exactly the 30 interior configurations of path32"
            )
        return np.ascontiguousarray(path), np.ascontiguousarray(coordinates)

    def evaluate_batch(
        self,
        *,
        batch_id: int,
        path32_A: Any,
        coordinates30_A: Any,
        role: str,
        state_in_hash: str,
        metadata: Mapping[str, Any] | None = None,
        continuous_state: Mapping[str, Any] | None = None,
    ) -> PaidInteriorResult:
        if self.failed:
            raise PermanentFailStop("this fresh label has permanently fail-stopped")
        if self.sealed:
            raise AccountingInvariantError("sealed label cannot pay new NF")
        if self._active is not None:
            raise AccountingInvariantError("a previous batch awaits its unique disposition")
        if batch_id != self.batch_count + 1:
            raise AccountingInvariantError("batch ids must be contiguous; replay/resume is forbidden")
        if batch_id > self.max_batches:
            raise AccountingInvariantError("next batch would exceed configured exact budget")
        if not role:
            raise AccountingInvariantError("role must identify current/trial/rejected state")
        if not re.fullmatch(r"[0-9a-f]{64}", state_in_hash):
            raise AccountingInvariantError("state_in_hash must be lowercase SHA-256")
        path, coordinates = self._validate_path_and_interior(path32_A, coordinates30_A)
        continuous_payload: dict[str, Any] | None = None
        if self.method == "deeplr":
            if continuous_state is None:
                raise AccountingInvariantError(
                    "DeepLR reserve requires paid continuous theta/parameter state"
                )
            required = {
                "continuous_representation",
                "paid_candidate_source",
                "paid_candidate_theta",
                "paid_candidate_theta_sha256",
                "formal_current_theta_sha256_before_query",
                "sample_parameters",
                "sample_parameters_sha256",
                "spline_knots",
                "spline_base_coefficients",
                "paid_coordinates30_reconstructed_sha256",
                "fixed32_paid_force_claim",
            }
            if set(continuous_state) != required:
                raise AccountingInvariantError("DeepLR continuous reserve fields are not exact")
            theta = canonical_array(continuous_state["paid_candidate_theta"])
            parameters = canonical_array(continuous_state["sample_parameters"])
            knots = canonical_array(continuous_state["spline_knots"])
            base = canonical_array(continuous_state["spline_base_coefficients"])
            if theta.ndim != 2 or theta.shape[0] != 92:
                raise AccountingInvariantError("DeepLR paid theta must have 92 coefficient vectors")
            if parameters.shape != (30,) or knots.shape != (98,) or base.shape[0] != 94:
                raise AccountingInvariantError("DeepLR spline/parameter shapes changed")
            if theta.shape[1] != coordinates.reshape((30, -1)).shape[1] or base.shape != (94, theta.shape[1]):
                raise AccountingInvariantError("DeepLR theta/base/coordinate dimensions disagree")
            if array_sha256(theta) != str(continuous_state["paid_candidate_theta_sha256"]):
                raise AccountingInvariantError("DeepLR paid theta SHA is false")
            if array_sha256(parameters) != str(continuous_state["sample_parameters_sha256"]):
                raise AccountingInvariantError("DeepLR sample-parameter SHA is false")
            if str(continuous_state["continuous_representation"]) != "clamped-cubic-C0-bspline":
                raise AccountingInvariantError("DeepLR continuous representation changed")
            candidate_source = str(continuous_state["paid_candidate_source"])
            if candidate_source not in {"current_initial", "pending_continuous_model"}:
                raise AccountingInvariantError("DeepLR paid candidate source is invalid")
            current_theta_before_sha = str(
                continuous_state["formal_current_theta_sha256_before_query"]
            )
            if not re.fullmatch(r"[0-9a-f]{64}", current_theta_before_sha):
                raise AccountingInvariantError("DeepLR pre-query current theta SHA is invalid")
            if role != candidate_source:
                raise AccountingInvariantError("DeepLR candidate source differs from paid role")
            if candidate_source == "current_initial" and current_theta_before_sha != array_sha256(theta):
                raise AccountingInvariantError("initial paid candidate differs from formal current")
            if int(continuous_state["fixed32_paid_force_claim"]) != 0:
                raise AccountingInvariantError("DeepLR fixed32 attachment cannot claim paid Force")
            coefficients = base.copy()
            coefficients[1:-1] += theta
            reconstructed = canonical_array(
                BSpline(knots, coefficients, 3, axis=0, extrapolate=False)(parameters)
            )
            paid_flat = coordinates.reshape((30, -1))
            if not np.array_equal(reconstructed, paid_flat):
                raise AccountingInvariantError(
                    "DeepLR theta/parameters do not reconstruct paid coordinates byte-for-byte"
                )
            continuous_payload = {
                "continuous_representation": str(continuous_state["continuous_representation"]),
                "paid_candidate_source": candidate_source,
                "paid_candidate_theta": theta,
                "paid_candidate_theta_sha256": array_sha256(theta),
                "formal_current_theta_sha256_before_query": current_theta_before_sha,
                "sample_parameters": parameters,
                "sample_parameters_sha256": array_sha256(parameters),
                "spline_knots": knots,
                "spline_base_coefficients": base,
                "paid_coordinates30_reconstructed_sha256": array_sha256(reconstructed),
                "fixed32_paid_force_claim": 0,
            }
            if str(continuous_state["paid_coordinates30_reconstructed_sha256"]) != array_sha256(
                paid_flat
            ):
                raise AccountingInvariantError(
                    "DeepLR caller reconstruction SHA differs from paid coordinates"
                )
        elif continuous_state is not None:
            raise AccountingInvariantError("discrete methods cannot attach DeepLR continuous state")
        nf_before = self.reserved_nf
        nf_after = nf_before + NF_PER_BATCH
        # Persist the whole paid state once, before the durable reservation.
        # Per-slot requests below reference immutable slices and their hashes;
        # this preserves exact rejected/trial geometry without 30 duplicate
        # coordinate files per batch.
        batch_coordinate_path = (
            self.batch_coordinate_dir / f"batch_{batch_id:04d}_paid_coordinates30_A.npz"
        )
        _exclusive_npz(
            batch_coordinate_path,
            coordinates30_A=coordinates,
            endpoint0_A=path[0],
            endpoint1_A=path[-1],
            path32_sha256=np.asarray(array_sha256(path)),
            state_role=np.asarray(role),
            **(
                {}
                if continuous_payload is None
                else {
                    "paid_candidate_theta": continuous_payload["paid_candidate_theta"],
                    "paid_candidate_theta_sha256": np.asarray(
                        continuous_payload["paid_candidate_theta_sha256"]
                    ),
                    "sample_parameters": continuous_payload["sample_parameters"],
                    "sample_parameters_sha256": np.asarray(
                        continuous_payload["sample_parameters_sha256"]
                    ),
                    "spline_knots": continuous_payload["spline_knots"],
                    "spline_base_coefficients": continuous_payload[
                        "spline_base_coefficients"
                    ],
                    "continuous_representation": np.asarray(
                        continuous_payload["continuous_representation"]
                    ),
                    "paid_candidate_source": np.asarray(
                        continuous_payload["paid_candidate_source"]
                    ),
                    "formal_current_theta_sha256_before_query": np.asarray(
                        continuous_payload["formal_current_theta_sha256_before_query"]
                    ),
                    "fixed32_paid_force_claim": np.asarray(0, dtype=np.int8),
                }
            ),
        )
        batch_coordinate_file_sha256 = file_sha256(batch_coordinate_path)
        self._append(
            "batch_reserve",
            batch_id=batch_id,
            nf_before=nf_before,
            nf_after=nf_after,
            slot_count=30,
            endpoint_slot_count=0,
            state_role=role,
            state_in_hash=state_in_hash,
            path32_sha256=array_sha256(path),
            endpoint_array_sha256=self.endpoint_hash,
            batch_coordinate_file=str(batch_coordinate_path.relative_to(self.run_dir)),
            batch_coordinate_file_sha256=batch_coordinate_file_sha256,
            slots=[{"slot_id": slot, "role": f"interior_{slot + 1:02d}"} for slot in range(30)],
            metadata=dict(metadata or {}),
            **(
                {}
                if continuous_payload is None
                else {
                    "continuous_state_semantics": "paid_candidate_theta_at_30_sample_parameters",
                    "paid_candidate_theta_sha256": continuous_payload[
                        "paid_candidate_theta_sha256"
                    ],
                    "paid_candidate_source": continuous_payload[
                        "paid_candidate_source"
                    ],
                    "formal_current_theta_sha256_before_query": continuous_payload[
                        "formal_current_theta_sha256_before_query"
                    ],
                    "sample_parameters_sha256": continuous_payload[
                        "sample_parameters_sha256"
                    ],
                    "paid_coordinates30_reconstructed_sha256": continuous_payload[
                        "paid_coordinates30_reconstructed_sha256"
                    ],
                    "fixed32_paid_force_claim": 0,
                }
            ),
        )
        self.reserved_nf = nf_after
        self.batch_count = batch_id
        self._active = {
            "batch_id": batch_id,
            "failed_slots": [],
            "role": role,
            "state_in_hash": state_in_hash,
            "path32_hash": array_sha256(path),
            "continuous_payload": continuous_payload,
        }

        energies = np.full(30, np.nan, dtype=np.float64)
        forces = np.full(coordinates.shape, np.nan, dtype=np.float64)

        def suppress_reserved_slots(first_slot: int, error: BaseException) -> None:
            """Give every already-reserved slot a durable terminal outcome.

            Resource fail-stop and non-``Exception`` control-flow interrupts
            must not start another physical call.  They also cannot turn the
            immutable +30 reservation into a partial batch, so each remaining
            slot receives an explicit request/failure pair linked to the one
            fsynced batch-coordinate package.
            """

            for remaining in range(first_slot, 30):
                remaining_nf = (batch_id - 1) * 30 + remaining + 1
                remaining_coordinate = coordinate_array(coordinates[remaining])
                self._append(
                    "force_request",
                    batch_id=batch_id,
                    slot_id=remaining,
                    role=f"interior_{remaining + 1:02d}",
                    state_role=role,
                    paid_nf=remaining_nf,
                    coordinate_sha256=array_sha256(remaining_coordinate),
                    coordinate_shape=list(remaining_coordinate.shape),
                    batch_coordinate_file=str(batch_coordinate_path.relative_to(self.run_dir)),
                    batch_coordinate_file_sha256=batch_coordinate_file_sha256,
                    batch_coordinate_slice_index=remaining,
                    joint_energy_full_force=True,
                    cached=False,
                    suppressed_after_fail_stop=True,
                )
                self._append(
                    "force_failure",
                    batch_id=batch_id,
                    slot_id=remaining,
                    role=f"interior_{remaining + 1:02d}",
                    paid_nf=remaining_nf,
                    error_type=type(error).__name__,
                    error_message="reserved slot not physically attempted after fail-stop",
                    oracle_attempt_index=None,
                    cached=False,
                    physical_attempt_made=False,
                    suppressed_after_fail_stop=True,
                )
                self._active["failed_slots"].append(remaining)

        for slot in range(30):
            paid_nf = (batch_id - 1) * 30 + slot + 1
            coordinate = coordinate_array(coordinates[slot])
            coordinate_hash = array_sha256(coordinate)
            self._append(
                "force_request",
                batch_id=batch_id,
                slot_id=slot,
                role=f"interior_{slot + 1:02d}",
                state_role=role,
                paid_nf=paid_nf,
                coordinate_sha256=coordinate_hash,
                coordinate_shape=list(coordinate.shape),
                batch_coordinate_file=str(batch_coordinate_path.relative_to(self.run_dir)),
                batch_coordinate_file_sha256=batch_coordinate_file_sha256,
                batch_coordinate_slice_index=slot,
                joint_energy_full_force=True,
                cached=False,
            )
            attempt_before = int(getattr(self.oracle, "attempt_count", 0))
            try:
                if self.before_slot is not None:
                    self.before_slot(batch_id, slot)
                result = self.oracle.evaluate(coordinate)
                attempt_after = int(getattr(self.oracle, "attempt_count", attempt_before + 1))
                if attempt_after != attempt_before + 1:
                    raise AccountingInvariantError("oracle must increment one physical attempt per request")
                energy, force = _unpack_oracle_result(result)
                force = coordinate_array(force)
                if force.shape != coordinate.shape or not np.isfinite(energy):
                    raise AccountingInvariantError("oracle returned invalid energy/full Force")
                self._append(
                    "force_response",
                    batch_id=batch_id,
                    slot_id=slot,
                    role=f"interior_{slot + 1:02d}",
                    paid_nf=paid_nf,
                    energy_eV=float(energy),
                    force_sha256=array_sha256(force),
                    oracle_attempt_index=attempt_after,
                    joint_energy_full_force=True,
                    cached=False,
                )
                energies[slot] = float(energy)
                forces[slot] = force
            except BaseException as error:
                attempt_after = int(getattr(self.oracle, "attempt_count", attempt_before))
                self._append(
                    "force_failure",
                    batch_id=batch_id,
                    slot_id=slot,
                    role=f"interior_{slot + 1:02d}",
                    paid_nf=paid_nf,
                    error_type=type(error).__name__,
                    error_message=str(error)[:2000],
                    oracle_attempt_index=(attempt_after if attempt_after > attempt_before else None),
                    cached=False,
                    physical_attempt_made=bool(attempt_after > attempt_before),
                )
                self._active["failed_slots"].append(slot)
                must_stop_physical = (
                    not isinstance(error, Exception)
                    or "Resource" in type(error).__name__
                    or isinstance(error, (MemoryError, RecursionError))
                )
                if must_stop_physical:
                    suppress_reserved_slots(slot + 1, error)
                    if not isinstance(error, Exception):
                        failed = tuple(int(value) for value in self._active["failed_slots"])
                        self._append(
                            "batch_force_complete",
                            batch_id=batch_id,
                            nf_after=nf_after,
                            requested_slots=30,
                            terminal_outcomes=30,
                            success=False,
                            failed_slots=list(failed),
                            milestone_full_force_receipt=None,
                            milestone_full_force_receipt_sha256=None,
                            interrupted_by=type(error).__name__,
                        )
                        raise
                    break
                # A normal oracle error does not waive any remaining paid slot.
                continue

        failed = tuple(int(slot) for slot in self._active["failed_slots"])
        milestone_receipt_file: str | None = None
        milestone_receipt_sha256: str | None = None
        if nf_after in self.milestones and not failed:
            receipt_path = (
                self.milestone_receipt_dir / f"nf{nf_after:05d}_batch_{batch_id:04d}_EF30.npz"
            )
            _exclusive_npz(
                receipt_path,
                coordinates30_A=coordinates,
                energies_eV=energies,
                forces_eV_per_A=forces,
                batch_id=np.asarray(batch_id, dtype="<i8"),
                nf_after=np.asarray(nf_after, dtype="<i8"),
            )
            milestone_receipt_file = str(receipt_path.relative_to(self.run_dir))
            milestone_receipt_sha256 = file_sha256(receipt_path)
        self._append(
            "batch_force_complete",
            batch_id=batch_id,
            nf_after=nf_after,
            requested_slots=30,
            terminal_outcomes=30,
            success=not failed,
            failed_slots=list(failed),
            milestone_full_force_receipt=milestone_receipt_file,
            milestone_full_force_receipt_sha256=milestone_receipt_sha256,
        )
        return PaidInteriorResult(
            batch_id=batch_id,
            nf_after=nf_after,
            energies=energies,
            forces=forces,
            success=not failed,
            failed_slots=failed,
        )

    def close_batch(
        self,
        *,
        batch_id: int,
        disposition: str,
        state_out_hash: str,
        path32_A: Any,
        evaluated_current: bool,
        optimizer_meta: Mapping[str, Any] | None = None,
        continuous_state: Mapping[str, Any] | None = None,
    ) -> None:
        active = self._active
        if active is None or active["batch_id"] != batch_id:
            raise AccountingInvariantError("close_batch must match the one active batch")
        if not re.fullmatch(r"[0-9a-f]{64}", state_out_hash):
            raise AccountingInvariantError("state_out_hash must be lowercase SHA-256")
        raw_path = canonical_array(path32_A)
        if raw_path.ndim == 2 and raw_path.shape[0] == 32:
            raw_interior = raw_path[1:-1]
        elif raw_path.ndim == 3 and raw_path.shape[0] == 32:
            raw_interior = raw_path[1:-1]
        else:
            raise AccountingInvariantError("path32 must contain exactly 32 configurations")
        path, coordinates = self._validate_path_and_interior(raw_path, raw_interior)
        metadata = dict(optimizer_meta or {})
        continuous_commit: dict[str, Any] = {}
        if self.method == "deeplr" and not active["failed_slots"]:
            if continuous_state is None:
                raise AccountingInvariantError("DeepLR commit requires continuous current state")
            required = {
                "continuous_state_semantics",
                "paid_candidate_theta_sha256",
                "current_theta_sha256",
                "pending_theta_sha256",
                "pending_theta_is_unevaluated",
                "sample_parameters_sha256",
                "paid_coordinates30_sha256",
                "current_continuous_state_sha256",
                "fixed32_geometric_attachment_sha256",
                "fixed32_is_geometric_attachment_only",
                "fixed32_paid_force_claim",
            }
            if set(continuous_state) != required:
                raise AccountingInvariantError("DeepLR continuous commit fields are not exact")
            paid_theta_sha = str(active["continuous_payload"]["paid_candidate_theta_sha256"])
            if (
                str(continuous_state["paid_candidate_theta_sha256"]) != paid_theta_sha
                or str(continuous_state["current_theta_sha256"]) != paid_theta_sha
            ):
                raise AccountingInvariantError("DeepLR paid candidate did not become current theta")
            if not bool(continuous_state["pending_theta_is_unevaluated"]):
                raise AccountingInvariantError("DeepLR pending theta must remain unevaluated")
            if not evaluated_current:
                raise AccountingInvariantError("successful DeepLR batch must evaluate current theta")
            if (
                str(continuous_state["sample_parameters_sha256"])
                != str(active["continuous_payload"]["sample_parameters_sha256"])
                or str(continuous_state["paid_coordinates30_sha256"])
                != str(active["continuous_payload"]["paid_coordinates30_reconstructed_sha256"])
            ):
                raise AccountingInvariantError("DeepLR commit detached from this paid sample batch")
            current_continuous_state_sha = str(
                continuous_state["current_continuous_state_sha256"]
            )
            if not re.fullmatch(r"[0-9a-f]{64}", current_continuous_state_sha):
                raise AccountingInvariantError("DeepLR current continuous-state SHA is invalid")
            # ``path`` is canonical (32,N,3); preserve that shape domain in
            # every fixed-32 attachment hash and never confuse it with the
            # flattened continuous model coefficients.
            attachment_sha = array_sha256(path)
            if (
                str(continuous_state["fixed32_geometric_attachment_sha256"])
                != attachment_sha
                or not bool(continuous_state["fixed32_is_geometric_attachment_only"])
                or int(continuous_state["fixed32_paid_force_claim"]) != 0
            ):
                raise AccountingInvariantError("DeepLR fixed32 geometric attachment claim is false")
            continuous_commit = {
                "continuous_state_semantics": str(
                    continuous_state["continuous_state_semantics"]
                ),
                "paid_candidate_theta_sha256": paid_theta_sha,
                "current_theta_sha256": paid_theta_sha,
                "pending_theta_sha256": str(continuous_state["pending_theta_sha256"]),
                "pending_theta_is_unevaluated": True,
                "sample_parameters_sha256": str(
                    continuous_state["sample_parameters_sha256"]
                ),
                "paid_coordinates30_sha256": str(
                    continuous_state["paid_coordinates30_sha256"]
                ),
                "current_continuous_state_sha256": current_continuous_state_sha,
                "fixed32_geometric_attachment_sha256": attachment_sha,
                "fixed32_is_geometric_attachment_only": True,
                "fixed32_paid_force_claim": 0,
            }
        elif self.method != "deeplr" and continuous_state is not None:
            raise AccountingInvariantError("discrete commit cannot attach continuous state")
        paid_selector = {}
        for key in (
            "paid_mean_normal_force_sq",
            "paid_rms_normal_force",
            "paid_max_energy_eV",
            "paid_p95_energy_eV",
            "paid_path_ase_residual",
        ):
            if key in metadata:
                value = float(metadata[key])
                if not np.isfinite(value):
                    raise AccountingInvariantError(f"non-finite paid selector metric {key}")
                paid_selector[key] = value
        failed = tuple(active["failed_slots"])
        if failed:
            if disposition != "failed":
                raise AccountingInvariantError("Force-failed batch may only fail-stop")
            self._append(
                "batch_failed",
                batch_id=batch_id,
                nf_after=self.reserved_nf,
                disposition=disposition,
                failed_slots=list(failed),
                state_out_hash=state_out_hash,
                path32_sha256=array_sha256(path),
                optimizer_metadata=metadata,
            )
            self._append(
                "run_fail_stop",
                batch_id=batch_id,
                nf=self.reserved_nf,
                reason="paid_force_failure",
                resume_allowed=False,
            )
            self.failed = True
        else:
            allowed = {
                "current_initial",
                "current_restart",
                "accepted",
                "rejected",
                "committed",
                "trial_pending",
                "evaluated_initial_continuous_model",
                "accepted_pending_continuous_model",
            }
            if disposition not in allowed:
                raise AccountingInvariantError(f"unrecognized paid disposition {disposition!r}")
            self._append(
                "batch_commit",
                batch_id=batch_id,
                nf_after=self.reserved_nf,
                disposition=disposition,
                state_in_hash=active["state_in_hash"],
                state_out_hash=state_out_hash,
                paid_state_role=active["role"],
                path32_sha256=array_sha256(path),
                paid_path32_sha256=active["path32_hash"],
                current_path32_sha256=array_sha256(path),
                evaluated_current=bool(evaluated_current),
                evaluated_current_semantics=(
                    "continuous_theta_paid_at_30_sample_parameters"
                    if self.method == "deeplr"
                    else "discrete_path32"
                ),
                current_state_kind=(
                    "continuous_spline_coefficients"
                    if self.method == "deeplr"
                    else "discrete_path32"
                ),
                current_path32_role=(
                    "geometric_attachment_only"
                    if self.method == "deeplr"
                    else "evaluated_discrete_current"
                ),
                fixed32_evaluated_current=(
                    False if self.method == "deeplr" else bool(evaluated_current)
                ),
                paid_geometry_kind=(
                    "continuous_model_at_30_sample_parameters"
                    if self.method == "deeplr"
                    else "discrete_path32_interior"
                ),
                update_count=1,
                best_so_far=False,
                optimizer_metadata=metadata,
                **paid_selector,
                **continuous_commit,
            )
            if self.reserved_nf in self.milestones:
                if not evaluated_current:
                    raise AccountingInvariantError(
                        "matched milestone must snapshot the current fully evaluated path"
                    )
                self._append(
                    "snapshot",
                    batch_id=batch_id,
                    nf=self.reserved_nf,
                    state_sha256=state_out_hash,
                    path32_sha256=array_sha256(path),
                    state_role=(
                        "evaluated_current_continuous_theta"
                        if self.method == "deeplr"
                        else "evaluated_current"
                    ),
                    path32_role=(
                        "geometric_attachment_only"
                        if self.method == "deeplr"
                        else "evaluated_discrete_current"
                    ),
                    fixed32_evaluated_current=(
                        False if self.method == "deeplr" else True
                    ),
                    evaluated=True,
                    best_so_far=False,
                    optimizer_metadata=metadata,
                    **paid_selector,
                    **continuous_commit,
                )
        self._active = None

    def seal(self) -> None:
        if self.failed or self._active is not None:
            raise PermanentFailStop("failed/active ledger cannot seal")
        if self.reserved_nf != self.terminal_nf or self.batch_count != self.max_batches:
            raise AccountingInvariantError("formal seal requires exact configured budget")
        self._append(
            "run_seal",
            nf=self.reserved_nf,
            batches=self.batch_count,
            endpoint_physical_calls=0,
            interior_physical_calls=self.reserved_nf,
        )
        self.sealed = True

    def fail_stop_active(self, reason: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        """Permanently close a paid batch after a non-oracle optimizer failure.

        All 30 Force outcomes already exist when this hook is used; it records
        that the method could not make a legal state disposition.  No commit or
        matched snapshot is produced, and the label cannot resume.
        """

        if self._active is None:
            if not self.failed:
                self._append(
                    "run_fail_stop",
                    nf=self.reserved_nf,
                    reason=str(reason),
                    resume_allowed=False,
                    metadata=dict(metadata or {}),
                )
                self.failed = True
            return
        active = self._active
        self._append(
            "batch_failed",
            batch_id=active["batch_id"],
            nf_after=self.reserved_nf,
            disposition="optimizer_failure_after_paid_force",
            reason=str(reason),
            failed_force_slots=list(active["failed_slots"]),
            state_in_hash=active["state_in_hash"],
            path32_sha256=active["path32_hash"],
            metadata=dict(metadata or {}),
        )
        self._append(
            "run_fail_stop",
            batch_id=active["batch_id"],
            nf=self.reserved_nf,
            reason=str(reason),
            resume_allowed=False,
        )
        self._active = None
        self.failed = True

    def close(self) -> None:
        if self._active is not None:
            raise AccountingInvariantError("cannot close with an active paid batch")
        if not self._stream.closed:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()


def audit_wal(
    path: Path | str,
    *,
    require_terminal: bool = False,
    expected_batches: int | None = None,
    expected_milestones: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Audit one paid-force WAL against an explicit budget/milestone contract.

    ``None`` preserves the historical NF9990 audit contract for old callers.
    New campaigns must pass both values explicitly; this avoids silently
    accepting a 200- or 1000-batch run under the legacy 333-batch constants.
    """

    terminal_batches = MAX_BATCHES if expected_batches is None else int(expected_batches)
    milestones = tuple(
        MATCHED_MILESTONES
        if expected_milestones is None
        else (int(value) for value in expected_milestones)
    )
    if terminal_batches <= 0:
        raise AccountingInvariantError("expected terminal batch count must be positive")
    terminal_nf = NF_PER_BATCH * terminal_batches
    if (
        not milestones
        or len(set(milestones)) != len(milestones)
        or tuple(sorted(milestones)) != milestones
        or any(value <= 0 or value % NF_PER_BATCH or value > terminal_nf for value in milestones)
    ):
        raise AccountingInvariantError("expected milestones are not sorted unique paid-batch NF")
    if require_terminal and milestones[-1] != terminal_nf:
        raise AccountingInvariantError("terminal milestone set must end at the expected budget")
    wal_path = Path(path).resolve()
    run_dir = wal_path.parent
    records = [json.loads(line) for line in wal_path.read_text(encoding="utf-8").splitlines()]
    if [record.get("sequence") for record in records] != list(range(len(records))):
        raise AccountingInvariantError("non-contiguous WAL sequence")
    previous_hash: str | None = None
    for record in records:
        if record.get("previous_event_sha256") != previous_hash:
            raise AccountingInvariantError("WAL previous-event hash chain is broken")
        recorded_hash = record.get("event_sha256")
        if not isinstance(recorded_hash, str):
            raise AccountingInvariantError("WAL event lacks its SHA-256")
        unsigned = dict(record)
        unsigned.pop("event_sha256", None)
        computed_hash = hashlib.sha256(_json_bytes(unsigned)).hexdigest()
        if recorded_hash != computed_hash:
            raise AccountingInvariantError("WAL event SHA-256 recomputation failed")
        previous_hash = recorded_hash
    reserves: dict[int, Mapping[str, Any]] = {}
    force_completes: dict[int, Mapping[str, Any]] = {}
    requests: dict[tuple[int, int], Mapping[str, Any]] = {}
    outcomes: dict[tuple[int, int], Mapping[str, Any]] = {}
    commits: dict[int, list[Mapping[str, Any]]] = {}
    snapshots: dict[int, Mapping[str, Any]] = {}
    run_open: list[Mapping[str, Any]] = []
    run_seals: list[Mapping[str, Any]] = []
    known_events = {
        "run_open", "batch_reserve", "force_request", "force_response",
        "force_failure", "batch_force_complete", "batch_commit", "snapshot",
        "batch_failed", "run_fail_stop", "run_seal",
    }
    for record in records:
        event = record.get("event")
        if event not in known_events:
            raise AccountingInvariantError(f"unknown WAL event {event!r}")
        if event == "run_open":
            run_open.append(record)
        elif event == "run_seal":
            run_seals.append(record)
        elif event == "batch_reserve":
            batch = int(record["batch_id"])
            if batch in reserves or int(record["nf_after"]) != 30 * batch:
                raise AccountingInvariantError("duplicate/misaligned batch reserve")
            if int(record["slot_count"]) != 30 or int(record["endpoint_slot_count"]) != 0:
                raise AccountingInvariantError("batch reserve is not exactly 30 interior/0 endpoint")
            reserves[batch] = record
        elif event == "force_request":
            key = (int(record["batch_id"]), int(record["slot_id"]))
            if key in requests or not 0 <= key[1] < 30:
                raise AccountingInvariantError("duplicate/out-of-range paid request")
            requests[key] = record
        elif event in {"force_response", "force_failure"}:
            key = (int(record["batch_id"]), int(record["slot_id"]))
            if key in outcomes:
                raise AccountingInvariantError("duplicate paid outcome")
            outcomes[key] = record
        elif event == "batch_force_complete":
            batch = int(record["batch_id"])
            if batch in force_completes:
                raise AccountingInvariantError("duplicate batch-force-complete event")
            force_completes[batch] = record
        elif event == "batch_commit":
            batch = int(record["batch_id"])
            commits.setdefault(batch, []).append(record)
        elif event == "snapshot":
            nf = int(record["nf"])
            if nf in snapshots:
                raise AccountingInvariantError("duplicate matched snapshot event")
            snapshots[nf] = record
    if len(run_open) != 1 or records[0].get("event") != "run_open":
        raise AccountingInvariantError("WAL must begin with exactly one run_open")
    if int(run_open[0].get("terminal_nf", -1)) != terminal_nf:
        raise AccountingInvariantError(
            "WAL run_open terminal budget differs from the expected audit contract"
        )
    if not reserves:
        raise AccountingInvariantError("WAL has no paid batch")
    batches = sorted(reserves)
    if batches != list(range(1, batches[-1] + 1)):
        raise AccountingInvariantError("batch ids are not contiguous")
    expected_keys = {(batch, slot) for batch in batches for slot in range(30)}
    if set(requests) != expected_keys or set(outcomes) != expected_keys:
        raise AccountingInvariantError("WAL contains a missing or extra paid request/outcome")
    if set(force_completes) != set(batches):
        raise AccountingInvariantError("WAL contains a missing or extra batch-force-complete")
    if set(commits) - set(batches):
        raise AccountingInvariantError("WAL contains a commit for an unreserved batch")
    if any(int(record.get("batch_id", 0)) not in batches for record in records
           if record.get("event") in {
               "batch_reserve", "force_request", "force_response", "force_failure",
               "batch_force_complete", "batch_commit", "snapshot", "batch_failed",
           }):
        raise AccountingInvariantError("WAL contains an event for an unreserved batch")
    previous_neb_current_path_hash: str | None = None
    previous_ivm_current_path_hash: str | None = None
    previous_deeplr_pending_theta_sha: str | None = None
    previous_deeplr_current_theta_sha: str | None = None
    reference_deeplr_knots: np.ndarray | None = None
    reference_deeplr_base: np.ndarray | None = None
    ode_restart_count = 0
    ivm_rejected_count = 0
    for batch in batches:
        keys = {(batch, slot) for slot in range(30)}
        if not keys <= requests.keys() or not keys <= outcomes.keys():
            raise AccountingInvariantError(f"batch {batch} lacks exact 30 requests/outcomes")
        reserve = reserves[batch]
        paid_hash = str(reserve["path32_sha256"])
        coordinate_package = run_dir / str(reserve["batch_coordinate_file"])
        if (
            not coordinate_package.is_file()
            or file_sha256(coordinate_package)
            != str(reserve["batch_coordinate_file_sha256"])
        ):
            raise AccountingInvariantError("paid coordinate package hash failed")
        with np.load(coordinate_package, allow_pickle=False) as archive:
            package_keys = set(archive.files)
            paid_interior = canonical_array(archive["coordinates30_A"])
            endpoint0 = canonical_array(archive["endpoint0_A"])
            endpoint1 = canonical_array(archive["endpoint1_A"])
            embedded_path_hash = str(np.asarray(archive["path32_sha256"]).reshape(()))
            embedded_role = str(np.asarray(archive["state_role"]).reshape(()))
        if paid_interior.ndim != 3 or paid_interior.shape[0] != 30 \
                or endpoint0.shape != paid_interior.shape[1:] \
                or endpoint1.shape != paid_interior.shape[1:]:
            raise AccountingInvariantError("paid coordinate package has invalid shapes")
        paid_path = np.concatenate((endpoint0[None], paid_interior, endpoint1[None]), axis=0)
        if array_sha256(paid_path) != paid_hash or embedded_path_hash != paid_hash:
            raise AccountingInvariantError("paid package coordinates disagree with reserve path SHA")
        if embedded_role != str(reserve["state_role"]):
            raise AccountingInvariantError("paid package state role differs from reserve")
        if str(reserve.get("method")) == "deeplr":
            continuous_keys = {
                "paid_candidate_theta",
                "paid_candidate_theta_sha256",
                "sample_parameters",
                "sample_parameters_sha256",
                "spline_knots",
                "spline_base_coefficients",
                "continuous_representation",
                "paid_candidate_source",
                "formal_current_theta_sha256_before_query",
                "fixed32_paid_force_claim",
            }
            if not continuous_keys <= package_keys:
                raise AccountingInvariantError("DeepLR paid package lacks continuous state")
            with np.load(coordinate_package, allow_pickle=False) as archive:
                theta = canonical_array(archive["paid_candidate_theta"])
                parameters = canonical_array(archive["sample_parameters"])
                knots = canonical_array(archive["spline_knots"])
                base = canonical_array(archive["spline_base_coefficients"])
                theta_sha = str(np.asarray(archive["paid_candidate_theta_sha256"]).reshape(()))
                parameters_sha = str(np.asarray(archive["sample_parameters_sha256"]).reshape(()))
                representation = str(np.asarray(archive["continuous_representation"]).reshape(()))
                candidate_source = str(np.asarray(archive["paid_candidate_source"]).reshape(()))
                current_theta_before_sha = str(
                    np.asarray(
                        archive["formal_current_theta_sha256_before_query"]
                    ).reshape(())
                )
                fixed32_claim = int(np.asarray(archive["fixed32_paid_force_claim"]).reshape(()))
            if (
                theta.shape[0] != 92
                or parameters.shape != (30,)
                or knots.shape != (98,)
                or base.shape != (94, theta.shape[1])
                or theta_sha != array_sha256(theta)
                or parameters_sha != array_sha256(parameters)
                or representation != "clamped-cubic-C0-bspline"
                or candidate_source not in {"current_initial", "pending_continuous_model"}
                or not re.fullmatch(r"[0-9a-f]{64}", current_theta_before_sha)
                or fixed32_claim != 0
            ):
                raise AccountingInvariantError("DeepLR paid continuous package is invalid")
            if (
                not np.array_equal(base[0].reshape(endpoint0.shape), endpoint0)
                or not np.array_equal(base[-1].reshape(endpoint1.shape), endpoint1)
            ):
                raise AccountingInvariantError("DeepLR spline base changed a frozen endpoint")
            h = 2.0**-11
            lower = h + (1.0 - 2.0 * h) * np.arange(30, dtype=np.float64) / 30.0
            upper = h + (1.0 - 2.0 * h) * (np.arange(30, dtype=np.float64) + 1.0) / 30.0
            if np.any(parameters < lower) or np.any(parameters >= upper):
                raise AccountingInvariantError("DeepLR sample parameters left fixed strata")
            if reference_deeplr_knots is None:
                reference_deeplr_knots = knots.copy()
                reference_deeplr_base = base.copy()
            elif not np.array_equal(knots, reference_deeplr_knots) or not np.array_equal(
                base, reference_deeplr_base
            ):
                raise AccountingInvariantError("DeepLR spline base/knots drifted across batches")
            expected_source = "current_initial" if batch == 1 else "pending_continuous_model"
            if candidate_source != expected_source or str(reserve.get("state_role")) != expected_source:
                raise AccountingInvariantError("DeepLR paid candidate role sequence is invalid")
            if batch == 1:
                if current_theta_before_sha != theta_sha or np.any(theta != 0.0):
                    raise AccountingInvariantError("DeepLR initial candidate is not zero current theta")
            elif theta_sha != previous_deeplr_pending_theta_sha:
                raise AccountingInvariantError("DeepLR pending theta did not become next paid candidate")
            if batch > 1 and current_theta_before_sha != previous_deeplr_current_theta_sha:
                raise AccountingInvariantError("DeepLR pre-query formal current theta chain broke")
            coefficients = base.copy()
            coefficients[1:-1] += theta
            reconstructed = np.asarray(
                BSpline(knots, coefficients, 3, axis=0, extrapolate=False)(parameters),
                dtype="<f8",
            )
            if not np.array_equal(reconstructed, paid_interior.reshape((30, -1))):
                raise AccountingInvariantError(
                    "DeepLR theta/parameters do not reconstruct paid coordinates byte-for-byte"
                )
            if (
                str(reserve.get("paid_candidate_theta_sha256")) != theta_sha
                or str(reserve.get("paid_candidate_source")) != candidate_source
                or str(reserve.get("formal_current_theta_sha256_before_query"))
                != current_theta_before_sha
                or str(reserve.get("sample_parameters_sha256")) != parameters_sha
                or str(reserve.get("paid_coordinates30_reconstructed_sha256"))
                != array_sha256(reconstructed)
                or int(reserve.get("fixed32_paid_force_claim", -1)) != 0
            ):
                raise AccountingInvariantError("DeepLR reserve continuous hashes disagree")
        for slot in range(30):
            request = requests[(batch, slot)]
            if (
                str(request.get("batch_coordinate_file"))
                != str(reserve["batch_coordinate_file"])
                or str(request.get("batch_coordinate_file_sha256"))
                != str(reserve["batch_coordinate_file_sha256"])
                or int(request.get("batch_coordinate_slice_index", -1)) != slot
                or str(request.get("coordinate_sha256")) != array_sha256(paid_interior[slot])
            ):
                raise AccountingInvariantError("paid request slice does not match durable package")
            if int(request.get("paid_nf", -1)) != (batch - 1) * 30 + slot + 1:
                raise AccountingInvariantError("paid request NF slot axis is inconsistent")
            outcome = outcomes[(batch, slot)]
            if int(outcome.get("paid_nf", -1)) != (batch - 1) * 30 + slot + 1:
                raise AccountingInvariantError("paid outcome NF slot axis is inconsistent")
        if any(requests[key].get("role", "").startswith("endpoint") for key in keys):
            raise AccountingInvariantError("endpoint entered paid request ledger")
        failures = [key for key in keys if outcomes[key]["event"] == "force_failure"]
        if failures and commits.get(batch):
            raise AccountingInvariantError("failed batch has optimizer commit")
        if not failures and len(commits.get(batch, [])) != 1:
            raise AccountingInvariantError("successful batch lacks exactly one disposition commit")
        if batch not in force_completes:
            raise AccountingInvariantError("batch lacks force-complete event")
        if not failures:
            commit = commits[batch][0]
            if str(commit.get("paid_path32_sha256")) != paid_hash:
                raise AccountingInvariantError("commit paid-path SHA disagrees with durable paid package")
            if str(commit.get("current_path32_sha256")) != str(commit["path32_sha256"]):
                raise AccountingInvariantError("commit current-path SHA fields disagree")
            paid_meta = commit.get("optimizer_metadata", {})
            if "paid_path_sha256" in paid_meta and str(paid_meta["paid_path_sha256"]) != paid_hash:
                raise AccountingInvariantError("optimizer paid-path link disagrees with reserve")
            if commit.get("method") == "neb_ase_ode":
                disposition = str(commit["disposition"])
                current_path_hash = str(commit["path32_sha256"])
                if str(commit["state_out_hash"]) != current_path_hash:
                    raise AccountingInvariantError("ASE NEB current state/path hashes disagree")
                if disposition in {"current_initial", "current_restart", "accepted"}:
                    if current_path_hash != paid_hash:
                        raise AccountingInvariantError(
                            f"ASE ODE {disposition} current is not the paid path"
                        )
                elif disposition == "rejected":
                    if previous_neb_current_path_hash is None:
                        raise AccountingInvariantError("ASE ODE rejected before an evaluated current")
                    if current_path_hash != previous_neb_current_path_hash:
                        raise AccountingInvariantError("ASE ODE rejection did not roll back to current")
                    if current_path_hash == paid_hash:
                        raise AccountingInvariantError("ASE ODE rejected trial equals rollback current")
                else:
                    raise AccountingInvariantError(
                        f"unexpected ASE ODE paid disposition {disposition!r}"
                    )
                if disposition == "current_restart":
                    ode_restart_count += 1
                previous_neb_current_path_hash = current_path_hash
            elif commit.get("method") == "ivm":
                disposition = str(commit["disposition"])
                current_path_hash = str(commit["path32_sha256"])
                if disposition in {"current_initial", "accepted"}:
                    if current_path_hash != paid_hash:
                        raise AccountingInvariantError(
                            f"IVM {disposition} current is not the complete paid path"
                        )
                elif disposition == "rejected":
                    if previous_ivm_current_path_hash is None:
                        raise AccountingInvariantError("IVM rejected before a paid current existed")
                    if current_path_hash != previous_ivm_current_path_hash:
                        raise AccountingInvariantError(
                            "IVM rejected paid trial did not roll back to the paid current"
                        )
                    if current_path_hash == paid_hash:
                        raise AccountingInvariantError(
                            "IVM rejected trial equals the rollback current"
                        )
                    if str(commit["paid_path32_sha256"]) != paid_hash:
                        raise AccountingInvariantError(
                            "IVM rejected paid-trial SHA differs from durable paid package"
                        )
                    if str(commit["current_path32_sha256"]) != previous_ivm_current_path_hash:
                        raise AccountingInvariantError(
                            "IVM rejected rollback-current SHA differs from prior paid current"
                        )
                    ivm_rejected_count += 1
                else:
                    raise AccountingInvariantError(
                        f"unexpected IVM paid disposition {disposition!r}"
                    )
                if not bool(commit.get("evaluated_current")):
                    raise AccountingInvariantError("IVM commit is not a paid evaluated current")
                previous_ivm_current_path_hash = current_path_hash
            elif commit.get("method") == "deeplr":
                if str(commit.get("paid_candidate_theta_sha256")) != theta_sha:
                    raise AccountingInvariantError("DeepLR commit paid theta differs from reserve")
                if str(commit.get("current_theta_sha256")) != theta_sha:
                    raise AccountingInvariantError("DeepLR current theta is not this batch candidate")
                if (
                    str(commit.get("sample_parameters_sha256")) != parameters_sha
                    or str(commit.get("paid_coordinates30_sha256"))
                    != array_sha256(reconstructed)
                ):
                    raise AccountingInvariantError("DeepLR commit is detached from paid samples")
                pending_theta_sha = str(commit.get("pending_theta_sha256", ""))
                if not re.fullmatch(r"[0-9a-f]{64}", pending_theta_sha):
                    raise AccountingInvariantError("DeepLR pending theta SHA is invalid")
                previous_deeplr_pending_theta_sha = pending_theta_sha
                previous_deeplr_current_theta_sha = theta_sha
                if not bool(commit.get("pending_theta_is_unevaluated")):
                    raise AccountingInvariantError("DeepLR pending theta lacks unevaluated status")
                paid_meta = commit.get("optimizer_metadata", {})
                expected_current_state_sha = _deeplr_current_continuous_state_sha256(
                    batch_id=batch,
                    arm=str(paid_meta.get("deeplr_arm", "")),
                    alpha=float(paid_meta.get("alpha_A2_per_eV", np.nan)),
                    theta=theta,
                    knots=knots,
                    base=base,
                )
                if str(commit.get("current_continuous_state_sha256")) != expected_current_state_sha:
                    raise AccountingInvariantError("DeepLR formal current-state SHA is false")
                if (
                    not bool(commit.get("fixed32_is_geometric_attachment_only"))
                    or int(commit.get("fixed32_paid_force_claim", -1)) != 0
                    or str(commit.get("fixed32_geometric_attachment_sha256"))
                    != str(commit.get("current_path32_sha256"))
                    or str(commit.get("current_state_kind"))
                    != "continuous_spline_coefficients"
                    or str(commit.get("current_path32_role"))
                    != "geometric_attachment_only"
                    or bool(commit.get("fixed32_evaluated_current"))
                    or str(commit.get("paid_geometry_kind"))
                    != "continuous_model_at_30_sample_parameters"
                    or str(commit.get("evaluated_current_semantics"))
                    != "continuous_theta_paid_at_30_sample_parameters"
                ):
                    raise AccountingInvariantError("DeepLR fixed32 attachment semantics are false")
                if str(commit.get("path32_sha256")) == paid_hash:
                    # Initial PL geometry can coincidentally coincide, but the
                    # claim remains explicitly non-paid; equality alone is not
                    # used as evidence of payment.
                    if int(commit.get("fixed32_paid_force_claim", -1)) != 0:
                        raise AccountingInvariantError("DeepLR fixed32 falsely claims paid path")
            nf = 30 * batch
            if nf in milestones:
                complete = force_completes[batch]
                receipt = complete.get("milestone_full_force_receipt")
                if not receipt or not complete.get("milestone_full_force_receipt_sha256"):
                    raise AccountingInvariantError("matched batch lacks paid-trial EF receipt")
                receipt_path = run_dir / str(receipt)
                if (
                    not receipt_path.is_file()
                    or file_sha256(receipt_path)
                    != str(complete["milestone_full_force_receipt_sha256"])
                ):
                    raise AccountingInvariantError("matched paid-trial EF receipt hash failed")
                with np.load(receipt_path, allow_pickle=False) as archive:
                    if int(np.asarray(archive["batch_id"]).reshape(())) != batch:
                        raise AccountingInvariantError("matched receipt batch link failed")
                    if int(np.asarray(archive["nf_after"]).reshape(())) != nf:
                        raise AccountingInvariantError("matched receipt NF link failed")
                    receipt_coordinates = canonical_array(archive["coordinates30_A"])
                    receipt_energies = canonical_array(archive["energies_eV"])
                    receipt_forces = canonical_array(archive["forces_eV_per_A"])
                if receipt_energies.shape != (30,) or receipt_forces.shape != paid_interior.shape:
                    raise AccountingInvariantError("matched EF receipt has invalid array shapes")
                for slot in range(30):
                    outcome = outcomes[(batch, slot)]
                    if outcome.get("event") != "force_response":
                        raise AccountingInvariantError("matched receipt slot is not a successful response")
                    if float(outcome.get("energy_eV")) != float(receipt_energies[slot]):
                        raise AccountingInvariantError("matched receipt energy differs from WAL response")
                    if str(outcome.get("force_sha256")) != array_sha256(receipt_forces[slot]):
                        raise AccountingInvariantError("matched receipt Force differs from WAL response hash")
                paid_coordinate_path = run_dir / str(reserve["batch_coordinate_file"])
                if (
                    not paid_coordinate_path.is_file()
                    or file_sha256(paid_coordinate_path)
                    != str(reserve["batch_coordinate_file_sha256"])
                ):
                    raise AccountingInvariantError("paid coordinate package hash failed")
                with np.load(paid_coordinate_path, allow_pickle=False) as archive:
                    paid_coordinates = canonical_array(archive["coordinates30_A"])
                if not np.array_equal(receipt_coordinates, paid_coordinates):
                    raise AccountingInvariantError("matched EF receipt is not the paid path")
                snap = snapshots.get(nf)
                if snap is None or int(snap["batch_id"]) != batch:
                    raise AccountingInvariantError("matched batch lacks current-state snapshot link")
                if str(snap["state_sha256"]) != str(commit["state_out_hash"]):
                    raise AccountingInvariantError("snapshot does not identify committed current state")
                if str(snap["path32_sha256"]) != str(commit["path32_sha256"]):
                    raise AccountingInvariantError("snapshot path differs from committed current path")
                if str(commit.get("method")) == "deeplr":
                    if (
                        str(snap.get("state_role"))
                        != "evaluated_current_continuous_theta"
                        or str(snap.get("path32_role"))
                        != "geometric_attachment_only"
                        or bool(snap.get("fixed32_evaluated_current"))
                        or
                        str(snap.get("current_theta_sha256")) != theta_sha
                        or str(snap.get("paid_candidate_theta_sha256")) != theta_sha
                        or str(snap.get("current_continuous_state_sha256"))
                        != expected_current_state_sha
                        or int(snap.get("fixed32_paid_force_claim", -1)) != 0
                    ):
                        raise AccountingInvariantError(
                            "DeepLR matched snapshot is not bound to paid/current theta"
                        )
                elif str(snap["state_role"]) != "evaluated_current":
                    raise AccountingInvariantError("matched snapshot is not evaluated current")
    if require_terminal:
        if batches[-1] != terminal_batches \
                or any(r["event"] == "force_failure" for r in outcomes.values()):
            raise AccountingInvariantError(
                f"terminal audit requires {terminal_batches} failure-free batches"
            )
        if len(run_seals) != 1 or records[-1].get("event") != "run_seal":
            raise AccountingInvariantError("terminal WAL must end with exactly one run seal")
        seal = run_seals[0]
        if int(seal.get("nf", -1)) != terminal_nf \
                or int(seal.get("batches", -1)) != terminal_batches \
                or int(seal.get("endpoint_physical_calls", -1)) != 0 \
                or int(seal.get("interior_physical_calls", -1)) != terminal_nf:
            raise AccountingInvariantError("terminal run seal budget/accounting fields drifted")
        if set(snapshots) != set(milestones):
            raise AccountingInvariantError("terminal WAL lacks the exact matched snapshot set")
    return {
        "pass": True,
        "batches": len(batches),
        "paid_nf": 30 * len(batches),
        "endpoint_physical_calls": 0,
        "requests": len(requests),
        "outcomes": len(outcomes),
        "matched_snapshot_paid_receipt_links": len(snapshots),
        "ase_ode_current_restart_batches": ode_restart_count,
        "ivm_rejected_paid_trial_rollbacks": ivm_rejected_count,
    }


def audit_deeplr_snapshot_bindings(
    run_dir: Path | str, *, require_terminal: bool = False
) -> dict[str, Any]:
    """Verify every matched DeepLR NPZ against its exact paid batch package.

    This is intentionally independent of dense diagnostics.  It proves that
    the continuous theta in each matched snapshot is the candidate evaluated
    at that batch's 30 persisted parameters, while its fixed-32 array remains
    an explicitly non-paid geometric attachment.
    """

    run = Path(run_dir).resolve()
    wal_path = run / "optimization_force_wal.jsonl"
    audit_wal(wal_path, require_terminal=require_terminal)
    records = [json.loads(line) for line in wal_path.read_text(encoding="utf-8").splitlines()]
    reserves = {
        int(record["batch_id"]): record
        for record in records
        if record.get("event") == "batch_reserve"
    }
    commits = {
        int(record["batch_id"]): record
        for record in records
        if record.get("event") == "batch_commit"
    }
    snapshots = {
        int(record["nf"]): record
        for record in records
        if record.get("event") == "snapshot"
    }
    if any(str(record.get("method")) != "deeplr" for record in reserves.values()):
        raise AccountingInvariantError("DeepLR snapshot binder received a non-DeepLR WAL")
    expected_nfs = set(MATCHED_MILESTONES if require_terminal else snapshots)
    if set(snapshots) != expected_nfs:
        raise AccountingInvariantError("DeepLR snapshot WAL set is not exact")
    file_hashes: dict[str, str] = {}
    for nf in sorted(expected_nfs):
        batch = nf // 30
        reserve = reserves[batch]
        commit = commits[batch]
        wal_snapshot = snapshots[nf]
        package_path = run / str(reserve["batch_coordinate_file"])
        snapshot_path = run / "snapshots" / f"nf{nf:05d}.npz"
        if not snapshot_path.is_file():
            raise AccountingInvariantError("DeepLR matched snapshot file is absent")
        with np.load(package_path, allow_pickle=False) as package:
            paid_coordinates = canonical_array(package["coordinates30_A"])
            paid_theta = canonical_array(package["paid_candidate_theta"])
            paid_parameters = canonical_array(package["sample_parameters"])
            package_knots = canonical_array(package["spline_knots"])
            package_base = canonical_array(package["spline_base_coefficients"])
        with np.load(snapshot_path, allow_pickle=False) as snapshot:
            required = {
                "nf", "batch_id", "method", "representation", "state_sha256",
                "knots", "base_coefficients", "theta", "current_theta_sha256",
                "current_continuous_state_sha256", "deeplr_arm", "alpha_A2_per_eV",
                "pending_theta", "pending_theta_sha256",
                "pending_theta_is_unevaluated_proposal",
                "paid_sample_parameters", "paid_sample_parameters_sha256",
                "paid_candidate_theta_sha256", "paid_coordinates30_sha256",
                "last_paid_batch_id", "fixed32_evaluation",
                "fixed32_geometric_attachment_sha256",
                "fixed32_evaluation_is_geometric_attachment_only",
                "fixed32_paid_force_claim", "paid_force_coordinates_are_not_fixed32",
            }
            if not required <= set(snapshot.files):
                raise AccountingInvariantError("DeepLR matched snapshot lacks binding fields")
            snap_nf = int(np.asarray(snapshot["nf"]).reshape(()))
            snap_batch = int(np.asarray(snapshot["batch_id"]).reshape(()))
            snap_method = str(np.asarray(snapshot["method"]).reshape(()))
            snap_representation = str(np.asarray(snapshot["representation"]).reshape(()))
            snap_state_sha = str(np.asarray(snapshot["state_sha256"]).reshape(()))
            theta = canonical_array(snapshot["theta"])
            parameters = canonical_array(snapshot["paid_sample_parameters"])
            knots = canonical_array(snapshot["knots"])
            base = canonical_array(snapshot["base_coefficients"])
            pending = canonical_array(snapshot["pending_theta"])
            fixed32 = canonical_array(snapshot["fixed32_evaluation"])
            node_parameters = canonical_array(snapshot["node_parameters"])
            arm = str(np.asarray(snapshot["deeplr_arm"]).reshape(()))
            alpha = float(np.asarray(snapshot["alpha_A2_per_eV"]).reshape(()))
            snapshot_scalars = {
                name: str(np.asarray(snapshot[name]).reshape(()))
                for name in (
                    "current_theta_sha256", "paid_candidate_theta_sha256",
                    "paid_sample_parameters_sha256", "paid_coordinates30_sha256",
                    "current_continuous_state_sha256", "pending_theta_sha256",
                    "fixed32_geometric_attachment_sha256",
                )
            }
            snapshot_integers = {
                name: int(np.asarray(snapshot[name]).reshape(()))
                for name in (
                    "last_paid_batch_id", "pending_theta_is_unevaluated_proposal",
                    "fixed32_evaluation_is_geometric_attachment_only",
                    "fixed32_paid_force_claim", "paid_force_coordinates_are_not_fixed32",
                )
            }
        if (
            snap_nf != nf or snap_batch != batch or snap_method != "deeplr"
            or snap_representation != "continuous_spline_coefficients"
            or snapshot_integers["last_paid_batch_id"] != batch
            or snap_state_sha != str(wal_snapshot["state_sha256"])
            or not np.array_equal(theta, paid_theta)
            or not np.array_equal(parameters, paid_parameters)
            or not np.array_equal(knots, package_knots)
            or not np.array_equal(base, package_base)
        ):
            raise AccountingInvariantError("DeepLR matched snapshot state/package link failed")
        coefficients = base.copy()
        coefficients[1:-1] += theta
        reconstructed = canonical_array(
            BSpline(knots, coefficients, 3, axis=0, extrapolate=False)(parameters)
        )
        if not np.array_equal(reconstructed, paid_coordinates.reshape((30, -1))):
            raise AccountingInvariantError("DeepLR matched snapshot cannot reconstruct paid coordinates")
        current_theta_sha = array_sha256(theta)
        parameters_sha = array_sha256(parameters)
        paid_coordinates_sha = array_sha256(reconstructed)
        current_state_sha = _deeplr_current_continuous_state_sha256(
            batch_id=batch, arm=arm, alpha=alpha, theta=theta, knots=knots, base=base
        )
        fixed32_reconstructed = canonical_array(
            BSpline(knots, coefficients, 3, axis=0, extrapolate=False)(
                node_parameters
            )
        ).reshape(fixed32.shape)
        if (
            snapshot_scalars["current_theta_sha256"] != current_theta_sha
            or snapshot_scalars["paid_candidate_theta_sha256"] != current_theta_sha
            or snapshot_scalars["paid_sample_parameters_sha256"] != parameters_sha
            or snapshot_scalars["paid_coordinates30_sha256"] != paid_coordinates_sha
            or snapshot_scalars["current_continuous_state_sha256"] != current_state_sha
            or str(commit.get("current_continuous_state_sha256")) != current_state_sha
            or str(commit.get("sample_parameters_sha256")) != parameters_sha
            or str(commit.get("paid_coordinates30_sha256")) != paid_coordinates_sha
            or snapshot_scalars["pending_theta_sha256"] != array_sha256(pending)
            or snapshot_integers["pending_theta_is_unevaluated_proposal"] != 1
            or not np.array_equal(fixed32, fixed32_reconstructed)
            or snapshot_scalars["fixed32_geometric_attachment_sha256"] != array_sha256(fixed32)
            or str(commit.get("fixed32_geometric_attachment_sha256")) != array_sha256(fixed32)
            or snapshot_integers["fixed32_evaluation_is_geometric_attachment_only"] != 1
            or snapshot_integers["fixed32_paid_force_claim"] != 0
            or snapshot_integers["paid_force_coordinates_are_not_fixed32"] != 1
        ):
            raise AccountingInvariantError("DeepLR matched snapshot semantic/hash binding failed")
        file_hashes[str(nf)] = file_sha256(snapshot_path)
    return {
        "pass": True,
        "matched_snapshot_count": len(file_hashes),
        "matched_snapshot_sha256": file_hashes,
    }


def audit_deeplr_terminal_completion(run_dir: Path | str) -> dict[str, Any]:
    """Bind the terminal completion seal to the exact nine snapshot files."""

    run = Path(run_dir).resolve()
    binding = audit_deeplr_snapshot_bindings(run, require_terminal=True)
    completion_path = run / "OPTIMIZATION_COMPLETE.json"
    if not completion_path.is_file():
        raise AccountingInvariantError("DeepLR terminal completion is absent")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    snapshot_map = completion.get("matched_snapshot_sha256")
    if snapshot_map != binding["matched_snapshot_sha256"]:
        raise AccountingInvariantError("DeepLR completion snapshot SHA map is not exact")
    expected_keys = {str(nf) for nf in MATCHED_MILESTONES}
    if set(snapshot_map) != expected_keys:
        raise AccountingInvariantError("DeepLR completion snapshot SHA keys are not exact")
    if completion.get("terminal_snapshot_sha256") != snapshot_map["9990"]:
        raise AccountingInvariantError("DeepLR terminal snapshot SHA differs from NF9990 map")
    wal_path = run / "optimization_force_wal.jsonl"
    if completion.get("optimization_wal_sha256") != file_sha256(wal_path):
        raise AccountingInvariantError("DeepLR completion WAL SHA is false")
    if completion.get("method") != "deeplr" or completion.get("paid_nf") != 9990:
        raise AccountingInvariantError("DeepLR completion identity/budget is false")
    return {**binding, "completion_sha256": file_sha256(completion_path)}
