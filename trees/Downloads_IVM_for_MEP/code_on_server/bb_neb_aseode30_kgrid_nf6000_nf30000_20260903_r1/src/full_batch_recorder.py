"""Lossless, durable archive for every paid ASE ODE--NEB force call.

The recorder is output-only: it receives arrays already returned by the one
paid joint energy/full-force batch.  It never owns or invokes a calculator.
Each 25-batch block starts with a float64 checkpoint; subsequent shards store
uint64 XOR deltas.  Reconstruction is bit exact.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np

from .accounting import array_sha256
from .integrity import (
    canonical_json_bytes,
    load_json_object,
    parse_json_object,
    sha256_file,
    write_json_new,
)


SHARD_SCHEMA = "bb-neb-kgrid-paid-batch-xor-delta-v3"
WAL_SCHEMA = "bb-neb-kgrid-paid-batch-wal-v3"
COMPLETE_SCHEMA = "bb-neb-kgrid-paid-batch-complete-v1"
BLOCK_SIZE = 25
ATOM_COORDINATES = 9396
ATOM_COUNT = ATOM_COORDINATES // 3
IMAGE_COUNT = 32
INTERIOR_COUNT = 30
ALLOWED_K = (0.1, 1.0, 10.0, 100.0)
ALLOWED_BATCH_LIMITS = (200, 1000)
WAL_NAME = "full_state.wal.jsonl"
COMPLETE_NAME = "FULL_STATE_COMPLETE.json"


class FullStateArchiveError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FullStateArchiveError(message)


def _f64(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    _require(array.shape == shape, f"{label} shape differs: {array.shape} != {shape}")
    _require(np.isfinite(array).all(), f"{label} contains a non-finite value")
    return array


def _path32(value: Any, label: str) -> np.ndarray:
    """Canonicalize a production path in the ledger's ``(32,N,3)`` domain."""

    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if array.shape == (IMAGE_COUNT, ATOM_COORDINATES):
        array = array.reshape((IMAGE_COUNT, ATOM_COUNT, 3))
    _require(
        array.shape == (IMAGE_COUNT, ATOM_COUNT, 3),
        f"{label} shape differs: {array.shape}",
    )
    _require(np.isfinite(array).all(), f"{label} contains a non-finite value")
    return np.ascontiguousarray(array, dtype="<f8")


def _force30(value: Any) -> np.ndarray:
    """Canonicalize paid full forces in the same shape used by the ledger."""

    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    if array.shape == (INTERIOR_COUNT, ATOM_COORDINATES):
        array = array.reshape((INTERIOR_COUNT, ATOM_COUNT, 3))
    _require(
        array.shape == (INTERIOR_COUNT, ATOM_COUNT, 3),
        f"paid interior full forces shape differs: {array.shape}",
    )
    _require(np.isfinite(array).all(), "paid interior full forces contain a non-finite value")
    return np.ascontiguousarray(array, dtype="<f8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file(path: Path, label: str, *, readonly: bool) -> None:
    status = path.lstat()
    _require(stat.S_ISREG(status.st_mode) and not path.is_symlink(),
             f"{label} is not a regular nonsymlink file")
    if readonly:
        _require(stat.S_IMODE(status.st_mode) & 0o222 == 0, f"{label} is not read-only")


def _load_endpoint_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        _require(
            set(archive.files)
            == {
                "coordinates_A",
                "energies_eV",
                "forces_eV_per_A",
                "source_dense200_sha256",
                "source_diagnostic_audit_sha256",
            },
            "endpoint cache key set changed",
        )
        coordinates = _f64(archive["coordinates_A"], (2, ATOM_COORDINATES),
                           "endpoint coordinates")
        energies = _f64(archive["energies_eV"], (2,), "endpoint energies")
        forces = _f64(archive["forces_eV_per_A"], (2, ATOM_COORDINATES),
                      "endpoint forces")
        for key in ("source_dense200_sha256", "source_diagnostic_audit_sha256"):
            digest = str(np.asarray(archive[key]).reshape(()))
            _require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                     f"endpoint provenance digest {key} is invalid")
    return coordinates, energies, forces


class FullBatchRecorder:
    """Append-only archive whose batch publication precedes ledger commit."""

    def __init__(
        self,
        run_dir: Path | str,
        endpoint_cache: Path | str,
        *,
        spring_k: float,
        terminal_batches: int,
        run_label: str,
        authorization_sha256: str,
        source_closure_sha256: str,
    ) -> None:
        self.root = Path(run_dir).resolve() / "paid_batch_full_state"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.wal = self.root / WAL_NAME
        self.complete_path = self.root / COMPLETE_NAME
        self.spring_k = float(spring_k)
        self.limit = int(terminal_batches)
        self.run_label = str(run_label)
        self.authorization_sha256 = str(authorization_sha256)
        self.source_closure_sha256 = str(source_closure_sha256)
        _require(self.spring_k in ALLOWED_K, "spring k is outside the frozen grid")
        _require(self.limit in ALLOWED_BATCH_LIMITS, "batch limit is not 200 or 1000")
        _require(bool(self.run_label), "run label is empty")
        for label, digest in (
            ("authorization", self.authorization_sha256),
            ("source closure", self.source_closure_sha256),
        ):
            _require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                     f"{label} SHA-256 is invalid")
        endpoint_path = Path(endpoint_cache).resolve(strict=True)
        _regular_file(endpoint_path, "endpoint cache", readonly=False)
        self.endpoint_cache_path = endpoint_path
        self.endpoint_cache_sha256 = sha256_file(endpoint_path)
        self.endpoint_coordinates, self.endpoint_energies, self.endpoint_forces = (
            _load_endpoint_cache(endpoint_path)
        )
        self.previous_record_sha256: str | None = None
        self.last_batch = 0
        self.previous_arrays: tuple[np.ndarray, ...] | None = None
        self.current_source_batch_id = 0
        self.current_state_sha256: str | None = None
        self.sealed = False
        descriptor = os.open(self.wal, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(descriptor)
        self._append(
            {
                "schema": WAL_SCHEMA,
                "event": "header",
                "run_label": self.run_label,
                "spring_k_eV_per_A2": self.spring_k,
                "terminal_batches": self.limit,
                "terminal_optimization_N_F": 30 * self.limit,
                "block_size": BLOCK_SIZE,
                "endpoint_cache_path": str(endpoint_path),
                "endpoint_cache_sha256": self.endpoint_cache_sha256,
                "authorization_sha256": self.authorization_sha256,
                "source_closure_sha256": self.source_closure_sha256,
                "endpoint_optimization_N_F": 0,
                "additional_OpenMM_calls": 0,
            }
        )

    def _append(self, unsigned: Mapping[str, Any]) -> dict[str, Any]:
        _require(not self.sealed, "sealed full-state archive cannot append")
        record = {**dict(unsigned), "previous_record_sha256": self.previous_record_sha256}
        record["record_sha256"] = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
        with self.wal.open("ab", buffering=0) as stream:
            stream.write(canonical_json_bytes(record) + b"\n")
            os.fsync(stream.fileno())
        self.previous_record_sha256 = str(record["record_sha256"])
        return record

    def disk_preflight(self) -> dict[str, int]:
        # Never assume XOR/compression savings: 10 MB/batch plus 1 GiB.
        required = self.limit * 10_000_000 + (1 << 30)
        free = shutil.disk_usage(self.root).free
        _require(free >= required, f"insufficient disk for full history: {free} < {required}")
        return {
            "free_bytes": int(free),
            "required_bytes": int(required),
            "terminal_batches": self.limit,
        }

    def record(
        self,
        *,
        batch_id: int,
        nf: int,
        role: str,
        disposition: str,
        paid_path32_A: Any,
        energies30_eV: Any,
        forces30_eV_per_A: Any,
        current_path32_A: Any,
        current_state_sha256: str,
    ) -> dict[str, Any]:
        _require(not self.sealed, "full-state archive is sealed")
        batch = int(batch_id)
        _require(batch == self.last_batch + 1 and batch <= self.limit,
                 "batch is non-contiguous or beyond the exact budget")
        _require(int(nf) == 30 * batch, "optimization N_F does not equal 30*batch")
        expected_role = "current_initial" if batch == 1 else "trial"
        _require(role == expected_role, "paid role sequence drifted")
        allowed = ("current_initial",) if batch == 1 else ("accepted", "rejected")
        _require(disposition in allowed, "paid disposition sequence drifted")

        paid_path = _path32(paid_path32_A, "paid path32")
        current_path = _path32(current_path32_A, "current path32")
        energies30 = _f64(energies30_eV, (INTERIOR_COUNT,), "paid interior energies")
        forces30 = _force30(forces30_eV_per_A)
        endpoint_coordinates = self.endpoint_coordinates.reshape((2, ATOM_COUNT, 3))
        _require(
            np.array_equal(paid_path[[0, -1]], endpoint_coordinates)
            and np.array_equal(current_path[[0, -1]], endpoint_coordinates),
            "fixed endpoint coordinates changed",
        )
        paid_hash = array_sha256(paid_path)
        current_hash = array_sha256(current_path)
        _require(str(current_state_sha256) == current_hash,
                 "declared current-state SHA differs from current path")
        if disposition in {"current_initial", "accepted"}:
            _require(np.array_equal(current_path, paid_path),
                     "accepted/current path is not the paid path")
            current_source_batch = batch
        else:
            _require(self.current_state_sha256 is not None and self.current_source_batch_id > 0,
                     "rejected trial has no prior evaluated current")
            _require(current_hash == self.current_state_sha256,
                     "rejected trial did not map to the prior accepted current")
            _require(not np.array_equal(current_path, paid_path),
                     "rejected paid trial duplicates the current path")
            current_source_batch = self.current_source_batch_id

        energy32 = np.empty(IMAGE_COUNT, dtype="<f8")
        full_force32 = np.empty((IMAGE_COUNT, ATOM_COUNT, 3), dtype="<f8")
        energy32[[0, -1]] = self.endpoint_energies
        full_force32[[0, -1]] = self.endpoint_forces.reshape((2, ATOM_COUNT, 3))
        energy32[1:-1] = energies30
        full_force32[1:-1] = forces30
        arrays = (paid_path, energy32, full_force32, current_path)
        checkpoint = (batch - 1) % BLOCK_SIZE == 0
        if checkpoint:
            encoded = arrays
            encoding = "full_float64"
            previous_batch_id = 0
        else:
            _require(self.previous_arrays is not None, "XOR delta lacks predecessor")
            encoded = tuple(
                np.bitwise_xor(array.view(np.uint64), previous.view(np.uint64))
                for array, previous in zip(arrays, self.previous_arrays)
            )
            encoding = "xor_uint64"
            previous_batch_id = batch - 1

        output = self.root / f"batch_{batch:04d}_nf{30 * batch:05d}.npz"
        _require(not output.exists(), "full-state shard would overwrite an existing file")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=output.name + ".", suffix=".tmp", dir=self.root
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                np.savez_compressed(
                    stream,
                    schema=np.asarray(SHARD_SCHEMA),
                    batch_id=np.asarray(batch, dtype=np.int64),
                    optimization_N_F=np.asarray(30 * batch, dtype=np.int64),
                    spring_k_eV_per_A2=np.asarray(self.spring_k, dtype=np.float64),
                    role=np.asarray(role),
                    disposition=np.asarray(disposition),
                    encoding=np.asarray(encoding),
                    block_size=np.asarray(BLOCK_SIZE, dtype=np.int64),
                    previous_batch_id=np.asarray(previous_batch_id, dtype=np.int64),
                    current_source_batch_id=np.asarray(current_source_batch, dtype=np.int64),
                    paid_path32=encoded[0],
                    energy32=encoded[1],
                    full_force32=encoded[2],
                    current_path32=encoded[3],
                    paid_path_sha256=np.asarray(paid_hash),
                    energy32_sha256=np.asarray(array_sha256(energy32)),
                    full_force32_sha256=np.asarray(array_sha256(full_force32)),
                    current_path_sha256=np.asarray(current_hash),
                    declared_current_state_sha256=np.asarray(current_state_sha256),
                    endpoint_cache_sha256=np.asarray(self.endpoint_cache_sha256),
                    endpoint_values_inherited=np.asarray(2, dtype=np.int64),
                    endpoint_optimization_N_F=np.asarray(0, dtype=np.int64),
                    additional_OpenMM_calls=np.asarray(0, dtype=np.int64),
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o444)
            os.link(temporary, output)
            _fsync_directory(self.root)
        finally:
            temporary.unlink(missing_ok=True)

        binding = {
            "schema": SHARD_SCHEMA,
            "batch_id": batch,
            "optimization_N_F": 30 * batch,
            "role": role,
            "disposition": disposition,
            "encoding": encoding,
            "previous_batch_id": previous_batch_id,
            "current_source_batch_id": current_source_batch,
            "relative_path": output.relative_to(Path(self.root).parent).as_posix(),
            "file_sha256": sha256_file(output),
            "paid_path_sha256": paid_hash,
            "energy32_sha256": array_sha256(energy32),
            "full_force32_sha256": array_sha256(full_force32),
            "current_path_sha256": current_hash,
            "endpoint_cache_sha256": self.endpoint_cache_sha256,
            "endpoint_optimization_N_F": 0,
            "additional_OpenMM_calls": 0,
        }
        # WAL event identity is distinct from the shard binding returned to
        # the optimizer ledger.  Put WAL fields last so the binding's shard
        # schema cannot overwrite them.
        self._append({**binding, "schema": WAL_SCHEMA, "event": "batch_published"})
        self.last_batch = batch
        self.previous_arrays = tuple(np.array(array, copy=True) for array in arrays)
        if disposition in {"current_initial", "accepted"}:
            self.current_source_batch_id = batch
            self.current_state_sha256 = current_hash
        return binding

    def seal(self) -> dict[str, Any]:
        _require(not self.sealed, "full-state archive already sealed")
        _require(self.last_batch == self.limit,
                 "full-state archive cannot seal before the exact batch budget")
        seal_record = self._append(
            {
                "schema": WAL_SCHEMA,
                "event": "terminal_seal",
                "run_label": self.run_label,
                "batch_count": self.last_batch,
                "terminal_optimization_N_F": 30 * self.last_batch,
                "current_source_batch_id": self.current_source_batch_id,
                "current_state_sha256": self.current_state_sha256,
                "endpoint_optimization_N_F": 0,
                "additional_OpenMM_calls": 0,
            }
        )
        self.sealed = True
        document = {
            "schema": COMPLETE_SCHEMA,
            "status": "SEALED",
            "run_label": self.run_label,
            "spring_k_eV_per_A2": self.spring_k,
            "paid_batches": self.last_batch,
            "paid_optimization_N_F": 30 * self.last_batch,
            "shard_count": self.last_batch,
            "block_size": BLOCK_SIZE,
            "terminal_current_source_batch_id": self.current_source_batch_id,
            "terminal_current_state_sha256": self.current_state_sha256,
            "wal_sha256": sha256_file(self.wal),
            "wal_terminal_record_sha256": seal_record["record_sha256"],
            "endpoint_cache_path": str(self.endpoint_cache_path),
            "endpoint_cache_sha256": self.endpoint_cache_sha256,
            "authorization_sha256": self.authorization_sha256,
            "source_closure_sha256": self.source_closure_sha256,
            "endpoint_optimization_N_F": 0,
            "additional_OpenMM_calls": 0,
            "encoding": "float64_checkpoint_plus_uint64_xor_delta",
            "bit_exact_reconstruction_required": True,
        }
        write_json_new(self.complete_path, document, readonly=True)
        return document


def _strict_wal_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(path.read_bytes().splitlines(), start=1):
        _require(bool(line), f"empty full-state WAL line {line_number}")
        record = parse_json_object(line, source=f"{path}:line {line_number}")
        _require(record.get("previous_record_sha256") == previous,
                 "full-state WAL previous-record chain is broken")
        recorded = record.get("record_sha256")
        unsigned = dict(record)
        unsigned.pop("record_sha256", None)
        computed = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
        _require(recorded == computed, "full-state WAL record SHA recomputation failed")
        previous = str(recorded)
        records.append(record)
    return records


def _decode_shard(
    path: Path, previous: tuple[np.ndarray, ...] | None, *, expected_checkpoint: bool
) -> tuple[tuple[np.ndarray, ...], dict[str, Any]]:
    _regular_file(path, "full-state shard", readonly=True)
    with np.load(path, allow_pickle=False) as archive:
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
        _require(set(archive.files) == expected_keys, "full-state shard key set changed")
        metadata = {
            key: np.asarray(archive[key]).reshape(()).item()
            for key in expected_keys - array_keys
        }
        encoded = tuple(
            np.asarray(archive[key])
            for key in ("paid_path32", "energy32", "full_force32", "current_path32")
        )
    _require(str(metadata["schema"]) == SHARD_SCHEMA, "full-state shard schema changed")
    if expected_checkpoint:
        _require(str(metadata["encoding"]) == "full_float64" and previous is None,
                 "XOR block does not start with a full checkpoint")
        arrays = tuple(np.ascontiguousarray(value, dtype="<f8") for value in encoded)
    else:
        _require(str(metadata["encoding"]) == "xor_uint64" and previous is not None,
                 "noninitial block shard is not an XOR delta")
        arrays = tuple(
            np.ascontiguousarray(
                np.bitwise_xor(np.asarray(value, dtype=np.uint64), prior.view(np.uint64))
                .view("<f8")
            )
            for value, prior in zip(encoded, previous)
        )
    for array, key in zip(
        arrays,
        ("paid_path_sha256", "energy32_sha256", "full_force32_sha256", "current_path_sha256"),
    ):
        _require(array_sha256(array) == str(metadata[key]),
                 f"bit-exact reconstruction hash failed for {key}")
    return arrays, metadata


def reconstruct_batch(root: Path | str, batch_id: int) -> tuple[np.ndarray, ...]:
    archive_root = Path(root).resolve(strict=True)
    batch = int(batch_id)
    _require(batch >= 1, "batch id must be positive")
    start = ((batch - 1) // BLOCK_SIZE) * BLOCK_SIZE + 1
    state: tuple[np.ndarray, ...] | None = None
    for index in range(start, batch + 1):
        matches = list(archive_root.glob(f"batch_{index:04d}_nf*.npz"))
        _require(len(matches) == 1, f"missing or duplicate full-state shard {index}")
        state, _ = _decode_shard(matches[0], state, expected_checkpoint=index == start)
    assert state is not None
    return state


def audit_full_state_archive(
    root: Path | str,
    *,
    spring_k: float,
    terminal_batches: int,
    run_label: str,
    endpoint_cache: Path | str,
    authorization_sha256: str,
    source_closure_sha256: str,
    optimizer_commits: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify every shard once, its WAL, mapping, and ledger-commit binding."""

    archive_root = Path(root).resolve(strict=True)
    _require(archive_root.is_dir() and not archive_root.is_symlink(),
             "full-state archive root is absent or unsafe")
    limit = int(terminal_batches)
    _require(float(spring_k) in ALLOWED_K and limit in ALLOWED_BATCH_LIMITS,
             "terminal k/batch contract is outside the frozen grid")
    endpoint_path = Path(endpoint_cache).resolve(strict=True)
    endpoint_coordinates, endpoint_energies, endpoint_forces = _load_endpoint_cache(endpoint_path)
    endpoint_coordinates = endpoint_coordinates.reshape((2, ATOM_COUNT, 3))
    endpoint_forces = endpoint_forces.reshape((2, ATOM_COUNT, 3))
    endpoint_sha = sha256_file(endpoint_path)
    wal = archive_root / WAL_NAME
    complete_path = archive_root / COMPLETE_NAME
    _regular_file(complete_path, "full-state completion", readonly=True)
    records = _strict_wal_records(wal)
    _require(len(records) == limit + 2, "full-state WAL record count changed")
    header, *body = records
    terminal = body.pop()
    expected_header = {
        "schema": WAL_SCHEMA,
        "event": "header",
        "run_label": str(run_label),
        "spring_k_eV_per_A2": float(spring_k),
        "terminal_batches": limit,
        "terminal_optimization_N_F": 30 * limit,
        "block_size": BLOCK_SIZE,
        "endpoint_cache_path": str(endpoint_path),
        "endpoint_cache_sha256": endpoint_sha,
        "authorization_sha256": str(authorization_sha256),
        "source_closure_sha256": str(source_closure_sha256),
        "endpoint_optimization_N_F": 0,
        "additional_OpenMM_calls": 0,
    }
    for key, value in expected_header.items():
        _require(header.get(key) == value, f"full-state WAL header field {key} changed")
    _require(terminal.get("schema") == WAL_SCHEMA and terminal.get("event") == "terminal_seal",
             "full-state WAL does not end with its unique terminal seal")
    _require(terminal.get("batch_count") == limit
             and terminal.get("terminal_optimization_N_F") == 30 * limit
             and terminal.get("run_label") == run_label
             and terminal.get("endpoint_optimization_N_F") == 0
             and terminal.get("additional_OpenMM_calls") == 0,
             "full-state WAL terminal seal fields changed")

    expected_names = {WAL_NAME, COMPLETE_NAME} | {
        f"batch_{batch:04d}_nf{30 * batch:05d}.npz" for batch in range(1, limit + 1)
    }
    _require({path.name for path in archive_root.iterdir()} == expected_names,
             "full-state archive contains a missing or extra file")
    commits = list(optimizer_commits or ())
    if optimizer_commits is not None:
        _require(len(commits) == limit, "optimizer commit count differs from full-state shards")

    previous_arrays: tuple[np.ndarray, ...] | None = None
    current_arrays: tuple[np.ndarray, ...] | None = None
    current_source_batch = 0
    accepted = 0
    rejected = 0
    for batch, record in enumerate(body, start=1):
        _require(record.get("schema") == WAL_SCHEMA
                 and record.get("event") == "batch_published"
                 and record.get("batch_id") == batch
                 and record.get("optimization_N_F") == 30 * batch,
                 f"full-state WAL batch topology failed at {batch}")
        shard = archive_root / f"batch_{batch:04d}_nf{30 * batch:05d}.npz"
        checkpoint = (batch - 1) % BLOCK_SIZE == 0
        if checkpoint:
            previous_arrays = None
        arrays, metadata = _decode_shard(
            shard, previous_arrays, expected_checkpoint=checkpoint
        )
        previous_arrays = arrays
        paid_path, energy32, full_force32, current_path = arrays
        _require(paid_path.shape == current_path.shape == (IMAGE_COUNT, ATOM_COUNT, 3)
                 and energy32.shape == (IMAGE_COUNT,)
                 and full_force32.shape == (IMAGE_COUNT, ATOM_COUNT, 3),
                 f"logical full-state shape failed at batch {batch}")
        _require(np.isfinite(paid_path).all() and np.isfinite(current_path).all()
                 and np.isfinite(energy32).all() and np.isfinite(full_force32).all(),
                 f"logical full-state contains non-finite data at batch {batch}")
        _require(np.array_equal(paid_path[[0, -1]], endpoint_coordinates)
                 and np.array_equal(current_path[[0, -1]], endpoint_coordinates)
                 and np.array_equal(energy32[[0, -1]], endpoint_energies)
                 and np.array_equal(full_force32[[0, -1]], endpoint_forces),
                 f"endpoint inheritance failed at batch {batch}")
        expected_role = "current_initial" if batch == 1 else "trial"
        disposition = str(record.get("disposition"))
        _require(record.get("role") == expected_role
                 and metadata["role"] == expected_role
                 and metadata["disposition"] == disposition,
                 f"full-state role/disposition mismatch at batch {batch}")
        if disposition in {"current_initial", "accepted"}:
            _require(np.array_equal(current_path, paid_path),
                     f"accepted/current mapping failed at batch {batch}")
            current_arrays = arrays
            current_source_batch = batch
            if disposition == "accepted":
                accepted += 1
        else:
            _require(disposition == "rejected" and current_arrays is not None,
                     f"invalid rejected mapping at batch {batch}")
            _require(np.array_equal(current_path, current_arrays[3])
                     and not np.array_equal(paid_path, current_path),
                     f"rejected trial rollback mapping failed at batch {batch}")
            rejected += 1
        _require(int(record.get("current_source_batch_id", -1)) == current_source_batch
                 and int(metadata["current_source_batch_id"]) == current_source_batch,
                 f"current-source mapping failed at batch {batch}")
        for key in (
            "paid_path_sha256", "energy32_sha256", "full_force32_sha256",
            "current_path_sha256", "endpoint_cache_sha256",
        ):
            _require(str(record.get(key)) == str(metadata[key]),
                     f"WAL/shard field {key} differs at batch {batch}")
        _require(record.get("file_sha256") == sha256_file(shard)
                 and record.get("relative_path")
                 == f"paid_batch_full_state/{shard.name}"
                 and record.get("endpoint_optimization_N_F") == 0
                 and record.get("additional_OpenMM_calls") == 0
                 and int(metadata["endpoint_optimization_N_F"]) == 0
                 and int(metadata["additional_OpenMM_calls"]) == 0,
                 f"full-state shard publication binding failed at batch {batch}")
        if optimizer_commits is not None:
            commit = commits[batch - 1]
            binding_keys = (
                "batch_id", "optimization_N_F", "role", "disposition",
                "encoding", "previous_batch_id", "current_source_batch_id",
                "relative_path", "file_sha256", "paid_path_sha256", "energy32_sha256",
                "full_force32_sha256", "current_path_sha256", "endpoint_cache_sha256",
                "endpoint_optimization_N_F", "additional_OpenMM_calls",
            )
            archive_binding = commit.get("optimizer_metadata", {}).get(
                "full_state_archive", {}
            )
            _require(commit.get("event") == "batch_commit"
                     and int(commit.get("batch_id", -1)) == batch
                     and int(commit.get("nf_after", -1)) == 30 * batch
                     and commit.get("paid_state_role") == expected_role
                     and commit.get("disposition") == disposition
                     and commit.get("paid_path32_sha256") == record["paid_path_sha256"]
                     and commit.get("current_path32_sha256") == record["current_path_sha256"]
                     and archive_binding.get("schema") == SHARD_SCHEMA
                     and {key: archive_binding.get(key) for key in binding_keys}
                     == {key: record[key] for key in binding_keys}
                     and set(archive_binding) == {"schema", *binding_keys},
                     f"optimizer ledger/full-state binding failed at batch {batch}")

    _require(accepted >= 1 and accepted + rejected == limit - 1,
             "ODE accepted/rejected full-state count is impossible")
    _require(current_arrays is not None
             and terminal.get("current_source_batch_id") == current_source_batch
             and terminal.get("current_state_sha256") == array_sha256(current_arrays[3]),
             "terminal full-state current mapping failed")
    complete = load_json_object(complete_path, require_canonical=True)
    expected_complete = {
        "schema": COMPLETE_SCHEMA,
        "status": "SEALED",
        "run_label": str(run_label),
        "spring_k_eV_per_A2": float(spring_k),
        "paid_batches": limit,
        "paid_optimization_N_F": 30 * limit,
        "shard_count": limit,
        "block_size": BLOCK_SIZE,
        "terminal_current_source_batch_id": current_source_batch,
        "terminal_current_state_sha256": terminal["current_state_sha256"],
        "wal_sha256": sha256_file(wal),
        "wal_terminal_record_sha256": terminal["record_sha256"],
        "endpoint_cache_path": str(endpoint_path),
        "endpoint_cache_sha256": endpoint_sha,
        "authorization_sha256": str(authorization_sha256),
        "source_closure_sha256": str(source_closure_sha256),
        "endpoint_optimization_N_F": 0,
        "additional_OpenMM_calls": 0,
        "encoding": "float64_checkpoint_plus_uint64_xor_delta",
        "bit_exact_reconstruction_required": True,
    }
    _require(complete == expected_complete,
             "full-state completion document differs from the verified archive")
    return {
        "pass": True,
        "paid_batches": limit,
        "paid_optimization_N_F": 30 * limit,
        "shards": limit,
        "accepted_trials": accepted,
        "rejected_trials": rejected,
        "terminal_current_source_batch_id": current_source_batch,
        "terminal_current_state_sha256": terminal["current_state_sha256"],
        "endpoint_optimization_N_F": 0,
        "additional_OpenMM_calls": 0,
        "wal_sha256": sha256_file(wal),
        "completion_sha256": sha256_file(complete_path),
    }
