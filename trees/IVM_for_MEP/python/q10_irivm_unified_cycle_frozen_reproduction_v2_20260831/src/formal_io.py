"""Fail-closed portable I/O primitives for the formal campaign.

This module contains no optimizer or metric logic. All registered paths are
POSIX relative paths under an explicitly declared root. Final files are written
through a same-directory temporary file, fsynced, validated when applicable,
and atomically installed with ``os.replace``.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Iterator, Mapping


class FormalIOError(RuntimeError):
    """Raised when a portable-path, hash, lock, or atomic-write gate fails."""


@dataclass(frozen=True)
class CampaignPaths:
    source_file: Path
    campaign_root: Path
    repository_root: Path


def derive_campaign_paths(source_file: str | os.PathLike[str]) -> CampaignPaths:
    """Derive final-layout roots without embedding a username or home path.

    The final layout is ``REPOSITORY_ROOT/python/CAMPAIGN/src/file.py``.
    Therefore ``file.parents[1]`` is the campaign root and ``file.parents[3]``
    is the repository root. The staging tree intentionally derives a synthetic
    repository root; no external dependency is resolved from staging.
    """

    source = Path(source_file).resolve(strict=True)
    if len(source.parents) < 4:
        raise FormalIOError(f"source is too shallow for campaign layout: {source}")
    campaign = source.parents[1]
    repository = source.parents[3]
    if campaign.parent.parent != repository:
        raise FormalIOError("campaign/repository root derivation is inconsistent")
    return CampaignPaths(source, campaign, repository)


def _reject_symlink_components(root: Path, relative: PurePosixPath) -> None:
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise FormalIOError(f"symlink component is not registered: {cursor}")


def resolve_registered_path(
    root: str | os.PathLike[str],
    relative_path: str,
    *,
    must_exist: bool = True,
    allow_symlinks: bool = False,
) -> Path:
    """Resolve one registered POSIX path and reject all escape mechanisms."""

    if not isinstance(relative_path, str) or not relative_path:
        raise FormalIOError("registered path must be a non-empty string")
    if "\\" in relative_path:
        raise FormalIOError("registered paths use POSIX separators only")
    portable = PurePosixPath(relative_path)
    if portable.is_absolute() or any(part in {"", ".", ".."} for part in portable.parts):
        raise FormalIOError(f"non-portable registered path: {relative_path!r}")

    root_path = Path(root).resolve(strict=True)
    if not allow_symlinks:
        _reject_symlink_components(root_path, portable)
    candidate = root_path.joinpath(*portable.parts)
    resolved = candidate.resolve(strict=must_exist)
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise FormalIOError(f"registered path escapes root: {relative_path!r}") from exc
    if must_exist and not resolved.is_file():
        raise FormalIOError(f"registered dependency is not a file: {resolved}")
    return resolved


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_canonical_json(path: str | os.PathLike[str]) -> Any:
    raw = Path(path).read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormalIOError(f"invalid UTF-8 JSON: {path}") from exc
    if raw != canonical_json_bytes(payload):
        raise FormalIOError(f"JSON is not canonical sorted compact UTF-8: {path}")
    return payload


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextlib.contextmanager
def atomic_output_path(
    destination: str | os.PathLike[str],
    *,
    refuse_overwrite: bool = True,
    suffix: str | None = None,
) -> Iterator[Path]:
    """Yield a same-directory temporary path and atomically install it.

    The caller writes the yielded path. On clean exit the file is fsynced and
    atomically renamed. On error the temporary file is unlinked.
    """

    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if refuse_overwrite and target.exists():
        raise FormalIOError(f"refusing to overwrite existing artifact: {target}")
    temp_suffix = target.suffix if suffix is None else suffix
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=temp_suffix, dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        yield temporary
        if not temporary.is_file():
            raise FormalIOError(f"atomic writer did not create file: {temporary}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if refuse_overwrite and target.exists():
            raise FormalIOError(f"destination appeared during write: {target}")
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_bytes(
    destination: str | os.PathLike[str],
    data: bytes,
    *,
    refuse_overwrite: bool = True,
) -> None:
    with atomic_output_path(destination, refuse_overwrite=refuse_overwrite, suffix="") as temp:
        with temp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())


def atomic_write_json(
    destination: str | os.PathLike[str],
    payload: Any,
    *,
    refuse_overwrite: bool = True,
) -> None:
    atomic_write_bytes(
        destination,
        canonical_json_bytes(payload),
        refuse_overwrite=refuse_overwrite,
    )


def atomic_savez_compressed(
    destination: str | os.PathLike[str],
    arrays: Mapping[str, Any],
    *,
    validator: Callable[[Path], None] | None = None,
    refuse_overwrite: bool = True,
) -> None:
    """Atomically write and optionally reopen/validate an NPZ archive."""

    import numpy as np

    with atomic_output_path(destination, refuse_overwrite=refuse_overwrite, suffix=".npz") as temp:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        with np.load(temp, allow_pickle=False) as archive:
            for key in archive.files:
                if archive[key].dtype.kind == "O":
                    raise FormalIOError(f"object array is forbidden in NPZ field {key}")
        if validator is not None:
            validator(temp)


def verify_registration_seal(
    registration_path: str | os.PathLike[str],
    seal_path: str | os.PathLike[str],
) -> tuple[dict[str, Any], str]:
    registration = Path(registration_path)
    payload = load_canonical_json(registration)
    if not isinstance(payload, dict):
        raise FormalIOError("registration root must be a JSON object")
    digest = sha256_file(registration)
    seal_raw = Path(seal_path).read_bytes()
    expected = f"{digest}\n".encode("ascii")
    if seal_raw != expected:
        raise FormalIOError("registration seal does not match exact registration bytes")
    return payload, digest


def verify_hash_manifest(
    root: str | os.PathLike[str], manifest: Mapping[str, str]
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for relative, expected in manifest.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise FormalIOError(f"invalid SHA-256 declaration for {relative}")
        path = resolve_registered_path(root, relative)
        actual = sha256_file(path)
        if actual != expected:
            raise FormalIOError(
                f"SHA-256 mismatch for {relative}: expected {expected}, got {actual}"
            )
        resolved[relative] = path
    return resolved


class HostLocalFcntlLock:
    """Unique nonblocking host-local advisory lock."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._descriptor: int | None = None

    def __enter__(self) -> "HostLocalFcntlLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_CREAT | os.O_RDWR
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            os.close(descriptor)
            raise FormalIOError(f"host-local campaign lock is already held: {self.path}") from exc
        self._descriptor = descriptor
        metadata = f"pid={os.getpid()}\n".encode("ascii")
        os.ftruncate(descriptor, 0)
        os.write(descriptor, metadata)
        os.fsync(descriptor)
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._descriptor is not None:
            try:
                fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            finally:
                os.close(self._descriptor)
                self._descriptor = None
