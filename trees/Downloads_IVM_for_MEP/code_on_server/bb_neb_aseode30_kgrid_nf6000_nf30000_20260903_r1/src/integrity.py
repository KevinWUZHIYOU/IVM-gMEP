"""Canonical hashing, exclusive artifacts, and whole-source identity."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Iterable, Mapping

import numpy as np


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST_NAME = "SOURCE_MANIFEST.json"
SOURCE_MANIFEST_SCHEMA = "bb-neb-aseode-kgrid-whole-source-manifest-v1"
FORBIDDEN_SOURCE_DIRECTORIES = frozenset(
    {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "archives",
        "campaign",
        "outputs",
        "results",
        "runs",
    }
)
FORBIDDEN_SUFFIXES = (".pyc", ".pyo")


class IntegrityError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_new(path: Path | str, value: Mapping[str, Any], *, readonly: bool = False) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(canonical_json_bytes(dict(value)) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    if readonly:
        os.chmod(target, 0o444)
    _fsync_directory(target.parent)
    return sha256_file(target)


def write_npz_new(path: Path | str, **arrays: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(target.parent)
    return sha256_file(target)


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise IntegrityError(f"duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def parse_json_object(
    payload: bytes | str,
    *,
    source: object = "<memory>",
    require_canonical: bool = False,
) -> dict[str, Any]:
    """Parse one JSON object while rejecting duplicate keys at every depth."""

    if isinstance(payload, bytes):
        raw = payload
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise IntegrityError(f"invalid UTF-8 JSON object: {source}") from error
    elif isinstance(payload, str):
        text = payload
        raw = payload.encode("utf-8")
    else:
        raise TypeError("JSON payload must be bytes or str")
    try:
        value = json.loads(text, object_pairs_hook=_unique_json_object)
    except IntegrityError as error:
        raise IntegrityError(f"duplicate key in JSON object {source}: {error}") from error
    except json.JSONDecodeError as error:
        raise IntegrityError(f"invalid JSON object: {source}") from error
    if not isinstance(value, dict):
        raise IntegrityError(f"JSON root is not an object: {source}")
    if require_canonical and raw != canonical_json_bytes(value) + b"\n":
        raise IntegrityError(f"JSON is not canonical: {source}")
    return value


def load_json_object(path: Path | str, *, require_canonical: bool = False) -> dict[str, Any]:
    source = Path(path)
    return parse_json_object(
        source.read_bytes(), source=source, require_canonical=require_canonical
    )


def is_within(path: Path | str, root: Path | str) -> bool:
    candidate = Path(path).resolve()
    parent = Path(root).resolve()
    return candidate == parent or parent in candidate.parents


def _regular_entry(root: Path, path: Path) -> dict[str, Any]:
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or path.is_symlink():
        raise IntegrityError(f"source contains a non-regular file: {path}")
    relative = path.relative_to(root).as_posix()
    return {
        "mode": format(stat.S_IMODE(status.st_mode), "04o"),
        "path": relative,
        "sha256": sha256_file(path),
        "size": status.st_size,
        "type": "regular",
    }


def scan_source_tree(root: Path | str = SOURCE_ROOT) -> list[dict[str, Any]]:
    source = Path(root).resolve(strict=True)
    entries: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(source, topdown=True):
        current = Path(directory)
        for name in list(directory_names):
            path = current / name
            if path.is_symlink():
                raise IntegrityError(f"source contains a symlink directory: {path}")
            if name in FORBIDDEN_SOURCE_DIRECTORIES:
                raise IntegrityError(f"source contains forbidden directory {name!r}: {path}")
        directory_names.sort()
        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(source).as_posix()
            if relative == SOURCE_MANIFEST_NAME:
                continue
            if name == ".DS_Store" or name.endswith(FORBIDDEN_SUFFIXES):
                raise IntegrityError(f"source contains forbidden generated file: {path}")
            entries.append(_regular_entry(source, path))
    entries.sort(key=lambda item: str(item["path"]))
    if not entries:
        raise IntegrityError("source tree is empty")
    return entries


def _core_paths(root: Path) -> tuple[str, ...]:
    path = root / "CORE_CLOSURE_FILES.txt"
    lines = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if lines != tuple(sorted(set(lines))):
        raise IntegrityError("CORE_CLOSURE_FILES.txt must be sorted and unique")
    for relative in lines:
        item = Path(relative)
        if item.is_absolute() or ".." in item.parts or not (root / item).is_file():
            raise IntegrityError(f"invalid core-closure path: {relative!r}")
    return lines


def scientific_core_map(root: Path | str = SOURCE_ROOT) -> dict[str, str]:
    source = Path(root).resolve(strict=True)
    return {relative: sha256_file(source / relative) for relative in _core_paths(source)}


def source_manifest_document(root: Path | str = SOURCE_ROOT) -> dict[str, Any]:
    source = Path(root).resolve(strict=True)
    entries = scan_source_tree(source)
    core = scientific_core_map(source)
    return {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "entries": entries,
        "file_count": len(entries),
        "scientific_core_files": core,
        "scientific_core_closure_sha256": canonical_json_sha256(core),
        "whole_source_closure_sha256": canonical_json_sha256(entries),
        "exclusions": ["self:SOURCE_MANIFEST.json"],
        "forbidden_source_directories": sorted(FORBIDDEN_SOURCE_DIRECTORIES),
    }


@dataclass(frozen=True)
class SourceIdentity:
    source_manifest_sha256: str
    whole_source_closure_sha256: str
    scientific_core_closure_sha256: str
    file_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_manifest_sha256": self.source_manifest_sha256,
            "whole_source_closure_sha256": self.whole_source_closure_sha256,
            "scientific_core_closure_sha256": self.scientific_core_closure_sha256,
            "file_count": self.file_count,
        }


def verify_source_identity(root: Path | str = SOURCE_ROOT) -> SourceIdentity:
    source = Path(root).resolve(strict=True)
    manifest_path = source / SOURCE_MANIFEST_NAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise IntegrityError("SOURCE_MANIFEST.json is absent or not a regular file")
    manifest = load_json_object(manifest_path, require_canonical=True)
    if manifest.get("schema") != SOURCE_MANIFEST_SCHEMA:
        raise IntegrityError("source-manifest schema differs")
    expected = source_manifest_document(source)
    if manifest != expected:
        raise IntegrityError("whole source tree differs from SOURCE_MANIFEST.json")
    return SourceIdentity(
        source_manifest_sha256=sha256_file(manifest_path),
        whole_source_closure_sha256=str(manifest["whole_source_closure_sha256"]),
        scientific_core_closure_sha256=str(
            manifest["scientific_core_closure_sha256"]
        ),
        file_count=int(manifest["file_count"]),
    )


def tree_file_manifest(
    root: Path | str, *, excluded_relative_paths: Iterable[str] = ()
) -> list[dict[str, Any]]:
    """Hash every regular file in a terminal run without following symlinks."""

    source = Path(root).resolve(strict=True)
    excluded = frozenset(str(value) for value in excluded_relative_paths)
    entries: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(source, topdown=True):
        current = Path(directory)
        for name in list(directory_names):
            path = current / name
            if path.is_symlink():
                raise IntegrityError(f"terminal tree contains symlink directory: {path}")
        directory_names.sort()
        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(source).as_posix()
            if relative in excluded:
                continue
            entries.append(_regular_entry(source, path))
    entries.sort(key=lambda item: str(item["path"]))
    return entries
