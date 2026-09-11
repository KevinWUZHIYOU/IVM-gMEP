#!/usr/bin/env python3
"""Stage the locked 32-image BB inputs without constructing OpenMM."""

from __future__ import annotations

import argparse
import io
import hashlib
import json
from pathlib import Path
import os
import zipfile

import numpy as np


EXPECTED = {
    "complex_fixed.pdb": (
        "26b58b4301c1e293c7fcd01878c3a208451621099419cbf5b41bd80eb96a0f49"
    ),
    "common_initial_path32.npz": (
        "6983365a5c39d117b5ec1ce4cb9afbaed1817c1190d27a567dea1fbce983e6cf"
    ),
}
EXPECTED_ARRAY_SHA256 = (
    "e6104f507303bc5e06abc557cad40ccd08924d21474e67073a3a9f6bd5823769"
)
ENDPOINT_SOURCE = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
    "bb_pip8_dsel_post30_b030_s32_a010_dense200_formal02_remote_archive_"
    "20260820_r1/runs/fresh-nonresume-pip8-dsel-post30-b030-s32-a010-"
    "dense200-formal02-remote-20260820-r1/dense200_data.npz"
)
ENDPOINT_SOURCE_AUDIT = ENDPOINT_SOURCE.with_name("DIAGNOSTIC_AUDIT.json")
ENDPOINT_SOURCE_SHA256 = (
    "c951b6bdfa356a838a5298740b44abc461344a712b3380ed3adda463a65c017c"
)
ENDPOINT_SOURCE_AUDIT_SHA256 = (
    "59534941c0a8748ed1a29c8dc4048a4d48693374c083610582a830d28f697fa7"
)
ENDPOINT_COORDINATE_SHA256 = (
    "5788e22f86df91ed5500e88a8dc053f7408bd2fb39414de121a6eec36b833db1"
)
ENDPOINT_ENERGY_SHA256 = (
    "24884695bc3bbecbf1e5007bf8f245bfe6d7eb5bf81725e986b8dd0ece088a31"
)
ENDPOINT_FORCE_SHA256 = (
    "c28ab9e8ad515beb3cfd5de8cc09af542ff9c80a96815d8d1b8b17d03fd8e01c"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_exclusive(source, destination):
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise RuntimeError(f"existing staged input drift: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(block)


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def deterministic_npz_exclusive(destination, arrays):
    """Write a portable NPZ with fixed ZIP metadata; never overwrite."""
    destination = Path(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    try:
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(arrays):
                stream = io.BytesIO()
                np.lib.format.write_array(
                    stream,
                    np.asarray(arrays[name]),
                    allow_pickle=False,
                )
                info = zipfile.ZipInfo(f"{name}.npy", (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, stream.getvalue())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument(
        "--endpoint-source",
        default=str(ENDPOINT_SOURCE),
        help="existing authoritative dense200 data; read-only and no OpenMM",
    )
    parser.add_argument(
        "--destination-root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve(strict=True)
    destination = Path(args.destination_root).resolve() / "inputs"
    for name, expected in EXPECTED.items():
        source = source_root / "inputs" / name
        if sha256(source) != expected:
            raise RuntimeError(f"canonical source hash mismatch: {source}")
        copy_exclusive(source, destination / name)
    with np.load(destination / "common_initial_path32.npz", allow_pickle=False) as data:
        path = np.ascontiguousarray(data["initial_path"], dtype=np.float64)
    if path.shape != (32, 9396):
        raise RuntimeError(f"unexpected canonical path shape: {path.shape}")
    manifest = json.loads(
        (Path(args.destination_root) / "inputs" / "INPUT_POINTERS.json").read_text()
    )
    array_hash = hashlib.sha256(path.tobytes()).hexdigest()
    if array_hash != EXPECTED_ARRAY_SHA256:
        raise RuntimeError("canonical path array hash mismatch")
    if array_hash != manifest["initial_path_array_sha256"]:
        raise RuntimeError("canonical path array hash mismatch")
    endpoint_source = Path(args.endpoint_source).resolve(strict=True)
    endpoint_audit = endpoint_source.with_name("DIAGNOSTIC_AUDIT.json")
    if sha256(endpoint_source) != ENDPOINT_SOURCE_SHA256:
        raise RuntimeError("authoritative dense200 endpoint source hash mismatch")
    if sha256(endpoint_audit) != ENDPOINT_SOURCE_AUDIT_SHA256:
        raise RuntimeError("authoritative endpoint source audit hash mismatch")
    source_audit = json.loads(endpoint_audit.read_text(encoding="utf-8"))
    if int(source_audit.get("endpoint_joint_energy_force_calls", -1)) != 2:
        raise RuntimeError("source audit does not certify two endpoint joint E/F calls")
    with np.load(endpoint_source, allow_pickle=False) as data:
        coordinates = np.ascontiguousarray(
            data["coordinates_A"][[0, -1]], dtype=np.float64
        )
        energies = np.ascontiguousarray(
            data["energies_eV"][[0, -1]], dtype=np.float64
        )
        forces = np.ascontiguousarray(
            data["forces_eV_per_A"][[0, -1]], dtype=np.float64
        )
    if array_sha256(coordinates) != ENDPOINT_COORDINATE_SHA256:
        raise RuntimeError("canonical endpoint coordinate hash mismatch")
    if array_sha256(energies) != ENDPOINT_ENERGY_SHA256:
        raise RuntimeError("canonical endpoint energy hash mismatch")
    if array_sha256(forces) != ENDPOINT_FORCE_SHA256:
        raise RuntimeError("canonical endpoint force hash mismatch")
    if not np.array_equal(coordinates, path[[0, -1]]):
        raise RuntimeError("cached endpoints do not match the 32-image path")
    cache_file = destination / "endpoint_joint_ef_cache.npz"
    deterministic_npz_exclusive(
        cache_file,
        {
            "coordinates_A": coordinates,
            "energies_eV": energies,
            "forces_eV_per_A": forces,
            "source_dense200_sha256": np.array(ENDPOINT_SOURCE_SHA256),
            "source_diagnostic_audit_sha256": np.array(
                ENDPOINT_SOURCE_AUDIT_SHA256
            ),
        },
    )
    with np.load(cache_file, allow_pickle=False) as cached:
        if array_sha256(cached["coordinates_A"]) != ENDPOINT_COORDINATE_SHA256:
            raise RuntimeError("staged endpoint cache coordinate drift")
        if array_sha256(cached["energies_eV"]) != ENDPOINT_ENERGY_SHA256:
            raise RuntimeError("staged endpoint cache energy drift")
        if array_sha256(cached["forces_eV_per_A"]) != ENDPOINT_FORCE_SHA256:
            raise RuntimeError("staged endpoint cache force drift")
    if sha256(cache_file) != manifest["files"]["endpoint_joint_ef_cache.npz"]:
        raise RuntimeError("staged endpoint cache file hash mismatch")
    print(
        "staged and verified canonical 32 x 9396 BB inputs plus inherited "
        "endpoint joint E/F cache; no OpenMM calls"
    )


if __name__ == "__main__":
    main()
