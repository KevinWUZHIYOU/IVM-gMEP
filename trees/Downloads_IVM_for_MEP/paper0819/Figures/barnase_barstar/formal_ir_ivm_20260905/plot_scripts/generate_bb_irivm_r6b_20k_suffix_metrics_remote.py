#!/usr/bin/env python3
"""Generate a zero-OpenMM compact metric ledger for R6B calls 516--693."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


N_IMAGES = 32
N_DOF = 9396
EV_TO_KCAL_PER_MOL = 23.0605
SCHEMA = "bb-irivm-r6b-native32-metric-suffix-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def scalar(archive: Mapping[str, Any], key: str) -> Any:
    value = np.asarray(archive[key])
    require(value.shape in {(), (1,)}, f"{key} is not scalar-like")
    return value.reshape(()).item()


def metric_triplet(path32: Any, energy32: Any, force32: Any):
    path = np.ascontiguousarray(np.asarray(path32, dtype=np.float64)).reshape(N_IMAGES, N_DOF)
    energy = np.ascontiguousarray(np.asarray(energy32, dtype=np.float64)).reshape(N_IMAGES)
    force = np.ascontiguousarray(np.asarray(force32, dtype=np.float64)).reshape(N_IMAGES, N_DOF)
    require(np.isfinite(path).all() and np.isfinite(energy).all() and np.isfinite(force).all(),
            "non-finite state")
    tangent = path[2:] - path[:-2]
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    require(np.all(tangent_norm > 1.0e-30), "degenerate central tangent")
    tangent /= tangent_norm
    internal_force = force[1:-1]
    perpendicular = internal_force - np.sum(internal_force * tangent, axis=1, keepdims=True) * tangent
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    require(np.all(segment > 0.0), "degenerate path segment")
    weights = 0.5 * (segment[:-1] + segment[1:])
    i_perp = float(np.sum(weights * np.linalg.norm(perpendicular, axis=1) ** 2))
    delta_e_eV = float(np.max(energy) - energy[0])
    force_norm = np.linalg.norm(force, axis=1)
    action_eV = float(np.sum(0.5 * (force_norm[:-1] + force_norm[1:]) * segment))
    require(i_perp >= 0.0 and delta_e_eV >= 0.0 and action_eV > 0.0
            and all(math.isfinite(v) for v in (i_perp, delta_e_eV, action_eV)),
            "invalid metric")
    return i_perp, delta_e_eV, action_eV, {
        "arc_length_A": float(np.sum(segment)),
        "minimum_segment_length_A": float(np.min(segment)),
        "maximum_segment_length_A": float(np.max(segment)),
        "minimum_force_norm_eV_per_A": float(np.min(force_norm)),
        "maximum_force_norm_eV_per_A": float(np.max(force_norm)),
    }


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-canonical-path", required=True)
    parser.add_argument("--generator-sha256", required=True)
    parser.add_argument("--start-call", type=int, default=516)
    parser.add_argument("--end-call", type=int, default=693)
    args = parser.parse_args()

    require("openmm" not in sys.modules and "simtk.openmm" not in sys.modules,
            "OpenMM was imported before analysis")
    primary = args.primary.resolve(strict=True)
    source_dir = args.source_dir.resolve(strict=True)
    verification = json.loads(primary.read_text(encoding="utf-8"))
    require(
        verification.get("schema") == "bb-irivm-selected-full-state-history-verification-v6b"
        and verification.get("status") == "PASS_R6B_NF30000_EXACT_PREFIX318_AND_FULL_STATE_CAPTURE",
        "R6B primary verifier is not PASS",
    )
    inventory = verification.get("path_inventory")
    require(isinstance(inventory, list) and len(inventory) == 1053, "R6B inventory differs")
    rows = []
    for call in range(args.start_call, args.end_call + 1):
        reference = inventory[call - 1]
        require(
            int(reference["call_per_image"]) == call
            and reference.get("formally_reportable") is True
            and reference.get("terminal_uncommitted_charged_tail") is False,
            f"call {call} is not formally reportable",
        )
        shard = (source_dir / f"call_{call:04d}_stable_last.npz").resolve(strict=True)
        shard_hash = sha256_file(shard)
        require(shard_hash == str(reference["shard_sha256"]), f"shard hash differs at call {call}")
        with np.load(shard, allow_pickle=False) as saved:
            path = np.asarray(saved["path32_A"], dtype=np.float64)
            energy = np.asarray(saved["energy_eV"], dtype=np.float64)
            force = np.asarray(saved["force_eV_per_A"], dtype=np.float64)
            saved_call = int(scalar(saved, "call_per_image"))
            nf = int(scalar(saved, "optimization_N_F"))
            stored_i = float(scalar(saved, "discrete_I_perp"))
            additional_calls = int(scalar(saved, "additional_physical_calls"))
        require(saved_call == call and nf == int(reference["optimization_N_F"]),
                f"call/NF identity differs at call {call}")
        require(additional_calls == 0, f"capture added calls at call {call}")
        path_hash = sha256_array(path)
        energy_hash = sha256_array(energy)
        force_hash = sha256_array(force)
        require(
            path_hash == str(reference["path_sha256"])
            and energy_hash == str(reference["energy_sha256"])
            and force_hash == str(reference["force_sha256"]),
            f"path/E/Force identity differs at call {call}",
        )
        i_perp, delta_e_eV, action_eV, diagnostics = metric_triplet(path, energy, force)
        require(np.asarray(i_perp).tobytes() == np.asarray(stored_i).tobytes(),
                f"stored/recomputed I_perp differs at call {call}")
        rows.append({
            "method": "IVM",
            "algorithm": "IR-IVM (inexact + equal-arclength reparameterization)",
            "optimization_N_F": nf,
            "call_per_image": call,
            "i_perp": i_perp,
            "delta_e_eV": delta_e_eV,
            "delta_e_kcal_per_mol": delta_e_eV * EV_TO_KCAL_PER_MOL,
            "geometric_action_eV": action_eV,
            "geometric_action_kcal_per_mol": action_eV * EV_TO_KCAL_PER_MOL,
            "path_sha256": path_hash,
            "energy_sha256": energy_hash,
            "force_sha256": force_hash,
            "source_shard": str(shard),
            "source_shard_sha256": shard_hash,
            "source_transport": "remote external-disk R6B full-state shard",
            "formally_reportable_under_R6B": True,
            **diagnostics,
        })
    require(len(rows) == args.end_call - args.start_call + 1, "suffix row count differs")
    require(rows[-1]["call_per_image"] == 693 and rows[-1]["optimization_N_F"] == 19993,
            "suffix does not end at call693/NF19993")
    require("openmm" not in sys.modules and "simtk.openmm" not in sys.modules,
            "OpenMM was imported during analysis")
    output = {
        "schema": SCHEMA,
        "status": "PASS_ZERO_OPENMM_ZERO_NEW_NOF",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {
            "canonical_path": args.generator_canonical_path,
            "sha256": args.generator_sha256,
            "execution_mode": "source streamed to remote python stdin; output returned only through existing Mutagen session",
        },
        "primary_verification": {
            "path": str(primary), "sha256": sha256_file(primary), "bytes": primary.stat().st_size
        },
        "source_directory": str(source_dir),
        "start_call": args.start_call,
        "end_call": args.end_call,
        "rows": rows,
        "metric_protocol": {
            "images": 32, "segments": 31, "densification": False,
            "i_perp": "central-difference tangent, normal full Force, local Cartesian Voronoi weights",
            "delta_e": "max(E_0,...,E_31)-E_0",
            "geometric_action_formula": "sum_i 0.5*(||F_i||+||F_(i+1)||)*||x_(i+1)-x_i||",
            "eV_to_kcal_per_mol": EV_TO_KCAL_PER_MOL,
        },
        "physical_calls": 0,
        "optimization_N_F_contribution": 0,
    }
    atomic_json(args.output, output)
    print(json.dumps({
        "output": str(args.output),
        "sha256": sha256_file(args.output),
        "bytes": args.output.stat().st_size,
        "rows": len(rows),
        "last_N_F": rows[-1]["optimization_N_F"],
        "physical_calls": 0,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
