#!/usr/bin/env python3
"""Create one immutable plotting pack spec from a completed joint-E/F bundle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
PLOTTER_PATH = HERE / "plot_bb_cost_to_precision_joint_ef_dense.py"


def _load_plotter():
    spec = importlib.util.spec_from_file_location("bb_joint_ef_cost_pack_freezer_plot", PLOTTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PLOT = _load_plotter()


def atomic_json_no_overwrite(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pack-id", required=True)
    parser.add_argument("--method", choices=PLOT.METHODS, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--bb-core", type=Path, required=True)
    parser.add_argument("--wal", type=Path, action="append", required=True)
    parser.add_argument("--ode-original-force-key")
    parser.add_argument("--ivm-replay-result", type=Path)
    parser.add_argument("--ivm-verification-report", type=Path)
    parser.add_argument("--ivm-verification-schema")
    parser.add_argument("--ivm-verification-status")
    args = parser.parse_args(list(argv) if argv is not None else None)

    destination = args.output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    paths = {
        name: value.expanduser().resolve(strict=True)
        for name, value in {
            "result": args.result,
            "audit": args.audit,
            "contract": args.contract,
            "manifest": args.manifest,
            "evaluator": args.evaluator,
            "topology": args.topology,
            "bb_core": args.bb_core,
        }.items()
    }
    wals = [path.expanduser().resolve(strict=True) for path in args.wal]
    contract = PLOT.read_json(paths["contract"])
    PLOT.require(contract.get("frozen") is True, "source contract is not frozen")
    PLOT.require(contract.get("method") == args.method, "source contract method mismatch")
    with np.load(paths["result"], allow_pickle=False) as saved:
        required = {
            "optimization_N_F", "selected_path_sha256", "discrete_I_perp",
            "densified_Delta_E_kcal_per_mol", "shard_path", "shard_sha256",
        }
        PLOT.require(required.issubset(saved.files), "source aggregate fields missing")
        nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        path_sha = np.asarray(saved["selected_path_sha256"]).astype(str)
        i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        delta_e = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
        shard_paths = [Path(value).expanduser().resolve(strict=True) for value in np.asarray(saved["shard_path"]).astype(str)]
        shard_sha = np.asarray(saved["shard_sha256"]).astype(str)
        labels = np.asarray(saved["labels"]).astype(str) if "labels" in saved.files else None
    count = len(nf)
    PLOT.require(
        count > 0
        and all(len(value) == count for value in (path_sha, i_perp, delta_e, shard_paths, shard_sha)),
        "source aggregate vector lengths mismatch",
    )

    if args.method == "IVM":
        required_ivm = (
            args.ivm_replay_result,
            args.ivm_verification_report,
            args.ivm_verification_schema,
            args.ivm_verification_status,
        )
        PLOT.require(all(value is not None for value in required_ivm), "IVM source-proof arguments are incomplete")
        replay = args.ivm_replay_result.expanduser().resolve(strict=True)
        verification = args.ivm_verification_report.expanduser().resolve(strict=True)
    else:
        PLOT.require(
            args.ivm_replay_result is None
            and args.ivm_verification_report is None
            and args.ivm_verification_schema is None
            and args.ivm_verification_status is None,
            "ODE-NEB pack may not use IVM proof arguments",
        )
        replay = verification = None

    points = []
    for index in range(count):
        row = PLOT._find_contract_row(contract, int(nf[index]), str(path_sha[index]))
        label = str(labels[index]) if labels is not None else str(row["label"])
        PLOT.require(label == str(row["label"]), "aggregate/contract label mismatch")
        PLOT.require(PLOT.sha256_file(shard_paths[index]) == shard_sha[index], "aggregate shard SHA mismatch")
        if args.method == "IVM":
            identifier = row.get("path_identifier", {})
            evidence = {
                "kind": "sealed_online_history",
                "calls_per_image": int(identifier.get("calls_per_image", -1)),
                "history_index": int(identifier.get("history_index", -1)),
                "replay_result": PLOT.file_binding(replay),
                "verification_report": PLOT.file_binding(verification),
                "verification_schema": args.ivm_verification_schema,
                "verification_status": args.ivm_verification_status,
            }
        else:
            with np.load(shard_paths[index], allow_pickle=False) as shard:
                candidates = [
                    key
                    for key in (
                        args.ode_original_force_key,
                        "paid_internal_force_eV_per_A",
                        "internal_diagnostic_force_eV_per_A",
                    )
                    if key and key in shard.files
                ]
            PLOT.require(candidates, f"NF{int(nf[index])} lacks an original internal-force field")
            evidence = {
                "kind": "recomputed_original_internal_force",
                "force_key": candidates[0],
            }
        points.append(
            {
                "label": label,
                "optimization_N_F": int(nf[index]),
                "selected_path_sha256": str(path_sha[index]),
                "discrete_I_perp": float(i_perp[index]),
                "densified_Delta_E_kcal_per_mol": float(delta_e[index]),
                "authoritative_shard": PLOT.file_binding(shard_paths[index]),
                "discrete_I_perp_evidence": evidence,
            }
        )

    payload = {
        "schema": PLOT.PACK_SCHEMA,
        "status": PLOT.PACK_STATUS,
        "created_utc": PLOT.utc_now(),
        "pack_id": args.pack_id,
        "method": args.method,
        "include_all_pack_points": True,
        "closure": {
            **{name: PLOT.file_binding(path) for name, path in paths.items()},
            "WALs": [PLOT.file_binding(path) for path in wals],
        },
        "points": points,
        "pack_spec_freezer": PLOT.file_binding(Path(__file__).resolve()),
        "physical_calls_by_pack_spec_freezer": 0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".validate.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        pack, reports = PLOT.validate_pack_spec(temporary)
        PLOT.require(pack["point_count"] == len(points) == len(reports), "pack validation count mismatch")
    finally:
        temporary.unlink()
    atomic_json_no_overwrite(destination, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "pack_spec": str(destination),
                "sha256": PLOT.sha256_file(destination),
                "method": args.method,
                "point_count": len(points),
                "optimization_N_F": [int(value) for value in nf],
                "physical_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
