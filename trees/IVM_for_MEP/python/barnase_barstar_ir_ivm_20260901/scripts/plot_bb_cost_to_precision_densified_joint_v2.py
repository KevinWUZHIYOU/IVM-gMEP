#!/usr/bin/env python3
"""Strict joint-E/F v2 adapter for the established densified BB figure.

Rendering and scientific comparison remain in the registered v1 plotter.  This
adapter replaces only its four-path loader with one that validates the v2
atomic shards, including all dense coordinates, energies, and full forces.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ORIGINAL_PLOTTER_PATH = HERE / "plot_bb_cost_to_precision_densified.py"
OLD = _load("bb_densified_plotter_v1_render", ORIGINAL_PLOTTER_PATH)
ODE = _load("bb_dense_ode_joint_v2_plot_validation", HERE / "evaluate_bb_dense_four_paths_joint_v2.py")
IVM = _load("bb_dense_ivm_joint_v2_plot_validation", HERE / "evaluate_bb_dense_four_ivm_joint_v2.py")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def scalar(saved, key: str):
    return np.asarray(saved[key]).item()


def file_binding(path: Path):
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": ODE.sha256_file(path)}


def load_dense_four_v2(result_raw: Path, audit_raw: Path, contract_raw: Path, method: str):
    result = result_raw.expanduser().resolve(strict=True)
    audit_path = audit_raw.expanduser().resolve(strict=True)
    contract_path = contract_raw.expanduser().resolve(strict=True)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    host_role = str(contract["paths"][0]["host_role"])
    if method == "ODE-NEB":
        require(audit.get("schema") == ODE.AUDIT_SCHEMA and audit.get("status") == "PASS" and audit.get("method") == method, "ODE v2 audit is not PASS")
        _, topology, _locks, rows = ODE.validate_and_load(contract_path, host_role)
        evaluator_sha = ODE.sha256_file(Path(ODE.__file__))
        topology_sha = ODE.sha256_file(topology)
        bb_core_sha = ODE.sha256_file(ODE.SRC / "bb_core.py")
        with np.load(result, allow_pickle=False) as aggregate:
            shard_paths = [Path(value).expanduser().resolve(strict=True) for value in np.asarray(aggregate["shard_path"]).astype(str)]
            shard_hashes = np.asarray(aggregate["shard_sha256"]).astype(str)
        require(len(shard_paths) == 4, "ODE v2 aggregate must bind four shards")
        shards = []
        for index, (row, path) in enumerate(zip(rows, shard_paths)):
            require(ODE.sha256_file(path) == shard_hashes[index], "ODE v2 shard file SHA mismatch")
            shards.append(ODE._validate_shard(path, row, ODE.sha256_file(contract_path), topology_sha, evaluator_sha, bb_core_sha))
        ODE._validate_aggregate(result, rows, shards, contract_path)
        discrete_wal = Path(str(audit["discrete_joint_ef_wal"])).expanduser().resolve(strict=True)
        dense_wal = Path(str(audit["dense_joint_ef_wal"])).expanduser().resolve(strict=True)
        require(ODE.sha256_file(discrete_wal) == audit.get("discrete_joint_ef_wal_sha256"), "ODE v2 discrete WAL/audit SHA mismatch")
        require(ODE.sha256_file(dense_wal) == audit.get("dense_joint_ef_wal_sha256"), "ODE v2 dense WAL/audit SHA mismatch")
        closure = ODE.validate_all_wal_shard_closure(rows, shards, discrete_wal, dense_wal)
        require(closure == audit.get("WAL_shard_pointwise_closure"), "ODE v2 recomputed WAL/shard closure differs from audit")
        official = np.asarray([row["official_path32_sha256"] for row in rows])
        engine_binding = None
    else:
        require(audit.get("schema") == IVM.AUDIT_SCHEMA and audit.get("status") == "PASS" and audit.get("method") == method, "IVM v2 audit is not PASS")
        _, topology, _locks, rows = IVM.validate_and_load(contract_path, host_role)
        evaluator_sha = ODE.sha256_file(Path(IVM.__file__))
        engine_sha = ODE.sha256_file(IVM.ENGINE_PATH)
        topology_sha = ODE.sha256_file(topology)
        bb_core_sha = ODE.sha256_file(IVM.ROOT / "src" / "bb_core.py")
        with np.load(result, allow_pickle=False) as aggregate:
            shard_paths = [Path(value).expanduser().resolve(strict=True) for value in np.asarray(aggregate["shard_path"]).astype(str)]
            shard_hashes = np.asarray(aggregate["shard_sha256"]).astype(str)
        require(len(shard_paths) == 4, "IVM v2 aggregate must bind four shards")
        shards = []
        for index, (row, path) in enumerate(zip(rows, shard_paths)):
            require(ODE.sha256_file(path) == shard_hashes[index], "IVM v2 shard file SHA mismatch")
            shards.append(IVM._validate_shard(path, row, ODE.sha256_file(contract_path), topology_sha, evaluator_sha, engine_sha, bb_core_sha))
        IVM._validate_aggregate(result, rows, shards, ODE.sha256_file(contract_path))
        online_wal = Path(str(audit["online_paid_state_wal"])).expanduser().resolve(strict=True)
        dense_wal = Path(str(audit["dense_joint_ef_wal"])).expanduser().resolve(strict=True)
        require(ODE.sha256_file(online_wal) == audit.get("online_paid_state_wal_sha256"), "IVM v2 online-state WAL/audit SHA mismatch")
        require(ODE.sha256_file(dense_wal) == audit.get("dense_joint_ef_wal_sha256"), "IVM v2 dense WAL/audit SHA mismatch")
        online_closure = IVM._validate_online_wal(rows, ODE._read_wal_events(online_wal))
        require(online_closure == audit.get("online_source_closure"), "IVM v2 recomputed online-state WAL closure differs from audit")
        dense_events = ODE._read_wal_events(dense_wal)
        dense_paths = []
        for row_index, (row, shard) in enumerate(zip(rows, shards)):
            dense_paths.append({
                "label": row["label"],
                **ODE._validate_stage_wal_closure(
                    dense_events, row_index, row, "dense_arc500_joint_EF",
                    np.asarray(shard["dense_coordinates_A"]),
                    np.asarray(shard["dense_energy_eV"]),
                    np.asarray(shard["dense_force_eV_per_A"]),
                ),
            })
        closure = {"status": "PASS", "lifetime": ODE._wal_lifetime_stats(dense_events), "paths": dense_paths}
        require(closure == audit.get("dense_WAL_shard_pointwise_closure"), "IVM v2 recomputed dense WAL/shard closure differs from audit")
        official = None
        engine_binding = file_binding(IVM.ENGINE_PATH)
    require(Path(str(audit["output"])).expanduser().resolve() == result and audit["output_sha256"] == ODE.sha256_file(result), f"{method} v2 result/audit mismatch")
    require(Path(str(audit["contract"])).expanduser().resolve() == contract_path and audit["contract_sha256"] == ODE.sha256_file(contract_path), f"{method} v2 contract/audit mismatch")
    require(int(audit.get("optimization_N_F_contribution", -1)) == 0 and int(audit.get("individual_incomplete_WAL_attempts_reused", -1)) == 0, f"{method} v2 accounting mismatch")
    manifest_path = Path(audit["shard_manifest"]).expanduser().resolve(strict=True)
    require(ODE.sha256_file(manifest_path) == audit["shard_manifest_sha256"], f"{method} v2 manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("status") == "PASS" and manifest.get("aggregate_sha256") == ODE.sha256_file(result) and len(manifest.get("shards", [])) == 4, f"{method} v2 manifest mismatch")
    for index, record in enumerate(manifest["shards"]):
        require(Path(record["path"]).expanduser().resolve() == shard_paths[index] and record["sha256"] == shard_hashes[index], f"{method} v2 manifest shard mismatch")
        for key in ("dense_coordinate_sha256", "dense_energy_sha256", "dense_force_sha256"):
            require(record[key] == str(scalar(shards[index], key)), f"{method} v2 manifest {key} mismatch")
        require(record.get("selected_path_sha256", record.get("path_sha256")) == rows[index]["path_sha256"], f"{method} v2 manifest selected path mismatch")
        if method == "IVM":
            require(record.get("online_energy_sha256") == rows[index]["online_energy_sha256"] and record.get("online_force_sha256") == rows[index]["online_force_sha256"], "IVM v2 manifest online E/F mismatch")
    with np.load(result, allow_pickle=False) as saved:
        nf = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        i_perp = np.asarray(saved["discrete_I_perp"], dtype=np.float64)
        delta_e = np.asarray(saved["densified_Delta_E_kcal_per_mol"], dtype=np.float64)
        path_hash = np.asarray(saved["selected_path_sha256"]).astype(str)
    require(tuple(nf) == OLD.EXPECTED_NF[method], f"{method} v2 N_F sequence mismatch")
    require(np.all(np.isfinite(i_perp)) and np.all(i_perp >= 0) and np.all(np.isfinite(delta_e)), f"{method} v2 metrics non-finite")
    data = {"number_of_forces": nf, "i_perp": i_perp, "delta_e": delta_e, "selected_path_sha256": path_hash}
    if official is not None:
        data["official_path32_sha256"] = official
    binding = {
        "result": file_binding(result), "audit": file_binding(audit_path), "contract": file_binding(contract_path),
        "topology": file_binding(topology), "bb_core": file_binding(ODE.SRC / "bb_core.py"),
        "evaluator": file_binding(Path(ODE.__file__) if method == "ODE-NEB" else Path(IVM.__file__)),
        "joint_engine": engine_binding, "manifest": file_binding(manifest_path),
        "authoritative_shards": [file_binding(path) for path in shard_paths],
        "WALs": [file_binding(online_wal if method == "IVM" else discrete_wal), file_binding(dense_wal)],
        "dense_full_force_vectors_verified": 4 * ODE.N_DENSE,
    }
    return data, binding


def main(argv: Iterable[str] | None = None) -> None:
    # Make the inherited contract validator and provenance identify this v2
    # adapter, while retaining the original renderer as a bound dependency.
    OLD.__file__ = str(Path(__file__).resolve())
    OLD.DIRECT_PLOT_DEPENDENCIES = (ORIGINAL_PLOTTER_PATH.resolve(), *OLD.DIRECT_PLOT_DEPENDENCIES)
    OLD.load_dense_four = load_dense_four_v2
    OLD.main(list(argv) if argv is not None else None)


if __name__ == "__main__":
    main()
