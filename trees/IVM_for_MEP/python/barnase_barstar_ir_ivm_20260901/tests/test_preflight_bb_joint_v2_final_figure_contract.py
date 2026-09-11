"""Synthetic zero-OpenMM test for the final joint-v2 figure preflight."""

from contextlib import redirect_stdout
import builtins
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "preflight_bb_joint_v2_final_figure_contract.py"


def load_module():
    spec = importlib.util.spec_from_file_location("joint_v2_figure_preflight_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class JointV2FigurePreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def bind(self, path: Path):
        return {"path": str(path.resolve()), "sha256": self.module.sha256_file(path)}

    def make_method(self, directory: Path, method: str, *, bad_iperp=False, bad_dense=False, duplicate_terminal=False):
        tag = "ivm" if method == "IVM" else "ode"
        metric_contract = directory / f"{tag}.contract.json"
        shards = []
        records = []
        rows = []
        arrays_by_row = []
        path_sha, dense_sha, energy_sha, force_sha, delta = [], [], [], [], []
        for index in range(4):
            selected = np.ascontiguousarray(np.arange(32)[:, None] * np.arange(1, 7)[None, :] + index, dtype=np.float64)
            dense = self.module.uniform_cartesian_arc500(selected)
            if bad_dense and index == 0:
                dense = dense.copy()
                dense[2, 0] += 0.25
            energy = np.ascontiguousarray(np.sum(dense * dense, axis=1), dtype=np.float64)
            force = np.ascontiguousarray(-2.0 * dense, dtype=np.float64)
            internal_energy = np.ascontiguousarray(np.sum(selected[1:-1] ** 2, axis=1), dtype=np.float64)
            internal_force = np.ascontiguousarray(-2.0 * selected[1:-1], dtype=np.float64)
            i_perp = self.module.discrete_iperp(selected, internal_force)
            if bad_iperp and index == 0:
                i_perp += 1.0
            hashes = tuple(self.module.sha256_array(value) for value in (selected, dense, energy, force))
            online_energy_sha = self.module.sha256_array(np.concatenate(([0.0], internal_energy, [0.0])))
            online_force_sha = self.module.sha256_array(np.vstack((np.zeros((1, 6)), internal_force, np.zeros((1, 6)))))
            barrier = float((np.max(energy) - energy[0]) * 23.0605)
            shard = directory / f"{tag}.shard{index}.npz"
            np.savez(
                shard, selected_path=selected, selected_path_sha256=np.asarray(hashes[0]),
                dense_coordinates_A=dense, dense_energy_eV=energy, dense_force_eV_per_A=force,
                dense_coordinate_sha256=np.asarray(hashes[1]), dense_energy_sha256=np.asarray(hashes[2]), dense_force_sha256=np.asarray(hashes[3]),
                internal_joint_energy_eV=internal_energy, internal_diagnostic_force_eV_per_A=internal_force,
                discrete_I_perp=np.asarray(i_perp),
                densified_Delta_E_kcal_per_mol=np.asarray(barrier), optimization_N_F_contribution=np.asarray(0),
            )
            shards.append(shard)
            path_sha.append(hashes[0]); dense_sha.append(hashes[1]); energy_sha.append(hashes[2]); force_sha.append(hashes[3]); delta.append(barrier)
            records.append({"path": str(shard.resolve()), "sha256": self.module.sha256_file(shard), "dense_coordinate_sha256": hashes[1], "dense_energy_sha256": hashes[2], "dense_force_sha256": hashes[3]})
            row = {"label": f"{tag}-{index}", "optimization_N_F": (index + 1) * 30, "path_sha256": hashes[0]}
            if method == "IVM":
                row.update({"online_energy_sha256": online_energy_sha, "online_force_sha256": online_force_sha, "online_full_state_source": {"sha256": f"source-{index}"}})
            rows.append(row)
            arrays_by_row.append((selected, internal_energy, internal_force, dense, energy, force, i_perp))
        metric_contract.write_text(json.dumps({
            "schema": "bb-dense-four-ivm-joint-ef-contract-v2" if method == "IVM" else "bb-dense-four-path-joint-ef-contract-v2",
            "frozen": True, "method": method, "jobs": 1, "paths": rows,
            "physical_call_policy": {"dense_joint_EF_per_path": 5, "dense_full_force_storage": True, "optimization_N_F_contribution": 0},
        }), encoding="utf-8")
        result = directory / f"{tag}.npz"
        np.savez(
            result, number_of_points=np.asarray(5), optimization_N_F_contribution=np.asarray(0),
            shard_path=np.asarray([str(path.resolve()) for path in shards]), shard_sha256=np.asarray([self.module.sha256_file(path) for path in shards]),
            selected_path_sha256=np.asarray(path_sha), dense_coordinate_sha256=np.asarray(dense_sha), energy_sha256=np.asarray(energy_sha), dense_force_sha256=np.asarray(force_sha),
            discrete_I_perp=np.asarray([values[6] for values in arrays_by_row]), densified_Delta_E_kcal_per_mol=np.asarray(delta),
        )
        manifest = directory / f"{tag}.manifest.json"
        manifest.write_text(json.dumps({"status": "PASS", "aggregate_sha256": self.module.sha256_file(result), "shards": records}), encoding="utf-8")
        wals = [directory / f"{tag}.{index}.wal.jsonl" for index in range(2)]
        first_events = [{"event": "header"}]
        dense_events = [{"event": "header"}]

        def add_stage(events, row_index, row, stage, coordinates, energies, forces, point_indices):
            events.append({"event": "path_stage_started", "row_index": row_index, "stage": stage})
            session = f"session-{row_index}-{stage}"
            for coordinate, energy_value, force_value, point in zip(coordinates, energies, forces, point_indices):
                common = {
                    "stage": stage, "row_index": row_index, "label": row["label"],
                    "optimization_N_F": row["optimization_N_F"], "path_sha256": row["path_sha256"],
                    "point_index": int(point), "coordinate_sha256": self.module.sha256_array(coordinate),
                    "session_id": session, "attempt_id": f"{session}:{point}",
                }
                events.append({"event": "attempt_started", **common})
                events.append({
                    "event": "attempt_succeeded", **common, "energy_eV": float(energy_value),
                    "energy_scalar_sha256": self.module.sha256_array(np.asarray(float(energy_value), dtype=np.float64)),
                    "force_sha256": self.module.sha256_array(force_value),
                })
            events.append({"event": "path_stage_data_complete_uncommitted", "row_index": row_index, "stage": stage})

        for row_index, (row, values) in enumerate(zip(rows, arrays_by_row)):
            selected, internal_energy, internal_force, dense, energy, force, _ = values
            if method == "ODE-NEB":
                add_stage(first_events, row_index, row, "original_discrete_internal", selected[1:-1], internal_energy, internal_force, range(1, 31))
            else:
                first_events.append({
                    "event": "verified_online_paid_full_state_reused", "row_index": row_index,
                    "path_sha256": row["path_sha256"], "online_energy_sha256": row["online_energy_sha256"],
                    "online_force_sha256": row["online_force_sha256"],
                    "source_sha256": row["online_full_state_source"]["sha256"], "new_physical_calls": 0,
                })
            add_stage(dense_events, row_index, row, "dense_arc500_joint_EF", dense, energy, force, range(5))
        wals[0].write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in first_events), encoding="utf-8")
        wals[1].write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in dense_events), encoding="utf-8")
        if method == "ODE-NEB":
            closure_paths = []
            for row_index, (row, values) in enumerate(zip(rows, arrays_by_row)):
                selected, internal_energy, internal_force, dense, energy, force, _ = values
                closure_paths.append({
                    "label": row["label"], "optimization_N_F": row["optimization_N_F"],
                    "discrete": self.module.validate_stage_wal_closure(first_events, row_index, row, "original_discrete_internal", selected[1:-1], internal_energy, internal_force),
                    "dense": self.module.validate_stage_wal_closure(dense_events, row_index, row, "dense_arc500_joint_EF", dense, energy, force),
                })
            wal_closure = {"status": "PASS", "discrete_lifetime": self.module.wal_lifetime_stats(first_events), "dense_lifetime": self.module.wal_lifetime_stats(dense_events), "paths": closure_paths}
        else:
            online_closure = self.module.validate_online_wal(rows, first_events)
            dense_closure = [{"label": row["label"], **self.module.validate_stage_wal_closure(dense_events, row_index, row, "dense_arc500_joint_EF", values[3], values[4], values[5])} for row_index, (row, values) in enumerate(zip(rows, arrays_by_row))]
            wal_closure = {"status": "PASS", "lifetime": self.module.wal_lifetime_stats(dense_events), "paths": dense_closure}
        if duplicate_terminal:
            duplicate = next(event.copy() for event in dense_events if event.get("event") == "attempt_succeeded")
            dense_events.append(duplicate)
            wals[1].write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in dense_events), encoding="utf-8")
        audit = directory / f"{tag}.audit.json"
        audit_payload = {
            "status": "PASS", "method": method, "output_sha256": self.module.sha256_file(result),
            "contract_sha256": self.module.sha256_file(metric_contract), "optimization_N_F_contribution": 0,
            "individual_incomplete_WAL_attempts_reused": 0, "shard_manifest_sha256": self.module.sha256_file(manifest),
        }
        if method == "ODE-NEB":
            audit_payload.update({"discrete_joint_ef_wal": str(wals[0].resolve()), "discrete_joint_ef_wal_sha256": self.module.sha256_file(wals[0]), "dense_joint_ef_wal": str(wals[1].resolve()), "dense_joint_ef_wal_sha256": self.module.sha256_file(wals[1]), "WAL_shard_pointwise_closure": wal_closure})
        else:
            audit_payload.update({"online_paid_state_wal": str(wals[0].resolve()), "online_paid_state_wal_sha256": self.module.sha256_file(wals[0]), "dense_joint_ef_wal": str(wals[1].resolve()), "dense_joint_ef_wal_sha256": self.module.sha256_file(wals[1]), "online_source_closure": online_closure, "dense_WAL_shard_pointwise_closure": wal_closure})
        audit.write_text(json.dumps(audit_payload), encoding="utf-8")
        closure = {"result": self.bind(result), "audit": self.bind(audit), "contract": self.bind(metric_contract), "manifest": self.bind(manifest), "authoritative_shards": [self.bind(path) for path in shards], "WALs": [self.bind(path) for path in wals]}
        return closure, shards, wals

    def make_figure_contract(self, directory: Path, ivm: dict, ode: dict) -> Path:
        contract = directory / "figure.contract.json"
        contract.write_text(json.dumps({
            "schema": "bb-densified-cost-to-precision-figure-contract-v1", "status": "FROZEN_DENSIFIED_INPUT_BINDINGS",
            "figure_output": str(directory / "figure.pdf"),
            "metric_protocol": {"arc500_stored_arrays_per_path": ["dense_coordinates_A", "dense_energy_eV", "dense_force_eV_per_A"], "authoritative_data_unit": "complete atomic per-path shard", "optimization_N_F_contribution": 0},
            "bindings": {"joint_EF_v2_direct_data_closure": {"IVM": ivm, "ODE-NEB": ode}},
        }), encoding="utf-8")
        return contract

    def test_complete_two_method_closure_passes_without_openmm_and_tamper_fails(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.multiple(self.module, N_DENSE=5, DIMENSION=6):
            directory = Path(temporary)
            ivm, _, _ = self.make_method(directory, "IVM")
            ode, ode_shards, ode_wals = self.make_method(directory, "ODE-NEB")
            contract = self.make_figure_contract(directory, ivm, ode)
            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "openmm" or name.startswith("openmm."):
                    raise AssertionError("preflight imported OpenMM")
                return original_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=guarded_import), redirect_stdout(io.StringIO()) as captured:
                self.module.main(["--contract", str(contract)])
            payload = json.loads(captured.getvalue())
            self.assertEqual(payload["status"], "ZERO_OPENMM_FINAL_FIGURE_CONTRACT_PREFLIGHT_PASS")
            self.assertEqual(payload["physical_calls"], 0)
            original_shard_bytes = ode_shards[0].read_bytes()
            with ode_shards[0].open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                self.module.main(["--contract", str(contract)])
            ode_shards[0].write_bytes(original_shard_bytes)
            contract_payload = json.loads(contract.read_text(encoding="utf-8"))
            with ode_wals[1].open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": "tampered_after_audit"}) + "\n")
            ode["WALs"][1] = self.bind(ode_wals[1])
            contract_payload["bindings"]["joint_EF_v2_direct_data_closure"]["ODE-NEB"] = ode
            contract.write_text(json.dumps(contract_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audit/dense_joint_ef_wal SHA mismatch"):
                self.module.main(["--contract", str(contract)])

    def test_rejects_internally_hash_consistent_wrong_discrete_iperp(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.multiple(self.module, N_DENSE=5, DIMENSION=6):
            directory = Path(temporary)
            ivm, _, _ = self.make_method(directory, "IVM")
            ode, _, _ = self.make_method(directory, "ODE-NEB", bad_iperp=True)
            contract = self.make_figure_contract(directory, ivm, ode)
            with self.assertRaisesRegex(ValueError, "discrete I_perp reconstruction mismatch"):
                self.module.main(["--contract", str(contract)])

    def test_rejects_hash_consistent_non_arc500_dense_coordinates(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.multiple(self.module, N_DENSE=5, DIMENSION=6):
            directory = Path(temporary)
            ivm, _, _ = self.make_method(directory, "IVM")
            ode, _, _ = self.make_method(directory, "ODE-NEB", bad_dense=True)
            contract = self.make_figure_contract(directory, ivm, ode)
            with self.assertRaisesRegex(ValueError, "dense coordinates differ from deterministic Cartesian arc500"):
                self.module.main(["--contract", str(contract)])

    def test_rejects_attempt_id_with_multiple_terminal_events(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.multiple(self.module, N_DENSE=5, DIMENSION=6):
            directory = Path(temporary)
            ivm, _, _ = self.make_method(directory, "IVM")
            ode, _, _ = self.make_method(directory, "ODE-NEB", duplicate_terminal=True)
            contract = self.make_figure_contract(directory, ivm, ode)
            with self.assertRaisesRegex(ValueError, "multiple terminal events"):
                self.module.main(["--contract", str(contract)])


if __name__ == "__main__":
    unittest.main()
