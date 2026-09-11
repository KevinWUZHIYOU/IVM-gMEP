"""Zero-OpenMM tests for the variable-density joint-E/F cost figure."""

from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
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
PLOT_PATH = ROOT / "scripts/plot_bb_cost_to_precision_joint_ef_dense.py"
FREEZER_PATH = ROOT / "scripts/freeze_bb_cost_to_precision_joint_ef_dense_figure_contract.py"
PACK_FREEZER_PATH = ROOT / "scripts/freeze_bb_joint_ef_cost_pack_spec.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PLOT = load("bb_joint_ef_cost_plot_test", PLOT_PATH)
FREEZER = load("bb_joint_ef_cost_freezer_test", FREEZER_PATH)
PACK_FREEZER = load("bb_joint_ef_cost_pack_freezer_test", PACK_FREEZER_PATH)


class JointEFCostPlotTests(unittest.TestCase):
    def small_context(self):
        stack = ExitStack()
        for module in (PLOT, FREEZER.PLOT, PACK_FREEZER.PLOT):
            stack.enter_context(mock.patch.object(module, "N_IMAGES", 4))
            stack.enter_context(mock.patch.object(module, "N_INTERNAL", 2))
            stack.enter_context(mock.patch.object(module, "DIMENSION", 6))
            stack.enter_context(mock.patch.object(module, "N_DENSE", 5))
        return stack

    @staticmethod
    def bind(module, path: Path) -> dict:
        return {"path": str(path.resolve()), "sha256": module.sha256_file(path)}

    def make_pack(
        self,
        directory: Path,
        method: str,
        nfs: tuple[int, ...],
        *,
        bad_dense: bool = False,
        bad_iperp: bool = False,
        bad_ivm_online_history: bool = False,
        bad_optimization_contribution: bool = False,
    ) -> Path:
        module = PLOT
        directory.mkdir(parents=True, exist_ok=True)
        tag = "ivm" if method == "IVM" else "ode"
        evaluator = directory / f"{tag}.evaluator.py"
        topology = directory / "topology.pdb"
        bb_core = directory / "bb_core.py"
        evaluator.write_text("# zero-OpenMM synthetic evaluator\n", encoding="utf-8")
        if not topology.exists():
            topology.write_text("synthetic topology\n", encoding="utf-8")
        if not bb_core.exists():
            bb_core.write_text("# synthetic metric source\n", encoding="utf-8")

        direction = np.linspace(0.5, 1.5, module.DIMENSION)
        selected_paths = []
        metrics = []
        deltas = []
        path_hashes = []
        shard_paths = []
        shard_hashes = []
        original_forces = []
        calls = []
        labels = []
        for index, nf in enumerate(nfs):
            curve = np.empty((module.N_IMAGES, module.DIMENSION), dtype=np.float64)
            for image in range(module.N_IMAGES):
                curve[image] = image * direction + (image * image) * 0.02 * (index + 1)
            dense = module.uniform_arc500(curve)
            if bad_dense and index == 0:
                dense = dense.copy()
                dense[2, 0] += 0.125
            energy = np.ascontiguousarray(
                np.linspace(-20.0, -18.0 + 0.1 * index, module.N_DENSE),
                dtype=np.float64,
            )
            dense_force = np.ascontiguousarray(-0.03 * dense + 0.01, dtype=np.float64)
            internal_force = np.ascontiguousarray(
                np.arange(module.N_INTERNAL * module.DIMENSION, dtype=np.float64).reshape(
                    module.N_INTERNAL, module.DIMENSION
                )
                * (0.001 + index * 0.0001)
                + 0.05,
                dtype=np.float64,
            )
            metric = module.discrete_iperp(curve, internal_force)
            if bad_iperp and index == 0:
                metric += 1.0
            delta = float((np.max(energy) - energy[0]) * module.EV_TO_KCAL)
            label = f"{method}_NF{nf}"
            call = index + 2
            shard = directory / f"{tag}.nf{nf}.npz"
            arrays = {
                "schema": np.asarray(f"synthetic-{tag}-joint-ef-shard-v1"),
                "status": np.asarray("PASS"),
                "method": np.asarray(method),
                "label": np.asarray(label),
                "optimization_N_F": np.asarray(nf, dtype=np.int64),
                "selected_path": curve,
                "selected_path_sha256": np.asarray(module.sha256_array(curve)),
                "discrete_I_perp": np.asarray(metric, dtype=np.float64),
                "dense_coordinates_A": dense,
                "dense_energy_eV": energy,
                "dense_force_eV_per_A": dense_force,
                "dense_coordinate_sha256": np.asarray(module.sha256_array(dense)),
                "dense_energy_sha256": np.asarray(module.sha256_array(energy)),
                "dense_force_sha256": np.asarray(module.sha256_array(dense_force)),
                "densified_Delta_E_kcal_per_mol": np.asarray(delta, dtype=np.float64),
                "number_of_points": np.asarray(module.N_DENSE, dtype=np.int64),
                "optimization_N_F_contribution": np.asarray(
                    1 if bad_optimization_contribution and index == 0 else 0,
                    dtype=np.int64,
                ),
            }
            if method == "ODE-NEB":
                arrays["paid_internal_force_eV_per_A"] = internal_force
            else:
                arrays["discrete_I_perp_source"] = np.asarray(
                    "sealed_verified_short_replay_online_history"
                )
            np.savez(shard, **arrays)
            selected_paths.append(curve)
            original_forces.append(internal_force)
            metrics.append(metric)
            deltas.append(delta)
            path_hashes.append(module.sha256_array(curve))
            shard_paths.append(shard.resolve())
            shard_hashes.append(module.sha256_file(shard))
            calls.append(call)
            labels.append(label)

        replay = directory / f"{tag}.replay.npz"
        verification = directory / f"{tag}.verification.json"
        if method == "IVM":
            np.savez(
                replay,
                stored_path_calls=np.asarray(calls, dtype=np.int64),
                stored_paths=np.asarray(selected_paths, dtype=np.float64),
                calls_per_image=np.asarray(calls, dtype=np.int64),
                total_internal_joint_evaluations=np.asarray(nfs, dtype=np.int64),
                i_perp=np.asarray(metrics, dtype=np.float64)
                + (
                    np.asarray([1.0] + [0.0] * (len(metrics) - 1), dtype=np.float64)
                    if bad_ivm_online_history
                    else 0.0
                ),
            )
            verification.write_text(
                json.dumps(
                    {
                        "schema": "synthetic-ivm-verification-v1",
                        "status": "PASS_EXACT_OUTPUT_ONLY_PREFIX",
                        "new_reporting_rows": [
                            {
                                "call_per_image": call,
                                "history_index": index,
                                "optimization_N_F": int(nf),
                                "selected_path_sha256": path_hashes[index],
                                "discrete_I_perp_eV2_per_A": float(metrics[index]),
                                "fully_evaluated_current_path": True,
                                "terminal_tail": False,
                            }
                            for index, (call, nf) in enumerate(zip(calls, nfs))
                        ],
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

        contract = directory / f"{tag}.contract.json"
        contract_rows = []
        for index, nf in enumerate(nfs):
            row = {
                "label": labels[index],
                "optimization_N_F": int(nf),
                "path_sha256": path_hashes[index],
            }
            if method == "IVM":
                row["online_discrete_I_perp"] = float(metrics[index])
            contract_rows.append(row)
        contract.write_text(
            json.dumps(
                {
                    "schema": f"synthetic-{tag}-contract-v1",
                    "frozen": True,
                    "method": method,
                    "evaluator": self.bind(module, evaluator),
                    "topology": self.bind(module, topology),
                    "bb_core": self.bind(module, bb_core),
                    "paths": contract_rows,
                    "physical_call_policy": {"optimization_N_F_contribution": 0},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        result = directory / f"{tag}.result.npz"
        np.savez(
            result,
            schema=np.asarray(f"synthetic-{tag}-result-v1"),
            status=np.asarray("PASS"),
            method=np.asarray(method),
            optimization_N_F=np.asarray(nfs, dtype=np.int64),
            selected_path_sha256=np.asarray(path_hashes),
            discrete_I_perp=np.asarray(metrics, dtype=np.float64),
            densified_Delta_E_kcal_per_mol=np.asarray(deltas, dtype=np.float64),
            shard_path=np.asarray([str(path) for path in shard_paths]),
            shard_sha256=np.asarray(shard_hashes),
            optimization_N_F_contribution=np.asarray(0, dtype=np.int64),
        )
        manifest = directory / f"{tag}.manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema": f"synthetic-{tag}-manifest-v1",
                    "status": "PASS",
                    "aggregate": str(result.resolve()),
                    "aggregate_sha256": module.sha256_file(result),
                    "contract": str(contract.resolve()),
                    "contract_sha256": module.sha256_file(contract),
                    "optimization_N_F_contribution": 0,
                    "shards": [
                        {
                            "optimization_N_F": int(nf),
                            "path": str(path),
                            "sha256": shard_hashes[index],
                        }
                        for index, (nf, path) in enumerate(zip(nfs, shard_paths))
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        wal = directory / f"{tag}.wal.jsonl"
        wal.write_text(json.dumps({"event": "closed", "optimization_N_F_contribution": 0}) + "\n", encoding="utf-8")
        audit = directory / f"{tag}.audit.json"
        audit.write_text(
            json.dumps(
                {
                    "schema": f"synthetic-{tag}-audit-v1",
                    "status": "PASS",
                    "method": method,
                    "output": self.bind(module, result),
                    "contract": self.bind(module, contract),
                    "manifest": self.bind(module, manifest),
                    "WAL": self.bind(module, wal),
                    "optimization_N_F_contribution": 0,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        spec = directory / f"{tag}.pack.json"
        spec.write_text(
            json.dumps(
                {
                    "schema": module.PACK_SCHEMA,
                    "status": module.PACK_STATUS,
                    "pack_id": f"synthetic-{tag}-pack",
                    "method": method,
                    "include_all_pack_points": True,
                    "pack_spec_freezer": self.bind(module, module.PACK_FREEZER_PATH),
                    "closure": {
                        "result": self.bind(module, result),
                        "audit": self.bind(module, audit),
                        "contract": self.bind(module, contract),
                        "manifest": self.bind(module, manifest),
                        "evaluator": self.bind(module, evaluator),
                        "topology": self.bind(module, topology),
                        "bb_core": self.bind(module, bb_core),
                        "WALs": [self.bind(module, wal)],
                    },
                    "points": [
                        {
                            "label": labels[index],
                            "optimization_N_F": int(nf),
                            "selected_path_sha256": path_hashes[index],
                            "discrete_I_perp": float(metrics[index]),
                            "densified_Delta_E_kcal_per_mol": float(deltas[index]),
                            "authoritative_shard": self.bind(module, shard_paths[index]),
                            "discrete_I_perp_evidence": (
                                {
                                    "kind": "sealed_online_history",
                                    "calls_per_image": int(calls[index]),
                                    "history_index": index,
                                    "replay_result": self.bind(module, replay),
                                    "verification_report": self.bind(module, verification),
                                    "verification_schema": "synthetic-ivm-verification-v1",
                                    "verification_status": "PASS_EXACT_OUTPUT_ONLY_PREFIX",
                                }
                                if method == "IVM"
                                else {
                                    "kind": "recomputed_original_internal_force",
                                    "force_key": "paid_internal_force_eV_per_A",
                                }
                            ),
                        }
                        for index, nf in enumerate(nfs)
                    ],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return spec, shard_paths[0]

    def freeze_contract(self, directory: Path, specs: list[Path]) -> tuple[Path, Path]:
        contract = directory / "figure.contract.json"
        figure = directory / "figure.pdf"
        argv = [
            "--output",
            str(contract),
            "--figure-output",
            str(figure),
            "--minimum-points-per-method",
            "3",
            "--maximum-points-per-method",
            "6",
        ]
        for spec in specs:
            argv.extend(("--pack-spec", str(spec)))
        with redirect_stdout(io.StringIO()):
            FREEZER.main(argv)
        return contract, figure

    def test_synthetic_preview_renders_no_openmm_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "preview.pdf"
            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "openmm" or name.startswith("openmm."):
                    raise AssertionError("plot imported OpenMM")
                return original_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=guarded_import), redirect_stdout(io.StringIO()):
                PLOT.main(["--synthetic-preview", "--output", str(output)])
            self.assertTrue(output.read_bytes().startswith(b"%PDF"))
            provenance = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(provenance["status"], "SYNTHETIC_STYLE_PREVIEW_NOT_SCIENTIFIC")
            self.assertEqual(provenance["point_counts"], {"IVM": 8, "ODE-NEB": 8})
            self.assertEqual(provenance["visible_arrow_text"], "numeric multiplier only")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                PLOT.main(["--synthetic-preview", "--output", str(output)])

    def test_variable_pack_freezer_to_plot_validate_only(self):
        with tempfile.TemporaryDirectory() as temporary, self.small_context():
            directory = Path(temporary)
            ivm, _ = self.make_pack(directory / "ivm", "IVM", (60, 150, 300))
            ode, _ = self.make_pack(directory / "ode", "ODE-NEB", (60, 150, 330))
            contract, figure = self.freeze_contract(directory, [ivm, ode])
            with redirect_stdout(io.StringIO()) as captured:
                PLOT.main(["--contract", str(contract), "--output", str(figure), "--validate-only"])
            provenance = json.loads(captured.getvalue())
            self.assertEqual(provenance["status"], "PASS_AUDITED_JOINT_EF_OBSERVED_ONLY")
            self.assertEqual(provenance["point_counts"], {"IVM": 3, "ODE-NEB": 3})
            self.assertEqual(provenance["diagnostic_joint_EF_optimization_N_F_contribution"], 0)
            self.assertFalse(figure.exists())

    def test_pack_spec_freezer_includes_every_aggregate_point(self):
        with tempfile.TemporaryDirectory() as temporary, self.small_context():
            directory = Path(temporary)
            source_spec, _ = self.make_pack(directory / "source", "ODE-NEB", (60, 150, 330))
            source = json.loads(source_spec.read_text())
            closure = source["closure"]
            output = directory / "rebuilt.pack.json"
            argv = [
                "--output", str(output),
                "--pack-id", "rebuilt-synthetic-ode",
                "--method", "ODE-NEB",
                "--result", closure["result"]["path"],
                "--audit", closure["audit"]["path"],
                "--contract", closure["contract"]["path"],
                "--manifest", closure["manifest"]["path"],
                "--evaluator", closure["evaluator"]["path"],
                "--topology", closure["topology"]["path"],
                "--bb-core", closure["bb_core"]["path"],
            ]
            for wal in closure["WALs"]:
                argv.extend(("--wal", wal["path"]))
            with redirect_stdout(io.StringIO()):
                PACK_FREEZER.main(argv)
            rebuilt = json.loads(output.read_text())
            self.assertTrue(rebuilt["include_all_pack_points"])
            self.assertEqual([row["optimization_N_F"] for row in rebuilt["points"]], [60, 150, 330])
            report, rows = PLOT.validate_pack_spec(output)
            self.assertEqual(report["point_count"], 3)
            self.assertEqual(len(rows), 3)

    def test_authoritative_dense_full_force_tamper_fails(self):
        with tempfile.TemporaryDirectory() as temporary, self.small_context():
            directory = Path(temporary)
            ivm, _ = self.make_pack(directory / "ivm", "IVM", (60, 150, 300))
            ode, shard = self.make_pack(directory / "ode", "ODE-NEB", (60, 150, 330))
            contract, figure = self.freeze_contract(directory, [ivm, ode])
            with shard.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                PLOT.main(["--contract", str(contract), "--output", str(figure), "--validate-only"])

    def test_hash_consistent_non_arc500_dense_coordinates_fail(self):
        with tempfile.TemporaryDirectory() as temporary, self.small_context():
            directory = Path(temporary)
            bad, _ = self.make_pack(directory / "bad", "ODE-NEB", (60, 150, 330), bad_dense=True)
            with self.assertRaisesRegex(ValueError, "differ from deterministic Cartesian arc500"):
                PLOT.validate_pack_spec(bad)

    def test_hash_consistent_wrong_discrete_iperp_fails(self):
        with tempfile.TemporaryDirectory() as temporary, self.small_context():
            directory = Path(temporary)
            bad, _ = self.make_pack(directory / "bad", "ODE-NEB", (60, 150, 330), bad_iperp=True)
            with self.assertRaisesRegex(ValueError, "discrete I_perp reconstruction mismatch"):
                PLOT.validate_pack_spec(bad)

    def test_nonzero_diagnostic_optimization_nf_fails(self):
        with tempfile.TemporaryDirectory() as temporary, self.small_context():
            directory = Path(temporary)
            bad, _ = self.make_pack(
                directory / "bad",
                "ODE-NEB",
                (60, 150, 330),
                bad_optimization_contribution=True,
            )
            with self.assertRaisesRegex(ValueError, "contaminated optimization N_F"):
                PLOT.validate_pack_spec(bad)

    def test_ivm_sealed_online_history_must_match_same_path_metric(self):
        with tempfile.TemporaryDirectory() as temporary, self.small_context():
            directory = Path(temporary)
            bad, _ = self.make_pack(
                directory / "bad",
                "IVM",
                (60, 150, 300),
                bad_ivm_online_history=True,
            )
            with self.assertRaisesRegex(ValueError, "online-history I_perp/N_F closure mismatch"):
                PLOT.validate_pack_spec(bad)


if __name__ == "__main__":
    unittest.main()
