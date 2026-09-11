from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts/plot_bb_irivm_nf8982_surface_evolution.py"


def load_plotter():
    spec = importlib.util.spec_from_file_location("bb_surface_evolution_tested", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestSurfaceEvolutionPlotter(unittest.TestCase):
    def test_real_verified_stored_paths_project_with_registered_nof_and_fit_grid(self):
        plotter = load_plotter()
        coordinate = ROOT / "runs/diagnostics/bb_irivm_nf8982_0k_section_coordinates_59x28_20260902_r2.npz"
        with np.load(coordinate, allow_pickle=False) as saved:
            surface = {
                "center": np.asarray(saved["center_path59_A"], dtype=float),
                "s": np.asarray(saved["center_s"], dtype=float),
                "normals": np.asarray(saved["normal_directions"], dtype=float),
                "n": np.asarray(saved["normal_coordinate_rms_A"], dtype=float),
                "initial_projected": np.column_stack((saved["center_s"], saved["common_initial_projected_n_A"])),
                "path_source_contract": Path(str(saved["base_evolution_contract"])).resolve(),
                "path_source_contract_sha256": str(saved["base_evolution_contract_sha256"]),
            }
        helper = ROOT / "scripts/plot_bb_irivm_evolution_hashbound.py"
        source = ROOT / "registrations/BB_IRIVM_EVOLUTION_FIGURE_CONTRACT_20260902_R1.json"
        contract = {
            "path_loader": plotter.file_binding(helper),
            "path_source_contract": plotter.file_binding(source),
            "panels": [
                {"ivm_optimization_N_F": ivm, "ode_neb_optimization_N_F": ode}
                for ivm, ode in plotter.EXPECTED_PANELS
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as stream:
            json.dump(contract, stream)
            contract_path = Path(stream.name)
        try:
            initial, panels, _ = plotter._load_paths(contract_path, contract, surface)
        finally:
            contract_path.unlink(missing_ok=True)
        self.assertEqual(initial.shape, (59, 2))
        self.assertEqual(
            [(row["ivm_optimization_N_F"], row["ode_neb_optimization_N_F"]) for row in panels],
            list(plotter.EXPECTED_PANELS),
        )
        for row in panels:
            for key in ("ivm_projected", "ode_neb_projected"):
                projection = row[key]
                self.assertGreaterEqual(float(projection[:, 1].min()), -0.12 - 2e-8)
                self.assertLessEqual(float(projection[:, 1].max()), 0.69 + 2e-8)
        self.assertAlmostEqual(float(initial[:, 1].max()), 0.6237545129305405, places=12)
        self.assertAlmostEqual(float(panels[0]["ivm_projected"][:, 1].max()), 0.5517741495459979, places=12)
        self.assertAlmostEqual(float(panels[0]["ode_neb_projected"][:, 1].max()), 0.6216242721925916, places=12)

    def test_synthetic_pdf_and_provenance_are_zero_physical_and_no_overwrite(self):
        plotter = load_plotter()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "bb_synthetic_preview.pdf"
            with mock.patch.object(plotter, "import_path", side_effect=AssertionError("scientific loader called")):
                plotter.main(["--synthetic-dry-run", "--output", str(output)])
            self.assertTrue(output.read_bytes().startswith(b"%PDF-"))
            provenance_path = output.with_suffix(".json")
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(provenance["status"], "PASS_SYNTHETIC_ZERO_PHYSICAL_DRY_RUN_NOT_SCIENTIFIC_DATA")
            self.assertTrue(provenance["summary"]["no_openmm_context_created"])
            self.assertEqual(provenance["summary"]["surface_shape"], [59, 28])
            self.assertEqual(
                [(row["ivm_optimization_N_F"], row["ode_neb_optimization_N_F"]) for row in provenance["summary"]["panels"]],
                list(plotter.EXPECTED_PANELS),
            )
            self.assertEqual(provenance["output_pdf"]["sha256"], plotter.sha256_file(output))
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                plotter.main(["--synthetic-dry-run", "--output", str(output)])

    def test_synthetic_validate_only_creates_nothing(self):
        plotter = load_plotter()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "synthetic_validate.pdf"
            plotter.main(["--synthetic-dry-run", "--validate-only", "--output", str(output)])
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".json").exists())

    def test_norm_round_trip_and_synthetic_inventory(self):
        plotter = load_plotter()
        norm = plotter.LogQuadraticNorm(0.0, 1.0e5, scale=40.0, curvature=2.5)
        values = np.asarray([0.0, 40.0, 240.0, 1e3, 1e4, 1e5])
        mapped = np.asarray(norm(values))
        self.assertTrue(np.all(np.diff(mapped) > 0.0))
        self.assertTrue(np.allclose(norm.inverse(mapped), values, rtol=1e-12, atol=1e-10))
        surface, initial, panels = plotter.synthetic_inputs()
        self.assertEqual(surface["energy_kcal"].shape, (59, 28))
        self.assertEqual(initial.shape, (59, 2))
        self.assertEqual(len(panels), 3)
        self.assertTrue(np.all(np.isfinite(surface["energy_kcal"])))

    def test_nonfinal_contract_fails_before_scientific_loading(self):
        plotter = load_plotter()
        with tempfile.TemporaryDirectory() as temporary:
            contract = Path(temporary) / "contract.json"
            contract.write_text(json.dumps({"schema": plotter.FIGURE_SCHEMA, "status": "TEMPLATE_NOT_FROZEN"}), encoding="utf-8")
            output = Path(temporary) / "candidate.pdf"
            with mock.patch.object(plotter, "_load_surface", side_effect=AssertionError("surface loaded")):
                with self.assertRaisesRegex(RuntimeError, "not final"):
                    plotter.main(["--contract", str(contract), "--output", str(output), "--validate-only"])


if __name__ == "__main__":
    unittest.main()
