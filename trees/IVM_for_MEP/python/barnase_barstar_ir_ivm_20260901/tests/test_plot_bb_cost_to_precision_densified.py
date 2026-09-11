"""Zero-OpenMM synthetic validation of both dense-four plotting bundles."""

import hashlib
import importlib.util
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("plot_bb_dense_test", ROOT / "scripts" / "plot_bb_cost_to_precision_densified.py")
PLOT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PLOT)
FREEZE_SPEC = importlib.util.spec_from_file_location("freeze_bb_dense_figure_test", ROOT / "scripts" / "freeze_bb_densified_cost_figure_contract.py")
FREEZER = importlib.util.module_from_spec(FREEZE_SPEC)
assert FREEZE_SPEC.loader is not None
FREEZE_SPEC.loader.exec_module(FREEZER)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path: Path):
    return {"path": str(path.resolve()), "sha256": sha_file(path)}


def write_wal(path: Path, events):
    path.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in events), encoding="utf-8")


def make_bundle(root: Path, method: str, ode_contract: Path = None):
    nf = np.asarray(PLOT.EXPECTED_NF[method], dtype=np.int64)
    calls = (2, 5, 10, 50)
    paths = np.empty((4, 32, 9396), dtype=np.float64)
    base = np.arange(32, dtype=np.float64)[:, None]
    for index in range(4):
        paths[index] = base + index * 1e-6
    source_records = []
    official = []
    if method == "ODE-NEB" and ode_contract is not None:
        frozen = json.loads(ode_contract.read_text(encoding="utf-8"))
        for index, record in enumerate(frozen["paths"]):
            archive = Path(record["source"]["path"])
            with np.load(archive, allow_pickle=False) as saved:
                paths[index] = np.concatenate((saved["endpoint0_A"][None], saved["coordinates30_A"], saved["endpoint1_A"][None]), axis=0).reshape(32, 9396)
            source_records.append(bind(archive))
            official.append(str(record["official_path32_sha256"]))
    path_hashes = np.asarray([PLOT.FINAL.sha256_array(row) for row in paths])
    labels = np.asarray([f"{method}_{index}" for index in range(4)])
    identifiers = [({"calls_per_image": calls[index]} if method == "IVM" else {"optimization_N_F": int(nf[index]), "paid_batch_id": index + 1}) for index in range(4)]
    topology = ROOT / "inputs" / "complex_fixed.pdb"
    bb_core = ROOT / "src" / "bb_core.py"
    evaluator = ROOT / "scripts" / "evaluate_bb_dense_four_paths.py"
    if method == "ODE-NEB":
        if ode_contract is None:
            for index, path in enumerate(paths):
                path3 = path.reshape(32, 3132, 3)
                official_hash = PLOT.official_path32_sha256(path3)
                archive = root / f"source_{index}.npz"
                np.savez_compressed(archive, endpoint0_A=path3[0], coordinates30_A=path3[1:-1], endpoint1_A=path3[-1], path32_sha256=np.asarray(official_hash))
                source_records.append(bind(archive))
                official.append(official_hash)
    else:
        replay = root / "replay.npz"
        np.savez_compressed(replay, stored_paths=paths)
        source_records = [bind(replay) for _ in range(4)]
    forces = np.full((4, 30, 9396), np.nan, dtype=np.float64)
    joint = np.full((4, 30), np.nan, dtype=np.float64)
    if method == "ODE-NEB":
        forces = np.zeros((4, 30, 9396), dtype=np.float64)
        forces[:, :, 0] = np.arange(1, 31, dtype=np.float64)[None, :]
        joint = np.arange(120, dtype=np.float64).reshape(4, 30)
        i_perp = np.asarray([PLOT.path_diagnostics(paths[index], np.zeros(32), forces[index])["i_perp"] for index in range(4)])
        force_hashes = np.asarray([PLOT.FINAL.sha256_array(row) for row in forces])
        joint_hashes = np.asarray([PLOT.FINAL.sha256_array(row) for row in joint])
    else:
        i_perp = np.asarray([6.8e7, 2.9e7, 8.9e5, 952.0])
        force_hashes = np.asarray(["", "", "", ""])
        joint_hashes = np.asarray(["", "", "", ""])
    dense_paths = [PLOT.uniform_arclength_resample(path, 500)[0] for path in paths]
    dense_hashes = np.asarray([PLOT.FINAL.sha256_array(row) for row in dense_paths])
    energies = np.asarray([np.linspace(-100.0, -90.0 + index, 500) for index in range(4)])
    energy_hashes = np.asarray([PLOT.FINAL.sha256_array(row) for row in energies])
    delta = (np.max(energies, axis=1) - energies[:, 0]) * 23.0605
    verified_rows = [{"call_per_image": calls[index], "optimization_N_F": int(nf[index]), "selected_path_sha256": path_hashes[index], "discrete_I_perp_eV2_per_A": float(i_perp[index]), "fully_evaluated_current_path": True, "terminal_tail": False} for index in range(4)]
    verification = root / "verification.json"
    verification.write_text(json.dumps({"status": "PASS_EXACT_OUTPUT_ONLY_PREFIX", "new_reporting_rows": verified_rows}) + "\n", encoding="utf-8")
    contract_paths = []
    for index in range(4):
        record = {"method": method, "label": labels[index], "optimization_N_F": int(nf[index]), "path_identifier": identifiers[index], "path_sha256": path_hashes[index], "source": source_records[index]}
        if method == "ODE-NEB":
            record["official_path32_sha256"] = official[index]
        contract_paths.append(record)
    contract = root / "contract.json"
    contract_payload = {"topology": bind(topology), "bb_core": bind(bb_core), "paths": contract_paths}
    if method == "ODE-NEB":
        contract_payload["evaluator"] = bind(evaluator)
    else:
        contract_payload["evaluator_engine"] = bind(evaluator)
        contract_payload["replay_verification"] = bind(verification)
    contract.write_text(json.dumps(contract_payload) + "\n", encoding="utf-8")
    force_wal = root / "force.wal.jsonl"
    force_events = [{"schema": "bb-dense-four-path-force-wal-v1", "contract": str(contract), "contract_sha256": sha_file(contract), "optimization_N_F_contribution": 0}]
    attempt = 0
    if method == "ODE-NEB":
        for row in range(4):
            for image in range(1, 31):
                attempt += 1
                common = {"row_index": row, "internal_image_index": image, "attempt": attempt, "path_sha256": path_hashes[row], "coordinate_sha256": PLOT.FINAL.sha256_array(paths[row, image])}
                force_events.extend(({"event": "attempt_started", **common}, {"event": "attempt_succeeded", **common, "energy_eV": float(joint[row, image - 1]), "force_sha256": PLOT.FINAL.sha256_array(forces[row, image - 1])}))
    else:
        force_events.extend({"event": "online_paid_history_reuse", "row_index": row, "path_sha256": path_hashes[row], "diagnostic_force_attempts": 0} for row in range(4))
    write_wal(force_wal, force_events)
    energy_wal = root / "energy.wal.jsonl"
    energy_events = [{"schema": "bb-dense-four-path-energy-wal-v1", "contract": str(contract), "contract_sha256": sha_file(contract), "optimization_N_F_contribution": 0}]
    attempt = 0
    for row in range(4):
        for point in range(500):
            attempt += 1
            common = {"row_index": row, "point_index": point, "attempt": attempt, "path_sha256": path_hashes[row], "coordinate_sha256": PLOT.FINAL.sha256_array(dense_paths[row][point])}
            energy_events.extend(({"event": "attempt_started", **common}, {"event": "attempt_succeeded", **common, "energy_eV": float(energies[row, point])}))
    write_wal(energy_wal, energy_events)
    result = root / "result.npz"
    np.savez_compressed(
        result, method=np.asarray(method), labels=labels,
        path_identifier_json=np.asarray([json.dumps(value, sort_keys=True) for value in identifiers]),
        optimization_N_F=nf, selected_path=paths, selected_path_sha256=path_hashes,
        discrete_I_perp=i_perp, densified_Delta_E_kcal_per_mol=delta,
        internal_diagnostic_force_eV_per_A=forces, internal_diagnostic_force_sha256=force_hashes,
        internal_joint_energy_eV=joint, internal_joint_energy_sha256=joint_hashes,
        diagnostic_force_attempts_per_path=np.full(4, 30 if method == "ODE-NEB" else 0),
        fixed_endpoint_diagnostic_force_attempts_per_path=np.zeros(4, dtype=np.int64),
        dense_coordinate_sha256=dense_hashes, energy_eV=energies, energy_sha256=energy_hashes,
        diagnostic_energy_attempts_lineage_per_path=np.full(4, 500), diagnostic_energy_attempts_fresh_per_path=np.full(4, 500),
        number_of_points=np.asarray(500), coordinate_space=np.asarray("unweighted_3N_cartesian"), sampling=np.asarray("uniform_cartesian_arclength"), interpolation=np.asarray("piecewise_linear"), grid_formula=np.asarray("s_j=j*L/499,j=0,...,499"),
        optimization_N_F_contribution=np.asarray(0), contract_sha256=np.asarray(sha_file(contract)),
        topology_sha256=np.asarray(sha_file(topology)), bb_core_sha256=np.asarray(sha_file(bb_core)), evaluator_sha256=np.asarray(sha_file(evaluator)),
    )
    records = []
    for index in range(4):
        records.append({"optimization_N_F": int(nf[index]), "path_sha256": path_hashes[index], "dense_coordinate_sha256": dense_hashes[index], "energy_sha256": energy_hashes[index], "discrete_I_perp": float(i_perp[index]), "densified_Delta_E_kcal_per_mol": float(delta[index]), "diagnostic_force_attempts": 30 if method == "ODE-NEB" else 0, "diagnostic_force_successes": 30 if method == "ODE-NEB" else 0, "fixed_endpoint_diagnostic_force_attempts": 0, "force_sha256": force_hashes[index], "joint_energy_sha256": joint_hashes[index], "diagnostic_energy_attempts_lineage": 500, "diagnostic_energy_attempts_fresh": 500, "diagnostic_energy_attempts_reused": 0, "optimization_N_F_contribution": 0})
    audit = root / "audit.json"
    audit.write_text(json.dumps({"schema": "bb-dense-four-path-dual-metric-audit-v1", "status": "PASS", "method": method, "output": str(result), "output_sha256": sha_file(result), "contract": str(contract), "contract_sha256": sha_file(contract), "path_count": 4, "optimization_N_F_contribution": 0, "force_wal": str(force_wal), "force_wal_sha256": sha_file(force_wal), "energy_wal": str(energy_wal), "energy_wal_sha256": sha_file(energy_wal), "paths": records}) + "\n", encoding="utf-8")
    return result, audit, contract, official


class DensePlotInputTests(unittest.TestCase):
    def test_ivm_bundle_validates_without_openmm(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, audit, contract, _official = make_bundle(Path(temporary), "IVM")
            data, provenance = PLOT.load_dense_four(result, audit, contract, "IVM")
            self.assertEqual(data["number_of_forces"].tolist(), [60, 150, 300, 1329])
            self.assertEqual(provenance["fresh_force_attempts"], 0)
            self.assertEqual(provenance["fresh_energy_attempts"], 2000)

    def test_ode_official_hashes_are_bound_and_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, audit, contract, official = make_bundle(Path(temporary), "ODE-NEB")
            data, provenance = PLOT.load_dense_four(result, audit, contract, "ODE-NEB")
            self.assertEqual(data["official_path32_sha256"].tolist(), official)
            self.assertEqual(provenance["fresh_force_attempts"], 120)

    def test_analysis_never_promotes_ratio_to_exact(self):
        ivm = {"number_of_forces": np.asarray([30, 60]), "i_perp": np.asarray([10.0, 1.0]), "delta_e": np.asarray([10.0, 1.0])}
        ode = {"number_of_forces": np.asarray([30, 90]), "i_perp": np.asarray([10.0, 1.0]), "delta_e": np.asarray([10.0, 1.0])}
        analysis = PLOT.analyze_observed_only(ivm, ode)
        self.assertEqual(analysis["all_cost_ratios"], "observed only")
        self.assertTrue(all(value["claim"] == "earliest_observed_first_passage" for value in analysis["ODE_NEB_passage_claim"].values()))

    def test_freezer_to_plot_validate_only_end_to_end(self):
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            ivm_dir = temporary_root / "ivm"
            ode_dir = temporary_root / "ode"
            ivm_dir.mkdir()
            ode_dir.mkdir()
            formal_ode_contract = ROOT / "registrations" / "diagnostics" / "BB_DENSE_COST_ODE_FOUR_20260902_R1.json"
            ivm_result, ivm_audit, ivm_contract, _ = make_bundle(ivm_dir, "IVM")
            ode_result, ode_audit, ode_contract, _ = make_bundle(ode_dir, "ODE-NEB", formal_ode_contract)
            old = {
                "ivm_result": ROOT / "runs/diagnostics/bb_irivm_selected_9990_arc500_20260902_r1.npz",
                "ivm_audit": ROOT / "runs/diagnostics/bb_irivm_selected_9990_arc500_20260902_r1.audit.json",
                "ivm_terminal_result": ROOT / "runs/production/ivm_bb_irivm_a1e0_r01_m05_inf9990_r1_central_state_ivm_cfge0354bb7178e_budget9990_storagecommitted.npz",
                "ivm_terminal_audit": ROOT / "runs/production/ivm_bb_irivm_a1e0_r01_m05_inf9990_r1_central_state_ivm_cfge0354bb7178e_budget9990_storagecommitted.audit.json",
                "ivm_long_verification": ROOT / "reports/verifications/BB_IRIVM_A1E0_R01_M05_INF9990_R1_VERIFICATION_20260902.json",
                "ivm_prefix_verification": ROOT / "reports/verifications/BB_IRIVM_SELECTED_9990_EXACT_4080_PREFIX_20260902_R1.json",
            }
            ode_pairs = [
                (ROOT / "runs/diagnostics/bb_ode_neb_arc500_preview_20260901_r1.npz", ROOT / "runs/diagnostics/bb_ode_neb_arc500_preview_20260901_r1.audit.json"),
                (ROOT / "runs/diagnostics/bb_ode_neb_nf4080_arc500_20260902_r1.npz", ROOT / "runs/diagnostics/bb_ode_neb_nf4080_arc500_20260902_r1.audit.json"),
            ]
            targeted = [
                (ROOT / "registrations/diagnostics/BB_ODE_NEB_TARGETED_ARC500_SELECTION_20260902_R1.json", ROOT / "runs/diagnostics/bb_ode_neb_targeted_nf8160_arc500_20260902_r1.npz", ROOT / "runs/diagnostics/bb_ode_neb_targeted_nf8160_arc500_20260902_r1.audit.json"),
                (ROOT / "registrations/diagnostics/BB_ODE_NEB_TARGETED_ARC500_SELECTION_20260902_R2.json", ROOT / "runs/diagnostics/bb_ode_neb_targeted_nf2040_arc500_20260902_r2a2.npz", ROOT / "runs/diagnostics/bb_ode_neb_targeted_nf2040_arc500_20260902_r2a2.audit.json"),
            ]
            timing = ROOT / "registrations/diagnostics/BB_IRIVM_ODE_NEB_COST_DENSIFICATION_SELECTION_20260902_R2.json"
            figure_output = temporary_root / "barnase_barstar_ivm_vs_ode_neb_cost_to_precision_densified.pdf"
            figure_contract = temporary_root / "figure_contract.json"
            freeze_argv = ["freeze", "--output", str(figure_contract), "--figure-output", str(figure_output)]
            for name, path in old.items():
                freeze_argv.extend(("--" + name.replace("_", "-"), str(path)))
            freeze_argv.extend(("--ivm-dense-result", str(ivm_result), "--ivm-dense-audit", str(ivm_audit), "--ivm-dense-contract", str(ivm_contract)))
            for result, audit in ode_pairs:
                freeze_argv.extend(("--ode-result", str(result), "--ode-audit", str(audit)))
            for selection, result, audit in targeted:
                freeze_argv.extend(("--ode-targeted-bundle", str(selection), str(result), str(audit)))
            freeze_argv.extend(("--ode-dense-result", str(ode_result), "--ode-dense-audit", str(ode_audit), "--ode-dense-contract", str(ode_contract), "--timing-correction", str(timing)))
            with mock.patch.object(sys, "argv", freeze_argv), redirect_stdout(io.StringIO()):
                FREEZER.main()
            plot_argv = []
            for name, path in old.items():
                plot_argv.extend(("--" + name.replace("_", "-"), str(path)))
            plot_argv.extend(("--ivm-dense-result", str(ivm_result), "--ivm-dense-audit", str(ivm_audit), "--ivm-dense-contract", str(ivm_contract)))
            for result, audit in ode_pairs:
                plot_argv.extend(("--ode-result", str(result), "--ode-audit", str(audit)))
            for selection, result, audit in targeted:
                plot_argv.extend(("--ode-targeted-bundle", str(selection), str(result), str(audit)))
            plot_argv.extend(("--ode-dense-result", str(ode_result), "--ode-dense-audit", str(ode_audit), "--ode-dense-contract", str(ode_contract), "--timing-correction", str(timing), "--figure-contract", str(figure_contract), "--output", str(figure_output), "--validate-only"))
            with redirect_stdout(io.StringIO()) as captured:
                PLOT.main(plot_argv)
            provenance = json.loads(captured.getvalue())
            self.assertEqual(provenance["all_cost_ratios"], "observed only")
            self.assertEqual(provenance["NF30_common_initial_reuse"]["status"], "PASS_BYTE_IDENTICAL_COMMON_INITIAL_RECEIPT_AND_PROTOCOL_REUSE")
            self.assertFalse(figure_output.exists())


if __name__ == "__main__":
    unittest.main()
