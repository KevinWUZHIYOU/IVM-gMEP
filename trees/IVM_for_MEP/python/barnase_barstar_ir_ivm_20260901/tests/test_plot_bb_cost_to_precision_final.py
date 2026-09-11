from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURE_CONTRACT_TEMPLATE = (
    ROOT
    / "registrations"
    / "BB_IRIVM_COST_TO_PRECISION_FIGURE_CONTRACT_TEMPLATE_20260902_R1.json"
)
sys.path.insert(0, str(ROOT / "scripts"))
import plot_bb_cost_to_precision_final as final_plot  # noqa: E402


def write_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def binding(path):
    return {"path": str(path.resolve()), "sha256": final_plot.sha256_file(path)}


def write_arc500(path, labels, nf, i_perp, delta_e, source_result=None):
    labels = np.asarray(labels)
    count = len(labels)
    selected = labels == final_plot.SELECTED_LABEL
    source_paths = np.full(count, "", dtype="U512")
    source_hashes = np.full(count, "", dtype="U64")
    if source_result is not None:
        source_paths[selected] = str(source_result.resolve())
        source_hashes[selected] = final_plot.sha256_file(source_result)
    np.savez_compressed(
        path,
        label=labels,
        optimization_N_F=np.asarray(nf, dtype=np.int64),
        discrete_I_perp=np.asarray(i_perp, dtype=np.float64),
        densified_Delta_E_kcal_per_mol=np.asarray(delta_e, dtype=np.float64),
        diagnostic_energy_successes_per_path=np.full(count, 500, dtype=np.int64),
        diagnostic_force_requests_per_path=np.zeros(count, dtype=np.int64),
        optimization_N_F_contribution=np.asarray(0, dtype=np.int64),
        number_of_points=np.asarray(500, dtype=np.int64),
        source_result=source_paths,
        source_result_sha256=source_hashes,
    )


def write_ode_bundle(
    path,
    nf,
    i_perp,
    delta_e,
    include_successes=True,
    official_hashes=None,
    target_selection=None,
):
    count = len(nf)
    if official_hashes is None:
        official_hashes = [f"{int(value):064x}" for value in nf]
    arrays = dict(
        optimization_N_F=np.asarray(nf, dtype=np.int64),
        discrete_I_perp=np.asarray(i_perp, dtype=np.float64),
        densified_Delta_E_kcal_per_mol=np.asarray(delta_e, dtype=np.float64),
        diagnostic_energy_attempts_per_path=np.full(count, 500, dtype=np.int64),
        diagnostic_force_requests_per_path=np.zeros(count, dtype=np.int64),
        number_of_points=np.asarray(500, dtype=np.int64),
        official_path32_sha256=np.asarray(official_hashes),
    )
    if include_successes:
        arrays["diagnostic_energy_successes_per_path"] = np.full(
            count, 500, dtype=np.int64
        )
    if target_selection is not None:
        arrays.update(
            coordinate_space=np.asarray("unweighted_3N_cartesian"),
            interpolation=np.asarray("piecewise_linear"),
            sampling=np.asarray("uniform_cartesian_arclength"),
            grid_formula=np.asarray("s_j=j*L/499,j=0,...,499"),
            target_selection=np.asarray(str(target_selection.resolve())),
            target_selection_sha256=np.asarray(final_plot.sha256_file(target_selection)),
            fixed_endpoint_optimization_N_F=np.asarray(0, dtype=np.int64),
        )
    np.savez_compressed(path, **arrays)


class FinalCostToPrecisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.shared_selected_result = self.root / "shared-selected.npz"
        np.savez_compressed(self.shared_selected_result, selected=np.asarray(1))
        self.shared_selected_audit = self.root / "shared-selected.audit.json"
        write_json(self.shared_selected_audit, {"status": "PASS"})
        self.topology = self.root / "complex_fixed.pdb"
        self.topology.write_text("MODEL\nENDMDL\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def selected_bundle(self, labels=None):
        terminal = self.root / "terminal.npz"
        np.savez_compressed(terminal, terminal=np.asarray(1))
        terminal_audit = self.root / "terminal.audit.json"
        write_json(terminal_audit, {"consistent": True})
        long_verification = self.root / "long-verification.json"
        write_json(
            long_verification,
            {
                "schema": "bb-irivm-long-budget-member-verification-v2",
                "status": "PASS",
                "label": final_plot.SELECTED_LABEL,
                "endpoint_optimization_N_F": 0,
                "actual_optimization_N_F": 9980,
                "bindings": {
                    "result": binding(terminal),
                    "audit": binding(terminal_audit),
                },
            },
        )
        prefix = self.root / "prefix.json"
        write_json(
            prefix,
            {
                "schema": "bb-irivm-selected-9990-exact-prefix-verification-v1",
                "status": "PASS_EXACT_PREFIX",
                "tolerance_fallback_used": False,
                "reference_eligible_prefix": [{}, {}, {}],
                "bindings": {
                    "selected_result": binding(terminal),
                    "selected_verification": binding(long_verification),
                },
            },
        )
        arc = self.root / "selected-arc500.npz"
        if labels is None:
            labels = [final_plot.COMMON_LABEL, final_plot.SELECTED_LABEL]
        write_arc500(
            arc,
            labels,
            [30, 1993],
            [1.0e6, 400.0],
            [10000.0, 600.0],
            terminal,
        )
        arc_audit = self.root / "selected-arc500.audit.json"
        write_json(
            arc_audit,
            {
                "schema": "bb-irivm-discrete-iperp-densified-barrier-audit-v1",
                "status": "PASS",
                "output": str(arc.resolve()),
                "output_sha256": final_plot.sha256_file(arc),
                "diagnostic_force_requests": 0,
                "optimization_N_F_contribution": 0,
            },
        )
        return arc, arc_audit, terminal, terminal_audit, long_verification, prefix

    def test_selected_terminal_chain_is_required_and_loaded(self):
        data, provenance = final_plot.load_selected_ivm(*self.selected_bundle())
        self.assertEqual(data["number_of_forces"].tolist(), [30, 1993])
        self.assertEqual(provenance["selected_label"], final_plot.SELECTED_LABEL)
        self.assertEqual(provenance["exact_prefix_state_count"], 3)

    def test_candidate_b_or_preview_label_is_rejected(self):
        paths = self.selected_bundle(
            [final_plot.COMMON_LABEL, "bb_irivm_a1e0_r01_m05_inf4080_r1"]
        )
        with self.assertRaisesRegex(ValueError, "candidate-B/preview"):
            final_plot.load_selected_ivm(*paths)

    def ode_bundle(self, name, schema, nf, include_successes=True):
        result = self.root / f"{name}.npz"
        write_ode_bundle(
            result,
            nf,
            np.linspace(1.0e6, 600.0, len(nf)),
            np.linspace(10000.0, 900.0, len(nf)),
            include_successes=include_successes,
        )
        audit = self.root / f"{name}.audit.json"
        write_json(
            audit,
            {
                "schema": schema,
                "status": "PASS",
                "output": str(result.resolve()),
                "output_sha256": final_plot.sha256_file(result),
                "diagnostic_force_requests": 0,
                "optimization_N_F_contribution": 0,
                "diagnostic_energy_attempts": 500 * len(nf),
                "diagnostic_energy_successes": 500 * len(nf),
            },
        )
        return result, audit

    def completed_record(self, result, audit):
        with np.load(result, allow_pickle=False) as saved:
            nfs = np.asarray(saved["optimization_N_F"], dtype=np.int64)
            hashes = np.asarray(saved["official_path32_sha256"]).astype(str)
        return {
            "result": str(result.resolve()),
            "result_sha256": final_plot.sha256_file(result),
            "audit": str(audit.resolve()),
            "audit_sha256": final_plot.sha256_file(audit),
            "optimization_N_F": [int(value) for value in nfs],
            "official_path32_sha256": [str(value) for value in hashes],
        }

    def paid_states(self, target_map):
        return [
            {
                "paid_batch_id": index,
                "optimization_N_F": 30 * index,
                "disposition": "current_initial" if index == 1 else "accepted",
                "ase_current_residual_eV_per_A": float(334 - index),
                "source_batch_id": index,
                "coordinate_archive": f"/sealed/batch_{index:04d}.npz",
                "coordinate_archive_sha256": f"{1000 + index:064x}",
                "official_path32_sha256": target_map.get(30 * index, f"{index:064x}"),
                "discrete_I_perp_eV2_per_A": None,
                "densified_Delta_E_kcal_per_mol": None,
            }
            for index in range(1, 334)
        ]

    def targeted_round(
        self,
        name,
        completed_pairs,
        target_nf,
        target_hash,
        round_index,
        paid_states,
        i_perp=450.0,
        delta_e=610.0,
        selected_pair=None,
    ):
        if selected_pair is None:
            selected_pair = (self.shared_selected_result, self.shared_selected_audit)
        source = self.root / f"{name}.paid-force.npz"
        np.savez_compressed(source, forces=np.asarray([1.0]))
        source_binding = {
            **binding(source),
            "optimization_N_F": target_nf,
            "paid_batch_id": target_nf // 30,
            "official_path32_sha256": target_hash,
            "discrete_I_perp_eV2_per_A": i_perp,
        }
        selection = self.root / f"{name}.selection.json"
        completed_records = [self.completed_record(*pair) for pair in completed_pairs]
        selection_payload = {
            "schema": final_plot.TARGETED_SELECTION_SCHEMA,
            "status": "FROZEN_ZERO_OPENMM_TARGET_SELECTION",
            "claim_mode": "observed",
            "physical_calls_made_by_selector": {"energy": 0, "force": 0},
            "optimization_N_F_contribution": 0,
            "paths_to_densify": [
                {
                    "optimization_N_F_first_occurrence": target_nf,
                    "official_path32_sha256": target_hash,
                    "exact_discrete_I_perp_eV2_per_A": i_perp,
                }
            ],
            "official_paid_states": json.loads(json.dumps(paid_states)),
            "claim_guard": {
                "Delta_E_monotonicity_assumed": False,
                "unmeasured_earlier_paths_forbid_exact_first_passage_claim": True,
            },
            "bindings": {
                "selected_IR_IVM_arc500": {
                    "result": str(selected_pair[0].resolve()),
                    "result_sha256": final_plot.sha256_file(selected_pair[0]),
                    "audit": str(selected_pair[1].resolve()),
                    "audit_sha256": final_plot.sha256_file(selected_pair[1]),
                },
                "paid_force_receipts": [source_binding],
                "completed_ODE_NEB_arc500": completed_records,
            },
        }
        write_json(selection, selection_payload)

        result = self.root / f"{name}.npz"
        contract = self.root / f"{name}.execution.json"
        contract_payload = {
            "schema": final_plot.TARGETED_EXECUTION_SCHEMA,
            "status": "AUTHORIZED_AFTER_FROZEN_TARGET_SELECTION",
            "physical_execution_authorized": True,
            "target_selection": binding(selection),
            "topology": binding(self.topology),
            "evaluator": binding(ROOT / "scripts" / "evaluate_targeted_ode_neb_arc500.py"),
            "implementation_bindings": {
                "target_selector": binding(ROOT / "scripts" / "freeze_targeted_ode_neb_arc500.py"),
                "common_arc500_evaluator": binding(ROOT / "scripts" / "evaluate_discrete_iperp_densified_barrier.py"),
                "bb_core": binding(ROOT / "src" / "bb_core.py"),
            },
            "output": str(result.resolve()),
            "execution_protocol": {
                "claim_mode": "observed",
                "evaluate_only_paths_to_densify": True,
                "distinct_retained_paths_only": True,
                "energy_evaluations_per_path": 500,
                "force_evaluations_per_path": 0,
                "optimization_N_F_contribution": 0,
                "fixed_endpoint_optimization_N_F": 0,
                "allow_exact_energy_reuse": False,
                "no_overwrite": True,
                "append_only_energy_WAL": True,
                "durable_attempt_started_before_each_physical_call": True,
                "fsync_each_attempt_start_and_terminal_record": True,
            },
            "lock_policy": {
                "jobs_per_host": 1,
                "acquire_both_before_first_physical_call": True,
                "hold_both_through_last_physical_call": True,
                "host_local_BB_locks": list(final_plot.CANONICAL_BB_LOCKS),
            },
        }
        write_json(contract, contract_payload)
        energies = np.linspace(-2.0, 3.0, 500, dtype=np.float64)
        selected_hash = f"{target_nf + 1:064x}"
        dense_hash = f"{target_nf + 2:064x}"
        energy_hash = final_plot.sha256_array(energies)
        wal = self.root / f"{name}.energy.wal.jsonl"
        wal_events = [
            {
                "schema": final_plot.TARGETED_WAL_SCHEMA,
                "number_of_points_per_path": 500,
                "force_requests": 0,
                "optimization_N_F_contribution": 0,
                "exact_path_reuse_enabled": False,
                "target_selection": str(selection.resolve()),
                "target_selection_sha256": final_plot.sha256_file(selection),
                "execution_contract": str(contract.resolve()),
                "execution_contract_sha256": final_plot.sha256_file(contract),
                "topology": str(self.topology.resolve()),
                "topology_sha256": final_plot.sha256_file(self.topology),
                "target_manifest": [{
                    "row_index": 0,
                    "optimization_N_F": target_nf,
                    "official_path32_sha256": target_hash,
                    "selected_path_sha256": selected_hash,
                    "dense_coordinate_sha256": dense_hash,
                }],
                "durable_attempt_started_before_each_physical_call": True,
                "fsync_each_attempt_start_and_terminal_record": True,
            },
            {
                "event": "path_start", "row_index": 0,
                "optimization_N_F": target_nf,
                "selected_path_sha256": selected_hash,
                "official_path32_sha256": target_hash,
                "dense_coordinate_sha256": dense_hash,
            },
        ]
        for point, energy in enumerate(energies):
            attempt = point + 1
            wal_events.extend((
                {
                    "event": "attempt_started", "row_index": 0,
                    "point_index": point, "attempt": attempt,
                    "selected_path_sha256": selected_hash,
                    "dense_coordinate_sha256": dense_hash,
                },
                {
                    "row_index": 0, "point_index": point, "attempt": attempt,
                    "status": "success", "energy_eV": float(energy),
                },
            ))
        wal.write_text("".join(json.dumps(event, sort_keys=True) + "\n" for event in wal_events), encoding="utf-8")
        np.savez_compressed(
            result,
            optimization_N_F=np.asarray([target_nf], dtype=np.int64),
            discrete_I_perp=np.asarray([i_perp]),
            densified_Delta_E_kcal_per_mol=np.asarray([delta_e]),
            official_path32_sha256=np.asarray([target_hash]),
            selected_path_sha256=np.asarray([selected_hash]),
            dense_coordinate_sha256=np.asarray([dense_hash]),
            energy_sha256=np.asarray([energy_hash]),
            energy_eV=energies[None],
            diagnostic_energy_attempts_per_path=np.asarray([500]),
            diagnostic_energy_successes_per_path=np.asarray([500]),
            diagnostic_energy_attempts_this_evaluation_per_path=np.asarray([500]),
            diagnostic_energy_reused_per_path=np.asarray([False]),
            diagnostic_force_requests_per_path=np.asarray([0]),
            exact_discrete_I_perp_source_kind=np.asarray(["paid_force_receipt"]),
            exact_discrete_I_perp_source=np.asarray([str(source.resolve())]),
            exact_discrete_I_perp_source_sha256=np.asarray([final_plot.sha256_file(source)]),
            exact_discrete_I_perp_source_optimization_N_F=np.asarray([target_nf]),
            number_of_points=np.asarray(500),
            coordinate_space=np.asarray("unweighted_3N_cartesian"),
            interpolation=np.asarray("piecewise_linear"),
            sampling=np.asarray("uniform_cartesian_arclength"),
            grid_formula=np.asarray("s_j=j*L/499,j=0,...,499"),
            optimization_N_F_contribution=np.asarray(0),
            fixed_endpoint_optimization_N_F=np.asarray(0),
            target_selection=np.asarray(str(selection.resolve())),
            target_selection_sha256=np.asarray(final_plot.sha256_file(selection)),
            execution_contract=np.asarray(str(contract.resolve())),
            execution_contract_sha256=np.asarray(final_plot.sha256_file(contract)),
            topology=np.asarray(str(self.topology.resolve())),
            topology_sha256=np.asarray(final_plot.sha256_file(self.topology)),
            energy_wal=np.asarray(str(wal.resolve())),
            energy_wal_sha256=np.asarray(final_plot.sha256_file(wal)),
            energy_wal_terminal_validation_schema=np.asarray(final_plot.TARGETED_WAL_VALIDATION_SCHEMA),
            energy_wal_exact_path_reuse_events=np.asarray(0),
            target_round_index=np.asarray(round_index),
            prior_targeted_bundle_count=np.asarray(round_index - 1),
        )
        audit = self.root / f"{name}.audit.json"
        audit_record = {
            "optimization_N_F": target_nf,
            "official_path32_sha256": target_hash,
            "selected_path_sha256": selected_hash,
            "dense_coordinate_sha256": dense_hash,
            "energy_sha256": energy_hash,
            "discrete_I_perp": i_perp,
            "densified_Delta_E_kcal_per_mol": delta_e,
            "diagnostic_energy_attempts": 500,
            "diagnostic_energy_successes": 500,
            "diagnostic_force_requests": 0,
            "exact_discrete_I_perp_source_kind": "paid_force_receipt",
            "exact_discrete_I_perp_source": str(source.resolve()),
            "exact_discrete_I_perp_source_sha256": final_plot.sha256_file(source),
            "exact_discrete_I_perp_source_optimization_N_F": target_nf,
        }
        lineage = [
            {
                "audit_schema": json.loads(
                    Path(record["audit"]).read_text(encoding="utf-8")
                )["schema"],
                **record,
            }
            for record in completed_records
        ]
        write_json(audit, {
            "schema": final_plot.TARGETED_ODE_AUDIT_SCHEMA,
            "status": "PASS", "output": str(result.resolve()),
            "output_sha256": final_plot.sha256_file(result),
            "target_selection": str(selection.resolve()),
            "target_selection_sha256": final_plot.sha256_file(selection),
            "execution_contract": str(contract.resolve()),
            "execution_contract_sha256": final_plot.sha256_file(contract),
            "energy_wal": str(wal.resolve()), "energy_wal_sha256": final_plot.sha256_file(wal),
            "energy_wal_terminal_validation": {
                "schema": final_plot.TARGETED_WAL_VALIDATION_SCHEMA, "status": "PASS",
                "header_and_target_manifest_bound_before_first_physical_call": True,
                "attempt_started_fsynced_before_each_physical_call": True,
                "terminal_record_fsynced_after_each_success": True,
                "exact_path_energy_reuse_events": 0, "failed_energy_events": 0,
                "successful_energy_events": 500, "line_count": 1002,
                "sha256": final_plot.sha256_file(wal),
            },
            "paid_N_F_state_count_verified": 333,
            "distinct_retained_path_count_verified": 284,
            "selected_distinct_retained_path_count": 1,
            "target_round_index": round_index,
            "prior_targeted_bundle_count": round_index - 1,
            "completed_prior_arc500_lineage": lineage,
            "fresh_diagnostic_energy_attempts": 500,
            "fresh_diagnostic_energy_successes": 500,
            "lineage_energy_evaluations": 500,
            "diagnostic_force_requests": 0, "optimization_N_F_contribution": 0,
            "fixed_endpoint_optimization_N_F": 0,
            "every_included_path_has_exactly_500_successful_energy_evaluations": True,
            "every_included_path_was_freshly_evaluated_once": True,
            "exact_path_energy_reuse_count": 0,
            "current_round_allow_exact_energy_reuse": False,
            "every_discrete_I_perp_is_bound_to_an_exact_paid_force_receipt": True,
            "records": [audit_record],
        })
        return selection, result, audit

    def test_official_old_bundle_and_nf4080_merge(self):
        old = self.ode_bundle(
            "old",
            "bb-ode-neb-arc500-preview-audit-v1",
            [240, 480, 990, 9990],
            include_successes=False,
        )
        extra = self.ode_bundle(
            "nf4080",
            "bb-ode-neb-nf4080-arc500-audit-v1",
            [4080],
        )
        data, bindings = final_plot.load_official_ode(
            [old[0], extra[0]], [old[1], extra[1]]
        )
        self.assertEqual(data["number_of_forces"].tolist(), [240, 480, 990, 4080, 9990])
        self.assertEqual(len(bindings), 2)

    def test_nonofficial_ode_milestone_is_rejected(self):
        bad = self.ode_bundle(
            "bad",
            "bb-ode-neb-arc500-preview-audit-v1",
            [240, 480, 990, 5000],
        )
        with self.assertRaisesRegex(ValueError, "nonofficial or missing"):
            final_plot.load_official_ode([bad[0]], [bad[1]])

    def test_old_bundles_and_frozen_targeted_result_merge(self):
        old = self.ode_bundle(
            "old-target",
            "bb-ode-neb-arc500-preview-audit-v1",
            [240, 480, 990, 9990],
            include_successes=False,
        )
        extra = self.ode_bundle(
            "nf4080-target",
            "bb-ode-neb-nf4080-arc500-audit-v1",
            [4080],
        )
        target_hash = f"{8160:064x}"
        states = self.paid_states({8160: target_hash})
        targeted = self.targeted_round(
            "round1", [old, extra], 8160, target_hash, 1, states
        )
        data, bindings = final_plot.load_official_ode(
            [old[0], extra[0]],
            [old[1], extra[1]],
            targeted_bundles=[targeted],
        )
        self.assertEqual(
            data["number_of_forces"].tolist(), [240, 480, 990, 4080, 8160, 9990]
        )
        self.assertEqual(len(data["official_paid_states"]), 333)
        self.assertEqual(bindings[-1]["target_round_index"], 1)

    def two_round_fixture(self, second_states=None):
        old = self.ode_bundle(
            "chain-old", "bb-ode-neb-arc500-preview-audit-v1",
            [240, 480, 990, 9990], include_successes=False,
        )
        extra = self.ode_bundle(
            "chain-4080", "bb-ode-neb-nf4080-arc500-audit-v1", [4080]
        )
        hash1, hash2 = f"{6000:064x}", f"{8160:064x}"
        states = self.paid_states({6000: hash1, 8160: hash2})
        round1 = self.targeted_round(
            "chain-round1", [old, extra], 6000, hash1, 1, states,
            i_perp=800.0, delta_e=700.0,
        )
        round2 = self.targeted_round(
            "chain-round2", [old, extra, (round1[1], round1[2])],
            8160, hash2, 2, states if second_states is None else second_states,
            i_perp=400.0, delta_e=600.0,
        )
        return old, extra, round1, round2, states

    def test_two_targeted_rounds_merge_in_chronological_chain(self):
        old, extra, round1, round2, _states = self.two_round_fixture()
        data, bindings = final_plot.load_official_ode(
            [old[0], extra[0]], [old[1], extra[1]],
            targeted_bundles=[round1, round2],
        )
        self.assertEqual(
            data["number_of_forces"].tolist(),
            [240, 480, 990, 4080, 6000, 8160, 9990],
        )
        targeted = [row for row in bindings if row.get("target_round_index")]
        self.assertEqual([row["target_round_index"] for row in targeted], [1, 2])

    def test_targeted_selection_audit_mismatch_is_rejected(self):
        old, extra, round1, round2, _states = self.two_round_fixture()
        with self.assertRaisesRegex(ValueError, "audit/selection mismatch"):
            final_plot.load_official_ode(
                [old[0], extra[0]], [old[1], extra[1]],
                targeted_bundles=[(round2[0], round1[1], round1[2])],
            )

    def test_targeted_iperp_and_paid_force_source_tamper_are_rejected(self):
        old, extra, round1, _round2, _states = self.two_round_fixture()
        with np.load(round1[1], allow_pickle=False) as saved:
            arrays = {key: np.asarray(saved[key]).copy() for key in saved.files}
        arrays["discrete_I_perp"][0] += 1.0
        np.savez_compressed(round1[1], **arrays)
        audit = json.loads(round1[2].read_text(encoding="utf-8"))
        audit["output_sha256"] = final_plot.sha256_file(round1[1])
        write_json(round1[2], audit)
        with self.assertRaisesRegex(ValueError, "I_perp mismatch"):
            final_plot.load_official_ode(
                [old[0], extra[0]], [old[1], extra[1]], targeted_bundles=[round1]
            )

        old, extra, round1, _round2, _states = self.two_round_fixture()
        selection = json.loads(round1[0].read_text(encoding="utf-8"))
        source = Path(selection["bindings"]["paid_force_receipts"][0]["path"])
        source.write_bytes(source.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "paid-force receipt"):
            final_plot.load_official_ode(
                [old[0], extra[0]], [old[1], extra[1]], targeted_bundles=[round1]
            )

    def test_targeted_wal_contract_fresh_and_no_reuse_tamper_are_rejected(self):
        for mode, pattern in (
            ("wal", "WAL binding"),
            ("contract", "execution protocol mismatch"),
            ("fresh", "fresh-call"),
            ("reuse", "reuse or incomplete"),
        ):
            with self.subTest(mode=mode):
                old, extra, round1, _round2, _states = self.two_round_fixture()
                audit = json.loads(round1[2].read_text(encoding="utf-8"))
                if mode == "wal":
                    wal = Path(audit["energy_wal"])
                    wal.write_bytes(wal.read_bytes() + b"{}\n")
                elif mode == "fresh":
                    audit["fresh_diagnostic_energy_attempts"] = 499
                    write_json(round1[2], audit)
                else:
                    with np.load(round1[1], allow_pickle=False) as saved:
                        arrays = {key: np.asarray(saved[key]).copy() for key in saved.files}
                    if mode == "reuse":
                        arrays["diagnostic_energy_reused_per_path"][0] = True
                    else:
                        contract_path = Path(audit["execution_contract"])
                        contract = json.loads(contract_path.read_text(encoding="utf-8"))
                        contract["execution_protocol"]["allow_exact_energy_reuse"] = True
                        write_json(contract_path, contract)
                        arrays["execution_contract_sha256"] = np.asarray(
                            final_plot.sha256_file(contract_path)
                        )
                        audit["execution_contract_sha256"] = final_plot.sha256_file(contract_path)
                    np.savez_compressed(round1[1], **arrays)
                    audit["output_sha256"] = final_plot.sha256_file(round1[1])
                    write_json(round1[2], audit)
                with self.assertRaisesRegex(ValueError, pattern):
                    final_plot.load_official_ode(
                        [old[0], extra[0]], [old[1], extra[1]],
                        targeted_bundles=[round1],
                    )

    def test_targeted_chain_omission_and_full_ledger_mismatch_are_rejected(self):
        old, extra, round1, _round2, states = self.two_round_fixture()
        hash2 = f"{8160:064x}"
        omitted = self.targeted_round(
            "omitted-round2", [old, extra], 8160, hash2, 2, states,
        )
        with self.assertRaisesRegex(ValueError, "omits an earlier targeted round"):
            final_plot.load_official_ode(
                [old[0], extra[0]], [old[1], extra[1]],
                targeted_bundles=[round1, omitted],
            )

        changed_states = json.loads(json.dumps(states))
        changed_states[10]["source_batch_id"] += 1
        mismatched = self.targeted_round(
            "mismatched-round2", [old, extra, (round1[1], round1[2])],
            8160, hash2, 2, changed_states,
        )
        with self.assertRaisesRegex(ValueError, "same full 333-state"):
            final_plot.load_official_ode(
                [old[0], extra[0]], [old[1], extra[1]],
                targeted_bundles=[round1, mismatched],
            )

    def test_targeted_result_without_selection_is_rejected(self):
        old = self.ode_bundle(
            "old-no-selection",
            "bb-ode-neb-arc500-preview-audit-v1",
            [240, 480, 990, 9990],
            include_successes=False,
        )
        target_hash = f"{8160:064x}"
        targeted = self.targeted_round(
            "round-no-selection", [old], 8160, target_hash, 1,
            self.paid_states({8160: target_hash}),
        )
        with self.assertRaisesRegex(ValueError, "requires --ode-target-selection"):
            final_plot.load_official_ode(
                [old[0], targeted[1]], [old[1], targeted[2]]
            )

    def test_target_selection_must_bind_final_selected_ivm_pair(self):
        old = self.ode_bundle(
            "old-selected-binding",
            "bb-ode-neb-arc500-preview-audit-v1",
            [240, 480, 990, 9990],
            include_successes=False,
        )
        selected_result = self.shared_selected_result
        selected_audit = self.shared_selected_audit
        target_hash = f"{8160:064x}"
        selection, _result, _audit = self.targeted_round(
            "selected-binding", [old], 8160, target_hash, 1,
            self.paid_states({8160: target_hash}),
            selected_pair=(selected_result, selected_audit),
        )
        final_plot.validate_selection_selected_ivm(
            selection, selected_result, selected_audit
        )
        other = self.root / "candidate-b.npz"
        np.savez_compressed(other, value=np.asarray(2))
        with self.assertRaisesRegex(ValueError, "final selected IR-IVM result"):
            final_plot.validate_selection_selected_ivm(
                selection, other, selected_audit
            )

    def test_final_figure_contract_binds_every_input_and_plotter(self):
        first = self.root / "first.npz"
        second = self.root / "second.json"
        np.savez_compressed(first, value=np.asarray(1))
        write_json(second, {"status": "PASS"})
        output = self.root / "figure.pdf"
        contract = self.root / "figure-contract.json"
        write_json(
            contract,
            {
                "schema": "bb-irivm-cost-to-precision-figure-contract-v1",
                "status": "FROZEN_FINAL_INPUT_BINDINGS",
                "figure_output": str(output.resolve()),
                "claim_policy": {
                    "Delta_E_monotonicity_assumed": False,
                    "allowed_sparse_claim": "earliest observed first passage",
                },
                "bindings": [
                    binding(first),
                    binding(second),
                    binding(Path(final_plot.__file__)),
                ],
            },
        )
        receipt = final_plot.validate_figure_contract(
            contract, output.resolve(), [first, second]
        )
        self.assertEqual(receipt["status"], "FROZEN_FINAL_INPUT_BINDINGS")
        unbound = self.root / "unbound.json"
        write_json(unbound, {"status": "PASS"})
        with self.assertRaisesRegex(ValueError, "does not bind required input"):
            final_plot.validate_figure_contract(
                contract, output.resolve(), [first, unbound]
            )

    def test_inert_template_rejects_and_fully_substituted_copy_passes(self):
        template = json.loads(FIGURE_CONTRACT_TEMPLATE.read_text(encoding="utf-8"))
        self.assertTrue(template["status"].startswith("INERT_TEMPLATE"))
        self.assertEqual(
            template["claim_policy"]["allowed_sparse_claim"],
            "earliest observed first passage",
        )
        self.assertFalse(template["claim_policy"]["Delta_E_monotonicity_assumed"])
        output = self.root / "activated-figure.pdf"
        with self.assertRaisesRegex(ValueError, "not activated and frozen"):
            final_plot.validate_figure_contract(
                FIGURE_CONTRACT_TEMPLATE, output.resolve(), []
            )

        activated = json.loads(json.dumps(template))
        activated["status"] = "FROZEN_FINAL_INPUT_BINDINGS"
        activated["figure_output"] = str(output.resolve())
        substituted_inputs = []
        counter = 0

        def substitute(value):
            nonlocal counter
            if isinstance(value, dict):
                path_value = value.get("path")
                hash_value = value.get("sha256")
                if (
                    isinstance(path_value, str)
                    and path_value.startswith("<")
                    and isinstance(hash_value, str)
                    and hash_value.startswith("<")
                ):
                    counter += 1
                    material = self.root / f"terminal-input-{counter:02d}.dat"
                    material.write_bytes(f"bound-input-{counter}\n".encode("ascii"))
                    value["path"] = str(material.resolve())
                    value["sha256"] = final_plot.sha256_file(material)
                    substituted_inputs.append(material)
                for child in value.values():
                    substitute(child)
            elif isinstance(value, list):
                for child in value:
                    substitute(child)

        substitute(activated)
        activated_path = self.root / "activated-contract.json"
        write_json(activated_path, activated)
        receipt = final_plot.validate_figure_contract(
            activated_path, output.resolve(), substituted_inputs
        )
        self.assertEqual(receipt["status"], "FROZEN_FINAL_INPUT_BINDINGS")
        self.assertEqual(len(substituted_inputs), 9)

    def test_plotter_has_single_panel_letter_pass_and_well_formed_guards(self):
        source = Path(final_plot.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count("figure.text("), 1)
        tree = ast.parse(source)
        require_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "require"
        ]
        self.assertTrue(require_calls)
        self.assertTrue(all(len(node.args) == 2 for node in require_calls))
        main_source = inspect.getsource(final_plot.main)
        self.assertIn('"--ode-target-selection"', main_source)
        self.assertIn('"--ode-targeted-bundle"', main_source)
        self.assertIn('"--figure-contract"', main_source)
        self.assertIn('output.with_suffix(".json")', main_source)
        self.assertIn('"command_argv"', main_source)

    def test_analysis_records_raw_points_envelopes_passage_and_ratio(self):
        ivm = {
            "number_of_forces": np.asarray([30, 100, 200]),
            "i_perp": np.asarray([100.0, 20.0, 5.0]),
            "delta_e": np.asarray([1000.0, 300.0, 100.0]),
        }
        ode = {
            "number_of_forces": np.asarray([30, 300, 600]),
            "i_perp": np.asarray([100.0, 20.0, 5.0]),
            "delta_e": np.asarray([1000.0, 300.0, 100.0]),
        }
        report = final_plot.analyze(ivm, ode)
        self.assertEqual(report["first_passage_number_of_forces"]["i_perp"], {"IVM": 200, "ODE-NEB": 600})
        self.assertEqual(report["ODE_NEB_over_IVM"]["i_perp"], 3.0)
        self.assertEqual(len(report["raw_points"]["IVM"]), 3)
        self.assertEqual(len(report["lower_envelopes"]["delta_e"]["ODE-NEB"]), 3)
        self.assertEqual(
            report["ODE_NEB_passage_claim"]["delta_e"]["claim"],
            "sampled_first_passage_only",
        )

    def test_nonmonotone_delta_e_is_only_earliest_observed_with_interval(self):
        ivm = {
            "number_of_forces": np.asarray([30, 50]),
            "i_perp": np.asarray([8.0, 2.0]),
            "delta_e": np.asarray([9.0, 3.0]),
        }
        ode = {
            "number_of_forces": np.asarray([30, 90]),
            "i_perp": np.asarray([9.0, 1.0]),
            "delta_e": np.asarray([10.0, 2.0]),
            "official_path32_sha256": np.asarray(["a", "c"]),
            "official_paid_states": [
                {"optimization_N_F": 30, "official_path32_sha256": "a"},
                {"optimization_N_F": 60, "official_path32_sha256": "b"},
                {"optimization_N_F": 90, "official_path32_sha256": "c"},
            ],
        }
        report = final_plot.analyze(ivm, ode)
        evidence = report["ODE_NEB_passage_claim"]["delta_e"]
        self.assertEqual(evidence["claim"], "earliest_observed_first_passage")
        self.assertEqual(evidence["possible_exact_first_passage_N_F_interval"], [60, 90])
        self.assertFalse(report["Delta_E_monotonicity_assumed"])


if __name__ == "__main__":
    unittest.main()
