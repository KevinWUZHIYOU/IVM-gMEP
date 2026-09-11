from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_targeted_ode_neb_arc500 as evaluator  # noqa: E402
import plot_bb_cost_to_precision_final as final_plot  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def binding(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": evaluator.sha256_file(path)}


def pair_binding(result: Path, audit: Path) -> dict:
    return {
        "result": str(result.resolve()),
        "result_sha256": evaluator.sha256_file(result),
        "audit": str(audit.resolve()),
        "audit_sha256": evaluator.sha256_file(audit),
    }


def write_synthetic_terminal_wal(state: dict, evaluated: list[dict]) -> None:
    header = {
        "schema": "bb-densified-energy-wal-v1",
        "number_of_points_per_path": 500,
        "force_requests": 0,
        "optimization_N_F_contribution": 0,
        "exact_path_reuse_enabled": False,
        **evaluator.wal_context(state),
    }
    lines = [header]
    attempt = 0
    for row_index, row in enumerate(evaluated):
        lines.append(
            {
                "event": "path_start",
                "row_index": row_index,
                "optimization_N_F": int(row["optimization_N_F"]),
                "selected_path_sha256": str(row["path_sha256"]),
                "official_path32_sha256": str(row["official_path32_sha256"]),
                "dense_coordinate_sha256": str(row["dense_coordinate_sha256"]),
            }
        )
        for point_index, energy in enumerate(row["energy_eV"]):
            attempt += 1
            lines.append(
                {
                    "event": "attempt_started",
                    "row_index": row_index,
                    "point_index": point_index,
                    "attempt": attempt,
                    "selected_path_sha256": str(row["path_sha256"]),
                    "dense_coordinate_sha256": str(
                        row["dense_coordinate_sha256"]
                    ),
                }
            )
            lines.append(
                {
                    "row_index": row_index,
                    "point_index": point_index,
                    "attempt": attempt,
                    "status": "success",
                    "energy_eV": float(energy),
                }
            )
    state["wal_output"].write_text(
        "".join(json.dumps(line, sort_keys=True) + "\n" for line in lines),
        encoding="utf-8",
    )


class TargetedEvaluatorFixture:
    def __init__(self, root: Path):
        self.root = root
        self.output = root / "targeted.npz"
        self.lock1 = root / "canonical.bb.lock"
        self.lock2 = root / "downloads.bb.lock"
        self.topology = root / "complex_fixed.pdb"
        self.topology.write_text("synthetic topology\n", encoding="utf-8")

        # One real, finite, positive-segment 32-image archive suffices for the
        # selected path.  Other synthetic ledger identities are never used as
        # coordinate sources because the authoritative reconstruction is
        # mocked only at its already-validated return boundary.
        atoms = evaluator.DIMENSION // 3
        endpoint0 = np.zeros((atoms, 3), dtype=np.float64)
        endpoint1 = np.zeros((atoms, 3), dtype=np.float64)
        endpoint1[0, 0] = 31.0
        coordinates = np.zeros((30, atoms, 3), dtype=np.float64)
        coordinates[:, 0, 0] = np.arange(1, 31, dtype=np.float64)
        path3 = np.concatenate(
            (endpoint0[None], coordinates, endpoint1[None]), axis=0
        )
        self.target_hash = evaluator.target_freezer.official_array_sha256(path3)
        self.common_path = np.ascontiguousarray(
            path3.reshape(evaluator.N_IMAGES, evaluator.DIMENSION)
        )
        self.common_hash = evaluator.sha256_array(self.common_path)
        self.common_archive = root / "common_initial_path32.npz"
        np.savez_compressed(self.common_archive, initial_path=self.common_path)
        self.archive = root / "batch_0200_paid_coordinates30_A.npz"
        np.savez_compressed(
            self.archive,
            coordinates30_A=coordinates,
            endpoint0_A=endpoint0,
            endpoint1_A=endpoint1,
            path32_sha256=np.asarray(self.target_hash),
        )
        self.archive_sha = evaluator.sha256_file(self.archive)
        self.target_nf = 6000
        self.exact_source = {
            self.target_hash: {
                "kind": "paid_force_receipt",
                "path": str(self.archive.resolve()),
                "sha256": self.archive_sha,
                "optimization_N_F": self.target_nf,
            }
        }
        endpoint1_b = np.zeros((atoms, 3), dtype=np.float64)
        endpoint1_b[0, 1] = 31.0
        coordinates_b = np.zeros((30, atoms, 3), dtype=np.float64)
        coordinates_b[:, 0, 1] = np.arange(1, 31, dtype=np.float64)
        path3_b = np.concatenate(
            (endpoint0[None], coordinates_b, endpoint1_b[None]), axis=0
        )
        self.target2_hash = evaluator.target_freezer.official_array_sha256(path3_b)
        self.target2_archive = root / "batch_0201_paid_coordinates30_A.npz"
        np.savez_compressed(
            self.target2_archive,
            coordinates30_A=coordinates_b,
            endpoint0_A=endpoint0,
            endpoint1_A=endpoint1_b,
            path32_sha256=np.asarray(self.target2_hash),
        )
        self.target2_archive_sha = evaluator.sha256_file(self.target2_archive)
        self.target2_nf = 6030
        self.exact_source2 = {
            "kind": "paid_force_receipt",
            "path": str(self.target2_archive.resolve()),
            "sha256": self.target2_archive_sha,
            "optimization_N_F": self.target2_nf,
        }

        self.paid_rows = []
        unique_hashes = [
            hashlib.sha256(f"synthetic-retained-{index}".encode()).hexdigest()
            for index in range(1, 285)
        ]
        unique_hashes[199] = self.target_hash
        unique_hashes[200] = self.target2_hash
        for batch_id in range(1, 334):
            if batch_id <= 284:
                path_hash = unique_hashes[batch_id - 1]
                source_batch = batch_id
                disposition = "current_initial" if batch_id == 1 else "accepted"
            else:
                path_hash = unique_hashes[-1]
                source_batch = 284
                disposition = "rejected"
            row_archive = self.archive
            row_archive_sha = self.archive_sha
            if batch_id == 201:
                row_archive = self.target2_archive
                row_archive_sha = self.target2_archive_sha
            self.paid_rows.append(
                {
                    "paid_batch_id": batch_id,
                    "optimization_N_F": batch_id * 30,
                    "disposition": disposition,
                    "ase_current_residual_eV_per_A": float(1000 - batch_id),
                    "source_batch_id": source_batch,
                    "coordinate_archive": str(row_archive.resolve()),
                    "coordinate_archive_sha256": row_archive_sha,
                    "official_path32_sha256": path_hash,
                }
            )
        self.target_source = self.paid_rows[199]
        self.target2_source = self.paid_rows[200]

        self.complete = root / "OPTIMIZATION_COMPLETE.json"
        self.optimization_audit = root / "OPTIMIZATION_AUDIT.json"
        self.optimization_wal = root / "optimization_force_wal.jsonl"
        self.endpoints = root / "frozen_endpoints_A.npy"
        write_json(self.complete, {"synthetic": True})
        write_json(self.optimization_audit, {"status": "PASS"})
        self.optimization_wal.write_text("{}\n", encoding="utf-8")
        np.save(self.endpoints, np.asarray([endpoint0, endpoint1]))
        self.official_bindings = {
            "optimization_complete": binding(self.complete),
            "optimization_audit": binding(self.optimization_audit),
            "optimization_wal": binding(self.optimization_wal),
            "frozen_endpoints": binding(self.endpoints),
        }

        self.terminal = root / "selected-terminal.npz"
        np.savez_compressed(self.terminal, terminal=np.asarray(1))
        self.terminal_audit = root / "selected-terminal.audit.json"
        write_json(self.terminal_audit, {"consistent": True})
        self.long_verification = root / "selected-long-verification.json"
        write_json(
            self.long_verification,
            {
                "schema": "bb-irivm-long-budget-member-verification-v2",
                "status": "PASS",
                "label": evaluator.SELECTED_IVM_LABEL,
                "endpoint_optimization_N_F": 0,
                "bindings": {
                    "result": binding(self.terminal),
                    "audit": binding(self.terminal_audit),
                },
            },
        )
        self.prefix_verification = root / "selected-prefix-verification.json"
        write_json(
            self.prefix_verification,
            {
                "schema": "bb-irivm-selected-9990-exact-prefix-verification-v1",
                "status": "PASS_EXACT_PREFIX",
                "tolerance_fallback_used": False,
                "bindings": {
                    "selected_result": binding(self.terminal),
                    "selected_verification": binding(self.long_verification),
                },
            },
        )
        self.metric_contract = root / "selected-metric-contract.json"
        write_json(
            self.metric_contract,
            {
                "schema": evaluator.SELECTED_IVM_CONTRACT_SCHEMA,
                "contract_id": "bb_irivm_selected_9990_arc500_20260902_r1",
                "status": "FROZEN_AFTER_TERMINAL_AND_EXACT_PREFIX_VERIFICATION_BEFORE_ANY_SELECTED_9990_DIAGNOSTIC_CALL",
                "runs": [
                    {
                        "label": evaluator.SELECTED_IVM_LABEL,
                        "registered_optimization_N_F": 9990,
                        "source_result": str(self.terminal.resolve()),
                        "source_result_sha256": evaluator.sha256_file(self.terminal),
                        "source_audit": str(self.terminal_audit.resolve()),
                        "source_audit_sha256": evaluator.sha256_file(
                            self.terminal_audit
                        ),
                        "terminal_verification": str(
                            self.long_verification.resolve()
                        ),
                        "terminal_verification_sha256": evaluator.sha256_file(
                            self.long_verification
                        ),
                        "exact_prefix_verification": str(
                            self.prefix_verification.resolve()
                        ),
                        "exact_prefix_verification_sha256": evaluator.sha256_file(
                            self.prefix_verification
                        ),
                        "eligible_committed_calls": [2],
                        "eligible_states_at_freeze": [
                            {
                                "history_index": 1,
                                "calls_per_image": 2,
                                "optimization_N_F": 9950,
                                "discrete_I_perp": 400.0,
                                "path_sha256": "a" * 64,
                            }
                        ],
                    }
                ],
                "bindings": {
                    "selected_long_member_verification": binding(
                        self.long_verification
                    ),
                    "exact_prefix_verification": binding(
                        self.prefix_verification
                    ),
                },
                "common_initial_path": {
                    "path": str(self.common_archive.resolve()),
                    "file_sha256": evaluator.sha256_file(self.common_archive),
                    "array_sha256": self.common_hash,
                },
                "eligible_path_policy": "post-initial reparameterization-committed stored paths only",
                "exact_reuse_preflight": {
                    "unique_selected_path_count_including_common_initial": 2,
                    "exact_reused_unique_path_count": 1,
                    "fresh_unique_path_count": 1,
                    "diagnostic_force_calls": 0,
                    "optimization_N_F_contribution": 0,
                },
            },
        )
        self.selected_result = root / "selected-arc500.npz"
        np.savez_compressed(
            self.selected_result,
            label=np.asarray(
                [evaluator.COMMON_IVM_LABEL, evaluator.SELECTED_IVM_LABEL]
            ),
            history_index=np.asarray([0, 1], dtype=np.int64),
            calls_per_image=np.asarray([1, 2], dtype=np.int64),
            optimization_N_F=np.asarray([30, 9950], dtype=np.int64),
            discrete_I_perp=np.asarray([1.0e6, 400.0]),
            densified_Delta_E_kcal_per_mol=np.asarray([1.0e4, 600.0]),
            selected_path_sha256=np.asarray([self.common_hash, "a" * 64]),
            source_result=np.asarray(["", str(self.terminal.resolve())]),
            source_result_sha256=np.asarray(
                ["", evaluator.sha256_file(self.terminal)]
            ),
            diagnostic_energy_attempts_per_path=np.asarray([500, 500], dtype=np.int64),
            diagnostic_energy_successes_per_path=np.asarray([500, 500], dtype=np.int64),
            diagnostic_force_requests_per_path=np.asarray([0, 0], dtype=np.int64),
            optimization_N_F_contribution=np.asarray(0, dtype=np.int64),
            number_of_points=np.asarray(500, dtype=np.int64),
            coordinate_space=np.asarray(evaluator.COORDINATE_SPACE),
            interpolation=np.asarray(evaluator.INTERPOLATION),
            sampling=np.asarray(evaluator.SAMPLING),
            grid_formula=np.asarray(evaluator.GRID_FORMULA),
            contract=np.asarray(str(self.metric_contract.resolve())),
            contract_sha256=np.asarray(evaluator.sha256_file(self.metric_contract)),
        )
        self.selected_audit = root / "selected-arc500.audit.json"
        write_json(
            self.selected_audit,
            {
                "schema": evaluator.SELECTED_IVM_AUDIT_SCHEMA,
                "status": "PASS",
                "output": str(self.selected_result.resolve()),
                "output_sha256": evaluator.sha256_file(self.selected_result),
                "contract": str(self.metric_contract.resolve()),
                "contract_sha256": evaluator.sha256_file(self.metric_contract),
                "diagnostic_force_requests": 0,
                "optimization_N_F_contribution": 0,
                "every_path_is_bound_to_exactly_500_successful_energy_evaluations": True,
                "selected_path_count_including_common_initial": 2,
                "diagnostic_energy_attempts": 1000,
                "diagnostic_energy_successes": 1000,
                "records": [
                    {
                        "label": evaluator.COMMON_IVM_LABEL,
                        "history_index": 0,
                        "optimization_N_F": 30,
                        "discrete_I_perp": 1.0e6,
                        "selected_path_sha256": self.common_hash,
                        "diagnostic_energy_attempts": 500,
                        "diagnostic_force_requests": 0,
                    },
                    {
                        "label": evaluator.SELECTED_IVM_LABEL,
                        "history_index": 1,
                        "optimization_N_F": 9950,
                        "discrete_I_perp": 400.0,
                        "selected_path_sha256": "a" * 64,
                        "diagnostic_energy_attempts": 500,
                        "diagnostic_force_requests": 0,
                    },
                ],
            },
        )
        self.thresholds, _ = evaluator.target_freezer.load_selected_thresholds(
            self.selected_result, self.selected_audit
        )

        self.core_result = root / "official-core.npz"
        core_nf = np.asarray([240, 480, 990, 9990], dtype=np.int64)
        core_hashes = np.asarray(
            [self.paid_rows[int(value // 30) - 1]["official_path32_sha256"] for value in core_nf]
        )
        np.savez_compressed(
            self.core_result,
            optimization_N_F=core_nf,
            discrete_I_perp=np.asarray([3e7, 3e6, 1e5, 600.0]),
            densified_Delta_E_kcal_per_mol=np.asarray([2e4, 1e4, 2e3, 900.0]),
            diagnostic_energy_attempts_per_path=np.full(4, 500, dtype=np.int64),
            diagnostic_force_requests_per_path=np.zeros(4, dtype=np.int64),
            number_of_points=np.asarray(500, dtype=np.int64),
            official_path32_sha256=core_hashes,
            topology=np.asarray(str(self.topology.resolve())),
            topology_sha256=np.asarray(evaluator.sha256_file(self.topology)),
            coordinate_space=np.asarray(evaluator.COORDINATE_SPACE),
            interpolation=np.asarray(evaluator.INTERPOLATION),
            sampling=np.asarray(evaluator.SAMPLING),
            grid_formula=np.asarray(evaluator.GRID_FORMULA),
        )
        self.core_audit = root / "official-core.audit.json"
        write_json(
            self.core_audit,
            {
                "schema": "bb-ode-neb-arc500-preview-audit-v1",
                "status": "PASS",
                "output": str(self.core_result.resolve()),
                "output_sha256": evaluator.sha256_file(self.core_result),
                "diagnostic_energy_attempts": 2000,
                "diagnostic_energy_successes": 2000,
                "diagnostic_force_requests": 0,
                "optimization_N_F_contribution": 0,
            },
        )

        self.selection = root / "selection.json"
        exact_i = {self.target_hash: 100.0}
        delta = {}
        i_passage = evaluator.target_freezer.classify_first_passage(
            self.paid_rows, exact_i, 400.0
        )
        delta_passage = evaluator.target_freezer.classify_first_passage(
            self.paid_rows, delta, 600.0
        )
        _chosen, rationale = evaluator.target_freezer.choose_paths_to_densify(
            self.paid_rows, delta, exact_i, 400.0, 600.0, "observed", 1
        )
        self.selection_payload = {
            "schema": evaluator.SELECTION_SCHEMA,
            "status": "FROZEN_ZERO_OPENMM_TARGET_SELECTION",
            "claim_mode": "observed",
            "physical_calls_made_by_selector": {"energy": 0, "force": 0},
            "optimization_N_F_contribution": 0,
            "execution_authorized_by_this_file": False,
            "thresholds_from_selected_IR_IVM": self.thresholds,
            "official_ODE_NEB_inventory": evaluator.target_freezer._inventory_summary(
                self.paid_rows, exact_i, delta
            ),
            "discrete_I_perp_first_passage": i_passage,
            "densified_Delta_E_first_passage_before_followup": delta_passage,
            "selection_rationale": rationale,
            "diagnostic_protocol_for_each_selected_path": {
                "coordinate_space": evaluator.COORDINATE_SPACE,
                "interpolation": evaluator.INTERPOLATION,
                "sampling": evaluator.SAMPLING,
                "number_of_points": 500,
                "include_both_endpoints": True,
                "grid_formula": evaluator.GRID_FORMULA,
                "energy_evaluations_per_distinct_path": 500,
                "force_evaluations_per_path": 0,
                "optimization_N_F_contribution": 0,
                "exact_hash_reuse_only": True,
            },
            "paths_to_densify": [
                {
                    "optimization_N_F_first_occurrence": self.target_nf,
                    "paid_batch_id_first_occurrence": 200,
                    "source_batch_id": 200,
                    "official_path32_sha256": self.target_hash,
                    "coordinate_archive": str(self.archive.resolve()),
                    "coordinate_archive_sha256": self.archive_sha,
                    "exact_discrete_I_perp_eV2_per_A": 100.0,
                }
            ],
            "official_paid_states": [
                {
                    **row,
                    "discrete_I_perp_eV2_per_A": (
                        100.0 if row["official_path32_sha256"] == self.target_hash else None
                    ),
                    "densified_Delta_E_kcal_per_mol": None,
                }
                for row in self.paid_rows
            ],
            "bindings": {
                **self.official_bindings,
                "selector": binding(ROOT / "scripts" / "freeze_targeted_ode_neb_arc500.py"),
                "selected_IR_IVM_arc500": pair_binding(
                    self.selected_result, self.selected_audit
                ),
                "paid_force_receipts": [
                    {
                        "path": str(self.archive.resolve()),
                        "sha256": self.archive_sha,
                        "optimization_N_F": self.target_nf,
                        "paid_batch_id": 200,
                        "official_path32_sha256": self.target_hash,
                        "discrete_I_perp_eV2_per_A": 100.0,
                    }
                ],
                "excluded_paid_force_receipts": [],
                "completed_ODE_NEB_arc500": [
                    {
                        **pair_binding(self.core_result, self.core_audit),
                        "optimization_N_F": [240, 480, 990, 9990],
                        "official_path32_sha256": [
                            str(value) for value in core_hashes
                        ],
                    }
                ],
            },
            "claim_guard": {
                "Delta_E_monotonicity_assumed": False,
                "unmeasured_earlier_paths_forbid_exact_first_passage_claim": True,
            },
        }
        write_json(self.selection, self.selection_payload)

        self.execution_contract = root / "execution.json"
        self.execution_payload = {
            "schema": evaluator.EXECUTION_SCHEMA,
            "status": "AUTHORIZED_AFTER_FROZEN_TARGET_SELECTION",
            "physical_execution_authorized": True,
            "target_selection": binding(self.selection),
            "topology": binding(self.topology),
            "evaluator": binding(ROOT / "scripts" / "evaluate_targeted_ode_neb_arc500.py"),
            "implementation_bindings": {
                "target_selector": binding(
                    ROOT / "scripts" / "freeze_targeted_ode_neb_arc500.py"
                ),
                "common_arc500_evaluator": binding(
                    ROOT / "scripts" / "evaluate_discrete_iperp_densified_barrier.py"
                ),
                "bb_core": binding(ROOT / "src" / "bb_core.py"),
            },
            "output": str(self.output.resolve()),
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
                "host_local_BB_locks": [str(self.lock1), str(self.lock2)],
            },
        }
        write_json(self.execution_contract, self.execution_payload)

    def patched_preflight(self) -> dict:
        return self.patched_preflight_custom(
            self.selection,
            self.execution_contract,
            self.output,
            {self.target_hash: 100.0},
            {},
            self.exact_source,
        )

    def patched_preflight_custom(
        self,
        selection: Path,
        execution_contract: Path,
        output: Path,
        exact_i: dict,
        delta: dict,
        exact_sources: dict,
    ) -> dict:
        with mock.patch.object(
            evaluator.target_freezer,
            "load_official_paid_ledger",
            return_value=(self.paid_rows, self.official_bindings),
        ), mock.patch.object(
            evaluator,
            "revalidate_metric_sources",
            return_value=(exact_i, delta, exact_sources),
        ), mock.patch.object(
            evaluator,
            "CANONICAL_HOST_LOCKS",
            (self.lock1.resolve(), self.lock2.resolve()),
        ), mock.patch.object(
            evaluator.common_arc500,
            "load_eligible_rows",
            return_value=(
                self.common_path,
                1.0e6,
                30,
                [
                    {
                        "label": evaluator.SELECTED_IVM_LABEL,
                        "history_index": 1,
                        "calls_per_image": 2,
                        "optimization_N_F": 9950,
                        "discrete_I_perp": 400.0,
                        "path_sha256": "a" * 64,
                    }
                ],
            ),
        ):
            return evaluator.preflight(selection, execution_contract, output)


class TargetedEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = TargetedEvaluatorFixture(Path(self.temporary.name))

    def tearDown(self):
        self.temporary.cleanup()

    def test_zero_openmm_preflight_validates_all333_and_target(self):
        state = self.fixture.patched_preflight()
        self.assertEqual(state["paid_state_count"], 333)
        self.assertEqual(state["distinct_retained_path_count"], 284)
        self.assertEqual(len(state["rows"]), 1)
        self.assertEqual(state["rows"][0]["optimization_N_F"], 6000)
        self.assertNotIn("openmm", sys.modules)

    def test_selection_binding_tamper_is_rejected(self):
        payload = read = json.loads(self.fixture.selection.read_text(encoding="utf-8"))
        payload["purpose"] = "post-contract tamper"
        write_json(self.fixture.selection, payload)
        with self.assertRaisesRegex(ValueError, "contract/selection binding"):
            self.fixture.patched_preflight()

    def test_fabricated_selected_arc500_row_outside_frozen_lineage_is_rejected(self):
        with np.load(self.fixture.selected_result, allow_pickle=False) as saved:
            arrays = {key: np.asarray(saved[key]).copy() for key in saved.files}
        arrays["optimization_N_F"][1] = 9949
        np.savez_compressed(self.fixture.selected_result, **arrays)
        selected_audit = json.loads(
            self.fixture.selected_audit.read_text(encoding="utf-8")
        )
        selected_audit["output_sha256"] = evaluator.sha256_file(
            self.fixture.selected_result
        )
        write_json(self.fixture.selected_audit, selected_audit)
        selection = json.loads(self.fixture.selection.read_text(encoding="utf-8"))
        selection["bindings"]["selected_IR_IVM_arc500"] = pair_binding(
            self.fixture.selected_result, self.fixture.selected_audit
        )
        write_json(self.fixture.selection, selection)
        execution = json.loads(
            self.fixture.execution_contract.read_text(encoding="utf-8")
        )
        execution["target_selection"] = binding(self.fixture.selection)
        write_json(self.fixture.execution_contract, execution)
        with self.assertRaisesRegex(ValueError, "frozen eligible path lineage"):
            self.fixture.patched_preflight()

    def test_force_protocol_tamper_is_rejected(self):
        payload = copy.deepcopy(self.fixture.selection_payload)
        payload["diagnostic_protocol_for_each_selected_path"][
            "force_evaluations_per_path"
        ] = 1
        with self.assertRaisesRegex(ValueError, "force_evaluations_per_path"):
            evaluator.validate_protocol(payload)

    def test_exact_mode_selection_is_explicitly_not_executable_without_exact_i_perp(self):
        payload = copy.deepcopy(self.fixture.selection_payload)
        payload["claim_mode"] = "exact"
        with self.assertRaisesRegex(ValueError, "only observed-mode"):
            evaluator.validate_protocol(payload)

    def test_paid_force_receipt_hash_tamper_is_rejected(self):
        receipt = Path(self.temporary.name) / "paid-force.npz"
        np.savez_compressed(receipt, forces=np.asarray([1.0]))
        record = binding(receipt)
        receipt.write_bytes(receipt.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "paid-force receipt binding drift"):
            evaluator.validate_file_binding(record, "paid-force receipt")

    def test_rejected_trial_cannot_be_selected_as_first_occurrence(self):
        record = copy.deepcopy(self.fixture.selection_payload["paths_to_densify"][0])
        source = copy.deepcopy(self.fixture.target_source)
        source["disposition"] = "rejected"
        with self.assertRaisesRegex(ValueError, "rejected trial"):
            evaluator.validate_target_records(
                {"paths_to_densify": [record]},
                [source],
                {self.fixture.target_hash: 100.0},
                {},
                self.fixture.exact_source,
            )

    def test_duplicate_target_path_is_rejected(self):
        record = self.fixture.selection_payload["paths_to_densify"][0]
        with self.assertRaisesRegex(ValueError, "repeats a path hash"):
            evaluator.validate_target_records(
                {"paths_to_densify": [copy.deepcopy(record), copy.deepcopy(record)]},
                self.fixture.paid_rows,
                {self.fixture.target_hash: 100.0},
                {},
                self.fixture.exact_source,
            )

    def test_target_path_hash_tamper_is_rejected(self):
        record = copy.deepcopy(self.fixture.selection_payload["paths_to_densify"][0])
        record["official_path32_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "outside paid ledger"):
            evaluator.validate_target_records(
                {"paths_to_densify": [record]},
                self.fixture.paid_rows,
                {self.fixture.target_hash: 100.0},
                {},
                self.fixture.exact_source,
            )

    def test_exact_paid_force_metric_is_required(self):
        record = self.fixture.selection_payload["paths_to_densify"][0]
        with self.assertRaisesRegex(ValueError, "lacks exact paid-force"):
            evaluator.validate_target_records(
                {"paths_to_densify": [record]},
                self.fixture.paid_rows,
                {},
                {},
                {},
            )

    def test_already_measured_path_is_rejected(self):
        record = self.fixture.selection_payload["paths_to_densify"][0]
        with self.assertRaisesRegex(ValueError, "already arc500-measured"):
            evaluator.validate_target_records(
                {"paths_to_densify": [record]},
                self.fixture.paid_rows,
                {self.fixture.target_hash: 100.0},
                {self.fixture.target_hash: 1.0},
                self.fixture.exact_source,
            )

    def test_inert_template_is_not_execution_authority(self):
        inert = ROOT / "registrations" / "diagnostics" / "BB_ODE_NEB_TARGETED_ARC500_EXECUTION_TEMPLATE_20260902_R1.json"
        payload = json.loads(inert.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "INERT_TEMPLATE_NOT_AN_AUTHORIZATION")
        self.assertFalse(payload["physical_execution_authorized"])
        with self.assertRaisesRegex(ValueError, "inert or not authorized"):
            evaluator.validate_execution_contract(
                inert,
                self.fixture.selection,
                self.fixture.topology,
                self.fixture.output,
            )

    def test_noncanonical_lock_pair_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "canonical host-local BB locks"):
            evaluator.validate_execution_contract(
                self.fixture.execution_contract,
                self.fixture.selection,
                self.fixture.topology,
                self.fixture.output,
            )

    def test_exclusive_atomic_publish_refuses_overwrite(self):
        path = Path(self.temporary.name) / "exclusive.json"
        evaluator.atomic_write_json_exclusive(path, {"first": 1})
        with self.assertRaises(FileExistsError):
            evaluator.atomic_write_json_exclusive(path, {"second": 2})
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"first": 1})

    def test_terminal_bundle_satisfies_final_plot_targeted_loader(self):
        state = self.fixture.patched_preflight()
        row = dict(state["rows"][0])
        energies = np.linspace(-10.0, 20.0, 500, dtype=np.float64)
        row.update(
            energy_eV=energies,
            energy_sha256=evaluator.sha256_array(energies),
            dense_coordinate_sha256=row["preflight_dense_coordinate_sha256"],
            densified_Delta_E_kcal_per_mol=30.0 * 23.0605,
            diagnostic_energy_reused=False,
            diagnostic_energy_attempts_this_evaluation=500,
            diagnostic_energy_attempts=500,
            diagnostic_energy_successes=500,
            diagnostic_force_requests=0,
        )
        write_synthetic_terminal_wal(state, [row])
        evaluator.save_terminal_artifacts(
            state, [row], [str(self.fixture.lock1), str(self.fixture.lock2)]
        )
        targeted_audit = self.fixture.output.with_suffix(".audit.json")
        data, _bindings = final_plot.load_official_ode(
            [self.fixture.core_result, self.fixture.output],
            [self.fixture.core_audit, targeted_audit],
            self.fixture.selection,
        )
        self.assertEqual(data["number_of_forces"].tolist(), [240, 480, 990, 6000, 9990])
        self.assertEqual(int(np.sum(data["number_of_forces"] == 6000)), 1)
        audit = json.loads(targeted_audit.read_text(encoding="utf-8"))
        self.assertEqual(audit["lineage_energy_evaluations"], 500)
        self.assertEqual(audit["diagnostic_force_requests"], 0)
        self.assertEqual(audit["fixed_endpoint_optimization_N_F"], 0)
        self.assertEqual(audit["exact_path_energy_reuse_count"], 0)
        self.assertFalse(audit["current_round_allow_exact_energy_reuse"])
        self.assertTrue(
            audit[
                "every_discrete_I_perp_is_bound_to_an_exact_paid_force_receipt"
            ]
        )
        self.assertEqual(
            audit["energy_wal_terminal_validation"]["successful_energy_events"],
            500,
        )
        self.assertEqual(
            audit["records"][0]["exact_discrete_I_perp_source_kind"],
            "paid_force_receipt",
        )
        self.assertNotIn("openmm", sys.modules)

    def test_complete_round_two_preflight_admits_only_exact_round_one_lineage(self):
        state = self.fixture.patched_preflight()
        row = dict(state["rows"][0])
        energies = np.linspace(-2.0, 3.0, 500, dtype=np.float64)
        round1_delta = 5.0 * 23.0605
        row.update(
            energy_eV=energies,
            energy_sha256=evaluator.sha256_array(energies),
            dense_coordinate_sha256=row["preflight_dense_coordinate_sha256"],
            densified_Delta_E_kcal_per_mol=round1_delta,
            diagnostic_energy_reused=False,
            diagnostic_energy_attempts_this_evaluation=500,
            diagnostic_energy_attempts=500,
            diagnostic_energy_successes=500,
            diagnostic_force_requests=0,
        )
        write_synthetic_terminal_wal(state, [row])
        evaluator.save_terminal_artifacts(
            state, [row], [str(self.fixture.lock1), str(self.fixture.lock2)]
        )
        prior_audit = self.fixture.output.with_suffix(".audit.json")

        exact_i = {
            self.fixture.target_hash: 100.0,
            self.fixture.target2_hash: 200.0,
        }
        delta = {self.fixture.target_hash: round1_delta}
        exact_sources = {
            self.fixture.target_hash: self.fixture.exact_source[
                self.fixture.target_hash
            ],
            self.fixture.target2_hash: self.fixture.exact_source2,
        }
        next_selection = copy.deepcopy(self.fixture.selection_payload)
        next_selection["bindings"]["completed_ODE_NEB_arc500"].append(
            {
                **pair_binding(self.fixture.output, prior_audit),
                "optimization_N_F": [self.fixture.target_nf],
                "official_path32_sha256": [self.fixture.target_hash],
            }
        )
        next_selection["bindings"]["paid_force_receipts"].append(
            {
                "path": str(self.fixture.target2_archive.resolve()),
                "sha256": self.fixture.target2_archive_sha,
                "optimization_N_F": self.fixture.target2_nf,
                "paid_batch_id": 201,
                "official_path32_sha256": self.fixture.target2_hash,
                "discrete_I_perp_eV2_per_A": 200.0,
            }
        )
        next_selection["official_ODE_NEB_inventory"] = (
            evaluator.target_freezer._inventory_summary(
                self.fixture.paid_rows, exact_i, delta
            )
        )
        next_selection["discrete_I_perp_first_passage"] = (
            evaluator.target_freezer.classify_first_passage(
                self.fixture.paid_rows, exact_i, 400.0
            )
        )
        next_selection["densified_Delta_E_first_passage_before_followup"] = (
            evaluator.target_freezer.classify_first_passage(
                self.fixture.paid_rows, delta, 600.0
            )
        )
        chosen, rationale = evaluator.target_freezer.choose_paths_to_densify(
            self.fixture.paid_rows,
            delta,
            exact_i,
            400.0,
            600.0,
            "observed",
            1,
        )
        self.assertEqual(
            [str(value["official_path32_sha256"]) for value in chosen],
            [self.fixture.target2_hash],
        )
        next_selection["selection_rationale"] = rationale
        next_selection["paths_to_densify"] = [
            {
                "optimization_N_F_first_occurrence": self.fixture.target2_nf,
                "paid_batch_id_first_occurrence": 201,
                "source_batch_id": 201,
                "official_path32_sha256": self.fixture.target2_hash,
                "coordinate_archive": str(
                    self.fixture.target2_archive.resolve()
                ),
                "coordinate_archive_sha256": self.fixture.target2_archive_sha,
                "exact_discrete_I_perp_eV2_per_A": 200.0,
            }
        ]
        next_selection["official_paid_states"] = [
            {
                **paid_row,
                "discrete_I_perp_eV2_per_A": exact_i.get(
                    str(paid_row["official_path32_sha256"])
                ),
                "densified_Delta_E_kcal_per_mol": delta.get(
                    str(paid_row["official_path32_sha256"])
                ),
            }
            for paid_row in self.fixture.paid_rows
        ]
        next_selection_path = Path(self.temporary.name) / "selection-round2.json"
        write_json(next_selection_path, next_selection)

        round2_output = Path(self.temporary.name) / "targeted-round2.npz"
        next_execution = copy.deepcopy(self.fixture.execution_payload)
        next_execution["target_selection"] = binding(next_selection_path)
        next_execution["output"] = str(round2_output.resolve())
        next_execution_path = Path(self.temporary.name) / "execution-round2.json"
        write_json(next_execution_path, next_execution)

        state2 = self.fixture.patched_preflight_custom(
            next_selection_path,
            next_execution_path,
            round2_output,
            exact_i,
            delta,
            exact_sources,
        )
        self.assertEqual(state2["target_round_index"], 2)
        self.assertEqual(state2["prior_targeted_bundle_count"], 1)
        self.assertEqual(len(state2["completed_prior_arc500_lineage"]), 2)
        self.assertEqual(len(state2["rows"]), 1)
        self.assertEqual(state2["rows"][0]["optimization_N_F"], 6030)
        self.assertEqual(
            state2["rows"][0]["official_path32_sha256"],
            self.fixture.target2_hash,
        )
        self.assertNotIn("openmm", sys.modules)

    def test_targeted_evaluator_writes_durable_precall_wal_without_openmm(self):
        state = self.fixture.patched_preflight()

        class FakeEnergyOnlyEvaluator:
            def __init__(self, _topology):
                self.backend = type("Backend", (), {"raw_joint_attempts": 0})()
                self.attempts = 0
                self.successes = 0
                self.failures = 0

            def evaluate(self, positions):
                self.attempts += 1
                self.successes += 1
                return float(np.asarray(positions)[0])

        with mock.patch.object(
            evaluator.common_arc500,
            "EnergyOnlyOpenMMEvaluator",
            FakeEnergyOnlyEvaluator,
        ):
            evaluated = evaluator.evaluate_targeted_paths(
                state["topology"],
                state["rows"],
                state["wal_output"],
                evaluator.wal_context(state),
            )
        receipt = evaluator.validate_terminal_energy_wal(state, evaluated)
        self.assertEqual(receipt["successful_energy_events"], 500)
        self.assertEqual(receipt["exact_path_energy_reuse_events"], 0)
        self.assertEqual(receipt["line_count"], 1002)
        self.assertNotIn("openmm", sys.modules)


if __name__ == "__main__":
    unittest.main()
