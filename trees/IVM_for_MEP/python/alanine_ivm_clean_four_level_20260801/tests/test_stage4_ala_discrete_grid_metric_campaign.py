from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


STUDY_ROOT = Path(__file__).resolve().parents[1]
SRC = STUDY_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_stage4_ala_discrete_grid_metric_campaign as campaign


def synthetic_paths(count: int = 4) -> np.ndarray:
    base = np.linspace(-1.0, 1.0, campaign.N_IMAGES * campaign.PATH_DOF).reshape(
        campaign.N_IMAGES, campaign.PATH_DOF
    )
    return np.asarray([base + 0.01 * index for index in range(count)], dtype=float)


def write_synthetic_result(path: Path, family: str) -> None:
    paths = synthetic_paths()
    arrays = {
        "total_internal_joint_evaluations": np.asarray([20, 40, 60, 80]),
        "calls_per_image": np.asarray([1, 2, 3, 4]),
        "stored_path_calls": np.asarray([1, 2, 3, 4]),
        "stored_paths": paths,
        "i_perp": np.asarray([0.4, 0.3, 0.2, 0.1]),
        "segment_length_cv": np.asarray([0.001, 0.002, 0.010, 0.003]),
        "segment_length_ratio": np.asarray([1.001, 1.002, 1.020, 1.003]),
    }
    if family == campaign.PRIMARY_FAMILY:
        arrays["reparam_committed"] = np.asarray(
            [False, True, True, False], dtype=bool
        )
        arrays["discrete_audit_stop_reason"] = np.asarray(
            [
                "evaluated-relative-natural-residual",
                "evaluated-inner-max-steps",
            ]
        )
    else:
        arrays["exact_no_reparam_complete_state_mask"] = np.asarray(
            [True, False, True, True], dtype=bool
        )
    np.savez_compressed(path, **arrays)


class DensePolicyTests(unittest.TestCase):
    def test_default_policy_is_configurable_doubling_20_to_160(self) -> None:
        policy = campaign.dense_policy()
        self.assertEqual(policy["subdivision_levels"], [20, 40, 80, 160])
        self.assertEqual(
            policy["convergence_tolerance_kcal_per_mol"], 0.001
        )
        self.assertEqual(policy["evaluated_level_count"], 4)
        self.assertEqual(
            policy["execution_rule"],
            "evaluate_every_preregistered_level_no_early_stop",
        )
        with self.assertRaisesRegex(campaign.MetricCampaignSafetyError, "doubling"):
            campaign.dense_policy((20, 60, 120))
        with self.assertRaisesRegex(campaign.MetricCampaignSafetyError, "positive"):
            campaign.dense_policy((20, 40), 0.0)

    def test_nested_grid_reuses_coarse_points_but_always_runs_to_160(self) -> None:
        path = np.asarray(
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]
        )
        calls = []

        def energy(point):
            calls.append(np.asarray(point).copy())
            return float(point[0])

        result = campaign.evaluate_nested_dense_barrier(
            path, energy, formal_shape=False
        )
        self.assertTrue(result["converged"])
        self.assertEqual(result["evaluated_levels"], [20, 40, 80, 160])
        self.assertEqual(result["selected_subdivisions"], 160)
        self.assertEqual(result["logical_energy_query_requests"], 321)
        self.assertEqual(len(calls), 321)
        self.assertEqual(
            [row["new_logical_energy_queries"] for row in result["level_records"]],
            [41, 40, 80, 160],
        )
        self.assertEqual(result["optimization_nf_charged"], 0)

    def test_nonconverged_path_reaches_configured_maximum(self) -> None:
        path = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )

        def energy(point):
            x_value = float(point[0])
            # The nearest grid point changes at every configured doubling.
            return -(x_value - 0.1337) ** 2

        result = campaign.evaluate_nested_dense_barrier(
            path,
            energy,
            convergence_tolerance_kcal=1e-30,
            formal_shape=False,
        )
        self.assertFalse(result["converged"])
        self.assertEqual(result["evaluated_levels"], [20, 40, 80, 160])
        self.assertEqual(result["logical_energy_query_requests"], 321)
        self.assertEqual(
            [row["new_logical_energy_queries"] for row in result["level_records"]],
            [41, 40, 80, 160],
        )

    def test_task_identity_binds_policy_and_topology(self) -> None:
        path_hash = "1" * 64
        topology_hash = "2" * 64
        default = campaign.path_task_id(
            path_hash, (22, 66), campaign.dense_policy(), topology_hash
        )
        tighter = campaign.path_task_id(
            path_hash, (22, 66), campaign.dense_policy(tolerance_kcal=0.0005), topology_hash
        )
        other_topology = campaign.path_task_id(
            path_hash, (22, 66), campaign.dense_policy(), "3" * 64
        )
        self.assertNotEqual(default, tighter)
        self.assertNotEqual(default, other_topology)
        self.assertIn(campaign.assigned_host(default), campaign.HOSTS)

    def test_forcefield_manifest_recursively_hashes_amber14_bundle(self) -> None:
        manifest = campaign.forcefield_runtime_manifest()
        self.assertEqual(manifest["root_forcefield_files"], ["amber14-all.xml"])
        self.assertEqual(
            set(manifest["files"]),
            {
                "amber14-all.xml",
                "amber14/DNA.OL15.xml",
                "amber14/RNA.OL3.xml",
                "amber14/lipid17.xml",
                "amber14/protein.ff14SB.xml",
            },
        )
        without_hash = {
            key: value for key, value in manifest.items() if key != "bundle_sha256"
        }
        self.assertEqual(
            manifest["bundle_sha256"], campaign.payload_sha256(without_hash)
        )

    def test_expected_coordinate_bytes_freeze_physical_cache_misses(self) -> None:
        repeated = np.zeros((campaign.N_IMAGES, campaign.PATH_DOF))
        accounting = campaign.expected_nested_coordinate_accounting(
            repeated, campaign.DEFAULT_LEVELS
        )
        self.assertEqual(
            accounting["expected_logical_energy_query_requests"], 3361
        )
        self.assertEqual(
            accounting["expected_diagnostic_raw_joint_evaluations"], 1
        )
        self.assertEqual(accounting["expected_diagnostic_cache_hits"], 3360)


class EligibilityTests(unittest.TestCase):
    def test_inexact_uses_initial_or_committed_and_raw_i_perp(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "inexact.npz"
            write_synthetic_result(result, campaign.PRIMARY_FAMILY)
            cell, paths = campaign.extract_cell_states(
                result, campaign.PRIMARY_FAMILY, "inexact-cell"
            )
        self.assertEqual(
            [state["trajectory_index"] for state in cell["states"]], [0, 1]
        )
        self.assertEqual(
            [state["raw_i_perp"] for state in cell["states"]], [0.4, 0.3]
        )
        self.assertEqual(cell["formal_complete_state_count"], 3)
        self.assertEqual(cell["complete_mesh_eligible_state_count"], 2)
        self.assertEqual(len(paths), 2)
        self.assertEqual(
            cell["inner_cap_stop_evidence"],
            {
                "source_field": "result.discrete_audit_stop_reason",
                "inner_cap_stop_token": "evaluated-inner-max-steps",
                "inner_cap_stop_count": 1,
                "inner_cap_binding": True,
                "stop_record_count": 2,
            },
        )

    def test_exact_uses_explicit_complete_mask_then_mesh_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "exact.npz"
            write_synthetic_result(result, campaign.SECONDARY_FAMILY)
            cell, _ = campaign.extract_cell_states(
                result, campaign.SECONDARY_FAMILY, "exact-cell"
            )
        self.assertEqual(
            [state["trajectory_index"] for state in cell["states"]], [0, 3]
        )
        self.assertEqual(
            cell["complete_state_definition"],
            "exact_no_reparam_complete_state_mask",
        )

    def test_exact_refuses_missing_explicit_complete_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "wrong.npz"
            write_synthetic_result(result, campaign.PRIMARY_FAMILY)
            with self.assertRaisesRegex(
                campaign.MetricCampaignSafetyError,
                "exact_no_reparam_complete_state_mask",
            ):
                campaign.extract_cell_states(
                    result, campaign.SECONDARY_FAMILY, "wrong-family"
                )

    def test_result_hash_is_frozen_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "result.npz"
            write_synthetic_result(result, campaign.PRIMARY_FAMILY)
            with self.assertRaisesRegex(
                campaign.MetricCampaignSafetyError, "source result hash drifted"
            ):
                campaign.extract_cell_states(
                    result,
                    campaign.PRIMARY_FAMILY,
                    "cell",
                    expected_result_sha256="0" * 64,
                )


class ContentAddressedManifestTests(unittest.TestCase):
    def test_decision_inputs_are_portable_inside_the_study_root(self) -> None:
        for relative in campaign.DECISION_INPUT_PATHS.values():
            resolved = (STUDY_ROOT / relative).resolve()
            self.assertEqual(
                resolved,
                STUDY_ROOT.resolve() / resolved.relative_to(STUDY_ROOT.resolve()),
            )
            self.assertTrue(resolved.is_file())
        snapshot = campaign._read_json(
            STUDY_ROOT
            / campaign.DECISION_INPUT_PATHS["portable_threshold_snapshot"]
        )
        self.assertEqual(
            snapshot, campaign.EXPECTED_PORTABLE_DECISION_SNAPSHOT
        )
        self.assertEqual(
            campaign.PROTOCOL_ID,
            "stage4_ala_discrete_grid_metric_campaign_v2",
        )

    def test_both_nonportable_v1_preregistrations_are_retired_unlaunched(self) -> None:
        provenance = campaign._superseded_v1_provenance(STUDY_ROOT)
        self.assertTrue(provenance["all_superseded_v1_outputs_absent"])
        self.assertEqual(
            [row["sha256"] for row in provenance["validated_superseded_preregistrations"]],
            [
                row["sha256"]
                for row in campaign.SUPERSEDED_V1_PREREGISTRATIONS
            ],
        )
        with mock.patch.object(campaign, "_route_exists", return_value=True):
            with self.assertRaisesRegex(
                campaign.MetricCampaignSafetyError, "zero-launch"
            ):
                campaign._superseded_v1_provenance(STUDY_ROOT)

    def test_same_path_across_cells_is_one_task_with_two_references(self) -> None:
        path = synthetic_paths(1)[0]
        path_hash = campaign.path_content_sha256(path)
        cells = []
        for family, cell_id, state_id in (
            (campaign.PRIMARY_FAMILY, "p", "a" * 64),
            (campaign.SECONDARY_FAMILY, "s", "b" * 64),
        ):
            cells.append(
                {
                    "family": family,
                    "cell_id": cell_id,
                    "source_result": f"/{cell_id}.npz",
                    "source_result_sha256": ("1" if cell_id == "p" else "2")
                    * 64,
                    "states": [
                        {
                            "state_id": state_id,
                            "trajectory_index": 0,
                            "optimization_nf": 20,
                            "raw_i_perp": 0.1,
                            "path_sha256": path_hash,
                        }
                    ],
                }
            )
        with tempfile.TemporaryDirectory() as temporary:
            tasks = campaign.build_unique_path_tasks(
                Path(temporary),
                "metric-unit",
                cells,
                {path_hash: path},
                campaign.dense_policy(),
                "3" * 64,
            )
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["reference_count"], 2)
        self.assertEqual(
            {state["path_task_id"] for cell in cells for state in cell["states"]},
            {tasks[0]["task_id"]},
        )

    def test_static_shards_are_a_disjoint_total_partition(self) -> None:
        tasks = [
            {"task_id": f"{index:064x}", "assigned_host": campaign.assigned_host(f"{index:064x}")}
            for index in range(1, 101)
        ]
        preregistration = {"tasks": tasks}
        local = campaign._tasks_for_host(preregistration, "local")
        remote = campaign._tasks_for_host(preregistration, "remote")
        self.assertEqual(len(local) + len(remote), len(tasks))
        self.assertFalse({row["task_id"] for row in local} & {row["task_id"] for row in remote})
        self.assertEqual(abs(len(local) - len(remote)), 0)

    def test_upstream_exact_aggregate_is_a_hard_input_freeze_barrier(self) -> None:
        registered = {"cells": {}}
        with mock.patch.object(
            campaign.upstream,
            "load_and_validate_preregistration",
            return_value=registered,
        ), mock.patch.object(
            campaign.infrastructure,
            "_validate_aggregate_file",
            side_effect=[{}, FileNotFoundError("successor exact aggregate absent")],
        ) as aggregate, mock.patch.object(
            campaign.infrastructure, "inspect_cell"
        ) as inspect:
            with self.assertRaises(FileNotFoundError):
                campaign.collect_validated_upstream_cells(
                    Path("/not-used"), Path("/not-used/preregistered.json")
                )
        self.assertEqual(aggregate.call_count, 2)
        inspect.assert_not_called()


class RealSuccessorIntegrationTests(unittest.TestCase):
    PREREGISTRATION = (
        STUDY_ROOT
        / "reports"
        / "stage4_ala_discrete_grid_campaign_successor_v2_"
        "1787604432554391000_preregistered.json"
    )

    @unittest.skipUnless(PREREGISTRATION.is_file(), "successor-v2 fixture absent")
    def test_real_successor_two_aggregate_manifest_and_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            preregistration_path = (
                Path(temporary) / "real_successor_metric_unit_preregistered.json"
            )
            payload = campaign.build_preregistration(
                STUDY_ROOT,
                preregistration_path,
                self.PREREGISTRATION,
            )
            self.assertEqual(
                payload["upstream_protocol_id"], campaign.upstream.PROTOCOL_ID
            )
            self.assertEqual(payload["source_cell_count"], 248)
            self.assertEqual(
                payload["family_cell_counts"],
                {campaign.PRIMARY_FAMILY: 240, campaign.SECONDARY_FAMILY: 8},
            )
            self.assertGreater(payload["eligible_state_reference_count"], 0)
            self.assertEqual(
                payload["incumbent_eligible_state_reference_count"], 18
            )
            self.assertEqual(
                payload["incumbent_inputs"]["result"]["sha256"],
                campaign.INCUMBENT_INPUTS["result"]["sha256"],
            )
            self.assertGreater(payload["unique_path_task_count"], 0)
            self.assertLess(
                payload["unique_path_task_count"],
                payload["eligible_state_reference_count"],
            )
            self.assertIn(
                "openmm_forcefield_bundle", payload["source_manifest"]
            )
            self.assertEqual(
                set(payload["cross_host_sentinel_tasks"]), set(campaign.HOSTS)
            )
            c50 = [
                cell
                for cell in payload["cells"]
                if cell["family"] == campaign.PRIMARY_FAMILY
                and cell["controls"]["inner_max_steps"] == 50
            ]
            self.assertEqual(len(c50), 40)
            self.assertTrue(
                all(
                    cell["inner_cap_stop_evidence"]["source_field"]
                    == "result.discrete_audit_stop_reason"
                    for cell in c50
                )
            )
            self.assertTrue(
                any(
                    cell["inner_cap_stop_evidence"]["inner_cap_binding"]
                    for cell in c50
                )
            )
            self.assertEqual(
                payload["decision_manifest"]["extracted_values"][
                    "reference_barrier_kcal_per_mol"
                ],
                campaign.REFERENCE_BARRIER_KCAL,
            )
            self.assertTrue(
                payload["predecessor_provenance"][
                    "all_superseded_v1_outputs_absent"
                ]
            )
            self.assertEqual(
                payload["decision_manifest"]["analysis_implementation"],
                payload["source_manifest"]["repository_files"][
                    "highalpha_analyzer"
                ],
            )
            self.assertEqual(
                payload["decision_manifest"][
                    "metric_orchestration_implementation"
                ],
                payload["source_manifest"]["repository_files"][
                    "metric_campaign"
                ],
            )
            preregistration_path.write_text(
                campaign.canonical_json_bytes(payload).decode("utf-8"),
                encoding="utf-8",
            )
            loaded = campaign.load_and_validate_preregistration(
                STUDY_ROOT, preregistration_path
            )
            self.assertEqual(loaded, payload)


class ArtifactAndRestartTests(unittest.TestCase):
    def _formal_metric_payload(self):
        path = synthetic_paths(1)[0]
        result = campaign.evaluate_nested_dense_barrier(
            path, lambda point: float(np.dot(point, point))
        )
        queries = result["logical_energy_query_requests"]
        result.update(
            {
                "diagnostic_raw_joint_evaluations": queries,
                "diagnostic_evaluator_joint_evaluations": queries,
                "diagnostic_energy_requests": queries,
                "diagnostic_force_requests": 0,
                "diagnostic_cache_hits": 0,
            }
        )
        return result

    @staticmethod
    def _runtime_identity(host: str):
        machine = {
            "hostname": f"unit-{host}",
            "platform_node": f"unit-{host}",
            "machine": "arm64",
            "node_id_sha256": "1" * 64,
        }
        identity = {
            "worker_host_label": host,
            "machine": machine,
            "machine_fingerprint": campaign.payload_sha256(machine),
            "system": "Darwin",
            "release": "unit",
            "python_version": "3.8.20",
            "python_executable": "/unit/python",
            "python_executable_sha256": "2" * 64,
            "scientific_distributions": {
                "numpy": "unit",
                "scipy": "unit",
                "OpenMM": "unit",
            },
        }
        identity["identity_sha256"] = campaign.payload_sha256(identity)
        return identity

    def test_metric_validator_keeps_diagnostic_and_optimization_counts_separate(self) -> None:
        payload = self._formal_metric_payload()
        task = {
            "expected_logical_energy_query_requests": payload[
                "logical_energy_query_requests"
            ],
            "expected_diagnostic_raw_joint_evaluations": payload[
                "diagnostic_raw_joint_evaluations"
            ],
            "expected_diagnostic_cache_hits": payload[
                "diagnostic_cache_hits"
            ],
        }
        validated = campaign._validate_metric_payload(
            payload, task, campaign.dense_policy()
        )
        self.assertEqual(validated["optimization_nf_charged"], 0)
        self.assertGreater(validated["diagnostic_raw_joint_evaluations"], 0)
        tampered = copy.deepcopy(payload)
        tampered["optimization_nf_charged"] = 1
        with self.assertRaisesRegex(
            campaign.PartialMetricTaskError, "optimization_nf_charged"
        ):
            campaign._validate_metric_payload(
                tampered, task, campaign.dense_policy()
            )

    def test_sequence_validates_skip_and_retries_only_absent_task(self) -> None:
        tasks = [{"task_id": "one"}, {"task_id": "two"}]
        complete = {"one"}
        launched = []

        def inspect(task):
            if task["task_id"] in complete:
                return "verified", {"task_id": task["task_id"]}
            return "absent", None

        def launch(task):
            launched.append(task["task_id"])
            complete.add(task["task_id"])

        rows = campaign.execute_task_sequence(tasks, inspect, launch)
        self.assertEqual(launched, ["two"])
        self.assertEqual(
            [row["action"] for row in rows], ["validated_skip", "executed"]
        )

    def test_task_artifact_preserves_actual_producer_runtime_on_skip(self) -> None:
        metric = self._formal_metric_payload()
        identity = self._runtime_identity("local")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            preregistration_path = directory / "metric_preregistered.json"
            preregistration_path.write_text("{}", encoding="utf-8")
            artifact_path = directory / "task.json"
            task = {
                "task_id": "a" * 64,
                "task_role": "eligible_unique_path",
                "assigned_host": "local",
                "path_sha256": "b" * 64,
                "source_locator": {"result": "/frozen/result.npz"},
                "reference_count": 1,
                "artifact": str(artifact_path),
                "expected_logical_energy_query_requests": metric[
                    "logical_energy_query_requests"
                ],
                "expected_diagnostic_raw_joint_evaluations": metric[
                    "diagnostic_raw_joint_evaluations"
                ],
                "expected_diagnostic_cache_hits": metric[
                    "diagnostic_cache_hits"
                ],
            }
            preregistration = {
                "dense_policy": campaign.dense_policy(),
                "optimizer_scientific_runtime_sha256": (
                    campaign.infrastructure._scientific_runtime_sha256(identity)
                ),
            }
            formal = campaign._formal_task_artifact(
                preregistration_path,
                preregistration,
                task,
                metric,
                identity,
            )
            campaign._write_json_exclusive(artifact_path, formal)
            state, row = campaign.inspect_task(
                preregistration_path, preregistration, task
            )
        self.assertEqual(state, "verified")
        self.assertEqual(
            row["producer_machine_fingerprint"], identity["machine_fingerprint"]
        )
        self.assertEqual(row["producer_provenance"], "artifact_attested_machine")

    def test_failed_attempt_never_publishes_formal_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            preregistration_path = temporary_path / "pre.json"
            preregistration_path.write_text("{}", encoding="utf-8")
            task = {
                "task_id": "a" * 64,
                "assigned_host": "local",
                "path_sha256": "b" * 64,
                "source_locator": {},
                "reference_count": 1,
                "artifact": str(temporary_path / "artifact.json"),
                "attempts_directory": str(temporary_path / "attempts"),
            }

            def fail(_task):
                raise RuntimeError("synthetic diagnostic failure")

            with self.assertRaisesRegex(RuntimeError, "synthetic"):
                campaign._launch_task(
                    preregistration_path,
                    {"dense_policy": campaign.dense_policy()},
                    task,
                    fail,
                    {},
                )
            self.assertFalse(Path(task["artifact"]).exists())
            failures = list(Path(task["attempts_directory"]).glob("*/failure.json"))
            self.assertEqual(len(failures), 1)

    def test_jobs_is_hard_one(self) -> None:
        self.assertEqual(campaign.validate_jobs(1), 1)
        for value in (0, 2, 8):
            with self.assertRaises(campaign.MetricCampaignSafetyError):
                campaign.validate_jobs(value)

    def test_thread_environment_is_fail_closed(self) -> None:
        complete = dict(campaign.THREAD_ENVIRONMENT)
        self.assertEqual(campaign.validate_thread_environment(complete), complete)
        incomplete = dict(complete)
        incomplete["OPENMM_CPU_THREADS"] = "2"
        with self.assertRaisesRegex(
            campaign.MetricCampaignSafetyError, "thread variable"
        ):
            campaign.validate_thread_environment(incomplete)

    def test_formal_highalpha_analyzer_is_source_pinned(self) -> None:
        self.assertEqual(
            campaign.HIGHALPHA_ANALYZER_RELATIVE,
            "src/analyze_stage4_ala_discrete_grid_metrics.py",
        )
        source_manifest = campaign._source_manifest(STUDY_ROOT)
        analyzer = source_manifest["repository_files"]["highalpha_analyzer"]
        decision = campaign._decision_manifest(STUDY_ROOT)
        self.assertEqual(decision["analysis_implementation"], analyzer)
        self.assertEqual(
            decision["analysis_rules"]["unique_winner_gate_policy_id"],
            campaign.DECISION_RULES["unique_winner_gate_policy_id"],
        )


class FormalComparisonTests(unittest.TestCase):
    def test_cost_to_precision_uses_minimum_observed_nf_without_interpolation(self) -> None:
        points = [(10, 0.5), (20, 0.2), (30, 0.4)]
        self.assertEqual(campaign.cost_to_precision(points, 0.3), 20.0)
        self.assertTrue(math.isinf(campaign.cost_to_precision(points, 0.1)))

    def test_two_metric_dominance_and_same_state_joint_gate(self) -> None:
        candidate = [
            {
                "optimization_nf": 10,
                "raw_i_perp": 3e-4,
                "barrier_error_kcal_per_mol": 0.03,
            },
            {
                "optimization_nf": 20,
                "raw_i_perp": 1e-4,
                "barrier_error_kcal_per_mol": 0.01,
            },
        ]
        comparator = [
            {
                "optimization_nf": 20,
                "raw_i_perp": 3e-4,
                "barrier_error_kcal_per_mol": 0.03,
            },
            {
                "optimization_nf": 40,
                "raw_i_perp": 1e-4,
                "barrier_error_kcal_per_mol": 0.01,
            },
        ]
        result = campaign.compare_candidate_metric_states(candidate, comparator)
        self.assertTrue(result["full_two_metric_dominance"])
        self.assertEqual(result["candidate_joint_target_nf"], 20.0)
        self.assertEqual(result["comparator_joint_target_nf"], 40.0)

    def test_pairwise_dominance_needs_strictness_on_only_one_metric_curve(self) -> None:
        candidate = [
            {
                "optimization_nf": 20,
                "raw_i_perp": 1e-4,
                "barrier_error_kcal_per_mol": 0.01,
            }
        ]
        comparator = [
            {
                "optimization_nf": 20,
                "raw_i_perp": 2e-4,
                "barrier_error_kcal_per_mol": 0.01,
            }
        ]
        result = campaign.compare_candidate_metric_states(candidate, comparator)
        self.assertTrue(result["i_perp_curve_strict_somewhere"])
        self.assertFalse(result["barrier_curve_strict_somewhere"])
        self.assertTrue(result["full_two_metric_dominance"])

    def test_any_nonconverged_formal_path_forces_no_winner(self) -> None:
        base_state = {
            "optimization_nf": 20,
            "raw_i_perp": 1e-4,
            "barrier_error_kcal_per_mol": 0.01,
            "path_task_id": "task-a",
            "dense_converged": True,
        }
        grid = [
            {"cell_id": "candidate", "states": [dict(base_state)]},
            {
                "cell_id": "other",
                "states": [
                    {
                        **base_state,
                        "path_task_id": "task-b",
                        "dense_converged": False,
                    }
                ],
            },
        ]
        incumbent = {"cell_id": "incumbent", "states": [dict(base_state)]}
        result = campaign.analyze_materialized_metric_cells(grid, incumbent)
        self.assertEqual(
            result["qualifier_status"],
            "not_determined_due_to_nonconverged_paths",
        )
        self.assertEqual(
            result["unique_winner_status"],
            "not_determined_due_to_nonconverged_paths",
        )
        self.assertIsNone(result["grid_only_nondominated"])
        self.assertIn(
            "candidate",
            result["provisional_nondominated_among_fully_converged_grid_cells"],
        )

    def test_unique_winner_must_dominate_grid_and_incumbent_and_reach_joint(self) -> None:
        def cell(cell_id, nf):
            return {
                "cell_id": cell_id,
                "states": [
                    {
                        "optimization_nf": nf,
                        "raw_i_perp": 1e-4,
                        "barrier_error_kcal_per_mol": 0.01,
                        "path_task_id": f"task-{cell_id}",
                        "dense_converged": True,
                    }
                ],
            }

        result = campaign.analyze_materialized_metric_cells(
            [cell("candidate", 10), cell("other", 20)],
            cell("incumbent", 30),
        )
        self.assertEqual(result["unique_winner_status"], "unique_winner")
        self.assertEqual(result["unique_winner_cell_id"], "candidate")
        self.assertEqual(result["grid_only_nondominated"], ["candidate"])
        self.assertEqual(
            result["incumbent_inclusive_nondominated"], ["candidate"]
        )

    def test_cross_host_sentinel_requires_every_level_within_tolerance(self) -> None:
        records = [
            {"subdivisions": level, "barrier_kcal_per_mol": 2.5}
            for level in campaign.DEFAULT_LEVELS
        ]
        metrics = {
            host: {"level_records": copy.deepcopy(records), "converged": True}
            for host in campaign.HOSTS
        }
        comparison = campaign.compare_cross_host_sentinel_metrics(metrics)
        self.assertTrue(comparison["all_levels_within_tolerance"])
        metrics["remote"]["converged"] = False
        comparison = campaign.compare_cross_host_sentinel_metrics(metrics)
        self.assertFalse(comparison["both_dense_converged"])
        self.assertTrue(
            comparison["sentinel_convergence_is_not_a_comparison_domain_gate"]
        )
        metrics["remote"]["level_records"][2]["barrier_kcal_per_mol"] += 1e-8
        with self.assertRaisesRegex(
            campaign.MetricCampaignSafetyError, "sentinel barriers"
        ):
            campaign.compare_cross_host_sentinel_metrics(metrics)


if __name__ == "__main__":
    unittest.main()
