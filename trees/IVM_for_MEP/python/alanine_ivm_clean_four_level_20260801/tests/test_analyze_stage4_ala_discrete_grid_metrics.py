from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import analyze_stage4_ala_discrete_grid_metrics as analysis


FORMAL = {
    "reference_barrier_kcal_per_mol": 2.0,
    "i_perp_target": 0.2,
    "barrier_error_target_kcal_per_mol": 0.2,
}


def state(
    state_id: str,
    index: int,
    nf: int,
    i_perp: float,
    barrier_error: float,
    *,
    complete: bool = True,
    mesh: bool = True,
    dense_converged: bool = True,
) -> dict[str, object]:
    return {
        "state_id": state_id,
        "trajectory_index": index,
        "optimization_nf": nf,
        "raw_i_perp": i_perp,
        "dense_barrier_kcal_per_mol": 2.0 + barrier_error,
        "formal_complete": complete,
        "mesh_eligible": mesh,
        "dense_converged": dense_converged,
        "path_task_id": f"task-{state_id}",
    }


def cell(
    cell_id: str,
    family: str,
    states: list[dict[str, object]],
    *,
    alpha: float = 1.0,
    ratio: float = 0.1,
    cap: int = 5,
    cap_binding: bool = False,
) -> dict[str, object]:
    payload = {
        "cell_id": cell_id,
        "family": family,
        "controls": {
            "alpha": alpha,
            "residual_ratio": ratio,
            "inner_max_steps": cap,
        },
        "states": states,
    }
    payload["inner_cap_stop_evidence"] = (
        {
            "source_field": "result.discrete_audit_stop_reason",
            "inner_cap_stop_token": "evaluated-inner-max-steps",
            "inner_cap_stop_count": int(cap_binding),
            "inner_cap_binding": cap_binding,
            "stop_record_count": 1,
        }
        if family == analysis.PRIMARY_FAMILY
        else None
    )
    return payload


def prepared(raw: dict[str, object]) -> dict[str, object]:
    return analysis.prepare_cell(raw, 2.0, 0.2, 0.2)


class GridMetricAnalysisTests(unittest.TestCase):
    def test_ctp_uses_minimum_observed_nf_without_interpolation_or_terminal_only(self):
        raw = cell(
            "nonmonotone",
            analysis.PRIMARY_FAMILY,
            [
                state("s0", 0, 10, 0.5, 0.5),
                state("s1", 1, 20, 0.1, 0.1),
                state("s2", 2, 30, 0.4, 0.4),
            ],
        )
        result = prepared(raw)
        self.assertEqual(result["targets"]["i_perp_ctp_nf"], 20)
        self.assertEqual(result["targets"]["barrier_error_ctp_nf"], 20)
        self.assertEqual(result["targets"]["same_state_joint_ctp_nf"], 20)
        self.assertEqual(result["raw_endpoint"]["trajectory_index"], 2)
        self.assertEqual(
            result["raw_endpoint"]["role"],
            "descriptive_only_not_used_for_ranking",
        )
        self.assertEqual(
            analysis.observed_ctp(result["_states"], "raw_i_perp", 0.3),
            20,
        )

    def test_joint_gate_requires_one_same_state(self):
        raw = cell(
            "split-targets",
            analysis.PRIMARY_FAMILY,
            [
                state("i-only", 0, 10, 0.1, 0.5),
                state("e-only", 1, 20, 0.5, 0.1),
            ],
        )
        result = prepared(raw)
        self.assertEqual(result["targets"]["i_perp_ctp_nf"], 10)
        self.assertEqual(result["targets"]["barrier_error_ctp_nf"], 20)
        self.assertIsNone(result["targets"]["same_state_joint_ctp_nf"])

    def test_full_curve_dominance_ignores_a_worse_late_endpoint(self):
        left = prepared(
            cell(
                "left",
                analysis.PRIMARY_FAMILY,
                [
                    state("l0", 0, 10, 0.1, 0.1),
                    state("l1", 1, 30, 0.9, 0.9),
                ],
            )
        )
        right = prepared(
            cell(
                "right",
                analysis.PRIMARY_FAMILY,
                [state("r0", 0, 20, 0.2, 0.2)],
            )
        )
        comparison = analysis.compare_cells(left, right)
        self.assertEqual(left["raw_endpoint"]["raw_i_perp"], 0.9)
        self.assertTrue(comparison["left_dominates"])
        self.assertEqual(comparison["relation"], "left_dominates")
        self.assertEqual(
            analysis.nondominated_cell_ids([left, right]), ["left"]
        )

    def test_manifest_eligibility_is_consumed_not_redefined(self):
        raw = cell(
            "bad-manifest",
            analysis.PRIMARY_FAMILY,
            [state("bad", 0, 10, 0.1, 0.1, complete=False)],
        )
        with self.assertRaisesRegex(
            analysis.MetricAnalysisError, "manifest-fixed eligible"
        ):
            prepared(raw)

    def test_control_effects_are_matched_for_each_fixed_context(self):
        cells = []
        for alpha in (1.0, 10.0):
            for ratio in (0.1, 0.3):
                for cap in (5, 10):
                    good = alpha == 1.0
                    cells.append(
                        prepared(
                            cell(
                                f"a{alpha}-r{ratio}-c{cap}",
                                analysis.PRIMARY_FAMILY,
                                [
                                    state(
                                        f"s-{alpha}-{ratio}-{cap}",
                                        0,
                                        10,
                                        0.1 if good else 0.3,
                                        0.1 if good else 0.3,
                                    )
                                ],
                                alpha=alpha,
                                ratio=ratio,
                                cap=cap,
                            )
                        )
                    )
        effects = analysis.control_variable_effects(cells)
        alpha_effect = effects["axes"]["alpha"]
        self.assertEqual(alpha_effect["matched_context_count"], 4)
        pair = alpha_effect["matched_pairwise_main_effect_counts"][0]
        self.assertEqual(pair["left_level"], 1.0)
        self.assertEqual(pair["right_level"], 10.0)
        self.assertEqual(pair["partial_order"]["left_dominates"], 4)
        self.assertEqual(len(alpha_effect["conditional_effects"]), 4)
        self.assertTrue(
            all(
                row["nondominated_levels"] == [1.0]
                for row in alpha_effect["conditional_effects"]
            )
        )

    def test_exact_vs_inexact_is_labeled_as_a_confounded_family_package(self):
        inexact = [
            prepared(
                cell(
                    "p5",
                    analysis.PRIMARY_FAMILY,
                    [state("p5s", 0, 10, 0.1, 0.1)],
                    alpha=1.0,
                    ratio=0.1,
                    cap=50,
                )
            ),
            prepared(
                cell(
                    "wrong-r",
                    analysis.PRIMARY_FAMILY,
                    [state("wrs", 0, 10, 0.1, 0.1)],
                    alpha=1.0,
                    ratio=0.3,
                    cap=5,
                )
            ),
        ]
        exact = [
            prepared(
                cell(
                    "exact",
                    analysis.SECONDARY_FAMILY,
                    [state("es", 0, 20, 0.2, 0.2)],
                    alpha=1.0,
                    ratio=0.1,
                    cap=5000,
                )
            )
        ]
        result = analysis.exact_vs_inexact_effects(inexact, exact)
        self.assertIn("descriptive confounded family-package", result["causal_label"])
        self.assertEqual(result["rows"][0]["comparison_count"], 1)
        self.assertEqual(
            result["rows"][0]["comparisons"][0]["inexact_cell_id"], "p5"
        )
        self.assertFalse(
            result["rows"][0]["comparisons"][0][
                "inexact_c50_cap_stop_evidence"
            ]["inner_cap_binding"]
        )

    def test_unique_winner_is_never_inferred_without_registered_rule(self):
        primary = cell(
            "primary",
            analysis.PRIMARY_FAMILY,
            [state("ps", 0, 10, 0.1, 0.1)],
            alpha=1.0,
            ratio=0.1,
            cap=50,
        )
        exact = cell(
            "exact",
            analysis.SECONDARY_FAMILY,
            [state("es", 0, 20, 0.2, 0.2)],
            alpha=1.0,
            ratio=0.1,
            cap=5000,
        )
        incumbent = cell(
            "incumbent",
            analysis.INCUMBENT_FAMILY,
            [state("is", 0, 30, 0.3, 0.3)],
            alpha=15.0,
            ratio=0.8,
            cap=10,
        )
        absent = analysis.analyze_cells([primary, exact], FORMAL, incumbent)
        self.assertEqual(
            absent["grid_only_nondominated"]["cell_ids"], ["primary"]
        )
        self.assertEqual(
            absent["incumbent_inclusive_nondominated"]["cell_ids"],
            ["primary"],
        )
        self.assertEqual(absent["winner"]["status"], "no_unique_winner")
        self.assertEqual(
            absent["winner"]["reason"],
            "supported_unique_winner_rule_not_preregistered",
        )

        registered = dict(FORMAL)
        registered["unique_winner_gate_policy_id"] = (
            analysis.SUPPORTED_UNIQUE_WINNER_RULE
        )
        selected = analysis.analyze_cells(
            [primary, exact], registered, incumbent
        )
        self.assertEqual(selected["winner"]["status"], "unique_winner")
        self.assertEqual(selected["winner"]["cell_id"], "primary")

    def test_nonconverged_dense_value_is_diagnostic_only_and_cannot_win(self):
        invalid = cell(
            "invalid",
            analysis.PRIMARY_FAMILY,
            [
                state(
                    "bad-dense",
                    0,
                    5,
                    0.01,
                    0.01,
                    dense_converged=False,
                )
            ],
            cap=50,
        )
        exact = cell(
            "exact",
            analysis.SECONDARY_FAMILY,
            [state("exact-state", 0, 20, 0.2, 0.2)],
            cap=5000,
        )
        incumbent = cell(
            "incumbent",
            analysis.INCUMBENT_FAMILY,
            [state("inc-state", 0, 30, 0.3, 0.3)],
        )
        registered = dict(FORMAL)
        registered["unique_winner_gate_policy_id"] = (
            analysis.SUPPORTED_UNIQUE_WINNER_RULE
        )
        output = analysis.analyze_cells(
            [invalid, exact], registered, incumbent
        )
        bad = next(
            row for row in output["cell_summaries"] if row["cell_id"] == "invalid"
        )
        self.assertIsNone(bad["targets"]["barrier_error_ctp_nf"])
        self.assertIsNone(bad["targets"]["same_state_joint_ctp_nf"])
        self.assertEqual(
            bad["observed_ctp_frontiers"]["barrier_error_kcal_per_mol"], []
        )
        self.assertEqual(
            output["grid_only_nondominated"]["status"],
            "not_determined_due_to_nonconverged_paths",
        )
        self.assertIsNone(output["grid_only_nondominated"]["cell_ids"])
        self.assertNotIn(
            "invalid",
            output["provisional_nondominated_among_fully_converged_cells"][
                "grid_only_cell_ids"
            ],
        )
        self.assertEqual(
            output["winner"]["status"],
            "not_determined_due_to_nonconverged_paths",
        )
        self.assertEqual(
            output["winner"]["reason"],
            "not_determined_due_to_nonconverged_paths",
        )

    def test_exact_duplicate_nf_frontier_keeps_best_observed_state(self):
        raw = cell(
            "duplicate-nf",
            analysis.SECONDARY_FAMILY,
            [
                state("early", 0, 10, 0.4, 0.4),
                state("late", 1, 10, 0.1, 0.1),
            ],
            cap=5000,
        )
        result = prepared(raw)
        i_frontier = result["observed_ctp_frontiers"]["raw_i_perp"]
        self.assertEqual(len(i_frontier), 1)
        self.assertEqual(i_frontier[0]["state_id"], "late")
        self.assertEqual(result["targets"]["same_state_joint_ctp_nf"], 10)


class FormalLoaderSchemaTests(unittest.TestCase):
    def test_loader_delegates_exact_aggregate_validation_and_fans_out_incumbent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            preregistration_path = root / "metric_preregistered.json"
            aggregate_path = root / "metric_aggregate.json"
            artifact_path = root / "task.json"
            preregistration_path.write_text("{}\n", encoding="utf-8")
            metric_payload = {
                "selected_barrier_kcal_per_mol": 2.1,
                "selected_subdivisions": 160,
                "converged": True,
            }
            artifact_path.write_text(
                json.dumps({"metric": metric_payload}), encoding="utf-8"
            )
            aggregate = {
                "status": "complete",
                "protocol_id": analysis.metric_campaign.PROTOCOL_ID,
                "preregistration": str(preregistration_path),
                "preregistration_sha256": analysis.metric_campaign.file_sha256(
                    preregistration_path
                ),
                "all_path_tasks_dense_converged": True,
                "nonconverged_path_task_count": 0,
            }
            aggregate_path.write_text(json.dumps(aggregate), encoding="utf-8")

            def registered_cell(index: int, family: str) -> dict[str, object]:
                evidence = (
                    {
                        "source_field": "result.discrete_audit_stop_reason",
                        "inner_cap_stop_token": "evaluated-inner-max-steps",
                        "inner_cap_stop_count": 0,
                        "inner_cap_binding": False,
                        "stop_record_count": 1,
                    }
                    if family == analysis.PRIMARY_FAMILY
                    else None
                )
                return {
                    "family": family,
                    "cell_id": f"cell-{index}",
                    "controls": {
                        "alpha": 1.0,
                        "residual_ratio": 0.1,
                        "inner_max_steps": 50,
                    },
                    "inner_cap_stop_evidence": evidence,
                    "states": [
                        {
                            "state_id": f"state-{index}",
                            "trajectory_index": 0,
                            "optimization_nf": 20,
                            "raw_i_perp": 0.1,
                            "formal_complete": True,
                            "mesh_eligible": True,
                            "path_task_id": "shared-task",
                        }
                    ],
                }

            grid = [
                registered_cell(index, analysis.PRIMARY_FAMILY)
                for index in range(240)
            ] + [
                registered_cell(240 + index, analysis.SECONDARY_FAMILY)
                for index in range(8)
            ]
            incumbent = registered_cell(999, analysis.INCUMBENT_FAMILY)
            incumbent["cell_id"] = "incumbent"
            preregistration = {
                "planned_outputs": {"aggregate": str(aggregate_path)},
                "tasks": [
                    {"task_id": "shared-task", "artifact": str(artifact_path)}
                ],
                "cells": grid,
                "incumbent_comparator": incumbent,
                "dense_policy": {},
            }
            inspected = {
                "task_id": "shared-task",
                "artifact": str(artifact_path),
            }
            with mock.patch.object(
                analysis.metric_campaign, "_root", return_value=root
            ), mock.patch.object(
                analysis.metric_campaign,
                "load_and_validate_preregistration",
                return_value=preregistration,
            ), mock.patch.object(
                analysis.metric_campaign,
                "validate_aggregate",
                return_value=aggregate,
            ) as aggregate_validator, mock.patch.object(
                analysis.metric_campaign,
                "inspect_task",
                return_value=("verified", inspected),
            ), mock.patch.object(
                analysis.metric_campaign,
                "_validate_metric_payload",
                return_value=metric_payload,
            ):
                cells, loaded_incumbent, loaded_prereg, provenance = (
                    analysis.load_formal_metric_cells(root, aggregate_path)
                )
            self.assertEqual(len(cells), 248)
            self.assertEqual(loaded_incumbent["cell_id"], "incumbent")
            self.assertEqual(loaded_prereg, preregistration)
            self.assertIn("metric_task_artifact_sha256", provenance)
            self.assertFalse(
                cells[0]["inner_cap_stop_evidence"]["inner_cap_binding"]
            )
            aggregate_validator.assert_called_once_with(
                root, preregistration_path, preregistration
            )


if __name__ == "__main__":
    unittest.main()
