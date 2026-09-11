#!/usr/bin/env python3
"""Static and synthetic gates for low-r coherence-conditioned recovery memory."""

from __future__ import annotations

import ast
import copy
import csv
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import q10_irivm_atomic_tangent_secant_core as method  # noqa: E402
import run_grid as runner  # noqa: E402


FROZEN_CHORD_COHERENCE = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "q10_irivm_chord_coherence_20260830/src/"
    "q10_irivm_atomic_tangent_secant_core.py"
)
V1_FORMAL_CAMPAIGN = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "q10_irivm_low_r_coherence_secant_20260830"
)
V1_FORMAL_CORE = V1_FORMAL_CAMPAIGN / "src/q10_irivm_atomic_tangent_secant_core.py"
V1_FORMAL_RUNNER = V1_FORMAL_CAMPAIGN / "src/run_grid.py"

class FakeCore:
    N_INTERNAL = 2
    N_TOTAL_IMAGES = 4

    @staticmethod
    def improved_tangents(path, energies):
        del path, energies
        return np.asarray([[1.0, 0.0], [1.0, 0.0]])

    @staticmethod
    def normal_basis(tangent):
        del tangent
        return np.asarray([[0.0], [1.0]])

    @staticmethod
    def _usable_warm_pairs(memory, basis, alpha, history_size):
        del memory, basis, alpha, history_size
        return []

    @staticmethod
    def _harmonic_preconditioner(basis, alpha, active_curvature, enabled):
        del basis, alpha, active_curvature, enabled
        return lambda vector: np.asarray(vector, dtype=np.float64)

    @staticmethod
    def _lbfgs_direction(gradient, pairs, apply_h0):
        del pairs
        return -np.asarray(apply_h0(gradient), dtype=np.float64)


def quadratic(point):
    value = np.asarray(point, dtype=np.float64)
    return float(0.5 * np.dot(value, value)), value.copy()


def make_evaluators(core, ledger):
    return [
        method.Evaluator(
            quadratic,
            ledger,
            charged=(0 < index < core.N_TOTAL_IMAGES - 1),
        )
        for index in range(core.N_TOTAL_IMAGES)
    ]


def synthetic_states(core, evaluators):
    pre_path = np.zeros((core.N_TOTAL_IMAGES, 2), dtype=np.float64)
    pre_path[:, 0] = np.arange(core.N_TOTAL_IMAGES, dtype=np.float64)
    pre_energy = np.zeros(core.N_TOTAL_IMAGES, dtype=np.float64)
    pre_gradient = np.zeros_like(pre_path)
    direct_path = pre_path.copy()
    direct_path[1, 1] = 1.0
    direct_path[2, 1] = 2.0
    attempt_start = np.asarray([item.calls for item in evaluators])
    direct_energy, direct_gradient = method.evaluate_path(
        core, direct_path, evaluators
    )
    return (
        (pre_path, pre_energy, pre_gradient),
        (direct_path, direct_energy, direct_gradient),
        attempt_start,
    )


def complete_projector(_core, raw_path, _direct_energies, _ceiling):
    return np.asarray(raw_path).copy(), 0.0, 1, 0, True


def deterministic_seed(strong):
    def seed(_core, raw_path, direct_energies, controls):
        del direct_energies
        weight = int(controls.m_max) - 1 if strong else 1
        return np.asarray(raw_path, dtype=np.float64).copy(), {
            "mode": "synthetic_explicit_branch",
            "saddle_image": 1,
            "adjacent_chord_cosine": 1.0 if strong else 0.0,
            "coherence_loss": 0.0 if strong else 1.0,
            "coherence_loss_limit": 1.0 / float((int(controls.m_max) - 1) ** 2),
            "amplified_direction_mismatch": 0.0 if strong else 1.0,
            "strong_recovery_selected": bool(strong),
            "saddle_spring_weight": float(weight),
            "spring_residual": 0.0,
            "spring_iterations": 1,
        }

    return seed


def complete_atomic_merge(
    core, _current, candidate, _ceiling, _alpha, _post_main_i
):
    return (
        candidate,
        np.ones(core.N_INTERNAL, dtype=bool),
        {
            "candidate_i_perp": 0.5,
            "candidate_equidistance": 0.0,
            "actual_proximal_merit": 1.0,
            "finite": True,
            "monotone_to_post_main_i_perp": True,
            "positive_actual_proximal_merit": True,
            "accepted_by_atomic_gate": True,
        },
    )


class AtomicTangentSecantContractTests(unittest.TestCase):
    PUBLICATION_COUNT_FIELDS = (
        "accepted_recovery_secants_certified_curvature",
        "accepted_recovery_secants_rejected_curvature",
        "accepted_recovery_secants_omitted_by_rank",
        "accepted_recovery_secants_added",
    )
    PUBLICATION_INDEX_FIELDS = (
        "accepted_recovery_secant_certified_indices",
        "accepted_recovery_secant_rejected_indices",
        "accepted_recovery_secant_omitted_indices",
        "accepted_recovery_secant_published_indices",
        "accepted_recovery_secant_published_normalized_curvatures",
    )

    def assert_publication_not_applicable(self, transaction):
        self.assertEqual(
            transaction["accepted_recovery_secant_publication_mode"],
            "not_applicable",
        )
        for name in self.PUBLICATION_COUNT_FIELDS:
            self.assertEqual(transaction[name], 0, name)
        for name in self.PUBLICATION_INDEX_FIELDS:
            self.assertEqual(transaction[name], [], name)

    def assert_publication_closure(self, transaction, n_internal):
        added = int(transaction["accepted_recovery_secants_added"])
        omitted = int(
            transaction["accepted_recovery_secants_omitted_by_rank"]
        )
        rejected = int(
            transaction["accepted_recovery_secants_rejected_curvature"]
        )
        certified = int(
            transaction["accepted_recovery_secants_certified_curvature"]
        )
        self.assertEqual(certified, added + omitted)
        self.assertEqual(n_internal, added + omitted + rejected)
        for name in self.PUBLICATION_INDEX_FIELDS[:-1]:
            values = transaction[name]
            self.assertEqual(values, sorted(set(values)), name)
        self.assertEqual(
            len(transaction["accepted_recovery_secant_certified_indices"]),
            certified,
        )
        self.assertEqual(
            len(transaction["accepted_recovery_secant_rejected_indices"]),
            rejected,
        )
        self.assertEqual(
            len(transaction["accepted_recovery_secant_omitted_indices"]),
            omitted,
        )
        self.assertEqual(
            len(transaction["accepted_recovery_secant_published_indices"]),
            added,
        )
        self.assertEqual(
            len(
                transaction[
                    "accepted_recovery_secant_published_normalized_curvatures"
                ]
            ),
            added,
        )

    def assert_transaction_nof_closure(self, transaction, n_total):
        direct = np.asarray(
            transaction["direct_joint_evaluation_vector"], dtype=np.int64
        )
        raw = np.asarray(
            transaction["raw_recovery_joint_evaluation_vector"], dtype=np.int64
        )
        projected = np.asarray(
            transaction["projected_recovery_joint_evaluation_vector"],
            dtype=np.int64,
        )
        recovery = np.asarray(
            transaction["recovery_joint_evaluation_vector"], dtype=np.int64
        )
        total = np.asarray(
            transaction["transaction_joint_evaluation_vector"], dtype=np.int64
        )
        for vector in (direct, raw, projected, recovery, total):
            self.assertEqual(vector.shape, (n_total,))
            self.assertEqual(int(vector[0]), 0)
            self.assertEqual(int(vector[-1]), 0)
        np.testing.assert_array_equal(raw, np.zeros(n_total, dtype=np.int64))
        np.testing.assert_array_equal(recovery, projected)
        np.testing.assert_array_equal(total, direct + projected)
        self.assertEqual(transaction["direct_joint_evaluations"], int(direct.sum()))
        self.assertEqual(transaction["raw_recovery_new_image_evaluations"], 0)
        self.assertEqual(
            transaction["projected_recovery_new_image_evaluations"],
            int(projected.sum()),
        )
        self.assertEqual(
            transaction["recovery_joint_evaluations"], int(recovery.sum())
        )
        self.assertEqual(
            transaction["transaction_joint_evaluations"], int(total.sum())
        )
        self.assertLessEqual(int(direct.max()), 1)
        self.assertLessEqual(int(projected.max()), 1)
        self.assertLessEqual(int(total.max()), 2)

    def recovery_call(
        self,
        *,
        budget=100,
        merge=complete_atomic_merge,
        i_values=(1.0, 2.0, 0.5),
        stationary_probe=False,
        strong_recovery_selected=False,
    ):
        core = FakeCore()
        ledger = method.Ledger(limit=budget)
        evaluators = make_evaluators(core, ledger)
        pre, direct, attempt_start = synthetic_states(core, evaluators)
        memories = [[], []]
        controls = method.Controls(
            10, method.ALPHAS[0], method.R_VALUES[0], method.M_MAX_VALUES[0]
        )
        with mock.patch.object(
            method,
            "internal_i_perp",
            side_effect=[
                (float(value), np.zeros(2)) for value in i_values
            ],
        ), mock.patch.object(
            method, "equidistance_integral", return_value=0.0
        ), mock.patch.object(
            method,
            "tangent_chord_constraint_project",
            side_effect=complete_projector,
        ), mock.patch.object(
            method,
            "chord_coherent_recovery_seed",
            side_effect=deterministic_seed(strong_recovery_selected),
        ), mock.patch.object(
            method, "atomic_recovery_merge", side_effect=merge
        ):
            updated, transaction, descriptor = method.single_synchronized_recovery(
                core,
                pre,
                direct,
                evaluators,
                memories,
                None,
                controls,
                0.0,
                attempt_start,
                stationary_probe_requested=stationary_probe,
            )
        return core, ledger, pre, direct, memories, updated, transaction, descriptor

    def test_recovery_threshold_is_post_main_and_does_not_read_r(self):
        def reject(core, current, _candidate, _ceiling, _alpha, _post_main_i):
            return (
                current,
                np.zeros(core.N_INTERNAL, dtype=bool),
                {
                    "candidate_i_perp": 1.5,
                    "candidate_equidistance": 0.0,
                    "actual_proximal_merit": -1.0,
                    "finite": True,
                    "monotone_to_post_main_i_perp": False,
                    "positive_actual_proximal_merit": False,
                    "accepted_by_atomic_gate": False,
                },
            )

        (
            _core,
            _ledger,
            pre,
            _direct,
            memories,
            updated,
            transaction,
            descriptor,
        ) = self.recovery_call(
            merge=reject, i_values=(1.0, 2.0)
        )
        np.testing.assert_array_equal(updated[0], pre[0])
        self.assertFalse(transaction["committed"])
        self.assertTrue(transaction["rolled_back"])
        self.assertEqual(transaction["acceptance_envelope_i_perp"], 1.0)
        self.assertFalse(transaction["main_r_read_by_recovery"])
        self.assert_publication_not_applicable(transaction)
        self.assertEqual([len(memory) for memory in memories], [0, 0])
        self.assertIsNotNone(descriptor)

    def test_acceptable_direct_reparameterization_skips_recovery(self):
        core = FakeCore()
        ledger = method.Ledger(limit=100)
        evaluators = make_evaluators(core, ledger)
        pre, direct, attempt_start = synthetic_states(core, evaluators)
        controls = method.Controls(
            10, method.ALPHAS[0], method.R_VALUES[0], method.M_MAX_VALUES[0]
        )
        with mock.patch.object(
            method,
            "internal_i_perp",
            side_effect=[(1.0, np.zeros(2)), (0.5, np.zeros(2))],
        ), mock.patch.object(
            method, "equidistance_integral", return_value=0.0
        ), mock.patch.object(
            method,
            "tangent_chord_constraint_project",
            side_effect=AssertionError("recovery should not run"),
        ):
            updated, transaction, descriptor = method.single_synchronized_recovery(
                core,
                pre,
                direct,
                evaluators,
                [[], []],
                None,
                controls,
                0.0,
                attempt_start,
            )
        np.testing.assert_array_equal(updated[0], direct[0])
        self.assertTrue(transaction["committed"])
        self.assertTrue(transaction["direct_within_post_main_i_perp"])
        self.assertFalse(transaction["projected_trial_evaluated"])
        self.assertIsNone(transaction["raw_pre_addition_direction"])
        self.assert_publication_not_applicable(transaction)
        self.assertIsNone(descriptor)

    def test_stationary_acceptable_direct_explicitly_exhausts_recovery(self):
        (
            _core,
            _ledger,
            _pre,
            _direct,
            _memories,
            _updated,
            transaction,
            _descriptor,
        ) = self.recovery_call(
            i_values=(1.0, 0.5, 0.25), stationary_probe=True
        )
        self.assertTrue(transaction["direct_within_post_main_i_perp"])
        self.assertTrue(transaction["stationary_normal_probe"])
        self.assertEqual(
            transaction["recovery_seed_mode"],
            "ordinary_normal_seed_for_stationarity_exhaustion",
        )
        self.assertTrue(transaction["projected_trial_evaluated"])
        self.assertEqual(
            transaction["accepted_recovery_secant_publication_mode"],
            "ordinary_all",
        )
        self.assertEqual(transaction["accepted_recovery_secants_added"], 2)
        self.assert_publication_closure(transaction, 2)

    def test_adjacent_chord_coherence_selects_only_registered_horizons(self):
        controls = method.Controls(10, 1.0e-3, 0.60, 4)
        energies = np.asarray([0.0, 2.0, 1.0, 0.0])
        straight = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]
        )
        bent = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]
        )

        def displaced_spring(_core, path, _energies):
            result = np.asarray(path, dtype=np.float64).copy()
            result[1, 1] += 0.1
            return result, 0.0, 1

        with mock.patch.object(
            method,
            "improved_spring_reparameterize",
            side_effect=displaced_spring,
        ):
            straight_seed, straight_audit = method.chord_coherent_recovery_seed(
                FakeCore(), straight, energies, controls
            )
            bent_seed, bent_audit = method.chord_coherent_recovery_seed(
                FakeCore(), bent, energies, controls
            )
        self.assertTrue(straight_audit["strong_recovery_selected"])
        self.assertEqual(straight_audit["saddle_spring_weight"], 3.0)
        self.assertFalse(bent_audit["strong_recovery_selected"])
        self.assertEqual(bent_audit["saddle_spring_weight"], 1.0)
        self.assertAlmostEqual(straight_seed[1, 1], 0.3)
        self.assertAlmostEqual(bent_seed[1, 1], 0.1)

    def test_same_cartesian_grid_and_only_three_controls(self):
        self.assertEqual(
            method.ALPHAS,
            (1.0e-3, 2.0e-3),
        )
        self.assertEqual(method.ALPHA_LABELS, ("1e-3", "2e-3"))
        self.assertTrue(all("." not in label for label in method.ALPHA_LABELS))
        self.assertEqual(method.R_VALUES, (0.60, 0.65))
        self.assertTrue(all(value < 0.7 for value in method.R_VALUES))
        self.assertEqual(method.M_MAX_VALUES, (4,))
        self.assertEqual(len(method.REGISTERED_CONTROLS), 8)
        by_resolution = {
            n: {
                (alpha, ratio, m_max)
                for resolution, alpha, ratio, m_max in method.REGISTERED_CONTROLS
                if resolution == n
            }
            for n in method.RESOLUTIONS
        }
        self.assertEqual(by_resolution[10], by_resolution[20])
        self.assertEqual(by_resolution[10], set(method.GRID_TRIPLES))
        self.assertGreaterEqual(min(method.M_MAX_VALUES), 4)
        self.assertEqual(
            tuple(method.Controls.__dataclass_fields__),
            ("n_internal", "alpha", "r", "m_max"),
        )
        with self.assertRaisesRegex(method.ContractError, "at least 4"):
            method.Controls(10, method.ALPHAS[0], method.R_VALUES[0], 3).validate()

    def test_only_recovery_and_provenance_functions_differ_from_0fafbb(self):
        self.assertEqual(
            runner.sha256(FROZEN_CHORD_COHERENCE),
            "0fafbb8127d5eb4572ba7248de81317e833c0f614d7b342d2939fa93479349c0",
        )

        def definitions(path, node_type):
            tree = ast.parse(Path(path).read_text(encoding="utf-8"))
            return {
                node.name: ast.dump(node, include_attributes=False)
                for node in tree.body
                if isinstance(node, node_type)
            }

        current_functions = definitions(method.THIS_FILE, ast.FunctionDef)
        frozen_functions = definitions(
            FROZEN_CHORD_COHERENCE, ast.FunctionDef
        )
        allowed_function_changes = {
            "validate_preregistration",
            "publish_accepted_recovery_secants",
            "single_synchronized_recovery",
        }
        provenance_helpers = {
            "outer_state_chain_log",
            "set_evaluation_trace_context",
            "transaction_state_chain_log",
            "validate_resolution_gate_token",
        }
        self.assertEqual(
            set(current_functions) - provenance_helpers,
            set(frozen_functions),
        )
        self.assertEqual(
            set(current_functions) - set(frozen_functions),
            provenance_helpers,
        )
        observed_changes = {
            name
            for name in set(current_functions) & set(frozen_functions)
            if current_functions[name] != frozen_functions[name]
        }
        self.assertEqual(
            observed_changes,
            allowed_function_changes
            | {"make_evaluators", "run", "run_one", "save_result"},
        )

        current_tree = ast.parse(method.THIS_FILE.read_text(encoding="utf-8"))
        frozen_tree = ast.parse(
            FROZEN_CHORD_COHERENCE.read_text(encoding="utf-8")
        )
        current_run = next(
            node
            for node in current_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        frozen_run = next(
            node
            for node in frozen_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )

        class RemoveRunLogging(ast.NodeTransformer):
            def __init__(self):
                self.state_chain_removed = []
                self.context_calls_removed = []
                self.return_trace_values_removed = []

            def visit_Expr(self, node):
                value = node.value
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Name)
                    and value.func.id == "set_evaluation_trace_context"
                    and len(value.args) == 3
                    and not value.keywords
                ):
                    self.context_calls_removed.append(
                        ast.dump(value, include_attributes=False)
                    )
                    return None
                if (
                    isinstance(value, ast.Call)
                    and isinstance(value.func, ast.Attribute)
                    and value.func.attr == "update"
                    and isinstance(value.func.value, ast.Name)
                    and len(value.args) == 1
                    and not value.keywords
                    and isinstance(value.args[0], ast.Call)
                    and isinstance(value.args[0].func, ast.Name)
                    and (
                        (
                            value.func.value.id == "transaction"
                            and value.args[0].func.id
                            == "transaction_state_chain_log"
                        )
                        or (
                            value.func.value.id == "stage_audit"
                            and value.args[0].func.id
                            == "outer_state_chain_log"
                        )
                    )
                ):
                    self.state_chain_removed.append(value.args[0].func.id)
                    return None
                return self.generic_visit(node)

            def visit_Dict(self, node):
                node = self.generic_visit(node)
                retained_keys = []
                retained_values = []
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "evaluation_miss_trace"
                    ):
                        self.return_trace_values_removed.append(
                            ast.dump(value, include_attributes=False)
                        )
                    else:
                        retained_keys.append(key)
                        retained_values.append(value)
                node.keys = retained_keys
                node.values = retained_values
                return node

        normalizer = RemoveRunLogging()
        current_run = normalizer.visit(copy.deepcopy(current_run))
        self.assertEqual(
            normalizer.state_chain_removed,
            ["outer_state_chain_log", "transaction_state_chain_log"],
        )
        expected_context_calls = [
            ast.dump(
                ast.parse(source).body[0].value,
                include_attributes=False,
            )
            for source in (
                'set_evaluation_trace_context(ledger, 0, "initial")',
                'set_evaluation_trace_context(ledger, outer, "main")',
                'set_evaluation_trace_context(ledger, outer, "direct")',
                'set_evaluation_trace_context(ledger, outer, "projected")',
            )
        ]
        self.assertEqual(
            normalizer.context_calls_removed, expected_context_calls
        )
        expected_return_trace = ast.dump(
            ast.parse(
                "[dict(item) for item in ledger.evaluation_miss_trace]",
                mode="eval",
            ).body,
            include_attributes=False,
        )
        self.assertEqual(
            normalizer.return_trace_values_removed, [expected_return_trace]
        )
        self.assertEqual(
            ast.dump(current_run, include_attributes=False),
            ast.dump(frozen_run, include_attributes=False),
        )

        current_make_evaluators = next(
            node
            for node in current_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "make_evaluators"
        )
        frozen_make_evaluators = next(
            node
            for node in frozen_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "make_evaluators"
        )
        current_make_evaluators = copy.deepcopy(current_make_evaluators)
        evaluator_calls = [
            node
            for node in ast.walk(current_make_evaluators)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "Evaluator"
        ]
        self.assertEqual(len(evaluator_calls), 1)
        image_keywords = [
            keyword
            for keyword in evaluator_calls[0].keywords
            if keyword.arg == "image_index"
        ]
        self.assertEqual(len(image_keywords), 1)
        self.assertEqual(
            ast.dump(image_keywords[0].value, include_attributes=False),
            ast.dump(ast.Name(id="index", ctx=ast.Load()), include_attributes=False),
        )
        evaluator_calls[0].keywords = [
            keyword
            for keyword in evaluator_calls[0].keywords
            if keyword.arg != "image_index"
        ]
        self.assertEqual(
            ast.dump(current_make_evaluators, include_attributes=False),
            ast.dump(frozen_make_evaluators, include_attributes=False),
        )

        current_run_one = next(
            node
            for node in current_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_one"
        )
        frozen_run_one = next(
            node
            for node in frozen_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_one"
        )

        class RemoveGateTokenProvenance(ast.NodeTransformer):
            def __init__(self):
                self.assignment_removed = 0
                self.keyword_removed = 0

            def visit_Assign(self, node):
                if (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "gate_token"
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "validate_resolution_gate_token"
                ):
                    self.assignment_removed += 1
                    return None
                return self.generic_visit(node)

            def visit_Call(self, node):
                node = self.generic_visit(node)
                if isinstance(node.func, ast.Name) and node.func.id == "save_result":
                    retained = []
                    for keyword in node.keywords:
                        if keyword.arg == "n20_gate_manifest_token":
                            self.keyword_removed += 1
                        else:
                            retained.append(keyword)
                    node.keywords = retained
                return node

        self.assertEqual(
            [argument.arg for argument in current_run_one.args.kwonlyargs],
            ["n20_gate_manifest_token"],
        )
        current_run_one.args.kwonlyargs = []
        current_run_one.args.kw_defaults = []
        token_normalizer = RemoveGateTokenProvenance()
        current_run_one = token_normalizer.visit(current_run_one)
        self.assertEqual(token_normalizer.assignment_removed, 1)
        self.assertEqual(token_normalizer.keyword_removed, 1)
        self.assertEqual(
            ast.dump(current_run_one, include_attributes=False),
            ast.dump(frozen_run_one, include_attributes=False),
        )

        current_save = next(
            node
            for node in current_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "save_result"
        )
        frozen_save = next(
            node
            for node in frozen_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "save_result"
        )
        current_save = copy.deepcopy(current_save)
        self.assertEqual(
            [argument.arg for argument in current_save.args.kwonlyargs],
            ["n20_gate_manifest_token"],
        )
        self.assertEqual(current_save.args.kw_defaults, [None])
        current_save.args.kwonlyargs = []
        current_save.args.kw_defaults = []
        expected_gate_assignment = ast.dump(
            ast.parse(
                "gate_token = validate_resolution_gate_token(controls, "
                "n20_gate_manifest_token)"
            ).body[0],
            include_attributes=False,
        )
        gate_assignments = [
            statement
            for statement in current_save.body
            if ast.dump(statement, include_attributes=False)
            == expected_gate_assignment
        ]
        self.assertEqual(len(gate_assignments), 1)
        current_save.body = [
            statement
            for statement in current_save.body
            if ast.dump(statement, include_attributes=False)
            != expected_gate_assignment
        ]

        def assigned_dict(function, target_name):
            matches = [
                statement.value
                for statement in function.body
                if isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id == target_name
                and isinstance(statement.value, ast.Dict)
            ]
            self.assertEqual(len(matches), 1)
            return matches[0]

        def string_key_map(dictionary):
            return {
                key.value: value
                for key, value in zip(dictionary.keys, dictionary.values)
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
            }

        current_payload = assigned_dict(current_save, "payload")
        frozen_payload = assigned_dict(frozen_save, "payload")
        current_payload_map = string_key_map(current_payload)
        frozen_payload_map = string_key_map(frozen_payload)
        expected_extra_payload = {
            "evaluation_miss_trace_json",
            "n20_gate_manifest_sha256_token",
            "recovery_secant_conservative_publication",
            "recovery_secant_normalized_curvature_formula",
            "recovery_secant_ordinary_publication",
            "recovery_secant_strong_publication",
            "recovery_secant_tie_break",
        }
        self.assertEqual(
            set(current_payload_map) - set(frozen_payload_map),
            expected_extra_payload,
        )
        self.assertEqual(set(frozen_payload_map) - set(current_payload_map), set())
        changed_shared_payload = {
            name
            for name in set(current_payload_map) & set(frozen_payload_map)
            if ast.dump(current_payload_map[name], include_attributes=False)
            != ast.dump(frozen_payload_map[name], include_attributes=False)
        }
        self.assertEqual(
            changed_shared_payload,
            {"recovery_definition", "rollback_commit_memory_transport"},
        )
        expected_changed_recovery_provenance = {
            "recovery_definition": (
                "zero-NoF adjacent-chord-coherent saddle seed, direct-energy "
                "improved-tangent chord-Newton projection, one evaluated "
                "projected band, unchanged atomic gate, coherence-conditioned "
                "accepted physical secant publication"
            ),
            "rollback_commit_memory_transport": (
                "publish the accepted pre-main snapshot plus "
                "coherence-conditioned curvature-certified "
                "direct-to-projected secants; rejected recovery trials never "
                "alter main memory"
            ),
        }
        for name, expected_text in expected_changed_recovery_provenance.items():
            self.assertEqual(
                ast.dump(current_payload_map[name], include_attributes=False),
                ast.dump(
                    ast.parse(
                        f"np.asarray({expected_text!r})", mode="eval"
                    ).body,
                    include_attributes=False,
                ),
                name,
            )
        expected_trace_serialization = ast.dump(
            ast.parse(
                "np.asarray([json.dumps(item, sort_keys=True) "
                'for item in result["evaluation_miss_trace"]])',
                mode="eval",
            ).body,
            include_attributes=False,
        )
        self.assertEqual(
            ast.dump(
                current_payload_map["evaluation_miss_trace_json"],
                include_attributes=False,
            ),
            expected_trace_serialization,
        )
        self.assertEqual(
            ast.dump(
                current_payload_map["n20_gate_manifest_sha256_token"],
                include_attributes=False,
            ),
            ast.dump(
                ast.parse("np.asarray(gate_token)", mode="eval").body,
                include_attributes=False,
            ),
        )
        expected_recovery_provenance = {
            "recovery_secant_normalized_curvature_formula": (
                "(s^T y)/(||s|| ||y||)"
            ),
            "recovery_secant_strong_publication": (
                "highest normalized-curvature certified pair only"
            ),
            "recovery_secant_conservative_publication": (
                "all curvature-certified pairs"
            ),
            "recovery_secant_ordinary_publication": (
                "all curvature-certified pairs"
            ),
            "recovery_secant_tie_break": (
                "smaller 1-based internal image index"
            ),
        }
        for name, expected_text in expected_recovery_provenance.items():
            self.assertEqual(
                ast.dump(current_payload_map[name], include_attributes=False),
                ast.dump(
                    ast.parse(
                        f"np.asarray({expected_text!r})", mode="eval"
                    ).body,
                    include_attributes=False,
                ),
                name,
            )

        retained_payload_keys = []
        retained_payload_values = []
        for key, value in zip(current_payload.keys, current_payload.values):
            name = (
                key.value
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                else None
            )
            if name in expected_extra_payload:
                continue
            retained_payload_keys.append(key)
            retained_payload_values.append(
                copy.deepcopy(frozen_payload_map[name])
                if name in changed_shared_payload
                else value
            )
        current_payload.keys = retained_payload_keys
        current_payload.values = retained_payload_values

        current_return = next(
            statement.value
            for statement in reversed(current_save.body)
            if isinstance(statement, ast.Return)
        )
        self.assertIsInstance(current_return, ast.Dict)
        receipt_token_values = [
            value
            for key, value in zip(current_return.keys, current_return.values)
            if isinstance(key, ast.Constant)
            and key.value == "n20_gate_manifest_sha256_token"
        ]
        self.assertEqual(len(receipt_token_values), 1)
        self.assertEqual(
            ast.dump(receipt_token_values[0], include_attributes=False),
            ast.dump(ast.Name(id="gate_token", ctx=ast.Load()), include_attributes=False),
        )
        retained_return_keys = []
        retained_return_values = []
        for key, value in zip(current_return.keys, current_return.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "n20_gate_manifest_sha256_token"
            ):
                continue
            retained_return_keys.append(key)
            retained_return_values.append(value)
        current_return.keys = retained_return_keys
        current_return.values = retained_return_values
        self.assertEqual(
            ast.dump(current_save, include_attributes=False),
            ast.dump(frozen_save, include_attributes=False),
        )

        for name in sorted(
            set(frozen_functions)
            - allowed_function_changes
            - {"make_evaluators", "run", "run_one", "save_result"}
        ):
            self.assertEqual(current_functions[name], frozen_functions[name], name)

        outer_logging_helper = next(
            node
            for node in current_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "outer_state_chain_log"
        )
        self.assertEqual(
            [argument.arg for argument in outer_logging_helper.args.args],
            ["outer_start_path", "post_main_state"],
        )
        helper_calls = {
            node.func.id
            for node in ast.walk(outer_logging_helper)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertEqual(helper_calls, {"all", "ContractError"})
        self.assertNotIn("evaluate_path", helper_calls)
        self.assertNotIn("query", helper_calls)
        helper_bound = {
            argument.arg for argument in outer_logging_helper.args.args
        }
        helper_bound.update(
            node.id
            for node in ast.walk(outer_logging_helper)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Param))
        )
        helper_loaded = {
            node.id
            for node in ast.walk(outer_logging_helper)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertEqual(
            helper_loaded - helper_bound, {"np", "all", "ContractError"}
        )
        self.assertEqual(
            {
                node.attr
                for node in ast.walk(outer_logging_helper)
                if isinstance(node, ast.Attribute)
            },
            {
                "all",
                "array_equal",
                "asarray",
                "float64",
                "isfinite",
                "ndim",
                "shape",
                "tolist",
            },
        )
        self.assertFalse(
            any(
                isinstance(node, (ast.Subscript, ast.Attribute))
                and isinstance(node.ctx, ast.Store)
                for node in ast.walk(outer_logging_helper)
            )
        )

        transaction_logging_helper = next(
            node
            for node in current_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "transaction_state_chain_log"
        )
        self.assertEqual(
            [argument.arg for argument in transaction_logging_helper.args.args],
            ["outer_start_path", "post_main_state", "committed_public_path"],
        )
        transaction_bound = {
            argument.arg for argument in transaction_logging_helper.args.args
        }
        transaction_bound.update(
            node.id
            for node in ast.walk(transaction_logging_helper)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Param))
        )
        transaction_loaded = {
            node.id
            for node in ast.walk(transaction_logging_helper)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertEqual(
            transaction_loaded - transaction_bound,
            {"np", "ContractError", "outer_state_chain_log"},
        )
        transaction_subscript_stores = [
            node
            for node in ast.walk(transaction_logging_helper)
            if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
        ]
        self.assertEqual(len(transaction_subscript_stores), 1)
        local_payload_store = transaction_subscript_stores[0]
        self.assertIsInstance(local_payload_store.value, ast.Name)
        self.assertEqual(local_payload_store.value.id, "payload")
        payload_key = local_payload_store.slice
        if isinstance(payload_key, ast.Index):
            payload_key = payload_key.value
        self.assertIsInstance(payload_key, ast.Constant)
        self.assertEqual(
            payload_key.value, "committed_public_path"
        )
        self.assertFalse(
            any(
                isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)
                for node in ast.walk(transaction_logging_helper)
            )
        )

        gate_validator = next(
            node
            for node in current_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "validate_resolution_gate_token"
        )
        self.assertEqual(
            [argument.arg for argument in gate_validator.args.args],
            ["controls", "token"],
        )
        gate_bound = {argument.arg for argument in gate_validator.args.args}
        gate_bound.update(
            node.id
            for node in ast.walk(gate_validator)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Param))
        )
        gate_loaded = {
            node.id
            for node in ast.walk(gate_validator)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        self.assertEqual(
            gate_loaded - gate_bound,
            {
                "ContractError",
                "N20_GATE_MANIFEST",
                "_sha256",
                "any",
                "int",
                "isinstance",
                "len",
                "str",
            },
        )

        trace_context_helper = next(
            node
            for node in current_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "set_evaluation_trace_context"
        )
        expected_trace_context_helper = ast.parse(
            '''def set_evaluation_trace_context(ledger, outer, phase):
    """Label subsequent miss records without changing evaluation control flow."""

    ledger.trace_outer = int(outer)
    ledger.trace_phase = str(phase)
'''
        ).body[0]
        self.assertEqual(
            ast.dump(trace_context_helper, include_attributes=False),
            ast.dump(expected_trace_context_helper, include_attributes=False),
        )

        current_classes = definitions(method.THIS_FILE, ast.ClassDef)
        frozen_classes = definitions(FROZEN_CHORD_COHERENCE, ast.ClassDef)
        self.assertEqual(set(current_classes), set(frozen_classes))
        self.assertEqual(
            {
                name: value
                for name, value in current_classes.items()
                if name not in {"Ledger", "Evaluator"}
            },
            {
                name: value
                for name, value in frozen_classes.items()
                if name not in {"Ledger", "Evaluator"}
            },
        )

        current_ledger = next(
            node
            for node in current_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Ledger"
        )
        frozen_ledger = next(
            node
            for node in frozen_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Ledger"
        )
        expected_ledger_fields_node = ast.parse(
            '''class ExpectedLedgerFields:
    evaluation_miss_trace: list = field(default_factory=list)
    trace_outer: int = -1
    trace_phase: str = "unassigned"
'''
        ).body[0]
        expected_ledger_fields = {
            statement.target.id: ast.dump(statement, include_attributes=False)
            for statement in expected_ledger_fields_node.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
        }
        current_ledger_field_statements = [
            statement
            for statement in current_ledger.body
            if isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id in expected_ledger_fields
        ]
        self.assertEqual(
            [statement.target.id for statement in current_ledger_field_statements],
            [
                "evaluation_miss_trace",
                "trace_outer",
                "trace_phase",
            ],
        )
        self.assertEqual(
            [
                ast.dump(statement, include_attributes=False)
                for statement in current_ledger_field_statements
            ],
            [
                expected_ledger_fields[name]
                for name in (
                    "evaluation_miss_trace",
                    "trace_outer",
                    "trace_phase",
                )
            ],
        )
        field_calls = [
            node
            for node in ast.walk(current_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "field"
        ]
        self.assertEqual(len(field_calls), 1)
        self.assertEqual(
            ast.dump(field_calls[0], include_attributes=False),
            ast.dump(
                ast.parse("field(default_factory=list)", mode="eval").body,
                include_attributes=False,
            ),
        )
        normalized_ledger = copy.deepcopy(current_ledger)
        normalized_ledger.body = [
            statement
            for statement in normalized_ledger.body
            if not (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id in expected_ledger_fields
            )
        ]
        self.assertEqual(
            ast.dump(normalized_ledger, include_attributes=False),
            ast.dump(frozen_ledger, include_attributes=False),
        )

        current_evaluator = next(
            node
            for node in current_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Evaluator"
        )
        frozen_evaluator = next(
            node
            for node in frozen_tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Evaluator"
        )
        normalized_evaluator = copy.deepcopy(current_evaluator)
        current_init = next(
            node
            for node in normalized_evaluator.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        self.assertEqual(
            [argument.arg for argument in current_init.args.args],
            ["self", "energy_gradient", "ledger", "charged", "image_index"],
        )
        self.assertEqual(
            [ast.dump(item, include_attributes=False) for item in current_init.args.defaults],
            [
                ast.dump(
                    ast.parse("None", mode="eval").body,
                    include_attributes=False,
                )
            ],
        )
        current_init.args.args = current_init.args.args[:-1]
        current_init.args.defaults = []
        expected_image_assignment = ast.dump(
            ast.parse(
                "self.image_index = -1 if image_index is None else int(image_index)"
            ).body[0],
            include_attributes=False,
        )
        image_assignments = [
            statement
            for statement in current_init.body
            if ast.dump(statement, include_attributes=False)
            == expected_image_assignment
        ]
        self.assertEqual(len(image_assignments), 1)
        current_init.body = [
            statement
            for statement in current_init.body
            if ast.dump(statement, include_attributes=False)
            != expected_image_assignment
        ]

        current_query = next(
            node
            for node in normalized_evaluator.body
            if isinstance(node, ast.FunctionDef) and node.name == "query"
        )
        frozen_query = next(
            node
            for node in frozen_evaluator.body
            if isinstance(node, ast.FunctionDef) and node.name == "query"
        )

        def charged_if(function):
            matches = [
                statement
                for statement in function.body
                if isinstance(statement, ast.If)
                and isinstance(statement.test, ast.Attribute)
                and isinstance(statement.test.value, ast.Name)
                and statement.test.value.id == "self"
                and statement.test.attr == "charged"
            ]
            self.assertEqual(len(matches), 1)
            return matches[0]

        current_charged = charged_if(current_query)
        frozen_charged = charged_if(frozen_query)
        self.assertEqual(len(current_charged.body), len(frozen_charged.body) + 1)
        self.assertEqual(
            [ast.dump(item, include_attributes=False) for item in current_charged.body[:-1]],
            [ast.dump(item, include_attributes=False) for item in frozen_charged.body],
        )
        expected_trace_append = ast.parse(
            '''self.ledger.evaluation_miss_trace.append(
    {
        "sequence": int(len(self.ledger.evaluation_miss_trace) + 1),
        "article_nf": int(self.ledger.total),
        "outer": int(self.ledger.trace_outer),
        "phase": str(self.ledger.trace_phase),
        "image_index": int(self.image_index),
        "coordinate_sha256": hashlib.sha256(key).hexdigest(),
    }
)'''
        ).body[0]
        self.assertEqual(
            ast.dump(current_charged.body[-1], include_attributes=False),
            ast.dump(expected_trace_append, include_attributes=False),
        )
        current_charged.body = current_charged.body[:-1]
        self.assertEqual(
            ast.dump(normalized_evaluator, include_attributes=False),
            ast.dump(frozen_evaluator, include_attributes=False),
        )

    def test_module_level_algorithm_and_executable_ast_matches_0fafbb(self):
        allowed_assignment_differences = {
            "EXPERIMENT_ID",
            "ALPHA_LABELS",
            "R_VALUES",
            "M_MAX_VALUES",
            "FROZEN_CHORD_COHERENCE_SOURCE",
            "AUDIT_EDGE_TEST_FILE",
            "N20_GATE_MANIFEST",
            "V1_FORMAL_REGISTRATION",
            "V1_FORMAL_CORE_SOURCE",
            "V1_FORMAL_AUDIT_SOURCE",
            "V1_FORMAL_N20_EVIDENCE_RESULT",
            "V1_FORMAL_N20_EVIDENCE_RECEIPT",
            "V1_FORMAL_N20_GATE_MANIFEST",
            "V1_FORMAL_N20_GATE_REPORT",
            "REQUIRED_DEPENDENCY_PATHS",
        }

        def assignment_name(node):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                return None
            target = node.targets[0]
            return target.id if isinstance(target, ast.Name) else None

        def module_contract(path, *, current_source):
            tree = ast.parse(Path(path).read_text(encoding="utf-8"))
            retained = []
            assignments = {}
            docstrings = 0
            dataclass_imports = 0
            for index, node in enumerate(tree.body):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if (
                    index == 0
                    and isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    docstrings += 1
                    continue
                name = assignment_name(node)
                if name is not None:
                    self.assertNotIn(name, assignments, "duplicate module assignment")
                    assignments[name] = ast.dump(node, include_attributes=False)
                    if name in allowed_assignment_differences:
                        continue
                if isinstance(node, ast.ImportFrom) and node.module == "dataclasses":
                    dataclass_imports += 1
                    observed = [alias.name for alias in node.names]
                    self.assertEqual(
                        observed,
                        ["dataclass", "field"]
                        if current_source
                        else ["dataclass"],
                    )
                    normalized = ast.ImportFrom(
                        module="dataclasses",
                        names=[ast.alias(name="dataclass")],
                        level=0,
                    )
                    retained.append(
                        ast.dump(normalized, include_attributes=False)
                    )
                    continue
                retained.append(ast.dump(node, include_attributes=False))
            self.assertEqual(docstrings, 1)
            self.assertEqual(dataclass_imports, 1)
            return retained, assignments

        current, current_assignments = module_contract(
            method.THIS_FILE, current_source=True
        )
        frozen, frozen_assignments = module_contract(
            FROZEN_CHORD_COHERENCE, current_source=False
        )
        self.assertEqual(current, frozen)
        observed_new = set(current_assignments) - set(frozen_assignments)
        observed_changed = {
            name
            for name in set(current_assignments) & set(frozen_assignments)
            if current_assignments[name] != frozen_assignments[name]
        }
        self.assertEqual(
            observed_new,
            {
                "FROZEN_CHORD_COHERENCE_SOURCE",
                "AUDIT_EDGE_TEST_FILE",
                "N20_GATE_MANIFEST",
                "V1_FORMAL_REGISTRATION",
                "V1_FORMAL_CORE_SOURCE",
                "V1_FORMAL_AUDIT_SOURCE",
                "V1_FORMAL_N20_EVIDENCE_RESULT",
                "V1_FORMAL_N20_EVIDENCE_RECEIPT",
                "V1_FORMAL_N20_GATE_MANIFEST",
                "V1_FORMAL_N20_GATE_REPORT",
            },
        )
        self.assertEqual(
            observed_changed,
            {
                "EXPERIMENT_ID",
                "ALPHA_LABELS",
                "R_VALUES",
                "M_MAX_VALUES",
                "REQUIRED_DEPENDENCY_PATHS",
            },
        )
        self.assertEqual(
            observed_new | observed_changed,
            allowed_assignment_differences,
        )

        algorithm_constants = (
            "ARTICLE_NF_BUDGET",
            "OUTER_LIMIT",
            "RESOLUTIONS",
            "HISTORY_SIZE",
            "ARMIJO",
            "BACKTRACK",
            "KKT_ATOL",
            "ROUNDING_FACTOR",
            "REPARAMETERIZATION_PERIOD",
            "TRUST_RADIUS_FRACTION",
            "TRUST_RADIUS_MAXIMUM",
            "SPRING_DTAU",
            "SPRING_MAX_ITER",
            "SPRING_TOLERANCE",
            "RECOVERY_EQUIDISTANCE_CONTRACTION",
            "RECOVERY_IPERP_ATOL",
            "RECOVERY_EQUIDISTANCE_ATOL",
            "TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT",
            "RECOVERY_BAND_EVALUATION_CAP",
            "RECOVERY_SECANT_CURVATURE",
            "THREAD_ENVIRONMENT",
        )
        for name in algorithm_constants:
            self.assertEqual(
                current_assignments[name], frozen_assignments[name], name
            )

    def test_v2_differs_from_formal_v1_only_by_campaign_provenance_metadata(self):
        """Keep the v2 numerical method executable AST identical to formal v1."""

        def named_definitions(tree, node_type):
            return {
                node.name: node
                for node in tree.body
                if isinstance(node, node_type)
            }

        def assignment_name(node):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                return None
            target = node.targets[0]
            return target.id if isinstance(target, ast.Name) else None

        class NormalizeLockLiteral(ast.NodeTransformer):
            LOCKS = {
                "locks/host-local-q10-irivm-low-r-coherence-secant.lock",
                "locks/host-local-q10-irivm-low-r-coherence-secant-v2.lock",
                "host-local-q10-irivm-low-r-coherence-secant.lock",
                "host-local-q10-irivm-low-r-coherence-secant-v2.lock",
            }

            def visit_Constant(self, node):
                if isinstance(node.value, str) and node.value in self.LOCKS:
                    return ast.copy_location(
                        ast.Constant(value="__CAMPAIGN_LOCAL_LOCK__"), node
                    )
                return node

        class RemoveRawDirectionLogging(ast.NodeTransformer):
            """Remove exactly the two logging-only raw-direction AST nodes."""

            def __init__(self):
                self.default_values = []
                self.assignment_values = []
                self.assignment_lines = []

            @staticmethod
            def is_log_target(node):
                if not (
                    isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "transaction"
                ):
                    return False
                index = node.slice
                if isinstance(index, ast.Index):
                    index = index.value
                return (
                    isinstance(index, ast.Constant)
                    and index.value == "raw_pre_addition_direction"
                )

            def visit_Dict(self, node):
                retained_keys = []
                retained_values = []
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "raw_pre_addition_direction"
                    ):
                        self.default_values.append(
                            ast.dump(value, include_attributes=False)
                        )
                    else:
                        retained_keys.append(key)
                        retained_values.append(value)
                node.keys = retained_keys
                node.values = retained_values
                return self.generic_visit(node)

            def visit_Assign(self, node):
                if (
                    len(node.targets) == 1
                    and self.is_log_target(node.targets[0])
                ):
                    self.assignment_values.append(
                        ast.dump(node.value, include_attributes=False)
                    )
                    self.assignment_lines.append(int(node.lineno))
                    return None
                return self.generic_visit(node)

        current_tree = ast.parse(method.THIS_FILE.read_text(encoding="utf-8"))
        v1_tree = ast.parse(V1_FORMAL_CORE.read_text(encoding="utf-8"))
        current_functions = named_definitions(current_tree, ast.FunctionDef)
        v1_functions = named_definitions(v1_tree, ast.FunctionDef)
        self.assertEqual(set(current_functions), set(v1_functions))
        changed_functions = {
            name
            for name in current_functions
            if ast.dump(current_functions[name], include_attributes=False)
            != ast.dump(v1_functions[name], include_attributes=False)
        }
        self.assertEqual(
            changed_functions,
            {"validate_preregistration", "single_synchronized_recovery"},
        )
        normalized_current_validation = NormalizeLockLiteral().visit(
            copy.deepcopy(current_functions["validate_preregistration"])
        )
        normalized_v1_validation = NormalizeLockLiteral().visit(
            copy.deepcopy(v1_functions["validate_preregistration"])
        )
        self.assertEqual(
            ast.dump(normalized_current_validation, include_attributes=False),
            ast.dump(normalized_v1_validation, include_attributes=False),
        )
        current_recovery = copy.deepcopy(
            current_functions["single_synchronized_recovery"]
        )
        raw_log_normalizer = RemoveRawDirectionLogging()
        normalized_current_recovery = raw_log_normalizer.visit(current_recovery)
        self.assertEqual(
            raw_log_normalizer.default_values,
            [
                ast.dump(
                    ast.parse("None", mode="eval").body,
                    include_attributes=False,
                )
            ],
        )
        expected_log_value = ast.parse(
            "np.asarray(directions, dtype=np.float64).tolist()", mode="eval"
        ).body
        self.assertEqual(
            raw_log_normalizer.assignment_values,
            [ast.dump(expected_log_value, include_attributes=False)],
        )
        self.assertEqual(len(raw_log_normalizer.assignment_lines), 1)
        candidate_path_additions = [
            node
            for node in ast.walk(
                current_functions["single_synchronized_recovery"]
            )
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "candidate_path"
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Add)
            and isinstance(node.value.left, ast.Name)
            and node.value.left.id == "path"
            and isinstance(node.value.right, ast.Name)
            and node.value.right.id == "candidate_displacement"
        ]
        self.assertEqual(len(candidate_path_additions), 1)
        self.assertLess(
            raw_log_normalizer.assignment_lines[0],
            int(candidate_path_additions[0].lineno),
        )
        self.assertEqual(
            ast.dump(normalized_current_recovery, include_attributes=False),
            ast.dump(
                v1_functions["single_synchronized_recovery"],
                include_attributes=False,
            ),
        )

        current_classes = named_definitions(current_tree, ast.ClassDef)
        v1_classes = named_definitions(v1_tree, ast.ClassDef)
        self.assertEqual(set(current_classes), set(v1_classes))
        for name in current_classes:
            self.assertEqual(
                ast.dump(current_classes[name], include_attributes=False),
                ast.dump(v1_classes[name], include_attributes=False),
                name,
            )

        current_assignments = {
            assignment_name(node): node
            for node in current_tree.body
            if assignment_name(node) is not None
        }
        v1_assignments = {
            assignment_name(node): node
            for node in v1_tree.body
            if assignment_name(node) is not None
        }
        evidence_metadata = {
            "V1_FORMAL_REGISTRATION",
            "V1_FORMAL_CORE_SOURCE",
            "V1_FORMAL_AUDIT_SOURCE",
            "V1_FORMAL_N20_EVIDENCE_RESULT",
            "V1_FORMAL_N20_EVIDENCE_RECEIPT",
            "V1_FORMAL_N20_GATE_MANIFEST",
            "V1_FORMAL_N20_GATE_REPORT",
        }
        self.assertEqual(
            set(current_assignments) - set(v1_assignments), evidence_metadata
        )
        changed_assignments = {
            name
            for name in set(current_assignments) & set(v1_assignments)
            if ast.dump(current_assignments[name], include_attributes=False)
            != ast.dump(v1_assignments[name], include_attributes=False)
        }
        self.assertEqual(
            changed_assignments, {"EXPERIMENT_ID", "REQUIRED_DEPENDENCY_PATHS"}
        )
        self.assertEqual(
            method.EXPERIMENT_ID,
            "q10-irivm-low-r-coherence-secant-v2-20260830",
        )
        self.assertEqual(
            {
                method.V1_FORMAL_REGISTRATION,
                method.V1_FORMAL_CORE_SOURCE,
                method.V1_FORMAL_AUDIT_SOURCE,
                method.V1_FORMAL_N20_EVIDENCE_RESULT,
                method.V1_FORMAL_N20_EVIDENCE_RECEIPT,
                method.V1_FORMAL_N20_GATE_MANIFEST,
                method.V1_FORMAL_N20_GATE_REPORT,
            },
            set(method.REQUIRED_DEPENDENCY_PATHS) & {
                method.V1_FORMAL_REGISTRATION,
                method.V1_FORMAL_CORE_SOURCE,
                method.V1_FORMAL_AUDIT_SOURCE,
                method.V1_FORMAL_N20_EVIDENCE_RESULT,
                method.V1_FORMAL_N20_EVIDENCE_RECEIPT,
                method.V1_FORMAL_N20_GATE_MANIFEST,
                method.V1_FORMAL_N20_GATE_REPORT,
            },
        )
        for path in (
            method.V1_FORMAL_REGISTRATION,
            method.V1_FORMAL_CORE_SOURCE,
            method.V1_FORMAL_AUDIT_SOURCE,
            method.V1_FORMAL_N20_EVIDENCE_RESULT,
            method.V1_FORMAL_N20_EVIDENCE_RECEIPT,
            method.V1_FORMAL_N20_GATE_MANIFEST,
            method.V1_FORMAL_N20_GATE_REPORT,
        ):
            self.assertTrue(path.is_file(), str(path))

        current_runner_tree = ast.parse((SRC / "run_grid.py").read_text())
        v1_runner_tree = ast.parse(V1_FORMAL_RUNNER.read_text())
        current_runner_functions = named_definitions(
            current_runner_tree, ast.FunctionDef
        )
        v1_runner_functions = named_definitions(v1_runner_tree, ast.FunctionDef)
        self.assertEqual(set(current_runner_functions), set(v1_runner_functions))
        changed_runner_functions = {
            name
            for name in current_runner_functions
            if ast.dump(current_runner_functions[name], include_attributes=False)
            != ast.dump(v1_runner_functions[name], include_attributes=False)
        }
        self.assertEqual(changed_runner_functions, {"main"})
        normalized_current_main = NormalizeLockLiteral().visit(
            copy.deepcopy(current_runner_functions["main"])
        )
        normalized_v1_main = NormalizeLockLiteral().visit(
            copy.deepcopy(v1_runner_functions["main"])
        )
        self.assertEqual(
            ast.dump(normalized_current_main, include_attributes=False),
            ast.dump(normalized_v1_main, include_attributes=False),
        )

    def test_publisher_ast_signature_and_free_name_contract(self):
        tree = ast.parse(method.THIS_FILE.read_text(encoding="utf-8"))
        publisher = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "publish_accepted_recovery_secants"
        )
        self.assertEqual(
            [argument.arg for argument in publisher.args.args],
            ["memories", "direct_state", "projected_state"],
        )
        self.assertEqual(
            [argument.arg for argument in publisher.args.kwonlyargs],
            ["strong_recovery_selected", "stationary_normal_probe"],
        )
        self.assertEqual(publisher.args.posonlyargs, [])
        self.assertIsNone(publisher.args.vararg)
        self.assertIsNone(publisher.args.kwarg)
        self.assertEqual(publisher.args.defaults, [])
        self.assertEqual(publisher.args.kw_defaults, [None, None])
        self.assertEqual(publisher.decorator_list, [])
        self.assertFalse(
            any(
                isinstance(node, (ast.Global, ast.Nonlocal))
                for node in ast.walk(publisher)
            )
        )

        bound_names = {
            argument.arg
            for argument in publisher.args.args + publisher.args.kwonlyargs
        }
        bound_names.update(
            node.id
            for node in ast.walk(publisher)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, (ast.Store, ast.Param))
        )
        loaded_names = {
            node.id
            for node in ast.walk(publisher)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        free_names = loaded_names - bound_names
        self.assertEqual(
            free_names,
            {
                "np",
                "RECOVERY_SECANT_CURVATURE",
                "HISTORY_SIZE",
                "ContractError",
                "bool",
                "float",
                "int",
                "len",
                "list",
                "sorted",
                "min",
                "range",
            },
        )
        self.assertEqual(
            {
                node.attr
                for node in ast.walk(publisher)
                if isinstance(node, ast.Attribute)
            },
            {
                "append",
                "asarray",
                "copy",
                "dot",
                "float64",
                "linalg",
                "norm",
                "ravel",
                "shape",
                "sort",
            },
        )
        predicate_nodes = [
            node.test for node in ast.walk(publisher) if isinstance(node, ast.If)
        ]
        predicate_nodes.extend(
            condition
            for node in ast.walk(publisher)
            if isinstance(node, ast.comprehension)
            for condition in node.ifs
        )
        allowed_predicate_names = bound_names | free_names
        for predicate in predicate_nodes:
            self.assertLessEqual(
                {
                    node.id
                    for node in ast.walk(predicate)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                },
                allowed_predicate_names,
            )

    def test_single_thread_gate_is_in_core_and_runner(self):
        good = {name: "1" for name in method.THREAD_ENVIRONMENT}
        good["PYTHONHASHSEED"] = "0"
        with mock.patch.dict(os.environ, good, clear=True):
            method.validate_single_thread_environment()
        bad = dict(good)
        bad["OPENBLAS_NUM_THREADS"] = "8"
        with mock.patch.dict(os.environ, bad, clear=True):
            with self.assertRaisesRegex(method.ContractError, "OPENBLAS"):
                method.validate_single_thread_environment()
        runner = (SRC / "run_grid.py").read_text(encoding="utf-8")
        self.assertIn("if args.jobs != 1", runner)
        self.assertIn('os.environ.get(name) != "1"', runner)
        self.assertIn('os.environ.get("PYTHONHASHSEED") != "0"', runner)
        self.assertIn("validate_single_thread_environment()", inspect.getsource(method.run))

    def test_runner_requires_explicit_resolution_and_authenticated_n20_gate(self):
        runner_source = inspect.getsource(runner.main)
        self.assertIn("explicit --n-internal 20 or --n-internal 10", runner_source)
        self.assertIn("refresh_n20_gate_audit()", runner_source)
        self.assertIn("validate_n20_gate_for_n10(registration)", runner_source)
        self.assertIn(
            "_, gate_manifest_token = validate_n20_gate_for_n10(registration)",
            runner_source,
        )
        self.assertIn('gate_manifest_token = "not_applicable"', runner_source)
        self.assertIn(
            "n20_gate_manifest_token=gate_manifest_token", runner_source
        )
        self.assertIn(
            'payload["n20_gate_manifest_sha256_token"]', runner_source
        )
        self.assertIn(
            'row.get("n20_gate_manifest_sha256_token")', runner_source
        )
        self.assertLess(
            runner_source.index("refresh_n20_gate_audit()"),
            runner_source.index("validate_n20_gate_for_n10(registration)"),
        )
        self.assertIn(
            "host-local-q10-irivm-low-r-coherence-secant-v2.lock",
            runner_source,
        )
        with mock.patch.object(runner.subprocess, "run") as launched:
            runner.refresh_n20_gate_audit()
        launched.assert_called_once_with(
            [
                sys.executable,
                str(method.AUDIT_FILE),
                "--n-internal",
                "20",
                "--output-dir",
                str(runner.N20_GATE_MANIFEST.parent),
            ],
            cwd=str(method.CAMPAIGN_ROOT),
            check=True,
        )

    def test_core_resolution_gate_token_contract(self):
        n20 = method.Controls(20, 1.0e-3, 0.60, 4)
        n10 = method.Controls(10, 1.0e-3, 0.60, 4)
        self.assertEqual(
            method.validate_resolution_gate_token(n20, "not_applicable"),
            "not_applicable",
        )
        with self.assertRaisesRegex(
            method.ContractError, "N20 gate token must be not_applicable"
        ):
            method.validate_resolution_gate_token(n20, "0" * 64)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "audit_manifest.json"
            manifest.write_text('{"authenticated": true}\n', encoding="utf-8")
            token = runner.sha256(manifest)
            with mock.patch.object(method, "N20_GATE_MANIFEST", manifest):
                self.assertEqual(
                    method.validate_resolution_gate_token(n10, token), token
                )
                with self.assertRaisesRegex(
                    method.ContractError, "token is stale"
                ):
                    method.validate_resolution_gate_token(n10, "0" * 64)
                with self.assertRaisesRegex(
                    method.ContractError, "lowercase 64-hex"
                ):
                    method.validate_resolution_gate_token(n10, "not_applicable")

    def test_n20_gate_rejects_fabricated_aggregate_without_results(self):
        registered_n20 = sum(
            int(item[0]) == 20 for item in method.REGISTERED_CONTROLS
        )
        registration = {
            "campaign_id": method.EXPERIMENT_ID,
            "implementation_sha256": runner.sha256(method.THIS_FILE),
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            registration_path = temporary / "registration.json"
            registration_path.write_text("{}\n", encoding="utf-8")
            manifest_path = temporary / "audit_manifest.json"
            manifest = {
                "campaign_id": method.EXPERIMENT_ID,
                "audit_status": "complete",
                "requested_resolutions": [20],
                "registration": str(registration_path),
                "registration_sha256": runner.sha256(registration_path),
                "core_source": str(method.THIS_FILE),
                "core_source_sha256": runner.sha256(method.THIS_FILE),
                "audit_script": str(method.AUDIT_FILE),
                "audit_script_sha256": runner.sha256(method.AUDIT_FILE),
                "integrity_failure_count": 0,
                "integrity_errors": [],
                "requested_member_count": registered_n20,
                "audited_member_count": registered_n20,
                "integrity_valid_member_count": registered_n20,
                "resolution_results": {
                    "20": {
                        "registered_member_count": registered_n20,
                        "integrity_valid_member_count": registered_n20,
                        "selectable_member_count": registered_n20,
                        "strict_pass_count": 1,
                    }
                },
                "all_requested_resolutions_strictly_pass": True,
            }
            manifest_path.write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with mock.patch.object(runner, "REGISTRATION", registration_path):
                with self.assertRaisesRegex(
                    RuntimeError, "artifact set drifted"
                ):
                    runner.validate_n20_gate_for_n10(
                        registration, manifest_path
                    )

    def test_n20_gate_authenticates_all_results_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            registration_path = (
                campaign / "registrations" / "CAMPAIGN_PREREGISTRATION.json"
            )
            registration_path.parent.mkdir(parents=True)
            registration_path.write_text("{}\n", encoding="utf-8")
            gate_dir = campaign / "reports" / "n20_gate_audit"
            gate_dir.mkdir(parents=True)
            run_dir = campaign / "runs" / "n20"
            receipt_dir = campaign / "reports" / "receipts" / "n20"
            run_dir.mkdir(parents=True)
            receipt_dir.mkdir(parents=True)

            expected_core_sha = runner.sha256(method.THIS_FILE)
            expected_registration_sha = runner.sha256(registration_path)
            registration = {
                "campaign_id": method.EXPERIMENT_ID,
                "implementation_sha256": expected_core_sha,
            }
            controls = [
                runner.controls_from_tuple(item)
                for item in method.REGISTERED_CONTROLS
                if int(item[0]) == 20
            ]
            rows = []
            receipt_paths = []
            for index, control in enumerate(controls):
                source = run_dir / (control.run_id + ".npz")
                per_image = np.zeros(22, dtype=np.int64)
                per_image[1:-1] = 1
                np.savez_compressed(
                    source,
                    experiment_id=np.asarray(method.EXPERIMENT_ID),
                    n20_gate_manifest_sha256_token=np.asarray(
                        "not_applicable"
                    ),
                    run_id=np.asarray(control.run_id),
                    config_json=np.asarray(
                        json.dumps(control.as_dict(), sort_keys=True)
                    ),
                    source_sha256=np.asarray(expected_core_sha),
                    registration_sha256=np.asarray(expected_registration_sha),
                    terminal_article_nf=np.asarray(20, dtype=np.int64),
                    last_complete_article_nf=np.asarray(20, dtype=np.int64),
                    per_image_joint_evaluations=per_image,
                    article_nf=np.asarray([20], dtype=np.int64),
                    article_usable_state=np.asarray([True]),
                    stop_reason=np.asarray("stationary"),
                    failure_message=np.asarray(""),
                )
                source_sha = runner.sha256(source)
                receipt_path = receipt_dir / (control.run_id + ".receipt.json")
                receipt = {
                    "status": "completed",
                    "n20_gate_manifest_sha256_token": "not_applicable",
                    "run_id": control.run_id,
                    "controls": control.as_dict(),
                    "output": str(source),
                    "output_sha256": source_sha,
                    "stop_reason": "stationary",
                    "failure_message": "",
                    "terminal_article_nf": 20,
                    "last_complete_article_nf": 20,
                }
                receipt_path.write_text(
                    json.dumps(receipt, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                receipt_paths.append(receipt_path)
                rows.append(
                    {
                        **control.as_dict(),
                        "run_id": control.run_id,
                        "integrity_valid": True,
                        "integrity_error": "",
                        "source": str(source),
                        "source_sha256": source_sha,
                        "receipt": str(receipt_path),
                        "receipt_sha256": runner.sha256(receipt_path),
                        "stop_reason": "stationary",
                        "failure_message": "",
                        "valid_for_selection": True,
                        "terminal_number_of_forces": 20,
                        "last_complete_number_of_forces": 20,
                        "strict_pass": index == 0,
                    }
                )

            summary_path = gate_dir / "candidate_summary.csv"
            with summary_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            checkpoint_path = gate_dir / "checkpoint_comparison.csv"
            checkpoint_path.write_text("header\n", encoding="utf-8")
            report_path = gate_dir / "AUDIT_REPORT.md"
            report_path.write_text("authenticated test report\n", encoding="utf-8")
            manifest_path = gate_dir / "audit_manifest.json"
            manifest = {
                "campaign_id": method.EXPERIMENT_ID,
                "audit_status": "complete",
                "audit_script": str(method.AUDIT_FILE),
                "audit_script_sha256": runner.sha256(method.AUDIT_FILE),
                "registration": str(registration_path),
                "registration_sha256": expected_registration_sha,
                "core_source": str(method.THIS_FILE),
                "core_source_sha256": expected_core_sha,
                "requested_resolutions": [20],
                "requested_member_count": len(controls),
                "audited_member_count": len(controls),
                "integrity_valid_member_count": len(controls),
                "integrity_failure_count": 0,
                "integrity_errors": [],
                "resolution_results": {
                    "20": {
                        "registered_member_count": len(controls),
                        "integrity_valid_member_count": len(controls),
                        "selectable_member_count": len(controls),
                        "strict_pass_count": 1,
                    }
                },
                "all_requested_resolutions_strictly_pass": True,
                "artifacts": {
                    "candidate_summary_csv": {
                        "path": str(summary_path),
                        "sha256": runner.sha256(summary_path),
                    },
                    "checkpoint_comparison_csv": {
                        "path": str(checkpoint_path),
                        "sha256": runner.sha256(checkpoint_path),
                    },
                    "human_report": {
                        "path": str(report_path),
                        "sha256": runner.sha256(report_path),
                    },
                },
            }
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
            with mock.patch.object(runner, "CAMPAIGN_ROOT", campaign), mock.patch.object(
                runner, "REGISTRATION", registration_path
            ):
                observed, observed_token = runner.validate_n20_gate_for_n10(
                    registration, manifest_path
                )
                self.assertEqual(observed, manifest)
                self.assertEqual(observed_token, runner.sha256(manifest_path))

                real_sha256 = runner.sha256
                manifest_hash_calls = {"count": 0}

                def changing_manifest_hash(target):
                    if Path(target).resolve() == manifest_path.resolve():
                        manifest_hash_calls["count"] += 1
                        return (
                            "a" * 64
                            if manifest_hash_calls["count"] == 1
                            else "b" * 64
                        )
                    return real_sha256(target)

                with mock.patch.object(
                    runner, "sha256", side_effect=changing_manifest_hash
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "manifest changed during validation"
                    ):
                        runner.validate_n20_gate_for_n10(
                            registration, manifest_path
                        )
                receipt_paths[0].write_text(
                    receipt_paths[0].read_text(encoding="utf-8") + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    RuntimeError, "result authentication failed"
                ):
                    runner.validate_n20_gate_for_n10(
                        registration, manifest_path
                    )

    def test_chord_projection_uses_direct_energies_and_no_force_query(self):
        class ProjectionCore:
            N_INTERNAL = 2
            N_TOTAL_IMAGES = 4

            def __init__(self):
                self.seen_energies = []

            def improved_tangents(self, path, energies):
                del path
                self.seen_energies.append(np.asarray(energies).copy())
                return np.asarray([[1.0, 0.0], [1.0, 0.0]])

        core = ProjectionCore()
        raw = np.asarray(
            [[0.0, 0.0], [1.0, 0.6], [2.0, -0.2], [3.0, 0.0]]
        )
        direct_energies = np.asarray([0.0, 4.0, 2.0, 1.0])
        projected, _, iterations, _, feasible = (
            method.tangent_chord_constraint_project(
                core, raw, direct_energies, 1.0e-8
            )
        )
        self.assertTrue(feasible)
        self.assertGreater(iterations, 0)
        self.assertLessEqual(method.equidistance_integral(projected), 1.0e-8 + 1.0e-14)
        self.assertGreater(len(core.seen_energies), 0)
        for seen in core.seen_energies:
            np.testing.assert_array_equal(seen, direct_energies)
        projection_source = inspect.getsource(method.tangent_chord_constraint_project)
        self.assertIn("core.improved_tangents", projection_source)
        self.assertIn("np.linalg.lstsq", projection_source)
        self.assertNotIn("evaluate_path", projection_source)
        self.assertNotIn("query(", projection_source)

    def test_recovery_raw_costs_zero_and_only_projected_band_is_evaluated(self):
        (
            core,
            ledger,
            _pre,
            direct,
            _memories,
            _updated,
            transaction,
            descriptor,
        ) = self.recovery_call()
        self.assertTrue(transaction["raw_proposal_constructed"])
        logged_direction = np.asarray(
            transaction["raw_pre_addition_direction"], dtype=np.float64
        )
        self.assertEqual(logged_direction.shape, direct[0].shape)
        expected_raw = np.asarray(direct[0], dtype=np.float64) + float(
            transaction["regularized_step_length"]
        ) * logged_direction
        expected_raw[[0, -1]] = np.asarray(direct[0], dtype=np.float64)[[0, -1]]
        np.testing.assert_array_equal(
            np.asarray(transaction["raw_candidate_path"], dtype=np.float64),
            expected_raw,
        )
        self.assertFalse(transaction["raw_trial_evaluated"])
        self.assertTrue(transaction["projected_trial_evaluated"])
        self.assertTrue(transaction["sync_trial_evaluated"])
        self.assertTrue(transaction["committed"])
        self.assertTrue(transaction["recovery_accepted_by_atomic_gate"])
        self.assertEqual(transaction["actual_proximal_merit"], 1.0)
        self.assertIsNotNone(transaction["projected_candidate_path"])
        self.assertIsNotNone(transaction["projected_candidate_energies"])
        self.assertIsNotNone(transaction["projected_candidate_gradients"])
        self.assertIsNone(descriptor)
        self.assertEqual(transaction["direct_joint_evaluations"], 2)
        self.assertEqual(transaction["raw_recovery_new_image_evaluations"], 0)
        self.assertEqual(transaction["projected_recovery_new_image_evaluations"], 2)
        self.assertEqual(transaction["transaction_joint_evaluations"], 4)
        self.assertEqual(ledger.total, 4)
        self.assert_transaction_nof_closure(transaction, core.N_TOTAL_IMAGES)
        source = inspect.getsource(method.single_synchronized_recovery)
        self.assertEqual(source.count("evaluate_path("), 1)
        self.assertIn('"main_r_read_by_recovery": False', source)
        self.assertIn("chord_coherent_recovery_seed", source)
        self.assertIn("atomic_recovery_merge", source)
        self.assertEqual(method.RECOVERY_BAND_EVALUATION_CAP, 1)
        self.assertEqual(method.TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT, 2)

    def test_atomic_merge_accepts_or_rejects_whole_band(self):
        core = FakeCore()
        current_path = np.zeros((4, 2), dtype=np.float64)
        candidate_path = current_path.copy()
        candidate_path[1:3, 1] = [0.1, 0.2]
        current = (current_path, np.zeros(4), np.zeros_like(current_path))
        candidate = (candidate_path, np.zeros(4), np.zeros_like(candidate_path))
        with mock.patch.object(
            method, "internal_i_perp", return_value=(5.0, None)
        ), mock.patch.object(method, "equidistance_integral", return_value=0.0):
            merged, accepted, audit = method.atomic_recovery_merge(
                core, current, candidate, 1.0, 1.0, 10.0
            )
        np.testing.assert_array_equal(merged[0], candidate_path)
        np.testing.assert_array_equal(accepted, np.ones(2, dtype=bool))
        self.assertTrue(audit["monotone_to_post_main_i_perp"])
        with mock.patch.object(
            method, "internal_i_perp", return_value=(11.0, None)
        ), mock.patch.object(method, "equidistance_integral", return_value=0.0):
            merged, accepted, audit = method.atomic_recovery_merge(
                core, current, candidate, 1.0, 1.0, 10.0
            )
        np.testing.assert_array_equal(merged[0], current_path)
        np.testing.assert_array_equal(accepted, np.zeros(2, dtype=bool))
        self.assertFalse(audit["positive_actual_proximal_merit"])

        energetic_current = (
            current_path,
            np.asarray([0.0, 10.0, 10.0, 0.0]),
            np.zeros_like(current_path),
        )
        energetic_candidate = (
            candidate_path,
            np.zeros(4),
            np.zeros_like(candidate_path),
        )
        with mock.patch.object(
            method, "internal_i_perp", return_value=(11.0, None)
        ), mock.patch.object(method, "equidistance_integral", return_value=0.0):
            merged, accepted, audit = method.atomic_recovery_merge(
                core,
                energetic_current,
                energetic_candidate,
                1.0,
                1.0,
                10.0,
            )
        np.testing.assert_array_equal(merged[0], candidate_path)
        np.testing.assert_array_equal(accepted, np.ones(2, dtype=bool))
        self.assertFalse(audit["monotone_to_post_main_i_perp"])
        self.assertTrue(audit["positive_actual_proximal_merit"])

    def test_complete_commit_publishes_curvature_certified_secants(self):
        (
            core,
            _ledger,
            _pre,
            _direct,
            memories,
            _updated,
            transaction,
            _descriptor,
        ) = self.recovery_call()
        self.assertEqual(transaction["merged_images"], core.N_INTERNAL)
        self.assertEqual(
            transaction["accepted_recovery_secant_publication_mode"],
            "conservative_all",
        )
        self.assertEqual(
            transaction["accepted_recovery_secants_certified_curvature"], 2
        )
        self.assertEqual(transaction["accepted_recovery_secants_added"], 2)
        self.assertEqual(
            transaction["accepted_recovery_secants_rejected_curvature"], 0
        )
        self.assertEqual([len(memory) for memory in memories], [1, 1])
        self.assert_publication_closure(transaction, core.N_INTERNAL)
        for memory in memories:
            step, change = memory[0]
            self.assertGreater(
                float(np.dot(step, change)),
                method.RECOVERY_SECANT_CURVATURE
                * np.linalg.norm(step)
                * np.linalg.norm(change),
            )

    def test_strong_commit_publishes_top_normalized_curvature_with_index_tie_break(self):
        (
            core,
            _ledger,
            _pre,
            _direct,
            memories,
            _updated,
            transaction,
            _descriptor,
        ) = self.recovery_call(strong_recovery_selected=True)
        self.assertTrue(transaction["strong_recovery_selected"])
        self.assertEqual(
            transaction["accepted_recovery_secant_publication_mode"],
            "strong_top1",
        )
        self.assertEqual(
            transaction["accepted_recovery_secant_certified_indices"], [1, 2]
        )
        self.assertEqual(
            transaction["accepted_recovery_secant_published_indices"], [1]
        )
        self.assertEqual(
            transaction["accepted_recovery_secant_omitted_indices"], [2]
        )
        self.assertEqual(
            transaction[
                "accepted_recovery_secant_published_normalized_curvatures"
            ],
            [1.0],
        )
        self.assertEqual([len(memory) for memory in memories], [1, 0])
        self.assert_publication_closure(transaction, core.N_INTERNAL)

    def test_strong_top1_selects_largest_curvature_and_partitions_rejection(self):
        direct_path = np.zeros((5, 2), dtype=np.float64)
        projected_path = direct_path.copy()
        projected_path[1:4, 0] = 1.0
        direct_gradients = np.zeros_like(direct_path)
        projected_gradients = direct_gradients.copy()
        projected_gradients[1] = [0.8, 0.6]
        projected_gradients[2] = [0.9, np.sqrt(1.0 - 0.9**2)]
        projected_gradients[3] = [-1.0, 0.0]
        memories = [[], [], []]
        audit = method.publish_accepted_recovery_secants(
            memories,
            (direct_path, np.zeros(5), direct_gradients),
            (projected_path, np.zeros(5), projected_gradients),
            strong_recovery_selected=True,
            stationary_normal_probe=False,
        )
        self.assertEqual(
            audit["accepted_recovery_secant_certified_indices"], [1, 2]
        )
        self.assertEqual(
            audit["accepted_recovery_secant_published_indices"], [2]
        )
        self.assertEqual(
            audit["accepted_recovery_secant_omitted_indices"], [1]
        )
        self.assertEqual(
            audit["accepted_recovery_secant_rejected_indices"], [3]
        )
        self.assertAlmostEqual(
            audit[
                "accepted_recovery_secant_published_normalized_curvatures"
            ][0],
            0.9,
        )
        self.assertEqual([len(memory) for memory in memories], [0, 1, 0])
        self.assert_publication_closure(audit, 3)

    def test_curvature_threshold_equality_is_strictly_rejected(self):
        threshold = float(method.RECOVERY_SECANT_CURVATURE)
        second = np.nextafter(
            np.sqrt(1.0 - threshold * threshold), np.inf
        )
        step = np.asarray([1.0, 0.0])
        change = np.asarray([threshold, second])
        self.assertEqual(float(np.linalg.norm(change)), 1.0)
        self.assertEqual(
            float(np.dot(step, change)),
            threshold * float(np.linalg.norm(step)) * float(np.linalg.norm(change)),
        )
        direct_path = np.zeros((3, 2), dtype=np.float64)
        projected_path = direct_path.copy()
        projected_path[1] = step
        direct_gradients = np.zeros_like(direct_path)
        projected_gradients = direct_gradients.copy()
        projected_gradients[1] = change
        memories = [[]]
        audit = method.publish_accepted_recovery_secants(
            memories,
            (direct_path, np.zeros(3), direct_gradients),
            (projected_path, np.zeros(3), projected_gradients),
            strong_recovery_selected=False,
            stationary_normal_probe=False,
        )
        self.assertEqual(
            audit["accepted_recovery_secants_certified_curvature"], 0
        )
        self.assertEqual(
            audit["accepted_recovery_secants_rejected_curvature"], 1
        )
        self.assertEqual(
            audit["accepted_recovery_secant_rejected_indices"], [1]
        )
        self.assertEqual(memories, [[]])
        self.assert_publication_closure(audit, 1)

    def test_zero_curvature_certified_pairs_are_closed_as_rejected(self):
        direct_path = np.zeros((4, 2), dtype=np.float64)
        projected_path = direct_path.copy()
        projected_path[1:3, 0] = [1.0, 2.0]
        direct_gradients = np.zeros_like(direct_path)
        projected_gradients = direct_gradients.copy()
        projected_gradients[1:3, 1] = [1.0, 2.0]
        memories = [[], []]
        audit = method.publish_accepted_recovery_secants(
            memories,
            (direct_path, np.zeros(4), direct_gradients),
            (projected_path, np.zeros(4), projected_gradients),
            strong_recovery_selected=True,
            stationary_normal_probe=False,
        )
        self.assertEqual(
            audit["accepted_recovery_secant_publication_mode"], "strong_top1"
        )
        self.assertEqual(
            audit["accepted_recovery_secants_certified_curvature"], 0
        )
        self.assertEqual(audit["accepted_recovery_secants_added"], 0)
        self.assertEqual(
            audit["accepted_recovery_secants_omitted_by_rank"], 0
        )
        self.assertEqual(
            audit["accepted_recovery_secants_rejected_curvature"], 2
        )
        self.assertEqual(
            audit["accepted_recovery_secant_rejected_indices"], [1, 2]
        )
        self.assertEqual([len(memory) for memory in memories], [0, 0])
        self.assert_publication_closure(audit, 2)

    def test_rejected_band_never_changes_accepted_memory(self):
        def reject(core, current, _candidate, _ceiling, _alpha, _post_main_i):
            return (
                current,
                np.zeros(core.N_INTERNAL, dtype=bool),
                {
                    "candidate_i_perp": 2.0,
                    "candidate_equidistance": 0.0,
                    "actual_proximal_merit": -1.0,
                    "finite": True,
                    "monotone_to_post_main_i_perp": False,
                    "positive_actual_proximal_merit": False,
                    "accepted_by_atomic_gate": False,
                },
            )

        (
            core,
            _ledger,
            pre,
            _direct,
            memories,
            updated,
            transaction,
            descriptor,
        ) = self.recovery_call(merge=reject)
        np.testing.assert_array_equal(updated[0], pre[0])
        self.assertFalse(transaction["committed"])
        self.assertTrue(transaction["rolled_back"])
        self.assertEqual(transaction["accepted_recovery_secants_added"], 0)
        self.assert_publication_not_applicable(transaction)
        self.assertEqual([len(memory) for memory in memories], [0, 0])
        self.assertIsNotNone(descriptor)
        self.assert_transaction_nof_closure(transaction, core.N_TOTAL_IMAGES)

    def test_partial_projected_band_is_charged_but_publishes_nothing(self):
        (
            core,
            ledger,
            pre,
            _direct,
            memories,
            updated,
            transaction,
            descriptor,
        ) = self.recovery_call(budget=3)
        np.testing.assert_array_equal(updated[0], pre[0])
        self.assertTrue(transaction["interrupted"])
        self.assertFalse(transaction["projected_trial_evaluated"])
        self.assertIsNone(transaction["projected_candidate_path"])
        self.assertIsNone(transaction["projected_candidate_energies"])
        self.assertIsNone(transaction["projected_candidate_gradients"])
        self.assertEqual(transaction["raw_recovery_new_image_evaluations"], 0)
        self.assertEqual(transaction["projected_recovery_new_image_evaluations"], 1)
        self.assertEqual(transaction["transaction_joint_evaluations"], 3)
        self.assertEqual(transaction["accepted_recovery_secants_added"], 0)
        self.assert_publication_not_applicable(transaction)
        self.assertEqual([len(memory) for memory in memories], [0, 0])
        self.assertIsNone(descriptor)
        self.assertEqual(ledger.total, 3)
        self.assertEqual(ledger.blocked_misses, 1)
        self.assert_transaction_nof_closure(transaction, core.N_TOTAL_IMAGES)

    def test_same_anchor_half_backtrack_is_zero_nof_and_capped_by_mmax(self):
        core = FakeCore()
        path = np.zeros((4, 2), dtype=np.float64)
        energies = np.zeros(4, dtype=np.float64)
        gradients = np.zeros((4, 2), dtype=np.float64)
        gradients[1:3, 1] = 1.0
        candidate = method.recovery_direction_candidate(
            core, path, energies, gradients, [[], []], 0.5, 10.0, "accepted"
        )
        candidate["base_displacement"] = (
            candidate["step_length"] * candidate["directions"]
        )
        fractions = []
        for attempts in (1, 2, 3):
            state = method.make_recovery_backtrack_state(
                path, energies, gradients, candidate, attempts
            )
            retry = method.recovery_backtrack_candidate(state, 0.5)
            fractions.append(retry["backtrack_fraction"])
        self.assertEqual(fractions, [0.5, 0.25, 0.125])
        attempts = 0
        decisions = []
        for _ in range(4):
            attempts, retry = method.same_anchor_recovery_retry_decision(
                attempts, True, 4
            )
            decisions.append(retry)
        self.assertEqual(attempts, 4)
        self.assertEqual(decisions, [True, True, True, False])
        self.assertNotIn(
            "evaluate_path", inspect.getsource(method.recovery_backtrack_candidate)
        )

    def test_optimizer_has_no_metric_or_resolution_specific_recovery_branch(self):
        recovery_source = inspect.getsource(method.single_synchronized_recovery)
        projection_source = inspect.getsource(method.tangent_chord_constraint_project)
        seed_source = inspect.getsource(method.chord_coherent_recovery_seed)
        publisher_source = inspect.getsource(
            method.publish_accepted_recovery_secants
        )
        atomic_gate_source = inspect.getsource(method.atomic_recovery_merge)
        combined = (
            recovery_source
            + projection_source
            + seed_source
            + publisher_source
            + atomic_gate_source
        )
        for forbidden in (
            "REFERENCE_BARRIER",
            "densified",
            "ARTICLE_N10",
            "ARTICLE_N20",
            "n_internal == 10",
            "n_internal == 20",
            "central_difference",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertIn("core.improved_tangents", projection_source)
        self.assertNotIn("controls.r", seed_source)
        self.assertNotIn("evaluate_path", seed_source)
        self.assertNotIn("query(", seed_source)
        self.assertIn("actual_proximal_merit", atomic_gate_source)
        self.assertIn("strong_recovery_selected", publisher_source)
        self.assertIn("stationary_normal_probe", publisher_source)
        self.assertIn(
            "strong_recovery_selected=bool(", recovery_source
        )
        publisher_tree = ast.parse(publisher_source)
        self.assertFalse(
            any(isinstance(node, ast.Global) for node in ast.walk(publisher_tree))
        )
        for forbidden in (
            "sha256",
            "digest",
            "path_hash",
            "controls.r",
            "ARTICLE_",
        ):
            self.assertNotIn(forbidden, publisher_source)


if __name__ == "__main__":
    unittest.main()
