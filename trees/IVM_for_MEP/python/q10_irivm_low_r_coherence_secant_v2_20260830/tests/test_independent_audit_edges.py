#!/usr/bin/env python3
"""Regression tests for independent low-r recovery-publication auditing."""

from __future__ import annotations

import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import audit_results as audit  # noqa: E402
import q10_irivm_atomic_tangent_secant_core as method  # noqa: E402


class FakeMetric:
    """Minimal improved-tangent provider for independent spring replay."""

    @staticmethod
    def improved_tangents(path, _energies):
        tangents = np.zeros((np.asarray(path).shape[0] - 2, 10))
        tangents[:, 0] = 1.0
        return tangents

    @staticmethod
    def energy_gradient(point, q=None):
        del q
        value = np.asarray(point, dtype=np.float64)
        return float(0.5 * np.dot(value, value)), value.copy()


CONTROLS = SimpleNamespace(n_internal=2, m_max=4)
ENERGIES = np.asarray([0.0, 2.0, 1.0, 0.0])


def publication_transaction(
    mode,
    certified,
    rejected,
    omitted,
    published,
    normalized_curvatures,
):
    return {
        "accepted_recovery_secants_enabled": True,
        "main_memory_updates_only_after_complete_recovery_commit": True,
        "accepted_recovery_secant_curvature_threshold": (
            audit.SECANT_CURVATURE_THRESHOLD
        ),
        "accepted_recovery_secant_publication_mode": mode,
        "accepted_recovery_secants_certified_curvature": len(certified),
        "accepted_recovery_secant_certified_indices": list(certified),
        "accepted_recovery_secants_rejected_curvature": len(rejected),
        "accepted_recovery_secant_rejected_indices": list(rejected),
        "accepted_recovery_secants_omitted_by_rank": len(omitted),
        "accepted_recovery_secant_omitted_indices": list(omitted),
        "accepted_recovery_secants_added": len(published),
        "accepted_recovery_secant_published_indices": list(published),
        "accepted_recovery_secant_published_normalized_curvatures": list(
            normalized_curvatures
        ),
    }


def degenerate_fallback_fixture():
    raw = np.zeros((4, 10), dtype=np.float64)
    raw[2, 0] = 1.0
    raw[3, 0] = 2.0
    spring, residual, iterations = audit.independent_improved_spring(
        FakeMetric(), raw, ENERGIES
    )
    transaction = {
        "raw_candidate_path": raw.tolist(),
        "recovery_projection_seed_path": spring.tolist(),
        "stationary_normal_probe": False,
        "recovery_seed_mode": "spring_fallback_degenerate_polyline",
        "saddle_image": 1,
        "adjacent_chord_cosine": None,
        "coherence_loss": None,
        "coherence_loss_limit": 1.0 / 9.0,
        "amplified_direction_mismatch": None,
        "strong_recovery_selected": False,
        "saddle_spring_weight": 1.0,
        "seed_spring_residual": residual,
        "seed_spring_iterations": iterations,
    }
    return raw, transaction


class StateChainAndGateTokenTests(unittest.TestCase):
    @staticmethod
    def evaluated_state(path):
        energies = []
        gradients = []
        for point in np.asarray(path):
            energy, gradient = FakeMetric.energy_gradient(point)
            energies.append(energy)
            gradients.append(gradient)
        return (
            np.asarray(path, dtype=np.float64),
            np.asarray(energies, dtype=np.float64),
            np.asarray(gradients, dtype=np.float64),
        )

    def state_chain_fixture(self):
        outer = np.zeros((4, 10), dtype=np.float64)
        outer[:, 0] = [0.0, 1.0, 2.0, 3.0]
        post = outer.copy()
        # Represents an accepted main movement served at the same NoF: the
        # public NoF ledger could replace `outer`, but this explicit chain may not.
        post[1:3, 0] += 0.25
        post_state = self.evaluated_state(post)
        stage = method.outer_state_chain_log(outer, post_state)
        outer_state = self.evaluated_state(outer)
        pre_i = audit.independent_gate_i_perp(FakeMetric(), *outer_state)
        post_i = audit.independent_gate_i_perp(FakeMetric(), *post_state)
        stage.update(
            {
                "pre_i_perp": pre_i,
                "candidate_i_perp": post_i,
                "post_main_i_perp": post_i,
                "rolled_back_by_i_perp_guard": False,
                "main_public_path_changed_bitwise": True,
            }
        )
        transaction = method.transaction_state_chain_log(outer, post_state, post)
        transaction.update(
            {
                "post_main_i_perp": post_i,
                "pre_i_perp": post_i,
                "acceptance_envelope_i_perp": post_i,
                "pre_equidistance": audit.independent_equidistance(post),
                "public_path_movement": float(
                    np.max(np.linalg.norm(post - outer, axis=1))
                ),
            }
        )
        return outer, post_state, stage, transaction

    def test_same_nof_main_movement_has_an_authenticated_state_chain(self):
        outer, post_state, stage, transaction = self.state_chain_fixture()
        stage_state = audit.audit_main_stage_state_chain_entry(
            FakeMetric(), stage, 2, outer, np.eye(10)
        )
        transaction_state = audit.audit_transaction_state_chain_entry(
            FakeMetric(), transaction, 2, outer, np.eye(10)
        )
        for name in (
            "outer_start_path",
            "post_main_path",
            "post_main_energies",
            "post_main_gradients",
        ):
            np.testing.assert_array_equal(stage_state[name], transaction_state[name])
        next_outer, movement = audit.close_transaction_state_chain_entry(
            transaction, transaction_state, post_state[0]
        )
        np.testing.assert_array_equal(next_outer, post_state[0])
        self.assertGreater(movement, 0.0)

    def test_state_chain_and_recomputed_metrics_reject_tampering(self):
        outer, post_state, stage, transaction = self.state_chain_fixture()
        stage_mutations = {
            "outer_start_path": lambda value: [
                value[0],
                [value[1][0] + 0.1] + value[1][1:],
                *value[2:],
            ],
            "post_main_path": lambda value: [
                value[0],
                [value[1][0] + 0.1] + value[1][1:],
                *value[2:],
            ],
            "post_main_energies": lambda value: [value[0] + 0.1, *value[1:]],
            "post_main_gradients": lambda value: [
                [value[0][0] + 0.1] + value[0][1:],
                *value[1:],
            ],
            "pre_i_perp": lambda value: value + 0.1,
            "post_main_i_perp": lambda value: value + 0.1,
            "main_public_path_changed_bitwise": lambda value: not value,
        }
        for field, mutate in stage_mutations.items():
            with self.subTest(stage_field=field):
                changed = copy.deepcopy(stage)
                changed[field] = mutate(changed[field])
                with self.assertRaises(audit.IntegrityError):
                    audit.audit_main_stage_state_chain_entry(
                        FakeMetric(), changed, 2, outer, np.eye(10)
                    )
        transaction_state = audit.audit_transaction_state_chain_entry(
            FakeMetric(), transaction, 2, outer, np.eye(10)
        )
        changed = copy.deepcopy(transaction)
        changed["committed_public_path"][1][0] += 0.1
        changed_state = audit.audit_transaction_state_chain_entry(
            FakeMetric(), changed, 2, outer, np.eye(10)
        )
        with self.assertRaises(audit.IntegrityError):
            audit.close_transaction_state_chain_entry(
                changed, changed_state, post_state[0]
            )
        changed = copy.deepcopy(transaction)
        changed["public_path_movement"] += 0.1
        with self.assertRaises(audit.IntegrityError):
            audit.close_transaction_state_chain_entry(
                changed, transaction_state, post_state[0]
            )

    def test_resolution_gate_token_is_mandatory_and_current(self):
        n20 = SimpleNamespace(n_internal=20)
        self.assertEqual(
            method.validate_resolution_gate_token(n20, "not_applicable"),
            "not_applicable",
        )
        with self.assertRaises(method.ContractError):
            method.validate_resolution_gate_token(n20, "0" * 64)
        n10 = SimpleNamespace(n_internal=10)
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "audit_manifest.json"
            manifest.write_text("authenticated synthetic gate\n", encoding="utf-8")
            token = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with mock.patch.object(method, "N20_GATE_MANIFEST", manifest):
                self.assertEqual(
                    method.validate_resolution_gate_token(n10, token), token
                )
                with self.assertRaises(method.ContractError):
                    method.validate_resolution_gate_token(n10, "0" * 64)
                with self.assertRaises(method.ContractError):
                    method.validate_resolution_gate_token(n10, "NOT_HEX")

    def test_per_outer_nof_chain_rejects_synchronized_timestamp_tampering(self):
        stage = {
            "image_audits": [{"new_calls": 2}, {"new_calls": 3}],
            "terminal_article_nf": 15,
        }
        running = audit.advance_main_stage_nof(stage, 10)
        self.assertEqual(running, 15)
        transaction = {
            "direct_joint_evaluation_vector": [0, 1, 1, 0],
            "recovery_joint_evaluation_vector": [0, 1, 0, 0],
        }
        self.assertEqual(
            audit.advance_transaction_nof(transaction, 17, running, 4), 18
        )
        changed_stage = copy.deepcopy(stage)
        changed_stage["terminal_article_nf"] = 14
        with self.assertRaises(audit.IntegrityError):
            audit.advance_main_stage_nof(changed_stage, 10)
        with self.assertRaises(audit.IntegrityError):
            audit.advance_transaction_nof(transaction, 16, running, 4)


class EvaluationTraceAndProjectionTests(unittest.TestCase):
    @staticmethod
    def trace_entry(sequence, outer, phase, image, point):
        return {
            "sequence": sequence,
            "article_nf": sequence,
            "outer": outer,
            "phase": phase,
            "image_index": image,
            "coordinate_sha256": audit.coordinate_sha256(point),
        }

    def test_ordered_trace_rejects_cross_stage_new_call_redistribution(self):
        initial = np.zeros((4, 10), dtype=np.float64)
        initial[:, 0] = [0.0, 1.0, 2.0, 3.0]
        main_one = initial[1].copy()
        main_one[1] = 1.0
        main_two = initial[2].copy()
        main_two[1] = 2.0
        trace = [
            self.trace_entry(1, 0, "initial", 1, initial[1]),
            self.trace_entry(2, 0, "initial", 2, initial[2]),
            self.trace_entry(3, 1, "main", 1, main_one),
            self.trace_entry(4, 2, "main", 2, main_two),
        ]
        stages = [
            {"image_audits": [{"new_calls": 1}, {"new_calls": 0}]},
            {"image_audits": [{"new_calls": 0}, {"new_calls": 1}]},
        ]
        transactions = [
            {
                "projected_attempt_path": None,
                "projected_trial_evaluated": False,
                "interrupted": False,
            },
            {
                "projected_attempt_path": None,
                "projected_trial_evaluated": False,
                "interrupted": False,
            },
        ]
        direct_paths = np.asarray([initial, initial])
        observed = audit.audit_evaluation_miss_trace(
            trace,
            2,
            initial,
            stages,
            transactions,
            direct_paths,
            FakeMetric(),
            4,
            "outer_limit",
        )
        np.testing.assert_array_equal(
            observed["main_vectors"][1], [0, 1, 0, 0]
        )
        changed = copy.deepcopy(stages)
        # Preserve the total NoF while moving one miss across stage boundaries,
        # as a synchronized terminal/public-timestamp forgery would do.
        changed[0]["image_audits"][1]["new_calls"] = 1
        changed[1]["image_audits"][1]["new_calls"] = 0
        with self.assertRaises(audit.IntegrityError):
            audit.audit_evaluation_miss_trace(
                trace,
                2,
                initial,
                changed,
                transactions,
                direct_paths,
                FakeMetric(),
                4,
                "outer_limit",
            )

    def test_projection_replay_authenticates_newton_path_and_metadata(self):
        seed = np.zeros((4, 10), dtype=np.float64)
        seed[:, 0] = [0.0, 0.2, 3.5, 4.0]
        path, residual, iterations, backtracks, feasible = (
            audit.independent_tangent_chord_projection(
                FakeMetric(), seed, ENERGIES, 0.0
            )
        )
        self.assertGreater(iterations, 0)
        transaction = {
            "raw_proposal_constructed": True,
            "recovery_projection_seed_path": seed.tolist(),
            "projected_attempt_path": path.tolist(),
            "recovery_projection_residual": residual,
            "recovery_projection_iterations": iterations,
            "recovery_projection_backtracks": backtracks,
            "recovery_projection_feasible": feasible,
        }
        observed = audit.audit_projection_replay(
            FakeMetric(), transaction, ENERGIES, 0.0
        )
        np.testing.assert_array_equal(observed["path"], path)
        mutations = {
            "projected_attempt_path": lambda value: [
                value[0],
                [value[1][0] + 0.25] + value[1][1:],
                *value[2:],
            ],
            "recovery_projection_residual": lambda value: value + 0.25,
            "recovery_projection_iterations": lambda value: value + 1,
            "recovery_projection_backtracks": lambda value: value + 1,
            "recovery_projection_feasible": lambda value: not value,
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(transaction)
                changed[field] = mutate(changed[field])
                with self.assertRaises(audit.IntegrityError):
                    audit.audit_projection_replay(
                        FakeMetric(), changed, ENERGIES, 0.0
                    )

    def test_unconstructed_raw_proposal_cannot_contain_projection_state(self):
        transaction = {
            "raw_proposal_constructed": False,
            "recovery_projection_seed_path": None,
            "projected_attempt_path": None,
            "recovery_projection_residual": None,
            "recovery_projection_iterations": 0,
            "recovery_projection_backtracks": 0,
            "recovery_projection_feasible": False,
        }
        audit.audit_projection_replay(FakeMetric(), transaction, ENERGIES, 0.0)
        transaction["projected_attempt_path"] = np.zeros((4, 10)).tolist()
        with self.assertRaises(audit.IntegrityError):
            audit.audit_projection_replay(
                FakeMetric(), transaction, ENERGIES, 0.0
            )

    def test_midrun_truncation_cannot_be_relabelled_stationary(self):
        transaction = {
            "outer": 1,
            "interrupted": False,
            "committed": True,
            "rolled_back": False,
            "projected_trial_evaluated": False,
            "completed_recovery_candidate_at_current_anchor": False,
            "same_anchor_recovery_attempt_count_before": 0,
            "same_anchor_recovery_attempt_count_after": 0,
            "same_anchor_recovery_attempt_index": 0,
            "same_anchor_recovery_attempt_cap_m_max": 4,
            "public_path_movement": 0.25,
            "recovery_backtrack_carry_allowed": False,
            "same_anchor_recovery_retry_allowed": False,
            "recovery_backtrack_lifecycle_action": (
                "clear_on_transaction_commit"
            ),
        }
        vectors = {
            "main_vectors": {1: np.asarray([0, 1, 0, 0])},
            "direct_vectors": {1: np.zeros(4, dtype=np.int64)},
            "projected_vectors": {1: np.zeros(4, dtype=np.int64)},
        }
        with self.assertRaises(audit.IntegrityError):
            audit.audit_stop_reason_semantics(
                "stationary",
                "",
                3,
                0,
                [{"outer": 1}],
                [transaction],
                1,
                4,
                vectors,
                "none",
            )
        vectors["main_vectors"][1][:] = 0
        self.assertEqual(
            audit.audit_stop_reason_semantics(
                "stationary",
                "",
                2,
                0,
                [{"outer": 1}],
                [transaction],
                1,
                4,
                vectors,
                "none",
            ),
            "stationary",
        )


class RawProposalAlgebraTests(unittest.TestCase):
    def setUp(self):
        self.path = np.zeros((4, 10), dtype=np.float64)
        self.path[:, 0] = [0.0, 1.0, 2.0, 3.0]
        self.energies = np.asarray([0.0, 1.0, 1.0, 0.0])
        self.gradients = np.zeros_like(self.path)
        self.gradients[1:3, 0] = 1.0
        self.alpha = 0.1
        self.logged_direction = np.zeros_like(self.path)
        self.logged_direction[1:3, 0] = -0.2
        self.g_dot_d = 0.0
        self.norm_squared = 0.0
        for image_index in range(1, self.path.shape[0] - 1):
            self.g_dot_d += float(
                np.dot(
                    self.gradients[image_index],
                    self.logged_direction[image_index],
                )
            )
            self.norm_squared += float(
                np.dot(
                    self.logged_direction[image_index],
                    self.logged_direction[image_index],
                )
            )
        self.base_step = float(
            np.clip(
                -self.alpha * self.g_dot_d / self.norm_squared,
                0.0,
                1.0,
            )
        )
        first_raw = self.path + self.base_step * self.logged_direction
        first_raw[0], first_raw[-1] = self.path[0], self.path[-1]
        self.base_displacement = first_raw - self.path

    def transaction(self, retry=False, carry=False):
        before = 1 if retry else 0
        fraction = 0.5 if retry else 1.0
        step = self.base_step * fraction
        raw = self.path + fraction * self.base_displacement
        predicted = (
            -step * self.g_dot_d
            - step * step * self.norm_squared / (2.0 * self.alpha)
        )
        name = "same_anchor_backtrack" if retry else "accepted_history"
        return {
            "total_atomic_trust_radius": audit.independent_recovery_trust_radius(
                self.path
            ),
            "recovery_backtrack_state_used": retry,
            "private_recovery_backtrack_state_entering": retry,
            "recovery_backtrack_anchor_matched": True,
            "raw_proposal_constructed": True,
            "raw_candidate_path": raw.tolist(),
            "raw_pre_addition_direction": self.logged_direction.tolist(),
            "regularized_step_length": step,
            "predicted_regularized_decrease": predicted,
            "recovery_backtrack_fraction": fraction,
            "warm_pair_count": 0,
            "selected_direction": name,
            "direction_candidates": [
                {
                    "name": name,
                    "predicted_regularized_decrease": predicted,
                    "regularized_step_length": step,
                    "base_regularized_step_length": self.base_step,
                    "backtrack_fraction": fraction,
                    "g_dot_d": self.g_dot_d,
                    "norm_squared": self.norm_squared,
                    "warm_pair_count": 0,
                }
            ],
            "same_anchor_recovery_attempt_count_before": before,
            "same_anchor_recovery_attempt_count_after": before + 1,
            "projected_trial_evaluated": True,
            "interrupted": False,
            "recovery_backtrack_carry_allowed": carry,
        }

    def audit(self, transaction, state):
        return audit.audit_raw_proposal_algebra(
            transaction,
            self.path,
            self.path,
            self.energies,
            self.gradients,
            self.alpha,
            4,
            state,
        )

    def test_first_step_and_same_anchor_fraction_are_independently_replayed(self):
        first = self.transaction(carry=True)
        state = self.audit(first, None)
        self.assertEqual(state["completed_attempts"], 1)
        retry = self.transaction(retry=True, carry=False)
        self.assertIsNone(self.audit(retry, state))

    def test_raw_algebra_and_exact_same_anchor_tampering_are_rejected(self):
        first = self.transaction(carry=True)
        state = self.audit(first, None)
        mutations = {
            "g_dot_d": lambda value: value + 0.1,
            "norm_squared": lambda value: value + 0.1,
            "predicted_regularized_decrease": lambda value: value + 0.1,
            "base_regularized_step_length": lambda value: value + 0.1,
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                changed = self.transaction(retry=True, carry=False)
                if field in changed["direction_candidates"][0]:
                    changed["direction_candidates"][0][field] = mutate(
                        changed["direction_candidates"][0][field]
                    )
                if field == "predicted_regularized_decrease":
                    changed[field] = mutate(changed[field])
                with self.assertRaises(audit.IntegrityError):
                    self.audit(changed, state)
        changed = self.transaction(retry=True, carry=False)
        changed["raw_candidate_path"][1][0] += 0.01
        with self.assertRaises(audit.IntegrityError):
            self.audit(changed, state)
        changed_energies = self.energies.copy()
        changed_energies[1] += 0.01
        with self.assertRaises(audit.IntegrityError):
            audit.audit_raw_proposal_algebra(
                self.transaction(retry=True, carry=False),
                self.path,
                self.path,
                changed_energies,
                self.gradients,
                self.alpha,
                4,
                state,
            )

    def test_true_tiny_direction_uses_logged_algebra_not_subtracted_sign(self):
        path = self.path.copy()
        path[1, 1] = 1.0
        path[1, 2] = 4.1640709306487675
        gradients = np.zeros_like(path)
        gradients[1, 1:3] = [0.1, -0.1]
        logged_direction = np.zeros_like(path)
        logged_direction[1, 1:3] = [
            5.551466273330682e-15,
            5.551466288599539e-15,
        ]
        logged_g_dot_d = 0.0
        logged_norm_squared = 0.0
        for image_index in range(1, path.shape[0] - 1):
            logged_g_dot_d += float(
                np.dot(gradients[image_index], logged_direction[image_index])
            )
            logged_norm_squared += float(
                np.dot(
                    logged_direction[image_index],
                    logged_direction[image_index],
                )
            )
        step = float(
            np.clip(
                -1.0e-3 * logged_g_dot_d / logged_norm_squared,
                0.0,
                1.0,
            )
        )
        prediction = float(
            -step * logged_g_dot_d
            - step * step * logged_norm_squared / (2.0 * 1.0e-3)
        )
        raw = path + step * logged_direction
        raw[0], raw[-1] = path[0], path[-1]
        transaction = {
            "total_atomic_trust_radius": (
                audit.independent_recovery_trust_radius(path)
            ),
            "recovery_backtrack_state_used": False,
            "private_recovery_backtrack_state_entering": False,
            "recovery_backtrack_anchor_matched": False,
            "raw_proposal_constructed": True,
            "raw_candidate_path": raw.tolist(),
            "raw_pre_addition_direction": logged_direction.tolist(),
            "regularized_step_length": step,
            "predicted_regularized_decrease": prediction,
            "recovery_backtrack_fraction": 1.0,
            "warm_pair_count": 0,
            "selected_direction": "accepted_history",
            "direction_candidates": [
                {
                    "name": "accepted_history",
                    "predicted_regularized_decrease": prediction,
                    "regularized_step_length": step,
                    "base_regularized_step_length": step,
                    "backtrack_fraction": 1.0,
                    "g_dot_d": logged_g_dot_d,
                    "norm_squared": logged_norm_squared,
                    "warm_pair_count": 0,
                }
            ],
            "same_anchor_recovery_attempt_count_before": 0,
            "same_anchor_recovery_attempt_count_after": 1,
            "projected_trial_evaluated": True,
            "interrupted": False,
            "recovery_backtrack_carry_allowed": False,
        }

        reconstructed_direction = raw - path
        self.assertGreater(float(np.sum(gradients * reconstructed_direction)), 0.0)
        self.assertEqual(step, 1.0)
        self.assertIsNone(
            audit.audit_raw_proposal_algebra(
                transaction,
                path,
                path,
                self.energies,
                gradients,
                1.0e-3,
                4,
                None,
            )
        )

        mutations = {
            "g_dot_d": lambda changed: changed[
                "direction_candidates"
            ][0].update({"g_dot_d": abs(logged_g_dot_d)}),
            "norm_squared": lambda changed: changed[
                "direction_candidates"
            ][0].update({"norm_squared": np.finfo(np.float64).tiny}),
            "regularized_step_length": lambda changed: (
                changed.update({"regularized_step_length": 0.5}),
                changed["direction_candidates"][0].update(
                    {
                        "regularized_step_length": 0.5,
                        "base_regularized_step_length": 0.5,
                    }
                ),
            ),
            "predicted_regularized_decrease": lambda changed: (
                changed.update(
                    {"predicted_regularized_decrease": 2.0 * prediction}
                ),
                changed["direction_candidates"][0].update(
                    {"predicted_regularized_decrease": 2.0 * prediction}
                ),
            ),
        }
        for field, mutate in mutations.items():
            with self.subTest(tampered=field):
                changed = copy.deepcopy(transaction)
                mutate(changed)
                with self.assertRaises(audit.IntegrityError):
                    audit.audit_raw_proposal_algebra(
                        changed,
                        path,
                        path,
                        self.energies,
                        gradients,
                        1.0e-3,
                        4,
                        None,
                    )

        changed = copy.deepcopy(transaction)
        forged_g_dot_d = -1.0e-14
        forged_norm_squared = 1.0e-16
        forged_step = 0.1
        forged_prediction = float(
            -forged_step * forged_g_dot_d
            - forged_step * forged_step * forged_norm_squared
            / (2.0 * 1.0e-3)
        )
        changed.update(
            {
                "regularized_step_length": forged_step,
                "predicted_regularized_decrease": forged_prediction,
            }
        )
        changed["direction_candidates"][0].update(
            {
                "g_dot_d": forged_g_dot_d,
                "norm_squared": forged_norm_squared,
                "regularized_step_length": forged_step,
                "base_regularized_step_length": forged_step,
                "predicted_regularized_decrease": forged_prediction,
            }
        )
        with self.assertRaises(audit.IntegrityError):
            audit.audit_raw_proposal_algebra(
                changed,
                path,
                path,
                self.energies,
                gradients,
                1.0e-3,
                4,
                None,
            )

        changed = copy.deepcopy(transaction)
        changed["raw_pre_addition_direction"][1][1] += 1.0e-12
        with self.assertRaises(audit.IntegrityError):
            audit.audit_raw_proposal_algebra(
                changed,
                path,
                path,
                self.energies,
                gradients,
                1.0e-3,
                4,
                None,
            )


class RecoverySeedEdgeTests(unittest.TestCase):
    def test_degenerate_polyline_replays_conservative_spring_fallback(self):
        raw, transaction = degenerate_fallback_fixture()
        self.assertIsNone(audit.independent_polyline_equal_arclength(raw))
        observed = audit.audit_chord_coherent_seed(
            FakeMetric(), CONTROLS, transaction, ENERGIES
        )
        self.assertEqual(
            observed,
            {
                "stationary": False,
                "strong": False,
                "weight": 1.0,
                "degenerate_polyline_fallback": True,
            },
        )

    def test_every_fallback_field_tamper_is_rejected(self):
        _raw, original = degenerate_fallback_fixture()
        mutations = {
            "recovery_projection_seed_path": lambda value: [
                value[0],
                [value[1][0] + 0.25] + value[1][1:],
                *value[2:],
            ],
            "recovery_seed_mode": lambda _value: (
                "polyline_spring_saddle_adjacent_chord_coherence"
            ),
            "saddle_image": lambda _value: 2,
            "adjacent_chord_cosine": lambda _value: 0.0,
            "coherence_loss": lambda _value: 0.0,
            "coherence_loss_limit": lambda value: value + 0.25,
            "amplified_direction_mismatch": lambda _value: 0.0,
            "strong_recovery_selected": lambda _value: True,
            "saddle_spring_weight": lambda _value: 3.0,
            "seed_spring_residual": lambda value: value + 0.25,
            "seed_spring_iterations": lambda value: value + 1,
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                transaction = copy.deepcopy(original)
                transaction[field] = mutate(transaction[field])
                with self.assertRaises(audit.IntegrityError):
                    audit.audit_chord_coherent_seed(
                        FakeMetric(), CONTROLS, transaction, ENERGIES
                    )

    def test_zero_resampled_saddle_chord_uses_cosine_minus_one(self):
        # Every raw segment is nonzero, but the first two equal-arclength
        # samples coincide after the 0 -> 1 -> 0 closed excursion.
        raw = np.zeros((4, 10), dtype=np.float64)
        raw[:, 0] = [0.0, 1.0, 0.0, 4.0]
        polyline = audit.independent_polyline_equal_arclength(raw)
        self.assertIsNotNone(polyline)
        np.testing.assert_array_equal(polyline[:, 0], [0.0, 0.0, 2.0, 4.0])
        spring, residual, iterations = audit.independent_improved_spring(
            FakeMetric(), raw, ENERGIES
        )
        seed = polyline.copy()
        seed[1] = spring[1]
        transaction = {
            "raw_candidate_path": raw.tolist(),
            "recovery_projection_seed_path": seed.tolist(),
            "stationary_normal_probe": False,
            "recovery_seed_mode": (
                "polyline_spring_saddle_adjacent_chord_coherence"
            ),
            "saddle_image": 1,
            "adjacent_chord_cosine": -1.0,
            "coherence_loss": 2.0,
            "coherence_loss_limit": 1.0 / 9.0,
            "amplified_direction_mismatch": 3.0 * np.sqrt(2.0),
            "strong_recovery_selected": False,
            "saddle_spring_weight": 1.0,
            "seed_spring_residual": residual,
            "seed_spring_iterations": iterations,
        }
        observed = audit.audit_chord_coherent_seed(
            FakeMetric(), CONTROLS, transaction, ENERGIES
        )
        self.assertFalse(observed["strong"])
        self.assertFalse(observed["degenerate_polyline_fallback"])
        self.assertEqual(observed["weight"], 1.0)


class SecantPublicationEdgeTests(unittest.TestCase):
    def setUp(self):
        self.direct_path = np.zeros((6, 10), dtype=np.float64)
        self.projected_path = self.direct_path.copy()
        self.projected_path[1:5, 0] = 1.0
        self.direct_gradient = np.zeros_like(self.direct_path)
        self.projected_gradient = self.direct_gradient.copy()
        self.projected_gradient[1, :2] = [1.0, 0.0]
        self.projected_gradient[2, :2] = [1.0, 1.0]
        self.projected_gradient[3, :2] = [-1.0, 0.0]
        self.projected_gradient[4, :2] = [1.0, 0.0]

    def audit_publication(self, transaction, mode):
        return audit.audit_secant_publication(
            transaction,
            4,
            recovery_committed=True,
            expected_mode=mode,
            direct_path=self.direct_path,
            direct_gradient=self.direct_gradient,
            projected_path=self.projected_path,
            projected_gradient=self.projected_gradient,
        )

    def test_strong_publishes_top_one_with_lower_index_tie_break(self):
        transaction = publication_transaction(
            "strong_top1", [1, 2, 4], [3], [2, 4], [1], [1.0]
        )
        observed = self.audit_publication(transaction, "strong_top1")
        self.assertEqual(observed["published"], 1)
        self.assertEqual(observed["omitted"], 2)

    def test_conservative_and_ordinary_publish_all_certified(self):
        curvatures = [1.0, 1.0 / np.sqrt(2.0), 1.0]
        for mode in ("conservative_all", "ordinary_all"):
            with self.subTest(mode=mode):
                transaction = publication_transaction(
                    mode, [1, 2, 4], [3], [], [1, 2, 4], curvatures
                )
                observed = self.audit_publication(transaction, mode)
                self.assertEqual(observed["published"], 3)
                self.assertEqual(observed["omitted"], 0)

    def test_strict_curvature_threshold_equality_is_rejected(self):
        direct_path = np.zeros((3, 10), dtype=np.float64)
        projected_path = direct_path.copy()
        projected_path[1, 0] = 1.0
        direct_gradient = np.zeros_like(direct_path)
        projected_gradient = direct_gradient.copy()
        projected_gradient[1, 0] = 1.0
        with mock.patch.object(audit, "SECANT_CURVATURE_THRESHOLD", 1.0):
            partition = audit.independent_secant_partition(
                direct_path,
                direct_gradient,
                projected_path,
                projected_gradient,
                "strong_top1",
            )
        self.assertEqual(partition["certified"], [])
        self.assertEqual(partition["rejected"], [1])
        self.assertEqual(partition["omitted"], [])
        self.assertEqual(partition["published"], [])

    def test_rank_omission_and_published_curvature_tampering_are_rejected(self):
        original = publication_transaction(
            "strong_top1", [1, 2, 4], [3], [2, 4], [1], [1.0]
        )
        mutations = (
            (
                "rank omission",
                {
                    "accepted_recovery_secant_omitted_indices": [2],
                    "accepted_recovery_secants_omitted_by_rank": 1,
                },
            ),
            (
                "normalized curvature",
                {
                    "accepted_recovery_secant_published_normalized_curvatures": [
                        0.5
                    ]
                },
            ),
        )
        for label, changes in mutations:
            with self.subTest(label=label):
                transaction = copy.deepcopy(original)
                transaction.update(changes)
                with self.assertRaises(audit.IntegrityError):
                    self.audit_publication(transaction, "strong_top1")


if __name__ == "__main__":
    unittest.main()
