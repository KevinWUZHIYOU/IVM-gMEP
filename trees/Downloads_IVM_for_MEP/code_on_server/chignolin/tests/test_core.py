from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from ase import Atoms

import run_ivm
from chignolin_core import (
    AuditedOpenMMCalculator,
    CentralDifferenceNEB,
    EvaluationBudgetReached,
    ImageEvaluator,
    OpenMMBackend,
    audit_counter_consistency,
    central_tangents,
    equidistance_diagnostics,
    evaluate_path,
    path_diagnostics,
    restore_evaluator_states,
    spring_reparameterize,
)
from run_ivm import (
    ForceFieldPreconditionerBank,
    ProjectedLBFGSMemory,
    adapt_proximal_alpha,
    central_hermite_spring_equilibrium,
    ffprecon_subproblem_acceptance_mask,
    ffprecon_update_mask,
    geometric_spring_reparameterization,
    euler_spring_reparameterization,
    greedy_bicriteria_recovery_acceptance,
    history_warm_start_is_active,
    residual_precision_cap,
    relative_iperp_precision_cap,
    rms_segment_trust_radius,
    greedy_global_iperp_acceptance,
    reparameterization_acceptance_mask,
    spring_proposal_to_target_cv,
    state_reparameterization_is_due,
    solve_contractive_proximal_band,
    solve_curve_projected_proximal_band,
    solve_lbfgsb_band,
    solve_projected_proximal_band,
)


class FakeBackend:
    def __init__(self, dimension=6):
        self.dimension = dimension
        self.n_atoms = dimension // 3
        self.raw_joint_evaluations = 0
        self.label = "fake"

    def evaluate_joint(self, positions_flat):
        x = np.asarray(positions_flat, dtype=float)
        self.raw_joint_evaluations += 1
        return float(0.5 * np.dot(x, x)), -x.copy()


def test_projected_lbfgs_separates_ivp_alpha_from_initial_curvature():
    memory = ProjectedLBFGSMemory(
        n_images=1, history_size=7, initial_curvature=70.0
    )
    gradient = np.array([0.0, 7.0, -14.0])
    tangent = np.array([1.0, 0.0, 0.0])
    direction = memory.direction(
        image_index=0,
        gradient=gradient,
        tangent=tangent,
        alpha=80.0,
    )
    np.testing.assert_allclose(direction, -gradient / 70.0)


def test_projected_lbfgs_can_use_physical_inverse_operator():
    memory = ProjectedLBFGSMemory(n_images=1, history_size=7)
    memory.observe(
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 2.0, 0.0]]),
    )
    memory.observe(
        np.array([[0.0, 0.1, 0.0]]),
        np.array([[0.0, 2.4, 0.0]]),
    )
    gradient = np.array([0.0, 1.0, 2.0])
    tangent = np.array([1.0, 0.0, 0.0])
    direction = memory.direction_with_operator(
        image_index=0,
        gradient=gradient,
        tangent=tangent,
        alpha=5.0,
        apply_initial_inverse=lambda vector: np.array(
            [vector[0], 0.5 * vector[1], 0.25 * vector[2]]
        ),
    )
    assert np.all(np.isfinite(direction))
    assert abs(np.dot(direction, tangent)) < 1e-14
    assert np.dot(direction, gradient) < 0.0


def test_adaptive_alpha_uses_trust_quality_not_a_call_schedule():
    grown, grow_action = adapt_proximal_alpha(
        2.0, quality=0.9, step_utilization=0.95, accepted_fraction=1.0,
        alpha_min=0.1, alpha_max=10.0,
    )
    held, hold_action = adapt_proximal_alpha(
        2.0, quality=0.6, step_utilization=0.95, accepted_fraction=1.0,
        alpha_min=0.1, alpha_max=10.0,
    )
    shrunk, shrink_action = adapt_proximal_alpha(
        2.0, quality=0.1, step_utilization=0.2, accepted_fraction=1.0,
        alpha_min=0.1, alpha_max=10.0,
    )
    assert (grown, grow_action) == (3.0, "grow")
    assert (held, hold_action) == (2.0, "hold")
    assert (shrunk, shrink_action) == (1.0, "shrink")


def test_curvature_warm_start_uses_a_state_trigger():
    assert not history_warm_start_is_active(False, 100.0, 50.0)
    assert not history_warm_start_is_active(True, 100.0, 101.0)
    assert history_warm_start_is_active(True, 100.0, 100.0)
    assert history_warm_start_is_active(True, -1.0, 1e9)


def test_ffprecon_red_black_masks_partition_internal_images():
    red = ffprecon_update_mask(20, 0, "red_black")
    black = ffprecon_update_mask(20, 1, "red_black")
    assert np.count_nonzero(red) == 10
    assert np.count_nonzero(black) == 10
    assert not np.any(red & black)
    assert np.all(red | black)
    assert np.all(ffprecon_update_mask(20, 7, "all"))


def test_residual_precision_cap_is_state_triggered():
    policy = "0.02:3,0.01:0.3"
    assert residual_precision_cap(5.0, 0.03, policy) == (5.0, np.inf)
    assert residual_precision_cap(5.0, 0.015, policy) == (3.0, 3.0)
    assert residual_precision_cap(5.0, 0.005, policy) == (0.3, 0.3)


def test_relative_iperp_precision_cap_is_dimensionless():
    policy = "1e-6:0.3"
    assert relative_iperp_precision_cap(5.0, 2e-6, policy) == (
        5.0,
        np.inf,
    )
    assert relative_iperp_precision_cap(5.0, 8e-7, policy) == (
        0.3,
        0.3,
    )


def test_state_trust_radius_scales_with_image_resolution():
    path = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    expected = 0.5 * 2.0 / np.sqrt(2.0)
    assert abs(rms_segment_trust_radius(path, 0.5) - expected) < 1e-14
    assert rms_segment_trust_radius(path, 0.5, maximum=0.25) == 0.25


def test_state_reparameterization_trigger_has_no_call_index():
    diagnostic = {"i_perp": 0.09, "segment_length_cv": 0.12}
    assert state_reparameterization_is_due(
        diagnostic,
        initial_i_perp=1e5,
        last_attempt_i_perp=np.inf,
        ready_relative_i_perp=1e-6,
        cv_trigger=0.10,
        retry_reduction=0.5,
    )
    assert not state_reparameterization_is_due(
        diagnostic,
        initial_i_perp=1e5,
        last_attempt_i_perp=0.1,
        ready_relative_i_perp=1e-6,
        cv_trigger=0.10,
        retry_reduction=0.5,
    )


def test_spring_proposal_chooses_scale_from_target_cv():
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    proposal, _, scale, before, after = spring_proposal_to_target_cv(
        path,
        target_cv=0.05,
        contraction=0.5,
        dtau=0.05,
        max_iter=1000,
        tolerance=1e-10,
    )
    assert 0.0 < scale <= 1.0
    assert after["segment_length_cv"] <= (
        0.5 * before["segment_length_cv"] + 1e-10
    )
    np.testing.assert_array_equal(proposal[0], path[0])
    np.testing.assert_array_equal(proposal[-1], path[-1])


def test_coordinate_change_without_property_request_does_not_count():
    backend = FakeBackend()
    evaluator = ImageEvaluator(backend, 0)
    x0 = np.arange(6.0)
    x1 = x0 + 1.0
    evaluator.query(x0, "force")
    before = evaluator.counters.joint_evaluations
    # The evaluator is deliberately not notified when external coordinates move.
    external = x1.copy()
    external += 2.0
    assert evaluator.counters.joint_evaluations == before


def test_many_coordinate_changes_count_only_the_final_requested_position():
    backend = FakeBackend()
    evaluator = ImageEvaluator(backend, 0)
    x0 = np.arange(6.0)
    evaluator.query(x0, "energy")
    external = x0.copy()
    external += 1.0
    external -= 3.0
    external *= 0.5
    assert evaluator.counters.joint_evaluations == 1
    evaluator.query(external, "force")
    assert evaluator.counters.joint_evaluations == 2
    assert backend.raw_joint_evaluations == 2


def test_change_then_restore_before_request_is_a_cache_hit():
    backend = FakeBackend()
    evaluator = ImageEvaluator(backend, 0)
    x0 = np.arange(6.0)
    evaluator.query(x0, "force")
    external = x0 + 0.5
    external = x0.copy()
    evaluator.query(external, "energy")
    assert evaluator.counters.joint_evaluations == 1
    assert evaluator.counters.cache_hits == 1


def test_joint_cache_force_then_energy_and_energy_then_force():
    backend = FakeBackend()
    evaluator = ImageEvaluator(backend, 0)
    x0 = np.arange(6.0)
    evaluator.query(x0, "force")
    evaluator.query(x0, "energy")
    evaluator.query(x0, "force")
    assert evaluator.counters.joint_evaluations == 1
    x1 = x0 + 1.0
    evaluator.query(x1, "energy")
    evaluator.query(x1, "force")
    assert evaluator.counters.joint_evaluations == 2
    evaluator.query(x0, "energy")
    assert evaluator.counters.joint_evaluations == 2
    assert backend.raw_joint_evaluations == 2


def test_one_ulp_coordinate_change_is_a_new_joint_evaluation():
    backend = FakeBackend()
    evaluator = ImageEvaluator(backend, 0)
    x0 = np.ones(6)
    evaluator.query(x0, "both")
    x1 = x0.copy()
    x1[3] = np.nextafter(x1[3], np.inf)
    evaluator.query(x1, "energy")
    assert evaluator.counters.joint_evaluations == 2
    assert backend.raw_joint_evaluations == 2


def test_mixed_request_accounting_and_cache_hit_identity():
    backend = FakeBackend()
    evaluator = ImageEvaluator(backend, 0)
    x0 = np.arange(6.0)
    x1 = x0 + 0.25
    evaluator.query(x0, "energy")
    evaluator.query(x0, "force")
    evaluator.query(x0, "both")
    evaluator.query(x1, "both")
    evaluator.query(x1, "energy")
    counters = evaluator.counters
    assert counters.energy_requests == 4
    assert counters.force_requests == 3
    assert counters.joint_evaluations == 2
    assert counters.cache_hits == 3
    assert counters.total_query_events == 5
    assert counters.both_requests == 2
    assert (
        counters.energy_requests
        + counters.force_requests
        - counters.both_requests
        == counters.total_query_events
    )
    assert (
        counters.cache_hits
        + counters.joint_evaluations
        + counters.budget_blocked_misses
        == counters.total_query_events
    )
    assert backend.raw_joint_evaluations == 2


def test_budget_blocks_only_a_new_coordinate_not_a_cache_hit():
    backend = FakeBackend()
    evaluator = ImageEvaluator(backend, 0, max_joint_evaluations=1)
    x0 = np.arange(6.0)
    evaluator.query(x0, "force")
    evaluator.query(x0, "energy")
    assert evaluator.counters.joint_evaluations == 1
    assert evaluator.counters.cache_hits == 1
    try:
        evaluator.query(x0 + 1.0, "force")
    except EvaluationBudgetReached:
        pass
    else:
        raise AssertionError("A new coordinate must respect the joint-call budget")
    assert evaluator.counters.joint_evaluations == 1
    assert evaluator.counters.budget_blocked_misses == 1
    assert evaluator.counters.total_query_events == 3
    assert (
        evaluator.counters.cache_hits
        + evaluator.counters.joint_evaluations
        + evaluator.counters.budget_blocked_misses
        == evaluator.counters.total_query_events
    )
    assert backend.raw_joint_evaluations == 1


def test_twenty_two_images_keep_independent_run_local_history_caches():
    backend = FakeBackend()
    evaluators = [ImageEvaluator(backend, index) for index in range(22)]
    coordinates = [np.arange(6.0) + index for index in range(22)]
    for evaluator, position in zip(evaluators, coordinates):
        evaluator.query(position, "both")
    for evaluator, position in zip(evaluators, coordinates):
        evaluator.query(position, "energy")
    evaluators[7].query(coordinates[7] + 0.25, "force")
    evaluators[7].query(coordinates[7], "energy")
    assert [item.counters.joint_evaluations for item in evaluators] == [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    report = audit_counter_consistency(backend, evaluators)
    assert report["consistent"]
    assert report["backend_raw_joint_evaluations"] == 23
    assert all(report["per_image_request_identity_ok"])
    assert all(report["per_image_query_outcome_identity_ok"])


def test_ase_force_then_energy_is_one_joint_evaluation():
    backend = FakeBackend(dimension=6)
    evaluator = ImageEvaluator(backend, 0)
    atoms = Atoms("H2", positions=np.arange(6.0).reshape(2, 3))
    atoms.calc = AuditedOpenMMCalculator(evaluator)
    atoms.get_forces()
    atoms.get_potential_energy()
    atoms.get_forces()
    assert evaluator.counters.joint_evaluations == 1
    assert evaluator.counters.force_requests == 2
    assert evaluator.counters.energy_requests == 1
    atoms.set_positions(atoms.get_positions() + 0.1)
    assert evaluator.counters.joint_evaluations == 1
    atoms.get_potential_energy()
    assert evaluator.counters.joint_evaluations == 2


def test_ase_neb_force_energy_sequence_does_not_double_count():
    backend = FakeBackend(dimension=3)
    images = [
        Atoms("H", positions=[[float(index), 0.1 * index, 0.0]])
        for index in range(4)
    ]
    evaluators = [ImageEvaluator(backend, index) for index in range(4)]
    for atoms, evaluator in zip(images, evaluators):
        atoms.calc = AuditedOpenMMCalculator(evaluator)
    neb = CentralDifferenceNEB(images, k=1.0)
    neb.get_forces()
    for image in images:
        image.get_potential_energy()
        image.get_forces()
    assert [item.counters.joint_evaluations for item in evaluators] == [1] * 4
    assert backend.raw_joint_evaluations == 4


def test_rejected_trial_counts_but_exact_return_reuses_run_local_history():
    backend = FakeBackend()
    evaluator = ImageEvaluator(backend, 0)
    accepted = np.arange(6.0)
    rejected = accepted + 0.25
    evaluator.query(accepted, "both")
    evaluator.query(rejected, "both")
    evaluator.query(accepted, "both")
    assert evaluator.counters.joint_evaluations == 2
    assert evaluator.counters.cache_hits == 1
    assert backend.raw_joint_evaluations == 2


def test_cache_history_is_not_shared_between_experiments():
    x = np.arange(6.0)
    first_backend = FakeBackend()
    first = ImageEvaluator(first_backend, 0)
    first.query(x, "both")

    second_backend = FakeBackend()
    second = ImageEvaluator(second_backend, 0)
    second.query(x, "both")
    assert first.counters.joint_evaluations == 1
    assert second.counters.joint_evaluations == 1
    assert first_backend.raw_joint_evaluations == 1
    assert second_backend.raw_joint_evaluations == 1


def test_checkpoint_restore_preserves_cache_and_counters():
    backend = FakeBackend()
    evaluator = ImageEvaluator(backend, 3)
    x = np.arange(6.0)
    evaluator.query(x, "force")
    state = evaluator.snapshot()
    restored_backend = FakeBackend()
    restored = ImageEvaluator(restored_backend, 3)
    restored.restore(state)
    restored.query(x, "energy")
    assert restored.counters.joint_evaluations == 1
    assert restored_backend.raw_joint_evaluations == 0


def test_checkpoint_restore_preserves_full_history_not_only_last_coordinate():
    backend = FakeBackend()
    evaluator = ImageEvaluator(backend, 3)
    x0 = np.arange(6.0)
    x1 = x0 + 0.5
    evaluator.query(x0, "both")
    evaluator.query(x1, "both")
    state = evaluator.snapshot()

    restored_backend = FakeBackend()
    restored = ImageEvaluator(restored_backend, 3)
    restored.restore(state)
    restored.query(x0, "energy")
    assert restored.counters.joint_evaluations == 2
    assert restored.counters.cache_hits == 1
    assert restored_backend.raw_joint_evaluations == 0


def test_full_checkpoint_restore_resynchronizes_backend_and_all_caches():
    backend = FakeBackend()
    evaluators = [ImageEvaluator(backend, index) for index in range(22)]
    coordinates = [np.arange(6.0) + index for index in range(22)]
    for evaluator, position in zip(evaluators, coordinates):
        evaluator.query(position, "both")
    states = [item.snapshot() for item in evaluators]

    restored_backend = FakeBackend()
    restored = [
        ImageEvaluator(restored_backend, index) for index in range(22)
    ]
    restore_evaluator_states(restored_backend, restored, states)
    assert restored_backend.raw_joint_evaluations == 22
    for evaluator, position in zip(restored, coordinates):
        evaluator.query(position, "energy")
    report = audit_counter_consistency(restored_backend, restored)
    assert report["consistent"]
    assert report["backend_raw_joint_evaluations"] == 22
    assert report["sum_image_joint_evaluations"] == 22


def test_real_openmm_joint_cache_matches_exact_coordinate_transitions():
    root = Path(
        os.environ.get(
            "CHIGNOLIN_TEST_ROOT", Path(__file__).resolve().parents[1]
        )
    )
    backend = OpenMMBackend(
        root / "inputs" / "chignolin_clean.pdb",
        platform_preference=("CPU",),
        precision="double",
        label="real-cache-test",
    )
    evaluator = ImageEvaluator(backend, 1)
    with np.load(root / "inputs" / "common_initial_path.npz") as saved:
        x0 = np.asarray(saved["initial_path"][1], dtype=float)
    evaluator.query(x0, "force")
    evaluator.query(x0.copy(), "energy")
    evaluator.query(x0, "both")
    assert evaluator.counters.joint_evaluations == 1
    assert evaluator.counters.cache_hits == 2
    x1 = x0.copy()
    x1[7] = np.nextafter(x1[7], np.inf)
    evaluator.query(x1, "energy")
    evaluator.query(x1, "force")
    evaluator.query(x0, "energy")
    assert evaluator.counters.joint_evaluations == 2
    assert evaluator.counters.energy_requests == 4
    assert evaluator.counters.force_requests == 3
    assert backend.raw_joint_evaluations == 2
    assert audit_counter_consistency(backend, [evaluator])["consistent"]


def test_force_field_preconditioner_is_normal_descent_and_call_free():
    root = Path(
        os.environ.get(
            "CHIGNOLIN_TEST_ROOT", Path(__file__).resolve().parents[1]
        )
    )
    backend = OpenMMBackend(
        root / "inputs" / "chignolin_clean.pdb",
        platform_preference=("CPU",),
        precision="double",
        label="real-ffprecon-test",
    )
    evaluators = [
        ImageEvaluator(backend, index) for index in range(22)
    ]
    with np.load(root / "inputs" / "common_initial_path.npz") as saved:
        path = np.asarray(saved["initial_path"], dtype=float)
    _, forces = evaluate_path(path, evaluators)
    tangents = central_tangents(path)
    gradients = -forces[1:-1]
    normal_gradients = gradients - np.sum(
        gradients * tangents, axis=1, keepdims=True
    ) * tangents
    bank = ForceFieldPreconditionerBank(
        backend,
        root / "inputs" / "chignolin_clean.pdb",
        c_stab=1.0,
    )
    raw_before = backend.raw_joint_evaluations
    bank.build(path)
    directions = bank.directions(
        normal_gradients,
        tangents,
        max_step=0.1,
        step_scales=np.ones(20),
    )
    assert backend.raw_joint_evaluations == raw_before
    np.testing.assert_allclose(
        np.sum(directions * tangents, axis=1), 0.0, atol=1e-11
    )
    assert np.all(np.sum(directions * normal_gradients, axis=1) < 0.0)


def test_lbfgsb_suboptimization_preserves_joint_call_accounting():
    backend = FakeBackend(dimension=6)
    evaluators = [ImageEvaluator(backend, index) for index in range(4)]
    path = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.4, 0.0, 0.2, -0.1, 0.0],
            [1.7, -0.3, 0.0, 0.5, 0.2, 0.0],
            [2.5, 0.0, 0.0, 0.8, 0.0, 0.0],
        ]
    )
    energies = np.zeros(4)
    forces = np.zeros_like(path)
    for index, evaluator in enumerate(evaluators):
        energies[index], forces[index] = evaluator.query(path[index], "both")
    initial_joint_calls = backend.raw_joint_evaluations
    (
        _,
        _,
        _,
        used_calls,
        synchronized_paths,
        _,
        _,
        synchronized_counters,
    ) = solve_lbfgsb_band(
        path,
        energies,
        forces,
        central_tangents(path),
        evaluators,
        alpha=1.0,
        max_step=0.1,
        initial_step=0.02,
        max_new_calls=4,
    )
    assert np.all(used_calls <= 4)
    assert len(synchronized_paths) == int(np.max(used_calls))
    assert len(synchronized_counters) == len(synchronized_paths)
    assert backend.raw_joint_evaluations == (
        initial_joint_calls + int(np.sum(used_calls))
    )
    report = audit_counter_consistency(backend, evaluators)
    assert report["consistent"]
    assert report["backend_raw_joint_evaluations"] == sum(
        report["per_image_joint_evaluations"]
    )
    for evaluator in evaluators:
        counters = evaluator.counters
        assert counters.energy_requests == counters.force_requests
        assert counters.both_requests == counters.total_query_events
        assert (
            counters.cache_hits
            + counters.joint_evaluations
            + counters.budget_blocked_misses
            == counters.total_query_events
        )


def test_projected_proximal_solver_reduces_kkt_without_tangent_null_space():
    backend = FakeBackend(dimension=6)
    evaluators = [ImageEvaluator(backend, index) for index in range(4)]
    path = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.4, 0.0, 0.2, -0.1, 0.0],
            [1.7, -0.3, 0.0, 0.5, 0.2, 0.0],
            [2.5, 0.0, 0.0, 0.8, 0.0, 0.0],
        ]
    )
    energies, forces = evaluate_path(path, evaluators)
    tangents = central_tangents(path)
    initial_diagnostic = path_diagnostics(path, energies, forces[1:-1])
    initial_calls = backend.raw_joint_evaluations
    (
        candidate,
        candidate_energies,
        candidate_forces,
        used_calls,
        synchronized_paths,
        _,
        _,
        _,
        report,
    ) = solve_projected_proximal_band(
        path,
        energies,
        forces,
        tangents,
        evaluators,
        alpha=10.0,
        max_step=0.2,
        max_new_calls=20,
        kkt_rtol=0.1,
    )
    displacement = candidate[1:-1] - path[1:-1]
    np.testing.assert_allclose(
        np.sum(displacement * tangents, axis=1),
        0.0,
        atol=1e-12,
    )
    assert np.all(report["final_kkt_l2"] < report["initial_kkt_l2"])
    assert np.all(report["relative_kkt"] <= 0.1 + 1e-10)
    final_diagnostic = path_diagnostics(
        candidate, candidate_energies, candidate_forces[1:-1]
    )
    assert final_diagnostic["i_perp"] < initial_diagnostic["i_perp"]
    assert len(synchronized_paths) == int(np.max(used_calls))
    assert backend.raw_joint_evaluations == initial_calls + int(
        np.sum(used_calls)
    )
    assert audit_counter_consistency(backend, evaluators)["consistent"]


def test_curve_projected_proximal_solver_is_equidistant_and_audited():
    backend = FakeBackend(dimension=3)
    evaluators = [ImageEvaluator(backend, index) for index in range(5)]
    angles = np.linspace(0.0, 1.2, 5)
    path = np.column_stack(
        (2.0 + np.cos(angles), np.sin(angles), np.zeros_like(angles))
    )
    energies, forces = evaluate_path(path, evaluators)
    initial_diagnostic = path_diagnostics(path, energies, forces[1:-1])
    initial_calls = backend.raw_joint_evaluations
    (
        candidate,
        candidate_energies,
        candidate_forces,
        used_calls,
        synchronized_paths,
        _,
        _,
        synchronized_counters,
        report,
    ) = solve_curve_projected_proximal_band(
        path,
        energies,
        forces,
        central_tangents(path),
        evaluators,
        alpha=1.0,
        max_step=0.2,
        max_new_calls=20,
        kkt_rtol=0.81,
    )
    assert np.all(report["converged"])
    assert np.all(report["relative_kkt"] <= 0.81 + 1e-12)
    assert len(synchronized_paths) == len(synchronized_counters)
    assert len(synchronized_paths) == int(np.max(used_calls))
    for trace_path in synchronized_paths:
        assert (
            equidistance_diagnostics(trace_path)["segment_length_cv"]
            < 1e-12
        )
    final_diagnostic = path_diagnostics(
        candidate, candidate_energies, candidate_forces[1:-1]
    )
    assert final_diagnostic["i_perp"] < initial_diagnostic["i_perp"]
    assert backend.raw_joint_evaluations == initial_calls + int(
        np.sum(used_calls)
    )
    assert audit_counter_consistency(backend, evaluators)["consistent"]


def test_contractive_proximal_solver_selects_alpha_and_solves_same_subproblem():
    backend = FakeBackend(dimension=6)
    evaluators = [ImageEvaluator(backend, index) for index in range(4)]
    path = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.8, 0.4, 0.0, 0.2, -0.1, 0.0],
            [1.7, -0.3, 0.0, 0.5, 0.2, 0.0],
            [2.5, 0.0, 0.0, 0.8, 0.0, 0.0],
        ]
    )
    energies, forces = evaluate_path(path, evaluators)
    tangents = central_tangents(path)
    initial_diagnostic = path_diagnostics(path, energies, forces[1:-1])
    initial_calls = backend.raw_joint_evaluations
    (
        candidate,
        candidate_energies,
        candidate_forces,
        used_calls,
        _,
        _,
        _,
        _,
        report,
    ) = solve_contractive_proximal_band(
        path,
        energies,
        forces,
        tangents,
        evaluators,
        alpha=1.0,
        max_step=0.2,
        max_new_calls=20,
        contraction_limit=0.8,
        kkt_rtol=1e-4,
    )
    assert report["effective_alpha"] == 0.5
    assert report["alpha_halvings"] == 1
    assert report["probe_accepted"]
    assert np.all(report["relative_kkt"] <= 1e-4 + 1e-10)
    final_diagnostic = path_diagnostics(
        candidate, candidate_energies, candidate_forces[1:-1]
    )
    assert final_diagnostic["i_perp"] < initial_diagnostic["i_perp"]
    assert backend.raw_joint_evaluations == initial_calls + int(
        np.sum(used_calls)
    )
    assert audit_counter_consistency(backend, evaluators)["consistent"]


def test_central_tangent_and_unormalized_i_perp():
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    tangent = central_tangents(path)
    np.testing.assert_allclose(tangent, [[1.0, 0.0, 0.0]] * 2)
    forces = np.array([[0.0, 2.0, 0.0], [0.0, 3.0, 0.0]])
    diag = path_diagnostics(path, np.zeros(4), forces)
    assert diag["i_perp"] == 13.0
    assert "normalized_i_perp" not in diag


def test_equidistance_integral_is_zero_only_for_uniform_straight_sampling():
    uniform = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    nonuniform = uniform.copy()
    nonuniform[1, 0] = 0.2
    uniform_diag = equidistance_diagnostics(uniform)
    nonuniform_diag = equidistance_diagnostics(nonuniform)
    assert uniform_diag["equidistance_integral"] == 0.0
    assert uniform_diag["segment_length_cv"] == 0.0
    assert nonuniform_diag["equidistance_integral"] > 0.0
    assert nonuniform_diag["segment_length_cv"] > 0.0
    ds = 1.0 / (nonuniform.shape[0] - 1)
    lengths = np.linalg.norm(np.diff(nonuniform, axis=0), axis=1)
    identity = np.sum(np.diff(lengths**2) ** 2) / (4.0 * ds**5)
    np.testing.assert_allclose(
        nonuniform_diag["equidistance_integral"], identity
    )


def test_central_neb_runtime_tangent_matches_shared_function():
    positions = [
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[1.0, 1.0, 0.0]]),
        np.array([[3.0, 1.0, 0.0]]),
    ]
    images = [Atoms("H", positions=p) for p in positions]
    neb = CentralDifferenceNEB(images, k=1.0)
    state = type("State", (), {})()
    spring1 = type("Spring", (), {"t": positions[1] - positions[0]})()
    spring2 = type("Spring", (), {"t": positions[2] - positions[1]})()
    got = neb.neb_method.get_tangent(state, spring1, spring2, 1).reshape(-1)
    expected = central_tangents(
        np.asarray([p.reshape(-1) for p in positions])
    )[0]
    np.testing.assert_allclose(got, expected)
    assert neb.method == "centraldifference"


def test_spring_reparameterization_uses_central_geometry():
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    reparameterized, residual = spring_reparameterize(path)
    lengths = np.linalg.norm(
        reparameterized[1:] - reparameterized[:-1], axis=1
    )
    assert residual < 1e-4
    assert np.max(lengths) - np.min(lengths) < 1e-3


def test_geometric_reparameterization_improves_spacing_without_backend_call():
    backend = FakeBackend()
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    candidate, _, applied, before, after = (
        geometric_spring_reparameterization(path, np.ones(2))
    )
    assert applied
    assert after["equidistance_integral"] < before["equidistance_integral"]
    assert after["segment_length_cv"] < before["segment_length_cv"]
    assert not np.array_equal(candidate, path)
    assert backend.raw_joint_evaluations == 0


def test_euler_spring_microstep_is_current_central_tangent_and_call_free():
    backend = FakeBackend()
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.3, 0.0],
            [1.8, 0.4, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    candidate, _, applied, before, after, _ = (
        euler_spring_reparameterization(path, np.ones(2), dtau=0.02)
    )
    assert applied
    displacement = candidate[1:-1] - path[1:-1]
    normal = displacement - (
        np.sum(displacement * central_tangents(path), axis=1)[:, None]
        * central_tangents(path)
    )
    np.testing.assert_allclose(normal, 0.0, atol=1e-14)
    assert after["equidistance_integral"] < before["equidistance_integral"]
    assert after["segment_length_cv"] <= before["segment_length_cv"]
    assert backend.raw_joint_evaluations == 0


def test_central_hermite_spring_equilibrium_is_call_free_and_more_uniform():
    backend = FakeBackend()
    angles = np.array([0.0, 0.08, 0.45, 1.05, 1.57])
    path = np.column_stack(
        (np.cos(angles), np.sin(angles), np.zeros_like(angles))
    )
    candidate, residual = central_hermite_spring_equilibrium(
        path, dense_points=4001
    )
    before = equidistance_diagnostics(path)
    after = equidistance_diagnostics(candidate)
    np.testing.assert_array_equal(candidate[[0, -1]], path[[0, -1]])
    assert after["equidistance_integral"] < before["equidistance_integral"]
    assert after["segment_length_cv"] < before["segment_length_cv"]
    assert residual < np.max(np.abs(np.diff(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    assert backend.raw_joint_evaluations == 0


def test_reparameterization_checks_objective_energy_and_central_force():
    current = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    candidate = current.copy()
    candidate[1, 1] = 1.0
    energies = np.zeros(5)
    candidate_energies = energies.copy()
    candidate_energies[2] = 0.1
    forces = np.zeros_like(current)
    forces[1:-1, 1] = 1.0
    candidate_forces = forces.copy()
    candidate_forces[3, 1] = 3.0
    accepted, checks = reparameterization_acceptance_mask(
        current,
        candidate,
        energies,
        candidate_energies,
        forces,
        candidate_forces,
        alpha=1.0,
        objective_tolerance_kcal=1.0,
        energy_tolerance_kcal=1.0,
        force_rtol=0.2,
    )
    assert not checks["objective_ok"][0]
    assert not checks["energy_ok"][1]
    assert not checks["force_ok"][2]
    assert not np.any(accepted)

    accepted_same, _ = reparameterization_acceptance_mask(
        current,
        current,
        energies,
        energies,
        forces,
        forces,
        alpha=1.0,
        objective_tolerance_kcal=1.0,
        energy_tolerance_kcal=1.0,
        force_rtol=0.2,
    )
    assert np.all(accepted_same)


def test_global_iperp_guard_greedily_keeps_only_improving_images():
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    energies = np.zeros(4)
    forces = np.zeros_like(path)
    forces[1:-1, 1] = 1.0
    candidate_forces = forces.copy()
    candidate_forces[1, 1] = 0.5
    candidate_forces[2, 1] = 3.0
    (
        accepted_path,
        accepted_energies,
        accepted_forces,
        accepted_diag,
        accepted,
    ) = greedy_global_iperp_acceptance(
        path,
        energies,
        forces,
        path,
        energies,
        candidate_forces,
        np.array([True, True]),
        global_iperp_rtol=0.0,
    )
    np.testing.assert_array_equal(accepted, [True, False])
    np.testing.assert_allclose(accepted_path, path)
    np.testing.assert_allclose(accepted_energies, energies)
    assert accepted_forces[1, 1] == 0.5
    assert accepted_forces[2, 1] == 1.0
    assert accepted_diag["i_perp"] < path_diagnostics(
        path, energies, forces[1:-1]
    )["i_perp"]


def test_global_best_acceptance_uses_no_new_evaluations_and_improves():
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    energies = np.zeros(4)
    forces = np.zeros_like(path)
    forces[1:-1, 1] = 2.0
    candidate_forces = forces.copy()
    candidate_forces[1, 1] = 1.5
    candidate_forces[2, 1] = 0.25
    result = greedy_global_iperp_acceptance(
        path,
        energies,
        forces,
        path,
        energies,
        candidate_forces,
        np.array([True, True]),
        global_iperp_rtol=0.0,
        order="global_best",
    )
    assert np.all(result[-1])
    assert result[3]["i_perp"] < path_diagnostics(
        path, energies, forces[1:-1]
    )["i_perp"]


def test_collective_iperp_accepts_coupled_central_tangent_improvement():
    current_path = np.array(
        [
            [0.82057869, -0.40166815, -0.87017237],
            [-1.89380714, -0.39365671, -0.03090325],
            [-0.08405436, -0.09379193, -1.12180963],
            [-0.06627389, -0.03867338, 1.29056204],
        ]
    )
    candidate_path = np.array(
        [
            [0.82057869, -0.40166815, -0.87017237],
            [-0.96044041, -0.46215123, -0.41405623],
            [-0.11654542, -0.39760257, -1.49302654],
            [-0.06627389, -0.03867338, 1.29056204],
        ]
    )
    current_forces = np.array(
        [
            [-0.05865015, -1.04330443, 0.60610675],
            [-0.10391555, 0.25000831, -0.18294004],
            [-0.72725445, -0.94795956, -0.23727600],
            [-0.54876034, 0.23390133, -0.00443170],
        ]
    )
    candidate_forces = np.array(
        [
            [-0.05865015, -1.04330443, 0.60610675],
            [-0.78506892, 0.28356896, -0.85437024],
            [-1.03549326, -1.09513636, -1.27491264],
            [-0.54876034, 0.23390133, -0.00443170],
        ]
    )
    energies = np.zeros(4)
    result = greedy_global_iperp_acceptance(
        current_path,
        energies,
        current_forces,
        candidate_path,
        energies,
        candidate_forces,
        np.array([True, True]),
        0.0,
        order="collective_first",
    )
    accepted_path, _, _, accepted_diag, accepted = result
    current_i_perp = path_diagnostics(
        current_path, energies, current_forces[1:-1]
    )["i_perp"]
    np.testing.assert_array_equal(accepted, [True, True])
    np.testing.assert_array_equal(accepted_path, candidate_path)
    assert accepted_diag["i_perp"] < current_i_perp


def test_bicriteria_recovery_rejects_spacing_destroying_merge():
    current_path = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    candidate_path = current_path.copy()
    candidate_path[1, 0] = 0.5
    candidate_path[2, 0] = 2.5
    energies = np.zeros(4)
    current_forces = np.zeros_like(current_path)
    current_forces[1:-1, 1] = 1.0
    candidate_forces = np.zeros_like(candidate_path)
    result = greedy_bicriteria_recovery_acceptance(
        current_path,
        energies,
        current_forces,
        candidate_path,
        energies,
        candidate_forces,
        np.array([True, True]),
        equidistance_ceiling=0.0,
    )
    np.testing.assert_array_equal(result[0], current_path)
    np.testing.assert_array_equal(result[-1], [False, False])


def test_bicriteria_recovery_accepts_force_improvement_at_fixed_spacing():
    path = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )
    energies = np.zeros(4)
    current_forces = np.zeros_like(path)
    current_forces[1:-1, 1] = 1.0
    candidate_forces = np.zeros_like(path)
    result = greedy_bicriteria_recovery_acceptance(
        path,
        energies,
        current_forces,
        path,
        energies,
        candidate_forces,
        np.array([True, True]),
        equidistance_ceiling=0.0,
    )
    np.testing.assert_array_equal(result[-1], [True, True])
    assert result[3]["i_perp"] == 0.0
    assert result[3]["equidistance_integral"] == 0.0


def test_bicriteria_recovery_tries_collective_coupled_state_first(
    monkeypatch,
):
    current_path = np.arange(15.0).reshape(5, 3)
    candidate_path = current_path.copy()
    candidate_path[1:-1, 1] += np.array([1.0, -1.0, 1.0])
    energies = np.zeros(5)
    forces = np.zeros((5, 3))

    def diagnostic(path, unused_energies, unused_internal_forces):
        moved = np.count_nonzero(
            np.linalg.norm(
                path[1:-1] - current_path[1:-1], axis=1
            )
            > 0.0
        )
        return {
            "i_perp": 0.5 if moved == 3 else (1.0 if moved == 0 else 2.0),
            "equidistance_integral": 0.5 if moved == 3 else 1.0,
        }

    monkeypatch.setattr(run_ivm, "path_diagnostics", diagnostic)
    result = greedy_bicriteria_recovery_acceptance(
        current_path,
        energies,
        forces,
        candidate_path,
        energies,
        forces,
        np.ones(3, dtype=bool),
        equidistance_ceiling=0.75,
        max_block_size=1,
    )
    np.testing.assert_array_equal(result[0], candidate_path)
    np.testing.assert_array_equal(result[-1], [True, True, True])
    assert result[3]["i_perp"] == 0.5


def test_ffprecon_proximal_acceptance_uses_ivp_objective():
    eligible, objective = ffprecon_subproblem_acceptance_mask(
        current_energies=np.array([2.0, 3.0]),
        candidate_energies=np.array([1.98, 2.999]),
        displacements=np.array([[0.2, 0.0], [0.2, 0.0]]),
        alpha=2.0,
        energy_rtol=0.0,
        mode="proximal_merit",
    )
    np.testing.assert_allclose(objective, [1.99, 3.009])
    np.testing.assert_array_equal(eligible, [True, False])


def test_ffprecon_proximal_acceptance_is_independent_of_coupled_force_gate():
    eligible, _ = ffprecon_subproblem_acceptance_mask(
        current_energies=np.array([2.0]),
        candidate_energies=np.array([1.8]),
        displacements=np.array([[0.1, 0.0]]),
        alpha=1.0,
        energy_rtol=0.0,
        mode="proximal_merit",
        current_per_image_l2=np.array([0.1]),
        candidate_per_image_l2=np.array([100.0]),
        force_rtol=0.0,
    )
    np.testing.assert_array_equal(eligible, [True])


def test_counter_audit_matches_backend():
    backend = FakeBackend()
    evaluators = [ImageEvaluator(backend, i) for i in range(3)]
    for i, evaluator in enumerate(evaluators):
        evaluator.query(np.arange(6.0) + i, "force")
    report = audit_counter_consistency(backend, evaluators)
    assert report["consistent"]
    assert report["backend_raw_joint_evaluations"] == 3
