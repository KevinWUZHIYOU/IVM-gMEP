"""Persistent unit and state-machine tests for the Ala2D V14 IR-IVM core.

These tests use only analytic/synthetic in-memory mean-force bands.  They do
not instantiate the restrained-MD oracle and therefore spend no formal NoF.
"""

from pathlib import Path
from types import SimpleNamespace
import json
import sys
from typing import Optional

import numpy as np
import pytest


CAMPAIGN = Path(__file__).resolve().parents[1]
SRC = CAMPAIGN / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ala2d_irivm_v14 as v14
from ala2d_article_neb_reparam import ArticleReparameterizationResult
from ala2d_oracles import OracleBand
from riemannian_torus_geometry import periodic_difference, unwrap_path


class SyntheticTenBlockOracle:
    """Deterministic, path-dependent ten-block oracle for core tests."""

    oracle_id = "v14-pytest-synthetic-ten-block"
    formal_evidence = False

    def __init__(self, noise: float):
        self.noise = float(noise)
        self.band_calls = 0
        self.call_count = 0

    def provenance(self):
        # Deliberately independent of call history so a resumed request has the
        # same immutable manifest as its first invocation.
        return {
            "oracle_id": self.oracle_id,
            "formal_evidence": False,
            "noise": self.noise,
        }

    def evaluate_band(self, full_path, band_id, band_directory=None):
        del band_directory
        self.band_calls += 1
        self.call_count += v14.N_INTERNAL
        centers = np.asarray(full_path[1:-1], dtype=np.float64)
        gradient = np.column_stack(
            (
                0.35 * np.sin(centers[:, 0]) + 0.1,
                0.28 * np.cos(centers[:, 1]) - 0.07,
            )
        )
        offsets = np.linspace(-1.0, 1.0, 10)
        pattern = np.column_stack(
            (
                np.linspace(0.7, 1.3, v14.N_INTERNAL),
                np.linspace(-1.2, -0.8, v14.N_INTERNAL),
            )
        )
        gradient_blocks = (
            gradient[:, None, :]
            + self.noise * offsets[None, :, None] * pattern[:, None, :]
        )
        metric = np.repeat(
            np.eye(v14.DIMENSION)[None, :, :], v14.N_INTERNAL, axis=0
        )
        metric_blocks = np.repeat(metric[:, None, :, :], 10, axis=1)
        frame_counts = np.full(v14.N_INTERNAL, 2501, dtype=np.int64)
        discarded = np.full(v14.N_INTERNAL, 125, dtype=np.int64)
        block_counts = v14._block_frame_counts(frame_counts - discarded, 10)
        gradient = v14._weighted_block_mean(gradient_blocks, block_counts)
        metric = v14._weighted_block_mean(metric_blocks, block_counts)
        return OracleBand(
            centers=centers.copy(),
            mean_coordinates=centers.copy(),
            grad_F=gradient,
            M=metric,
            grad_F_block_means=gradient_blocks,
            M_block_means=metric_blocks,
            seeds=np.arange(v14.N_INTERNAL, dtype=np.int64)
            + 1000 * int(band_id),
            thermostat_seeds=np.arange(v14.N_INTERNAL, dtype=np.int64)
            + 2000 * int(band_id),
            frame_counts=frame_counts,
            discarded_frame_counts=discarded,
            wall_seconds=np.zeros(v14.N_INTERNAL),
            nof=v14.N_INTERNAL,
            metadata={"synthetic": True, "band_id": int(band_id)},
        )


@pytest.fixture
def causal_initial():
    with np.load(
        CAMPAIGN
        / "inputs/reference/MFEP_INITIAL_PATH_current_ivm_state0.npz",
        allow_pickle=False,
    ) as loaded:
        return (
            np.asarray(loaded["full_path_init"], dtype=np.float64),
            np.asarray(loaded["M_init"], dtype=np.float64),
            np.asarray(loaded["F_init"], dtype=np.float64),
        )


def _load_state(run_directory: Path, state_index: int):
    return v14._load_npz_record(v14._state_path(run_directory, state_index))


def _assert_records_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        assert v14._array_equal(left[key], right[key]), key


def _run(
    run_directory: Path,
    initial,
    *,
    budget: int,
    stop: Optional[int] = None,
    noise: float = 1.0e-3,
    controls: Optional[v14.IRIVMControls] = None,
    oracle: Optional[SyntheticTenBlockOracle] = None,
):
    path, metric, gradient = initial
    selected_oracle = oracle or SyntheticTenBlockOracle(noise)
    selected_controls = controls or v14.IRIVMControls(
        alpha=1.0e-4, r=0.5, m_max=4
    )
    result = v14.run_irivm(
        oracle=selected_oracle,
        controls=selected_controls,
        initial_path=path,
        initial_M=metric,
        initial_grad_F=gradient,
        nof_budget=budget,
        stop_after_nof=stop,
        run_directory=run_directory,
    )
    return result, selected_oracle


def test_controls_enforce_three_grid_dimensions_and_frozen_safeguards():
    assert (
        v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4).validate().m_max
        == 4
    )
    for alpha in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="alpha"):
            v14.IRIVMControls(alpha=alpha, r=0.4, m_max=4).validate()
    for residual_fraction in (0.0, -0.1, 0.8, 1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="r"):
            v14.IRIVMControls(
                alpha=1.0e-4, r=residual_fraction, m_max=4
            ).validate()
    for maximum_steps in (True, 3, 3.5):
        with pytest.raises(ValueError, match="m_max"):
            v14.IRIVMControls(
                alpha=1.0e-4, r=0.4, m_max=maximum_steps
            ).validate()
    with pytest.raises(ValueError, match="history_size"):
        v14.IRIVMControls(
            alpha=1.0e-4, r=0.4, m_max=4, history_size=6
        ).validate()
    with pytest.raises(ValueError, match="residual_bound_kappa"):
        v14.IRIVMControls(
            alpha=1.0e-4,
            r=0.4,
            m_max=4,
            residual_bound_kappa=2.0,
        ).validate()


def test_initial_missing_se_is_explicit_and_all_later_bands_cost_twenty(
    tmp_path, causal_initial
):
    run_directory = tmp_path / "initial-special"
    _, oracle = _run(run_directory, causal_initial, budget=20)
    initial = _load_state(run_directory, 0)
    first_paid = _load_state(run_directory, 1)

    assert not initial["residual_standard_error_available"].any()
    assert np.isnan(initial["residual_standard_error"]).all()
    assert bool(initial["initial_no_se_special_case_available"])
    assert str(initial["controller_phase"].item()) == "INITIAL_NO_SE_SPECIAL_CASE"
    assert bool(first_paid["initial_no_se_special_case_used"])
    assert str(first_paid["selected_candidate_kind"].item()) == "H0"
    assert bool(first_paid["paid_action_is_movement"])
    assert int(first_paid["nof_this_complete_band"]) == 20
    assert int(first_paid["nof_mean_force_incremental"]) == 20
    assert int(first_paid["nof_mean_force_absolute_including_shared_initial"]) == 40
    assert int(first_paid["evaluated_candidate_count"]) == 1
    assert int(first_paid["preselection_mean_force_evaluations"]) == 0
    assert bool(first_paid["only_selected_complete_band_evaluated"])
    assert oracle.band_calls == 1
    assert oracle.call_count == 20
    trajectory = v14.load_public_trajectory(run_directory)
    np.testing.assert_array_equal(
        trajectory["nof_mean_force_incremental"], np.array([0, 20])
    )


def test_pause_resume_one_state_ahead_replay_and_budget_prefix_are_exact(
    tmp_path, causal_initial
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.5, m_max=4)
    resume_directory = tmp_path / "resume"
    resume_oracle = SyntheticTenBlockOracle(1.0e-3)

    paused, _ = _run(
        resume_directory,
        causal_initial,
        budget=80,
        stop=20,
        controls=controls,
        oracle=resume_oracle,
    )
    assert paused["paused_at_requested_stop"]
    assert resume_oracle.call_count == 20
    continued, _ = _run(
        resume_directory,
        causal_initial,
        budget=80,
        stop=60,
        controls=controls,
        oracle=resume_oracle,
    )
    assert continued["nof_mean_force_incremental"] == 60
    checkpoint_at_60 = (resume_directory / "controller_checkpoint.npz").read_bytes()
    completed, _ = _run(
        resume_directory,
        causal_initial,
        budget=80,
        stop=80,
        controls=controls,
        oracle=resume_oracle,
    )
    assert completed["completed"]
    assert resume_oracle.call_count == 80

    # Simulate an interruption after immutable state 4 was published but before
    # the replaceable checkpoint advanced from state 3.  Resume must replay the
    # existing public state without another oracle call.
    (resume_directory / "controller_checkpoint.npz").write_bytes(
        checkpoint_at_60
    )
    replayed, _ = _run(
        resume_directory,
        causal_initial,
        budget=80,
        stop=80,
        controls=controls,
        oracle=resume_oracle,
    )
    assert replayed["completed"]
    assert resume_oracle.call_count == 80
    np.testing.assert_array_equal(
        v14.load_public_trajectory(resume_directory)[
            "nof_mean_force_incremental"
        ],
        np.arange(0, 81, 20),
    )
    analyzer_view = v14.load_public_trajectory(resume_directory)
    for key in (
        "selected_reparameterization_output_residual",
        "state_kind",
        "segment_length_relative_deviation",
        "full_path_unwrapped",
        "M",
        "grad_F",
        "nof_mean_force_incremental",
    ):
        assert key in analyzer_view

    short_directory = tmp_path / "budget-40"
    long_prefix_directory = tmp_path / "budget-80-stopped-40"
    _run(short_directory, causal_initial, budget=40, controls=controls)
    _run(
        long_prefix_directory,
        causal_initial,
        budget=80,
        stop=40,
        controls=controls,
    )
    for state_index in range(3):
        _assert_records_equal(
            _load_state(short_directory, state_index),
            _load_state(long_prefix_directory, state_index),
        )


def test_exact_center_hold_reuses_path_certificate_and_pools_all_blocks(
    tmp_path, causal_initial
):
    run_directory = tmp_path / "hold"
    _, oracle = _run(run_directory, causal_initial, budget=40, noise=1.0e6)
    source = _load_state(run_directory, 1)
    held = _load_state(run_directory, 2)

    assert str(held["selected_candidate_kind"].item()) == "HOLD"
    assert str(held["state_kind"].item()) == v14.HOLD_STATE_KIND
    assert np.array_equal(source["full_path_unwrapped"], held["full_path_unwrapped"])
    assert bool(held["hold_reused_exact_path_bytes"])
    assert int(held["candidate_reparameterization_calls"]) == 0
    assert not bool(held["reparameterization_applied_before_evaluation"])
    assert not bool(held["paid_action_is_movement"])
    assert int(held["inner_step"]) == int(source["next_inner_step"])
    assert int(held["next_inner_step"]) == int(source["next_inner_step"])
    assert int(held["joint_lbfgs_pair_count_next"]) == int(
        source["joint_lbfgs_pair_count_next"]
    )
    assert not bool(held["secant_stored_after_transition"])
    np.testing.assert_array_equal(
        held["secant_s_composite_actual_u"], np.zeros(v14.N_INTERNAL)
    )
    assert bool(held["hold_reused_source_reparameterization_certificate"])
    for key in (
        "selected_reparameterization_audit_sha256",
        "selected_reparameterization_iterations",
        "selected_reparameterization_output_residual",
    ):
        assert v14._array_equal(held[key], source[key]), key

    expected_gradient_blocks = np.concatenate(
        (
            source["grad_F_block_means"],
            held["oracle_raw_grad_F_block_means"],
        ),
        axis=1,
    )
    expected_metric_blocks = np.concatenate(
        (source["M_block_means"], held["oracle_raw_M_block_means"]), axis=1
    )
    np.testing.assert_array_equal(
        held["grad_F_block_means"], expected_gradient_blocks
    )
    np.testing.assert_array_equal(held["M_block_means"], expected_metric_blocks)
    expected_counts = np.concatenate(
        (source["block_frame_counts"], held["oracle_raw_block_frame_counts"]),
        axis=1,
    )
    np.testing.assert_array_equal(held["block_frame_counts"], expected_counts)
    np.testing.assert_allclose(
        held["grad_F"],
        v14._weighted_block_mean(expected_gradient_blocks, expected_counts),
        atol=0.0,
    )
    np.testing.assert_allclose(
        held["M"],
        v14._weighted_block_mean(expected_metric_blocks, expected_counts),
        atol=0.0,
    )
    assert int(held["block_count"]) == 20
    assert int(held["oracle_raw_block_count"]) == 10
    assert int(held["nof_this_complete_band"]) == 20
    assert oracle.call_count == 40


def _analytic_controller(controls, *, inner_step=0, state_index=0):
    with np.load(
        CAMPAIGN
        / "inputs/reference/MFEP_INITIAL_PATH_current_ivm_state0.npz",
        allow_pickle=False,
    ) as loaded:
        path = np.asarray(loaded["full_path_init"], dtype=np.float64)
    metric = np.repeat(np.eye(2)[None, :, :], v14.N_INTERNAL, axis=0)
    provisional_gradient = np.zeros((v14.N_INTERNAL, 2), dtype=np.float64)
    frame = v14._frozen_frame(path, metric, provisional_gradient)
    component = 10.0 / np.sqrt(v14.N_INTERNAL)
    gradient = frame["basis"] * component
    blocks = np.repeat(gradient[:, None, :], 10, axis=1)
    controller = v14._new_controller(
        path,
        metric,
        gradient,
        blocks,
        controls,
        np.repeat(metric[:, None, :, :], 10, axis=1),
    )
    if state_index:
        controller["state_index"] = state_index
        controller["nof"] = state_index * v14.N_INTERNAL
        controller["inner_step"] = inner_step
        controller["outer_anchor_lower_norm"] = np.asarray(10.0)
        controller["outer_anchor_upper_norm"] = np.asarray(10.0)
        controller["outer_anchor_bounds_ready"] = True
        controller["observation_state_indices"][0] = state_index
        controller["current_block_frame_counts"] = np.full(
            (v14.N_INTERNAL, 10), 10, dtype=np.int64
        )
        refreshed = v14._residual_statistics(
            path=controller["current_path"],
            grad=controller["current_grad"],
            grad_blocks=controller["current_grad_blocks"],
            frozen_path=controller["frozen_path"],
            frozen_basis=controller["frozen_basis"],
            frozen_G=controller["frozen_G"],
            alpha=controls.alpha,
            block_frame_counts=controller["current_block_frame_counts"],
        )
        controller["lbfgs_anchor_u"] = refreshed.normal_coordinate
        controller["lbfgs_anchor_raw_residual"] = refreshed.raw_residual
        controller["lbfgs_anchor_standard_error"] = refreshed.standard_error
    v14._validate_controller(controller, controls)
    return controller


def _identity_reparameterization(path, **unused):
    del unused
    result_path = unwrap_path(np.asarray(path, dtype=np.float64))
    segment_lengths = np.linalg.norm(
        periodic_difference(result_path[1:], result_path[:-1]), axis=1
    )
    return ArticleReparameterizationResult(
        path=result_path,
        iteration_count=1,
        stopping_pre_update_spring_residual=0.0,
        output_spring_residual=0.0,
        segment_lengths=segment_lengths,
        maximum_neighbor_length_difference=0.0,
        actual_chord_relative_deviation=0.0,
        actual_I_arc=0.0,
        audit_sha256="identity-reparameterization-for-pytest",
    )


def test_proximal_score_contains_quadratic_and_rejects_oversized_step(
    monkeypatch,
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls)
    statistics = v14._residual_statistics(
        path=controller["current_path"],
        grad=controller["current_grad"],
        grad_blocks=controller["current_grad_blocks"],
        frozen_path=controller["frozen_path"],
        frozen_basis=controller["frozen_basis"],
        frozen_G=controller["frozen_G"],
        alpha=controls.alpha,
    )
    _, _, radii = v14._trust_radii(controller["frozen_path"], controls)
    monkeypatch.setattr(v14, "reparameterize_article_neb", _identity_reparameterization)

    small = v14._build_candidate(
        label="small",
        direction=-controls.alpha * statistics.effective_residual,
        controller=controller,
        statistics=statistics,
        radii=radii,
        controls=controls,
    )
    oversized = v14._build_candidate(
        label="oversized",
        direction=-3.0 * controls.alpha * statistics.effective_residual,
        controller=controller,
        statistics=statistics,
        radii=radii,
        controls=controls,
    )
    assert small.valid
    assert small.residual_linear_decrease > small.proximal_quadratic_penalty
    assert small.admissibility_score > 0.0
    assert not oversized.valid
    assert oversized.error == "post_reparameterization_admissibility_score_nonpositive"
    assert oversized.residual_linear_decrease > 0.0
    assert oversized.proximal_quadratic_penalty > oversized.residual_linear_decrease
    assert oversized.admissibility_score < 0.0
    np.testing.assert_allclose(
        small.admissibility_score,
        small.residual_linear_decrease - small.proximal_quadratic_penalty,
    )
    np.testing.assert_allclose(
        oversized.admissibility_score,
        oversized.residual_linear_decrease
        - oversized.proximal_quadratic_penalty,
    )


def test_actual_post_reparameterization_delta_is_audited_and_secant_shift_once(
    tmp_path, causal_initial
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.5, m_max=4)
    run_directory = tmp_path / "actual-post-r"
    _run(run_directory, causal_initial, budget=40, controls=controls)
    first = _load_state(run_directory, 1)
    second = _load_state(run_directory, 2)

    assert str(first["selected_candidate_kind"].item()) == "H0"
    actual_u = v14._normal_coordinate(
        first["full_path_unwrapped"],
        first["frozen_reference_path_unwrapped"],
        first["frozen_reference_normal_basis"],
        first["frozen_reference_G"],
    )
    np.testing.assert_allclose(
        first["h0_actual_normal_coordinate_after_reparameterization"],
        actual_u,
        rtol=0.0,
        atol=2.0e-15,
    )
    actual_delta = actual_u - first["source_normal_coordinate"]
    np.testing.assert_allclose(
        first["h0_actual_normal_step_after_reparameterization"],
        actual_delta,
        rtol=0.0,
        atol=2.0e-15,
    )
    # The real article map changes the nominal normal proposal, so this also
    # catches implementations that save the pre-reparameterization direction.
    assert np.max(
        np.abs(actual_delta - first["h0_capped_direction"])
    ) > 1.0e-10

    basis = second["frozen_reference_normal_basis"]
    frozen_path = second["frozen_reference_path_unwrapped"]
    frozen_G = second["frozen_reference_G"]
    source_u = v14._normal_coordinate(
        second["source_paid_path_unwrapped"], frozen_path, basis, frozen_G
    )
    next_u = v14._normal_coordinate(
        second["full_path_unwrapped"], frozen_path, basis, frozen_G
    )
    secant_s = next_u - source_u
    projected_gradient_change = np.einsum(
        "ni,ni->n", basis, second["grad_F"] - first["grad_F"]
    )
    expected_y = projected_gradient_change + secant_s / controls.alpha
    duplicated_shift_y = projected_gradient_change + 2.0 * secant_s / controls.alpha
    np.testing.assert_allclose(
        second["secant_s_composite_actual_u"], secant_s, rtol=0.0, atol=2.0e-14
    )
    np.testing.assert_allclose(
        second["secant_y_raw_proximal_residual"],
        expected_y,
        rtol=2.0e-13,
        atol=2.0e-12,
    )
    assert np.linalg.norm(
        second["secant_y_raw_proximal_residual"] - duplicated_shift_y
    ) > 1.0e-6


def _engineered_advance(controls, *, inner_step, target_residual_norm):
    controller = _analytic_controller(
        controls, inner_step=inner_step, state_index=1
    )
    statistics = v14._residual_statistics(
        path=controller["current_path"],
        grad=controller["current_grad"],
        grad_blocks=controller["current_grad_blocks"],
        frozen_path=controller["frozen_path"],
        frozen_basis=controller["frozen_basis"],
        frozen_G=controller["frozen_G"],
        alpha=controls.alpha,
    )
    delta = np.full(v14.N_INTERNAL, 1.0e-5, dtype=np.float64)
    selected_path = controller["current_path"].copy()
    selected_path[1:-1] += controller["frozen_basis"] * delta[:, None]
    selected_path = unwrap_path(selected_path)
    actual_u = v14._normal_coordinate(
        selected_path,
        controller["frozen_path"],
        controller["frozen_basis"],
        controller["frozen_G"],
    )
    target_component = target_residual_norm / np.sqrt(v14.N_INTERNAL)
    projected_gradient = target_component - actual_u / controls.alpha
    gradient = controller["frozen_basis"] * projected_gradient[:, None]
    gradient_blocks = np.repeat(gradient[:, None, :], 10, axis=1)
    metric = np.repeat(np.eye(2)[None, :, :], v14.N_INTERNAL, axis=0)
    metric_blocks = np.repeat(metric[:, None, :, :], 10, axis=1)
    prepared = SimpleNamespace(
        source_checkpoint_sha256=v14._checkpoint_digest(controller),
        selected_path=selected_path,
        selected_label="H0",
        source_statistics=statistics,
        latch_after=False,
        initial_no_se_special_case=False,
    )
    after, audit = v14._advance_controller(
        controller,
        prepared,
        M=metric,
        grad=gradient,
        grad_blocks=gradient_blocks,
        M_blocks=metric_blocks,
        frame_counts=np.full(v14.N_INTERNAL, 2501, dtype=np.int64),
        discarded_frame_counts=np.full(v14.N_INTERNAL, 125, dtype=np.int64),
        controls=controls,
    )
    return after, audit


def test_r_and_m_max_produce_real_engineered_state_branches():
    low_r = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=8)
    high_r = v14.IRIVMControls(alpha=1.0e-4, r=0.7, m_max=8)
    after_low_r, audit_low_r = _engineered_advance(
        low_r, inner_step=1, target_residual_norm=5.5
    )
    after_high_r, audit_high_r = _engineered_advance(
        high_r, inner_step=1, target_residual_norm=5.5
    )
    assert not audit_low_r["r_stop"]
    assert not audit_low_r["stopped_inner"]
    assert after_low_r["outer_index"] == 0
    assert after_low_r["inner_step"] == 2
    assert audit_high_r["r_stop"]
    assert audit_high_r["stopped_inner"]
    assert audit_high_r["inner_stop_reason"] == "global_r_interval_stop"
    assert after_high_r["outer_index"] == 1
    assert after_high_r["inner_step"] == 0
    assert not np.array_equal(
        after_low_r["frozen_path"], after_high_r["frozen_path"]
    )

    cap_four = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    cap_eight = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=8)
    after_four, audit_four = _engineered_advance(
        cap_four, inner_step=3, target_residual_norm=9.0
    )
    after_eight, audit_eight = _engineered_advance(
        cap_eight, inner_step=3, target_residual_norm=9.0
    )
    assert audit_four["m_max_stop"]
    assert audit_four["stopped_inner"]
    assert audit_four["inner_stop_reason"] == "m_max_movement_stop"
    assert after_four["outer_index"] == 1
    assert after_four["inner_step"] == 0
    assert not audit_eight["m_max_stop"]
    assert not audit_eight["stopped_inner"]
    assert after_eight["outer_index"] == 0
    assert after_eight["inner_step"] == 4
    assert not np.array_equal(after_four["frozen_path"], after_eight["frozen_path"])


def test_unequal_np_array_split_weights_mean_se_and_covariance_exactly():
    block_values = np.array([1.0, 3.0, 8.0])
    blocks = np.zeros((v14.N_INTERNAL, 3, 2), dtype=np.float64)
    blocks[:, :, 0] = block_values
    counts = v14._block_frame_counts(
        np.full(v14.N_INTERNAL, 23, dtype=np.int64), 3
    )
    np.testing.assert_array_equal(counts[0], np.array([8, 8, 7]))
    expected_mean = float(np.dot(counts[0], block_values) / 23.0)
    weighted = v14._weighted_block_mean(blocks, counts)
    np.testing.assert_allclose(weighted[:, 0], expected_mean)

    covariance, available = v14._gradient_covariance_of_mean(blocks, counts)
    expected_variance = float(
        np.dot(counts[0], (block_values - expected_mean) ** 2) / (2.0 * 23.0)
    )
    assert available.all()
    np.testing.assert_allclose(covariance[:, 0, 0], expected_variance)

    path = np.column_stack(
        (np.linspace(-1.0, 1.0, v14.N_PATH), np.linspace(0.2, 0.8, v14.N_PATH))
    )
    metric = np.repeat(np.eye(2)[None, :, :], v14.N_INTERNAL, axis=0)
    basis = np.repeat(np.array([[1.0, 0.0]]), v14.N_INTERNAL, axis=0)
    stats = v14._residual_statistics(
        path=path,
        grad=weighted,
        grad_blocks=blocks,
        frozen_path=path,
        frozen_basis=basis,
        frozen_G=metric,
        alpha=1.0e-4,
        block_frame_counts=counts,
    )
    np.testing.assert_allclose(stats.standard_error**2, expected_variance)

    equal_counts = np.full((v14.N_INTERNAL, 3), 5, dtype=np.int64)
    equal_stats = v14._residual_statistics(
        path=path,
        grad=blocks.mean(axis=1),
        grad_blocks=blocks,
        frozen_path=path,
        frozen_basis=basis,
        frozen_G=metric,
        alpha=1.0e-4,
        block_frame_counts=equal_counts,
    )
    np.testing.assert_allclose(
        equal_stats.standard_error,
        np.std(block_values, ddof=1) / np.sqrt(3.0),
    )


def test_run_rejects_multiblock_historical_state_zero(tmp_path, causal_initial):
    path, metric, gradient = causal_initial
    with pytest.raises(ValueError, match="state 0.*exactly one"):
        v14.run_irivm(
            oracle=SyntheticTenBlockOracle(1.0e-3),
            controls=v14.IRIVMControls(alpha=1.0e-4, r=0.5, m_max=4),
            initial_path=path,
            initial_M=metric,
            initial_grad_F=gradient,
            initial_grad_F_block_means=np.repeat(gradient[:, None, :], 2, axis=1),
            initial_M_block_means=np.repeat(metric[:, None, :, :], 2, axis=1),
            nof_budget=20,
            run_directory=tmp_path / "bad-state-zero",
        )


def test_exact_score_tie_prefers_h0_and_no_candidate_is_terminal(monkeypatch):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls, inner_step=2, state_index=2)
    controller["lbfgs_s"][:2] = np.eye(2, v14.N_INTERNAL) * 1.0e-4
    controller["lbfgs_y"][:2] = np.eye(2, v14.N_INTERNAL)
    controller["lbfgs_count"] = 2
    monkeypatch.setattr(v14, "reparameterize_article_neb", _identity_reparameterization)
    monkeypatch.setattr(
        v14,
        "joint_lbfgs_direction",
        lambda gradient, history_s, history_y, diagonal_h0: -controls.alpha
        * gradient,
    )
    prepared, terminal = v14._prepare_update(controller, controls)
    assert terminal == {}
    assert prepared is not None
    assert prepared.selected_label == "H0"
    assert not prepared.hold_selected

    initial = _analytic_controller(controls)
    monkeypatch.setattr(
        v14,
        "_build_candidate",
        lambda label, direction, controller, statistics, radii, controls: v14._invalid_candidate(
            label, "engineered-invalid"
        ),
    )
    prepared, terminal = v14._prepare_update(initial, controls)
    assert prepared is None
    assert terminal["phase"] == "TERMINAL"
    assert terminal["reason"] == "no_valid_motion_candidate_terminal"


def test_zero_nof_terminal_is_durable_and_resume_makes_no_oracle_call(
    tmp_path, causal_initial
):
    path, metric, _ = causal_initial
    zero = np.zeros((v14.N_INTERNAL, 2), dtype=np.float64)
    oracle = SyntheticTenBlockOracle(1.0e-3)
    kwargs = dict(
        oracle=oracle,
        controls=v14.IRIVMControls(alpha=1.0e-4, r=0.5, m_max=4),
        initial_path=path,
        initial_M=metric,
        initial_grad_F=zero,
        nof_budget=40,
        run_directory=tmp_path / "durable-terminal",
    )
    first = v14.run_irivm(**kwargs)
    second = v14.run_irivm(**kwargs)
    assert first["phase"] == second["phase"] == "TERMINAL"
    assert first["nof_mean_force_incremental"] == 0
    assert oracle.call_count == 0


def test_reprojection_and_global_latch_use_warm_scalar_without_cold(
    monkeypatch, tmp_path
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls, inner_step=3, state_index=3)
    frame = {
        "path": controller["frozen_path"],
        "basis": controller["frozen_basis"],
        "G": controller["frozen_G"],
    }
    paths = np.zeros_like(controller["observation_paths"])
    gradients = np.zeros_like(controller["observation_gradients"])
    covariances = np.zeros_like(controller["observation_gradient_covariances"])
    available = np.zeros_like(controller["observation_covariance_available"])
    indices = np.zeros_like(controller["observation_state_indices"])
    for slot, coordinate in enumerate((-2.0e-5, -1.0e-5, 0.0)):
        paths[slot] = controller["frozen_path"]
        paths[slot, 1:-1] += controller["frozen_basis"] * coordinate
        available[slot] = True
        indices[slot] = slot + 1
    observations = {
        "observation_paths": paths,
        "observation_gradients": gradients,
        "observation_gradient_covariances": covariances,
        "observation_covariance_available": available,
        "observation_state_indices": indices,
        "observation_count": np.asarray(3),
    }
    rebuilt = v14._reproject_observation_history(
        observations, frame, controls, controls.alpha, -1
    )
    assert rebuilt["lbfgs_count"] == 2
    np.testing.assert_allclose(rebuilt["lbfgs_y"][:2], rebuilt["lbfgs_s"][:2] / controls.alpha)
    assert rebuilt["warm_safeguard_source_state_index"] == 3

    controller["lbfgs_latched"] = True
    controller["lbfgs_count"] = 0
    controller["warm_safeguard_inverse_scale"] = np.asarray(controls.alpha)
    controller["warm_safeguard_source_state_index"] = 2
    monkeypatch.setattr(v14, "reparameterize_article_neb", _identity_reparameterization)
    prepared, terminal = v14._prepare_update(controller, controls)
    assert terminal == {}
    assert prepared is not None
    assert prepared.selected_label == "WARM_SCALAR_SAFEGUARD"
    assert not prepared.h0_candidate.valid
    assert "cold_forbidden" in prepared.h0_candidate.error

    band = SyntheticTenBlockOracle(0.0).evaluate_band(
        prepared.selected_path, band_id=4
    )
    after, audit = v14._advance_controller(
        controller,
        prepared,
        M=band.M,
        grad=band.grad_F,
        grad_blocks=band.grad_F_block_means,
        M_blocks=band.M_block_means,
        frame_counts=band.frame_counts,
        discarded_frame_counts=band.discarded_frame_counts,
        controls=controls,
    )
    assert audit["m_max_stop"]
    assert after["outer_index"] == controller["outer_index"] + 1
    assert after["lbfgs_latched"]
    assert after["warm_safeguard_source_state_index"] == 2
    checkpoint_directory = tmp_path / "global-latch-checkpoint"
    v14._save_checkpoint(checkpoint_directory, after)
    restored = v14._controller_from_checkpoint(
        v14._load_npz_record(checkpoint_directory / "controller_checkpoint.npz"),
        controls,
    )
    assert restored["lbfgs_latched"]
    assert v14._controllers_equal(after, restored)
    next_prepared, next_terminal = v14._prepare_update(after, controls)
    assert next_terminal == {}
    assert next_prepared is not None
    assert next_prepared.selected_label == "WARM_SCALAR_SAFEGUARD"
    assert not next_prepared.h0_candidate.valid


def test_unready_anchor_hold_rebuilds_frame_from_weighted_pool(monkeypatch):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls, state_index=1)
    offsets = np.linspace(-100.0, 100.0, 10)
    controller["current_grad_blocks"] = controller["frozen_basis"][:, None, :] * (
        1.0 + offsets[None, :, None]
    )
    controller["current_grad"] = v14._weighted_block_mean(
        controller["current_grad_blocks"], controller["current_block_frame_counts"]
    )
    controller["outer_anchor_bounds_ready"] = False
    controller["outer_anchor_lower_norm"] = np.asarray(np.nan)
    controller["outer_anchor_upper_norm"] = np.asarray(np.nan)
    current_stats = v14._residual_statistics(
        path=controller["current_path"],
        grad=controller["current_grad"],
        grad_blocks=controller["current_grad_blocks"],
        frozen_path=controller["frozen_path"],
        frozen_basis=controller["frozen_basis"],
        frozen_G=controller["frozen_G"],
        alpha=controls.alpha,
        block_frame_counts=controller["current_block_frame_counts"],
    )
    controller["lbfgs_anchor_raw_residual"] = current_stats.raw_residual
    controller["lbfgs_anchor_standard_error"] = current_stats.standard_error
    controller["observation_gradients"][0] = controller["current_grad"]
    controller["observation_gradient_covariances"][0], controller[
        "observation_covariance_available"
    ][0] = v14._gradient_covariance_of_mean(
        controller["current_grad_blocks"], controller["current_block_frame_counts"]
    )
    v14._validate_controller(controller, controls)
    prepared, terminal = v14._prepare_update(controller, controls)
    assert terminal == {} and prepared is not None and prepared.hold_selected

    new_metric_blocks = np.repeat(
        np.diag([2.0, 1.0])[None, None, :, :],
        v14.N_INTERNAL,
        axis=0,
    )
    new_metric_blocks = np.repeat(new_metric_blocks, 10, axis=1)
    new_gradient_blocks = controller["current_grad_blocks"].copy()
    retained = np.full(v14.N_INTERNAL, 2376, dtype=np.int64)
    new_counts = v14._block_frame_counts(retained, 10)
    after, audit = v14._advance_controller(
        controller,
        prepared,
        M=v14._weighted_block_mean(new_metric_blocks, new_counts),
        grad=v14._weighted_block_mean(new_gradient_blocks, new_counts),
        grad_blocks=new_gradient_blocks,
        M_blocks=new_metric_blocks,
        frame_counts=np.full(v14.N_INTERNAL, 2501, dtype=np.int64),
        discarded_frame_counts=np.full(v14.N_INTERNAL, 125, dtype=np.int64),
        controls=controls,
    )
    assert audit["frame_rebuilt"]
    np.testing.assert_allclose(after["frozen_M"], after["current_M"])
    assert not np.allclose(after["frozen_M"], controller["frozen_M"])


def test_post_reparameterization_path_must_remain_inside_anchor_trust(monkeypatch):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls)
    statistics = v14._residual_statistics(
        path=controller["current_path"],
        grad=controller["current_grad"],
        grad_blocks=controller["current_grad_blocks"],
        frozen_path=controller["frozen_path"],
        frozen_basis=controller["frozen_basis"],
        frozen_G=controller["frozen_G"],
        alpha=controls.alpha,
        block_frame_counts=controller["current_block_frame_counts"],
    )
    _, _, radii = v14._trust_radii(controller["frozen_path"], controls)

    def escaping_reparameterization(path, **unused):
        del path, unused
        escaped = controller["frozen_path"].copy()
        escaped[1:-1] += controller["frozen_basis"] * (2.0 * radii[:, None])
        base = _identity_reparameterization(escaped)
        return base

    monkeypatch.setattr(v14, "reparameterize_article_neb", escaping_reparameterization)
    candidate = v14._build_candidate(
        label="escape",
        direction=-controls.alpha * statistics.effective_residual,
        controller=controller,
        statistics=statistics,
        radii=radii,
        controls=controls,
    )
    assert not candidate.valid
    assert candidate.error == (
        "post_reparameterization_path_exceeds_frozen_anchor_trust_radii"
    )


def test_replay_rejects_intermediate_pooled_phase_latch_and_secant_corruption(
    tmp_path, causal_initial
):
    run_directory = tmp_path / "strict-intermediate-replay"
    _run(run_directory, causal_initial, budget=60, noise=1.0e-3)
    state_path = v14._state_path(run_directory, 1)
    original_bytes = state_path.read_bytes()

    def perturb_array(record, key):
        record[key] = np.asarray(record[key]).copy()
        record[key].flat[0] += 1

    mutations = {
        "grad_F_block_means": lambda record: perturb_array(
            record, "grad_F_block_means"
        ),
        "M_block_means": lambda record: perturb_array(record, "M_block_means"),
        "block_frame_counts": lambda record: perturb_array(
            record, "block_frame_counts"
        ),
        "controller_phase_after_paid_band": lambda record: record.__setitem__(
            "controller_phase_after_paid_band", np.asarray("CORRUPTED")
        ),
        "one_way_takeover_latched_next": lambda record: record.__setitem__(
            "one_way_takeover_latched_next",
            np.asarray(not bool(record["one_way_takeover_latched_next"])),
        ),
        "secant_y_raw_proximal_residual": lambda record: perturb_array(
            record, "secant_y_raw_proximal_residual"
        ),
        "selected_reparameterization_audit_sha256": lambda record: record.__setitem__(
            "selected_reparameterization_audit_sha256", np.asarray("CORRUPTED")
        ),
    }
    for key, mutate in mutations.items():
        record = v14._load_npz_record(state_path)
        mutate(record)
        with state_path.open("wb") as stream:
            np.savez_compressed(stream, **record)
        with pytest.raises(v14.IRIVMError, match=key):
            v14.load_public_trajectory(run_directory)
        state_path.write_bytes(original_bytes)

    trajectory = v14.load_public_trajectory(run_directory)
    assert len(trajectory["grad_F_block_means"]) == 4
    assert len(trajectory["M_block_means"]) == 4
    assert len(trajectory["block_frame_counts"]) == 4
    assert trajectory["controller_phase"].shape == (4,)
    assert trajectory["one_way_takeover_latched"].shape == (4,)

    manifest_path = run_directory / "run_manifest.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["implementation_source_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(v14.IRIVMError, match="implementation source hash"):
        v14.load_public_trajectory(run_directory)
    manifest_path.write_bytes(original_manifest)
    v14.load_public_trajectory(run_directory)
