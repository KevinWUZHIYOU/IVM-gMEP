"""Persistent unit and state-machine tests for the Ala2D V17 IR-IVM core.

These tests use only analytic/synthetic in-memory mean-force bands.  They do
not instantiate the restrained-MD oracle and therefore spend no formal NoF.
"""

from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
from typing import Optional

import numpy as np
import pytest


CAMPAIGN = Path(__file__).resolve().parents[1]
SRC = CAMPAIGN / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ala2d_irivm_v17 as v14
from ala2d_article_neb_reparam import ArticleReparameterizationResult
from ala2d_oracles import OracleBand
from riemannian_torus_geometry import (
    arclength_residual,
    periodic_difference,
    unwrap_path,
)


class SyntheticTenBlockOracle:
    """Deterministic, path-dependent ten-block oracle for core tests."""

    oracle_id = "v15-pytest-synthetic-ten-block"
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
        CAMPAIGN / "tests/fixtures/causal_initial_state0.npz",
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
        CAMPAIGN / "tests/fixtures/causal_initial_state0.npz",
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


def _stale_outer_controller(controls, *, inner_step=1, state_index=1):
    """Return a valid paid controller whose current path moved from its anchor."""

    controller = _analytic_controller(
        controls, inner_step=inner_step, state_index=state_index
    )
    controller["current_path"] = controller["current_path"].copy()
    controller["current_path"][1:-1] += (
        controller["frozen_basis"] * 1.0e-5
    )
    last = int(controller["observation_count"]) - 1
    controller["observation_paths"][last] = controller["current_path"]
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
    controller["lbfgs_anchor_u"] = statistics.normal_coordinate.copy()
    controller["lbfgs_anchor_raw_residual"] = statistics.raw_residual.copy()
    controller["lbfgs_anchor_standard_error"] = statistics.standard_error.copy()
    v14._validate_controller(controller, controls)
    assert not v14._frozen_outer_is_current_paid_frame(controller)
    return controller


def _identity_reparameterization(path, **unused):
    audit_source_path = np.asarray(
        unused.pop("audit_source_path", path), dtype=np.float64
    )
    controls = {
        "k_s": float(unused.get("k_s", 1.0)),
        "dtau": float(unused.get("dtau", 0.5)),
        "tolerance": float(unused.get("tolerance", 1.0e-6)),
        "max_iterations": int(unused.get("max_iterations", 20000)),
    }
    result_path = unwrap_path(np.asarray(path, dtype=np.float64))
    segment_lengths = np.linalg.norm(
        periodic_difference(result_path[1:], result_path[:-1]), axis=1
    )
    neighbor_difference = float(np.max(np.abs(np.diff(segment_lengths))))
    chord_deviation = float(
        np.max(
            np.abs(segment_lengths / float(np.mean(segment_lengths)) - 1.0)
        )
    )
    provisional = ArticleReparameterizationResult(
        path=result_path,
        iteration_count=1,
        stopping_pre_update_spring_residual=0.0,
        output_spring_residual=0.0,
        segment_lengths=segment_lengths,
        maximum_neighbor_length_difference=neighbor_difference,
        actual_chord_relative_deviation=chord_deviation,
        actual_I_arc=float(arclength_residual(result_path)),
        audit_sha256="",
    )
    audit = hashlib.sha256(
        v14._canonical_json(
            provisional.audit_payload(source_path=audit_source_path, **controls)
        )
    ).hexdigest()
    return ArticleReparameterizationResult(
        path=provisional.path,
        iteration_count=provisional.iteration_count,
        stopping_pre_update_spring_residual=(
            provisional.stopping_pre_update_spring_residual
        ),
        output_spring_residual=provisional.output_spring_residual,
        segment_lengths=provisional.segment_lengths,
        maximum_neighbor_length_difference=(
            provisional.maximum_neighbor_length_difference
        ),
        actual_chord_relative_deviation=(
            provisional.actual_chord_relative_deviation
        ),
        actual_I_arc=provisional.actual_I_arc,
        audit_sha256=audit,
    )


def test_effective_proximal_value_is_diagnostic_not_structural_validity(
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
    assert small.proposal_effective_linear_term > small.proposal_quadratic_penalty
    assert small.proposal_effective_proximal_value > 0.0
    assert oversized.valid
    assert oversized.error == "none"
    assert oversized.proposal_effective_linear_term > 0.0
    assert (
        oversized.proposal_quadratic_penalty
        > oversized.proposal_effective_linear_term
    )
    assert oversized.proposal_effective_proximal_value < 0.0
    np.testing.assert_allclose(
        small.proposal_effective_proximal_value,
        small.proposal_effective_linear_term - small.proposal_quadratic_penalty,
    )
    np.testing.assert_allclose(
        oversized.proposal_effective_proximal_value,
        oversized.proposal_effective_linear_term
        - oversized.proposal_quadratic_penalty,
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
        hold_selected=False,
        warm_validation_selected=False,
        frozen_warm_candidate=None,
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


def _set_noisy_projected_residual(
    controller, controls, *, residual_norm, component_standard_error
):
    """Replace the current paid estimate by an analytic ten-block residual."""

    block_count = 10
    offsets = np.linspace(-1.0, 1.0, block_count)
    offset_scale = (
        float(component_standard_error)
        * np.sqrt(block_count)
        / np.std(offsets, ddof=1)
    )
    residual_center = float(residual_norm) / np.sqrt(v14.N_INTERNAL)
    residual_blocks = residual_center + offset_scale * offsets
    current_u = v14._normal_coordinate(
        controller["current_path"],
        controller["frozen_path"],
        controller["frozen_basis"],
        controller["frozen_G"],
    )
    projected_gradient_blocks = (
        residual_blocks[None, :] - current_u[:, None] / controls.alpha
    )
    gradient_blocks = (
        controller["frozen_basis"][:, None, :]
        * projected_gradient_blocks[:, :, None]
    )
    block_counts = np.full(
        (v14.N_INTERNAL, block_count), 10, dtype=np.int64
    )
    gradient = v14._weighted_block_mean(gradient_blocks, block_counts)
    controller["current_grad_blocks"] = gradient_blocks
    controller["current_grad"] = gradient
    controller["current_block_frame_counts"] = block_counts
    latest = int(controller["observation_count"]) - 1
    controller["observation_gradients"][latest] = gradient
    covariance, available = v14._gradient_covariance_of_mean(
        gradient_blocks, block_counts
    )
    controller["observation_gradient_covariances"][latest] = covariance
    controller["observation_covariance_available"][latest] = available
    statistics = v14._residual_statistics(
        path=controller["current_path"],
        grad=gradient,
        grad_blocks=gradient_blocks,
        frozen_path=controller["frozen_path"],
        frozen_basis=controller["frozen_basis"],
        frozen_G=controller["frozen_G"],
        alpha=controls.alpha,
        block_frame_counts=block_counts,
    )
    controller["lbfgs_anchor_u"] = statistics.normal_coordinate.copy()
    controller["lbfgs_anchor_raw_residual"] = statistics.raw_residual.copy()
    controller["lbfgs_anchor_standard_error"] = statistics.standard_error.copy()
    v14._validate_controller(controller, controls)
    return statistics


def _target_band_in_source_frame(
    controller, selected_path, controls, *, residual_norm, component_standard_error
):
    block_count = 10
    offsets = np.linspace(-1.0, 1.0, block_count)
    offset_scale = (
        float(component_standard_error)
        * np.sqrt(block_count)
        / np.std(offsets, ddof=1)
    )
    residual_center = float(residual_norm) / np.sqrt(v14.N_INTERNAL)
    residual_blocks = residual_center + offset_scale * offsets
    actual_u = v14._normal_coordinate(
        selected_path,
        controller["frozen_path"],
        controller["frozen_basis"],
        controller["frozen_G"],
    )
    projected_gradient_blocks = (
        residual_blocks[None, :] - actual_u[:, None] / controls.alpha
    )
    gradient_blocks = (
        controller["frozen_basis"][:, None, :]
        * projected_gradient_blocks[:, :, None]
    )
    block_counts = np.full(
        (v14.N_INTERNAL, block_count), 10, dtype=np.int64
    )
    metric = np.repeat(np.eye(2)[None, :, :], v14.N_INTERNAL, axis=0)
    metric_blocks = np.repeat(metric[:, None, :, :], block_count, axis=1)
    return {
        "M": metric,
        "grad": v14._weighted_block_mean(gradient_blocks, block_counts),
        "grad_blocks": gradient_blocks,
        "M_blocks": metric_blocks,
        "frame_counts": np.full(v14.N_INTERNAL, 110, dtype=np.int64),
        "discarded_frame_counts": np.full(v14.N_INTERNAL, 10, dtype=np.int64),
    }


def test_r_interval_ambiguity_keeps_moving_until_clear_stop_or_m_max(monkeypatch):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.7, m_max=4)
    monkeypatch.setattr(v14, "reparameterize_article_neb", _identity_reparameterization)

    controller = _analytic_controller(controls, inner_step=1, state_index=1)
    current = _set_noisy_projected_residual(
        controller,
        controls,
        residual_norm=6.0,
        component_standard_error=0.1,
    )
    assert current.lower_norm < controls.r * 10.0 < current.upper_norm

    prepared, terminal = v14._prepare_update(controller, controls)
    assert terminal == {}
    assert prepared is not None
    assert not prepared.hold_selected
    assert prepared.selected_label == "H0"
    assert not np.array_equal(prepared.selected_path, controller["current_path"])

    ambiguous_band = _target_band_in_source_frame(
        controller,
        prepared.selected_path,
        controls,
        residual_norm=6.0,
        component_standard_error=0.1,
    )
    after_ambiguous, ambiguous_audit = v14._advance_controller(
        controller, prepared, controls=controls, **ambiguous_band
    )
    assert ambiguous_audit["residual_decision_ambiguous"]
    assert not ambiguous_audit["stopped_inner"]
    assert ambiguous_audit["inner_stop_reason"] == (
        "continue_same_frozen_outer_r_stop_not_resolved"
    )
    assert after_ambiguous["outer_index"] == controller["outer_index"]
    assert after_ambiguous["inner_step"] == 2

    later, later_terminal = v14._prepare_update(after_ambiguous, controls)
    assert later_terminal == {}
    assert later is not None and not later.hold_selected
    clear_band = _target_band_in_source_frame(
        after_ambiguous,
        later.selected_path,
        controls,
        residual_norm=4.0,
        component_standard_error=0.1,
    )
    after_clear, clear_audit = v14._advance_controller(
        after_ambiguous, later, controls=controls, **clear_band
    )
    assert clear_audit["r_stop"]
    assert clear_audit["stopped_inner"]
    assert clear_audit["inner_stop_reason"] == "global_r_interval_stop"
    assert after_clear["outer_index"] == controller["outer_index"] + 1

    capped = _analytic_controller(controls, inner_step=3, state_index=3)
    _set_noisy_projected_residual(
        capped,
        controls,
        residual_norm=6.0,
        component_standard_error=0.1,
    )
    cap_prepared, cap_terminal = v14._prepare_update(capped, controls)
    assert cap_terminal == {}
    assert cap_prepared is not None and not cap_prepared.hold_selected
    cap_band = _target_band_in_source_frame(
        capped,
        cap_prepared.selected_path,
        controls,
        residual_norm=6.0,
        component_standard_error=0.1,
    )
    after_cap, cap_audit = v14._advance_controller(
        capped, cap_prepared, controls=controls, **cap_band
    )
    assert cap_audit["residual_decision_ambiguous"]
    assert cap_audit["m_max_stop"]
    assert cap_audit["inner_stop_reason"] == "m_max_movement_stop"
    assert after_cap["outer_index"] == capped["outer_index"] + 1


def _engineered_candidate(label, controller, step, score):
    selected_path = np.asarray(controller["current_path"], dtype=np.float64).copy()
    selected_path[1:-1] += controller["frozen_basis"] * step[:, None]
    reparameterization = _identity_reparameterization(selected_path)
    return v14.Candidate(
        label=label,
        valid=True,
        error="none",
        raw_direction=step.copy(),
        capped_direction=step.copy(),
        common_cap_scale=1.0,
        common_cap_truncated=False,
        raw_path=selected_path.copy(),
        reparameterization=reparameterization,
        actual_normal_coordinate=step.copy(),
        actual_normal_step=step.copy(),
        proposal_effective_linear_term=float(score) + 0.5,
        proposal_quadratic_penalty=0.5,
        proposal_effective_proximal_value=float(score),
        anchor_cap_theoretical_scale=1.0,
        anchor_cap_binary_bracket_used=False,
        anchor_cap_binary_bracket_iterations=0,
        anchor_cap_source_distances=np.zeros(v14.N_INTERNAL),
        anchor_cap_raw_path_distances=np.zeros(v14.N_INTERNAL),
        post_r_backtrack_count=0,
        post_r_backtrack_factor=1.0,
        total_direction_scale=1.0,
        post_r_attempt_count=1,
        post_r_attempt_scales=np.asarray([1.0]),
        post_r_attempt_status=np.asarray(["accepted_inside_anchor_trust"]),
        post_r_attempt_raw_path_sha256=np.asarray(["engineered"]),
        post_r_attempt_reparameterization_audit_sha256=np.asarray(
            [reparameterization.audit_sha256]
        ),
        post_r_attempt_max_anchor_distance=np.asarray([0.0]),
        trust_globalization_exhausted=False,
    )


@pytest.mark.parametrize(
    "lbfgs_score, expected_label, expected_resolved",
    [
        (1.005, "H0", False),
        (1.020, "LBFGS", True),
    ],
)
def test_pre_takeover_uses_h0_unless_lbfgs_advantage_is_resolved(
    monkeypatch, lbfgs_score, expected_label, expected_resolved
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls, inner_step=0, state_index=1)
    _set_noisy_projected_residual(
        controller,
        controls,
        residual_norm=6.0,
        component_standard_error=0.1,
    )
    controller["lbfgs_s"][:2] = np.eye(2, v14.N_INTERNAL) * 1.0e-4
    controller["lbfgs_y"][:2] = np.eye(2, v14.N_INTERNAL)
    controller["lbfgs_count"] = 2
    h0_step = np.full(v14.N_INTERNAL, -0.010, dtype=np.float64)
    lbfgs_step = np.full(v14.N_INTERNAL, -0.020, dtype=np.float64)

    def fake_build(label, direction, controller, statistics, radii, controls):
        del direction, statistics, radii, controls
        if label == "H0":
            return _engineered_candidate(label, controller, h0_step, 1.0)
        return _engineered_candidate(label, controller, lbfgs_step, lbfgs_score)

    monkeypatch.setattr(v14, "_build_candidate", fake_build)
    def fake_raw_evidence(candidate, statistics, controls):
        del statistics
        value = float(candidate.proposal_effective_proximal_value)
        standard_error = 1.0e-3
        return {
            "linear": value + 0.5,
            "quadratic": 0.5,
            "score": value,
            "standard_error": standard_error,
            "lower_bound": value - controls.residual_bound_kappa * standard_error,
            "upper_bound": value + controls.residual_bound_kappa * standard_error,
        }

    monkeypatch.setattr(
        v14, "_candidate_raw_improvement_evidence", fake_raw_evidence
    )
    monkeypatch.setattr(
        v14,
        "joint_lbfgs_direction",
        lambda gradient, history_s, history_y, diagonal_h0: -gradient,
    )
    prepared, terminal = v14._prepare_update(controller, controls)
    assert terminal == {}
    assert prepared is not None
    assert prepared.selected_label == expected_label
    assert not prepared.hold_selected
    assert prepared.lbfgs_raw_advantage_resolved_positive is expected_resolved
    assert prepared.strict_lbfgs_win is expected_resolved
    assert prepared.latch_after is expected_resolved
    np.testing.assert_allclose(
        prepared.lbfgs_raw_advantage_resolution_threshold,
        controls.secant_noise_z
        * prepared.raw_improvement_difference_standard_error,
    )
    if expected_label == "H0":
        # This reproduces the V14 414-HOLD mechanism: L-BFGS has a small point
        # advantage, but it is not resolved, while H0's own movement LCB is
        # comfortably positive.  V15 moves with H0 instead of resampling.
        assert prepared.lbfgs_raw_improvement_advantage > 0.0
        assert (
            prepared.lbfgs_raw_improvement_advantage
            < prepared.lbfgs_raw_advantage_resolution_threshold
        )
        assert prepared.selected_raw_improvement_lower_bound > 0.0


def test_unresolved_advantage_cannot_bypass_nonpositive_h0_lcb(monkeypatch):
    """A positive-LCB L-BFGS candidate cannot enter through an unresolved D."""

    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls, inner_step=0, state_index=1)
    statistics = _set_noisy_projected_residual(
        controller,
        controls,
        residual_norm=6.0,
        component_standard_error=0.1,
    )
    controller["lbfgs_s"][:2] = np.eye(2, v14.N_INTERNAL) * 1.0e-4
    controller["lbfgs_y"][:2] = np.eye(2, v14.N_INTERNAL)
    controller["lbfgs_count"] = 2
    h0_step = np.full(v14.N_INTERNAL, -0.100, dtype=np.float64)
    lbfgs_step = np.full(v14.N_INTERNAL, -0.010, dtype=np.float64)

    def fake_build(label, direction, controller, statistics, radii, controls):
        del direction, statistics, radii, controls
        if label == "H0":
            return _engineered_candidate(label, controller, h0_step, 0.010)
        return _engineered_candidate(label, controller, lbfgs_step, 0.020)

    monkeypatch.setattr(v14, "_build_candidate", fake_build)
    def raw_evidence(candidate, statistics, controls):
        del statistics
        if candidate.label == "H0":
            mean, se = 0.010, 0.010
        else:
            mean, se = 0.01001, 0.001
        return {
            "linear": mean + 0.5,
            "quadratic": 0.5,
            "score": mean,
            "standard_error": se,
            "lower_bound": mean - controls.residual_bound_kappa * se,
            "upper_bound": mean + controls.residual_bound_kappa * se,
        }

    monkeypatch.setattr(v14, "_candidate_raw_improvement_evidence", raw_evidence)
    monkeypatch.setattr(
        v14,
        "joint_lbfgs_direction",
        lambda gradient, history_s, history_y, diagonal_h0: -gradient,
    )
    prepared, terminal = v14._prepare_update(controller, controls)
    assert terminal == {}
    assert prepared is not None and prepared.hold_selected
    assert not prepared.lbfgs_raw_advantage_resolved_positive
    assert prepared.hold_reason == (
        "pre_takeover_h0_raw_improvement_lower_bound_nonpositive"
    )
    assert prepared.h0_raw_improvement["lower_bound"] <= 0.0
    assert prepared.lbfgs_raw_improvement["lower_bound"] > 0.0
    assert not prepared.warm_validation_selected
    assert not prepared.latch_after
    np.testing.assert_array_equal(prepared.selected_path, controller["current_path"])


def test_resolved_raw_advantage_does_not_take_over_when_warm_lcb_is_nonpositive(
    monkeypatch,
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls, inner_step=0, state_index=1)
    _set_noisy_projected_residual(
        controller,
        controls,
        residual_norm=6.0,
        component_standard_error=0.1,
    )
    controller["lbfgs_s"][:2] = np.eye(2, v14.N_INTERNAL) * 1.0e-4
    controller["lbfgs_y"][:2] = np.eye(2, v14.N_INTERNAL)
    controller["lbfgs_count"] = 2
    h0_step = np.full(v14.N_INTERNAL, -0.001, dtype=np.float64)
    lbfgs_step = np.full(v14.N_INTERNAL, -0.100, dtype=np.float64)

    def fake_build(label, direction, controller, statistics, radii, controls):
        del direction, statistics, radii, controls
        if label == "H0":
            return _engineered_candidate(label, controller, h0_step, 0.001)
        return _engineered_candidate(label, controller, lbfgs_step, 0.100)

    monkeypatch.setattr(v14, "_build_candidate", fake_build)
    def raw_evidence(candidate, statistics, controls):
        del statistics
        if candidate.label == "H0":
            mean, se = 0.010, 0.001
        else:
            mean, se = 0.210, 0.100
        return {
            "linear": mean + 0.5,
            "quadratic": 0.5,
            "score": mean,
            "standard_error": se,
            "lower_bound": mean - controls.residual_bound_kappa * se,
            "upper_bound": mean + controls.residual_bound_kappa * se,
        }

    monkeypatch.setattr(v14, "_candidate_raw_improvement_evidence", raw_evidence)
    monkeypatch.setattr(
        v14,
        "joint_lbfgs_direction",
        lambda gradient, history_s, history_y, diagonal_h0: -gradient,
    )
    prepared, terminal = v14._prepare_update(controller, controls)
    assert terminal == {}
    assert prepared is not None and not prepared.hold_selected
    assert prepared.selected_label == "H0"
    assert prepared.lbfgs_raw_advantage_resolved_positive
    assert (
        prepared.lbfgs_raw_improvement_advantage
        > prepared.lbfgs_raw_advantage_resolution_threshold
    )
    assert prepared.lbfgs_raw_improvement["lower_bound"] <= 0.0
    assert prepared.h0_raw_improvement["lower_bound"] > 0.0
    assert not prepared.strict_lbfgs_win
    assert not prepared.latch_after


@pytest.mark.parametrize(
    "h0_error, lbfgs_error",
    [
        (
            "post_reparameterization_path_exceeds_frozen_anchor_trust_radii",
            "engineered_nonrequired_warm_failure",
        ),
        (
            "engineered_required_h0_failure",
            "post_reparameterization_path_exceeds_frozen_anchor_trust_radii",
        ),
    ],
)
def test_pre_takeover_structurally_invalid_h0_is_terminal_not_a_statistical_hold(
    monkeypatch, h0_error, lbfgs_error
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls, inner_step=0, state_index=1)
    _set_noisy_projected_residual(
        controller,
        controls,
        residual_norm=6.0,
        component_standard_error=0.1,
    )
    controller["lbfgs_s"][:2] = np.eye(2, v14.N_INTERNAL) * 1.0e-4
    controller["lbfgs_y"][:2] = np.eye(2, v14.N_INTERNAL)
    controller["lbfgs_count"] = 2
    v14._validate_controller(controller, controls)

    def fake_build(label, direction, controller, statistics, radii, controls):
        del direction, controller, statistics, radii, controls
        return v14._invalid_candidate(
            label, h0_error if label == "H0" else lbfgs_error
        )

    monkeypatch.setattr(v14, "_build_candidate", fake_build)
    monkeypatch.setattr(
        v14,
        "joint_lbfgs_direction",
        lambda gradient, history_s, history_y, diagonal_h0: -gradient,
    )
    prepared, terminal = v14._prepare_update(controller, controls)
    assert prepared is None
    assert terminal["phase"] == "TERMINAL"
    assert terminal["reason"] == "no_valid_motion_candidate_terminal"
    assert terminal["h0_error"] == h0_error
    assert terminal["warm_error"] == lbfgs_error


def test_inner_zero_hold_cannot_trigger_r_stop_and_replays(monkeypatch, tmp_path):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls, inner_step=0, state_index=1)
    _set_noisy_projected_residual(
        controller,
        controls,
        residual_norm=0.5,
        component_standard_error=0.3,
    )
    monkeypatch.setattr(v14, "reparameterize_article_neb", _identity_reparameterization)
    prepared, terminal = v14._prepare_update(controller, controls)
    assert terminal == {}
    assert prepared is not None
    assert prepared.hold_selected
    assert prepared.hold_reason == (
        "pre_takeover_h0_raw_improvement_lower_bound_nonpositive"
    )
    np.testing.assert_array_equal(prepared.selected_path, controller["current_path"])

    target = _target_band_in_source_frame(
        controller,
        prepared.selected_path,
        controls,
        residual_norm=0.5,
        component_standard_error=0.01,
    )
    band = OracleBand(
        centers=prepared.selected_path[1:-1].copy(),
        mean_coordinates=prepared.selected_path[1:-1].copy(),
        grad_F=target["grad"],
        M=target["M"],
        grad_F_block_means=target["grad_blocks"],
        M_block_means=target["M_blocks"],
        seeds=np.arange(v14.N_INTERNAL, dtype=np.int64),
        thermostat_seeds=np.arange(v14.N_INTERNAL, dtype=np.int64) + 100,
        frame_counts=target["frame_counts"],
        discarded_frame_counts=target["discarded_frame_counts"],
        wall_seconds=np.zeros(v14.N_INTERNAL),
        nof=v14.N_INTERNAL,
        metadata={"engineered_inner_zero_hold": True},
    )
    after, advance = v14._advance_controller(
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
    pooled = advance["new_statistics_in_source_frozen_frame"]
    assert pooled.upper_norm <= (
        controls.r * float(controller["outer_anchor_lower_norm"])
    )
    assert not advance[
        "r_stop_decision_eligible_after_at_least_one_movement"
    ]
    assert not advance["r_stop"]
    assert not advance["stopped_inner"]
    assert advance["inner_stop_reason"] == (
        "movement_evidence_hold_resample_continue"
    )
    assert not advance["movement_committed"]
    assert after["outer_index"] == controller["outer_index"]
    assert after["inner_step"] == controller["inner_step"] == 0

    run_directory = tmp_path / "inner-zero-hold-r-gate-replay"
    v14._atomic_publish_npz_once(
        v14._state_path(run_directory, 1),
        full_path_unwrapped=controller["current_path"],
        selected_reparameterization_audit_sha256=np.asarray("source-certificate"),
        selected_reparameterization_iterations=np.asarray(1, dtype=np.int64),
        selected_reparameterization_output_residual=np.asarray(0.0),
    )
    metrics = v14.compute_article_metrics(
        after["current_path"], after["current_M"], after["current_grad"]
    )
    v14._save_public_state(
        run_directory,
        controller,
        after,
        prepared,
        {},
        band,
        advance,
        metrics,
    )
    row = v14._load_npz_record(v14._state_path(run_directory, 2))
    assert not bool(row["r_stop_decision_eligible_after_at_least_one_movement"])
    assert not bool(row["r_global_stop"])
    assert int(row["next_outer_index"]) == int(controller["outer_index"])
    assert int(row["next_inner_step"]) == 0
    replayed = v14._replay_one_state(controller, row, controls)
    assert v14._controllers_equal(after, replayed)
    v14._save_checkpoint(run_directory, after)
    restored = v14._controller_from_checkpoint(
        v14._load_npz_record(run_directory / "controller_checkpoint.npz"),
        controls,
    )
    assert v14._controllers_equal(after, restored)


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
    assert terminal["reason"] == "no_structurally_valid_motion_candidate_terminal"


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
        observations, frame, controls, controls.alpha, -1, 0
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


@pytest.mark.parametrize(
    "invalid_error",
    [
        "post_reparameterization_path_exceeds_frozen_anchor_trust_radii",
        "ArticleReparameterizationError:engineered-post-takeover-failure",
    ],
)
def test_latched_structurally_invalid_warm_is_terminal_not_a_validation_hold(
    monkeypatch, invalid_error
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)

    noisy = _analytic_controller(controls, inner_step=0, state_index=1)
    _set_noisy_projected_residual(
        noisy,
        controls,
        residual_norm=6.0,
        component_standard_error=0.1,
    )
    noisy["lbfgs_latched"] = True
    noisy["lbfgs_count"] = 0
    noisy["warm_safeguard_inverse_scale"] = np.asarray(controls.alpha)
    noisy["warm_safeguard_source_state_index"] = 1
    v14._validate_controller(noisy, controls)

    monkeypatch.setattr(
        v14,
        "_build_candidate",
        lambda label, direction, controller, statistics, radii, controls: (
            v14._invalid_candidate(label, invalid_error)
        ),
    )
    prepared, terminal = v14._prepare_update(noisy, controls)
    assert prepared is None
    assert terminal["phase"] == "TERMINAL"
    assert terminal["reason"] == "no_freezeable_warm_motion_candidate_terminal"
    assert terminal["globally_latched"]
    assert terminal["warm_error"] == invalid_error

    exact = _analytic_controller(controls, inner_step=0, state_index=1)
    exact["lbfgs_latched"] = True
    exact["lbfgs_count"] = 0
    exact["warm_safeguard_inverse_scale"] = np.asarray(controls.alpha)
    exact["warm_safeguard_source_state_index"] = 1
    v14._validate_controller(exact, controls)
    exact_prepared, exact_terminal = v14._prepare_update(exact, controls)
    assert exact_prepared is None
    assert exact_terminal["phase"] == "TERMINAL"
    assert exact_terminal["reason"] == "no_freezeable_warm_motion_candidate_terminal"
    assert exact_terminal["globally_latched"]


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
        escaped = controller["frozen_path"].copy()
        escaped[1:-1] += controller["frozen_basis"] * (2.0 * radii[:, None])
        base = _identity_reparameterization(
            escaped, audit_source_path=path, **unused
        )
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
        "post_reparameterization_trust_globalization_exhausted"
    )
    assert candidate.trust_globalization_exhausted
    assert candidate.post_r_attempt_count == v14.POST_R_BACKTRACK_LIMIT + 1
    np.testing.assert_array_equal(
        candidate.post_r_attempt_scales,
        candidate.common_cap_scale
        * np.exp2(-np.arange(v14.POST_R_BACKTRACK_LIMIT + 1)),
    )


def test_anchor_aware_common_cap_accounts_for_accumulated_displacement():
    """The cap is on anchor-to-proposal displacement, not just new motion."""

    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls)
    anchor = controller["frozen_path"].copy()
    basis = np.repeat(np.asarray([[1.0, 0.0]]), v14.N_INTERNAL, axis=0)
    current = anchor.copy()
    current[1:-1, 0] += 0.8
    direction = np.ones(v14.N_INTERNAL, dtype=np.float64)
    radii = np.ones(v14.N_INTERNAL, dtype=np.float64)

    cap = v14._whole_band_common_cap(
        direction,
        basis,
        radii,
        current_path=current,
        frozen_path=anchor,
        secant_norm_floor=controls.secant_norm_floor,
    )

    assert cap.valid
    assert cap.scale < 0.21
    assert cap.scale > 0.19
    assert cap.truncated
    np.testing.assert_allclose(cap.source_distances, 0.8, atol=2.0e-15)
    assert np.all(cap.raw_path_distances <= radii)
    # The V15 direction-only cap would have accepted scale one here.
    assert np.max(np.linalg.norm(basis * direction[:, None], axis=1) / radii) <= 1.0


def test_anchor_cap_roundoff_fallback_is_exactly_64_binary64_steps(monkeypatch):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls)
    current = controller["current_path"].copy()
    anchor = current.copy()
    basis = np.repeat(np.asarray([[1.0, 0.0]]), v14.N_INTERNAL, axis=0)
    direction = np.full(v14.N_INTERNAL, 0.1, dtype=np.float64)
    radii = np.ones(v14.N_INTERNAL, dtype=np.float64)
    original_ray = v14._raw_path_on_direction_ray
    observed_scales = []

    def engineered_binary64_check(current_path, normal, reduced, scale):
        observed_scales.append(float(scale))
        raw = original_ray(current_path, normal, reduced, scale)
        # Emulate a strict binary64 reconstruction that is infeasible at and
        # above a known boundary even though the analytic cap is one.
        if float(scale) >= 0.75:
            raw[1:-1, 0] = anchor[1:-1, 0] + 2.0
        return raw

    monkeypatch.setattr(
        v14, "_raw_path_on_direction_ray", engineered_binary64_check
    )
    cap = v14._whole_band_common_cap(
        direction,
        basis,
        radii,
        current_path=current,
        frozen_path=anchor,
        secant_norm_floor=controls.secant_norm_floor,
    )

    assert cap.valid
    assert cap.theoretical_scale == 1.0
    assert cap.binary_bracket_used
    assert cap.binary_bracket_iterations == v14.ANCHOR_CAP_BRACKET_ITERATIONS == 64
    assert cap.scale < 0.75
    assert np.nextafter(cap.scale, np.inf) >= 0.75
    assert np.all(cap.raw_path_distances <= radii)
    # theoretical check + verified zero + 64 midpoints + final verification
    assert len(observed_scales) == 67


def test_anchor_cap_at_boundary_outward_is_real_typed_trust_exhaustion():
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls)
    anchor = controller["frozen_path"].copy()
    current = anchor.copy()
    current[1, 0] += 1.0 - 1.0e-15
    basis = np.repeat(np.asarray([[1.0, 0.0]]), v14.N_INTERNAL, axis=0)
    direction = np.zeros(v14.N_INTERNAL, dtype=np.float64)
    direction[0] = 1.0
    radii = np.ones(v14.N_INTERNAL, dtype=np.float64)

    cap = v14._whole_band_common_cap(
        direction,
        basis,
        radii,
        current_path=current,
        frozen_path=anchor,
        secant_norm_floor=controls.secant_norm_floor,
    )
    assert not cap.valid
    assert cap.error == "anchor_aware_common_cap_exhausted_below_secant_norm_floor"
    assert 0.0 < cap.scale <= controls.secant_norm_floor

    candidate_controller = dict(controller)
    candidate_controller["current_path"] = current
    candidate_controller["frozen_basis"] = basis
    statistics = SimpleNamespace(effective_residual=-direction)
    candidate = v14._build_candidate(
        label="boundary-outward",
        direction=direction,
        controller=candidate_controller,
        statistics=statistics,
        radii=radii,
        controls=controls,
    )
    assert not candidate.valid
    assert candidate.trust_globalization_exhausted
    assert candidate.error == cap.error


def test_post_r_fixed_half_backtracking_rebuilds_and_accepts_third_scale(
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
        block_frame_counts=controller["current_block_frame_counts"],
    )
    _, _, radii = v14._trust_radii(controller["frozen_path"], controls)
    calls = []

    def first_two_escape(path, **unused):
        calls.append(np.asarray(path, dtype=np.float64).copy())
        if len(calls) <= 2:
            escaped = controller["frozen_path"].copy()
            escaped[1:-1] += controller["frozen_basis"] * (2.0 * radii)[:, None]
            return _identity_reparameterization(
                escaped, audit_source_path=path, **unused
            )
        return _identity_reparameterization(path, **unused)

    monkeypatch.setattr(v14, "reparameterize_article_neb", first_two_escape)
    candidate = v14._build_candidate(
        label="half-backtrack",
        direction=-controls.alpha * statistics.effective_residual,
        controller=controller,
        statistics=statistics,
        radii=radii,
        controls=controls,
    )

    assert candidate.valid
    assert candidate.post_r_backtrack_count == 2
    assert candidate.post_r_backtrack_factor == 0.25
    assert candidate.post_r_attempt_count == 3
    np.testing.assert_array_equal(
        candidate.post_r_attempt_scales,
        candidate.common_cap_scale * np.asarray([1.0, 0.5, 0.25]),
    )
    assert candidate.post_r_attempt_status.tolist() == [
        "post_r_path_outside_anchor_trust",
        "post_r_path_outside_anchor_trust",
        "accepted_inside_anchor_trust",
    ]
    assert len(set(candidate.post_r_attempt_raw_path_sha256.tolist())) == 3
    assert np.all(
        v14._anchor_trust_distances(
            candidate.reparameterization.path, controller["frozen_path"]
        )
        <= radii
    )


def test_post_r_endpoint_change_is_strictly_rejected_at_every_attempt(monkeypatch):
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

    def endpoint_changing_result(path, **unused):
        changed = np.asarray(path, dtype=np.float64).copy()
        changed[0, 0] += 1.0e-6
        return _identity_reparameterization(
            changed, audit_source_path=path, **unused
        )

    monkeypatch.setattr(v14, "reparameterize_article_neb", endpoint_changing_result)
    candidate = v14._build_candidate(
        label="bad-endpoint",
        direction=-controls.alpha * statistics.effective_residual,
        controller=controller,
        statistics=statistics,
        radii=radii,
        controls=controls,
    )

    assert not candidate.valid
    assert candidate.trust_globalization_exhausted
    assert candidate.post_r_attempt_count == v14.POST_R_BACKTRACK_LIMIT + 1
    assert all(
        "moved a fixed endpoint" in status
        for status in candidate.post_r_attempt_status.tolist()
    )


def _trust_exhausted_candidate(label):
    return v14._invalid_candidate(
        label,
        "post_reparameterization_trust_globalization_exhausted",
        trust_globalization_exhausted=True,
    )


def test_nonrequired_pre_takeover_warm_exhaustion_cannot_refresh(monkeypatch):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls, state_index=1)
    controller["lbfgs_s"][:2] = np.eye(2, v14.N_INTERNAL) * 1.0e-4
    controller["lbfgs_y"][:2] = np.eye(2, v14.N_INTERNAL)
    controller["lbfgs_count"] = 2
    h0_step = np.full(v14.N_INTERNAL, -1.0e-5)

    def fake_build(label, direction, controller, statistics, radii, controls):
        del direction, statistics, radii, controls
        if label == "H0":
            return _engineered_candidate(label, controller, h0_step, 1.0)
        return _trust_exhausted_candidate(label)

    monkeypatch.setattr(v14, "_build_candidate", fake_build)
    monkeypatch.setattr(
        v14,
        "joint_lbfgs_direction",
        lambda gradient, history_s, history_y, diagonal_h0: -gradient,
    )
    outcome = v14._prepare_update(controller, controls)

    assert outcome.prepared is not None
    assert outcome.prepared.selected_label == "H0"
    assert outcome.proposal_controller is controller
    assert outcome.zero_nof_event == {}
    assert outcome.terminal == {}


@pytest.mark.parametrize("latched", [False, True])
def test_required_structural_anchor_cap_failure_terminals_without_hold_or_refresh(
    monkeypatch, latched
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls, state_index=1)
    _set_noisy_projected_residual(
        controller,
        controls,
        residual_norm=6.0,
        component_standard_error=0.1,
    )
    controller["lbfgs_latched"] = latched
    if latched:
        controller["lbfgs_count"] = 0
        controller["warm_safeguard_inverse_scale"] = np.asarray(controls.alpha)
        controller["warm_safeguard_source_state_index"] = 1
    v14._validate_controller(controller, controls)
    required_label = "WARM_SCALAR_SAFEGUARD" if latched else "H0"

    def structural_failure(label, direction, controller, statistics, radii, controls):
        del direction, controller, statistics, radii, controls
        if label == required_label:
            return v14._invalid_candidate(
                label, "anchor_aware_common_cap_nonfinite_quadratic"
            )
        return _trust_exhausted_candidate(label)

    monkeypatch.setattr(v14, "_build_candidate", structural_failure)
    outcome = v14._prepare_update(controller, controls)

    assert outcome.prepared is None
    assert outcome.zero_nof_event == {}
    assert outcome.proposal_controller is controller
    assert outcome.terminal["phase"] == "TERMINAL"
    assert outcome.terminal["reason"] == (
        "required_candidate_anchor_cap_structural_failure"
    )
    assert outcome.terminal["required_candidate_label"] == required_label


@pytest.mark.parametrize("latched,required_label", [(False, "H0"), (True, "WARM_SCALAR_SAFEGUARD")])
def test_required_trust_exhaustion_refreshes_once_and_preserves_paid_state(
    monkeypatch, latched, required_label
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _stale_outer_controller(
        controls, inner_step=2, state_index=2
    )
    controller["lbfgs_latched"] = latched
    if latched:
        controller["lbfgs_count"] = 0
        controller["warm_safeguard_inverse_scale"] = np.asarray(controls.alpha)
        controller["warm_safeguard_source_state_index"] = 2
    protected = {
        key: np.asarray(controller[key]).copy()
        for key in (
            "current_path",
            "current_M",
            "current_grad",
            "current_grad_blocks",
            "current_M_blocks",
            "current_block_frame_counts",
            "observation_paths",
            "observation_gradients",
            "observation_gradient_covariances",
            "observation_covariance_available",
            "observation_state_indices",
        )
    }
    original_state = controller["state_index"]
    original_nof = controller["nof"]
    original_outer = controller["outer_index"]
    build_calls = []

    def fake_build(label, direction, controller, statistics, radii, controls):
        del direction, statistics, radii, controls
        build_calls.append((label, int(controller["outer_index"])))
        if int(controller["outer_index"]) == original_outer:
            return _trust_exhausted_candidate(label)
        step = np.full(v14.N_INTERNAL, -1.0e-5)
        return _engineered_candidate(label, controller, step, 1.0)

    monkeypatch.setattr(v14, "_build_candidate", fake_build)
    outcome = v14._prepare_update(controller, controls)
    refreshed = outcome.proposal_controller

    assert outcome.prepared is not None
    assert outcome.prepared.selected_label == required_label
    assert outcome.zero_nof_event["mean_force_evaluations"] == 0
    assert outcome.zero_nof_event["required_candidate_label"] == required_label
    assert refreshed["state_index"] == original_state
    assert refreshed["nof"] == original_nof
    assert refreshed["outer_index"] == original_outer + 1
    assert refreshed["inner_step"] == 0
    assert refreshed["lbfgs_latched"] is latched
    assert refreshed["trust_refresh_used_path_sha256"] == v14._array_sha256(
        controller["current_path"]
    )
    assert outcome.zero_nof_event["proposal_controller_checkpoint_sha256"] == (
        v14._checkpoint_digest(refreshed)
    )
    for key, value in protected.items():
        np.testing.assert_array_equal(refreshed[key], value, err_msg=key)
    assert int(refreshed["observation_count"]) == int(controller["observation_count"])
    assert [row[0] for row in build_calls] == [required_label, required_label]

    oracle = SyntheticTenBlockOracle(0.0)
    band = oracle.evaluate_band(outcome.prepared.selected_path, band_id=3)
    after, _ = v14._advance_controller(
        refreshed,
        outcome.prepared,
        M=band.M,
        grad=band.grad_F,
        grad_blocks=band.grad_F_block_means,
        M_blocks=band.M_block_means,
        frame_counts=band.frame_counts,
        discarded_frame_counts=band.discarded_frame_counts,
        controls=controls,
    )
    assert oracle.call_count == v14.N_INTERNAL
    assert after["nof"] == original_nof + v14.N_INTERNAL
    assert after["trust_refresh_used_path_sha256"] == "none"


def test_second_required_exhaustion_at_same_path_is_typed_terminal(monkeypatch):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _stale_outer_controller(controls, state_index=1)
    monkeypatch.setattr(
        v14,
        "_build_candidate",
        lambda label, direction, controller, statistics, radii, controls: (
            _trust_exhausted_candidate(label)
        ),
    )

    first = v14._prepare_update(controller, controls)
    assert first.prepared is None
    assert first.zero_nof_event["mean_force_evaluations"] == 0
    assert first.terminal["reason"].endswith("after_one_outer_refresh")
    assert first.proposal_controller["outer_index"] == controller["outer_index"] + 1

    second = v14._prepare_update(first.proposal_controller, controls)
    assert second.prepared is None
    assert second.zero_nof_event == {}
    assert second.proposal_controller is first.proposal_controller
    assert second.terminal["same_path_refresh_previously_used"] is True
    assert second.proposal_controller["outer_index"] == first.proposal_controller[
        "outer_index"
    ]


def test_required_exhaustion_in_an_already_fresh_outer_terminals_without_refresh(
    monkeypatch,
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls, state_index=1)
    monkeypatch.setattr(
        v14,
        "_build_candidate",
        lambda label, direction, controller, statistics, radii, controls: (
            _trust_exhausted_candidate(label)
        ),
    )

    outcome = v14._prepare_update(controller, controls)

    assert outcome.prepared is None
    assert outcome.proposal_controller is controller
    assert outcome.zero_nof_event == {}
    assert outcome.terminal["outer_was_already_fresh"] is True
    assert outcome.terminal["reason"].endswith("already_fresh_outer")
    assert outcome.proposal_controller["state_index"] == 1
    assert outcome.proposal_controller["nof"] == v14.N_INTERNAL
    assert outcome.proposal_controller["outer_index"] == controller["outer_index"]


def test_same_aggregate_frame_with_stale_observation_history_may_refresh(monkeypatch):
    """Freshness includes reconstructed history, not only path and mean M/g."""

    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _analytic_controller(controls, state_index=1)
    current_path = controller["current_path"].copy()
    prior_path = current_path.copy()
    prior_path[1:-1] -= controller["frozen_basis"] * 1.0e-5
    controller["observation_paths"][0] = prior_path
    controller["observation_paths"][1] = current_path
    controller["observation_gradients"][0] = controller["current_grad"]
    controller["observation_gradients"][1] = controller["current_grad"]
    controller["observation_gradient_covariances"][:2] = 0.0
    controller["observation_covariance_available"][:2] = True
    controller["observation_state_indices"][:2] = np.asarray([0, 1])
    controller["observation_count"] = 2
    v14._validate_controller(controller, controls)
    assert np.array_equal(controller["current_path"], controller["frozen_path"])
    assert np.array_equal(controller["current_M"], controller["frozen_M"])
    assert np.array_equal(controller["current_grad"], controller["frozen_grad"])
    assert not v14._frozen_outer_is_current_paid_frame(controller)
    monkeypatch.setattr(
        v14,
        "_build_candidate",
        lambda label, direction, controller, statistics, radii, controls: (
            _trust_exhausted_candidate(label)
        ),
    )

    outcome = v14._prepare_update(controller, controls)

    assert outcome.prepared is None
    assert outcome.zero_nof_event["mean_force_evaluations"] == 0
    assert outcome.proposal_controller["lbfgs_count"] == 1
    assert outcome.terminal["reason"].endswith("after_one_outer_refresh")


def test_refresh_followed_by_hold_keeps_marker_and_forbids_second_refresh(
    monkeypatch,
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller = _stale_outer_controller(controls, state_index=1)
    _set_noisy_projected_residual(
        controller,
        controls,
        residual_norm=6.0,
        component_standard_error=0.1,
    )
    source_outer = int(controller["outer_index"])

    def exhaust_then_weak_move(label, direction, controller, statistics, radii, controls):
        del direction, statistics, radii, controls
        if int(controller["outer_index"]) == source_outer:
            return _trust_exhausted_candidate(label)
        return _engineered_candidate(
            label,
            controller,
            np.full(v14.N_INTERNAL, -0.1),
            0.01,
        )

    monkeypatch.setattr(v14, "_build_candidate", exhaust_then_weak_move)
    outcome = v14._prepare_update(controller, controls)
    assert outcome.prepared is not None and outcome.prepared.hold_selected
    assert outcome.zero_nof_event["mean_force_evaluations"] == 0
    marker = outcome.proposal_controller["trust_refresh_used_path_sha256"]
    assert marker == v14._array_sha256(controller["current_path"])

    oracle = SyntheticTenBlockOracle(0.1)
    band = oracle.evaluate_band(outcome.prepared.selected_path, band_id=2)
    after, advance = v14._advance_controller(
        outcome.proposal_controller,
        outcome.prepared,
        M=band.M,
        grad=band.grad_F,
        grad_blocks=band.grad_F_block_means,
        M_blocks=band.M_block_means,
        frame_counts=band.frame_counts,
        discarded_frame_counts=band.discarded_frame_counts,
        controls=controls,
    )
    assert advance["hold_resample"]
    assert after["trust_refresh_used_path_sha256"] == marker
    assert oracle.band_calls == 1

    monkeypatch.setattr(
        v14,
        "_build_candidate",
        lambda label, direction, controller, statistics, radii, controls: (
            _trust_exhausted_candidate(label)
        ),
    )
    second = v14._prepare_update(after, controls)
    assert second.prepared is None
    assert second.zero_nof_event == {}
    assert second.terminal["same_path_refresh_previously_used"] is True
    assert second.proposal_controller["outer_index"] == after["outer_index"]


def test_refresh_event_is_paid_once_replayed_and_corruption_fails_closed(
    monkeypatch, tmp_path, causal_initial
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)

    def refresh_then_move(label, direction, controller, statistics, radii, controls):
        del direction, statistics, radii, controls
        if int(controller["outer_index"]) == 0:
            return _trust_exhausted_candidate(label)
        return _engineered_candidate(
            label,
            controller,
            np.full(v14.N_INTERNAL, -1.0e-5),
            1.0,
        )

    monkeypatch.setattr(v14, "_build_candidate", refresh_then_move)
    monkeypatch.setattr(
        v14, "_frozen_outer_is_current_paid_frame", lambda controller: False
    )
    run_directory = tmp_path / "refresh-paid-replay"
    oracle = SyntheticTenBlockOracle(0.0)
    result, _ = _run(
        run_directory,
        causal_initial,
        budget=20,
        stop=20,
        controls=controls,
        oracle=oracle,
    )
    assert result["nof_mean_force_incremental"] == 20
    assert oracle.band_calls == 1
    row = _load_state(run_directory, 1)
    assert bool(row["zero_nof_trust_refresh_applied"])
    event = json.loads(str(row["zero_nof_trust_refresh_event_json"].item()))
    assert event["mean_force_evaluations"] == 0
    assert str(row["proposal_controller_checkpoint_sha256"].item()) == event[
        "proposal_controller_checkpoint_sha256"
    ]
    assert str(row["trust_refresh_used_path_sha256_next"].item()) == "none"
    v14.load_public_trajectory(run_directory)

    # Rewind only the replaceable checkpoint. The immutable paid state must
    # replay through the same zero-NoF refresh without another oracle call.
    path, metric, gradient = causal_initial
    initial_controller = v14._new_controller(
        path,
        metric,
        gradient,
        gradient[:, None, :],
        controls,
        metric[:, None, :, :],
        False,
    )
    v14._save_checkpoint(run_directory, initial_controller)
    resumed, _ = _run(
        run_directory,
        causal_initial,
        budget=20,
        stop=20,
        controls=controls,
        oracle=oracle,
    )
    assert resumed["nof_mean_force_incremental"] == 20
    assert oracle.band_calls == 1

    state_path = v14._state_path(run_directory, 1)
    original = state_path.read_bytes()
    mutations = {
        "zero_nof_trust_refresh_event_json": lambda record: record.__setitem__(
            "zero_nof_trust_refresh_event_json", np.asarray("{}")
        ),
        "proposal_controller_checkpoint_sha256": lambda record: record.__setitem__(
            "proposal_controller_checkpoint_sha256", np.asarray("0" * 64)
        ),
        "trust_refresh_used_path_sha256_before": lambda record: record.__setitem__(
            "trust_refresh_used_path_sha256_before", np.asarray("none")
        ),
        "trust_refresh_used_path_sha256_next": lambda record: record.__setitem__(
            "trust_refresh_used_path_sha256_next", np.asarray("1" * 64)
        ),
    }
    for key, mutate in mutations.items():
        corrupted = v14._load_npz_record(state_path)
        mutate(corrupted)
        with state_path.open("wb") as stream:
            np.savez_compressed(stream, **corrupted)
        try:
            with pytest.raises(v14.IRIVMError, match=key):
                v14.load_public_trajectory(run_directory)
        finally:
            state_path.write_bytes(original)


def test_refresh_terminal_rederives_on_resume_without_an_oracle_call(
    monkeypatch, tmp_path, causal_initial
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    monkeypatch.setattr(
        v14,
        "_build_candidate",
        lambda label, direction, controller, statistics, radii, controls: (
            _trust_exhausted_candidate(label)
        ),
    )
    monkeypatch.setattr(
        v14, "_frozen_outer_is_current_paid_frame", lambda controller: False
    )
    oracle = SyntheticTenBlockOracle(0.0)
    run_directory = tmp_path / "refresh-terminal"
    first, _ = _run(
        run_directory,
        causal_initial,
        budget=20,
        controls=controls,
        oracle=oracle,
    )
    second, _ = _run(
        run_directory,
        causal_initial,
        budget=20,
        controls=controls,
        oracle=oracle,
    )

    assert first["phase"] == second["phase"] == "TERMINAL"
    assert first["nof_mean_force_incremental"] == 0
    assert first["zero_nof_trust_refresh_event"]["mean_force_evaluations"] == 0
    durable = v14._controller_from_checkpoint(
        v14._load_npz_record(run_directory / "controller_checkpoint.npz"),
        controls,
    )
    assert durable["outer_index"] == 0
    assert durable["trust_refresh_used_path_sha256"] == "none"
    assert first["outer_index_next"] == 1
    assert first["terminal_proposal_controller_checkpoint_sha256"] == first[
        "zero_nof_trust_refresh_event"
    ]["proposal_controller_checkpoint_sha256"]
    assert first["same_index_zero_nof_refresh_checkpoint_published"] is False
    assert second["terminal_rederived_on_resume"] is True
    assert oracle.band_calls == 0


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
        try:
            with pytest.raises(v14.IRIVMError, match=key):
                v14.load_public_trajectory(run_directory)
        finally:
            state_path.write_bytes(original_bytes)

    initial_state_path = v14._state_path(run_directory, 0)
    original_initial_bytes = initial_state_path.read_bytes()
    initial_mutations = {
        "controller_phase": lambda record: record.__setitem__(
            "controller_phase", np.asarray("CORRUPTED")
        ),
        "state_kind": lambda record: record.__setitem__(
            "state_kind", np.asarray("CORRUPTED")
        ),
        "joint_lbfgs_pair_count": lambda record: record.__setitem__(
            "joint_lbfgs_pair_count", np.asarray(1, dtype=np.int64)
        ),
        "one_way_takeover_latched": lambda record: record.__setitem__(
            "one_way_takeover_latched", np.asarray(True, dtype=bool)
        ),
        "warm_safeguard_inverse_scale": lambda record: perturb_array(
            record, "warm_safeguard_inverse_scale"
        ),
        "warm_safeguard_source_state_index": lambda record: record.__setitem__(
            "warm_safeguard_source_state_index", np.asarray(0, dtype=np.int64)
        ),
        "initial_no_se_special_case_available": lambda record: record.__setitem__(
            "initial_no_se_special_case_available", np.asarray(False, dtype=bool)
        ),
        "initial_no_se_special_case_used": lambda record: record.__setitem__(
            "initial_no_se_special_case_used", np.asarray(True, dtype=bool)
        ),
        "frozen_reference_normal_basis": lambda record: perturb_array(
            record, "frozen_reference_normal_basis"
        ),
        "raw_proximal_residual": lambda record: perturb_array(
            record, "raw_proximal_residual"
        ),
        "h0_inverse_diagonal_scale": lambda record: perturb_array(
            record, "h0_inverse_diagonal_scale"
        ),
        "reparameterization_applied_before_evaluation": lambda record: (
            record.__setitem__(
                "reparameterization_applied_before_evaluation",
                np.asarray(True, dtype=bool),
            )
        ),
        "normal_mapping_recomputed_after_reparameterization": lambda record: (
            record.__setitem__(
                "normal_mapping_recomputed_after_reparameterization",
                np.asarray(True, dtype=bool),
            )
        ),
        "selected_reparameterization_audit_sha256": lambda record: (
            record.__setitem__(
                "selected_reparameterization_audit_sha256",
                np.asarray("CORRUPTED"),
            )
        ),
    }
    for key, mutate in initial_mutations.items():
        record = v14._load_npz_record(initial_state_path)
        mutate(record)
        with initial_state_path.open("wb") as stream:
            np.savez_compressed(stream, **record)
        try:
            with pytest.raises(v14.IRIVMError, match=key):
                v14.load_public_trajectory(run_directory)
        finally:
            initial_state_path.write_bytes(original_initial_bytes)

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


def _latched_warm_validation_preparation(monkeypatch, controls, *, state_index=1):
    controller = _analytic_controller(
        controls, inner_step=0, state_index=state_index
    )
    _set_noisy_projected_residual(
        controller,
        controls,
        residual_norm=0.5,
        component_standard_error=0.3,
    )
    controller["lbfgs_latched"] = True
    controller["lbfgs_count"] = 0
    controller["warm_safeguard_inverse_scale"] = np.asarray(controls.alpha)
    controller["warm_safeguard_source_state_index"] = state_index
    v14._validate_controller(controller, controls)
    monkeypatch.setattr(
        v14, "reparameterize_article_neb", _identity_reparameterization
    )
    outcome = v14._prepare_update(controller, controls)
    assert outcome.terminal == {}
    prepared = outcome.prepared
    assert prepared is not None
    assert prepared.selected_label == "WARM_VALIDATION_HOLD"
    assert prepared.hold_selected and prepared.warm_validation_selected
    assert prepared.lbfgs_raw_improvement["lower_bound"] <= 0.0
    return controller, prepared


def _oracle_band_from_target(target, selected_path, *, band_id):
    return OracleBand(
        centers=np.asarray(selected_path[1:-1], dtype=np.float64).copy(),
        mean_coordinates=np.asarray(selected_path[1:-1], dtype=np.float64).copy(),
        grad_F=np.asarray(target["grad"], dtype=np.float64),
        M=np.asarray(target["M"], dtype=np.float64),
        grad_F_block_means=np.asarray(target["grad_blocks"], dtype=np.float64),
        M_block_means=np.asarray(target["M_blocks"], dtype=np.float64),
        seeds=np.arange(v14.N_INTERNAL, dtype=np.int64) + 1000 * band_id,
        thermostat_seeds=np.arange(v14.N_INTERNAL, dtype=np.int64)
        + 2000 * band_id,
        frame_counts=np.asarray(target["frame_counts"], dtype=np.int64),
        discarded_frame_counts=np.asarray(
            target["discarded_frame_counts"], dtype=np.int64
        ),
        wall_seconds=np.zeros(v14.N_INTERNAL),
        nof=v14.N_INTERNAL,
        metadata={"synthetic_warm_validation": True, "band_id": int(band_id)},
    )


def _fresh_band_along_frozen_delta(
    controller,
    prepared,
    controls,
    *,
    raw_delta_multiplier,
    component_standard_error,
    band_id,
):
    """Build real fresh blocks with P_raw=c*delta/alpha in one frozen frame."""

    frozen = prepared.frozen_warm_candidate
    delta = np.asarray(frozen["actual_normal_step"], dtype=np.float64)
    raw_center = float(raw_delta_multiplier) * delta / controls.alpha
    block_count = 10
    offsets = np.linspace(-1.0, 1.0, block_count)
    offset_scale = (
        float(component_standard_error)
        * np.sqrt(block_count)
        / np.std(offsets, ddof=1)
    )
    raw_blocks = raw_center[:, None] + offset_scale * offsets[None, :]
    source_u = v14._normal_coordinate(
        controller["current_path"],
        controller["frozen_path"],
        controller["frozen_basis"],
        controller["frozen_G"],
    )
    projected_blocks = raw_blocks - source_u[:, None] / controls.alpha
    basis_norm_squared = np.einsum(
        "ni,ni->n", controller["frozen_basis"], controller["frozen_basis"]
    )
    gradient_blocks = (
        controller["frozen_basis"][:, None, :]
        * (projected_blocks / basis_norm_squared[:, None])[:, :, None]
    )
    block_counts = np.full((v14.N_INTERNAL, block_count), 10, dtype=np.int64)
    metric = np.asarray(controller["current_M"], dtype=np.float64)
    target = {
        "M": metric,
        "grad": v14._weighted_block_mean(gradient_blocks, block_counts),
        "grad_blocks": gradient_blocks,
        "M_blocks": np.repeat(metric[:, None, :, :], block_count, axis=1),
        "frame_counts": np.full(v14.N_INTERNAL, 110, dtype=np.int64),
        "discarded_frame_counts": np.full(v14.N_INTERNAL, 10, dtype=np.int64),
    }
    return _oracle_band_from_target(
        target, prepared.selected_path, band_id=band_id
    )


@pytest.mark.parametrize(
    "mean,se,expected",
    [
        (2.262, 1.0, "unresolved"),
        (-2.262, 1.0, "confirmed_bad"),
        (2.262 + 1.0e-12, 1.0, "validated_positive"),
        (-2.262 + 1.0e-12, 1.0, "unresolved"),
    ],
)
def test_warm_validation_interval_boundaries_are_registered_three_way_rule(
    mean, se, expected
):
    classification, lower, upper = v14._classify_raw_improvement_interval(
        mean, se, 2.262
    )
    assert classification == expected
    assert lower == mean - 2.262 * se
    assert upper == mean + 2.262 * se


def test_warm_candidate_freezes_path_delta_frame_and_hashes(monkeypatch):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller, first = _latched_warm_validation_preparation(
        monkeypatch, controls
    )
    frozen = first.frozen_warm_candidate
    second = v14._prepare_update(controller, controls).prepared
    assert second is not None and second.warm_validation_selected
    repeated = second.frozen_warm_candidate
    for key in (
        "candidate_sha256",
        "frame_sha256",
        "request_sha256",
        "source_checkpoint_sha256",
    ):
        assert repeated[key] == frozen[key]
    for key in (
        "path",
        "raw_path",
        "source_path",
        "frozen_path",
        "frozen_M",
        "frozen_grad",
        "frozen_tangent",
        "frozen_G",
        "frozen_basis",
        "actual_normal_coordinate",
        "actual_normal_step",
    ):
        np.testing.assert_array_equal(repeated[key], frozen[key])
    expected = v14._candidate_raw_improvement_evidence(
        first.lbfgs_candidate, first.source_statistics, controls
    )
    assert frozen["source_score"] == expected["score"]
    assert frozen["source_score_lower_bound"] == expected["lower_bound"]


@pytest.mark.parametrize(
    "residual_norm,component_se,expected_classification",
    [
        (6.0, 0.01, "validated_positive"),
        (0.0, 0.01, "confirmed_bad"),
        (0.1, 0.3, "unresolved"),
    ],
)
def test_fresh_raw_validation_three_branches_cost_exactly_one_band(
    monkeypatch, residual_norm, component_se, expected_classification
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller, prepared = _latched_warm_validation_preparation(
        monkeypatch, controls
    )
    target = _target_band_in_source_frame(
        controller,
        prepared.selected_path,
        controls,
        residual_norm=residual_norm,
        component_standard_error=component_se,
    )
    after, audit = v14._advance_controller(
        controller, prepared, controls=controls, **target
    )
    validation = audit["warm_validation"]
    assert validation["classification"] == expected_classification
    assert after["nof"] == controller["nof"] + v14.N_INTERNAL
    assert audit["completed_inner_step"] == controller["inner_step"]
    assert not audit["movement_committed"]
    np.testing.assert_array_equal(after["current_path"], controller["current_path"])
    assert validation["fresh_raw_block_count"] == 10
    assert not validation["used_pooled_source_estimate"]
    assert not validation["used_shrinkage_centered_residual"]
    assert not np.array_equal(
        validation["fresh_raw_residual"],
        audit["new_statistics_in_source_frozen_frame"].raw_residual,
    )
    if expected_classification == "validated_positive":
        assert after["pending_warm_status"] == v14.WARM_VALIDATION_PENDING_COMMIT
        assert audit["warm_validation_disposition"] == (
            "frozen_candidate_pending_commit"
        )
        assert after["warm_restart_used_path_sha256"] == "none"
    else:
        assert after["pending_warm_status"] == v14.WARM_VALIDATION_PENDING_NONE
        assert audit["warm_validation_disposition"] == "zero_nof_warm_restart"
        assert after["lbfgs_latched"]
        assert after["lbfgs_count"] == 0
        assert after["curvature_epoch_floor_state_index"] == after["state_index"]
        assert float(after["warm_safeguard_inverse_scale"]) == controls.alpha
        assert after["warm_restart_used_path_sha256"] == v14._array_sha256(
            after["current_path"]
        )
        assert audit["warm_restart_event"]["validation_classification"] == (
            expected_classification
        )


def test_positive_validation_checkpoints_then_commits_exact_original_candidate(
    monkeypatch, tmp_path
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller, prepared = _latched_warm_validation_preparation(
        monkeypatch, controls
    )
    frozen = prepared.frozen_warm_candidate
    validation_target = _target_band_in_source_frame(
        controller,
        prepared.selected_path,
        controls,
        residual_norm=6.0,
        component_standard_error=0.01,
    )
    after_validation, validation_audit = v14._advance_controller(
        controller, prepared, controls=controls, **validation_target
    )
    assert validation_audit["warm_validation"]["classification"] == (
        "validated_positive"
    )
    checkpoint_directory = tmp_path / "positive-validation-checkpoint"
    v14._save_checkpoint(checkpoint_directory, after_validation)
    restored = v14._controller_from_checkpoint(
        v14._load_npz_record(checkpoint_directory / "controller_checkpoint.npz"),
        controls,
    )
    assert v14._controllers_equal(after_validation, restored)
    registered_request_sha = restored["pending_warm_request_sha256"]
    restored["pending_warm_request_sha256"] = "bogus"
    with pytest.raises(v14.IRIVMError, match="request digest differs"):
        v14._validate_controller(restored, controls)
    restored["pending_warm_request_sha256"] = registered_request_sha
    v14._validate_controller(restored, controls)

    monkeypatch.setattr(
        v14,
        "_build_candidate",
        lambda *args, **kwargs: pytest.fail(
            "validated commit must not regenerate an adaptive candidate"
        ),
    )
    commit = v14._prepare_update(restored, controls).prepared
    assert commit is not None and commit.frozen_warm_commit
    assert not commit.warm_validation_selected and not commit.hold_selected
    np.testing.assert_array_equal(commit.selected_path, frozen["path"])
    committed_digest = hashlib.sha256(
        v14._canonical_json(v14._candidate_digest(commit.lbfgs_candidate))
    ).hexdigest()
    assert committed_digest == frozen["candidate_sha256"]

    target = _target_band_in_source_frame(
        restored,
        commit.selected_path,
        controls,
        residual_norm=3.0,
        component_standard_error=0.01,
    )
    after_commit, commit_audit = v14._advance_controller(
        restored, commit, controls=controls, **target
    )
    assert commit_audit["movement_committed"]
    assert commit_audit["completed_inner_step"] == controller["inner_step"] + 1
    assert after_commit["nof"] == controller["nof"] + 2 * v14.N_INTERNAL
    np.testing.assert_array_equal(after_commit["current_path"], frozen["path"])
    assert (
        after_commit["pending_warm_status"]
        == v14.WARM_VALIDATION_PENDING_NONE
    )


@pytest.mark.parametrize(
    "residual_norm,component_se,expected",
    [(0.0, 0.01, "confirmed_bad"), (0.1, 0.3, "unresolved")],
)
def test_warm_restart_is_bounded_and_old_curvature_cannot_resurrect(
    monkeypatch, tmp_path, residual_norm, component_se, expected
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller, _ = _latched_warm_validation_preparation(
        monkeypatch, controls, state_index=3
    )
    # Register two old observations and old pairs.  The restart floor must
    # exclude them from every later reprojection, not merely zero the arrays.
    controller["observation_paths"][1] = controller["current_path"].copy()
    controller["observation_paths"][1, 1:-1] -= (
        controller["frozen_basis"] * 1.0e-5
    )
    controller["observation_paths"][2] = controller["current_path"]
    controller["observation_gradients"][1:3] = controller["current_grad"]
    controller["observation_gradient_covariances"][1:3] = controller[
        "observation_gradient_covariances"
    ][0]
    controller["observation_covariance_available"][1:3] = controller[
        "observation_covariance_available"
    ][0]
    controller["observation_state_indices"][:3] = (1, 2, 3)
    controller["observation_count"] = 3
    controller["lbfgs_s"][:2] = np.eye(2, v14.N_INTERNAL) * 1.0e-4
    controller["lbfgs_y"][:2] = np.eye(2, v14.N_INTERNAL)
    controller["lbfgs_count"] = 2
    v14._validate_controller(controller, controls)
    monkeypatch.setattr(
        v14,
        "joint_lbfgs_direction",
        lambda gradient, history_s, history_y, diagonal_h0: -controls.alpha
        * gradient,
    )
    prepared = v14._prepare_update(controller, controls).prepared
    assert prepared is not None and prepared.warm_validation_selected
    target = _target_band_in_source_frame(
        controller,
        prepared.selected_path,
        controls,
        residual_norm=residual_norm,
        component_standard_error=component_se,
    )
    after, audit = v14._advance_controller(
        controller, prepared, controls=controls, **target
    )
    assert audit["warm_validation"]["classification"] == expected
    assert after["lbfgs_count"] == 0
    rebuilt = v14._rebuilt_outer_fields_at_current_paid(after, controls)
    assert int(rebuilt["lbfgs_count"]) == 0

    checkpoint_directory = tmp_path / f"restart-{expected}"
    v14._save_checkpoint(checkpoint_directory, after)
    restored = v14._controller_from_checkpoint(
        v14._load_npz_record(
            checkpoint_directory / "controller_checkpoint.npz"
        ),
        controls,
    )
    assert v14._controllers_equal(after, restored)

    # Force a second nonpositive same-path decision without altering the
    # post-restart anchor state.  Even when its lower bound is still zero, the
    # durable marker must bypass the generic anchor HOLD and stop before a
    # second validation band can be requested.

    def nonpositive_raw(candidate, statistics, controls):
        del statistics
        if not candidate.valid:
            return {
                key: np.nan
                for key in (
                    "linear",
                    "quadratic",
                    "score",
                    "standard_error",
                    "lower_bound",
                    "upper_bound",
                )
            }
        return {
            "linear": 1.0,
            "quadratic": 1.0,
            "score": 0.0,
            "standard_error": 0.1,
            "lower_bound": -controls.residual_bound_kappa * 0.1,
            "upper_bound": controls.residual_bound_kappa * 0.1,
        }

    monkeypatch.setattr(
        v14, "_candidate_raw_improvement_evidence", nonpositive_raw
    )
    second = v14._prepare_update(restored, controls)
    assert second.prepared is None
    assert second.terminal["phase"] == "TERMINAL"
    assert second.terminal["reason"] == (
        "warm_candidate_failed_after_one_same_path_restart"
    )
    assert second.terminal["same_path_warm_restart_previously_used"]


def test_pooled_r_stop_precedes_positive_validation_commit(monkeypatch):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.7, m_max=4)
    controller, _ = _latched_warm_validation_preparation(
        monkeypatch, controls
    )
    controller["inner_step"] = 1
    controller["outer_anchor_lower_norm"] = np.asarray(4.0)
    controller["outer_anchor_upper_norm"] = np.asarray(4.0)
    controller["outer_anchor_bounds_ready"] = True
    v14._validate_controller(controller, controls)
    prepared = v14._prepare_update(controller, controls).prepared
    assert prepared is not None and prepared.warm_validation_selected
    target = _target_band_in_source_frame(
        controller,
        prepared.selected_path,
        controls,
        residual_norm=1.0,
        component_standard_error=0.01,
    )
    after, audit = v14._advance_controller(
        controller, prepared, controls=controls, **target
    )
    assert audit["warm_validation"]["classification"] == "validated_positive"
    assert audit["r_stop"]
    assert audit["warm_validation_disposition"] == "discarded_by_pooled_r_stop"
    assert after["pending_warm_status"] == v14.WARM_VALIDATION_PENDING_NONE
    assert after["outer_index"] == controller["outer_index"] + 1
    np.testing.assert_array_equal(after["current_path"], controller["current_path"])


def test_validation_public_state_replays_to_identical_pending_checkpoint(
    monkeypatch, tmp_path
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    controller, prepared = _latched_warm_validation_preparation(
        monkeypatch, controls
    )
    target = _target_band_in_source_frame(
        controller,
        prepared.selected_path,
        controls,
        residual_norm=6.0,
        component_standard_error=0.01,
    )
    band = _oracle_band_from_target(target, prepared.selected_path, band_id=2)
    after, advance = v14._advance_controller(
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
    run_directory = tmp_path / "warm-validation-replay"
    v14._atomic_publish_npz_once(
        v14._state_path(run_directory, 1),
        full_path_unwrapped=controller["current_path"],
        selected_reparameterization_audit_sha256=np.asarray("source-certificate"),
        selected_reparameterization_iterations=np.asarray(1, dtype=np.int64),
        selected_reparameterization_output_residual=np.asarray(0.0),
    )
    metrics = v14.compute_article_metrics(
        after["current_path"], after["current_M"], after["current_grad"]
    )
    v14._save_public_state(
        run_directory, controller, after, prepared, {}, band, advance, metrics
    )
    row = v14._load_npz_record(v14._state_path(run_directory, 2))
    assert str(row["state_kind"].item()) == v14.WARM_VALIDATION_STATE_KIND
    assert str(row["warm_validation_classification"].item()) == (
        "validated_positive"
    )
    assert int(row["warm_validation_band_mean_force_cost"]) == v14.N_INTERNAL
    assert not bool(row["warm_validation_incremented_movement_count"])
    replayed = v14._replay_one_state(controller, row, controls)
    assert v14._controllers_equal(after, replayed)


def test_validation_at_budget_is_complete_and_future_commit_preserves_prefix(
    monkeypatch, tmp_path, causal_initial
):
    """A pending next action must not invalidate its exact paid prefix."""

    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    shared = {}
    original_prepare = v14._prepare_update
    original_evidence = v14._candidate_raw_improvement_evidence

    def registered_source_evidence(candidate, statistics, active_controls):
        if candidate.valid and candidate.label in {
            "LBFGS",
            "WARM_SCALAR_SAFEGUARD",
        }:
            quadratic = float(
                np.dot(
                    candidate.actual_normal_step,
                    candidate.actual_normal_step,
                )
                / (2.0 * active_controls.alpha)
            )
            standard_error = 0.1
            return {
                "linear": quadratic,
                "quadratic": quadratic,
                "score": 0.0,
                "standard_error": standard_error,
                "lower_bound": -active_controls.residual_bound_kappa
                * standard_error,
                "upper_bound": active_controls.residual_bound_kappa
                * standard_error,
            }
        return original_evidence(candidate, statistics, active_controls)

    def forced_validation_prepare(controller, active_controls):
        if (
            int(controller["state_index"]) == 1
            and str(controller["pending_warm_status"])
            == v14.WARM_VALIDATION_PENDING_NONE
        ):
            controller["lbfgs_latched"] = True
            controller["lbfgs_count"] = 0
            controller["inner_step"] = 0
            controller["warm_safeguard_inverse_scale"] = np.asarray(
                active_controls.alpha
            )
            controller["warm_safeguard_source_state_index"] = int(
                controller["state_index"]
            )
            outcome = original_prepare(controller, active_controls)
            assert outcome.prepared is not None
            assert outcome.prepared.warm_validation_selected
            # Construct fresh raw residual blocks in the frozen source frame
            # with P_raw=-2*delta/alpha.  Thus Psi_raw is strictly positive by
            # construction, independent of the real candidate's sign pattern.
            shared["validation_band"] = _fresh_band_along_frozen_delta(
                outcome.proposal_controller,
                outcome.prepared,
                active_controls,
                raw_delta_multiplier=-2.0,
                component_standard_error=1.0e-8,
                band_id=2,
            )
            return outcome
        return original_prepare(controller, active_controls)

    class BudgetBoundaryOracle(SyntheticTenBlockOracle):
        oracle_id = "v17-pytest-positive-validation-at-budget"

        def evaluate_band(self, full_path, band_id, band_directory=None):
            if int(band_id) == 2 and "validation_band" in shared:
                del band_directory
                self.band_calls += 1
                self.call_count += v14.N_INTERNAL
                band = shared["validation_band"]
                np.testing.assert_array_equal(
                    band.centers, np.asarray(full_path)[1:-1]
                )
                return band
            return super().evaluate_band(full_path, band_id, band_directory)

    monkeypatch.setattr(
        v14, "_candidate_raw_improvement_evidence", registered_source_evidence
    )
    monkeypatch.setattr(v14, "_prepare_update", forced_validation_prepare)

    path, metric, gradient = causal_initial

    # At an exact registered horizon the validation state itself is a complete,
    # scoreable prefix even though the already validated movement is pending.
    short_directory = tmp_path / "pending-at-budget"
    short_oracle = BudgetBoundaryOracle(1.0e-3)
    short = v14.run_irivm(
        oracle=short_oracle,
        controls=controls,
        initial_path=path,
        initial_M=metric,
        initial_grad_F=gradient,
        nof_budget=40,
        stop_after_nof=40,
        run_directory=short_directory,
    )
    assert short["completed"] and short["budget_prefix_complete"]
    assert short["requested_prefix_complete"]
    assert short["pending_next_action"]
    assert short["pending_validated_commit_requires_another_band"]
    assert short["pending_warm_status"] == v14.WARM_VALIDATION_PENDING_COMMIT
    assert short_oracle.call_count == 40
    short_state = _load_state(short_directory, 2)
    assert str(short_state["state_kind"].item()) == v14.WARM_VALIDATION_STATE_KIND
    assert not bool(short_state["paid_action_is_movement"])

    # With a longer declared budget, stopping at the same prefix must produce
    # byte-identical public states.  Resuming buys exactly one new target band
    # and commits the original frozen candidate without rewriting the prefix.
    shared.clear()
    long_directory = tmp_path / "pending-then-commit"
    long_oracle = BudgetBoundaryOracle(1.0e-3)
    first_band = v14.run_irivm(
        oracle=long_oracle,
        controls=controls,
        initial_path=path,
        initial_M=metric,
        initial_grad_F=gradient,
        nof_budget=60,
        stop_after_nof=20,
        run_directory=long_directory,
    )
    assert first_band["nof_mean_force_incremental"] == 20
    checkpoint_at_20 = (
        long_directory / "controller_checkpoint.npz"
    ).read_bytes()
    paused = v14.run_irivm(
        oracle=long_oracle,
        controls=controls,
        initial_path=path,
        initial_M=metric,
        initial_grad_F=gradient,
        nof_budget=60,
        stop_after_nof=40,
        run_directory=long_directory,
    )
    assert not paused["completed"]
    assert paused["requested_prefix_complete"]
    assert paused["pending_next_action"]
    prefix_bytes = {
        index: v14._state_path(long_directory, index).read_bytes()
        for index in range(3)
    }
    # Crash window: the validation public row is durable but the replaceable
    # checkpoint still names its source.  Resume must replay that row to the
    # pending state without purchasing a duplicate validation band.
    (long_directory / "controller_checkpoint.npz").write_bytes(
        checkpoint_at_20
    )
    resumed = v14.run_irivm(
        oracle=long_oracle,
        controls=controls,
        initial_path=path,
        initial_M=metric,
        initial_grad_F=gradient,
        nof_budget=60,
        stop_after_nof=60,
        run_directory=long_directory,
    )
    assert resumed["completed"] and resumed["budget_prefix_complete"]
    assert not resumed["pending_next_action"]
    assert resumed["pending_warm_status"] == v14.WARM_VALIDATION_PENDING_NONE
    assert long_oracle.call_count == 60
    for index, encoded in prefix_bytes.items():
        assert v14._state_path(long_directory, index).read_bytes() == encoded
    committed = _load_state(long_directory, 3)
    assert bool(committed["paid_action_is_movement"])
    assert str(committed["selected_candidate_kind"].item()) in {
        "LBFGS",
        "WARM_SCALAR_SAFEGUARD",
    }
    assert int(committed["nof_this_complete_band"]) == v14.N_INTERNAL


@pytest.mark.parametrize(
    "raw_multiplier,component_se,expected_classification",
    [
        (10.0, 0.01, "confirmed_bad"),
        (-0.5, 0.1, "unresolved"),
    ],
)
def test_nonpositive_validation_public_replay_then_terminal_uses_no_new_band(
    monkeypatch,
    tmp_path,
    causal_initial,
    raw_multiplier,
    component_se,
    expected_classification,
):
    controls = v14.IRIVMControls(alpha=1.0e-4, r=0.4, m_max=4)
    shared = {}
    original_prepare = v14._prepare_update
    original_evidence = v14._candidate_raw_improvement_evidence

    def nonpositive_warm_evidence(candidate, statistics, active_controls):
        if candidate.valid and candidate.label in {
            "LBFGS",
            "WARM_SCALAR_SAFEGUARD",
        }:
            quadratic = float(
                np.dot(
                    candidate.actual_normal_step,
                    candidate.actual_normal_step,
                )
                / (2.0 * active_controls.alpha)
            )
            standard_error = 0.1
            return {
                "linear": quadratic,
                "quadratic": quadratic,
                "score": 0.0,
                "standard_error": standard_error,
                "lower_bound": -active_controls.residual_bound_kappa
                * standard_error,
                "upper_bound": active_controls.residual_bound_kappa
                * standard_error,
            }
        return original_evidence(candidate, statistics, active_controls)

    def forced_validation_prepare(controller, active_controls):
        if (
            int(controller["state_index"]) == 1
            and str(controller["pending_warm_status"])
            == v14.WARM_VALIDATION_PENDING_NONE
        ):
            controller["lbfgs_latched"] = True
            controller["lbfgs_count"] = 0
            controller["inner_step"] = 0
            controller["warm_safeguard_inverse_scale"] = np.asarray(
                active_controls.alpha
            )
            controller["warm_safeguard_source_state_index"] = int(
                controller["state_index"]
            )
            outcome = original_prepare(controller, active_controls)
            assert outcome.prepared is not None
            assert outcome.prepared.warm_validation_selected
            shared["validation_band"] = _fresh_band_along_frozen_delta(
                outcome.proposal_controller,
                outcome.prepared,
                active_controls,
                raw_delta_multiplier=raw_multiplier,
                component_standard_error=component_se,
                band_id=2,
            )
            return outcome
        return original_prepare(controller, active_controls)

    class NonpositiveValidationOracle(SyntheticTenBlockOracle):
        oracle_id = (
            "v17-pytest-nonpositive-validation-" + expected_classification
        )

        def evaluate_band(self, full_path, band_id, band_directory=None):
            if int(band_id) == 2 and "validation_band" in shared:
                del band_directory
                self.band_calls += 1
                self.call_count += v14.N_INTERNAL
                band = shared["validation_band"]
                np.testing.assert_array_equal(
                    band.centers, np.asarray(full_path)[1:-1]
                )
                return band
            return super().evaluate_band(full_path, band_id, band_directory)

    monkeypatch.setattr(
        v14, "_candidate_raw_improvement_evidence", nonpositive_warm_evidence
    )
    monkeypatch.setattr(v14, "_prepare_update", forced_validation_prepare)
    path, metric, gradient = causal_initial
    run_directory = tmp_path / expected_classification
    oracle = NonpositiveValidationOracle(1.0e-3)
    v14.run_irivm(
        oracle=oracle,
        controls=controls,
        initial_path=path,
        initial_M=metric,
        initial_grad_F=gradient,
        nof_budget=60,
        stop_after_nof=20,
        run_directory=run_directory,
    )
    checkpoint_at_20 = (
        run_directory / "controller_checkpoint.npz"
    ).read_bytes()
    stopped_at_validation = v14.run_irivm(
        oracle=oracle,
        controls=controls,
        initial_path=path,
        initial_M=metric,
        initial_grad_F=gradient,
        nof_budget=60,
        stop_after_nof=40,
        run_directory=run_directory,
    )
    assert stopped_at_validation["nof_mean_force_incremental"] == 40
    validation_bytes = v14._state_path(run_directory, 2).read_bytes()
    validation_row = _load_state(run_directory, 2)
    assert str(validation_row["warm_validation_classification"].item()) == (
        expected_classification
    )
    assert str(validation_row["warm_validation_disposition"].item()) == (
        "zero_nof_warm_restart"
    )
    assert not bool(validation_row["paid_action_is_movement"])
    assert oracle.call_count == 40

    # Recreate the crash window with the public validation one state ahead.
    (run_directory / "controller_checkpoint.npz").write_bytes(
        checkpoint_at_20
    )
    terminal = v14.run_irivm(
        oracle=oracle,
        controls=controls,
        initial_path=path,
        initial_M=metric,
        initial_grad_F=gradient,
        nof_budget=60,
        stop_after_nof=60,
        run_directory=run_directory,
    )
    assert terminal["phase"] == "TERMINAL"
    assert terminal["nof_mean_force_incremental"] == 40
    assert terminal["algorithmic_zero_nof_stop"]["reason"] == (
        "warm_candidate_failed_after_one_same_path_restart"
    )
    assert oracle.call_count == 40
    assert v14._state_path(run_directory, 2).read_bytes() == validation_bytes

    resumed_terminal = v14.run_irivm(
        oracle=oracle,
        controls=controls,
        initial_path=path,
        initial_M=metric,
        initial_grad_F=gradient,
        nof_budget=60,
        stop_after_nof=60,
        run_directory=run_directory,
    )
    assert resumed_terminal["terminal_rederived_on_resume"]
    assert resumed_terminal["algorithmic_zero_nof_stop"] == terminal[
        "algorithmic_zero_nof_stop"
    ]
    assert oracle.call_count == 40
