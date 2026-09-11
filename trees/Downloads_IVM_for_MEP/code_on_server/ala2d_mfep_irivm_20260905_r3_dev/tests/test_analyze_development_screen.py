import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import ala2d_irivm_v15 as v15_core
import analyze_development_screen as analyzer
from analyze_development_screen import (
    AnalysisError,
    correct_path_guard,
    evaluate_parameter_rules,
    exact_indices,
    normalized_auc,
    passing_sort_key,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = np.arange(0, 2001, 20, dtype=np.int64)
TAIL = np.arange(1620, 2001, 20, dtype=np.int64)
TAIL_SLICE = slice(81, 101)
SEEDS = (1401, 1402, 1403)


def test_analyzer_is_bound_to_v15_identity_and_core_source():
    assert analyzer.ANALYSIS_SCHEMA == (
        "ala2d.irivm.v15.development_screen.analysis.v1"
    )
    assert analyzer.INVENTORY_SCHEMA == (
        "ala2d.irivm.v15.development_screen.nonselecting_inventory.v1"
    )
    assert analyzer.IMPLEMENTATION_ID.startswith("ala2d-riemannian-v15")
    assert Path("src/ala2d_irivm_v15.py") in analyzer.screen_runner.SOURCE_RELATIVES
    assert Path("src/ala2d_irivm_v14.py") not in analyzer.screen_runner.SOURCE_RELATIVES


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("algorithm_variant", "v14_development_frozen_noise_aware_joint_lbfgs"),
        ("relative_residual_ambiguous_action", "exact-center HOLD/resample"),
        ("r_stop_is_one_sided_confident_stop", False),
        ("ambiguous_candidate_policy", "HOLD on unresolved direction comparison"),
    ],
)
def test_analyzer_rejects_non_v15_core_manifest_semantics(field, bad_value):
    plan = analyzer.screen_runner.load_frozen_plan(ROOT)
    spec = plan.run_specifications()[0]
    initial_path, initial_M, initial_gradient = analyzer.screen_runner._load_initial(ROOT)
    oracle = analyzer.screen_runner._instantiate_oracle(ROOT, plan, spec)
    manifest = v15_core._manifest(
        oracle,
        spec.parameter.controls,
        plan.nof_budget,
        initial_path,
        initial_M,
        initial_gradient,
        initial_gradient[:, None, :],
        initial_M[:, None, :, :],
    )
    analyzer._run_core_manifest_checks(
        core_manifest=manifest,
        spec=spec,
        plan=plan,
        root=ROOT,
        initial_path=initial_path,
        initial_M=initial_M,
        initial_gradient=initial_gradient,
    )
    tampered = dict(manifest)
    tampered[field] = bad_value
    with pytest.raises(AnalysisError, match="core manifest field"):
        analyzer._run_core_manifest_checks(
            core_manifest=tampered,
            spec=spec,
            plan=plan,
            root=ROOT,
            initial_path=initial_path,
            initial_M=initial_M,
            initial_gradient=initial_gradient,
        )


def _plan():
    return json.loads(
        (ROOT / "registrations/DEVELOPMENT_SCREEN_PLAN.json").read_text(
            encoding="utf-8"
        )
    )


def _references():
    path = np.column_stack((np.linspace(-0.5, 0.5, 22), np.zeros(22)))
    minimax_path = np.column_stack(
        (np.linspace(-0.5, 0.5, 201), np.zeros(201))
    )
    saddle = np.asarray([0.0, 0.0], dtype=np.float64)
    return path, minimax_path, saddle


def _trajectory(
    *,
    I_perp=0.8,
    I_arc=0.8,
    barrier_error=0.4,
    barrier_reference=1.1433272313575527,
):
    path, _, _ = _references()
    count = CHECKPOINTS.size
    return {
        "nof_mean_force_incremental": CHECKPOINTS.copy(),
        "full_path_unwrapped": np.repeat(path[None, :, :], count, axis=0),
        "I_perp_G": np.full(count, I_perp, dtype=np.float64),
        "I_arc": np.full(count, I_arc, dtype=np.float64),
        "Delta_F_barrier": np.full(
            count, barrier_reference + barrier_error, dtype=np.float64
        ),
        "selected_reparameterization_output_residual": np.full(
            count, 0.5e-6, dtype=np.float64
        ),
    }


def _evidence():
    plan = _plan()
    barrier_reference = plan["posthoc_scoring"][
        "barrier_reference_kcal_per_mol"
    ]
    baseline = _trajectory(
        I_perp=1.0,
        I_arc=1.0,
        barrier_error=1.0,
        barrier_reference=barrier_reference,
    )
    deterministic = _trajectory(
        I_perp=0.75,
        I_arc=0.75,
        barrier_error=0.3,
        barrier_reference=barrier_reference,
    )
    synthetic = {
        seed: _trajectory(
            I_perp=0.8,
            I_arc=0.8,
            barrier_error=0.4,
            barrier_reference=barrier_reference,
        )
        for seed in SEEDS
    }
    return synthetic, deterministic, baseline, barrier_reference, plan


def _evaluate(synthetic, deterministic, baseline, barrier_reference, plan):
    _, minimax_path, saddle = _references()
    return evaluate_parameter_rules(
        synthetic,
        deterministic,
        baseline,
        barrier_reference,
        plan,
        minimax_path=minimax_path,
        saddle_point=saddle,
    )


def _set_tail(trajectory, metric, values, barrier_reference):
    values = np.asarray(values, dtype=np.float64)
    assert values.shape == (20,)
    if metric == "Delta_F_error":
        trajectory["Delta_F_barrier"][TAIL_SLICE] = barrier_reference + values
    else:
        trajectory[metric][TAIL_SLICE] = values


def _copy_evidence():
    return copy.deepcopy(_evidence())


def test_exact_indices_requires_exact_registered_checkpoints_even_for_zero_nof_stop():
    trajectory = {"nof_mean_force_incremental": np.asarray([0, 20, 40, 60])}
    np.testing.assert_array_equal(
        exact_indices(trajectory, np.asarray([20, 60]), "complete"), [1, 3]
    )
    with pytest.raises(AnalysisError, match="missing.*20"):
        exact_indices(
            {"nof_mean_force_incremental": np.asarray([0])},
            np.asarray([20]),
            "zero-NoF early stop",
        )


def test_normalized_auc_is_trapezoidal_and_divided_by_checkpoint_width():
    values = np.asarray([1.0, 3.0, 2.0])
    checkpoints = np.asarray([2.0, 4.0, 8.0])
    expected = (0.5 * (1.0 + 3.0) * 2.0 + 0.5 * (3.0 + 2.0) * 4.0) / 6.0
    assert normalized_auc(values, checkpoints) == pytest.approx(expected)


@pytest.mark.parametrize("metric", ["I_perp_G", "Delta_F_error"])
def test_front_is_strict_for_noisy_metrics_but_I_arc_allows_equality(metric):
    synthetic, deterministic, baseline, barrier_reference, plan = _copy_evidence()
    front_index = int(400 // 20)
    for trajectory in synthetic.values():
        if metric == "Delta_F_error":
            trajectory["Delta_F_barrier"][front_index] = barrier_reference + 1.0
        else:
            trajectory[metric][front_index] = 1.0
    assert not _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]

    synthetic, deterministic, baseline, barrier_reference, plan = _copy_evidence()
    for trajectory in synthetic.values():
        trajectory["I_arc"][front_index] = 1.0
    assert _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]


@pytest.mark.parametrize("metric", ["I_perp_G", "Delta_F_error"])
def test_noisy_tail_checkpoint_win_fraction_accepts_exactly_eighty_percent(metric):
    synthetic, deterministic, baseline, barrier_reference, plan = _copy_evidence()
    exactly_sixteen_wins = np.full(20, 0.9)
    exactly_sixteen_wins[15:19] = 1.1
    exactly_sixteen_wins[-1] = 0.4
    for trajectory in synthetic.values():
        _set_tail(trajectory, metric, exactly_sixteen_wins, barrier_reference)
    assert _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]

    only_fifteen_wins = np.full(20, 0.9)
    only_fifteen_wins[14:19] = 1.1
    only_fifteen_wins[-1] = 0.4
    for trajectory in synthetic.values():
        _set_tail(trajectory, metric, only_fifteen_wins, barrier_reference)
    assert not _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]


@pytest.mark.parametrize("metric", ["I_perp_G", "Delta_F_error"])
def test_noisy_tail_median_must_be_strictly_lower(metric):
    synthetic, deterministic, baseline, barrier_reference, plan = _copy_evidence()
    baseline_tail = np.r_[np.full(10, 0.5), np.full(6, 2.0), np.full(4, 0.1)]
    candidate_tail = np.r_[np.full(10, 0.4), np.full(6, 0.6), np.full(4, 1.5)]
    # Keep the terminal barrier inside the independently registered
    # correct-path limit without changing either distribution's median.
    baseline_tail[[0, -1]] = baseline_tail[[-1, 0]]
    candidate_tail[[0, -1]] = candidate_tail[[-1, 0]]
    _set_tail(baseline, metric, baseline_tail, barrier_reference)
    assert np.median(candidate_tail) == np.median(baseline_tail)
    assert normalized_auc(candidate_tail, TAIL) < normalized_auc(baseline_tail, TAIL)
    assert np.mean(candidate_tail < baseline_tail) == pytest.approx(0.8)
    for trajectory in synthetic.values():
        _set_tail(trajectory, metric, candidate_tail, barrier_reference)
    assert not _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]


@pytest.mark.parametrize("metric", ["I_perp_G", "Delta_F_error"])
def test_noisy_tail_median_auc_must_be_strictly_lower(metric):
    synthetic, deterministic, baseline, barrier_reference, plan = _copy_evidence()
    # Four interior losses have trapezoidal weight four.  The terminal win is
    # fixed at 0.4 so it also satisfies the independent 0.5 barrier-error
    # guard.  Solving for ``high`` makes the curve AUC exactly one.
    high = (19.0 - (0.9 * 15.0 - 0.5 * (0.9 - 0.4))) / 4.0
    median_curve = np.full(20, 0.9)
    median_curve[14:18] = high
    median_curve[-1] = 0.4
    assert normalized_auc(median_curve, TAIL) == pytest.approx(1.0)
    assert np.median(median_curve) < 1.0
    assert np.mean(median_curve < 1.0) == pytest.approx(0.8)

    for index in range(20):
        if index == 19:
            rows = [0.4, 0.4, 0.4]
        else:
            low_seed = index % 2
            rows = [
                median_curve[index],
                median_curve[index],
                median_curve[index] + 0.2,
            ]
            rows[low_seed] -= 0.2
        for seed, value in zip(SEEDS, rows):
            values = synthetic[seed][
                "Delta_F_barrier" if metric == "Delta_F_error" else metric
            ]
            values[81 + index] = (
                barrier_reference + value if metric == "Delta_F_error" else value
            )
    individual = []
    for trajectory in synthetic.values():
        if metric == "Delta_F_error":
            values = np.abs(
                trajectory["Delta_F_barrier"][TAIL_SLICE] - barrier_reference
            )
        else:
            values = trajectory[metric][TAIL_SLICE]
        individual.append(normalized_auc(values, TAIL))
    assert sum(value < 1.0 for value in individual) == 2
    assert not _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]


@pytest.mark.parametrize("metric", ["I_perp_G", "Delta_F_error"])
def test_noisy_tail_requires_two_individual_seed_aucs_to_be_lower(metric):
    synthetic, deterministic, baseline, barrier_reference, plan = _copy_evidence()
    for index in range(20):
        if index == 19:
            rows = (0.4, 0.4, 0.4)
        else:
            rows = (0.8, 1.3, 0.9) if index % 2 == 0 else (0.8, 0.9, 1.3)
        for seed, value in zip(SEEDS, rows):
            values = synthetic[seed][
                "Delta_F_barrier" if metric == "Delta_F_error" else metric
            ]
            values[81 + index] = (
                barrier_reference + value if metric == "Delta_F_error" else value
            )
    assert not _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]


def test_I_arc_tail_uses_nonstrict_checkpoint_median_and_strict_individual_auc():
    synthetic, deterministic, baseline, barrier_reference, plan = _copy_evidence()
    roles = (0.8, 1.0, 1.05)
    for index in range(20):
        rotated = roles[index % 3 :] + roles[: index % 3]
        for seed, value in zip(SEEDS, rotated):
            synthetic[seed]["I_arc"][81 + index] = value
    medians = np.median(
        np.stack([synthetic[seed]["I_arc"][TAIL_SLICE] for seed in SEEDS]),
        axis=0,
    )
    np.testing.assert_array_equal(medians, np.ones(20))
    assert all(
        normalized_auc(synthetic[seed]["I_arc"][TAIL_SLICE], TAIL) < 1.0
        for seed in SEEDS
    )
    assert _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]

    for seed, value in zip(SEEDS, (0.8, 1.0, 1.0)):
        synthetic[seed]["I_arc"][TAIL_SLICE] = value
    assert not _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]

    for seed, value in zip(SEEDS, (0.8, 0.9, 1.0)):
        synthetic[seed]["I_arc"][TAIL_SLICE] = value
    assert _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]


def test_deterministic_front_guard_cannot_be_rescued_by_three_synthetic_repetitions():
    synthetic, deterministic, baseline, barrier_reference, plan = _copy_evidence()
    deterministic["I_perp_G"][20] = 1.0
    assert not _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]


@pytest.mark.parametrize("metric", ["I_perp_G", "I_arc", "Delta_F_error"])
def test_deterministic_tail_auc_is_strictly_lower_for_each_metric(metric):
    synthetic, deterministic, baseline, barrier_reference, plan = _copy_evidence()
    _set_tail(deterministic, metric, np.ones(20), barrier_reference)
    if metric == "Delta_F_error":
        # Keep the separately tested terminal correct-path gate satisfied while
        # preserving an exactly unit trapezoidal AUC.
        values = deterministic["Delta_F_barrier"][TAIL_SLICE] - barrier_reference
        values[-1] = 0.4
        values[-2] = 1.3
        deterministic["Delta_F_barrier"][TAIL_SLICE] = barrier_reference + values
    source = (
        deterministic["Delta_F_barrier"][TAIL_SLICE] - barrier_reference
        if metric == "Delta_F_error"
        else deterministic[metric][TAIL_SLICE]
    )
    assert normalized_auc(source, TAIL) == pytest.approx(1.0)
    assert not _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]


def test_correct_path_guard_is_required_for_deterministic_and_every_seed():
    synthetic, deterministic, baseline, barrier_reference, plan = _copy_evidence()
    synthetic[1403]["selected_reparameterization_output_residual"][-1] = 1.0e-6
    assert not _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]

    synthetic[1403]["selected_reparameterization_output_residual"][-1] = 0.999e-6
    assert _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]

    deterministic["Delta_F_barrier"][-1] = barrier_reference + 0.5000001
    assert not _evaluate(
        synthetic, deterministic, baseline, barrier_reference, plan
    )["pass"]


def test_direct_correct_path_guard_checks_each_registered_geometric_limit():
    plan = _plan()
    guard = plan["posthoc_scoring"]["correct_path_guard"]
    path, minimax_path, saddle = _references()
    assert correct_path_guard(
        path,
        0.5,
        np.nextafter(1.0e-6, 0.0),
        minimax_path,
        saddle,
        guard,
    )["pass"]

    cases = []

    long_jump = path.copy()
    long_jump[11, 1] = 0.5
    cases.append(
        (long_jump, 0.5, 0.5e-6, saddle, "maximum_segment_rad")
    )

    corridor_spike = path.copy()
    corridor_spike[:7, 1] = (0.0, 0.27, 0.54, 0.8, 0.54, 0.27, 0.0)
    cases.append(
        (
            corridor_spike,
            0.5,
            0.5e-6,
            saddle,
            "maximum_nearest_fes_minimax_path_distance_rad",
        )
    )

    rms_offset = path.copy()
    rms_offset[:, 1] = 0.5
    cases.append(
        (
            rms_offset,
            0.5,
            0.5e-6,
            saddle,
            "rms_nearest_fes_minimax_path_distance_rad",
        )
    )
    cases.append(
        (
            path,
            0.5,
            0.5e-6,
            np.asarray([0.0, 0.61]),
            "minimax_saddle_proximity_rad",
        )
    )
    cases.append(
        (
            path,
            0.5000001,
            0.5e-6,
            saddle,
            "maximum_barrier_error_kcal_per_mol",
        )
    )
    cases.append(
        (
            path,
            0.5,
            1.0e-6,
            saddle,
            "article_reparameterization_output_residual_strictly_less_than",
        )
    )
    for candidate_path, barrier_error, residual, candidate_saddle, failure in cases:
        result = correct_path_guard(
            candidate_path,
            barrier_error,
            residual,
            minimax_path,
            candidate_saddle,
            guard,
        )
        assert not result["pass"]
        assert failure in result["failures"]


def test_zero_nof_early_stop_is_not_silently_carried_forward():
    synthetic, deterministic, baseline, barrier_reference, plan = _copy_evidence()
    for key, value in list(synthetic[1402].items()):
        if isinstance(value, np.ndarray) and value.shape[:1] == CHECKPOINTS.shape:
            synthetic[1402][key] = value[:1]
    with pytest.raises(AnalysisError, match="missing"):
        _evaluate(synthetic, deterministic, baseline, barrier_reference, plan)


def test_passing_sort_key_applies_every_registered_tie_break_in_order():
    base_result = {
        "pass": True,
        "selection_metrics": {
            "joint_front_middle_checkpoint_win_count": 200,
            "minimum_component_tail_checkpoint_win_fraction": 0.8,
            "worst_individual_synthetic_seed_tail_I_perp_G_auc_ratio": 0.9,
            "worst_individual_synthetic_seed_tail_Delta_F_error_auc_ratio": 0.9,
            "deterministic_worst_all_metric_checkpoint_ratio": 0.9,
        },
    }
    base_controls = {"alpha": 0.00013, "r": 0.7, "m_max": 8}
    field_cases = [
        ("joint_front_middle_checkpoint_win_count", 201),
        ("minimum_component_tail_checkpoint_win_fraction", 0.85),
        ("worst_individual_synthetic_seed_tail_I_perp_G_auc_ratio", 0.85),
        ("worst_individual_synthetic_seed_tail_Delta_F_error_auc_ratio", 0.85),
        ("deterministic_worst_all_metric_checkpoint_ratio", 0.85),
    ]
    for field, preferred_value in field_cases:
        preferred_result = copy.deepcopy(base_result)
        preferred_result["selection_metrics"][field] = preferred_value
        rows = [
            ("base", base_result, base_controls),
            ("preferred", preferred_result, base_controls),
        ]
        assert sorted(
            rows, key=lambda row: passing_sort_key(row[1], row[2])
        )[0][0] == "preferred"

    for field, preferred_value in (
        ("alpha", 0.00010),
        ("r", 0.4),
        ("m_max", 4),
    ):
        preferred_controls = copy.deepcopy(base_controls)
        preferred_controls[field] = preferred_value
        rows = [
            ("base", base_result, base_controls),
            ("preferred", base_result, preferred_controls),
        ]
        assert sorted(
            rows, key=lambda row: passing_sort_key(row[1], row[2])
        )[0][0] == "preferred"


def test_passing_sort_key_is_lexicographic_in_the_registered_priority_order():
    def result(joint, fraction, I_ratio, F_ratio, deterministic_ratio):
        return {
            "pass": True,
            "selection_metrics": {
                "joint_front_middle_checkpoint_win_count": joint,
                "minimum_component_tail_checkpoint_win_fraction": fraction,
                "worst_individual_synthetic_seed_tail_I_perp_G_auc_ratio": I_ratio,
                "worst_individual_synthetic_seed_tail_Delta_F_error_auc_ratio": F_ratio,
                "deterministic_worst_all_metric_checkpoint_ratio": deterministic_ratio,
            },
        }

    ordinary_result = result(200, 0.8, 0.9, 0.9, 0.9)
    ordinary_controls = {"alpha": 0.00013, "r": 0.7, "m_max": 8}
    cases = [
        (
            result(201, 0.0, 99.0, 99.0, 99.0),
            ordinary_controls,
            result(200, 1.0, 0.0, 0.0, 0.0),
            {"alpha": 0.00010, "r": 0.4, "m_max": 4},
        ),
        (
            result(200, 0.9, 99.0, 99.0, 99.0),
            ordinary_controls,
            result(200, 0.8, 0.0, 0.0, 0.0),
            {"alpha": 0.00010, "r": 0.4, "m_max": 4},
        ),
        (
            result(200, 0.8, 0.8, 99.0, 99.0),
            ordinary_controls,
            result(200, 0.8, 0.9, 0.0, 0.0),
            {"alpha": 0.00010, "r": 0.4, "m_max": 4},
        ),
        (
            result(200, 0.8, 0.9, 0.8, 99.0),
            ordinary_controls,
            result(200, 0.8, 0.9, 0.9, 0.0),
            {"alpha": 0.00010, "r": 0.4, "m_max": 4},
        ),
        (
            result(200, 0.8, 0.9, 0.9, 0.8),
            {"alpha": 0.00016, "r": 0.7, "m_max": 8},
            ordinary_result,
            {"alpha": 0.00010, "r": 0.4, "m_max": 4},
        ),
        (
            ordinary_result,
            {"alpha": 0.00010, "r": 0.7, "m_max": 8},
            ordinary_result,
            {"alpha": 0.00013, "r": 0.4, "m_max": 4},
        ),
        (
            ordinary_result,
            {"alpha": 0.00013, "r": 0.4, "m_max": 8},
            ordinary_result,
            {"alpha": 0.00013, "r": 0.7, "m_max": 4},
        ),
    ]
    for preferred_result, preferred_controls, other_result, other_controls in cases:
        assert passing_sort_key(preferred_result, preferred_controls) < passing_sort_key(
            other_result, other_controls
        )


def test_analysis_binding_includes_campaign_analysis_source(tmp_path):
    root = tmp_path / "campaign"
    (root / "src").mkdir(parents=True)
    (root / "src/analyze_development_screen.py").write_text("analyzer-v1\n")
    campaign_source = root / "src/campaign_analysis.py"
    campaign_source.write_text("helper-v1\n")
    first = analyzer._analysis_source_hashes(root)
    assert set(first) == {
        "src/analyze_development_screen.py",
        "src/campaign_analysis.py",
    }
    campaign_source.write_text("helper-v2\n")
    second = analyzer._analysis_source_hashes(root)
    assert second["src/campaign_analysis.py"] != first["src/campaign_analysis.py"]


def test_blind_scorer_evaluates_every_public_path_and_ignores_stored_metrics(
    monkeypatch, tmp_path
):
    class FakeOracle:
        instances = []

        def __init__(self, **kwargs):
            del kwargs
            self.call_count = 0
            self.instances.append(self)

        def evaluate_band(self, path, band_id):
            del band_id
            count = np.asarray(path).shape[0] - 2
            self.call_count += count
            return SimpleNamespace(
                M=np.repeat(np.eye(2)[None, :, :], count, axis=0),
                grad_F=np.zeros((count, 2), dtype=np.float64),
            )

    monkeypatch.setattr(analyzer, "SurrogateOracle", FakeOracle)
    base_path = np.column_stack((np.linspace(-0.5, 0.5, 22), np.zeros(22)))
    paths = np.repeat(base_path[None, :, :], 4, axis=0)
    trajectory = {
        "nof_mean_force_incremental": np.arange(4, dtype=np.int64) * 20,
        "full_path_unwrapped": paths,
        "state_kind": np.asarray(["common_initial", "trial", "trial", "trial"]),
        "I_perp_G": np.full(4, 999.0),
        "I_arc": np.full(4, 999.0),
        "Delta_F_barrier": np.full(4, 999.0),
        "selected_reparameterization_output_residual": np.asarray(
            [np.nan, 1e-7, 1e-7, 1e-7]
        ),
    }
    scored = analyzer.score_public_paths_blind(
        trajectory,
        fes_npz=tmp_path / "unused_fes.npz",
        metric_bank_npz=tmp_path / "unused_metric.npz",
        label="fake public trajectory",
    )
    assert len(FakeOracle.instances) == 1
    assert FakeOracle.instances[0].call_count == 4 * 20
    np.testing.assert_array_equal(scored["I_perp_G"], np.zeros(4))
    np.testing.assert_array_equal(scored["Delta_F_barrier"], np.zeros(4))
    assert not np.array_equal(scored["I_perp_G"], trajectory["I_perp_G"])


def test_public_state_ledger_hash_tamper_fails_before_analysis(monkeypatch, tmp_path):
    plan = analyzer.screen_runner.load_frozen_plan(ROOT)
    spec = plan.run_specifications()[0]
    screen_root = tmp_path / "screen"
    manifest_path = screen_root / analyzer.screen_runner.EXECUTION_MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(b"manifest-binding\n")
    run_directory = screen_root / spec.relative_directory
    state_directory = run_directory / "states"
    state_directory.mkdir(parents=True)
    binding = analyzer.screen_runner._run_binding_payload(
        spec=spec, plan=plan, manifest_path=manifest_path
    )
    (run_directory / analyzer.screen_runner.RUN_BINDING_NAME).write_bytes(
        analyzer._canonical_json_bytes(binding)
    )
    (run_directory / "run_manifest.json").write_text("{}\n")
    state_paths = []
    for index in range(3):
        path = state_directory / f"state_{index:06d}.npz"
        path.write_bytes(f"state-{index}\n".encode())
        state_paths.append(path)
    ledger = {
        "schema_version": analyzer.screen_runner.LEDGER_SCHEMA,
        "formal_evidence": False,
        "implementation_id": analyzer.IMPLEMENTATION_ID,
        "execution_manifest_sha256": analyzer.sha256_file(manifest_path),
        "run_binding_sha256": analyzer.sha256_file(
            run_directory / analyzer.screen_runner.RUN_BINDING_NAME
        ),
        "core_run_manifest_sha256": analyzer.sha256_file(
            run_directory / "run_manifest.json"
        ),
        "oracle_kind": spec.oracle_kind,
        "master_seed": spec.master_seed,
        "parameter_id": spec.parameter.parameter_id,
        "state_count_including_common_initial": 3,
        "last_incremental_query_equivalents": 40,
        "states": [
            {
                "state_index": index,
                "incremental_query_equivalents": index * 20,
                "relative_file": f"states/state_{index:06d}.npz",
                "sha256": analyzer.sha256_file(path),
            }
            for index, path in enumerate(state_paths)
        ],
    }
    (run_directory / analyzer.screen_runner.PUBLIC_STATE_LEDGER_NAME).write_bytes(
        analyzer._canonical_json_bytes(ledger)
    )
    completion = {
        "completed": False,
        "paused_at_requested_stop": False,
        "algorithmic_zero_nof_stop": {"reason": "no admissible move"},
        "nof_mean_force_incremental": 40,
        "nof_budget": 2000,
        "state_count_including_common_initial": 3,
        "implementation_id": analyzer.IMPLEMENTATION_ID,
        "development_only": True,
        "run_directory": str(run_directory.resolve()),
    }
    (run_directory / "completion.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    initial_path = np.column_stack((np.linspace(-0.5, 0.5, 22), np.zeros(22)))
    initial_M = np.repeat(np.eye(2)[None, :, :], 20, axis=0)
    initial_gradient = np.zeros((20, 2), dtype=np.float64)
    fake_trajectory = {
        "nof_mean_force_incremental": np.asarray([0, 20, 40], dtype=np.int64),
        "full_path_unwrapped": np.repeat(initial_path[None, :, :], 3, axis=0),
        "M": np.repeat(initial_M[None, :, :, :], 3, axis=0),
        "grad_F": np.repeat(initial_gradient[None, :, :], 3, axis=0),
    }
    monkeypatch.setattr(analyzer, "_run_core_manifest_checks", lambda **kwargs: None)
    monkeypatch.setattr(analyzer, "load_public_trajectory", lambda path: fake_trajectory)
    record, _ = analyzer.validate_one_run(
        root=ROOT,
        screen_root=screen_root,
        plan=plan,
        spec=spec,
        execution_manifest_path=manifest_path,
        initial_path=initial_path,
        initial_M=initial_M,
        initial_gradient=initial_gradient,
    )
    assert record["status"] == "algorithmic_zero_nof_stop_before_exact_2000"
    assert not record["complete"]

    state_paths[1].write_bytes(b"tampered\n")
    with pytest.raises(AnalysisError, match="hash differs"):
        analyzer.validate_one_run(
            root=ROOT,
            screen_root=screen_root,
            plan=plan,
            spec=spec,
            execution_manifest_path=manifest_path,
            initial_path=initial_path,
            initial_M=initial_M,
            initial_gradient=initial_gradient,
        )
    state_paths[1].write_bytes(b"state-1\n")
    (state_directory / "state_000003.npz").write_bytes(b"late-unledgered-state\n")
    with pytest.raises(AnalysisError, match="ledger/file state counts differ"):
        analyzer.validate_one_run(
            root=ROOT,
            screen_root=screen_root,
            plan=plan,
            spec=spec,
            execution_manifest_path=manifest_path,
            initial_path=initial_path,
            initial_M=initial_M,
            initial_gradient=initial_gradient,
        )


def test_allow_incomplete_is_inventory_only_even_when_all_48_are_complete(
    monkeypatch, tmp_path
):
    plan = analyzer.screen_runner.load_frozen_plan(ROOT)
    root = tmp_path / "workspace/code_on_server/campaign"
    (root / "src").mkdir(parents=True)
    (root / "src/analyze_development_screen.py").write_text("analyzer\n")
    (root / "src/campaign_analysis.py").write_text("analysis helper\n")
    screen_root = root / "runs/development_screen"
    screen_root.mkdir(parents=True)
    manifest_path = screen_root / analyzer.screen_runner.EXECUTION_MANIFEST_NAME
    manifest_path.write_text("{}\n")

    monkeypatch.setattr(analyzer.screen_runner, "load_frozen_plan", lambda value: plan)
    monkeypatch.setattr(analyzer.screen_runner, "_validate_required_files", lambda value: None)
    monkeypatch.setattr(analyzer.screen_runner, "_validate_formal_pdfs", lambda value: {})
    monkeypatch.setattr(
        analyzer,
        "validate_execution_manifest",
        lambda **kwargs: (manifest_path, {}, {"verified": True}),
    )
    initial_path = np.column_stack((np.linspace(-0.5, 0.5, 22), np.zeros(22)))
    initial_M = np.repeat(np.eye(2)[None, :, :], 20, axis=0)
    initial_gradient = np.zeros((20, 2), dtype=np.float64)
    monkeypatch.setattr(
        analyzer.screen_runner,
        "_load_initial",
        lambda value: (initial_path, initial_M, initial_gradient),
    )

    def fake_validate(**kwargs):
        spec = kwargs["spec"]
        return (
            {
                "ordinal": spec.ordinal,
                "oracle_kind": spec.oracle_kind,
                "master_seed": spec.master_seed,
                "parameter_id": spec.parameter.parameter_id,
                "relative_directory": spec.relative_directory.as_posix(),
                "status": "complete_exact_0_to_2000",
                "complete": True,
                "last_incremental_query_equivalents": 2000,
            },
            {"unused": np.asarray(1)},
        )

    monkeypatch.setattr(analyzer, "validate_one_run", fake_validate)
    captured = {}

    def fake_publish(**kwargs):
        captured.update(kwargs)
        return {"status": "COMPLETE_INVENTORY_NO_SELECTION"}

    monkeypatch.setattr(analyzer, "publish_nonselecting_inventory", fake_publish)
    monkeypatch.setattr(
        analyzer,
        "validate_barrier_reference",
        lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("inventory-only mode must not start scoring")
        ),
    )
    result = analyzer.analyze_development_screen(
        root=root, screen_root=screen_root, allow_incomplete=True
    )
    assert result["status"] == "COMPLETE_INVENTORY_NO_SELECTION"
    assert len(captured["inventory"]) == 48
    assert all(item["complete"] for item in captured["inventory"])


def test_nonselecting_inventory_csv_is_self_identifying_and_has_no_ranking(tmp_path):
    plan = analyzer.screen_runner.load_frozen_plan(ROOT)
    inventory = [
        {
            "ordinal": spec.ordinal,
            "oracle_kind": spec.oracle_kind,
            "master_seed": spec.master_seed,
            "parameter_id": spec.parameter.parameter_id,
            "relative_directory": spec.relative_directory.as_posix(),
            "status": "missing",
            "complete": False,
            "last_incremental_query_equivalents": None,
        }
        for spec in plan.run_specifications()
    ]
    payload = analyzer.publish_nonselecting_inventory(
        root=tmp_path,
        plan=plan,
        inventory=inventory,
        binding={"fixture": "nonselecting-inventory"},
    )
    assert payload["status"] == "INCOMPLETE_DEVELOPMENT_SCREEN"
    assert payload["selection_performed"] is False
    assert payload["recommended_parameter_id"] is None
    assert payload["diagnostic_parameter_id"] is None
    assert payload["artifacts"]["static_png"] == []
    csv_path = tmp_path / payload["artifacts"]["inventory_csv"]["relative_path"]
    header = csv_path.read_text().splitlines()[0].split(",")
    assert header[:3] == [
        "artifact_scope",
        "formal_evidence",
        "analysis_input_sha256",
    ]
    first_row = csv_path.read_text().splitlines()[1]
    assert first_row.startswith("DEVELOPMENT ONLY,false,")
    report = json.loads(
        (tmp_path / "reports/development_screen_analysis.json").read_text()
    )
    assert report["analysis_input_sha256"] == payload["analysis_input_sha256"]
