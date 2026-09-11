"""Synthetic, zero-MD tests for the V17 formal plotting boundary."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ala2d_irivm_v17 import (  # noqa: E402
    HOLD_STATE_KIND,
    IMPLEMENTATION_ID,
    STATE_KIND,
    WARM_VALIDATION_STATE_KIND,
)
from formal_contract import ACCEPTANCE_SCHEMA, PAID_STATE_KINDS  # noqa: E402
import plot_registered_figures as plotter  # noqa: E402
from riemannian_torus_geometry import compute_article_metrics  # noqa: E402


def _draft():
    return json.loads(
        (ROOT / "registrations/V17_FORMAL_CAMPAIGN_DRAFT.json").read_text(
            encoding="utf-8"
        )
    )


def _registration_for_pure_trajectory_tests():
    registration = _draft()
    registration["candidate"]["selected_parameter"] = {
        "parameter_id": "synthetic-v17-candidate",
        "alpha": 1.0e-4,
        "r": 0.7,
        "m_max": 4,
    }
    registration["seeds"]["optimization"] = [9101, 9102, 9103]
    registration["seeds"]["contour"] = 9102
    registration["plotting"]["contour_seed"] = 9102
    return registration


def _straight_path(offset: float = 0.0):
    t = np.linspace(0.0, 1.0, 22)
    path = np.column_stack((t, 0.5 * t))
    path[1:-1, 1] += offset
    return path


def _trajectory(terminal_kind: str):
    nof = np.arange(0, 2001, 20, dtype=np.int64)
    count = nof.size
    base = _straight_path()
    kinds = np.asarray(["common_initial"] + [STATE_KIND] * (count - 1), dtype="U64")
    kinds[8] = HOLD_STATE_KIND
    kinds[12] = WARM_VALIDATION_STATE_KIND
    kinds[-1] = terminal_kind
    holds = np.asarray(
        [False] + [kind != STATE_KIND for kind in kinds[1:]], dtype=bool
    )
    validations = kinds == WARM_VALIDATION_STATE_KIND
    movements = np.asarray(
        [False] + [kind == STATE_KIND for kind in kinds[1:]], dtype=bool
    )
    movement_count = np.cumsum(movements.astype(np.int64))
    paths = np.empty((count, 22, 2), dtype=np.float64)
    paths[0] = base
    for index in range(1, count):
        paths[index] = paths[index - 1]
        if movements[index]:
            paths[index, 1:-1, 1] += 1.0e-7
    M = np.repeat(np.eye(2)[None, None, :, :], count * 20, axis=0).reshape(
        count, 20, 2, 2
    )
    grad = np.zeros((count, 20, 2), dtype=np.float64)
    metrics = [
        compute_article_metrics(path, metric, force)
        for path, metric, force in zip(paths, M, grad)
    ]
    return {
        "state_index": np.arange(count, dtype=np.int64),
        "nof_mean_force_incremental": nof,
        "nof_mean_force_absolute_including_shared_initial": nof + 20,
        "implementation_id": np.full(count, IMPLEMENTATION_ID),
        "development_only": np.zeros(count, dtype=bool),
        "state_kind": kinds,
        "hold_resample_selected": holds,
        "warm_validation_selected": validations,
        "paid_action_is_movement": movements,
        "movement_count": movement_count,
        "warm_validation_band_mean_force_cost": np.where(
            validations, 20, 0
        ).astype(np.int64),
        "warm_validation_fresh_raw_block_count": np.where(
            validations, 10, 0
        ).astype(np.int64),
        "warm_validation_used_pooled_source_estimate": np.zeros(count, dtype=bool),
        "warm_validation_used_shrinkage_centered_residual": np.zeros(count, dtype=bool),
        "warm_validation_incremented_movement_count": np.zeros(count, dtype=bool),
        "zero_nof_trust_refresh_applied": np.zeros(count, dtype=bool),
        "full_path_unwrapped": paths,
        "M": M,
        "grad_F": grad,
        "I_perp_G": np.asarray([value["I_perp_G"] for value in metrics]),
        "I_arc": np.asarray([value["I_arc"] for value in metrics]),
        "Delta_F_barrier": np.asarray(
            [value["Delta_F_barrier"] for value in metrics]
        ),
        "segment_length_relative_deviation": np.asarray(
            [value["segment_length_relative_deviation"] for value in metrics]
        ),
        "selected_reparameterization_output_residual": np.concatenate(
            ([np.nan], np.zeros(count - 1))
        ),
    }


def _common(trajectory):
    return {
        "full_path_init": np.array(trajectory["full_path_unwrapped"][0], copy=True),
        "M_init": np.array(trajectory["M"][0], copy=True),
        "F_init": np.array(trajectory["grad_F"][0], copy=True),
        "I_perp_G": np.asarray(trajectory["I_perp_G"][0]),
        "I_arc": np.asarray(trajectory["I_arc"][0]),
        "Delta_F_barrier": np.asarray(trajectory["Delta_F_barrier"][0]),
    }


def _completion(tmp_path: Path, terminal_kind: str, *, pending: bool):
    return {
        "completed": True,
        "budget_prefix_complete": True,
        "nof_mean_force_incremental": 2000,
        "nof_budget": 2000,
        "requested_prefix_complete": True,
        "budget_fully_spent": True,
        "paused_at_requested_stop": False,
        "state_count_including_common_initial": 101,
        "implementation_id": IMPLEMENTATION_ID,
        "development_only": False,
        "run_directory": str(tmp_path.resolve()),
        "pending_next_action": pending,
        "pending_validated_commit_requires_another_band": pending,
        "phase": (
            "VALIDATED_FROZEN_WARM_PENDING_COMMIT" if pending else "READY"
        ),
        "pending_warm_status": (
            "validated_positive_pending_commit" if pending else "none"
        ),
        "terminal_kind_for_test_only": terminal_kind,
    }


def test_checked_in_draft_fails_before_hash_or_output_probe(monkeypatch, tmp_path):
    registration_path = ROOT / "registrations/V17_FORMAL_CAMPAIGN_DRAFT.json"
    touched = []

    def forbidden(*args, **kwargs):
        touched.append((args, kwargs))
        raise AssertionError("the DRAFT gate was bypassed")

    monkeypatch.setattr(plotter, "_validate_hash_table", forbidden)
    with pytest.raises(plotter.PlotInputError, match="DRAFT"):
        plotter.load_sealed_registration(registration_path)
    assert touched == []
    assert not (tmp_path / "figures").exists()


def test_direct_render_with_draft_creates_no_directory(tmp_path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(plotter.PlotInputError, match="DRAFT"):
        plotter.render_registered_figures(
            {"registration": _draft()}, output, formal=False
        )
    assert not output.exists()


def test_optimization_and_contour_seeds_are_registration_driven():
    registration = _registration_for_pure_trajectory_tests()
    assert plotter._registered_seeds(registration) == (9101, 9102, 9103)
    assert plotter._single_registered_seed(registration, "contour") == 9102


def test_prior_formal_or_development_seeds_are_rejected():
    registration = _registration_for_pure_trajectory_tests()
    registration["seeds"]["optimization"] = [9101, 41, 9103]
    with pytest.raises(plotter.PlotInputError, match="reuse prior seeds"):
        plotter._registered_seeds(registration)


@pytest.mark.parametrize("terminal_kind", PAID_STATE_KINDS)
def test_plot_trajectory_accepts_each_v17_paid_terminal_kind(terminal_kind):
    registration = _registration_for_pure_trajectory_tests()
    trajectory = _trajectory(terminal_kind)
    result = plotter._validate_trajectory(
        trajectory, 9101, registration, _common(trajectory)
    )
    assert result["nof_mean_force_incremental"][-1] == 2000
    assert result["state_kind"][-1] == terminal_kind
    assert np.array_equal(
        result["full_path_unwrapped"][-1], trajectory["full_path_unwrapped"][-1]
    )


def test_plot_trajectory_rejects_missing_1980_instead_of_carrying_forward():
    registration = _registration_for_pure_trajectory_tests()
    trajectory = _trajectory(STATE_KIND)
    for key, value in tuple(trajectory.items()):
        array = np.asarray(value)
        if array.shape and array.shape[0] == 101:
            trajectory[key] = np.delete(array, -2, axis=0)
    with pytest.raises(plotter.PlotInputError, match="exact 0:20:budget"):
        plotter._validate_trajectory(
            trajectory, 9101, registration, _common(_trajectory(STATE_KIND))
        )


def test_pending_validation_completion_preserves_exact_2000_source_path(tmp_path):
    trajectory = _trajectory(WARM_VALIDATION_STATE_KIND)
    manifest = {"implementation_id": IMPLEMENTATION_ID}
    completion = _completion(
        tmp_path, WARM_VALIDATION_STATE_KIND, pending=True
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "completion.json").write_text(json.dumps(completion), encoding="utf-8")
    path = plotter._validate_run_completion(
        tmp_path, tmp_path / "run_manifest.json", trajectory, 9101
    )
    assert path == tmp_path / "completion.json"
    assert np.array_equal(
        trajectory["full_path_unwrapped"][-1],
        trajectory["full_path_unwrapped"][-2],
    )


def test_acceptance_consumer_imports_the_single_shared_schema():
    assert plotter.ACCEPTANCE_SCHEMA == ACCEPTANCE_SCHEMA
    source = Path(plotter.__file__).read_text(encoding="utf-8")
    assert "accepted_report_schema(analysis)" in source
    assert "acceptance_v1" not in source
    assert "acceptance_v2" not in source


def test_acceptance_report_consumer_accepts_only_shared_v17_schema(tmp_path):
    root = tmp_path / "campaign"
    registration_directory = root / "registrations"
    report_directory = root / "reports"
    registration_directory.mkdir(parents=True)
    report_directory.mkdir()
    registration = _registration_for_pure_trajectory_tests()
    registration_path = registration_directory / "sealed.json"
    registration_path.write_text(json.dumps(registration), encoding="utf-8")
    analysis = {
        "schema_version": ACCEPTANCE_SCHEMA,
        "implementation_id": IMPLEMENTATION_ID,
        "campaign_id": registration["campaign_id"],
        "candidate_parameter_id": "synthetic-v17-candidate",
        "optimization_seeds": [9101, 9102, 9103],
        "cost_history": {
            "optimization_seeds": [9101, 9102, 9103],
            "front_middle": {"pass": True},
            "tail": {"pass": True},
            "pass": True,
        },
        "correct_path": {"pass": True},
        "terminal_noise_test": {"decision": "pass", "pass": True},
        "decision": "pass",
        "pass": True,
    }
    report = {
        "registration_sha256": plotter.sha256_file(registration_path),
        "stage": "acceptance",
        "pass": True,
        "decision": "pass",
        "figure_replacement_authorized": True,
        "figure_files_modified_by_this_stage": False,
        "terminal_artifacts_revalidated_without_new_mean_force_evaluations": True,
        "winner_parameter_id": "synthetic-v17-candidate",
        "acceptance_analysis": analysis,
    }
    report_path = report_directory / "acceptance.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    loaded, winner, runs, references = plotter.validate_acceptance_report(
        report_path, registration, registration_path, root
    )
    assert loaded["acceptance_analysis"]["schema_version"] == ACCEPTANCE_SCHEMA
    assert winner == "synthetic-v17-candidate"
    assert tuple(runs) == (9101, 9102, 9103)
    assert references == {}

    report["acceptance_analysis"]["schema_version"] = "obsolete-schema"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(plotter.PlotInputError, match=ACCEPTANCE_SCHEMA):
        plotter.validate_acceptance_report(
            report_path, registration, registration_path, root
        )


def test_plotter_has_no_md_call_site_or_obsolete_public_identity():
    source = Path(plotter.__file__).read_text(encoding="utf-8")
    assert "evaluate_band(" not in source
    assert "reparameterized_inner_trial" not in source
    assert "riemannian-v13" not in source
    assert "CONFIRMATION_SEEDS" not in source


def test_contour_indices_require_exact_public_rows():
    nof = np.arange(0, 2001, 20, dtype=np.int64)
    assert np.array_equal(
        plotter._exact_indices(nof, np.asarray([0, 200, 400, 2000]), "test"),
        np.asarray([0, 10, 20, 100]),
    )
    with pytest.raises(plotter.PlotInputError, match="exact requested"):
        plotter._exact_indices(nof, np.asarray([0, 210, 2000]), "test")
