import json
from pathlib import Path

import numpy as np
import pytest

import run_development_screen as runner
from ala2d_irivm_v15 import IMPLEMENTATION_ID, load_public_trajectory
from run_development_screen import (
    EXECUTION_MANIFEST_NAME,
    PUBLIC_STATE_LEDGER_NAME,
    RUN_BINDING_NAME,
    ROOT,
    ScreenError,
    execute_specification,
    list_payload,
    load_frozen_plan,
    select_specification,
)


WORKSPACE = ROOT.parents[1]


def test_frozen_plan_expands_to_exact_declared_order_without_writing():
    plan = load_frozen_plan(ROOT)
    payload = list_payload(plan)
    assert plan.raw["plan_id"] == "ala2d-irivm-v15-development-screen-r1"
    assert plan.raw["status"] == "frozen-before-first-v15-grid-result"
    assert IMPLEMENTATION_ID.startswith("ala2d-riemannian-v15")
    assert payload["combination_count"] == 12
    assert payload["run_count"] == 48
    assert payload["parameter_ids_in_frozen_order"] == [
        "a10em5_r0p4_m4",
        "a10em5_r0p4_m8",
        "a10em5_r0p7_m4",
        "a10em5_r0p7_m8",
        "a13em5_r0p4_m4",
        "a13em5_r0p4_m8",
        "a13em5_r0p7_m4",
        "a13em5_r0p7_m8",
        "a16em5_r0p4_m4",
        "a16em5_r0p4_m8",
        "a16em5_r0p7_m4",
        "a16em5_r0p7_m8",
    ]
    runs = payload["runs_in_frozen_order"]
    assert [row["oracle_kind"] for row in runs[:12]] == ["deterministic"] * 12
    assert [row["master_seed"] for row in runs[12:24]] == [1401] * 12
    assert [row["master_seed"] for row in runs[24:36]] == [1402] * 12
    assert [row["master_seed"] for row in runs[36:48]] == [1403] * 12


def test_frozen_plan_records_v15_decisions_and_keeps_v14_as_rationale_only():
    plan = load_frozen_plan(ROOT)
    algorithm = plan.raw["algorithm"]
    assert "U(P_j) <= r L(P_0)" in algorithm["relative_residual_stop_policy"]
    assert "continue with an admissible movement" in algorithm[
        "relative_residual_stop_policy"
    ]
    assert "H0 is the required fallback" in algorithm[
        "unresolved_lbfgs_advantage_policy"
    ]
    assert "if that H0 lower bound is nonpositive" in algorithm[
        "unresolved_lbfgs_advantage_policy"
    ]
    assert "cannot bypass an unresolved takeover comparison" in algorithm[
        "unresolved_lbfgs_advantage_policy"
    ]
    assert "after takeover H0 remains forbidden" in algorithm[
        "post_takeover_policy"
    ]
    revision = plan.raw["revision_rationale"]
    assert revision["prior_decision"] == "NO_COMBINATION_PASSED"
    assert revision["prior_complete_exact_run_count"] == 48
    assert revision["role_in_v15"].startswith("Revision rationale only.")


def test_cli_list_never_enters_manifest_or_execution_path(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("list attempted to publish an execution manifest")

    production_screen_root = ROOT / "runs/development_screen"
    existed_before = production_screen_root.exists()
    manifest_existed_before = (
        production_screen_root / EXECUTION_MANIFEST_NAME
    ).exists()
    monkeypatch.setattr(runner, "ensure_execution_manifest", forbidden)
    assert runner.main(["list"]) == 0
    capsys.readouterr()
    assert production_screen_root.exists() == existed_before
    assert (
        production_screen_root / EXECUTION_MANIFEST_NAME
    ).exists() == manifest_existed_before


def test_plan_hash_binding_fails_closed_on_a_modified_copy(tmp_path):
    copied_root = tmp_path / "campaign"
    plan_target = copied_root / "registrations/DEVELOPMENT_SCREEN_PLAN.json"
    plan_target.parent.mkdir(parents=True)
    payload = json.loads(
        (ROOT / "registrations/DEVELOPMENT_SCREEN_PLAN.json").read_text(
            encoding="utf-8"
        )
    )
    payload["parameter_grid"]["r"] = [0.4, 0.6]
    plan_target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ScreenError, match="plan SHA256 differs"):
        load_frozen_plan(copied_root)


def test_execution_manifest_refuses_unbound_root_level_artifact(tmp_path):
    plan = load_frozen_plan(ROOT)
    screen_root = tmp_path / "development_screen"
    screen_root.mkdir()
    (screen_root / "unregistered.txt").write_text("not part of the frozen layout\n")
    with pytest.raises(ScreenError, match="unexpected root-level"):
        runner.ensure_execution_manifest(
            root=ROOT,
            workspace=WORKSPACE,
            screen_root=screen_root,
            plan=plan,
        )


def test_short_deterministic_budget_is_atomic_traceable_and_resumable(tmp_path):
    plan = load_frozen_plan(ROOT)
    specification = select_specification(
        plan,
        oracle_kind="deterministic",
        parameter_id="a10em5_r0p4_m4",
        master_seed=None,
    )
    screen_root = tmp_path / "development_screen"
    first = execute_specification(
        spec=specification,
        plan=plan,
        stop_after_nof=20,
        root=ROOT,
        workspace=WORKSPACE,
        screen_root=screen_root,
    )
    run_directory = screen_root / "deterministic/a10em5_r0p4_m4"
    assert first["before_incremental_query_equivalents"] == 0
    assert first["after_incremental_query_equivalents"] == 20
    assert first["new_query_equivalents_this_invocation"] == 20
    assert first["paused_at_requested_stop"]
    assert (screen_root / EXECUTION_MANIFEST_NAME).is_file()
    assert (run_directory / RUN_BINDING_NAME).is_file()
    assert (run_directory / "controller_checkpoint.npz").is_file()
    assert (run_directory / PUBLIC_STATE_LEDGER_NAME).is_file()
    core_manifest = json.loads(
        (run_directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert core_manifest["algorithm_variant"] == (
        "v15_development_one_sided_r_resolved_lbfgs"
    )
    assert core_manifest["relative_residual_ambiguous_action"] == (
        "continue_movement_same_frozen_outer"
    )
    assert core_manifest["r_stop_is_one_sided_confident_stop"] is True
    assert "choose valid H0" in core_manifest["ambiguous_candidate_policy"]
    assert "never HOLD solely" in core_manifest["ambiguous_candidate_policy"]
    trajectory = load_public_trajectory(run_directory)
    np.testing.assert_array_equal(trajectory["state_index"], [0, 1])
    np.testing.assert_array_equal(trajectory["nof_mean_force_incremental"], [0, 20])
    ledger = json.loads(
        (run_directory / PUBLIC_STATE_LEDGER_NAME).read_text(encoding="utf-8")
    )
    assert ledger["state_count_including_common_initial"] == 2
    assert [row["incremental_query_equivalents"] for row in ledger["states"]] == [
        0,
        20,
    ]

    second = execute_specification(
        spec=specification,
        plan=plan,
        stop_after_nof=20,
        root=ROOT,
        workspace=WORKSPACE,
        screen_root=screen_root,
    )
    assert second["before_incremental_query_equivalents"] == 20
    assert second["after_incremental_query_equivalents"] == 20
    assert second["new_query_equivalents_this_invocation"] == 0
    np.testing.assert_array_equal(
        load_public_trajectory(run_directory)["nof_mean_force_incremental"],
        [0, 20],
    )


def test_existing_mismatched_run_binding_fails_closed(tmp_path):
    plan = load_frozen_plan(ROOT)
    specification = select_specification(
        plan,
        oracle_kind="deterministic",
        parameter_id="a10em5_r0p4_m4",
        master_seed=None,
    )
    screen_root = tmp_path / "development_screen"
    execute_specification(
        spec=specification,
        plan=plan,
        stop_after_nof=0,
        root=ROOT,
        workspace=WORKSPACE,
        screen_root=screen_root,
    )
    binding_path = (
        screen_root / "deterministic/a10em5_r0p4_m4" / RUN_BINDING_NAME
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["r"] = 0.7
    binding_path.write_text(json.dumps(binding), encoding="utf-8")
    with pytest.raises(ScreenError, match="run binding differs"):
        execute_specification(
            spec=specification,
            plan=plan,
            stop_after_nof=20,
            root=ROOT,
            workspace=WORKSPACE,
            screen_root=screen_root,
        )


def test_short_synthetic_run_uses_seeded_directory_and_development_oracle(tmp_path):
    plan = load_frozen_plan(ROOT)
    specification = select_specification(
        plan,
        oracle_kind="synthetic",
        parameter_id="a10em5_r0p4_m4",
        master_seed=1401,
    )
    screen_root = tmp_path / "development_screen"
    result = execute_specification(
        spec=specification,
        plan=plan,
        stop_after_nof=20,
        root=ROOT,
        workspace=WORKSPACE,
        screen_root=screen_root,
    )
    run_directory = screen_root / "synthetic/seed_1401/a10em5_r0p4_m4"
    assert Path(result["run_directory"]) == run_directory
    assert result["new_query_equivalents_this_invocation"] == 20
    manifest = json.loads(
        (run_directory / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["development_only"] is True
    assert manifest["oracle_formal_evidence"] is False
    assert manifest["oracle_provenance"]["master_seed"] == 1401
    assert manifest["oracle_provenance"]["noise_scale"] == 1.0
    np.testing.assert_array_equal(
        load_public_trajectory(run_directory)["nof_mean_force_incremental"],
        [0, 20],
    )


@pytest.mark.parametrize("stop_after", [-20, 1, 2020])
def test_noncheckpoint_or_out_of_range_stop_is_rejected(tmp_path, stop_after):
    plan = load_frozen_plan(ROOT)
    specification = plan.run_specifications()[0]
    with pytest.raises(ScreenError, match="multiple of 20"):
        execute_specification(
            spec=specification,
            plan=plan,
            stop_after_nof=stop_after,
            root=ROOT,
            workspace=WORKSPACE,
            screen_root=tmp_path / "development_screen",
        )
