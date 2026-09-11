"""Focused runner, replay, NoF-audit, and lock tests for the R6 draft.

Only deterministic synthetic arrays and temporary directories are used.  The
restrained-MD oracle is never constructed and no external MD program runs.
"""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest


CAMPAIGN = Path(__file__).resolve().parents[1]
SRC = CAMPAIGN / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import ala2d_irivm_v17 as core
from ala2d_oracles import OracleBand
import formal_contract as contract
import run_v17_formal_campaign as runner


class CountingSyntheticOracle:
    """Ten-block analytic oracle used only to count requested bands."""

    oracle_id = "r6-pytest-counting-synthetic"
    formal_evidence = False

    def __init__(self, noise: float = 1.0e-3):
        self.noise = float(noise)
        self.band_calls = 0
        self.call_count = 0
        self.requested_band_ids: List[int] = []
        self.requested_paths: List[np.ndarray] = []

    def provenance(self) -> Mapping[str, object]:
        return {
            "oracle_id": self.oracle_id,
            "formal_evidence": False,
            "noise": self.noise,
            "test_only": True,
        }

    def evaluate_band(
        self,
        full_path: np.ndarray,
        band_id: int,
        band_directory: Optional[Path] = None,
    ) -> OracleBand:
        del band_directory
        path = np.asarray(full_path, dtype=np.float64)
        centers = path[1:-1]
        self.band_calls += 1
        self.call_count += core.N_INTERNAL
        self.requested_band_ids.append(int(band_id))
        self.requested_paths.append(path.copy())

        mean = np.column_stack(
            (
                0.2 * np.sin(centers[:, 0]) + 0.1,
                0.15 * np.cos(centers[:, 1]) - 0.03,
            )
        )
        offsets = np.linspace(-1.0, 1.0, 10)
        pattern = np.column_stack(
            (
                np.linspace(0.7, 1.3, core.N_INTERNAL),
                np.linspace(-1.2, -0.8, core.N_INTERNAL),
            )
        )
        gradient_blocks = (
            mean[:, None, :]
            + self.noise * offsets[None, :, None] * pattern[:, None, :]
        )
        metric = np.repeat(
            np.eye(core.DIMENSION)[None, :, :], core.N_INTERNAL, axis=0
        )
        metric_blocks = np.repeat(metric[:, None, :, :], 10, axis=1)
        frame_counts = np.full(core.N_INTERNAL, 100, dtype=np.int64)
        discarded = np.zeros(core.N_INTERNAL, dtype=np.int64)
        block_counts = core._block_frame_counts(frame_counts, 10)
        gradient = core._weighted_block_mean(gradient_blocks, block_counts)
        metric = core._weighted_block_mean(metric_blocks, block_counts)
        return OracleBand(
            centers=centers.copy(),
            mean_coordinates=centers.copy(),
            grad_F=gradient,
            M=metric,
            grad_F_block_means=gradient_blocks,
            M_block_means=metric_blocks,
            seeds=np.arange(core.N_INTERNAL, dtype=np.int64)
            + 1000 * int(band_id),
            thermostat_seeds=np.arange(core.N_INTERNAL, dtype=np.int64)
            + 2000 * int(band_id),
            frame_counts=frame_counts,
            discarded_frame_counts=discarded,
            wall_seconds=np.zeros(core.N_INTERNAL, dtype=np.float64),
            nof=core.N_INTERNAL,
            metadata={"test_only": True, "band_id": int(band_id)},
        )


def synthetic_initial() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinate = np.linspace(-1.0, 1.0, core.N_PATH)
    path = np.column_stack((coordinate, 0.5 * coordinate))
    metric = np.repeat(
        np.eye(core.DIMENSION)[None, :, :], core.N_INTERNAL, axis=0
    )
    centers = path[1:-1]
    gradient = np.column_stack(
        (
            0.2 * np.sin(centers[:, 0]) + 0.1,
            0.15 * np.cos(centers[:, 1]) - 0.03,
        )
    )
    return path, metric, gradient


def controls() -> core.IRIVMControls:
    return core.IRIVMControls(alpha=1.0e-4, r=0.5, m_max=4)


def run_synthetic(
    run_directory: Path,
    oracle: CountingSyntheticOracle,
    *,
    budget: int,
    stop: Optional[int] = None,
) -> Mapping[str, object]:
    path, metric, gradient = synthetic_initial()
    return core.run_irivm(
        oracle=oracle,
        controls=controls(),
        initial_path=path,
        initial_M=metric,
        initial_grad_F=gradient,
        nof_budget=budget,
        stop_after_nof=stop,
        run_directory=run_directory,
    )


def load_state(run_directory: Path, index: int) -> Dict[str, np.ndarray]:
    return core._load_npz_record(core._state_path(run_directory, index))


def assert_npz_records_equal(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> None:
    assert left.keys() == right.keys()
    for key in left:
        assert core._array_equal(left[key], right[key]), key


def draft_registration() -> Dict[str, object]:
    return json.loads(
        runner.DRAFT_REGISTRATION.read_text(encoding="utf-8")
    )


def sealed_registration() -> Dict[str, object]:
    """Minimal fully typed future registration for pure validator tests."""

    implementation_hash = hashlib.sha256(
        (CAMPAIGN / "src/ala2d_irivm_v17.py").read_bytes()
    ).hexdigest()
    return {
        "schema_version": contract.EXECUTABLE_REGISTRATION_SCHEMA,
        "status": contract.SEALED_STATUS,
        "formal_evidence": True,
        "new_restrained_md_authorized": True,
        "executable": True,
        "candidate": {
            "selected_parameter": {
                "parameter_id": "a1em4_r0p5_m4",
                "alpha": 1.0e-4,
                "r": 0.5,
                "m_max": 4,
            },
            "implementation_id": core.IMPLEMENTATION_ID,
            "implementation_sha256": implementation_hash,
            "development_exact_trajectory_count_required": 288,
            "development_plan_path": "../r5/registrations/plan.json",
            "development_plan_sha256": "1" * 64,
            "development_execution_manifest_path": "../r5/manifest.json",
            "development_execution_manifest_sha256": "2" * 64,
            "development_analysis_path": "../r5/reports/analysis.json",
            "development_analysis_sha256": "3" * 64,
            "selection_receipt_path": "registrations/selection_receipt.json",
            "selection_receipt_sha256": "4" * 64,
        },
        "seeds": {
            "optimization": [2101, 2102, 2103],
            "terminal_diagnostic": list(range(2201, 2211)),
            "bootstrap": 2301,
            "contour": 2102,
            "forbidden_prior_seed_sets": [
                [41, 42, 43],
                [1401, 1402, 1403],
                [1701, 1702, 1703],
            ],
            "novelty_attestation_path": "registrations/seed_novelty.json",
            "novelty_attestation_sha256": "5" * 64,
        },
        "execution": {
            "jobs": 1,
            "lock_kind": "host-global fcntl",
            "lock_path_relative_to_workspace": (
                contract.HOST_GLOBAL_LOCK_RELATIVE_TO_WORKSPACE
            ),
            "run_root_must_be_absent": True,
            "registered_formal_md_hostname": "synthetic-test-host",
            "cross_host_execution_authorized": False,
            "maximum_attempts_per_logical_evaluation": 3,
            "mutagen_preflight_required": True,
            "mutagen": {
                "executable": "/synthetic/mutagen",
                "executable_sha256": "6" * 64,
                "version": "0.18.1",
                "session_identifier": "synthetic-session",
                "session_name": "synthetic-name",
                "local_root": "/synthetic/local",
                "remote_protocol": "ssh",
                "remote_user": "synthetic-user",
                "remote_host": "synthetic-host",
                "remote_root": "/synthetic/remote",
                "mode": "two-way-resolved",
                "hash": "sha256",
                "required_status": "Watching for changes",
                "endpoints_connected_and_scanned_required": True,
                "paused_must_be_false": True,
            },
        },
        "hashes": {
            "algorithm": "sha256",
            "source": {"src/ala2d_irivm_v17.py": "7" * 64},
            "input": {"inputs/reference/common.npz": "8" * 64},
            "runtime": {"/synthetic/gmx": "9" * 64},
        },
        "nof": {
            "mean_force_vector_dimension": 2,
            "complete_band": core.N_INTERNAL,
            "incremental_budget_per_optimization_run": 2000,
            "common_initial_incremental_cost": 0,
            "repeated_center_is_charged_again": True,
            "hold_and_validation_bands_are_charged": True,
            "zero_nof_refresh_creates_public_state": False,
            "hidden_candidate_or_recovery_evaluations": False,
        },
    }


def audit_trajectory() -> Dict[str, object]:
    """Full 0:20:2000 trajectory including HOLD and validation bands."""

    count = 101
    kind_values = [contract.COMMON_INITIAL_STATE_KIND] + [
        core.STATE_KIND
    ] * (count - 1)
    kind_values[50] = core.HOLD_STATE_KIND
    kind_values[100] = core.WARM_VALIDATION_STATE_KIND
    # Convert only after assigning the longest label; assigning it into a
    # fixed-width NumPy string array would silently truncate the fixture.
    kinds = np.asarray(kind_values)
    paths = np.zeros((count, core.N_PATH, core.DIMENSION), dtype=np.float64)
    base = np.linspace(-1.0, 1.0, core.N_PATH)
    paths[0, :, 0] = base
    paths[0, :, 1] = 0.5 * base
    movement_count = np.zeros(count, dtype=np.int64)
    for index in range(1, count):
        paths[index] = paths[index - 1]
        if str(kinds[index]) == core.STATE_KIND:
            paths[index, 1:-1, 0] += index * 1.0e-7
            movement_count[index] = movement_count[index - 1] + 1
        else:
            movement_count[index] = movement_count[index - 1]
    nof = np.arange(count, dtype=np.int64) * core.N_INTERNAL
    movement = kinds == core.STATE_KIND
    hold = np.asarray(
        [False] + [str(kind) != core.STATE_KIND for kind in kinds[1:]],
        dtype=bool,
    )
    validation = kinds == core.WARM_VALIDATION_STATE_KIND
    validation_cost = np.where(validation, core.N_INTERNAL, 0).astype(np.int64)
    validation_blocks = np.where(validation, 10, 0).astype(np.int64)
    metric = np.repeat(
        np.eye(core.DIMENSION)[None, None, :, :],
        count * core.N_INTERNAL,
        axis=0,
    ).reshape(count, core.N_INTERNAL, core.DIMENSION, core.DIMENSION)
    return {
        "implementation_id": np.asarray([core.IMPLEMENTATION_ID] * count),
        "development_only": np.zeros(count, dtype=bool),
        "state_index": np.arange(count, dtype=np.int64),
        "nof_mean_force_incremental": nof,
        "nof_mean_force_absolute_including_shared_initial": nof + 20,
        "state_kind": kinds,
        "full_path_unwrapped": paths,
        "M": metric,
        "grad_F": np.zeros((count, core.N_INTERNAL, core.DIMENSION)),
        "I_perp_G": np.linspace(3.0, 1.0, count),
        "I_arc": np.linspace(0.3, 0.1, count),
        "Delta_F_barrier": np.linspace(2.0, 2.4, count),
        "segment_length_relative_deviation": np.zeros(count),
        "selected_reparameterization_output_residual": np.zeros(count),
        "hold_resample_selected": hold,
        "warm_validation_selected": validation,
        "warm_validation_band_mean_force_cost": validation_cost,
        "warm_validation_incremented_movement_count": np.zeros(count, dtype=bool),
        "warm_validation_fresh_raw_block_count": validation_blocks,
        "warm_validation_used_pooled_source_estimate": np.zeros(
            count, dtype=bool
        ),
        "warm_validation_used_shrinkage_centered_residual": np.zeros(
            count, dtype=bool
        ),
        "paid_action_is_movement": movement,
        "movement_count": movement_count,
        "zero_nof_trust_refresh_applied": np.zeros(count, dtype=bool),
    }


def complete_attempt_records(
    trajectory: Mapping[str, object],
    *,
    parameter_id: str = "a1em4_r0p5_m4",
    master_seed: int = 2101,
) -> List[Dict[str, object]]:
    paths = np.asarray(trajectory["full_path_unwrapped"], dtype=np.float64)
    records: List[Dict[str, object]] = []
    for band_id in range(1, paths.shape[0]):
        for image_index in range(core.N_INTERNAL):
            attempt_id = f"b{band_id:06d}-i{image_index:02d}-t0-p0"
            identity = {
                "attempt_id": attempt_id,
                "parameter_id": parameter_id,
                "master_seed": master_seed,
                "band_id": band_id,
                "image_index": image_index,
                "center": paths[band_id, image_index + 1].tolist(),
                "registration_sha256": "a" * 64,
                "registration_binding_sha256": "b" * 64,
                "seed": master_seed + 10000 * band_id + image_index,
                "thermostat_seed": master_seed + 20000 * band_id + image_index,
                "seeded_mdp_sha256": "c" * 64,
                "gmx_sha256": "d" * 64,
                "plumed_kernel_sha256": "e" * 64,
                "input_sha256": {"synthetic-input": "f" * 64},
            }
            records.append({**identity, "status": "started"})
            records.append(
                {
                    **identity,
                    "status": "succeeded",
                    "nof_returned": 1,
                    "result_npz_sha256": "1" * 64,
                    "processed_topology_sha256": "2" * 64,
                    "restrained_tpr_sha256": "3" * 64,
                }
            )
    return records


@pytest.mark.parametrize(
    "command",
    (
        "run-optimization",
        "run-terminal-diagnostics",
        "accept",
        "plot-formal",
    ),
)
def test_every_draft_command_fails_before_any_side_effect(
    monkeypatch: pytest.MonkeyPatch,
    command: str,
):
    reached: List[str] = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        reached.append("side effect")
        raise AssertionError("draft gate was bypassed")

    monkeypatch.setattr(runner, "host_global_ala_lock", forbidden)
    monkeypatch.setattr(runner, "RestrainedMDOracle", forbidden)
    monkeypatch.setattr(runner, "run_irivm", forbidden)
    monkeypatch.setattr(runner, "atomic_write_once_json", forbidden)
    before = runner.DRAFT_REGISTRATION.read_bytes()
    with pytest.raises(runner.FormalCampaignError, match="DRAFT|disabled|sealed"):
        runner.main(["--registration", str(runner.DRAFT_REGISTRATION), command])
    assert reached == []
    assert runner.DRAFT_REGISTRATION.read_bytes() == before
    assert not (CAMPAIGN / "runs").exists()


def test_validate_draft_is_read_only_and_explicitly_non_executable(capsys):
    before = runner.DRAFT_REGISTRATION.read_bytes()
    assert runner.main(["validate-draft"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "valid non-executable draft"
    assert payload["formal_evidence"] is False
    assert payload["new_restrained_md_authorized"] is False
    assert payload["implementation_id"] == core.IMPLEMENTATION_ID
    assert runner.DRAFT_REGISTRATION.read_bytes() == before
    assert not (CAMPAIGN / "runs").exists()


def test_changing_only_draft_status_cannot_resolve_placeholders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    registration = draft_registration()
    registration["status"] = contract.SEALED_STATUS
    path = tmp_path / "status-only.json"
    path.write_text(json.dumps(registration), encoding="utf-8")

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("a side-effect boundary was reached")

    monkeypatch.setattr(runner, "RestrainedMDOracle", forbidden)
    monkeypatch.setattr(runner, "run_irivm", forbidden)
    with pytest.raises(runner.FormalCampaignError):
        runner.main(["--registration", str(path), "run-optimization"])


@pytest.mark.parametrize("jobs", [True, False, 0, 2, 4, None])
def test_future_sealed_registration_accepts_only_integer_jobs_one(jobs: object):
    registration = sealed_registration()
    registration["execution"]["jobs"] = jobs
    with pytest.raises(runner.FormalCampaignError, match="jobs=1"):
        runner.validate_executable_registration(registration)


def test_future_sealed_registration_has_one_exact_host_global_lock():
    registration = sealed_registration()
    candidate, optimization, diagnostic = runner.validate_executable_registration(
        registration
    )
    assert candidate.parameter_id == "a1em4_r0p5_m4"
    assert optimization == (2101, 2102, 2103)
    assert diagnostic == tuple(range(2201, 2211))

    for replacement in (
        "host-local fcntl",
        "per-run fcntl",
        "host-global threading.Lock",
    ):
        mutated = deepcopy(registration)
        mutated["execution"]["lock_kind"] = replacement
        with pytest.raises(runner.FormalCampaignError, match="host-global"):
            runner.validate_executable_registration(mutated)

    mutated = deepcopy(registration)
    mutated["execution"]["lock_path_relative_to_workspace"] = (
        "r6/seed-2101.lock"
    )
    with pytest.raises(runner.FormalCampaignError, match="lock path"):
        runner.validate_executable_registration(mutated)


def test_future_seal_cannot_disable_mutagen_or_omit_hash_tables():
    registration = sealed_registration()
    registration["execution"]["mutagen_preflight_required"] = False
    with pytest.raises(runner.FormalCampaignError, match="Mutagen"):
        runner.validate_executable_registration(registration)

    registration = sealed_registration()
    registration["execution"]["mutagen"][
        "endpoints_connected_and_scanned_required"
    ] = False
    with pytest.raises(runner.FormalCampaignError, match="Mutagen"):
        runner.validate_executable_registration(registration)

    registration = sealed_registration()
    registration["hashes"]["source"] = {}
    with pytest.raises(runner.FormalCampaignError, match="hash table"):
        runner.validate_executable_registration(registration)


def test_host_global_fcntl_lock_excludes_a_second_os_process(tmp_path: Path):
    script = "\n".join(
        (
            "import sys",
            "from pathlib import Path",
            "from run_v17_formal_campaign import host_global_ala_lock",
            "with host_global_ala_lock(workspace=Path(sys.argv[1]), purpose='child'):",
            "    print('LOCKED', flush=True)",
            "    sys.stdin.readline()",
        )
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(SRC) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(tmp_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "LOCKED"
        expected = tmp_path.resolve() / contract.HOST_GLOBAL_LOCK_RELATIVE_TO_WORKSPACE
        assert runner.host_global_lock_path(tmp_path) == expected
        for purpose in ("different-seed", "different-parameter"):
            with pytest.raises(runner.FormalCampaignError, match="host-global"):
                with runner.host_global_ala_lock(
                    workspace=tmp_path, purpose=purpose
                ):
                    raise AssertionError("second process acquired the Ala lock")
    finally:
        if process.poll() is None and process.stdin is not None:
            try:
                process.stdin.write("release\n")
                process.stdin.flush()
            except BrokenPipeError:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
    assert process.returncode == 0
    with runner.host_global_ala_lock(workspace=tmp_path, purpose="after-release"):
        assert expected.is_file()


def test_core_pays_exactly_one_20_image_band_for_movement_and_hold(
    tmp_path: Path,
):
    movement_oracle = CountingSyntheticOracle(noise=1.0e-3)
    movement_directory = tmp_path / "movement"
    result = run_synthetic(movement_directory, movement_oracle, budget=20)
    assert result["nof_mean_force_incremental"] == 20
    assert movement_oracle.band_calls == 1
    assert movement_oracle.call_count == 20
    movement = load_state(movement_directory, 1)
    assert str(movement["state_kind"].item()) == core.STATE_KIND
    assert bool(movement["paid_action_is_movement"])
    assert int(movement["nof_this_complete_band"]) == 20
    assert int(movement["preselection_mean_force_evaluations"]) == 0

    hold_oracle = CountingSyntheticOracle(noise=1.0e6)
    hold_directory = tmp_path / "hold"
    result = run_synthetic(hold_directory, hold_oracle, budget=40)
    assert result["nof_mean_force_incremental"] == 40
    assert hold_oracle.band_calls == 2
    assert hold_oracle.call_count == 40
    source = load_state(hold_directory, 1)
    held = load_state(hold_directory, 2)
    assert str(held["state_kind"].item()) == core.HOLD_STATE_KIND
    assert not bool(held["paid_action_is_movement"])
    assert int(held["nof_this_complete_band"]) == 20
    np.testing.assert_array_equal(
        held["full_path_unwrapped"], source["full_path_unwrapped"]
    )


def test_resume_never_rewrites_a_paid_prefix_or_repeats_its_evaluations(
    tmp_path: Path,
):
    run_directory = tmp_path / "resume-prefix"
    oracle = CountingSyntheticOracle()
    run_synthetic(run_directory, oracle, budget=60, stop=20)
    run_synthetic(run_directory, oracle, budget=60, stop=40)
    assert oracle.band_calls == 2
    prefix = {
        index: core._state_path(run_directory, index).read_bytes()
        for index in range(3)
    }
    checkpoint_at_40 = (run_directory / "controller_checkpoint.npz").read_bytes()

    completed = run_synthetic(run_directory, oracle, budget=60, stop=60)
    assert completed["completed"]
    assert oracle.band_calls == 3
    assert oracle.call_count == 60
    for index, encoded in prefix.items():
        assert core._state_path(run_directory, index).read_bytes() == encoded

    # Simulate a public row one state ahead of the replaceable checkpoint.
    (run_directory / "controller_checkpoint.npz").write_bytes(checkpoint_at_40)
    replayed = run_synthetic(run_directory, oracle, budget=60, stop=60)
    assert replayed["completed"]
    assert oracle.band_calls == 3
    assert oracle.call_count == 60
    for index, encoded in prefix.items():
        assert core._state_path(run_directory, index).read_bytes() == encoded


def test_short_and_long_registered_budgets_have_identical_common_state_prefixes(
    tmp_path: Path,
):
    short = tmp_path / "short"
    long = tmp_path / "long"
    run_synthetic(short, CountingSyntheticOracle(), budget=40)
    run_synthetic(long, CountingSyntheticOracle(), budget=60, stop=40)
    for index in range(3):
        assert_npz_records_equal(load_state(short, index), load_state(long, index))


def test_zero_nof_refresh_and_terminal_never_enter_the_public_paid_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    run_directory = tmp_path / "zero-nof"
    oracle = CountingSyntheticOracle()
    run_synthetic(run_directory, oracle, budget=40, stop=20)
    assert oracle.band_calls == 1
    before = {
        path.name: path.read_bytes()
        for path in (run_directory / "states").glob("state_*.npz")
    }
    original_build = core._build_candidate
    original_fresh = core._frozen_outer_is_current_paid_frame

    def exhausted_after_state_zero(
        label, direction, controller, statistics, radii, controls
    ):
        if int(controller["state_index"]) == 0:
            return original_build(
                label=label,
                direction=direction,
                controller=controller,
                statistics=statistics,
                radii=radii,
                controls=controls,
            )
        return core._invalid_candidate(
            label,
            "post_reparameterization_trust_globalization_exhausted",
            trust_globalization_exhausted=True,
        )

    def stale_after_state_zero(controller):
        if int(controller["state_index"]) == 0:
            return original_fresh(controller)
        return False

    monkeypatch.setattr(core, "_build_candidate", exhausted_after_state_zero)
    monkeypatch.setattr(
        core, "_frozen_outer_is_current_paid_frame", stale_after_state_zero
    )
    stopped = run_synthetic(run_directory, oracle, budget=40, stop=40)
    assert stopped["phase"] == "TERMINAL"
    assert stopped["nof_mean_force_incremental"] == 20
    assert stopped["zero_nof_trust_refresh_event"]["mean_force_evaluations"] == 0
    assert stopped["same_index_zero_nof_refresh_checkpoint_published"] is False
    assert oracle.band_calls == 1
    assert oracle.call_count == 20
    after = {
        path.name: path.read_bytes()
        for path in (run_directory / "states").glob("state_*.npz")
    }
    assert after == before

    resumed = run_synthetic(run_directory, oracle, budget=40, stop=40)
    assert resumed["terminal_rederived_on_resume"] is True
    assert oracle.band_calls == 1
    assert {
        path.name: path.read_bytes()
        for path in (run_directory / "states").glob("state_*.npz")
    } == before


def test_complete_attempt_audit_counts_every_paid_kind_as_20_nof():
    trajectory = audit_trajectory()
    records = complete_attempt_records(trajectory)
    result = runner.audit_md_attempt_records(
        records,
        trajectory,
        parameter_id="a1em4_r0p5_m4",
        master_seed=2101,
    )
    assert result["paid_band_count"] == 100
    assert result["unique_logical_evaluation_count"] == 2000
    assert result["realized_mean_force_nof"] == 2000
    assert result["failed_before_mean_force_attempt_count"] == 0
    assert result["hidden_logical_evaluation_count"] == 0
    assert result["pass"] is True


def test_failed_before_mean_force_retry_is_unique_and_costs_zero():
    trajectory = audit_trajectory()
    records = complete_attempt_records(trajectory)
    successful = records[:2]
    identity = {
        key: value
        for key, value in successful[0].items()
        if key not in {"attempt_id", "status", "nof_returned"}
    }
    failed_id = "b000001-i00-t1-p0"
    records.extend(
        [
            {**identity, "attempt_id": failed_id, "status": "started"},
            {
                **identity,
                "attempt_id": failed_id,
                "status": "failed_before_mean_force",
                "nof_returned": 0,
            },
        ]
    )
    result = runner.audit_md_attempt_records(
        records,
        trajectory,
        parameter_id="a1em4_r0p5_m4",
        master_seed=2101,
    )
    assert result["realized_mean_force_nof"] == 2000
    assert result["failed_before_mean_force_attempt_count"] == 1


def test_retry_must_keep_the_complete_logical_identity_and_registered_limit():
    trajectory = audit_trajectory()
    base_records = complete_attempt_records(trajectory)
    base_start = base_records[0]

    identity_changed = deepcopy(base_records)
    changed_start = deepcopy(base_start)
    changed_start["attempt_id"] = "b000001-i00-t1-p0"
    changed_start["seed"] = int(changed_start["seed"]) + 1
    changed_terminal = {
        **changed_start,
        "status": "failed_before_mean_force",
        "nof_returned": 0,
    }
    identity_changed.extend([changed_start, changed_terminal])
    with pytest.raises(runner.FormalCampaignError):
        runner.audit_md_attempt_records(
            identity_changed,
            trajectory,
            parameter_id="a1em4_r0p5_m4",
            master_seed=2101,
        )

    too_many = deepcopy(base_records)
    for attempt_number in (1, 2, 3):
        failed_start = deepcopy(base_start)
        failed_start["attempt_id"] = (
            f"b000001-i00-t{attempt_number}-p0"
        )
        failed_terminal = {
            **failed_start,
            "status": "failed_before_mean_force",
            "nof_returned": 0,
        }
        too_many.extend([failed_start, failed_terminal])
    # The successful t0 attempt plus three failed attempts exceeds the
    # registered maximum of three attempts for this logical evaluation.
    with pytest.raises(runner.FormalCampaignError):
        runner.audit_md_attempt_records(
            too_many,
            trajectory,
            parameter_id="a1em4_r0p5_m4",
            master_seed=2101,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "duplicate_event",
        "second_success_same_logical",
        "failed_attempt_charged_one",
        "ineligible_mean_force",
        "hidden_band",
    ),
)
def test_attempt_audit_rejects_duplicates_bad_charges_and_hidden_evaluations(
    mutation: str,
):
    trajectory = audit_trajectory()
    records = complete_attempt_records(trajectory)
    if mutation == "duplicate_event":
        records.append(deepcopy(records[0]))
    elif mutation == "second_success_same_logical":
        duplicate = deepcopy(records[0])
        duplicate["attempt_id"] = "b000001-i00-t1-p0"
        records.extend(
            [
                duplicate,
                {
                    **duplicate,
                    "status": "succeeded",
                    "nof_returned": 1,
                },
            ]
        )
    elif mutation == "failed_attempt_charged_one":
        failed = deepcopy(records[1])
        failed["attempt_id"] = "b000001-i00-t1-p0"
        failed["status"] = "failed_before_mean_force"
        failed["nof_returned"] = 1
        started = deepcopy(failed)
        started["status"] = "started"
        started.pop("nof_returned")
        records.extend([started, failed])
    elif mutation == "ineligible_mean_force":
        records[1]["status"] = "mean_force_returned_but_ineligible"
    elif mutation == "hidden_band":
        hidden = deepcopy(records[0])
        hidden["attempt_id"] = "b000101-i00-t0-p0"
        hidden["band_id"] = 101
        records.extend(
            [
                hidden,
                {**hidden, "status": "succeeded", "nof_returned": 1},
            ]
        )
    with pytest.raises(runner.FormalCampaignError):
        runner.audit_md_attempt_records(
            records,
            trajectory,
            parameter_id="a1em4_r0p5_m4",
            master_seed=2101,
        )


def test_atomic_attempt_records_and_jsonl_are_one_unique_ledger(tmp_path: Path):
    run_directory = tmp_path / "attempt-ledger"
    atomic = run_directory / "md_attempt_records"
    atomic.mkdir(parents=True)
    identity = {
        "attempt_id": "b000001-i00-t0-p0",
        "parameter_id": "a1em4_r0p5_m4",
        "master_seed": 2101,
        "band_id": 1,
        "image_index": 0,
        "center": [0.0, 0.0],
    }
    start = {**identity, "status": "started"}
    terminal = {**identity, "status": "succeeded", "nof_returned": 1}
    for record in (start, terminal):
        path = atomic / f"{record['attempt_id']}.{record['status']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
    # A duplicate JSONL event is not allowed to become a second ledger fact.
    ledger = run_directory / "md_attempts.jsonl"
    ledger.write_text(
        "\n".join((json.dumps(start), json.dumps(start), json.dumps(terminal)))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(runner.FormalCampaignError):
        runner.load_atomic_attempt_records(run_directory)


def _evaluate_band_call_count(path: Path) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "evaluate_band"
        for node in ast.walk(tree)
    )


def test_formal_call_graph_has_one_paid_band_boundary_and_no_consumer_query():
    assert _evaluate_band_call_count(CAMPAIGN / "src/ala2d_irivm_v17.py") == 1
    for relative in (
        "src/formal_contract.py",
        "src/run_v17_formal_campaign.py",
        "src/campaign_acceptance.py",
        "src/plot_registered_figures.py",
    ):
        assert _evaluate_band_call_count(CAMPAIGN / relative) == 0, relative
    runner_source = (CAMPAIGN / "src/run_v17_formal_campaign.py").read_text(
        encoding="utf-8"
    )
    assert "campaign_analysis" not in runner_source
