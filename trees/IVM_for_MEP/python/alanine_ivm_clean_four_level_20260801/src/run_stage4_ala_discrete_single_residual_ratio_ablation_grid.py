#!/usr/bin/env python3
"""Preregister, then separately launch the Ala residual-ratio ablation.

Preregistration freezes two new ratios (0.90 and 0.50), hashes the formal
barrier-blind alpha-refinement decision that selected alpha=12.5 as a
non-winner ablation center, and hash-locks the already completed r=0.80
candidate and same-alpha legacy control.  Launching requires that immutable
preregistration and runs only the two new ratio members.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import run_stage4_ala_discrete_single_residual_ratio_ablation as member


PROTOCOL_ID = "stage4_ala_discrete_single_residual_ratio_ablation_v1"
FORMAL_ANALYSIS_ID = (
    "stage4_ala_discrete_single_alpha_refinement_iperp_ctp_v1"
)
FORMAL_REFINEMENT_ANALYSIS = (
    "reports/"
    "stage4_ala_discrete_single_alpha_refinement_matched_v1_N254_r1_"
    "1787252714040428000_iperp_ctp_readonly_v1_1787253163640931000.json"
)
FORMAL_REFINEMENT_ANALYSIS_SHA256 = (
    "8ff9d9824eb2d3719d28f73f33b92b62e115ff9f7449bf60be054298d7ba5212"
)
COMMON_INITIAL_PATH = "inputs/common_initial_path_equidistant.npz"
COMMON_INITIAL_FILE_SHA256 = (
    "4a08066e1f0a89ad1d0ee6ecc442d7255d511217576d1ff65e9cb5e2164d891b"
)
COMMON_INITIAL_ARRAY_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)
REFERENCE_RESULT = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "ivm_stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "r08c10_a12p5_N254_discrete_b040_r1_central_state_ivm_"
    "cfgfaf71a0dca2c_budget254.npz"
)
REFERENCE_AUDIT = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/audits/"
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "r08c10_a12p5_N254_discrete_b040_r1.json"
)
REFERENCE_COMMAND = (
    "logs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "r08c10_a12p5_N254_discrete_b040_r1/command.json"
)
LEGACY_RESULT = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "ivm_stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "legacy_stop_a12p5_N254_legacy_r1_central_state_ivm_"
    "cfg706082ab7a04_budget254.npz"
)
LEGACY_AUDIT = (
    "runs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/audits/"
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "legacy_stop_a12p5_N254_legacy_r1.json"
)
LEGACY_COMMAND = (
    "logs/stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1/members/"
    "stage4_ala_discrete_single_fixedalpha_alpha_refinement_v1_"
    "legacy_stop_a12p5_N254_legacy_r1/command.json"
)
EXPECTED_REUSE_HASHES = {
    "reference_result": "15df24a92b0a4a73e972be0574b5979d2e4eb7e7b980bc76cd58d474f9d8e835",
    "reference_audit": "1cf4ec9619b41f7f5d3dc39e007e0d7fc95323f78d6900befa64f365f3b5bc5f",
    "reference_command": "628c7ca2d276bb48e46bbe903a1ee8ee9e5416cf9c032f69ecf1a82ffcfdf0ac",
    "legacy_result": "236c9e8f623fa297619d403163a33419a09f3069ef3d151cddd9dafb9cd11c70",
    "legacy_audit": "7783ba8a42491ce8d23f7b321ee310a7fd47b48d1df23e4e42d24b30a211d2db",
    "legacy_command": "db1c35940aea1e9d94a35ff32eab186c7af8e38d3453935b34ea833810783131",
}
REPORTING_ANCHORS = (
    1.0,
    0.1,
    0.05,
    0.02,
    0.01,
    0.005,
    0.002,
    0.001,
    5e-4,
    2e-4,
    1.5e-4,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _hashed(path: Path, expected: str | None = None) -> dict[str, str]:
    resolved = path.resolve()
    actual = file_sha256(resolved)
    if expected is not None and actual != expected:
        raise RuntimeError(f"Hash drift: {resolved}")
    return {"path": str(resolved), "sha256": actual}


def _source_manifest(root: Path) -> dict[str, dict[str, str]]:
    sources = {
        "launcher": Path(__file__).resolve(),
        "member_wrapper": (
            root / "src/run_stage4_ala_discrete_single_residual_ratio_ablation.py"
        ),
        "frozen_alpha_refinement_wrapper": (
            root / "src/run_stage4_ala_discrete_single_alpha_refinement.py"
        ),
        "frozen_single_wrapper": (
            root / "src/run_stage4_ala_discrete_single_fixedalpha.py"
        ),
        "frozen_single_implementation": (
            root / "src/run_stage4_ala_dsel_single_fixedalpha.py"
        ),
        "legacy_command_helper": root / "src/run_stage4_uniform_optimized.py",
        "core_runner": (
            root / "src/run_ivm_v9_spring_projected_production_20260730.py"
        ),
        "ala_core": root / "src/ala_core.py",
        "frozen_discrete_b040": root / "src/frozen_discrete_b040.py",
        "formal_refinement_analyzer": (
            root / "src/analyze_stage4_ala_discrete_single_alpha_refinement_iperp_ctp.py"
        ),
    }
    return {name: _hashed(path) for name, path in sources.items()}


def _common_initial_record(root: Path) -> dict[str, str]:
    path = (root / COMMON_INITIAL_PATH).resolve()
    record = _hashed(path, COMMON_INITIAL_FILE_SHA256)
    with np.load(path, allow_pickle=False) as saved:
        initial = np.ascontiguousarray(
            np.asarray(saved["initial_path"], dtype=np.float64)
        )
    array_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    if array_hash != COMMON_INITIAL_ARRAY_SHA256:
        raise RuntimeError("Canonical initial-path array hash drifted")
    return {**record, "initial_path_array_sha256": array_hash}


def _verify_formal_decision(root: Path) -> tuple[dict[str, str], dict[str, object]]:
    path = (root / FORMAL_REFINEMENT_ANALYSIS).resolve()
    record = _hashed(path, FORMAL_REFINEMENT_ANALYSIS_SHA256)
    payload = _read_json(path)
    if payload.get("analysis_id") != FORMAL_ANALYSIS_ID:
        raise RuntimeError("Unexpected formal refinement analysis")
    if payload.get("read_only_formal_input_audit") is not True:
        raise RuntimeError("Formal refinement analysis is not read-only")
    if payload.get("barrier_data_accessed") is not False:
        raise RuntimeError("Formal refinement analysis accessed barrier data")
    if payload.get("barrier_computation_or_densification_performed") is not False:
        raise RuntimeError("Formal refinement analysis computed a barrier")
    if payload.get("barrier_remains_embargoed") is not True:
        raise RuntimeError("Formal refinement analysis released the barrier")
    if payload.get("barrier_unembargoed") is not False:
        raise RuntimeError("Formal refinement analysis unembargoed the barrier")
    if payload.get("joint_full_curve_dominator_alphas") != []:
        raise RuntimeError("Stop ablation requires no alpha joint winner")
    if float(payload.get("stop_ablation_center_alpha")) != member.CANONICAL_ALPHA:
        raise RuntimeError("Formal analysis did not select alpha=12.5 center")
    if payload.get("stop_ablation_center_is_not_a_winner") is not True:
        raise RuntimeError("Ablation center was incorrectly called a winner")
    analyzer = Path(str(payload["analysis_script"])).resolve()
    if file_sha256(analyzer) != str(payload["analysis_script_sha256"]):
        raise RuntimeError("Formal refinement analyzer source drifted")
    official = payload.get("official_stage4_v4")
    if not isinstance(official, dict):
        raise RuntimeError("Formal analysis has no official comparator")
    archive = Path(str(official["archive"])).resolve()
    if file_sha256(archive) != str(official["archive_sha256"]):
        raise RuntimeError("Official comparator hash drifted")
    return record, payload


def _reuse_records(root: Path) -> dict[str, dict[str, str]]:
    paths = {
        "reference_result": root / REFERENCE_RESULT,
        "reference_audit": root / REFERENCE_AUDIT,
        "reference_command": root / REFERENCE_COMMAND,
        "legacy_result": root / LEGACY_RESULT,
        "legacy_audit": root / LEGACY_AUDIT,
        "legacy_command": root / LEGACY_COMMAND,
    }
    records = {
        name: _hashed(path, EXPECTED_REUSE_HASHES[name])
        for name, path in paths.items()
    }
    reference_command = json.loads(paths["reference_command"].read_text())
    if reference_command != member.canonical_reference_command(root):
        raise RuntimeError("Frozen r=0.8 command no longer matches its builder")
    reference_audit = _read_json(paths["reference_audit"])
    legacy_audit = _read_json(paths["legacy_audit"])
    if reference_audit.get("barrier_fields_accessed") is not False:
        raise RuntimeError("Reused r=0.8 audit did not preserve embargo")
    if legacy_audit.get("barrier_fields_accessed") is not False:
        raise RuntimeError("Reused legacy audit did not preserve embargo")
    if float(reference_audit["alpha"]) != member.CANONICAL_ALPHA:
        raise RuntimeError("Reused candidate is not alpha=12.5")
    if float(legacy_audit["alpha"]) != member.CANONICAL_ALPHA:
        raise RuntimeError("Reused control is not alpha=12.5")
    return records


def _mechanism_evidence(root: Path) -> dict[str, object]:
    result = (root / REFERENCE_RESULT).resolve()
    with np.load(result, allow_pickle=False) as saved:
        selections = np.asarray(saved["discrete_audit_selection"], dtype="U16")
        histories = np.asarray(saved["discrete_audit_history_count"], dtype=int)
        residuals = np.asarray(
            saved["discrete_audit_global_relative_residual"], dtype=float
        )
        return {
            "source": str(result),
            "barrier_fields_accessed": False,
            "frozen_subproblem_solve_count": int(residuals.size),
            "joint_step_count": int(selections.size),
            "h0_selection_count": int(np.count_nonzero(selections == "H0")),
            "lbfgs_selection_count": int(np.count_nonzero(selections == "LBFGS")),
            "maximum_history_count": int(np.max(histories)),
            "motivation": (
                "r=0.90 probes earlier outer reparameterization; r=0.50 "
                "probes deeper inner solves and mature m7"
            ),
        }


def _registered_commands(root: Path) -> dict[str, str]:
    return {
        member.ratio_tag(ratio): member.compact_command_sha256(
            member.single_run_command(root, ratio)[0]
        )
        for ratio in member.NEW_RATIOS
    }


def _assert_no_outputs(root: Path) -> None:
    for ratio in member.NEW_RATIOS:
        paths = member.member_paths(root, ratio)
        result_dir = paths["result_dir"]
        audit = paths["audit"]
        member_log = paths["member_log"]
        launcher_log = paths["launcher_log"]
        label = str(paths["label"])
        assert isinstance(result_dir, Path)
        assert isinstance(audit, Path)
        assert isinstance(member_log, Path)
        assert isinstance(launcher_log, Path)
        existing = sorted(
            result_dir.glob(
                f"ivm_{label}_central_state_ivm_cfg*_budget{member.CANONICAL_BUDGET}.npz"
            )
        )
        if existing or audit.exists() or member_log.exists() or launcher_log.exists():
            target = existing[0] if existing else audit
            raise FileExistsError(f"Refusing pre-existing ratio data: {target}")


def build_preregistration(root: Path, planned_report: Path) -> dict[str, object]:
    _assert_no_outputs(root)
    formal_record, formal = _verify_formal_decision(root)
    reuse = _reuse_records(root)
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "planned_report": str(planned_report.resolve()),
        "one_factor_ablation": "discrete_residual_ratio",
        "fixed_alpha": member.CANONICAL_ALPHA,
        "new_residual_ratios": list(member.NEW_RATIOS),
        "reused_residual_ratio": member.REFERENCE_RATIO,
        "display_grid": [0.9, 0.8, 0.5],
        "new_member_tags": [member.ratio_tag(value) for value in member.NEW_RATIOS],
        "budget_per_image": member.CANONICAL_BUDGET,
        "replicate": member.CANONICAL_REPLICATE,
        "evaluated_step_cap": member.EVALUATED_STEP_CAP,
        "lbfgs_history": member.LBFGS_HISTORY,
        "reparameterization_policy": "state_curve/per_image from first state",
        "core_stop_mode_selector": member.CORE_STOP_MODE_SELECTOR,
        "core_stop_mode_selector_note": (
            "legacy API name r08c10; actual ratio is the separately registered "
            "--discrete-residual-ratio value"
        ),
        "stop_rule_template": "R/R0 <= registered ratio OR evaluated steps == 10",
        "scientific_command_contract": (
            "relative to the frozen alpha12.5 r=0.8 command, only "
            "--discrete-residual-ratio changes; output-dir and run-label are "
            "routing-only changes"
        ),
        "reference_ratio_candidate_reused_not_rerun": True,
        "same_alpha_legacy_control_reused_not_rerun": True,
        "single_run_from_canonical_common_path": True,
        "eligible_record_definition": "initial plus reparam_committed",
        "mesh_gate": {
            "segment_length_cv_max": member.MESH_CV_MAX,
            "segment_length_ratio_max": member.MESH_RATIO_MAX,
        },
        "cost_to_precision_definition": (
            "minimum actual optimization NF at I_perp <= threshold; no interpolation"
        ),
        "reporting_anchors": list(REPORTING_ANCHORS),
        "formal_selection_rule": (
            "full-curve Iperp CTP on complete mesh-eligible states versus the "
            "same-alpha legacy control and separately official Stage-IV v4"
        ),
        "primary_success_criterion": (
            "a ratio candidate must weakly dominate every reachable threshold "
            "and be strict at least once versus both comparators; its tightest "
            "eligible I_perp must also be no larger than each comparator's "
            "tightest eligible I_perp"
        ),
        "multiple_primary_winner_tiebreak": [
            "lexicographically smallest candidate-minus-official NF vector at official thresholds ordered tight-to-loose",
            "lexicographically smallest candidate-minus-legacy NF vector ordered tight-to-loose",
            "smallest absolute log-distance from reused r=0.8",
        ],
        "no_primary_dominator_rule": {
            "winner_claim_allowed": False,
            "ratio_lock_allowed": False,
            "post_hoc_grid_expansion_allowed": False,
            "barrier_unembargo_allowed": False,
        },
        "terminal_or_raw_iperp_ranking_allowed": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "sparse_barrier_role": "embargoed_not_read_not_reported_not_ranked",
        "densified_energy_barrier_role": "embargoed_until_primary_ratio_lock",
        "formal_refinement_analysis": formal_record,
        "canonical_common_initial_path": _common_initial_record(root),
        "formal_refinement_decision": {
            "joint_full_curve_dominator_alphas": [],
            "stop_ablation_center_alpha": formal["stop_ablation_center_alpha"],
            "stop_ablation_center_is_not_a_winner": True,
        },
        "reused_inputs": reuse,
        "reference_mechanism_evidence": _mechanism_evidence(root),
        "registered_command_hashes": _registered_commands(root),
        "run_source_manifest": _source_manifest(root),
        "independent_member_paths_and_logs": True,
        "overwrite_or_force_allowed": False,
    }


def _verify_manifest(manifest: object, name: str) -> None:
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(f"Invalid {name}")
    for record in manifest.values():
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid {name} record")
        if file_sha256(Path(str(record["path"]))) != str(record["sha256"]):
            raise RuntimeError(f"Hash drift in {name}")


def validate_preregistration(protocol: dict[str, object], root: Path) -> None:
    if protocol.get("status") != "preregistered":
        raise RuntimeError("Input is not a preregistration")
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Unexpected ratio-ablation protocol")
    if tuple(float(value) for value in protocol["new_residual_ratios"]) != (
        member.NEW_RATIOS
    ):
        raise RuntimeError("Preregistered ratio grid drifted")
    if float(protocol["fixed_alpha"]) != member.CANONICAL_ALPHA:
        raise RuntimeError("Preregistered alpha drifted")
    if protocol.get("barrier_fields_accessed_by_wrapper_or_launcher") is not False:
        raise RuntimeError("Preregistration broke the barrier embargo")
    planned = Path(str(protocol["planned_report"])).resolve()
    if planned.parent != (root / "reports").resolve() or not planned.name.startswith(
        f"{PROTOCOL_ID}_N254_r1_"
    ):
        raise RuntimeError("Planned report uses the wrong namespace")
    formal = protocol["formal_refinement_analysis"]
    if not isinstance(formal, dict):
        raise RuntimeError("Missing formal decision hash")
    _hashed(Path(str(formal["path"])), str(formal["sha256"]))
    _verify_formal_decision(root)
    common = protocol.get("canonical_common_initial_path")
    if not isinstance(common, dict):
        raise RuntimeError("Missing canonical common initial path")
    if common != _common_initial_record(root):
        raise RuntimeError("Canonical common initial path manifest drifted")
    _verify_manifest(protocol["reused_inputs"], "reused inputs")
    _verify_manifest(protocol["run_source_manifest"], "run sources")
    if protocol["registered_command_hashes"] != _registered_commands(root):
        raise RuntimeError("Registered ratio commands changed")
    _assert_no_outputs(root)


def _run_process(
    command: list[str], root: Path, stdout_path: Path, stderr_path: Path
) -> tuple[int, float, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "OPENMM_CPU_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=root,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed.returncode, time.time() - started, completed.stderr


def _run_cell(root: Path, ratio: float) -> dict[str, object]:
    paths = member.member_paths(root, ratio)
    launcher_log = paths["launcher_log"]
    audit_path = paths["audit"]
    assert isinstance(launcher_log, Path)
    assert isinstance(audit_path, Path)
    launcher_log.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(root / "src/run_stage4_ala_discrete_single_residual_ratio_ablation.py"),
        "--root",
        str(root),
        "--residual-ratio",
        f"{ratio:g}",
    ]
    (launcher_log / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    returncode, wall_seconds, stderr = _run_process(
        command,
        root,
        launcher_log / "stdout.txt",
        launcher_log / "stderr.txt",
    )
    row: dict[str, object] = {
        "residual_ratio": ratio,
        "ratio_tag": member.ratio_tag(ratio),
        "returncode": returncode,
        "wall_seconds": wall_seconds,
    }
    if returncode:
        row["error"] = stderr[-4000:]
        return row
    audit = _read_json(audit_path)
    if audit.get("barrier_fields_accessed") is not False:
        raise RuntimeError("Member audit broke the barrier embargo")
    if float(audit["residual_ratio"]) != ratio:
        raise RuntimeError("Member audit ratio mismatch")
    result = Path(str(audit["result"])).resolve()
    if file_sha256(result) != str(audit["result_sha256"]):
        raise RuntimeError("Member result hash mismatch")
    row.update(
        {
            "member_audit": str(audit_path),
            "member_audit_sha256": file_sha256(audit_path),
            "result": str(result),
            "result_sha256": file_sha256(result),
            "final_optimization_NF": int(audit["final_optimization_NF"]),
            "mesh_eligible_complete_state_count": int(
                audit["mesh_eligible_complete_state_count"]
            ),
        }
    )
    return row


def preregister(root: Path) -> Path:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stem = f"{PROTOCOL_ID}_N254_r1_{time.time_ns()}"
    preregistration = reports / f"{stem}_preregistered.json"
    planned_report = reports / f"{stem}.json"
    _write_json_exclusive(
        preregistration, build_preregistration(root, planned_report)
    )
    return preregistration


def launch(root: Path, preregistration: Path, jobs: int) -> Path:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    preregistration = preregistration.expanduser().resolve()
    protocol = _read_json(preregistration)
    validate_preregistration(protocol, root)
    final_report = Path(str(protocol["planned_report"])).resolve()
    if final_report.exists():
        raise FileExistsError(f"Refusing to overwrite {final_report}")
    rows: list[dict[str, object]] = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(_run_cell, root, ratio): ratio
            for ratio in member.NEW_RATIOS
        }
        for future in concurrent.futures.as_completed(futures):
            ratio = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "residual_ratio": ratio,
                    "ratio_tag": member.ratio_tag(ratio),
                    "returncode": -1,
                    "error": repr(exc),
                }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    order = {value: index for index, value in enumerate(member.NEW_RATIOS)}
    rows.sort(key=lambda row: order[float(row["residual_ratio"])])
    complete = all(int(row.get("returncode", -1)) == 0 for row in rows)
    report = {
        **protocol,
        "status": "complete" if complete else "incomplete",
        "preregistration": str(preregistration),
        "preregistration_sha256": file_sha256(preregistration),
        "execution_jobs_operational_only": jobs,
        "started_unix": started,
        "finished_unix": time.time(),
        "formal_ranking_deferred_to_separate_readonly_iperp_ctp": True,
        "results": rows,
    }
    _write_json_exclusive(final_report, report)
    if not complete:
        raise SystemExit(1)
    return final_report


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preregister-only", action="store_true")
    action.add_argument("--preregistration", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main() -> None:
    args = argument_parser().parse_args()
    root = args.root.expanduser().resolve()
    if args.preregister_only:
        if args.jobs != 1:
            raise ValueError("--jobs applies only to execution")
        print(preregister(root))
        return
    print(launch(root, args.preregistration, args.jobs))


if __name__ == "__main__":
    main()
