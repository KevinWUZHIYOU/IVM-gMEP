#!/usr/bin/env python3
"""Preregister, then separately launch the Ala evaluated-step-cap ablation."""

from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

import run_stage4_ala_discrete_single_max_steps_cap_ablation as member


PROTOCOL_ID = "stage4_ala_discrete_single_max_evaluated_steps_cap_ablation_v1"
FORMAL_RATIO_ANALYSIS_ID = "stage4_ala_discrete_single_residual_ratio_iperp_ctp_v1"
FORMAL_RATIO_ANALYSIS = (
    "reports/stage4_ala_discrete_single_residual_ratio_ablation_v1_N254_r1_"
    "1787253500284697000_iperp_ctp_readonly_v1_1787253931892862000.json"
)
FORMAL_RATIO_ANALYSIS_SHA256 = (
    "49daff701dc66a7f17bcd8ae8f766f805685679a13f54ef633af604a04706373"
)
RATIO_PREREGISTRATION = (
    "reports/stage4_ala_discrete_single_residual_ratio_ablation_v1_N254_r1_"
    "1787253500284697000_preregistered.json"
)
RATIO_PREREGISTRATION_SHA256 = (
    "eba63e07899480b5a63af31a6b80c0262e97902a241a9b6a9bbc87f4e8c099d4"
)
RATIO_REPORT = (
    "reports/stage4_ala_discrete_single_residual_ratio_ablation_v1_N254_r1_"
    "1787253500284697000.json"
)
RATIO_REPORT_SHA256 = (
    "f4e1f3e9f32704e80d51dd1b5ebd56b40b5a56d2bde0350ef21a908e84004d2c"
)
COMMON_INITIAL_PATH = "inputs/common_initial_path_equidistant.npz"
COMMON_INITIAL_FILE_SHA256 = (
    "4a08066e1f0a89ad1d0ee6ecc442d7255d511217576d1ff65e9cb5e2164d891b"
)
COMMON_INITIAL_ARRAY_SHA256 = (
    "cdc9330623efd3e125dfe40ecf5f4440fa90c32ef5ba8db840824b4280c69e6e"
)
REFERENCE_STEP_DISTRIBUTION = {1: 4, 3: 12, 4: 3, 6: 1}
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
    path = path.resolve()
    actual = file_sha256(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f"Hash drift: {path}")
    return {"path": str(path), "sha256": actual}


def _verify_manifest(manifest: object, name: str) -> None:
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(f"Invalid {name}")
    for key, record in manifest.items():
        if not isinstance(record, dict):
            raise RuntimeError(f"Invalid {name}.{key}")
        _hashed(Path(str(record["path"])), str(record["sha256"]))


def _common_initial_record(root: Path) -> dict[str, str]:
    path = (root / COMMON_INITIAL_PATH).resolve()
    record = _hashed(path, COMMON_INITIAL_FILE_SHA256)
    with np.load(path, allow_pickle=False) as saved:
        initial = np.ascontiguousarray(np.asarray(saved["initial_path"], dtype=np.float64))
    array_hash = hashlib.sha256(initial.tobytes()).hexdigest()
    if array_hash != COMMON_INITIAL_ARRAY_SHA256:
        raise RuntimeError("Canonical initial-path array hash drifted")
    return {**record, "initial_path_array_sha256": array_hash}


def _formal_ratio_chain(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    prereg_path = (root / RATIO_PREREGISTRATION).resolve()
    report_path = (root / RATIO_REPORT).resolve()
    analysis_path = (root / FORMAL_RATIO_ANALYSIS).resolve()
    records = {
        "ratio_preregistration": _hashed(prereg_path, RATIO_PREREGISTRATION_SHA256),
        "ratio_report": _hashed(report_path, RATIO_REPORT_SHA256),
        "formal_ratio_analyzer_output": _hashed(
            analysis_path, FORMAL_RATIO_ANALYSIS_SHA256
        ),
    }
    prereg = _read_json(prereg_path)
    report = _read_json(report_path)
    analysis = _read_json(analysis_path)
    if prereg.get("status") != "preregistered":
        raise RuntimeError("Ratio preregistration is invalid")
    if report.get("status") != "complete":
        raise RuntimeError("Ratio report is incomplete")
    if Path(str(report["preregistration"])).resolve() != prereg_path or str(
        report["preregistration_sha256"]
    ) != file_sha256(prereg_path):
        raise RuntimeError("Ratio report/preregistration mismatch")
    if analysis.get("analysis_id") != FORMAL_RATIO_ANALYSIS_ID:
        raise RuntimeError("Unexpected formal ratio analyzer")
    if Path(str(analysis["ratio_preregistration"])).resolve() != prereg_path:
        raise RuntimeError("Formal ratio analysis points to another preregistration")
    if Path(str(analysis["ratio_report"])).resolve() != report_path:
        raise RuntimeError("Formal ratio analysis points to another report")
    if str(analysis["ratio_preregistration_sha256"]) != file_sha256(prereg_path) or str(
        analysis["ratio_report_sha256"]
    ) != file_sha256(report_path):
        raise RuntimeError("Formal ratio input hash mismatch")
    if analysis.get("joint_full_curve_dominator_ratios") != []:
        raise RuntimeError("Cap ablation requires no ratio primary dominator")
    if analysis.get("ratio_lock_allowed") is not False or analysis.get(
        "winner_claim_made"
    ) is not False:
        raise RuntimeError("Formal ratio analysis incorrectly locked a winner")
    if analysis.get("barrier_data_accessed") is not False or analysis.get(
        "barrier_computation_or_densification_performed"
    ) is not False or analysis.get("barrier_remains_embargoed") is not True:
        raise RuntimeError("Formal ratio analysis broke the barrier embargo")
    analyzer = Path(str(analysis["analysis_script"])).resolve()
    if file_sha256(analyzer) != str(analysis["analysis_script_sha256"]):
        raise RuntimeError("Formal ratio analyzer source drifted")
    official = analysis["official_stage4_v4"]
    official_path = Path(str(official["archive"])).resolve()
    if file_sha256(official_path) != str(official["archive_sha256"]):
        raise RuntimeError("Official comparator hash drifted")
    records["official_stage4_v4"] = _hashed(
        official_path, str(official["archive_sha256"])
    )
    return records, {"prereg": prereg, "report": report, "analysis": analysis}


def _reuse_records(root: Path, ratio_prereg: dict[str, object]) -> dict[str, dict[str, str]]:
    reused = ratio_prereg.get("reused_inputs")
    if not isinstance(reused, dict) or set(reused) != {
        "reference_result",
        "reference_audit",
        "reference_command",
        "legacy_result",
        "legacy_audit",
        "legacy_command",
    }:
        raise RuntimeError("Ratio preregistration has invalid reused inputs")
    _verify_manifest(reused, "ratio reused inputs")
    reference_command = json.loads(
        Path(str(reused["reference_command"]["path"])).read_text(encoding="utf-8")
    )
    legacy_command = json.loads(
        Path(str(reused["legacy_command"]["path"])).read_text(encoding="utf-8")
    )
    if reference_command != member.canonical_reference_command(root):
        raise RuntimeError("Frozen cap=10 command differs from its builder")
    member.frozen_refinement.assert_matched_pair_commands(
        reference_command, legacy_command, member.CANONICAL_ALPHA
    )
    reference_audit = _read_json(Path(str(reused["reference_audit"]["path"])))
    legacy_audit = _read_json(Path(str(reused["legacy_audit"]["path"])))
    for audit, role in ((reference_audit, "candidate"), (legacy_audit, "matched_control")):
        if audit.get("barrier_fields_accessed") is not False:
            raise RuntimeError("Reused audit broke the barrier embargo")
        if float(audit["alpha"]) != member.CANONICAL_ALPHA or audit["member_role"] != role:
            raise RuntimeError("Reused audit identity mismatch")
    return {name: dict(record) for name, record in reused.items()}


def _reference_mechanism_evidence(
    reused: dict[str, dict[str, str]]
) -> dict[str, object]:
    result = Path(reused["reference_result"]["path"])
    with np.load(result, allow_pickle=False) as saved:
        outer = np.asarray(saved["discrete_audit_outer"], dtype=int)
        steps = np.asarray(saved["discrete_audit_step"], dtype=int)
        selections = np.asarray(saved["discrete_audit_selection"], dtype="U16")
        histories = np.asarray(saved["discrete_audit_history_count"], dtype=int)
    per_solve = Counter(Counter(outer.tolist()).values())
    distribution = {int(key): int(value) for key, value in sorted(per_solve.items())}
    if distribution != REFERENCE_STEP_DISTRIBUTION:
        raise RuntimeError("Frozen cap=10 step-count distribution changed")
    if int(np.max(steps)) + 1 != 6:
        raise RuntimeError("Frozen cap=10 maximum evaluated-step count is not six")
    return {
        "source": str(result.resolve()),
        "source_sha256": file_sha256(result),
        "barrier_fields_accessed": False,
        "per_solve_evaluated_step_count_distribution": {
            str(key): value for key, value in distribution.items()
        },
        "maximum_observed_evaluated_steps": 6,
        "cap_greater_than_six_has_no_effect_on_this_frozen_reference_trajectory": True,
        "h0_selection_count": int(np.count_nonzero(selections == "H0")),
        "lbfgs_selection_count": int(np.count_nonzero(selections == "LBFGS")),
        "maximum_history_count": int(np.max(histories)),
        "ablation_rationale": {
            "cap3": "increases outer/reparameterization opportunities relative to deeper solves",
            "cap2": "further increases outer opportunities while retaining an LBFGS-eligible second step",
            "cap1": "H0-only mechanism endpoint because every fixed-path solve ends before LBFGS can take over",
        },
    }


def _source_manifest(root: Path) -> dict[str, dict[str, str]]:
    sources = {
        "launcher": Path(__file__).resolve(),
        "member_wrapper": root / "src/run_stage4_ala_discrete_single_max_steps_cap_ablation.py",
        "frozen_alpha_refinement_wrapper": root / "src/run_stage4_ala_discrete_single_alpha_refinement.py",
        "frozen_single_wrapper": root / "src/run_stage4_ala_discrete_single_fixedalpha.py",
        "frozen_single_implementation": root / "src/run_stage4_ala_dsel_single_fixedalpha.py",
        "legacy_command_helper": root / "src/run_stage4_uniform_optimized.py",
        "core_runner": root / "src/run_ivm_v9_spring_projected_production_20260730.py",
        "ala_core": root / "src/ala_core.py",
        "frozen_discrete_b040": root / "src/frozen_discrete_b040.py",
        "ratio_launcher": root / "src/run_stage4_ala_discrete_single_residual_ratio_ablation_grid.py",
        "ratio_member_wrapper": root / "src/run_stage4_ala_discrete_single_residual_ratio_ablation.py",
        "formal_ratio_analyzer": root / "src/analyze_stage4_ala_discrete_single_residual_ratio_iperp_ctp.py",
    }
    return {name: _hashed(path) for name, path in sources.items()}


def _registered_commands(root: Path) -> dict[str, str]:
    return {
        member.cap_tag(cap): member.compact_command_sha256(
            member.single_run_command(root, cap)[0]
        )
        for cap in member.NEW_CAPS
    }


def _assert_no_outputs(root: Path) -> None:
    for cap in member.NEW_CAPS:
        paths = member.member_paths(root, cap)
        result_dir, audit, member_log, launcher_log = (
            paths["result_dir"], paths["audit"], paths["member_log"], paths["launcher_log"]
        )
        label = str(paths["label"])
        assert isinstance(result_dir, Path) and isinstance(audit, Path) and isinstance(member_log, Path) and isinstance(launcher_log, Path)
        existing = sorted(result_dir.glob(f"ivm_{label}_central_state_ivm_cfg*_budget254.npz"))
        if existing or audit.exists() or member_log.exists() or launcher_log.exists():
            target = existing[0] if existing else audit
            raise FileExistsError(f"Refusing pre-existing cap data: {target}")


def build_preregistration(root: Path, planned_report: Path) -> dict[str, object]:
    _assert_no_outputs(root)
    formal_records, formal = _formal_ratio_chain(root)
    reused = _reuse_records(root, formal["prereg"])
    evidence = _reference_mechanism_evidence(reused)
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "planned_report": str(planned_report.resolve()),
        "one_factor_ablation": "maximum_evaluated_steps_per_frozen_subproblem",
        "fixed_alpha": member.CANONICAL_ALPHA,
        "fixed_residual_ratio": member.CANONICAL_RATIO,
        "new_max_evaluated_steps_caps": list(member.NEW_CAPS),
        "reused_reference_cap": member.REFERENCE_CAP,
        "display_caps": [1, 2, 3, 10],
        "new_member_tags": [member.cap_tag(cap) for cap in member.NEW_CAPS],
        "budget_per_image": member.CANONICAL_BUDGET,
        "replicate": member.CANONICAL_REPLICATE,
        "lbfgs_history": member.LBFGS_HISTORY,
        "reparameterization_policy": "state_curve/per_image from first state",
        "core_stop_mode_selector": member.CORE_STOP_MODE_SELECTOR,
        "core_stop_mode_selector_note": "historical API name; cap is the separately registered max-steps value",
        "stop_rule_template": "R/R0 <= 0.8 OR evaluated accepted steps == registered cap",
        "scientific_command_contract": (
            "relative to frozen alpha12.5/r0.8/cap10, only --discrete-inner-max-steps changes; "
            "output-dir and run-label are routing-only"
        ),
        "cap10_candidate_reused_not_rerun": True,
        "same_alpha_legacy_control_reused_not_rerun": True,
        "single_run_from_canonical_common_path": True,
        "eligible_record_definition": "initial plus reparam_committed",
        "mesh_gate": {"segment_length_cv_max": 0.0065, "segment_length_ratio_max": 1.03},
        "cost_to_precision_definition": "minimum actual optimization NF at I_perp <= threshold; no interpolation",
        "reporting_anchors": list(REPORTING_ANCHORS),
        "formal_selection_rule": (
            "full-curve Iperp CTP on complete mesh-eligible states versus same-alpha legacy and separately official Stage-IV v4"
        ),
        "primary_lock_eligible_caps": list(member.PRIMARY_LOCK_ELIGIBLE_CAPS),
        "primary_success_criterion": (
            "cap in {2,3,10}, at least one observed LBFGS selection, full-curve dominance with at least one strict threshold "
            "and no tightest-precision loss versus both same-alpha legacy and official Stage-IV v4"
        ),
        "cap1_role": {
            "mechanism_only": True,
            "h0_only_endpoint": True,
            "primary_winner_eligible": False,
            "reason": "one evaluated step ends each solve before an L-BFGS takeover can occur",
        },
        "multiple_primary_winner_tiebreak": [
            "lexicographically smallest candidate-minus-official NF vector tight-to-loose",
            "lexicographically smallest candidate-minus-legacy NF vector tight-to-loose",
            "smallest absolute difference from reused cap10",
        ],
        "no_primary_dominator_rule": {
            "winner_claim_allowed": False,
            "cap_lock_allowed": False,
            "post_hoc_grid_expansion_allowed": False,
            "barrier_unembargo_allowed": False,
        },
        "terminal_or_raw_iperp_ranking_allowed": False,
        "barrier_fields_accessed_by_wrapper_or_launcher": False,
        "sparse_barrier_role": "embargoed_not_read_not_reported_not_ranked",
        "densified_energy_barrier_role": "embargoed_until_primary_cap_lock",
        "formal_ratio_chain": formal_records,
        "formal_ratio_decision": {
            "joint_full_curve_dominator_ratios": [],
            "ratio_lock_status": formal["analysis"]["ratio_lock_status"],
            "winner_claim_made": False,
            "barrier_remains_embargoed": True,
        },
        "canonical_common_initial_path": _common_initial_record(root),
        "reused_inputs": reused,
        "reference_mechanism_evidence": evidence,
        "registered_command_hashes": _registered_commands(root),
        "run_source_manifest": _source_manifest(root),
        "independent_member_paths_and_logs": True,
        "overwrite_or_force_allowed": False,
    }


def validate_preregistration(protocol: dict[str, object], root: Path) -> None:
    if protocol.get("status") != "preregistered" or protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Invalid cap-ablation preregistration")
    if tuple(int(value) for value in protocol["new_max_evaluated_steps_caps"]) != member.NEW_CAPS:
        raise RuntimeError("Preregistered cap grid drifted")
    if float(protocol["fixed_alpha"]) != 12.5 or float(protocol["fixed_residual_ratio"]) != 0.8:
        raise RuntimeError("Preregistered alpha/ratio drifted")
    if tuple(int(value) for value in protocol["primary_lock_eligible_caps"]) != member.PRIMARY_LOCK_ELIGIBLE_CAPS:
        raise RuntimeError("Primary-eligible cap set drifted")
    if protocol["cap1_role"]["primary_winner_eligible"] is not False:
        raise RuntimeError("Cap1 cannot be primary-winner eligible")
    if protocol.get("barrier_fields_accessed_by_wrapper_or_launcher") is not False:
        raise RuntimeError("Preregistration broke the barrier embargo")
    planned = Path(str(protocol["planned_report"])).resolve()
    if planned.parent != (root / "reports").resolve() or not planned.name.startswith(f"{PROTOCOL_ID}_N254_r1_"):
        raise RuntimeError("Planned report uses the wrong namespace")
    formal_records, formal = _formal_ratio_chain(root)
    if protocol["formal_ratio_chain"] != formal_records:
        raise RuntimeError("Formal ratio chain drifted")
    if protocol["canonical_common_initial_path"] != _common_initial_record(root):
        raise RuntimeError("Canonical common path drifted")
    _verify_manifest(protocol["reused_inputs"], "reused inputs")
    if protocol["reused_inputs"] != _reuse_records(root, formal["prereg"]):
        raise RuntimeError("Reused input manifest drifted")
    if protocol["reference_mechanism_evidence"] != _reference_mechanism_evidence(protocol["reused_inputs"]):
        raise RuntimeError("Reference mechanism evidence drifted")
    _verify_manifest(protocol["run_source_manifest"], "run sources")
    if protocol["registered_command_hashes"] != _registered_commands(root):
        raise RuntimeError("Registered cap commands changed")
    _assert_no_outputs(root)


def _run_process(command: list[str], root: Path, stdout_path: Path, stderr_path: Path) -> tuple[int, float, str]:
    environment = os.environ.copy()
    environment.update({
        "OPENMM_CPU_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    })
    started = time.time()
    completed = subprocess.run(command, cwd=root, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return completed.returncode, time.time() - started, completed.stderr


def _run_cell(root: Path, cap: int) -> dict[str, object]:
    paths = member.member_paths(root, cap)
    launcher_log, audit_path = paths["launcher_log"], paths["audit"]
    assert isinstance(launcher_log, Path) and isinstance(audit_path, Path)
    launcher_log.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(root / "src/run_stage4_ala_discrete_single_max_steps_cap_ablation.py"),
        "--root", str(root), "--max-evaluated-steps", str(cap),
    ]
    (launcher_log / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    returncode, wall_seconds, stderr = _run_process(command, root, launcher_log / "stdout.txt", launcher_log / "stderr.txt")
    row: dict[str, object] = {"max_evaluated_steps": cap, "cap_tag": member.cap_tag(cap), "returncode": returncode, "wall_seconds": wall_seconds}
    if returncode:
        row["error"] = stderr[-4000:]
        return row
    audit = _read_json(audit_path)
    if audit.get("barrier_fields_accessed") is not False or int(audit["max_evaluated_steps"]) != cap:
        raise RuntimeError("Cap member audit identity/embargo mismatch")
    result = Path(str(audit["result"])).resolve()
    if file_sha256(result) != str(audit["result_sha256"]):
        raise RuntimeError("Cap member result hash mismatch")
    row.update({
        "member_audit": str(audit_path), "member_audit_sha256": file_sha256(audit_path),
        "result": str(result), "result_sha256": file_sha256(result),
        "final_optimization_NF": int(audit["final_optimization_NF"]),
        "mesh_eligible_complete_state_count": int(audit["mesh_eligible_complete_state_count"]),
        "h0_selection_count": int(audit["h0_selection_count"]),
        "lbfgs_selection_count": int(audit["lbfgs_selection_count"]),
        "lbfgs_takeover_observed": bool(audit["lbfgs_takeover_observed"]),
    })
    return row


def preregister(root: Path) -> Path:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stem = f"{PROTOCOL_ID}_N254_r1_{time.time_ns()}"
    preregistration = reports / f"{stem}_preregistered.json"
    planned_report = reports / f"{stem}.json"
    _write_json_exclusive(preregistration, build_preregistration(root, planned_report))
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
    rows = []
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_cell, root, cap): cap for cap in member.NEW_CAPS}
        for future in concurrent.futures.as_completed(futures):
            cap = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {"max_evaluated_steps": cap, "cap_tag": member.cap_tag(cap), "returncode": -1, "error": repr(exc)}
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    rows.sort(key=lambda row: int(row["max_evaluated_steps"]))
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
