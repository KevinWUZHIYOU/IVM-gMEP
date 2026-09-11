#!/usr/bin/env python3
"""Supersede the unlaunched cap<=3 protocol and preregister cap>3 v2."""

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

import run_stage4_ala_discrete_single_max_steps_cap_ablation_gt3 as member
import run_stage4_ala_discrete_single_max_steps_cap_ablation_grid as superseded_grid


PROTOCOL_ID = (
    "stage4_ala_discrete_single_max_evaluated_steps_cap_ablation_gt3_v2"
)
SUPERSEDED_PROTOCOL_ID = (
    "stage4_ala_discrete_single_max_evaluated_steps_cap_ablation_v1"
)
SUPERSEDED_PREREGISTRATION = (
    "reports/stage4_ala_discrete_single_max_evaluated_steps_cap_ablation_v1_"
    "N254_r1_1787254333339866000_preregistered.json"
)
SUPERSEDED_PREREGISTRATION_SHA256 = (
    "cb0fd4ef2368eed44f83180452d3939c9be44fb7cb8736379464e176394890c9"
)
SUPERSEDED_POLICY_VERSION = (
    "stage4_ala_discrete_single_max_evaluated_steps_cap_ablation_v1"
)
SUPERSESSION_REASON = (
    "user objective updated maxstep>3 before any launch"
)
SUPERSEDED_SOURCE_HASHES = {
    "wrapper": "d928c5afe373e8edb2e56772bcc8a790117b7d1fc458d0f70509ce985984d540",
    "launcher": "879adfe45ce260a27d361c9eb27c3f8f73a277fb93c75b28f270ca622bf4cdbf",
    "tests": "caf136d8e4eb4fc6db6ffefb81d677e4c6d8dd090580a9f9e8d17286d6170796",
}
SUPERSEDED_SOURCE_PATHS = {
    "wrapper": "src/run_stage4_ala_discrete_single_max_steps_cap_ablation.py",
    "launcher": "src/run_stage4_ala_discrete_single_max_steps_cap_ablation_grid.py",
    "tests": "tests/test_stage4_ala_discrete_single_max_steps_cap_ablation.py",
}
PRIMARY_SUCCESS_CRITERION = (
    "cap in {4,5,6,10}, at least one observed LBFGS selection, full-curve "
    "dominance with at least one strict threshold and no tightest-precision "
    "loss versus both comparators"
)
FORMAL_SELECTION_RULE = (
    "full-curve Iperp CTP on complete mesh-eligible states versus same-alpha "
    "legacy and separately official Stage-IV v4"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hashed(path: Path, expected: str | None = None) -> dict[str, str]:
    path = path.resolve()
    actual = file_sha256(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f"Hash drift: {path}")
    return {"path": str(path), "sha256": actual}


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _superseded_source_records(root: Path) -> dict[str, dict[str, str]]:
    return {
        name: _hashed(root / SUPERSEDED_SOURCE_PATHS[name], expected)
        for name, expected in SUPERSEDED_SOURCE_HASHES.items()
    }


def _superseded_output_evidence(root: Path) -> dict[str, object]:
    run_root = root / "runs" / SUPERSEDED_POLICY_VERSION
    log_root = root / "logs" / SUPERSEDED_POLICY_VERSION
    run_files = sorted(path for path in run_root.rglob("*") if path.is_file()) if run_root.exists() else []
    log_files = sorted(path for path in log_root.rglob("*") if path.is_file()) if log_root.exists() else []
    prereg = _read_json((root / SUPERSEDED_PREREGISTRATION).resolve())
    planned = Path(str(prereg["planned_report"])).resolve()
    if run_files or log_files or planned.exists():
        raise RuntimeError("Superseded v1 was launched or has output artifacts")
    return {
        "checked_run_root": str(run_root.resolve()),
        "checked_log_root": str(log_root.resolve()),
        "run_file_count": 0,
        "log_file_count": 0,
        "planned_report": str(planned),
        "planned_report_exists": False,
        "proof": "no v1 run/log files and no planned formal report before supersession",
    }


def build_supersession_payload(root: Path) -> dict[str, object]:
    old_prereg = (root / SUPERSEDED_PREREGISTRATION).resolve()
    _hashed(old_prereg, SUPERSEDED_PREREGISTRATION_SHA256)
    payload = _read_json(old_prereg)
    if payload.get("status") != "preregistered" or payload.get("protocol_id") != SUPERSEDED_PROTOCOL_ID:
        raise RuntimeError("Old v1 preregistration identity mismatch")
    return {
        "status": "superseded_unlaunched",
        "superseded_protocol_id": SUPERSEDED_PROTOCOL_ID,
        "superseding_protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "reason": SUPERSESSION_REASON,
        "new_hard_constraint": "max_evaluated_steps > 3",
        "superseded_preregistration": _hashed(
            old_prereg, SUPERSEDED_PREREGISTRATION_SHA256
        ),
        "superseded_source_files_unchanged": _superseded_source_records(root),
        "zero_output_evidence": _superseded_output_evidence(root),
        "superseded_preregistration_remains_immutable": True,
        "barrier_data_accessed": False,
    }


def _verify_supersession_sidecar(record: object, root: Path) -> dict[str, object]:
    if not isinstance(record, dict):
        raise RuntimeError("Missing supersession sidecar record")
    path = Path(str(record["path"])).resolve()
    _hashed(path, str(record["sha256"]))
    payload = _read_json(path)
    if (
        payload.get("status") != "superseded_unlaunched"
        or payload.get("superseded_protocol_id") != SUPERSEDED_PROTOCOL_ID
        or payload.get("superseding_protocol_id") != PROTOCOL_ID
    ):
        raise RuntimeError("Invalid supersession sidecar")
    if payload.get("reason") != SUPERSESSION_REASON:
        raise RuntimeError("Supersession sidecar reason drifted")
    if payload.get("new_hard_constraint") != "max_evaluated_steps > 3":
        raise RuntimeError("Supersession sidecar lost the user hard constraint")
    if payload.get("superseded_preregistration_remains_immutable") is not True:
        raise RuntimeError("Supersession sidecar no longer preserves v1")
    if payload.get("barrier_data_accessed") is not False:
        raise RuntimeError("Supersession sidecar broke the barrier embargo")
    if payload.get("zero_output_evidence") != _superseded_output_evidence(root):
        raise RuntimeError("Superseded v1 zero-output evidence changed")
    if payload.get("superseded_source_files_unchanged") != _superseded_source_records(root):
        raise RuntimeError("Superseded v1 source hashes changed")
    if payload.get("superseded_preregistration") != _hashed(
        root / SUPERSEDED_PREREGISTRATION, SUPERSEDED_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("Superseded preregistration hash changed")
    return payload


def _source_manifest(root: Path) -> dict[str, dict[str, str]]:
    inherited = superseded_grid._source_manifest(root)
    sources = {
        "gt3_launcher": _hashed(Path(__file__).resolve()),
        "gt3_member_wrapper": _hashed(
            root / "src/run_stage4_ala_discrete_single_max_steps_cap_ablation_gt3.py"
        ),
        "superseded_v1_launcher": _hashed(
            root / "src/run_stage4_ala_discrete_single_max_steps_cap_ablation_grid.py",
            SUPERSEDED_SOURCE_HASHES["launcher"],
        ),
        "superseded_v1_member_wrapper": _hashed(
            root / "src/run_stage4_ala_discrete_single_max_steps_cap_ablation.py",
            SUPERSEDED_SOURCE_HASHES["wrapper"],
        ),
    }
    for name, record in inherited.items():
        sources[f"inherited_{name}"] = record
    return sources


def _registered_commands(root: Path) -> dict[str, str]:
    return {
        member.cap_tag(cap): member.compact_command_sha256(
            member.single_run_command(root, cap)[0]
        )
        for cap in member.NEW_CAPS
    }


def _gt3_reference_mechanism_evidence(
    reused: dict[str, dict[str, str]]
) -> dict[str, object]:
    evidence = dict(superseded_grid._reference_mechanism_evidence(reused))
    evidence.pop("ablation_rationale", None)
    evidence["gt3_ablation_rationale"] = {
        "cap4": (
            "smallest user-eligible cap; on the frozen cap10 reference it "
            "truncates the sole six-step solve after four evaluated steps"
        ),
        "cap5": (
            "intermediate user-eligible cap; on the frozen cap10 reference it "
            "truncates the sole six-step solve after five evaluated steps"
        ),
        "cap6": (
            "observed-depth endpoint equal to the maximum six evaluated steps "
            "seen in the frozen cap10 reference"
        ),
    }
    evidence["all_new_caps_satisfy_user_gt3_constraint"] = True
    evidence["all_primary_caps_require_observed_lbfgs_takeover"] = True
    return evidence


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
            raise FileExistsError(f"Refusing pre-existing GT3 cap data: {target}")


def build_preregistration(
    root: Path, planned_report: Path, sidecar_record: dict[str, str]
) -> dict[str, object]:
    _assert_no_outputs(root)
    _verify_supersession_sidecar(sidecar_record, root)
    formal_records, formal = superseded_grid._formal_ratio_chain(root)
    reused = superseded_grid._reuse_records(root, formal["prereg"])
    evidence = _gt3_reference_mechanism_evidence(reused)
    return {
        "status": "preregistered",
        "protocol_id": PROTOCOL_ID,
        "created_unix": time.time(),
        "planned_report": str(planned_report.resolve()),
        "user_hard_constraint": "max_evaluated_steps > 3",
        "supersession_sidecar": dict(sidecar_record),
        "superseded_v1_preregistration": _hashed(
            root / SUPERSEDED_PREREGISTRATION, SUPERSEDED_PREREGISTRATION_SHA256
        ),
        "superseded_v1_was_never_launched": True,
        "one_factor_ablation": "maximum_evaluated_steps_per_frozen_subproblem",
        "fixed_alpha": 12.5,
        "fixed_residual_ratio": 0.8,
        "new_max_evaluated_steps_caps": [4, 5, 6],
        "reused_reference_cap": 10,
        "display_caps": [4, 5, 6, 10],
        "new_member_tags": ["c04", "c05", "c06"],
        "budget_per_image": 254,
        "replicate": "r1",
        "lbfgs_history": 7,
        "reparameterization_policy": "state_curve/per_image from first state",
        "core_stop_mode_selector": "r08c10",
        "stop_rule_template": "R/R0 <= 0.8 OR evaluated accepted steps == registered cap",
        "scientific_command_contract": (
            "relative to frozen alpha12.5/r0.8/cap10, only --discrete-inner-max-steps changes; output-dir and run-label are routing-only"
        ),
        "cap10_candidate_reused_not_rerun": True,
        "same_alpha_legacy_control_reused_not_rerun": True,
        "single_run_from_canonical_common_path": True,
        "eligible_record_definition": "initial plus reparam_committed",
        "mesh_gate": {"segment_length_cv_max": 0.0065, "segment_length_ratio_max": 1.03},
        "cost_to_precision_definition": "minimum actual optimization NF at I_perp <= threshold; no interpolation",
        "reporting_anchors": list(superseded_grid.REPORTING_ANCHORS),
        "formal_selection_rule": FORMAL_SELECTION_RULE,
        "primary_lock_eligible_caps": [4, 5, 6, 10],
        "primary_success_criterion": PRIMARY_SUCCESS_CRITERION,
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
        "canonical_common_initial_path": superseded_grid._common_initial_record(root),
        "reused_inputs": reused,
        "reference_mechanism_evidence": evidence,
        "registered_command_hashes": _registered_commands(root),
        "run_source_manifest": _source_manifest(root),
        "independent_member_paths_and_logs": True,
        "overwrite_or_force_allowed": False,
    }


def validate_preregistration(protocol: dict[str, object], root: Path) -> None:
    if protocol.get("status") != "preregistered" or protocol.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError("Invalid GT3 cap preregistration")
    if tuple(int(value) for value in protocol["new_max_evaluated_steps_caps"]) != (4, 5, 6):
        raise RuntimeError("GT3 cap grid drifted")
    if not all(int(value) > 3 for value in protocol["new_max_evaluated_steps_caps"]):
        raise RuntimeError("GT3 hard constraint violated")
    if protocol.get("user_hard_constraint") != "max_evaluated_steps > 3":
        raise RuntimeError("GT3 preregistration lost the user hard constraint")
    if protocol.get("display_caps") != [4, 5, 6, 10] or protocol.get(
        "new_member_tags"
    ) != ["c04", "c05", "c06"]:
        raise RuntimeError("GT3 display caps/tags drifted")
    if (
        float(protocol["fixed_alpha"]) != 12.5
        or float(protocol["fixed_residual_ratio"]) != 0.8
        or int(protocol["budget_per_image"]) != 254
        or str(protocol["replicate"]) != "r1"
        or int(protocol["lbfgs_history"]) != 7
        or protocol.get("core_stop_mode_selector") != "r08c10"
        or protocol.get("reparameterization_policy")
        != "state_curve/per_image from first state"
    ):
        raise RuntimeError("GT3 frozen configuration drifted")
    if protocol.get("eligible_record_definition") != "initial plus reparam_committed":
        raise RuntimeError("GT3 complete-state definition drifted")
    if protocol.get("mesh_gate") != {
        "segment_length_cv_max": 0.0065,
        "segment_length_ratio_max": 1.03,
    }:
        raise RuntimeError("GT3 mesh gate drifted")
    if protocol.get("cost_to_precision_definition") != (
        "minimum actual optimization NF at I_perp <= threshold; no interpolation"
    ):
        raise RuntimeError("GT3 CTP definition drifted")
    if protocol.get("formal_selection_rule") != FORMAL_SELECTION_RULE:
        raise RuntimeError("GT3 formal selection rule drifted")
    if tuple(int(value) for value in protocol["primary_lock_eligible_caps"]) != (4, 5, 6, 10):
        raise RuntimeError("Primary-eligible cap set drifted")
    if protocol.get("primary_success_criterion") != PRIMARY_SUCCESS_CRITERION:
        raise RuntimeError("GT3 primary criterion drifted")
    no_winner = protocol.get("no_primary_dominator_rule")
    if not isinstance(no_winner, dict) or no_winner != {
        "winner_claim_allowed": False,
        "cap_lock_allowed": False,
        "post_hoc_grid_expansion_allowed": False,
        "barrier_unembargo_allowed": False,
    }:
        raise RuntimeError("GT3 no-dominator rule drifted")
    if protocol.get("terminal_or_raw_iperp_ranking_allowed") is not False:
        raise RuntimeError("GT3 preregistration enabled terminal ranking")
    if protocol.get("barrier_fields_accessed_by_wrapper_or_launcher") is not False:
        raise RuntimeError("Preregistration broke barrier embargo")
    planned = Path(str(protocol["planned_report"])).resolve()
    if planned.parent != (root / "reports").resolve() or not planned.name.startswith(f"{PROTOCOL_ID}_N254_r1_"):
        raise RuntimeError("Planned report namespace drifted")
    _verify_supersession_sidecar(protocol["supersession_sidecar"], root)
    if protocol["superseded_v1_preregistration"] != _hashed(
        root / SUPERSEDED_PREREGISTRATION, SUPERSEDED_PREREGISTRATION_SHA256
    ):
        raise RuntimeError("Superseded prereg hash drifted")
    formal_records, formal = superseded_grid._formal_ratio_chain(root)
    if protocol["formal_ratio_chain"] != formal_records:
        raise RuntimeError("Formal ratio chain drifted")
    if protocol["canonical_common_initial_path"] != superseded_grid._common_initial_record(root):
        raise RuntimeError("Common initial path drifted")
    superseded_grid._verify_manifest(protocol["reused_inputs"], "reused inputs")
    if protocol["reused_inputs"] != superseded_grid._reuse_records(root, formal["prereg"]):
        raise RuntimeError("Reused input manifest drifted")
    if protocol["reference_mechanism_evidence"] != _gt3_reference_mechanism_evidence(
        protocol["reused_inputs"]
    ):
        raise RuntimeError("Mechanism evidence drifted")
    superseded_grid._verify_manifest(protocol["run_source_manifest"], "run sources")
    if protocol["run_source_manifest"] != _source_manifest(root):
        raise RuntimeError("GT3 run source manifest drifted")
    if protocol["registered_command_hashes"] != _registered_commands(root):
        raise RuntimeError("Registered GT3 commands changed")
    _assert_no_outputs(root)


def _run_process(command: list[str], root: Path, stdout: Path, stderr: Path) -> tuple[int, float, str]:
    environment = os.environ.copy()
    environment.update({
        "OPENMM_CPU_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    })
    started = time.time()
    completed = subprocess.run(command, cwd=root, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    return completed.returncode, time.time() - started, completed.stderr


def _run_cell(root: Path, cap: int) -> dict[str, object]:
    paths = member.member_paths(root, cap)
    launcher_log, audit_path = paths["launcher_log"], paths["audit"]
    assert isinstance(launcher_log, Path) and isinstance(audit_path, Path)
    launcher_log.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(root / "src/run_stage4_ala_discrete_single_max_steps_cap_ablation_gt3.py"),
        "--root", str(root), "--max-evaluated-steps", str(cap),
    ]
    (launcher_log / "command.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    returncode, wall_seconds, error = _run_process(command, root, launcher_log / "stdout.txt", launcher_log / "stderr.txt")
    row: dict[str, object] = {"max_evaluated_steps": cap, "cap_tag": member.cap_tag(cap), "returncode": returncode, "wall_seconds": wall_seconds}
    if returncode:
        row["error"] = error[-4000:]
        return row
    audit = _read_json(audit_path)
    if audit.get("barrier_fields_accessed") is not False or int(audit["max_evaluated_steps"]) != cap:
        raise RuntimeError("GT3 member audit mismatch")
    result = Path(str(audit["result"])).resolve()
    if file_sha256(result) != str(audit["result_sha256"]):
        raise RuntimeError("GT3 result hash mismatch")
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


def preregister(root: Path) -> tuple[Path, Path]:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    unique = time.time_ns()
    sidecar = reports / (
        f"{SUPERSEDED_PROTOCOL_ID}_N254_r1_1787254333339866000_"
        f"superseded_by_gt3_v2_{unique}.json"
    )
    _write_json_exclusive(sidecar, build_supersession_payload(root))
    sidecar_record = _hashed(sidecar)
    stem = f"{PROTOCOL_ID}_N254_r1_{unique}"
    preregistration = reports / f"{stem}_preregistered.json"
    planned_report = reports / f"{stem}.json"
    _write_json_exclusive(
        preregistration,
        build_preregistration(root, planned_report, sidecar_record),
    )
    return sidecar, preregistration


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
        sidecar, preregistration = preregister(root)
        print(sidecar)
        print(preregistration)
        return
    print(launch(root, args.preregistration, args.jobs))


if __name__ == "__main__":
    main()
