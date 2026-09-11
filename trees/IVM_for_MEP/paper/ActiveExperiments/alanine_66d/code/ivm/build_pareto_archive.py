#!/usr/bin/env python3
"""Build a reference-free, force-accounted Pareto archive of complete IVM states."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from ala_core import (
    ImageEvaluator,
    OpenMMBackend,
    TOPOLOGY_FILENAME,
    array_sha256,
    atomic_save_npz,
    densified_barrier,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def complete_candidate_calls(combined, activation: float) -> np.ndarray:
    stored_force = np.asarray(combined["stored_force_evaluations"], dtype=int)
    stored_phase = np.asarray(combined["stored_phase"], dtype=int)
    stored_i = np.asarray(combined["i_perp"], dtype=float)
    candidates = {
        int(force)
        for force, phase, i_perp in zip(stored_force, stored_phase, stored_i)
        if int(phase) == 0 and float(i_perp) <= activation
    }
    phase_b_path = Path(str(scalar(combined, "phase_b_result")))
    with np.load(phase_b_path, allow_pickle=False) as phase_b:
        offset = int(scalar(combined, "phase_transition_force_offset"))
        for force, i_perp, committed in zip(
            np.asarray(phase_b["total_internal_joint_evaluations"], dtype=int),
            np.asarray(phase_b["i_perp"], dtype=float),
            np.asarray(phase_b["reparam_committed"], dtype=bool),
        ):
            if bool(committed) and float(i_perp) <= activation:
                candidates.add(offset + int(force))
    return np.asarray(sorted(candidates), dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-label", required=True)
    parser.add_argument("--activation-iperp", type=float, default=7e-3)
    parser.add_argument("--target-iperp", type=float, default=1.9e-4)
    parser.add_argument("--screen-subdivisions", type=int, default=10)
    parser.add_argument("--barrier-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--continue-after-target",
        action="store_true",
        help=(
            "Screen every remaining complete state after the first joint target; "
            "used only for uncensored audit archives."
        ),
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    result_path = args.result.expanduser().resolve()
    with np.load(result_path, allow_pickle=False) as combined:
        calls = complete_candidate_calls(combined, args.activation_iperp)
        stored_force = np.asarray(combined["stored_force_evaluations"], dtype=int)
        stored_paths = np.asarray(combined["stored_paths"], dtype=float)
        history_force = np.asarray(combined["total_force_evaluations"], dtype=int)
        history_i = np.asarray(combined["i_perp"], dtype=float)
        alpha = float(scalar(combined, "alpha"))
        common_initial_sha256 = str(scalar(combined, "common_initial_path_sha256"))
        combined_final_force = int(history_force[-1])

    path_by_call = {}
    i_by_call = {}
    for call in calls:
        path_matches = np.flatnonzero(stored_force == call)
        history_matches = np.flatnonzero(history_force == call)
        if not path_matches.size or not history_matches.size:
            raise RuntimeError(f"Candidate call {call} is not exactly stored")
        path_by_call[int(call)] = stored_paths[int(path_matches[-1])]
        i_by_call[int(call)] = float(history_i[int(history_matches[-1])])

    backend = OpenMMBackend(
        root / "inputs" / TOPOLOGY_FILENAME,
        platform_preference=("CPU",),
        precision="double",
        label="pareto_barrier10_online_screen",
    )
    evaluator = ImageEvaluator(backend, image_index=0)
    incumbent_i = math.inf
    incumbent_barrier = math.inf
    candidate_i = []
    candidate_barrier = []
    candidate_cost = []
    candidate_accepted = []
    processed_calls = []
    selected_call = -1
    target_reached = False
    first_target_call = -1
    first_target_i = math.inf
    first_target_screen_cost = 0

    for call in calls:
        before = int(backend.raw_joint_evaluations)
        barrier, _ = densified_barrier(
            path_by_call[int(call)], evaluator, args.screen_subdivisions
        )
        cost = int(backend.raw_joint_evaluations) - before
        i_perp = i_by_call[int(call)]
        accepted = bool(
            i_perp <= incumbent_i + 1e-15
            and barrier <= incumbent_barrier + args.barrier_tolerance
        )
        if accepted:
            incumbent_i = i_perp
            incumbent_barrier = barrier
            selected_call = int(call)
        processed_calls.append(int(call))
        candidate_i.append(i_perp)
        candidate_barrier.append(barrier)
        candidate_cost.append(cost)
        candidate_accepted.append(accepted)
        if accepted and i_perp <= args.target_iperp:
            if first_target_call < 0:
                target_reached = True
                first_target_call = int(call)
                first_target_i = float(i_perp)
                first_target_screen_cost = int(backend.raw_joint_evaluations)
            if not args.continue_after_target:
                break

    screen_cost = int(backend.raw_joint_evaluations)
    if selected_call >= 0:
        selected_path = np.asarray(path_by_call[selected_call], dtype=float)
        selected_i = float(i_by_call[selected_call])
        if args.continue_after_target:
            optimization_cost = combined_final_force
        else:
            optimization_cost = (
                first_target_call if target_reached else combined_final_force
            )
    else:
        selected_path = np.empty((0, 0), dtype=float)
        selected_i = math.inf
        optimization_cost = combined_final_force
    total_charged = optimization_cost + screen_cost
    first_target_total_charged = (
        first_target_call + first_target_screen_cost if target_reached else -1
    )
    full_audit_total_charged = combined_final_force + screen_cost

    output_dir = root / "runs" / "pareto_archive"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"pareto_{args.output_label}.npz"
    atomic_save_npz(
        output,
        policy_version=np.array("stage4_uniform_v4_pareto_archive"),
        source_result=np.array(str(result_path)),
        source_result_sha256=np.array(file_sha256(result_path)),
        common_initial_path_sha256=np.array(common_initial_sha256),
        alpha=np.array(alpha),
        activation_iperp=np.array(args.activation_iperp),
        target_iperp=np.array(args.target_iperp),
        screen_subdivisions=np.array(args.screen_subdivisions),
        barrier_tolerance=np.array(args.barrier_tolerance),
        reference_barrier_used=np.array(False),
        complete_state_candidates_only=np.array(True),
        candidate_calls=np.asarray(processed_calls, dtype=int),
        candidate_i_perp=np.asarray(candidate_i, dtype=float),
        candidate_screen_barrier_kcal=np.asarray(candidate_barrier, dtype=float),
        candidate_screen_force_equivalent_cost=np.asarray(candidate_cost, dtype=int),
        candidate_accepted=np.asarray(candidate_accepted, dtype=bool),
        accepted_calls=np.asarray(processed_calls, dtype=int)[
            np.asarray(candidate_accepted, dtype=bool)
        ],
        selected_call=np.array(selected_call),
        selected_i_perp=np.array(selected_i),
        selected_screen_barrier_kcal=np.array(incumbent_barrier),
        selected_path=selected_path,
        selected_path_sha256=np.array(
            array_sha256(selected_path) if selected_path.size else ""
        ),
        target_reached=np.array(target_reached),
        continue_after_target=np.array(args.continue_after_target),
        first_target_call=np.array(first_target_call),
        first_target_i_perp=np.array(first_target_i),
        first_target_screen_force_equivalent_evaluations=np.array(
            first_target_screen_cost
        ),
        first_target_total_charged_force_equivalent_evaluations=np.array(
            first_target_total_charged
        ),
        optimization_force_evaluations_at_selection=np.array(optimization_cost),
        screen_force_equivalent_evaluations=np.array(screen_cost),
        total_charged_force_equivalent_evaluations=np.array(total_charged),
        full_audit_optimization_force_evaluations=np.array(combined_final_force),
        full_audit_screen_force_equivalent_evaluations=np.array(screen_cost),
        full_audit_total_charged_force_equivalent_evaluations=np.array(
            full_audit_total_charged
        ),
        materialization_mode=np.array(
            "deterministic reference-free online-equivalent prefix replay"
        ),
    )
    audit = {
        "status": "target_reached" if target_reached else "target_not_reached",
        "policy_version": "stage4_uniform_v4_pareto_archive",
        "source_result": str(result_path),
        "output": str(output),
        "alpha": "inf" if math.isinf(alpha) else alpha,
        "activation_iperp": args.activation_iperp,
        "target_iperp": args.target_iperp,
        "screen_subdivisions": args.screen_subdivisions,
        "reference_barrier_used": False,
        "candidate_calls": processed_calls,
        "accepted_calls": [
            call for call, accepted in zip(processed_calls, candidate_accepted) if accepted
        ],
        "selected_call": selected_call,
        "selected_i_perp": selected_i if math.isfinite(selected_i) else None,
        "selected_screen_barrier_kcal": (
            incumbent_barrier if math.isfinite(incumbent_barrier) else None
        ),
        "continue_after_target": args.continue_after_target,
        "first_target_call": first_target_call,
        "first_target_i_perp": (
            first_target_i if math.isfinite(first_target_i) else None
        ),
        "first_target_screen_force_equivalent_evaluations": (
            first_target_screen_cost
        ),
        "first_target_total_charged_force_equivalent_evaluations": (
            first_target_total_charged
        ),
        "optimization_force_evaluations_at_selection": optimization_cost,
        "screen_force_equivalent_evaluations": screen_cost,
        "total_charged_force_equivalent_evaluations": total_charged,
        "full_audit_optimization_force_evaluations": combined_final_force,
        "full_audit_screen_force_equivalent_evaluations": screen_cost,
        "full_audit_total_charged_force_equivalent_evaluations": (
            full_audit_total_charged
        ),
        "backend_raw_joint_evaluations": int(backend.raw_joint_evaluations),
        "evaluator_cache_misses": int(evaluator.counters.joint_evaluations),
        "accounting_consistent": bool(
            int(backend.raw_joint_evaluations)
            == int(evaluator.counters.joint_evaluations)
            == int(sum(candidate_cost))
        ),
        "materialization_mode": (
            "deterministic reference-free online-equivalent prefix replay"
        ),
    }
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
