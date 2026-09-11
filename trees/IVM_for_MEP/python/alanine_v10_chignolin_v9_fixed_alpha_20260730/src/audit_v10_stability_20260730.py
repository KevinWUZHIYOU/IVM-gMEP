#!/usr/bin/env python3
"""Fail-closed stability audit for one Ala V10 production trajectory.

The optimization run is charged only for its joint energy/force calls. Dense
barriers are an independent diagnostic and are never put on the force-cost
axis. This audit checks the complete raw accepted path, not a cumulative-best
or cross-alpha envelope.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from audit_force_calls import audit_result
from run_ala_v10_policy_20260730 import POLICIES


def scalar(saved, name):
    value = np.asarray(saved[name])
    if value.size != 1:
        raise ValueError(f"{name} must be scalar")
    return value.reshape(()).item()


def alpha_equal(left: float, right: float) -> bool:
    if math.isinf(left) or math.isinf(right):
        return math.isinf(left) and math.isinf(right)
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-14)


def path_change_indices(paths: np.ndarray) -> np.ndarray:
    """Keep the first state and every genuine accepted coordinate change."""
    if len(paths) == 0:
        raise ValueError("stored path history is empty")
    changed = [0]
    for index in range(1, len(paths)):
        if not np.array_equal(paths[index], paths[changed[-1]]):
            changed.append(index)
    return np.asarray(changed, dtype=int)


def upward_events(calls, values, *, atol, rtol, start_call, relative_to_best):
    events = []
    best_value = float(values[0])
    best_call = int(calls[0])
    for previous, current in zip(range(len(values) - 1), range(1, len(values))):
        if int(calls[current]) <= int(start_call):
            if float(values[current]) < best_value:
                best_value = float(values[current])
                best_call = int(calls[current])
            continue
        reference_value = best_value if relative_to_best else float(values[previous])
        reference_call = best_call if relative_to_best else int(calls[previous])
        allowed = reference_value * (1.0 + float(rtol)) + float(atol)
        if float(values[current]) > allowed:
            events.append(
                {
                    "reference_call": reference_call,
                    "to_call": int(calls[current]),
                    "reference_value": reference_value,
                    "to_value": float(values[current]),
                    "increase": float(values[current] - reference_value),
                    "allowed_value": float(allowed),
                }
            )
        if float(values[current]) < best_value:
            best_value = float(values[current])
            best_call = int(calls[current])
    return events


def load_dense(path: Path):
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "calls",
            "adaptive_barrier_kcal",
            "final_level_difference_kcal",
            "source_result",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"dense diagnostic missing fields: {sorted(missing)}")
        return {
            "source_result": str(scalar(saved, "source_result")),
            "calls": np.asarray(saved["calls"], dtype=int),
            "barrier": np.asarray(saved["adaptive_barrier_kcal"], dtype=float),
            "resolution": np.asarray(saved["final_level_difference_kcal"], dtype=float),
        }


def audit(args) -> dict:
    result_path = Path(args.result).expanduser().resolve()
    force_report = audit_result(result_path)
    failures = list(force_report["failures"])
    with np.load(result_path, allow_pickle=False) as saved:
        required = {
            "stored_paths",
            "stored_path_calls",
            "calls_per_image",
            "i_perp",
            "active_alpha",
            "alpha",
            "alpha_mode",
            "solver",
            "reparam_mode",
            "plbfgs_kkt_rtol",
            "inner_calls",
            "state_curve_direct_trial_calls",
            "state_globalization",
            "state_curve_direct_objective_acceptance",
            "state_kkt_stall_tighten_target",
            "active_plbfgs_kkt_rtol",
            "stop_iperp",
            "initial_path",
            "initial_path_sha256",
            "segment_length_cv",
        }
        missing = required.difference(saved.files)
        if missing:
            raise ValueError(f"result missing fields: {sorted(missing)}")
        history_calls = np.asarray(saved["calls_per_image"], dtype=int)
        history_iperp = np.asarray(saved["i_perp"], dtype=float)
        history_spacing_cv = np.asarray(saved["segment_length_cv"], dtype=float)
        initial_path = np.asarray(saved["initial_path"], dtype=float)
        initial_path_sha256 = str(scalar(saved, "initial_path_sha256"))
        if args.expected_image_count is not None and initial_path.shape[0] != int(
            args.expected_image_count
        ):
            failures.append(
                f"initial path has {initial_path.shape[0]} images, expected "
                f"{int(args.expected_image_count)}"
            )
        if (
            args.expected_initial_path_sha256 is not None
            and initial_path_sha256 != args.expected_initial_path_sha256
        ):
            failures.append(
                "initial path hash differs from the declared experiment input"
            )
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
        stored_paths = np.asarray(saved["stored_paths"], dtype=float)
        call_to_history = {int(call): index for index, call in enumerate(history_calls)}
        if len(call_to_history) != len(history_calls):
            failures.append("calls_per_image contains duplicate labels after runner deduplication")
        if any(int(call) not in call_to_history for call in stored_calls):
            failures.append("stored path call label has no matching metric record")
        changed = path_change_indices(stored_paths)
        raw_calls = stored_calls[changed]
        raw_iperp = np.asarray(
            [history_iperp[call_to_history[int(call)]] for call in raw_calls],
            dtype=float,
        )
        raw_spacing_cv = np.asarray(
            [history_spacing_cv[call_to_history[int(call)]] for call in raw_calls],
            dtype=float,
        )
        active_alpha = np.asarray(saved["active_alpha"], dtype=float)
        nominal_alpha = float(scalar(saved, "alpha"))
        if args.expected_alpha is not None:
            expected_alpha = float(args.expected_alpha)
            if not alpha_equal(nominal_alpha, expected_alpha):
                failures.append(
                    f"nominal alpha {nominal_alpha:g} differs from expected {expected_alpha:g}"
                )
        else:
            expected_alpha = nominal_alpha
        if not all(alpha_equal(float(value), expected_alpha) for value in active_alpha):
            failures.append("active_alpha is not fixed at the declared alpha")
        policy = POLICIES[args.policy]
        actual = {
            "solver": str(scalar(saved, "solver")),
            "alpha_mode": str(scalar(saved, "alpha_mode")),
            "reparam_mode": str(scalar(saved, "reparam_mode")),
            "state_globalization": str(scalar(saved, "state_globalization")),
            "kkt_rtol": float(scalar(saved, "plbfgs_kkt_rtol")),
            "inner_calls": int(scalar(saved, "inner_calls")),
            "direct_trial_calls": int(scalar(saved, "state_curve_direct_trial_calls")),
            "direct_objective_acceptance": str(
                scalar(saved, "state_curve_direct_objective_acceptance")
            ),
            "stall_kkt_target": float(
                scalar(saved, "state_kkt_stall_tighten_target")
            ),
            "kkt_iperp_trigger": float(
                scalar(saved, "state_kkt_iperp_tighten_trigger")
                if "state_kkt_iperp_tighten_trigger" in saved.files
                else -1.0
            ),
            "stop_iperp": float(scalar(saved, "stop_iperp")),
        }
        expected = {
            "solver": "state_ivm",
            "alpha_mode": "fixed",
            "reparam_mode": "state_curve_projected",
            "state_globalization": policy["state_globalization"],
            "kkt_rtol": float(policy.get("kkt_rtol", 0.7)),
            "inner_calls": int(policy.get("inner_calls", 30)),
            "direct_trial_calls": int(policy.get("inner_calls", 30)),
            "direct_objective_acceptance": policy.get(
                "direct_objective_acceptance", "per_image"
            ),
            "stall_kkt_target": float(policy.get("stall_kkt_target", -1.0)),
            "kkt_iperp_trigger": float(policy.get("kkt_iperp_trigger", -1.0)),
            "stop_iperp": float(policy.get("stop_iperp", -1.0)),
        }
        for name, value in expected.items():
            if isinstance(value, float):
                matches = math.isclose(actual[name], value, rel_tol=0.0, abs_tol=1e-14)
            else:
                matches = actual[name] == value
            if not matches:
                failures.append(f"{name}={actual[name]!r}, expected {value!r}")
        active_kkt = np.asarray(saved["active_plbfgs_kkt_rtol"], dtype=float)
        allowed_kkt = {expected["kkt_rtol"]}
        if expected["stall_kkt_target"] > 0.0:
            allowed_kkt.add(expected["stall_kkt_target"])
        if any(
            not any(math.isclose(value, item, rel_tol=0.0, abs_tol=1e-14)
                    for item in allowed_kkt)
            for value in active_kkt
        ):
            failures.append("active KKT tolerance contains an undeclared value")
        if np.any(np.diff(active_kkt) > 1e-14):
            failures.append("active KKT tolerance loosens after tightening")

    iperp_events = upward_events(
        raw_calls,
        raw_iperp,
        atol=args.iperp_atol,
        rtol=args.iperp_rtol,
        start_call=args.initial_transient_max_call,
        relative_to_best=False,
    )
    if iperp_events:
        failures.append(f"raw accepted I_perp increases {len(iperp_events)} times")
    if float(np.max(raw_spacing_cv)) > float(args.max_spacing_cv):
        failures.append(
            f"raw accepted path spacing CV reaches {float(np.max(raw_spacing_cv)):.6g}, "
            f"above the declared limit {float(args.max_spacing_cv):.6g}"
        )

    dense_report = None
    if args.dense is not None:
        dense_path = Path(args.dense).expanduser().resolve()
        dense = load_dense(dense_path)
        if Path(dense["source_result"]).resolve() != result_path:
            failures.append("dense diagnostic was generated from a different result")
        if np.any(np.diff(dense["calls"]) <= 0):
            failures.append("dense diagnostic calls are not strictly increasing")
        barrier_events = upward_events(
            dense["calls"],
            dense["barrier"],
            atol=args.dense_tolerance_kcal,
            rtol=0.0,
            start_call=args.initial_transient_max_call,
            relative_to_best=True,
        )
        if barrier_events:
            failures.append(
                "densified barrier increases beyond the declared "
                f"{float(args.dense_tolerance_kcal):g} kcal/mol resolution "
                f"{len(barrier_events)} times"
            )
        dense_report = {
            "path": str(dense_path),
            "records": int(len(dense["calls"])),
            "calls": [int(item) for item in dense["calls"]],
            "minimum_kcal": float(np.min(dense["barrier"])),
            "final_kcal": float(dense["barrier"][-1]),
            "max_final_level_difference_kcal": float(np.max(dense["resolution"])),
            "upward_events": barrier_events,
        }
    elif args.require_dense:
        failures.append("independent dense-barrier diagnostic is required but absent")

    return {
        "result": str(result_path),
        "policy": args.policy,
        "expected_alpha": "inf" if math.isinf(expected_alpha) else expected_alpha,
        "passed": not failures,
        "failures": failures,
        "force_counter_audit": force_report,
        "raw_accepted_path": {
            "records": int(len(raw_calls)),
            "calls": [int(item) for item in raw_calls],
            "minimum_i_perp": float(np.min(raw_iperp)),
            "final_i_perp": float(raw_iperp[-1]),
            "maximum_spacing_cv": float(np.max(raw_spacing_cv)),
            "final_spacing_cv": float(raw_spacing_cv[-1]),
            "upward_events": iperp_events,
        },
        "dense_barrier": dense_report,
        "criterion": {
            "initial_transient_max_call": int(args.initial_transient_max_call),
            "iperp_atol": float(args.iperp_atol),
            "iperp_rtol": float(args.iperp_rtol),
            "dense_tolerance_kcal": float(args.dense_tolerance_kcal),
            "max_spacing_cv": float(args.max_spacing_cv),
            "expected_image_count": args.expected_image_count,
            "expected_initial_path_sha256": args.expected_initial_path_sha256,
            "raw_trajectory_only": True,
            "dense_barrier_guard": "no cumulative rise from the prior running minimum beyond tolerance",
            "diagnostic_cost_excluded_from_optimization_axis": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument(
        "--policy",
        choices=tuple(POLICIES),
        required=True,
    )
    parser.add_argument("--expected-alpha", type=float)
    parser.add_argument("--dense")
    parser.add_argument("--require-dense", action="store_true")
    parser.add_argument("--initial-transient-max-call", type=int, default=5)
    parser.add_argument("--iperp-atol", type=float, default=1e-12)
    parser.add_argument("--iperp-rtol", type=float, default=1e-10)
    parser.add_argument("--dense-tolerance-kcal", type=float, default=0.05)
    parser.add_argument("--max-spacing-cv", type=float, default=1e-3)
    parser.add_argument("--expected-image-count", type=int)
    parser.add_argument("--expected-initial-path-sha256")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(args)
    output = Path(args.root).expanduser().resolve() / "reports" / args.output
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
