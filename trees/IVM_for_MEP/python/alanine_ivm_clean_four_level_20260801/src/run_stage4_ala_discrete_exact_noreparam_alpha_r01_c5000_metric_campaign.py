#!/usr/bin/env python3
"""Audited densified metrics for the completed exact/no-reparam Ala alpha grid.

The optimizer inputs are the eight frozen successor-v2 runs with ``r=0.1``,
no reparameterization, a 5000-NF budget, and a nonbinding 5000-step
implementation safety cap.  Every state marked complete by
``exact_no_reparam_complete_state_mask`` is included.  Mesh CV and segment
length ratio are retained for diagnosis, but they are not an eligibility gate.

Execution and aggregation reuse the already-audited r=0.1 metric campaign.
This wrapper changes its protocol identity, registered sources, optimizer
loader, complete-state extractor, and aggregate mesh annotations inside a
short-lived context; the base module is restored before every public call
returns.  Densified energy evaluation remains 20/40/80/160 subdivisions per
original segment, selected at 160, with diagnostic evaluations excluded from
optimizer Number of Forces.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import run_stage4_ala_discrete_alpha_r01_c5000_metric_campaign as base
import run_stage4_ala_discrete_exact_noreparam_alpha_grid_member_v2 as member
import run_stage4_ala_discrete_grid_campaign_successor_v2 as upstream
import run_stage4_ala_discrete_grid_metric_campaign as metric_v2


PROTOCOL_ID = (
    "stage4_ala_discrete_exact_noreparam_alpha_r01_c5000_metric_campaign_v1"
)
EXPECTED_CELL_COUNT = 8
ALPHAS = tuple(float(10**power) for power in range(-1, 7))
RESIDUAL_RATIO = 0.1
INNER_MAX_STEPS = 5000
TOTAL_INTERNAL_NF_BUDGET = 5000
SECONDARY_FAMILY = upstream.SECONDARY_FAMILY

FROZEN_OPTIMIZER_PREREGISTRATION_RELATIVE = (
    "reports/stage4_ala_discrete_grid_campaign_successor_v2_"
    "1787604432554391000_preregistered.json"
)
FROZEN_OPTIMIZER_PREREGISTRATION_SHA256 = (
    "d3ebf3239931dfde104342b791d8ed096b4ddef597c1bee0de4cb32b4b289aa3"
)
FROZEN_OPTIMIZER_AGGREGATE_RELATIVE = (
    "reports/stage4_ala_discrete_grid_campaign_successor_v2_"
    "1787604432554391000_exact_no_reparam_aggregate.json"
)
FROZEN_OPTIMIZER_AGGREGATE_SHA256 = (
    "a2307bd55d7b165d07e3d83c61c6c5b41f5449b7db72d66f540ffaf74cf0556e"
)

SOURCE_PATHS = {
    "wrapper": (
        "src/run_stage4_ala_discrete_exact_noreparam_alpha_r01_c5000_"
        "metric_campaign.py"
    ),
    "base": "src/run_stage4_ala_discrete_alpha_r01_c5000_metric_campaign.py",
    "metric_v2": "src/run_stage4_ala_discrete_grid_metric_campaign.py",
    "upstream": "src/run_stage4_ala_discrete_grid_campaign_successor_v2.py",
    "member": (
        "src/run_stage4_ala_discrete_exact_noreparam_alpha_grid_member_v2.py"
    ),
    "runtime": (
        "src/run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim_v2.py"
    ),
    "campaign_infrastructure": "src/run_stage4_ala_discrete_grid_campaign.py",
    "ala_core": "src/ala_core.py",
    "topology": "inputs/ala2_fixed.pdb",
}

ELIGIBILITY_POLICY = {
    "formal_complete": "result.exact_no_reparam_complete_state_mask_is_true",
    "terminal_partial_state_excluded": True,
    "mesh_gate": "not_applied",
    "mesh_diagnostics_retained": {
        "segment_length_cv": "result.segment_length_cv",
        "segment_length_ratio": "result.segment_length_ratio",
        "mesh_eligible_annotation": {
            "segment_length_cv_max_inclusive": metric_v2.MESH_CV_MAX,
            "segment_length_ratio_max_inclusive": metric_v2.MESH_RATIO_MAX,
        },
    },
    "i_perp": "raw_result_field_no_recompute_no_cummin_no_interpolation",
}

DENSE_METRIC_POLICY = dict(base.DENSE_METRIC_POLICY)
ACCOUNTING_POLICY = dict(base.ACCOUNTING_POLICY)
CONCURRENCY_POLICY = dict(base.CONCURRENCY_POLICY)

MetricCampaignSafetyError = base.MetricCampaignSafetyError


def _require_bool_vector(
    saved: np.lib.npyio.NpzFile, key: str, size: int
) -> np.ndarray:
    raw = np.asarray(saved[key])
    if raw.dtype.kind != "b" or raw.shape != (size,):
        raise MetricCampaignSafetyError(f"{key} must be a boolean trajectory vector")
    return raw.astype(bool, copy=False)


def _extract_complete_states(
    result_path: Path,
    cell_id: str,
    expected_result_sha256: str,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Extract every exact-complete state and annotate, but do not gate, mesh."""

    result_path = result_path.expanduser().resolve()
    if (
        not result_path.is_file()
        or base.file_sha256(result_path) != expected_result_sha256
    ):
        raise MetricCampaignSafetyError(
            f"frozen exact/no-reparam result hash drifted: {result_path}"
        )
    required = {
        "calls_per_image",
        "stored_path_calls",
        "stored_paths",
        "total_internal_joint_evaluations",
        "i_perp",
        "segment_length_cv",
        "segment_length_ratio",
        "exact_no_reparam_complete_state_mask",
        "discrete_audit_stop_reason",
        "active_reparam_every",
        "reparam_attempted",
        "reparam_committed",
        "reparam_direct_joint_evaluations",
        "reparam_recovery_joint_evaluations",
    }
    with np.load(result_path, allow_pickle=False) as saved:
        missing = sorted(required.difference(saved.files))
        if missing:
            raise MetricCampaignSafetyError(
                f"exact metric input is missing {', '.join(missing)}: {result_path}"
            )
        cost = np.asarray(saved["total_internal_joint_evaluations"], dtype=np.int64)
        calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=np.int64)
        paths = np.asarray(saved["stored_paths"], dtype=np.float64)
        i_perp = np.asarray(saved["i_perp"], dtype=np.float64)
        cv = np.asarray(saved["segment_length_cv"], dtype=np.float64)
        ratio = np.asarray(saved["segment_length_ratio"], dtype=np.float64)
        size = int(cost.size)
        if size < 1 or any(
            vector.shape != (size,)
            for vector in (cost, calls, stored_calls, i_perp, cv, ratio)
        ):
            raise MetricCampaignSafetyError("exact metric trajectory is misaligned")
        if paths.shape != (size, metric_v2.N_IMAGES, metric_v2.PATH_DOF):
            raise MetricCampaignSafetyError("exact stored paths are misaligned")
        if not np.array_equal(calls, stored_calls):
            raise MetricCampaignSafetyError("stored path labels differ from call labels")
        if (
            cost[0] < 0
            or np.any(np.diff(cost) < 0)
            or np.any(cost > TOTAL_INTERNAL_NF_BUDGET)
        ):
            raise MetricCampaignSafetyError("exact optimization NF is invalid")
        if not (
            np.all(np.isfinite(i_perp))
            and np.all(i_perp >= 0.0)
            and np.all(np.isfinite(cv))
            and np.all(cv >= 0.0)
            and np.all(np.isfinite(ratio))
            and np.all(ratio >= 1.0)
        ):
            raise MetricCampaignSafetyError("raw I_perp or mesh diagnostics are invalid")

        complete = _require_bool_vector(
            saved, "exact_no_reparam_complete_state_mask", size
        )
        if not np.any(complete):
            raise MetricCampaignSafetyError("result has no exact-complete state")
        mesh = (cv <= metric_v2.MESH_CV_MAX) & (
            ratio <= metric_v2.MESH_RATIO_MAX
        )
        stop_reasons = np.asarray(saved["discrete_audit_stop_reason"], dtype="U64")
        cap_count = int(
            np.count_nonzero(stop_reasons == "evaluated-inner-max-steps")
        )
        if cap_count != 0:
            raise MetricCampaignSafetyError(
                "the 5000-step implementation safety cap constrained a solve"
            )
        active_reparam = np.asarray(saved["active_reparam_every"], dtype=np.int64)
        attempted = np.asarray(saved["reparam_attempted"], dtype=bool)
        committed = np.asarray(saved["reparam_committed"], dtype=bool)
        direct_nf = np.asarray(
            saved["reparam_direct_joint_evaluations"], dtype=np.int64
        )
        recovery_nf = np.asarray(
            saved["reparam_recovery_joint_evaluations"], dtype=np.int64
        )
        if (
            active_reparam.shape != (size,)
            or attempted.shape != (size,)
            or committed.shape != (size,)
            or direct_nf.shape != (size,)
            or recovery_nf.shape != (size,)
            or np.any(active_reparam != 0)
            or np.any(attempted)
            or np.any(committed)
            or np.any(direct_nf != 0)
            or np.any(recovery_nf != 0)
        ):
            raise MetricCampaignSafetyError(
                "an exact/no-reparam optimizer result contains reparameterization"
            )

        states: List[Dict[str, Any]] = []
        unique_paths: Dict[str, np.ndarray] = {}
        for index_value in np.flatnonzero(complete):
            index = int(index_value)
            path = metric_v2._canonical_path(paths[index])
            path_hash = metric_v2.path_content_sha256(path)
            previous = unique_paths.get(path_hash)
            if previous is not None and not np.array_equal(previous, path):
                raise MetricCampaignSafetyError("path SHA256 collision inside a run")
            unique_paths[path_hash] = path.copy()
            nf = int(cost[index])
            states.append(
                {
                    "state_id": metric_v2._state_id(
                        expected_result_sha256, index, nf, path_hash
                    ),
                    "trajectory_index": index,
                    "optimization_nf": nf,
                    "calls_per_image": int(calls[index]),
                    "raw_i_perp": float(i_perp[index]),
                    "segment_length_cv": float(cv[index]),
                    "segment_length_ratio": float(ratio[index]),
                    "formal_complete": True,
                    "mesh_eligible": bool(mesh[index]),
                    "path_sha256": path_hash,
                }
            )

    mesh_complete_count = int(np.count_nonzero(complete & mesh))
    cell = {
        "family": SECONDARY_FAMILY,
        "cell_id": str(cell_id),
        "source_result": str(result_path),
        "source_result_sha256": expected_result_sha256,
        "source_trajectory_record_count": size,
        "formal_complete_state_count": int(np.count_nonzero(complete)),
        "complete_mesh_eligible_state_count": mesh_complete_count,
        "complete_mesh_ineligible_state_count": int(
            np.count_nonzero(complete) - mesh_complete_count
        ),
        "complete_state_definition": "exact_no_reparam_complete_state_mask",
        "metric_eligibility_definition": (
            "all exact-complete states; mesh diagnostics retained but not gated"
        ),
        "raw_i_perp_field": "i_perp",
        "inner_cap_stop_evidence": {
            "source_field": "result.discrete_audit_stop_reason",
            "inner_cap_stop_token": "evaluated-inner-max-steps",
            "inner_cap_stop_count": 0,
            "inner_cap_binding": False,
            "stop_record_count": int(stop_reasons.size),
        },
        "reparameterization_evidence": {
            "active_reparam_every_unique": [0],
            "reparam_attempt_count": 0,
            "reparam_commit_count": 0,
            "reparam_direct_joint_evaluations": 0,
            "reparam_recovery_joint_evaluations": 0,
        },
        "states": states,
    }
    return cell, unique_paths


def _frozen_optimizer_path(root: Path, relative: str, expected_sha: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_file() or base.file_sha256(path) != expected_sha:
        raise MetricCampaignSafetyError(f"frozen optimizer input drifted: {path}")
    return path


def _load_optimizer_inputs(
    root: Path, preregistration_path: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], Dict[str, Any]]:
    preregistration_path = preregistration_path.expanduser().resolve()
    frozen_prereg = _frozen_optimizer_path(
        root,
        FROZEN_OPTIMIZER_PREREGISTRATION_RELATIVE,
        FROZEN_OPTIMIZER_PREREGISTRATION_SHA256,
    )
    if preregistration_path != frozen_prereg:
        raise MetricCampaignSafetyError(
            "formal metrics require the frozen successor-v2 preregistration"
        )
    frozen_aggregate = _frozen_optimizer_path(
        root,
        FROZEN_OPTIMIZER_AGGREGATE_RELATIVE,
        FROZEN_OPTIMIZER_AGGREGATE_SHA256,
    )
    registered = upstream.load_and_validate_preregistration(root, frozen_prereg)
    if registered.get("protocol_id") != upstream.PROTOCOL_ID:
        raise MetricCampaignSafetyError("optimizer protocol identity drifted")
    with upstream._successor_context():
        aggregate = upstream.legacy._validate_aggregate_file(
            root, frozen_prereg, registered, SECONDARY_FAMILY
        )
    aggregate_path = Path(
        str(registered["planned_outputs"]["secondary_aggregate"])
    ).resolve()
    if aggregate_path != frozen_aggregate or aggregate.get("status") != "complete":
        raise MetricCampaignSafetyError("exact/no-reparam aggregate is not frozen complete")
    if (
        int(aggregate.get("expected_cell_count", -1)) != EXPECTED_CELL_COUNT
        or int(aggregate.get("unique_cell_count", -1)) != EXPECTED_CELL_COUNT
    ):
        raise MetricCampaignSafetyError("exact/no-reparam aggregate coverage drifted")

    registered_cells = registered.get("cells", {}).get(SECONDARY_FAMILY)
    if not isinstance(registered_cells, list) or len(registered_cells) != EXPECTED_CELL_COUNT:
        raise MetricCampaignSafetyError("optimizer preregistration must contain eight exact runs")

    cells: List[Dict[str, Any]] = []
    paths: Dict[str, np.ndarray] = {}
    observed_alphas: List[float] = []
    for registered_cell in registered_cells:
        if not isinstance(registered_cell, dict):
            raise MetricCampaignSafetyError("optimizer run registration is malformed")
        controls = registered_cell.get("controls")
        canonical = registered_cell.get("canonical_scientific_spec")
        if not isinstance(controls, dict) or not isinstance(canonical, dict):
            raise MetricCampaignSafetyError("optimizer controls/specification are missing")
        alpha = float(controls.get("alpha", float("nan")))
        ratio = float(controls.get("residual_ratio", float("nan")))
        cap = int(controls.get("inner_max_steps", -1))
        budget = int(controls.get("total_internal_nf_budget", -1))
        if (
            not math.isfinite(alpha)
            or alpha not in ALPHAS
            or ratio != RESIDUAL_RATIO
            or cap != INNER_MAX_STEPS
            or budget != TOTAL_INTERNAL_NF_BUDGET
            or controls.get("scientific_maxstep") != "unbounded_subject_to_budget"
            or canonical.get("reparameterization") != "disabled"
            or canonical.get("inner_cap_required_nonbinding") is not True
            or canonical.get("runtime_policy_id") != member.runtime.POLICY_ID
            or canonical.get("runtime_implementation_policy_id")
            != member.runtime.IMPLEMENTATION_POLICY_ID
        ):
            raise MetricCampaignSafetyError("optimizer exact/no-reparam controls drifted")
        with upstream._successor_context():
            state, validated = upstream.legacy.inspect_cell(root, registered_cell)
        if state != "verified" or validated is None:
            raise MetricCampaignSafetyError("optimizer run artifact is not verified")
        result_path = Path(str(validated["result"])).resolve()
        result_sha = str(validated["result_sha256"])
        cell, cell_paths = _extract_complete_states(
            result_path, str(registered_cell["cell_id"]), result_sha
        )
        cell["controls"] = {
            "alpha": alpha,
            "residual_ratio": ratio,
            "inner_max_steps": cap,
            "scientific_maxstep": "unbounded_subject_to_budget",
            "reparameterization": "disabled",
            "total_internal_nf_budget": budget,
        }
        cell["member_audit"] = str(validated["member_audit"])
        cell["member_audit_sha256"] = str(validated["member_audit_sha256"])
        cell["scientific_command_sha256"] = str(
            validated["scientific_command_sha256"]
        )
        cells.append(cell)
        observed_alphas.append(alpha)
        for path_hash, path in cell_paths.items():
            previous = paths.get(path_hash)
            if previous is not None and not np.array_equal(previous, path):
                raise MetricCampaignSafetyError("path SHA256 collision across runs")
            paths[path_hash] = path
    if sorted(observed_alphas) != sorted(ALPHAS) or len(set(observed_alphas)) != len(ALPHAS):
        raise MetricCampaignSafetyError("optimizer alpha coverage is not exact")
    cells.sort(key=lambda item: float(item["controls"]["alpha"]))
    return cells, paths, {
        "preregistration": base._file_record(frozen_prereg),
        "aggregate": base._file_record(frozen_aggregate),
        "protocol_id": upstream.PROTOCOL_ID,
        "family": SECONDARY_FAMILY,
        "member_policy_version": member.MEMBER_POLICY_VERSION,
        "runtime_policy_id": member.runtime.POLICY_ID,
        "runtime_implementation_policy_id": member.runtime.IMPLEMENTATION_POLICY_ID,
        "aggregate_scientific_runtime_sha256": str(
            aggregate["scientific_runtime_sha256"]
        ),
    }


_BASE_REBUILD_AGGREGATE = base._rebuild_aggregate


def _rebuild_aggregate(
    root: Path,
    preregistration_path: Path,
    preregistration: Mapping[str, Any],
) -> Dict[str, Any]:
    payload = _BASE_REBUILD_AGGREGATE(
        root, preregistration_path, preregistration
    )
    state_lookup = {
        (str(cell["cell_id"]), int(state["trajectory_index"])): state
        for cell in preregistration["cells"]
        for state in cell["states"]
    }
    for row in payload["state_metrics"]:
        key = (str(row["cell_id"]), int(row["trajectory_index"]))
        row["mesh_eligible"] = bool(state_lookup[key]["mesh_eligible"])
    mesh_count = sum(bool(row["mesh_eligible"]) for row in payload["state_metrics"])
    payload["formal_complete_state_reference_count"] = len(payload["state_metrics"])
    payload["mesh_eligible_state_reference_count"] = mesh_count
    payload["mesh_ineligible_state_reference_count"] = (
        len(payload["state_metrics"]) - mesh_count
    )
    payload["mesh_gate_applied"] = False
    for summary in payload["alpha_summaries"]:
        rows = [
            row
            for row in payload["state_metrics"]
            if float(row["alpha"]) == float(summary["alpha"])
        ]
        summary["formal_complete_state_count"] = len(rows)
        summary["mesh_eligible_state_count"] = sum(
            bool(row["mesh_eligible"]) for row in rows
        )
        summary["mesh_ineligible_state_count"] = sum(
            not bool(row["mesh_eligible"]) for row in rows
        )
    return payload


@contextmanager
def _base_context() -> Iterator[None]:
    replacements = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "EXPECTED_CELL_COUNT": EXPECTED_CELL_COUNT,
        "ALPHAS": ALPHAS,
        "RESIDUAL_RATIO": RESIDUAL_RATIO,
        "INNER_MAX_STEPS": INNER_MAX_STEPS,
        "TOTAL_INTERNAL_NF_BUDGET": TOTAL_INTERNAL_NF_BUDGET,
        "SOURCE_PATHS": SOURCE_PATHS,
        "ELIGIBILITY_POLICY": ELIGIBILITY_POLICY,
        "DENSE_METRIC_POLICY": DENSE_METRIC_POLICY,
        "ACCOUNTING_POLICY": ACCOUNTING_POLICY,
        "CONCURRENCY_POLICY": CONCURRENCY_POLICY,
        "optimizer": upstream,
        "member": member,
        "_load_optimizer_inputs": _load_optimizer_inputs,
        "_rebuild_aggregate": _rebuild_aggregate,
    }
    previous = {name: getattr(base, name) for name in replacements}
    for name, value in replacements.items():
        setattr(base, name, value)
    try:
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def build_preregistration(
    root: Path,
    preregistration_path: Path,
    optimizer_preregistration_path: Path,
) -> Dict[str, Any]:
    with _base_context():
        return base.build_preregistration(
            root, preregistration_path, optimizer_preregistration_path
        )


def preregister(root: Path, optimizer_preregistration_path: Path) -> Path:
    with _base_context():
        return base.preregister(root, optimizer_preregistration_path)


def load_and_validate_preregistration(root: Path, path: Path) -> Dict[str, Any]:
    with _base_context():
        return base.load_and_validate_preregistration(root, path)


def run(
    root: Path,
    preregistration_path: Path,
    jobs: int = 1,
    wait_for_lock: bool = False,
) -> Path:
    with _base_context():
        return base.run(
            root,
            preregistration_path,
            jobs=jobs,
            wait_for_lock=wait_for_lock,
        )


def create_aggregate(root: Path, preregistration_path: Path) -> Path:
    with _base_context():
        return base.create_aggregate(root, preregistration_path)


def validate_aggregate(root: Path, preregistration_path: Path) -> Dict[str, Any]:
    with _base_context():
        return base.validate_aggregate(root, preregistration_path)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    preregister_parser = subparsers.add_parser("preregister")
    preregister_parser.add_argument(
        "--optimizer-preregistration", type=Path, required=True
    )
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--preregistration", type=Path, required=True)
    run_parser.add_argument("--jobs", type=int, default=1)
    run_parser.add_argument("--wait-for-lock", action="store_true")
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--preregistration", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--preregistration", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = argument_parser().parse_args(argv)
    if args.action == "preregister":
        path = preregister(args.root, args.optimizer_preregistration)
    elif args.action == "run":
        path = run(
            args.root,
            args.preregistration,
            jobs=args.jobs,
            wait_for_lock=args.wait_for_lock,
        )
    elif args.action == "aggregate":
        path = create_aggregate(args.root, args.preregistration)
    elif args.action == "validate":
        registered = load_and_validate_preregistration(
            args.root, args.preregistration
        )
        aggregate_path = Path(str(registered["planned_outputs"]["aggregate"]))
        if aggregate_path.exists():
            validate_aggregate(args.root, args.preregistration)
        path = args.preregistration.resolve()
    else:
        raise AssertionError("unreachable action")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
