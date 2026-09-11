#!/usr/bin/env python3
"""Redraw the long alpha audit at one exact common NF=5000 cutoff.

The raw panels retain only completed optimization states at or before the
cutoff.  When the last completed state is below the cutoff, its incumbent is
carried horizontally to NF=5000.  The Stage-IV joint frontier admits only
fully screened candidates whose cumulative charged cost is at most 5000.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from plot_alpha_full_budget_v6 import (
    ALPHAS,
    EXPECTED_INITIAL_SHA256,
    TARGET_IPERP,
    AlphaRun,
    alpha_tag,
    configure_style,
    load_stage1_minimum_force,
    load_stage4_minimum_force,
    make_audit,
    make_budget_frontier,
    make_full_history,
    record_frontier,
    scalar,
    validate_alpha_grid,
    write_data,
)


COMMON_FORCE_CUTOFF = 5000
VERSION = "v8_cut5000"
ARCHIVE_PREFIX = "clean_stage4_v7_min5000"


def truncated_raw(item: AlphaRun) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Return completed raw states through cutoff plus an explicit cutoff point."""
    mask = np.asarray(item.raw_force) <= COMMON_FORCE_CUTOFF
    force = np.asarray(item.raw_force[mask], dtype=int)
    iperp = np.asarray(item.raw_iperp[mask], dtype=float)
    if force.size == 0:
        raise RuntimeError(f"No completed state before cutoff in {item.raw_source}")
    last_actual = int(force[-1])
    carried = last_actual < COMMON_FORCE_CUTOFF
    if carried:
        force = np.append(force, COMMON_FORCE_CUTOFF)
        iperp = np.append(iperp, iperp[-1])
    if int(force[-1]) != COMMON_FORCE_CUTOFF:
        raise RuntimeError(f"Cutoff endpoint mismatch in {item.raw_source}")
    return force, iperp, last_actual, carried


def truncate_stage1(long_records: list[AlphaRun]) -> list[AlphaRun]:
    records = []
    for item in long_records:
        force, iperp, last_actual, carried = truncated_raw(item)
        actual_count = len(force) - int(carried)
        frontier_force, frontier_iperp = record_frontier(
            force[:actual_count], iperp[:actual_count]
        )
        records.append(
            AlphaRun(
                family="stage1_exact_like_r0p1_no_resampling_commonNF5000",
                alpha=item.alpha,
                raw_source=item.raw_source,
                archive_source=None,
                raw_force=force,
                raw_iperp=iperp,
                frontier_force=frontier_force,
                frontier_iperp=frontier_iperp,
                terminal_charged_force=COMMON_FORCE_CUTOFF,
                terminal_best_iperp=float(frontier_iperp[-1]),
                raw_stop_reason="common_force_cutoff",
                target_reached=False,
                screen_force=0,
                initial_sha256=item.initial_sha256,
                protocol_budget=(
                    f"{item.protocol_budget}; deterministic completed-state prefix; "
                    f"common charged cutoff NF={COMMON_FORCE_CUTOFF}"
                ),
                common_force_cutoff=COMMON_FORCE_CUTOFF,
                last_actual_raw_force=last_actual,
                cutoff_carried_forward=carried,
            )
        )
    validate_alpha_grid(records, "Stage I common-cutoff")
    return records


def truncate_stage4(long_records: list[AlphaRun]) -> list[AlphaRun]:
    records = []
    for item in long_records:
        force, iperp, last_actual, carried = truncated_raw(item)
        actual_count = len(force) - int(carried)
        actual_force = force[:actual_count]
        actual_iperp = iperp[:actual_count]

        if item.archive_source is None:
            raise RuntimeError(f"Missing Stage-IV archive for alpha={item.alpha}")
        with np.load(item.archive_source, allow_pickle=False) as archive:
            candidate_calls = np.asarray(archive["candidate_calls"], dtype=int)
            candidate_iperp = np.asarray(archive["candidate_i_perp"], dtype=float)
            screen_cost = np.asarray(
                archive["candidate_screen_force_equivalent_cost"], dtype=int
            )
            accepted = np.asarray(archive["candidate_accepted"], dtype=bool)
            reference_used = bool(scalar(archive, "reference_barrier_used"))
        if reference_used:
            raise RuntimeError(f"Reference barrier leaked into {item.archive_source}")
        if not (
            len(candidate_calls)
            == len(candidate_iperp)
            == len(screen_cost)
            == len(accepted)
        ):
            raise RuntimeError(f"Invalid candidate archive in {item.archive_source}")

        charged_candidate = candidate_calls + np.cumsum(screen_cost)
        completed_screen = charged_candidate <= COMMON_FORCE_CUTOFF
        accepted_in_budget = completed_screen & accepted
        if candidate_calls.size:
            raw_prefix = actual_force < int(candidate_calls[0])
            joint_force = np.concatenate(
                (actual_force[raw_prefix], charged_candidate[accepted_in_budget])
            )
            joint_iperp = np.concatenate(
                (actual_iperp[raw_prefix], candidate_iperp[accepted_in_budget])
            )
        else:
            joint_force, joint_iperp = actual_force, actual_iperp
        frontier_force, frontier_iperp = record_frontier(joint_force, joint_iperp)
        if frontier_force.size == 0 or frontier_force[-1] > COMMON_FORCE_CUTOFF:
            raise RuntimeError(f"Invalid cutoff frontier in {item.archive_source}")

        target_force = (
            item.target_force
            if item.target_force is not None
            and item.target_force <= COMMON_FORCE_CUTOFF
            else None
        )
        target_iperp = item.target_iperp if target_force is not None else None
        target_reached = bool(
            target_force is not None
            and target_iperp is not None
            and target_iperp <= TARGET_IPERP
        )
        records.append(
            AlphaRun(
                family=(
                    "stage4_inexact_resample_reference_free_pareto_commonNF5000"
                ),
                alpha=item.alpha,
                raw_source=item.raw_source,
                archive_source=item.archive_source,
                raw_force=force,
                raw_iperp=iperp,
                frontier_force=frontier_force,
                frontier_iperp=frontier_iperp,
                terminal_charged_force=COMMON_FORCE_CUTOFF,
                terminal_best_iperp=float(frontier_iperp[-1]),
                raw_stop_reason="common_force_cutoff",
                target_reached=target_reached,
                screen_force=int(np.sum(screen_cost[completed_screen])),
                initial_sha256=item.initial_sha256,
                protocol_budget=(
                    f"{item.protocol_budget}; deterministic completed-state prefix; "
                    f"only fully screened candidates with total charged cost "
                    f"<= {COMMON_FORCE_CUTOFF}"
                ),
                target_force=target_force,
                target_iperp=target_iperp,
                common_force_cutoff=COMMON_FORCE_CUTOFF,
                last_actual_raw_force=last_actual,
                cutoff_carried_forward=carried,
            )
        )
    validate_alpha_grid(records, "Stage IV common-cutoff")
    return records


def validate_common_cutoff(stage1: list[AlphaRun], stage4: list[AlphaRun]) -> None:
    for item in stage1 + stage4:
        if (
            int(item.raw_force[-1]) != COMMON_FORCE_CUTOFF
            or item.terminal_charged_force != COMMON_FORCE_CUTOFF
            or np.any(item.raw_force > COMMON_FORCE_CUTOFF)
            or np.any(item.frontier_force > COMMON_FORCE_CUTOFF)
            or item.initial_sha256 != EXPECTED_INITIAL_SHA256
        ):
            raise RuntimeError(f"Common-cutoff validation failed for {item.family}")
    selected = next(item for item in stage4 if math.isclose(item.alpha, 20.0))
    if selected.target_force != 4808 or not selected.target_reached:
        raise RuntimeError("Expected alpha=20 joint target at charged NF=4808")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "figures/clean_four_level_audit/alpha_common5000_v8"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    stage1_long = load_stage1_minimum_force(
        root,
        minimum_terminal_force=COMMON_FORCE_CUTOFF,
        minimum_macro_budget=1000,
    )
    stage4_long = load_stage4_minimum_force(
        root,
        minimum_terminal_force=COMMON_FORCE_CUTOFF,
        minimum_phase_b=600,
        archive_prefix=ARCHIVE_PREFIX,
    )
    stage1 = truncate_stage1(stage1_long)
    stage4 = truncate_stage4(stage4_long)
    validate_common_cutoff(stage1, stage4)

    raw_endpoint_label = r"common $N_F=5000$ cutoff"
    charged_endpoint_label = r"common charged $N_F=5000$ cutoff"
    outputs = []
    outputs += make_full_history(
        stage1,
        stage4,
        output_dir,
        version=VERSION,
        stage1_title=r"Stage I: exact-like, no resampling (common $N_F=5000$)",
        stage4_title=r"Stage IV: optimized inexact + resampling (common $N_F=5000$)",
        raw_xmax=5500,
        raw_endpoint_label=raw_endpoint_label,
    )
    outputs += make_budget_frontier(
        stage1,
        stage4,
        output_dir,
        version=VERSION,
        complete_audit=True,
        endpoint_label=charged_endpoint_label,
        minimum_ymax=6500,
    )
    outputs += make_audit(
        stage1,
        stage4,
        output_dir,
        version=VERSION,
        stage1_title=r"Stage I history through common $N_F=5000$",
        stage4_title=r"Stage IV history through common $N_F=5000$",
        complete_audit=True,
        raw_xmax=5500,
        raw_endpoint_label=raw_endpoint_label,
        endpoint_label=charged_endpoint_label,
        minimum_ymax=6500,
    )
    csv_path, manifest_path = write_data(
        stage1,
        stage4,
        output_dir,
        version=VERSION,
        stage1_protocol=(
            "exact-like IVM; r=0.1; no resampling; identical policy across alpha; "
            "completed-state prefix through common NF=5000; last completed incumbent "
            "carried to the exact cutoff when necessary"
        ),
        stage4_protocol=(
            "uniform optimized inexact IVM + resampling; Phase A=34; Phase B>=600; "
            "identical policy across alpha; completed raw states through NF=5000; "
            "reference-free barrier10 screening; only fully screened candidates at "
            "cumulative charged cost <=5000 enter the joint Pareto frontier"
        ),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete_common_nf5000_cutoff_audit",
            "purpose": (
                "Compare every fixed-alpha trajectory at one exact NF=5000 "
                "budget without using any post-cutoff state."
            ),
            "common_force_cutoff": COMMON_FORCE_CUTOFF,
            "all_curve_endpoints_equal_cutoff": True,
            "source_audit_version": "v7_min5000",
            "stage4_cost_accounting": (
                "optimization calls plus completed reference-free screen work"
            ),
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "common_force_cutoff": COMMON_FORCE_CUTOFF,
                "all_curve_endpoints_equal_cutoff": True,
                "alpha20_first_joint_target_force": 4808,
                "outputs": [str(path) for path in outputs],
                "data": str(csv_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
