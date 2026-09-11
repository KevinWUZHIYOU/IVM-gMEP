#!/usr/bin/env python3
"""Materialize the predeclared v3 online-stop prefix from completed v2 runs.

Stopping cannot influence a deterministic trajectory before the first state at
which the rule fires.  This exact-prefix replay is therefore useful while an
independent online rerun is pending, but its provenance is explicitly retained
and it must never be described as that independent rerun.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


TARGET = 1.5e-4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def label(alpha: float) -> str:
    if math.isinf(alpha):
        return "inf"
    return f"{alpha:g}".replace(".", "p").replace("-", "m")


def scalar(saved, key):
    return np.asarray(saved[key]).item()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result_dir = root / "runs/stage4_inexact_optimized_resample"
    sources = sorted(result_dir.glob("stage4_uniform_v2_a*_phaseA80_phaseB220.npz"))
    if len(sources) != 10:
        raise RuntimeError(f"Expected ten complete v2 sources, found {len(sources)}")
    reports = []
    for source in sources:
        with np.load(source, allow_pickle=False) as saved:
            alpha = float(saved["alpha"])
            phase_b_path = Path(str(scalar(saved, "phase_b_result")))
            if not phase_b_path.exists():
                raise FileNotFoundError(phase_b_path)
            phase_a_offset = int(saved["phase_transition_force_offset"])
        with np.load(phase_b_path, allow_pickle=False) as phase_b:
            eligible = np.flatnonzero(
                np.asarray(phase_b["reparam_committed"], dtype=bool)
                & (np.asarray(phase_b["i_perp"], dtype=float) <= TARGET)
            )
            if eligible.size:
                selected_phase_b_index = int(eligible[0])
                selected_phase_b_call = int(phase_b["calls_per_image"][selected_phase_b_index])
                selected_phase_b_force = int(
                    phase_b["total_internal_joint_evaluations"][selected_phase_b_index]
                )
                target_force = phase_a_offset + selected_phase_b_force
                fired = True
            else:
                selected_phase_b_index = len(phase_b["i_perp"]) - 1
                selected_phase_b_call = int(phase_b["calls_per_image"][-1])
                selected_phase_b_force = int(phase_b["total_internal_joint_evaluations"][-1])
                target_force = phase_a_offset + selected_phase_b_force
                fired = False

        with np.load(source, allow_pickle=False) as saved:
            force = np.asarray(saved["total_force_evaluations"], dtype=int)
            history_keep = np.flatnonzero(force <= target_force)
            if not history_keep.size or int(force[history_keep[-1]]) != target_force:
                raise RuntimeError(f"Target force {target_force} absent from {source}")
            stop = int(history_keep[-1]) + 1
            stored_force = np.asarray(saved["stored_force_evaluations"], dtype=int)
            stored_keep = np.flatnonzero(stored_force <= target_force)
            if not stored_keep.size or int(stored_force[stored_keep[-1]]) != target_force:
                raise RuntimeError(f"Target stored path {target_force} absent from {source}")
            stored_stop = int(stored_keep[-1]) + 1
            output = result_dir / (
                f"stage4_uniform_v3_a{label(alpha)}_phaseA80_phaseB220.npz"
            )
            np.savez_compressed(
                output,
                policy_version=np.array("stage4_uniform_v3"),
                alpha=np.array(alpha),
                alpha_mode=np.array("fixed_across_both_phases"),
                common_initial_path=np.array(str(scalar(saved, "common_initial_path"))),
                common_initial_path_sha256=np.array(str(scalar(saved, "common_initial_path_sha256"))),
                phase_labels=np.asarray(saved["phase_labels"]),
                phase_a_budget=np.array(int(saved["phase_a_budget"])),
                phase_b_budget=np.array(int(saved["phase_b_budget"])),
                phase_a_result=np.array(str(scalar(saved, "phase_a_result"))),
                phase_b_result=np.array(str(scalar(saved, "phase_b_result"))),
                phase_a_result_sha256=np.array(str(scalar(saved, "phase_a_result_sha256"))),
                phase_b_result_sha256=np.array(str(scalar(saved, "phase_b_result_sha256"))),
                phase_a_configuration_sha256=np.array(str(scalar(saved, "phase_a_configuration_sha256"))),
                phase_b_configuration_sha256=np.array(str(scalar(saved, "phase_b_configuration_sha256"))),
                phase_transition_macro_index=np.array(int(saved["phase_transition_macro_index"])),
                phase_transition_force_offset=np.array(phase_a_offset),
                macro_calls=np.asarray(saved["macro_calls"][:stop]),
                total_force_evaluations=force[:stop],
                i_perp=np.asarray(saved["i_perp"][:stop]),
                sparse_barrier_kcal=np.asarray(saved["sparse_barrier_kcal"][:stop]),
                physical_fmax=np.asarray(saved["physical_fmax"][:stop]),
                segment_length_cv=np.asarray(saved["segment_length_cv"][:stop]),
                equidistance_relative_defect=np.asarray(saved["equidistance_relative_defect"][:stop]),
                phase=np.asarray(saved["phase"][:stop]),
                stored_force_evaluations=stored_force[:stored_stop],
                stored_macro_calls=np.asarray(saved["stored_macro_calls"][:stored_stop]),
                stored_phase=np.asarray(saved["stored_phase"][:stored_stop]),
                stored_paths=np.asarray(saved["stored_paths"][:stored_stop]),
                final_path=np.asarray(saved["stored_paths"][stored_stop - 1]),
                raw_iperp_rises=np.array(
                    int(np.count_nonzero(np.diff(saved["i_perp"][:stop]) > 1e-12))
                ),
                stop_complete_iperp=np.array(TARGET),
                stop_rule_fired=np.array(fired),
                stop_reason=np.array(
                    "complete_reparameterized_i_perp_target"
                    if fired
                    else "maximum_phase_b_budget"
                ),
                selected_phase_b_call=np.array(selected_phase_b_call),
                selected_phase_b_force_evaluations=np.array(selected_phase_b_force),
                materialization_mode=np.array("exact_prefix_of_completed_v2_trajectory"),
                materialized_from=np.array(str(source)),
                materialized_from_sha256=np.array(file_sha256(source)),
                independent_online_rerun_status=np.array("pending_resource_release"),
            )
        report = {
            "alpha": "inf" if math.isinf(alpha) else alpha,
            "source": str(source),
            "source_sha256": file_sha256(source),
            "output": str(output),
            "output_sha256": file_sha256(output),
            "stop_rule_fired": fired,
            "selected_phase_b_call": selected_phase_b_call,
            "selected_total_force_evaluations": target_force,
            "selected_i_perp": float(np.load(output)["i_perp"][-1]),
            "materialization_mode": "exact_prefix_of_completed_v2_trajectory",
            "independent_online_rerun_status": "pending_resource_release",
        }
        output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        reports.append(report)
        print(json.dumps(report, sort_keys=True))
    report_path = root / "reports/stage4_v3_exact_prefix_materialization.json"
    report_path.write_text(
        json.dumps(
            {
                "online_rule": "first accepted reparam_committed macrostate with raw I_perp <= 1.5e-4",
                "causality": "the stopping decision cannot change any earlier deterministic state",
                "independent_online_rerun_required": True,
                "results": reports,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

