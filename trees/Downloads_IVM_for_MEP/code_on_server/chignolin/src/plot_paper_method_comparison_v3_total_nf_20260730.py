#!/usr/bin/env python3
"""Run the paper comparison with an exact total-force IVM truncation.

The upstream V2 plotter already uses total path force counts on every axis but
selected the IVM window indirectly by ``max calls/image <= 500``.  This wrapper
changes only that view policy: IVM is retained through complete operator-split
macrostates with audited total ``N_F <= 10,000``.  A late physical half-step
is not displayed until its geometry-only spring projection is committed.  The
non-IVM 1000-calls/image baselines and every plotting/scoring definition
remain unchanged.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


SOURCE = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/"
    "src/plot_paper_method_comparison_v2_20260729.py"
)
IVM_TOTAL_FORCE_LIMIT = 10000


def load_module():
    specification = importlib.util.spec_from_file_location(
        "chignolin_paper_comparison_v2", SOURCE
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load {SOURCE}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def argument_value(name: str, default=None):
    if name not in sys.argv:
        return default
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise RuntimeError(f"Missing value after {name}")
    return sys.argv[index + 1]


def enrich_summary() -> None:
    root = Path(argument_value("--root")).expanduser().resolve()
    output_directory = argument_value(
        "--output-directory",
        "figures/paper_methods_total_nf_20260727a",
    )
    result = Path(argument_value("--ivm-result")).expanduser().resolve()
    summary_path = (
        root / output_directory / "chignolin_method_comparison_summary.json"
    )
    summary = json.loads(summary_path.read_text())
    with np.load(result, allow_pickle=False) as saved:
        selected = summary["selected_ivm"]
        selected.update(
            {
                "relative_kkt_forcing_factor": float(
                    np.asarray(saved["plbfgs_kkt_rtol"]).item()
                ),
                "maximum_new_joint_evaluations_per_image_per_subproblem": int(
                    np.asarray(saved["inner_calls"]).item()
                ),
                "synchronized_trial_call_cap": int(
                    np.asarray(
                        saved["state_curve_direct_trial_calls"]
                    ).item()
                ),
                "projection_method": str(
                    np.asarray(
                        saved["state_curve_projection_method"]
                    ).item()
                ),
                "projection_uses_spline": bool(
                    np.asarray(
                        saved["state_curve_projection_uses_spline"]
                    ).item()
                ),
                "projection_openmm_calls": int(
                    np.asarray(
                        saved["state_curve_projection_openmm_calls"]
                    ).item()
                ),
                "free_polish_relative_defect_trigger": float(
                    np.asarray(
                        saved[
                            "state_free_polish_relative_defect_trigger"
                        ]
                    ).item()
                ),
                "free_polish_relative_defect_target": float(
                    np.asarray(
                        saved[
                            "state_free_polish_relative_defect_target"
                        ]
                    ).item()
                ),
                "complete_macrostate_policy": (
                    "During free polish, publish a physical half-step only "
                    "after its geometry-only spring correction is committed."
                ),
            }
        )
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, summary_path)


def main() -> None:
    module = load_module()
    original_load_run = module.load_run

    def load_run_total_force(path: Path, maximum_call: int = 500):
        if int(maximum_call) != 500:
            return original_load_run(path, maximum_call=maximum_call)
        full = original_load_run(path, maximum_call=10**9)
        total_force = np.asarray(
            full["number_of_force_evaluations"], dtype=np.int64
        )
        with np.load(Path(path).resolve(), allow_pickle=False) as saved:
            if (
                "state_free_polish_active" in saved.files
                and "reparam_committed" in saved.files
            ):
                free_polish = np.asarray(
                    saved["state_free_polish_active"], dtype=bool
                )
                spring_committed = np.asarray(
                    saved["reparam_committed"], dtype=bool
                )
                complete = (~free_polish) | spring_committed
            else:
                complete = np.ones(len(total_force), dtype=bool)
        if len(complete) != len(total_force):
            raise RuntimeError(
                f"IVM macrostate mask has wrong length for {path}"
            )
        keep = (total_force <= IVM_TOTAL_FORCE_LIMIT) & complete
        if not np.any(keep):
            raise RuntimeError(
                f"No IVM record at N_F <= {IVM_TOTAL_FORCE_LIMIT}: {path}"
            )
        length = len(total_force)
        cropped = {}
        for key, value in full.items():
            if (
                isinstance(value, np.ndarray)
                and value.ndim >= 1
                and value.shape[0] == length
            ):
                cropped[key] = value[keep]
            else:
                cropped[key] = value
        if (
            int(cropped["number_of_force_evaluations"][-1])
            > IVM_TOTAL_FORCE_LIMIT
        ):
            raise RuntimeError("IVM total-force truncation failed")
        return cropped

    module.load_run = load_run_total_force
    module.main()
    enrich_summary()


if __name__ == "__main__":
    main()
