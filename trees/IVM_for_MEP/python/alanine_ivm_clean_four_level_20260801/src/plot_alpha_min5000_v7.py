#!/usr/bin/env python3
"""Plot the uncensored fixed-alpha audit with every raw trajectory above NF=5000."""

from __future__ import annotations

import json
from pathlib import Path

from plot_alpha_full_budget_v6 import (
    configure_style,
    load_stage1_minimum_force,
    load_stage4_minimum_force,
    make_audit,
    make_budget_frontier,
    make_full_history,
    write_data,
)


MINIMUM_TERMINAL_FORCE = 5000
VERSION = "v7_min5000"
ARCHIVE_PREFIX = "clean_stage4_v7_min5000"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "figures/clean_four_level_audit/alpha_min5000_v7"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    stage1 = load_stage1_minimum_force(
        root,
        minimum_terminal_force=MINIMUM_TERMINAL_FORCE,
        minimum_macro_budget=1000,
    )
    stage4 = load_stage4_minimum_force(
        root,
        minimum_terminal_force=MINIMUM_TERMINAL_FORCE,
        minimum_phase_b=600,
        archive_prefix=ARCHIVE_PREFIX,
    )
    if not all(item.raw_force[-1] > MINIMUM_TERMINAL_FORCE for item in stage1 + stage4):
        raise RuntimeError("At least one plotted raw trajectory does not exceed NF=5000")

    outputs = []
    outputs += make_full_history(
        stage1,
        stage4,
        output_dir,
        version=VERSION,
        stage1_title=r"Stage I: exact-like, no resampling (all $N_F>5000$)",
        stage4_title=r"Stage IV: optimized inexact + resampling (all $N_F>5000$)",
    )
    outputs += make_budget_frontier(
        stage1,
        stage4,
        output_dir,
        version=VERSION,
        complete_audit=True,
    )
    outputs += make_audit(
        stage1,
        stage4,
        output_dir,
        version=VERSION,
        stage1_title=r"Stage I full history (all $N_F>5000$)",
        stage4_title=r"Stage IV uncensored history (all $N_F>5000$)",
        complete_audit=True,
    )
    csv_path, manifest_path = write_data(
        stage1,
        stage4,
        output_dir,
        version=VERSION,
        stage1_protocol=(
            "exact-like IVM; r=0.1; no resampling; same update policy; "
            "minimum macro budget 1000 and alpha=1 extended to 2500 so every "
            "reported trajectory exceeds 5000 whole-path force evaluations"
        ),
        stage4_protocol=(
            "uniform optimized inexact IVM + resampling; Phase A=34; Phase B>=600; "
            "alpha=1 and alpha=10 extended to Phase B=1000; complete-I_perp early "
            "stop disabled; every raw trajectory exceeds 5000 force evaluations; "
            "reference-free barrier10 screen continues after first target and all "
            "screen work is charged"
        ),
    )
    result = {
        "minimum_terminal_force": MINIMUM_TERMINAL_FORCE,
        "all_raw_trajectories_exceed_minimum": True,
        "outputs": [str(path) for path in outputs],
        "data": str(csv_path),
        "manifest": str(manifest_path),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
