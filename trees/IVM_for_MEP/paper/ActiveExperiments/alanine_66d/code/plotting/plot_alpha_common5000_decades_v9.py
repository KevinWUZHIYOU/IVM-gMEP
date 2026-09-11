#!/usr/bin/env python3
"""Redraw the NF=5000 audit using only decade-spaced fixed alpha values."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from plot_alpha_common5000_v8 import (
    COMMON_FORCE_CUTOFF,
    truncate_stage1,
    truncate_stage4,
    validate_common_cutoff,
)
from plot_alpha_full_budget_v6 import (
    alpha_tag,
    configure_style,
    load_stage1_minimum_force,
    load_stage4_minimum_force,
    make_audit,
    make_budget_frontier,
    make_full_history,
    write_data,
)


DECADE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
DECADE_LABELS = ("1e-2", "1e-1", "1e0", "1e1", "1e2", "1e3", "1e4")
VERSION = "v9_cut5000_decades"
ARCHIVE_PREFIX = "clean_stage4_v7_min5000"


def filter_decades(records):
    selected = [
        item
        for item in records
        if any(np.isclose(item.alpha, alpha) for alpha in DECADE_ALPHAS)
    ]
    observed = tuple(alpha_tag(item.alpha) for item in selected)
    expected = tuple(alpha_tag(alpha) for alpha in DECADE_ALPHAS)
    if observed != expected:
        raise RuntimeError(f"Decade alpha grid mismatch: {observed} != {expected}")
    return selected


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "figures/clean_four_level_audit/alpha_common5000_decades_v9"
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()

    stage1_all = truncate_stage1(
        load_stage1_minimum_force(
            root,
            minimum_terminal_force=COMMON_FORCE_CUTOFF,
            minimum_macro_budget=1000,
        )
    )
    stage4_all = truncate_stage4(
        load_stage4_minimum_force(
            root,
            minimum_terminal_force=COMMON_FORCE_CUTOFF,
            minimum_phase_b=600,
            archive_prefix=ARCHIVE_PREFIX,
        )
    )
    validate_common_cutoff(stage1_all, stage4_all)
    stage1 = filter_decades(stage1_all)
    stage4 = filter_decades(stage4_all)

    raw_endpoint_label = r"common $N_F=5000$ cutoff"
    charged_endpoint_label = r"common charged $N_F=5000$ cutoff"
    outputs = []
    outputs += make_full_history(
        stage1,
        stage4,
        output_dir,
        version=VERSION,
        stage1_title=r"Stage I: exact-like, no resampling (decade $\alpha$; $N_F=5000$)",
        stage4_title=r"Stage IV: optimized inexact + resampling (decade $\alpha$; $N_F=5000$)",
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
        stage1_title=r"Stage I history: decade $\alpha$ through $N_F=5000$",
        stage4_title=r"Stage IV history: decade $\alpha$ through $N_F=5000$",
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
            "v8 common-NF5000 Stage-I records; display restricted to the decade "
            "alpha grid 1e-2 through 1e4"
        ),
        stage4_protocol=(
            "v8 common charged-NF5000 Stage-IV records; display restricted to the "
            "decade alpha grid 1e-2 through 1e4; reference-free screen costs retained"
        ),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "status": "complete_common_nf5000_decade_alpha_redraw",
            "purpose": (
                "Reduce visual clutter by retaining only fixed alpha values at "
                "integer powers of ten while preserving the exact v8 cutoff data."
            ),
            "alpha_grid": list(DECADE_LABELS),
            "selected_alpha": None,
            "common_force_cutoff": COMMON_FORCE_CUTOFF,
            "all_curve_endpoints_equal_cutoff": True,
            "source_cutoff_version": "v8_cut5000",
            "new_dynamics_run": False,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "alpha_grid": list(DECADE_LABELS),
                "common_force_cutoff": COMMON_FORCE_CUTOFF,
                "outputs": [str(path) for path in outputs],
                "data": str(csv_path),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
