#!/usr/bin/env python3
"""Freeze complete joint-E/F pack specs for the final BB cost figure.

This command is zero-OpenMM and performs the same array-level checks used by
the renderer before publishing an immutable figure contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Iterable


HERE = Path(__file__).resolve().parent
PLOTTER_PATH = HERE / "plot_bb_cost_to_precision_joint_ef_dense.py"


def _load_plotter():
    spec = importlib.util.spec_from_file_location("bb_joint_ef_cost_plotter_freeze", PLOTTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PLOT = _load_plotter()


def atomic_json_no_overwrite(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    parser.add_argument("--pack-spec", type=Path, action="append", required=True)
    parser.add_argument("--minimum-points-per-method", type=int, default=8)
    parser.add_argument("--maximum-points-per-method", type=int, default=12)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    destination = args.output.expanduser().resolve()
    figure = args.figure_output.expanduser().resolve()
    PLOT.require(destination.suffix.lower() == ".json", "figure contract must be JSON")
    PLOT.require(figure.suffix.lower() == ".pdf", "figure output must be PDF")
    PLOT.require(
        2 <= args.minimum_points_per_method <= args.maximum_points_per_method <= 64,
        "invalid point-count policy",
    )
    PLOT.require(not figure.exists(), "refusing to bind an existing figure output")

    pack_records = []
    points = {method: [] for method in PLOT.METHODS}
    packs = []
    for raw in args.pack_spec:
        pack_path = raw.expanduser().resolve(strict=True)
        pack, rows = PLOT.validate_pack_spec(pack_path)
        pack_records.append(PLOT.file_binding(pack_path))
        packs.append(pack)
        points[pack["method"]].extend(rows)
    for method in PLOT.METHODS:
        observed = len(points[method])
        PLOT.require(
            args.minimum_points_per_method <= observed <= args.maximum_points_per_method,
            f"{method} has {observed} points outside the requested density",
        )
        nf = [row["optimization_N_F"] for row in points[method]]
        PLOT.require(len(nf) == len(set(nf)), f"{method} repeats N_F across packs")
    topology = {pack["closure"]["topology"]["sha256"] for pack in packs}
    bb_core = {pack["closure"]["bb_core"]["sha256"] for pack in packs}
    PLOT.require(len(topology) == len(bb_core) == 1, "packs do not share topology/bb_core")
    _arrays, observed_analysis = PLOT.analyze_observed(points)

    payload = {
        "schema": PLOT.FIGURE_SCHEMA,
        "status": PLOT.FIGURE_STATUS,
        "created_utc": PLOT.utc_now(),
        "figure_output": str(figure),
        "figure_output_policy": {"overwrite": False, "sidecar_provenance": str(figure.with_suffix('.json'))},
        "point_count_policy": {
            "minimum_per_method": args.minimum_points_per_method,
            "maximum_per_method": args.maximum_points_per_method,
            "actual": {method: len(points[method]) for method in PLOT.METHODS},
        },
        "claim_policy": {
            "main_curve": "observed lower envelope",
            "show_all_raw_points_faintly": True,
            "interpolation": "forbidden",
            "first_passage_claim": "earliest observed first passage only",
            "visible_arrow_text": "numeric multiplier only",
            "arrow_text_position": "right of vertical arrow at arithmetic line midpoint",
        },
        "metric_protocol": {
            "discrete_metric": "I_perp_on_original_32_image_path",
            "densified_metric": "Delta_E_on_same_path_arc500",
            "coordinate_space": "unweighted_3N_cartesian",
            "sampling": "uniform_cartesian_arclength",
            "interpolation": "piecewise_linear",
            "grid_formula": "s_j=j*L/499,j=0,...,499",
            "points": PLOT.N_DENSE,
            "include_both_endpoints": True,
            "full_force_storage": True,
            "dense_force_use": "archival validation only",
            "optimization_N_F_contribution": 0,
        },
        "selection_policy": {
            "all_points_from_every_bound_pack_are_included": True,
            "post_metric_point_omission": "forbidden",
            "duplicate_optimization_N_F": "forbidden",
        },
        "observed_analysis_at_freeze": observed_analysis,
        "pack_specs": pack_records,
        "dependencies": {
            "plotter": PLOT.file_binding(PLOTTER_PATH),
            "freezer": PLOT.file_binding(Path(__file__).resolve()),
            "muller_alanine_style_reference": PLOT.file_binding(PLOT.STYLE_REFERENCE),
        },
        "zero_physical_freeze": {
            "OpenMM_contexts_constructed": 0,
            "energy_calls": 0,
            "force_calls": 0,
            "optimization_N_F_contribution": 0,
        },
    }
    if args.validate_only:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        return
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    atomic_json_no_overwrite(destination, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "contract": str(destination),
                "sha256": PLOT.sha256_file(destination),
                "figure_output": str(figure),
                "point_counts": payload["point_count_policy"]["actual"],
                "physical_calls": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
