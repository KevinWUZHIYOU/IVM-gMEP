#!/usr/bin/env python3
"""Render the call-515 constrained-relaxation diagnostic evolution plot.

This script consumes frozen arrays only.  It never constructs an OpenMM context
and makes zero energy or force calls.  The frozen 15 x 9 pilot is extended by
the independently audited n=0.72 top row to form a 15 x 10 diagnostic surface.
The surface is not fully relaxed; that limitation is recorded in provenance
without obscuring the plotted energy field.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
R2_ANALYSIS = HERE / "bb_discrete32_analysis_r2.py"
R2_MANIFEST = HERE / "ANALYSIS_INPUTS_R2.final.json"
PILOT_ROOT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
    "runs/diagnostics/"
    "bb_irivm_call515_nf14983_terminal_uncommitted_tail_"
    "constrained_relaxed_pilot_15x9_20260903_r1"
)
PILOT_RESULT = PILOT_ROOT / "result.npz"
PILOT_AUDIT = PILOT_ROOT / "audit.json"
TOP_ROW_ROOT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
    "runs/diagnostics/"
    "bb_irivm_call515_nf14983_constrained_relaxed_toprow_"
    "n0p72_15s_20260903_r1"
)
TOP_ROW_RESULT = TOP_ROW_ROOT / "result.npz"
TOP_ROW_AUDIT = TOP_ROW_ROOT / "audit.json"
COORDINATE_MANIFEST = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
    "runs/diagnostics/"
    "bb_irivm_call515_nf14983_terminal_uncommitted_tail_"
    "constrained_relaxed_pilot_coordinates_15x9_20260903_r1b.npz"
)
IR_SHARD_ROOT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
    "runs/full_state_history_replay_r5b/"
    "bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b_observer/shards"
)
ODE_RUN_ROOT = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/"
    "bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/runs/"
    "fresh-nonresume-bb-neb-aseode-k0p1-formal-nf30000-remote-20260903-r1"
)
ODE_SNAPSHOT_ROOT = ODE_RUN_ROOT / "snapshots"
ODE_COMPLETE = ODE_RUN_ROOT / "OPTIMIZATION_COMPLETE.json"
DEFAULT_OUTPUT = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/"
    "barnase_barstar_irivm_constrained_relaxed_evolution.pdf"
)
EV_TO_KCAL = 23.06054783061903
COLOR_SCALE = 40.0
DEFAULT_COLOR_CURVATURE = 2.0
FILLED_CONTOUR_BANDS = 28  # Previous 40 bands multiplied by the requested 0.7.
REFERENCE_STYLE_PDF = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/"
    "barnase_branch_local_true0k_evolution.pdf"
)
REFERENCE_STYLE_PDF_SHA256 = "26625a3f4a3a23c4d4302be6283e8a07879d7ca6fe569abec276f2e0ee8f9394"
N_ATOMS = 3132
EXPECTED_RESULT_SHA256 = (
    "b1bc1f8e2fc45867c3cb61a0d9e4cff6e006e7b396500b1fa9c6f9e94f11d39e"
)
EXPECTED_COORDINATE_SHA256 = (
    "1e0acc34ff629d3bb3c5eb7aa633d6af72eeaf8289aa98205acb8882caa754ec"
)
PILOT_RESULT_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-result-r1-v1"
PILOT_AUDIT_SCHEMA = "bb-irivm-call515-constrained-relaxed-pilot-audit-r1-v1"
TOP_ROW_SCHEMA = "bb-irivm-call515-constrained-relaxed-toprow-n0p72-r1-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_binding(
    record: Any, role: str, expected_path: Path | None = None,
) -> Path:
    """Validate a canonical {path, sha256, bytes} frozen-file binding."""
    require(isinstance(record, dict), f"{role} binding is not an object")
    require(set(record) == {"path", "sha256", "bytes"},
            f"{role} binding fields differ")
    path = Path(str(record["path"])).expanduser().resolve(strict=True)
    if expected_path is not None:
        expected = expected_path.expanduser().resolve(strict=True)
        require(path == expected, f"{role} binding path differs")
    require(path.stat().st_size == int(record["bytes"]),
            f"{role} binding byte count differs")
    require(sha256_file(path) == str(record["sha256"]),
            f"{role} binding hash differs")
    return path


def validate_wal_accounting(
    path: Path, role: str, expected_calls: int, expected_points: int,
    expected_result: Path, event_style: str,
) -> dict[str, int]:
    """Cross-check the sealed physical-call ledger, not merely its file hash."""
    lines = path.read_text(encoding="utf-8").splitlines()
    require(lines, f"{role} WAL is empty")
    header = json.loads(lines[0])
    require(header.get("optimization_N_F_contribution") == 0,
            f"{role} WAL optimization accounting differs")
    if event_style == "pilot15x9":
        started_name, succeeded_name = "attempt_started", "attempt_succeeded"
        failed_name, point_name, published_name = (
            "attempt_failed", "point_completed", "result_published"
        )
        identity_key, point_key = "attempt_id", "point_id"
    elif event_style == "toprow":
        started_name, succeeded_name = "started", "succeeded"
        failed_name, point_name, published_name = (
            "failed", "point_complete", "published"
        )
        identity_key, point_key = "id", "point"
    else:
        raise RuntimeError(f"unknown WAL event style: {event_style}")

    started: set[str] = set()
    succeeded: set[str] = set()
    completed_points: set[int] = set()
    terminal_reasons: list[str] = []
    completed_call_sum = 0
    published = 0
    old_ordinals: list[int] = []
    for encoded in lines[1:]:
        event = json.loads(encoded)
        kind = event.get("event")
        if kind == started_name:
            identity = str(event[identity_key])
            require(identity not in started, f"{role} WAL duplicates a started call")
            started.add(identity)
            if event_style == "pilot15x9":
                old_ordinals.append(int(event["physical_force_call_ordinal"]))
        elif kind == succeeded_name:
            identity = str(event[identity_key])
            require(identity in started and identity not in succeeded,
                    f"{role} WAL has an orphan/duplicate successful call")
            succeeded.add(identity)
        elif kind == failed_name:
            raise RuntimeError(f"{role} WAL contains a failed physical call")
        elif kind == point_name:
            point = int(event[point_key])
            require(point not in completed_points,
                    f"{role} WAL duplicates a completed point")
            completed_points.add(point)
            terminal_reasons.append(str(event["terminal_reason"] if event_style
                                        == "pilot15x9" else event["reason"]))
            completed_call_sum += int(event["physical_force_calls"] if event_style
                                      == "pilot15x9" else event["calls"])
        elif kind == published_name:
            published += 1
            verify_binding(event.get("result"), f"{role} published result", expected_result)
        else:
            raise RuntimeError(f"{role} WAL contains unknown event {kind!r}")

    require(started == succeeded and len(started) == expected_calls,
            f"{role} WAL started/succeeded call accounting differs")
    require(len(completed_points) == expected_points
            and completed_call_sum == expected_calls,
            f"{role} WAL completed-point accounting differs")
    require(published == 1, f"{role} WAL publication accounting differs")
    if event_style == "pilot15x9":
        require(old_ordinals == list(range(1, expected_calls + 1)),
                f"{role} WAL physical-call ordinals are not contiguous")
    return {
        "calls": len(started),
        "points": len(completed_points),
        "converged": terminal_reasons.count("converged"),
        "capped": terminal_reasons.count("maximum_update_steps"),
        "cached": terminal_reasons.count("fixed_endpoint_cached_below_threshold"),
    }


def scalar(saved: Any, key: str) -> Any:
    value = saved[key]
    return value.item() if value.ndim == 0 else value


def load_module():
    spec = importlib.util.spec_from_file_location("bb_discrete32_r2_for_relaxed15", R2_ANALYSIS)
    require(spec is not None and spec.loader is not None, "cannot load R2 analysis module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_ir_snapshots() -> dict[int, dict[str, Any]]:
    inventory = []
    for path in sorted(IR_SHARD_ROOT.glob("call_*_stable_last.npz")):
        with np.load(path, allow_pickle=False) as saved:
            inventory.append((
                int(scalar(saved, "optimization_N_F")),
                int(scalar(saved, "call_per_image")),
                path,
            ))
    require(len(inventory) == 515, "IR-IVM full-state shard inventory differs")
    selected = {}
    expected = {2000: (1993, 72), 10000: (9985, 352), 15000: (14983, 515)}
    for target, expected_identity in expected.items():
        eligible = [row for row in inventory if row[0] <= target]
        require(eligible, f"no IR-IVM state exists at or below NF{target}")
        actual_nf, call_per_image, path = max(eligible)
        require((actual_nf, call_per_image) == expected_identity,
                f"IR-IVM state selected for target NF{target} differs")
        with np.load(path, allow_pickle=False) as saved:
            require(bool(scalar(saved, "fully_evaluated_current_path")),
                    f"IR-IVM NF{actual_nf} path is not fully evaluated")
            require(int(scalar(saved, "additional_physical_calls")) == 0,
                    f"IR-IVM NF{actual_nf} shard reports extra physical calls")
            path32 = np.asarray(saved["path32_A"], dtype=np.float64).reshape(32, 3132, 3)
            require(str(scalar(saved, "path_sha256"))
                    == hashlib.sha256(path32.tobytes()).hexdigest(),
                    f"IR-IVM NF{actual_nf} path hash differs")
        selected[target] = {
            "target_N_F": target,
            "optimization_N_F": actual_nf,
            "call_per_image": call_per_image,
            "path32": path32,
            "path": path,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "committed_by_final_reparameterization": call_per_image <= 318,
        }
    return selected


def load_ode_snapshots() -> dict[int, dict[str, Any]]:
    complete = json.loads(ODE_COMPLETE.read_text(encoding="utf-8"))
    require(complete.get("status") == "OPTIMIZATION_COMPLETE"
            and int(complete.get("paid_nf", -1)) == 30000
            and int(complete.get("endpoint_physical_calls", -1)) == 0,
            "ODE-NEB completion identity differs")
    snapshot_hashes = complete.get("snapshot_sha256", {})
    expected = {
        2000: (2040, 68, "nf02040.npz"),
        10000: (9990, 333, "nf09990.npz"),
        30000: (30000, 1000, "nf30000.npz"),
    }
    selected = {}
    for target, (actual_nf, batch_id, filename) in expected.items():
        path = ODE_SNAPSHOT_ROOT / filename
        require(sha256_file(path) == snapshot_hashes.get(str(actual_nf)),
                f"ODE-NEB NF{actual_nf} snapshot seal hash differs")
        with np.load(path, allow_pickle=False) as saved:
            require(int(scalar(saved, "optimization_N_F")) == actual_nf
                    and int(scalar(saved, "batch_id")) == batch_id,
                    f"ODE-NEB snapshot selected for target NF{target} differs")
            require(int(scalar(saved, "evaluated_current")) == 1,
                    f"ODE-NEB NF{actual_nf} snapshot is not evaluated current")
            require(str(scalar(saved, "method"))
                    == "ase.mep.neb.NEB+NEBOptimizer(method=ODE)"
                    and str(scalar(saved, "tangent"))
                    == "tau_i = normalize(R_(i+1) - R_(i-1))"
                    and float(scalar(saved, "spring_k_eV_per_A2")) == 0.1,
                    f"ODE-NEB NF{actual_nf} method identity differs")
            path32 = np.asarray(saved["nodes"], dtype=np.float64).reshape(32, 3132, 3)
            require(np.all(np.isfinite(path32)),
                    f"ODE-NEB NF{actual_nf} path contains non-finite values")
        selected[target] = {
            "target_N_F": target,
            "optimization_N_F": actual_nf,
            "batch_id": batch_id,
            "path32": path32,
            "path": path,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    return selected


def load_inputs(module):
    require(sha256_file(PILOT_RESULT) == EXPECTED_RESULT_SHA256,
            "relaxed-pilot result hash differs")
    require(sha256_file(COORDINATE_MANIFEST) == EXPECTED_COORDINATE_SHA256,
            "relaxed-pilot coordinate-manifest hash differs")
    audit = json.loads(PILOT_AUDIT.read_text(encoding="utf-8"))
    require(audit.get("schema") == PILOT_AUDIT_SCHEMA
            and audit.get("status") == "PASS",
            "relaxed-pilot audit identity differs")
    verify_binding(audit.get("result"), "15x9 result", PILOT_RESULT)
    verify_binding(
        audit.get("coordinate_manifest"), "15x9 coordinate manifest",
        COORDINATE_MANIFEST,
    )
    for key in (
        "contract", "preauthorization_preflight", "authorization",
        "coordinate_audit", "endpoint_joint_ef_cache",
    ):
        verify_binding(audit.get(key), f"15x9 {key}")
    pilot_wal = verify_binding(audit.get("force_wal"), "15x9 force_wal")
    require(int(audit.get("independent_points", -1)) == 135
            and int(audit.get("fresh_relaxed_points", -1)) == 133
            and int(audit.get("fresh_converged_points", -1)) == 24
            and int(audit.get("max_step_points", -1)) == 109
            and int(audit.get("fixed_cached_endpoint_anchors", -1)) == 2,
            "15x9 audit point accounting differs")
    require(int(audit.get("physical_force_calls_total", -1)) == 6258
            and int(audit.get("physical_force_calls_succeeded", -1)) == 6258
            and int(audit.get("physical_force_calls_failed", -1)) == 0
            and int(audit.get("maximum_authorized_force_calls", -1)) == 6783
            and int(audit.get("energy_only_calls", -1)) == 0
            and int(audit.get("optimization_N_F_contribution", -1)) == 0,
            "15x9 audit physical-call accounting differs")
    pilot_wal_stats = validate_wal_accounting(
        pilot_wal, "15x9", 6258, 135, PILOT_RESULT, "pilot15x9"
    )
    require(pilot_wal_stats == {
        "calls": 6258, "points": 135, "converged": 24,
        "capped": 109, "cached": 2,
    }, "15x9 WAL terminal-state accounting differs")

    top_audit = json.loads(TOP_ROW_AUDIT.read_text(encoding="utf-8"))
    require(top_audit.get("schema") == TOP_ROW_SCHEMA + "-audit"
            and top_audit.get("status") == "PASS",
            "n=0.72 top-row audit identity differs")
    verify_binding(top_audit.get("result"), "n=0.72 result", TOP_ROW_RESULT)
    for key in ("contract", "preflight", "authorization"):
        verify_binding(top_audit.get(key), f"n=0.72 {key}")
    top_wal = verify_binding(top_audit.get("wal"), "n=0.72 wal")
    top_audit_calls = int(top_audit.get("physical_force_calls_total", -1))
    require(int(top_audit.get("points", -1)) == 15
            and int(top_audit.get("maximum_force_calls", -1)) == 765
            and int(top_audit.get("optimization_N_F_contribution", -1)) == 0,
            "n=0.72 audit point/budget accounting differs")
    top_wal_stats = validate_wal_accounting(
        top_wal, "n=0.72", top_audit_calls, 15, TOP_ROW_RESULT, "toprow"
    )

    manifest = module.BASE_MODULE.load_json(R2_MANIFEST)
    require(manifest.get("status") == "FINAL_INPUTS_FROZEN",
            "R2 comparison inputs are not frozen")
    ir_snapshots = load_ir_snapshots()
    ode_snapshots = load_ode_snapshots()

    with np.load(PILOT_RESULT, allow_pickle=False) as saved:
        require(str(scalar(saved, "schema")) == PILOT_RESULT_SCHEMA,
                "relaxed-pilot result schema differs")
        require(str(scalar(saved, "status")) == "PASS",
                "relaxed-pilot result status differs")
        require(Path(str(scalar(saved, "coordinate_manifest"))).resolve(strict=True)
                == COORDINATE_MANIFEST.resolve(strict=True)
                and str(scalar(saved, "coordinate_manifest_sha256"))
                == EXPECTED_COORDINATE_SHA256,
                "relaxed-pilot result coordinate binding differs")
        pilot = {key: np.asarray(saved[key]) for key in (
            "s_values", "n_values_rms_A", "final_energy_eV",
            "final_projected_force_eV_A", "converged", "terminal_reason",
            "fixed_endpoint_anchor_mask", "physical_force_calls_per_point",
        )}
        pilot["physical_force_calls_total"] = int(
            scalar(saved, "physical_force_calls_total")
        )
        pilot["optimization_N_F_contribution"] = int(
            scalar(saved, "optimization_N_F_contribution")
        )
        pilot["formally_reportable"] = bool(scalar(saved, "formally_reportable"))
        pilot["maximum_authorized_force_calls"] = int(
            scalar(saved, "maximum_authorized_force_calls")
        )
        pilot["held_host_locks"] = np.asarray(saved["held_host_locks"])

    with np.load(COORDINATE_MANIFEST, allow_pickle=False) as saved:
        require(str(scalar(saved, "schema"))
                == "bb-irivm-call515-constrained-relaxed-pilot-"
                   "coordinate-manifest-r1b-v1"
                and str(scalar(saved, "status"))
                == "PASS_ZERO_PHYSICS_FIXED_15X9_PILOT",
                "relaxed-pilot coordinate identity differs")
        coordinate = {key: np.asarray(saved[key]) for key in (
            "s_values", "n_values_rms_A", "initial_coordinates_A", "fixed_nu",
            "source_s_indices",
        )}
        source_manifest = Path(str(scalar(saved, "source_coordinate_manifest")))
        source_manifest_sha256 = str(scalar(saved, "source_coordinate_manifest_sha256"))

    require(sha256_file(source_manifest) == source_manifest_sha256,
            "source 59x28 coordinate-manifest hash differs")

    with np.load(source_manifest, allow_pickle=False) as saved:
        source = {key: np.asarray(saved[key], dtype=np.float64) for key in (
            "common_initial_path59_A", "center_path59_A",
        )}

    s = np.asarray(pilot["s_values"], dtype=np.float64)
    n = np.asarray(pilot["n_values_rms_A"], dtype=np.float64)
    energy = np.asarray(pilot["final_energy_eV"], dtype=np.float64)
    reasons = np.asarray(pilot["terminal_reason"])
    converged = np.asarray(pilot["converged"], dtype=bool)
    anchors = np.asarray(pilot["fixed_endpoint_anchor_mask"], dtype=bool)
    indices = np.asarray(coordinate["source_s_indices"], dtype=np.int64)
    zero_index = int(np.argmin(np.abs(n)))
    center = np.asarray(coordinate["initial_coordinates_A"][:, zero_index], dtype=np.float64)
    normals = np.asarray(coordinate["fixed_nu"], dtype=np.float64)
    initial = np.asarray(source["common_initial_path59_A"][indices], dtype=np.float64)

    require(s.shape == (15,) and n.shape == (9,) and energy.shape == (15, 9),
            "relaxed-pilot grid dimensions differ")
    require(center.shape == normals.shape == initial.shape == (15, N_ATOMS, 3),
            "relaxed-pilot projection arrays differ")
    require(np.array_equal(coordinate["s_values"], s)
            and np.array_equal(coordinate["n_values_rms_A"], n),
            "relaxed-pilot result/coordinate grids differ")
    require(np.all(np.isfinite(energy)), "relaxed-pilot energy contains non-finite values")
    require(int(np.count_nonzero(converged & ~anchors)) == 24,
            "fresh convergence count differs")
    require(int(np.count_nonzero(reasons == "maximum_update_steps")) == 109,
            "maximum-step count differs")
    require(int(np.count_nonzero(anchors)) == 2, "cached endpoint count differs")
    require(pilot["physical_force_calls_total"] == 6258,
            "physical force-call count differs")
    require(int(np.sum(pilot["physical_force_calls_per_point"])) == 6258
            and pilot["maximum_authorized_force_calls"] == 6783,
            "15x9 result force-call accounting differs")
    require(pilot["optimization_N_F_contribution"] == 0,
            "pilot unexpectedly contributes optimization N_F")
    require(pilot["formally_reportable"] is False,
            "pilot formal-reportability flag differs")

    with np.load(TOP_ROW_RESULT, allow_pickle=False) as saved:
        require(str(scalar(saved, "schema")) == TOP_ROW_SCHEMA + "-result"
                and str(scalar(saved, "status")) == "PASS",
                "n=0.72 top-row result identity differs")
        top = {key: np.asarray(saved[key]) for key in (
            "s_values", "initial_coordinates_A", "final_coordinates_A",
            "final_energy_eV", "final_raw_force_eV_A",
            "final_projected_force_eV_A", "held_host_locks",
        )}
        top["n_value_rms_A"] = float(scalar(saved, "n_value_rms_A"))
        top["physical_force_calls_total"] = int(
            scalar(saved, "physical_force_calls_total")
        )
        top["optimization_N_F_contribution"] = int(
            scalar(saved, "optimization_N_F_contribution")
        )

    require(top["s_values"].shape == (15,)
            and np.array_equal(top["s_values"], s)
            and top["n_value_rms_A"] == 0.72,
            "n=0.72 top-row grid differs")
    require(top["initial_coordinates_A"].shape == (15, N_ATOMS, 3)
            and top["final_coordinates_A"].shape == (15, N_ATOMS, 3)
            and top["final_energy_eV"].shape == (15,)
            and top["final_raw_force_eV_A"].shape == (15, N_ATOMS, 3)
            and top["final_projected_force_eV_A"].shape == (15, N_ATOMS, 3),
            "n=0.72 top-row array dimensions differ")
    expected_top_initial = np.ascontiguousarray(
        coordinate["initial_coordinates_A"][:, -1]
        + (0.72 - 0.69) * math.sqrt(N_ATOMS) * normals,
        dtype=np.float64,
    )
    require(np.array_equal(top["initial_coordinates_A"], expected_top_initial),
            "n=0.72 initial-coordinate construction differs")
    for key in (
        "initial_coordinates_A", "final_coordinates_A", "final_energy_eV",
        "final_raw_force_eV_A", "final_projected_force_eV_A",
    ):
        require(np.all(np.isfinite(top[key])),
                f"n=0.72 {key} contains non-finite values")
    top_calls = top["physical_force_calls_total"]
    require(top_calls == int(top_audit.get("physical_force_calls_total", -1))
            and 15 <= top_calls <= 765
            and top["optimization_N_F_contribution"] == 0
            and list(top["held_host_locks"]) == top_audit.get("held_host_locks"),
            "n=0.72 result/audit physical-call accounting differs")

    top_fmax = np.max(
        np.linalg.norm(top["final_projected_force_eV_A"], axis=2), axis=1
    )
    top_converged = top_fmax <= 0.10
    top_reasons = np.where(top_converged, "converged", "maximum_update_steps")
    top_capped = int(np.count_nonzero(~top_converged))
    require(top_wal_stats["calls"] == top_calls
            and top_wal_stats["converged"] == int(np.count_nonzero(top_converged))
            and top_wal_stats["capped"] == top_capped
            and top_wal_stats["cached"] == 0,
            "n=0.72 WAL/result terminal-state accounting differs")

    pilot["final_energy_eV"] = np.column_stack((energy, top["final_energy_eV"]))
    pilot["converged"] = np.column_stack((converged, top_converged))
    pilot["terminal_reason"] = np.column_stack((reasons, top_reasons))
    pilot["fixed_endpoint_anchor_mask"] = np.column_stack((
        anchors, np.zeros(15, dtype=bool),
    ))
    pilot["grid_shape"] = (15, 10)
    pilot["pilot15x9_physical_force_calls"] = 6258
    pilot["top_row_physical_force_calls"] = top_calls
    pilot["physical_force_calls_total"] = 6258 + top_calls
    pilot["pilot15x9_fresh_converged"] = 24
    pilot["pilot15x9_capped"] = 109
    pilot["top_row_fresh_converged"] = int(np.count_nonzero(top_converged))
    pilot["top_row_capped"] = top_capped
    pilot["fresh_converged"] = 24 + int(np.count_nonzero(top_converged))
    pilot["capped"] = 109 + top_capped
    n = np.concatenate((n, np.asarray([0.72], dtype=np.float64)))
    require(pilot["final_energy_eV"].shape == (15, 10)
            and np.array_equal(n, np.asarray([
                -0.12, -0.06, 0.0, 0.12, 0.24, 0.36, 0.48, 0.60, 0.69, 0.72,
            ])),
            "assembled 15x10 relaxed diagnostic grid differs")
    require(pilot["fresh_converged"] + pilot["capped"] + 2 == 150,
            "assembled 15x10 terminal-state accounting differs")

    return (
        ir_snapshots, ode_snapshots, pilot, s, n, center, normals, initial,
        audit, top_audit, source_manifest,
    )


def plot(
    module, ir_snapshots, ode_snapshots, pilot, s, n, center, normals, initial,
    color_curvature: float,
):
    base = module.BASE_MODULE
    base.configure_style()
    require(sha256_file(REFERENCE_STYLE_PDF) == REFERENCE_STYLE_PDF_SHA256,
            "reference PDF used for direct stroke-style audit has changed")
    require(math.isfinite(color_curvature) and color_curvature > 0.0,
            "color curvature must be finite and positive")
    energy = np.asarray(pilot["final_energy_eV"], dtype=np.float64)
    zero_index = int(np.argmin(np.abs(n)))
    reference_eV = float(energy[0, zero_index])
    relative = np.maximum((energy - reference_eV) * EV_TO_KCAL, 0.0)
    maximum = float(np.max(relative))
    require(maximum > 0.0, "relaxed-pilot energy range is empty")
    norm = base.LogQuadraticNorm(
        0.0, maximum, scale=COLOR_SCALE, curvature=color_curvature
    )
    filled_levels = norm.inverse(np.linspace(0.0, 1.0, FILLED_CONTOUR_BANDS + 1))
    sm, nm = np.meshgrid(s, n, indexing="ij")

    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.38), sharex=True, sharey=True)
    initial_n = np.sum((initial - center) * normals, axis=(1, 2)) / math.sqrt(N_ATOMS)
    initial_n[[0, -1]] = 0.0
    require(float(initial_n.min()) >= -0.12 - 2.0e-8
            and float(initial_n.max()) <= 0.72 + 2.0e-8,
            "common initial path leaves the assembled 15 x 10 envelope")

    contour = None
    panels = (
        None,
        (2000, 2000, "IR-IVM 2k | ODE-NEB 2k"),
        (10000, 10000, "IR-IVM 10k* | ODE-NEB 10k"),
        (15000, 30000, "IR-IVM 15k* | ODE-NEB 30k"),
    )
    for panel, axis in enumerate(axes):
        contour = axis.contourf(
            sm, nm, relative, levels=filled_levels, cmap="cividis",
            norm=norm, antialiased=False,
        )
        # PDF viewers can introduce dashed-looking seams between adjacent vector
        # fill polygons even with antialiased=False. A fill-only PDF reproduces
        # them without any white contour strokes. Render only this background at
        # the 600 dpi save resolution; contours, paths, axes and text stay vector.
        if hasattr(contour, "set_rasterized"):
            contour.set_rasterized(True)
        else:  # Matplotlib versions that expose individual contour collections.
            for collection in contour.collections:
                collection.set_rasterized(True)
        axis.contour(
            # Directly audited from the 2026-07-24 reference PDF content stream:
            # /A3 CA=.28, .25 w, 1 G. Later plotting scripts differ from this PDF.
            # Use the actual filled-contour boundaries, not an independently
            # spaced set: every internal color-band boundary gets one line.
            sm, nm, relative, levels=contour.levels[1:-1], colors="white",
            linewidths=0.25, linestyles="solid", alpha=0.28,
        )

        if panel == 0:
            axis.plot(
                s, initial_n, color=base.INITIAL_COLOR, linewidth=1.55,
                marker="o", markersize=2.15, markevery=1,
                label="Initial path", zorder=5,
            )
            title = "Initial path"
        else:
            ir_target, ode_target, title = panels[panel]
            ir_state = ir_snapshots[ir_target]
            ode = ode_snapshots[ode_target]
            for path32, color, linewidth, marker, label, zorder in (
                (ode["path32"], base.ODE_COLOR, 1.45, "o", "ODE-NEB", 5),
                (ir_state["path32"], base.IVM_COLOR, 1.75, "s", "IR-IVM", 6),
            ):
                projected = base.project_path(path32, center, s, normals)
                require(projected[:, 1].min() >= -0.12 - 2.0e-8
                        and projected[:, 1].max() <= 0.72 + 2.0e-8,
                        f"{label} path leaves the 15 x 10 diagnostic envelope")
                axis.plot(
                    projected[:, 0], projected[:, 1], color=color, marker=marker,
                    markevery=2, markersize=2.15, linewidth=linewidth,
                    label=label, zorder=zorder,
                )

        axis.set_title(title, fontsize=6.8, pad=3.0)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-0.12, 0.72)
        axis.set_xticks((0.0, 0.5, 1.0))
        axis.tick_params(direction="out", length=2.8)
        axis.text(
            0.035, 0.965, f"({chr(97 + panel)})", transform=axis.transAxes,
            va="top", ha="left", fontsize=8.5, fontweight="bold",
        )
        for spine in axis.spines.values():
            spine.set_linewidth(0.8)

    for axis in axes[:-1]:
        axis.get_xticklabels()[-1].set_visible(False)
    axes[0].set_ylabel(r"RMS path-normal $n$ ($\AA$)")
    figure.supxlabel(r"path progress $s$", x=0.48, y=0.035)
    handles = [
        Line2D([], [], color=base.ODE_COLOR, marker="o", markersize=2.15,
               markeredgecolor="black", markeredgewidth=0.35,
               linewidth=1.45, label="ODE-NEB"),
        Line2D([], [], color=base.IVM_COLOR, marker="s", markersize=2.15,
               markeredgecolor="black", markeredgewidth=0.35,
               linewidth=1.75, label="IR-IVM"),
    ]
    axes[-1].legend(
        handles=handles, frameon=True, facecolor="white", edgecolor="#B3B3B3",
        framealpha=0.90, loc="upper right", handlelength=2.0,
    )

    color_axis = figure.add_axes([0.905, 0.23, 0.014, 0.67])
    colorbar = figure.colorbar(contour, cax=color_axis)
    tick_candidates = ((0.0, "0"), (40.0, "40"), (240.0, "240"),
                       (1.0e3, r"$10^3$"), (1.0e4, r"$10^4$"),
                       (1.0e6, r"$10^6$"))
    ticks = [value for value, _label in tick_candidates if value <= maximum]
    labels = [label for value, label in tick_candidates if value <= maximum]
    if maximum > ticks[-1] * 1.05:
        # Avoid crowding a near-maximum decade tick against the exact top tick.
        if len(ticks) > 1 and float(norm(ticks[-1])) > 0.90:
            ticks.pop()
            labels.pop()
        exponent = int(math.floor(math.log10(maximum)))
        mantissa = maximum / (10.0 ** exponent)
        ticks.append(maximum)
        labels.append(rf"${mantissa:.1f}\times10^{{{exponent}}}$")
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(labels)
    colorbar.set_label(
        r"$E_{\mathrm{0K}}-E_{\mathrm{bound}}$ (kcal mol$^{-1}$)"
    )
    figure.subplots_adjust(
        left=0.075, right=0.885, bottom=0.23, top=0.90, wspace=0.08
    )
    return figure, {
        "reference_energy_eV": reference_eV,
        "display_maximum_kcal_per_mol": maximum,
        "stored_energy_values_changed": False,
        "color_norm": {
            "name": "LogQuadraticNorm", "scale": COLOR_SCALE,
            "curvature": color_curvature,
        },
        "filled_contour_levels": len(filled_levels),
        "filled_contour_bands": FILLED_CONTOUR_BANDS,
        "line_contour_levels": len(filled_levels) - 2,
        "white_contours_match_internal_fill_boundaries": True,
        "white_contour_levels_kcal_per_mol": filled_levels[1:-1].tolist(),
        "white_contour_style": {
            "source": "direct audit of reference PDF content/ExtGState, not later source version",
            "reference_pdf": str(REFERENCE_STYLE_PDF),
            "reference_pdf_sha256": REFERENCE_STYLE_PDF_SHA256,
            "color": "white", "linewidth_pt": 0.25,
            "alpha": 0.28, "linestyle": "solid",
        },
        "filled_contour_antialiased": False,
        "filled_contour_rasterized": True,
        "filled_contour_raster_dpi": 600,
        "white_contours_paths_axes_text_remain_vector": True,
        "fill_seam_diagnosis": "dashed-looking seams reproduced in fill-only vector PDF",
        "capped_points_overlaid_on_surface": False,
        "capped_points_retained_in_provenance": int(pilot["capped"]),
        "panel_budget_labels": [0, 2000, 10000, {"IR-IVM": 15000, "ODE-NEB": 30000}],
        "actual_ir_ivm_N_F": [0, 1993, 9985, 14983],
        "actual_ode_neb_N_F": [0, 2040, 9990, 30000],
        "initial_projection_maximum_rms_A": float(initial_n.max()),
        "sampled_surface_minimum_rms_A": float(n.min()),
        "sampled_surface_maximum_rms_A": float(n.max()),
        "plot_xlim": [0.0, 1.0],
        "plot_ylim_rms_A": [-0.12, 0.72],
        "initial_points_outside_sampled_surface": 0,
        "energy_extrapolated_outside_sampled_surface": False,
    }


def atomic_save_pdf(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.stage.", suffix=".pdf", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.04,
            metadata={
                "Title": "Barnase-Barstar constrained-relaxation 15 x 10 diagnostic",
                "Creator": "plot_bb_relaxed15x9_diagnostic_evolution.py",
            },
        )
        require(temporary.stat().st_size > 0, "staged PDF is empty")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--color-curvature", type=float, default=DEFAULT_COLOR_CURVATURE,
        help="positive LogQuadraticNorm curvature (default: 2)",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    module = load_module()
    (
        ir_snapshots, ode_snapshots, pilot, s, n, center, normals, initial,
        audit, top_audit, source_manifest,
    ) = load_inputs(module)
    figure, display = plot(
        module, ir_snapshots, ode_snapshots, pilot, s, n, center, normals, initial,
        args.color_curvature,
    )
    try:
        atomic_save_pdf(figure, output)
    finally:
        plt.close(figure)

    provenance_path = output.with_suffix(".provenance.json")
    provenance = {
        "schema": "bb-irivm-constrained-relaxed-diagnostic-evolution-v2",
        "status": "DIAGNOSTIC_NOT_FORMALLY_REPORTABLE",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scientific_interpretation": {
            "surface": (
                "frozen 15x9 constrained-relaxation pilot terminal energies "
                "plus independently audited n=0.72 constrained-relaxation top row"
            ),
            "grid_shape_s_by_n": [15, 10],
            "grid_points": 150,
            "fresh_grid_points": 148,
            "fresh_converged": int(pilot["fresh_converged"]),
            "maximum_update_steps": int(pilot["capped"]),
            "cached_fixed_endpoints": 2,
            "fully_relaxed_surface": False,
            "diagnostic_status": "PARTIALLY_CONVERGED_CONSTRAINED_RELAXATION_SURFACE",
            "nonconverged_grid_points_are_visually_overlaid": False,
            "nonconverged_grid_points_are_accounted_here": True,
            "parts": {
                "frozen_15x9": {
                    "grid_points": 135,
                    "fresh_grid_points": 133,
                    "fresh_converged": int(pilot["pilot15x9_fresh_converged"]),
                    "maximum_update_steps": int(pilot["pilot15x9_capped"]),
                    "cached_fixed_endpoints": 2,
                },
                "fresh_n0p72_top_row": {
                    "grid_points": 15,
                    "fresh_grid_points": 15,
                    "fresh_converged": int(pilot["top_row_fresh_converged"]),
                    "maximum_update_steps": int(pilot["top_row_capped"]),
                    "cached_fixed_endpoints": 0,
                    "convergence_criterion_projected_fmax_eV_per_A": 0.10,
                },
            },
            "terminal_uncommitted_center_path": True,
            "ir_ivm_N_F_9985_and_14983_state_role": (
                "charged fully evaluated returned-tail states not committed "
                "by the final reparameterization"
            ),
        },
        "accounting": {
            "frozen_15x9_physical_joint_energy_force_calls": int(
                pilot["pilot15x9_physical_force_calls"]
            ),
            "fresh_n0p72_top_row_physical_joint_energy_force_calls": int(
                pilot["top_row_physical_force_calls"]
            ),
            "combined_surface_physical_joint_energy_force_calls": int(
                pilot["physical_force_calls_total"]
            ),
            "combined_surface_maximum_authorized_calls": 6783 + 765,
            "energy_only_calls": 0,
            "optimization_N_F_contribution": 0,
            "plotting_physical_calls": 0,
        },
        "inputs": {
            "frozen_15x9_result": {
                "path": str(PILOT_RESULT), "sha256": sha256_file(PILOT_RESULT),
                "bytes": PILOT_RESULT.stat().st_size,
            },
            "frozen_15x9_audit": {
                "path": str(PILOT_AUDIT), "sha256": sha256_file(PILOT_AUDIT),
                "bytes": PILOT_AUDIT.stat().st_size, "status": audit.get("status"),
            },
            "fresh_n0p72_top_row_result": {
                "path": str(TOP_ROW_RESULT), "sha256": sha256_file(TOP_ROW_RESULT),
                "bytes": TOP_ROW_RESULT.stat().st_size,
            },
            "fresh_n0p72_top_row_audit": {
                "path": str(TOP_ROW_AUDIT), "sha256": sha256_file(TOP_ROW_AUDIT),
                "bytes": TOP_ROW_AUDIT.stat().st_size,
                "status": top_audit.get("status"),
            },
            "frozen_15x9_coordinates": {
                "path": str(COORDINATE_MANIFEST),
                "sha256": sha256_file(COORDINATE_MANIFEST),
                "bytes": COORDINATE_MANIFEST.stat().st_size,
            },
            "source_59x28_coordinates": {
                "path": str(source_manifest), "sha256": sha256_file(source_manifest),
                "bytes": source_manifest.stat().st_size,
            },
            "comparison_manifest": {
                "path": str(R2_MANIFEST), "sha256": sha256_file(R2_MANIFEST),
                "bytes": R2_MANIFEST.stat().st_size,
            },
            "ode_neb_optimization_complete": {
                "path": str(ODE_COMPLETE), "sha256": sha256_file(ODE_COMPLETE),
                "bytes": ODE_COMPLETE.stat().st_size,
            },
            **{
                f"ir_ivm_snapshot_target_N_F_{target}": {
                    "path": str(state["path"]),
                    "sha256": state["sha256"],
                    "bytes": state["bytes"],
                    "actual_optimization_N_F": state["optimization_N_F"],
                    "call_per_image": state["call_per_image"],
                    "committed_by_final_reparameterization": state[
                        "committed_by_final_reparameterization"
                    ],
                }
                for target, state in sorted(ir_snapshots.items())
            },
            **{
                f"ode_neb_snapshot_target_N_F_{target}": {
                    "path": str(state["path"]),
                    "sha256": state["sha256"],
                    "bytes": state["bytes"],
                    "actual_optimization_N_F": state["optimization_N_F"],
                    "batch_id": state["batch_id"],
                }
                for target, state in sorted(ode_snapshots.items())
            },
            "plot_code": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "bytes": Path(__file__).resolve().stat().st_size,
            },
        },
        "display": display,
        "output": {
            "path": str(output), "sha256": sha256_file(output),
            "bytes": output.stat().st_size,
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": provenance["status"],
        "pdf": str(output),
        "provenance": str(provenance_path),
        "plotting_physical_calls": 0,
    }, indent=2))


if __name__ == "__main__":
    main()
