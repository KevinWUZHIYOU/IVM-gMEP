#!/usr/bin/env python3
"""Preregister and compute the discrete Ala geometric-action diagnostic.

The optimizer artifacts are read-only.  Every selected 22-image path is
reevaluated with one persistent OpenMM CPU context.  These offline diagnostic
calls are counted separately and never added to the optimization NoF axis.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

for key in (
    "OPENMM_CPU_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[key] = "1"

import numpy as np


REPO = Path(__file__).resolve().parents[5]
STUDY = REPO / "python/alanine_ivm_clean_four_level_20260801"
LEGACY = REPO / "python/alanine_v10_chignolin_v9_fixed_alpha_20260730"
CORE_DIR = STUDY / "src"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from ala_core import (  # noqa: E402
    EV_TO_KCAL,
    TANGENT_DEFINITION,
    OpenMMBackend,
    central_tangents,
    local_path_weights,
    path_diagnostics,
)


FIGURE_MANIFEST = (
    REPO / "output/pdf/alanine66d_delta_aware_grid_ivm_replacement_1x2_preview_manifest.json"
)
FIGURE_SOURCE = (
    REPO / "output/pdf/alanine66d_delta_aware_grid_ivm_replacement_1x2_preview_source.csv"
)
GRID_PREREGISTRATION = (
    STUDY
    / "reports/stage4_ala_discrete_grid_metric_campaign_v2_1787608620070816000_preregistered.json"
)
CHIGNOLIN_EXAMPLE = (
    REPO / "python/chignolin_ir_ivm_20260826/src/postprocess_geometric_action.py"
)
TOPOLOGY = STUDY / "inputs/ala2_fixed.pdb"
SUPERSEDED_CONTRACTS = (
    STUDY / "reports/ala_discrete_geometric_action_v1_preregistered.json",
    STUDY / "reports/ala_discrete_geometric_action_v2_preregistered.json",
    STUDY / "reports/ala_discrete_geometric_action_v3_preregistered.json",
    STUDY / "reports/ala_discrete_geometric_action_v4_preregistered.json",
)
CONTRACT = STUDY / "reports/ala_discrete_geometric_action_v5_preregistered.json"
OUT = STUDY / "runs/diagnostics/ala_discrete_geometric_action_v5.npz"
HOST_LOCK = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.ala_global_heavy_task.lock")
CELL_ID = "a736dd230285d61c"

ODE_RESULT = LEGACY / "formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz"
ODE_METRICS = LEGACY / "formal_figure_inputs/metrics_ala_0to1_N22_k50_alpha0.0001_ode_results.npz"
BASELINE_RESULTS = {
    "static_neb": LEGACY / "runs/neb/neb_publication90k20260802r2_central_k30_alpha0p013_static_budget4500.npz",
    "original_string": LEGACY / "runs/publication_long90k_20260802/publication_long90k_20260802_publication90k20260802r2_original_nolambda_dt0p014_r1_cfg587b411dd523_budget4500.npz",
    "obsm_fire": LEGACY / "runs/publication_long90k_20260802/publication_long90k_20260802_publication90k20260802r3_fire_lam0p85_inner3_r1_nocap_cfg75c2b3203822_budget4510.npz",
    "obsm_bfgs": LEGACY / "runs/publication_long90k_20260802/publication_long90k_20260802_publication90k_robust_20260802r2_bfgs_lam0p3_inner1_r1_nocap_cfg9c7658b737c3_budget8000.npz",
}
METHOD_ORDER = (
    "ivm",
    "ode_neb",
    "static_neb",
    "original_string",
    "obsm_fire",
    "obsm_bfgs",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def discrete_action(path: np.ndarray, forces: np.ndarray):
    """Return the full-gradient trapezoid integral on the original polygon.

    This is intentionally identical to the Chignolin example: production
    central tangents are validated, interior weights are exactly
    ``local_path_weights``, and the complete physical force norm is used.
    Force and gradient norms are identical because ``force = -grad(V)``.
    """
    path = np.asarray(path, dtype=float)
    forces = np.asarray(forces, dtype=float)
    if path.shape != forces.shape or path.ndim != 2 or len(path) < 3:
        raise ValueError("Path/force shapes must match with at least three nodes")
    if not np.all(np.isfinite(path)) or not np.all(np.isfinite(forces)):
        raise ValueError("Non-finite geometry or force")
    central_tangents(path)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if np.any(lengths <= 0):
        raise ValueError("Zero-length segment")
    norms = np.linalg.norm(forces, axis=1)
    interior = float(np.dot(local_path_weights(path), norms[1:-1]))
    endpoint = float(0.5 * (lengths[0] * norms[0] + lengths[-1] * norms[-1]))
    segment_sum = float(np.dot(lengths, 0.5 * (norms[:-1] + norms[1:])))
    np.testing.assert_allclose(segment_sum, interior + endpoint, rtol=5e-15)
    return segment_sum, endpoint, norms, lengths


def _scalar(value):
    return np.asarray(value).item()


def _grid_cell() -> dict:
    payload = json.loads(GRID_PREREGISTRATION.read_text())
    cells = [row for row in payload["cells"] if row["cell_id"] == CELL_ID]
    if len(cells) != 1:
        raise RuntimeError(f"Expected one registered grid result for {CELL_ID}")
    cell = cells[0]
    if cell["family"] != "inexact_reparam" or cell["controls"] != {
        "alpha": 10.0,
        "residual_ratio": 0.7,
        "inner_max_steps": 5,
        "total_internal_nf_budget": 5000,
    }:
        raise RuntimeError("Selected IVM controls changed")
    return cell


def selected_rows() -> list[dict]:
    """Bind every diagnostic path to an optimization record and its NoF."""
    rows: list[dict] = []
    cell = _grid_cell()
    ivm_result = Path(cell["source_result"])
    ivm_result_hash = sha256(ivm_result)
    if ivm_result_hash != cell["source_result_sha256"]:
        raise RuntimeError("Selected IVM result hash changed")
    with np.load(ivm_result, allow_pickle=False) as saved:
        if _scalar(saved["tangent_definition"]) != TANGENT_DEFINITION:
            raise RuntimeError("IVM tangent definition changed")
        paths = np.asarray(saved["stored_paths"], dtype=float)
        for state in cell["states"]:
            if not state["formal_complete"] or not state["mesh_eligible"]:
                raise RuntimeError("IVM state is not eligible")
            index = int(state["trajectory_index"])
            path = paths[index]
            if array_sha256(path) != state["path_sha256"]:
                raise RuntimeError(f"IVM path hash mismatch at {index}")
            rows.append(
                dict(
                    method="ivm",
                    source_result=str(ivm_result),
                    source_result_sha256=ivm_result_hash,
                    source_record_index=index,
                    source_call=int(state["calls_per_image"]),
                    number_of_forces=int(state["optimization_nf"]),
                    i_perp=float(state["raw_i_perp"]),
                    path=path,
                    selected_path_sha256=state["path_sha256"],
                    eligibility="formal_complete_and_mesh_eligible",
                )
            )

    ode_result_hash = sha256(ODE_RESULT)
    with np.load(ODE_RESULT, allow_pickle=False) as saved, np.load(
        ODE_METRICS, allow_pickle=False
    ) as metrics:
        paths = np.asarray(saved["images_seq"], dtype=float)
        calls = np.asarray(saved["count_seq"], dtype=int)
        if not np.array_equal(calls, np.asarray(metrics["count_seq"], dtype=int)):
            raise RuntimeError("ODE path/metric call indices differ")
        iperp = np.asarray(metrics["loss_perp"], dtype=float)
        for index, (path, call, value) in enumerate(zip(paths, calls, iperp)):
            rows.append(
                dict(
                    method="ode_neb",
                    source_result=str(ODE_RESULT),
                    source_result_sha256=ode_result_hash,
                    source_record_index=index,
                    source_call=int(call),
                    number_of_forces=int(2 + 20 * call),
                    i_perp=float(value),
                    iperp_verification_applicable=False,
                    path=path,
                    selected_path_sha256=array_sha256(path),
                    eligibility="all_saved_states",
                )
            )

    for method, source in BASELINE_RESULTS.items():
        source_hash = sha256(source)
        with np.load(source, allow_pickle=False) as saved:
            if _scalar(saved["tangent_definition"]) != TANGENT_DEFINITION:
                raise RuntimeError(f"{method} tangent definition changed")
            paths = np.asarray(saved["stored_paths"], dtype=float)
            stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)
            calls = np.asarray(saved["calls_per_image"], dtype=int)
            costs = np.asarray(saved["counter_vectors"], dtype=int).sum(axis=1)
            iperp = np.asarray(saved["i_perp"], dtype=float)
            for path_index, (path, call) in enumerate(zip(paths, stored_calls)):
                matches = np.flatnonzero(calls == call)
                if not len(matches):
                    raise RuntimeError(f"Cannot bind {method} saved call {call}")
                # This is the legacy figure loader's exact policy: when a
                # cumulative call count repeats, bind the saved path to the
                # last history record at that call.
                trajectory_index = int(matches[-1])
                rows.append(
                    dict(
                        method=method,
                        source_result=str(source),
                        source_result_sha256=source_hash,
                        source_record_index=path_index,
                        source_call=int(call),
                        number_of_forces=int(costs[trajectory_index]),
                        i_perp=float(iperp[trajectory_index]),
                        iperp_verification_applicable=True,
                        path=path,
                        selected_path_sha256=array_sha256(path),
                        eligibility="all_saved_paths",
                    )
                )

    for row in rows:
        row.setdefault("iperp_verification_applicable", True)
        if row["path"].shape != (22, 66):
            raise RuntimeError(f"Unexpected Ala path shape: {row['path'].shape}")
        central_tangents(row["path"])
    if tuple(dict.fromkeys(row["method"] for row in rows)) != METHOD_ORDER:
        raise RuntimeError("Method order changed")
    return rows


def register() -> None:
    if CONTRACT.exists():
        raise FileExistsError(f"Refusing to overwrite {CONTRACT}")
    rows = selected_rows()
    source_files = {
        Path(__file__).resolve(),
        CORE_DIR / "ala_core.py",
        CHIGNOLIN_EXAMPLE,
        FIGURE_MANIFEST,
        FIGURE_SOURCE,
        GRID_PREREGISTRATION,
        TOPOLOGY,
        ODE_RESULT,
        ODE_METRICS,
        *BASELINE_RESULTS.values(),
        Path(_grid_cell()["source_result"]),
    }
    payload = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Additional discrete geometric-action cost-to-precision diagnostic; no optimizer rerun",
        "formula": "sum_i 0.5*(norm(gradV_i)+norm(gradV_(i+1)))*norm(x_(i+1)-x_i)",
        "gradient": "full physical gradient; force=-gradV so their Euclidean norms agree",
        "tangent": TANGENT_DEFINITION,
        "tangent_reference_example": str(CHIGNOLIN_EXAMPLE),
        "quadrature": "polygonal arc-length trapezoid",
        "interior_weights": "ala_core.local_path_weights; identical policy to the Chignolin example",
        "endpoint_policy": "include half-segment endpoint terms",
        "spatial_nodes": 22,
        "densification": False,
        "coordinate_space": "unweighted 66D Cartesian, Angstrom",
        "output_unit": "kcal/mol",
        "ev_to_kcal": EV_TO_KCAL,
        "selected_ivm_controls": _grid_cell()["controls"],
        "selected_ivm_cell_id": CELL_ID,
        "counts": {m: sum(row["method"] == m for row in rows) for m in METHOD_ORDER},
        "selected_path_count": len(rows),
        "expected_diagnostic_joint_evaluations": len(rows) * 22,
        "selected_paths_sha256": array_sha256(
            np.asarray([row["selected_path_sha256"] for row in rows])
        ),
        "jobs": 1,
        "openmm_cpu_threads": 1,
        "host_local_ala_lock": str(HOST_LOCK),
        "diagnostic_cost_excluded_from_optimization_axis": True,
        "optimization_cost": "source optimizer cumulative counter; all 22 images, including endpoints",
        "iperp_verification_rtol": 1e-5,
        "iperp_verification_atol": 1e-8,
        "iperp_verification_exclusion": {
            "method": "ode_neb",
            "source_record_indices": "all",
            "reason": "legacy ODE loss_perp predates the article's uniform central-tangent diagnostic; path forces are checked directly instead",
        },
        "ode_stored_force_verification": {
            "reference_field": "F_original",
            "verified_source_record_indices": "1 through 4637 inclusive",
            "excluded_source_record_index": 0,
            "exclusion_reason": "legacy F_original[0] is an all-zero placeholder",
            "relative_l2_tolerance": 1e-4,
            "maximum_absolute_tolerance_ev_per_angstrom": 1e-4,
        },
        "supersedes_failed_preregistrations": [str(path) for path in SUPERSEDED_CONTRACTS],
        "figure_policy": "append geometric action as a third panel; common Number-of-Forces scale across all panels",
        "sources": {str(path): sha256(path) for path in sorted(source_files)},
    }
    CONTRACT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def compute() -> None:
    config = json.loads(CONTRACT.read_text())
    for name, expected in config["sources"].items():
        observed = sha256(Path(name))
        if observed != expected:
            raise RuntimeError(f"Registered source drift: {name}")
    if OUT.exists() or OUT.with_suffix(".audit.json").exists():
        raise FileExistsError("Refusing to overwrite the registered diagnostic")
    rows = selected_rows()
    selected_hash = array_sha256(
        np.asarray([row["selected_path_sha256"] for row in rows])
    )
    if selected_hash != config["selected_paths_sha256"]:
        raise RuntimeError("Selected path set changed after preregistration")

    HOST_LOCK.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(HOST_LOCK, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise RuntimeError(f"Host-local Ala lock is held: {HOST_LOCK}")

    try:
        backend = OpenMMBackend(
            TOPOLOGY,
            platform_preference=("CPU",),
            precision="double",
            label="offline_discrete_geometric_action",
        )
        with np.load(ODE_RESULT, allow_pickle=False) as ode_saved:
            ode_force_reference = np.asarray(ode_saved["F_original"], dtype=float)
        ode_force_relative_l2 = []
        ode_force_maximum_absolute = []
        for position, row in enumerate(rows):
            before = backend.raw_joint_evaluations
            evaluated = [backend.evaluate_joint(point) for point in row["path"]]
            energies = np.asarray([value[0] for value in evaluated], dtype=float)
            forces = np.asarray([value[1] for value in evaluated], dtype=float)
            if backend.raw_joint_evaluations - before != 22:
                raise RuntimeError("Per-path diagnostic force count changed")
            action, endpoint, norms, lengths = discrete_action(row["path"], forces)
            check = path_diagnostics(row["path"], energies, forces[1:-1])["i_perp"]
            if row["method"] == "ode_neb" and row["source_record_index"] > 0:
                reference_force = ode_force_reference[row["source_record_index"]]
                difference = forces[1:-1] - reference_force
                relative_l2 = float(
                    np.linalg.norm(difference) / max(np.linalg.norm(forces[1:-1]), 1e-30)
                )
                maximum_absolute = float(np.max(np.abs(difference)))
                if (
                    relative_l2
                    > config["ode_stored_force_verification"]["relative_l2_tolerance"]
                    or maximum_absolute
                    > config["ode_stored_force_verification"][
                        "maximum_absolute_tolerance_ev_per_angstrom"
                    ]
                ):
                    raise RuntimeError(
                        f"ODE saved-force verification failed at record "
                        f"{row['source_record_index']}: rel={relative_l2}, max={maximum_absolute}"
                    )
                ode_force_relative_l2.append(relative_l2)
                ode_force_maximum_absolute.append(maximum_absolute)
            if row["iperp_verification_applicable"] and not np.isclose(
                check,
                row["i_perp"],
                rtol=config["iperp_verification_rtol"],
                atol=config["iperp_verification_atol"],
            ):
                raise RuntimeError(
                    f"I_perp verification failed for {row['method']} record "
                    f"{row['source_record_index']}: {check} != {row['i_perp']}"
                )
            row.update(
                action_kcal=action * EV_TO_KCAL,
                endpoint_term_kcal=endpoint * EV_TO_KCAL,
                recomputed_i_perp=float(check),
                gradient_norm_ev_per_angstrom=norms,
                segment_length_angstrom=lengths,
                energy_ev=energies,
                diagnostic_joint_evaluations=22,
            )
            done = position + 1
            if done % 50 == 0 or done == len(rows):
                print(
                    f"{done}/{len(rows)} paths; diagnostic NoF={backend.raw_joint_evaluations}; "
                    f"{row['method']}, optimization NF={row['number_of_forces']}, "
                    f"S={row['action_kcal']:.10g}",
                    flush=True,
                )

        if backend.raw_joint_evaluations != config["expected_diagnostic_joint_evaluations"]:
            raise RuntimeError("Total diagnostic force count changed")
        keys = (
            "method",
            "source_result",
            "source_result_sha256",
            "source_record_index",
            "source_call",
            "number_of_forces",
                "i_perp",
                "recomputed_i_perp",
                "iperp_verification_applicable",
            "selected_path_sha256",
            "eligibility",
            "action_kcal",
            "endpoint_term_kcal",
            "diagnostic_joint_evaluations",
            "gradient_norm_ev_per_angstrom",
            "segment_length_angstrom",
            "energy_ev",
        )
        OUT.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            OUT,
            **{key: np.asarray([row[key] for row in rows]) for key in keys},
            contract_sha256=np.asarray(sha256(CONTRACT)),
        )
        audit = {
            "status": "validated",
            "contract": str(CONTRACT),
            "contract_sha256": sha256(CONTRACT),
            "result": str(OUT),
            "result_sha256": sha256(OUT),
            "selected_path_count": len(rows),
            "counts": config["counts"],
            "diagnostic_joint_evaluations": backend.raw_joint_evaluations,
            "diagnostic_cost_excluded_from_optimization_axis": True,
            "backend_platform": backend.platform_name,
            "openmm_cpu_threads": 1,
            "maximum_iperp_absolute_difference": max(
                abs(row["recomputed_i_perp"] - row["i_perp"]) for row in rows
                if row["iperp_verification_applicable"]
            ),
            "maximum_iperp_relative_difference": max(
                abs(row["recomputed_i_perp"] - row["i_perp"])
                / max(abs(row["i_perp"]), 1e-30)
                for row in rows
                if row["iperp_verification_applicable"]
            ),
            "maximum_ode_saved_force_relative_l2_difference": max(
                ode_force_relative_l2
            ),
            "maximum_ode_saved_force_absolute_difference_ev_per_angstrom": max(
                ode_force_maximum_absolute
            ),
        }
        OUT.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        print(json.dumps(audit, indent=2), flush=True)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    parser.add_argument("--compute", action="store_true")
    args = parser.parse_args()
    if args.register == args.compute:
        parser.error("Choose exactly one of --register or --compute")
    register() if args.register else compute()


if __name__ == "__main__":
    main()
