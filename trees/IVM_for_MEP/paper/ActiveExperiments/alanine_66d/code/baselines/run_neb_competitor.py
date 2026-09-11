#!/usr/bin/env python3
"""Run one central-difference alanine NEB configuration (ODE or static)."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from ase.io import read
from ase.mep.neb import NEBOptimizer

from ala_core import (
    CentralDifferenceNEB,
    EvaluationBudgetReached,
    OpenMMBackend,
    TANGENT_DEFINITION,
    TOPOLOGY_FILENAME,
    array_sha256,
    attach_audited_calculators,
    atomic_save_npz,
    audit_counter_consistency,
    counter_arrays,
    load_common_path,
    make_evaluators,
    path_diagnostics,
    write_json,
)


PATH_MILESTONES = {1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500}


def run(args) -> Path:
    root = Path(args.root).expanduser().resolve()
    initial_path_file = Path(args.initial_path_file).expanduser()
    if not initial_path_file.is_absolute():
        initial_path_file = root / initial_path_file
    initial_path = load_common_path(initial_path_file)
    template = read(str(root / "inputs" / TOPOLOGY_FILENAME))
    backend = OpenMMBackend(
        root / "inputs" / TOPOLOGY_FILENAME,
        platform_preference=("CPU",),
        precision="double",
        label="neb_optimization",
    )
    evaluators = make_evaluators(backend, internal_limit=args.budget)
    images = attach_audited_calculators(initial_path, template, evaluators)
    neb = CentralDifferenceNEB(
        images,
        k=args.spring_k,
        climb=False,
        parallel=False,
        remove_rotation_and_translation=False,
    )
    if neb.neb_method.tangent_definition != TANGENT_DEFINITION:
        raise RuntimeError("NEB tangent implementation failed runtime audit")

    history = {
        "calls_per_image": [],
        "mean_calls_per_internal_image": [],
        "total_internal_joint_evaluations": [],
        "optimizer_fmax": [],
        "neb_residual": [],
        "physical_fmax": [],
        "i_perp": [],
        "barrier_kcal": [],
        "equidistance_integral": [],
        "i_arc": [],
        "max_unit_spring_force": [],
        "segment_length_cv": [],
        "segment_length_ratio": [],
        "wall_seconds": [],
        "counter_vectors": [],
        "energy_request_vectors": [],
        "force_request_vectors": [],
        "cache_hit_vectors": [],
    }
    stored_paths = []
    stored_path_calls = []
    last_recorded_path = None
    start_time = time.time()
    counter_sync_violation = False

    def current_path():
        return np.asarray(
            [image.get_positions().reshape(-1) for image in neb.images], dtype=float
        )

    def record(force=False):
        nonlocal counter_sync_violation, last_recorded_path
        neb_force = np.asarray(neb.get_forces(), dtype=float).reshape(-1, 3)
        counters = counter_arrays(evaluators)
        internal = counters["joint_evaluations"][1:-1]
        call = int(np.max(internal))
        if history["calls_per_image"] and call == history["calls_per_image"][-1]:
            if not force:
                return
            if np.array_equal(current_path(), stored_paths[-1]) if stored_paths else False:
                return
        if int(np.max(internal) - np.min(internal)) > 1:
            counter_sync_violation = True
        path = current_path()
        last_recorded_path = path.copy()
        energies = np.asarray(neb.energies, dtype=float)
        physical_forces = np.asarray(neb.real_forces[1:-1], dtype=float).reshape(
            path.shape[0] - 2, -1
        )
        diagnostic = path_diagnostics(path, energies, physical_forces)
        optimizer_fmax = float(np.max(np.linalg.norm(neb_force, axis=1)))
        residual = float(neb.get_residual())
        history["calls_per_image"].append(call)
        history["mean_calls_per_internal_image"].append(float(np.mean(internal)))
        history["total_internal_joint_evaluations"].append(int(np.sum(internal)))
        history["optimizer_fmax"].append(optimizer_fmax)
        history["neb_residual"].append(residual)
        history["physical_fmax"].append(diagnostic["fmax"])
        history["i_perp"].append(diagnostic["i_perp"])
        history["barrier_kcal"].append(diagnostic["barrier_kcal"])
        history["equidistance_integral"].append(
            diagnostic["equidistance_integral"]
        )
        history["i_arc"].append(diagnostic["i_arc"])
        history["max_unit_spring_force"].append(
            diagnostic["max_unit_spring_force"]
        )
        history["segment_length_cv"].append(diagnostic["segment_length_cv"])
        history["segment_length_ratio"].append(
            diagnostic["segment_length_ratio"]
        )
        history["wall_seconds"].append(time.time() - start_time)
        history["counter_vectors"].append(counters["joint_evaluations"].copy())
        history["energy_request_vectors"].append(
            counters["energy_requests"].copy()
        )
        history["force_request_vectors"].append(counters["force_requests"].copy())
        history["cache_hit_vectors"].append(counters["cache_hits"].copy())
        if (
            args.save_all_paths
            or (args.save_all_until > 0 and call <= args.save_all_until)
            or call in PATH_MILESTONES
            or (
                args.path_interval > 0
                and call >= 500
                and call % args.path_interval == 0
            )
        ):
            stored_paths.append(path.copy())
            stored_path_calls.append(call)
        print(
            f"[NEB central k={args.spring_k:g} alpha={args.alpha:g} "
            f"mode={args.optimizer}] calls={call:4d} "
            f"Iperp={diagnostic['i_perp']:.6e} "
            f"barrier={diagnostic['barrier_kcal']:.4f} "
            f"fmax={diagnostic['fmax']:.5f} "
            f"spacing_CV={diagnostic['segment_length_cv']:.4f}",
            flush=True,
        )

    stop_reason = "optimizer_steps"
    budget_reached = False
    try:
        record(force=True)
        optimizer = NEBOptimizer(
            neb,
            method=args.optimizer,
            alpha=args.alpha,
            logfile=None,
            verbose=0,
        )
        optimizer.attach(record, interval=1)
        optimizer.run(fmax=args.target_fmax, steps=args.budget * 3)
        stop_reason = "fmax"
    except EvaluationBudgetReached:
        stop_reason = "evaluation_budget"
        budget_reached = True
    finally:
        if not budget_reached:
            try:
                record(force=True)
            except EvaluationBudgetReached:
                pass

    if last_recorded_path is None:
        raise RuntimeError("NEB run ended without one evaluated path record")
    # Static NEB may hold an unevaluated local trial after the next OpenMM
    # request is blocked by the budget.  Publish only the last evaluated path.
    final_path = last_recorded_path.copy()
    if not stored_path_calls or stored_path_calls[-1] != history["calls_per_image"][-1]:
        stored_paths.append(final_path.copy())
        stored_path_calls.append(history["calls_per_image"][-1])
    label_prefix = f"{args.run_label}_" if args.run_label else ""
    tag = (
        f"neb_{label_prefix}central_k{args.spring_k:g}_alpha{args.alpha:g}_"
        f"{args.optimizer}_budget{args.budget}"
    ).replace(".", "p")
    output = root / "runs" / "neb" / f"{tag}.npz"
    atomic_save_npz(
        output,
        method=np.array("NEB"),
        run_label=np.array(args.run_label or "tuning"),
        tangent_definition=np.array(TANGENT_DEFINITION),
        initial_path=initial_path,
        initial_path_file=np.array(str(initial_path_file)),
        initial_path_sha256=np.array(array_sha256(initial_path)),
        final_path=final_path,
        stored_paths=np.asarray(stored_paths),
        stored_path_calls=np.asarray(stored_path_calls, dtype=int),
        spring_k=np.array(args.spring_k),
        optimizer=np.array(args.optimizer),
        optimizer_alpha=np.array(args.alpha),
        budget=np.array(args.budget),
        path_interval=np.array(args.path_interval),
        save_all_until=np.array(args.save_all_until),
        stop_reason=np.array(stop_reason),
        counter_sync_violation=np.array(counter_sync_violation),
        **{name: np.asarray(values) for name, values in history.items()},
    )
    audit = audit_counter_consistency(backend, evaluators)
    audit.update(
        {
            "result_file": str(output),
            "tangent_definition": TANGENT_DEFINITION,
            "counter_sync_violation": counter_sync_violation,
            "stop_reason": stop_reason,
        }
    )
    write_json(output.with_suffix(".audit.json"), audit)
    if not audit["consistent"] or counter_sync_violation:
        raise RuntimeError(f"NEB counter audit failed: {audit}")
    print(f"Saved {output}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--initial-path-file",
        default="inputs/common_initial_path_equidistant.npz",
        help=(
            "Common IDPP-derived path used by both methods. Relative paths "
            "are resolved below --root."
        ),
    )
    parser.add_argument("--run-label", default="")
    parser.add_argument("--spring-k", type=float, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument(
        "--optimizer",
        choices=("ode", "static"),
        default="ode",
        help=(
            "ASE NEBOptimizer mode. In static mode alpha is the explicit "
            "Euler step; adaptive ODE does not use alpha in its update."
        ),
    )
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--target-fmax", type=float, default=0.0)
    parser.add_argument("--save-all-paths", action="store_true")
    parser.add_argument("--save-all-until", type=int, default=0)
    parser.add_argument("--path-interval", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
