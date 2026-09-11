#!/usr/bin/env python3
"""Run audited ASE improved-tangent ODE-NEB for one spring constant."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.mep.neb import NEB, NEBOptimizer

from leps_core import (
    N_IMAGES,
    N_INTERNAL,
    TANGENT_DEFINITION,
    EvaluationBudgetReached,
    LEPSBackend,
    array_sha256,
    atomic_save_npz,
    attach_audited_calculators,
    audit_counter_consistency,
    counter_arrays,
    densified_barrier,
    evaluate_path,
    load_common_path,
    make_evaluators,
    path_diagnostics,
    project_relative,
    reference_saddle,
    write_json,
)


class BudgetedNEBOptimizer(NEBOptimizer):
    """Prevent an ODE trial from partially crossing the per-image budget."""

    def __init__(self, *args, evaluators, budget, **kwargs):
        super().__init__(*args, **kwargs)
        self.evaluators = evaluators
        self.budget = int(budget)
        self.last_accepted_x = self.neb.get_positions().reshape(-1).copy()

    def force_function(self, x):
        proposed = np.asarray(x, dtype=float).reshape(
            (N_INTERNAL * 2, 3)
        )
        for i in range(N_INTERNAL):
            flat = proposed[2 * i : 2 * (i + 1)].reshape(-1)
            evaluator = self.evaluators[i + 1]
            if (
                evaluator.counters.joint_evaluations >= self.budget
                and not evaluator.has_cached(flat)
            ):
                raise EvaluationBudgetReached("ODE-NEB evaluation budget reached")
        return super().force_function(x)

    def callback(self, x, f=None):
        self.last_accepted_x = np.asarray(x, dtype=float).copy()
        super().callback(x, f)

    def restore_last_accepted(self):
        self.neb.set_positions(
            self.last_accepted_x.reshape((N_INTERNAL * 2, 3))
        )


class NEBTrace:
    def __init__(self, neb, evaluators, saddle_barrier):
        self.neb = neb
        self.evaluators = evaluators
        self.saddle_barrier = float(saddle_barrier)
        self.rows = []

    def record(self):
        path = np.asarray(
            [image.get_positions().reshape(-1) for image in self.neb.images],
            dtype=float,
        )
        energies, forces = evaluate_path(path, self.evaluators)
        diagnostic = path_diagnostics(path, energies, forces[1:-1])
        dense, _ = densified_barrier(path, subdivisions=80)
        counters = counter_arrays(self.evaluators)
        internal = counters["joint_evaluations"][1:-1]
        call = int(np.max(internal))
        try:
            neb_residual = float(self.neb.get_residual())
        except Exception:
            neb_residual = np.nan
        row = {
            "call": call,
            "path": path.copy(),
            "energies": energies.copy(),
            "forces": forces.copy(),
            "i_perp": diagnostic["i_perp"],
            "fmax": diagnostic["fmax"],
            "neb_residual": neb_residual,
            "barrier": diagnostic["barrier"],
            "dense_barrier": dense,
            "barrier_error": abs(dense - self.saddle_barrier),
            "i_arc": diagnostic["i_arc"],
            "equidistance_integral": diagnostic["equidistance_integral"],
            "spring_force_max_unit": diagnostic["spring_force_max_unit"],
            "equidistance_relative_defect": diagnostic[
                "equidistance_relative_defect"
            ],
            "segment_length_cv": diagnostic["segment_length_cv"],
            "segment_length_ratio": diagnostic["segment_length_ratio"],
            "counter_vector": counters["joint_evaluations"].copy(),
            "energy_request_vector": counters["energy_requests"].copy(),
            "force_request_vector": counters["force_requests"].copy(),
            "cache_hit_vector": counters["cache_hits"].copy(),
            "mean_calls": float(np.mean(internal)),
            "total_internal_calls": int(np.sum(internal)),
        }
        if self.rows and self.rows[-1]["call"] == call:
            self.rows[-1] = row
        else:
            self.rows.append(row)

    def arrays(self):
        payload = {
            "calls_per_image": np.asarray([row["call"] for row in self.rows])
        }
        for key in self.rows[0]:
            if key != "call":
                payload[key] = np.asarray([row[key] for row in self.rows])
        return payload


def run(args):
    root = args.root.expanduser().resolve()
    initial_path_file = args.initial_path_file.expanduser()
    if not initial_path_file.is_absolute():
        initial_path_file = root / initial_path_file
    initial_path = load_common_path(initial_path_file)
    backend = LEPSBackend(label=args.run_label)
    evaluators = make_evaluators(backend, internal_limit=args.budget)
    images = attach_audited_calculators(initial_path, evaluators)
    neb = NEB(
        images,
        k=float(args.k),
        climb=False,
        parallel=False,
        remove_rotation_and_translation=False,
        method="improvedtangent",
    )
    saddle = reference_saddle()
    trace = NEBTrace(neb, evaluators, saddle["forward_barrier"])
    trace.record()
    optimizer = BudgetedNEBOptimizer(
        neb,
        method=args.optimizer,
        logfile=None,
        trajectory=None,
        evaluators=evaluators,
        budget=args.budget,
        alpha=args.alpha,
        rtol=args.ode_rtol,
        C1=args.ode_c1,
        C2=args.ode_c2,
    )
    optimizer.attach(trace.record, interval=1)
    converged = False
    stop_reason = "unknown"
    try:
        converged = bool(
            optimizer.run(fmax=args.neb_fmax, steps=args.max_ode_steps)
        )
        stop_reason = (
            "fmax" if converged else f"{args.optimizer}_step_limit"
        )
    except EvaluationBudgetReached:
        stop_reason = "evaluation_budget"
    except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
        stop_reason = f"numerical_failure_{type(exc).__name__}"
    finally:
        if np.all(np.isfinite(optimizer.last_accepted_x)):
            optimizer.restore_last_accepted()
        else:
            last_path = trace.rows[-1]["path"]
            neb.set_positions(last_path[1:-1].reshape((N_INTERNAL * 2, 3)))
        try:
            trace.record()
        except (EvaluationBudgetReached, FloatingPointError, ValueError):
            # Static Euler can exhaust one image exactly at the boundary while
            # restoring a band whose diagnostic state was not yet cached.
            # The last complete trace row remains the honest final record.
            pass

    audit = audit_counter_consistency(backend, evaluators)
    if not audit["consistent"]:
        raise RuntimeError("LEPS NEB counter audit failed")
    configuration = {
        "method": "NEB",
        "optimizer": f"ASE NEBOptimizer {args.optimizer}",
        "neb_method": "improvedtangent",
        "run_label": args.run_label,
        "k": args.k,
        "ode_rtol": args.ode_rtol,
        "ode_c1": args.ode_c1,
        "ode_c2": args.ode_c2,
        "alpha": (
            args.alpha
            if args.optimizer == "static"
            else "not used by NEBOptimizer(method='ode')"
        ),
        "neb_fmax": args.neb_fmax,
        "max_ode_steps": args.max_ode_steps,
        "budget": args.budget,
        "initial_path_file": project_relative(initial_path_file, root),
        "initial_path_sha256": array_sha256(initial_path),
        "tangent_definition": TANGENT_DEFINITION,
        "stop_reason": stop_reason,
        "converged": converged,
    }
    config_sha = hashlib.sha256(
        json.dumps(configuration, sort_keys=True).encode()
    ).hexdigest()
    output = (
        root
        / "runs"
        / "neb"
        / f"{args.run_label}_cfg{config_sha[:12]}_budget{args.budget}.npz"
    )
    final_path = np.asarray(
        [image.get_positions().reshape(-1) for image in neb.images], dtype=float
    )
    atomic_save_npz(
        output,
        method=np.asarray("NEB"),
        optimizer=np.asarray(args.optimizer),
        neb_method=np.asarray("improvedtangent"),
        run_label=np.asarray(args.run_label),
        configuration_sha256=np.asarray(config_sha),
        tangent_definition=np.asarray(TANGENT_DEFINITION),
        initial_path=initial_path,
        initial_path_sha256=np.asarray(configuration["initial_path_sha256"]),
        final_path=final_path,
        spring_constant=np.asarray(args.k),
        static_step_size=np.asarray(
            args.alpha if args.optimizer == "static" else np.nan
        ),
        stop_reason=np.asarray(stop_reason),
        converged=np.asarray(converged),
        reference_barrier=np.asarray(saddle["forward_barrier"]),
        counter_audit_consistent=np.asarray(audit["consistent"]),
        backend_raw_joint_evaluations=np.asarray(
            audit["backend_raw_joint_evaluations"]
        ),
        **trace.arrays(),
    )
    write_json(
        output.with_suffix(".audit.json"),
        {
            "configuration": configuration,
            "configuration_sha256": config_sha,
            "counter_audit": audit,
            "result": project_relative(output, root),
        },
    )
    print(output)
    return output


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--initial-path-file",
        type=Path,
        default=Path("inputs/common_initial_path.npz"),
    )
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--k", type=float, required=True)
    parser.add_argument(
        "--optimizer", choices=("ode", "static"), default="ode"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Explicit Euler step used only by optimizer=static.",
    )
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--neb-fmax", type=float, default=1e-8)
    parser.add_argument("--max-ode-steps", type=int, default=100000)
    parser.add_argument("--ode-rtol", type=float, default=0.1)
    parser.add_argument("--ode-c1", type=float, default=1e-2)
    parser.add_argument("--ode-c2", type=float, default=2.0)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
