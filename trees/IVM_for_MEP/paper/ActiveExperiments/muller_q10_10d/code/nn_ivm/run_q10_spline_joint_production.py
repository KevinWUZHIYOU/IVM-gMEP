from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from q10_nn_ivm_fourier import FourierConfig, run_experiment


ROOT = Path("muller_q10_ivm")
RUN_ROOT = ROOT / "runs/nnivm_spline_joint_production_locked"
REPORT_PATH = ROOT / "reports/q10_spline_joint_production_locked.json"
SEEDS = (20260812, 20260813, 20260814)


def trajectory(path: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {
            "number_of_forces": np.asarray(data["accepted_number_of_forces"], dtype=int),
            "dense_i_perp": np.asarray(data["dense_i_perp"], dtype=float),
            "dense_barrier_error": np.asarray(data["dense_barrier_error"], dtype=float),
        }


def evaluate(seed: int) -> dict:
    common = {
        "alpha": 1e-3,
        "feature_mode": "bspline",
        "inner_iterations": 3,
        "sampling_mode": "golden_shift",
        "sampling_jitter": 0.3,
        "sampling_seed": seed,
        "max_energy_anchor_count": 0,
        "certificate_points": 5001,
    }
    stage1 = run_experiment(
        RUN_ROOT,
        FourierConfig(
            **common,
            force_budget=3200,
            max_outer=40,
            network_width=12,
            tangent_stencil_step=0.002,
            role=f"locked_seed{seed}_stage1",
        ),
    )
    stage2 = run_experiment(
        RUN_ROOT,
        FourierConfig(
            **common,
            force_budget=3700,
            cost_offset=3200,
            initial_path_file=stage1["output"],
            max_outer=10,
            network_width=14,
            tangent_stencil_step=0.001,
            role=f"locked_seed{seed}_stage2",
        ),
    )
    stage3 = run_experiment(
        RUN_ROOT,
        FourierConfig(
            **common,
            force_budget=4000,
            cost_offset=3700,
            initial_path_file=stage2["output"],
            max_outer=6,
            network_width=8,
            tangent_stencil_step=0.0001,
            role=f"locked_seed{seed}_stage3",
        ),
    )

    pieces = [trajectory(stage1["output"]), trajectory(stage2["output"]), trajectory(stage3["output"])]
    costs = np.concatenate([pieces[0]["number_of_forces"]] + [piece["number_of_forces"][1:] for piece in pieces[1:]])
    dense_i = np.concatenate([pieces[0]["dense_i_perp"]] + [piece["dense_i_perp"][1:] for piece in pieces[1:]])
    dense_barrier = np.concatenate(
        [pieces[0]["dense_barrier_error"]]
        + [piece["dense_barrier_error"][1:] for piece in pieces[1:]]
    )
    tail = costs >= 3500
    all_theory = all(stage["theory_audit_pass"] for stage in (stage1, stage2, stage3))
    result = {
        "seed": seed,
        "stages": [stage1, stage2, stage3],
        "number_of_forces": costs.tolist(),
        "dense_i_perp": dense_i.tolist(),
        "dense_barrier_error": dense_barrier.tolist(),
        "final_dense_i_perp": float(dense_i[-1]),
        "final_dense_barrier_error": float(dense_barrier[-1]),
        "tail_max_dense_i_perp": float(np.max(dense_i[tail])),
        "tail_max_dense_barrier_error": float(np.max(dense_barrier[tail])),
        "maximum_inactive_coordinate": float(stage3["maximum_inactive_coordinate"]),
        "theory_audit_pass": bool(all_theory),
    }
    result["main_acceptance_pass"] = bool(
        result["final_dense_i_perp"] <= 5.0
        and result["final_dense_barrier_error"] <= 1e-5
        and result["theory_audit_pass"]
    )
    result["tail_stability_pass"] = bool(
        result["tail_max_dense_i_perp"] <= 7.5
        and result["tail_max_dense_barrier_error"] <= 5e-4
    )
    result["reproduction_precision_pass"] = bool(
        result["final_dense_i_perp"] <= 8.0
        and result["final_dense_barrier_error"] <= 1e-4
    )
    return result


def main() -> None:
    results = []
    with ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(evaluate, seed): seed for seed in SEEDS}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"seed={result['seed']}",
                f"I={result['final_dense_i_perp']:.8e}",
                f"dE={result['final_dense_barrier_error']:.8e}",
                f"tail_I={result['tail_max_dense_i_perp']:.8e}",
                f"tail_dE={result['tail_max_dense_barrier_error']:.8e}",
                flush=True,
            )

    results.sort(key=lambda item: item["seed"])
    report = {
        "method": "locked three-stage cubic-spline NN-IVM",
        "acceptance_rules": {
            "hard_theory_gates": {
                "fixed_alpha": 1e-3,
                "moving_force_samples": 20,
                "delta_optimization": "unconstrained",
                "objective": "projected IVM proximal objective only",
                "tangent": "NEB improved tangent",
                "forbidden": [
                    "Lagrangian",
                    "mixing",
                    "trust clipping",
                    "output clipping",
                    "barrier penalty",
                    "dense-certificate feedback",
                ],
            },
            "main_at_4000_forces": {"dense_i_perp_max": 5.0, "dense_barrier_error_max": 1e-5},
            "tail_3500_to_4000": {"dense_i_perp_max": 7.5, "dense_barrier_error_max": 5e-4},
            "three_run_reproduction": {"dense_i_perp_max": 8.0, "dense_barrier_error_max": 1e-4},
            "stretch_at_3000_forces": {"dense_i_perp_max": 5.0, "dense_barrier_error_max": 1e-6},
        },
        "locked_schedule": [
            {"force_interval": [0, 3200], "width": 12, "tangent_stencil_step": 0.002},
            {"force_interval": [3200, 3700], "width": 14, "tangent_stencil_step": 0.001},
            {"force_interval": [3700, 4000], "width": 8, "tangent_stencil_step": 0.0001},
        ],
        "results": results,
    }
    report["all_main_acceptance_pass"] = all(item["main_acceptance_pass"] for item in results)
    report["all_tail_stability_pass"] = all(item["tail_stability_pass"] for item in results)
    report["all_reproduction_precision_pass"] = all(
        item["reproduction_precision_pass"] for item in results
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps({key: report[key] for key in report if key.startswith("all_")}, indent=2))


if __name__ == "__main__":
    main()
