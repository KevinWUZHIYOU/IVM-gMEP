"""Reference loading and exact-prefix scoring for Ala2D V17 formal IR-IVM."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import heapq

import numpy as np

from ala2d_oracles import KJ_PER_KCAL
from formal_contract import validate_public_trajectory
from riemannian_torus_geometry import compute_article_metrics


Array = np.ndarray


def load_common_initial(path: Path) -> Dict[str, Array]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.array(loaded[key], copy=True) for key in loaded.files}


def validate_causal_shared_initial(
    current_npz: Path,
    common: Dict[str, Array],
) -> None:
    """Fail closed unless the shared observation is actual current-IVM state 0.

    Matching only the path geometry is insufficient: the sampled metric and
    full mean-force vector determine the first update.  All three objects must
    therefore be the exact arrays from which the archived state-1 path was
    produced.
    """

    required_common = (
        "C1",
        "C2",
        "full_path_init",
        "M_init",
        "F_init",
        "historical_initial_nof",
        "incremental_nof",
        "mean_force_vector_dimension",
    )
    missing = [key for key in required_common if key not in common]
    if missing:
        raise ValueError(
            "shared initial observation lacks causal-provenance fields: "
            f"{missing}"
        )
    with np.load(current_npz, allow_pickle=False) as loaded:
        required_current = (
            "phi_ref_iter_0",
            "M_ref_iter_0",
            "F_ref_iter_0",
            "path_iter_0",
            "phi_ref_iter_1",
        )
        missing_current = [key for key in required_current if key not in loaded.files]
        if missing_current:
            raise ValueError(
                "historical current IVM lacks causal-prefix fields: "
                f"{missing_current}"
            )
        phi0 = np.asarray(loaded["phi_ref_iter_0"], dtype=np.float64)
        expected_path = np.vstack(
            [
                np.asarray(common["C1"], dtype=np.float64),
                phi0,
                np.asarray(common["C2"], dtype=np.float64),
            ]
        )
        identities = {
            "path": np.array_equal(
                np.asarray(common["full_path_init"], dtype=np.float64),
                expected_path,
            ),
            "sampled metric": np.array_equal(
                np.asarray(common["M_init"], dtype=np.float64),
                np.asarray(loaded["M_ref_iter_0"], dtype=np.float64),
            ),
            "mean force": np.array_equal(
                np.asarray(common["F_init"], dtype=np.float64),
                np.asarray(loaded["F_ref_iter_0"], dtype=np.float64),
            ),
            "state-0 proposal/state-1 path": np.array_equal(
                np.asarray(loaded["path_iter_0"], dtype=np.float64),
                np.asarray(loaded["phi_ref_iter_1"], dtype=np.float64),
            ),
        }
    failed = [label for label, matches in identities.items() if not matches]
    if failed:
        raise ValueError(
            "shared initial observation is not the causal current-IVM state 0: "
            + ", ".join(failed)
        )
    n_internal = int(phi0.shape[0])
    if not (
        int(np.asarray(common["historical_initial_nof"]).item()) == n_internal
        and int(np.asarray(common["incremental_nof"]).item()) == 0
        and int(np.asarray(common["mean_force_vector_dimension"]).item()) == 2
    ):
        raise ValueError("shared initial NoF metadata violates the registered convention")


def load_historical_current_ivm(
    current_npz: Path,
    common_npz: Path,
    maximum_count: int = 100,
) -> Dict[str, Array]:
    """Recompute the actual plotted current-IVM trajectory from raw states.

    Count zero is the explicitly shared *causal* state-0 observation.  Its
    path, sampled metric, and full mean-force vector are required to match the
    corresponding historical current-IVM arrays exactly; every later state is
    then taken from the same archived L-BFGS-Recycle trajectory.
    """

    common = load_common_initial(common_npz)
    validate_causal_shared_initial(current_npz, common)
    nof: List[int] = [0]
    paths: List[Array] = [common["full_path_init"]]
    metrics: List[Dict[str, Array]] = [
        compute_article_metrics(
            common["full_path_init"], common["M_init"], common["F_init"]
        )
    ]
    M_values: List[Array] = [common["M_init"]]
    gradients: List[Array] = [common["F_init"]]
    with np.load(current_npz, allow_pickle=False) as loaded:
        for count in range(1, int(maximum_count) + 1):
            path = np.vstack(
                [common["C1"], loaded[f"phi_ref_iter_{count}"], common["C2"]]
            )
            M = np.asarray(loaded[f"M_ref_iter_{count}"], dtype=np.float64)
            gradient_key = (
                f"F_ref_iter_{count}"
                if f"F_ref_iter_{count}" in loaded.files
                else f"g_ref_iter_{count}"
            )
            gradient = np.asarray(loaded[gradient_key], dtype=np.float64)
            nof.append(20 * count)
            paths.append(path)
            M_values.append(M)
            gradients.append(gradient)
            metrics.append(compute_article_metrics(path, M, gradient))
    return {
        "nof_mean_force_incremental": np.asarray(nof, dtype=np.int64),
        "full_path_unwrapped": np.stack(paths),
        "M": np.stack(M_values),
        "grad_F": np.stack(gradients),
        "I_perp_G": np.asarray([float(item["I_perp_G"]) for item in metrics]),
        "I_arc": np.asarray([float(item["I_arc"]) for item in metrics]),
        "Delta_F_barrier": np.asarray(
            [float(item["Delta_F_barrier"]) for item in metrics]
        ),
    }


def fes_minimax_barrier_reference(fes_npz: Path, common_npz: Path) -> Dict[str, object]:
    """Independent grid minimax barrier between the two registered endpoints.

    The metadynamics FES is stored in GROMACS/PLUMED kJ mol^-1.  The returned
    barrier is converted to kcal mol^-1 before comparison to the integrated
    mean-force curve.
    """

    common = load_common_initial(common_npz)
    with np.load(fes_npz, allow_pickle=False) as loaded:
        X = np.asarray(loaded["X"], dtype=np.float64)
        Y = np.asarray(loaded["Y"], dtype=np.float64)
        Z_kj = np.asarray(loaded["Z"], dtype=np.float64)
    x_axis = np.deg2rad(X[0])
    y_axis = np.deg2rad(Y[:, 0])
    endpoints = [common["C1"], common["C2"]]
    indices = [
        (
            int(np.argmin(np.abs(y_axis - endpoint[1]))),
            int(np.argmin(np.abs(x_axis - endpoint[0]))),
        )
        for endpoint in endpoints
    ]
    start, target = indices
    ny, nx = Z_kj.shape
    cost = np.full((ny, nx), np.inf, dtype=np.float64)
    predecessor = np.full((ny, nx, 2), -1, dtype=np.int64)
    cost[start] = Z_kj[start]
    queue: List[Tuple[float, Tuple[int, int]]] = [(float(cost[start]), start)]
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    while queue:
        value, (row, column) = heapq.heappop(queue)
        if value != cost[row, column]:
            continue
        if (row, column) == target:
            break
        for delta_row, delta_column in neighbors:
            next_row = (row + delta_row) % ny
            next_column = (column + delta_column) % nx
            next_value = max(value, float(Z_kj[next_row, next_column]))
            if next_value < cost[next_row, next_column]:
                cost[next_row, next_column] = next_value
                predecessor[next_row, next_column] = (row, column)
                heapq.heappush(queue, (next_value, (next_row, next_column)))
    if not np.isfinite(cost[target]):
        raise RuntimeError("FES minimax path was not found")

    grid_path: List[Tuple[int, int]] = []
    cursor = target
    while cursor != start:
        grid_path.append(cursor)
        previous = tuple(int(value) for value in predecessor[cursor])
        if previous[0] < 0:
            raise RuntimeError("broken FES minimax predecessor chain")
        cursor = previous
    grid_path.append(start)
    grid_path.reverse()
    angular_path = np.asarray(
        [[x_axis[column], y_axis[row]] for row, column in grid_path],
        dtype=np.float64,
    )
    saddle_kj = float(cost[target])
    start_kj = float(Z_kj[start])
    return {
        "method": "periodic eight-neighbor minimax path on frozen 315x315 FES grid",
        "barrier_kcal_per_mol": (saddle_kj - start_kj) / KJ_PER_KCAL,
        "saddle_kj_per_mol": saddle_kj,
        "start_kj_per_mol": start_kj,
        "start_grid_index": [int(start[0]), int(start[1])],
        "target_grid_index": [int(target[0]), int(target[1])],
        "grid_path": angular_path,
        "grid_path_point_count": len(grid_path),
    }


def score_trajectory(
    candidate: Dict[str, Array],
    baseline: Dict[str, Array],
    barrier_reference: float,
    maximum_nof: int,
) -> Dict[str, object]:
    """Score a trajectory at every common complete-band NoF checkpoint."""

    checkpoints = np.arange(20, int(maximum_nof) + 1, 20, dtype=np.int64)
    baseline_nof = np.asarray(baseline["nof_mean_force_incremental"], dtype=np.int64)
    candidate_nof = np.asarray(candidate["nof_mean_force_incremental"], dtype=np.int64)
    if baseline_nof[0] != 0 or candidate_nof[0] != 0:
        raise ValueError("both trajectories must contain the common zero-cost initial state")

    # V17 publishes every paid HOLD and independent-validation band.  Require
    # the exact prefix and score the path actually public at each cost; never
    # carry forward the last movement or inspect an unpaid pending candidate.
    validate_public_trajectory(candidate, expected_budget=int(maximum_nof))

    def exact(values: Array, costs: Array, label: str) -> Array:
        indices = np.searchsorted(costs, checkpoints)
        valid = indices < costs.size
        if not np.all(valid):
            raise ValueError(f"{label} does not contain every exact checkpoint")
        if not np.array_equal(costs[indices], checkpoints):
            raise ValueError(f"{label} does not contain every exact checkpoint")
        return np.asarray(values, dtype=np.float64)[indices]

    candidate_I = exact(candidate["I_perp_G"], candidate_nof, "candidate")
    candidate_A = exact(candidate["I_arc"], candidate_nof, "candidate")
    candidate_D = np.abs(
        exact(candidate["Delta_F_barrier"], candidate_nof, "candidate")
        - barrier_reference
    )
    baseline_I = exact(baseline["I_perp_G"], baseline_nof, "current IVM")
    baseline_A = exact(baseline["I_arc"], baseline_nof, "current IVM")
    baseline_D = np.abs(
        exact(baseline["Delta_F_barrier"], baseline_nof, "current IVM")
        - barrier_reference
    )
    epsilon = 1.0e-14
    ratios = np.column_stack(
        [
            candidate_I / np.maximum(baseline_I, epsilon),
            candidate_A / np.maximum(baseline_A, epsilon),
            candidate_D / np.maximum(baseline_D, epsilon),
        ]
    )
    dominance = np.column_stack(
        [candidate_I < baseline_I, candidate_A <= baseline_A, candidate_D < baseline_D]
    )
    joint = np.all(dominance, axis=1)
    return {
        "checkpoints": checkpoints,
        "dominance_count": int(np.sum(joint)),
        "checkpoint_count": int(checkpoints.size),
        "component_dominance_counts": [int(value) for value in np.sum(dominance, axis=0)],
        "first_failure_nof": int(checkpoints[np.flatnonzero(~joint)[0]]) if not np.all(joint) else None,
        "worst_joint_ratio": float(np.max(ratios)),
        "mean_log_joint_ratio": float(np.mean(np.log(np.maximum(ratios, 1.0e-300)))),
        "terminal_joint_ratio": [float(value) for value in ratios[-1]],
        "strict_all_three_pass": bool(np.all(joint)),
    }

