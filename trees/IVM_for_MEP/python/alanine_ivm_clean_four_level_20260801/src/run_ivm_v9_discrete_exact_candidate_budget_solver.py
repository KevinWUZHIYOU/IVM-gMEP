#!/usr/bin/env python3
"""Candidate-aware complete-band budget adapter for exact Ala solves.

This is the ``discrete_b040`` loop with one exact-runtime-only addition: before
each joint trial, count the actual uncached internal images.  The whole band is
either affordable and evaluated, or refused before any query.  In particular,
a legitimate 19-miss/1-hit band costs 19 NF and remains a complete evaluated
band.  The frozen production source is not edited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


POLICY_ID = "ala-exact-candidate-aware-complete-band-budget-v1"
STOP_REASON = "total-internal-nf-complete-band-budget"


class CandidateAwareBudgetSafetyError(RuntimeError):
    """Fail closed if candidate-aware accounting drifts."""


@dataclass(frozen=True)
class CandidateBudgetEvent:
    required_internal_misses: int
    remaining_internal_nf_before: int
    stopped_before_query: bool
    joint_steps_before: int


@dataclass
class InstalledCandidateAwareBudgetSolver:
    frozen_module: Any
    budget_installed: Any
    original_solve: Any
    original_implementation_id: str
    events: list[CandidateBudgetEvent] = field(default_factory=list)
    solve_wrapper: Any = None
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        if self.frozen_module.solve_frozen_discrete_b040_band is self.solve_wrapper:
            self.frozen_module.solve_frozen_discrete_b040_band = self.original_solve
        self.frozen_module.IMPLEMENTATION_ID = self.original_implementation_id
        self.restored = True


def install_candidate_aware_budget_solver(
    frozen_module: Any, budget_installed: Any
) -> InstalledCandidateAwareBudgetSolver:
    """Install the exact-only complete-band solver for one fresh process."""

    original_solve = frozen_module.solve_frozen_discrete_b040_band
    original_id = str(frozen_module.IMPLEMENTATION_ID)
    installed = InstalledCandidateAwareBudgetSolver(
        frozen_module=frozen_module,
        budget_installed=budget_installed,
        original_solve=original_solve,
        original_implementation_id=original_id,
    )

    def solve_candidate_aware(
        path,
        energies,
        forces,
        tangents,
        evaluators,
        alpha,
        max_step,
        max_new_calls,
        *,
        inner_max_steps=10,
        stop_mode="r08",
        relative_residual=0.8,
        history_size=7,
        kkt_atol=1.0e-8,
        preconditioner_bank=None,
        active_images=None,
    ):
        f = frozen_module
        path = np.asarray(path, dtype=np.float64)
        energies = np.asarray(energies, dtype=np.float64)
        forces = np.asarray(forces, dtype=np.float64)
        tangents = np.asarray(tangents, dtype=np.float64)
        n_internal = path.shape[0] - 2
        if path.ndim != 2 or path.shape[0] < 3:
            raise ValueError("fixed path must contain endpoints and internal images")
        if energies.shape != (path.shape[0],) or forces.shape != path.shape:
            raise ValueError("fixed path energy/force shape differs")
        if tangents.shape != (n_internal, path.shape[1]):
            raise ValueError("frozen tangent shape differs")
        if not (
            np.all(np.isfinite(path))
            and np.all(np.isfinite(energies))
            and np.all(np.isfinite(forces))
            and np.all(np.isfinite(tangents))
        ):
            raise ValueError("fixed path state must be finite")
        tangent_norms = np.linalg.norm(tangents, axis=1)
        if not np.allclose(tangent_norms, 1.0, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("frozen central tangents must be unit vectors")
        if active_images is not None:
            active = np.asarray(active_images, dtype=bool)
            if active.shape != (n_internal,) or not np.all(active):
                raise ValueError("joint discrete solver requires every internal image")
        if type(inner_max_steps) is not int or inner_max_steps < 1:
            raise ValueError("inner_max_steps must be a positive exact int")
        if type(history_size) is not int or history_size < 1:
            raise ValueError("history_size must be a positive exact int")
        if stop_mode not in f.VALID_STOP_MODES:
            raise ValueError("stop_mode must be r08c10, r08, or c10")
        relative_residual = float(relative_residual)
        if not 0.0 < relative_residual < 1.0:
            raise ValueError("relative_residual must lie in (0,1)")
        if int(max_new_calls) < 0:
            raise ValueError("max_new_calls must be nonnegative")
        inverse_alpha = f._proximal_inverse_alpha(alpha)

        reference_path = path.copy()
        current_path = path.copy()
        current_energies = energies.copy()
        current_forces = forces.copy()
        starting_counters = f.counter_arrays(evaluators)
        start_calls = starting_counters["joint_evaluations"][1:-1].copy()
        displacement = np.zeros_like(path[1:-1])
        gradient = f.project_normal(-forces[1:-1], tangents)
        residual = f.project_normal(
            gradient + inverse_alpha * displacement, tangents
        )
        initial_per_image = np.linalg.norm(residual, axis=1)
        initial_residual = f._norm(residual) / np.sqrt(float(n_internal))
        if initial_residual <= 0.0:
            raise f.DiscreteB040SafetyError(
                "initial natural residual must be positive"
            )

        history_s = []
        history_y = []
        joint_steps = 0
        synchronized_paths = []
        synchronized_energies = []
        synchronized_forces = []
        synchronized_counters = []
        direction_selection = []
        h0_raw_decrease = []
        lbfgs_raw_decrease = []
        h0_capped_decrease = []
        lbfgs_capped_decrease = []
        h0_cap_scale = []
        lbfgs_cap_scale = []
        history_count = []
        curvature_status = []
        stop_reason = "not-started"

        def used_calls():
            return (
                f.counter_arrays(evaluators)["joint_evaluations"][1:-1]
                - start_calls
            ).astype(int)

        while True:
            residual_l2 = f._norm(residual) / np.sqrt(float(n_internal))
            relative = residual_l2 / initial_residual
            if joint_steps > 0 and residual_l2 <= float(kkt_atol):
                stop_reason = "evaluated-absolute-natural-residual"
                break
            if (
                stop_mode in f.RESIDUAL_STOP_MODES
                and joint_steps > 0
                and relative <= relative_residual
            ):
                stop_reason = "evaluated-relative-natural-residual"
                break
            if joint_steps >= int(inner_max_steps):
                stop_reason = "evaluated-inner-max-steps"
                break
            current_used = used_calls()
            if np.any(current_used >= int(max_new_calls)):
                stop_reason = "per-image-new-call-limit"
                break

            def apply_h0(value):
                return f.apply_block_h0_inverse(
                    value, tangents, preconditioner_bank, alpha
                )

            h0_inverse_gradient = apply_h0(residual)
            raw_h0 = -np.asarray(h0_inverse_gradient, dtype=np.float64)
            if history_s:
                raw_lbfgs = f.joint_lbfgs_direction(
                    residual, history_s, history_y, apply_h0
                )
            else:
                raw_lbfgs = np.array(raw_h0, dtype=np.float64, copy=True)
            h0 = f.screen_direction("H0", raw_h0, residual, tangents, max_step)
            lbfgs = f.screen_direction(
                "LBFGS", raw_lbfgs, residual, tangents, max_step
            )
            selected = f.select_direction(h0, lbfgs)
            step = np.asarray(selected["step"], dtype=np.float64)
            trial_displacement = f.project_normal(displacement + step, tangents)
            trial_path = reference_path.copy()
            trial_path[1:-1] = reference_path[1:-1] + trial_displacement
            if not np.array_equal(
                trial_path[0], reference_path[0]
            ) or not np.array_equal(trial_path[-1], reference_path[-1]):
                raise f.DiscreteB040SafetyError("fixed endpoints changed")

            blocked = np.asarray(
                [
                    current_used[index] >= int(max_new_calls)
                    and not evaluators[index + 1].has_cached(
                        trial_path[index + 1]
                    )
                    for index in range(n_internal)
                ],
                dtype=bool,
            )
            if np.any(blocked):
                stop_reason = "preflight-per-image-new-call-limit"
                break

            required_misses = int(
                np.count_nonzero(
                    [
                        not evaluators[index + 1].has_cached(
                            trial_path[index + 1]
                        )
                        for index in range(n_internal)
                    ]
                )
            )
            remaining = int(budget_installed.budget) - int(
                budget_installed.successful_internal_joint_evaluations
            )
            stopped = required_misses > remaining
            installed.events.append(
                CandidateBudgetEvent(
                    required_internal_misses=required_misses,
                    remaining_internal_nf_before=remaining,
                    stopped_before_query=stopped,
                    joint_steps_before=joint_steps,
                )
            )
            budget_installed.full_band_preflight_calls += 1
            if stopped:
                budget_installed.full_band_preflight_crops += 1
                if joint_steps == 0:
                    budget_installed.full_band_preflight_zero_step_stops += 1
                budget_installed.complete_band_budget_stop_count += 1
                raise budget_installed.ala_core_module.EvaluationBudgetReached(
                    "candidate-aware total-NF preflight refused an incomplete band"
                )

            before_total = int(
                budget_installed.successful_internal_joint_evaluations
            )
            trial_energies, trial_forces = f.evaluate_path(
                trial_path, evaluators
            )
            paid = int(budget_installed.successful_internal_joint_evaluations) - (
                before_total
            )
            if paid != required_misses:
                raise CandidateAwareBudgetSafetyError(
                    "actual candidate misses differ from preflight"
                )
            trial_gradient = f.project_normal(-trial_forces[1:-1], tangents)
            trial_residual = f.project_normal(
                trial_gradient + inverse_alpha * trial_displacement,
                tangents,
            )
            if not (
                np.all(np.isfinite(trial_energies))
                and np.all(np.isfinite(trial_forces))
                and np.all(np.isfinite(trial_residual))
            ):
                raise f.DiscreteB040SafetyError("evaluated trial is non-finite")

            s_value = trial_displacement - displacement
            y_value = trial_residual - residual
            status = f.append_curvature(
                history_s, history_y, s_value, y_value, history_size
            )
            displacement = trial_displacement
            current_path = trial_path
            current_energies = trial_energies
            current_forces = trial_forces
            gradient = trial_gradient
            residual = trial_residual
            joint_steps += 1

            direction_selection.append(selected["label"])
            h0_raw_decrease.append(h0["raw_decrease"])
            lbfgs_raw_decrease.append(lbfgs["raw_decrease"])
            h0_capped_decrease.append(h0["capped_decrease"])
            lbfgs_capped_decrease.append(lbfgs["capped_decrease"])
            h0_cap_scale.append(h0["cap_scale"])
            lbfgs_cap_scale.append(lbfgs["cap_scale"])
            history_count.append(len(history_s))
            curvature_status.append(status)
            synchronized_paths.append(current_path.copy())
            synchronized_energies.append(current_energies.copy())
            synchronized_forces.append(current_forces.copy())
            synchronized_counters.append(
                {
                    name: values.copy()
                    for name, values in f.counter_arrays(evaluators).items()
                }
            )

        final_per_image = np.linalg.norm(residual, axis=1)
        final_residual = f._norm(residual) / np.sqrt(float(n_internal))
        relative_per_image = final_per_image / np.maximum(
            initial_per_image, 1.0e-30
        )
        global_relative = final_residual / initial_residual
        global_converged = bool(
            final_residual <= float(kkt_atol)
            or (
                stop_mode in f.RESIDUAL_STOP_MODES
                and global_relative <= relative_residual
            )
        )
        used = used_calls()
        hit_call_limit = used >= int(max_new_calls)
        per_image_target = np.maximum(
            float(kkt_atol), relative_residual * initial_per_image
        )
        report = {
            "initial_kkt_l2": initial_per_image,
            "final_kkt_l2": final_per_image,
            "relative_kkt": relative_per_image,
            "converged": final_per_image <= per_image_target,
            "hit_call_limit": hit_call_limit,
            "accepted_steps": np.full(n_internal, joint_steps, dtype=int),
            "backtracks": np.zeros(n_internal, dtype=int),
            "curvature_restarts": np.zeros(n_internal, dtype=int),
            "discrete_b040": True,
            "implementation_id": f.IMPLEMENTATION_ID,
            "global_initial_residual_l2": float(initial_residual),
            "global_final_residual_l2": float(final_residual),
            "global_relative_residual": float(global_relative),
            "global_converged": global_converged,
            "stop_reason": str(stop_reason),
            "joint_steps": int(joint_steps),
            "paid_trials": int(len(synchronized_paths)),
            "direction_selection": np.asarray(direction_selection, dtype="U8"),
            "h0_raw_predicted_decrease": np.asarray(h0_raw_decrease, dtype=float),
            "lbfgs_raw_predicted_decrease": np.asarray(
                lbfgs_raw_decrease, dtype=float
            ),
            "h0_capped_predicted_decrease": np.asarray(
                h0_capped_decrease, dtype=float
            ),
            "lbfgs_capped_predicted_decrease": np.asarray(
                lbfgs_capped_decrease, dtype=float
            ),
            "h0_cap_scale": np.asarray(h0_cap_scale, dtype=float),
            "lbfgs_cap_scale": np.asarray(lbfgs_cap_scale, dtype=float),
            "history_count": np.asarray(history_count, dtype=int),
            "curvature_status": np.asarray(curvature_status, dtype="U64"),
            "frozen_path_sha256": f._array_sha256(reference_path),
            "frozen_tangent_sha256": f._array_sha256(tangents),
            "all_returned_states_evaluated": True,
            "endpoints_bitwise_fixed": bool(
                np.array_equal(current_path[0], reference_path[0])
                and np.array_equal(current_path[-1], reference_path[-1])
            ),
        }
        return (
            current_path,
            current_energies,
            current_forces,
            used,
            synchronized_paths,
            synchronized_energies,
            synchronized_forces,
            synchronized_counters,
            report,
        )

    installed.solve_wrapper = solve_candidate_aware
    frozen_module.solve_frozen_discrete_b040_band = solve_candidate_aware
    frozen_module.IMPLEMENTATION_ID = "+".join((original_id, POLICY_ID))
    return installed
