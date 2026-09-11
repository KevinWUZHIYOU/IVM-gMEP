# -*- coding: utf-8 -*-
"""
Standalone uncoupled FENEB implementation for comparison with IVM.

Algorithm per outer iteration
-----------------------------
1. Evaluate restrained-MD mean coordinates, M, and grad F at every image.
2. Move the band to the restrained-MD mean coordinates.
3. Form the MFEP field M grad F.
4. Apply the G-orthogonal normal projector, G=M^{-1}.
5. Perform one globally normalized physical update.
6. Separately relax spring forces until the images are evenly distributed.
7. Use the reparameterized band as the reference band for the next iteration.

The target condition is

    P_G (M grad F) = 0.

This is identical to the MFEP condition used in the supplied IVM code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np


Array = np.ndarray


# ============================================================
# Restrained-MD oracle output
# ============================================================

@dataclass
class OracleResult:
    q_mean: Array
    M: Array
    grad_F: Array
    state: Any = None
    metadata: Optional[Dict[str, Any]] = None


# ============================================================
# Periodic angular-coordinate helpers
# ============================================================

def as_periods(
    periods: Optional[Sequence[float] | float],
    dimension: int,
) -> Array:
    if periods is None:
        return np.full(
            dimension,
            np.inf,
            dtype=float,
        )

    periods = np.asarray(
        periods,
        dtype=float,
    )

    if periods.ndim == 0:
        periods = np.full(
            dimension,
            float(periods),
            dtype=float,
        )

    return periods.reshape(dimension)


def wrap_periodic(
    q: Array,
    periods: Optional[Sequence[float] | float],
) -> Array:
    q = np.asarray(
        q,
        dtype=float,
    )

    periods_array = as_periods(
        periods,
        q.shape[-1],
    )

    result = q.copy()

    for coordinate_index, period in enumerate(
        periods_array
    ):
        if np.isfinite(period) and period > 0.0:
            result[..., coordinate_index] = (
                (
                    result[..., coordinate_index]
                    + 0.5 * period
                )
                % period
                - 0.5 * period
            )

    return result


def periodic_difference(
    q1: Array,
    q0: Array,
    periods: Optional[Sequence[float] | float],
) -> Array:
    """
    Return the shortest periodic displacement q1-q0.
    """
    q1 = np.asarray(
        q1,
        dtype=float,
    )

    q0 = np.asarray(
        q0,
        dtype=float,
    )

    difference = q1 - q0

    periods_array = as_periods(
        periods,
        difference.shape[-1],
    )

    for coordinate_index, period in enumerate(
        periods_array
    ):
        if np.isfinite(period) and period > 0.0:
            difference[..., coordinate_index] -= (
                period
                * np.round(
                    difference[..., coordinate_index]
                    / period
                )
            )

    return difference


def unwrap_near(
    q_wrapped: Array,
    reference_unwrapped: Array,
    periods: Optional[Sequence[float] | float],
) -> Array:
    """
    Select the branch of q_wrapped nearest to reference_unwrapped.
    """
    return (
        np.asarray(
            reference_unwrapped,
            dtype=float,
        )
        + periodic_difference(
            q_wrapped,
            reference_unwrapped,
            periods,
        )
    )


def shortest_periodic_interpolation(
    q0: Array,
    q1: Array,
    n_internal: int,
    periods: Optional[Sequence[float] | float],
) -> Array:
    """
    Generate an unwrapped straight path containing both endpoints.
    """
    q0 = np.asarray(
        q0,
        dtype=float,
    ).reshape(-1)

    q1 = np.asarray(
        q1,
        dtype=float,
    ).reshape(-1)

    displacement = periodic_difference(
        q1,
        q0,
        periods,
    )

    interpolation_parameter = np.linspace(
        0.0,
        1.0,
        n_internal + 2,
    )

    return (
        q0[None, :]
        + interpolation_parameter[:, None]
        * displacement[None, :]
    )


# ============================================================
# Path tangent
# ============================================================

def central_difference_tangent(
    full_path: Array,
) -> Array:
    """
    Same tangent used in the supplied IVM code:

        path[i+1] - path[i-1].
    """
    full_path = np.asarray(
        full_path,
        dtype=float,
    )

    return (
        full_path[2:]
        - full_path[:-2]
    )


# ============================================================
# G-orthogonal projectors
# ============================================================

def compute_G_projectors(
    tangent: Array,
    M_all: Array,
) -> Tuple[Array, Array, Array]:
    """
    Reproduce the G_projectors() calculation in the supplied IVM.

    G = M^{-1}

    P_parallel = t (t^T G)/(t^T G t)
    P_perp     = I-P_parallel
    """
    tangent = np.asarray(
        tangent,
        dtype=float,
    )

    M_all = np.asarray(
        M_all,
        dtype=float,
    )

    n_internal, dimension = tangent.shape

    if M_all.shape != (
        n_internal,
        dimension,
        dimension,
    ):
        raise ValueError(
            f"M shape {M_all.shape} is inconsistent "
            f"with tangent shape {tangent.shape}."
        )

    G_all = np.linalg.inv(
        M_all
    )

    identity = np.eye(
        dimension,
        dtype=float,
    )

    P_parallel_all = np.zeros(
        (
            n_internal,
            dimension,
            dimension,
        ),
        dtype=float,
    )

    P_perp_all = np.zeros_like(
        P_parallel_all
    )

    for image_index in range(n_internal):
        tangent_i = tangent[image_index]
        G_i = G_all[image_index]

        denominator = float(
            tangent_i
            @ G_i
            @ tangent_i
        )

        if (
            not np.isfinite(denominator)
            or denominator <= 1.0e-30
        ):
            raise FloatingPointError(
                "Invalid G-tangent norm at image "
                f"{image_index}: {denominator}"
            )

        P_parallel_i = (
            np.outer(
                tangent_i,
                G_i @ tangent_i,
            )
            / denominator
        )

        P_parallel_all[image_index] = (
            P_parallel_i
        )

        P_perp_all[image_index] = (
            identity
            - P_parallel_i
        )

    return (
        G_all,
        P_parallel_all,
        P_perp_all,
    )


def apply_projectors(
    projectors: Array,
    vectors: Array,
) -> Array:
    return np.einsum(
        "nij,nj->ni",
        np.asarray(
            projectors,
            dtype=float,
        ),
        np.asarray(
            vectors,
            dtype=float,
        ),
    )


# ============================================================
# Euclidean perpendicular projection for diagnostics
# ============================================================

def euclidean_perpendicular(
    vectors: Array,
    tangent: Array,
) -> Array:
    vectors = np.asarray(
        vectors,
        dtype=float,
    )

    tangent = np.asarray(
        tangent,
        dtype=float,
    )

    tangent_norm = np.linalg.norm(
        tangent,
        axis=1,
        keepdims=True,
    )

    tangent_unit = (
        tangent
        / (tangent_norm + 1.0e-16)
    )

    tangential_component = np.sum(
        vectors * tangent_unit,
        axis=1,
        keepdims=True,
    )

    return (
        vectors
        - tangential_component
        * tangent_unit
    )


# ============================================================
# IVM-compatible path metrics
# ============================================================

def recover_free_energy_internal(
    internal_path: Array,
    grad_F: Array,
) -> Tuple[Array, Array]:
    """
    Match recover_V_along_curve() in the IVM code.

    The supplied IVM reconstructs F using internal images only.
    """
    internal_path = np.asarray(
        internal_path,
        dtype=float,
    )

    grad_F = np.asarray(
        grad_F,
        dtype=float,
    )

    n_internal = len(internal_path)

    if n_internal < 2:
        raise ValueError(
            "At least two internal images are required."
        )

    ds = 1.0 / float(
        n_internal - 1
    )

    phi_prime = np.zeros_like(
        internal_path
    )

    phi_prime[1:-1] = (
        internal_path[2:]
        - internal_path[:-2]
    ) / (2.0 * ds)

    phi_prime[0] = (
        internal_path[1]
        - internal_path[0]
    ) / ds

    phi_prime[-1] = (
        internal_path[-1]
        - internal_path[-2]
    ) / ds

    dF_ds = np.sum(
        grad_F * phi_prime,
        axis=1,
    )

    increments = (
        0.5
        * (
            dF_ds[:-1]
            + dF_ds[1:]
        )
        * ds
    )

    free_energy = np.zeros(
        n_internal,
        dtype=float,
    )

    free_energy[1:] = np.cumsum(
        increments
    )

    free_energy -= (
        np.min(free_energy)
        - 1.0e-6
    )

    return (
        free_energy,
        dF_ds,
    )


def compute_ivm_compatible_metrics(
    full_path: Array,
    M_all: Array,
    grad_F: Array,
) -> Dict[str, Any]:
    """
    Compute diagnostics using the same definitions as the supplied IVM.
    """
    full_path = np.asarray(
        full_path,
        dtype=float,
    )

    M_all = np.asarray(
        M_all,
        dtype=float,
    )

    grad_F = np.asarray(
        grad_F,
        dtype=float,
    )

    internal_path = (
        full_path[1:-1]
    )

    tangent = central_difference_tangent(
        full_path
    )

    (
        G_all,
        P_parallel_G,
        P_perp_G,
    ) = compute_G_projectors(
        tangent,
        M_all,
    )

    Mg = np.einsum(
        "nij,nj->ni",
        M_all,
        grad_F,
    )

    perpG_Mg = apply_projectors(
        P_perp_G,
        Mg,
    )

    perpE_Mg = euclidean_perpendicular(
        Mg,
        tangent,
    )

    norm_perp_E_sq = np.sum(
        perpG_Mg ** 2,
        axis=1,
    )

    norm_perp_E = np.sqrt(
        np.maximum(
            norm_perp_E_sq,
            0.0,
        )
    )

    fmax = float(
        np.max(
            norm_perp_E
        )
    )

    norm_perp_G_sq = np.einsum(
        "ni,nij,nj->n",
        perpG_Mg,
        G_all,
        perpG_Mg,
    )

    norm_perp_G_sq = np.maximum(
        norm_perp_G_sq,
        0.0,
    )

    # Same G-arclength quadrature weights used by IVM.
    segment_minus = (
        internal_path
        - full_path[:-2]
    )

    segment_plus = (
        full_path[2:]
        - internal_path
    )

    segment_minus_G_sq = np.einsum(
        "ni,nij,nj->n",
        segment_minus,
        G_all,
        segment_minus,
    )

    segment_plus_G_sq = np.einsum(
        "ni,nij,nj->n",
        segment_plus,
        G_all,
        segment_plus,
    )

    segment_minus_G = np.sqrt(
        np.maximum(
            segment_minus_G_sq,
            0.0,
        )
    )

    segment_plus_G = np.sqrt(
        np.maximum(
            segment_plus_G_sq,
            0.0,
        )
    )

    node_weights_G = 0.5 * (
        segment_minus_G
        + segment_plus_G
    )

    I_perp = float(
        np.sum(
            norm_perp_G_sq
            * node_weights_G
        )
    )

    I_perp_euclid_mean = float(
        np.mean(
            norm_perp_E_sq
        )
    )

    # G-angle, matching the IVM formula.
    tangent_G_norm = np.sqrt(
        np.maximum(
            np.einsum(
                "ni,nij,nj->n",
                tangent,
                G_all,
                tangent,
            ),
            0.0,
        )
    )

    grad_M_norm = np.sqrt(
        np.maximum(
            np.einsum(
                "ni,nij,nj->n",
                grad_F,
                M_all,
                grad_F,
            ),
            0.0,
        )
    )

    numerator = np.sum(
        tangent * grad_F,
        axis=1,
    )

    cosine = (
        numerator
        / (
            tangent_G_norm
            * grad_M_norm
            + 1.0e-16
        )
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    angle_residual = float(
        np.mean(
            1.0 - cosine ** 2
        )
    )

    free_energy, dF_ds = (
        recover_free_energy_internal(
            internal_path,
            grad_F,
        )
    )

    barrier = float(
        np.max(free_energy)
        - np.min(free_energy)
    )

    return {
        "G_all": G_all,
        "P_parallel_G": P_parallel_G,
        "P_perp_G": P_perp_G,

        "tangent": tangent,

        "Mg": Mg,
        "perpG_Mg": perpG_Mg,
        "perpE_Mg": perpE_Mg,

        "norm_perp_E": norm_perp_E,
        "norm_perp_G_sq": norm_perp_G_sq,

        "node_weights_G": node_weights_G,

        "I_perp": I_perp,
        "I_perp_euclid_mean": (
            I_perp_euclid_mean
        ),

        "fmax": fmax,
        "angle_residual": angle_residual,

        "free_energy": free_energy,
        "dF_ds": dF_ds,
        "barrier": barrier,
    }


# ============================================================
# Spring-only reparameterization
#
# This is deliberately identical to the IVM reparam_by_neb().
# ============================================================

def reparam_by_neb(
    full_path: Array,
    k_s: float = 1.0,
    dtau: float = 1.0e-3,
    max_iter: int = 5000,
    tol: float = 1.0e-3,
    verbose: bool = False,
) -> Tuple[Array, float, int, bool]:
    """
    Spring-only optimization:

        F_sp = k_s (l_plus-l_minus) tau

        X <- X + dtau F_sp

    This matches the supplied IVM implementation.
    """
    X = np.asarray(
        full_path,
        dtype=float,
    ).copy()

    residual = float("inf")
    converged = False
    used_iterations = 0

    for iteration in range(
        int(max_iter)
    ):
        segments = (
            X[1:]
            - X[:-1]
        )

        segment_lengths = np.linalg.norm(
            segments,
            axis=1,
        )

        length_minus = (
            segment_lengths[:-1]
        )

        length_plus = (
            segment_lengths[1:]
        )

        tangent = (
            X[2:]
            - X[:-2]
        )

        tangent_norm = np.linalg.norm(
            tangent,
            axis=1,
            keepdims=True,
        )

        tangent_unit = (
            tangent
            / (
                tangent_norm
                + 1.0e-16
            )
        )

        spring_force = (
            float(k_s)
            * (
                length_plus
                - length_minus
            )[:, None]
            * tangent_unit
        )

        residual = float(
            np.max(
                np.linalg.norm(
                    spring_force,
                    axis=1,
                )
            )
        )

        if residual < float(tol):
            converged = True
            used_iterations = iteration

            if verbose:
                print(
                    "  [FENEB reparam] "
                    f"converged at iteration "
                    f"{iteration}, "
                    f"max|F_sp|={residual:.6e}"
                )

            break

        X[1:-1] += (
            float(dtau)
            * spring_force
        )

        used_iterations = iteration + 1

    if not converged and verbose:
        print(
            "  [FENEB reparam] "
            f"hit max_iter={max_iter}, "
            f"max|F_sp|={residual:.6e}"
        )

    return (
        X,
        residual,
        used_iterations,
        converged,
    )


# ============================================================
# Uncoupled metric-FENEB optimizer
# ============================================================

class MetricFENEB:
    """
    FENEB extended to the same MFEP condition used by IVM.

    Parameters
    ----------
    oracle
        Callable with signature:

            oracle(q_ref_wrapped, image_index, previous_state)

        returning OracleResult.

    physical_step
        FENEB lambda in coordinate units (radians here).

    max_updates
        Number of physical FENEB updates. The oracle is evaluated
        max_updates+1 times, including the final diagnostic.
    """

    def __init__(
        self,
        endpoints: Array,
        n_internal: int,
        oracle: Callable[
            [Array, int, Any],
            OracleResult,
        ],
        *,
        periods: Optional[
            Sequence[float] | float
        ] = None,
        initial_path: Optional[Array] = None,
        physical_step: float = 0.005,
        spring_constant: float = 1.0,
        reparam_dtau: float = 1.0e-3,
        reparam_tolerance: float = 1.0e-3,
        reparam_max_iter: int = 5000,
        max_updates: int = 200,
        fmax_tolerance: float = 1.0e-2,
        early_stop: bool = False,
        callback: Optional[
            Callable[[Dict[str, Any]], None]
        ] = None,
        verbose: bool = True,
    ):
        endpoints = np.asarray(
            endpoints,
            dtype=float,
        )

        if (
            endpoints.ndim != 2
            or endpoints.shape[0] != 2
        ):
            raise ValueError(
                "endpoints must have shape "
                "(2, dimension)."
            )

        self.endpoints = endpoints.copy()
        self.dimension = endpoints.shape[1]

        self.n_internal = int(
            n_internal
        )

        self.n_total = (
            self.n_internal + 2
        )

        self.periods = as_periods(
            periods,
            self.dimension,
        )

        if initial_path is None:
            initial_path = (
                shortest_periodic_interpolation(
                    endpoints[0],
                    endpoints[1],
                    self.n_internal,
                    self.periods,
                )
            )

        initial_path = np.asarray(
            initial_path,
            dtype=float,
        )

        expected_shape = (
            self.n_total,
            self.dimension,
        )

        if initial_path.shape != expected_shape:
            raise ValueError(
                f"initial_path shape "
                f"{initial_path.shape}; "
                f"expected {expected_shape}."
            )

        self.reference_path = (
            initial_path.copy()
        )

        self.reference_path[0] = (
            initial_path[0]
        )

        self.reference_path[-1] = (
            initial_path[-1]
        )

        self.left_endpoint = (
            self.reference_path[0].copy()
        )

        self.right_endpoint = (
            self.reference_path[-1].copy()
        )

        self.oracle = oracle

        self.physical_step = float(
            physical_step
        )

        self.spring_constant = float(
            spring_constant
        )

        self.reparam_dtau = float(
            reparam_dtau
        )

        self.reparam_tolerance = float(
            reparam_tolerance
        )

        self.reparam_max_iter = int(
            reparam_max_iter
        )

        self.max_updates = int(
            max_updates
        )

        self.fmax_tolerance = float(
            fmax_tolerance
        )

        self.early_stop = bool(
            early_stop
        )

        self.callback = callback
        self.verbose = bool(verbose)

        self.oracle_states = [
            None
            for _ in range(
                self.n_internal
            )
        ]

        self.total_reparam_iterations = 0

    # --------------------------------------------------------
    # Restrained MD at every internal image
    # --------------------------------------------------------
    def sample_band(self):
        mean_path = (
            self.reference_path.copy()
        )

        M_all = np.zeros(
            (
                self.n_internal,
                self.dimension,
                self.dimension,
            ),
            dtype=float,
        )

        grad_F_all = np.zeros(
            (
                self.n_internal,
                self.dimension,
            ),
            dtype=float,
        )

        metadata_all = []

        for internal_index in range(
            self.n_internal
        ):
            path_index = (
                internal_index + 1
            )

            q_ref_unwrapped = (
                self.reference_path[
                    path_index
                ]
            )

            q_ref_wrapped = wrap_periodic(
                q_ref_unwrapped,
                self.periods,
            )

            result = self.oracle(
                q_ref_wrapped.copy(),
                internal_index,
                self.oracle_states[
                    internal_index
                ],
            )

            if not isinstance(
                result,
                OracleResult,
            ):
                raise TypeError(
                    "oracle must return "
                    "OracleResult."
                )

            q_mean_wrapped = np.asarray(
                result.q_mean,
                dtype=float,
            ).reshape(
                self.dimension
            )

            q_mean_unwrapped = unwrap_near(
                q_mean_wrapped,
                q_ref_unwrapped,
                self.periods,
            )

            M_i = np.asarray(
                result.M,
                dtype=float,
            ).reshape(
                self.dimension,
                self.dimension,
            )

            # Same symmetrization as FES_metric.
            M_i = 0.5 * (
                M_i + M_i.T
            )

            grad_F_i = np.asarray(
                result.grad_F,
                dtype=float,
            ).reshape(
                self.dimension
            )

            if not np.all(
                np.isfinite(
                    q_mean_unwrapped
                )
            ):
                raise FloatingPointError(
                    "Non-finite q_mean at image "
                    f"{internal_index}."
                )

            if not np.all(
                np.isfinite(M_i)
            ):
                raise FloatingPointError(
                    "Non-finite M at image "
                    f"{internal_index}."
                )

            if not np.all(
                np.isfinite(grad_F_i)
            ):
                raise FloatingPointError(
                    "Non-finite gradient at image "
                    f"{internal_index}."
                )

            mean_path[
                path_index
            ] = q_mean_unwrapped

            M_all[
                internal_index
            ] = M_i

            grad_F_all[
                internal_index
            ] = grad_F_i

            self.oracle_states[
                internal_index
            ] = result.state

            metadata_all.append(
                {}
                if result.metadata is None
                else dict(
                    result.metadata
                )
            )

        mean_path[0] = (
            self.left_endpoint
        )

        mean_path[-1] = (
            self.right_endpoint
        )

        return (
            mean_path,
            M_all,
            grad_F_all,
            metadata_all,
        )

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------
    def run(self):
        final_record = None

        # max_updates+1 oracle sweeps:
        #
        # sweep 0:
        #     initial diagnostic + update 1
        #
        # sweep max_updates:
        #     final diagnostic only
        for outer_index in range(
            self.max_updates + 1
        ):
            (
                mean_path,
                M_all,
                grad_F_all,
                metadata_all,
            ) = self.sample_band()

            metrics = (
                compute_ivm_compatible_metrics(
                    mean_path,
                    M_all,
                    grad_F_all,
                )
            )

            perpendicular_gradient = (
                metrics["perpG_Mg"]
            )

            physical_force = (
                -perpendicular_gradient
            )

            fmax = float(
                metrics["fmax"]
            )

            is_final_diagnostic = (
                outer_index
                == self.max_updates
            )

            convergence_reached = (
                fmax
                <= self.fmax_tolerance
            )

            should_stop_early = (
                self.early_stop
                and convergence_reached
            )

            intermediate_path = (
                mean_path.copy()
            )

            reparameterized_path = (
                mean_path.copy()
            )

            reparam_residual = 0.0
            reparam_iterations = 0
            reparam_converged = True
            maximum_physical_step = 0.0

            if (
                not is_final_diagnostic
                and not should_stop_early
            ):
                # Original FENEB-style global normalization:
                #
                # q_int = <q> + lambda F_perp/max|F_perp|
                if fmax > 1.0e-30:
                    physical_displacement = (
                        self.physical_step
                        * physical_force
                        / fmax
                    )
                else:
                    physical_displacement = (
                        np.zeros_like(
                            physical_force
                        )
                    )

                intermediate_path[
                    1:-1
                ] += physical_displacement

                intermediate_path[0] = (
                    self.left_endpoint
                )

                intermediate_path[-1] = (
                    self.right_endpoint
                )

                maximum_physical_step = float(
                    np.max(
                        np.linalg.norm(
                            physical_displacement,
                            axis=1,
                        )
                    )
                )

                (
                    reparameterized_path,
                    reparam_residual,
                    reparam_iterations,
                    reparam_converged,
                ) = reparam_by_neb(
                    intermediate_path,
                    k_s=self.spring_constant,
                    dtau=self.reparam_dtau,
                    max_iter=(
                        self.reparam_max_iter
                    ),
                    tol=(
                        self.reparam_tolerance
                    ),
                    verbose=(
                        self.verbose
                        and outer_index % 10 == 0
                    ),
                )

                self.total_reparam_iterations += (
                    reparam_iterations
                )

            record = {
                "outer_index": int(
                    outer_index
                ),

                "count": int(
                    outer_index
                ),

                "is_final_diagnostic": bool(
                    is_final_diagnostic
                ),

                "convergence_reached": bool(
                    convergence_reached
                ),

                "reference_path": (
                    self.reference_path.copy()
                ),

                "reference_path_wrapped": (
                    wrap_periodic(
                        self.reference_path,
                        self.periods,
                    )
                ),

                "mean_path": (
                    mean_path.copy()
                ),

                "mean_path_wrapped": (
                    wrap_periodic(
                        mean_path,
                        self.periods,
                    )
                ),

                "intermediate_path": (
                    intermediate_path.copy()
                ),

                "intermediate_path_wrapped": (
                    wrap_periodic(
                        intermediate_path,
                        self.periods,
                    )
                ),

                "reparameterized_path": (
                    reparameterized_path.copy()
                ),

                "reparameterized_path_wrapped": (
                    wrap_periodic(
                        reparameterized_path,
                        self.periods,
                    )
                ),

                "M_all": M_all.copy(),
                "grad_F_all": (
                    grad_F_all.copy()
                ),

                "physical_force": (
                    physical_force.copy()
                ),

                "physical_displacement": (
                    intermediate_path[1:-1]
                    - mean_path[1:-1]
                ),

                "maximum_physical_step": float(
                    maximum_physical_step
                ),

                "reparam_residual": float(
                    reparam_residual
                ),

                "reparam_iterations": int(
                    reparam_iterations
                ),

                "reparam_converged": bool(
                    reparam_converged
                ),

                "total_reparam_iterations": int(
                    self.total_reparam_iterations
                ),

                "metrics": metrics,
                "oracle_metadata": metadata_all,
            }

            final_record = record

            if self.callback is not None:
                self.callback(record)

            if (
                is_final_diagnostic
                or should_stop_early
            ):
                self.reference_path = (
                    mean_path.copy()
                )

                return {
                    "converged": bool(
                        convergence_reached
                    ),

                    "outer_index": int(
                        outer_index
                    ),

                    "updates_completed": int(
                        outer_index
                    ),

                    "final_path": (
                        mean_path.copy()
                    ),

                    "final_path_wrapped": (
                        wrap_periodic(
                            mean_path,
                            self.periods,
                        )
                    ),

                    "final_record": record,
                }

            self.reference_path = (
                reparameterized_path.copy()
            )

        raise RuntimeError(
            "Unexpected termination of "
            "MetricFENEB.run()."
        )