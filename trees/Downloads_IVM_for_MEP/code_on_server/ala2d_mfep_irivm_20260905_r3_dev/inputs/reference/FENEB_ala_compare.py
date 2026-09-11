# -*- coding: utf-8 -*-

"""
Alanine-dipeptide metric-FENEB calculation designed for direct comparison
with the supplied IVM implementation.

Important matched settings
--------------------------
IVM and FENEB both use:

    N internal images       = 20
    total images            = 22
    endpoints               = identical
    initial path            = straight interpolation
    random seed             = 41
    discard fraction        = 0.05
    M_gradient              = identical
    target residual         = P_G M grad F
    G                       = M^{-1}
    spring k                = 1.0
    spring dtau             = 1e-3
    spring tolerance        = 1e-3
    spring maximum steps    = 5000
    physical-update budget  = 200
    oracle calls per image  = 201 including final diagnostic

The FENEB-specific physical update is

    q_int = q_mean
            - lambda * P_G M grad F
            / max_i |P_G M grad F|.

The spring relaxation is then performed separately.
"""

from __future__ import annotations

import os
import time
import traceback

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


from feneb_core import (
    MetricFENEB,
    OracleResult,
    shortest_periodic_interpolation,
)


# ============================================================
# Import the user's existing MD oracle and plotting function
# ============================================================

import IVM_shell as ala_ivm


M_gradient = ala_ivm.M_gradient
fig_save = ala_ivm.fig_save

if hasattr(
    ala_ivm,
    "GROMACS_QUIET",
):
    ala_ivm.GROMACS_QUIET = True


# ============================================================
# Numerical settings
# ============================================================

device = torch.device("cpu")
dtype = torch.float64

SEED = 41

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Same endpoints as IVM
# ============================================================

C1 = np.array(
    [-2.60866606, 2.77661136],
    dtype=float,
)

C2 = np.array(
    [-1.33047160, 0.99118795],
    dtype=float,
)

ENDPOINTS = np.stack(
    [C1, C2],
    axis=0,
)


# ============================================================
# Same image number as IVM
# ============================================================

N = 20
N_INTERNAL = N
N_TOTAL_IMAGES = N + 2


# ============================================================
# Angular periodicity
# ============================================================

PERIODS = np.array(
    [
        2.0 * np.pi,
        2.0 * np.pi,
    ],
    dtype=float,
)


# ============================================================
# Same restrained-MD settings as IVM
# ============================================================

DISCARD_FRAC = 0.05
TMIN = None
TMAX = None

# M_gradient/FES_gradient uses kappa=1000 in the native
# GROMACS/PLUMED energy unit.
KAPPA_RAW = 1000.0

# M_gradient returns the gradient in kJ/mol/rad.
KJ_TO_KCAL = 1.0 / 4.184


# ============================================================
# FENEB physical-step grid
#
# Because FENEB globally normalizes the force, lambda is the
# maximum physical displacement in radians per outer update.
# ============================================================

GRID_PHYSICAL_STEP = [
    1.0e-1,
    5.0e-2,
    2.0e-2,
    # 1.0e-2,
    # 5.0e-3,
    # 2.0e-3,
    # 1.0e-3,
]


# ============================================================
# Same spring reparameterization as IVM
# ============================================================

REPARAM_KS = 1.0
REPARAM_DTAU = 1.0e-3
REPARAM_TOL = 1.0e-3
REPARAM_MAXITER = 5000


# ============================================================
# Same total first-order/update budget as IVM
#
# IVM:
#     TOTAL_INNER_STEPS=200
#
# FENEB:
#     200 physical updates + final diagnostic.
# ============================================================

MAX_FENEB_UPDATES = 200

# To ensure a fixed-budget comparison, do not stop before 200.
EARLY_STOP = False

# Still record whether this tolerance has been reached.
FENEB_FMAX_TOL = 1.0e-2


# ============================================================
# Save settings
# ============================================================

BASE_PATH = (
    "/Users/wuzhiyou/Code/IVM/"
    "MFEP_ala/amber14sb_vacuo"
)

FORCE_RERUN = True

SAVE_EVERY_STATE = True
SAVE_FIGS = True

# Save at 0,4,8,...,200 to match IVM count_seq.
SAVE_FIG_EVERY_COUNT = 4

XLIM = (-np.pi, 0.0)
YLIM = (0.0, np.pi)


# ============================================================
# Oracle
# ============================================================

class AlanineFENEBOracle:
    """
    Adapter around the exact same M_gradient() used by IVM.

    M_gradient returns:

        M, grad_F_raw

    where

        grad_F_raw
        = -KAPPA_RAW * mean(q_sample-q_ref).

    Therefore:

        q_mean
        = q_ref - grad_F_raw/KAPPA_RAW.

    This reconstructs the restrained-MD mean coordinate required by FENEB
    without modifying IVM_shell.py.
    """

    def __init__(self):
        self.total_image_calls = 0

        self.per_image_calls = np.zeros(
            N_INTERNAL,
            dtype=int,
        )

    def __call__(
        self,
        q_reference_wrapped,
        image_index,
        previous_state,
    ):
        del previous_state

        q_reference_wrapped = np.asarray(
            q_reference_wrapped,
            dtype=float,
        ).reshape(2)

        q_tensor = torch.tensor(
            q_reference_wrapped.reshape(
                1,
                2,
            ),
            dtype=dtype,
            device=device,
        )

        M_i, grad_F_raw = M_gradient(
            q_tensor,
            discard_frac=DISCARD_FRAC,
            tmin=TMIN,
            tmax=TMAX,
        )

        M_i = np.asarray(
            M_i,
            dtype=float,
        ).reshape(
            2,
            2,
        )

        M_i = 0.5 * (
            M_i + M_i.T
        )

        grad_F_raw = np.asarray(
            grad_F_raw,
            dtype=float,
        ).reshape(2)

        # Same unit conversion as IVM.
        grad_F_kcal = (
            grad_F_raw
            * KJ_TO_KCAL
        )

        # Reconstruct q_mean before converting the gradient.
        #
        # grad_F_raw = -KAPPA_RAW * mean(q-q_ref)
        #
        # q_mean = q_ref - grad_F_raw/KAPPA_RAW
        q_mean = (
            q_reference_wrapped
            - grad_F_raw / KAPPA_RAW
        )

        q_mean = (
            (q_mean + np.pi)
            % (2.0 * np.pi)
            - np.pi
        )

        if not np.all(
            np.isfinite(M_i)
        ):
            raise FloatingPointError(
                f"Non-finite M at image "
                f"{image_index}:\n{M_i}"
            )

        if not np.all(
            np.isfinite(
                grad_F_kcal
            )
        ):
            raise FloatingPointError(
                f"Non-finite gradient at image "
                f"{image_index}: "
                f"{grad_F_kcal}"
            )

        if not np.all(
            np.isfinite(q_mean)
        ):
            raise FloatingPointError(
                f"Non-finite q_mean at image "
                f"{image_index}: {q_mean}"
            )

        self.total_image_calls += 1

        self.per_image_calls[
            image_index
        ] += 1

        return OracleResult(
            q_mean=q_mean,
            M=M_i,
            grad_F=grad_F_kcal,
            state=None,
            metadata={
                "q_reference": (
                    q_reference_wrapped.copy()
                ),

                "q_mean": (
                    q_mean.copy()
                ),

                "M": M_i.copy(),

                "grad_F_raw": (
                    grad_F_raw.copy()
                ),

                "grad_F_kcal": (
                    grad_F_kcal.copy()
                ),

                "image_call": int(
                    self.per_image_calls[
                        image_index
                    ]
                ),
            },
        )


# ============================================================
# File helpers
# ============================================================

def float_tag(value):
    return f"{float(value):g}"


def atomic_savez(
    filename,
    **kwargs,
):
    temporary_filename = (
        filename + ".tmp.npz"
    )

    np.savez(
        temporary_filename,
        **kwargs,
    )

    os.replace(
        temporary_filename,
        filename,
    )


def result_paths(
    physical_step,
):
    step_tag = float_tag(
        physical_step
    )

    case_number = (
        f"MFEP_FENEB_COMPARE"
        f"_N{N_INTERNAL}"
        f"_lambda{step_tag}"
        f"_Nupdates{MAX_FENEB_UPDATES}"
        f"_Kspring{REPARAM_KS:g}"
        f"_dtau{REPARAM_DTAU:g}"
        f"_reparamTol{REPARAM_TOL:g}"
        f"_sameIVMsettings"
    )

    folder_path = os.path.join(
        BASE_PATH,
        case_number,
    )

    figure_path = os.path.join(
        folder_path,
        "figures",
    )

    result_file = os.path.join(
        folder_path,
        f"alanine_{case_number}.npz",
    )

    done_file = os.path.join(
        folder_path,
        "DONE",
    )

    failed_file = os.path.join(
        folder_path,
        "FAILED",
    )

    return (
        case_number,
        folder_path,
        figure_path,
        result_file,
        done_file,
        failed_file,
    )


# ============================================================
# Single experiment
# ============================================================

def run_single_experiment(
    physical_step,
):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    physical_step = float(
        physical_step
    )

    (
        case_number,
        folder_path,
        figure_path,
        result_file,
        done_file,
        failed_file,
    ) = result_paths(
        physical_step
    )

    os.makedirs(
        folder_path,
        exist_ok=True,
    )

    os.makedirs(
        figure_path,
        exist_ok=True,
    )

    if FORCE_RERUN:
        for marker in [
            done_file,
            failed_file,
        ]:
            if os.path.exists(marker):
                os.remove(marker)

    if (
        not FORCE_RERUN
        and os.path.exists(done_file)
        and os.path.exists(result_file)
    ):
        print(
            f"[SKIP] lambda="
            f"{physical_step:g}"
        )

        with np.load(
            result_file,
            allow_pickle=True,
        ) as loaded:
            return {
                "physical_step": (
                    physical_step
                ),

                "failed": False,
                "skipped": True,

                "final_I_perp": float(
                    loaded[
                        "I_perp_seq"
                    ][-1]
                ),

                "final_fmax": float(
                    loaded[
                        "fmax_seq"
                    ][-1]
                ),

                "result_file": (
                    result_file
                ),
            }

    print("\n" + "=" * 110)
    print("ALANINE MFEP — FENEB/IVM FAIR COMPARISON")
    print("=" * 110)
    print(f"Internal images         : {N_INTERNAL}")
    print(f"Total images            : {N_TOTAL_IMAGES}")
    print(f"Physical updates        : {MAX_FENEB_UPDATES}")
    print(f"Oracle calls per image  : {MAX_FENEB_UPDATES + 1}")
    print(f"Physical step lambda    : {physical_step:g} rad")
    print(f"Spring k                : {REPARAM_KS:g}")
    print(f"Spring dtau             : {REPARAM_DTAU:g}")
    print(f"Spring tolerance        : {REPARAM_TOL:g}")
    print(f"Spring maximum iter     : {REPARAM_MAXITER}")
    print(f"Discard fraction        : {DISCARD_FRAC:g}")
    print(f"Early stop              : {EARLY_STOP}")
    print(f"Output                   : {folder_path}")
    print("=" * 110)

    initial_path = (
        shortest_periodic_interpolation(
            C1,
            C2,
            N_INTERNAL,
            PERIODS,
        )
    )

    oracle = AlanineFENEBOracle()

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # History
    # --------------------------------------------------------
    history = {
        "index_seq": [],
        "count_seq": [],

        "total_image_md_calls_seq": [],
        "per_image_md_calls_seq": [],
        "path_gradient_sweeps_seq": [],

        "I_perp_seq": [],
        "I_perp_euclid_mean_seq": [],
        "fmax_seq": [],
        "cos_seq": [],

        "reparam_resid_seq": [],
        "reparam_iterations_seq": [],
        "reparam_converged_seq": [],
        "total_reparam_iterations_seq": [],

        "max_normal_step_seq": [],

        "barrier_seq": [],
        "free_energy_profile_seq": [],

        "reference_path_seq": [],
        "mean_path_seq": [],
        "intermediate_path_seq": [],
        "reparameterized_path_seq": [],

        "reference_path_unwrapped_seq": [],
        "mean_path_unwrapped_seq": [],
        "intermediate_path_unwrapped_seq": [],
        "reparameterized_path_unwrapped_seq": [],

        "M_ref_seq": [],
        "G_ref_seq": [],
        "F_ref_seq": [],
        "Mg_ref_seq": [],

        "perpG_Mg_seq": [],
        "perpE_Mg_seq": [],
        "P_perp_G_seq": [],
        "node_weights_seq": [],
    }

    def save_current_result():
        atomic_savez(
            result_file,

            # ------------------------------------------------
            # IVM-compatible keys
            # ------------------------------------------------
            index_seq=np.asarray(
                history[
                    "index_seq"
                ],
                dtype=int,
            ),

            count_seq=np.asarray(
                history[
                    "count_seq"
                ],
                dtype=int,
            ),

            I_perp_seq=np.asarray(
                history[
                    "I_perp_seq"
                ],
                dtype=float,
            ),

            I_perp_euclid_mean_seq=np.asarray(
                history[
                    "I_perp_euclid_mean_seq"
                ],
                dtype=float,
            ),

            fmax_seq=np.asarray(
                history[
                    "fmax_seq"
                ],
                dtype=float,
            ),

            cos_seq=np.asarray(
                history[
                    "cos_seq"
                ],
                dtype=float,
            ),

            reparam_resid_seq=np.asarray(
                history[
                    "reparam_resid_seq"
                ],
                dtype=float,
            ),

            max_normal_step_seq=np.asarray(
                history[
                    "max_normal_step_seq"
                ],
                dtype=float,
            ),

            # ------------------------------------------------
            # Cost counters
            # ------------------------------------------------
            total_image_md_calls_seq=np.asarray(
                history[
                    "total_image_md_calls_seq"
                ],
                dtype=int,
            ),

            per_image_md_calls_seq=np.asarray(
                history[
                    "per_image_md_calls_seq"
                ],
                dtype=int,
            ),

            path_gradient_sweeps_seq=np.asarray(
                history[
                    "path_gradient_sweeps_seq"
                ],
                dtype=float,
            ),

            # ------------------------------------------------
            # FENEB diagnostics
            # ------------------------------------------------
            reparam_iterations_seq=np.asarray(
                history[
                    "reparam_iterations_seq"
                ],
                dtype=int,
            ),

            reparam_converged_seq=np.asarray(
                history[
                    "reparam_converged_seq"
                ],
                dtype=bool,
            ),

            total_reparam_iterations_seq=np.asarray(
                history[
                    "total_reparam_iterations_seq"
                ],
                dtype=int,
            ),

            barrier_seq=np.asarray(
                history[
                    "barrier_seq"
                ],
                dtype=float,
            ),

            free_energy_profile_seq=np.asarray(
                history[
                    "free_energy_profile_seq"
                ],
                dtype=float,
            ),

            # ------------------------------------------------
            # Paths
            # ------------------------------------------------
            reference_path_seq=np.asarray(
                history[
                    "reference_path_seq"
                ],
                dtype=float,
            ),

            mean_path_seq=np.asarray(
                history[
                    "mean_path_seq"
                ],
                dtype=float,
            ),

            intermediate_path_seq=np.asarray(
                history[
                    "intermediate_path_seq"
                ],
                dtype=float,
            ),

            reparameterized_path_seq=np.asarray(
                history[
                    "reparameterized_path_seq"
                ],
                dtype=float,
            ),

            reference_path_unwrapped_seq=np.asarray(
                history[
                    "reference_path_unwrapped_seq"
                ],
                dtype=float,
            ),

            mean_path_unwrapped_seq=np.asarray(
                history[
                    "mean_path_unwrapped_seq"
                ],
                dtype=float,
            ),

            intermediate_path_unwrapped_seq=np.asarray(
                history[
                    "intermediate_path_unwrapped_seq"
                ],
                dtype=float,
            ),

            reparameterized_path_unwrapped_seq=np.asarray(
                history[
                    "reparameterized_path_unwrapped_seq"
                ],
                dtype=float,
            ),

            # ------------------------------------------------
            # Oracle and projected fields
            # ------------------------------------------------
            M_ref_seq=np.asarray(
                history[
                    "M_ref_seq"
                ],
                dtype=float,
            ),

            G_ref_seq=np.asarray(
                history[
                    "G_ref_seq"
                ],
                dtype=float,
            ),

            F_ref_seq=np.asarray(
                history[
                    "F_ref_seq"
                ],
                dtype=float,
            ),

            Mg_ref_seq=np.asarray(
                history[
                    "Mg_ref_seq"
                ],
                dtype=float,
            ),

            perpG_Mg_seq=np.asarray(
                history[
                    "perpG_Mg_seq"
                ],
                dtype=float,
            ),

            perpE_Mg_seq=np.asarray(
                history[
                    "perpE_Mg_seq"
                ],
                dtype=float,
            ),

            P_perp_G_seq=np.asarray(
                history[
                    "P_perp_G_seq"
                ],
                dtype=float,
            ),

            node_weights_seq=np.asarray(
                history[
                    "node_weights_seq"
                ],
                dtype=float,
            ),

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------
            method=np.asarray(
                [
                    "uncoupled_metric_FENEB"
                ],
                dtype=str,
            ),

            target_condition=np.asarray(
                [
                    "P_G_M_grad_F_equals_zero"
                ],
                dtype=str,
            ),

            N_internal=np.asarray(
                [N_INTERNAL],
                dtype=int,
            ),

            N_total_images=np.asarray(
                [N_TOTAL_IMAGES],
                dtype=int,
            ),

            endpoints=ENDPOINTS.copy(),
            periods=PERIODS.copy(),

            physical_step=np.asarray(
                [physical_step],
                dtype=float,
            ),

            spring_constant=np.asarray(
                [REPARAM_KS],
                dtype=float,
            ),

            reparam_dtau=np.asarray(
                [REPARAM_DTAU],
                dtype=float,
            ),

            reparam_tolerance=np.asarray(
                [REPARAM_TOL],
                dtype=float,
            ),

            reparam_max_iter=np.asarray(
                [REPARAM_MAXITER],
                dtype=int,
            ),

            max_updates=np.asarray(
                [MAX_FENEB_UPDATES],
                dtype=int,
            ),

            discard_frac=np.asarray(
                [DISCARD_FRAC],
                dtype=float,
            ),

            kappa_raw=np.asarray(
                [KAPPA_RAW],
                dtype=float,
            ),

            early_stop=np.asarray(
                [EARLY_STOP],
                dtype=bool,
            ),
        )

    # --------------------------------------------------------
    # Callback
    # --------------------------------------------------------
    def record_state(record):
        count = int(
            record["count"]
        )

        metrics = record[
            "metrics"
        ]

        history[
            "index_seq"
        ].append(count)

        history[
            "count_seq"
        ].append(count)

        history[
            "total_image_md_calls_seq"
        ].append(
            int(
                oracle.total_image_calls
            )
        )

        history[
            "per_image_md_calls_seq"
        ].append(
            oracle.per_image_calls.copy()
        )

        history[
            "path_gradient_sweeps_seq"
        ].append(
            float(
                oracle.total_image_calls
                / N_INTERNAL
            )
        )

        history[
            "I_perp_seq"
        ].append(
            float(
                metrics["I_perp"]
            )
        )

        history[
            "I_perp_euclid_mean_seq"
        ].append(
            float(
                metrics[
                    "I_perp_euclid_mean"
                ]
            )
        )

        history[
            "fmax_seq"
        ].append(
            float(
                metrics["fmax"]
            )
        )

        history[
            "cos_seq"
        ].append(
            float(
                metrics[
                    "angle_residual"
                ]
            )
        )

        history[
            "reparam_resid_seq"
        ].append(
            float(
                record[
                    "reparam_residual"
                ]
            )
        )

        history[
            "reparam_iterations_seq"
        ].append(
            int(
                record[
                    "reparam_iterations"
                ]
            )
        )

        history[
            "reparam_converged_seq"
        ].append(
            bool(
                record[
                    "reparam_converged"
                ]
            )
        )

        history[
            "total_reparam_iterations_seq"
        ].append(
            int(
                record[
                    "total_reparam_iterations"
                ]
            )
        )

        history[
            "max_normal_step_seq"
        ].append(
            float(
                record[
                    "maximum_physical_step"
                ]
            )
        )

        history[
            "barrier_seq"
        ].append(
            float(
                metrics["barrier"]
            )
        )

        history[
            "free_energy_profile_seq"
        ].append(
            metrics[
                "free_energy"
            ].copy()
        )

        history[
            "reference_path_seq"
        ].append(
            record[
                "reference_path_wrapped"
            ].copy()
        )

        history[
            "mean_path_seq"
        ].append(
            record[
                "mean_path_wrapped"
            ].copy()
        )

        history[
            "intermediate_path_seq"
        ].append(
            record[
                "intermediate_path_wrapped"
            ].copy()
        )

        history[
            "reparameterized_path_seq"
        ].append(
            record[
                "reparameterized_path_wrapped"
            ].copy()
        )

        history[
            "reference_path_unwrapped_seq"
        ].append(
            record[
                "reference_path"
            ].copy()
        )

        history[
            "mean_path_unwrapped_seq"
        ].append(
            record[
                "mean_path"
            ].copy()
        )

        history[
            "intermediate_path_unwrapped_seq"
        ].append(
            record[
                "intermediate_path"
            ].copy()
        )

        history[
            "reparameterized_path_unwrapped_seq"
        ].append(
            record[
                "reparameterized_path"
            ].copy()
        )

        history[
            "M_ref_seq"
        ].append(
            record[
                "M_all"
            ].copy()
        )

        history[
            "G_ref_seq"
        ].append(
            metrics[
                "G_all"
            ].copy()
        )

        history[
            "F_ref_seq"
        ].append(
            record[
                "grad_F_all"
            ].copy()
        )

        history[
            "Mg_ref_seq"
        ].append(
            metrics[
                "Mg"
            ].copy()
        )

        history[
            "perpG_Mg_seq"
        ].append(
            metrics[
                "perpG_Mg"
            ].copy()
        )

        history[
            "perpE_Mg_seq"
        ].append(
            metrics[
                "perpE_Mg"
            ].copy()
        )

        history[
            "P_perp_G_seq"
        ].append(
            metrics[
                "P_perp_G"
            ].copy()
        )

        history[
            "node_weights_seq"
        ].append(
            metrics[
                "node_weights_G"
            ].copy()
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        print(
            f"[FENEB lambda="
            f"{physical_step:g}] "
            f"count={count:>3d}  "
            f"I_perp="
            f"{metrics['I_perp']:.4e}  "
            f"I_perp_euclid_mean="
            f"{metrics['I_perp_euclid_mean']:.4e}  "
            f"fmax="
            f"{metrics['fmax']:.4e}  "
            f"max_normal_step="
            f"{record['maximum_physical_step']:.4e} rad  "
            f"reparam_resid="
            f"{record['reparam_residual']:.4e}  "
            f"MD_calls/image="
            f"{oracle.per_image_calls[0]:>3d}  "
            f"elapsed={elapsed:.1f}s"
            + (
                "  [FINAL DIAGNOSTIC]"
                if record[
                    "is_final_diagnostic"
                ]
                else ""
            )
        )

        # Save figures at exactly the same count spacing as IVM.
        should_save_figure = (
            SAVE_FIGS
            and (
                count == 0
                or count
                % SAVE_FIG_EVERY_COUNT
                == 0
                or count
                == MAX_FENEB_UPDATES
            )
        )

        if should_save_figure:
            try:
                x_reference = (
                    record[
                        "mean_path_wrapped"
                    ][1:-1]
                )

                x_intermediate = (
                    record[
                        "intermediate_path_wrapped"
                    ][1:-1]
                )

                x_reparameterized = (
                    record[
                        "reparameterized_path_wrapped"
                    ][1:-1]
                )

                fig_save(
                    count,

                    x_reference,
                    x_intermediate,
                    x_intermediate,
                    x_reparameterized,

                    xlim=XLIM,
                    ylim=YLIM,

                    # Same three vectors used in IVM figures.
                    force1=-metrics[
                        "perpG_Mg"
                    ],

                    force2=-record[
                        "grad_F_all"
                    ],

                    force3=-metrics[
                        "perpE_Mg"
                    ],

                    figure_path=figure_path,
                )

            except Exception as exc:
                print(
                    "[WARNING] fig_save failed "
                    f"at count={count}: {exc}"
                )

            plt.close("all")

        if SAVE_EVERY_STATE:
            save_current_result()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = MetricFENEB(
        endpoints=ENDPOINTS,
        n_internal=N_INTERNAL,
        oracle=oracle,

        periods=PERIODS,
        initial_path=initial_path,

        physical_step=physical_step,

        spring_constant=(
            REPARAM_KS
        ),

        reparam_dtau=(
            REPARAM_DTAU
        ),

        reparam_tolerance=(
            REPARAM_TOL
        ),

        reparam_max_iter=(
            REPARAM_MAXITER
        ),

        max_updates=(
            MAX_FENEB_UPDATES
        ),

        fmax_tolerance=(
            FENEB_FMAX_TOL
        ),

        early_stop=EARLY_STOP,

        callback=record_state,
        verbose=True,
    )

    result = optimizer.run()

    save_current_result()

    elapsed_total = (
        time.perf_counter()
        - start_time
    )

    final_I_perp = float(
        history[
            "I_perp_seq"
        ][-1]
    )

    final_I_perp_euclid_mean = float(
        history[
            "I_perp_euclid_mean_seq"
        ][-1]
    )

    final_fmax = float(
        history[
            "fmax_seq"
        ][-1]
    )

    final_cos = float(
        history[
            "cos_seq"
        ][-1]
    )

    final_barrier = float(
        history[
            "barrier_seq"
        ][-1]
    )

    final_count = int(
        history[
            "count_seq"
        ][-1]
    )

    with open(
        done_file,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "completed\n"
            f"method=uncoupled_metric_FENEB\n"
            f"physical_step={physical_step}\n"
            f"N_internal={N_INTERNAL}\n"
            f"N_total={N_TOTAL_IMAGES}\n"
            f"max_updates={MAX_FENEB_UPDATES}\n"
            f"final_count={final_count}\n"
            f"oracle_calls_per_image="
            f"{oracle.per_image_calls[0]}\n"
            f"total_image_calls="
            f"{oracle.total_image_calls}\n"
            f"final_I_perp="
            f"{final_I_perp}\n"
            f"final_I_perp_euclid_mean="
            f"{final_I_perp_euclid_mean}\n"
            f"final_fmax={final_fmax}\n"
            f"final_cos={final_cos}\n"
            f"final_barrier="
            f"{final_barrier}\n"
            f"elapsed_seconds="
            f"{elapsed_total}\n"
            f"result_file={result_file}\n"
        )

    print("\n" + "=" * 110)
    print("FENEB EXPERIMENT FINISHED")
    print("=" * 110)
    print(
        f"Physical step          : "
        f"{physical_step:g}"
    )
    print(
        f"Final count            : "
        f"{final_count}"
    )
    print(
        f"Oracle calls/image     : "
        f"{oracle.per_image_calls[0]}"
    )
    print(
        f"Total image calls      : "
        f"{oracle.total_image_calls}"
    )
    print(
        f"Final I_perp           : "
        f"{final_I_perp:.6e}"
    )
    print(
        f"Final Euclidean mean   : "
        f"{final_I_perp_euclid_mean:.6e}"
    )
    print(
        f"Final fmax             : "
        f"{final_fmax:.6e}"
    )
    print(
        f"Final angle residual   : "
        f"{final_cos:.6e}"
    )
    print(
        f"Final barrier          : "
        f"{final_barrier:.6f}"
    )
    print(
        f"Elapsed seconds        : "
        f"{elapsed_total:.1f}"
    )
    print(
        f"Result file            : "
        f"{result_file}"
    )
    print("=" * 110)

    return {
        "physical_step": (
            physical_step
        ),

        "failed": False,
        "skipped": False,

        "converged": bool(
            result["converged"]
        ),

        "final_count": (
            final_count
        ),

        "oracle_calls_per_image": int(
            oracle.per_image_calls[0]
        ),

        "total_image_calls": int(
            oracle.total_image_calls
        ),

        "final_I_perp": (
            final_I_perp
        ),

        "final_I_perp_G": (
            final_I_perp
        ),

        "final_I_perp_euclid_mean": (
            final_I_perp_euclid_mean
        ),

        "final_fmax": (
            final_fmax
        ),

        "final_cos": (
            final_cos
        ),

        "final_barrier": (
            final_barrier
        ),

        "elapsed_seconds": (
            elapsed_total
        ),

        "result_file": (
            result_file
        ),

        "folder_path": (
            folder_path
        ),
    }


# ============================================================
# Summary
# ============================================================

def save_summary(
    sweep_results,
):
    summary_file = os.path.join(
        BASE_PATH,
        (
            "FENEB_IVM_FAIR_COMPARISON_"
            f"N{N_INTERNAL}_"
            f"UPDATES{MAX_FENEB_UPDATES}_"
            "SUMMARY.npz"
        ),
    )

    atomic_savez(
        summary_file,

        results=np.asarray(
            sweep_results,
            dtype=object,
        ),

        physical_step_grid=np.asarray(
            GRID_PHYSICAL_STEP,
            dtype=float,
        ),

        N_internal=np.asarray(
            [N_INTERNAL],
            dtype=int,
        ),

        N_total_images=np.asarray(
            [N_TOTAL_IMAGES],
            dtype=int,
        ),

        max_updates=np.asarray(
            [MAX_FENEB_UPDATES],
            dtype=int,
        ),

        oracle_calls_per_image=np.asarray(
            [MAX_FENEB_UPDATES + 1],
            dtype=int,
        ),

        spring_constant=np.asarray(
            [REPARAM_KS],
            dtype=float,
        ),

        reparam_dtau=np.asarray(
            [REPARAM_DTAU],
            dtype=float,
        ),

        reparam_tolerance=np.asarray(
            [REPARAM_TOL],
            dtype=float,
        ),

        endpoints=ENDPOINTS.copy(),
    )

    return summary_file


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 120)
    print("ALANINE MFEP — FENEB GRID FOR IVM COMPARISON")
    print("=" * 120)
    print(
        f"Internal images        : "
        f"{N_INTERNAL}"
    )
    print(
        f"Total images           : "
        f"{N_TOTAL_IMAGES}"
    )
    print(
        f"Physical-step grid     : "
        f"{GRID_PHYSICAL_STEP}"
    )
    print(
        f"Physical updates       : "
        f"{MAX_FENEB_UPDATES}"
    )
    print(
        f"Oracle calls/image     : "
        f"{MAX_FENEB_UPDATES + 1}"
    )
    print(
        f"Spring settings        : "
        f"k={REPARAM_KS}, "
        f"dtau={REPARAM_DTAU}, "
        f"tol={REPARAM_TOL}, "
        f"maxiter={REPARAM_MAXITER}"
    )
    print(
        f"Base path              : "
        f"{BASE_PATH}"
    )
    print("=" * 120)

    sweep_results = []

    for experiment_index, physical_step in enumerate(
        GRID_PHYSICAL_STEP,
        start=1,
    ):
        print("\n" + "#" * 120)
        print(
            f"RUNNING FENEB "
            f"{experiment_index}/"
            f"{len(GRID_PHYSICAL_STEP)}"
        )
        print(
            f"physical step = "
            f"{physical_step:g}"
        )
        print("#" * 120)

        (
            case_number,
            folder_path,
            figure_path,
            result_file,
            done_file,
            failed_file,
        ) = result_paths(
            physical_step
        )

        try:
            result = run_single_experiment(
                physical_step
            )

        except KeyboardInterrupt:
            summary_file = save_summary(
                sweep_results
            )

            print(
                "\nPartial summary saved to:"
            )
            print(summary_file)

            raise

        except Exception as exc:
            print(
                f"\n[ERROR] physical_step="
                f"{physical_step:g} failed."
            )

            print(
                f"Type   : "
                f"{type(exc).__name__}"
            )

            print(
                f"Reason : {exc}"
            )

            traceback.print_exc()

            os.makedirs(
                folder_path,
                exist_ok=True,
            )

            with open(
                failed_file,
                "w",
                encoding="utf-8",
            ) as file:
                file.write(
                    "failed\n"
                    f"physical_step="
                    f"{physical_step}\n"
                    f"error_type="
                    f"{type(exc).__name__}\n"
                    f"error_message="
                    f"{str(exc)}\n"
                )

            result = {
                "physical_step": float(
                    physical_step
                ),

                "failed": True,
                "skipped": False,

                "error_type": (
                    type(exc).__name__
                ),

                "error_message": (
                    str(exc)
                ),

                "result_file": (
                    result_file
                ),
            }

        sweep_results.append(
            result
        )

        current_summary = save_summary(
            sweep_results
        )

        print(
            "\nCurrent summary:"
        )
        print(current_summary)

    successful_results = [
        result
        for result in sweep_results
        if not result.get(
            "failed",
            False,
        )
    ]

    failed_results = [
        result
        for result in sweep_results
        if result.get(
            "failed",
            False,
        )
    ]

    successful_results = sorted(
        successful_results,
        key=lambda result: result.get(
            "final_I_perp",
            np.inf,
        ),
    )

    print("\n" + "=" * 150)
    print("FENEB GRID FINAL SUMMARY")
    print("=" * 150)

    for rank, result in enumerate(
        successful_results,
        start=1,
    ):
        print(
            f"rank={rank:2d}"
            f" | lambda="
            f"{result['physical_step']:>10g}"
            f" | I_perp="
            f"{result['final_I_perp']:.6e}"
            f" | I_euclid_mean="
            f"{result['final_I_perp_euclid_mean']:.6e}"
            f" | fmax="
            f"{result['final_fmax']:.6e}"
            f" | barrier="
            f"{result['final_barrier']:.6f}"
            f" | MD calls/image="
            f"{result['oracle_calls_per_image']:>4d}"
            f" | elapsed="
            f"{result['elapsed_seconds']:.1f}s"
        )

    if failed_results:
        print("\nFAILED CONFIGURATIONS")

        for result in failed_results:
            print(
                f"lambda="
                f"{result['physical_step']:g}"
                f" | "
                f"{result.get('error_type', '')}: "
                f"{result.get('error_message', '')}"
            )

    print("=" * 150)

    final_summary = save_summary(
        sweep_results
    )

    print("\nFinal summary:")
    print(final_summary)

    if successful_results:
        best = successful_results[0]

        print(
            "\nBest FENEB configuration "
            "by final I_perp:"
        )

        print(
            f"  physical step = "
            f"{best['physical_step']:g}"
        )

        print(
            f"  final I_perp  = "
            f"{best['final_I_perp']:.6e}"
        )

        print(
            f"  final fmax    = "
            f"{best['final_fmax']:.6e}"
        )

        print(
            f"  result file   = "
            f"{best['result_file']}"
        )


if __name__ == "__main__":
    main()