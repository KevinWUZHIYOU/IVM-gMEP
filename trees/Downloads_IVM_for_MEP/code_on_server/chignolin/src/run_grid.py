#!/usr/bin/env python3
"""Parallel, restart-safe orchestration for the Chignolin tuning grids."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PYTHON = Path(sys.executable).resolve()


def ivm_early_commands(root: Path):
    alphas = (0.03, 0.3, 1.0, 3.0, 10.0, 80.0)
    max_steps = (0.1, 0.3, 1.0, 2.0)
    solvers = ("lbfgs", "lbfgsb")
    inner_calls = (3, 9)
    script = root / "src" / "run_ivm.py"
    for solver, alpha, max_step, inner in itertools.product(
        solvers, alphas, max_steps, inner_calls
    ):
        name = (
            f"ivm50_{solver}_a{alpha:g}_s{max_step:g}_inner{inner}"
        ).replace(".", "p")
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--solver",
            solver,
            "--alpha",
            str(alpha),
            "--max-step",
            str(max_step),
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--inner-calls",
            str(inner),
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--budget",
            "50",
        ]
        yield name, command


def neb_500_commands(root: Path):
    spring_constants = (0.1, 1, 5, 10, 20, 50, 80, 100, 500, 1000, 2000)
    fixed_alpha = 1e-4
    script = root / "src" / "run_neb.py"
    for spring_k in spring_constants:
        name = (
            f"neb500_k{spring_k:g}_ode"
        ).replace(".", "p")
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--spring-k",
            str(spring_k),
            "--alpha",
            str(fixed_alpha),
            "--optimizer",
            "ode",
            "--budget",
            "500",
        ]
        yield name, command


def ivm_mid_commands(root: Path):
    """Twelve independently restarted early/middle schedules to 200 calls."""
    script = root / "src" / "run_ivm.py"
    schedules = [
        ("a1_s1_i9", "lbfgsb", 1, "", 1, "", 9, "", 0),
        ("a3_s1_i9", "lbfgsb", 3, "", 1, "", 9, "", 0),
        ("a1_s0p3_i9", "lbfgsb", 1, "", 0.3, "", 9, "", 0),
        ("a3_s0p3_i9", "lbfgsb", 3, "", 0.3, "", 9, "", 0),
        ("a0p3_s1_i9", "lbfgsb", 0.3, "", 1, "", 9, "", 0),
        ("a3_s2_i9", "lbfgsb", 3, "", 2, "", 9, "", 0),
        ("a1to0p3_s1to0p3", "lbfgsb", 1, "50:0.3", 1, "50:0.3", 9, "", 0),
        ("a1to3_s1", "lbfgsb", 1, "50:3", 1, "", 9, "", 0),
        ("a3to1_s1", "lbfgsb", 3, "50:1", 1, "", 9, "", 0),
        ("a1_s2to1to0p3", "lbfgsb", 1, "", 2, "20:1,100:0.3", 9, "", 0),
        ("a1_s1_i9to5", "lbfgsb", 1, "", 1, "", 9, "50:5", 0),
        (
            "hybrid_warm5",
            "hybrid",
            0.03,
            "5:1",
            0.5,
            "2:0.3,5:1",
            1,
            "5:9",
            5,
        ),
    ]
    for (
        name,
        solver,
        alpha,
        alpha_schedule,
        max_step,
        max_step_schedule,
        inner,
        inner_schedule,
        switch,
    ) in schedules:
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--solver",
            solver,
            "--hybrid-switch-call",
            str(switch),
            "--alpha",
            str(alpha),
            "--alpha-schedule",
            alpha_schedule,
            "--max-step",
            str(max_step),
            "--max-step-schedule",
            max_step_schedule,
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--inner-calls",
            str(inner),
            "--inner-calls-schedule",
            inner_schedule,
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--budget",
            "200",
            "--save-all-paths",
        ]
        yield f"ivm200_{name}", command


def ivm_late_commands(root: Path):
    """Protected late-polish trials for the selected alpha=1 early schedule."""
    script = root / "src" / "run_ivm.py"
    schedules = [
        ("control", "", "", "", ""),
        ("protect", "", "", "", "200:0.1,350:0.02"),
        (
            "soften250",
            "250:0.3",
            "250:0.3",
            "",
            "200:0.1,350:0.02",
        ),
        (
            "accelerate_then_polish",
            "250:3,400:0.3",
            "400:0.2",
            "",
            "350:0.1,450:0.0",
        ),
        (
            "refresh_late",
            "",
            "",
            "200:5,350:3",
            "200:0.1,350:0.02",
        ),
    ]
    for name, alpha_schedule, step_schedule, inner_schedule, rtol_schedule in schedules:
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--solver",
            "lbfgsb",
            "--alpha",
            "1",
            "--alpha-schedule",
            alpha_schedule,
            "--max-step",
            "1",
            "--max-step-schedule",
            step_schedule,
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--inner-calls",
            "9",
            "--inner-calls-schedule",
            inner_schedule,
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--iperp-rtol-schedule",
            rtol_schedule,
            "--budget",
            "500",
        ]
        yield f"ivm500_late_{name}", command


def ivm_reparam_screen_commands(root: Path):
    """Screen audited optional reparameterization frequency and tolerances."""
    script = root / "src" / "run_ivm.py"
    frequencies = (1, 2, 3, 5, 10, 20)
    acceptance_settings = (
        ("strict", 1.0, 1.0, 0.2),
        ("balanced", 5.0, 5.0, 0.5),
        ("permissive", 20.0, 10.0, 1.0),
    )
    for frequency, (
        acceptance_name,
        objective_tolerance,
        energy_tolerance,
        force_rtol,
    ) in itertools.product(frequencies, acceptance_settings):
        name = (
            f"ivm200_reparam{frequency}_{acceptance_name}"
        )
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "reparam_screen",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            "5",
            "--alpha",
            "0.03",
            "--alpha-schedule",
            "5:1",
            "--max-step",
            "0.5",
            "--max-step-schedule",
            "2:0.3,5:1",
            "--history",
            "7",
            "--reparam-every",
            str(frequency),
            "--reparam-objective-tolerance-kcal",
            str(objective_tolerance),
            "--reparam-energy-tolerance-kcal",
            str(energy_tolerance),
            "--reparam-force-rtol",
            str(force_rtol),
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            "5:9",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--budget",
            "200",
            "--save-all-paths",
        ]
        yield name, command


def ivm_reparam_late_commands(root: Path):
    """Extend diverse audited reparameterization schedules to 500 calls."""
    script = root / "src" / "run_ivm.py"
    schedules = (
        ("rep10_all_bal", 10, "", 5.0, 5.0, 0.5),
        ("rep10_off250_bal", 10, "250:0", 5.0, 5.0, 0.5),
        ("rep10_off350_bal", 10, "350:0", 5.0, 5.0, 0.5),
        ("rep10_to20_200_bal", 10, "200:20", 5.0, 5.0, 0.5),
        ("rep10_to5_250_bal", 10, "250:5", 5.0, 5.0, 0.5),
        ("late200_rep10_bal", 0, "200:10", 5.0, 5.0, 0.5),
        ("rep20_all_bal", 20, "", 5.0, 5.0, 0.5),
        ("rep3_to10_100_off350_bal", 3, "100:10,350:0", 5.0, 5.0, 0.5),
        ("rep10_all_strict", 10, "", 1.0, 1.0, 0.2),
        ("rep10_all_perm", 10, "", 20.0, 10.0, 1.0),
    )
    for (
        name,
        frequency,
        frequency_schedule,
        objective_tolerance,
        energy_tolerance,
        force_rtol,
    ) in schedules:
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "reparam_late",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            "5",
            "--alpha",
            "0.03",
            "--alpha-schedule",
            "5:1,250:3,400:0.3",
            "--max-step",
            "0.5",
            "--max-step-schedule",
            "2:0.3,5:1,400:0.2",
            "--history",
            "7",
            "--reparam-every",
            str(frequency),
            "--reparam-every-schedule",
            frequency_schedule,
            "--reparam-objective-tolerance-kcal",
            str(objective_tolerance),
            "--reparam-energy-tolerance-kcal",
            str(energy_tolerance),
            "--reparam-force-rtol",
            str(force_rtol),
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            "5:9",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--iperp-rtol-schedule",
            "350:0.1,450:0.0",
            "--budget",
            "500",
            "--save-all-paths",
        ]
        yield f"ivm500_{name}", command


def ivm_aggressive_late_commands(root: Path):
    """Search aggressive-to-polish L-BFGS-B schedules through 500 calls."""
    script = root / "src" / "run_ivm.py"
    current_alpha = "5:1,250:3,400:0.3"
    alpha_a = "5:1,150:3,300:10,400:1,450:0.3"
    alpha_b = "5:1,100:3,250:10,400:3,450:0.3"
    alpha_c = "5:1,100:10,300:3,400:1,450:0.3"
    alpha_d = "5:1,200:10,350:3,450:0.3"
    alpha_e = "5:3,150:10,350:1,450:0.1"
    alpha_f = "5:0.3,100:1,200:3,350:1,450:0.1"
    current_step = "2:0.3,5:1,400:0.2"
    current_rtol = "350:0.1,450:0.0"
    rtol_half = "5:0.5,50:0.2,250:0.1,400:0.0"
    rtol_fifth = "5:0.2,100:0.1,350:0.0"
    schedules = (
        ("control", current_alpha, current_step, 9, current_rtol, 0),
        ("alpha_a", alpha_a, current_step, 9, current_rtol, 0),
        ("alpha_b", alpha_b, current_step, 9, current_rtol, 0),
        ("alpha_c", alpha_c, current_step, 9, current_rtol, 0),
        ("alpha_d", alpha_d, current_step, 9, current_rtol, 0),
        ("alpha_e", alpha_e, current_step, 9, current_rtol, 0),
        ("alpha_f", alpha_f, current_step, 9, current_rtol, 0),
        ("b_rtol_half", alpha_b, current_step, 9, rtol_half, 0),
        ("b_rtol_fifth", alpha_b, current_step, 9, rtol_fifth, 0),
        ("b_rtol_zero", alpha_b, current_step, 9, "5:0.0", 0),
        ("b_inner5", alpha_b, current_step, 5, rtol_half, 0),
        ("b_inner15", alpha_b, current_step, 15, rtol_half, 0),
        ("b_inner25", alpha_b, current_step, 25, rtol_half, 0),
        (
            "b_step_soft",
            alpha_b,
            "2:0.3,5:1,350:0.5,450:0.1",
            9,
            rtol_half,
            0,
        ),
        (
            "b_step_wide",
            alpha_b,
            "2:0.3,5:2,300:1,400:0.3",
            9,
            rtol_half,
            0,
        ),
        ("b_rtol_half_rep10", alpha_b, current_step, 9, rtol_half, 10),
    )
    for name, alpha_schedule, step_schedule, inner_calls, rtol, reparam in schedules:
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "aggressive_late",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            "5",
            "--alpha",
            "0.03",
            "--alpha-schedule",
            alpha_schedule,
            "--max-step",
            "0.5",
            "--max-step-schedule",
            step_schedule,
            "--history",
            "7",
            "--reparam-every",
            str(reparam),
            "--reparam-objective-tolerance-kcal",
            "1.0",
            "--reparam-energy-tolerance-kcal",
            "1.0",
            "--reparam-force-rtol",
            "0.2",
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            f"5:{inner_calls}",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--iperp-rtol-schedule",
            rtol,
            "--budget",
            "500",
            "--save-all-paths",
        ]
        yield f"ivm500_aggressive_{name}", command


def ivm_warmup_screen_commands(root: Path):
    """Screen early hybrid handoff and acceptance with the inner-25 solver."""
    script = root / "src" / "run_ivm.py"
    schedules = [
        ("s5_r05", 5, "5:1,100:3,250:10,400:3,450:0.3", 0.5),
        ("s5_r04", 5, "5:1,100:3,250:10,400:3,450:0.3", 0.4),
        ("s5_r03", 5, "5:1,100:3,250:10,400:3,450:0.3", 0.3),
        ("s5_r02", 5, "5:1,100:3,250:10,400:3,450:0.3", 0.2),
        ("s5_r01", 5, "5:1,100:3,250:10,400:3,450:0.3", 0.1),
        ("s5_r00", 5, "5:1,100:3,250:10,400:3,450:0.3", 0.0),
        ("s8_hold", 8, "8:1,100:3,250:10,400:3,450:0.3", 0.5),
        (
            "s8_a001",
            8,
            "5:0.01,8:1,100:3,250:10,400:3,450:0.3",
            0.5,
        ),
        ("s12_hold", 12, "12:1,100:3,250:10,400:3,450:0.3", 0.5),
        (
            "s12_a001_5",
            12,
            "5:0.01,12:1,100:3,250:10,400:3,450:0.3",
            0.5,
        ),
        (
            "s12_a001_8",
            12,
            "8:0.01,12:1,100:3,250:10,400:3,450:0.3",
            0.5,
        ),
        (
            "s12_a0015_8",
            12,
            "8:0.015,12:1,100:3,250:10,400:3,450:0.3",
            0.5,
        ),
        ("s16_hold", 16, "16:1,100:3,250:10,400:3,450:0.3", 0.5),
        (
            "s16_a001",
            16,
            "8:0.01,16:1,100:3,250:10,400:3,450:0.3",
            0.5,
        ),
        (
            "s20_a001",
            20,
            "8:0.01,20:1,100:3,250:10,400:3,450:0.3",
            0.5,
        ),
    ]
    for name, switch, alpha_schedule, handoff_rtol in schedules:
        rtol_schedule = (
            f"{switch}:{handoff_rtol:g},50:0.2,250:0.1,400:0.0"
        )
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "warmup_screen",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            str(switch),
            "--alpha",
            "0.03",
            "--alpha-schedule",
            alpha_schedule,
            "--max-step",
            "0.5",
            "--max-step-schedule",
            f"2:0.3,{switch}:1,400:0.2",
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            f"{switch}:25",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--iperp-rtol-schedule",
            rtol_schedule,
            "--budget",
            "100",
            "--save-all-paths",
        ]
        yield f"ivm100_warmup_{name}", command


def ivm_global_guard_screen_commands(root: Path):
    """Screen path-level I_perp acceptance after per-image IVP checks."""
    script = root / "src" / "run_ivm.py"
    settings = []
    for global_rtol in (0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        settings.append((f"per05_g{global_rtol:g}", 0.5, global_rtol))
    for global_rtol in (0.0, 0.01, 0.05):
        settings.append((f"per1_g{global_rtol:g}", 1.0, global_rtol))
    for global_rtol in (0.0, 0.01):
        settings.append((f"per01_g{global_rtol:g}", 0.1, global_rtol))
    for name, per_image_rtol, global_rtol in settings:
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "global_guard_screen",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            "5",
            "--alpha",
            "0.03",
            "--alpha-schedule",
            "5:1,100:3,250:10,400:3,450:0.3",
            "--max-step",
            "0.5",
            "--max-step-schedule",
            "2:0.3,5:1,400:0.2",
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            "5:25",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--iperp-rtol-schedule",
            f"5:{per_image_rtol:g},50:0.2,250:0.1,400:0.0",
            "--global-iperp-rtol",
            str(global_rtol),
            "--budget",
            "100",
            "--save-all-paths",
        ]
        yield f"ivm100_global_{name}", command


def ivm_initial_step_screen_commands(root: Path):
    """Screen the first L-BFGS-B displacement with the inner-25 solver."""
    script = root / "src" / "run_ivm.py"
    for initial_step in (0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5):
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "initial_step_screen",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            "5",
            "--alpha",
            "0.03",
            "--alpha-schedule",
            "5:1,100:3,250:10,400:3,450:0.3",
            "--max-step",
            "0.5",
            "--max-step-schedule",
            "2:0.3,5:1,400:0.2",
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--lbfgsb-initial-step",
            str(initial_step),
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            "5:25",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--iperp-rtol-schedule",
            "5:0.5,50:0.2,250:0.1,400:0.0",
            "--global-iperp-rtol",
            "0.001",
            "--budget",
            "100",
            "--save-all-paths",
        ]
        yield f"ivm100_initial_{initial_step:g}".replace(".", "p"), command


def ivm_inner_transition_commands(root: Path):
    """Combine the faster early inner-15 phase with inner-25 late polish."""
    script = root / "src" / "run_ivm.py"
    schedules = [
        ("t50", 50, 0.0, 1e9, ""),
        ("t75", 75, 0.0, 1e9, ""),
        ("t100", 100, 0.0, 1e9, ""),
        ("t150", 150, 0.0, 1e9, ""),
        ("t200", 200, 0.0, 1e9, ""),
        ("t50_i005", 50, 0.05, 1e9, ""),
        ("t100_i005", 100, 0.05, 1e9, ""),
        ("t50_guard50", 50, 0.0, 0.001, "50:1000000000"),
        ("t100_guard50", 100, 0.0, 0.001, "50:1000000000"),
        ("t100_i005_guard50", 100, 0.05, 0.001, "50:1000000000"),
    ]
    for name, transition, initial_step, global_rtol, global_schedule in schedules:
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "inner_transition",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            "5",
            "--alpha",
            "0.03",
            "--alpha-schedule",
            "5:1,100:3,250:10,400:3,450:0.3",
            "--max-step",
            "0.5",
            "--max-step-schedule",
            "2:0.3,5:1,400:0.2",
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--lbfgsb-initial-step",
            str(initial_step),
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            f"5:15,{transition}:25",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--iperp-rtol-schedule",
            "5:0.5,50:0.2,250:0.1,400:0.0",
            "--global-iperp-rtol",
            str(global_rtol),
            "--global-iperp-rtol-schedule",
            global_schedule,
            "--budget",
            "500",
            "--save-all-paths",
        ]
        yield f"ivm500_transition_{name}", command


def ivm_initial_step_schedule_commands(root: Path):
    """Use a nonzero L-BFGS-B initial displacement only during warmup."""
    script = root / "src" / "run_ivm.py"
    for initial_step, off_call in itertools.product(
        (0.03, 0.05, 0.1),
        (20, 30, 40, 50),
    ):
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "initial_step_schedule",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            "5",
            "--alpha",
            "0.03",
            "--alpha-schedule",
            "5:1,100:3,250:10,400:3,450:0.3",
            "--max-step",
            "0.5",
            "--max-step-schedule",
            "2:0.3,5:1,400:0.2",
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--lbfgsb-initial-step",
            str(initial_step),
            "--lbfgsb-initial-step-schedule",
            f"{off_call}:0",
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            "5:15,50:25",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--iperp-rtol-schedule",
            "5:0.5,50:0.2,250:0.1,400:0.0",
            "--global-iperp-rtol",
            "0.001",
            "--global-iperp-rtol-schedule",
            "50:1000000000",
            "--budget",
            "500",
            "--save-all-paths",
        ]
        name = f"ivm500_initial_{initial_step:g}_off{off_call}".replace(
            ".", "p"
        )
        yield name, command


def ivm_handoff_alignment_commands(root: Path):
    """Align early frozen-tangent subproblems with the NEB-envelope gaps."""
    script = root / "src" / "run_ivm.py"
    for switch, initial_step in itertools.product(
        (5, 6, 7, 8),
        (0.03, 0.05),
    ):
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "handoff_alignment",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            str(switch),
            "--alpha",
            "0.03",
            "--alpha-schedule",
            f"{switch}:1,100:3,250:10,400:3,450:0.3",
            "--max-step",
            "0.5",
            "--max-step-schedule",
            f"2:0.3,{switch}:1,400:0.2",
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--lbfgsb-initial-step",
            str(initial_step),
            "--lbfgsb-initial-step-schedule",
            "40:0",
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            f"{switch}:15,50:25",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--iperp-rtol-schedule",
            f"{switch}:0.5,50:0.2,250:0.1,400:0.0",
            "--global-iperp-rtol",
            "0.001",
            "--global-iperp-rtol-schedule",
            "50:1000000000",
            "--budget",
            "500",
            "--save-all-paths",
        ]
        name = f"ivm500_handoff_s{switch}_i{initial_step:g}".replace(
            ".", "p"
        )
        yield name, command


def ivm_two_stage_initial_screen_commands(root: Path):
    """Tune the first two frozen-tangent subproblem initial displacements."""
    script = root / "src" / "run_ivm.py"
    first_steps = (0.015, 0.02, 0.025, 0.03, 0.035, 0.04)
    second_steps = (0.0, 0.015, 0.03, 0.05, 0.1)
    for first_step, second_step in itertools.product(
        first_steps, second_steps
    ):
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "two_stage_initial_screen",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            "5",
            "--alpha",
            "0.03",
            "--alpha-schedule",
            "5:1,100:3,250:10,400:3,450:0.3",
            "--max-step",
            "0.5",
            "--max-step-schedule",
            "2:0.3,5:1,400:0.2",
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--lbfgsb-initial-step",
            str(first_step),
            "--lbfgsb-initial-step-schedule",
            f"20:{second_step:g},40:0",
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            "5:15,50:25",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--iperp-rtol-schedule",
            "5:0.5,50:0.2,250:0.1,400:0.0",
            "--global-iperp-rtol",
            "0.001",
            "--global-iperp-rtol-schedule",
            "50:1000000000",
            "--budget",
            "100",
            "--save-all-paths",
        ]
        name = (
            f"ivm100_two_initial_{first_step:g}_{second_step:g}"
        ).replace(".", "p")
        yield name, command


def ivm_two_stage_initial_refine_commands(root: Path):
    """Refine the best two-stage initial displacements around 0.015-0.02 A."""
    script = root / "src" / "run_ivm.py"
    first_steps = (0.015, 0.0175, 0.02)
    second_steps = (0.01, 0.0125, 0.015, 0.0175, 0.02)
    for first_step, second_step in itertools.product(
        first_steps, second_steps
    ):
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "two_stage_initial_refine",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            "5",
            "--alpha",
            "0.03",
            "--alpha-schedule",
            "5:1,100:3,250:10,400:3,450:0.3",
            "--max-step",
            "0.5",
            "--max-step-schedule",
            "2:0.3,5:1,400:0.2",
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--lbfgsb-initial-step",
            str(first_step),
            "--lbfgsb-initial-step-schedule",
            f"20:{second_step:g},40:0",
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            "5:15,50:25",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--iperp-rtol-schedule",
            "5:0.5,50:0.2,250:0.1,400:0.0",
            "--global-iperp-rtol",
            "0.001",
            "--global-iperp-rtol-schedule",
            "50:1000000000",
            "--budget",
            "100",
            "--save-all-paths",
        ]
        name = (
            f"ivm100_two_refine_{first_step:g}_{second_step:g}"
        ).replace(".", "p")
        yield name, command


def ivm_two_stage_alpha_screen_commands(root: Path):
    """Tune prox alpha independently for the first two subproblems."""
    script = root / "src" / "run_ivm.py"
    first_alphas = (0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0)
    second_alphas = (0.3, 0.5, 1.0, 2.0, 3.0)
    for first_alpha, second_alpha in itertools.product(
        first_alphas, second_alphas
    ):
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "two_stage_alpha_screen",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            "5",
            "--alpha",
            "0.03",
            "--alpha-schedule",
            (
                f"5:{first_alpha:g},20:{second_alpha:g},"
                "100:3,250:10,400:3,450:0.3"
            ),
            "--max-step",
            "0.5",
            "--max-step-schedule",
            "2:0.3,5:1,400:0.2",
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--lbfgsb-initial-step",
            "0.015",
            "--lbfgsb-initial-step-schedule",
            "20:0.01,40:0",
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            "5:15,50:25",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1.0",
            "--iperp-rtol-schedule",
            "5:0.5,50:0.2,250:0.1,400:0.0",
            "--global-iperp-rtol",
            "0.001",
            "--global-iperp-rtol-schedule",
            "50:1000000000",
            "--budget",
            "100",
            "--save-all-paths",
        ]
        name = (
            f"ivm100_two_alpha_{first_alpha:g}_{second_alpha:g}"
        ).replace(".", "p")
        yield name, command


def ivm_precision_500_commands(root: Path):
    """Retune calls 400–500 from the strongest audited early schedule."""
    script = root / "src" / "run_ivm.py"
    schedules = [
        # name, alpha schedule, step schedule, per-image tolerance,
        # global tolerance, restart calls, reparameterization schedule,
        # reparameterization initial scale, initial-step schedule
        (
            "baseline",
            "5:1,100:3,250:10,400:3,450:0.3",
            "2:0.3,5:1,400:0.2",
            "5:0.5,50:0.2,250:0.1,400:0",
            "50:1000000000",
            "",
            "",
            1.0,
            "30:0",
        ),
        (
            "keep_a10",
            "5:1,100:3,250:10",
            "2:0.3,5:1,400:0.2",
            "5:0.5,50:0.2,250:0.1,400:0",
            "50:1000000000",
            "",
            "",
            1.0,
            "30:0",
        ),
        (
            "late_a3",
            "5:1,100:3,250:10,400:3",
            "2:0.3,5:1,400:0.2",
            "5:0.5,50:0.2,250:0.1,400:0",
            "50:1000000000",
            "",
            "",
            1.0,
            "30:0",
        ),
        (
            "late_a1",
            "5:1,100:3,250:10,400:1",
            "2:0.3,5:1,400:0.2",
            "5:0.5,50:0.2,250:0.1,400:0",
            "50:1000000000",
            "",
            "",
            1.0,
            "30:0",
        ),
        (
            "cooperative_a3",
            "5:1,100:3,250:10,400:3",
            "2:0.3,5:1,400:0.2",
            "5:0.5,50:0.2,250:0.1,400:1",
            "50:1000000000,400:0",
            "400",
            "",
            1.0,
            "30:0",
        ),
        (
            "cooperative_a10",
            "5:1,100:3,250:10",
            "2:0.3,5:1,400:0.1",
            "5:0.5,50:0.2,250:0.1,400:1",
            "50:1000000000,400:0",
            "400",
            "",
            1.0,
            "30:0",
        ),
        (
            "late_reparam",
            "5:1,100:3,250:10,400:3,450:0.3",
            "2:0.3,5:1,400:0.2",
            "5:0.5,50:0.2,250:0.1,400:0.2",
            "50:1000000000,400:0",
            "400",
            "400:5",
            0.1,
            "30:0",
        ),
        (
            "late_kick",
            "5:1,100:3,250:10,400:3,450:0.3",
            "2:0.3,5:1,400:0.2",
            "5:0.5,50:0.2,250:0.1,400:0.1",
            "50:1000000000,400:0",
            "400,450",
            "",
            1.0,
            "30:0,400:0.01,450:0",
        ),
    ]
    for (
        name,
        alpha_schedule,
        step_schedule,
        iperp_schedule,
        global_schedule,
        restart_calls,
        reparam_schedule,
        reparam_scale,
        initial_step_schedule,
    ) in schedules:
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "precision500_screen",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            "5",
            "--alpha",
            "0.03",
            "--alpha-schedule",
            alpha_schedule,
            "--max-step",
            "0.5",
            "--max-step-schedule",
            step_schedule,
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--reparam-every-schedule",
            reparam_schedule,
            "--reparam-initial-step-scale",
            str(reparam_scale),
            "--restart-at-calls",
            restart_calls,
            "--lbfgsb-initial-step",
            "0.1",
            "--lbfgsb-initial-step-schedule",
            initial_step_schedule,
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            "5:15,50:25",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1",
            "--iperp-rtol-schedule",
            iperp_schedule,
            "--global-iperp-rtol",
            "0.001",
            "--global-iperp-rtol-schedule",
            global_schedule,
            "--budget",
            "500",
            "--save-all-paths",
        ]
        yield f"ivm500_precision_{name}", command


def ivm_theory_trust_screen_commands(root: Path):
    """Pure frozen-tangent IVP subminimization with persistent L-BFGS."""
    script = root / "src" / "run_ivm.py"
    for alpha, max_step in itertools.product(
        (3.0, 10.0, 30.0, 80.0), (0.05, 0.1, 0.3)
    ):
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "theory_trust_screen",
            "--solver",
            "lbfgs",
            "--alpha",
            str(alpha),
            "--max-step",
            str(max_step),
            "--history",
            "7",
            "--lbfgs-initial-curvature",
            "70",
            "--reparam-every",
            "0",
            "--inner-calls",
            "5",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1000000000",
            "--global-iperp-rtol",
            "1000000000",
            "--budget",
            "500",
            "--save-all-paths",
        ]
        name = (
            f"ivm500_theory_trust_a{alpha:g}_s{max_step:g}"
        ).replace(".", "p")
        yield name, command


def ivm_theory_lbfgsb_late_screen_commands(root: Path):
    """Strong early schedule followed by objective-only IVP subproblems."""
    script = root / "src" / "run_ivm.py"
    for late_alpha, late_step in itertools.product(
        (1.0, 3.0, 5.0, 10.0, 20.0), (0.05, 0.1, 0.2)
    ):
        command = [
            str(PYTHON),
            str(script),
            "--root",
            str(root),
            "--run-label",
            "theory_lbfgsb_late_screen",
            "--solver",
            "hybrid",
            "--hybrid-switch-call",
            "5",
            "--alpha",
            "0.03",
            "--alpha-schedule",
            f"5:1,100:3,250:10,400:3,450:0.3,500:{late_alpha:g}",
            "--max-step",
            "0.5",
            "--max-step-schedule",
            f"2:0.3,5:1,400:0.2,500:{late_step:g}",
            "--history",
            "7",
            "--reparam-every",
            "0",
            "--restart-at-calls",
            "500",
            "--lbfgsb-initial-step",
            "0.05",
            "--lbfgsb-initial-step-schedule",
            "40:0",
            "--inner-calls",
            "1",
            "--inner-calls-schedule",
            "5:15,50:25",
            "--energy-rtol",
            "1e-10",
            "--iperp-rtol",
            "1",
            "--iperp-rtol-schedule",
            "5:0.5,50:0.2,250:0.1,400:0,500:1000000000",
            "--global-iperp-rtol",
            "0.001",
            "--global-iperp-rtol-schedule",
            "50:1000000000",
            "--budget",
            "800",
            "--save-all-paths",
        ]
        name = (
            f"ivm800_theory_lbfgsb_a{late_alpha:g}_s{late_step:g}"
        ).replace(".", "p")
        yield name, command


def ivm_refined_theory_candidates(root: Path):
    """Best early warm start plus stable objective-only late IVP solves."""
    script = root / "src" / "run_ivm.py"
    for transition in (400, 450, 500):
        command = [
            str(PYTHON), str(script),
            "--root", str(root),
            "--run-label", "refined_theory_candidate",
            "--solver", "hybrid",
            "--hybrid-switch-call", "5",
            "--alpha", "0.03",
            "--alpha-schedule",
            f"5:1,100:3,250:10,400:3,450:0.3,{transition}:3",
            "--max-step", "0.5",
            "--max-step-schedule",
            f"2:0.3,5:1,400:0.2,{transition}:0.05",
            "--history", "7",
            "--reparam-every", "0",
            "--restart-at-calls", str(transition),
            "--lbfgsb-initial-step", "0.015",
            "--lbfgsb-initial-step-schedule", "20:0.01,40:0",
            "--inner-calls", "1",
            "--inner-calls-schedule", "5:15,50:25",
            "--energy-rtol", "1e-10",
            "--iperp-rtol", "1",
            "--iperp-rtol-schedule",
            f"5:0.5,50:0.2,250:0.1,400:0,{transition}:1000000000",
            "--global-iperp-rtol", "0.001",
            "--global-iperp-rtol-schedule", "50:1000000000",
            "--budget", "800",
            "--save-all-paths",
        ]
        yield f"ivm800_refined_transition_{transition}", command


def ivm_theory_inner_screen_commands(root: Path):
    """Tune tangent refresh frequency in exact frozen-tangent IVP solves."""
    script = root / "src" / "run_ivm.py"
    for late_alpha, late_step, late_inner_calls in itertools.product(
        (1.0, 2.0, 3.0), (0.02, 0.05), (2, 5, 10)
    ):
        command = [
            str(PYTHON), str(script),
            "--root", str(root),
            "--run-label", "theory_inner_screen",
            "--solver", "hybrid",
            "--hybrid-switch-call", "5",
            "--alpha", "0.03",
            "--alpha-schedule",
            f"5:1,100:3,250:10,400:3,450:0.3,500:{late_alpha:g}",
            "--max-step", "0.5",
            "--max-step-schedule",
            f"2:0.3,5:1,400:0.2,500:{late_step:g}",
            "--history", "7",
            "--reparam-every", "0",
            "--restart-at-calls", "500",
            "--lbfgsb-initial-step", "0.05",
            "--lbfgsb-initial-step-schedule", "40:0",
            "--inner-calls", "1",
            "--inner-calls-schedule",
            f"5:15,50:25,500:{late_inner_calls}",
            "--energy-rtol", "1e-10",
            "--iperp-rtol", "1",
            "--iperp-rtol-schedule",
            "5:0.5,50:0.2,250:0.1,400:0,500:1000000000",
            "--global-iperp-rtol", "0.001",
            "--global-iperp-rtol-schedule", "50:1000000000",
            "--budget", "1000",
        ]
        name = (
            "ivm1000_theory_inner_"
            f"a{late_alpha:g}_s{late_step:g}_n{late_inner_calls}"
        ).replace(".", "p")
        yield name, command


def ivm_early_trust_screen_commands(root: Path):
    """Screen the first frozen-tangent IVP trust region, then restore baseline."""
    script = root / "src" / "run_ivm.py"
    for initial_step, early_max_step in itertools.product(
        (0.015, 0.025, 0.05, 0.075), (0.2, 0.4, 0.6, 0.8, 1.0)
    ):
        command = [
            str(PYTHON), str(script),
            "--root", str(root),
            "--run-label", "early_trust_screen",
            "--solver", "hybrid",
            "--hybrid-switch-call", "5",
            "--alpha", "0.03",
            "--alpha-schedule",
            "5:1,100:3,250:10,400:3,450:0.3",
            "--max-step", "0.5",
            "--max-step-schedule",
            f"2:0.3,5:{early_max_step:g},20:1,400:0.2",
            "--history", "7",
            "--reparam-every", "0",
            "--lbfgsb-initial-step", f"{initial_step:g}",
            "--lbfgsb-initial-step-schedule", "40:0",
            "--inner-calls", "1",
            "--inner-calls-schedule", "5:15,50:25",
            "--energy-rtol", "1e-10",
            "--iperp-rtol", "1",
            "--iperp-rtol-schedule",
            "5:0.5,50:0.2,250:0.1,400:0",
            "--global-iperp-rtol", "0.001",
            "--global-iperp-rtol-schedule", "50:1000000000",
            "--budget", "100",
        ]
        name = (
            "ivm100_early_trust_"
            f"i{initial_step:g}_s{early_max_step:g}"
        ).replace(".", "p")
        yield name, command


def ivm_early_trust_refine_commands(root: Path):
    """Refine the Pareto-optimal early IVP initial guess and trust radius."""
    script = root / "src" / "run_ivm.py"
    for initial_step, early_max_step in itertools.product(
        (0.016, 0.018, 0.020, 0.022, 0.024), (0.5, 0.6, 0.7)
    ):
        command = [
            str(PYTHON), str(script),
            "--root", str(root),
            "--run-label", "early_trust_refine",
            "--solver", "hybrid",
            "--hybrid-switch-call", "5",
            "--alpha", "0.03",
            "--alpha-schedule",
            "5:1,100:3,250:10,400:3,450:0.3",
            "--max-step", "0.5",
            "--max-step-schedule",
            f"2:0.3,5:{early_max_step:g},20:1,400:0.2",
            "--history", "7",
            "--reparam-every", "0",
            "--lbfgsb-initial-step", f"{initial_step:g}",
            "--lbfgsb-initial-step-schedule", "40:0",
            "--inner-calls", "1",
            "--inner-calls-schedule", "5:15,50:25",
            "--energy-rtol", "1e-10",
            "--iperp-rtol", "1",
            "--iperp-rtol-schedule",
            "5:0.5,50:0.2,250:0.1,400:0",
            "--global-iperp-rtol", "0.001",
            "--global-iperp-rtol-schedule", "50:1000000000",
            "--budget", "100",
        ]
        name = (
            "ivm100_early_trust_refine_"
            f"i{initial_step:g}_s{early_max_step:g}"
        ).replace(".", "p")
        yield name, command


def ivm_early_trust_two_stage_commands(root: Path):
    """Decouple calls 5-20 and calls 20-40 IVP trust radii."""
    script = root / "src" / "run_ivm.py"
    for initial_step, early_max_step, second_max_step in itertools.product(
        (0.0215, 0.0220, 0.0225), (0.68, 0.70, 0.72), (0.6, 0.8, 1.0)
    ):
        command = [
            str(PYTHON), str(script),
            "--root", str(root),
            "--run-label", "early_trust_two_stage",
            "--solver", "hybrid",
            "--hybrid-switch-call", "5",
            "--alpha", "0.03",
            "--alpha-schedule",
            "5:1,100:3,250:10,400:3,450:0.3",
            "--max-step", "0.5",
            "--max-step-schedule",
            (
                f"2:0.3,5:{early_max_step:g},"
                f"20:{second_max_step:g},40:1,400:0.2"
            ),
            "--history", "7",
            "--reparam-every", "0",
            "--lbfgsb-initial-step", f"{initial_step:g}",
            "--lbfgsb-initial-step-schedule", "40:0",
            "--inner-calls", "1",
            "--inner-calls-schedule", "5:15,50:25",
            "--energy-rtol", "1e-10",
            "--iperp-rtol", "1",
            "--iperp-rtol-schedule",
            "5:0.5,50:0.2,250:0.1,400:0",
            "--global-iperp-rtol", "0.001",
            "--global-iperp-rtol-schedule", "50:1000000000",
            "--budget", "100",
        ]
        name = (
            "ivm100_early_trust_two_stage_"
            f"i{initial_step:g}_s{early_max_step:g}_s2{second_max_step:g}"
        ).replace(".", "p")
        yield name, command


def run_one(item, log_dir: Path, cpu_threads: int):
    name, command = item
    log_path = log_dir / f"{name}.log"
    start = time.time()
    environment = os.environ.copy()
    environment["OPENMM_CPU_THREADS"] = str(cpu_threads)
    environment["OMP_NUM_THREADS"] = str(cpu_threads)
    environment["MKL_NUM_THREADS"] = str(cpu_threads)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(command[1]).parent)
    with log_path.open("w") as handle:
        process = subprocess.run(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
        )
    return {
        "name": name,
        "command": command,
        "returncode": int(process.returncode),
        "wall_seconds": time.time() - start,
        "log": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--grid",
        choices=(
            "ivm-early",
            "ivm-mid",
            "ivm-late",
            "ivm-reparam-screen",
            "ivm-reparam-late",
            "ivm-aggressive-late",
            "ivm-warmup-screen",
            "ivm-global-guard-screen",
            "ivm-initial-step-screen",
            "ivm-inner-transition",
            "ivm-initial-step-schedule",
            "ivm-handoff-alignment",
            "ivm-two-stage-initial-screen",
            "ivm-two-stage-initial-refine",
            "ivm-two-stage-alpha-screen",
            "ivm-precision-500",
            "ivm-theory-trust-screen",
            "ivm-theory-lbfgsb-late-screen",
            "ivm-refined-theory-candidates",
            "ivm-theory-inner-screen",
            "ivm-early-trust-screen",
            "ivm-early-trust-refine",
            "ivm-early-trust-two-stage",
            "neb-500",
        ),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    generators = {
        "ivm-early": ivm_early_commands,
        "ivm-mid": ivm_mid_commands,
        "ivm-late": ivm_late_commands,
        "ivm-reparam-screen": ivm_reparam_screen_commands,
        "ivm-reparam-late": ivm_reparam_late_commands,
        "ivm-aggressive-late": ivm_aggressive_late_commands,
        "ivm-warmup-screen": ivm_warmup_screen_commands,
        "ivm-global-guard-screen": ivm_global_guard_screen_commands,
        "ivm-initial-step-screen": ivm_initial_step_screen_commands,
        "ivm-inner-transition": ivm_inner_transition_commands,
        "ivm-initial-step-schedule": ivm_initial_step_schedule_commands,
        "ivm-handoff-alignment": ivm_handoff_alignment_commands,
        "ivm-two-stage-initial-screen": ivm_two_stage_initial_screen_commands,
        "ivm-two-stage-initial-refine": ivm_two_stage_initial_refine_commands,
        "ivm-two-stage-alpha-screen": ivm_two_stage_alpha_screen_commands,
        "ivm-precision-500": ivm_precision_500_commands,
        "ivm-theory-trust-screen": ivm_theory_trust_screen_commands,
        "ivm-theory-lbfgsb-late-screen": (
            ivm_theory_lbfgsb_late_screen_commands
        ),
        "ivm-refined-theory-candidates": ivm_refined_theory_candidates,
        "ivm-theory-inner-screen": ivm_theory_inner_screen_commands,
        "ivm-early-trust-screen": ivm_early_trust_screen_commands,
        "ivm-early-trust-refine": ivm_early_trust_refine_commands,
        "ivm-early-trust-two-stage": ivm_early_trust_two_stage_commands,
        "neb-500": neb_500_commands,
    }
    items = list(generators[args.grid](root))
    log_dir = root / "logs" / args.grid
    log_dir.mkdir(parents=True, exist_ok=True)
    results = []
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = {
            executor.submit(run_one, item, log_dir, args.cpu_threads): item[0]
            for item in items
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), start=1
        ):
            result = future.result()
            results.append(result)
            state = "ok" if result["returncode"] == 0 else "FAILED"
            print(
                f"[{completed:03d}/{len(items):03d}] {state:6s} "
                f"{result['name']} {result['wall_seconds']:.1f}s",
                flush=True,
            )

    summary = {
        "grid": args.grid,
        "workers": args.workers,
        "cpu_threads_per_worker": args.cpu_threads,
        "wall_seconds": time.time() - start,
        "n_total": len(results),
        "n_success": sum(item["returncode"] == 0 for item in results),
        "n_failed": sum(item["returncode"] != 0 for item in results),
        "results": sorted(results, key=lambda item: item["name"]),
    }
    summary_path = root / "reports" / f"{args.grid}_grid_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(temporary, summary_path)
    print(f"Summary: {summary_path}", flush=True)
    if summary["n_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
