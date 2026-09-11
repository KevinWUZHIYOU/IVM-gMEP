#!/usr/bin/env python3
"""Independent 1000-call candidates for every non-IVM method."""

from __future__ import annotations

from pathlib import Path

import run_selected_method_productions as production


production_command_for = production.command_for
production_output_glob = production.output_glob


def command_for(
    python: Path,
    root: Path,
    namespace: str,
    name: str,
    config: dict,
    budget: int,
) -> list[str]:
    if config["method"] != "ode_neb":
        return production_command_for(
            python, root, namespace, name, config, budget
        )
    return [
        str(python),
        str(root / "src" / "run_neb.py"),
        "--root",
        str(root),
        "--initial-path-file",
        "inputs/common_initial_path_equidistant.npz",
        "--run-label",
        f"paper_production_{name}",
        "--spring-k",
        str(config["spring_k"]),
        "--alpha",
        "0.0001",
        "--optimizer",
        "ode",
        "--budget",
        str(budget),
        "--target-fmax",
        "0",
        "--save-all-paths",
    ]


def output_glob(
    root: Path, namespace: str, name: str, method: str
) -> str:
    if method == "ode_neb":
        return f"runs/neb/neb_paper_production_{name}_central_*.npz"
    return production_output_glob(root, namespace, name, method)


def obsm_config(lambda_mix: float, inner_steps: int, fire_dt: float) -> dict:
    return {
        "method": "obsm_fire",
        "lambda_mix": float(lambda_mix),
        "inner_steps": int(inner_steps),
        "fire_dt": float(fire_dt),
        "fire_beta": 1.0,
        "fire_mass": 1.0,
        "reparam_interval": 1,
        "selection_role": "focused 1000-call OBSM candidate",
    }


productions = {
    "extended1000_original_dt0p012525": {
        "method": "original_string",
        "dt": 0.012525,
        "reparam_interval": 1,
        "selection_role": "unified 300/500-call candidate",
    },
    "extended1000_original_dt0p01255": {
        "method": "original_string",
        "dt": 0.01255,
        "reparam_interval": 1,
        "selection_role": "unified 300/500-call candidate",
    },
    "extended1000_original_dt0p012575": {
        "method": "original_string",
        "dt": 0.012575,
        "reparam_interval": 1,
        "selection_role": "unified 300/500-call candidate",
    },
    "extended1000_static_k40_a0p012525": {
        "method": "static_neb",
        "spring_k": 40.0,
        "alpha": 0.012525,
        "selection_role": "unified 300/500-call candidate",
    },
    "extended1000_static_k40_a0p01255": {
        "method": "static_neb",
        "spring_k": 40.0,
        "alpha": 0.01255,
        "selection_role": "unified 300/500-call candidate",
    },
    "extended1000_static_k40_a0p012575": {
        "method": "static_neb",
        "spring_k": 40.0,
        "alpha": 0.012575,
        "selection_role": "unified 300/500-call candidate",
    },
}

for spring_k in (0.1, 1.0, 5.0, 10.0, 20.0, 35.0, 50.0, 80.0):
    tag = str(spring_k).replace(".", "p")
    productions[f"extended1000_ode_k{tag}"] = {
        "method": "ode_neb",
        "spring_k": float(spring_k),
        "alpha_interface_only": 0.0001,
        "selection_role": "focused 1000-call ODE-NEB k candidate",
    }

for lambda_mix in (0.9, 0.95):
    lambda_tag = str(lambda_mix).replace(".", "p")
    for inner_steps, fire_dts in (
        (5, (0.115, 0.12, 0.125)),
        (7, (0.12,)),
        (10, (0.115, 0.12, 0.125)),
        (15, (0.115, 0.12)),
        (20, (0.12,)),
    ):
        for fire_dt in fire_dts:
            dt_tag = str(fire_dt).replace(".", "p")
            name = (
                f"extended1000_obsm_l{lambda_tag}_n{inner_steps}_"
                f"dt{dt_tag}"
            )
            productions[name] = obsm_config(
                lambda_mix, inner_steps, fire_dt
            )

production.command_for = command_for
production.output_glob = output_glob
production.PRODUCTIONS = productions


if __name__ == "__main__":
    production.main()
