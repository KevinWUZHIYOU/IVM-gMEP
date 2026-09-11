#!/usr/bin/env python3
"""Extend fixed-alpha IVM sensitivity through alpha=1000."""

from __future__ import annotations

from pathlib import Path

import run_ivm_fixed_alpha_sensitivity as sensitivity


sensitivity.ALPHAS = (
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    20.0,
    30.0,
    100.0,
    300.0,
    1000.0,
)

base_command_for = sensitivity.command_for


def command_for(
    python: Path, root: Path, alpha: float, budget: int
) -> list[str]:
    command = base_command_for(python, root, alpha, budget)
    index = command.index("--alpha-max") + 1
    command[index] = str(max(20.0, float(alpha)))
    return command


sensitivity.command_for = command_for


if __name__ == "__main__":
    sensitivity.main()
