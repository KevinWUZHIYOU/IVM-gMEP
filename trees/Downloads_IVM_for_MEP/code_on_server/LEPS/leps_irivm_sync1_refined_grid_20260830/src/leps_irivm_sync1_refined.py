#!/usr/bin/env python3
"""Frozen sync-one LEPS IR-IVM on the preregistered refined grid."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


THIS_FILE = Path(__file__).resolve()
CAMPAIGN_ROOT = THIS_FILE.parents[1]
LEPS_ROOT = THIS_FILE.parents[2]
FROZEN_SOURCE = (
    LEPS_ROOT
    / "leps_irivm_ala_chig_every1_sync1_20260830/src/leps_irivm_sync1.py"
)
spec = importlib.util.spec_from_file_location("_frozen_leps_sync1_for_refined_grid", FROZEN_SOURCE)
if spec is None or spec.loader is None:
    raise ImportError(FROZEN_SOURCE)
frozen = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = frozen
spec.loader.exec_module(frozen)

parent = frozen.parent
base = frozen.base
EXPERIMENT_ID = "leps-irivm-sync1-refined-grid-20260830"
REGISTRATION = CAMPAIGN_ROOT / "registrations/CAMPAIGN_PREREGISTRATION.json"
ARTICLE_NF_BUDGET = 3000

# Clean Cartesian refinement.  These remain the only tuned controls.
ALPHAS = (20.0, 50.0, 100.0, 200.0, 500.0)
R_VALUES = (0.001, 0.003, 0.01, 0.03, 0.1)
M_MAX_VALUES = (3, 4, 5, 6, 8, 10)
GRID = tuple(
    (alpha, r, m_max)
    for alpha in ALPHAS
    for r in R_VALUES
    for m_max in M_MAX_VALUES
)

# Rebind the frozen implementation's contract globals to this preregistered
# campaign without changing any optimizer or recovery operation.
frozen.THIS_FILE = THIS_FILE
frozen.CAMPAIGN_ROOT = CAMPAIGN_ROOT
frozen.EXPERIMENT_ID = EXPERIMENT_ID
frozen.REGISTRATION = REGISTRATION
frozen.ARTICLE_NF_BUDGET = ARTICLE_NF_BUDGET
frozen.ALPHAS = ALPHAS
frozen.R_VALUES = R_VALUES
frozen.M_MAX_VALUES = M_MAX_VALUES
frozen.GRID = GRID
for module in (parent, parent.current, base):
    module.EXPERIMENT_ID = EXPERIMENT_ID
    module.CAMPAIGN_ROOT = CAMPAIGN_ROOT
    module.REGISTRATION = REGISTRATION
    module.ARTICLE_NF_BUDGET = ARTICLE_NF_BUDGET
    module.ALPHAS = ALPHAS
    module.R_VALUES = R_VALUES
    module.M_MAX_VALUES = M_MAX_VALUES
    module.GRID = GRID
base.__file__ = str(THIS_FILE)

Controls = frozen.Controls
ContractError = frozen.ContractError
BudgetReached = frozen.BudgetReached
validate_preregistration = base.validate_preregistration
run = frozen.run
run_one = frozen.run_one


if __name__ == "__main__":
    raise SystemExit("use run_grid.py")
