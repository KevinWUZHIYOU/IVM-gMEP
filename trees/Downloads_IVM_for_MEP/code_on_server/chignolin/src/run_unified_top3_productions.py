#!/usr/bin/env python3
"""Independent 500-call productions for unified-score top-three candidates."""

from __future__ import annotations

import run_selected_method_productions as production


production.PRODUCTIONS = {
    "original_rank01_dt0p01255": {
        "method": "original_string",
        "dt": 0.01255,
        "reparam_interval": 1,
        "selection_role": "unified absolute screening rank 1 for method",
    },
    "original_rank02_dt0p012575": {
        "method": "original_string",
        "dt": 0.012575,
        "reparam_interval": 1,
        "selection_role": "unified absolute screening rank 2 for method",
    },
    "original_rank03_dt0p012525": {
        "method": "original_string",
        "dt": 0.012525,
        "reparam_interval": 1,
        "selection_role": "unified absolute screening rank 3 for method",
    },
    "static_rank01_k40_a0p012575": {
        "method": "static_neb",
        "spring_k": 40.0,
        "alpha": 0.012575,
        "selection_role": "unified absolute screening rank 1 for method",
    },
    "static_rank02_k40_a0p01255": {
        "method": "static_neb",
        "spring_k": 40.0,
        "alpha": 0.01255,
        "selection_role": "unified absolute screening rank 2 for method",
    },
    "static_rank03_k40_a0p012525": {
        "method": "static_neb",
        "spring_k": 40.0,
        "alpha": 0.012525,
        "selection_role": "unified absolute screening rank 3 for method",
    },
}


if __name__ == "__main__":
    production.main()
