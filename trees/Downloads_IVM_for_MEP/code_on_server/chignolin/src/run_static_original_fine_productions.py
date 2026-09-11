#!/usr/bin/env python3
"""Independent productions selected by the 20260727e fine screening."""

from __future__ import annotations

import run_selected_method_productions as production


production.PRODUCTIONS = {
    "static_neb_fine": {
        "method": "static_neb",
        "spring_k": 40.0,
        "alpha": 0.0126,
        "selection_role": (
            "rank 1 in reports/static_fine_20260727e/manifest.json"
        ),
    },
    "original_string_fine": {
        "method": "original_string",
        "dt": 0.0125,
        "reparam_interval": 1,
        "selection_role": (
            "rank 1 in reports/original_fine_20260727e/manifest.json"
        ),
    },
}


if __name__ == "__main__":
    production.main()
