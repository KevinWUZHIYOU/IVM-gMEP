#!/usr/bin/env python3
"""Independent confirmation runs for the four selected 1000-call methods."""

from __future__ import annotations

import run_extended1000_candidates as extended


extended.production.PRODUCTIONS = {
    "confirmed1000_original_dt0p012575": {
        **extended.productions["extended1000_original_dt0p012575"],
        "selection_role": (
            "independent confirmation after unified dense selection"
        ),
    },
    "confirmed1000_static_k40_a0p012575": {
        **extended.productions["extended1000_static_k40_a0p012575"],
        "selection_role": (
            "independent confirmation after unified dense selection"
        ),
    },
    "confirmed1000_ode_k50": {
        **extended.productions["extended1000_ode_k50p0"],
        "selection_role": (
            "independent confirmation after unified dense selection"
        ),
    },
    "confirmed1000_obsm_l0p9_n5_dt0p12": {
        **extended.productions[
            "extended1000_obsm_l0p9_n5_dt0p12"
        ],
        "selection_role": (
            "independent confirmation after unified dense selection"
        ),
    },
}


if __name__ == "__main__":
    extended.production.main()
