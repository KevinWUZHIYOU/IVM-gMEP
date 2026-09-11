#!/usr/bin/env python3
"""Diagnostic scan of one paid N20 recovery proposal at NoF 80."""

from __future__ import annotations

import json
import os

import numpy as np

from python.q10_irivm_recovery_monotone_exploration_20260831.src import (
    q10_fraction_scan_optimizer as candidate,
)
from python.q10_irivm_unified_cycle_frozen_reproduction_v2_20260831.src.final_audit_core import (
    _public_metrics,
)


FRACTIONS = (
    1.05,
    1.10,
    1.15,
    1.20,
    1.25,
    1.30,
    1.35,
    1.40,
    1.45,
    1.50,
    1.60,
    1.70,
    1.80,
    2.00,
    2.20,
    2.40,
    2.60,
    2.80,
    3.00,
)


def main() -> None:
    rows = []
    for fraction in FRACTIONS:
        os.environ["Q10_DIAGNOSTIC_RECOVERY_FRACTION"] = str(fraction)
        result = candidate.run_clean(20, 1.0e-3, 0.3, 8, budget=80)
        records = result["records"]
        paths = np.asarray([item["path"] for item in records])
        metrics_i, metrics_delta = _public_metrics(paths, result["q_matrix"])
        public_nf = np.asarray([item["nf"] for item in records])
        index_60 = int(np.flatnonzero(public_nf <= 60)[-1])
        index_80 = int(np.flatnonzero(public_nf <= 80)[-1])
        audit = next(
            item for item in result["recovery_audits"]
            if int(item.get("nf", -1)) == 80
        )
        rows.append(
            {
                "fraction": fraction,
                "i_perp_60": float(metrics_i[index_60]),
                "i_perp_80": float(metrics_i[index_80]),
                "i_nonincreasing": bool(
                    metrics_i[index_80] <= metrics_i[index_60]
                ),
                "delta_e_80": float(metrics_delta[index_80]),
                "regularized_prediction": float(
                    audit["post_spring_regularized_prediction"]
                ),
                "public_action": str(audit["public_action"]),
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
