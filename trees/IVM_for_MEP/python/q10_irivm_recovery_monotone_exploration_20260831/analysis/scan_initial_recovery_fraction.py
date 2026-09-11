#!/usr/bin/env python3
"""Offline diagnostic scan of the first N20 recovery direction scale.

This is not an optimizer: it freezes the online state after NoF 60, evaluates
scaled proposals from that same state, and reports their two public metrics.
It is used only to determine whether a one-band, recovery-only monotone
replacement exists before designing an online rule.
"""

from __future__ import annotations

import numpy as np

from python.q10_irivm_recovery_monotone_exploration_20260831.src import (
    q10_exact_envelope_optimizer as candidate,
)
from python.q10_irivm_unified_cycle_frozen_reproduction_v2_20260831.src.final_audit_core import (
    REFERENCE_BARRIER,
    _public_metrics,
    load_and_audit_article,
)


def main() -> None:
    prefix = candidate.run_clean(
        n_internal=20,
        alpha=1.0e-3,
        ratio=0.3,
        m_max=8,
        budget=60,
    )
    anchor_record = prefix["records"][-1]
    anchor = tuple(
        np.asarray(anchor_record[key], dtype=np.float64).copy()
        for key in ("path", "energies", "gradients")
    )
    memories = prefix["memories"]
    controls = prefix["controls"]
    core, potential = candidate.alg._load_backend(20)
    ledger = candidate.alg.Ledger(limit=100_000)
    evaluators = candidate.alg.make_evaluators(core, potential, ledger)

    root = candidate.ROOT
    data_root = root / "paper/ActiveExperiments/muller_q10_10d/data"
    with np.load(data_root / "inputs/Q_data.npz", allow_pickle=False) as archive:
        q = np.asarray(archive["Q"], dtype=np.float64)
    article = load_and_audit_article(
        data_root / "frozen_runs/tuned_ivm_alpha1e3_selected.npz", q, 20
    )
    checkpoint_index = int(np.flatnonzero(article.nof == 80)[0])
    article_i = float(article.i_perp[checkpoint_index])
    article_delta = float(article.delta_e[checkpoint_index])
    article_barrier_error = abs(article_delta - REFERENCE_BARRIER)
    anchor_i, anchor_delta = _public_metrics(anchor[0][None, ...], q)

    fractions = sorted(
        set(
            np.linspace(0.05, 4.0, 80).tolist()
            + [
                controls.r,
                1.0 - controls.r,
                1.0 + controls.r,
                1.0 / (1.0 - controls.r),
                1.0 / controls.r,
                np.ceil(controls.r * controls.m_max) - 1.0,
                np.ceil(controls.r * controls.m_max),
            ]
        )
    )
    rows = []
    for fraction in fractions:
        proposal, audit = candidate.recovery_proposal_spring(
            core, anchor, memories, controls, fraction, "warm"
        )
        if proposal is None:
            continue
        energies, gradients = candidate.alg.evaluate_path(
            core, proposal, evaluators
        )
        state = (proposal, energies, gradients)
        public_i, public_delta = _public_metrics(proposal[None, ...], q)
        barrier_error = abs(float(public_delta[0]) - REFERENCE_BARRIER)
        assessment = candidate.assess_recovery_candidate(
            core,
            anchor,
            state,
            float(anchor_i[0]),
            float(anchor_i[0]),
            controls,
            normal_relaxation=False,
        )
        rows.append(
            {
                "fraction": float(fraction),
                "i_perp": float(public_i[0]),
                "delta_e": float(public_delta[0]),
                "barrier_error": barrier_error,
                "monotone": bool(public_i[0] <= anchor_i[0]),
                "strict_i_at_80": bool(public_i[0] < article_i),
                "strict_barrier_at_80": bool(
                    barrier_error < article_barrier_error
                ),
                "predicted": float(assessment["predicted_actual_step"]),
                "actual": float(assessment["actual_merit"]),
                "step": float(audit["step"]),
            }
        )

    print(
        "anchor",
        float(anchor_i[0]),
        float(anchor_delta[0]),
        "article80",
        article_i,
        article_delta,
        article_barrier_error,
    )
    print("fraction i_perp delta_e berr mono I80 B80 pred actual")
    for row in rows:
        if row["monotone"] or abs(row["fraction"] - round(row["fraction"])) < 1e-12:
            print(
                f"{row['fraction']:.12g} {row['i_perp']:.12g} "
                f"{row['delta_e']:.12g} {row['barrier_error']:.12g} "
                f"{int(row['monotone'])} {int(row['strict_i_at_80'])} "
                f"{int(row['strict_barrier_at_80'])} "
                f"{row['predicted']:.12g} {row['actual']:.12g}"
            )


if __name__ == "__main__":
    main()
