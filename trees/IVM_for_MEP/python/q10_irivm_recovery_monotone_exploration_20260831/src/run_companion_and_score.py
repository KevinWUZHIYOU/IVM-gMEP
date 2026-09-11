#!/usr/bin/env python3
"""Run and independently score the companion-validated QM10 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from python.q10_irivm_unified_cycle_frozen_reproduction_v2_20260831.src.final_audit_core import (
    _public_metrics,
    load_and_audit_article,
    locf_indices,
    strict_checkpoint_comparison,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "runs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paths(n_internal: int) -> tuple[Path, Path]:
    base = REPOSITORY_ROOT / "paper/ActiveExperiments/muller_q10_10d"
    if n_internal == 10:
        base = base / "image12"
        article = base / "data/frozen_runs/tuned_ivm_alpha2e3_selected.npz"
    elif n_internal == 20:
        article = base / "data/frozen_runs/tuned_ivm_alpha1e3_selected.npz"
    else:
        raise ValueError(n_internal)
    return base / "data/inputs/Q_data.npz", article


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, choices=(10, 20), required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--r", type=float, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--budget", type=int, default=3000)
    parser.add_argument("--label-suffix", default="")
    parser.add_argument(
        "--variant",
        choices=("companion_validated",),
        default="companion_validated",
    )
    arguments = parser.parse_args()

    from . import q10_companion_validated_optimizer as candidate

    result = candidate.run_clean(
        arguments.n,
        arguments.alpha,
        arguments.r,
        arguments.m,
        budget=arguments.budget,
    )
    records = result["records"]
    public_nof = np.asarray([item["nf"] for item in records], dtype=np.int64)
    public_phase = np.asarray([item["phase"] for item in records], dtype="U64")
    public_path = np.asarray([item["path"] for item in records], dtype=np.float64)

    q_path, article_path = _paths(arguments.n)
    with np.load(q_path, allow_pickle=False) as archive:
        q = np.asarray(archive["Q"], dtype=np.float64)
    public_i, public_delta = _public_metrics(public_path, q)
    article = load_and_audit_article(article_path, q, arguments.n)
    checkpoints = article.nof[1:]
    candidate_indices = locf_indices(public_nof, checkpoints)
    comparison = strict_checkpoint_comparison(
        public_i[candidate_indices],
        article.i_perp[1:],
        public_delta[candidate_indices],
        article.delta_e[1:],
    )
    rises = np.diff(public_i)
    rise_indices = np.flatnonzero(rises > 0.0) + 1

    controls_label = (
        f"{arguments.variant}_n{arguments.n}_a{arguments.alpha:.12g}_"
        f"r{arguments.r:.12g}_m{arguments.m}_b{arguments.budget}"
        f"{arguments.label_suffix}"
    ).replace(".", "p")
    output_directory = OUTPUT_ROOT / controls_label
    output_directory.mkdir(parents=True, exist_ok=False)
    trajectory_path = output_directory / "trajectory.npz"
    np.savez_compressed(
        trajectory_path,
        public_nf=public_nof,
        public_phase=public_phase,
        public_path=public_path,
        public_i_perp=public_i,
        public_delta_e=public_delta,
        checkpoint_nof=checkpoints,
        checkpoint_candidate_state_index=candidate_indices,
        checkpoint_i_perp_strict=comparison["i_perp_strict"],
        checkpoint_barrier_strict=comparison["barrier_strict"],
        checkpoint_joint_strict=comparison["joint_strict"],
    )
    summary = {
        "algorithm_sha256": _sha256(Path(candidate.__file__)),
        "variant": arguments.variant,
        "n_internal": arguments.n,
        "alpha": arguments.alpha,
        "r": arguments.r,
        "m_max": arguments.m,
        "budget": arguments.budget,
        "terminal_nof": int(result["terminal_nf"]),
        "stop_reason": str(result["stop_reason"]),
        "nof_breakdown": result["nof_breakdown"],
        "nof_per_image_sum": int(np.sum(result["per_image_joint_evaluations"])),
        "public_state_count": int(public_nof.size),
        "i_perp_rise_count": int(rise_indices.size),
        "maximum_i_perp_rise": (
            float(np.max(rises[rises > 0.0])) if rise_indices.size else 0.0
        ),
        "rise_events": [
            {
                "state_index": int(index),
                "nof": int(public_nof[index]),
                "phase": str(public_phase[index]),
                "before": float(public_i[index - 1]),
                "after": float(public_i[index]),
                "rise": float(rises[index - 1]),
            }
            for index in rise_indices.tolist()
        ],
        "checkpoint_count": int(checkpoints.size),
        "strict_i_perp_count": int(np.count_nonzero(comparison["i_perp_strict"])),
        "strict_barrier_count": int(np.count_nonzero(comparison["barrier_strict"])),
        "strict_joint_count": int(np.count_nonzero(comparison["joint_strict"])),
        "trusted_nonmonotone_public_commits": int(
            result["trusted_nonmonotone_commits"]
        ),
        "fraction_choice_events": [
            {
                "outer": int(item.get("outer", -1)),
                "nof": int(item.get("nf", -1)),
                "public_action": str(item.get("public_action", "")),
                "selected_fraction": float(item["zero_nof_fraction_choice"]),
                "options": item["zero_nof_fraction_options"],
            }
            for item in result["recovery_audits"]
            if "zero_nof_fraction_choice" in item
        ],
        "trajectory_relative_path": trajectory_path.name,
    }
    summary_path = output_directory / "summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
