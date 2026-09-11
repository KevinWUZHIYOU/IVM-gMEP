#!/usr/bin/env python3
"""Build an auditable synthetic-noise calibration from the failed V13 MD.

The resulting archive is development data only.  It records how individual
ten-block restrained-MD estimates differed from the independent FES spline
and from the smooth historical metric interpolant.  No new molecular dynamics
is run here and the archive is never valid as formal performance evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import scipy

import ala2d_oracles
import riemannian_torus_geometry
from ala2d_oracles import SurrogateOracle


IMPLEMENTATION_ID = "ala2d-v13-confirmed-md-noise-calibration-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _require_completed_run(run_directory: Path) -> List[Path]:
    completion_path = run_directory / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("completed") is not True:
        raise RuntimeError(f"source run is not complete: {run_directory}")
    if int(completion.get("nof_mean_force_incremental", -1)) != 2000:
        raise RuntimeError(f"source run does not contain exactly 2000 NoF: {run_directory}")
    if not str(completion.get("implementation_id", "")).startswith(
        "ala2d-riemannian-v13-"
    ):
        raise RuntimeError(f"source run is not a V13 trajectory: {run_directory}")
    state_paths = sorted((run_directory / "states").glob("state_*.npz"))
    paid = [path for path in state_paths if path.name != "state_000000.npz"]
    if len(paid) != 100:
        raise RuntimeError(
            f"expected 100 paid 20-image states in {run_directory}, found {len(paid)}"
        )
    expected_names = [f"state_{index:06d}.npz" for index in range(1, 101)]
    if [path.name for path in paid] != expected_names:
        raise RuntimeError(f"source paid-state sequence is not exactly 1..100: {run_directory}")
    return paid


def _iter_state_records(
    run_directories: Sequence[Path], surrogate: SurrogateOracle
) -> Iterable[Dict[str, np.ndarray]]:
    for repeat_index, run_directory in enumerate(run_directories):
        for state_path in _require_completed_run(run_directory):
            with np.load(state_path, allow_pickle=False) as loaded:
                if not bool(np.asarray(loaded["complete_band"]).item()):
                    raise RuntimeError(f"incomplete source state: {state_path}")
                centers = np.asarray(loaded["centers"], dtype=np.float64)
                gradient = np.asarray(loaded["grad_F"], dtype=np.float64)
                metric = np.asarray(loaded["M"], dtype=np.float64)
                gradient_blocks = np.asarray(
                    loaded["grad_F_block_means"], dtype=np.float64
                )
                metric_blocks = np.asarray(
                    loaded["M_block_means"], dtype=np.float64
                )
                state_index = int(np.asarray(loaded["state_index"]).item())
                nof = int(
                    np.asarray(loaded["nof_mean_force_incremental"]).item()
                )
                implementation_id = str(
                    np.asarray(loaded["implementation_id"]).item()
                )
                source_seeds = np.asarray(loaded["md_seeds"], dtype=np.int64)
                frame_counts = np.asarray(loaded["frame_counts"], dtype=np.int64)
                discarded_frame_counts = np.asarray(
                    loaded["discarded_frame_counts"], dtype=np.int64
                )
            expected_index = int(state_path.stem.rsplit("_", 1)[-1])
            if state_index != expected_index or nof != 20 * state_index:
                raise RuntimeError(f"state-index/NoF mismatch in {state_path}")
            if not implementation_id.startswith("ala2d-riemannian-v13-"):
                raise RuntimeError(f"unexpected implementation in {state_path}")
            if centers.shape != (20, 2):
                raise RuntimeError(f"unexpected center shape in {state_path}: {centers.shape}")
            if gradient_blocks.shape != (20, 10, 2):
                raise RuntimeError(
                    f"unexpected gradient-block shape in {state_path}: "
                    f"{gradient_blocks.shape}"
                )
            if metric_blocks.shape != (20, 10, 2, 2):
                raise RuntimeError(
                    f"unexpected metric-block shape in {state_path}: {metric_blocks.shape}"
                )

            truth_gradient = surrogate._gradient(centers)
            truth_metric = surrogate._metric(centers)
            yield {
                "centers": centers,
                "gradient_md_surrogate_residual": gradient - truth_gradient,
                "metric_md_surrogate_residual": metric - truth_metric,
                # Center within each paid image.  Synthetic band estimates are
                # supplied separately by gradient_error/metric_error.
                "gradient_block_residual": gradient_blocks
                - np.mean(gradient_blocks, axis=1, keepdims=True),
                "metric_block_residual": metric_blocks
                - np.mean(metric_blocks, axis=1, keepdims=True),
                "repeat_index": np.full(20, repeat_index, dtype=np.int64),
                "state_index": np.full(20, state_index, dtype=np.int64),
                "image_index": np.arange(20, dtype=np.int64),
                "source_seed": source_seeds,
                "source_frame_count": frame_counts,
                "source_discarded_frame_count": discarded_frame_counts,
            }


def build(
    run_directories: Sequence[Path],
    fes_npz: Path,
    metric_bank_npz: Path,
    output_npz: Path,
    output_provenance: Path,
) -> None:
    resolved_runs = [Path(path).resolve() for path in run_directories]
    surrogate = SurrogateOracle(fes_npz=fes_npz, metric_bank_npz=metric_bank_npz)
    records = list(_iter_state_records(resolved_runs, surrogate))
    if len(records) != 300:
        raise RuntimeError(f"expected 300 paid bands across three repeats, got {len(records)}")

    array_keys = (
        "centers",
        "gradient_md_surrogate_residual",
        "metric_md_surrogate_residual",
        "gradient_block_residual",
        "metric_block_residual",
        "repeat_index",
        "state_index",
        "image_index",
        "source_seed",
        "source_frame_count",
        "source_discarded_frame_count",
    )
    arrays = {
        key: np.concatenate([np.asarray(record[key]) for record in records], axis=0)
        for key in array_keys
    }
    if arrays["centers"].shape != (6000, 2):
        raise RuntimeError(f"unexpected calibration size: {arrays['centers'].shape}")

    base_provenance = surrogate.provenance()
    arrays.update(
        {
            "implementation_id": np.asarray(IMPLEMENTATION_ID),
            "formal_evidence": np.asarray(False, dtype=bool),
            "source_repeat_count": np.asarray(3, dtype=np.int64),
            "source_paid_band_count": np.asarray(300, dtype=np.int64),
            "source_image_evaluation_count": np.asarray(6000, dtype=np.int64),
            "block_count": np.asarray(10, dtype=np.int64),
            "fes_npz_sha256": np.asarray(base_provenance["fes_npz_sha256"]),
            "metric_bank_npz_sha256": np.asarray(
                base_provenance["metric_bank_npz_sha256"]
            ),
        }
    )
    _atomic_npz(output_npz, **arrays)

    source_files = []
    for run_directory in resolved_runs:
        completion_path = run_directory / "completion.json"
        source_files.append(
            {
                "path": str(completion_path),
                "sha256": _sha256(completion_path),
                "role": "completed-run ledger",
            }
        )
        for state_path in _require_completed_run(run_directory):
            source_files.append(
                {
                    "path": str(state_path),
                    "sha256": _sha256(state_path),
                    "role": "one complete paid 20-image MD band",
                }
            )

    payload: Dict[str, object] = {
        "implementation_id": IMPLEMENTATION_ID,
        "status": "development-only synthetic-noise calibration",
        "formal_evidence": False,
        "new_molecular_dynamics_run": False,
        "meaning": (
            "Nearest-neighbour resampling bank for MD estimator error and ten-block "
            "fluctuations; it cannot establish algorithm performance."
        ),
        "construction": {
            "repeats": 3,
            "paid_bands_per_repeat": 100,
            "internal_images_per_band": 20,
            "blocks_per_image": 10,
            "mean_force_evaluations_reused": 6000,
            "additional_mean_force_evaluations": 0,
            "state_zero_excluded": True,
            "gradient_md_surrogate_residual_definition": "stored grad_F minus deterministic FES-spline gradient; includes finite-sample fluctuation, restraint/FES mismatch, and other local model discrepancy",
            "metric_md_surrogate_residual_definition": "stored M minus smooth historical-M interpolant; includes finite-sample fluctuation and local model discrepancy",
            "block_residual_definition": "stored block estimate minus its ten-block arithmetic mean",
        },
        "base_surrogate": base_provenance,
        "output_npz": str(output_npz.resolve()),
        "output_npz_sha256": _sha256(output_npz),
        "builder": str(Path(__file__).resolve()),
        "builder_sha256": _sha256(Path(__file__).resolve()),
        "imported_source": {
            str(Path(ala2d_oracles.__file__).resolve()): _sha256(
                Path(ala2d_oracles.__file__).resolve()
            ),
            str(Path(riemannian_torus_geometry.__file__).resolve()): _sha256(
                Path(riemannian_torus_geometry.__file__).resolve()
            ),
        },
        "runtime": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "source_files": source_files,
    }
    manifest_digest = hashlib.sha256(
        "".join(item["sha256"] for item in source_files).encode("ascii")
    ).hexdigest()
    payload["ordered_source_sha256_digest"] = manifest_digest
    _atomic_json(output_provenance, payload)


def _default_paths() -> Dict[str, object]:
    campaign = Path(__file__).resolve().parents[1]
    r1 = campaign.parent / "ala2d_mfep_irivm_20260905_r1"
    return {
        "runs": [
            r1 / "runs/md/pilot/a13em5_r0p5_m4/seed_41",
            r1 / "runs/md/confirm/a13em5_r0p5_m4/seed_42",
            r1 / "runs/md/confirm/a13em5_r0p5_m4/seed_43",
        ],
        "fes": campaign / "inputs/reference/fes_XYZ.npz",
        "metric_bank": campaign / "inputs/reference/current_ivm.npz",
        "output": campaign / "inputs/noise_calibration/r1_v13_confirmed_block_noise.npz",
        "provenance": campaign
        / "inputs/noise_calibration/r1_v13_confirmed_block_noise.provenance.json",
    }


def main() -> None:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=Path, dest="runs")
    parser.add_argument("--fes", type=Path, default=defaults["fes"])
    parser.add_argument("--metric-bank", type=Path, default=defaults["metric_bank"])
    parser.add_argument("--output", type=Path, default=defaults["output"])
    parser.add_argument("--provenance", type=Path, default=defaults["provenance"])
    arguments = parser.parse_args()
    build(
        run_directories=arguments.runs or defaults["runs"],
        fes_npz=arguments.fes,
        metric_bank_npz=arguments.metric_bank,
        output_npz=arguments.output,
        output_provenance=arguments.provenance,
    )


if __name__ == "__main__":
    main()
