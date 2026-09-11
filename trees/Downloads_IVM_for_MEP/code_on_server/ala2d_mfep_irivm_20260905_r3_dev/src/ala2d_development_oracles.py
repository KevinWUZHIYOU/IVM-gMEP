"""Development-only calibrated noisy oracle for Ala2D IR-IVM screening.

This module never starts GROMACS.  It overlays resampled local MD--surrogate
residuals from three completed V13 repeats on the independent deterministic surrogate.
Every query is labelled non-formal and is counted only as a query equivalent.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from ala2d_oracles import OracleBand, OracleError, SurrogateOracle, _sha256_file
from riemannian_torus_geometry import TWO_PI, wrap_periodic


Array = np.ndarray


def _synthetic_seed(master_seed: int, band_id: int, image_index: int, stream: str) -> int:
    payload = (
        f"ala2d-calibrated-synthetic|{int(master_seed)}|{int(band_id)}|"
        f"{int(image_index)}|{stream}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (2**31 - 2)


def _project_spd(matrices: Array, floor: float = 1.0e-8) -> Array:
    symmetric = 0.5 * (matrices + np.swapaxes(matrices, -1, -2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.maximum(eigenvalues, float(floor))
    return np.einsum("...ik,...k,...jk->...ij", eigenvectors, clipped, eigenvectors)


class CalibratedSyntheticNoiseOracle:
    """Fixed-seed, spatially local resampling of observed MD fluctuations.

    The oracle is intended to reject fragile algorithm variants before new MD
    is spent.  It is explicitly not an independent validation set: the noise
    bank comes from the failed V13 campaign and performance claims must rely on
    newly preregistered restrained-MD repeats.
    """

    oracle_id = "fes-surrogate-plus-v13-local-md-noise-development-v1"
    formal_evidence = False

    def __init__(
        self,
        fes_npz: Path,
        metric_bank_npz: Path,
        calibration_npz: Path,
        calibration_provenance_json: Path,
        master_seed: int,
        neighbor_count: int = 64,
        neighbor_bandwidth_rad: float = 0.20,
        maximum_nearest_calibration_distance_rad: float = 0.60,
        maximum_farthest_neighbor_distance_rad: float = 0.70,
        noise_scale: float = 1.0,
        metric_noise: bool = True,
    ) -> None:
        if int(neighbor_count) < 1:
            raise ValueError("neighbor_count must be positive")
        if not np.isfinite(neighbor_bandwidth_rad) or float(neighbor_bandwidth_rad) <= 0.0:
            raise ValueError("neighbor_bandwidth_rad must be positive")
        if (
            not np.isfinite(maximum_nearest_calibration_distance_rad)
            or float(maximum_nearest_calibration_distance_rad) <= 0.0
        ):
            raise ValueError("maximum nearest calibration distance must be positive")
        if (
            not np.isfinite(maximum_farthest_neighbor_distance_rad)
            or
            float(maximum_farthest_neighbor_distance_rad)
            < float(maximum_nearest_calibration_distance_rad)
        ):
            raise ValueError("farthest-neighbor support limit must exceed nearest limit")
        if not np.isfinite(noise_scale) or float(noise_scale) < 0.0:
            raise ValueError("noise_scale must be nonnegative")

        self.base = SurrogateOracle(fes_npz=fes_npz, metric_bank_npz=metric_bank_npz)
        self.calibration_npz = Path(calibration_npz).resolve()
        self.calibration_provenance_json = Path(calibration_provenance_json).resolve()
        self.master_seed = int(master_seed)
        self.neighbor_count = int(neighbor_count)
        self.neighbor_bandwidth_rad = float(neighbor_bandwidth_rad)
        self.maximum_nearest_calibration_distance_rad = float(
            maximum_nearest_calibration_distance_rad
        )
        self.maximum_farthest_neighbor_distance_rad = float(
            maximum_farthest_neighbor_distance_rad
        )
        self.noise_scale = float(noise_scale)
        self.metric_noise = bool(metric_noise)
        self.call_count = 0

        provenance = json.loads(
            self.calibration_provenance_json.read_text(encoding="utf-8")
        )
        if provenance.get("formal_evidence") is not False:
            raise OracleError("noise calibration is not marked development-only")
        if provenance.get("output_npz_sha256") != _sha256_file(self.calibration_npz):
            raise OracleError("noise calibration archive does not match its provenance")

        with np.load(self.calibration_npz, allow_pickle=False) as loaded:
            if bool(np.asarray(loaded["formal_evidence"]).item()):
                raise OracleError("noise calibration archive is incorrectly formal")
            base_provenance = self.base.provenance()
            expected_fes = str(np.asarray(loaded["fes_npz_sha256"]).item())
            expected_metric = str(np.asarray(loaded["metric_bank_npz_sha256"]).item())
            if expected_fes != base_provenance["fes_npz_sha256"]:
                raise OracleError("FES archive differs from noise-calibration base")
            if expected_metric != base_provenance["metric_bank_npz_sha256"]:
                raise OracleError("metric bank differs from noise-calibration base")
            self.centers = np.asarray(loaded["centers"], dtype=np.float64)
            if str(np.asarray(loaded["implementation_id"]).item()) != (
                "ala2d-v13-confirmed-md-noise-calibration-v1"
            ):
                raise OracleError("unexpected noise-calibration implementation")
            expected_counts = {
                "source_repeat_count": 3,
                "source_paid_band_count": 300,
                "source_image_evaluation_count": 6000,
                "block_count": 10,
            }
            for name, expected in expected_counts.items():
                if int(np.asarray(loaded[name]).item()) != expected:
                    raise OracleError(f"unexpected noise-calibration {name}")
            self.gradient_md_surrogate_residual = np.asarray(
                loaded["gradient_md_surrogate_residual"], dtype=np.float64
            )
            self.metric_md_surrogate_residual = np.asarray(
                loaded["metric_md_surrogate_residual"], dtype=np.float64
            )
            self.gradient_block_residual = np.asarray(
                loaded["gradient_block_residual"], dtype=np.float64
            )
            self.metric_block_residual = np.asarray(
                loaded["metric_block_residual"], dtype=np.float64
            )
            self.repeat_index = np.asarray(loaded["repeat_index"], dtype=np.int64)
            self.state_index = np.asarray(loaded["state_index"], dtype=np.int64)
            self.image_index = np.asarray(loaded["image_index"], dtype=np.int64)
            self.source_seed = np.asarray(loaded["source_seed"], dtype=np.int64)
            self.source_frame_count = np.asarray(
                loaded["source_frame_count"], dtype=np.int64
            )
            self.source_discarded_frame_count = np.asarray(
                loaded["source_discarded_frame_count"], dtype=np.int64
            )

        if self.centers.shape != (6000, 2):
            raise OracleError(f"unexpected noise calibration shape {self.centers.shape}")
        if self.gradient_block_residual.shape != (6000, 10, 2):
            raise OracleError("unexpected calibrated gradient-block shape")
        if self.metric_block_residual.shape != (6000, 10, 2, 2):
            raise OracleError("unexpected calibrated metric-block shape")
        expected_shapes = {
            "gradient_md_surrogate_residual": (
                self.gradient_md_surrogate_residual,
                (6000, 2),
            ),
            "metric_md_surrogate_residual": (
                self.metric_md_surrogate_residual,
                (6000, 2, 2),
            ),
            "repeat_index": (self.repeat_index, (6000,)),
            "state_index": (self.state_index, (6000,)),
            "image_index": (self.image_index, (6000,)),
            "source_seed": (self.source_seed, (6000,)),
            "source_frame_count": (self.source_frame_count, (6000,)),
            "source_discarded_frame_count": (
                self.source_discarded_frame_count,
                (6000,),
            ),
        }
        for name, (value, shape) in expected_shapes.items():
            if value.shape != shape:
                raise OracleError(f"unexpected calibrated {name} shape {value.shape}")
        if not np.array_equal(np.unique(self.repeat_index), np.arange(3)):
            raise OracleError("calibration repeat indices are incomplete")
        if not np.array_equal(np.unique(self.state_index), np.arange(1, 101)):
            raise OracleError("calibration state indices are incomplete")
        if not np.array_equal(np.unique(self.image_index), np.arange(20)):
            raise OracleError("calibration image indices are incomplete")
        if np.any(self.source_frame_count <= 0) or np.any(
            self.source_discarded_frame_count < 0
        ):
            raise OracleError("calibration frame counts are invalid")
        if not all(
            np.all(np.isfinite(value))
            for value in (
                self.centers,
                self.gradient_md_surrogate_residual,
                self.metric_md_surrogate_residual,
                self.gradient_block_residual,
                self.metric_block_residual,
            )
        ):
            raise OracleError("noise calibration contains non-finite values")
        if self.neighbor_count > self.centers.shape[0]:
            raise ValueError("neighbor_count cannot exceed unique calibration sources")

        shifts = np.asarray(
            [(i * TWO_PI, j * TWO_PI) for i in (-1, 0, 1) for j in (-1, 0, 1)],
            dtype=np.float64,
        )
        tiled_centers = np.concatenate(
            [wrap_periodic(self.centers) + shift for shift in shifts], axis=0
        )
        self._source_indices = np.tile(np.arange(self.centers.shape[0]), len(shifts))
        self._tree = cKDTree(tiled_centers)

    def provenance(self) -> Dict[str, object]:
        return {
            "oracle_id": self.oracle_id,
            "formal_evidence": False,
            "meaning": "fixed local MD--surrogate-residual development screen; not MD evidence",
            "base_surrogate": self.base.provenance(),
            "calibration_npz": str(self.calibration_npz),
            "calibration_npz_sha256": _sha256_file(self.calibration_npz),
            "calibration_provenance_json": str(self.calibration_provenance_json),
            "calibration_provenance_json_sha256": _sha256_file(
                self.calibration_provenance_json
            ),
            "master_seed": self.master_seed,
            "neighbor_count": self.neighbor_count,
            "neighbor_bandwidth_rad": self.neighbor_bandwidth_rad,
            "maximum_nearest_calibration_distance_rad": (
                self.maximum_nearest_calibration_distance_rad
            ),
            "maximum_farthest_neighbor_distance_rad": (
                self.maximum_farthest_neighbor_distance_rad
            ),
            "noise_scale": self.noise_scale,
            "metric_noise": self.metric_noise,
        }

    def _choose_source(
        self, center: Array, band_id: int, image_index: int
    ) -> Tuple[int, float, float]:
        count = self.neighbor_count
        distances, tiled_indices = self._tree.query(
            wrap_periodic(np.asarray(center, dtype=np.float64)), k=count, workers=1
        )
        distances = np.atleast_1d(np.asarray(distances, dtype=np.float64))
        tiled_indices = np.atleast_1d(np.asarray(tiled_indices, dtype=np.int64))
        source_indices = self._source_indices[tiled_indices]
        weights = np.exp(
            -0.5 * (distances / max(self.neighbor_bandwidth_rad, 1.0e-12)) ** 2
        )
        if not np.all(np.isfinite(weights)) or float(np.sum(weights)) <= 0.0:
            weights = np.ones_like(distances)
        weights = weights / np.sum(weights)
        rng = np.random.default_rng(
            _synthetic_seed(self.master_seed, band_id, image_index, "source")
        )
        return (
            int(rng.choice(source_indices, p=weights)),
            float(distances[0]),
            float(distances[-1]),
        )

    def evaluate_band(
        self,
        full_path: Array,
        band_id: int,
        band_directory: Optional[Path] = None,
    ) -> OracleBand:
        del band_directory
        path = np.asarray(full_path, dtype=np.float64)
        if path.shape != (22, 2):
            raise OracleError(f"expected a 22-image path, got {path.shape}")
        centers = path[1:-1]
        truth_gradient = self.base._gradient(centers)
        truth_metric = self.base._metric(centers)
        choices = [
            self._choose_source(center, band_id, i) for i, center in enumerate(centers)
        ]
        selected = np.asarray([item[0] for item in choices], dtype=np.int64)
        nearest_distances = np.asarray([item[1] for item in choices], dtype=np.float64)
        farthest_neighbor_distances = np.asarray(
            [item[2] for item in choices], dtype=np.float64
        )
        if float(np.max(nearest_distances)) > self.maximum_nearest_calibration_distance_rad:
            raise OracleError(
                "synthetic-noise path lies outside the nearest-point calibration support"
            )
        if (
            float(np.max(farthest_neighbor_distances))
            > self.maximum_farthest_neighbor_distance_rad
        ):
            raise OracleError(
                "synthetic-noise path lacks the required local-neighbor calibration support"
            )

        gradient_center = truth_gradient + self.noise_scale * self.gradient_md_surrogate_residual[
            selected
        ]
        gradient_blocks = gradient_center[:, None, :] + self.noise_scale * self.gradient_block_residual[
            selected
        ]
        retained_counts = (
            self.source_frame_count[selected] - self.source_discarded_frame_count[selected]
        )
        quotient = retained_counts // 10
        remainder = retained_counts % 10
        block_counts = quotient[:, None] + (
            np.arange(10)[None, :] < remainder[:, None]
        )
        gradient = np.einsum("nk,nki->ni", block_counts, gradient_blocks) / retained_counts[:, None]
        if self.metric_noise:
            metric_center = _project_spd(
                truth_metric
                + self.noise_scale * self.metric_md_surrogate_residual[selected]
            )
            metric_blocks = _project_spd(
                metric_center[:, None, :, :]
                + self.noise_scale * self.metric_block_residual[selected]
            )
            metric = np.einsum(
                "nk,nkij->nij", block_counts, metric_blocks
            ) / retained_counts[:, None, None]
        else:
            metric = truth_metric
            metric_blocks = np.repeat(metric[:, None, :, :], 10, axis=1)
        if self.noise_scale == 0.0:
            # Preserve the deterministic oracle bit pattern at the zero-noise
            # boundary; its repeated blocks still satisfy weighted aggregation.
            gradient = truth_gradient.copy()
            metric = truth_metric.copy()

        seeds = np.asarray(
            [
                _synthetic_seed(self.master_seed, band_id, image_index, "estimate")
                for image_index in range(20)
            ],
            dtype=np.int64,
        )
        thermostat_seeds = np.asarray(
            [
                _synthetic_seed(self.master_seed, band_id, image_index, "thermostat")
                for image_index in range(20)
            ],
            dtype=np.int64,
        )
        self.call_count += 20
        return OracleBand(
            centers=centers.copy(),
            mean_coordinates=centers.copy(),
            grad_F=gradient,
            M=metric,
            grad_F_block_means=gradient_blocks,
            M_block_means=metric_blocks,
            seeds=seeds,
            thermostat_seeds=thermostat_seeds,
            frame_counts=self.source_frame_count[selected].copy(),
            discarded_frame_counts=self.source_discarded_frame_count[selected].copy(),
            wall_seconds=np.zeros(20, dtype=np.float64),
            nof=20,
            metadata={
                "oracle_id": self.oracle_id,
                "formal_evidence": False,
                "query_equivalents_only": True,
                "band_id": int(band_id),
                "master_seed": self.master_seed,
                "selected_calibration_indices": selected.tolist(),
                "selected_source_repeat_indices": self.repeat_index[selected].tolist(),
                "selected_source_state_indices": self.state_index[selected].tolist(),
                "selected_source_image_indices": self.image_index[selected].tolist(),
                "nearest_calibration_distance_rad": nearest_distances.tolist(),
                "farthest_selected_neighbor_distance_rad": (
                    farthest_neighbor_distances.tolist()
                ),
                "observed_maximum_nearest_calibration_distance_rad": float(
                    np.max(nearest_distances)
                ),
                "observed_maximum_farthest_selected_neighbor_distance_rad": float(
                    np.max(farthest_neighbor_distances)
                ),
                "nearest_calibration_support_limit_rad": (
                    self.maximum_nearest_calibration_distance_rad
                ),
                "farthest_neighbor_support_limit_rad": (
                    self.maximum_farthest_neighbor_distance_rad
                ),
            },
        )
