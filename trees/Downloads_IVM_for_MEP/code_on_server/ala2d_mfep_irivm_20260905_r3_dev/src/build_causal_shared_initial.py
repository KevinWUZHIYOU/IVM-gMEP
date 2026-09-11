"""Build the causal shared initial state for the historical Ala2D IVM.

The legacy ``MFEP_INITIAL_PATH_common.npz`` has the right straight-line
geometry, but its sampled metric and free-energy gradient are not the arrays
stored at state 0 of ``current_ivm.npz``.  Replacing only the plotted state-0
metrics therefore splices together observations from two different restrained
MD evaluations.  This builder instead copies the complete historical state-0
observation and recomputes the article metrics from that observation.

The legacy file is used only for the fixed endpoints (and as an auditable
geometry cross-check); it is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import sys
import zipfile
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from riemannian_torus_geometry import compute_article_metrics


Array = np.ndarray
SCHEMA_VERSION = "ala2d.causal_shared_initial.v1"
EXPECTED_C1 = np.asarray([-2.60866606, 2.77661136], dtype=np.float64)
EXPECTED_C2 = np.asarray([-1.33047160, 0.99118795], dtype=np.float64)
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class CausalInitialError(RuntimeError):
    """Raised when the historical state cannot be reconstructed safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_npz(path: Path) -> Dict[str, Array]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.array(loaded[key], copy=True) for key in loaded.files}


def _require_keys(data: Mapping[str, Array], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise CausalInitialError(f"{label} is missing required keys: {missing}")


def _atomic_write_bytes(path: Path, payload: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _deterministic_npz(arrays: Mapping[str, Array]) -> bytes:
    """Return a byte-reproducible, pickle-free compressed NPZ payload."""

    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for key, value in arrays.items():
            if "/" in key or "\\" in key:
                raise CausalInitialError(f"unsafe NPZ key: {key!r}")
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise CausalInitialError(f"object arrays are forbidden: {key}")
            npy = io.BytesIO()
            np.lib.format.write_array(npy, array, allow_pickle=False)
            member = zipfile.ZipInfo(f"{key}.npy", FIXED_ZIP_TIMESTAMP)
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, npy.getvalue(), compress_type=zipfile.ZIP_DEFLATED)
    return output.getvalue()


def _json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _scalar(value: object, dtype: object = np.float64) -> Array:
    return np.asarray(value, dtype=dtype)


def build_causal_shared_initial(
    current_ivm_npz: Path,
    legacy_common_npz: Path,
    output_npz: Path,
    provenance_json: Optional[Path] = None,
    *,
    overwrite: bool = False,
) -> Dict[str, object]:
    """Reconstruct and save the actual state 0 of the historical current IVM.

    One evaluation at each of the 20 internal CV images produced the full
    two-component mean-force vector and the local metric.  Those 20 historical
    mean-force evaluations are real, but the shared initial state is displayed
    at incremental NoF 0; the next saved historical state is consequently at
    incremental NoF 20.
    """

    current_ivm_npz = Path(current_ivm_npz).resolve()
    legacy_common_npz = Path(legacy_common_npz).resolve()
    output_npz = Path(output_npz).resolve()
    if provenance_json is None:
        provenance_json = output_npz.with_suffix(".provenance.json")
    provenance_json = Path(provenance_json).resolve()

    if output_npz in {current_ivm_npz, legacy_common_npz}:
        raise CausalInitialError("output NPZ must not replace either source NPZ")
    if provenance_json in {current_ivm_npz, legacy_common_npz, output_npz}:
        raise CausalInitialError("provenance JSON must be a distinct output")
    if output_npz.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {output_npz}")
    if provenance_json.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing output: {provenance_json}"
        )

    current = _load_npz(current_ivm_npz)
    legacy = _load_npz(legacy_common_npz)
    _require_keys(
        current,
        [
            "index_seq",
            "phi_ref_iter_0",
            "M_ref_iter_0",
            "F_ref_iter_0",
            "arc_ref_iter_0",
            "path_iter_0",
            "phi_ref_iter_1",
        ],
        "historical current IVM archive",
    )
    _require_keys(
        legacy,
        [
            "C1",
            "C2",
            "N",
            "phi_init",
            "full_path_init",
            "M_init",
            "F_init",
        ],
        "legacy common-initial archive",
    )

    index_seq = np.asarray(current["index_seq"])
    if index_seq.ndim != 1 or index_seq.size == 0 or int(index_seq[0]) != 0:
        raise CausalInitialError("historical current IVM does not begin at state 0")

    C1 = np.asarray(legacy["C1"], dtype=np.float64)
    C2 = np.asarray(legacy["C2"], dtype=np.float64)
    if not np.array_equal(C1, EXPECTED_C1) or not np.array_equal(C2, EXPECTED_C2):
        raise CausalInitialError("endpoint source differs from the current-IVM fixed endpoints")

    phi = np.asarray(current["phi_ref_iter_0"], dtype=np.float64)
    M = np.asarray(current["M_ref_iter_0"], dtype=np.float64)
    grad_F = np.asarray(current["F_ref_iter_0"], dtype=np.float64)
    arc = np.asarray(current["arc_ref_iter_0"], dtype=np.float64)
    n_internal = int(np.asarray(legacy["N"]).item())
    if phi.shape != (n_internal, 2):
        raise CausalInitialError(f"unexpected phi_ref_iter_0 shape: {phi.shape}")
    if M.shape != (n_internal, 2, 2):
        raise CausalInitialError(f"unexpected M_ref_iter_0 shape: {M.shape}")
    if grad_F.shape != (n_internal, 2):
        raise CausalInitialError(f"unexpected F_ref_iter_0 shape: {grad_F.shape}")
    if arc.shape != (n_internal,):
        raise CausalInitialError(f"unexpected arc_ref_iter_0 shape: {arc.shape}")
    if not all(np.all(np.isfinite(value)) for value in (phi, M, grad_F, arc)):
        raise CausalInitialError("state-0 observation contains non-finite values")

    if "g_ref_iter_0" in current and not np.array_equal(
        grad_F, np.asarray(current["g_ref_iter_0"], dtype=np.float64)
    ):
        raise CausalInitialError("F_ref_iter_0 and g_ref_iter_0 disagree")
    if not np.array_equal(current["path_iter_0"], current["phi_ref_iter_1"]):
        raise CausalInitialError(
            "path_iter_0 is not the next reference state phi_ref_iter_1"
        )

    full_path = np.vstack([C1, phi, C2])
    legacy_path = np.asarray(legacy["full_path_init"], dtype=np.float64)
    if legacy_path.shape != full_path.shape:
        raise CausalInitialError("legacy and historical state-0 paths have different shapes")
    path_max_abs_difference = float(np.max(np.abs(full_path - legacy_path)))
    path_rms_difference = float(
        np.sqrt(np.mean(np.sum((full_path - legacy_path) ** 2, axis=1)))
    )
    if path_max_abs_difference > 1.0e-12:
        raise CausalInitialError(
            "legacy common geometry is not the historical current-IVM state-0 geometry"
        )

    metrics = compute_article_metrics(full_path, M, grad_F)
    legacy_M = np.asarray(legacy["M_init"], dtype=np.float64)
    legacy_grad_F = np.asarray(legacy["F_init"], dtype=np.float64)
    legacy_metrics = compute_article_metrics(legacy_path, legacy_M, legacy_grad_F)
    generator_path = Path(__file__).resolve()
    current_sha = _sha256(current_ivm_npz)
    legacy_sha = _sha256(legacy_common_npz)
    generator_sha = _sha256(generator_path)

    arrays: Dict[str, Array] = {
        "C1": C1,
        "C2": C2,
        "N": _scalar(n_internal, np.int64),
        "phi_init": phi,
        "full_path_init": full_path,
        "M_init": M,
        "F_init": grad_F,
        "arc_init": arc,
        "count": _scalar(0.0),
        "I_perp_G": _scalar(metrics["I_perp_G"]),
        "I_arc": _scalar(metrics["I_arc"]),
        "Delta_F_barrier": _scalar(metrics["Delta_F_barrier"]),
        "Delta_F_final": _scalar(metrics["Delta_F_final"]),
        "free_energy_profile": np.asarray(
            metrics["free_energy_profile"], dtype=np.float64
        ),
        "perp_G_Mg_init": np.asarray(metrics["perp_G_Mg"], dtype=np.float64),
        "weights_G_init": np.asarray(metrics["weights_G"], dtype=np.float64),
        "metric_eigenvalues_init": np.asarray(
            metrics["metric_eigenvalues"], dtype=np.float64
        ),
        "euclidean_segment_lengths_init": np.asarray(
            metrics["euclidean_segment_lengths"], dtype=np.float64
        ),
        "schema_version": _scalar(SCHEMA_VERSION, np.str_),
        "source_state_index": _scalar(0, np.int64),
        "historical_initial_nof": _scalar(n_internal, np.int64),
        "incremental_nof": _scalar(0, np.int64),
        "mean_force_vector_dimension": _scalar(2, np.int64),
        "source_current_ivm_sha256": _scalar(current_sha, np.str_),
        "source_legacy_common_sha256": _scalar(legacy_sha, np.str_),
        "generator_sha256": _scalar(generator_sha, np.str_),
        "numpy_version": _scalar(np.__version__, np.str_),
        "source_phi_key": _scalar("phi_ref_iter_0", np.str_),
        "source_M_key": _scalar("M_ref_iter_0", np.str_),
        "source_grad_F_key": _scalar("F_ref_iter_0", np.str_),
        "source_arc_key": _scalar("arc_ref_iter_0", np.str_),
        "source_C1_key": _scalar("C1", np.str_),
        "source_C2_key": _scalar("C2", np.str_),
    }
    npz_payload = _deterministic_npz(arrays)
    _atomic_write_bytes(output_npz, npz_payload, overwrite)
    output_sha = _sha256(output_npz)

    provenance: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "file": generator_path.name,
            "sha256": generator_sha,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "byteorder": sys.byteorder,
        },
        "inputs": {
            "historical_current_ivm": {
                "file": current_ivm_npz.name,
                "sha256": current_sha,
            },
            "legacy_common_initial": {
                "file": legacy_common_npz.name,
                "sha256": legacy_sha,
                "role": "fixed endpoints and geometry cross-check only",
            },
        },
        "output": {"file": output_npz.name, "sha256": output_sha},
        "key_mapping": {
            "C1": "legacy_common_initial:C1 (verified against fixed current-IVM endpoint)",
            "C2": "legacy_common_initial:C2 (verified against fixed current-IVM endpoint)",
            "phi_init": "historical_current_ivm:phi_ref_iter_0",
            "full_path_init": "vstack(C1, phi_ref_iter_0, C2)",
            "M_init": "historical_current_ivm:M_ref_iter_0",
            "F_init": "historical_current_ivm:F_ref_iter_0",
            "arc_init": "historical_current_ivm:arc_ref_iter_0",
            "article_metrics_and_profile": (
                "recomputed from full_path_init, M_init, and F_init by "
                "riemannian_torus_geometry.compute_article_metrics"
            ),
        },
        "causal_prefix": {
            "state_0": "phi_ref_iter_0 with M_ref_iter_0 and F_ref_iter_0",
            "state_1": "phi_ref_iter_1 with M_ref_iter_1 and F_ref_iter_1",
            "archived_identity": "path_iter_0 is bitwise equal to phi_ref_iter_1",
            "reason": (
                "The state at incremental NoF 0 is now the exact observation from "
                "which the archived current-IVM update produced state 1; no metrics "
                "from a separate restrained-MD realization are spliced into the prefix."
            ),
        },
        "nof": {
            "name": "number of mean-force evaluations",
            "counting_unit": (
                "one complete two-component mean-force vector evaluated at one 2D "
                "collective-variable image counts as one NoF"
            ),
            "internal_image_count": n_internal,
            "historical_initial_absolute_nof": n_internal,
            "shared_initial_incremental_nof": 0,
            "next_historical_state_incremental_nof": n_internal,
            "explanation": (
                "The archived state 0 required one mean-force evaluation at each of "
                "20 internal images (20 absolute NoF).  Because both methods inherit "
                "that already observed band, the comparison axis starts it at "
                "incremental NoF 0; state 1 adds the next 20 evaluations."
            ),
        },
        "geometry_audit": {
            "legacy_vs_causal_path_max_abs_rad": path_max_abs_difference,
            "legacy_vs_causal_path_rms_rad": path_rms_difference,
            "tolerance_rad": 1.0e-12,
        },
        "legacy_observation_audit": {
            "M_max_abs_difference": float(np.max(np.abs(M - legacy_M))),
            "M_rms_difference": float(np.sqrt(np.mean((M - legacy_M) ** 2))),
            "grad_F_max_abs_difference_kcal_per_mol_rad": float(
                np.max(np.abs(grad_F - legacy_grad_F))
            ),
            "grad_F_rms_difference_kcal_per_mol_rad": float(
                np.sqrt(np.mean((grad_F - legacy_grad_F) ** 2))
            ),
            "old_metrics": {
                "I_perp_G": float(legacy_metrics["I_perp_G"]),
                "I_arc": float(legacy_metrics["I_arc"]),
                "Delta_F_barrier_kcal_per_mol": float(
                    legacy_metrics["Delta_F_barrier"]
                ),
                "Delta_F_final_kcal_per_mol": float(legacy_metrics["Delta_F_final"]),
            },
            "causal_minus_old": {
                "I_perp_G": float(metrics["I_perp_G"] - legacy_metrics["I_perp_G"]),
                "I_arc": float(metrics["I_arc"] - legacy_metrics["I_arc"]),
                "Delta_F_barrier_kcal_per_mol": float(
                    metrics["Delta_F_barrier"] - legacy_metrics["Delta_F_barrier"]
                ),
                "Delta_F_final_kcal_per_mol": float(
                    metrics["Delta_F_final"] - legacy_metrics["Delta_F_final"]
                ),
            },
        },
        "metrics": {
            "I_perp_G": float(metrics["I_perp_G"]),
            "I_arc": float(metrics["I_arc"]),
            "Delta_F_barrier_kcal_per_mol": float(metrics["Delta_F_barrier"]),
            "Delta_F_final_kcal_per_mol": float(metrics["Delta_F_final"]),
        },
    }
    _atomic_write_bytes(provenance_json, _json_bytes(provenance), overwrite)
    return provenance


def _default_paths() -> Dict[str, Path]:
    root = Path(__file__).resolve().parents[1]
    reference = root / "inputs" / "reference"
    return {
        "current": reference / "current_ivm.npz",
        "legacy": reference / "MFEP_INITIAL_PATH_common.npz",
        "output": reference / "MFEP_INITIAL_PATH_current_ivm_state0.npz",
        "provenance": (
            reference / "MFEP_INITIAL_PATH_current_ivm_state0.provenance.json"
        ),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    defaults = _default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-ivm", type=Path, default=defaults["current"])
    parser.add_argument("--legacy-common", type=Path, default=defaults["legacy"])
    parser.add_argument("--output", type=Path, default=defaults["output"])
    parser.add_argument("--provenance", type=Path, default=defaults["provenance"])
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the named new output and sidecar, never either source",
    )
    args = parser.parse_args(argv)
    result = build_causal_shared_initial(
        args.current_ivm,
        args.legacy_common,
        args.output,
        args.provenance,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
