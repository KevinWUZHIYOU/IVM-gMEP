#!/usr/bin/env python3
"""Build the audited NF<=15000 geometric-action cost comparison.

This is a zero-physical-call analysis.  It reads the already stored 32-image
Cartesian paths and full Forces for formal IR-IVM and ODE-NEB states and
evaluates the path-dependent geometric action

    S_hat(phi) = integral_phi |grad V| d ell

with the original 32 images (31 segments) and the segmentwise trapezoidal
rule.  OpenMM's stored Force is ``-grad V``, so its Euclidean norm is used.
No densification, interpolation, smoothing, or mass weighting is applied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


sys.dont_write_bytecode = True

WORKSPACE = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP")
FIGURE_DIR = WORKSPACE / "Figures/barnase_barstar/active_candidates"
OUTPUT_STEM = FIGURE_DIR / "barnase_barstar_discrete32_nf15000_geometric_action_cost"
DATA_JSON = OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_data.json")
PDF = OUTPUT_STEM.with_suffix(".pdf")
PROVENANCE_MD = OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_provenance.md")
AUDIT_JSON = OUTPUT_STEM.with_name(OUTPUT_STEM.name + "_audit.json")

COMPARISON_JSON = FIGURE_DIR / "barnase_barstar_discrete32_nf15000_comparison.json"
ODE_CACHE_JSON = WORKSPACE / (
    "code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/"
    "analysis/formal_discrete32_cache_r2_audited_v2_local_mutagen/"
    "formal_discrete32_cache_through_batch_0501.json"
)
ODE_SOURCE_ROOT = WORKSPACE / (
    "code_on_server/bb_neb_aseode30_kgrid_nf6000_nf30000_20260903_r1"
)
ODE_RUN_DIR = WORKSPACE / (
    "code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/"
    "runs/fresh-nonresume-bb-neb-aseode-k0p1-formal-nf30000-remote-20260903-r1"
)
ODE_FULL_STATE_DIR = ODE_RUN_DIR / "paid_batch_full_state"
REGRESSION_RESULT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
    "runs/diagnostics/bb_ivm_ode_neb_discrete32_geometric_action_20260902_r1.npz"
)

SCHEMA = "bb-discrete32-nf15000-geometric-action-cost-v1"
AUDIT_SCHEMA = "bb-discrete32-nf15000-geometric-action-cost-audit-v1"
FORMULA_ID = "original32-segment-trapezoid-int-norm-gradV-dell-v1"
FORMULA_TEXT = (
    "S_hat=sum_{i=0}^{30} 0.5*(||F_i||_2+||F_{i+1}||_2)"
    "*||x_{i+1}-x_i||_2"
)
EV_TO_KCAL_PER_MOL = 23.0605
N_IMAGES = 32
N_DOF = 9396
ODE_BATCHES = 500
ODE_MAX_NF = 15000
IR_MAX_FORMAL_NF = 8982
IVM_COLOR = "#0072B2"
ODE_COLOR = "#D55E00"
SHA256_ALPHABET = set("0123456789abcdef")


class AnalysisError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AnalysisError(message)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def scalar(archive: Mapping[str, Any], key: str) -> Any:
    require(key in archive, f"archive lacks {key}")
    value = np.asarray(archive[key])
    require(value.shape in {(), (1,)}, f"{key} is not scalar-like")
    return value.reshape(()).item()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )


def geometric_action(path32: Any, force32: Any) -> tuple[float, dict[str, float]]:
    path = np.ascontiguousarray(np.asarray(path32, dtype=np.float64)).reshape(
        N_IMAGES, N_DOF
    )
    force = np.ascontiguousarray(np.asarray(force32, dtype=np.float64)).reshape(
        N_IMAGES, N_DOF
    )
    require(np.isfinite(path).all() and np.isfinite(force).all(),
            "geometric-action state contains a non-finite value")
    segment = np.linalg.norm(np.diff(path, axis=0), axis=1)
    force_norm = np.linalg.norm(force, axis=1)
    require(segment.shape == (31,) and force_norm.shape == (32,),
            "geometric-action topology differs from 32 images / 31 segments")
    require(np.all(segment > 0.0), "geometric-action path has a degenerate segment")
    increments = 0.5 * (force_norm[:-1] + force_norm[1:]) * segment
    value_eV = float(np.sum(increments, dtype=np.float64))
    require(math.isfinite(value_eV) and value_eV > 0.0,
            "geometric action is not finite and positive")
    return value_eV, {
        "arc_length_A": float(np.sum(segment, dtype=np.float64)),
        "minimum_segment_length_A": float(np.min(segment)),
        "maximum_segment_length_A": float(np.max(segment)),
        "minimum_force_norm_eV_per_A": float(np.min(force_norm)),
        "maximum_force_norm_eV_per_A": float(np.max(force_norm)),
    }


def lower_frontier(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    best = math.inf
    result: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: int(value["optimization_N_F"])):
        value = float(row["geometric_action_kcal_per_mol"])
        tolerance = 16.0 * np.finfo(float).eps * max(
            1.0, abs(best) if math.isfinite(best) else 1.0
        )
        if value < best - tolerance:
            result.append(dict(row))
            best = value
    require(result, "strict lower frontier is empty")
    return result


def first_passage(rows: Sequence[Mapping[str, Any]], threshold: float) -> int | None:
    for row in sorted(rows, key=lambda value: int(value["optimization_N_F"])):
        if float(row["geometric_action_kcal_per_mol"]) <= threshold:
            return int(row["optimization_N_F"])
    return None


def import_recorder_module():
    source_text = str(ODE_SOURCE_ROOT)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from src import full_batch_recorder  # type: ignore

    imported = Path(full_batch_recorder.__file__).resolve(strict=True)
    expected = (ODE_SOURCE_ROOT / "src/full_batch_recorder.py").resolve(strict=True)
    require(imported == expected, "noncanonical full-state decoder was imported")
    return full_batch_recorder


def decode_transport_shard(path: Path, previous: tuple[np.ndarray, ...] | None,
                           *, expected_checkpoint: bool, recorder):
    """Decode a Mutagen replica while retaining all frozen decoder checks.

    The canonical decoder additionally requires mode 0444.  Mutagen's local
    transport replica deliberately has mode 0600 while the remote source is
    immutable.  Here file identity is instead bound to the audited cache's
    per-batch artifact SHA-256, and all schema, encoding, and logical-array
    hashes from the canonical decoder are repeated exactly.
    """

    require(path.is_file() and not path.is_symlink(),
            f"full-state transport shard is absent or unsafe: {path}")
    array_keys = {"paid_path32", "energy32", "full_force32", "current_path32"}
    expected_keys = {
        "schema", "batch_id", "optimization_N_F", "spring_k_eV_per_A2",
        "role", "disposition", "encoding", "block_size", "previous_batch_id",
        "current_source_batch_id", *array_keys, "paid_path_sha256",
        "energy32_sha256", "full_force32_sha256", "current_path_sha256",
        "declared_current_state_sha256", "endpoint_cache_sha256",
        "endpoint_values_inherited", "endpoint_optimization_N_F",
        "additional_OpenMM_calls",
    }
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == expected_keys,
                "full-state transport shard key set changed")
        metadata = {
            key: np.asarray(archive[key]).reshape(()).item()
            for key in expected_keys - array_keys
        }
        encoded = tuple(
            np.asarray(archive[key])
            for key in ("paid_path32", "energy32", "full_force32", "current_path32")
        )
    require(str(metadata["schema"]) == str(recorder.SHARD_SCHEMA),
            "full-state transport shard schema changed")
    if expected_checkpoint:
        require(str(metadata["encoding"]) == "full_float64" and previous is None,
                "transport XOR block does not start with a full checkpoint")
        arrays = tuple(np.ascontiguousarray(value, dtype="<f8") for value in encoded)
    else:
        require(str(metadata["encoding"]) == "xor_uint64" and previous is not None,
                "transport noninitial shard is not an XOR delta")
        arrays = tuple(
            np.ascontiguousarray(
                np.bitwise_xor(
                    np.asarray(value, dtype=np.uint64), prior.view(np.uint64)
                ).view("<f8")
            )
            for value, prior in zip(encoded, previous)
        )
    for array, key in zip(
        arrays,
        ("paid_path_sha256", "energy32_sha256", "full_force32_sha256", "current_path_sha256"),
    ):
        require(recorder.array_sha256(array) == str(metadata[key]),
                f"transport bit-exact reconstruction hash failed for {key}")
    return arrays, metadata


def build_ivm_history(reference_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(reference_rows) == 318, "formal IR-IVM reference history is not 318 states")
    result: list[dict[str, Any]] = []
    for index, reference in enumerate(reference_rows, 1):
        source = reference.get("source")
        require(isinstance(source, Mapping), f"IR-IVM row {index} lacks source binding")
        shard = Path(str(source.get("shard", ""))).resolve(strict=True)
        require(sha256_file(shard) == str(source.get("shard_sha256")),
                f"IR-IVM shard hash differs at row {index}")
        with np.load(shard, allow_pickle=False) as saved:
            path = np.asarray(saved["path32_A"], dtype=np.float64)
            energy = np.asarray(saved["energy_eV"], dtype=np.float64)
            force = np.asarray(saved["force_eV_per_A"], dtype=np.float64)
            nf = int(scalar(saved, "optimization_N_F"))
            call = int(scalar(saved, "call_per_image"))
            extra_calls = int(scalar(saved, "additional_physical_calls"))
        require(extra_calls == 0 and nf == int(reference["optimization_N_F"]),
                f"IR-IVM budget/call identity differs at row {index}")
        require(sha256_array(path) == str(reference["path_sha256"])
                and sha256_array(energy) == str(reference["energy_sha256"])
                and sha256_array(force) == str(reference["force_sha256"]),
                f"IR-IVM path/E/Force identity differs at row {index}")
        action_eV, diagnostics = geometric_action(path, force)
        result.append({
            "method": "IVM",
            "optimization_N_F": nf,
            "call_per_image": call,
            "geometric_action_eV": action_eV,
            "geometric_action_kcal_per_mol": action_eV * EV_TO_KCAL_PER_MOL,
            "path_sha256": str(reference["path_sha256"]),
            "energy_sha256": str(reference["energy_sha256"]),
            "force_sha256": str(reference["force_sha256"]),
            "source_shard": str(shard),
            "source_shard_sha256": str(source["shard_sha256"]),
            **diagnostics,
        })
    require(result[-1]["optimization_N_F"] == IR_MAX_FORMAL_NF,
            "formal IR-IVM history does not end at NF8982")
    return result


def build_ode_history(reference_rows: Sequence[Mapping[str, Any]],
                      cache_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    require(len(reference_rows) == ODE_BATCHES and len(cache_rows) >= ODE_BATCHES,
            "ODE-NEB NF15000 reference/cache axis differs from 500 batches")
    recorder = import_recorder_module()
    result: list[dict[str, Any]] = []
    previous_arrays = None
    current = None
    current_action = None
    current_diagnostics = None
    current_source_batch_id = 0
    for batch_id in range(1, ODE_BATCHES + 1):
        reference = reference_rows[batch_id - 1]
        cache = cache_rows[batch_id - 1]
        require(int(reference["optimization_N_F"]) == 30 * batch_id
                and int(cache["optimization_N_F"]) == 30 * batch_id,
                f"ODE-NEB budget axis differs at batch {batch_id}")
        require(str(reference["path_sha256"]) == str(cache["current_path_sha256"])
                and str(reference["energy_sha256"]) == str(cache["current_energy_sha256"])
                and str(reference["force_sha256"]) == str(cache["current_force_sha256"]),
                f"NF30000 prefix is not bit-identical to the published NF15000 run at batch {batch_id}")
        checkpoint = (batch_id - 1) % int(recorder.BLOCK_SIZE) == 0
        if checkpoint:
            previous_arrays = None
        shard = ODE_FULL_STATE_DIR / f"batch_{batch_id:04d}_nf{30 * batch_id:05d}.npz"
        require(sha256_file(shard) == str(cache["source_artifact_sha256"]),
                f"ODE transport shard file hash differs at batch {batch_id}")
        arrays, metadata = decode_transport_shard(
            shard, previous_arrays, expected_checkpoint=checkpoint, recorder=recorder
        )
        previous_arrays = arrays
        paid_path, energy, force, current_path = arrays
        disposition = str(metadata["disposition"])
        if disposition in {"current_initial", "accepted"}:
            current = (paid_path, energy, force)
            current_action, current_diagnostics = geometric_action(paid_path, force)
            current_source_batch_id = batch_id
        else:
            require(disposition == "rejected", f"unknown ODE disposition at batch {batch_id}")
        require(current is not None and current_action is not None
                and current_diagnostics is not None,
                f"ODE current state is unavailable at batch {batch_id}")
        require(np.array_equal(current_path, current[0])
                and int(metadata["current_source_batch_id"]) == current_source_batch_id
                and int(cache["current_source_batch_id"]) == current_source_batch_id,
                f"ODE accepted/rejected current-state mapping differs at batch {batch_id}")
        require(sha256_array(current[0]) == str(reference["path_sha256"])
                and sha256_array(current[1]) == str(reference["energy_sha256"])
                and sha256_array(current[2]) == str(reference["force_sha256"]),
                f"decoded ODE current path/E/Force differs at batch {batch_id}")
        source_shard = ODE_FULL_STATE_DIR / (
            f"batch_{current_source_batch_id:04d}_nf{30 * current_source_batch_id:05d}.npz"
        )
        result.append({
            "method": "ODE-NEB",
            "optimization_N_F": 30 * batch_id,
            "batch_id": batch_id,
            "disposition": disposition,
            "current_source_batch_id": current_source_batch_id,
            "geometric_action_eV": current_action,
            "geometric_action_kcal_per_mol": current_action * EV_TO_KCAL_PER_MOL,
            "path_sha256": str(reference["path_sha256"]),
            "energy_sha256": str(reference["energy_sha256"]),
            "force_sha256": str(reference["force_sha256"]),
            "current_source_shard": str(source_shard),
            "current_source_shard_sha256": sha256_file(source_shard),
            **current_diagnostics,
        })
    require(result[-1]["optimization_N_F"] == ODE_MAX_NF,
            "ODE-NEB comparison does not end at NF15000")
    return result


def regression_check(histories: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    require(REGRESSION_RESULT.is_file(), "independent NF8982/NF9990 action result is absent")
    with np.load(REGRESSION_RESULT, allow_pickle=False) as saved:
        methods = [str(value) for value in np.asarray(saved["method"]).tolist()]
        nfs = [int(value) for value in np.asarray(saved["optimization_N_F"]).tolist()]
        values = [float(value) for value in np.asarray(
            saved["total_path_dependent_geometric_action_kcal_per_mol"]
        ).tolist()]
    expected = dict(zip(zip(methods, nfs), values))
    checks = {}
    for method, nf in (("IVM", 8982), ("ODE-NEB", 9990)):
        row = next(value for value in histories[method]
                   if int(value["optimization_N_F"]) == nf)
        measured = float(row["geometric_action_kcal_per_mol"])
        reference = float(expected[(method, nf)])
        difference = abs(measured - reference)
        tolerance = 1.0e-10 * max(1.0, abs(reference))
        require(difference <= tolerance,
                f"{method} NF{nf} action differs from independent result")
        checks[f"{method}_NF{nf}"] = {
            "measured_kcal_per_mol": measured,
            "independent_reference_kcal_per_mol": reference,
            "absolute_difference_kcal_per_mol": difference,
            "tolerance_kcal_per_mol": tolerance,
        }
    return {
        "source": str(REGRESSION_RESULT),
        "source_sha256": sha256_file(REGRESSION_RESULT),
        "checks": checks,
    }


def compute_data() -> dict[str, Any]:
    comparison = load_json(COMPARISON_JSON)
    histories = comparison.get("histories")
    require(isinstance(histories, Mapping)
            and set(histories) == {"IVM", "ODE-NEB", "Previous IVM"},
            "published NF15000 comparison method set differs")
    cache = load_json(ODE_CACHE_JSON)
    require(cache.get("status") == "PARTIAL_NOT_FINAL"
            and int(cache.get("processed_batches", -1)) >= ODE_BATCHES
            and int(cache.get("target_optimization_N_F", -1)) == 30000
            and cache.get("physical_calls_started") == 0,
            "NF30000 prefix cache is not a compatible zero-call audited cache")
    ivm = build_ivm_history(histories["IVM"])
    ode = build_ode_history(histories["ODE-NEB"], cache["rows"])
    methods = {"IVM": ivm, "ODE-NEB": ode}
    frontiers = {method: lower_frontier(rows) for method, rows in methods.items()}
    threshold = max(
        min(float(row["geometric_action_kcal_per_mol"]) for row in ivm),
        min(float(row["geometric_action_kcal_per_mol"]) for row in ode),
    )
    ivm_nf = first_passage(ivm, threshold)
    ode_nf = first_passage(ode, threshold)
    require(ivm_nf is not None and ode_nf is not None,
            "common geometric-action threshold has no first passage")
    regression = regression_check(methods)
    output = {
        "schema": SCHEMA,
        "status": "PASS_ZERO_PHYSICAL_CALLS",
        "created_utc": utc_now(),
        "quantity": {
            "name": "fixed-endpoint path-dependent geometric action",
            "latex": r"\widehat{S}(\varphi)=\int_{\varphi}|\nabla V|\,\mathrm{d}\ell",
            "formula_id": FORMULA_ID,
            "discrete_formula": FORMULA_TEXT,
            "force_identity": "stored OpenMM Force = -grad V; Euclidean norms are equal",
            "coordinate_metric": "unweighted Cartesian Euclidean",
            "number_of_images": 32,
            "number_of_segments": 31,
            "densification": False,
            "interpolation": False,
            "smoothing": False,
            "mass_weighting": False,
            "units_before_conversion": "eV",
            "units_plotted": "kcal mol^-1",
            "eV_to_kcal_per_mol": EV_TO_KCAL_PER_MOL,
        },
        "cost_protocol": {
            "vertical_axis": "optimization Number of Forces",
            "fixed_endpoint_optimization_N_F": 0,
            "additional_physical_calls": 0,
            "ir_ivm_formal_history_end_N_F": IR_MAX_FORMAL_NF,
            "ir_ivm_terminal_uncommitted_tail_included": False,
            "ode_neb_history_end_N_F": ODE_MAX_NF,
            "stored_state_selection": "current accepted path after each paid batch; rejected trials retain the preceding current path",
        },
        "source_bindings": {
            "published_nf15000_comparison": {
                "path": str(COMPARISON_JSON), "sha256": sha256_file(COMPARISON_JSON)
            },
            "audited_nf30000_prefix_cache": {
                "path": str(ODE_CACHE_JSON), "sha256": sha256_file(ODE_CACHE_JSON),
                "note": "its first 500 path/E/Force hashes are all bit-identical to the sealed NF15000 comparison",
            },
            "ode_full_state_directory": str(ODE_FULL_STATE_DIR),
            "ode_decoder": {
                "path": str(ODE_SOURCE_ROOT / "src/full_batch_recorder.py"),
                "sha256": sha256_file(ODE_SOURCE_ROOT / "src/full_batch_recorder.py"),
            },
        },
        "identity_checks": {
            "ode_nf30000_prefix_rows_checked_against_published_nf15000": 500,
            "ode_path_hash_matches": 500,
            "ode_energy_hash_matches": 500,
            "ode_force_hash_matches": 500,
            "ir_rows_checked_against_published_nf15000": 318,
            "regression": regression,
        },
        "histories": methods,
        "strict_lower_frontiers": frontiers,
        "common_threshold": {
            "definition": "larger of the two best recorded actions",
            "geometric_action_kcal_per_mol": threshold,
            "IVM_first_passage_N_F": ivm_nf,
            "ODE-NEB_first_passage_N_F": ode_nf,
            "Number_of_Forces_ratio": max(ivm_nf, ode_nf) / min(ivm_nf, ode_nf),
        },
        "summary": {
            method: {
                "states": len(rows),
                "last_recorded_N_F": int(rows[-1]["optimization_N_F"]),
                "last_recorded_action_kcal_per_mol": float(
                    rows[-1]["geometric_action_kcal_per_mol"]
                ),
                "best_recorded_action_kcal_per_mol": min(
                    float(row["geometric_action_kcal_per_mol"]) for row in rows
                ),
                "strict_lower_frontier_points": len(frontiers[method]),
            }
            for method, rows in methods.items()
        },
    }
    return output


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.4,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def force_formatter(value: float, _position: int) -> str:
    return f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"


def plot_data(data: Mapping[str, Any]) -> plt.Figure:
    require(data.get("schema") == SCHEMA and data.get("status") == "PASS_ZERO_PHYSICAL_CALLS",
            "geometric-action data pack is not complete")
    configure_style()
    figure, axis = plt.subplots(figsize=(4.10, 2.75))
    styles = {
        "IVM": dict(color=IVM_COLOR, marker="o", linestyle="-", linewidth=1.85),
        "ODE-NEB": dict(color=ODE_COLOR, marker="s", linestyle="--", linewidth=1.65),
    }
    plotted_values: list[float] = []
    for method in ("IVM", "ODE-NEB"):
        rows = data["strict_lower_frontiers"][method]
        x = [float(row["geometric_action_kcal_per_mol"]) for row in rows]
        y = [int(row["optimization_N_F"]) for row in rows]
        plotted_values.extend(x)
        axis.plot(x, y, label=method, markersize=3.0,
                  markevery=max(1, len(rows) // 10), **styles[method])
    threshold_record = data["common_threshold"]
    threshold = float(threshold_record["geometric_action_kcal_per_mol"])
    ivm_nf = int(threshold_record["IVM_first_passage_N_F"])
    ode_nf = int(threshold_record["ODE-NEB_first_passage_N_F"])
    if ivm_nf != ode_nf:
        low, high = sorted((ivm_nf, ode_nf))
        axis.annotate("", xy=(threshold, high), xytext=(threshold, low),
                      arrowprops={"arrowstyle": "<->", "color": "#202020", "lw": 0.8})
        axis.annotate(
            f"{float(threshold_record['Number_of_Forces_ratio']):.1f}×",
            xy=(threshold, 0.5 * (low + high)), xytext=(6, 0),
            textcoords="offset points", va="center", ha="left",
            fontsize=7.3, color="#202020",
        )
    require(min(plotted_values) > 0.0, "logarithmic action axis requires positive values")
    low_limit = min(plotted_values) / 1.18
    high_limit = max(plotted_values) * 1.18
    axis.set_xscale("log")
    axis.set_xlim(high_limit, low_limit)
    axis.set_ylim(0, ODE_MAX_NF)
    axis.set_xlabel(r"Geometric action $\widehat{S}(\varphi)$ (kcal mol$^{-1}$)")
    axis.set_ylabel("Number of Forces")
    axis.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(force_formatter))
    axis.grid(color="#D0D0D0", linewidth=0.45, alpha=0.55)
    axis.legend(frameon=False, loc="upper center", ncol=2,
                bbox_to_anchor=(0.5, 1.17))
    figure.subplots_adjust(left=0.17, right=0.97, bottom=0.20, top=0.82)
    return figure


def build_plot() -> None:
    data = load_json(DATA_JSON)
    figure = plot_data(data)
    PDF.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PDF, bbox_inches="tight", metadata={
        "Title": "Barnase--Barstar geometric-action cost comparison",
        "Subject": "Discrete 32-image geometric action versus Number of Forces",
        "Author": "IVM for MEP analysis",
        "Creator": "Matplotlib",
    })
    plt.close(figure)
    require(PDF.is_file() and PDF.stat().st_size > 10_000, "PDF output is absent or implausibly small")
    audit = {
        "schema": AUDIT_SCHEMA,
        "status": "PASS_PENDING_VISUAL_CONFIRMATION",
        "created_utc": utc_now(),
        "script": {"path": str(Path(__file__).resolve()),
                   "sha256": sha256_file(Path(__file__).resolve())},
        "data": {"path": str(DATA_JSON), "sha256": sha256_file(DATA_JSON)},
        "pdf": {"path": str(PDF), "sha256": sha256_file(PDF), "bytes": PDF.stat().st_size},
        "structural_checks": {
            "methods": ["IVM", "ODE-NEB"],
            "x_scale": "logarithmic descending (worse to better)",
            "y_scale": "linear 0 to 15000",
            "additional_physical_calls": 0,
            "densification": False,
        },
    }
    atomic_json(AUDIT_JSON, audit)
    summary = data["summary"]
    threshold = data["common_threshold"]
    provenance = f"""# Barnase--Barstar NF15000 geometric-action cost figure

Status: `PASS_PENDING_VISUAL_CONFIRMATION`

## Output

- PDF: `{PDF}`
- PDF SHA-256: `{sha256_file(PDF)}`
- Data: `{DATA_JSON}`
- Data SHA-256: `{sha256_file(DATA_JSON)}`
- Plot/analysis code: `{Path(__file__).resolve()}`
- Code SHA-256: `{sha256_file(Path(__file__).resolve())}`

## Quantity and discretization

The plotted horizontal coordinate is the path-dependent geometric action
`S_hat(phi) = integral_phi |grad V| d ell`.  Since the stored OpenMM Force is
`-grad V`, the calculation uses its Euclidean norm.  On the original 32
images it is evaluated over the 31 direct Cartesian segments by
`{FORMULA_TEXT}`.  There is no densification, interpolation, smoothing, or
mass weighting.  The native `eV` result is multiplied by
`{EV_TO_KCAL_PER_MOL}` to give `kcal mol^-1`.

## Cost and state semantics

- Fixed endpoints are inherited and contribute `0` optimization Number of Forces.
- This analysis makes `0` new physical calls and contributes `0` Number of Forces.
- IVM includes the 318 formally reportable stored states through the last
  complete reparameterization commit at NF `{IR_MAX_FORMAL_NF}`.  The charged
  terminal uncommitted tail through NF 14983 is not presented as an IVM state.
- ODE-NEB includes the current accepted path after every paid batch through
  NF `{ODE_MAX_NF}`.  A rejected paid trial consumes its 30 Forces but retains
  the previous current path.
- The NF30000 rerun prefix is used only as a storage replica: all 500 path,
  energy, and Force hashes through NF15000 are bit-identical to the sealed
  published NF15000 comparison.

## Numerical summary

- IVM: last NF `{summary['IVM']['last_recorded_N_F']}`, last action
  `{summary['IVM']['last_recorded_action_kcal_per_mol']:.12g}` kcal mol^-1,
  best recorded `{summary['IVM']['best_recorded_action_kcal_per_mol']:.12g}`.
- ODE-NEB: last NF `{summary['ODE-NEB']['last_recorded_N_F']}`, last action
  `{summary['ODE-NEB']['last_recorded_action_kcal_per_mol']:.12g}` kcal mol^-1,
  best recorded `{summary['ODE-NEB']['best_recorded_action_kcal_per_mol']:.12g}`.
- Common threshold: `{threshold['geometric_action_kcal_per_mol']:.12g}` kcal mol^-1;
  first reached by IVM at NF `{threshold['IVM_first_passage_N_F']}` and by
  ODE-NEB at NF `{threshold['ODE-NEB_first_passage_N_F']}`, a
  `{threshold['Number_of_Forces_ratio']:.6g}`-fold Number-of-Forces ratio.

## Source chain

- Published NF15000 comparison: `{COMPARISON_JSON}`
  (`{sha256_file(COMPARISON_JSON)}`).
- Audited NF30000 prefix cache: `{ODE_CACHE_JSON}`
  (`{sha256_file(ODE_CACHE_JSON)}`).
- ODE full-state source: `{ODE_FULL_STATE_DIR}`.
- Each of the 318 IR-IVM source shard paths and hashes and each ODE current
  source batch path/hash are recorded row-by-row in the data JSON.
- Independent regression checks at IVM NF8982 and ODE-NEB NF9990 are recorded
  in the data JSON and agree within the declared floating-point tolerance.
"""
    atomic_text(PROVENANCE_MD, provenance)
    print(json.dumps({
        "pdf": str(PDF), "pdf_sha256": sha256_file(PDF),
        "data": str(DATA_JSON), "audit": str(AUDIT_JSON),
        "provenance": str(PROVENANCE_MD),
        "summary": summary, "common_threshold": threshold,
    }, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("compute", "plot", "verify"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "compute":
        data = compute_data()
        atomic_json(DATA_JSON, data)
        print(json.dumps({
            "data": str(DATA_JSON), "sha256": sha256_file(DATA_JSON),
            "summary": data["summary"], "common_threshold": data["common_threshold"],
            "identity_checks": data["identity_checks"],
        }, indent=2, sort_keys=True))
    elif args.mode == "plot":
        build_plot()
    else:
        data = load_json(DATA_JSON)
        require(data.get("schema") == SCHEMA
                and data.get("status") == "PASS_ZERO_PHYSICAL_CALLS",
                "data-pack schema/status differs")
        require(data["cost_protocol"]["additional_physical_calls"] == 0,
                "data pack reports nonzero physical calls")
        require(len(data["histories"]["IVM"]) == 318
                and len(data["histories"]["ODE-NEB"]) == 500,
                "data-pack method history lengths differ")
        require(data["histories"]["IVM"][-1]["optimization_N_F"] == 8982
                and data["histories"]["ODE-NEB"][-1]["optimization_N_F"] == 15000,
                "data-pack terminal budgets differ")
        print("PASS: audited NF15000 geometric-action data pack")


if __name__ == "__main__":
    main()
