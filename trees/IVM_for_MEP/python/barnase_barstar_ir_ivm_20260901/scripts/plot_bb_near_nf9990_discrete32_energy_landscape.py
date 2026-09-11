#!/usr/bin/env python3
"""Plot the three near-NF9990 Barnase--Barstar paths from 32 images only.

Every displayed vertex is one of the original 32 path images.  The abscissa is
the cumulative Cartesian arc length of those same images and adjacent vertices
are joined by straight plotting segments.  This script neither densifies nor
interpolates a path and performs no physical energy/force evaluation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


EV_TO_KCAL_PER_MOL = 23.0605
N_IMAGES = 32
N_ATOMS = 3132
DIMENSION = N_ATOMS * 3
EXPECTED_IR_PATH_SHA256 = "46f54b316a6138d294150c810c14b5eedb14921a4a02717f61239cce9b7965bb"
EXPECTED_ENDPOINT_FILE_SHA256 = "5cd4e757c99f0784d8b0781d05d8083070c8cfaad2670ead26999b3a4134e99a"
EXPECTED_ENDPOINT_COORDINATE_SHA256 = "5788e22f86df91ed5500e88a8dc053f7408bd2fb39414de121a6eec36b833db1"
EXPECTED_ENDPOINT_ENERGY_SHA256 = "24884695bc3bbecbf1e5007bf8f245bfe6d7eb5bf81725e986b8dd0ece088a31"
EXPECTED_ODE_FILE_SHA256 = "0f8c84b0e11fa191db9163c13fca8f10504aa685952a4c728b4043b5eba40ae0"
EXPECTED_PREVIOUS_IVM_FILE_SHA256 = "80f01ca006da806dda8fa67c833133cac9654514e7bf25d62cb9992e7e2a1db0"

ROOT = Path("/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901")
ENDPOINT_CACHE = ROOT / "inputs/endpoint_joint_ef_cache.npz"
ODE_SOURCE = ROOT / (
    "inputs/baselines/ode_neb_source_paths_nf240_480_990_9990/"
    "nf09990_batch_0333_EF30.npz"
)
PREVIOUS_IVM_SOURCE = ROOT / (
    "inputs/baselines/original_ivm_source_snapshots_nf30_240_480_990_9990/"
    "nf09990.npz"
)

IR_COLOR = "#0072B2"
ODE_COLOR = "#D55E00"
PREVIOUS_IVM_COLOR = "#6A6A6A"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def scalar(saved, key: str):
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} must be scalar")
    return value.item()


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.1,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_endpoints() -> tuple[np.ndarray, np.ndarray, dict]:
    require(sha256_file(ENDPOINT_CACHE) == EXPECTED_ENDPOINT_FILE_SHA256, "endpoint-cache file drift")
    with np.load(ENDPOINT_CACHE, allow_pickle=False) as saved:
        coordinates = np.asarray(saved["coordinates_A"], dtype=np.float64)
        energies = np.asarray(saved["energies_eV"], dtype=np.float64)
    require(coordinates.shape == (2, DIMENSION), "endpoint coordinate shape drift")
    require(energies.shape == (2,), "endpoint energy shape drift")
    require(np.all(np.isfinite(coordinates)) and np.all(np.isfinite(energies)), "bad endpoint cache")
    require(sha256_array(coordinates) == EXPECTED_ENDPOINT_COORDINATE_SHA256, "endpoint coordinate drift")
    require(sha256_array(energies) == EXPECTED_ENDPOINT_ENERGY_SHA256, "endpoint energy drift")
    return coordinates, energies, {
        "path": str(ENDPOINT_CACHE.resolve(strict=True)),
        "sha256": sha256_file(ENDPOINT_CACHE),
        "coordinate_sha256": sha256_array(coordinates),
        "energy_sha256": sha256_array(energies),
    }


def load_ir(path: Path, endpoint_coordinates: np.ndarray, endpoint_energies: np.ndarray) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    audit_path = resolved.with_suffix(".audit.json")
    require(audit_path.exists(), "IR energy result lacks its PASS audit")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    require(audit.get("status") == "PASS", "IR energy audit is not PASS")
    result_binding = audit.get("result", {})
    if isinstance(result_binding, dict):
        require(result_binding.get("path") == str(resolved), "IR audit/result path mismatch")
        require(result_binding.get("sha256") == sha256_file(resolved), "IR audit/result hash mismatch")
    else:
        require(result_binding == str(resolved), "IR audit/result path mismatch")
        require(audit.get("result_sha256") == sha256_file(resolved), "IR audit/result hash mismatch")
    with np.load(resolved, allow_pickle=False) as saved:
        require("original_path32_A" in saved.files, "IR energy result lacks original_path32_A")
        require("original_path32_energy_eV" in saved.files, "IR energy result lacks original_path32_energy_eV")
        coordinates = np.asarray(saved["original_path32_A"], dtype=np.float64).reshape(N_IMAGES, DIMENSION)
        energies = np.asarray(saved["original_path32_energy_eV"], dtype=np.float64)
        source_nf = int(scalar(saved, "center_source_optimization_N_F"))
        source_call = int(scalar(saved, "center_source_call_per_image"))
        embedded_path_hash = str(scalar(saved, "original_path32_sha256"))
        embedded_energy_hash = str(scalar(saved, "original_path32_energy_sha256"))
        require(str(scalar(saved, "status")) == "PASS", "IR energy result is not PASS")
    require(source_nf == 8982 and source_call == 318, "IR selected-state identity drift")
    require(energies.shape == (N_IMAGES,), "IR energy shape drift")
    require(np.all(np.isfinite(coordinates)) and np.all(np.isfinite(energies)), "bad IR energy result")
    require(np.array_equal(coordinates[[0, -1]], endpoint_coordinates), "IR endpoints differ from cache")
    require(np.array_equal(energies[[0, -1]], endpoint_energies), "IR endpoint energies differ from cache")
    require(embedded_path_hash == EXPECTED_IR_PATH_SHA256, "IR NF8982 path is not the frozen selected path")
    require(sha256_array(coordinates) == embedded_path_hash, "IR path hash mismatch")
    require(sha256_array(energies) == embedded_energy_hash, "IR energy hash mismatch")
    curve = make_curve("IVM", source_nf, coordinates, energies, resolved, embedded_path_hash, embedded_energy_hash)
    curve["source_audit"] = str(audit_path)
    curve["source_audit_sha256"] = sha256_file(audit_path)
    return curve


def load_ode(endpoint_coordinates: np.ndarray, endpoint_energies: np.ndarray) -> dict:
    require(sha256_file(ODE_SOURCE) == EXPECTED_ODE_FILE_SHA256, "ODE--NEB source file drift")
    with np.load(ODE_SOURCE, allow_pickle=False) as saved:
        internal_coordinates = np.asarray(saved["coordinates30_A"], dtype=np.float64).reshape(30, DIMENSION)
        internal_energies = np.asarray(saved["energies_eV"], dtype=np.float64)
        nf = int(scalar(saved, "nf_after"))
    coordinates = np.concatenate((endpoint_coordinates[:1], internal_coordinates, endpoint_coordinates[1:]), axis=0)
    energies = np.concatenate((endpoint_energies[:1], internal_energies, endpoint_energies[1:]))
    require(nf == 9990, "ODE--NEB source NF drift")
    return make_curve("ODE-NEB", nf, coordinates, energies, ODE_SOURCE.resolve(strict=True))


def load_previous_ivm(endpoint_coordinates: np.ndarray, endpoint_energies: np.ndarray) -> dict:
    require(sha256_file(PREVIOUS_IVM_SOURCE) == EXPECTED_PREVIOUS_IVM_FILE_SHA256, "previous IVM source file drift")
    with np.load(PREVIOUS_IVM_SOURCE, allow_pickle=False) as saved:
        coordinates = np.asarray(saved["nodes"], dtype=np.float64).reshape(N_IMAGES, DIMENSION)
        internal_energies = np.asarray(saved["current_energies_eV"], dtype=np.float64)
        nf_array = np.asarray(saved["nf"], dtype=np.int64).reshape(-1)
    require(nf_array.size == 1 and int(nf_array[0]) == 9990, "previous IVM source NF drift")
    require(np.array_equal(coordinates[[0, -1]], endpoint_coordinates), "previous IVM endpoints differ from cache")
    energies = np.concatenate((endpoint_energies[:1], internal_energies, endpoint_energies[1:]))
    return make_curve("Previous IVM", 9990, coordinates, energies, PREVIOUS_IVM_SOURCE.resolve(strict=True))


def make_curve(
    method: str,
    nf: int,
    coordinates: np.ndarray,
    energies: np.ndarray,
    source: Path,
    path_hash: str | None = None,
    energy_hash: str | None = None,
) -> dict:
    band = np.ascontiguousarray(np.asarray(coordinates, dtype=np.float64).reshape(N_IMAGES, DIMENSION))
    values = np.ascontiguousarray(np.asarray(energies, dtype=np.float64))
    require(values.shape == (N_IMAGES,), f"{method}: energy shape drift")
    require(np.all(np.isfinite(band)) and np.all(np.isfinite(values)), f"{method}: non-finite data")
    segment_lengths = np.linalg.norm(np.diff(band, axis=0), axis=1)
    require(np.all(segment_lengths > 0.0), f"{method}: degenerate segment")
    arc = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    relative = (values - values[0]) * EV_TO_KCAL_PER_MOL
    return {
        "method": method,
        "optimization_N_F": int(nf),
        "coordinates": band,
        "energy_eV": values,
        "arc_length_A": arc,
        "relative_energy_kcal_per_mol": relative,
        "discrete_Delta_E_kcal_per_mol": float(np.max(relative)),
        "peak_image_index": int(np.argmax(relative)),
        "source": str(source),
        "source_sha256": sha256_file(source),
        "path_sha256": path_hash or sha256_array(band),
        "energy_sha256": energy_hash or sha256_array(values),
    }


def save_atomic(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + f".tmp.{os.getpid()}" + output.suffix)
    fig.savefig(
        temporary,
        bbox_inches="tight",
        metadata={
            "Title": "Barnase--Barstar near-NF9990 discrete 32-image energy profiles",
            "Subject": "Original 32 images joined directly; no densification or interpolation",
        },
    )
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir-energy-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()

    configure_style()
    endpoint_coordinates, endpoint_energies, endpoint_record = load_endpoints()
    curves = [
        load_ir(args.ir_energy_result, endpoint_coordinates, endpoint_energies),
        load_ode(endpoint_coordinates, endpoint_energies),
        load_previous_ivm(endpoint_coordinates, endpoint_energies),
    ]

    styles = {
        "IVM": {"color": IR_COLOR, "linestyle": "-", "marker": "s", "zorder": 4},
        "ODE-NEB": {"color": ODE_COLOR, "linestyle": "-", "marker": "o", "zorder": 3},
        "Previous IVM": {"color": PREVIOUS_IVM_COLOR, "linestyle": "--", "marker": "^", "zorder": 2},
    }
    fig, axis = plt.subplots(figsize=(4.15, 2.85))
    for curve in curves:
        style = styles[curve["method"]]
        axis.plot(
            curve["arc_length_A"],
            curve["relative_energy_kcal_per_mol"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.45,
            marker=style["marker"],
            markersize=3.0,
            markeredgewidth=0.45,
            markeredgecolor="white",
            label=f"{curve['method']}  ($N_F={curve['optimization_N_F']}$)",
            zorder=style["zorder"],
        )
    axis.set_xlabel(r"Cartesian arc length $s$ ($\AA$)")
    axis.set_ylabel(r"$E(s)-E(0)$ (kcal mol$^{-1}$)")
    axis.set_title("Discrete 32-image energy profiles", pad=5)
    axis.set_xlim(0.0, max(float(curve["arc_length_A"][-1]) for curve in curves))
    axis.set_ylim(bottom=0.0)
    axis.grid(True, color="#D8D8D8", linewidth=0.45, alpha=0.72)
    axis.legend(frameon=False, loc="upper right", handlelength=2.3)
    axis.tick_params(direction="out", length=3)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.19, top=0.90)

    output = args.output.expanduser().resolve()
    provenance = args.provenance.expanduser().resolve()
    save_atomic(fig, output)
    plt.close(fig)

    record = {
        "schema": "bb-near-nf9990-discrete32-energy-landscape-provenance-v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure": {"path": str(output), "sha256": sha256_file(output)},
        "plot_script": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
        "endpoint_cache": endpoint_record,
        "sampling": {
            "number_of_images_per_path": N_IMAGES,
            "abscissa": "cumulative Cartesian arc length over the same 32 images",
            "connection": "adjacent original images joined by straight plotting segments",
            "densification": "none",
            "interpolation": "none",
            "physical_calls_by_plotter": 0,
        },
        "curves": [],
    }
    for curve in curves:
        curve_record = {
            key: value
            for key, value in curve.items()
            if key not in {"coordinates", "energy_eV", "arc_length_A", "relative_energy_kcal_per_mol"}
        }
        curve_record.update(
            {
                "total_discrete_arc_length_A": float(curve["arc_length_A"][-1]),
                "arc_length_sha256": sha256_array(curve["arc_length_A"]),
                "relative_energy_sha256": sha256_array(curve["relative_energy_kcal_per_mol"]),
            }
        )
        record["curves"].append(curve_record)
    provenance.parent.mkdir(parents=True, exist_ok=True)
    temporary = provenance.with_name(provenance.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, provenance)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
