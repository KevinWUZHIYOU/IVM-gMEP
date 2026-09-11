#!/usr/bin/env python3
"""Plot BB path evolution on the new IR-IVM-NF8982-centered 0 K section.

The legacy evolution PDF supplies visual language only.  In final mode the
background must be the hash-bound R4 59x28 surface result, while all IVM and
ODE-NEB curves are loaded from the already verified stored-path sources.
This script never imports or constructs an OpenMM context.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
FIGURE_SCHEMA = "bb-irivm-nf8982-surface-evolution-figure-contract-v1"
FIGURE_STATUS = "FINAL_INPUTS_FROZEN_AFTER_R4_SURFACE_PASS"
PROVENANCE_SCHEMA = "bb-irivm-nf8982-surface-evolution-provenance-v1"
IVM_COLOR = "#0072B2"
ODE_COLOR = "#D55E00"
INITIAL_COLOR = "#7B3294"
EV_TO_KCAL = 23.060547830619
EXPECTED_PANELS = ((210, 240), (429, 480), (8982, 9990))
N_IMAGES = 32
N_ATOMS = 3132
POINTS = 59
NORMAL_POINTS = 28
ZERO_INDEX = 4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def file_binding(path: Path) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def resolve_binding(contract_path: Path, record: Mapping[str, object], role: str) -> Path:
    require(isinstance(record, Mapping), f"{role} binding is not an object")
    path = Path(str(record.get("path", ""))).expanduser()
    if not path.is_absolute():
        path = contract_path.parent / path
    path = path.resolve(strict=True)
    require(sha256_file(path) == str(record.get("sha256", "")), f"{role} SHA-256 mismatch")
    return path


def import_path(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class LogQuadraticNorm(Normalize):
    def __init__(self, vmin: float, vmax: float, scale: float = 40.0, curvature: float = 2.5):
        super().__init__(vmin=vmin, vmax=vmax, clip=True)
        self.scale = float(scale)
        self.curvature = float(curvature)
        self._span = np.log1p((self.vmax - self.vmin) / self.scale)

    def __call__(self, value, clip=None):
        result, scalar = self.process_value(value)
        raw = np.clip((result.data - self.vmin) / self.scale, 0.0, (self.vmax - self.vmin) / self.scale)
        logarithmic = np.log1p(raw) / self._span
        mapped = 1.0 - np.power(1.0 - logarithmic, self.curvature)
        output = np.ma.array(mapped, mask=result.mask, copy=False)
        return output[0] if scalar else output

    def inverse(self, value):
        mapped = np.asarray(value, dtype=float)
        logarithmic = 1.0 - np.power(np.clip(1.0 - mapped, 0.0, 1.0), 1.0 / self.curvature)
        return self.vmin + self.scale * np.expm1(logarithmic * self._span)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 8.3,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def arc_coordinate(path: np.ndarray) -> np.ndarray:
    flat = np.asarray(path, dtype=np.float64).reshape(len(path), -1)
    lengths = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    require(np.all(np.isfinite(lengths)) and np.all(lengths > 0.0), "path contains a degenerate segment")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return cumulative / cumulative[-1]


def resample(path: np.ndarray, target_s: np.ndarray) -> np.ndarray:
    band = np.asarray(path, dtype=np.float64).reshape(len(path), -1)
    source_s = arc_coordinate(band)
    result = np.stack(
        [np.interp(target_s, source_s, band[:, column]) for column in range(band.shape[1])],
        axis=1,
    )
    return result.reshape(len(target_s), *np.asarray(path).shape[1:])


def project_band(path: np.ndarray, center: np.ndarray, center_s: np.ndarray, normals: np.ndarray) -> np.ndarray:
    band = np.asarray(path, dtype=np.float64)
    require(band.shape == (N_IMAGES, N_ATOMS, 3), "projected path must be 32x3132x3")
    matched = resample(band, center_s)
    n_value = np.sum((matched - center) * normals, axis=(1, 2)) / np.sqrt(float(N_ATOMS))
    n_value[[0, -1]] = 0.0
    require(np.all(np.isfinite(n_value)), "path projection is non-finite")
    return np.column_stack((center_s, n_value))


def _load_surface(contract_path: Path, contract: dict) -> dict:
    evaluator_path = resolve_binding(contract_path, contract["surface_evaluator"], "R4 surface evaluator")
    evaluator = import_path(evaluator_path, "bb_ir_surface_plot_validation")
    execution_contract = resolve_binding(contract_path, contract["surface_execution_contract"], "surface execution contract")
    authorization = resolve_binding(contract_path, contract["surface_authorization"], "surface authorization")
    result = resolve_binding(contract_path, contract["surface_result"], "surface result")
    audit = resolve_binding(contract_path, contract["surface_audit"], "surface audit")
    wal = resolve_binding(contract_path, contract["surface_wal"], "surface WAL")
    state = evaluator.load_contract(execution_contract, authorization, "local", result, evaluator.SURFACE_STAGE)
    expected_result, expected_audit, expected_wal = evaluator.stage_artifacts(state, evaluator.SURFACE_STAGE)
    require((result, audit, wal) == (expected_result, expected_audit, expected_wal), "surface artifact paths differ from execution contract")
    path_result = resolve_binding(contract_path, contract["original_path32_result"], "original-path result")
    path_audit = resolve_binding(contract_path, contract["original_path32_audit"], "original-path audit")
    path_wal = resolve_binding(contract_path, contract["original_path32_wal"], "original-path WAL")
    expected_path_result, expected_path_audit, expected_path_wal = evaluator.stage_artifacts(state, evaluator.PATH_STAGE)
    require(
        (path_result, path_audit, path_wal)
        == (expected_path_result, expected_path_audit, expected_path_wal),
        "original-path artifact paths differ from execution contract",
    )
    _, wal_state = evaluator.read_wal(wal, evaluator.expected_wal_header(state, evaluator.SURFACE_STAGE))
    evaluator.validate_publication(wal_state, result, evaluator.SURFACE_STAGE, required=True)
    evaluator.validate_result(result, state, evaluator.SURFACE_STAGE, wal_state)
    evaluator.validate_audit(audit, state, evaluator.SURFACE_STAGE, wal_state)
    _, path_wal_state = evaluator.read_wal(path_wal, evaluator.expected_wal_header(state, evaluator.PATH_STAGE))
    evaluator.validate_publication(path_wal_state, path_result, evaluator.PATH_STAGE, required=True)
    evaluator.validate_result(path_result, state, evaluator.PATH_STAGE, path_wal_state)
    evaluator.validate_audit(path_audit, state, evaluator.PATH_STAGE, path_wal_state)
    with np.load(result, allow_pickle=False) as saved:
        require(str(np.asarray(saved["schema"]).item()) == evaluator.RESULT_SCHEMA, "surface result schema mismatch")
        require(str(np.asarray(saved["stage"]).item()) == evaluator.SURFACE_STAGE, "surface result stage mismatch")
        coordinates = np.asarray(saved["surface_coordinates_A"], dtype=np.float64)
        energies = np.asarray(saved["surface_energy_eV"], dtype=np.float64)
        hashes = np.asarray(saved["surface_coordinate_sha256"]).astype(str)
        fresh = np.asarray(saved["fresh_evaluated_mask"], dtype=bool)
        inherited = np.asarray(saved["inherited_endpoint_mask"], dtype=bool)
    require(coordinates.shape == (POINTS, NORMAL_POINTS, N_ATOMS, 3), "surface coordinate shape mismatch")
    require(energies.shape == fresh.shape == inherited.shape == (POINTS, NORMAL_POINTS), "surface value/mask shape mismatch")
    require(hashes.shape == (POINTS, NORMAL_POINTS), "surface coordinate hash shape mismatch")
    require(np.all(np.isfinite(coordinates)) and np.all(np.isfinite(energies)), "surface contains non-finite values")
    require(int(fresh.sum()) == 1650 and int(inherited.sum()) == 2 and np.all(fresh ^ inherited), "surface fresh/inherited partition mismatch")
    require(inherited[0, ZERO_INDEX] and inherited[-1, ZERO_INDEX], "surface inherited cells are not the two fixed endpoints")
    coordinate_manifest = resolve_binding(contract_path, contract["coordinate_manifest"], "R2 coordinate manifest")
    with np.load(coordinate_manifest, allow_pickle=False) as saved:
        center = np.asarray(saved["center_path59_A"], dtype=np.float64)
        center_s = np.asarray(saved["center_s"], dtype=np.float64)
        normals = np.asarray(saved["normal_directions"], dtype=np.float64)
        n_grid = np.asarray(saved["normal_coordinate_rms_A"], dtype=np.float64)
        initial = np.asarray(saved["common_initial_path59_A"], dtype=np.float64)
        initial_n = np.asarray(saved["common_initial_projected_n_A"], dtype=np.float64)
        base_contract = Path(str(np.asarray(saved["base_evolution_contract"]).item())).expanduser().resolve(strict=True)
        base_contract_sha256 = str(np.asarray(saved["base_evolution_contract_sha256"]).item())
    require(np.array_equal(coordinates[:, ZERO_INDEX], center), "surface n=0 column differs from center")
    require(np.array_equal(center_s, np.linspace(0.0, 1.0, POINTS)), "center s grid mismatch")
    expected_n = np.arange(-0.12, 0.6901, 0.03, dtype=np.float64)
    expected_n[ZERO_INDEX] = 0.0
    require(np.array_equal(n_grid, expected_n), "surface n grid mismatch")
    require(center.shape == normals.shape == initial.shape == (POINTS, N_ATOMS, 3), "surface basis shape mismatch")
    require(np.allclose(np.sum((initial - center) * normals, axis=(1, 2)) / np.sqrt(float(N_ATOMS)), initial_n, rtol=0.0, atol=2e-12), "stored initial projection mismatch")
    relative = np.maximum((energies - energies[0, ZERO_INDEX]) * EV_TO_KCAL, 0.0)
    return {
        "s": center_s,
        "n": n_grid,
        "energy_kcal": relative,
        "center": center,
        "normals": normals,
        "initial_projected": np.column_stack((center_s, initial_n)),
        "bindings": {
            "surface_evaluator": evaluator_path,
            "surface_execution_contract": execution_contract,
            "surface_authorization": authorization,
            "surface_result": result,
            "surface_audit": audit,
            "surface_wal": wal,
            "original_path32_result": path_result,
            "original_path32_audit": path_audit,
            "original_path32_wal": path_wal,
            "coordinate_manifest": coordinate_manifest,
        },
        "call_accounting": {"stored_values": 1652, "fresh_energy_only": 1650, "inherited_endpoints": 2, "force_calls": 0, "optimization_N_F": 0},
        "path_source_contract": base_contract,
        "path_source_contract_sha256": base_contract_sha256,
    }


def _load_paths(contract_path: Path, contract: dict, surface: dict) -> tuple[np.ndarray, list[dict], dict]:
    helper_path = resolve_binding(contract_path, contract["path_loader"], "verified path loader")
    source_contract_path = resolve_binding(contract_path, contract["path_source_contract"], "path source contract")
    require(source_contract_path == surface["path_source_contract"], "surface basis and path loader use different source contracts")
    require(sha256_file(source_contract_path) == surface["path_source_contract_sha256"], "surface basis source-contract hash mismatch")
    helper = import_path(helper_path, "bb_verified_path_loader")
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    gates = helper.validate_json_gates_before_selected_npz(source_contract_path, source_contract)
    with np.load(gates["paths"]["selected_result"], allow_pickle=False) as saved:
        selected = helper.collect_selected_committed_paths(saved, gates["verification"], int(gates["actual_nf"]))
    ledger_path, ledger_rows, ledger_sources = helper.load_ode_neb_paid_path_ledger(source_contract_path, source_contract["ode_neb_paid_path_ledger"])
    ode_paths = {}
    ode_metadata = {}
    for record in source_contract["ode_neb_snapshots"]:
        archive, band, metadata = helper.load_ode_neb_snapshot(source_contract_path, record, ledger_rows)
        ode_paths[int(metadata["optimization_N_F"])] = np.asarray(band, dtype=np.float64)
        ode_metadata[int(metadata["optimization_N_F"])] = {"archive": str(archive), **metadata}
    requested = tuple((int(row["ivm_optimization_N_F"]), int(row["ode_neb_optimization_N_F"])) for row in contract["panels"])
    require(requested == EXPECTED_PANELS, "figure panel NoF pairs changed")
    panels = []
    for ivm_nf, ode_nf in requested:
        require(ivm_nf in selected and ode_nf in ode_paths, "requested stored path is unavailable")
        ivm = np.asarray(selected[ivm_nf]["path"], dtype=np.float64).reshape(N_IMAGES, N_ATOMS, 3)
        ode = np.asarray(ode_paths[ode_nf], dtype=np.float64).reshape(N_IMAGES, N_ATOMS, 3)
        require(np.array_equal(ivm[[0, -1]], ode[[0, -1]]), "IVM/ODE fixed endpoints differ")
        ivm_projection = project_band(ivm, surface["center"], surface["s"], surface["normals"])
        ode_projection = project_band(ode, surface["center"], surface["s"], surface["normals"])
        for projection in (ivm_projection, ode_projection):
            require(float(projection[:, 1].min()) >= float(surface["n"][0]) - 2e-8 and float(projection[:, 1].max()) <= float(surface["n"][-1]) + 2e-8, "projected path falls outside frozen surface")
        panels.append(
            {
                "ivm_optimization_N_F": ivm_nf,
                "ode_neb_optimization_N_F": ode_nf,
                "ivm_projected": ivm_projection,
                "ode_neb_projected": ode_projection,
                "ivm_path_sha256": helper.canonical_le_float64_sha256(ivm),
                "ode_neb_path_sha256": helper.canonical_le_float64_sha256(ode),
                "ivm_projected_sha256": sha256_array(ivm_projection),
                "ode_neb_projected_sha256": sha256_array(ode_projection),
                "ode_metadata": ode_metadata[ode_nf],
            }
        )
    return surface["initial_projected"], panels, {"path_loader": helper_path, "path_source_contract": source_contract_path, "selected_gates": gates["paths"], "ode_ledger": ledger_path, "ode_ledger_sources": ledger_sources}


def synthetic_inputs() -> tuple[dict, np.ndarray, list[dict]]:
    s = np.linspace(0.0, 1.0, POINTS)
    n = np.arange(-0.12, 0.6901, 0.03)
    n[ZERO_INDEX] = 0.0
    sm, nm = np.meshgrid(s, n, indexing="ij")
    valley = 0.015 + 0.055 * np.sin(np.pi * sm) ** 2
    energy = 18.0 * np.sin(np.pi * sm) ** 2 + 1800.0 * (nm - valley) ** 2 + 15.0 * np.sin(3.0 * np.pi * sm) ** 2
    initial_n = 0.62 * np.sin(np.pi * s) ** 1.3
    initial = np.column_stack((s, initial_n))
    panels = []
    for index, (ivm_nf, ode_nf) in enumerate(EXPECTED_PANELS):
        decay = (0.72, 0.34, 0.03)[index]
        ivm_n = decay * initial_n + (0.03 - 0.008 * index) * np.sin(2 * np.pi * s)
        ode_n = (0.98, 0.62, 0.10)[index] * initial_n + 0.035 * np.sin(2.5 * np.pi * s)
        ivm_n[[0, -1]] = 0.0
        ode_n[[0, -1]] = 0.0
        panels.append({"ivm_optimization_N_F": ivm_nf, "ode_neb_optimization_N_F": ode_nf, "ivm_projected": np.column_stack((s, ivm_n)), "ode_neb_projected": np.column_stack((s, ode_n))})
    return {"s": s, "n": n, "energy_kcal": energy}, initial, panels


def plot_evolution(surface: Mapping[str, np.ndarray], initial: np.ndarray, panels: Sequence[Mapping[str, object]]) -> plt.Figure:
    configure_style()
    energy = np.asarray(surface["energy_kcal"], dtype=float)
    maximum = float(np.max(energy))
    require(np.all(np.isfinite(energy)) and maximum > 0.0, "invalid display energy surface")
    norm = LogQuadraticNorm(0.0, maximum, scale=40.0, curvature=2.5)
    filled_levels = norm.inverse(np.linspace(0.0, 1.0, 41))
    line_levels = norm.inverse(np.linspace(0.0, 1.0, 20))
    s_mesh, n_mesh = np.meshgrid(surface["s"], surface["n"], indexing="ij")
    figure, axes = plt.subplots(1, 4, figsize=(7.2, 2.38), sharex=True, sharey=True)
    contour = None
    for panel_index, axis in enumerate(axes):
        contour = axis.contourf(s_mesh, n_mesh, energy, levels=filled_levels, cmap="cividis", norm=norm, antialiased=True, extend="max")
        axis.contour(s_mesh, n_mesh, energy, levels=line_levels, colors="#F3F3F3", linewidths=0.32, alpha=0.58)
        if panel_index == 0:
            axis.plot(initial[:, 0], initial[:, 1], color=INITIAL_COLOR, linewidth=1.6, marker="o", markersize=2.2, markevery=4, label="Initial path", zorder=5)
            title = "Initial path"
        else:
            row = panels[panel_index - 1]
            axis.plot(row["ode_neb_projected"][:, 0], row["ode_neb_projected"][:, 1], color=ODE_COLOR, linewidth=1.45, marker="o", markersize=2.15, markevery=4, label="ODE-NEB", zorder=5)
            axis.plot(row["ivm_projected"][:, 0], row["ivm_projected"][:, 1], color=IVM_COLOR, linewidth=1.8, marker="s", markersize=2.15, markevery=4, label="IVM", zorder=6)
            title = f"IVM {int(row['ivm_optimization_N_F'])} | ODE-NEB {int(row['ode_neb_optimization_N_F'])}"
        axis.set_title(title, pad=5.0, fontsize=7.8 if panel_index else 8.5)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(float(surface["n"][0]), float(surface["n"][-1]))
        axis.set_xticks((0.0, 0.5, 1.0))
        axis.tick_params(length=2.8)
        axis.text(0.035, 0.965, chr(97 + panel_index), transform=axis.transAxes, va="top", ha="left", fontsize=8.7, fontweight="bold")
    axes[0].set_ylabel(r"RMS path-normal $n$ ($\AA$)")
    axes[-1].legend(frameon=True, facecolor="white", edgecolor="#B3B3B3", framealpha=0.90, loc="upper right", handlelength=2.0)
    figure.supxlabel(r"Path progress $s$", x=0.48, y=0.035)
    color_axis = figure.add_axes([0.905, 0.23, 0.014, 0.67])
    colorbar = figure.colorbar(contour, cax=color_axis)
    ticks = [value for value in (0.0, 40.0, 240.0, 1e3, 1e4, 1e6) if value <= maximum]
    if not ticks or ticks[-1] < maximum / 2.0:
        ticks.append(maximum)
    colorbar.set_ticks(ticks)
    labels = []
    for value in ticks:
        if value == 0:
            labels.append("0")
        elif value < 1000:
            labels.append(f"{value:.0f}")
        else:
            exponent = int(np.floor(np.log10(value)))
            mantissa = value / (10.0**exponent)
            labels.append(rf"$10^{{{exponent}}}$" if abs(mantissa - 1.0) < 1e-10 else rf"${mantissa:.1f}\times10^{{{exponent}}}$")
    colorbar.set_ticklabels(labels)
    colorbar.set_label(r"$E_{\mathrm{0K}}-E_{\mathrm{bound}}$ (kcal mol$^{-1}$)")
    figure.subplots_adjust(left=0.075, right=0.885, bottom=0.23, top=0.90, wspace=0.08)
    return figure


def write_pdf_no_overwrite(figure: plt.Figure, output: Path) -> None:
    require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=".tmp.pdf", dir=str(output.parent))
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(temporary, format="pdf", bbox_inches="tight")
        require(temporary.read_bytes().startswith(b"%PDF-"), "matplotlib did not create a PDF")
        os.link(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_no_overwrite(path: Path, payload: dict) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def provenance_tree(value):
    if isinstance(value, Path):
        return file_binding(value)
    if isinstance(value, Mapping):
        return {str(key): provenance_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [provenance_tree(item) for item in value]
    return value


def main(argv=None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(raw)
    output = args.output.expanduser().resolve()
    require(output.suffix.lower() == ".pdf", "output must be PDF")
    provenance_path = output.with_suffix(".json")
    if args.synthetic_dry_run:
        require(args.contract is None, "synthetic dry-run must not accept a scientific contract")
        require("synthetic" in output.stem, "synthetic output filename must say synthetic")
        surface, initial, panels = synthetic_inputs()
        source_bindings = {}
        status = "PASS_SYNTHETIC_ZERO_PHYSICAL_DRY_RUN_NOT_SCIENTIFIC_DATA"
    else:
        require(args.contract is not None, "final mode requires --contract")
        contract_path = args.contract.expanduser().resolve(strict=True)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        require(contract.get("schema") == FIGURE_SCHEMA and contract.get("status") == FIGURE_STATUS, "figure contract is not final")
        require(resolve_binding(contract_path, contract["plotter"], "plotter") == Path(__file__).resolve(), "plotter path mismatch")
        require(output == Path(str(contract["output_pdf"])).expanduser().resolve(), "output differs from frozen figure contract")
        require(provenance_path == Path(str(contract["output_provenance"])).expanduser().resolve(), "provenance path differs from contract")
        surface = _load_surface(contract_path, contract)
        initial, panels, path_sources = _load_paths(contract_path, contract, surface)
        source_bindings = {**surface["bindings"], **path_sources, "figure_contract": contract_path, "style_reference_pdf": resolve_binding(contract_path, contract["style_reference_pdf"], "legacy style reference")}
        status = "PASS_FINAL_R4_IR_CENTERED_EVOLUTION"
    summary = {
        "status": status,
        "surface_shape": list(np.asarray(surface["energy_kcal"]).shape),
        "surface_n_range_A": [float(surface["n"][0]), float(surface["n"][-1])],
        "surface_energy_display_range_kcal_per_mol": [float(np.min(surface["energy_kcal"])), float(np.max(surface["energy_kcal"]))],
        "panels": [
            {
                "ivm_optimization_N_F": int(row["ivm_optimization_N_F"]),
                "ode_neb_optimization_N_F": int(row["ode_neb_optimization_N_F"]),
                "ivm_cartesian_path_sha256": row.get("ivm_path_sha256"),
                "ode_neb_cartesian_path_sha256": row.get("ode_neb_path_sha256"),
                "ivm_projected_sha256": sha256_array(row["ivm_projected"]),
                "ode_neb_projected_sha256": sha256_array(row["ode_neb_projected"]),
            }
            for row in panels
        ],
        "projection": "each stored 32-image path is independently resampled to the frozen 59-point Cartesian arc-length grid; signed RMS displacement is projected on the frozen IR-NF8982 normal",
        "surface_call_accounting": surface.get(
            "call_accounting",
            {
                "status": "synthetic-only; no physical data or calls",
                "fresh_energy_only": 0,
                "force_calls": 0,
                "optimization_N_F": 0,
            },
        ),
        "surface_display_policy": "raw finite eV values remain in the R4 result; the color field displays max((E-E_bound)*23.060547830619, 0)",
        "style": {"reference": "legacy PDF visual language only", "layout": "1x4", "cmap": "cividis", "norm": "log1p scale40 followed by concave power2.5", "ivm_color": IVM_COLOR, "ode_neb_color": ODE_COLOR, "initial_color": INITIAL_COLOR},
        "no_openmm_context_created": True,
    }
    if args.validate_only:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    figure = plot_evolution(surface, initial, panels)
    write_pdf_no_overwrite(figure, output)
    plt.close(figure)
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "status": status,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_pdf": file_binding(output),
        "plotter": file_binding(Path(__file__).resolve()),
        "source_bindings": provenance_tree(source_bindings),
        "summary": summary,
        "invocation": {"argv": [str(value) for value in raw], "python": str(Path(sys.executable).resolve()), "working_directory": str(Path.cwd().resolve())},
    }
    write_json_no_overwrite(provenance_path, provenance)
    print(json.dumps({"status": status, "pdf": file_binding(output), "provenance": file_binding(provenance_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
