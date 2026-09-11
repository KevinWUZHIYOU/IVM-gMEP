#!/usr/bin/env python3
"""Render the formal BB IR-IVM-20k / ODE-NEB-30k profile figure.

This is a zero-physics renderer.  The energy profiles use the already saved
native 32-image energies, while all four molecular configurations are selected
from and rendered from the accepted IR-IVM call-693 path (optimization
N_F=19,993).  No path is densified, no potential is evaluated, and no force is
evaluated.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPOSITORY = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP")
PROJECT = Path("/Users/wuzhiyou/IVM_for_MEP")
HERE = Path(__file__).resolve().parent

IR_SHARD = (
    PROJECT
    / "python/barnase_barstar_ir_ivm_20260901/reports/plot_inputs/"
    "bb_irivm_nf20k_nf30k_20260904_r1/call_0693_stable_last.npz"
)
IR_PRIMARY_AUDIT = (
    PROJECT
    / "python/barnase_barstar_ir_ivm_20260901/reports/verifications/"
    "BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF30000_20260904_R6B_PRIMARY.json"
)
ODE_CACHE = (
    REPOSITORY
    / "code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/"
    "analysis/formal_discrete32_cache_r2_audited_v2_local_mutagen/"
    "formal_current_state_nf30000.npz"
)
ODE_COMPARISON = (
    REPOSITORY
    / "Figures/barnase_barstar/active_candidates/"
    "barnase_barstar_discrete32_ode_nf30000_r2_comparison.json"
)
ODE_COMPLETION_AUDIT = (
    REPOSITORY
    / "Figures/barnase_barstar/active_candidates/"
    "barnase_barstar_discrete32_ode_nf30000_r2_completion_audit.json"
)
ODE_TERMINAL_AUDIT = (
    REPOSITORY
    / "code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/"
    "terminal_audits/"
    "fresh-nonresume-bb-neb-aseode-k0p1-formal-nf30000-remote-20260903-r1."
    "terminal-audit.json"
)
TOPOLOGY_PDB = (
    PROJECT / "python/barnase_barstar_ir_ivm_20260901/inputs/complex_fixed.pdb"
)
REFERENCE_FIGURE = (
    REPOSITORY / "paper0819/Figures/barnase_barstar/"
    "barnase_ivm_snapshots_plus_energy.pdf"
)
OUTPUT = (
    REPOSITORY / "output/pdf/"
    "barnase_barstar_irivm20k_ode30k_snapshots_plus_energy.pdf"
)
PROVENANCE = OUTPUT.with_suffix(".provenance.json")
TRACEABILITY = OUTPUT.with_suffix(".traceability.md")
ASSET_DIR = OUTPUT.with_suffix("").with_name(OUTPUT.stem + "_assets")
TRAJECTORY = ASSET_DIR / "irivm_call0693_nf19993_separated_to_bound.pdb"
CHIMERAX_SCRIPT = ASSET_DIR / "render_irivm_call0693_configurations.cxc"
CHIMERAX = Path("/Applications/ChimeraX-1.12.app/Contents/MacOS/ChimeraX")

# Keep the exact conversion used by the audited BB discrete32 analysis pack.
EV_TO_KCAL = 23.0605
N_IMAGES = 32
N_ATOMS = 3132
N_DOF = N_ATOMS * 3
IR_CALL = 693
IR_NF = 19993
ODE_NF = 30000
IR_COLOR = "#0072B2"
ODE_COLOR = "#D55E00"
SNAPSHOT_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
SNAPSHOT_LABELS = ("Pre-barrier", "Barrier region", "Post-barrier", "Bound complex")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def scalar(saved: Any, key: str) -> Any:
    value = np.asarray(saved[key])
    require(value.shape == (), f"{key} is not scalar")
    return value.item()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage.", suffix=path.suffix, dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_pdf(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage.", suffix=".pdf", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        figure.savefig(
            temporary,
            format="pdf",
            dpi=600,
            bbox_inches="tight",
            metadata={
                "Title": "Barnase-Barstar IVM 20k and ODE-NEB 30k energy profiles",
                "Subject": "Four configurations are selected from the accepted IVM 20k path",
                "Creator": Path(__file__).name,
            },
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} is not a JSON object")
    return value


def load_ir() -> dict[str, Any]:
    audit = load_json(IR_PRIMARY_AUDIT)
    require(
        audit.get("status")
        == "PASS_R6B_NF30000_EXACT_PREFIX318_AND_FULL_STATE_CAPTURE",
        "IR-IVM primary history audit is not the required PASS record",
    )
    rows = [
        row for row in audit.get("path_inventory", [])
        if int(row.get("call_per_image", -1)) == IR_CALL
    ]
    require(len(rows) == 1, "IR-IVM call 693 is not unique in the primary audit")
    row = rows[0]
    require(
        int(row["optimization_N_F"]) == IR_NF
        and row.get("formally_reportable") is True
        and row.get("terminal_uncommitted_charged_tail") is False,
        "IR-IVM call 693 is not the accepted/formally reportable 20k state",
    )
    require(sha256_file(IR_SHARD) == row["shard_sha256"], "IR shard hash differs")
    with np.load(IR_SHARD, allow_pickle=False) as saved:
        path = np.asarray(saved["path32_A"], dtype=np.float64).reshape(
            N_IMAGES, N_ATOMS, 3
        )
        energy = np.asarray(saved["energy_eV"], dtype=np.float64).reshape(N_IMAGES)
        force = np.asarray(saved["force_eV_per_A"], dtype=np.float64).reshape(
            N_IMAGES, N_DOF
        )
        embedded = {
            "path_sha256": str(scalar(saved, "path_sha256")),
            "energy_sha256": str(scalar(saved, "energy_sha256")),
            "force_sha256": str(scalar(saved, "force_sha256")),
        }
        require(
            int(scalar(saved, "call_per_image")) == IR_CALL
            and int(scalar(saved, "optimization_N_F")) == IR_NF
            and bool(scalar(saved, "fully_evaluated_current_path"))
            and int(scalar(saved, "additional_physical_calls")) == 0,
            "IR shard counters or complete-state flags differ",
        )
    computed = {
        "path_sha256": sha256_array(path.reshape(N_IMAGES, N_DOF)),
        "energy_sha256": sha256_array(energy),
        "force_sha256": sha256_array(force),
    }
    require(computed == embedded, "IR embedded array hashes differ")
    require(all(computed[key] == row[key] for key in computed),
            "IR arrays differ from the primary history audit")
    return {
        "method": "IVM",
        "optimization_N_F": IR_NF,
        "path": path,
        "energy_eV": energy,
        "array_hashes": computed,
        "shard": binding(IR_SHARD),
        "primary_audit": binding(IR_PRIMARY_AUDIT),
        "formally_reportable": True,
    }


def load_ode() -> dict[str, Any]:
    comparison = load_json(ODE_COMPARISON)
    completion = load_json(ODE_COMPLETION_AUDIT)
    terminal = load_json(ODE_TERMINAL_AUDIT)
    require(
        comparison.get("status")
        == "COMPLETE_ZERO_FORCE_DISCRETE32_NF30000_R2_ANALYSIS",
        "ODE comparison pack is not complete",
    )
    require(
        completion.get("status") == "PASS_FINAL_R2_FIGURES_QA_AND_PROVENANCE_CLOSED",
        "ODE figure completion audit is not PASS",
    )
    require(
        terminal.get("status") == "PASS"
        and terminal.get("formal_figure_eligible") is True
        and int(terminal.get("paid_optimization_N_F", -1)) == ODE_NF,
        "ODE terminal audit is not the formal NF30000 PASS record",
    )
    history = comparison.get("histories", {}).get("ODE-NEB", [])
    require(len(history) == 1000, "ODE comparison history does not contain 1000 batches")
    row = history[-1]
    require(int(row.get("optimization_N_F", -1)) == ODE_NF,
            "ODE comparison history does not end at NF30000")
    with np.load(ODE_CACHE, allow_pickle=False) as saved:
        path = np.asarray(saved["path32_A"], dtype=np.float64).reshape(
            N_IMAGES, N_ATOMS, 3
        )
        energy = np.asarray(saved["energy32_eV"], dtype=np.float64).reshape(N_IMAGES)
        force = np.asarray(saved["full_force32_eV_per_A"], dtype=np.float64).reshape(
            N_IMAGES, N_DOF
        )
        embedded = {
            "path_sha256": str(scalar(saved, "path_sha256")),
            "energy_sha256": str(scalar(saved, "energy_sha256")),
            "force_sha256": str(scalar(saved, "force_sha256")),
        }
        require(
            str(scalar(saved, "schema"))
            == "bb-ode-formal-discrete32-selected-state-v1"
            and int(scalar(saved, "optimization_N_F")) == ODE_NF
            and int(scalar(saved, "batch_id")) == 1000
            and int(scalar(saved, "current_source_batch_id")) == 1000
            and int(scalar(saved, "endpoint_optimization_N_F")) == 0
            and int(scalar(saved, "additional_physical_calls")) == 0,
            "ODE selected-state cache counters differ",
        )
    computed = {
        "path_sha256": sha256_array(path),
        "energy_sha256": sha256_array(energy),
        "force_sha256": sha256_array(force),
    }
    require(computed == embedded, "ODE embedded array hashes differ")
    require(all(computed[key] == row[key] for key in computed),
            "ODE arrays differ from the formal comparison pack")
    require(
        float((np.max(energy) - energy[0]) * EV_TO_KCAL)
        == float(row["delta_e32_kcal_per_mol"]),
        "ODE stored-energy barrier differs from the formal comparison pack",
    )
    return {
        "method": "ODE-NEB",
        "optimization_N_F": ODE_NF,
        "path": path,
        "energy_eV": energy,
        "array_hashes": computed,
        "cache": binding(ODE_CACHE),
        "comparison": binding(ODE_COMPARISON),
        "completion_audit": binding(ODE_COMPLETION_AUDIT),
        "terminal_audit": binding(ODE_TERMINAL_AUDIT),
        "formally_reportable": True,
    }


def path_coordinate(path: np.ndarray) -> np.ndarray:
    flat = np.asarray(path, dtype=np.float64).reshape(N_IMAGES, N_DOF)
    segment = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    require(np.all(segment > 0.0), "path has a degenerate Cartesian segment")
    cumulative = np.concatenate(([0.0], np.cumsum(segment)))
    coordinate = cumulative / cumulative[-1]
    require(
        coordinate.shape == (N_IMAGES,)
        and coordinate[0] == 0.0
        and coordinate[-1] == 1.0
        and np.all(np.diff(coordinate) > 0.0),
        "normalized Cartesian arc-length coordinate differs",
    )
    return coordinate


def select_ir_configurations(energy_eV: np.ndarray) -> tuple[list[int], list[int]]:
    barrier = int(np.argmax(energy_eV))
    require(2 <= barrier <= N_IMAGES - 3, "IR energy maximum is too close to an endpoint")
    indices = [barrier + 2, barrier, barrier - 2, 0]
    frames = [N_IMAGES - index for index in indices]
    require(len(set(indices)) == 4 and all(0 <= index < N_IMAGES for index in indices),
            "configuration selection is not four distinct native IR images")
    return indices, frames


def write_multimodel_pdb(template: Path, path: np.ndarray, output: Path) -> None:
    lines = template.read_text(encoding="utf-8").splitlines()
    atom_lines = [line for line in lines if line.startswith(("ATOM  ", "HETATM"))]
    require(len(atom_lines) == N_ATOMS, "PDB template atom count differs")
    header = [
        line for line in lines
        if not line.startswith(("ATOM  ", "HETATM", "CONECT", "MODEL", "ENDMDL", "END"))
    ]
    rendered = []
    rendered.extend(header)
    for model, coordinates in enumerate(np.asarray(path)[::-1], start=1):
        rendered.append(f"MODEL     {model:4d}")
        for source, xyz in zip(atom_lines, coordinates):
            rendered.append(
                source[:30]
                + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
                + source[54:]
            )
        rendered.append("ENDMDL")
    rendered.append("END")
    atomic_text(output, "\n".join(rendered) + "\n")


def write_chimerax_script(frames: list[int]) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "close all",
        "windowsize 1200 1200",
        f'open "{TRAJECTORY}" coordsets true autoStyle false',
        "set bgColor white",
        "preset 1",
        "camera ortho",
        "lighting gentle",
        "material dull",
        "graphics silhouettes true color #202020 width 1.5",
        "show #1 models",
        "show #1 atoms",
        f"coordset #1 {frames[0]}",
        "view #1 pad 0.08",
        "hide #1 atoms",
        "cartoon #1",
        "show #1 cartoons",
        "cartoon style #1 width 1.8 thickness 0.35 xsection oval",
        "color #1/C #4C78A8 target c",
        "color #1/F #F28E2B target c",
        "turn y 8",
        "turn x -6",
    ]
    for frame in frames:
        image_path = ASSET_DIR / f"irivm_frame_{frame:02d}.png"
        lines.extend([
            f"coordset #1 {frame}",
            "wait 2",
            f'save "{image_path}" width 1800 height 1800 supersample 3 '
            "transparentBackground false",
        ])
    lines.append("exit")
    atomic_text(CHIMERAX_SCRIPT, "\n".join(lines) + "\n")


def prepare_configuration_assets(ir: dict[str, Any], frames: list[int], render: bool) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    write_multimodel_pdb(TOPOLOGY_PDB, ir["path"], TRAJECTORY)
    write_chimerax_script(frames)
    if render:
        require(CHIMERAX.is_file(), "ChimeraX executable is unavailable")
        subprocess.run(
            # ChimeraX 1.12 on this macOS host has no OSMesa library, so its
            # documented --offscreen path cannot create an OpenGL context.
            # The ordinary renderer is still deterministic under the frozen
            # command script and exits as soon as the four PNGs are saved.
            [str(CHIMERAX), "--exit", str(CHIMERAX_SCRIPT)],
            check=True,
            cwd=REPOSITORY,
        )
    for frame in frames:
        image_path = ASSET_DIR / f"irivm_frame_{frame:02d}.png"
        require(image_path.is_file() and image_path.stat().st_size > 10000,
                f"configuration image is absent or unexpectedly small: {image_path}")


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def render_figure(
    ir: dict[str, Any], ode: dict[str, Any], indices: list[int], frames: list[int]
) -> dict[str, Any]:
    configure_style()
    ir_relative = (ir["energy_eV"] - ir["energy_eV"][0]) * EV_TO_KCAL
    ode_relative = (ode["energy_eV"] - ode["energy_eV"][0]) * EV_TO_KCAL
    x_ir = 1.0 - path_coordinate(ir["path"])[::-1]
    x_ode = 1.0 - path_coordinate(ode["path"])[::-1]
    ir_display = ir_relative[::-1]
    ode_display = ode_relative[::-1]
    require(np.all(np.diff(x_ir) > 0.0) and np.all(np.diff(x_ode) > 0.0),
            "display arc-length coordinates are not strictly increasing")

    figure = plt.figure(figsize=(7.2, 5.15))
    grid = figure.add_gridspec(
        2, 4, height_ratios=(1.0, 1.10), hspace=0.48, wspace=0.03
    )
    image_axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
    profile = figure.add_subplot(grid[1, :])
    image_bindings = []
    for panel, (axis, frame, label, color) in enumerate(
        zip(image_axes, frames, SNAPSHOT_LABELS, SNAPSHOT_COLORS)
    ):
        image_path = ASSET_DIR / f"irivm_frame_{frame:02d}.png"
        image = plt.imread(image_path)
        require(image.ndim in (2, 3) and image.shape[0] == 1800 and image.shape[1] == 1800,
                f"configuration image dimensions differ: {image_path}")
        crop = int(0.025 * min(image.shape[:2]))
        axis.imshow(image[crop:-crop, crop:-crop])
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.65)
            spine.set_edgecolor("#A6A6A6")
        axis.text(
            0.035, 0.965, f"({chr(97 + panel)})", transform=axis.transAxes,
            ha="left", va="top", fontsize=8.5, fontweight="bold",
        )
        axis.text(
            0.97, 0.035, label, transform=axis.transAxes,
            ha="right", va="bottom", fontsize=7.2,
            bbox={"facecolor": "white", "alpha": 0.80, "edgecolor": "none", "pad": 1.2},
        )
        axis.plot(
            [0.0, 1.0], [0.0, 0.0], transform=axis.transAxes,
            color=color, linewidth=3.0, clip_on=False,
        )
        image_bindings.append(binding(image_path))

    profile.plot(
        x_ode, ode_display, "--s", color=ODE_COLOR, linewidth=1.35,
        markersize=2.7, markerfacecolor="white", markeredgewidth=0.65,
        label=r"ODE-NEB ($N_F=30\mathrm{k}$)", zorder=2,
    )
    profile.plot(
        x_ir, ir_display, "-o", color=IR_COLOR, linewidth=1.8,
        markersize=2.7, markerfacecolor="white", markeredgewidth=0.65,
        label=r"IVM ($N_F\approx20\mathrm{k}$)", zorder=3,
    )
    display_indices = [N_IMAGES - 1 - index for index in indices]
    selected_x = [float(x_ir[index]) for index in display_indices]
    selected_y = [float(ir_display[index]) for index in display_indices]
    markers = ("o", "*", "s", "D")
    sizes = (34, 65, 34, 38)
    for panel, (x_value, y_value, color, marker, size) in enumerate(
        zip(selected_x, selected_y, SNAPSHOT_COLORS, markers, sizes)
    ):
        profile.scatter(
            x_value, y_value, s=size, marker=marker, facecolor=color,
            edgecolor="#202020", linewidth=0.65, zorder=6,
        )
        profile.annotate(
            f"({chr(97 + panel)})", (x_value, y_value), xytext=(0, 7),
            textcoords="offset points", color=color, ha="center", va="bottom",
            fontsize=7.5, fontweight="bold",
        )
    profile.axhline(0.0, color="#808080", linewidth=0.6)
    profile.set_xlim(-0.02, 1.02)
    ymax = max(float(np.max(ir_display)), float(np.max(ode_display)))
    profile.set_ylim(-0.04 * ymax, 1.13 * ymax)
    profile.set_xlabel(r"Normalized Cartesian arc length (separated $\rightarrow$ bound)")
    profile.set_ylabel(r"Energy relative to bound (kcal mol$^{-1}$)")
    profile.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.60)
    profile.legend(frameon=False, loc="upper left")
    profile.spines["top"].set_visible(False)
    profile.spines["right"].set_visible(False)
    profile.set_title(
        "IVM pathway configurations and energy profile",
        loc="left", fontsize=9.0, fontweight="bold", pad=5.0,
    )

    for axis, x_value, y_value, color in zip(
        image_axes, selected_x, selected_y, SNAPSHOT_COLORS
    ):
        start = axis.transAxes.transform((0.52, -0.02))
        end = profile.transData.transform((x_value, y_value))
        start_figure = figure.transFigure.inverted().transform(start)
        end_figure = figure.transFigure.inverted().transform(end)
        figure.add_artist(plt.Line2D(
            [start_figure[0], end_figure[0]],
            [start_figure[1], end_figure[1]],
            transform=figure.transFigure, color=color, linewidth=0.8,
            linestyle="--", alpha=0.52, zorder=0,
        ))
    figure.subplots_adjust(left=0.09, right=0.985, bottom=0.105, top=0.985)
    atomic_pdf(figure, OUTPUT)
    plt.close(figure)
    return {
        "ir_barrier_index_bound_to_separated_zero_based": int(np.argmax(ir_relative)),
        "ir_barrier_kcal_per_mol": float(np.max(ir_relative)),
        "ode_barrier_kcal_per_mol": float(np.max(ode_relative)),
        "selected_ir_indices_bound_to_separated_zero_based": indices,
        "selected_ir_frames_separated_to_bound_one_based": frames,
        "selected_labels": list(SNAPSHOT_LABELS),
        "selected_arc_coordinates_separated_to_bound": selected_x,
        "selected_relative_energies_kcal_per_mol": selected_y,
        "profile_points_per_method": N_IMAGES,
        "paths_densified_or_interpolated": False,
        "configuration_images": image_bindings,
    }


def write_provenance(ir: dict[str, Any], ode: dict[str, Any], plot: dict[str, Any]) -> None:
    record = {
        "schema": "bb-irivm20k-ode30k-snapshots-energy-provenance-v1",
        "status": "PASS_FORMAL_STORED32_ZERO_PHYSICS_RENDER",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scientific_claim": {
            "ivm": "accepted/formally reportable IR-IVM call 693 nearest 20k",
            "ode_neb": "formal accepted ASE ODE-NEB state at exactly NF30000",
            "all_four_configurations": "selected exclusively from the IVM NF19993 path",
        },
        "metric_and_display_protocol": {
            "energy": "stored native 32-image energy; each profile relative to the common bound endpoint",
            "eV_to_kcal_per_mol": EV_TO_KCAL,
            "horizontal_coordinate": "normalized Cartesian arc length computed separately on each stored 32-image path",
            "display_direction": "separated to bound",
            "display_budget_labels": {
                "IVM": "N_F approximately 20k",
                "ODE-NEB": "N_F = 30k",
            },
            "configuration_selection": "one native image two index positions on each side of the IVM discrete barrier, the IVM barrier image, and the bound endpoint",
            "path_densification": False,
            "energy_or_force_recomputation": False,
        },
        "ivm": {key: value for key, value in ir.items() if key not in {"path", "energy_eV"}},
        "ode_neb": {key: value for key, value in ode.items() if key not in {"path", "energy_eV"}},
        "configuration_selection": plot,
        "render_assets": {
            "topology_template": binding(TOPOLOGY_PDB),
            "derived_multimodel_pdb": binding(TRAJECTORY),
            "derived_render_only_PDB_coordinate_precision_A": 0.001,
            "chimerax_script": binding(CHIMERAX_SCRIPT),
            "chimerax_executable": str(CHIMERAX),
            "view_and_chain_colors": "frozen from the prior manuscript figure renderer",
        },
        "reference_style_figure": binding(REFERENCE_FIGURE),
        "code": binding(Path(__file__)),
        "output": binding(OUTPUT),
        "physical_calls": 0,
        "optimization_N_F_contribution": 0,
        "openmm_contexts_constructed": 0,
    }
    atomic_text(
        PROVENANCE,
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    traceability = f"""# BB IVM-20k / ODE-NEB-30k configurations and energy profile

Status: `PASS_FORMAL_STORED32_ZERO_PHYSICS_RENDER`

## Figure meaning

The energy-profile panel compares the accepted IR-IVM call {IR_CALL} state at optimization $N_F={IR_NF:,}$ with the formal ASE ODE-NEB state at $N_F={ODE_NF:,}$. Both curves connect their 32 stored energies directly; no densification, interpolation, new energy evaluation, or new force evaluation was used.

All four molecular panels are configurations from the accepted IR-IVM $N_F={IR_NF:,}$ path. In bound-to-separated zero-based path indexing they are `{plot['selected_ir_indices_bound_to_separated_zero_based']}`; in the derived separated-to-bound multimodel PDB they are one-based frames `{plot['selected_ir_frames_separated_to_bound_one_based']}`. They denote pre-barrier, barrier region, post-barrier, and the bound complex, respectively.

## Sources

- IR-IVM state: `{IR_SHARD}`
- IR-IVM primary history audit: `{IR_PRIMARY_AUDIT}`
- ODE-NEB state cache: `{ODE_CACHE}`
- ODE-NEB comparison pack: `{ODE_COMPARISON}`
- ODE-NEB terminal audit: `{ODE_TERMINAL_AUDIT}`
- PDB topology/template: `{TOPOLOGY_PDB}`
- Renderer: `{Path(__file__).resolve()}`
- Machine-readable provenance: `{PROVENANCE}`

## Accounting

This plotting workflow made 0 physical energy/force calls, constructed 0 OpenMM contexts, and contributed 0 to optimization $N_F$. ChimeraX was used only to render cartoons from already saved Cartesian coordinates.
"""
    atomic_text(TRACEABILITY, traceability)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-configurations", action="store_true",
        help="regenerate the four ChimeraX PNGs before assembling the PDF",
    )
    args = parser.parse_args()
    ir = load_ir()
    ode = load_ode()
    require(np.array_equal(ir["path"][[0, -1]], ode["path"][[0, -1]]),
            "IVM and ODE-NEB do not share byte-identical fixed endpoints")
    indices, frames = select_ir_configurations(ir["energy_eV"])
    prepare_configuration_assets(ir, frames, args.render_configurations)
    plot = render_figure(ir, ode, indices, frames)
    write_provenance(ir, ode, plot)
    print(json.dumps({
        "status": "PASS_FORMAL_STORED32_ZERO_PHYSICS_RENDER",
        "output": binding(OUTPUT),
        "provenance": binding(PROVENANCE),
        "traceability": binding(TRACEABILITY),
        "selected_ir_indices": indices,
        "selected_ir_frames": frames,
        "physical_calls": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
