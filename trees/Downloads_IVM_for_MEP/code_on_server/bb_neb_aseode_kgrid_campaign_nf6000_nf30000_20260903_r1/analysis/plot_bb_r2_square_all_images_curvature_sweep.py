#!/usr/bin/env python3
"""Render seven square-panel BB evolution variants in the formal BB style.

The audited R2 surface, four saved native 32-image path states, labels and
energy values are inherited without modification.  Each real image is directly
projected onto the saved 59-point coordinate basis; paths are neither resampled
nor densified, and every image is marked.  Path colors and artists otherwise
match the formal BB evolution figure and panels remain square.  The requested
contour density is 0.6 times the previous 28 filled bands: 17 bands, hence 16
internal white contours.  A segmented color norm maps 0, 50, 240 and the true
maximum energy to 0, 0.07, 0.90 and 1; curvature acts only on 50--240.

No energy or force backend is imported by this script.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402


HERE = Path(__file__).resolve().parent
REPOSITORY = Path("/Users/wuzhiyou/Downloads/IVM_for_MEP")
BASE_WRAPPER = HERE / "plot_bb_r2_merged_relaxed15x10_evolution.py"
BASE_WRAPPER_SHA256 = (
    "040f5e996f5c23b9a8f34b37a3fc7c1540f076e17c2e9ff1f5b770f6503e69cb"
)
PATH_STYLE_PDF = (
    REPOSITORY
    / "Figures/barnase_barstar/barnase_branch_local_true0k_evolution.pdf"
)
PATH_STYLE_PDF_SHA256 = (
    "26625a3f4a3a23c4d4302be6283e8a07879d7ca6fe569abec276f2e0ee8f9394"
)
PATH_STYLE_SOURCE = (
    REPOSITORY
    / "code_on_server/barnase_barstar_chigstyle/src/plot_manuscript_candidate.py"
)
PATH_STYLE_SOURCE_SHA256 = (
    "aa30cd8530c67df66de511868db88711c2614193396aefc93627d59be060151c"
)
OUTPUT_DIRECTORY = REPOSITORY / "output/pdf"
CANONICAL_OUTPUT = (
    OUTPUT_DIRECTORY / "barnase_barstar_irivm_constrained_relaxed_evolution.pdf"
)
MANIFEST = (
    OUTPUT_DIRECTORY
    / "barnase_barstar_irivm_constrained_relaxed_evolution_"
      "square_all_points_curvature_sweep.manifest.json"
)
PREVIEW_OUTPUT = (
    OUTPUT_DIRECTORY
    / "barnase_barstar_irivm_constrained_relaxed_evolution_"
      "midband_native32_curvature2_preview.pdf"
)
INITIAL_PATH32 = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
    "inputs/common_initial_path32.npz"
)
INITIAL_PATH32_SHA256 = (
    "6983365a5c39d117b5ec1ce4cb9afbaed1817c1190d27a567dea1fbce983e6cf"
)
PROJECTION_COORDINATE_SOURCE = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/"
    "runs/diagnostics/bb_irivm_call515_nf14983_terminal_uncommitted_tail_"
    "0k_section_coordinates_59x28_20260903_r2.npz"
)
PROJECTION_COORDINATE_SOURCE_SHA256 = (
    "8d84d1a5e2881306e02cacfb035d16e21f8a02666911f90cc57d89dde6611bea"
)

# Preserve the user's requested enumeration in the manifest and output loop.
CURVATURES = (1.0, 1.5, 1.25, 0.7, 2.0, 3.0, 4.0)

# Exact line/marker/color values used by the formal BB reference plot.
REFERENCE_PATH_STYLES = {
    "Initial path": {
        "color": "#7B3294", "linestyle": "-", "linewidth": 1.55,
        "marker": "o", "markersize": 2.15, "markevery": 1, "zorder": 5,
    },
    "IVM": {
        "color": "#0072B2", "linestyle": "-", "linewidth": 1.75,
        "marker": "s", "markersize": 2.15, "markevery": 1, "zorder": 6,
    },
    "ODE-NEB": {
        "color": "#D55E00", "linestyle": "-", "linewidth": 1.45,
        "marker": "o", "markersize": 2.15, "markevery": 1, "zorder": 5,
    },
}
PREVIOUS_FILLED_CONTOUR_BANDS = 28
CONTOUR_DENSITY_FACTOR = 0.6
REQUESTED_FILLED_CONTOUR_BANDS = round(
    PREVIOUS_FILLED_CONTOUR_BANDS * CONTOUR_DENSITY_FACTOR
)
CONTOUR_COUNT_ROUNDING_IS_EXACT = REQUESTED_FILLED_CONTOUR_BANDS == 17
LOW_ENERGY_ANCHOR = 50.0
HIGH_ENERGY_ANCHOR = 240.0
LOW_COLOR_FRACTION = 0.07
MIDDLE_COLOR_FRACTION = 0.83
HIGH_COLOR_FRACTION = 0.10
MIDDLE_COLOR_END = LOW_COLOR_FRACTION + MIDDLE_COLOR_FRACTION
N_IMAGES = 32
N_ATOMS = 3132
N_DOF = 9396


class SegmentedMidrangeNorm(Normalize):
    """Allocate 7%, 83%, 10% of color space to three energy ranges.

    Curvature acts only on the 50--240 kcal/mol middle segment.  The low and
    high segments remain linear and every segment is continuous and monotone.
    """

    def __init__(self, vmin, vmax, scale=40.0, curvature=2.0):
        del scale  # retained only for call compatibility with the base plot
        super().__init__(vmin=float(vmin), vmax=float(vmax), clip=True)
        self.curvature = float(curvature)
        require(self.vmin == 0.0, "segmented norm requires zero reference energy")
        require(self.vmax > HIGH_ENERGY_ANCHOR,
                "segmented norm requires an energy maximum above 240 kcal/mol")
        require(math.isfinite(self.curvature) and self.curvature > 0.0,
                "segmented norm curvature must be finite and positive")

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        energy = np.clip(result.data, self.vmin, self.vmax)
        mapped = np.empty_like(energy, dtype=np.float64)
        low = energy <= LOW_ENERGY_ANCHOR
        middle = (energy > LOW_ENERGY_ANCHOR) & (energy <= HIGH_ENERGY_ANCHOR)
        high = energy > HIGH_ENERGY_ANCHOR
        mapped[low] = LOW_COLOR_FRACTION * energy[low] / LOW_ENERGY_ANCHOR
        local = ((energy[middle] - LOW_ENERGY_ANCHOR)
                 / (HIGH_ENERGY_ANCHOR - LOW_ENERGY_ANCHOR))
        curved = 1.0 - np.power(1.0 - local, self.curvature)
        mapped[middle] = LOW_COLOR_FRACTION + MIDDLE_COLOR_FRACTION * curved
        mapped[high] = MIDDLE_COLOR_END + HIGH_COLOR_FRACTION * (
            (energy[high] - HIGH_ENERGY_ANCHOR)
            / (self.vmax - HIGH_ENERGY_ANCHOR)
        )
        output = np.ma.array(mapped, mask=result.mask, copy=False)
        return output[0] if is_scalar else output

    def inverse(self, value):
        mapped = np.clip(np.asarray(value, dtype=np.float64), 0.0, 1.0)
        energy = np.empty_like(mapped, dtype=np.float64)
        low = mapped <= LOW_COLOR_FRACTION
        middle = (mapped > LOW_COLOR_FRACTION) & (mapped <= MIDDLE_COLOR_END)
        high = mapped > MIDDLE_COLOR_END
        energy[low] = LOW_ENERGY_ANCHOR * mapped[low] / LOW_COLOR_FRACTION
        local = ((mapped[middle] - LOW_COLOR_FRACTION)
                 / MIDDLE_COLOR_FRACTION)
        uncurved = 1.0 - np.power(1.0 - local, 1.0 / self.curvature)
        energy[middle] = (LOW_ENERGY_ANCHOR
                          + (HIGH_ENERGY_ANCHOR - LOW_ENERGY_ANCHOR) * uncurved)
        energy[high] = HIGH_ENERGY_ANCHOR + (
            (self.vmax - HIGH_ENERGY_ANCHOR)
            * (mapped[high] - MIDDLE_COLOR_END) / HIGH_COLOR_FRACTION
        )
        return energy


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def load_file_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.stage.", suffix=".json", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def curvature_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def output_for_curvature(value: float) -> Path:
    if value == 2.0:
        return CANONICAL_OUTPUT
    return OUTPUT_DIRECTORY / (
        "barnase_barstar_irivm_constrained_relaxed_evolution_"
        f"square_all_points_curvature_{curvature_tag(value)}.pdf"
    )


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.view(np.uint8)).hexdigest()


def load_native_source_array(path: Path, expected_path: np.ndarray) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as saved:
        keys = [key for key in ("path32_A", "nodes", "initial_path") if key in saved.files]
        require(len(keys) == 1, f"native path source {path} has ambiguous array keys")
        key = keys[0]
        raw = np.asarray(saved[key], dtype=np.float64)
    require(raw.shape == (N_IMAGES, N_DOF),
            f"native source {path} does not store exact shape (32, 9396)")
    shaped = raw.reshape(N_IMAGES, N_ATOMS, 3)
    require(np.array_equal(shaped, np.asarray(expected_path, dtype=np.float64)),
            f"native source {path} differs from the already validated path")
    return {
        "source": binding(path),
        "array_key": key,
        "native_stored_shape": list(raw.shape),
        "native_array_sha256": array_sha256(raw),
    }


def load_projection_basis() -> dict[str, object]:
    require(binding(PROJECTION_COORDINATE_SOURCE)["sha256"]
            == PROJECTION_COORDINATE_SOURCE_SHA256,
            "native-path projection coordinate source changed")
    with np.load(PROJECTION_COORDINATE_SOURCE, allow_pickle=False) as saved:
        center = np.asarray(saved["center_path59_A"], dtype=np.float64)
        s = np.asarray(saved["center_s"], dtype=np.float64)
        normals = np.asarray(saved["normal_directions"], dtype=np.float64)
    require(center.shape == normals.shape == (59, N_ATOMS, 3)
            and s.shape == (59,), "projection basis dimensions differ")
    require(np.all(np.diff(s) > 0.0) and s[0] == 0.0 and s[-1] == 1.0,
            "projection basis path coordinate differs")
    normal_norm = np.linalg.norm(normals.reshape(59, N_DOF), axis=1)
    require(np.allclose(normal_norm, 1.0, rtol=0.0, atol=1.0e-12),
            "projection basis normals are not unit vectors")
    return {"center": center, "s": s, "normals": normals,
            "source": binding(PROJECTION_COORDINATE_SOURCE)}


def project_native_path32(path32: np.ndarray, basis: dict[str, object]) -> np.ndarray:
    """Project each real image onto the 59-point center polyline directly."""
    path = np.asarray(path32, dtype=np.float64)
    require(path.shape == (N_IMAGES, N_ATOMS, 3),
            "in-memory native path does not have 32 images")
    flat = path.reshape(N_IMAGES, N_DOF)
    center = np.asarray(basis["center"], dtype=np.float64).reshape(59, N_DOF)
    s = np.asarray(basis["s"], dtype=np.float64)
    normals = np.asarray(basis["normals"], dtype=np.float64).reshape(59, N_DOF)
    segments = center[1:] - center[:-1]
    segment_norm2 = np.einsum("ij,ij->i", segments, segments)
    require(np.all(segment_norm2 > 0.0), "projection center polyline is degenerate")
    projected = np.empty((N_IMAGES, 2), dtype=np.float64)
    for image_index, image in enumerate(flat):
        fraction = np.clip(
            np.einsum("ij,ij->i", image - center[:-1], segments) / segment_norm2,
            0.0, 1.0,
        )
        closest = center[:-1] + fraction[:, None] * segments
        residual = image - closest
        segment_index = int(np.argmin(np.einsum("ij,ij->i", residual, residual)))
        local = float(fraction[segment_index])
        direction = ((1.0 - local) * normals[segment_index]
                     + local * normals[segment_index + 1])
        direction /= np.linalg.norm(direction)
        projected[image_index, 0] = (
            (1.0 - local) * s[segment_index] + local * s[segment_index + 1]
        )
        projected[image_index, 1] = (
            np.dot(residual[segment_index], direction) / math.sqrt(N_ATOMS)
        )
    projected[[0, -1], 1] = 0.0
    require(np.all(np.isfinite(projected))
            and np.all(np.diff(projected[:, 0]) >= -1.0e-12),
            "native 32-image projection is non-finite or reverses path progress")
    return projected


def path_snapshot_records(inputs: tuple) -> dict[str, object]:
    ir_snapshots, ode_snapshots = inputs[:2]
    ir_records = {}
    for target, row in sorted(ir_snapshots.items()):
        source = binding(Path(row["path"]))
        require(source["sha256"] == row["sha256"]
                and source["bytes"] == row["bytes"],
                f"IR-IVM target {target} source binding differs")
        ir_records[str(target)] = {
            "target_N_F": int(row["target_N_F"]),
            "optimization_N_F": int(row["optimization_N_F"]),
            "call_per_image": int(row["call_per_image"]),
            "committed_by_final_reparameterization": bool(
                row["committed_by_final_reparameterization"]
            ),
            "source": source,
            "saved_images": 32,
            "native_path": load_native_source_array(Path(row["path"]), row["path32"]),
        }
    ode_records = {}
    for target, row in sorted(ode_snapshots.items()):
        source = binding(Path(row["path"]))
        require(source["sha256"] == row["sha256"]
                and source["bytes"] == row["bytes"],
                f"ODE-NEB target {target} source binding differs")
        ode_records[str(target)] = {
            "target_N_F": int(row["target_N_F"]),
            "optimization_N_F": int(row["optimization_N_F"]),
            "batch_id": int(row["batch_id"]),
            "source": source,
            "saved_images": 32,
            "native_path": load_native_source_array(Path(row["path"]), row["path32"]),
        }
    return {"IR-IVM": ir_records, "ODE-NEB": ode_records}


def build_native_projections(inputs: tuple) -> tuple[list[dict[str, np.ndarray]], dict]:
    ir_snapshots, ode_snapshots = inputs[:2]
    basis = load_projection_basis()
    require(sha256(INITIAL_PATH32) == INITIAL_PATH32_SHA256,
            "native common initial path changed")
    with np.load(INITIAL_PATH32, allow_pickle=False) as saved:
        initial_raw = np.asarray(saved["initial_path"], dtype=np.float64)
    require(initial_raw.shape == (N_IMAGES, N_DOF),
            "native initial path does not store exact shape (32, 9396)")
    initial = initial_raw.reshape(N_IMAGES, N_ATOMS, 3)
    panels = [
        {"Initial path": project_native_path32(initial, basis)},
        {"ODE-NEB": project_native_path32(ode_snapshots[2000]["path32"], basis),
         "IR-IVM": project_native_path32(ir_snapshots[2000]["path32"], basis)},
        {"ODE-NEB": project_native_path32(ode_snapshots[10000]["path32"], basis),
         "IR-IVM": project_native_path32(ir_snapshots[10000]["path32"], basis)},
        {"ODE-NEB": project_native_path32(ode_snapshots[30000]["path32"], basis),
         "IR-IVM": project_native_path32(ir_snapshots[15000]["path32"], basis)},
    ]
    all_values = np.concatenate([value for panel in panels for value in panel.values()])
    require(all(value.shape == (N_IMAGES, 2)
                for panel in panels for value in panel.values()),
            "a native projection does not contain exactly 32 plotted points")
    return panels, {
        "projection_coordinate_source": basis["source"],
        "initial_path": {
            "source": binding(INITIAL_PATH32),
            "array_key": "initial_path",
            "native_stored_shape": list(initial_raw.shape),
            "native_array_sha256": array_sha256(initial_raw),
        },
        "projection_method": (
            "each native image is projected to its nearest point on the saved "
            "59-point center polyline; no path resampling or path densification"
        ),
        "projected_points_per_path": N_IMAGES,
        "projected_s_range": [float(all_values[:, 0].min()),
                              float(all_values[:, 0].max())],
        "projected_n_range_A": [float(all_values[:, 1].min()),
                                float(all_values[:, 1].max())],
    }


def find_single_path_line(axis: plt.Axes, label: str) -> Line2D:
    candidates = [line for line in axis.lines if line.get_label() == label]
    require(len(candidates) == 1, f"expected exactly one {label!r} line")
    return candidates[0]


def apply_reference_path_style(
    line: Line2D, label: str, projected: np.ndarray,
) -> dict[str, object]:
    style = REFERENCE_PATH_STYLES[label]
    require(projected.shape == (N_IMAGES, 2),
            f"{label} does not contain 32 directly projected native images")
    line.set_data(projected[:, 0], projected[:, 1])
    original_color = line.get_color()
    require(original_color.lower() == style["color"].lower(),
            f"{label} inherited color differs from the formal BB reference")
    line.set_color(style["color"])
    line.set_linestyle(style["linestyle"])
    line.set_linewidth(style["linewidth"])
    line.set_marker(style["marker"])
    line.set_markersize(style["markersize"])
    line.set_markevery(style["markevery"])
    line.set_zorder(style["zorder"])
    # The formal BB source does not override marker face, marker edge, or
    # opacity.  Keep the fresh Matplotlib defaults already present on the line.
    displayed_markers = len(range(0, len(line.get_xdata()), style["markevery"]))
    return {
        "label": label,
        "color": line.get_color(),
        "color_inherited_unchanged": original_color == line.get_color(),
        "linestyle": style["linestyle"],
        "linewidth_pt": style["linewidth"],
        "marker": style["marker"],
        "markersize_pt": style["markersize"],
        "markerfacecolor_effective": line.get_markerfacecolor(),
        "markeredgecolor_effective": line.get_markeredgecolor(),
        "markeredgewidth_pt": line.get_markeredgewidth(),
        "alpha": line.get_alpha(),
        "defaults_retained_for_marker_face_edge_and_alpha": True,
        "markevery": style["markevery"],
        "zorder": style["zorder"],
        "stored_path_points": len(line.get_xdata()),
        "displayed_marker_points": displayed_markers,
        "projected_coordinate_sha256": array_sha256(projected),
        "path_resampled_or_densified": False,
    }


def adapt_figure(
    figure: plt.Figure, native_projections: list[dict[str, np.ndarray]],
) -> dict[str, object]:
    require(len(figure.axes) == 5, "expected four panels and one colorbar")
    require(len(native_projections) == 4, "native projection panel count differs")
    axes = figure.axes[:4]
    color_axis = figure.axes[4]

    styles: list[dict[str, object]] = []
    initial_line = find_single_path_line(axes[0], "Initial path")
    styles.append(apply_reference_path_style(
        initial_line, "Initial path", native_projections[0]["Initial path"]
    ))
    require(len(initial_line.get_xdata()) == N_IMAGES,
            "native initial projection must retain all 32 images")

    for panel_index, axis in enumerate(axes[1:], start=1):
        ode = find_single_path_line(axis, "ODE-NEB")
        ivm = find_single_path_line(axis, "IR-IVM")
        styles.append(apply_reference_path_style(
            ode, "ODE-NEB", native_projections[panel_index]["ODE-NEB"]
        ))
        styles.append(apply_reference_path_style(
            ivm, "IVM", native_projections[panel_index]["IR-IVM"]
        ))
        require(len(ode.get_xdata()) == N_IMAGES and len(ivm.get_xdata()) == N_IMAGES,
                "native path projection must retain all 32 images")

    # Rebuild only the existing legend handles so the legend represents the
    # newly requested path artists.  Labels, order, placement, frame, and BB
    # colors remain unchanged.
    old_legend = axes[-1].get_legend()
    require(old_legend is not None, "registered BB legend is missing")
    old_legend.remove()
    ode_line = find_single_path_line(axes[-1], "ODE-NEB")
    ivm_line = find_single_path_line(axes[-1], "IR-IVM")
    handles = [
        Line2D(
            [], [], color="#D55E00", linestyle="-", linewidth=1.45,
            marker="o", markersize=2.15, label="ODE-NEB",
        ),
        Line2D(
            [], [], color="#0072B2", linestyle="-", linewidth=1.75,
            marker="s", markersize=2.15, label="IR-IVM",
        ),
    ]
    axes[-1].legend(
        handles=handles, frameon=True, facecolor="white", edgecolor="#B3B3B3",
        framealpha=0.90, loc="upper right", handlelength=2.0,
    )

    all_n = np.concatenate([
        projected[:, 1] for panel in native_projections for projected in panel.values()
    ])
    original_ylim = [-0.12, 0.72]
    expanded_ylim = [original_ylim[0], original_ylim[1]]
    if float(all_n.min()) < original_ylim[0]:
        expanded_ylim[0] = math.floor(float(all_n.min()) * 100.0) / 100.0
    if float(all_n.max()) > original_ylim[1]:
        expanded_ylim[1] = math.ceil(float(all_n.max()) * 100.0) / 100.0
    for axis in axes:
        axis.set_ylim(*expanded_ylim)

    for axis in axes:
        axis.set_box_aspect(1.0)

    # Keep the registered horizontal layout.  Align the colorbar vertically
    # to the now-square panels without changing its x position or width.
    figure.canvas.draw()
    panel_position = axes[-1].get_position()
    color_position = color_axis.get_position()
    color_axis.set_position([
        color_position.x0, panel_position.y0,
        color_position.width, panel_position.height,
    ])
    figure.canvas.draw()

    box_ratios = []
    for index, axis in enumerate(axes):
        extent = axis.get_window_extent()
        ratio = float(extent.width / extent.height)
        require(abs(ratio - 1.0) <= 1.0e-10,
                f"panel {index} is not square: width/height={ratio}")
        box_ratios.append(ratio)

    return {
        "path_style_reference_pdf": binding(PATH_STYLE_PDF),
        "path_style_reference_source": binding(PATH_STYLE_SOURCE),
        "path_artists": styles,
        "formal_bb_marker_shapes_colors_and_linewidths": True,
        "user_requested_markevery": 1,
        "native_projected_points_per_path": N_IMAGES,
        "underlying_saved_images_per_compared_path": 32,
        "native_images_directly_projected": True,
        "path_resampled_or_densified": False,
        "original_plot_ylim_rms_A": original_ylim,
        "final_plot_ylim_rms_A": expanded_ylim,
        "native_projected_n_range_A": [float(all_n.min()), float(all_n.max())],
        "panel_box_aspect": 1.0,
        "panel_width_over_height": box_ratios,
        "colorbar_vertical_extent_aligned_to_square_panels": True,
        "colors_changed": False,
        "saved_surface_and_native_path_values_changed": False,
        "display_projection_changed_from_15_resampled_to_32_native": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-curvature2", action="store_true")
    args = parser.parse_args()
    require(sha256(BASE_WRAPPER) == BASE_WRAPPER_SHA256,
            "R2 plotting wrapper changed")
    require(sha256(PATH_STYLE_PDF) == PATH_STYLE_PDF_SHA256,
            "path-style reference PDF changed")
    require(sha256(PATH_STYLE_SOURCE) == PATH_STYLE_SOURCE_SHA256,
            "path-style reference source changed")

    wrapper = load_file_module("bb_r2_registered_wrapper", BASE_WRAPPER)
    _adapter, legacy, analysis, inputs, report = wrapper.load_validated_inputs()
    require(CONTOUR_COUNT_ROUNDING_IS_EXACT,
            "requested contour rounding no longer gives 17 bands")
    require(legacy.FILLED_CONTOUR_BANDS == PREVIOUS_FILLED_CONTOUR_BANDS,
            "inherited filled-contour count differs from the audited 28-band base")
    legacy.FILLED_CONTOUR_BANDS = REQUESTED_FILLED_CONTOUR_BANDS
    analysis.BASE_MODULE.LogQuadraticNorm = SegmentedMidrangeNorm
    snapshots = path_snapshot_records(inputs)
    native_projections, native_projection_audit = build_native_projections(inputs)

    outputs = []
    curvatures = (2.0,) if args.preview_curvature2 else CURVATURES
    for curvature in curvatures:
        figure, inherited_display = legacy.plot(
            analysis, *inputs, color_curvature=curvature
        )
        inherited_display["color_norm"] = {
            "name": "SegmentedMidrangeNorm",
            "anchors_kcal_per_mol": [0.0, LOW_ENERGY_ANCHOR,
                                     HIGH_ENERGY_ANCHOR,
                                     inherited_display["display_maximum_kcal_per_mol"]],
            "mapped_color_coordinates": [0.0, LOW_COLOR_FRACTION,
                                         MIDDLE_COLOR_END, 1.0],
            "segment_color_fractions": [LOW_COLOR_FRACTION,
                                        MIDDLE_COLOR_FRACTION,
                                        HIGH_COLOR_FRACTION],
            "middle_segment_curvature": curvature,
            "curvature_applies_only_to_50_240_segment": True,
            "low_and_high_segments_linear": True,
        }
        try:
            style_audit = adapt_figure(figure, native_projections)
            inherited_display["plot_ylim_rms_A"] = style_audit[
                "final_plot_ylim_rms_A"
            ]
            output = (PREVIEW_OUTPUT if args.preview_curvature2
                      else output_for_curvature(curvature)).resolve()
            legacy.atomic_save_pdf(figure, output)
        finally:
            plt.close(figure)

        require(output.is_file() and output.stat().st_size > 0,
                f"missing output for curvature {curvature:g}")
        provenance = {
            "schema": "bb-irivm-r2-midband-native32-curvature-variant-v3",
            "status": "PASS_ZERO_PHYSICS_PLOT_VARIANT",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "color_curvature": curvature,
            "output": binding(output),
            "base_plot_wrapper": binding(BASE_WRAPPER),
            "inherited_display": inherited_display,
            "requested_style_audit": style_audit,
            "native_projection_audit": native_projection_audit,
            "unchanged_scientific_inputs": {
                "surface_grid": binding(wrapper.SURFACE_GRID),
                "surface_sources": binding(wrapper.SURFACE_SOURCES),
                "path_snapshots": snapshots,
                "fresh_converged": report["fresh_converged"],
                "fresh_maxstep": report["fresh_maxstep"],
            },
            "physical_calls": 0,
            "optimization_N_F_contribution": 0,
            "formally_reportable": False,
        }
        provenance_path = output.with_suffix(".provenance.json")
        atomic_json(provenance_path, provenance)
        outputs.append({
            "curvature": curvature,
            "pdf": binding(output),
            "provenance": binding(provenance_path),
            "panel_width_over_height": style_audit["panel_width_over_height"],
        })

    if args.preview_curvature2:
        require(len(outputs) == 1 and outputs[0]["curvature"] == 2.0,
                "curvature-2 preview output differs")
        print(json.dumps({
            "status": "PASS_ZERO_PHYSICS_CURVATURE2_PREVIEW",
            "pdf": outputs[0]["pdf"],
            "provenance": outputs[0]["provenance"],
            "native_projected_points_per_path": N_IMAGES,
            "plot_ylim_rms_A": style_audit["final_plot_ylim_rms_A"],
        }, indent=2, sort_keys=True, allow_nan=False))
        return 0

    require(len(outputs) == 7, "curvature sweep must contain seven PDFs")
    manifest = {
        "schema": "bb-irivm-r2-midband-native32-curvature-sweep-v3",
        "status": "PASS_SEVEN_ZERO_PHYSICS_PDF_VARIANTS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested_curvatures": list(CURVATURES),
        "outputs": outputs,
        "code": binding(Path(__file__)),
        "base_plot_wrapper": binding(BASE_WRAPPER),
        "path_style_reference_pdf": binding(PATH_STYLE_PDF),
        "path_style_reference_source": binding(PATH_STYLE_SOURCE),
        "shared_scientific_inputs": {
            "surface_grid": binding(wrapper.SURFACE_GRID),
            "surface_sources": binding(wrapper.SURFACE_SOURCES),
            "path_snapshots": snapshots,
            "fresh_converged": report["fresh_converged"],
            "fresh_maxstep": report["fresh_maxstep"],
        },
        "all_variants_use_identical_data": True,
        "path_markevery": 1,
        "native_projected_points_per_path": N_IMAGES,
        "path_resampled_or_densified": False,
        "native_projection_audit": native_projection_audit,
        "color_norm": {
            "name": "SegmentedMidrangeNorm",
            "anchors_kcal_per_mol_except_vmax": [0.0, LOW_ENERGY_ANCHOR,
                                                 HIGH_ENERGY_ANCHOR],
            "mapped_color_coordinates": [0.0, LOW_COLOR_FRACTION,
                                         MIDDLE_COLOR_END, 1.0],
            "segment_color_fractions": [LOW_COLOR_FRACTION,
                                        MIDDLE_COLOR_FRACTION,
                                        HIGH_COLOR_FRACTION],
            "curvature_applies_only_to_50_240_segment": True,
        },
        "previous_filled_contour_bands": PREVIOUS_FILLED_CONTOUR_BANDS,
        "requested_contour_density_factor": CONTOUR_DENSITY_FACTOR,
        "filled_contour_bands": REQUESTED_FILLED_CONTOUR_BANDS,
        "filled_contour_boundaries": REQUESTED_FILLED_CONTOUR_BANDS + 1,
        "internal_white_contours": REQUESTED_FILLED_CONTOUR_BANDS - 1,
        "all_variants_have_square_panels": True,
        "colors_changed": False,
        "physical_calls": 0,
    }
    atomic_json(MANIFEST, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "manifest": str(MANIFEST),
        "outputs": [{"curvature": row["curvature"], "pdf": row["pdf"]["path"]}
                    for row in outputs],
    }, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
