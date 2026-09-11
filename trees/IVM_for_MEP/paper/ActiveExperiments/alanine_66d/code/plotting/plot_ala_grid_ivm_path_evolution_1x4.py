#!/usr/bin/env python3
"""Draw the registered Ala path evolution used beside manuscript Fig. 6.

The IVM path is the registered inexact, reparameterized run with
alpha=10, r=0.7, and m_max=5.  ODE--NEB is the same k=50 frozen baseline
used in Fig. 6.  The middle panels use the latest actual saved paths at or
before 100 and 1000 force evaluations.  The last panel compares each method's
final reported path: IVM's last formal complete mesh-eligible state and the
terminal saved ODE--NEB path.  No special endpoint/minimum symbols are drawn.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[5]
STUDY = REPO / "python/alanine_ivm_clean_four_level_20260801"
LEGACY = REPO / "python/alanine_v10_chignolin_v9_fixed_alpha_20260730"
OUTPUT_DIR = REPO / "output/pdf"
STEM = "alanine66d_grid_ivm_path_evolution_1x4"

GRID_PREREGISTRATION = (
    STUDY
    / "reports/stage4_ala_discrete_grid_metric_campaign_v2_"
    "1787608620070816000_preregistered.json"
)
COMMON_INITIAL = STUDY / "inputs/common_initial_path_equidistant.npz"
ODE_RESULT = (
    LEGACY
    / "formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz"
)
RAMA_CACHE = LEGACY / "formal_figure_inputs/ala2_scan_cache.pkl"
PROJECTION_SOURCE = LEGACY / "src/plot_v49_formal_ala_1x3.py"
FIGURE6_MANIFEST = REPO / "output/pdf/alanine66d_ivm_geometric_action_1x3_manifest.json"

CELL_ID = "a736dd230285d61c"
EXPECTED_CONTROLS = {
    "alpha": 10.0,
    "residual_ratio": 0.7,
    "inner_max_steps": 5,
    "total_internal_nf_budget": 5000,
}
EXPECTED_HASHES = {
    GRID_PREREGISTRATION: "5fcfad3228e72c267ea1d768dd6f677528ae1fc8e8c6f180ea0d3498503b98a3",
    COMMON_INITIAL: "4a08066e1f0a89ad1d0ee6ecc442d7255d511217576d1ff65e9cb5e2164d891b",
    ODE_RESULT: "935561709ecbdfe87af31b78b45335a66545356b3e62f331a4a9fbb8125d9a09",
    RAMA_CACHE: "1eab9d20f7767a264b5164dadb8b2b5dd6f759b3de37dabef217933375a5ce9d",
    PROJECTION_SOURCE: "4af8feee0843206f17fc6f4b1b19eb52743719a4f553821a8a041ac9d06626dd",
}
EXPECTED_IVM_RESULT_SHA256 = (
    "0da1bbab469de81ca5b9229c4ebef656b7688ccb79c822ec14f0f4c92226d161"
)
EXPECTED_IVM_SELECTION = (
    (
        100,
        100,
        4,
        "1205bcd469d3c155d6ffbc4556177bbee7a90702fe6c5445f6b39262546a3756",
        False,
        True,
    ),
    (
        1000,
        993,
        52,
        "1e33327d9c3ed467e5b0d3c6e1a2860f9f4b09dc464fc6f36fab48cfe3484836",
        False,
        False,
    ),
)
EXPECTED_ODE_SELECTION = (
    (100, 82, 2),
    (1000, 982, 41),
)
EXPECTED_FINAL_SELECTION = {
    "ivm": (3587, 235),
    "ode_neb": (100022, 4637),
}

EV_TO_KCAL = 23.0605
COLOR_ODE = "#D55E00"
COLOR_IVM = "#0072B2"
COLOR_INITIAL = "#888888"
MARKER_EDGEWIDTH = 0.4


class _IgnoredAseObject:
    """Placeholder for ASE objects stored beside the numeric PES arrays."""

    def __new__(cls, *_args, **_kwargs):
        return super().__new__(cls)

    def __setstate__(self, state):
        self.state = state


class _ContourCacheUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("ase."):
            return _IgnoredAseObject
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_projection_module():
    spec = importlib.util.spec_from_file_location("ala_path_projection", PROJECTION_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load projection source: {PROJECTION_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.8,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def validate_frozen_inputs() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Frozen input drift for {path}: expected {expected}, got {observed}"
            )


def load_panels():
    preregistration = json.loads(GRID_PREREGISTRATION.read_text(encoding="utf-8"))
    matches = [cell for cell in preregistration["cells"] if cell["cell_id"] == CELL_ID]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one registered cell {CELL_ID}")
    cell = matches[0]
    if cell["family"] != "inexact_reparam" or cell["controls"] != EXPECTED_CONTROLS:
        raise RuntimeError(f"Selected IVM controls changed: {cell['controls']}")

    ivm_result = Path(cell["source_result"])
    if (
        str(cell["source_result_sha256"]) != EXPECTED_IVM_RESULT_SHA256
        or sha256(ivm_result) != EXPECTED_IVM_RESULT_SHA256
    ):
        raise RuntimeError("Selected IVM result hash changed")

    state_by_nf = {
        int(state["optimization_nf"]): state
        for state in cell["states"]
        if state["formal_complete"] and state["mesh_eligible"]
    }
    if len(state_by_nf) != 22:
        raise RuntimeError("Expected 22 formal complete mesh-eligible IVM states")

    with np.load(COMMON_INITIAL, allow_pickle=False) as saved:
        initial_path = np.asarray(saved["initial_path"], dtype=float)
    with np.load(ivm_result, allow_pickle=False) as saved:
        ivm_initial = np.asarray(saved["initial_path"], dtype=float)
        ivm_paths = np.asarray(saved["stored_paths"], dtype=float)
        ivm_calls = np.asarray(saved["stored_path_calls"], dtype=int)
        ivm_nf = np.asarray(saved["total_internal_joint_evaluations"], dtype=int)
        ivm_committed = np.asarray(saved["reparam_committed"], dtype=bool)
        ivm_cv = np.asarray(saved["segment_length_cv"], dtype=float)
        ivm_ratio = np.asarray(saved["segment_length_ratio"], dtype=float)
    np.testing.assert_array_equal(initial_path, ivm_initial)

    panels = [
        {
            "panel": "a",
            "title": r"$N_F=0$",
            "budget_nf": 0,
            "entries": [
                {
                    "method": "Initial (IDPP)",
                    "actual_nf": 0,
                    "source_record_index": None,
                    "source_call": 0,
                    "path": initial_path,
                    "path_sha256": array_sha256(initial_path),
                    "source_file": COMMON_INITIAL,
                    "formal_complete": True,
                    "mesh_eligible": True,
                }
            ],
        }
    ]

    with np.load(ODE_RESULT, allow_pickle=False) as saved:
        ode_paths = np.asarray(saved["images_seq"], dtype=float)
        ode_calls = np.asarray(saved["count_seq"], dtype=int)
    ode_nf = 20 * ode_calls + 2

    for panel_index, (ivm_expected, ode_expected) in enumerate(
        zip(EXPECTED_IVM_SELECTION, EXPECTED_ODE_SELECTION), start=1
    ):
        (
            budget,
            actual_ivm_nf,
            ivm_index,
            expected_ivm_path_hash,
            expected_formal_complete,
            expected_mesh_eligible,
        ) = ivm_expected
        if int(ivm_nf[ivm_index]) != actual_ivm_nf:
            raise RuntimeError(f"IVM cost ledger changed at record {ivm_index}")
        observed_ivm_path_hash = array_sha256(ivm_paths[ivm_index])
        if observed_ivm_path_hash != expected_ivm_path_hash:
            raise RuntimeError(f"IVM path hash changed at record {ivm_index}")
        formal_complete = bool(ivm_index == 0 or ivm_committed[ivm_index])
        mesh_eligible = bool(ivm_cv[ivm_index] <= 0.0065 and ivm_ratio[ivm_index] <= 1.03)
        if (formal_complete, mesh_eligible) != (
            expected_formal_complete,
            expected_mesh_eligible,
        ):
            raise RuntimeError(f"IVM intermediate-state status changed at {ivm_index}")

        ode_budget, expected_ode_nf, expected_ode_index = ode_expected
        if ode_budget != budget:
            raise RuntimeError("IVM/ODE panel budgets are not aligned")
        eligible = np.flatnonzero(ode_nf <= budget)
        if not eligible.size:
            raise RuntimeError(f"No ODE--NEB path at or before N_F={budget}")
        ode_index = int(eligible[-1])
        actual_ode_nf = int(ode_nf[ode_index])
        if (actual_ode_nf, ode_index) != (expected_ode_nf, expected_ode_index):
            raise RuntimeError(
                f"ODE--NEB checkpoint changed for budget {budget}: "
                f"{actual_ode_nf}, {ode_index}"
            )

        panels.append(
            {
                "panel": chr(ord("a") + panel_index),
                "title": rf"$N_F\leq {budget:,}$",
                "budget_nf": budget,
                "entries": [
                    {
                        "method": "ODE-NEB",
                        "actual_nf": actual_ode_nf,
                        "source_record_index": ode_index,
                        "source_call": int(ode_calls[ode_index]),
                        "path": ode_paths[ode_index],
                        "path_sha256": array_sha256(ode_paths[ode_index]),
                        "source_file": ODE_RESULT,
                        "formal_complete": True,
                        "mesh_eligible": None,
                    },
                    {
                        "method": "IVM",
                        "actual_nf": actual_ivm_nf,
                        "source_record_index": ivm_index,
                        "source_call": int(ivm_calls[ivm_index]),
                        "path": ivm_paths[ivm_index],
                        "path_sha256": observed_ivm_path_hash,
                        "source_file": ivm_result,
                        "formal_complete": formal_complete,
                        "mesh_eligible": mesh_eligible,
                    },
                ],
            }
        )

    final_ivm_nf, final_ivm_index = EXPECTED_FINAL_SELECTION["ivm"]
    final_ode_nf, final_ode_index = EXPECTED_FINAL_SELECTION["ode_neb"]
    final_state = state_by_nf.get(final_ivm_nf)
    if (
        final_state is None
        or int(final_state["trajectory_index"]) != final_ivm_index
        or int(ivm_nf[final_ivm_index]) != final_ivm_nf
        or array_sha256(ivm_paths[final_ivm_index])
        != str(final_state["path_sha256"])
    ):
        raise RuntimeError("Final formal complete mesh-eligible IVM path changed")
    if (
        final_ode_index != len(ode_paths) - 1
        or int(ode_nf[final_ode_index]) != final_ode_nf
    ):
        raise RuntimeError("Terminal ODE--NEB path changed")
    panels.append(
        {
            "panel": "d",
            "title": "Final paths",
            "budget_nf": "final",
            "entries": [
                {
                    "method": "ODE-NEB",
                    "actual_nf": final_ode_nf,
                    "source_record_index": final_ode_index,
                    "source_call": int(ode_calls[final_ode_index]),
                    "path": ode_paths[final_ode_index],
                    "path_sha256": array_sha256(ode_paths[final_ode_index]),
                    "source_file": ODE_RESULT,
                    "formal_complete": True,
                    "mesh_eligible": None,
                },
                {
                    "method": "IVM",
                    "actual_nf": final_ivm_nf,
                    "source_record_index": final_ivm_index,
                    "source_call": int(ivm_calls[final_ivm_index]),
                    "path": ivm_paths[final_ivm_index],
                    "path_sha256": str(final_state["path_sha256"]),
                    "source_file": ivm_result,
                    "formal_complete": True,
                    "mesh_eligible": True,
                },
            ],
        }
    )
    return panels, cell, ivm_result


def path_style(method: str) -> dict:
    if method == "IVM":
        return {
            "color": COLOR_IVM,
            "linestyle": "-",
            "linewidth": 1.85,
            "marker": "o",
            "markersize": 3.1,
            "zorder": 5,
        }
    if method == "ODE-NEB":
        return {
            "color": COLOR_ODE,
            "linestyle": "--",
            "linewidth": 1.65,
            "marker": "s",
            "markersize": 2.9,
            "zorder": 4,
        }
    return {
        "color": COLOR_INITIAL,
        "linestyle": "--",
        "linewidth": 1.6,
        "marker": "D",
        "markersize": 2.9,
        "zorder": 3,
    }


def draw_figure(projection, rama, panels, output: Path) -> None:
    phi_grid, psi_grid = np.meshgrid(
        np.asarray(rama["phi_g"], dtype=float),
        np.asarray(rama["psi_g"], dtype=float),
        indexing="ij",
    )
    energy_grid = (
        np.asarray(rama["PES"], dtype=float)
        - float(np.nanmin(np.asarray(rama["PES"], dtype=float)))
    ) * EV_TO_KCAL

    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=200)
    figure.subplots_adjust(
        left=0.04,
        right=0.96,
        bottom=0.12,
        top=0.90,
        wspace=0.18,
        hspace=0.0,
    )
    contour = None
    for axis, panel in zip(axes, panels):
        contour = axis.contourf(
            phi_grid,
            psi_grid,
            energy_grid,
            levels=np.linspace(0.0, 15.0, 16),
            cmap="plasma",
            extend="max",
        )
        axis.contour(
            phi_grid,
            psi_grid,
            energy_grid,
            levels=np.linspace(0.0, 15.0, 8),
            colors="white",
            linewidths=0.3,
            alpha=0.5,
        )
        for entry in panel["entries"]:
            phi, psi = projection.path_to_dihedrals(entry["path"])
            style = path_style(entry["method"])
            axis.plot(
                phi,
                psi,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=MARKER_EDGEWIDTH,
                markevery=1,
                label=entry["method"],
                zorder=style["zorder"],
            )
        axis.set_xlim(-180, 0)
        axis.set_ylim(0, 180)
        axis.set_xticks(range(-180, 1, 60))
        axis.set_yticks(range(0, 181, 60))
        axis.set_xlabel(r"$\phi$ (deg)")
        axis.set_ylabel(r"$\psi$ (deg)")
        axis.set_aspect("equal")
        axis.set_title(panel["title"], fontsize=10)
        axis.legend(
            loc="upper right",
            fontsize=6.5,
            facecolor="white",
            edgecolor="black",
            framealpha=0.9,
        )

    if contour is None:
        raise RuntimeError("No contour artist was created")
    colorbar = figure.colorbar(
        contour, ax=axes, fraction=0.03, pad=0.02, location="right"
    )
    colorbar.set_label(r"$\Delta E$ (kcal mol$^{-1}$)", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="pdf",
        metadata={
            "Title": "Alanine 66D registered IVM path evolution",
            "Subject": "Same selected IVM and ODE--NEB experiments as manuscript Fig. 6",
        },
    )
    plt.close(figure)


def source_rows(projection, panels):
    rows = []
    for panel in panels:
        for entry in panel["entries"]:
            phi, psi = projection.path_to_dihedrals(entry["path"])
            source_file = Path(entry["source_file"])
            for image_index, (phi_value, psi_value) in enumerate(zip(phi, psi)):
                rows.append(
                    {
                        "panel": panel["panel"],
                        "panel_title": panel["title"],
                        "budget_NF": panel["budget_nf"],
                        "method": entry["method"],
                        "actual_NF": entry["actual_nf"],
                        "source_record_index": (
                            ""
                            if entry["source_record_index"] is None
                            else entry["source_record_index"]
                        ),
                        "source_call": entry["source_call"],
                        "image_index": image_index,
                        "phi_deg": float(phi_value),
                        "psi_deg": float(psi_value),
                        "selected_path_sha256": entry["path_sha256"],
                        "source_file": str(source_file),
                        "source_file_sha256": sha256(source_file),
                        "formal_complete": entry["formal_complete"],
                        "mesh_eligible": (
                            "" if entry["mesh_eligible"] is None else entry["mesh_eligible"]
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_frozen_inputs()
    configure_style()
    projection = load_projection_module()
    with RAMA_CACHE.open("rb") as handle:
        rama = _ContourCacheUnpickler(handle).load()
    panels, cell, ivm_result = load_panels()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output_dir / f"{STEM}.pdf"
    csv_path = args.output_dir / f"{STEM}_source.csv"
    manifest_path = args.output_dir / f"{STEM}_manifest.json"
    existing = [path for path in (pdf_path, csv_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {existing}")

    draw_figure(projection, rama, panels, pdf_path)
    rows = source_rows(projection, panels)
    write_csv(csv_path, rows)
    manifest = {
        "status": "registered_and_cited_in_paper0819",
        "purpose": "Ramachandran projection of the same selected IVM and ODE--NEB experiments as manuscript Fig. 6",
        "selected_cell_id": CELL_ID,
        "selected_ivm_controls": cell["controls"],
        "selected_ivm_result": {
            "path": str(ivm_result),
            "sha256": sha256(ivm_result),
        },
        "panel_policy": {
            "initial": "selected IVM IDPP path before charged optimization evaluations",
            "milestones": [
                "latest saved paths at or before N_F=100",
                "latest saved paths at or before N_F=1000",
                "final reported paths: last formal complete mesh-eligible IVM state and terminal saved ODE--NEB path",
            ],
            "intermediate_selection": "latest saved path of each method at or before the displayed common force budget",
            "panels": [
                {
                    "panel": panel["panel"],
                    "title": panel["title"],
                    "budget_NF": panel["budget_nf"],
                    "entries": [
                        {
                            "method": entry["method"],
                            "actual_NF": entry["actual_nf"],
                            "source_record_index": entry["source_record_index"],
                            "source_call": entry["source_call"],
                            "selected_path_sha256": entry["path_sha256"],
                            "formal_complete": entry["formal_complete"],
                            "mesh_eligible": entry["mesh_eligible"],
                        }
                        for entry in panel["entries"]
                    ],
                }
                for panel in panels
            ],
        },
        "style": {
            "background": "plasma, unchanged from the superseded manuscript Fig. 5",
            "ivm": path_style("IVM"),
            "ode_neb": path_style("ODE-NEB"),
            "initial": path_style("Initial (IDPP)"),
            "marker_facecolor": "white",
            "marker_edgewidth": MARKER_EDGEWIDTH,
            "markevery": 1,
            "special_endpoint_or_minimum_markers": "none",
        },
        "projection": {
            "source": str(PROJECTION_SOURCE),
            "source_sha256": sha256(PROJECTION_SOURCE),
            "phi_atom_indices": [4, 6, 8, 14],
            "psi_atom_indices": [6, 8, 14, 16],
            "wrap_degrees": "[-180, 180)",
        },
        "inputs": {
            "grid_preregistration": {
                "path": str(GRID_PREREGISTRATION),
                "sha256": sha256(GRID_PREREGISTRATION),
            },
            "common_initial": {
                "path": str(COMMON_INITIAL),
                "sha256": sha256(COMMON_INITIAL),
            },
            "ode_result": {
                "path": str(ODE_RESULT),
                "sha256": sha256(ODE_RESULT),
            },
            "ramachandran_cache": {
                "path": str(RAMA_CACHE),
                "sha256": sha256(RAMA_CACHE),
            },
            "figure6_manifest": {
                "path": str(FIGURE6_MANIFEST),
                "sha256": sha256(FIGURE6_MANIFEST),
            },
            "plotter": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256(Path(__file__).resolve()),
            },
        },
        "outputs": {
            "pdf": {"path": str(pdf_path), "sha256": sha256(pdf_path)},
            "source_csv": {"path": str(csv_path), "sha256": sha256(csv_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pdf": str(pdf_path), "panels": manifest["panel_policy"]["panels"]}, indent=2))


if __name__ == "__main__":
    main()
