#!/usr/bin/env python3
"""Publication Ala energy profile and ball-and-stick configurations.

Reuse the exact final paths in the registered path-evolution figure and the
already audited native-image energy evaluations in the geometric-action
cache. This renderer does not evaluate a potential or force.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
STUDY = REPO / "python/alanine_ivm_clean_four_level_20260801"
STEM = "alanine66d_energy_and_configurations"
OUT = REPO / "output/pdf"
ASSETS = OUT / f"{STEM}_assets"
SNAPSHOT_RENDERER_ARCHIVE = ASSETS / "render_ala_ballstick_snapshot_source.py"
PATH_PLOTTER = HERE / "plot_ala_grid_ivm_path_evolution_1x4.py"
RENDERER = HERE / "render_ala_ballstick.py"
ENERGY_CACHE = STUDY / "runs/diagnostics/ala_discrete_geometric_action_v5.npz"
ENERGY_AUDIT = ENERGY_CACHE.with_suffix(".audit.json")
ENERGY_CONTRACT = STUDY / "reports/ala_discrete_geometric_action_v5_preregistered.json"
TOPOLOGY = STUDY / "inputs/ala2_fixed.pdb"
EXPECTED_CACHE_SHA = "d19d8efacf4b48b31ace4a87ff6857b04f65c6be9b3a1172db9e4506f58076eb"
EXPECTED_CONTRACT_SHA = "48eb99096e6aa1e569f6ef15a3b5c86dac0affcbaea2c563df42f60dcd55ffba"
EXPECTED_PATH_PLOTTER_SHA = "b5252cd9ce40a8ec23ae9add9dcc8e9cdeb5e75ba6feed26950c16ea77808313"
EV_TO_KCAL = 23.0605
IVM_COLOR, ODE_COLOR = "#0072B2", "#D55E00"
NUMBER_COLOR = "#005A86"
NUMBER_SIZE = 9.5
PHI_COLOR, PSI_COLOR = "#087F77", "#834A9A"
PLANE_COLORS = ["#2A9D8F", "#D9A234", "#A16AC0"]
PLANE_ATOMS = [[4, 6, 8], [6, 8, 14], [8, 14, 16]]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def load_data():
    if sha256(PATH_PLOTTER) != EXPECTED_PATH_PLOTTER_SHA:
        raise RuntimeError("Registered path-selection code changed")
    source = module(PATH_PLOTTER, "ala_registered_paths")
    source.validate_frozen_inputs()
    panels, cell, _ = source.load_panels()
    projection = source.load_projection_module()
    if tuple(projection.PHI_IDX) != (4, 6, 8, 14) or tuple(projection.PSI_IDX) != (6, 8, 14, 16):
        raise RuntimeError("Article dihedral convention changed")
    audit = json.loads(ENERGY_AUDIT.read_text())
    if (audit["status"] != "validated" or sha256(ENERGY_CACHE) != EXPECTED_CACHE_SHA
            or audit["result_sha256"] != EXPECTED_CACHE_SHA
            or sha256(ENERGY_CONTRACT) != EXPECTED_CONTRACT_SHA
            or audit["contract_sha256"] != EXPECTED_CONTRACT_SHA
            or audit["diagnostic_cost_excluded_from_optimization_axis"] is not True):
        raise RuntimeError("Audited energy cache identity changed")
    profiles = []
    with np.load(ENERGY_CACHE, allow_pickle=False) as saved:
        for entry in panels[-1]["entries"]:
            method_id = "ivm" if entry["method"] == "IVM" else "ode_neb"
            match = np.flatnonzero(
                (saved["method"] == method_id)
                & (saved["source_record_index"] == entry["source_record_index"])
                & (saved["number_of_forces"] == entry["actual_nf"])
                & (saved["selected_path_sha256"] == entry["path_sha256"])
            )
            if len(match) != 1:
                raise RuntimeError("No unique cached energy for the selected path")
            row = int(match[0])
            if saved["source_result_sha256"][row] != sha256(entry["source_file"]):
                raise RuntimeError("Energy cache is linked to another optimizer result")
            path = np.asarray(entry["path"], dtype=np.float64)
            energy = np.asarray(saved["energy_ev"][row], dtype=np.float64)
            lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
            np.testing.assert_allclose(lengths, saved["segment_length_angstrom"][row], atol=1e-12, rtol=1e-12)
            progress = np.r_[0.0, np.cumsum(lengths)] / lengths.sum()
            phi, psi = projection.path_to_dihedrals(path)
            profiles.append({
                **entry, "source_file": str(entry["source_file"]), "method_id": method_id,
                "energy_cache_index": row, "energy_ev": energy,
                "relative_energy_kcal": (energy - energy[0]) * EV_TO_KCAL,
                "progress": progress, "phi_deg": phi, "psi_deg": psi,
                "path": path.reshape(22, 22, 3),
            })
    ivm = next(p for p in profiles if p["method_id"] == "ivm")
    peak = int(np.argmax(ivm["relative_energy_kcal"]))
    rising = np.flatnonzero(ivm["relative_energy_kcal"][:peak] >= ivm["relative_energy_kcal"][peak] / 2)
    if len(rising) == 0 or peak in (0, 21):
        raise RuntimeError("Expected an internal sampled maximum and rising-branch image")
    selected = [0, int(rising[0]), peak, 21]
    if selected != [0, 8, 12, 21]:
        raise RuntimeError(f"Frozen representative image selection changed: {selected}")
    return profiles, selected, source, audit


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "mathtext.fontset": "stixsans", "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    })


def draw(profiles, selected, renders, pdf, png):
    style()
    ivm = next(p for p in profiles if p["method_id"] == "ivm")
    height_mm = 112
    y_top = lambda old: 1 - (1-old)*112/height_mm
    fig = plt.figure(figsize=(180 / 25.4, height_mm / 25.4), facecolor="white")
    fig.text(.022, y_top(.971), "a", weight="bold", fontsize=11, va="top")
    titles = ["Reactant", "Rising branch", "Transition state", "Product"]
    lefts = [.055, .290, .525, .760]
    width = .221
    for j, (idx, title, rendered, left) in enumerate(zip(selected, titles, renders["images"], lefts)):
        # Match the renderer's 4:3 canvas in physical units, without stretching.
        ax = fig.add_axes([left, y_top(.615), width, width * 180 / height_mm * .75])
        ax.imshow(plt.imread(rendered["path"]), extent=(0, 1, 0, 1), aspect="auto")
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis("off")
        draw_dihedrals_on_configuration(ax, rendered)
        title_artist = fig.text(left + width/2, y_top(.941), title, ha="center", va="top", fontsize=9, weight="bold")
        title_box = title_artist.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        fig.text(left + width/2, y_top(.907), f"Image {idx + 1} of 22", ha="center", va="top", fontsize=6.8, color="#727A80")
        # Numbers associate actual configurations with the sampled energy profile.
        fig.text(title_box.x0 - 13 / (fig.get_figwidth()*72),
                 (title_box.y0 + title_box.y1)/2, str(j+1), ha="center", va="center",
                 color="white", fontsize=NUMBER_SIZE, weight="bold",
                 bbox={"boxstyle":"circle,pad=.30", "facecolor":NUMBER_COLOR,
                       "edgecolor":"white", "linewidth":.8})
        fig.text(left + width/2, y_top(.580), rf"$\phi={ivm['phi_deg'][idx]:.1f}^\circ$", ha="center", color=PHI_COLOR, fontsize=8.4)
        fig.text(left + width/2, y_top(.547), rf"$\psi={ivm['psi_deg'][idx]:.1f}^\circ$", ha="center", color=PSI_COLOR, fontsize=8.4)
    # Restrained element key, separate from the two method colors.
    for x, label, color in zip([.785, .837, .889, .941], ["C", "N", "O", "H"],
                               ["#4B535E", "#376DC4", "#D6423B", "#DCE1E8"]):
        fig.add_artist(Line2D([x], [y_top(.501)], transform=fig.transFigure, marker="o",
                             ms=3.6, mfc=color, mec="#7B838B", mew=.3, ls="none"))
        fig.text(x+.009, y_top(.501), label, fontsize=6.4, va="center", color="#414A52")
    for x, label, color in zip([.085, .312, .501],
            [r"$C_{\mathrm{ACE}}-N-C_\alpha$", r"$N-C_\alpha-C$", r"$C_\alpha-C-N_{\mathrm{NME}}$"],
            PLANE_COLORS):
        fig.add_artist(Polygon([[x,.492],[x+.016,.492],[x+.013,.510]],
                              transform=fig.transFigure, facecolor=color, edgecolor=color,
                              alpha=.60, linewidth=.5))
        fig.text(x+.024, .501, label, fontsize=6.7, va="center", color="#414A52")
    fig.text(.023, .465, "b", weight="bold", fontsize=11, va="top")
    ax = fig.add_axes([.103, .123*112/height_mm, .871, .326*112/height_mm])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", lw=.4, color="#DFE4E8")
    for p in profiles:
        is_ivm = p["method_id"] == "ivm"
        nf_text = f"{p['actual_nf']:,}".replace(",", "{,}")
        ax.plot(p["progress"], p["relative_energy_kcal"], color=IVM_COLOR if is_ivm else ODE_COLOR,
                lw=1.35 if is_ivm else 1.05, ls="-" if is_ivm else (0, (4, 2.5)),
                marker="o" if is_ivm else "s", ms=2.9 if is_ivm else 2.7,
                mfc="white", mew=.7, zorder=4 if is_ivm else 3,
                label=rf"{p['method']}  ($N_F={nf_text}$)")
    for j, idx in enumerate(selected):
        x,y=ivm["progress"][idx],ivm["relative_energy_kcal"][idx]
        ax.plot([x,x],[-.14,y], color="#C5D5DF", lw=.65, ls=(0,(2,3)), zorder=1)
        ax.plot(x,y,"o",color=IVM_COLOR,ms=5,mec="white",mew=.7,zorder=5)
        ax.annotate(str(j+1), xy=(x,y), xytext=(0,16),textcoords="offset points",ha="center",va="center",
            color="white", fontsize=NUMBER_SIZE, weight="bold",
            bbox={"boxstyle":"circle,pad=.30","fc":NUMBER_COLOR,"ec":"white","lw":.8},
            arrowprops={"arrowstyle":"-","color":NUMBER_COLOR,"lw":.7,"shrinkA":1.5,"shrinkB":4},
            zorder=6)
    ax.set(xlim=(-.025,1.025),ylim=(-.17,2.89),xticks=np.linspace(0,1,6),yticks=np.arange(0,2.6,.5))
    ax.set_xlabel(r"Normalized Cartesian arc length, $s$ (reactant $\rightarrow$ product)",labelpad=6)
    ax.set_ylabel(r"$V(x_i)-V(x_0)$ (kcal mol$^{-1}$)",labelpad=6)
    ax.legend(loc="upper left", frameon=False, handlelength=2.8, borderaxespad=.15)
    fig.text(.976, .037*112/height_mm, "22 stored images per path", ha="right", color="#69757E", fontsize=6.5)
    fig.savefig(pdf, dpi=600, metadata={"Title":"Alanine dipeptide: energy profile and molecular configurations", "Creator":Path(__file__).name})
    fig.savefig(png, dpi=300)
    plt.close(fig)


def draw_dihedrals_on_configuration(ax, rendered):
    """Schematic transparent plane overlays on the unchanged molecular view.

    Triangle vertices are exact projected atom centers. No enlarged surrogate
    molecule, changed camera, or projected 2D angle is used. The two labels
    point to the corresponding true central bonds, not to a planar angle.
    """
    positions = np.asarray(rendered["atom_xy_normalized"])
    for atoms, color in zip(PLANE_ATOMS, PLANE_COLORS):
        vertices = positions[atoms]
        ax.add_patch(Polygon(vertices, facecolor=color, edgecolor="none", alpha=.32, zorder=3))
        ax.add_patch(Polygon(vertices, fill=False, edgecolor=color, linewidth=.75,
                             alpha=.92, joinstyle="round", zorder=4))
    for symbol, bond, text_position, color, curvature in [
        (r"$\phi$", [6, 8], (.19, .27), PHI_COLOR, -.15),
        (r"$\psi$", [8, 14], (.80, .27), PSI_COLOR, .15),
    ]:
        midpoint = positions[bond].mean(axis=0)
        ax.annotate(symbol, xy=midpoint, xytext=text_position,
                    fontsize=10.5, color=color, ha="center", va="center",
                    arrowprops={"arrowstyle":"-|>","color":color,"lw":.8,
                                "connectionstyle":f"arc3,rad={curvature}",
                                "shrinkA":2,"shrinkB":1,"mutation_scale":5},
                    zorder=5)


def export(profiles, selected, renders, source, audit):
    csv_path = OUT / f"{STEM}_source.csv"
    rows = []
    for p in profiles:
        for i in range(22):
            rows.append({"method":p["method"],"optimization_nf":p["actual_nf"],
                "source_record_index":p["source_record_index"],"source_call":p["source_call"],
                "image_index_zero_based":i,"image_number_one_based":i+1,"normalized_cartesian_arclength":p["progress"][i],
                "energy_ev":p["energy_ev"][i],"relative_energy_kcal_mol":p["relative_energy_kcal"][i],
                "phi_deg":p["phi_deg"][i],"psi_deg":p["psi_deg"][i],
                "configuration_number":selected.index(i)+1 if p["method"]=="IVM" and i in selected else "",
                "source_result":p["source_file"],"source_result_sha256":sha256(p["source_file"]),
                "path_sha256":p["path_sha256"],"energy_cache_row":p["energy_cache_index"]})
    with csv_path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    coordinate_path = ASSETS / "selected_paths_and_energies.npz"
    np.savez_compressed(coordinate_path, **{f"{p['method_id']}_{k}":p[k]
        for p in profiles for k in ("path","energy_ev","progress","phi_deg","psi_deg")},
        ivm_snapshot_indices=np.asarray(selected))
    manifest={
        "status":"validated_and_generated", "created_utc":datetime.now(timezone.utc).isoformat(),
        "figure":bind(OUT/f"{STEM}.pdf"),"preview":bind(OUT/f"{STEM}.png"),"source_table":bind(csv_path),
        "source_coordinates":bind(coordinate_path),"plotter":bind(Path(__file__)),"renderer":bind(RENDERER),
        "registered_path_selector":bind(PATH_PLOTTER),"topology":bind(TOPOLOGY),
        "energy_source":bind(ENERGY_CACHE),"energy_audit":bind(ENERGY_AUDIT),"energy_contract":bind(ENERGY_CONTRACT),
        "dihedral_helper":bind(source.PROJECTION_SOURCE),
        "ivm_controls":{"alpha":10,"r":.7,"m_max":5,"budget":5000},
        "profiles":[{k:p[k] for k in ("method","method_id","actual_nf","source_record_index","source_call","source_file","path_sha256","formal_complete","mesh_eligible","energy_cache_index")} for p in profiles],
        "snapshot_indices_zero_based":selected,
        "snapshot_selection":"IVM reactant; first pre-peak stored image reaching half the native sampled barrier; native maximum-energy image; product",
        "endpoint_inclusion":{"reactant":0,"product":21},
        "configuration_number_style":{
            "numbers":[1,2,3,4], "font_size_pt":NUMBER_SIZE, "font_weight":"bold",
            "badge_fill":NUMBER_COLOR, "text_color":"white", "outline_color":"white",
            "configuration_placement":"immediately before each configuration title, vertically aligned to its center",
            "energy_placement":"16 points above the matching IVM sample, with a short connecting leader",
        },
        "energy_definition":"native cached V(x_i)-V(x_0) for each path; eV converted by 23.0605; no new force or energy evaluation",
        "arclength_definition":"cumulative Euclidean distance in ORIGINAL 66D Cartesian coordinates, normalized separately per stored path; no display alignment applied to metric",
        "profile_interpolation":"straight connections between 22 stored samples only; no smoothing or densification",
        "transition_state_label":"Transition state: image 13, selected as the highest-energy stored image on the reported path",
        "native_peak_caveat":"maximum of 22 sampled images, distinct from densified Delta E",
        "phi_atoms_zero_based":[4,6,8,14],"psi_atoms_zero_based":[6,8,14,16],
        "phi_definition":"C(ACE)-N(ALA)-CA(ALA)-C(ALA)","psi_definition":"N(ALA)-CA(ALA)-C(ALA)-N(NME)",
        "angles":"signed degrees in [-180,180), computed on original coordinates before rigid alignment",
        "new_physical_evaluations":0,"optimization_nf_added":0,
        "reused_energy_diagnostic_historical_evaluations_for_two_paths":44,
        "noF_convention":"IVM total_internal_joint_evaluations over 20 moving images; ODE frozen 20*count_seq+2; no convention silently changed",
        "rendering":renders,
        "dihedral_annotations":{
            "placement":"directly on each of the four unchanged molecular configurations; no separate dihedral panels",
            "triangles_zero_based":PLANE_ATOMS, "plane_colors":PLANE_COLORS,
            "phi_planes":[0,1], "psi_planes":[1,2],
            "phi_shared_bond":[6,8], "psi_shared_bond":[8,14],
            "geometry":"exact atom-center projections from the original shared camera",
            "display":"schematic transparent vector overlays, not opaque depth-rendered surfaces",
        },
        "snapshot_renderer_archive":bind(SNAPSHOT_RENDERER_ARCHIVE),
    }
    (OUT/f"{STEM}_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-renders",action="store_true")
    parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args()
    profiles,selected,source,audit=load_data()
    if args.validate_only:
        print(json.dumps({"status":"validated","snapshots":selected,"profiles":[{"method":p["method"],"nf":p["actual_nf"],"sampled_barrier_kcal":max(p["relative_energy_kcal"])} for p in profiles]},indent=2))
        return
    OUT.mkdir(parents=True,exist_ok=True);ASSETS.mkdir(parents=True,exist_ok=True)
    render_json=ASSETS/"rendering.json"
    ivm=next(p for p in profiles if p["method"]=="IVM")
    if args.reuse_renders:
        renders=json.loads(render_json.read_text())
        expected_coordinate_sha = hashlib.sha256(np.ascontiguousarray(ivm["path"][selected]).tobytes()).hexdigest()
        if (renders["input_coordinates_sha256"] != expected_coordinate_sha
                or renders["renderer_source_sha256"] not in (sha256(RENDERER),sha256(SNAPSHOT_RENDERER_ARCHIVE))
                or len(renders["images"]) != len(selected)):
            raise RuntimeError("Saved renders no longer match coordinates or renderer")
        for image in renders["images"]:
            if sha256(image["path"]) != image["sha256"]:
                raise RuntimeError("Saved molecular PNG changed")
    else:
        renderer=module(RENDERER,"ala_ballstick_renderer")
        renders=renderer.render_configurations(ivm["path"][selected],ASSETS,
            names=["reactant","rising_branch","peak_image","product"],image_size=1600,reference=ivm["path"][0],quality="final")
        render_json.write_text(json.dumps(renders,indent=2)+"\n")
    draw(profiles,selected,renders,OUT/f"{STEM}.pdf",OUT/f"{STEM}.png")
    export(profiles,selected,renders,source,audit)
    print(json.dumps({"pdf":str(OUT/f"{STEM}.pdf"),"snapshots":selected,"new_physical_evaluations":0},indent=2))


if __name__=="__main__":
    main()
