#!/usr/bin/env python3
"""Plot cumulative discrete geometric action on each original 32-image path.

The plot connects the 32 stored vertices directly.  No densification,
interpolation, smoothing, or new physical evaluation is performed here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
EVALUATOR = ROOT / "scripts/evaluate_bb_discrete32_geometric_action.py"
IR_COLOR = "#0072B2"
ODE_COLOR = "#D55E00"


def load_evaluator():
    spec = importlib.util.spec_from_file_location("bb_discrete32_action_plotter", EVALUATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


def render_pdf_temporary(fig, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.{os.getpid()}{output.suffix}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary figure: {temporary}")
    fig.savefig(
        temporary,
        bbox_inches="tight",
        metadata={
            "Title": "Barnase--Barstar discrete 32-image path-dependent geometric action",
            "Subject": "Original 32 images joined directly; no densification or interpolation",
        },
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    return temporary


def write_json_temporary(path: Path, payload: dict) -> Path:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"stale temporary provenance: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return temporary


def recover_or_complete_publication(evaluator, publication: Path, output: Path, provenance: Path, result: Path) -> dict:
    plan = json.loads(publication.read_text(encoding="utf-8"))
    evaluator.require(plan.get("schema") == "bb-discrete32-geometric-action-figure-publication-v1", "figure publication schema mismatch")
    evaluator.require(plan.get("status") == "PREPARED_FOR_ATOMIC_NO_OVERWRITE_LINKS", "figure publication status mismatch")
    evaluator.require(plan.get("plotter") == evaluator.file_binding(Path(__file__).resolve()), "figure publication plotter mismatch")
    evaluator.require(plan.get("result") == evaluator.file_binding(result), "figure publication result mismatch")
    targets = (("figure", output), ("provenance", provenance))
    for role, target in targets:
        binding = plan.get(role, {})
        evaluator.require(binding.get("path") == str(target), f"publication {role} path mismatch")
        expected_hash = str(binding.get("sha256", ""))
        expected_bytes = int(binding.get("bytes", -1))
        temporary = Path(str(binding.get("temporary_path", ""))).expanduser().resolve()
        if target.exists():
            evaluator.require(evaluator.sha256_file(target) == expected_hash, f"published {role} hash mismatch")
            evaluator.require(target.stat().st_size == expected_bytes, f"published {role} size mismatch")
        else:
            evaluator.require(temporary.exists(), f"neither published nor recoverable temporary {role} exists")
            evaluator.require(evaluator.sha256_file(temporary) == expected_hash, f"temporary {role} hash mismatch")
            evaluator.require(temporary.stat().st_size == expected_bytes, f"temporary {role} size mismatch")
            os.link(temporary, target)
            descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    record = json.loads(provenance.read_text(encoding="utf-8"))
    evaluator.require(record.get("status") == "PASS", "published provenance is not PASS")
    evaluator.require(record.get("figure") == {"path": str(output), "sha256": evaluator.sha256_file(output)}, "provenance/figure mismatch")
    evaluator.require(record.get("publication_plan") == str(publication), "provenance/publication path mismatch")
    for role in ("figure", "provenance"):
        temporary = Path(plan[role]["temporary_path"])
        if temporary.exists():
            temporary.unlink()
    return record


def validate_and_load(evaluator, result: Path) -> dict:
    state = evaluator.load_contract(evaluator.CONTRACT, evaluator.AUTHORIZATION, result)
    evaluator.require(result.exists(), "geometric-action result does not exist")
    evaluator.require(state["audit"].exists(), "geometric-action result lacks audit")
    evaluator.require(state["wal"].exists(), "geometric-action result lacks joint E/F WAL")
    evaluator.require(state["result_publication"].exists(), "geometric-action result lacks atomic publication")
    sources = evaluator.load_sources()
    wal_state = evaluator.read_wal(state["wal"], evaluator.expected_wal_header(state))
    evaluator.validate_result(result, state, sources, wal_state)
    evaluator.validate_result_publication(state)
    audit = evaluator.validate_audit(state["audit"], state, wal_state)

    with np.load(result, allow_pickle=False) as saved:
        methods = np.asarray(saved["method"]).astype(str)
        nfs = np.asarray(saved["optimization_N_F"], dtype=np.int64)
        coordinates = np.ascontiguousarray(saved["coordinates_A"], dtype=np.float64)
        forces = np.ascontiguousarray(saved["force_eV_per_A"], dtype=np.float64)
        arc = np.ascontiguousarray(saved["arc_length_A"], dtype=np.float64)
        cumulative_eV = np.ascontiguousarray(
            saved["cumulative_path_dependent_geometric_action_eV"], dtype=np.float64
        )
        cumulative_kcal = np.ascontiguousarray(
            saved["cumulative_path_dependent_geometric_action_kcal_per_mol"], dtype=np.float64
        )
        path_hashes = np.asarray(saved["path_sha256"]).astype(str)
        force_hashes = np.asarray(saved["force_sha256"]).astype(str)
        number_images = int(evaluator.scalar(saved, "number_of_original_images"))
        number_segments = int(evaluator.scalar(saved, "number_of_original_segments"))
        densification = str(evaluator.scalar(saved, "densification"))
        interpolation = str(evaluator.scalar(saved, "interpolation"))
        smoothing = str(evaluator.scalar(saved, "smoothing"))

    evaluator.require(methods.tolist() == ["IVM", "ODE-NEB"], "method inventory/order mismatch")
    evaluator.require(nfs.tolist() == [evaluator.IR_OPTIMIZATION_NF, evaluator.ODE_OPTIMIZATION_NF], "optimization N_F mismatch")
    evaluator.require(coordinates.shape == (2, 32, evaluator.DIMENSION), "coordinate shape is not two original 32-image paths")
    evaluator.require(forces.shape == coordinates.shape, "full-force shape mismatch")
    evaluator.require(arc.shape == (2, 32) and cumulative_eV.shape == (2, 32), "curve shape mismatch")
    evaluator.require(cumulative_kcal.shape == (2, 32), "converted curve shape mismatch")
    evaluator.require(number_images == 32 and number_segments == 31, "original-image count mismatch")
    evaluator.require(densification == "none" and interpolation == "none" and smoothing == "none", "resampling policy mismatch")
    evaluator.require(np.all(np.isfinite(coordinates)) and np.all(np.isfinite(forces)), "non-finite stored E/F data")
    for index in range(2):
        evaluator.require(evaluator.sha256_array(coordinates[index]) == path_hashes[index], "stored path hash mismatch")
        evaluator.require(evaluator.sha256_array(forces[index]) == force_hashes[index], "stored force hash mismatch")
        recomputed = evaluator.discrete_path_dependent_action(coordinates[index], forces[index])
        evaluator.require(np.array_equal(recomputed["arc_length_A"], arc[index]), "arc-length recomputation mismatch")
        evaluator.require(np.array_equal(recomputed["cumulative_action_eV"], cumulative_eV[index]), "action recomputation mismatch")
        evaluator.require(
            np.array_equal(recomputed["cumulative_action_eV"] * evaluator.EV_TO_KCAL_PER_MOL, cumulative_kcal[index]),
            "action-unit conversion mismatch",
        )
    return {
        "state": state,
        "audit": audit,
        "methods": methods,
        "optimization_N_F": nfs,
        "coordinates": coordinates,
        "forces": forces,
        "arc_length_A": arc,
        "cumulative_action_eV": cumulative_eV,
        "cumulative_action_kcal_per_mol": cumulative_kcal,
        "path_sha256": path_hashes,
        "force_sha256": force_hashes,
    }


def main(argv=None) -> None:
    evaluator = load_evaluator()
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, default=evaluator.RESULT)
    parser.add_argument("--output", type=Path, default=evaluator.FIGURE)
    parser.add_argument("--provenance", type=Path, default=evaluator.PROVENANCE)
    args = parser.parse_args(argv)
    result = args.result.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    provenance = args.provenance.expanduser().resolve()
    publication = evaluator.FIGURE_PUBLICATION.resolve()
    evaluator.require(result == evaluator.RESULT.resolve(), "noncanonical geometric-action result")
    evaluator.require(output == evaluator.FIGURE.resolve(), "noncanonical geometric-action figure")
    evaluator.require(provenance == evaluator.PROVENANCE.resolve(), "noncanonical geometric-action provenance")
    data = validate_and_load(evaluator, result)
    if publication.exists():
        record = recover_or_complete_publication(evaluator, publication, output, provenance, result)
        print(json.dumps(record, indent=2, sort_keys=True))
        return
    evaluator.require(not output.exists() and not provenance.exists(), "orphan figure/provenance without publication plan")

    configure_style()
    styles = {
        "IVM": {"color": IR_COLOR, "marker": "s", "zorder": 4},
        "ODE-NEB": {"color": ODE_COLOR, "marker": "o", "zorder": 3},
    }
    fig, axis = plt.subplots(figsize=(4.15, 2.85))
    for index, method in enumerate(data["methods"]):
        style = styles[str(method)]
        # A single 32-vertex Line2D is deliberate: adjacent original images are
        # connected directly and no interpolated display vertices are created.
        axis.plot(
            data["arc_length_A"][index],
            data["cumulative_action_kcal_per_mol"][index],
            color=style["color"],
            linestyle="-",
            linewidth=1.45,
            marker=style["marker"],
            markersize=3.0,
            markeredgewidth=0.45,
            markeredgecolor="white",
            label=f"{method}  ($N_F={int(data['optimization_N_F'][index])}$)",
            zorder=style["zorder"],
        )
    axis.set_xlabel(r"Cartesian arc length $\ell$ ($\AA$)")
    axis.set_ylabel(r"Cumulative $\widehat{S}$ (kcal mol$^{-1}$)")
    axis.set_title("Discrete 32-image path-dependent geometric action", pad=5)
    axis.set_xlim(0.0, float(np.max(data["arc_length_A"][:, -1])))
    axis.set_ylim(bottom=0.0)
    axis.grid(True, color="#D8D8D8", linewidth=0.45, alpha=0.72)
    axis.legend(frameon=False, loc="upper left", handlelength=2.3)
    axis.tick_params(direction="out", length=3)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.19, top=0.90)
    pdf_temporary = render_pdf_temporary(fig, output)
    plt.close(fig)

    state = data["state"]
    record = {
        "schema": "bb-discrete32-path-dependent-geometric-action-figure-provenance-v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "figure": {"path": str(output), "sha256": evaluator.sha256_file(pdf_temporary)},
        "publication_plan": str(publication),
        "plotter": evaluator.file_binding(Path(__file__).resolve()),
        "evaluator": evaluator.file_binding(EVALUATOR),
        "result": evaluator.file_binding(result),
        "result_publication": evaluator.file_binding(state["result_publication"]),
        "result_audit": evaluator.file_binding(state["audit"]),
        "joint_EF_WAL": evaluator.file_binding(state["wal"]),
        "contract": evaluator.file_binding(state["contract_path"]),
        "authorization": evaluator.file_binding(state["authorization_path"]),
        "zero_openmm_preflight": evaluator.file_binding(state["preauth_path"]),
        "manuscript": evaluator.file_binding(evaluator.MANUSCRIPT),
        "scientific_quantity": "fixed-endpoint path-dependent part of the geometric action",
        "formula_id": evaluator.FORMULA_ID,
        "formula_text": evaluator.FORMULA_TEXT,
        "sampling": {
            "number_of_original_images_per_path": 32,
            "number_of_direct_segments_per_path": 31,
            "abscissa": "each path's cumulative unweighted Cartesian arc length",
            "connection": "adjacent original images connected directly",
            "densification": "none",
            "interpolation": "none",
            "smoothing": "none",
        },
        "physical_calls_by_plotter": 0,
        "optimization_N_F_contribution_by_diagnostic": 0,
        "curves": [],
    }
    for index, method in enumerate(data["methods"]):
        record["curves"].append(
            {
                "method": str(method),
                "optimization_N_F": int(data["optimization_N_F"][index]),
                "path_sha256": str(data["path_sha256"][index]),
                "force_sha256": str(data["force_sha256"][index]),
                "arc_length_sha256": evaluator.sha256_array(data["arc_length_A"][index]),
                "cumulative_action_eV_sha256": evaluator.sha256_array(data["cumulative_action_eV"][index]),
                "total_arc_length_A": float(data["arc_length_A"][index, -1]),
                "total_path_dependent_geometric_action_eV": float(data["cumulative_action_eV"][index, -1]),
                "total_path_dependent_geometric_action_kcal_per_mol": float(
                    data["cumulative_action_kcal_per_mol"][index, -1]
                ),
            }
        )
    provenance_temporary = None
    publication_prepared = False
    try:
        provenance_temporary = write_json_temporary(provenance, record)
        plan = {
            "schema": "bb-discrete32-geometric-action-figure-publication-v1",
            "status": "PREPARED_FOR_ATOMIC_NO_OVERWRITE_LINKS",
            "prepared_utc": datetime.now(timezone.utc).isoformat(),
            "plotter": evaluator.file_binding(Path(__file__).resolve()),
            "result": evaluator.file_binding(result),
            "figure": {
                "path": str(output),
                "sha256": evaluator.sha256_file(pdf_temporary),
                "bytes": pdf_temporary.stat().st_size,
                "temporary_path": str(pdf_temporary),
            },
            "provenance": {
                "path": str(provenance),
                "sha256": evaluator.sha256_file(provenance_temporary),
                "bytes": provenance_temporary.stat().st_size,
                "temporary_path": str(provenance_temporary),
            },
            "physical_calls": 0,
        }
        evaluator.atomic_json_no_overwrite(publication, plan)
        publication_prepared = True
        published = recover_or_complete_publication(evaluator, publication, output, provenance, result)
    finally:
        if not publication_prepared:
            for temporary in (pdf_temporary, provenance_temporary):
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    print(json.dumps(published, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
