#!/usr/bin/env python3
"""Formal-style dense NN-IVM comparison for Muller-22."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_q10_article_muller22 import extend_styles, load_formal_namespace


ORDER = ("IVM", "NN-IVM", "OBSM-FIRE", "OBSM-LBFGS", "Original String", "ODE-NEB", "Static NEB")


def main() -> None:
    stage = Path(__file__).resolve().parents[1]
    namespace = load_formal_namespace(stage.parent, stage)
    extend_styles(namespace)
    style = namespace["METHOD_STYLE"]
    style["NN-IVM"] = {
        "linestyle": "-", "linewidth": 2.15, "color": "#D1495B",
        "marker": "o", "markersize": 3.0, "zorder": 18,
    }
    report_path = stage / "reports" / "q10_nn_ivm_dense_comparison.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    files = {row["method"]: stage / row["output"] for row in report["runs"]}
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.75))
    for label in ORDER:
        archive = np.load(files[label], allow_pickle=False)
        x = np.asarray(archive["number_of_forces"], dtype=float)
        i_perp = np.maximum(np.asarray(archive["dense_i_perp"], dtype=float), 1.0e-30)
        barrier = np.maximum(np.asarray(archive["dense_delta_e_error"], dtype=float), 1.0e-16)
        keep = x <= 5000
        options = dict(style[label])
        markevery = max(1, int(np.count_nonzero(keep) / 10))
        axes[0].plot(x[keep], i_perp[keep], label=label, markevery=markevery, **options)
        axes[1].plot(x[keep], barrier[keep], label=label, markevery=markevery, **options)
    for axis in axes:
        axis.set_xlim(0.0, 5000.0)
        axis.set_yscale("log")
        axis.set_xlabel("Number of Forces")
        axis.grid(True, which="major", alpha=0.22, linewidth=0.55)
        axis.tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel(r"Densified $I_{\perp}$")
    axes[1].set_ylabel(r"Densified $|\Delta E-\Delta E^*|$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.75, wspace=0.27)
    output_dir = stage / "figures" / "nn_ivm"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "muller22_q10_nn_ivm_dense_comparison"
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600 if suffix == "png" else None)
    plt.close(fig)
    manifest = {
        "case": report["case"],
        "figure_status": "production theory-audited fixed-alpha NN-IVM comparison",
        "source_report": str(report_path),
        "method_order": list(ORDER),
        "ivm_highest_zorder": True,
        "xscale": "linear",
        "xlim": [0, 5000],
        "yscale": "log",
        "metrics": ["5001-point densified I_perp", "5001-point densified Delta E error"],
        "tangent": "NEB Henkelman-Jonsson improved tangent",
        "main_tex_updated": False,
    }
    stem.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
