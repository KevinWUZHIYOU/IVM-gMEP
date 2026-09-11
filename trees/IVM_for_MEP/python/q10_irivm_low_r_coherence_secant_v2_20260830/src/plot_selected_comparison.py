#!/usr/bin/env python3
"""Plot the final audited low-r IR-IVM winners against Article IVM."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


SCRIPT = Path(__file__).resolve()
CAMPAIGN = SCRIPT.parents[1]
REPOSITORY = SCRIPT.parents[3]
AUDIT_MANIFEST = CAMPAIGN / "reports/result_audit/audit_manifest.json"
METRIC_SOURCE = REPOSITORY / "python/unified_ir_ivm_20260827/src/q10_article_metrics.py"
ARTICLE = {
    10: REPOSITORY
    / "paper/ActiveExperiments/muller_q10_10d/image12/data/frozen_runs/"
    "tuned_ivm_alpha2e3_selected.npz",
    20: REPOSITORY
    / "paper/ActiveExperiments/muller_q10_10d/data/frozen_runs/"
    "tuned_ivm_alpha1e3_selected.npz",
}
OUTPUT_PDF = REPOSITORY / "output/pdf/q10_irivm_low_r_strict_comparison.pdf"
QA_PNG = REPOSITORY / "output/pdf/q10_irivm_low_r_strict_comparison.png"
REFERENCE_DELTA_E = 67.50188060819488
ARTICLE_COLOR = "#0072B2"
IRIVM_COLOR = "#D55E00"
SHADE_COLOR = "#DCECE5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def scalar(archive: np.lib.npyio.NpzFile, key: str) -> str:
    require(key in archive.files, f"candidate NPZ is missing {key!r}")
    return str(np.asarray(archive[key]).item())


def load_metric(audit: dict):
    require(Path(audit["metric_source"]).resolve() == METRIC_SOURCE.resolve(),
            "final audit names an unexpected metric source")
    require(sha256(METRIC_SOURCE) == audit["metric_source_sha256"],
            "article metric source hash no longer matches the final audit")
    spec = importlib.util.spec_from_file_location(
        "q10_low_r_strict_figure_metric", METRIC_SOURCE
    )
    require(spec is not None and spec.loader is not None,
            "could not load the article metric module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    require(float(module.REFERENCE_BARRIER) == REFERENCE_DELTA_E,
            "metric reference barrier has changed")
    require(int(module.LINEAR_DENSE_COUNT) == 500,
            "metric densification count has changed")
    return module


def verify_registered_dependency(audit: dict, path: Path) -> None:
    resolved = path.resolve()
    matches = [
        row for row in audit["dependency_hash_audit"]
        if Path(row["path"]).resolve() == resolved
    ]
    require(len(matches) == 1, f"missing unique dependency audit for {resolved}")
    row = matches[0]
    observed = sha256(resolved)
    require(bool(row["valid"]), f"dependency audit failed for {resolved}")
    require(row["observed_sha256"] == row["registered_sha256"] == observed,
            f"dependency hash mismatch for {resolved}")


def expected_gate_token(audit: dict, n_internal: int) -> str:
    gate = audit["staged_resolution_gate"]
    require(bool(gate["independently_authenticated_for_n10"]),
            "N=20 staging gate was not independently authenticated")
    if n_internal == 20:
        return str(gate["n20_token"])
    token = str(gate["n10_manifest_sha256_token"])
    manifest = Path(gate["fixed_n20_manifest"]).resolve()
    require(sha256(manifest) == token,
            "N=10 gate token does not match the fixed N=20 audit manifest")
    return token


def verify_candidate(audit: dict, n_internal: int) -> tuple[dict, dict]:
    resolution = audit["resolution_results"][str(n_internal)]
    winner = resolution.get("strict_winner")
    require(isinstance(winner, dict), f"N={n_internal} has no strict winner")
    require(bool(winner["strict_pass"]), f"N={n_internal} winner is not strict")
    for key in (
        "i_perp_strictly_better_checkpoints",
        "delta_e_error_strictly_better_checkpoints",
        "both_strictly_better_checkpoints",
    ):
        require(int(winner[key]) == 51,
                f"N={n_internal} winner does not have 51/51 for {key}")

    run_id = str(winner["run_id"])
    candidate_path = Path(winner["source"]).resolve()
    receipt_path = Path(winner["receipt"]).resolve()
    expected_candidate = (CAMPAIGN / f"runs/n{n_internal}/{run_id}.npz").resolve()
    expected_receipt = (
        CAMPAIGN / f"reports/receipts/n{n_internal}/{run_id}.receipt.json"
    ).resolve()
    require(candidate_path == expected_candidate,
            f"N={n_internal} winner NPZ is outside its formal run location")
    require(receipt_path == expected_receipt,
            f"N={n_internal} receipt is outside its formal receipt location")
    require(sha256(candidate_path) == winner["source_sha256"],
            f"N={n_internal} winner NPZ hash mismatch")
    require(sha256(receipt_path) == winner["receipt_sha256"],
            f"N={n_internal} winner receipt hash mismatch")

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt["status"] == "completed", f"N={n_internal} run is incomplete")
    require(str(receipt["run_id"]) == run_id, f"N={n_internal} receipt run mismatch")
    require(Path(receipt["output"]).resolve() == candidate_path,
            f"N={n_internal} receipt output path mismatch")
    require(receipt["output_sha256"] == winner["source_sha256"],
            f"N={n_internal} receipt output hash mismatch")
    token = expected_gate_token(audit, n_internal)
    require(str(receipt["n20_gate_manifest_sha256_token"]) == token,
            f"N={n_internal} receipt gate token mismatch")

    with np.load(candidate_path, allow_pickle=False) as archive:
        require(scalar(archive, "run_id") == run_id,
                f"N={n_internal} NPZ run identity mismatch")
        require(scalar(archive, "n20_gate_manifest_sha256_token") == token,
                f"N={n_internal} NPZ gate token mismatch")
        config = json.loads(scalar(archive, "config_json"))
        controls = {
            "n_internal": int(config["n_internal"]),
            "alpha": float(config["alpha"]),
            "alpha_label": str(config["alpha_label"]),
            "r": float(config["r"]),
            "m_max": int(config["m_max"]),
        }
        expected_controls = {
            "n_internal": int(winner["n_internal"]),
            "alpha": float(winner["alpha"]),
            "alpha_label": str(winner["alpha_label"]),
            "r": float(winner["r"]),
            "m_max": int(winner["m_max"]),
        }
        require(controls == expected_controls == receipt["controls"],
                f"N={n_internal} controls disagree across audit, NPZ, and receipt")
        nof = np.asarray(archive["article_nf"], dtype=np.int64)
        paths = np.asarray(archive["path_history"], dtype=np.float64)
        q_matrix = np.asarray(archive["q_matrix"], dtype=np.float64)
        require("article_usable_state" in archive.files,
                f"N={n_internal} candidate lacks its public-state mask")
        usable = np.asarray(archive["article_usable_state"], dtype=bool)
    require(paths.shape[0] == nof.size == usable.size,
            f"N={n_internal} candidate trajectory lengths disagree")
    require(bool(np.all(usable)),
            f"N={n_internal} candidate contains a non-public plotted state")
    require(bool(np.all(np.diff(nof) > 0)),
            f"N={n_internal} candidate NoF is not strictly increasing")
    require(int(nof[-1]) == int(winner["last_complete_number_of_forces"]),
            f"N={n_internal} candidate terminal NoF disagrees with final audit")
    return winner, {
        "path": candidate_path,
        "receipt": receipt_path,
        "config": controls,
        "nof": nof,
        "paths": paths,
        "q": q_matrix,
    }


def load_article(audit: dict, n_internal: int) -> dict:
    source = ARTICLE[n_internal].resolve()
    verify_registered_dependency(audit, source)
    with np.load(source, allow_pickle=False) as archive:
        nof = np.asarray(archive["total_force_evaluations"], dtype=np.int64)
        paths = np.asarray(archive["path_history"], dtype=np.float64)
        q_matrix = np.asarray(archive["q_matrix"], dtype=np.float64)
    require(paths.shape[0] == nof.size,
            f"Article N={n_internal} trajectory lengths disagree")
    require(bool(np.all(np.diff(nof) > 0)),
            f"Article N={n_internal} NoF is not strictly increasing")
    return {"path": source, "nof": nof, "paths": paths, "q": q_matrix}


def trajectory_metrics(metric, paths: np.ndarray, q_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    i_perp = []
    barrier_error = []
    for path in paths:
        i_perp.append(metric.article_path_metric(path, q=q_matrix)[0])
        delta_e = metric.linear500_barrier(path, q=q_matrix)
        barrier_error.append(abs(delta_e - REFERENCE_DELTA_E))
    values_i = np.asarray(i_perp, dtype=np.float64)
    values_e = np.asarray(barrier_error, dtype=np.float64)
    require(bool(np.all(np.isfinite(values_i))) and bool(np.all(values_i > 0.0)),
            "non-positive or non-finite I_perp cannot be shown on the log axis")
    require(bool(np.all(np.isfinite(values_e))) and bool(np.all(values_e > 0.0)),
            "non-positive or non-finite barrier error cannot be shown on the log axis")
    return values_i, values_e


def locf(nof: np.ndarray, values: np.ndarray, checkpoints: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(nof, checkpoints, side="right") - 1
    require(bool(np.all(indices >= 0)), "a checkpoint precedes the first public state")
    return np.asarray(values)[indices]


def draw_public_step(
    axis,
    nof: np.ndarray,
    values: np.ndarray,
    *,
    color: str,
    label: str,
    stop: int,
    linewidth: float,
) -> None:
    axis.step(
        nof,
        values,
        where="post",
        color=color,
        linewidth=linewidth,
        label=label,
        solid_capstyle="butt",
        zorder=3,
    )
    if int(nof[-1]) < int(stop):
        axis.plot(
            [int(nof[-1]), int(stop)],
            [float(values[-1]), float(values[-1])],
            color=color,
            linewidth=linewidth,
            linestyle=(0, (3.0, 2.0)),
            solid_capstyle="butt",
            zorder=3,
        )


def alpha_math(label: str) -> str:
    mantissa, exponent = label.split("e", maxsplit=1)
    return rf"{mantissa}\mathrm{{e}}{{{int(exponent):+d}}}"


def main() -> None:
    audit = json.loads(AUDIT_MANIFEST.read_text(encoding="utf-8"))
    require(audit["audit_status"] == "complete", "final audit is not complete")
    require(bool(audit["all_requested_resolutions_strictly_pass"]),
            "final audit did not strictly pass both resolutions")
    require(audit["requested_resolutions"] == [10, 20],
            "final audit resolution set has changed")
    require(float(audit["reference_delta_e"]) == REFERENCE_DELTA_E,
            "final audit reference barrier has changed")
    require(int(audit["checkpoint_rule"]["count_per_resolution"]) == 51,
            "final audit checkpoint count has changed")
    metric = load_metric(audit)
    verify_registered_dependency(audit, METRIC_SOURCE)

    selected = {}
    for n_internal in (10, 20):
        winner, candidate = verify_candidate(audit, n_internal)
        article = load_article(audit, n_internal)
        require(np.array_equal(candidate["q"], article["q"]),
                f"N={n_internal} candidate and Article Q matrices differ")
        candidate_i, candidate_e = trajectory_metrics(
            metric, candidate["paths"], candidate["q"]
        )
        article_i, article_e = trajectory_metrics(
            metric, article["paths"], article["q"]
        )
        lower, upper = [
            int(value)
            for value in audit["checkpoint_rule"]["ranges"][str(n_internal)]
        ]
        require(upper == int(article["nof"][-1]),
                f"N={n_internal} checkpoint range does not end at Article IVM")
        checkpoints = np.rint(np.linspace(lower, upper, 51)).astype(np.int64)
        i_wins = locf(candidate["nof"], candidate_i, checkpoints) < locf(
            article["nof"], article_i, checkpoints
        )
        e_wins = locf(candidate["nof"], candidate_e, checkpoints) < locf(
            article["nof"], article_e, checkpoints
        )
        require(int(np.count_nonzero(i_wins)) == 51,
                f"N={n_internal} plotted I_perp does not reproduce 51/51")
        require(int(np.count_nonzero(e_wins)) == 51,
                f"N={n_internal} plotted barrier error does not reproduce 51/51")
        require(int(np.count_nonzero(i_wins & e_wins)) == 51,
                f"N={n_internal} plotted metrics do not reproduce joint 51/51")
        selected[n_internal] = {
            "winner": winner,
            "candidate": candidate,
            "article": article,
            "candidate_i": candidate_i,
            "candidate_e": candidate_e,
            "article_i": article_i,
            "article_e": article_e,
            "checkpoint_range": (lower, upper),
        }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.3,
            "axes.linewidth": 0.7,
            "axes.titleweight": "semibold",
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.top": False,
            "ytick.right": False,
            "legend.fontsize": 8.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(180.0 / 25.4, 126.0 / 25.4),
        sharex="col",
    )

    for column, n_internal in enumerate((10, 20)):
        item = selected[n_internal]
        candidate = item["candidate"]
        article = item["article"]
        stop = int(article["nof"][-1])
        lower, upper = item["checkpoint_range"]
        top = axes[0, column]
        bottom = axes[1, column]
        for axis in (top, bottom):
            axis.axvspan(
                lower,
                upper,
                color=SHADE_COLOR,
                alpha=0.58,
                linewidth=0.0,
                zorder=0,
            )
            axis.grid(True, which="major", color="#D5DADD", linewidth=0.48)
            axis.grid(True, which="minor", color="#ECEFF0", linewidth=0.30)
            axis.set_yscale("log")
            axis.set_xlim(0, stop)
            axis.set_axisbelow(True)

        draw_public_step(
            top,
            article["nof"],
            item["article_i"],
            color=ARTICLE_COLOR,
            label="Article IVM",
            stop=stop,
            linewidth=1.35,
        )
        draw_public_step(
            top,
            candidate["nof"],
            item["candidate_i"],
            color=IRIVM_COLOR,
            label="IR-IVM (strict winner)",
            stop=stop,
            linewidth=1.55,
        )
        draw_public_step(
            bottom,
            article["nof"],
            item["article_e"],
            color=ARTICLE_COLOR,
            label="Article IVM",
            stop=stop,
            linewidth=1.35,
        )
        draw_public_step(
            bottom,
            candidate["nof"],
            item["candidate_e"],
            color=IRIVM_COLOR,
            label="IR-IVM (strict winner)",
            stop=stop,
            linewidth=1.55,
        )

        config = candidate["config"]
        top.set_title(
            rf"$N_{{\rm int}}={n_internal}$"
            + "\n"
            + rf"IR-IVM: $\alpha={alpha_math(config['alpha_label'])},\ "
            + rf"r={config['r']:g},\ m_{{\max}}={config['m_max']}$",
            pad=5.0,
        )
        top.text(
            0.965,
            0.93,
            "51/51 checkpoints\n(both metrics)",
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=7.7,
            color="#245A46",
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.82,
                "pad": 1.6,
            },
            zorder=5,
        )
        bottom.set_xlabel("Number of Forces")

    axes[0, 0].set_ylabel(r"Article $I_{\perp}$ (HJ-improved)")
    axes[1, 0].set_ylabel(
        r"500-point densified $|\Delta E-\Delta E_{\rm ref}|$"
    )
    for axis, letter in zip(axes.flat, ("a", "b", "c", "d")):
        axis.text(
            -0.115,
            1.035,
            letter,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.2,
            fontweight="bold",
            clip_on=False,
        )

    legend_handles = (
        Line2D([0], [0], color=ARTICLE_COLOR, linewidth=1.45),
        Line2D([0], [0], color=IRIVM_COLOR, linewidth=1.65),
    )
    figure.legend(
        legend_handles,
        ("Article IVM", "IR-IVM (strict winner)"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.993),
        ncol=2,
        frameon=False,
        handlelength=3.0,
        columnspacing=2.3,
    )
    figure.subplots_adjust(
        left=0.123,
        right=0.985,
        bottom=0.115,
        top=0.825,
        wspace=0.25,
        hspace=0.12,
    )
    figure.text(
        0.5,
        0.012,
        "Shading: 51 latter-half same-NoF checkpoints.  "
        "Dashed IR-IVM: final public state carried forward after stopping.  "
        + rf"$\Delta E_{{\rm ref}}={REFERENCE_DELTA_E:.14f}$.",
        ha="center",
        va="bottom",
        fontsize=6.5,
        color="#4F5559",
    )

    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "Title": "QM10 Article IVM and low-r IR-IVM strict comparison",
        "Author": "",
        "Subject": "Formally audited same-NoF comparison",
        "Keywords": "QM10, IVM, IR-IVM",
        "Creator": str(SCRIPT),
        "Producer": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(OUTPUT_PDF, metadata=metadata)
    figure.savefig(QA_PNG, dpi=240)
    plt.close(figure)
    print(
        json.dumps(
            {
                "output_pdf": str(OUTPUT_PDF),
                "output_pdf_sha256": sha256(OUTPUT_PDF),
                "qa_png": str(QA_PNG),
                "qa_png_sha256": sha256(QA_PNG),
                "plot_source": str(SCRIPT),
                "plot_source_sha256": sha256(SCRIPT),
                "strict_winners": {
                    str(n): selected[n]["winner"]["run_id"] for n in (10, 20)
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
