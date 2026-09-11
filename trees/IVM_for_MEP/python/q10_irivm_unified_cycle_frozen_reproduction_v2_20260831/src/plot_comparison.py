"""Draw the registered 2x2 comparison from authenticated audit CSV files only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .formal_io import (
    FormalIOError,
    atomic_output_path,
    atomic_write_bytes,
    atomic_write_json,
    derive_campaign_paths,
    load_canonical_json,
    resolve_registered_path,
    sha256_file,
)


EXPECTED_COUNTS = {10: 79, 20: 98}
METHODS = ("Article IVM", "IR-IVM")
COLORS = {"Article IVM": "#444444", "IR-IVM": "#C44E52"}


class PlotGateError(RuntimeError):
    pass


def _read_authenticated_csv(
    audit_root: Path, relative: str, expected_sha256: str
) -> list[dict[str, str]]:
    path = resolve_registered_path(audit_root, relative)
    if sha256_file(path) != expected_sha256:
        raise PlotGateError(f"audit CSV hash mismatch: {relative}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise PlotGateError(f"invalid boolean CSV value: {value!r}")


def _curve(rows: list[dict[str, str]], resolution: int, method: str) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if int(row["n_internal"]) == resolution and row["method"] == method
    ]
    selected.sort(key=lambda row: int(row["nof"]))
    if not selected:
        raise PlotGateError(f"missing public metric curve for N={resolution}, {method}")
    nof = np.asarray([int(row["nof"]) for row in selected], dtype=np.int64)
    if np.any(np.diff(nof) <= 0):
        raise PlotGateError(f"non-increasing curve NoF for N={resolution}, {method}")
    return selected


def _verify_checkpoint_pass(rows: list[dict[str, str]], resolution: int) -> int:
    selected = [row for row in rows if int(row["n_internal"]) == resolution]
    expected = EXPECTED_COUNTS[resolution]
    if len(selected) != expected:
        raise PlotGateError(
            f"checkpoint table count for N={resolution} is {len(selected)}, expected {expected}"
        )
    checkpoints = np.asarray([int(row["checkpoint_nof"]) for row in selected], dtype=np.int64)
    if np.any(np.diff(checkpoints) <= 0):
        raise PlotGateError("checkpoint table is not strictly ordered")
    if not all(
        _truth(row["i_perp_strict"])
        and _truth(row["barrier_strict"])
        and _truth(row["joint_strict"])
        for row in selected
    ):
        raise PlotGateError(f"N={resolution} is not jointly strict at every checkpoint")
    return expected


def _verify_curve_endpoint(
    rows: list[dict[str, str]], resolution: int, method: str, terminal: int
) -> None:
    if int(rows[-1]["nof"]) != terminal:
        raise PlotGateError(
            f"N={resolution} {method} curve does not end at Article terminal {terminal}"
        )
    extension = [_truth(row["is_locf_extension"]) for row in rows]
    if method == "Article IVM" and any(extension):
        raise PlotGateError("Article curve may not contain a LOCF extension")
    if any(extension[:-1]):
        raise PlotGateError("LOCF extension is permitted only at the final plotting row")


def generate_figure(
    audit_manifest_path: Path,
    output_pdf: Path,
    output_png: Path,
    manifest_path: Path,
    provenance_path: Path = None,
) -> None:
    audit_manifest = load_canonical_json(audit_manifest_path)
    if audit_manifest.get("outcome") != "complete_scientific_pass":
        raise PlotGateError("figure requires complete_scientific_pass audit outcome")
    if audit_manifest.get("integrity_complete") is not True:
        raise PlotGateError("figure requires an integrity-complete audit")
    if audit_manifest.get("audit_source_sha256") is None:
        raise PlotGateError("audit manifest lacks its independent auditor identity")
    artifacts = audit_manifest.get("artifacts", {})
    audit_root = audit_manifest_path.parent
    checkpoint_spec = artifacts.get("checkpoint_comparison.csv")
    trajectory_spec = artifacts.get("public_trajectory_metrics.csv")
    if not isinstance(checkpoint_spec, dict) or not isinstance(trajectory_spec, dict):
        raise PlotGateError("audit manifest lacks authenticated comparison CSV declarations")
    checkpoint_rows = _read_authenticated_csv(
        audit_root, checkpoint_spec["relative_path"], checkpoint_spec["sha256"]
    )
    trajectory_rows = _read_authenticated_csv(
        audit_root, trajectory_spec["relative_path"], trajectory_spec["sha256"]
    )
    counts = {resolution: _verify_checkpoint_pass(checkpoint_rows, resolution) for resolution in (10, 20)}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.2), sharex="col")
    controls = audit_manifest.get("registered_controls", {})
    for column, resolution in enumerate((10, 20)):
        frozen = controls.get(str(resolution), {})
        title = (
            rf"$N_{{\mathrm{{int}}}}={resolution}$; "
            rf"$\alpha={frozen.get('alpha_label', '?')}$, "
            rf"$r={frozen.get('r', '?')}$, $m_{{\max}}={frozen.get('m_max', '?')}$"
        )
        axes[0, column].set_title(title, fontsize=10.5)
        for method in METHODS:
            rows = _curve(trajectory_rows, resolution, method)
            max_checkpoint = max(
                int(row["checkpoint_nof"])
                for row in checkpoint_rows
                if int(row["n_internal"]) == resolution
            )
            _verify_curve_endpoint(rows, resolution, method, max_checkpoint)
            nof = np.asarray([int(row["nof"]) for row in rows], dtype=np.int64)
            i_perp = np.asarray([float(row["i_perp"]) for row in rows], dtype=np.float64)
            barrier = np.asarray([float(row["barrier_error"]) for row in rows], dtype=np.float64)
            if not np.all(np.isfinite(i_perp)) or not np.all(np.isfinite(barrier)):
                raise PlotGateError("metric curve contains a non-finite value")
            if np.any(i_perp <= 0.0) or np.any(barrier <= 0.0):
                raise PlotGateError("log-scale metric curve contains a non-positive value")
            extension = np.asarray([_truth(row["is_locf_extension"]) for row in rows])
            actual = ~extension
            for row_index, values in enumerate((i_perp, barrier)):
                axis = axes[row_index, column]
                axis.step(
                    nof[actual],
                    values[actual],
                    where="post",
                    color=COLORS[method],
                    linewidth=1.8,
                    label=method if column == 0 and row_index == 0 else None,
                )
                if np.any(extension):
                    first_extension = int(np.flatnonzero(extension)[0])
                    start = max(0, first_extension - 1)
                    axis.step(
                        nof[start:],
                        values[start:],
                        where="post",
                        color=COLORS[method],
                        linewidth=1.6,
                        linestyle="--",
                    )
        max_checkpoint = max(
            int(row["checkpoint_nof"])
            for row in checkpoint_rows
            if int(row["n_internal"]) == resolution
        )
        for row_index in range(2):
            axes[row_index, column].set_yscale("log")
            axes[row_index, column].set_xlim(0, max_checkpoint)
            axes[row_index, column].grid(True, which="both", alpha=0.22, linewidth=0.6)
        axes[0, column].text(
            0.98,
            0.05,
            f"{counts[resolution]}/{counts[resolution]} Article checkpoints\n"
            r"($I_\perp$ and $\Delta E$ both strict)",
            transform=axes[0, column].transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
        )

    axes[0, 0].set_ylabel(r"$I_\perp$")
    axes[1, 0].set_ylabel(r"$|\Delta E_{\mathrm{linear500}}-\Delta E_{\mathrm{ref}}|$")
    axes[1, 0].set_xlabel("Number of Forces")
    axes[1, 1].set_xlabel("Number of Forces")
    figure.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.005))
    figure.tight_layout(rect=(0, 0, 1, 0.965))

    with atomic_output_path(output_pdf, suffix=".pdf") as temporary:
        figure.savefig(temporary, format="pdf", bbox_inches="tight")
    with atomic_output_path(output_png, suffix=".png") as temporary:
        figure.savefig(temporary, format="png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    if provenance_path is None:
        provenance_path = manifest_path.with_name("FIGURE_PROVENANCE.md")
    provenance_lines = [
        "# QM10 IR-IVM comparison figure provenance",
        "",
        "This 2x2 figure is drawn only from the two hash-authenticated tables issued by the "
        "complete independent audit. No optimizer archive or Article archive is read by the "
        "plotting program.",
        "",
        f"- Audit manifest SHA-256: `{sha256_file(audit_manifest_path)}`",
        f"- Checkpoint table SHA-256: `{checkpoint_spec['sha256']}`",
        f"- Public trajectory table SHA-256: `{trajectory_spec['sha256']}`",
        f"- Plot source SHA-256: `{sha256_file(Path(__file__))}`",
        f"- PDF SHA-256: `{sha256_file(output_pdf)}`",
        f"- PNG SHA-256: `{sha256_file(output_png)}`",
        "- N=10 plotting range: through the Article terminal at 951 NoF; a final dashed "
        "segment denotes LOCF when the formal IR-IVM run stopped earlier.",
        "- N=20 plotting range: truncated at the Article terminal at 2377 NoF.",
        "- Metrics: article-language discrete I_perp and the absolute error of the "
        "coordinate-wise linear-500 densified energy barrier.",
        "",
    ]
    atomic_write_bytes(
        provenance_path, ("\n".join(provenance_lines)).encode("utf-8")
    )

    figure_manifest = {
        "schema_version": 1,
        "status": "awaiting_poppler_visual_qa",
        "audit_manifest_relative_path": str(audit_manifest_path.name),
        "audit_manifest_sha256": sha256_file(audit_manifest_path),
        "plot_source_sha256": sha256_file(Path(__file__)),
        "outputs": {
            output_pdf.name: sha256_file(output_pdf),
            output_png.name: sha256_file(output_png),
            provenance_path.name: sha256_file(provenance_path),
        },
        "authenticated_inputs": {
            "checkpoint_comparison.csv": checkpoint_spec["sha256"],
            "public_trajectory_metrics.csv": trajectory_spec["sha256"],
            "registration_sha256": audit_manifest["registration"]["sha256"],
            "result_sha256": {
                str(item["n_internal"]): item["result_sha256"]
                for item in audit_manifest["runs"]
            },
            "receipt_sha256": {
                str(item["n_internal"]): item["receipt_sha256"]
                for item in audit_manifest["runs"]
            },
        },
        "checkpoint_counts": {str(key): value for key, value in counts.items()},
        "methods": list(METHODS),
        "densification_count": 500,
        "reference_barrier": 67.50188060819488,
        "qa_png_sha256": None,
        "cryptographic_provenance_complete": True,
        "formal_provenance_complete": False,
    }
    atomic_write_json(manifest_path, figure_manifest)


def main() -> int:
    paths = derive_campaign_paths(__file__)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit-manifest",
        type=Path,
        default=paths.campaign_root / "reports/audit/audit_manifest.json",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=paths.campaign_root / "figures/q10_unified_cycle_irivm_vs_article_2x2.pdf",
    )
    parser.add_argument(
        "--png",
        type=Path,
        default=paths.campaign_root / "figures/q10_unified_cycle_irivm_vs_article_2x2.png",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=paths.campaign_root / "figures/figure_manifest.json",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=paths.campaign_root / "figures/FIGURE_PROVENANCE.md",
    )
    arguments = parser.parse_args()
    try:
        generate_figure(
            arguments.audit_manifest,
            arguments.pdf,
            arguments.png,
            arguments.manifest,
            arguments.provenance,
        )
    except (FormalIOError, PlotGateError, OSError, KeyError, ValueError) as exc:
        print(f"PLOT REFUSED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
