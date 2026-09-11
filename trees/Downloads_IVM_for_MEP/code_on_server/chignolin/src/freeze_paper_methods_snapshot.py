#!/usr/bin/env python3
"""Freeze the Chignolin method-comparison code, figures, and provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
import shutil
import sys
from pathlib import Path


SOURCE_FILES = (
    "src/chignolin_core.py",
    "src/forcefield_preconditioner.py",
    "src/prepare_experiment.py",
    "src/run_ivm.py",
    "src/run_neb.py",
    "src/run_string_family.py",
    "src/run_obsm.py",
    "src/postprocess_dense_barrier.py",
    "src/run_method_sweep.py",
    "src/run_selected_method_productions.py",
    "src/run_selected_dense_diagnostics.py",
    "src/run_ivm_fixed_alpha_sensitivity.py",
    "src/run_ivm_alpha_dense_diagnostics.py",
    "src/plot_paper_method_comparison.py",
    "src/plot_ivm_fixed_alpha_sensitivity_total_nf.py",
    "src/freeze_paper_methods_snapshot.py",
    "tests/test_core.py",
    "tests/test_string_methods.py",
    "tests/run_tests.py",
)
REPORT_NAMESPACES = (
    "static_broad_20260727a",
    "static_refine_20260727a",
    "original_broad_20260727a",
    "original_refine_20260727a",
    "obsm_fire_broad_20260727a",
    "obsm_fire_dt_extension_20260727a",
    "obsm_fire_dt_high_20260727a",
    "obsm_fire_beta_reparam_20260727a",
    "paper_method_productions_20260727a",
    "paper_total_nf_dense_20260727a",
    "ivm_fixed_alpha_sensitivity_total_nf_20260727a",
)
FIGURE_DIRECTORY = "figures/paper_methods_total_nf_20260727a"
REPORT_PATH = (
    "reports/chignolin_method_comparison_total_nf_20260727a/README.md"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_with_record(
    root: Path,
    snapshot: Path,
    relative_source: str,
    relative_destination: str,
) -> dict:
    source = root / relative_source
    destination = snapshot / relative_destination
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "source": str(source.resolve()),
        "snapshot_path": relative_destination,
        "sha256": sha256(destination),
        "bytes": destination.stat().st_size,
    }


def package_versions() -> dict:
    versions = {"python": sys.version, "platform": platform.platform()}
    for name in ("numpy", "scipy", "openmm", "ase", "matplotlib"):
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "unknown")
    return versions


def referenced_results(root: Path) -> list[dict]:
    records = []
    summary = json.loads(
        (
            root
            / FIGURE_DIRECTORY
            / "chignolin_method_comparison_summary.json"
        ).read_text()
    )
    for method, payload in summary["methods"].items():
        for role in ("run", "run_audit", "dense"):
            path = Path(payload[role]).resolve()
            records.append(
                {
                    "method": method,
                    "role": role,
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    alpha_manifest = json.loads(
        (
            root
            / "reports"
            / "ivm_fixed_alpha_sensitivity_total_nf_20260727a"
            / "manifest.json"
        ).read_text()
    )
    dense_manifest = json.loads(
        (
            root
            / "reports"
            / "ivm_fixed_alpha_sensitivity_total_nf_20260727a"
            / "dense_manifest.json"
        ).read_text()
    )
    for record in alpha_manifest["records"]:
        for role, key in (
            ("fixed_alpha_run", "result_file"),
            ("fixed_alpha_audit", "audit_file"),
        ):
            path = Path(record[key]).resolve()
            records.append(
                {
                    "method": f"IVM alpha={record['alpha']:g}",
                    "role": role,
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    for record in dense_manifest["records"]:
        for role, key in (
            ("fixed_alpha_dense", "result_file"),
            ("fixed_alpha_dense_audit", "audit_file"),
        ):
            path = Path(record[key]).resolve()
            records.append(
                {
                    "method": f"IVM alpha={record['alpha']:g}",
                    "role": role,
                    "path": str(path),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--snapshot-name", default="paper_methods_total_nf_20260727a"
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    snapshot = root / "snapshots" / args.snapshot_name
    if snapshot.exists() and any(snapshot.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite nonempty snapshot: {snapshot}"
        )
    snapshot.mkdir(parents=True, exist_ok=True)

    copied = []
    for relative in SOURCE_FILES:
        copied.append(
            copy_with_record(root, snapshot, relative, relative)
        )
    copied.append(
        copy_with_record(root, snapshot, REPORT_PATH, "README.md")
    )
    for namespace in REPORT_NAMESPACES:
        source_dir = root / "reports" / namespace
        for filename in ("manifest.json", "dense_manifest.json"):
            source = source_dir / filename
            if source.is_file():
                destination = (
                    f"manifests/{namespace}_{filename}"
                )
                copied.append(
                    copy_with_record(
                        root,
                        snapshot,
                        str(source.relative_to(root)),
                        destination,
                    )
                )
    figure_root = root / FIGURE_DIRECTORY
    for source in sorted(figure_root.rglob("*")):
        if source.is_file():
            relative = source.relative_to(figure_root)
            copied.append(
                copy_with_record(
                    root,
                    snapshot,
                    str(source.relative_to(root)),
                    str(Path("figures") / relative),
                )
            )

    references = referenced_results(root)
    with (snapshot / "referenced_result_manifest.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "role", "path", "sha256", "bytes"),
        )
        writer.writeheader()
        writer.writerows(references)
    manifest = {
        "snapshot_name": args.snapshot_name,
        "source_root": str(root),
        "created_by": str(Path(__file__).resolve()),
        "initial_path_sha256": (
            "57b61ab0fd61d2a177c91e6b9e70b7ff72b40f7f4f03c41f91d80d8efa5e2996"
        ),
        "primary_cost_axis": "counter_vectors.sum(axis=1)",
        "primary_cost_label": "Number of Force Evaluations",
        "cost_includes_all_22_images": True,
        "diagnostic_calls_excluded_from_optimization_axis": True,
        "environment": package_versions(),
        "copied_artifacts": copied,
        "referenced_large_results": references,
    }
    output = snapshot / "manifest.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
