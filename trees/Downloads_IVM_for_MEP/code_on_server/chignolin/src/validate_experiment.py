#!/usr/bin/env python3
"""Fail-closed validation of Chignolin production outputs and audits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from chignolin_core import TANGENT_DEFINITION, array_sha256, load_common_path


def scalar_text(value) -> str:
    return str(np.asarray(value).item())


def validate_result(
    path: Path,
    common_path: np.ndarray,
    common_path_file: Path,
    role: str,
):
    failures = []
    with np.load(path, allow_pickle=False) as saved:
        files = set(saved.files)
        initial_path = np.asarray(saved["initial_path"], dtype=float)
        calls = np.asarray(saved["calls_per_image"], dtype=int)
        counter_vectors = np.asarray(saved["counter_vectors"], dtype=int)
        mean_calls = np.asarray(
            saved["mean_calls_per_internal_image"], dtype=float
        )
        total_internal = np.asarray(
            saved["total_internal_joint_evaluations"], dtype=int
        )
        tangent = scalar_text(saved["tangent_definition"])
        run_label = scalar_text(saved["run_label"]) if "run_label" in files else ""
        stored_calls = np.asarray(saved["stored_path_calls"], dtype=int)

        if not np.array_equal(initial_path, common_path):
            failures.append("initial_path differs from common IDPP path")
        if scalar_text(saved["initial_path_sha256"]) != array_sha256(common_path):
            failures.append("initial_path SHA-256 mismatch")
        declared_initial_path_file = Path(
            scalar_text(saved["initial_path_file"])
        ).expanduser().resolve()
        if declared_initial_path_file != common_path_file:
            failures.append(
                "declared initial_path_file differs from validated common path"
            )
        if tangent != TANGENT_DEFINITION:
            failures.append(f"unexpected tangent definition: {tangent}")
        if "normalized_i_perp" in files:
            failures.append("forbidden normalized_i_perp output exists")
        if counter_vectors.shape != (len(calls), 22):
            failures.append(
                f"counter_vectors shape {counter_vectors.shape} != {(len(calls), 22)}"
            )
        else:
            internal = counter_vectors[:, 1:-1]
            if not np.array_equal(np.max(internal, axis=1), calls):
                failures.append("x-axis is not max internal-image cache misses")
            if not np.allclose(np.mean(internal, axis=1), mean_calls):
                failures.append("mean internal-image call history mismatch")
            if not np.array_equal(np.sum(internal, axis=1), total_internal):
                failures.append("internal-image sum history mismatch")
            if np.any(np.diff(counter_vectors, axis=0) < 0):
                failures.append("per-image counter history decreases")
        if np.any(np.diff(calls) <= 0):
            failures.append("call labels are not strictly increasing")
        if role == "ivm":
            if "production" not in run_label.lower():
                failures.append(
                    f"IVM run_label {run_label!r} is not marked production"
                )
            if not np.array_equal(calls, np.arange(1, 501)):
                failures.append("IVM production curve is not exactly calls 1..500")
            if not np.array_equal(stored_calls, calls):
                failures.append("IVM production does not store every call path")
            if any(name.startswith("merged") for name in files):
                failures.append("IVM production file contains merged-run metadata")

    audit_path = path.with_suffix(".audit.json")
    if not audit_path.exists():
        failures.append("missing result counter audit JSON")
        audit = {}
    else:
        audit = json.loads(audit_path.read_text())
        if not audit.get("consistent", False):
            failures.append("result counter audit is inconsistent")
        if (
            audit.get("backend_raw_joint_evaluations")
            != audit.get("sum_image_joint_evaluations")
        ):
            failures.append("backend raw calls != summed per-image cache misses")
        if counter_vectors.size and (
            list(counter_vectors[-1])
            != audit.get("per_image_joint_evaluations")
        ):
            failures.append("last stored counter vector != final audit vector")
    return {
        "path": str(path),
        "role": role,
        "passed": not failures,
        "failures": failures,
        "records": int(len(calls)),
        "final_max_calls_per_internal_image": int(calls[-1]),
        "final_sum_over_20": float(total_internal[-1] / 20.0),
        "final_total_internal_openmm_evaluations": int(total_internal[-1]),
        "final_total_22_image_openmm_evaluations": audit.get(
            "backend_raw_joint_evaluations"
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--initial-path-file",
        default="",
        help=(
            "Common path to validate. If omitted, use the path declared by "
            "the IVM production result."
        ),
    )
    parser.add_argument("--ivm", required=True)
    parser.add_argument("--neb", default="")
    parser.add_argument("--dense-ivm", default="")
    parser.add_argument("--dense-neb", default="")
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    if args.initial_path_file:
        common_path_file = Path(args.initial_path_file).expanduser()
        if not common_path_file.is_absolute():
            common_path_file = root / common_path_file
        common_path_file = common_path_file.resolve()
    else:
        with np.load(Path(args.ivm).resolve(), allow_pickle=False) as saved:
            common_path_file = Path(
                scalar_text(saved["initial_path_file"])
            ).expanduser().resolve()
    common_path = load_common_path(common_path_file)

    results = [
        validate_result(
            Path(args.ivm).resolve(),
            common_path,
            common_path_file,
            "ivm",
        )
    ]
    if args.neb:
        results.append(
            validate_result(
                Path(args.neb).resolve(),
                common_path,
                common_path_file,
                "neb",
            )
        )

    source_failures = []
    for source in (root / "src").glob("*.py"):
        if source.name == Path(__file__).name:
            continue
        text = source.read_text().lower()
        if "improvedtangent" in text or "improved tangent" in text:
            source_failures.append(
                f"forbidden tangent implementation reference in {source.name}"
            )

    diagnostic_results = []
    for label, dense_path in (
        ("ivm", args.dense_ivm),
        ("neb", args.dense_neb),
    ):
        if not dense_path:
            continue
        path = Path(dense_path).resolve()
        with np.load(path, allow_pickle=False) as saved:
            backend = int(saved["diagnostic_backend_raw_joint_evaluations"])
            misses = int(saved["diagnostic_evaluator_cache_misses"])
            calls = np.asarray(saved["calls"], dtype=int)
        diagnostic_results.append(
            {
                "role": label,
                "path": str(path),
                "records": int(len(calls)),
                "backend_raw_joint_evaluations": backend,
                "diagnostic_cache_misses": misses,
                "passed": backend == misses,
                "excluded_from_optimization_axis": True,
            }
        )

    failures = [
        failure
        for result in results
        for failure in result["failures"]
    ] + source_failures
    failures.extend(
        f"{result['role']} diagnostic backend/cache mismatch"
        for result in diagnostic_results
        if not result["passed"]
    )
    if len(results) == 2:
        with np.load(results[0]["path"], allow_pickle=False) as ivm_saved, np.load(
            results[1]["path"], allow_pickle=False
        ) as neb_saved:
            if not np.array_equal(
                ivm_saved["initial_path"], neb_saved["initial_path"]
            ):
                failures.append("IVM and NEB initial paths are not elementwise equal")

    report = {
        "passed": not failures,
        "tangent_definition": TANGENT_DEFINITION,
        "common_initial_path_file": str(common_path_file),
        "common_initial_path_sha256": array_sha256(common_path),
        "results": results,
        "diagnostics": diagnostic_results,
        "source_failures": source_failures,
        "failures": failures,
    }
    output = root / "reports" / "production_validation.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
