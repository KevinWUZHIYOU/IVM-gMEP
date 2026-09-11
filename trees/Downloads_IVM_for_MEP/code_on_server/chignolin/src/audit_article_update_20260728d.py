"""Read-only final audit for the Chignolin article-update experiment bundle."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin"
)
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures" / "article_update_20260728d"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def scalar(value):
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else array


def audit_productions() -> dict:
    manifest = read_json(
        REPORTS / "confirmed1000_productions_20260728c" / "manifest.json"
    )
    rows = []
    initial_hashes = set()
    for record in manifest["records"]:
        result_path = Path(record["result_file"])
        audit_path = Path(record["audit_file"])
        audit = read_json(audit_path)
        with np.load(result_path, allow_pickle=False) as saved:
            counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
            calls = np.asarray(saved["calls_per_image"], dtype=np.int64)
            initial_hashes.add(str(scalar(saved["initial_path_sha256"])))
            final_total = int(counters[-1].sum())
            final_max_internal = int(counters[-1, 1:-1].max())
            final_min_internal = int(counters[-1, 1:-1].min())
            monotone = bool(np.all(np.diff(counters, axis=0) >= 0))
            final_vector_equal = bool(
                np.array_equal(
                    counters[-1],
                    np.asarray(audit["per_image_joint_evaluations"], dtype=np.int64),
                )
            )
            calls_equal = bool(np.array_equal(calls, counters.max(axis=1)))
        assert record["status"] == "completed"
        assert record["valid"] is True
        assert audit["consistent"] is True
        assert audit["backend_raw_joint_evaluations"] == audit[
            "sum_image_joint_evaluations"
        ]
        assert all(audit["per_image_request_identity_ok"])
        assert all(audit["per_image_query_outcome_identity_ok"])
        assert final_vector_equal
        assert monotone
        assert calls_equal
        assert final_total == record["final_number_of_force_evaluations"]
        assert final_max_internal == record["final_max_calls_per_internal_image"]
        assert final_max_internal <= 1000
        rows.append(
            {
                "name": record["name"],
                "max_internal": final_max_internal,
                "min_internal": final_min_internal,
                "total_force_evaluations": final_total,
                "backend_equals_image_sum": True,
                "final_counter_vector_equal": final_vector_equal,
                "counter_monotone": monotone,
            }
        )
    assert len(initial_hashes) == 1
    return {"common_initial_path_hash": initial_hashes.pop(), "rows": rows}


def audit_dense() -> dict:
    manifest = read_json(
        REPORTS / "confirmed1000_dense_20260728c" / "manifest.json"
    )
    rows = []
    for record in manifest["records"]:
        audit = read_json(Path(record["audit_file"]))
        with np.load(Path(record["result_file"]), allow_pickle=False) as saved:
            saved_backend = int(
                scalar(saved["diagnostic_backend_raw_joint_evaluations"])
            )
            saved_misses = int(
                scalar(saved["diagnostic_evaluator_cache_misses"])
            )
            excluded = bool(
                audit["counts_excluded_from_optimization_axis"]
            )
        assert record["status"] == "completed"
        assert record["returncode"] == 0
        assert (
            audit["diagnostic_backend_raw_joint_evaluations"]
            == audit["diagnostic_evaluator_cache_misses"]
            == record["diagnostic_joint_evaluations"]
            == saved_backend
            == saved_misses
        )
        assert audit["counts_excluded_from_optimization_axis"] is True
        assert excluded
        rows.append(
            {
                "name": record["name"],
                "diagnostic_joint_evaluations": saved_backend,
                "subdivisions": record["selected_subdivision_counts"],
                "excluded_from_optimization_axis": excluded,
            }
        )
    return {"rows": rows}


def audit_fixed_alpha() -> dict:
    report_dir = REPORTS / "ivm_fixed_alpha_extended_20260728a"
    manifest = read_json(report_dir / "manifest.json")
    dense_manifest = read_json(report_dir / "dense_manifest.json")
    expected = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 20.0, 30.0, 100.0, 300.0, 1000.0]
    observed = []
    rows = []
    for record in manifest["records"]:
        alpha = float(record["alpha"])
        observed.append(alpha)
        with np.load(Path(record["result_file"]), allow_pickle=False) as saved:
            active = np.asarray(saved["active_alpha"], dtype=float)
            alpha_mode = str(scalar(saved["alpha_mode"]))
            stored_alpha = float(scalar(saved["alpha"]))
            stored_alpha_max = float(scalar(saved["alpha_max"]))
            counters = np.asarray(saved["counter_vectors"], dtype=np.int64)
        audit = read_json(Path(record["audit_file"]))
        assert record["valid"] is True
        assert alpha_mode == "fixed"
        assert stored_alpha == alpha
        assert stored_alpha_max >= alpha
        assert np.all(active == alpha)
        assert audit["consistent"] is True
        assert audit["backend_raw_joint_evaluations"] == audit[
            "sum_image_joint_evaluations"
        ]
        assert int(counters[-1].sum()) == record[
            "final_number_of_force_evaluations"
        ]
        rows.append(
            {
                "alpha": alpha,
                "status": record["status"],
                "n_records": int(record["n_records"]),
                "final_total_force_evaluations": int(
                    record["final_number_of_force_evaluations"]
                ),
                "active_alpha_constant": True,
            }
        )
    assert observed == expected
    for record in dense_manifest["records"]:
        for checkpoint in record["checkpoints"]:
            assert checkpoint["actual_number_of_force_evaluations"] == checkpoint[
                "requested_force_budget"
            ]
        audit = read_json(Path(record["audit_file"]))
        assert audit["diagnostic_backend_raw_joint_evaluations"] == audit[
            "diagnostic_evaluator_cache_misses"
        ]
        assert audit["counts_excluded_from_optimization_axis"] is True
    return {
        "alphas": observed,
        "max_alpha": max(observed),
        "requested_force_budgets": dense_manifest[
            "requested_total_force_budgets"
        ],
        "rows": rows,
    }


def audit_figure_data() -> dict:
    summary = read_json(FIGURES / "chignolin_method_comparison_summary.json")
    method_rows = []
    for method, details in summary["methods"].items():
        with np.load(Path(details["run"]), allow_pickle=False) as saved:
            final_total = int(
                np.asarray(saved["counter_vectors"], dtype=np.int64)[-1].sum()
            )
        assert final_total == details["final_number_of_force_evaluations"]
        method_rows.append(
            {
                "method": method,
                "final_total_force_evaluations": final_total,
                "single_run": str(details["run"]),
            }
        )
    assert summary["primary_cost_axis"] == "counter_vectors.sum(axis=1)"
    assert summary["primary_cost_label"] == "Number of Force Evaluations"
    assert summary["cost_includes_all_22_images"] is True
    assert summary["dense_diagnostic_calls_excluded_from_optimization_axis"] is True

    alpha_csv = FIGURES / "data" / "chignolin_ivm_fixed_alpha_sensitivity.csv"
    alpha_values = set()
    for row in csv.DictReader(alpha_csv.open()):
        alpha_values.add(float(row["alpha"]))
    assert max(alpha_values) == 1000.0
    assert alpha_values == {
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
        20.0,
        30.0,
        100.0,
        300.0,
        1000.0,
    }
    return {
        "methods": method_rows,
        "primary_cost_axis": summary["primary_cost_axis"],
        "primary_cost_label": summary["primary_cost_label"],
        "fixed_alpha_csv_max": max(alpha_values),
    }


def main() -> None:
    output = {
        "productions": audit_productions(),
        "dense": audit_dense(),
        "fixed_alpha": audit_fixed_alpha(),
        "figure_data": audit_figure_data(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
