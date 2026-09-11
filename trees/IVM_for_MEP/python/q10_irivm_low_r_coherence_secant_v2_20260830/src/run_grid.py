#!/usr/bin/env python3
"""Run one staged resolution of the preregistered low-r QM10 campaign."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from q10_irivm_atomic_tangent_secant_core import (
    ARTICLE_NF_BUDGET,
    AUDIT_FILE,
    CAMPAIGN_ROOT,
    EXPERIMENT_ID,
    REGISTRATION,
    REGISTERED_CONTROLS,
    THIS_FILE,
    Controls,
    run_one,
    validate_preregistration,
)


THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
N20_GATE_MANIFEST = (
    CAMPAIGN_ROOT / "reports/n20_gate_audit/audit_manifest.json"
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fields = (
        "status",
        "n20_gate_manifest_sha256_token",
        "n_internal",
        "alpha",
        "alpha_label",
        "r",
        "m_max",
        "run_id",
        "output",
        "output_sha256",
        "stop_reason",
        "failure_message",
        "terminal_article_nf",
        "last_complete_article_nf",
        "accepted_states",
        "transactions",
        "committed_transactions",
        "rolled_back_transactions",
        "recovery_backtrack_states_cleared_total",
        "wall_seconds",
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})
    os.replace(temporary, path)


def runtime_identity():
    return {
        "sys_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_ENVIRONMENT
        },
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }


def controls_from_tuple(item):
    return Controls(int(item[0]), float(item[1]), float(item[2]), int(item[3]))


def refresh_n20_gate_audit():
    """Regenerate the N20 gate with the registered audit under this runtime."""

    subprocess.run(
        [
            sys.executable,
            str(AUDIT_FILE),
            "--n-internal",
            "20",
            "--output-dir",
            str(N20_GATE_MANIFEST.parent),
        ],
        cwd=str(CAMPAIGN_ROOT),
        check=True,
    )


def _gate_csv_bool(value, field):
    if value == "True":
        return True
    if value == "False":
        return False
    raise RuntimeError("N10 N20-audit gate has invalid boolean: " + field)


def _gate_scalar(archive, name):
    if name not in archive:
        raise RuntimeError("N10 N20-audit gate NPZ lacks: " + name)
    value = np.asarray(archive[name])
    if value.shape != ():
        raise RuntimeError("N10 N20-audit gate NPZ scalar drifted: " + name)
    return value.item()


def validate_n20_gate_for_n10(registration, manifest_path=N20_GATE_MANIFEST):
    """Require a complete, authenticated strict N20 audit before N10 starts."""

    path = Path(manifest_path)
    if not path.is_file():
        raise RuntimeError("N10 is gated by the missing N20 audit manifest: " + str(path))
    validated_manifest_token = sha256(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    registered_n20 = sum(int(item[0]) == 20 for item in REGISTERED_CONTROLS)
    resolution = manifest.get("resolution_results", {}).get("20", {})
    expected_registration_sha = sha256(REGISTRATION)
    expected_core_sha = sha256(THIS_FILE)
    expected_audit_sha = sha256(AUDIT_FILE)
    required = {
        "campaign_id": manifest.get("campaign_id") == EXPERIMENT_ID,
        "audit_status": manifest.get("audit_status") == "complete",
        "requested_resolutions": manifest.get("requested_resolutions") == [20],
        "registration_path": manifest.get("registration") == str(REGISTRATION),
        "registration_sha256": (
            manifest.get("registration_sha256") == expected_registration_sha
        ),
        "core_source": manifest.get("core_source") == str(THIS_FILE),
        "core_source_sha256": (
            manifest.get("core_source_sha256") == expected_core_sha
        ),
        "audit_script": manifest.get("audit_script") == str(AUDIT_FILE),
        "audit_script_sha256": (
            manifest.get("audit_script_sha256") == expected_audit_sha
        ),
        "registration_campaign": (
            registration.get("campaign_id") == EXPERIMENT_ID
        ),
        "registration_core_sha256": (
            registration.get("implementation_sha256") == expected_core_sha
        ),
        "integrity_failure_count": (
            int(manifest.get("integrity_failure_count", -1)) == 0
        ),
        "integrity_errors": manifest.get("integrity_errors") == [],
        "requested_member_count": (
            int(manifest.get("requested_member_count", -1)) == registered_n20
        ),
        "audited_member_count": (
            int(manifest.get("audited_member_count", -1)) == registered_n20
        ),
        "integrity_valid_member_count": (
            int(manifest.get("integrity_valid_member_count", -1))
            == registered_n20
        ),
        "resolution_registered_member_count": (
            int(resolution.get("registered_member_count", -1))
            == registered_n20
        ),
        "resolution_integrity_valid_member_count": (
            int(resolution.get("integrity_valid_member_count", -1))
            == registered_n20
        ),
        "resolution_selectable_member_count": (
            int(resolution.get("selectable_member_count", -1))
            == registered_n20
        ),
        "resolution_strict_pass_count": (
            int(resolution.get("strict_pass_count", 0)) > 0
        ),
        "all_requested_resolutions_strictly_pass": (
            manifest.get("all_requested_resolutions_strictly_pass") is True
        ),
    }
    failed = sorted(name for name, valid in required.items() if not valid)
    if failed:
        raise RuntimeError(
            "N10 N20-audit gate failed: " + ", ".join(failed)
        )

    expected_artifacts = {
        "candidate_summary_csv": "candidate_summary.csv",
        "checkpoint_comparison_csv": "checkpoint_comparison.csv",
        "human_report": "AUDIT_REPORT.md",
    }
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
        raise RuntimeError("N10 N20-audit gate artifact set drifted")
    authenticated_artifacts = {}
    for name, filename in expected_artifacts.items():
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise RuntimeError("N10 N20-audit gate artifact record drifted: " + name)
        artifact_path = Path(record.get("path", ""))
        expected_path = path.parent / filename
        if (
            not artifact_path.is_file()
            or artifact_path.resolve() != expected_path.resolve()
            or record.get("sha256") != sha256(artifact_path)
        ):
            raise RuntimeError("N10 N20-audit gate artifact authentication failed: " + name)
        authenticated_artifacts[name] = artifact_path

    with authenticated_artifacts["candidate_summary_csv"].open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    expected_controls = {
        control.run_id: control
        for control in (
            controls_from_tuple(item)
            for item in REGISTERED_CONTROLS
            if int(item[0]) == 20
        )
    }
    observed_run_ids = [row.get("run_id", "") for row in rows]
    if (
        len(rows) != registered_n20
        or len(set(observed_run_ids)) != registered_n20
        or set(observed_run_ids) != set(expected_controls)
    ):
        raise RuntimeError("N10 N20-audit gate summary member identity drifted")

    strict_pass_count = 0
    for row in rows:
        run_id = row["run_id"]
        control = expected_controls[run_id]
        expected_config = control.as_dict()
        try:
            row_identity_ok = bool(
                int(row.get("n_internal", -1)) == 20
                and float(row.get("alpha", "nan")) == float(control.alpha)
                and row.get("alpha_label") == control.alpha_label
                and float(row.get("r", "nan")) == float(control.r)
                and int(row.get("m_max", -1)) == int(control.m_max)
            )
        except (TypeError, ValueError):
            row_identity_ok = False
        if not row_identity_ok:
            raise RuntimeError("N10 N20-audit gate summary control drifted: " + run_id)
        if (
            not _gate_csv_bool(row.get("integrity_valid"), "integrity_valid")
            or row.get("integrity_error", "") != ""
            or not _gate_csv_bool(
                row.get("valid_for_selection"), "valid_for_selection"
            )
            or row.get("failure_message", "") != ""
        ):
            raise RuntimeError("N10 N20-audit gate summary member is not valid: " + run_id)
        strict_pass = _gate_csv_bool(row.get("strict_pass"), "strict_pass")
        strict_pass_count += int(strict_pass)

        source = Path(row.get("source", ""))
        receipt_path = Path(row.get("receipt", ""))
        expected_source = CAMPAIGN_ROOT / "runs" / "n20" / (run_id + ".npz")
        expected_receipt = (
            CAMPAIGN_ROOT
            / "reports"
            / "receipts"
            / "n20"
            / (run_id + ".receipt.json")
        )
        if (
            not source.is_file()
            or source.resolve() != expected_source.resolve()
            or row.get("source_sha256") != sha256(source)
            or not receipt_path.is_file()
            or receipt_path.resolve() != expected_receipt.resolve()
            or row.get("receipt_sha256") != sha256(receipt_path)
        ):
            raise RuntimeError("N10 N20-audit gate result authentication failed: " + run_id)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("run_id") != run_id
            or receipt.get("controls") != expected_config
            or str(Path(receipt.get("output", "")).resolve()) != str(source.resolve())
            or receipt.get("output_sha256") != sha256(source)
            or receipt.get("status") not in ("completed", "existing_validated")
            or receipt.get("n20_gate_manifest_sha256_token")
            != "not_applicable"
        ):
            raise RuntimeError("N10 N20-audit gate receipt identity drifted: " + run_id)

        with np.load(source, allow_pickle=False) as archive:
            if (
                str(_gate_scalar(archive, "experiment_id")) != EXPERIMENT_ID
                or str(
                    _gate_scalar(
                        archive, "n20_gate_manifest_sha256_token"
                    )
                )
                != "not_applicable"
                or str(_gate_scalar(archive, "run_id")) != run_id
                or str(_gate_scalar(archive, "source_sha256")) != expected_core_sha
                or str(_gate_scalar(archive, "registration_sha256"))
                != expected_registration_sha
            ):
                raise RuntimeError("N10 N20-audit gate NPZ identity drifted: " + run_id)
            try:
                stored_config = json.loads(str(_gate_scalar(archive, "config_json")))
            except Exception as error:
                raise RuntimeError(
                    "N10 N20-audit gate NPZ config drifted: " + run_id
                ) from error
            terminal = int(_gate_scalar(archive, "terminal_article_nf"))
            last_complete = int(
                _gate_scalar(archive, "last_complete_article_nf")
            )
            stop_reason = str(_gate_scalar(archive, "stop_reason"))
            failure_message = str(_gate_scalar(archive, "failure_message"))
            per_image = np.asarray(
                archive["per_image_joint_evaluations"], dtype=np.int64
            )
            article_nf = np.asarray(archive["article_nf"], dtype=np.int64)
            usable = np.asarray(archive["article_usable_state"], dtype=bool)
        nof_ok = bool(
            stored_config == expected_config
            and per_image.shape == (22,)
            and np.all(per_image >= 0)
            and int(per_image[0]) == 0
            and int(per_image[-1]) == 0
            and int(np.sum(per_image)) == terminal
            and article_nf.ndim == 1
            and article_nf.size > 0
            and np.all(np.diff(article_nf) > 0)
            and int(article_nf[0]) == 20
            and int(article_nf[-1]) == last_complete == terminal
            and 20 <= terminal <= ARTICLE_NF_BUDGET
            and usable.shape == article_nf.shape
            and np.all(usable)
        )
        receipt_ok = bool(
            receipt.get("stop_reason") == stop_reason
            and receipt.get("failure_message", "") == failure_message == ""
            and int(receipt.get("terminal_article_nf", -1)) == terminal
            and int(receipt.get("last_complete_article_nf", -1))
            == last_complete
            and row.get("stop_reason") == stop_reason
            and int(row.get("terminal_number_of_forces", -1)) == terminal
            and int(row.get("last_complete_number_of_forces", -1))
            == last_complete
            and stop_reason in ("budget", "stationary", "outer_limit")
        )
        if not nof_ok or not receipt_ok:
            raise RuntimeError("N10 N20-audit gate Number-of-Forces drifted: " + run_id)

    if (
        strict_pass_count <= 0
        or strict_pass_count != int(resolution.get("strict_pass_count", -1))
    ):
        raise RuntimeError("N10 N20-audit gate strict-pass summary drifted")
    if sha256(path) != validated_manifest_token:
        raise RuntimeError("N10 N20-audit gate manifest changed during validation")
    return manifest, validated_manifest_token


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="debug/resume convenience; a formal complete campaign omits this",
    )
    parser.add_argument("--n-internal", type=int, choices=(10, 20), default=None)
    args = parser.parse_args()
    if args.n_internal is None:
        raise RuntimeError(
            "formal execution requires explicit --n-internal 20 or --n-internal 10"
        )
    if args.jobs != 1:
        raise RuntimeError("formal campaign is serial-only: jobs must be 1")
    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            raise RuntimeError(name + " must equal 1")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("PYTHONHASHSEED must equal 0")

    registration = validate_preregistration()
    if args.n_internal == 10:
        refresh_n20_gate_audit()
        _, gate_manifest_token = validate_n20_gate_for_n10(registration)
    else:
        gate_manifest_token = "not_applicable"
    controls = [controls_from_tuple(item) for item in REGISTERED_CONTROLS]
    controls = [item for item in controls if item.n_internal == args.n_internal]
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be nonnegative")
        controls = controls[: args.limit]

    lock = (
        CAMPAIGN_ROOT
        / "locks/host-local-q10-irivm-low-r-coherence-secant-v2.lock"
    )
    lock.parent.mkdir(parents=True, exist_ok=True)
    report_suffix = f"_n{int(args.n_internal)}"
    report_json = CAMPAIGN_ROOT / f"reports/grid_execution{report_suffix}.json"
    report_csv = CAMPAIGN_ROOT / f"reports/grid_execution{report_suffix}.csv"
    receipt_root = CAMPAIGN_ROOT / "reports/receipts"
    rows = []
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        for index, item in enumerate(controls, start=1):
            destination = (
                CAMPAIGN_ROOT
                / "runs"
                / f"n{item.n_internal}"
                / f"{item.run_id}.npz"
            )
            started = time.perf_counter()
            if destination.exists():
                receipt_path = (
                    receipt_root
                    / f"n{item.n_internal}"
                    / f"{item.run_id}.receipt.json"
                )
                if not receipt_path.exists():
                    raise RuntimeError("existing result lacks receipt: " + str(destination))
                row = json.loads(receipt_path.read_text(encoding="utf-8"))
                if row.get("output_sha256") != sha256(destination):
                    raise RuntimeError("existing result hash differs from receipt")
                with np.load(destination, allow_pickle=False) as payload:
                    stored_experiment = str(payload["experiment_id"].item())
                    stored_gate_manifest_token = str(
                        payload["n20_gate_manifest_sha256_token"].item()
                    )
                    stored_run_id = str(payload["run_id"].item())
                    stored_config = json.loads(str(payload["config_json"].item()))
                    stored_source = str(payload["source_sha256"].item())
                    stored_registration = str(
                        payload["registration_sha256"].item()
                    )
                    q_matrix = np.asarray(payload["q_matrix"], dtype=np.float64)
                    stored_q_sha = str(payload["q_sha256"].item())
                    terminal_nf = int(payload["terminal_article_nf"].item())
                    per_image = np.asarray(
                        payload["per_image_joint_evaluations"], dtype=np.int64
                    )
                    article_nf = np.asarray(payload["article_nf"], dtype=np.int64)
                if stored_experiment != EXPERIMENT_ID:
                    raise RuntimeError("existing result experiment ID drifted")
                if stored_gate_manifest_token != gate_manifest_token:
                    raise RuntimeError("existing result N20-gate token drifted")
                if stored_run_id != item.run_id or stored_config != item.as_dict():
                    raise RuntimeError("existing result control identity drifted")
                if stored_source != sha256(THIS_FILE):
                    raise RuntimeError("existing result source hash drifted")
                if stored_registration != sha256(REGISTRATION):
                    raise RuntimeError("existing result registration hash drifted")
                computed_q_sha = hashlib.sha256(
                    np.ascontiguousarray(q_matrix).tobytes()
                ).hexdigest()
                if stored_q_sha != computed_q_sha:
                    raise RuntimeError("existing result Q hash drifted")
                if (
                    terminal_nf < 0
                    or terminal_nf > ARTICLE_NF_BUDGET
                    or np.any(per_image < 0)
                    or int(np.sum(per_image)) != terminal_nf
                    or article_nf.ndim != 1
                    or article_nf.size == 0
                    or np.any(np.diff(article_nf) < 0)
                    or int(article_nf[-1]) != terminal_nf
                ):
                    raise RuntimeError("existing result Number-of-Forces ledger drifted")
                if row.get("run_id") != item.run_id:
                    raise RuntimeError("existing receipt run ID drifted")
                if (
                    row.get("n20_gate_manifest_sha256_token")
                    != gate_manifest_token
                ):
                    raise RuntimeError("existing receipt N20-gate token drifted")
                if row.get("controls") != item.as_dict():
                    raise RuntimeError("existing receipt controls drifted")
                if str(Path(row.get("output", "")).resolve()) != str(
                    destination.resolve()
                ):
                    raise RuntimeError("existing receipt output path drifted")
                row["status"] = (
                    "failed"
                    if row.get("stop_reason") == "numerical_failure"
                    else "existing_validated"
                )
                row["wall_seconds"] = 0.0
            else:
                try:
                    _, _, receipt = run_one(
                        item,
                        n20_gate_manifest_token=gate_manifest_token,
                    )
                    with np.load(destination, allow_pickle=False) as payload:
                        generated_gate_manifest_token = str(
                            payload[
                                "n20_gate_manifest_sha256_token"
                            ].item()
                        )
                    if (
                        generated_gate_manifest_token != gate_manifest_token
                        or receipt.get("n20_gate_manifest_sha256_token")
                        != gate_manifest_token
                    ):
                        raise RuntimeError("new result N20-gate token drifted")
                    row = dict(receipt)
                    row.update(
                        {
                            "status": (
                                "failed"
                                if receipt["stop_reason"] == "numerical_failure"
                                else "completed"
                            ),
                            "output_sha256": sha256(destination),
                            "wall_seconds": time.perf_counter() - started,
                        }
                    )
                    receipt_path = (
                        receipt_root
                        / f"n{item.n_internal}"
                        / f"{item.run_id}.receipt.json"
                    )
                    atomic_json(receipt_path, row)
                except Exception as error:
                    row = {
                        "status": "failed",
                        **item.as_dict(),
                        "run_id": item.run_id,
                        "n20_gate_manifest_sha256_token": gate_manifest_token,
                        "output": str(destination),
                        "failure_message": f"{type(error).__name__}: {error}",
                        "wall_seconds": time.perf_counter() - started,
                    }
            rows.append(row)
            write_csv(report_csv, rows)
            atomic_json(
                report_json,
                {
                    "campaign_id": registration["campaign_id"],
                    "runtime": runtime_identity(),
                    "requested_member_count": len(controls),
                    "processed_member_count": len(rows),
                    "completed_or_existing": sum(
                        value["status"] in ("completed", "existing_validated")
                        for value in rows
                    ),
                    "failed": sum(value["status"] == "failed" for value in rows),
                    "rows": rows,
                },
            )
            if index == 1 or index % 25 == 0 or index == len(controls):
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "total": len(controls),
                            "failed": sum(
                                value["status"] == "failed" for value in rows
                            ),
                            "current": item.as_dict(),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    failed = [row for row in rows if row["status"] == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} registered runs failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
