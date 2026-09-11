"""O_EXCL compact metric traces produced from verified full-state batches."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any, Mapping

import numpy as np

from .accounting import file_sha256
from .integrity import (
    is_within,
    load_json_object,
    verify_source_identity,
    write_json_new,
)
from .kgrid_metric_evidence import visit_candidate_evidence
from .kgrid_selection import (
    BUDGET_GRID,
    K_VALUES,
    _metric_pair,
    select_verified_compact_metric_traces,
)
from .kgrid_spec import INITIAL_ARRAY_SHA256, build_run


TRACE_SCHEMA = "bb-ase-ode-kgrid-compact-metric-trace-v1"
COMPACT_AUDIT_SCHEMA = "bb-neb-aseode-kgrid-compact-trace-audit-v1"
METRIC_SOURCE = "recomputed from verified path32/E32/full-force32"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _readonly_regular(path, label):
    status = path.lstat()
    if not stat.S_ISREG(status.st_mode) or path.is_symlink() \
            or stat.S_IMODE(status.st_mode) & 0o222:
        raise ValueError(f"{label} is not a read-only regular nonsymlink file")


def _initial_path_sha256(value):
    array = np.asarray(value, dtype="<f8").reshape((32, -1, 3))
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _recompute_compact_trace(run_dir, terminal_audit_path, *, expected_k):
    """Rebuild a compact trace from independently reaudited full history."""

    rows = []
    source_rows = []
    current = {"path": None, "energy": None, "force": None}
    def consume(row):
        paid_path = np.asarray(row["paid_path32_A"], dtype=np.float64)
        disposition = str(row["disposition"])
        if disposition in {"current_initial", "accepted"}:
            current["path"] = paid_path
            current["energy"] = np.asarray(row["energy32_eV"], dtype=np.float64)
            current["force"] = np.asarray(row["full_force32_eV_per_A"], dtype=np.float64)
        elif current["path"] is None:
            raise ValueError("rejected batch precedes evaluated current")
        path = current["path"]
        i_perp, delta_e = _metric_pair(
            path, current["energy"], current["force"]
        )
        rows.append({"optimization_N_F": int(row["optimization_nf"]),
                     "i_perp": i_perp, "delta_e32_eV": delta_e,
                     "disposition": disposition,
                     "current_path_sha256": _initial_path_sha256(path)})
        source_rows.append({key: row[key] for key in (
            "batch_id", "optimization_nf", "role", "disposition",
            "source_artifact_sha256", "ledger_commit_event_sha256",
            "full_state_wal_event_sha256")})
    audit_sha = visit_candidate_evidence(
        run_dir, terminal_audit_path, expected_k=expected_k, consumer=consume
    )
    raw = json.dumps(source_rows, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    if len(rows) != 200 or [row["optimization_N_F"] for row in rows] \
            != BUDGET_GRID.tolist():
        raise ValueError("recomputed compact trace lacks the exact 200-budget grid")
    if rows[0]["disposition"] != "current_initial" \
            or rows[0]["current_path_sha256"] != INITIAL_ARRAY_SHA256:
        raise ValueError("first paid state is not the frozen common initial path")
    document = {
        "schema": TRACE_SCHEMA,
        "status": "COMPLETE",
        "spring_k_eV_per_A2": float(expected_k),
        "terminal_audit": {
            "path": str(Path(terminal_audit_path).resolve()),
            "sha256": audit_sha,
        },
        "screening_terminal_optimization_N_F": 6000,
        "batch_count": 200,
        "endpoint_optimization_N_F": 0,
        "metric_source": METRIC_SOURCE,
        "common_initial_path_sha256": rows[0]["current_path_sha256"],
        "source_closure_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": rows,
    }
    return document


def freeze_compact_trace(run_dir, terminal_audit_path, output, *, expected_k):
    output = Path(output).resolve()
    document = _recompute_compact_trace(
        run_dir, terminal_audit_path, expected_k=expected_k
    )
    write_json_new(output, document, readonly=True)
    return document


def _load_compact_trace_structure(path):
    path = Path(path).resolve(strict=True)
    _readonly_regular(path, "compact trace")
    doc = load_json_object(path, require_canonical=True)
    if doc.get("schema") != TRACE_SCHEMA or doc.get("status") != "COMPLETE":
        raise ValueError("compact trace is not complete")
    expected_keys = {
        "schema", "status", "spring_k_eV_per_A2", "terminal_audit",
        "screening_terminal_optimization_N_F", "batch_count",
        "endpoint_optimization_N_F", "metric_source",
        "common_initial_path_sha256", "source_closure_sha256", "rows",
    }
    if set(doc) != expected_keys or float(doc["spring_k_eV_per_A2"]) not in K_VALUES \
            or doc["screening_terminal_optimization_N_F"] != 6000 \
            or doc["batch_count"] != 200 or doc["endpoint_optimization_N_F"] != 0 \
            or doc["metric_source"] != METRIC_SOURCE:
        raise ValueError("compact trace top-level contract differs")
    rows = doc.get("rows")
    if not isinstance(rows, list) or len(rows) != 200:
        raise ValueError("compact trace does not contain 200 budgets")
    if [r.get("optimization_N_F") for r in rows] != BUDGET_GRID.tolist():
        raise ValueError("compact trace budget axis differs")
    for index, row in enumerate(rows):
        if set(row) != {
            "optimization_N_F", "i_perp", "delta_e32_eV", "disposition",
            "current_path_sha256",
        } or not math.isfinite(float(row["i_perp"])) \
                or not math.isfinite(float(row["delta_e32_eV"])) \
                or float(row["i_perp"]) < 0 or float(row["delta_e32_eV"]) < 0 \
                or not _SHA256.fullmatch(str(row["current_path_sha256"])):
            raise ValueError("compact trace row contract differs")
        expected_dispositions = {"current_initial"} if index == 0 else {"accepted", "rejected"}
        if row["disposition"] not in expected_dispositions:
            raise ValueError("compact trace disposition sequence differs")
        if index and row["disposition"] == "rejected" \
                and row["current_path_sha256"] != rows[index - 1]["current_path_sha256"]:
            raise ValueError("compact trace rejected-current hash did not roll back")
    if doc["common_initial_path_sha256"] != rows[0]["current_path_sha256"] \
            or doc["common_initial_path_sha256"] != INITIAL_ARRAY_SHA256 \
            or not _SHA256.fullmatch(str(doc["source_closure_sha256"])):
        raise ValueError("compact trace initial/source closure differs")
    audit = doc.get("terminal_audit")
    if not isinstance(audit, dict) or set(audit) != {"path", "sha256"} \
            or not isinstance(audit.get("path"), str) \
            or not audit["path"] or not Path(audit["path"]).is_absolute() \
            or not _SHA256.fullmatch(str(audit.get("sha256", ""))):
        raise ValueError("compact trace terminal audit binding is malformed")
    return doc


def load_compact_trace(path):
    """Origin-host verification: reopen raw history and recompute every row."""

    doc = _load_compact_trace_structure(path)
    audit = doc["terminal_audit"]
    if file_sha256(Path(audit["path"]).resolve(strict=True)) != audit["sha256"]:
        raise ValueError("compact trace terminal audit binding failed")
    audit_document = load_json_object(Path(audit["path"]), require_canonical=True)
    run_dir = audit_document.get("run_dir")
    candidate = audit_document.get("candidate", {})
    if (
        audit_document.get("schema")
        != "bb-neb-aseode-kgrid-external-terminal-audit-v1"
        or audit_document.get("status") != "PASS"
        or audit_document.get("selection_eligible") is not True
        or not isinstance(candidate, Mapping)
        or candidate.get("phase") != "screening"
        or float(candidate.get("spring_k_eV_per_A2", -1))
        != float(doc.get("spring_k_eV_per_A2", -2))
        or int(audit_document.get("paid_batches", -1)) != 200
        or int(audit_document.get("paid_optimization_N_F", -1)) != 6000
    ):
        raise ValueError("compact trace terminal audit candidate contract failed")
    expected = _recompute_compact_trace(
        run_dir, audit["path"], expected_k=float(doc["spring_k_eV_per_A2"])
    )
    if doc != expected:
        raise ValueError("compact trace differs from streamed full-history recomputation")
    return doc


def write_compact_trace_audit(trace_path, output):
    """On the origin host, independently rederive and attest one trace."""

    trace_path = Path(trace_path).resolve(strict=True)
    output = Path(output).resolve()
    if is_within(output, Path(__file__).resolve().parents[1]):
        raise ValueError("compact audit receipt must be external to frozen source")
    trace = load_compact_trace(trace_path)
    terminal_path = Path(trace["terminal_audit"]["path"]).resolve(strict=True)
    terminal = load_json_object(terminal_path, require_canonical=True)
    authorization_path = Path(terminal["authorization"]["path"]).resolve(strict=True)
    authorization = load_json_object(authorization_path, require_canonical=True)
    candidate_record = terminal.get("candidate")
    if not isinstance(candidate_record, Mapping):
        raise ValueError("terminal audit candidate binding is absent")
    candidate = build_run(
        spring_k=float(candidate_record.get("spring_k_eV_per_A2", -1)),
        phase=str(candidate_record.get("phase", "")),
        run_label=str(candidate_record.get("run_label", "")),
    )
    if candidate_record != candidate.as_dict() or candidate.phase != "screening" \
            or candidate.spring_k != float(trace["spring_k_eV_per_A2"]) \
            or candidate.batches != 200 or candidate.terminal_nf != 6000:
        raise ValueError("compact audit source is not an exact screening candidate")
    authorization_binding = terminal.get("authorization")
    if not isinstance(authorization_binding, Mapping) \
            or set(authorization_binding) != {"path", "sha256"} \
            or authorization_binding["path"] != str(authorization_path) \
            or file_sha256(authorization_path) != authorization_binding["sha256"]:
        raise ValueError("compact audit authorization binding failed")
    source = verify_source_identity()
    document = {
        "schema": COMPACT_AUDIT_SCHEMA,
        "status": "verified-go",
        "origin_machine_id": authorization["target_machine"]["machine_id"],
        "source_identity": source.as_dict(),
        "candidate": candidate.as_dict(),
        "authorization": dict(authorization_binding),
        "spring_k_eV_per_A2": trace["spring_k_eV_per_A2"],
        "screening_terminal_optimization_N_F": 6000,
        "endpoint_optimization_N_F": 0,
        "common_initial_path_sha256": trace["common_initial_path_sha256"],
        "trace_source_closure_sha256": trace["source_closure_sha256"],
        "trace": {"origin_path": str(trace_path), "sha256": file_sha256(trace_path)},
        "terminal_audit": {
            "origin_path": str(terminal_path),
            "sha256": file_sha256(terminal_path),
        },
        "terminal_tree_closure_sha256": terminal["terminal_tree_closure_sha256"],
        "optimization_wal_sha256": terminal["optimization_wal_sha256"],
        "full_state_wal_sha256": terminal["full_state_wal_sha256"],
        "full_state_completion_sha256": terminal["full_state_completion_sha256"],
        "physical_force_calls_by_compact_audit": 0,
        "OpenMM_contexts_constructed_by_compact_audit": 0,
    }
    write_json_new(output, document, readonly=True)
    return document


def load_portable_compact_trace(trace_binding, compact_audit_binding):
    """Verify an origin-attested trace without reopening origin raw data."""

    if not isinstance(trace_binding, Mapping) \
            or set(trace_binding) != {"path", "sha256"} \
            or not isinstance(compact_audit_binding, Mapping) \
            or set(compact_audit_binding) != {"path", "sha256"}:
        raise ValueError("portable compact trace/audit bindings are not exact")
    source = verify_source_identity()
    trace_path = Path(trace_binding["path"]).resolve(strict=True)
    audit_path = Path(compact_audit_binding["path"]).resolve(strict=True)
    _readonly_regular(trace_path, "portable compact trace")
    _readonly_regular(audit_path, "portable compact audit")
    if file_sha256(trace_path) != trace_binding.get("sha256") \
            or file_sha256(audit_path) != compact_audit_binding.get("sha256"):
        raise ValueError("portable compact trace/audit hash failed")
    trace = _load_compact_trace_structure(trace_path)
    audit = load_json_object(audit_path, require_canonical=True)
    expected_audit_keys = {
        "schema", "status", "origin_machine_id", "source_identity",
        "candidate", "authorization",
        "spring_k_eV_per_A2", "screening_terminal_optimization_N_F",
        "endpoint_optimization_N_F", "common_initial_path_sha256",
        "trace_source_closure_sha256", "trace", "terminal_audit",
        "terminal_tree_closure_sha256", "optimization_wal_sha256",
        "full_state_wal_sha256", "full_state_completion_sha256",
        "physical_force_calls_by_compact_audit",
        "OpenMM_contexts_constructed_by_compact_audit",
    }
    trace_origin = audit.get("trace")
    terminal_origin = audit.get("terminal_audit")
    candidate_record = audit.get("candidate")
    authorization = audit.get("authorization")
    try:
        candidate = build_run(
            spring_k=float(candidate_record.get("spring_k_eV_per_A2", -1)),
            phase=str(candidate_record.get("phase", "")),
            run_label=str(candidate_record.get("run_label", "")),
        ) if isinstance(candidate_record, Mapping) else None
    except (TypeError, ValueError, RuntimeError):
        candidate = None
    if set(audit) != expected_audit_keys \
            or not isinstance(trace_origin, Mapping) \
            or set(trace_origin) != {"origin_path", "sha256"} \
            or not isinstance(terminal_origin, Mapping) \
            or set(terminal_origin) != {"origin_path", "sha256"} \
            or not isinstance(trace_origin.get("origin_path"), str) \
            or not trace_origin["origin_path"] \
            or not Path(trace_origin["origin_path"]).is_absolute() \
            or not isinstance(terminal_origin.get("origin_path"), str) \
            or not terminal_origin["origin_path"] \
            or not Path(terminal_origin["origin_path"]).is_absolute() \
            or terminal_origin["origin_path"] != trace["terminal_audit"]["path"] \
            or candidate is None or candidate_record != candidate.as_dict() \
            or candidate.phase != "screening" or candidate.batches != 200 \
            or candidate.terminal_nf != 6000 \
            or candidate.spring_k != float(trace["spring_k_eV_per_A2"]) \
            or not isinstance(authorization, Mapping) \
            or set(authorization) != {"path", "sha256"} \
            or not isinstance(authorization.get("path"), str) \
            or not authorization["path"] or not Path(authorization["path"]).is_absolute() \
            or not _SHA256.fullmatch(str(authorization.get("sha256", ""))) \
            or not _SHA256.fullmatch(str(trace_origin.get("sha256", ""))) \
            or not _SHA256.fullmatch(str(terminal_origin.get("sha256", ""))) \
            or audit.get("schema") != COMPACT_AUDIT_SCHEMA \
            or audit.get("status") != "verified-go" \
            or audit.get("source_identity") != source.as_dict() \
            or audit.get("spring_k_eV_per_A2") != trace["spring_k_eV_per_A2"] \
            or audit.get("screening_terminal_optimization_N_F") != 6000 \
            or audit.get("endpoint_optimization_N_F") != 0 \
            or audit.get("common_initial_path_sha256") \
            != trace["common_initial_path_sha256"] \
            or audit.get("trace_source_closure_sha256") != trace["source_closure_sha256"] \
            or trace_origin.get("sha256") != trace_binding["sha256"] \
            or terminal_origin.get("sha256") \
            != trace["terminal_audit"]["sha256"] \
            or audit.get("physical_force_calls_by_compact_audit") != 0 \
            or audit.get("OpenMM_contexts_constructed_by_compact_audit") != 0:
        raise ValueError("portable compact audit/source/provenance contract failed")
    for key in (
        "origin_machine_id", "terminal_tree_closure_sha256",
        "optimization_wal_sha256", "full_state_wal_sha256",
        "full_state_completion_sha256",
    ):
        value = str(audit.get(key, ""))
        if key != "origin_machine_id" and not _SHA256.fullmatch(value):
            raise ValueError(f"portable compact audit {key} is malformed")
        if key == "origin_machine_id" and not value:
            raise ValueError("portable compact audit origin machine is empty")
    return trace, audit


def select_from_compact_traces(bindings, *, selection_protocol_path):
    protocol = Path(selection_protocol_path).resolve(strict=True)
    traces = {}
    verified_candidates = []
    for binding in bindings:
        if set(binding) != {"path", "sha256", "compact_audit"}:
            raise ValueError("selection compact binding fields are not exact")
        path = Path(binding["path"]).resolve(strict=True)
        trace_binding = {"path": str(path), "sha256": binding["sha256"]}
        trace, compact_audit = load_portable_compact_trace(
            trace_binding, binding["compact_audit"]
        )
        k = float(trace["spring_k_eV_per_A2"])
        if k in traces:
            raise ValueError("duplicate compact trace k")
        traces[k] = trace
        receipt_record = {
            "spring_k_eV_per_A2": k,
            "candidate": compact_audit["candidate"],
            "authorization": compact_audit["authorization"],
            "terminal_audit": trace["terminal_audit"],
            "trace": trace_binding,
            "compact_audit": {
                "path": str(Path(binding["compact_audit"]["path"]).resolve()),
                "sha256": binding["compact_audit"]["sha256"],
            },
            "origin_machine_id": compact_audit["origin_machine_id"],
            "terminal_tree_closure_sha256": compact_audit[
                "terminal_tree_closure_sha256"
            ],
        }
        verified_candidates.append({
            "spring_k_eV_per_A2": k,
            "optimization_N_F": [row["optimization_N_F"] for row in trace["rows"]],
            "i_perp": [row["i_perp"] for row in trace["rows"]],
            "delta_e32_eV": [row["delta_e32_eV"] for row in trace["rows"]],
            "dispositions": [row["disposition"] for row in trace["rows"]],
            "current_path_sha256": [
                row["current_path_sha256"] for row in trace["rows"]
            ],
            "common_initial_path_sha256": trace["common_initial_path_sha256"],
            "source_closure_sha256": trace["source_closure_sha256"],
            "terminal_audit_sha256": trace["terminal_audit"]["sha256"],
            "receipt_record": receipt_record,
        })
    if set(traces) != set(K_VALUES):
        raise ValueError("selection requires exact four-k compact traces")
    initial_hashes = {trace["common_initial_path_sha256"] for trace in traces.values()}
    if len(initial_hashes) != 1:
        raise ValueError("four compact traces do not share the exact initial path")
    return select_verified_compact_metric_traces(
        verified_candidates,
        selection_protocol={"path": str(protocol), "sha256": file_sha256(protocol)},
    )


def freeze_selection_receipt(bindings, *, selection_protocol_path, output):
    document = select_from_compact_traces(
        bindings, selection_protocol_path=selection_protocol_path
    )
    write_json_new(Path(output).resolve(), document, readonly=True)
    return document
