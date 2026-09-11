"""Build selector evidence only from a sealed screening run.

This module performs no energy/Force evaluation.  It verifies the terminal
audit receipt and both append-only histories, then decodes every paid batch
from the immutable full-state shards.  Metric values are deliberately not an
input to this boundary.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping

from .accounting import audit_wal, file_sha256
from .full_batch_recorder import (
    COMPLETE_NAME,
    WAL_NAME,
    audit_full_state_archive,
    reconstruct_batch,
)
from .integrity import load_json_object
from .kgrid_selection import SCREENING_TERMINAL_NF
from .kgrid_spec import SCREENING_MILESTONES


AUDIT_SCHEMA = "bb-neb-aseode-kgrid-external-terminal-audit-v1"
TERMINAL_BATCHES = 200
NF_PER_BATCH = 30
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class MetricEvidenceError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MetricEvidenceError(message)


def _canonical_file(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    _require(resolved.is_file() and not resolved.is_symlink(), f"{label} is unsafe")
    return resolved


def _binding(receipt: Mapping[str, Any], name: str, expected: Path) -> None:
    digest = str(receipt.get(name, ""))
    _require(bool(_SHA256.fullmatch(digest)), f"terminal audit {name} hash is malformed")
    _require(file_sha256(expected) == digest,
             f"terminal audit {name} hash recomputation failed")


def _wal_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        _require(bool(line), f"empty optimization WAL line {number}")
        value = json.loads(line)
        _require(isinstance(value, dict), f"optimization WAL line {number} is not an object")
        records.append(value)
    return records


def _reaudit_terminal(run: Path, authorization_path: Path) -> dict[str, Any]:
    # Lazy import keeps this evidence module free of any calculator import at
    # module load.  audit_terminal itself is contractually zero-Force.
    from .kgrid_terminal_audit import audit_terminal
    return audit_terminal(run_dir=run, authorization_path=authorization_path)


def load_candidate_evidence(
    run_dir: Path | str,
    terminal_audit_path: Path | str,
    *,
    expected_k: float,
    _batch_consumer=None,
    _retain_batches: bool = True,
) -> dict[str, object]:
    """Return real, array-bearing input for ``select_best_k``.

    The terminal audit is a gate, not a source of metrics.  All array values
    come from bit-exact shard reconstruction after independent WAL/archive
    validation.
    """

    run = Path(run_dir).resolve(strict=True)
    _require(run.is_dir() and not run.is_symlink(), "screening run directory is unsafe")
    audit_path = _canonical_file(Path(terminal_audit_path), "terminal audit")
    receipt = load_json_object(audit_path, require_canonical=True)
    _require(receipt.get("schema") == AUDIT_SCHEMA, "terminal audit schema differs")
    _require(receipt.get("status") == "PASS", "terminal audit status is not PASS")
    _require(Path(str(receipt.get("run_dir", ""))).resolve() == run,
             "terminal audit is for another run")
    _require(receipt.get("terminal_eligible") is True
             and receipt.get("selection_eligible") is True,
             "terminal audit is not selection eligible")
    candidate = receipt.get("candidate")
    _require(isinstance(candidate, Mapping), "terminal audit lacks candidate")
    _require(float(candidate.get("spring_k_eV_per_A2")) == float(expected_k)
             and candidate.get("phase") == "screening",
             "terminal audit candidate differs")
    _require(int(receipt.get("paid_batches", -1)) == TERMINAL_BATCHES
             and int(receipt.get("paid_optimization_N_F", -1)) == SCREENING_TERMINAL_NF,
             "terminal audit is not the complete NF6000 screening run")
    authorization = receipt.get("authorization")
    _require(isinstance(authorization, Mapping)
             and set(authorization) == {"path", "sha256"},
             "terminal audit authorization binding is not exact")
    authorization_path = _canonical_file(
        Path(str(authorization["path"])), "execution authorization"
    )
    _require(str(authorization["sha256"]) == file_sha256(authorization_path),
             "execution authorization hash recomputation failed")
    _require(_reaudit_terminal(run, authorization_path) == receipt,
             "canonical terminal audit recomputation differs from receipt")

    optimization_wal = _canonical_file(run / "optimization_force_wal.jsonl", "optimization WAL")
    archive_root = run / "paid_batch_full_state"
    full_wal = _canonical_file(archive_root / WAL_NAME, "full-state WAL")
    full_complete_path = _canonical_file(
        archive_root / COMPLETE_NAME, "full-state completion"
    )
    _binding(receipt, "optimization_wal_sha256", optimization_wal)
    _binding(receipt, "full_state_wal_sha256", full_wal)
    _binding(receipt, "full_state_completion_sha256", full_complete_path)

    records = _wal_records(optimization_wal)
    commits = [record for record in records if record.get("event") == "batch_commit"]
    requests: Dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("event") == "force_request":
            requests.setdefault(int(record["batch_id"]), []).append(record)
    audit_wal(
        optimization_wal,
        require_terminal=True,
        expected_batches=TERMINAL_BATCHES,
        expected_milestones=SCREENING_MILESTONES,
    )

    completion = load_json_object(full_complete_path, require_canonical=True)
    _require(float(completion.get("spring_k_eV_per_A2")) == float(expected_k),
             "full-state completion spring coefficient differs")
    archive_audit = audit_full_state_archive(
        archive_root,
        spring_k=float(expected_k),
        terminal_batches=TERMINAL_BATCHES,
        run_label=str(completion["run_label"]),
        endpoint_cache=str(completion["endpoint_cache_path"]),
        authorization_sha256=str(completion["authorization_sha256"]),
        source_closure_sha256=str(completion["source_closure_sha256"]),
        optimizer_commits=commits,
    )
    _require(archive_audit.get("pass") is True, "full-state archive audit did not pass")
    _require(len(commits) == TERMINAL_BATCHES and set(requests) == set(range(1, 201)),
             "attempt/commit mapping is incomplete")

    full_records = _wal_records(full_wal)
    published = {
        int(record["batch_id"]): record
        for record in full_records
        if record.get("event") == "batch_published"
    }
    _require(set(published) == set(range(1, 201)), "full-state WAL mapping is incomplete")
    batches = []
    for batch_id in range(1, 201):
        commit = commits[batch_id - 1]
        publication = published[batch_id]
        attempts = sorted(int(row["paid_nf"]) for row in requests[batch_id])
        expected_attempts = list(range(30 * batch_id - 29, 30 * batch_id + 1))
        _require(attempts == expected_attempts, f"batch {batch_id} attempt axis differs")
        shard = _canonical_file(
            archive_root / f"batch_{batch_id:04d}_nf{30 * batch_id:05d}.npz",
            f"batch {batch_id} shard",
        )
        _require(publication.get("file_sha256") == file_sha256(shard),
                 f"batch {batch_id} shard hash recomputation failed")
        paid_path, energy32, full_force32, current_path = reconstruct_batch(
            archive_root, batch_id
        )
        row = {
            "batch_id": batch_id,
            "optimization_nf": 30 * batch_id,
            "spring_k_eV_per_A2": float(expected_k),
            "role": str(publication["role"]),
            "disposition": str(publication["disposition"]),
            "paid_path32_A": paid_path.reshape((32, -1)),
            "energy32_eV": energy32,
            "full_force32_eV_per_A": full_force32.reshape((32, -1)),
            "current_path32_A": current_path.reshape((32, -1)),
            "paid_force_attempt_indices": attempts,
            "endpoint_optimization_nf": 0,
            "source_artifact_sha256": str(publication["file_sha256"]),
            "ledger_commit_event_sha256": str(commit["event_sha256"]),
            "full_state_wal_event_sha256": str(publication["record_sha256"]),
        }
        if _batch_consumer is not None:
            _batch_consumer(row)
        if _retain_batches:
            batches.append(row)
    return {"terminal_audit_sha256": file_sha256(audit_path), "batches": batches}


def visit_candidate_evidence(run_dir, terminal_audit_path, *, expected_k, consumer):
    """Validate one run and deliver one reconstructed batch at a time."""
    _require(callable(consumer), "batch consumer is not callable")
    result = load_candidate_evidence(
        run_dir, terminal_audit_path, expected_k=expected_k,
        _batch_consumer=consumer, _retain_batches=False,
    )
    _require(result["batches"] == [], "streaming evidence retained batch arrays")
    return str(result["terminal_audit_sha256"])
