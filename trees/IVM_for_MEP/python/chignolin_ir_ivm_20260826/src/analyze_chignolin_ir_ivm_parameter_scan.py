#!/usr/bin/env python3
"""Summarize audited Chignolin IR-IVM parameter-scan results.

Inputs are CSV or JSON documents written by
``audit_chignolin_ir_ivm_results.py``.  This program performs no molecular
simulation and does not evaluate energies, forces, or dense-path barriers.

Three outputs are written:

* ``parameter_settings.csv``: one row per (alpha, r, m_max, budget) setting;
* ``parameter_effects_long.csv``: three tidy rows per setting, one for each
  parameter axis, ready for parameter-effect plots; and
* ``parameter_scan_analysis.json``: the same records plus an article-language
  summary and complete input provenance.

An audited setting that has not reached the I_perp threshold is represented as
right-censored at its last complete path state.  That censoring Number of
Forces is never reported as a first-passage cost.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


IPERP_THRESHOLD = 0.21242215133173672
TOTAL_NF_OFFSET = 2
EXPECTED_REPARAM_MODE = "state_curve"
EXPECTED_SELECTOR_MODE = "one_way_takeover"
EXPECTED_ETA = 1.1
EXPECTED_USABLE_RULE = "record_index_0_or_reparam_committed"

STATUS_REACHED = "reached"
STATUS_NOT_REACHED = "not_reached"
STATUS_AUDIT_FAILED = "audit_failed"

STATUS_ZH = {
    STATUS_REACHED: "达到 I_perp 门槛",
    STATUS_NOT_REACHED: "截至最后完整路径状态仍未达到 I_perp 门槛",
    STATUS_AUDIT_FAILED: "结果审计失败",
}

SETTINGS_FILENAME = "parameter_settings.csv"
EFFECTS_FILENAME = "parameter_effects_long.csv"
JSON_FILENAME = "parameter_scan_analysis.json"


class AnalysisError(RuntimeError):
    """Raised when an audited summary is malformed or internally inconsistent."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _text(row: Mapping[str, Any], key: str, allow_missing: bool = False) -> Optional[str]:
    value = row.get(key)
    if _missing(value):
        if allow_missing:
            return None
        raise AnalysisError("missing text field: " + key)
    return str(value)


def _float(
    row: Mapping[str, Any], key: str, allow_missing: bool = False
) -> Optional[float]:
    value = row.get(key)
    if _missing(value):
        if allow_missing:
            return None
        raise AnalysisError("missing numeric field: " + key)
    if isinstance(value, bool):
        raise AnalysisError("boolean supplied for numeric field: " + key)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise AnalysisError("invalid numeric field: " + key)
    if not math.isfinite(result):
        raise AnalysisError("non-finite numeric field: " + key)
    return result


def _int(row: Mapping[str, Any], key: str, allow_missing: bool = False) -> Optional[int]:
    value = row.get(key)
    if _missing(value):
        if allow_missing:
            return None
        raise AnalysisError("missing integer field: " + key)
    if isinstance(value, bool):
        raise AnalysisError("boolean supplied for integer field: " + key)
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError):
        raise AnalysisError("invalid integer field: " + key)
    if not math.isfinite(numeric) or numeric != result:
        raise AnalysisError("non-integral field: " + key)
    return result


def _bool(
    row: Mapping[str, Any], key: str, allow_missing: bool = False
) -> Optional[bool]:
    value = row.get(key)
    if _missing(value):
        if allow_missing:
            return None
        raise AnalysisError("missing Boolean field: " + key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    raise AnalysisError("invalid Boolean field: " + key)


def _require_equal(observed: Any, expected: Any, key: str) -> None:
    if observed != expected:
        raise AnalysisError(
            "audited metadata drift at {}: observed {!r}, expected {!r}".format(
                key, observed, expected
            )
        )


def _setting_id(alpha: float, ratio: float, cap: int, budget: int) -> str:
    return "alpha={:g}|r={:g}|m_max={}|internal_nf_budget={}".format(
        alpha, ratio, cap, budget
    )


def _normalize_failed(
    row: Mapping[str, Any], source: Path, source_threshold: Optional[float]
) -> Dict[str, Any]:
    label = _text(row, "run_label")
    alpha = _float(row, "alpha")
    ratio = _float(row, "residual_ratio_r")
    cap = _int(row, "inner_max_steps_m_max")
    budget = _int(row, "internal_nf_budget")
    threshold = _float(row, "i_perp_threshold", allow_missing=True)
    if threshold is None:
        threshold = source_threshold
    if threshold is None:
        threshold = IPERP_THRESHOLD
    _require_equal(threshold, IPERP_THRESHOLD, "i_perp_threshold")
    if alpha is None or alpha <= 0.0:
        raise AnalysisError("alpha must be positive")
    if ratio is None or not (0.0 < ratio < 1.0):
        raise AnalysisError("r must lie strictly between zero and one")
    if cap is None or cap <= 0 or budget is None or budget <= 0:
        raise AnalysisError("m_max and the internal NF budget must be positive")
    reason = _text(row, "audit_failure_reason", allow_missing=True)
    return {
        "parameter_setting_id": _setting_id(alpha, ratio, cap, budget),
        "run_label": label,
        "alpha": alpha,
        "residual_ratio_r": ratio,
        "inner_max_steps_m_max": cap,
        "internal_nf_budget": budget,
        "i_perp_threshold": threshold,
        "analysis_status": STATUS_AUDIT_FAILED,
        "analysis_status_zh": STATUS_ZH[STATUS_AUDIT_FAILED],
        "audit_failure_reason": reason,
        "first_threshold_reached": False,
        "first_threshold_state_index": None,
        "first_threshold_state_kind": None,
        "first_threshold_total_nf": None,
        "first_threshold_i_perp": None,
        "last_complete_reparam_state_available": False,
        "last_complete_reparam_state_index": None,
        "last_complete_reparam_total_nf": None,
        "last_complete_reparam_i_perp": None,
        "last_usable_state_kind": None,
        "last_usable_state_index": None,
        "last_usable_total_nf": None,
        "last_usable_i_perp": None,
        "right_censoring_total_nf": None,
        "terminal_total_nf": None,
        "reparam_attempt_count": None,
        "reparam_accept_count": None,
        "recovery_event_count": None,
        "recovery_joint_evaluations": None,
        "result_npz": None,
        "result_npz_sha256": None,
        "configuration_sha256": None,
        "source_audit_files": [str(source)],
    }


def _normalize_passed(
    row: Mapping[str, Any], source: Path, source_threshold: Optional[float]
) -> Dict[str, Any]:
    label = _text(row, "run_label")
    alpha = _float(row, "alpha")
    ratio = _float(row, "residual_ratio_r")
    cap = _int(row, "inner_max_steps_m_max")
    budget = _int(row, "internal_nf_budget")
    eta = _float(row, "eta")
    threshold = _float(row, "i_perp_threshold")
    if source_threshold is not None:
        _require_equal(threshold, source_threshold, "source/row i_perp_threshold")
    _require_equal(threshold, IPERP_THRESHOLD, "i_perp_threshold")
    _require_equal(_text(row, "reparam_mode"), EXPECTED_REPARAM_MODE, "reparam_mode")
    _require_equal(_text(row, "selector_mode"), EXPECTED_SELECTOR_MODE, "selector_mode")
    _require_equal(eta, EXPECTED_ETA, "eta")
    _require_equal(
        _text(row, "usable_state_rule"), EXPECTED_USABLE_RULE, "usable_state_rule"
    )
    if alpha is None or alpha <= 0.0:
        raise AnalysisError("alpha must be positive")
    if ratio is None or not (0.0 < ratio < 1.0):
        raise AnalysisError("r must lie strictly between zero and one")
    if cap is None or cap <= 0 or budget is None or budget <= 0:
        raise AnalysisError("m_max and the internal NF budget must be positive")

    terminal_internal_nf = _int(row, "terminal_internal_nf")
    terminal_total_nf = _int(row, "terminal_total_nf")
    last_index = _int(row, "last_complete_state_index")
    last_kind = _text(row, "last_complete_state_kind")
    last_internal_nf = _int(row, "last_complete_internal_nf")
    last_total_nf = _int(row, "last_complete_total_nf")
    last_iperp = _float(row, "last_complete_i_perp")
    usable_count = _int(row, "usable_state_count")
    reached = _bool(row, "first_threshold_reached")

    if terminal_internal_nf is None or terminal_total_nf is None:
        raise AnalysisError("terminal NF fields are absent")
    if terminal_total_nf != terminal_internal_nf + TOTAL_NF_OFFSET:
        raise AnalysisError("terminal total NF is not internal NF plus two")
    if terminal_internal_nf > budget:
        raise AnalysisError("audited setting exceeds its internal NF budget")
    if last_internal_nf is None or last_total_nf is None:
        raise AnalysisError("last complete NF fields are absent")
    if last_total_nf != last_internal_nf + TOTAL_NF_OFFSET:
        raise AnalysisError("last complete total NF is not internal NF plus two")
    if last_total_nf > terminal_total_nf:
        raise AnalysisError("last complete state lies after the terminal ledger")
    if last_index is None or last_index < 0 or usable_count is None or usable_count < 1:
        raise AnalysisError("last complete state metadata is invalid")
    if last_kind not in ("initial", "reparam_committed"):
        raise AnalysisError("last complete state kind is unknown")
    if last_iperp is None or last_iperp < 0.0:
        raise AnalysisError("last complete I_perp is invalid")

    partial_tail_present = _bool(row, "terminal_partial_tail_present")
    partial_tail_quarantined = _bool(row, "terminal_partial_tail_quarantined")
    partial_tail_nf = _int(row, "terminal_partial_tail_internal_nf")
    expected_tail = terminal_internal_nf - last_internal_nf
    if partial_tail_nf != expected_tail:
        raise AnalysisError("terminal partial-tail NF is inconsistent")
    if partial_tail_present != (expected_tail > 0):
        raise AnalysisError("terminal partial-tail presence is inconsistent")
    if partial_tail_present and not partial_tail_quarantined:
        raise AnalysisError("terminal partial tail was not quarantined")

    first_index = _int(row, "first_threshold_state_index", allow_missing=True)
    first_kind = _text(row, "first_threshold_state_kind", allow_missing=True)
    first_internal_nf = _int(row, "first_threshold_internal_nf", allow_missing=True)
    first_total_nf = _int(row, "first_threshold_total_nf", allow_missing=True)
    first_iperp = _float(row, "first_threshold_i_perp", allow_missing=True)
    first_values = (first_index, first_kind, first_internal_nf, first_total_nf, first_iperp)
    if reached:
        if any(value is None for value in first_values):
            raise AnalysisError("reached setting lacks first-passage metadata")
        if first_kind not in ("initial", "reparam_committed"):
            raise AnalysisError("first-passage state kind is unknown")
        if first_total_nf != first_internal_nf + TOTAL_NF_OFFSET:
            raise AnalysisError("first-passage total NF is not internal NF plus two")
        if first_total_nf > last_total_nf or first_index > last_index:
            raise AnalysisError("first-passage state lies after the last complete state")
        if first_iperp > threshold:
            raise AnalysisError("reported first-passage I_perp exceeds its threshold")
        status = STATUS_REACHED
        censoring_nf: Optional[int] = None
    else:
        if any(value is not None for value in first_values):
            raise AnalysisError("unreached setting contains first-passage metadata")
        if last_iperp <= threshold:
            raise AnalysisError("unreached setting already satisfies the I_perp threshold")
        status = STATUS_NOT_REACHED
        censoring_nf = last_total_nf

    reparam_attempts = _int(row, "reparam_attempt_count")
    reparam_accepts = _int(row, "reparam_accept_count")
    recovery_events = _int(row, "recovery_event_count")
    recovery_nf = _int(row, "recovery_joint_evaluations")
    if (
        reparam_attempts is None
        or reparam_accepts is None
        or reparam_attempts < reparam_accepts
        or reparam_accepts < 0
        or recovery_events is None
        or recovery_events < 0
        or recovery_nf is None
        or recovery_nf < 0
    ):
        raise AnalysisError("reparameterization/recovery counts are invalid")
    if last_kind == "reparam_committed" and reparam_accepts < 1:
        raise AnalysisError("last committed reparameterization lacks an accept count")

    result_sha = _text(row, "result_npz_sha256")
    configuration_sha = _text(row, "configuration_sha256")
    if len(result_sha) != 64 or len(configuration_sha) != 64:
        raise AnalysisError("result or configuration SHA-256 is malformed")

    return {
        "parameter_setting_id": _setting_id(alpha, ratio, cap, budget),
        "run_label": label,
        "alpha": alpha,
        "residual_ratio_r": ratio,
        "inner_max_steps_m_max": cap,
        "internal_nf_budget": budget,
        "eta": eta,
        "i_perp_threshold": threshold,
        "analysis_status": status,
        "analysis_status_zh": STATUS_ZH[status],
        "audit_failure_reason": None,
        "first_threshold_reached": reached,
        "first_threshold_state_index": first_index,
        "first_threshold_state_kind": first_kind,
        "first_threshold_total_nf": first_total_nf,
        "first_threshold_i_perp": first_iperp,
        "last_complete_reparam_state_available": last_kind == "reparam_committed",
        "last_complete_reparam_state_index": (
            last_index if last_kind == "reparam_committed" else None
        ),
        "last_complete_reparam_total_nf": (
            last_total_nf if last_kind == "reparam_committed" else None
        ),
        "last_complete_reparam_i_perp": (
            last_iperp if last_kind == "reparam_committed" else None
        ),
        "last_usable_state_kind": last_kind,
        "last_usable_state_index": last_index,
        "last_usable_total_nf": last_total_nf,
        "last_usable_i_perp": last_iperp,
        "right_censoring_total_nf": censoring_nf,
        "terminal_total_nf": terminal_total_nf,
        "terminal_partial_tail_internal_nf": partial_tail_nf,
        "terminal_partial_tail_quarantined": partial_tail_quarantined,
        "reparam_attempt_count": reparam_attempts,
        "reparam_accept_count": reparam_accepts,
        "recovery_event_count": recovery_events,
        "recovery_joint_evaluations": recovery_nf,
        "result_npz": _text(row, "result_npz"),
        "result_npz_sha256": result_sha,
        "configuration_sha256": configuration_sha,
        "source_audit_files": [str(source)],
    }


def _normalize_row(
    row: Mapping[str, Any], source: Path, source_threshold: Optional[float]
) -> Dict[str, Any]:
    audit_status = _text(row, "audit_status")
    if audit_status == "passed":
        return _normalize_passed(row, source, source_threshold)
    if audit_status == "failed":
        return _normalize_failed(row, source, source_threshold)
    raise AnalysisError("unknown audit status: {!r}".format(audit_status))


def _load_source(path: Path) -> Tuple[List[Mapping[str, Any]], Optional[float]]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise AnalysisError("audit summary does not exist: {}".format(path))
    if path.suffix.lower() == ".json":
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AnalysisError("cannot read audit JSON {}: {}".format(path, exc))
        if not isinstance(document, dict) or not isinstance(document.get("results"), list):
            raise AnalysisError("audit JSON lacks a results array: {}".format(path))
        if document.get("schema_version") != "chignolin-ir-ivm-result-audit-v1":
            raise AnalysisError("unsupported audit JSON schema: {}".format(path))
        declared = document.get("result_count")
        if declared != len(document["results"]):
            raise AnalysisError("audit JSON result count is inconsistent: {}".format(path))
        threshold_value = document.get("i_perp_threshold")
        if isinstance(threshold_value, bool):
            raise AnalysisError("audit JSON threshold is invalid")
        try:
            threshold = float(threshold_value)
        except (TypeError, ValueError):
            raise AnalysisError("audit JSON threshold is invalid")
        if not math.isfinite(threshold):
            raise AnalysisError("audit JSON threshold is non-finite")
        return document["results"], threshold
    if path.suffix.lower() == ".csv":
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise AnalysisError("audit CSV has no header: {}".format(path))
                rows = list(reader)
        except OSError as exc:
            raise AnalysisError("cannot read audit CSV {}: {}".format(path, exc))
        if not rows:
            raise AnalysisError("audit CSV has no result rows: {}".format(path))
        return rows, None
    raise AnalysisError("audit input must be CSV or JSON: {}".format(path))


def _same_record(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    ignored = {"source_audit_files"}
    left_core = {key: value for key, value in left.items() if key not in ignored}
    right_core = {key: value for key, value in right.items() if key not in ignored}
    return left_core == right_core


def load_parameter_settings(paths: Sequence[Path]) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    """Load, validate, and deduplicate audited parameter settings."""

    if not paths:
        raise AnalysisError("at least one audit CSV or JSON is required")
    by_label: Dict[str, Dict[str, Any]] = {}
    source_records: List[Dict[str, str]] = []
    seen_sources = set()
    for raw_path in paths:
        source = Path(raw_path).expanduser().resolve()
        if source in seen_sources:
            continue
        seen_sources.add(source)
        rows, source_threshold = _load_source(source)
        source_records.append(
            {
                "path": str(source),
                "sha256": _sha256_file(source),
                "format": source.suffix.lower().lstrip("."),
            }
        )
        for row in rows:
            if not isinstance(row, Mapping):
                raise AnalysisError("audit result row is not an object: {}".format(source))
            normalized = _normalize_row(row, source, source_threshold)
            label = str(normalized["run_label"])
            previous = by_label.get(label)
            if previous is None:
                by_label[label] = normalized
                continue
            if not _same_record(previous, normalized):
                raise AnalysisError("duplicate run label has conflicting audited results: " + label)
            sources = list(previous["source_audit_files"])
            for item in normalized["source_audit_files"]:
                if item not in sources:
                    sources.append(item)
            previous["source_audit_files"] = sources

    settings = sorted(
        by_label.values(),
        key=lambda row: (
            float(row["alpha"]),
            float(row["residual_ratio_r"]),
            int(row["inner_max_steps_m_max"]),
            int(row["internal_nf_budget"]),
            str(row["run_label"]),
        ),
    )
    by_controls: Dict[Tuple[float, float, int, int], str] = {}
    for row in settings:
        controls = (
            float(row["alpha"]),
            float(row["residual_ratio_r"]),
            int(row["inner_max_steps_m_max"]),
            int(row["internal_nf_budget"]),
        )
        label = str(row["run_label"])
        if controls in by_controls and by_controls[controls] != label:
            raise AnalysisError(
                "more than one audited run represents the same parameter setting: {}".format(
                    row["parameter_setting_id"]
                )
            )
        by_controls[controls] = label
    return settings, source_records


def parameter_effect_records(settings: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Expand each setting into alpha, r, and m_max plotting records."""

    result: List[Dict[str, Any]] = []
    definitions = (
        ("alpha", "alpha", "log10"),
        ("residual_ratio_r", "r", "linear"),
        ("inner_max_steps_m_max", "m_max", "discrete"),
    )
    for setting in settings:
        status = str(setting["analysis_status"])
        if status == STATUS_REACHED:
            observation = "first_passage"
            plotted_nf = setting["first_threshold_total_nf"]
        elif status == STATUS_NOT_REACHED:
            observation = "right_censored"
            plotted_nf = setting["right_censoring_total_nf"]
        else:
            observation = "audit_failed"
            plotted_nf = None
        for key, symbol, scale in definitions:
            value = setting[key]
            result.append(
                {
                    "parameter_setting_id": setting["parameter_setting_id"],
                    "run_label": setting["run_label"],
                    "parameter_name": key,
                    "parameter_symbol": symbol,
                    "parameter_value": value,
                    "parameter_scale": scale,
                    "parameter_value_log10": (
                        math.log10(float(value)) if key == "alpha" else None
                    ),
                    "alpha": setting["alpha"],
                    "residual_ratio_r": setting["residual_ratio_r"],
                    "inner_max_steps_m_max": setting["inner_max_steps_m_max"],
                    "internal_nf_budget": setting["internal_nf_budget"],
                    "analysis_status": status,
                    "analysis_status_zh": setting["analysis_status_zh"],
                    "observation_type": observation,
                    "threshold_reached": setting["first_threshold_reached"],
                    "number_of_forces_for_plot": plotted_nf,
                    "first_threshold_total_nf": setting["first_threshold_total_nf"],
                    "right_censoring_total_nf": setting["right_censoring_total_nf"],
                    "last_complete_reparam_total_nf": setting[
                        "last_complete_reparam_total_nf"
                    ],
                    "last_complete_reparam_i_perp": setting[
                        "last_complete_reparam_i_perp"
                    ],
                    "i_perp_threshold": setting["i_perp_threshold"],
                }
            )
    return result


def _article_summary(settings: Sequence[Mapping[str, Any]]) -> Tuple[List[str], Optional[Dict[str, Any]]]:
    reached = [row for row in settings if row["analysis_status"] == STATUS_REACHED]
    not_reached = [
        row for row in settings if row["analysis_status"] == STATUS_NOT_REACHED
    ]
    failed = [row for row in settings if row["analysis_status"] == STATUS_AUDIT_FAILED]
    sentences = [
        (
            "在 I_perp <= {:.17g} 的判据下，共汇总 {} 组参数设置：{} 组达到门槛，"
            "{} 组截至最后一个完整路径状态仍未达到，{} 组结果审计失败。"
        ).format(IPERP_THRESHOLD, len(settings), len(reached), len(not_reached), len(failed)),
        "首次达到和末状态均只采用初始路径或已提交的完整重参数化路径状态。",
    ]
    if not reached:
        sentences.append("目前尚无通过审计的参数设置达到该 I_perp 门槛。")
        return sentences, None
    best = min(
        reached,
        key=lambda row: (
            int(row["first_threshold_total_nf"]),
            float(row["first_threshold_i_perp"]),
            float(row["alpha"]),
            float(row["residual_ratio_r"]),
            int(row["inner_max_steps_m_max"]),
        ),
    )
    sentences.append(
        (
            "达到门槛的设置中，alpha={:g}, r={:g}, m_max={} 首次达到时所需的 "
            "Number of Forces 最少，为 {}。"
        ).format(
            float(best["alpha"]),
            float(best["residual_ratio_r"]),
            int(best["inner_max_steps_m_max"]),
            int(best["first_threshold_total_nf"]),
        )
    )
    best_summary = {
        "parameter_setting_id": best["parameter_setting_id"],
        "run_label": best["run_label"],
        "alpha": best["alpha"],
        "residual_ratio_r": best["residual_ratio_r"],
        "inner_max_steps_m_max": best["inner_max_steps_m_max"],
        "internal_nf_budget": best["internal_nf_budget"],
        "first_threshold_total_nf": best["first_threshold_total_nf"],
        "first_threshold_i_perp": best["first_threshold_i_perp"],
        "last_complete_reparam_total_nf": best["last_complete_reparam_total_nf"],
        "last_complete_reparam_i_perp": best["last_complete_reparam_i_perp"],
    }
    return sentences, best_summary


SETTINGS_FIELDS = (
    "parameter_setting_id",
    "run_label",
    "alpha",
    "residual_ratio_r",
    "inner_max_steps_m_max",
    "internal_nf_budget",
    "eta",
    "i_perp_threshold",
    "analysis_status",
    "analysis_status_zh",
    "audit_failure_reason",
    "first_threshold_reached",
    "first_threshold_state_index",
    "first_threshold_state_kind",
    "first_threshold_total_nf",
    "first_threshold_i_perp",
    "last_complete_reparam_state_available",
    "last_complete_reparam_state_index",
    "last_complete_reparam_total_nf",
    "last_complete_reparam_i_perp",
    "last_usable_state_kind",
    "last_usable_state_index",
    "last_usable_total_nf",
    "last_usable_i_perp",
    "right_censoring_total_nf",
    "terminal_total_nf",
    "terminal_partial_tail_internal_nf",
    "terminal_partial_tail_quarantined",
    "reparam_attempt_count",
    "reparam_accept_count",
    "recovery_event_count",
    "recovery_joint_evaluations",
    "result_npz",
    "result_npz_sha256",
    "configuration_sha256",
    "source_audit_files",
)

EFFECT_FIELDS = (
    "parameter_setting_id",
    "run_label",
    "parameter_name",
    "parameter_symbol",
    "parameter_value",
    "parameter_scale",
    "parameter_value_log10",
    "alpha",
    "residual_ratio_r",
    "inner_max_steps_m_max",
    "internal_nf_budget",
    "analysis_status",
    "analysis_status_zh",
    "observation_type",
    "threshold_reached",
    "number_of_forces_for_plot",
    "first_threshold_total_nf",
    "right_censoring_total_nf",
    "last_complete_reparam_total_nf",
    "last_complete_reparam_i_perp",
    "i_perp_threshold",
)


def _csv_text(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        output: Dict[str, Any] = {}
        for key in fields:
            value = row.get(key)
            if key == "source_audit_files" and isinstance(value, list):
                value = ";".join(str(item) for item in value)
            output[key] = "" if value is None else value
        writer.writerow(output)
    return buffer.getvalue()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_analysis(
    settings: Sequence[Mapping[str, Any]],
    effects: Sequence[Mapping[str, Any]],
    sources: Sequence[Mapping[str, str]],
    output_dir: Path,
) -> Dict[str, Path]:
    output_dir = Path(output_dir).expanduser().resolve()
    settings_path = output_dir / SETTINGS_FILENAME
    effects_path = output_dir / EFFECTS_FILENAME
    json_path = output_dir / JSON_FILENAME
    summary_zh, best = _article_summary(settings)
    status_counts = {
        status: sum(row["analysis_status"] == status for row in settings)
        for status in (STATUS_REACHED, STATUS_NOT_REACHED, STATUS_AUDIT_FAILED)
    }
    document = {
        "schema_version": "chignolin-ir-ivm-parameter-scan-analysis-v1",
        "i_perp_threshold": IPERP_THRESHOLD,
        "number_of_forces_definition": "sum of all 22 image joint E/F cache misses",
        "usable_path_state_rule": EXPECTED_USABLE_RULE,
        "not_reached_interpretation": (
            "right-censored at the last complete path state; censoring NF is not a first-passage cost"
        ),
        "dense_barrier_computed": False,
        "pdf_generated": False,
        "source_audit_summaries": list(sources),
        "parameter_setting_count": len(settings),
        "status_counts": status_counts,
        "best_reached_setting": best,
        "article_summary_zh": summary_zh,
        "parameter_settings": list(settings),
        "parameter_effect_records": list(effects),
    }
    serialized = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _atomic_write(settings_path, _csv_text(settings, SETTINGS_FIELDS))
    _atomic_write(effects_path, _csv_text(effects, EFFECT_FIELDS))
    _atomic_write(json_path, serialized)
    return {
        "parameter_settings_csv": settings_path,
        "parameter_effects_long_csv": effects_path,
        "parameter_scan_analysis_json": json_path,
    }


def analyze(paths: Sequence[Path], output_dir: Path) -> Dict[str, Any]:
    settings, sources = load_parameter_settings(paths)
    effects = parameter_effect_records(settings)
    outputs = write_analysis(settings, effects, sources, output_dir)
    return {"settings": settings, "effects": effects, "sources": sources, "outputs": outputs}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_summary", nargs="+", help="auditor-generated CSV/JSON")
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    result = analyze(
        [Path(item) for item in args.audit_summary], Path(args.output_dir)
    )
    outputs = result["outputs"]
    print(
        "analyzed {} parameter setting(s); settings={} effects={} JSON={}".format(
            len(result["settings"]),
            outputs["parameter_settings_csv"],
            outputs["parameter_effects_long_csv"],
            outputs["parameter_scan_analysis_json"],
        )
    )


if __name__ == "__main__":
    main()
