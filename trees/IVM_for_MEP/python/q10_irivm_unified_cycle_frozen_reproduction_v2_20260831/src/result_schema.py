"""Versioned, pickle-free NPZ contract for one frozen QM10 reproduction.

The archive keeps every charged energy/gradient value and every HJ spring
input/output in full.  The much larger stream of improved-tangent calls is
verified online by a separately implemented HJ formula and retained as three
ordered 32-byte digests per call, parent/context codes, context counts, and a
transcript hash.  Spring tangents remain independently replayable from each
saved spring input and fixed energy vector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = 2
DIMENSION = 10
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_NOF_CATEGORIES = (
    "initial_band",
    "recovery_band_trial",
    "normal_subproblem",
    "direct_spring_check",
    "synchronized_recovery_trial",
)


class ResultSchemaError(RuntimeError):
    """Raised when a result cannot cross the formal artifact boundary."""


@dataclass(frozen=True)
class ResultDimensions:
    n_internal: int
    public_states: int
    evaluations: int
    nof_categories: int
    main_events: int
    recovery_events: int
    recovery_states: int
    reparam_events: int
    tangent_events: int
    memory_pairs: int


@dataclass(frozen=True)
class ValidationReport:
    schema_version: int
    dimensions: ResultDimensions
    terminal_nof: int
    budget: int

    def to_jsonable(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dimensions"] = asdict(self.dimensions)
        return payload


UNICODE_SCALARS = {
    "campaign_id", "run_id", "config_json", "registration_sha256",
    "registration_seal_sha256", "algorithm_source_sha256", "optimizer_sha256",
    "adapter_sha256", "runtime_sha256", "backend_sha256", "runner_sha256",
    "dependency_manifest_json", "exploratory_source_sha256",
    "preformal_equivalence_sha256", "instrumentation_equivalence_sha256",
    "runtime_identity_json",
    "optimizer_summary_json", "q_sha256", "tangent_definition",
    "reparameterization_definition", "tangent_transcript_sha256",
    "stop_reason", "failure_message",
}
JSON_SCALARS = {
    "config_json", "dependency_manifest_json", "runtime_identity_json",
    "optimizer_summary_json",
}
SHA256_SCALARS = {
    "registration_sha256", "registration_seal_sha256",
    "algorithm_source_sha256", "optimizer_sha256", "adapter_sha256",
    "runtime_sha256", "backend_sha256", "runner_sha256",
    "exploratory_source_sha256", "preformal_equivalence_sha256",
    "instrumentation_equivalence_sha256", "q_sha256",
    "tangent_transcript_sha256",
}
FLOAT_SCALARS = {"spring_dtau", "spring_tolerance"}
INT_SCALARS = {
    "schema_version", "spring_max_iter", "budget", "terminal_nof",
    "blocked_uncached_misses", "tangent_call_count", "tangent_mismatch_count",
    "tangent_reparam_call_count",
}
BOOL_SCALARS = {"working_equals_public_at_exit"}
INT_VECTORS = {
    "public_nf", "public_outer", "evaluation_sequence", "evaluation_nof",
    "evaluation_outer", "evaluation_image_index", "per_image_joint_evaluations",
    "nof_category_counts", "recovery_state_nf", "reparam_event_id",
    "reparam_outer", "reparam_nof_at_entry", "reparam_iterations",
    "reparam_tangent_call_start", "reparam_tangent_call_count",
    "tangent_event_id", "tangent_parent_reparam_event_id",
    "tangent_context_counts", "memory_image_index", "memory_pair_order",
}
FLOAT_ARRAYS = {
    "q_matrix", "public_path_history", "public_energy_history",
    "public_gradient_history", "evaluation_coordinate", "evaluation_energy",
    "evaluation_gradient", "recovery_anchor_path", "recovery_candidate_path",
    "recovery_public_before_path", "reparam_input_path", "reparam_fixed_energy",
    "reparam_output_path", "reparam_final_residual", "memory_s", "memory_y",
}
BOOL_VECTORS = {"public_state_changed"}
INT16_VECTORS = {"tangent_context_code"}
UINT8_ARRAYS = {
    "tangent_path_sha256", "tangent_energy_sha256", "tangent_output_sha256"
}
UNICODE_VECTORS = {
    "public_phase", "evaluation_phase", "evaluation_coordinate_sha256",
    "optimizer_evaluation_trace_json", "nof_category_names", "main_event_json",
    "recovery_event_json", "reparam_caller", "tangent_context_names",
}
REQUIRED_FIELDS = (
    UNICODE_SCALARS | FLOAT_SCALARS | INT_SCALARS | BOOL_SCALARS | INT_VECTORS
    | FLOAT_ARRAYS | BOOL_VECTORS | INT16_VECTORS | UINT8_ARRAYS
    | UNICODE_VECTORS
)


def coordinate_sha256(coordinate: np.ndarray) -> str:
    canonical = np.ascontiguousarray(
        np.asarray(coordinate, dtype="<f8").reshape(DIMENSION)
    )
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def q_matrix_sha256(matrix: np.ndarray) -> str:
    canonical = np.ascontiguousarray(
        np.asarray(matrix, dtype="<f8").reshape(DIMENSION, DIMENSION)
    )
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def tangent_transcript_sha256(mapping: Mapping[str, Any]) -> str:
    """Recompute the ordered compact tangent transcript digest."""

    event_ids = _array(mapping, "tangent_event_id")
    parents = _array(mapping, "tangent_parent_reparam_event_id")
    codes = _array(mapping, "tangent_context_code")
    names = _array(mapping, "tangent_context_names").tolist()
    path_digests = _array(mapping, "tangent_path_sha256")
    energy_digests = _array(mapping, "tangent_energy_sha256")
    output_digests = _array(mapping, "tangent_output_sha256")
    digest = hashlib.sha256(b"q10-tangent-transcript-v2\x00")
    for index in range(event_ids.shape[0]):
        code = int(codes[index])
        if code < 0 or code >= len(names):
            raise ResultSchemaError("tangent context code lies outside names")
        context = str(names[code]).encode("utf-8")
        digest.update(struct.pack(">Q", int(event_ids[index])))
        digest.update(struct.pack(">q", int(parents[index])))
        digest.update(struct.pack(">I", len(context)))
        digest.update(context)
        digest.update(np.ascontiguousarray(path_digests[index]).tobytes())
        digest.update(np.ascontiguousarray(energy_digests[index]).tobytes())
        digest.update(np.ascontiguousarray(output_digests[index]).tobytes())
    return digest.hexdigest()


def _array(mapping: Mapping[str, Any], name: str) -> np.ndarray:
    try:
        return np.asarray(mapping[name])
    except KeyError as exc:
        raise ResultSchemaError(f"missing NPZ field: {name}") from exc


def _scalar(mapping: Mapping[str, Any], name: str) -> Any:
    value = _array(mapping, name)
    if value.shape != ():
        raise ResultSchemaError(f"{name} must be a scalar, got {value.shape}")
    return value.item()


def _require_dtype(array: np.ndarray, dtype: Any, name: str) -> None:
    if array.dtype != np.dtype(dtype):
        raise ResultSchemaError(
            f"{name} dtype must be {np.dtype(dtype)}, got {array.dtype}"
        )


def _require_unicode(array: np.ndarray, name: str) -> None:
    if array.dtype.kind != "U":
        raise ResultSchemaError(f"{name} must use Unicode dtype, got {array.dtype}")


def _require_shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    if array.shape != shape:
        raise ResultSchemaError(f"{name} shape must be {shape}, got {array.shape}")


def _parse_json_text(raw: str, label: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ResultSchemaError(f"{label} is not valid JSON") from exc


def _parse_json_scalar(mapping: Mapping[str, Any], name: str) -> Any:
    return _parse_json_text(str(_scalar(mapping, name)), name)


def _infer_dimensions(mapping: Mapping[str, Any]) -> ResultDimensions:
    paths = _array(mapping, "public_path_history")
    if paths.ndim != 3 or paths.shape[2] != DIMENSION:
        raise ResultSchemaError("public_path_history must have shape [S,N+2,10]")
    states, total, _ = paths.shape
    n_internal = total - 2
    if n_internal not in {10, 20} or states <= 0:
        raise ResultSchemaError("formal result must have N_internal 10 or 20")
    return ResultDimensions(
        n_internal=n_internal,
        public_states=states,
        evaluations=int(_array(mapping, "evaluation_sequence").shape[0]),
        nof_categories=int(_array(mapping, "nof_category_names").shape[0]),
        main_events=int(_array(mapping, "main_event_json").shape[0]),
        recovery_events=int(_array(mapping, "recovery_event_json").shape[0]),
        recovery_states=int(_array(mapping, "recovery_state_nf").shape[0]),
        reparam_events=int(_array(mapping, "reparam_event_id").shape[0]),
        tangent_events=int(_array(mapping, "tangent_event_id").shape[0]),
        memory_pairs=int(_array(mapping, "memory_image_index").shape[0]),
    )


def validate_result_mapping(
    mapping: Mapping[str, Any], *, strict_extra_fields: bool = True
) -> ValidationReport:
    keys = set(mapping.keys())
    missing = REQUIRED_FIELDS - keys
    extra = keys - REQUIRED_FIELDS
    if missing:
        raise ResultSchemaError(f"missing NPZ fields: {sorted(missing)}")
    if strict_extra_fields and extra:
        raise ResultSchemaError(f"unregistered NPZ fields: {sorted(extra)}")

    for name in UNICODE_SCALARS:
        value = _array(mapping, name)
        _require_shape(value, (), name)
        _require_unicode(value, name)
    for name in FLOAT_SCALARS:
        value = _array(mapping, name)
        _require_shape(value, (), name)
        _require_dtype(value, np.float64, name)
    for name in INT_SCALARS:
        value = _array(mapping, name)
        _require_shape(value, (), name)
        _require_dtype(value, np.int64, name)
    for name in BOOL_SCALARS:
        value = _array(mapping, name)
        _require_shape(value, (), name)
        _require_dtype(value, np.bool_, name)
    for name in INT_VECTORS:
        value = _array(mapping, name)
        if value.ndim != 1:
            raise ResultSchemaError(f"{name} must be one-dimensional")
        _require_dtype(value, np.int64, name)
    for name in FLOAT_ARRAYS:
        value = _array(mapping, name)
        _require_dtype(value, np.float64, name)
        if not np.all(np.isfinite(value)):
            raise ResultSchemaError(f"{name} contains a non-finite value")
    for name in BOOL_VECTORS:
        value = _array(mapping, name)
        if value.ndim != 1:
            raise ResultSchemaError(f"{name} must be one-dimensional")
        _require_dtype(value, np.bool_, name)
    for name in INT16_VECTORS:
        value = _array(mapping, name)
        if value.ndim != 1:
            raise ResultSchemaError(f"{name} must be one-dimensional")
        _require_dtype(value, np.int16, name)
    for name in UINT8_ARRAYS:
        _require_dtype(_array(mapping, name), np.uint8, name)
    for name in UNICODE_VECTORS:
        value = _array(mapping, name)
        if value.ndim != 1:
            raise ResultSchemaError(f"{name} must be one-dimensional")
        _require_unicode(value, name)

    if int(_scalar(mapping, "schema_version")) != SCHEMA_VERSION:
        raise ResultSchemaError("schema_version drifted")
    for name in JSON_SCALARS:
        _parse_json_scalar(mapping, name)
    for name in (
        "main_event_json", "recovery_event_json", "optimizer_evaluation_trace_json"
    ):
        for index, raw in enumerate(_array(mapping, name).tolist()):
            decoded = _parse_json_text(str(raw), f"{name}[{index}]")
            if not isinstance(decoded, dict):
                raise ResultSchemaError(f"{name}[{index}] must be a JSON object")
    for name in SHA256_SCALARS:
        if SHA256_RE.fullmatch(str(_scalar(mapping, name))) is None:
            raise ResultSchemaError(f"{name} is not a lowercase SHA-256 digest")
    if str(_scalar(mapping, "algorithm_source_sha256")) != str(
        _scalar(mapping, "optimizer_sha256")
    ):
        raise ResultSchemaError("optimizer_sha256 must alias algorithm source")

    dims = _infer_dimensions(mapping)
    n, total = dims.n_internal, dims.n_internal + 2
    s, t, c = dims.public_states, dims.evaluations, dims.nof_categories
    m, r, k = dims.main_events, dims.recovery_events, dims.recovery_states
    p, h = dims.reparam_events, dims.memory_pairs
    context_count = int(_array(mapping, "tangent_context_names").shape[0])
    expected_shapes = {
        "q_matrix": (DIMENSION, DIMENSION),
        "public_nf": (s,), "public_outer": (s,), "public_phase": (s,),
        "public_path_history": (s, total, DIMENSION),
        "public_energy_history": (s, total),
        "public_gradient_history": (s, total, DIMENSION),
        "public_state_changed": (s,), "evaluation_sequence": (t,),
        "evaluation_nof": (t,), "evaluation_outer": (t,),
        "evaluation_phase": (t,), "evaluation_image_index": (t,),
        "evaluation_coordinate": (t, DIMENSION),
        "evaluation_coordinate_sha256": (t,), "evaluation_energy": (t,),
        "evaluation_gradient": (t, DIMENSION),
        "optimizer_evaluation_trace_json": (t,),
        "per_image_joint_evaluations": (total,), "nof_category_names": (c,),
        "nof_category_counts": (c,), "main_event_json": (m,),
        "recovery_event_json": (r,), "recovery_state_nf": (k,),
        "recovery_anchor_path": (k, total, DIMENSION),
        "recovery_candidate_path": (k, total, DIMENSION),
        "recovery_public_before_path": (k, total, DIMENSION),
        "reparam_event_id": (p,), "reparam_caller": (p,),
        "reparam_outer": (p,), "reparam_nof_at_entry": (p,),
        "reparam_input_path": (p, total, DIMENSION),
        "reparam_fixed_energy": (p, total),
        "reparam_output_path": (p, total, DIMENSION),
        "reparam_iterations": (p,), "reparam_final_residual": (p,),
        "reparam_tangent_call_start": (p,), "reparam_tangent_call_count": (p,),
        "tangent_event_id": (dims.tangent_events,),
        "tangent_parent_reparam_event_id": (dims.tangent_events,),
        "tangent_context_names": (context_count,),
        "tangent_context_code": (dims.tangent_events,),
        "tangent_context_counts": (context_count,),
        "tangent_path_sha256": (dims.tangent_events, 32),
        "tangent_energy_sha256": (dims.tangent_events, 32),
        "tangent_output_sha256": (dims.tangent_events, 32),
        "memory_image_index": (h,),
        "memory_pair_order": (h,), "memory_s": (h, DIMENSION),
        "memory_y": (h, DIMENSION),
    }
    for name, shape in expected_shapes.items():
        _require_shape(_array(mapping, name), shape, name)

    config = _parse_json_scalar(mapping, "config_json")
    if not isinstance(config, dict) or int(config.get("n_internal", -1)) != n:
        raise ResultSchemaError("config_json.n_internal disagrees with result")
    budget = int(_scalar(mapping, "budget"))
    terminal = int(_scalar(mapping, "terminal_nof"))
    if budget != 3000 or not (n <= terminal <= budget):
        raise ResultSchemaError("invalid frozen budget or terminal NoF")
    if int(_scalar(mapping, "blocked_uncached_misses")) < 0:
        raise ResultSchemaError("blocked_uncached_misses cannot be negative")
    if not bool(_scalar(mapping, "working_equals_public_at_exit")):
        raise ResultSchemaError("working state differs from public state at exit")
    if float(_scalar(mapping, "spring_dtau")) != 0.05:
        raise ResultSchemaError("spring_dtau differs from 0.05")
    if int(_scalar(mapping, "spring_max_iter")) != 500:
        raise ResultSchemaError("spring_max_iter differs from 500")
    if float(_scalar(mapping, "spring_tolerance")) != 1.0e-5:
        raise ResultSchemaError("spring_tolerance differs from 1e-5")
    if q_matrix_sha256(_array(mapping, "q_matrix")) != str(
        _scalar(mapping, "q_sha256")
    ):
        raise ResultSchemaError("q_matrix does not match q_sha256")

    public_nf = _array(mapping, "public_nf")
    if int(public_nf[0]) != n or int(public_nf[-1]) != terminal:
        raise ResultSchemaError("public NoF must begin at N and end at terminal")
    if np.any(np.diff(public_nf) <= 0):
        raise ResultSchemaError("public_nf must be strictly increasing")
    changed = _array(mapping, "public_state_changed")
    if not bool(changed[0]):
        raise ResultSchemaError("initial state must be marked present/changed")
    public_paths = _array(mapping, "public_path_history")
    for index in range(1, s):
        actual = not np.array_equal(public_paths[index], public_paths[index - 1])
        if bool(changed[index]) != actual:
            raise ResultSchemaError(f"public_state_changed mismatch at {index}")
    if not np.all(public_paths[:, 0] == public_paths[0, 0]):
        raise ResultSchemaError("left endpoint changed")
    if not np.all(public_paths[:, -1] == public_paths[0, -1]):
        raise ResultSchemaError("right endpoint changed")

    expected_trace = np.arange(1, t + 1, dtype=np.int64)
    if not np.array_equal(_array(mapping, "evaluation_sequence"), expected_trace):
        raise ResultSchemaError("evaluation_sequence must equal 1..T")
    if not np.array_equal(_array(mapping, "evaluation_nof"), expected_trace):
        raise ResultSchemaError("evaluation_nof must equal 1..T")
    if t != terminal:
        raise ResultSchemaError("evaluation trace length does not close to NoF")
    image_indices = _array(mapping, "evaluation_image_index")
    if np.any((image_indices < 1) | (image_indices > n)):
        raise ResultSchemaError("charged trace contains an endpoint")
    per_image = _array(mapping, "per_image_joint_evaluations")
    reconstructed = np.bincount(image_indices, minlength=total).astype(np.int64)
    if not np.array_equal(per_image, reconstructed):
        raise ResultSchemaError("per-image NoF ledger does not close")
    categories = tuple(_array(mapping, "nof_category_names").tolist())
    if categories != EXPECTED_NOF_CATEGORIES:
        raise ResultSchemaError("NoF category names/order drifted")
    category_counts = _array(mapping, "nof_category_counts")
    phases = _array(mapping, "evaluation_phase")
    observed_counts = np.asarray(
        [int(np.count_nonzero(phases == name)) for name in categories],
        dtype=np.int64,
    )
    if not np.array_equal(category_counts, observed_counts):
        raise ResultSchemaError("evaluation phases disagree with NoF breakdown")
    if int(np.sum(category_counts)) != terminal:
        raise ResultSchemaError("NoF category ledger does not close")
    coordinates = _array(mapping, "evaluation_coordinate")
    coordinate_hashes = _array(mapping, "evaluation_coordinate_sha256")
    optimizer_trace = _array(mapping, "optimizer_evaluation_trace_json")
    for index in range(t):
        digest = coordinate_sha256(coordinates[index])
        if digest != str(coordinate_hashes[index]):
            raise ResultSchemaError(f"coordinate hash mismatch at {index}")
        core_event = _parse_json_text(
            str(optimizer_trace[index]), f"optimizer_evaluation_trace_json[{index}]"
        )
        if (
            int(core_event.get("sequence", -1)) != index + 1
            or int(core_event.get("article_nf", -1)) != index + 1
            or int(core_event.get("image_index", -1)) != int(image_indices[index])
            or str(core_event.get("coordinate_sha256", "")) != digest
        ):
            raise ResultSchemaError(f"optimizer trace mismatch at {index}")

    recovery_nf = _array(mapping, "recovery_state_nf")
    if k and np.any((recovery_nf < n) | (recovery_nf > terminal)):
        raise ResultSchemaError("recovery state NoF lies outside the run")
    if p:
        if not np.array_equal(
            _array(mapping, "reparam_event_id"), np.arange(1, p + 1)
        ):
            raise ResultSchemaError("reparam_event_id must equal 1..P")
        iterations = _array(mapping, "reparam_iterations")
        counts = _array(mapping, "reparam_tangent_call_count")
        starts = _array(mapping, "reparam_tangent_call_start")
        if np.any((iterations < 1) | (iterations > 500)):
            raise ResultSchemaError("reparam_iterations outside 1..500")
        if not np.array_equal(iterations, counts):
            raise ResultSchemaError("every spring iteration must call one HJ tangent")
        if np.any(starts < 0) or np.any(starts + counts > dims.tangent_events):
            raise ResultSchemaError("spring tangent range lies outside transcript")
        if p > 1 and np.any(starts[1:] < starts[:-1] + counts[:-1]):
            raise ResultSchemaError("spring tangent ranges overlap or reorder")
        if np.any(_array(mapping, "reparam_final_residual") < 0.0):
            raise ResultSchemaError("spring residual cannot be negative")
        inputs = _array(mapping, "reparam_input_path")
        outputs = _array(mapping, "reparam_output_path")
        if not np.all(inputs[:, (0, -1)] == outputs[:, (0, -1)]):
            raise ResultSchemaError("spring reparameterization moved an endpoint")
        if any(not item for item in _array(mapping, "reparam_caller").tolist()):
            raise ResultSchemaError("empty reparameterization caller")

    tangent_total = int(_scalar(mapping, "tangent_call_count"))
    event_ids = _array(mapping, "tangent_event_id")
    parents = _array(mapping, "tangent_parent_reparam_event_id")
    context_codes = _array(mapping, "tangent_context_code")
    if tangent_total != dims.tangent_events:
        raise ResultSchemaError("tangent_call_count disagrees with transcript length")
    if not np.array_equal(
        event_ids, np.arange(1, tangent_total + 1, dtype=np.int64)
    ):
        raise ResultSchemaError("tangent_event_id must equal 1..G")
    if np.any((parents < -1) | (parents == 0) | (parents > p)):
        raise ResultSchemaError("invalid tangent parent spring ID")
    if tangent_total < 0 or int(_scalar(mapping, "tangent_mismatch_count")) != 0:
        raise ResultSchemaError("runtime tangent verification did not close cleanly")
    reparam_tangents = int(np.sum(_array(mapping, "reparam_tangent_call_count")))
    if reparam_tangents != int(_scalar(mapping, "tangent_reparam_call_count")):
        raise ResultSchemaError("spring tangent call count does not close")
    if reparam_tangents != int(np.count_nonzero(parents > 0)):
        raise ResultSchemaError("parent-linked spring tangent count does not close")
    starts = _array(mapping, "reparam_tangent_call_start")
    counts = _array(mapping, "reparam_tangent_call_count")
    for index in range(p):
        start, count = int(starts[index]), int(counts[index])
        if not np.all(parents[start:start + count] == index + 1):
            raise ResultSchemaError("spring tangent slice has wrong parent ID")
    context_names = _array(mapping, "tangent_context_names").tolist()
    if any(not item for item in context_names) or len(set(context_names)) != len(context_names):
        raise ResultSchemaError("tangent context names are empty or duplicated")
    if tangent_total and (
        np.any(context_codes < 0) or np.any(context_codes >= len(context_names))
    ):
        raise ResultSchemaError("tangent context code lies outside names")
    if np.any(_array(mapping, "tangent_context_counts") < 0):
        raise ResultSchemaError("negative tangent context count")
    context_counts = _array(mapping, "tangent_context_counts")
    reconstructed_contexts = np.bincount(
        np.asarray(context_codes, dtype=np.int64), minlength=len(context_names)
    ).astype(np.int64)
    if not np.array_equal(context_counts, reconstructed_contexts):
        raise ResultSchemaError("tangent context counts disagree with event codes")
    if int(np.sum(context_counts)) != tangent_total:
        raise ResultSchemaError("tangent context counts do not close")
    if tangent_transcript_sha256(mapping) != str(
        _scalar(mapping, "tangent_transcript_sha256")
    ):
        raise ResultSchemaError("tangent transcript SHA-256 mismatch")

    memory_images = _array(mapping, "memory_image_index")
    memory_orders = _array(mapping, "memory_pair_order")
    if h and np.any((memory_images < 1) | (memory_images > n)):
        raise ResultSchemaError("terminal recovery memory has invalid image index")
    if np.any(memory_orders < 0):
        raise ResultSchemaError("terminal recovery memory has negative order")
    if len(set(zip(memory_images.tolist(), memory_orders.tolist()))) != h:
        raise ResultSchemaError("terminal recovery memory labels are duplicated")

    return ValidationReport(SCHEMA_VERSION, dims, terminal, budget)


def validate_result_archive(
    path: str | Path, *, strict_extra_fields: bool = True
) -> ValidationReport:
    with np.load(Path(path), allow_pickle=False) as archive:
        return validate_result_mapping(
            archive, strict_extra_fields=strict_extra_fields
        )
