"""Write-after observation adapter for the frozen QM10 IR-IVM optimizer.

The adapter monkeypatches only call boundaries.  Every numerical return object
is passed back unchanged; copies, independent tangent checks, and hashes are
made only after the frozen implementation has produced a value.  The formal
optimizer source is not edited by this module.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
import hashlib
import json
import struct
import sys
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .optimizer_interface import FrozenControls
from .result_schema import EXPECTED_NOF_CATEGORIES, SCHEMA_VERSION


class AdapterObservationError(RuntimeError):
    pass


def _canonical_f64(value: Any) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype="<f8"))


def _unicode(values: Sequence[Any]) -> np.ndarray:
    text = [str(value) for value in values]
    width = max([len(value) for value in text] or [1])
    return np.asarray(text, dtype="<U%d" % width)


def _unicode_scalar(value: Any) -> np.ndarray:
    return np.asarray(str(value), dtype="U")


def _stack(values: Sequence[Any], shape: Tuple[int, ...]) -> np.ndarray:
    if not values:
        return np.empty((0,) + shape, dtype=np.float64)
    return np.ascontiguousarray(np.stack(values), dtype=np.float64)


def _array_sha256(value: Any, domain: str) -> bytes:
    array = _canonical_f64(value)
    digest = hashlib.sha256()
    digest.update(b"q10-formal-array-v2\x00")
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00<f8\x00")
    digest.update(struct.pack(">Q", int(array.ndim)))
    for extent in array.shape:
        digest.update(struct.pack(">Q", int(extent)))
    digest.update(array.tobytes(order="C"))
    return digest.digest()


def _coordinate_sha256(value: Any) -> str:
    return hashlib.sha256(
        _canonical_f64(value).reshape(-1).tobytes(order="C")
    ).hexdigest()


def independent_hj_improved_tangents(path: Any, energies: Any) -> np.ndarray:
    """Independent transcription of the frozen HJ improved-tangent formula."""

    points = np.asarray(path, dtype=np.float64)
    values = np.asarray(energies, dtype=np.float64)
    if points.ndim != 2 or values.shape != (points.shape[0],):
        raise AdapterObservationError("invalid improved-tangent input shape")
    tangents = np.empty(
        (points.shape[0] - 2, points.shape[1]), dtype=np.float64
    )
    for output_index, image_index in enumerate(range(1, points.shape[0] - 1)):
        forward = points[image_index + 1] - points[image_index]
        backward = points[image_index] - points[image_index - 1]
        e_previous, e_current, e_next = values[image_index - 1:image_index + 2]
        if e_next > e_current > e_previous:
            tangent = forward.copy()
        elif e_next < e_current < e_previous:
            tangent = backward.copy()
        else:
            delta_plus = abs(e_next - e_current)
            delta_minus = abs(e_previous - e_current)
            delta_maximum = max(delta_plus, delta_minus)
            delta_minimum = min(delta_plus, delta_minus)
            if e_next > e_previous:
                tangent = forward * delta_maximum + backward * delta_minimum
            else:
                tangent = forward * delta_minimum + backward * delta_maximum
        norm = float(np.linalg.norm(tangent))
        if norm <= 1.0e-14:
            tangent = forward + backward
            norm = float(np.linalg.norm(tangent))
        if norm <= 1.0e-14:
            raise FloatingPointError("degenerate independent improved tangent")
        tangents[output_index] = tangent / norm
    return tangents


@dataclass
class EvaluationEvent:
    sequence: int
    nof: int
    outer: int
    phase: str
    image_index: int
    coordinate: np.ndarray
    energy: float
    gradient: np.ndarray


@dataclass
class TangentEvent:
    event_id: int
    context: str
    parent_reparam_event_id: int
    path_sha256: bytes
    energy_sha256: bytes
    output_sha256: bytes


@dataclass
class ReparameterizationEvent:
    event_id: int
    caller: str
    outer: int
    nof_at_entry: int
    input_path: np.ndarray
    fixed_energy: np.ndarray
    tangent_call_start: int
    output_path: Optional[np.ndarray] = None
    iterations: Optional[int] = None
    final_residual: Optional[float] = None
    tangent_call_count: Optional[int] = None


@dataclass
class RuntimeObserver:
    evaluations: List[EvaluationEvent] = field(default_factory=list)
    tangents: List[TangentEvent] = field(default_factory=list)
    reparameterizations: List[ReparameterizationEvent] = field(default_factory=list)
    public_outer_by_nf: Dict[int, int] = field(default_factory=dict)
    tangent_mismatch_count: int = 0
    ledger: Optional[Any] = None
    _active_reparam_event_id: int = -1
    _active_reparam_caller: Optional[str] = None

    @staticmethod
    def _runtime_frames() -> Iterator[Any]:
        frame = sys._getframe(1)
        while frame is not None:
            yield frame
            frame = frame.f_back

    def current_outer(self) -> int:
        for frame in self._runtime_frames():
            if frame.f_code.co_name == "run_clean" and "outer" in frame.f_locals:
                return int(frame.f_locals["outer"])
        return -1

    def current_evaluation_phase(self) -> str:
        names = []
        for frame in self._runtime_frames():
            name = frame.f_code.co_name
            names.append(name)
            if name == "evaluate_band_counted" and "label" in frame.f_locals:
                label = str(frame.f_locals["label"])
                if label in EXPECTED_NOF_CATEGORIES:
                    return label
        if "synchronized_spring_recovery" in names:
            return "synchronized_recovery_trial"
        if "solve_main_stage_counted" in names or "solve_main_stage" in names:
            return "normal_subproblem"
        raise AdapterObservationError(
            "charged evaluation route has no registered NoF category: "
            + " -> ".join(names[:12])
        )

    def bind_ledger(self, ledger: Any) -> None:
        if self.ledger is None:
            self.ledger = ledger
        elif self.ledger is not ledger:
            raise AdapterObservationError("one observer saw multiple NoF ledgers")

    def record_evaluation(
        self, evaluator: Any, coordinate: Any, energy: Any, gradient: Any
    ) -> None:
        self.bind_ledger(evaluator.ledger)
        sequence = int(evaluator.ledger.total)
        if sequence != len(self.evaluations) + 1:
            raise AdapterObservationError("charged evaluation sequence lost closure")
        self.evaluations.append(
            EvaluationEvent(
                sequence=sequence,
                nof=sequence,
                outer=self.current_outer(),
                phase=self.current_evaluation_phase(),
                image_index=int(evaluator.image_index),
                coordinate=_canonical_f64(coordinate).copy(),
                energy=float(energy),
                gradient=_canonical_f64(gradient).copy(),
            )
        )

    def record_tangent(self, path: Any, energies: Any, output: Any) -> None:
        expected = independent_hj_improved_tangents(path, energies)
        observed = np.asarray(output)
        if not np.array_equal(expected, observed):
            self.tangent_mismatch_count += 1
            raise AdapterObservationError(
                "sealed improved_tangents disagrees with independent HJ formula"
            )
        caller = sys._getframe(1).f_back.f_code.co_name
        if self._active_reparam_event_id > 0:
            context = "spring:" + str(self._active_reparam_caller)
        else:
            context = str(caller)
        self.tangents.append(
            TangentEvent(
                event_id=len(self.tangents) + 1,
                context=context,
                parent_reparam_event_id=int(self._active_reparam_event_id),
                path_sha256=_array_sha256(path, "tangent-input-path"),
                energy_sha256=_array_sha256(energies, "tangent-input-energy"),
                output_sha256=_array_sha256(output, "tangent-output"),
            )
        )

    def begin_reparameterization(
        self, caller: str, path: Any, energies: Any
    ) -> int:
        if self._active_reparam_event_id != -1:
            raise AdapterObservationError("nested HJ spring call is forbidden")
        if self.ledger is None:
            raise AdapterObservationError("HJ spring preceded the initial force ledger")
        event_id = len(self.reparameterizations) + 1
        self.reparameterizations.append(
            ReparameterizationEvent(
                event_id=event_id,
                caller=str(caller),
                outer=self.current_outer(),
                nof_at_entry=int(self.ledger.total),
                input_path=_canonical_f64(path).copy(),
                fixed_energy=_canonical_f64(energies).copy(),
                tangent_call_start=len(self.tangents),
            )
        )
        self._active_reparam_event_id = event_id
        self._active_reparam_caller = str(caller)
        return event_id

    def finish_reparameterization(
        self,
        event_id: int,
        output_path: Any,
        final_residual: Any,
        iterations: Any,
    ) -> None:
        if int(event_id) != self._active_reparam_event_id:
            raise AdapterObservationError("HJ spring event nesting drifted")
        event = self.reparameterizations[event_id - 1]
        event.output_path = _canonical_f64(output_path).copy()
        event.final_residual = float(final_residual)
        event.iterations = int(iterations)
        event.tangent_call_count = len(self.tangents) - event.tangent_call_start
        if event.tangent_call_count != event.iterations:
            raise AdapterObservationError(
                "each HJ spring iteration must make one improved-tangent call"
            )
        self._active_reparam_event_id = -1
        self._active_reparam_caller = None

    def abort_reparameterization(self, event_id: int) -> None:
        if int(event_id) == self._active_reparam_event_id:
            self._active_reparam_event_id = -1
            self._active_reparam_caller = None

    def validate(self, result: Mapping[str, Any]) -> None:
        if self.ledger is None:
            raise AdapterObservationError("formal run produced no force ledger")
        terminal = int(result["terminal_nf"])
        if len(self.evaluations) != terminal or int(self.ledger.total) != terminal:
            raise AdapterObservationError("full evaluation values do not close to NoF")
        trace = list(result["evaluation_miss_trace"])
        if len(trace) != terminal:
            raise AdapterObservationError("optimizer miss trace does not close")
        for event, original in zip(self.evaluations, trace):
            if (
                int(original["sequence"]) != event.sequence
                or int(original["article_nf"]) != event.nof
                or int(original["image_index"]) != event.image_index
                or str(original["coordinate_sha256"])
                != _coordinate_sha256(event.coordinate)
            ):
                raise AdapterObservationError(
                    "observer differs from optimizer miss trace at %d" % event.sequence
                )
        phase_counts = Counter(event.phase for event in self.evaluations)
        breakdown = dict(result["nof_breakdown"])
        if tuple(breakdown.keys()) != EXPECTED_NOF_CATEGORIES:
            raise AdapterObservationError("optimizer NoF category order drifted")
        for name in EXPECTED_NOF_CATEGORIES:
            if int(phase_counts[name]) != int(breakdown[name]):
                raise AdapterObservationError("NoF phase count drifted for " + name)
        if self._active_reparam_event_id != -1:
            raise AdapterObservationError("formal result ended in an HJ spring")
        if any(
            event.output_path is None
            or event.iterations is None
            or event.final_residual is None
            or event.tangent_call_count is None
            for event in self.reparameterizations
        ):
            raise AdapterObservationError("incomplete HJ spring provenance")
        if self.tangent_mismatch_count != 0:
            raise AdapterObservationError("runtime HJ tangent mismatch was recorded")


class AuditedCoreProxy:
    def __init__(self, raw_core: Any, observer: RuntimeObserver):
        self._raw_core = raw_core
        self._observer = observer
        self._raw_improved_tangents = raw_core.improved_tangents

    def __getattr__(self, name: str) -> Any:
        if name == "improved_tangents":
            return self.improved_tangents
        return getattr(self._raw_core, name)

    def improved_tangents(self, path: Any, energies: Any, *args: Any, **kwargs: Any) -> Any:
        result = self._raw_improved_tangents(path, energies, *args, **kwargs)
        self._observer.record_tangent(path, energies, result)
        return result


SPRING_CALLERS = {
    "recovery_proposal_spring": "recovery_proposal_spring",
    "synchronized_spring_recovery": "synchronized_spring_recovery",
    "build_severe_rejection_hj_spring_retry": "severe_rejection_hj_spring_retry",
    "prepare_normal_stream_exit_hj_spring": "normal_stream_exit_hj_spring",
    "run_clean": "post_main_direct_hj_spring",
}


@contextmanager
def observed_optimizer(optimizer: Any, observer: RuntimeObserver) -> Iterator[None]:
    """Install and restore the four transparent observation hooks."""

    algorithm_runtime = optimizer.alg
    original_load_backend = algorithm_runtime._load_backend
    original_query = algorithm_runtime.Evaluator.query
    original_spring = optimizer.hj_nearest_neighbor_spring_reparameterize
    original_append_record = optimizer.append_record

    def load_backend_observed(n_internal: int) -> Tuple[Any, Any]:
        raw_core, potential = original_load_backend(n_internal)
        return AuditedCoreProxy(raw_core, observer), potential

    def query_observed(evaluator: Any, value: Any) -> Any:
        observer.bind_ledger(evaluator.ledger)
        before = int(evaluator.ledger.total)
        result = original_query(evaluator, value)
        after = int(evaluator.ledger.total)
        if bool(evaluator.charged):
            if after == before + 1:
                observer.record_evaluation(evaluator, value, result[0], result[1])
            elif after != before:
                raise AdapterObservationError("one query changed NoF by more than one")
        elif after != before:
            raise AdapterObservationError("endpoint query changed NoF")
        return result

    def spring_observed(core: Any, path: Any, energies: Any) -> Any:
        raw_caller = sys._getframe(1).f_code.co_name
        if raw_caller not in SPRING_CALLERS:
            raise AdapterObservationError("unregistered HJ spring caller: " + raw_caller)
        event_id = observer.begin_reparameterization(
            SPRING_CALLERS[raw_caller], path, energies
        )
        try:
            result = original_spring(core, path, energies)
        except BaseException:
            observer.abort_reparameterization(event_id)
            raise
        observer.finish_reparameterization(
            event_id, result[0], result[1], result[2]
        )
        return result

    def append_record_observed(
        records: Any, nf: Any, phase: Any, state: Any
    ) -> None:
        original_append_record(records, nf, phase, state)
        observer.public_outer_by_nf[int(nf)] = observer.current_outer()

    algorithm_runtime._load_backend = load_backend_observed
    algorithm_runtime.Evaluator.query = query_observed
    optimizer.hj_nearest_neighbor_spring_reparameterize = spring_observed
    optimizer.append_record = append_record_observed
    try:
        yield
    finally:
        optimizer.append_record = original_append_record
        optimizer.hj_nearest_neighbor_spring_reparameterize = original_spring
        algorithm_runtime.Evaluator.query = original_query
        algorithm_runtime._load_backend = original_load_backend


def run_registered_member(
    optimizer: Any, controls: FrozenControls
) -> Tuple[Mapping[str, Any], RuntimeObserver]:
    controls.validate()
    if not callable(getattr(optimizer, "run_clean", None)):
        raise AdapterObservationError("sealed optimizer lacks run_clean")
    observer = RuntimeObserver()
    with observed_optimizer(optimizer, observer):
        result = optimizer.run_clean(
            int(controls.n_internal),
            float(controls.alpha),
            float(controls.r),
            int(controls.m_max),
            budget=int(controls.budget),
        )
    observer.validate(result)
    returned = result["controls"]
    if (
        int(returned.n_internal), float(returned.alpha), float(returned.r),
        int(returned.m_max)
    ) != (
        int(controls.n_internal), float(controls.alpha), float(controls.r),
        int(controls.m_max)
    ):
        raise AdapterObservationError("optimizer returned different controls")
    if str(returned.run_id) != controls.run_id:
        raise AdapterObservationError("optimizer and runner run IDs disagree")
    return result, observer


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        if np.isnan(value):
            return {"__nonfinite_float__": "nan"}
        return {"__nonfinite_float__": "inf" if value > 0 else "-inf"}
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise AdapterObservationError("cannot encode optimizer audit type: %r" % type(value))


def _json_text(value: Any) -> str:
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False,
    )


def _digest_matrix(events: Sequence[TangentEvent], attribute: str) -> np.ndarray:
    if not events:
        return np.empty((0, 32), dtype=np.uint8)
    return np.vstack(
        [np.frombuffer(getattr(event, attribute), dtype=np.uint8) for event in events]
    )


def _tangent_transcript_sha256(events: Sequence[TangentEvent]) -> str:
    digest = hashlib.sha256(b"q10-tangent-transcript-v2\x00")
    for event in events:
        context = event.context.encode("utf-8")
        digest.update(struct.pack(">Q", int(event.event_id)))
        digest.update(struct.pack(">q", int(event.parent_reparam_event_id)))
        digest.update(struct.pack(">I", len(context)))
        digest.update(context)
        digest.update(event.path_sha256)
        digest.update(event.energy_sha256)
        digest.update(event.output_sha256)
    return digest.hexdigest()


def result_npz_fields(
    result: Mapping[str, Any],
    observer: RuntimeObserver,
    controls: FrozenControls,
    identity: Mapping[str, Any],
) -> Dict[str, np.ndarray]:
    """Losslessly convert optimizer output and observer data to schema v2."""

    observer.validate(result)
    records = list(result["records"])
    if not records:
        raise AdapterObservationError("optimizer returned no public record")
    total_images = int(controls.n_internal) + 2
    public_paths = _stack(
        [item["path"] for item in records], (total_images, 10)
    )
    public_changed = np.ones(len(records), dtype=np.bool_)
    for index in range(1, len(records)):
        public_changed[index] = not np.array_equal(
            public_paths[index], public_paths[index - 1]
        )

    evaluations = observer.evaluations
    reparams = observer.reparameterizations
    tangents = observer.tangents
    contexts = sorted(set(event.context for event in tangents))
    context_codes = {name: index for index, name in enumerate(contexts)}
    tangent_code = np.asarray(
        [context_codes[event.context] for event in tangents], dtype=np.int16
    )
    tangent_counts = np.bincount(
        tangent_code.astype(np.int64), minlength=len(contexts)
    ).astype(np.int64)

    recovery_states = list(result["recovery_states"])
    memory_images = []
    memory_orders = []
    memory_s = []
    memory_y = []
    for image_offset, image_memory in enumerate(result["memories"], start=1):
        for pair_order, pair in enumerate(image_memory):
            memory_images.append(image_offset)
            memory_orders.append(pair_order)
            memory_s.append(pair[0])
            memory_y.append(pair[1])

    breakdown = dict(result["nof_breakdown"])
    if tuple(breakdown.keys()) != EXPECTED_NOF_CATEGORIES:
        raise AdapterObservationError("NoF category order drifted before serialization")
    result_identity = dict(identity["result_identity"])
    dependency_manifest = dict(identity["dependency_manifest"])
    summary = {
        "held_recovery_steps": int(result["held_recovery_steps"]),
        "trusted_nonmonotone_commits": int(result["trusted_nonmonotone_commits"]),
        "provisional_rollbacks": int(result["provisional_rollbacks"]),
        "max_provisional_depth": int(result["max_provisional_depth"]),
    }

    arrays = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int64),
        "campaign_id": _unicode_scalar(identity["campaign_id"]),
        "run_id": _unicode_scalar(controls.run_id),
        "config_json": _unicode_scalar(_json_text(controls.as_dict())),
        "registration_sha256": _unicode_scalar(identity["registration_sha256"]),
        "registration_seal_sha256": _unicode_scalar(identity["registration_seal_sha256"]),
        "algorithm_source_sha256": _unicode_scalar(result_identity["algorithm_source_sha256"]),
        "optimizer_sha256": _unicode_scalar(result_identity["optimizer_sha256"]),
        "adapter_sha256": _unicode_scalar(result_identity["adapter_sha256"]),
        "runtime_sha256": _unicode_scalar(result_identity["runtime_sha256"]),
        "backend_sha256": _unicode_scalar(result_identity["backend_sha256"]),
        "runner_sha256": _unicode_scalar(result_identity["runner_sha256"]),
        "dependency_manifest_json": _unicode_scalar(_json_text(dependency_manifest)),
        "exploratory_source_sha256": _unicode_scalar(result_identity["exploratory_source_sha256"]),
        "preformal_equivalence_sha256": _unicode_scalar(result_identity["preformal_equivalence_sha256"]),
        "instrumentation_equivalence_sha256": _unicode_scalar(
            result_identity["instrumentation_equivalence_sha256"]
        ),
        "runtime_identity_json": _unicode_scalar(_json_text(identity["runtime_identity"])),
        "optimizer_summary_json": _unicode_scalar(_json_text(summary)),
        "q_matrix": np.ascontiguousarray(result["q_matrix"], dtype=np.float64),
        "q_sha256": _unicode_scalar(result_identity["q_sha256"]),
        "tangent_definition": _unicode_scalar("Henkelman-Jonsson improved tangent everywhere"),
        "reparameterization_definition": _unicode_scalar("nearest-neighbour (L[i+1]-L[i])*tau_i HJ spring flow"),
        "spring_dtau": np.asarray(0.05, dtype=np.float64),
        "spring_max_iter": np.asarray(500, dtype=np.int64),
        "spring_tolerance": np.asarray(1.0e-5, dtype=np.float64),
        "stop_reason": _unicode_scalar(result["stop_reason"]),
        "failure_message": _unicode_scalar(""),
        "budget": np.asarray(controls.budget, dtype=np.int64),
        "terminal_nof": np.asarray(result["terminal_nf"], dtype=np.int64),
        "public_nf": np.asarray([item["nf"] for item in records], dtype=np.int64),
        "public_outer": np.asarray(
            [observer.public_outer_by_nf[int(item["nf"])] for item in records],
            dtype=np.int64,
        ),
        "public_phase": _unicode([item["phase"] for item in records]),
        "public_path_history": public_paths,
        "public_energy_history": _stack(
            [item["energies"] for item in records], (total_images,)
        ),
        "public_gradient_history": _stack(
            [item["gradients"] for item in records], (total_images, 10)
        ),
        "public_state_changed": public_changed,
        "working_equals_public_at_exit": np.asarray(
            bool(result["working_equals_public_at_exit"]), dtype=np.bool_
        ),
        "evaluation_sequence": np.asarray([item.sequence for item in evaluations], dtype=np.int64),
        "evaluation_nof": np.asarray([item.nof for item in evaluations], dtype=np.int64),
        "evaluation_outer": np.asarray([item.outer for item in evaluations], dtype=np.int64),
        "evaluation_phase": _unicode([item.phase for item in evaluations]),
        "evaluation_image_index": np.asarray([item.image_index for item in evaluations], dtype=np.int64),
        "evaluation_coordinate": _stack([item.coordinate for item in evaluations], (10,)),
        "evaluation_coordinate_sha256": _unicode([_coordinate_sha256(item.coordinate) for item in evaluations]),
        "evaluation_energy": np.asarray([item.energy for item in evaluations], dtype=np.float64),
        "evaluation_gradient": _stack([item.gradient for item in evaluations], (10,)),
        "optimizer_evaluation_trace_json": _unicode([_json_text(item) for item in result["evaluation_miss_trace"]]),
        "per_image_joint_evaluations": np.asarray(result["per_image_joint_evaluations"], dtype=np.int64),
        "nof_category_names": _unicode(EXPECTED_NOF_CATEGORIES),
        "nof_category_counts": np.asarray([breakdown[name] for name in EXPECTED_NOF_CATEGORIES], dtype=np.int64),
        "blocked_uncached_misses": np.asarray(observer.ledger.blocked_misses, dtype=np.int64),
        "main_event_json": _unicode([_json_text(item) for item in result["main_audits"]]),
        "recovery_event_json": _unicode([_json_text(item) for item in result["recovery_audits"]]),
        "recovery_state_nf": np.asarray([item["nf"] for item in recovery_states], dtype=np.int64),
        "recovery_anchor_path": _stack([item["anchor_path"] for item in recovery_states], (total_images, 10)),
        "recovery_candidate_path": _stack([item["candidate_path"] for item in recovery_states], (total_images, 10)),
        "recovery_public_before_path": _stack([item["public_path_before"] for item in recovery_states], (total_images, 10)),
        "reparam_event_id": np.asarray([item.event_id for item in reparams], dtype=np.int64),
        "reparam_caller": _unicode([item.caller for item in reparams]),
        "reparam_outer": np.asarray([item.outer for item in reparams], dtype=np.int64),
        "reparam_nof_at_entry": np.asarray([item.nof_at_entry for item in reparams], dtype=np.int64),
        "reparam_input_path": _stack([item.input_path for item in reparams], (total_images, 10)),
        "reparam_fixed_energy": _stack([item.fixed_energy for item in reparams], (total_images,)),
        "reparam_output_path": _stack([item.output_path for item in reparams], (total_images, 10)),
        "reparam_iterations": np.asarray([item.iterations for item in reparams], dtype=np.int64),
        "reparam_final_residual": np.asarray([item.final_residual for item in reparams], dtype=np.float64),
        "reparam_tangent_call_start": np.asarray([item.tangent_call_start for item in reparams], dtype=np.int64),
        "reparam_tangent_call_count": np.asarray([item.tangent_call_count for item in reparams], dtype=np.int64),
        "tangent_call_count": np.asarray(len(tangents), dtype=np.int64),
        "tangent_mismatch_count": np.asarray(observer.tangent_mismatch_count, dtype=np.int64),
        "tangent_reparam_call_count": np.asarray(sum(int(item.tangent_call_count) for item in reparams), dtype=np.int64),
        "tangent_event_id": np.asarray([item.event_id for item in tangents], dtype=np.int64),
        "tangent_parent_reparam_event_id": np.asarray([item.parent_reparam_event_id for item in tangents], dtype=np.int64),
        "tangent_context_names": _unicode(contexts),
        "tangent_context_code": tangent_code,
        "tangent_context_counts": tangent_counts,
        "tangent_path_sha256": _digest_matrix(tangents, "path_sha256"),
        "tangent_energy_sha256": _digest_matrix(tangents, "energy_sha256"),
        "tangent_output_sha256": _digest_matrix(tangents, "output_sha256"),
        "tangent_transcript_sha256": _unicode_scalar(_tangent_transcript_sha256(tangents)),
        "memory_image_index": np.asarray(memory_images, dtype=np.int64),
        "memory_pair_order": np.asarray(memory_orders, dtype=np.int64),
        "memory_s": _stack(memory_s, (10,)),
        "memory_y": _stack(memory_y, (10,)),
    }
    return arrays
