#!/usr/bin/env python3
"""Corrected frame-bound output observer for the R3 full-state replay.

The production ``record`` closure owns counters/history but deliberately does
not close over energies or forces.  Its immediate caller (``run`` or
``attempt_reparameterization``) owns the current path, energies, forces, and
diagnostic.  R3 resolves those code objects prospectively and accepts exactly
that two-frame relationship.  It then delegates durable storage to the frozen
R2 shard writer, whose storage implementation itself was not the R2 failure.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from types import CodeType

from full_state_history_recorder import (
    FullStateHistoryRecorder,
    require,
    sha256_file,
)


EXPECTED_PRODUCTION_SHA256 = (
    "991c11ff1f8eab014a6e54b4bf8ba610b0e52d071b202dc131b03bfabe6143fa"
)
ALLOWED_STATE_FRAME_NAMES = frozenset({"run", "attempt_reparameterization"})


@dataclass(frozen=True)
class FrameCodeContract:
    source_path: Path
    source_sha256: str
    record_code: CodeType
    allowed_state_codes: frozenset[CodeType]


def nested_codes(code: CodeType):
    yield code
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            yield from nested_codes(constant)


def freeze_frame_code_contract(
    runner_module,
    expected_source_sha256: str = EXPECTED_PRODUCTION_SHA256,
) -> FrameCodeContract:
    source_path = Path(runner_module.__file__).expanduser().resolve(strict=True)
    observed_sha256 = sha256_file(source_path)
    require(
        observed_sha256 == str(expected_source_sha256),
        "production runner source hash differs from the prospective frame contract",
    )
    run_function = getattr(runner_module, "run", None)
    require(callable(run_function) and isinstance(run_function.__code__, CodeType), "runner lacks run code")
    codes = list(nested_codes(run_function.__code__))
    record_codes = [code for code in codes if code.co_name == "record"]
    require(len(record_codes) == 1, "production runner must contain exactly one record closure")
    state_codes = frozenset(
        code for code in codes if code.co_name in ALLOWED_STATE_FRAME_NAMES
    )
    require(
        {code.co_name for code in state_codes} == ALLOWED_STATE_FRAME_NAMES,
        "production runner state-frame inventory drift",
    )
    require(
        all(Path(code.co_filename).expanduser().resolve() == source_path for code in (record_codes[0], *state_codes)),
        "frame code does not originate from the hash-bound production source",
    )
    return FrameCodeContract(
        source_path=source_path,
        source_sha256=observed_sha256,
        record_code=record_codes[0],
        allowed_state_codes=state_codes,
    )


def install_output_observer(
    runner_module,
    recorder: FullStateHistoryRecorder,
    *,
    expected_source_sha256: str = EXPECTED_PRODUCTION_SHA256,
):
    """Install a zero-query observer and return an idempotent restore call."""

    contract = freeze_frame_code_contract(runner_module, expected_source_sha256)
    original = runner_module.should_store_path
    require(
        getattr(original, "__module__", None) == "storage_policy"
        and getattr(original, "__name__", None) == "should_store_path",
        "unexpected original storage-policy callable",
    )

    def observed_should_store_path(
        *, mode, save_all_paths, call, is_initial, reparam_committed
    ):
        current_frame = inspect.currentframe()
        record_frame = current_frame.f_back if current_frame is not None else None
        state_frame = record_frame.f_back if record_frame is not None else None
        try:
            require(
                record_frame is not None
                and record_frame.f_code is contract.record_code,
                "observer was not called by the exact hash-bound record closure",
            )
            require(
                state_frame is not None
                and state_frame.f_code in contract.allowed_state_codes,
                "record was not called by an allowed hash-bound state frame",
            )
            record_local = record_frame.f_locals
            state_local = state_frame.f_locals
            require(
                all(
                    name in record_local
                    for name in ("path", "counters", "history", "diagnostic")
                ),
                "record frame lacks its expected current-state closure values",
            )
            require(
                all(
                    name in state_local
                    for name in ("path", "energies", "forces", "diagnostic")
                ),
                "state frame lacks current path/energy/full-force values",
            )
            require(
                record_local["path"] is state_local["path"]
                and record_local["diagnostic"] is state_local["diagnostic"],
                "record/state frames do not refer to the same current path and diagnostic",
            )
            recorder.record(
                call,
                state_local["path"],
                state_local["energies"],
                state_local["forces"],
                record_local["counters"],
                record_local["history"],
                state_local["diagnostic"],
            )
            return original(
                mode=mode,
                save_all_paths=save_all_paths,
                call=call,
                is_initial=is_initial,
                reparam_committed=reparam_committed,
            )
        finally:
            del current_frame, record_frame, state_frame

    runner_module.should_store_path = observed_should_store_path

    def restore() -> None:
        if runner_module.should_store_path is observed_should_store_path:
            runner_module.should_store_path = original

    return restore
