#!/usr/bin/env python3
"""Isolated successor member for the signature-safe exact/no-reparam grid.

This adapter deliberately leaves the preregistered v1 member and its failed
artifacts untouched.  The scientific controls are inherited from v1, while
the runtime, cell identity, run labels, and artifact namespaces are v2-only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator

import run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim_v2 as runtime
import run_stage4_ala_discrete_exact_noreparam_alpha_grid_member as legacy


MEMBER_POLICY_VERSION = (
    "stage4_ala_discrete_exact_noreparam_alpha_grid_v2_call_signature"
)
SUPERSEDED_MEMBER_POLICY_VERSION = legacy.MEMBER_POLICY_VERSION
FAMILY = legacy.FAMILY
ALPHAS = legacy.ALPHAS
ALPHA_TEXT = legacy.ALPHA_TEXT
HOST_SHARDS = legacy.HOST_SHARDS
COMMON_INITIAL_ARRAY_SHA256 = legacy.COMMON_INITIAL_ARRAY_SHA256
TOTAL_INTERNAL_NF_BUDGET = runtime.TOTAL_INTERNAL_NF_BUDGET
PER_IMAGE_SAFETY_BUDGET = runtime.PER_IMAGE_SAFETY_BUDGET
RESIDUAL_RATIO = runtime.RESIDUAL_RATIO
INNER_MAX_STEPS = runtime.INNER_MAX_STEPS
ETA = runtime.ETA
HISTORY = runtime.HISTORY
KKT_ATOL = runtime.KKT_ATOL
PHYSICAL_MAX_STEP = runtime.PHYSICAL_MAX_STEP
REPLICATE = legacy.REPLICATE


def _require_alpha(value: Any) -> float:
    return legacy._require_alpha(value)


def alpha_text(value: Any) -> str:
    return legacy.alpha_text(value)


def alpha_tag(value: Any) -> str:
    return legacy.alpha_tag(value)


def assigned_host(value: Any, execution_host: Any = None) -> str:
    return legacy.assigned_host(value, execution_host)


def canonical_cell_spec(
    value: Any, execution_host: Any = None
) -> dict[str, object]:
    spec = dict(legacy.canonical_cell_spec(value, execution_host))
    spec.update(
        {
            "member_policy_version": MEMBER_POLICY_VERSION,
            "runtime_policy_id": runtime.POLICY_ID,
            "runtime_implementation_policy_id": runtime.IMPLEMENTATION_POLICY_ID,
            "successor_of_member_policy_version": (
                SUPERSEDED_MEMBER_POLICY_VERSION
            ),
        }
    )
    return spec


def cell_id(value: Any, execution_host: Any = None) -> str:
    return legacy.hashlib.sha256(
        json.dumps(
            canonical_cell_spec(value, execution_host),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]


def run_label(value: Any, execution_host: Any = None) -> str:
    alpha = _require_alpha(value)
    host = assigned_host(alpha, execution_host)
    return (
        f"{MEMBER_POLICY_VERSION}_{FAMILY}_{host}_{cell_id(alpha, host)}_"
        f"a{alpha_tag(alpha)}_r01_unbounded_NF{TOTAL_INTERNAL_NF_BUDGET}_"
        f"{REPLICATE}"
    )


def _replace_option(command: list[str], flag: str, value: str) -> None:
    positions = [index for index, token in enumerate(command) if token == flag]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise RuntimeError(f"expected exactly one valued {flag}")
    command[positions[0] + 1] = value


def compact_command_sha256(command: list[str]) -> str:
    return legacy.compact_command_sha256(command)


def single_run_command(
    root: Path, value: Any, execution_host: Any = None
) -> tuple[list[str], str]:
    root = root.expanduser().resolve()
    alpha = _require_alpha(value)
    host = assigned_host(alpha, execution_host)
    command, _ = legacy.single_run_command(root, alpha, host)
    command[1] = str(
        (
            root
            / "src/run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim_v2.py"
        ).resolve()
    )
    label = run_label(alpha, host)
    _replace_option(
        command,
        "--output-dir",
        f"runs/{MEMBER_POLICY_VERSION}/members/{host}",
    )
    _replace_option(command, "--run-label", label)
    assert_registered_command(command, root, alpha, host)
    return command, label


def assert_registered_command(
    command: list[str],
    root: Path,
    value: Any,
    execution_host: Any = None,
) -> None:
    root = root.expanduser().resolve()
    alpha = _require_alpha(value)
    host = assigned_host(alpha, execution_host)
    expected, _ = legacy.single_run_command(root, alpha, host)
    normalized = list(command)
    if len(normalized) != len(expected):
        raise RuntimeError("v2 exact/no-reparam command length drifted")
    expected_runtime = (
        root
        / "src/run_ivm_v9_discrete_ffrbk_takeover_exact_noreparam_shim_v2.py"
    ).resolve()
    if Path(normalized[1]).resolve() != expected_runtime:
        raise RuntimeError("v2 exact/no-reparam runtime path drifted")
    normalized[1] = expected[1]
    _replace_option(
        normalized,
        "--output-dir",
        f"runs/{legacy.MEMBER_POLICY_VERSION}/members/{host}",
    )
    _replace_option(normalized, "--run-label", legacy.run_label(alpha, host))
    if normalized != expected:
        raise RuntimeError("v2 changed a v1 scientific command token")
    options = legacy.command_helper.command_options(command)
    if options.get("--output-dir") != (
        f"runs/{MEMBER_POLICY_VERSION}/members/{host}"
    ) or options.get("--run-label") != run_label(alpha, host):
        raise RuntimeError("v2 member routing drifted")


def member_paths(
    root: Path, value: Any, execution_host: Any = None
) -> dict[str, Path | str | float]:
    root = root.expanduser().resolve()
    alpha = _require_alpha(value)
    host = assigned_host(alpha, execution_host)
    label = run_label(alpha, host)
    return {
        "alpha": alpha,
        "alpha_text": alpha_text(alpha),
        "assigned_host": host,
        "cell_id": cell_id(alpha, host),
        "label": label,
        "result_dir": root / "runs" / MEMBER_POLICY_VERSION / "members" / host,
        "audit": (
            root
            / "runs"
            / MEMBER_POLICY_VERSION
            / "audits"
            / host
            / f"{label}.json"
        ),
        "member_log": (
            root / "logs" / MEMBER_POLICY_VERSION / "members" / host / label
        ),
        "worker_log": (
            root / "logs" / MEMBER_POLICY_VERSION / "workers" / host / label
        ),
    }


def expected_result_glob(value: Any, execution_host: Any = None) -> str:
    return (
        f"ivm_{run_label(value, execution_host)}_central_state_ivm_cfg*_"
        f"budget{PER_IMAGE_SAFETY_BUDGET}.npz"
    )


@contextmanager
def _v2_validation_context() -> Iterator[None]:
    previous = legacy.runtime
    legacy.runtime = runtime
    try:
        yield
    finally:
        legacy.runtime = previous


def validate_result(
    result: Path, value: Any, execution_host: Any = None
) -> dict[str, object]:
    with _v2_validation_context():
        summary = legacy.validate_result(result, value, execution_host)
    implementation_id = _validate_v2_implementation_identity(result)
    summary.update(
        {
            "discrete_implementation_id": implementation_id,
            "runtime_v2_implementation_policy_verified": True,
        }
    )
    return summary


def _validate_v2_implementation_identity(result: Path) -> str:
    with legacy.np.load(result.expanduser().resolve(), allow_pickle=False) as saved:
        if "discrete_implementation_id" not in saved.files:
            raise RuntimeError("v2 result lacks discrete implementation identity")
        implementation_id = str(
            legacy.np.asarray(saved["discrete_implementation_id"]).item()
        )
    if runtime.IMPLEMENTATION_POLICY_ID not in implementation_id.split("+"):
        raise RuntimeError(
            "result implementation ID omits the v2 completion-audit policy"
        )
    return implementation_id


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    legacy._write_json_exclusive(path, payload)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", required=True)
    parser.add_argument("--execution-host", choices=HOST_SHARDS)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    alpha = _require_alpha(args.alpha)
    host = assigned_host(alpha, args.execution_host)
    command, label = single_run_command(root, alpha, host)
    paths = member_paths(root, alpha, host)
    result_dir = Path(paths["result_dir"])
    audit_path = Path(paths["audit"])
    log_dir = Path(paths["member_log"])
    existing = sorted(result_dir.glob(expected_result_glob(alpha, host)))
    if existing or audit_path.exists() or log_dir.exists():
        target = (
            existing[0]
            if existing
            else audit_path
            if audit_path.exists()
            else log_dir
        )
        raise FileExistsError(
            f"refusing to overwrite v2 exact/no-reparam cell: {target}"
        )
    result_dir.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    wall_seconds = legacy.legacy.run_command(
        command, root, log_dir / "stdout.txt", log_dir / "stderr.txt"
    )
    result = legacy.legacy.find_result(
        result_dir, label, PER_IMAGE_SAFETY_BUDGET
    )
    summary = validate_result(result, alpha, host)
    payload: dict[str, object] = {
        "status": "complete",
        "member_policy_version": MEMBER_POLICY_VERSION,
        "superseded_member_policy_version": SUPERSEDED_MEMBER_POLICY_VERSION,
        "runtime_policy_id": runtime.POLICY_ID,
        "runtime_implementation_policy_id": runtime.IMPLEMENTATION_POLICY_ID,
        "family": FAMILY,
        "priority": "secondary_after_inexact_reparam_grid",
        "alpha": alpha,
        "alpha_text": alpha_text(alpha),
        "assigned_host": host,
        "default_assigned_host": assigned_host(alpha),
        "execution_host_override": args.execution_host,
        "cell_id": cell_id(alpha, host),
        "canonical_cell_spec": canonical_cell_spec(alpha, host),
        "residual_ratio": RESIDUAL_RATIO,
        "scientific_maxstep": "unbounded_subject_to_total_NF_budget",
        "implementation_safety_inner_cap": INNER_MAX_STEPS,
        "inner_cap_required_nonbinding": True,
        "physical_max_step": PHYSICAL_MAX_STEP,
        "total_internal_nf_budget": TOTAL_INTERNAL_NF_BUDGET,
        "per_image_runner_budget_is_safety_only": PER_IMAGE_SAFETY_BUDGET,
        "one_way_ffrbk_operator_lbfgs": True,
        "eta": ETA,
        "history": HISTORY,
        "kkt_atol": KKT_ATOL,
        "reparameterization_enabled": False,
        "energy_backtracking_enabled": False,
        "result": str(result),
        "result_sha256": legacy.legacy.file_sha256(result),
        "scientific_command_sha256": compact_command_sha256(command),
        "command": command,
        "wall_seconds": wall_seconds,
        **summary,
    }
    _write_json_exclusive(audit_path, payload)
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
