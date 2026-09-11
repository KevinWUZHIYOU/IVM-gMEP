#!/usr/bin/env python3
"""Run the frozen Ala2D V15 development screen without starting MD.

The command-line interface is deliberately safe by default: ``list`` only
validates and prints the 48 preregistered trajectories.  ``run-one`` requires
one exact parameter identifier and oracle repetition; ``run-all`` is the only
entry point that traverses the complete frozen order.  Both execution modes
write a campaign-level, write-once source/input/runtime manifest before any
trajectory artifact and rely on :func:`ala2d_irivm_v15.run_irivm` for atomic
public states and replaceable exact-resume checkpoints.

This module can instantiate only the two development oracles named in the
frozen plan.  It contains no restrained-MD execution path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy

from ala2d_development_oracles import CalibratedSyntheticNoiseOracle
from ala2d_irivm_v15 import (
    IMPLEMENTATION_ID,
    IRIVMControls,
    IRIVMError,
    N_INTERNAL,
    load_public_trajectory,
    run_irivm,
)
from ala2d_oracles import SurrogateOracle


ROOT = Path(__file__).resolve().parents[1]
PLAN_RELATIVE = Path("registrations/DEVELOPMENT_SCREEN_PLAN.json")
PLAN_PATH = ROOT / PLAN_RELATIVE
EXPECTED_PLAN_SHA256 = "38c9c8aee4974a692949ce5b665979cfba12453562c7d4e303d18cf49f4cbd4c"
EXPECTED_IMPLEMENTATION_ID = (
    "ala2d-riemannian-v15-development-one-sided-r-resolved-lbfgs"
)
EXPECTED_IMPLEMENTATION_SOURCE_SHA256 = (
    "07ff2d70216bae7d3f95997805835b652bc755ab8f16cb4e82a489ae69ba56dc"
)
EXECUTION_MANIFEST_NAME = "execution_manifest.json"
RUN_BINDING_NAME = "screen_run_binding.json"
PUBLIC_STATE_LEDGER_NAME = "public_state_ledger.json"
SCREEN_SCHEMA = "ala2d.irivm.v15.development_screen.execution.v1"
RUN_BINDING_SCHEMA = "ala2d.irivm.v15.development_screen.run_binding.v1"
LEDGER_SCHEMA = "ala2d.irivm.v15.development_screen.public_state_ledger.v1"

INITIAL_RELATIVE = Path("inputs/reference/MFEP_INITIAL_PATH_current_ivm_state0.npz")
INITIAL_PROVENANCE_RELATIVE = Path(
    "inputs/reference/MFEP_INITIAL_PATH_current_ivm_state0.provenance.json"
)
FES_RELATIVE = Path("inputs/reference/fes_XYZ.npz")
METRIC_BANK_RELATIVE = Path("inputs/reference/current_ivm.npz")
CALIBRATION_RELATIVE = Path(
    "inputs/noise_calibration/r1_v13_confirmed_block_noise.npz"
)
CALIBRATION_PROVENANCE_RELATIVE = Path(
    "inputs/noise_calibration/r1_v13_confirmed_block_noise.provenance.json"
)

FORMAL_PDF_SHA256 = {
    Path("Figures/ala2_2d_FreeEnergy/MFEP_IVM_vs_FENEB_bestpair_threepanel.pdf"):
        "ed57a7089730e760862ac435545d7e9fc8534601ee8c9008635b1a16d54b9932",
    Path("Figures/ala2_2d_FreeEnergy/MFEP_IVM_vs_FENEB_contour_fourpanel.pdf"):
        "e872be9fd9d4cea83df67f1f94fe04e6a06ff3d5093108a4afc07877f8b993e1",
}

SOURCE_RELATIVES = (
    Path("src/run_development_screen.py"),
    Path("src/ala2d_irivm_v15.py"),
    Path("src/ala2d_development_oracles.py"),
    Path("src/ala2d_oracles.py"),
    Path("src/ala2d_article_neb_reparam.py"),
    Path("src/riemannian_torus_geometry.py"),
)

SPECIFICATION_RELATIVES = (
    Path("reports/V15_STATE_MACHINE_DESIGN.md"),
    Path("reports/DEVELOPMENT_SCREEN_EXPLANATION.md"),
    Path("DEVELOPMENT_RATIONALE.md"),
)


class ScreenError(RuntimeError):
    """Fail-closed development-screen configuration or persistence error."""


@dataclass(frozen=True)
class ParameterCombination:
    parameter_id: str
    alpha_label: str
    controls: IRIVMControls


@dataclass(frozen=True)
class RunSpecification:
    ordinal: int
    oracle_kind: str
    master_seed: Optional[int]
    parameter: ParameterCombination

    @property
    def relative_directory(self) -> Path:
        if self.oracle_kind == "deterministic":
            return Path("deterministic") / self.parameter.parameter_id
        if self.oracle_kind == "synthetic":
            if self.master_seed is None:
                raise ScreenError("synthetic run lacks a fixed master seed")
            return (
                Path("synthetic")
                / f"seed_{self.master_seed}"
                / self.parameter.parameter_id
            )
        raise ScreenError(f"unsupported oracle kind {self.oracle_kind!r}")


@dataclass(frozen=True)
class FrozenPlan:
    raw: Mapping[str, object]
    sha256: str
    combinations: Tuple[ParameterCombination, ...]
    seeds: Tuple[int, ...]
    nof_budget: int
    checkpoint_spacing: int

    def run_specifications(self) -> Tuple[RunSpecification, ...]:
        rows: List[RunSpecification] = []
        ordinal = 0
        for parameter in self.combinations:
            rows.append(RunSpecification(ordinal, "deterministic", None, parameter))
            ordinal += 1
        for seed in self.seeds:
            for parameter in self.combinations:
                rows.append(RunSpecification(ordinal, "synthetic", seed, parameter))
                ordinal += 1
        return tuple(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _atomic_write_once_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically create JSON once, or require identical canonical content."""

    expected = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ScreenError(f"write-once JSON path is not a regular file: {path}")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        try:
            observed = path.read_bytes()
        except OSError as error:
            raise ScreenError(f"cannot read existing write-once file {path}") from error
        if observed != expected:
            raise ScreenError(f"existing write-once JSON differs from request: {path}")
        return
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(expected)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # A partial exclusive-create file is evidence of an interrupted first
        # writer.  Keep it so the next invocation fails closed rather than
        # silently claiming a clean first publication.
        raise


def _atomic_replace_json(path: Path, payload: Mapping[str, object]) -> None:
    encoded = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _require_mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ScreenError(f"{context} must be a JSON object")
    return value


def _require_exact(value: object, expected: object, context: str) -> None:
    if value != expected or type(value) is not type(expected):
        raise ScreenError(f"{context} differs from the frozen value {expected!r}")


def _decimal_equal(left: object, right: str) -> bool:
    try:
        return Decimal(str(left)) == Decimal(right)
    except InvalidOperation:
        return False


def _number_token(value: float) -> str:
    decimal = Decimal(str(value)).normalize()
    token = format(decimal, "g").lower()
    return token.replace("-", "m").replace("+", "").replace(".", "p")


def _alpha_label_token(label: str) -> str:
    try:
        value = Decimal(label)
    except InvalidOperation as error:
        raise ScreenError(f"invalid frozen alpha notation {label!r}") from error
    if not value.is_finite() or value <= 0:
        raise ScreenError(f"invalid frozen alpha notation {label!r}")
    return label.lower().replace("+", "").replace("-", "m").replace(".", "p")


def _parameter_id(alpha_label: str, r: float, m_max: int) -> str:
    return f"a{_alpha_label_token(alpha_label)}_r{_number_token(r)}_m{m_max}"


def load_frozen_plan(root: Path = ROOT) -> FrozenPlan:
    """Load and exhaustively validate the exact frozen development plan."""

    root = Path(root).resolve()
    path = root / PLAN_RELATIVE
    if not path.is_file():
        raise ScreenError(f"frozen development plan is missing: {path}")
    observed_hash = _sha256_file(path)
    if observed_hash != EXPECTED_PLAN_SHA256:
        raise ScreenError(
            "frozen development plan SHA256 differs from the runner binding: "
            f"expected {EXPECTED_PLAN_SHA256}, observed {observed_hash}"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScreenError("cannot parse the frozen development plan") from error
    raw = _require_mapping(raw, "development plan")

    _require_exact(raw.get("schema_version"), "1.1.0", "schema_version")
    _require_exact(
        raw.get("plan_id"),
        "ala2d-irivm-v15-development-screen-r1",
        "plan_id",
    )
    _require_exact(
        raw.get("status"),
        "frozen-before-first-v15-grid-result",
        "status",
    )
    _require_exact(raw.get("formal_evidence"), False, "formal_evidence")
    _require_exact(
        raw.get("new_restrained_md_authorized"),
        False,
        "new_restrained_md_authorized",
    )

    algorithm = _require_mapping(raw.get("algorithm"), "algorithm")
    required_prefix = algorithm.get("required_implementation_prefix")
    _require_exact(required_prefix, "ala2d-riemannian-v15", "implementation prefix")
    _require_exact(
        algorithm.get("required_implementation_sha256"),
        EXPECTED_IMPLEMENTATION_SOURCE_SHA256,
        "implementation source SHA256",
    )
    if IMPLEMENTATION_ID != EXPECTED_IMPLEMENTATION_ID:
        raise ScreenError(
            f"core implementation {IMPLEMENTATION_ID!r} differs from the frozen "
            f"V15 implementation {EXPECTED_IMPLEMENTATION_ID!r}"
        )
    if not IMPLEMENTATION_ID.startswith(str(required_prefix)):
        raise ScreenError("frozen V15 implementation violates its required prefix")
    implementation_path = root / "src/ala2d_irivm_v15.py"
    if (
        not implementation_path.is_file()
        or implementation_path.is_symlink()
        or _sha256_file(implementation_path) != EXPECTED_IMPLEMENTATION_SOURCE_SHA256
    ):
        raise ScreenError(
            "V15 implementation source bytes differ from the frozen plan binding"
        )
    _require_exact(
        algorithm.get("mathematical_description"),
        "frozen-frame projected proximal root solver on article-reparameterized public paths",
        "mathematical description",
    )
    _require_exact(
        algorithm.get("design_specification"),
        "reports/V15_STATE_MACHINE_DESIGN.md",
        "design specification",
    )
    _require_exact(
        algorithm.get("optimizer_may_use_article_performance_metrics"),
        False,
        "optimizer access to article metrics",
    )
    _require_exact(
        algorithm.get("unselected_candidate_oracle_queries"),
        False,
        "unselected candidate queries",
    )
    _require_exact(
        algorithm.get("exact_center_resampling_is_public"),
        True,
        "public resampling policy",
    )
    _require_exact(
        algorithm.get("resampling_reuses_identical_path_bytes"),
        True,
        "exact-center resampling policy",
    )
    _require_exact(
        algorithm.get("relative_residual_stop_policy"),
        "After at least one movement, stop the frozen inner solve only when the one-sided confidence condition U(P_j) <= r L(P_0) is true. If it is false, including interval overlap or any other unresolved stop decision, continue with an admissible movement until a clear stop or m_max; do not purchase an exact-center resample merely to resolve r.",
        "one-sided relative-residual stop policy",
    )
    _require_exact(
        algorithm.get("unresolved_lbfgs_advantage_policy"),
        "Before global warm takeover, select L-BFGS only when its positive score advantage over H0 is statistically resolved. Otherwise H0 is the required fallback: move with H0 only when H0 is valid and its own improvement lower confidence bound is strictly positive; if that H0 lower bound is nonpositive with positive reducible uncertainty, HOLD is allowed. The nonselected L-BFGS candidate's own lower bound cannot bypass an unresolved takeover comparison.",
        "unresolved L-BFGS advantage policy",
    )
    _require_exact(
        algorithm.get("post_takeover_policy"),
        "The one-way warm takeover and its bounded warm scalar safeguard are unchanged; after takeover H0 remains forbidden. An invalid warm candidate with positive reducible uncertainty selects exact-center HOLD, while an invalid warm candidate without reducible uncertainty terminates without another query.",
        "post-takeover policy",
    )

    revision = _require_mapping(raw.get("revision_rationale"), "revision_rationale")
    _require_exact(
        revision.get("prior_plan_id"),
        "ala2d-irivm-v14-development-screen-r1",
        "prior plan identifier",
    )
    _require_exact(
        revision.get("prior_plan_sha256"),
        "2399bfa55445374644d72faec09c29a6b645b31b9fd081faddbc90022a527dd8",
        "prior plan SHA256",
    )
    _require_exact(
        revision.get("prior_analysis_sha256"),
        "fcbb44d0a90f0b4975de7196ff9664e6a5469d3f0fa447b25e82d7817e815c98",
        "prior analysis SHA256",
    )
    _require_exact(
        revision.get("prior_complete_exact_run_count"),
        48,
        "prior complete exact run count",
    )
    _require_exact(
        revision.get("prior_decision"),
        "NO_COMBINATION_PASSED",
        "prior decision",
    )

    grid = _require_mapping(raw.get("parameter_grid"), "parameter_grid")
    alpha_values = grid.get("alpha")
    alpha_labels = grid.get("alpha_integer_mantissa_notation")
    r_values = grid.get("r")
    m_values = grid.get("m_max")
    _require_exact(alpha_values, [0.00010, 0.00013, 0.00016], "alpha grid")
    _require_exact(alpha_labels, ["10e-5", "13e-5", "16e-5"], "alpha notation")
    _require_exact(r_values, [0.4, 0.7], "r grid")
    _require_exact(m_values, [4, 8], "m_max grid")
    _require_exact(
        grid.get("complete_factorial_combination_count"),
        12,
        "factorial combination count",
    )
    for value, label in zip(alpha_values, alpha_labels):
        if not _decimal_equal(value, label):
            raise ScreenError(f"alpha value {value!r} does not match notation {label!r}")

    combinations: List[ParameterCombination] = []
    seen_ids = set()
    for alpha_index, r_value, m_value in itertools.product(
        range(len(alpha_values)), r_values, m_values
    ):
        controls = IRIVMControls(
            alpha=float(alpha_values[alpha_index]),
            r=float(r_value),
            m_max=int(m_value),
        ).validate()
        label = str(alpha_labels[alpha_index])
        parameter_id = _parameter_id(label, controls.r, controls.m_max)
        if parameter_id in seen_ids:
            raise ScreenError(f"parameter identifier collision: {parameter_id}")
        seen_ids.add(parameter_id)
        combinations.append(ParameterCombination(parameter_id, label, controls))
    if len(combinations) != 12:
        raise ScreenError("frozen grid did not expand to exactly 12 combinations")

    oracles = _require_mapping(raw.get("oracles"), "oracles")
    deterministic = _require_mapping(oracles.get("deterministic"), "deterministic oracle")
    _require_exact(
        deterministic.get("class"),
        "ala2d_oracles.SurrogateOracle",
        "deterministic oracle class",
    )
    _require_exact(deterministic.get("repetitions"), 1, "deterministic repetitions")
    _require_exact(
        deterministic.get("formal_evidence"), False, "deterministic evidence flag"
    )

    synthetic = _require_mapping(
        oracles.get("calibrated_synthetic_noise"), "synthetic oracle"
    )
    _require_exact(
        synthetic.get("class"),
        "ala2d_development_oracles.CalibratedSyntheticNoiseOracle",
        "synthetic oracle class",
    )
    _require_exact(synthetic.get("master_seeds"), [1401, 1402, 1403], "synthetic seeds")
    _require_exact(synthetic.get("neighbor_count"), 64, "synthetic neighbor_count")
    _require_exact(
        synthetic.get("neighbor_bandwidth_rad"), 0.2, "synthetic neighbor bandwidth"
    )
    _require_exact(
        synthetic.get("maximum_nearest_calibration_distance_rad"),
        0.6,
        "synthetic nearest support limit",
    )
    _require_exact(
        synthetic.get("maximum_farthest_neighbor_distance_rad"),
        0.7,
        "synthetic farthest-neighbor support limit",
    )
    _require_exact(synthetic.get("noise_scale"), 1.0, "synthetic noise scale")
    _require_exact(synthetic.get("metric_noise"), True, "synthetic metric noise")
    _require_exact(
        synthetic.get("calibration"),
        CALIBRATION_RELATIVE.as_posix(),
        "synthetic calibration path",
    )
    _require_exact(synthetic.get("formal_evidence"), False, "synthetic evidence flag")

    horizon = _require_mapping(raw.get("cost_horizon"), "cost_horizon")
    _require_exact(horizon.get("internal_images"), N_INTERNAL, "internal image count")
    _require_exact(
        horizon.get("query_equivalents_per_complete_band"),
        N_INTERNAL,
        "complete-band query cost",
    )
    _require_exact(
        horizon.get("incremental_query_equivalent_budget_per_run"),
        2000,
        "per-run budget",
    )
    _require_exact(horizon.get("checkpoint_spacing"), 20, "checkpoint spacing")
    _require_exact(
        horizon.get("common_initial_incremental_cost"), 0, "common initial cost"
    )
    _require_exact(
        horizon.get("development_query_equivalents_are_not_formal_nof"),
        True,
        "development NoF disclaimer",
    )

    posthoc = _require_mapping(raw.get("posthoc_scoring"), "posthoc_scoring")
    _require_exact(
        posthoc.get("algorithm_access_to_scoring_oracle"),
        False,
        "scoring-oracle access",
    )
    selection = _require_mapping(raw.get("selection_rule"), "selection_rule")
    _require_exact(
        selection.get("selection_is_not_formal_acceptance"),
        True,
        "selection evidence disclaimer",
    )
    prerequisites = raw.get("required_before_execution")
    if not isinstance(prerequisites, list) or len(prerequisites) != 3:
        raise ScreenError("required_before_execution must contain the three frozen gates")

    return FrozenPlan(
        raw=raw,
        sha256=observed_hash,
        combinations=tuple(combinations),
        seeds=tuple(int(seed) for seed in synthetic["master_seeds"]),
        nof_budget=int(horizon["incremental_query_equivalent_budget_per_run"]),
        checkpoint_spacing=int(horizon["checkpoint_spacing"]),
    )


def _validate_required_files(root: Path) -> None:
    required = (
        PLAN_RELATIVE,
        INITIAL_RELATIVE,
        INITIAL_PROVENANCE_RELATIVE,
        FES_RELATIVE,
        METRIC_BANK_RELATIVE,
        CALIBRATION_RELATIVE,
        CALIBRATION_PROVENANCE_RELATIVE,
    ) + SOURCE_RELATIVES + SPECIFICATION_RELATIVES
    for relative in required:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ScreenError(f"required regular, non-symlink file is missing: {path}")


def _validate_formal_pdfs(workspace: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for relative, expected in FORMAL_PDF_SHA256.items():
        path = workspace / relative
        if not path.is_file() or path.is_symlink():
            raise ScreenError(f"formal comparison PDF is missing: {path}")
        observed = _sha256_file(path)
        if observed != expected:
            raise ScreenError(
                f"formal comparison PDF changed before development acceptance: {path}; "
                f"expected {expected}, observed {observed}"
            )
        hashes[relative.as_posix()] = observed
    return hashes


def _load_initial(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial_path = root / INITIAL_RELATIVE
    provenance_path = root / INITIAL_PROVENANCE_RELATIVE
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScreenError("cannot parse causal-initial provenance") from error
    expected_output = provenance.get("output", {})
    if not isinstance(expected_output, dict) or expected_output.get("sha256") != _sha256_file(
        initial_path
    ):
        raise ScreenError("causal-initial archive differs from its provenance")
    historical = provenance.get("inputs", {}).get("historical_current_ivm", {})
    if not isinstance(historical, dict) or historical.get("sha256") != _sha256_file(
        root / METRIC_BANK_RELATIVE
    ):
        raise ScreenError("historical current-IVM archive differs from initial provenance")

    with np.load(initial_path, allow_pickle=False) as loaded:
        required_keys = {
            "full_path_init",
            "M_init",
            "F_init",
            "C1",
            "C2",
            "N",
            "schema_version",
            "source_state_index",
            "incremental_nof",
            "mean_force_vector_dimension",
            "source_current_ivm_sha256",
        }
        if not required_keys.issubset(loaded.files):
            missing = sorted(required_keys.difference(loaded.files))
            raise ScreenError(f"causal-initial archive is missing keys: {missing}")
        path = np.asarray(loaded["full_path_init"], dtype=np.float64)
        metric = np.asarray(loaded["M_init"], dtype=np.float64)
        gradient = np.asarray(loaded["F_init"], dtype=np.float64)
        c1 = np.asarray(loaded["C1"], dtype=np.float64)
        c2 = np.asarray(loaded["C2"], dtype=np.float64)
        scalar_checks = {
            "N": (int(np.asarray(loaded["N"]).item()), 20),
            "source_state_index": (
                int(np.asarray(loaded["source_state_index"]).item()),
                0,
            ),
            "incremental_nof": (int(np.asarray(loaded["incremental_nof"]).item()), 0),
            "mean_force_vector_dimension": (
                int(np.asarray(loaded["mean_force_vector_dimension"]).item()),
                2,
            ),
        }
        if str(np.asarray(loaded["schema_version"]).item()) != (
            "ala2d.causal_shared_initial.v1"
        ):
            raise ScreenError("unexpected causal-initial schema")
        if str(np.asarray(loaded["source_current_ivm_sha256"]).item()) != _sha256_file(
            root / METRIC_BANK_RELATIVE
        ):
            raise ScreenError("causal-initial embedded current-IVM hash differs")
    for name, (observed, expected) in scalar_checks.items():
        if observed != expected:
            raise ScreenError(f"causal-initial {name} is {observed}, expected {expected}")
    if path.shape != (22, 2) or metric.shape != (20, 2, 2) or gradient.shape != (20, 2):
        raise ScreenError("causal-initial arrays have unexpected shapes")
    if not all(np.all(np.isfinite(value)) for value in (path, metric, gradient)):
        raise ScreenError("causal-initial arrays contain non-finite values")
    if not np.array_equal(path[0], c1) or not np.array_equal(path[-1], c2):
        raise ScreenError("causal-initial endpoints differ from C1/C2")
    if float(np.min(np.linalg.eigvalsh(0.5 * (metric + metric.swapaxes(1, 2))))) <= 0.0:
        raise ScreenError("causal-initial metric is not positive definite")
    return path, metric, gradient


def _runtime_payload() -> Dict[str, object]:
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full_version": sys.version,
        "byteorder": sys.byteorder,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
    }


def _hashed_mapping(root: Path, relatives: Iterable[Path]) -> Dict[str, str]:
    return {relative.as_posix(): _sha256_file(root / relative) for relative in relatives}


def _execution_manifest_payload(
    *, root: Path, workspace: Path, screen_root: Path, plan: FrozenPlan
) -> Dict[str, object]:
    formal_hashes = _validate_formal_pdfs(workspace)
    return {
        "schema_version": SCREEN_SCHEMA,
        "plan_id": plan.raw["plan_id"],
        "plan_sha256": plan.sha256,
        "formal_evidence": False,
        "new_restrained_md_authorized": False,
        "query_cost_name": "development query equivalents, not formal NoF",
        "implementation_id": IMPLEMENTATION_ID,
        "run_order": [
            {
                "ordinal": spec.ordinal,
                "oracle_kind": spec.oracle_kind,
                "master_seed": spec.master_seed,
                "parameter_id": spec.parameter.parameter_id,
                "alpha": spec.parameter.controls.alpha,
                "alpha_integer_mantissa_notation": spec.parameter.alpha_label,
                "r": spec.parameter.controls.r,
                "m_max": spec.parameter.controls.m_max,
                "relative_directory": spec.relative_directory.as_posix(),
            }
            for spec in plan.run_specifications()
        ],
        "run_count": 48,
        "per_run_incremental_query_equivalent_budget": plan.nof_budget,
        "complete_band_query_equivalents": N_INTERNAL,
        "common_initial_incremental_cost": 0,
        "root": str(root.resolve()),
        "screen_root": str(screen_root.resolve()),
        "source_sha256": _hashed_mapping(root, SOURCE_RELATIVES),
        "specification_sha256": _hashed_mapping(root, SPECIFICATION_RELATIVES),
        "input_sha256": _hashed_mapping(
            root,
            (
                INITIAL_RELATIVE,
                INITIAL_PROVENANCE_RELATIVE,
                FES_RELATIVE,
                METRIC_BANK_RELATIVE,
                CALIBRATION_RELATIVE,
                CALIBRATION_PROVENANCE_RELATIVE,
            ),
        ),
        "formal_pdf_unchanged_sha256": formal_hashes,
        "runtime": _runtime_payload(),
        "persistence": {
            "campaign_manifest": "write-once canonical JSON",
            "run_binding": "write-once canonical JSON",
            "public_states": "atomic publish-once NPZ in V15 core",
            "controller_checkpoint": "atomic replaceable NPZ in V15 core",
            "resume": "deterministic replay validates every durable public state",
            "public_state_ledger": "atomic replaceable hash index",
        },
    }


def ensure_execution_manifest(
    *, root: Path, workspace: Path, screen_root: Path, plan: FrozenPlan
) -> Path:
    """Publish the campaign manifest before any trajectory result exists."""

    manifest_path = screen_root / EXECUTION_MANIFEST_NAME
    _validate_screen_root_layout(screen_root)
    if not manifest_path.exists():
        for relative in (Path("deterministic"), Path("synthetic")):
            existing = screen_root / relative
            if existing.exists() and (
                not existing.is_dir() or existing.is_symlink()
            ):
                raise ScreenError(f"development result family path is unsafe: {existing}")
            if existing.exists() and any(existing.iterdir()):
                raise ScreenError(
                    "development results exist without the required execution manifest"
                )
    payload = _execution_manifest_payload(
        root=root, workspace=workspace, screen_root=screen_root, plan=plan
    )
    _atomic_write_once_json(manifest_path, payload)
    return manifest_path


def _validate_screen_root_layout(screen_root: Path) -> None:
    """Reject root-level artifacts outside the frozen screen persistence layout."""

    screen_root = Path(screen_root)
    if not screen_root.exists():
        return
    if not screen_root.is_dir() or screen_root.is_symlink():
        raise ScreenError(
            f"development-screen root is not a regular directory: {screen_root}"
        )
    allowed = {EXECUTION_MANIFEST_NAME, "deterministic", "synthetic"}
    unexpected = sorted(entry.name for entry in screen_root.iterdir() if entry.name not in allowed)
    if unexpected:
        raise ScreenError(
            f"unexpected root-level development-screen artifacts: {unexpected}"
        )
    manifest_path = screen_root / EXECUTION_MANIFEST_NAME
    if manifest_path.exists() and (manifest_path.is_symlink() or not manifest_path.is_file()):
        raise ScreenError("development execution manifest path is unsafe")
    for name in ("deterministic", "synthetic"):
        family = screen_root / name
        if family.exists() and (family.is_symlink() or not family.is_dir()):
            raise ScreenError(f"development result family path is unsafe: {family}")


def _instantiate_oracle(root: Path, plan: FrozenPlan, spec: RunSpecification) -> object:
    common = {
        "fes_npz": root / FES_RELATIVE,
        "metric_bank_npz": root / METRIC_BANK_RELATIVE,
    }
    if spec.oracle_kind == "deterministic":
        oracle = SurrogateOracle(**common)
        expected_class = "ala2d_oracles.SurrogateOracle"
    elif spec.oracle_kind == "synthetic":
        if spec.master_seed not in plan.seeds:
            raise ScreenError("synthetic seed is not in the frozen plan")
        config = plan.raw["oracles"]["calibrated_synthetic_noise"]
        oracle = CalibratedSyntheticNoiseOracle(
            **common,
            calibration_npz=root / CALIBRATION_RELATIVE,
            calibration_provenance_json=root / CALIBRATION_PROVENANCE_RELATIVE,
            master_seed=int(spec.master_seed),
            neighbor_count=int(config["neighbor_count"]),
            neighbor_bandwidth_rad=float(config["neighbor_bandwidth_rad"]),
            maximum_nearest_calibration_distance_rad=float(
                config["maximum_nearest_calibration_distance_rad"]
            ),
            maximum_farthest_neighbor_distance_rad=float(
                config["maximum_farthest_neighbor_distance_rad"]
            ),
            noise_scale=float(config["noise_scale"]),
            metric_noise=bool(config["metric_noise"]),
        )
        expected_class = (
            "ala2d_development_oracles.CalibratedSyntheticNoiseOracle"
        )
    else:
        raise ScreenError(f"unsupported development oracle {spec.oracle_kind!r}")
    actual_class = f"{oracle.__class__.__module__}.{oracle.__class__.__name__}"
    if actual_class != expected_class:
        raise ScreenError(
            f"instantiated oracle class {actual_class!r}, expected {expected_class!r}"
        )
    if bool(getattr(oracle, "formal_evidence", True)):
        raise ScreenError("development runner refuses a formal or MD oracle")
    provenance = oracle.provenance()
    if provenance.get("formal_evidence") is not False:
        raise ScreenError("development oracle provenance lacks formal_evidence=false")
    return oracle


def _run_binding_payload(
    *, spec: RunSpecification, plan: FrozenPlan, manifest_path: Path
) -> Dict[str, object]:
    return {
        "schema_version": RUN_BINDING_SCHEMA,
        "plan_id": plan.raw["plan_id"],
        "plan_sha256": plan.sha256,
        "execution_manifest_sha256": _sha256_file(manifest_path),
        "implementation_id": IMPLEMENTATION_ID,
        "formal_evidence": False,
        "new_restrained_md_authorized": False,
        "oracle_kind": spec.oracle_kind,
        "master_seed": spec.master_seed,
        "ordinal_in_frozen_run_order": spec.ordinal,
        "parameter_id": spec.parameter.parameter_id,
        "alpha": spec.parameter.controls.alpha,
        "alpha_integer_mantissa_notation": spec.parameter.alpha_label,
        "r": spec.parameter.controls.r,
        "m_max": spec.parameter.controls.m_max,
        "incremental_query_equivalent_budget": plan.nof_budget,
        "complete_band_query_equivalents": N_INTERNAL,
        "relative_run_directory": spec.relative_directory.as_posix(),
    }


def _validate_existing_run_root(run_directory: Path, binding: Mapping[str, object]) -> None:
    allowed = {
        RUN_BINDING_NAME,
        "run_manifest.json",
        "controller_checkpoint.npz",
        "states",
        "status.json",
        "completion.json",
        PUBLIC_STATE_LEDGER_NAME,
    }
    if not run_directory.exists():
        return
    if not run_directory.is_dir() or run_directory.is_symlink():
        raise ScreenError(f"run path is not a regular directory: {run_directory}")
    entries = list(run_directory.iterdir())
    binding_path = run_directory / RUN_BINDING_NAME
    if entries and not binding_path.is_file():
        raise ScreenError(f"existing result lacks a run binding: {run_directory}")
    if binding_path.exists() and binding_path.read_bytes() != _canonical_json_bytes(binding):
        raise ScreenError(f"existing run binding differs: {binding_path}")
    unexpected = sorted(entry.name for entry in entries if entry.name not in allowed)
    if unexpected:
        raise ScreenError(f"unexpected artifacts in {run_directory}: {unexpected}")
    expected_regular_files = {
        RUN_BINDING_NAME,
        "run_manifest.json",
        "controller_checkpoint.npz",
        "status.json",
        "completion.json",
        PUBLIC_STATE_LEDGER_NAME,
    }
    unsafe_files = sorted(
        entry.name
        for entry in entries
        if entry.name in expected_regular_files
        and (entry.is_symlink() or not entry.is_file())
    )
    if unsafe_files:
        raise ScreenError(f"unsafe run artifacts in {run_directory}: {unsafe_files}")
    states = run_directory / "states"
    if states.exists():
        if not states.is_dir() or states.is_symlink():
            raise ScreenError("public states path is not a regular directory")
        bad_states = sorted(
            child.name
            for child in states.iterdir()
            if child.is_symlink()
            or not child.is_file()
            or not child.name.startswith("state_")
            or child.suffix != ".npz"
        )
        if bad_states:
            raise ScreenError(f"unexpected public-state artifacts: {bad_states}")


def _public_state_ledger(
    *, run_directory: Path, spec: RunSpecification, manifest_path: Path
) -> Dict[str, object]:
    trajectory = load_public_trajectory(run_directory)
    state_paths = sorted((run_directory / "states").glob("state_*.npz"))
    if len(state_paths) != int(trajectory["state_index"].size):
        raise ScreenError("validated public trajectory and state-file count differ")
    records = []
    for array_index, state_path in enumerate(state_paths):
        state_index = int(trajectory["state_index"][array_index])
        nof = int(trajectory["nof_mean_force_incremental"][array_index])
        records.append(
            {
                "state_index": state_index,
                "incremental_query_equivalents": nof,
                "relative_file": f"states/{state_path.name}",
                "sha256": _sha256_file(state_path),
            }
        )
    return {
        "schema_version": LEDGER_SCHEMA,
        "formal_evidence": False,
        "implementation_id": IMPLEMENTATION_ID,
        "execution_manifest_sha256": _sha256_file(manifest_path),
        "run_binding_sha256": _sha256_file(run_directory / RUN_BINDING_NAME),
        "core_run_manifest_sha256": _sha256_file(run_directory / "run_manifest.json"),
        "oracle_kind": spec.oracle_kind,
        "master_seed": spec.master_seed,
        "parameter_id": spec.parameter.parameter_id,
        "state_count_including_common_initial": len(records),
        "last_incremental_query_equivalents": records[-1][
            "incremental_query_equivalents"
        ],
        "states": records,
    }


@contextmanager
def _exclusive_screen_lock(screen_root: Path):
    """Prevent concurrent writers to any run in one development screen."""

    identity = hashlib.sha256(str(screen_root.resolve()).encode("utf-8")).hexdigest()[:20]
    lock_path = Path(tempfile.gettempdir()) / f"ala2d-v15-development-{identity}.lock"
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ScreenError(
                f"another process holds the host-local development-screen lock: {lock_path}"
            ) from error
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _execute_specification_unlocked(
    *,
    spec: RunSpecification,
    plan: FrozenPlan,
    stop_after_nof: int,
    root: Path = ROOT,
    workspace: Optional[Path] = None,
    screen_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Execute or exactly resume one preregistered development trajectory."""

    root = Path(root).resolve()
    workspace = root.parents[1] if workspace is None else Path(workspace).resolve()
    screen_root = (
        root / "runs/development_screen"
        if screen_root is None
        else Path(screen_root).resolve()
    )
    if spec not in plan.run_specifications():
        raise ScreenError("run specification is not in the frozen run order")
    if (
        type(stop_after_nof) is not int
        or stop_after_nof < 0
        or stop_after_nof > plan.nof_budget
        or stop_after_nof % plan.checkpoint_spacing
    ):
        raise ScreenError("stop_after_nof must be a multiple of 20 in [0, 2000]")

    _validate_required_files(root)
    initial_path, initial_M, initial_gradient = _load_initial(root)
    manifest_path = ensure_execution_manifest(
        root=root, workspace=workspace, screen_root=screen_root, plan=plan
    )
    run_directory = screen_root / spec.relative_directory
    resolved_parent = run_directory.parent.resolve()
    if screen_root.resolve() != resolved_parent and screen_root.resolve() not in resolved_parent.parents:
        raise ScreenError("resolved run directory escapes the development-screen root")
    binding = _run_binding_payload(spec=spec, plan=plan, manifest_path=manifest_path)
    _validate_existing_run_root(run_directory, binding)
    run_directory.mkdir(parents=True, exist_ok=True)
    _atomic_write_once_json(run_directory / RUN_BINDING_NAME, binding)

    before_nof = 0
    if (run_directory / "states").is_dir():
        before = load_public_trajectory(run_directory)
        before_nof = int(before["nof_mean_force_incremental"][-1])
    oracle = _instantiate_oracle(root, plan, spec)
    status = run_irivm(
        oracle=oracle,
        controls=spec.parameter.controls,
        initial_path=initial_path,
        initial_M=initial_M,
        initial_grad_F=initial_gradient,
        nof_budget=plan.nof_budget,
        run_directory=run_directory,
        resume=True,
        stop_after_nof=stop_after_nof,
    )
    after_nof = int(status["nof_mean_force_incremental"])
    expected_new_queries = after_nof - before_nof
    if expected_new_queries < 0 or int(getattr(oracle, "call_count", -1)) != expected_new_queries:
        raise ScreenError(
            "oracle query ledger differs from newly published complete bands: "
            f"oracle={getattr(oracle, 'call_count', None)}, expected={expected_new_queries}"
        )
    ledger = _public_state_ledger(
        run_directory=run_directory, spec=spec, manifest_path=manifest_path
    )
    _atomic_replace_json(run_directory / PUBLIC_STATE_LEDGER_NAME, ledger)
    return {
        "ordinal": spec.ordinal,
        "oracle_kind": spec.oracle_kind,
        "master_seed": spec.master_seed,
        "parameter_id": spec.parameter.parameter_id,
        "run_directory": str(run_directory),
        "before_incremental_query_equivalents": before_nof,
        "after_incremental_query_equivalents": after_nof,
        "new_query_equivalents_this_invocation": expected_new_queries,
        "completed": bool(status["completed"]),
        "paused_at_requested_stop": bool(status["paused_at_requested_stop"]),
        "algorithmic_zero_nof_stop": status["algorithmic_zero_nof_stop"],
        "public_state_ledger_sha256": _sha256_file(
            run_directory / PUBLIC_STATE_LEDGER_NAME
        ),
    }


def execute_specification(
    *,
    spec: RunSpecification,
    plan: FrozenPlan,
    stop_after_nof: int,
    root: Path = ROOT,
    workspace: Optional[Path] = None,
    screen_root: Optional[Path] = None,
) -> Dict[str, object]:
    """Execute one run under an exclusive host-local screen writer lock."""

    resolved_root = Path(root).resolve()
    resolved_screen_root = (
        resolved_root / "runs/development_screen"
        if screen_root is None
        else Path(screen_root).resolve()
    )
    with _exclusive_screen_lock(resolved_screen_root):
        result = _execute_specification_unlocked(
            spec=spec,
            plan=plan,
            stop_after_nof=stop_after_nof,
            root=resolved_root,
            workspace=workspace,
            screen_root=resolved_screen_root,
        )
        # Recompute every pre-execution binding before reporting success.  A
        # concurrent source/input/PDF change therefore leaves an explicitly
        # failed invocation rather than an apparently valid mixed-code result.
        ensure_execution_manifest(
            root=resolved_root,
            workspace=(
                resolved_root.parents[1]
                if workspace is None
                else Path(workspace).resolve()
            ),
            screen_root=resolved_screen_root,
            plan=plan,
        )
        return result


def select_specification(
    plan: FrozenPlan,
    *,
    oracle_kind: str,
    parameter_id: str,
    master_seed: Optional[int],
) -> RunSpecification:
    matches = [
        spec
        for spec in plan.run_specifications()
        if spec.oracle_kind == oracle_kind
        and spec.parameter.parameter_id == parameter_id
        and spec.master_seed == master_seed
    ]
    if len(matches) != 1:
        raise ScreenError(
            "requested oracle/seed/parameter tuple is not one frozen run specification"
        )
    return matches[0]


def list_payload(plan: FrozenPlan) -> Dict[str, object]:
    return {
        "mode": "validation-and-list-only",
        "plan_id": plan.raw["plan_id"],
        "plan_sha256": plan.sha256,
        "formal_evidence": False,
        "new_restrained_md_authorized": False,
        "combination_count": len(plan.combinations),
        "run_count": len(plan.run_specifications()),
        "per_run_incremental_query_equivalent_budget": plan.nof_budget,
        "parameter_ids_in_frozen_order": [
            parameter.parameter_id for parameter in plan.combinations
        ],
        "runs_in_frozen_order": [
            {
                "ordinal": spec.ordinal,
                "oracle_kind": spec.oracle_kind,
                "master_seed": spec.master_seed,
                "parameter_id": spec.parameter.parameter_id,
                "relative_directory": spec.relative_directory.as_posix(),
            }
            for spec in plan.run_specifications()
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "list", help="validate the frozen plan and print all runs; write nothing"
    )
    one = subparsers.add_parser("run-one", help="run or resume one exact frozen tuple")
    one.add_argument("--oracle", choices=("deterministic", "synthetic"), required=True)
    one.add_argument("--parameter-id", required=True)
    one.add_argument("--seed", type=int)
    one.add_argument("--stop-after", type=int, default=2000)
    all_parser = subparsers.add_parser(
        "run-all", help="sequentially run all 48 trajectories in frozen order"
    )
    all_parser.add_argument("--stop-after", type=int, default=2000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    plan = load_frozen_plan(ROOT)
    _validate_required_files(ROOT)
    # ``ROOT`` is ``<workspace>/code_on_server/<campaign>``.
    workspace = ROOT.parents[1]
    _validate_formal_pdfs(workspace)
    if arguments.command == "list":
        print(json.dumps(list_payload(plan), indent=2, sort_keys=True))
        return 0
    if arguments.command == "run-one":
        if arguments.oracle == "deterministic" and arguments.seed is not None:
            raise ScreenError("deterministic run-one forbids --seed")
        if arguments.oracle == "synthetic" and arguments.seed is None:
            raise ScreenError("synthetic run-one requires one frozen --seed")
        spec = select_specification(
            plan,
            oracle_kind=arguments.oracle,
            parameter_id=arguments.parameter_id,
            master_seed=arguments.seed,
        )
        result = execute_specification(
            spec=spec,
            plan=plan,
            stop_after_nof=arguments.stop_after,
            root=ROOT,
            workspace=workspace,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if arguments.command == "run-all":
        results = []
        for spec in plan.run_specifications():
            result = execute_specification(
                spec=spec,
                plan=plan,
                stop_after_nof=arguments.stop_after,
                root=ROOT,
                workspace=workspace,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        print(
            json.dumps(
                {
                    "completed_invocation_count": len(results),
                    "frozen_run_count": len(plan.run_specifications()),
                    "stop_after": arguments.stop_after,
                },
                sort_keys=True,
            )
        )
        return 0
    raise ScreenError(f"unsupported command {arguments.command!r}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ScreenError, IRIVMError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
