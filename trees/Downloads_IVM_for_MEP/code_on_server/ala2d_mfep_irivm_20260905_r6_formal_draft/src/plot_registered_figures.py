"""Fail-closed construction and rendering of the registered Ala2D figures.

The module deliberately separates evidence binding from rendering.  Calling
``build_registered_plot_data`` performs all registration, acceptance-report,
run-binding, trajectory, and reference-input checks without importing
matplotlib or writing a file.  Rendering happens only through
``render_registered_figures`` or the command-line entry point.

The command defaults to a campaign-local preview directory.  Formal PDFs are
eligible for atomic replacement only when ``--formal`` is explicit, the final
acceptance report says ``pass: true``, and every registered gate is satisfied.
No molecular dynamics is started here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from ala2d_irivm_v17 import (
    IMPLEMENTATION_ID,
    N_INTERNAL,
    load_public_trajectory,
)
from formal_contract import (
    ACCEPTANCE_SCHEMA,
    EXECUTABLE_REGISTRATION_SCHEMA,
    FORMAL_INCREMENTAL_NOF_BUDGET,
    PAID_STATE_KINDS,
    SEALED_STATUS,
    FormalContractError,
    accepted_report_schema,
    assert_registration_is_not_executable,
    validate_public_trajectory,
)
from riemannian_torus_geometry import compute_article_metrics, wrap_periodic


Array = np.ndarray
PathLike = Union[str, Path]
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MODULE_PATH = Path(__file__).resolve()
CAMPAIGN_ROOT = MODULE_PATH.parents[1]
DEFAULT_REGISTRATION = CAMPAIGN_ROOT / "registrations/V17_FORMAL_CAMPAIGN_DRAFT.json"
DEFAULT_ACCEPTANCE_REPORT = CAMPAIGN_ROOT / "reports/V17_FORMAL_ACCEPTANCE.json"
DEFAULT_PREVIEW_DIRECTORY = CAMPAIGN_ROOT / "figures/v17_formal_preview"
CAUSAL_COMMON_SCHEMA = "ala2d.causal_shared_initial.v1"
CAUSAL_COMMON_RELATIVE_PATH = (
    "inputs/reference/MFEP_INITIAL_PATH_current_ivm_state0.npz"
)
CAUSAL_PROVENANCE_RELATIVE_PATH = (
    "inputs/reference/MFEP_INITIAL_PATH_current_ivm_state0.provenance.json"
)
LEGACY_COMMON_RELATIVE_PATH = "inputs/reference/MFEP_INITIAL_PATH_common.npz"
HISTORICAL_CURRENT_RELATIVE_PATH = "inputs/reference/current_ivm.npz"
PLOTTING_SOURCE_RELATIVE_PATH = "src/plot_registered_figures.py"
CAUSAL_BUILDER_RELATIVE_PATH = "src/build_causal_shared_initial.py"
MEAN_FORCE_AXIS_LABEL = "Number of mean-force evaluations"
MEAN_FORCE_NOF_UNIT = (
    "One restrained-MD evaluation at one restraint center that jointly returns "
    "the full two-component mean-force vector counts as one NoF."
)
CONTOUR_NOF = np.asarray([0, 200, 400, 2000], dtype=np.int64)
RUN_MANIFEST_NOF_DEFINITION = (
    "one estimate of the complete two-component mean-force vector at one "
    "internal collective-variable point equals one NoF"
)


class PlotInputError(RuntimeError):
    """Raised when plotting evidence is missing, ambiguous, or unregistered."""


def sha256_file(path: PathLike) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> Dict[str, object]:
    if not path.is_file():
        raise PlotInputError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PlotInputError(f"cannot read valid JSON from {label}: {path}") from error
    if not isinstance(value, dict):
        raise PlotInputError(f"{label} root must be a JSON object")
    return value


def _mapping(parent: Mapping[str, object], key: str, label: str = "registration") -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise PlotInputError(f"{label}.{key} must be an object")
    return value


def _integer(parent: Mapping[str, object], key: str, label: str = "registration") -> int:
    value = parent.get(key)
    if type(value) is not int:
        raise PlotInputError(f"{label}.{key} must be an integer")
    return int(value)


def _number(parent: Mapping[str, object], key: str, label: str = "registration") -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlotInputError(f"{label}.{key} must be numerical")
    result = float(value)
    if not np.isfinite(result):
        raise PlotInputError(f"{label}.{key} must be finite")
    return result


def _boolean(parent: Mapping[str, object], key: str, label: str = "registration") -> bool:
    value = parent.get(key)
    if type(value) is not bool:
        raise PlotInputError(f"{label}.{key} must be boolean")
    return bool(value)


def _text(parent: Mapping[str, object], key: str, label: str = "registration") -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise PlotInputError(f"{label}.{key} must be nonempty text")
    return value


def _inside(root: Path, candidate: Path, label: str) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise PlotInputError(f"{label} escapes the campaign root: {candidate}") from error
    return candidate


def _campaign_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise PlotInputError(f"{label} must be a nonempty path")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return _inside(root, path, label)


def _validate_hash_table(
    root: Path,
    table: Mapping[str, object],
    label: str,
) -> Dict[str, str]:
    if not table:
        raise PlotInputError(f"registration hashes.{label} cannot be empty")
    verified: Dict[str, str] = {}
    for raw_path, raw_digest in sorted(table.items()):
        if not isinstance(raw_path, str) or not raw_path:
            raise PlotInputError(f"registration hashes.{label} contains an invalid path")
        if not isinstance(raw_digest, str) or SHA256_PATTERN.fullmatch(raw_digest) is None:
            raise PlotInputError(
                f"registration hashes.{label}[{raw_path!r}] is not a sealed SHA-256 digest"
            )
        path = _campaign_path(root, raw_path, f"hashes.{label}[{raw_path!r}]")
        if not path.is_file():
            raise PlotInputError(f"registered {label} file is missing: {path}")
        observed = sha256_file(path)
        if observed != raw_digest:
            raise PlotInputError(
                f"registered {label} hash mismatch for {raw_path}: "
                f"expected {raw_digest}, observed {observed}"
            )
        verified[str(path)] = observed
    return verified


def load_sealed_registration(path: PathLike) -> Tuple[Dict[str, object], Path, Dict[str, str]]:
    """Load a sealed registration and verify all registered source/input files.

    Returns ``(registration, campaign_root, verified_file_hashes)``.  Runtime
    binaries are not reopened here because the plotting process does not use
    them; their sealed identities remain preserved in the registration and in
    the run-level provenance.
    """

    registration_path = Path(path).resolve()
    registration = _load_json(registration_path, "campaign registration")
    root = registration_path.parents[1]
    # This is deliberately the first semantic gate.  In particular, a checked-in
    # DRAFT must fail here, before source/input probing and before any output
    # directory can be created by the caller.
    try:
        assert_registration_is_not_executable(registration)
    except FormalContractError as error:
        raise PlotInputError(str(error)) from error
    if registration.get("schema_version") != EXECUTABLE_REGISTRATION_SCHEMA:
        raise PlotInputError("registration does not use the executable V17 schema")
    if registration.get("status") != SEALED_STATUS:
        raise PlotInputError("registration was not sealed before the first formal MD call")
    candidate = _mapping(registration, "candidate")
    if candidate.get("implementation_id") != IMPLEMENTATION_ID:
        raise PlotInputError("registration selects a different implementation")
    hashes = _mapping(registration, "hashes")
    if hashes.get("algorithm") != "sha256":
        raise PlotInputError("registration hashes.algorithm must be 'sha256'")
    verified = {str(registration_path): sha256_file(registration_path)}
    verified.update(_validate_hash_table(root, _mapping(hashes, "source", "hashes"), "source"))
    verified.update(_validate_hash_table(root, _mapping(hashes, "input", "hashes"), "input"))
    source_hashes = _mapping(hashes, "source", "hashes")
    plotting_digest = source_hashes.get(PLOTTING_SOURCE_RELATIVE_PATH)
    if not isinstance(plotting_digest, str) or plotting_digest != sha256_file(MODULE_PATH):
        raise PlotInputError(
            "the executing plotting source is not the source sealed in the registration"
        )
    return registration, root, verified


def _require_registered_input(
    root: Path,
    registration: Mapping[str, object],
    path: Path,
    label: str,
) -> str:
    """Require a consumed reference file to be in the sealed input table."""

    relative = path.resolve().relative_to(root.resolve()).as_posix()
    inputs = _mapping(_mapping(registration, "hashes"), "input", "hashes")
    expected = inputs.get(relative)
    if not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected) is None:
        raise PlotInputError(f"{label} is not sealed as a registered input: {relative}")
    observed = sha256_file(path)
    if observed != expected:
        raise PlotInputError(
            f"registered input hash mismatch for {label}: expected {expected}, "
            f"observed {observed}"
        )
    return observed


def _registered_seeds(registration: Mapping[str, object]) -> Tuple[int, ...]:
    values = _mapping(registration, "seeds").get("optimization")
    if (
        not isinstance(values, list)
        or len(values) != 3
        or any(type(value) is not int for value in values)
    ):
        raise PlotInputError("registration.seeds.optimization must contain three integers")
    seeds = tuple(int(value) for value in values)
    if len(set(seeds)) != len(seeds):
        raise PlotInputError("registration.seeds.optimization contains duplicates")
    forbidden = {
        int(item)
        for group in _mapping(registration, "seeds").get("forbidden_prior_seed_sets", [])
        if isinstance(group, list)
        for item in group
        if type(item) is int
    }
    reused = sorted(forbidden.intersection(seeds))
    if reused:
        raise PlotInputError(f"formal optimization seeds reuse prior seeds: {reused}")
    return seeds


def _single_registered_seed(registration: Mapping[str, object], key: str) -> int:
    value = _mapping(registration, "seeds").get(key)
    if type(value) is int:
        return int(value)
    if isinstance(value, list) and len(value) == 1 and type(value[0]) is int:
        return int(value[0])
    raise PlotInputError(f"registration.seeds.{key} must identify exactly one seed")


def _validate_nof_contract(registration: Mapping[str, object]) -> Mapping[str, object]:
    """Reject any axis that is not the registered mean-force-evaluation count."""

    nof = _mapping(registration, "nof")
    expected_integers = {
        "incremental_budget_per_optimization_run": FORMAL_INCREMENTAL_NOF_BUDGET,
        "complete_band": N_INTERNAL,
        "mean_force_vector_dimension": 2,
        "common_initial_incremental_cost": 0,
        "shared_initial_absolute_cost": N_INTERNAL,
    }
    for key, expected in expected_integers.items():
        if _integer(nof, key, "nof") != expected:
            raise PlotInputError(
                f"nof.{key} must be {expected} mean-force evaluations"
            )
    expected_booleans = {
        "repeated_center_is_charged_again": True,
        "hold_and_validation_bands_are_charged": True,
        "zero_nof_refresh_creates_public_state": False,
        "hidden_candidate_or_recovery_evaluations": False,
    }
    for key, expected in expected_booleans.items():
        if _boolean(nof, key, "nof") is not expected:
            raise PlotInputError(f"nof.{key} differs from the registered NoF semantics")
    unit = _text(nof, "unit", "nof")
    if "complete two-component mean-force vector" not in unit:
        raise PlotInputError("nof.unit is not the registered vector mean-force cost")
    return nof


def _validate_referenced_report(
    root: Path,
    final_report: Mapping[str, object],
    *,
    path_key: str,
    digest_key: str,
    stage: str,
    registration_digest: str,
    winner: str,
) -> Tuple[Path, Dict[str, object]]:
    report_path = _campaign_path(root, final_report.get(path_key), f"acceptance.{path_key}")
    expected_digest = final_report.get(digest_key)
    if not isinstance(expected_digest, str) or SHA256_PATTERN.fullmatch(expected_digest) is None:
        raise PlotInputError(f"acceptance.{digest_key} is not a SHA-256 digest")
    if not report_path.is_file():
        raise PlotInputError(f"referenced {stage} report is missing: {report_path}")
    observed_digest = sha256_file(report_path)
    if observed_digest != expected_digest:
        raise PlotInputError(f"referenced report hash mismatch: {report_path}")
    referenced = _load_json(report_path, f"referenced {stage} report")
    if referenced.get("stage") != stage:
        raise PlotInputError(f"referenced report has the wrong stage: {report_path}")
    if referenced.get("registration_sha256") != registration_digest:
        raise PlotInputError(f"referenced report is bound to another registration: {report_path}")
    if referenced.get("winner_parameter_id") != winner:
        raise PlotInputError(f"referenced report selects another parameter: {report_path}")
    return report_path, referenced


def validate_acceptance_report(
    path: PathLike,
    registration: Mapping[str, object],
    registration_path: PathLike,
    campaign_root: PathLike,
) -> Tuple[Dict[str, object], str, Dict[int, Path], Dict[str, str]]:
    """Bind a V17 full-pass acceptance report to its prospectively sealed runs."""

    report_path = Path(path).resolve()
    report = _load_json(report_path, "final acceptance report")
    registration_digest = sha256_file(registration_path)
    if report.get("registration_sha256") != registration_digest:
        raise PlotInputError("final acceptance report is bound to another registration")
    if report.get("stage") != "acceptance":
        raise PlotInputError("final acceptance report stage is not 'acceptance'")
    if report.get("pass") is not True:
        raise PlotInputError("final acceptance report does not record pass: true")
    if report.get("decision") != "pass":
        raise PlotInputError("final acceptance report decision is not 'pass'")
    if report.get("figure_replacement_authorized") is not True:
        raise PlotInputError("final acceptance report does not authorize figure replacement")
    if report.get("figure_files_modified_by_this_stage") is not False:
        raise PlotInputError("acceptance stage did not preserve the formal figure inputs")
    if (
        report.get(
            "terminal_artifacts_revalidated_without_new_mean_force_evaluations"
        )
        is not True
    ):
        raise PlotInputError(
            "final acceptance did not revalidate terminal artifacts without new evaluations"
        )
    analysis = report.get("acceptance_analysis")
    if not isinstance(analysis, Mapping):
        raise PlotInputError("final acceptance report lacks acceptance_analysis")
    try:
        accepted_report_schema(analysis)
    except FormalContractError as error:
        raise PlotInputError(
            f"acceptance analysis must use {ACCEPTANCE_SCHEMA}: {error}"
        ) from error
    if analysis.get("campaign_id") != registration.get("campaign_id"):
        raise PlotInputError("acceptance analysis is bound to another campaign")
    if analysis.get("decision") != "pass" or analysis.get("pass") is not True:
        raise PlotInputError("nested acceptance analysis is not a full pass")
    # ``pass: true`` is necessary but not sufficient for a formal figure.  The
    # registered complete rule names the front/middle history, noisy tail,
    # correct-path gate, and terminal paired-MD test.  Require the corresponding
    # machine-readable sections to agree with the top-level decision.
    cost_history = analysis.get("cost_history")
    correct_path = analysis.get("correct_path")
    terminal = analysis.get("terminal_noise_test")
    if not isinstance(cost_history, Mapping) or cost_history.get("pass") is not True:
        raise PlotInputError("final acceptance report lacks a passing cost_history section")
    for section_name in ("front_middle", "tail"):
        section = cost_history.get(section_name)
        if not isinstance(section, Mapping) or section.get("pass") is not True:
            raise PlotInputError(
                f"final acceptance report lacks passing cost_history.{section_name}"
            )
    if not isinstance(correct_path, Mapping) or correct_path.get("pass") is not True:
        raise PlotInputError("final acceptance report lacks a passing correct_path section")
    if (
        not isinstance(terminal, Mapping)
        or terminal.get("pass") is not True
        or terminal.get("decision") != "pass"
    ):
        raise PlotInputError(
            "final acceptance report lacks a passing terminal_noise_test section"
        )
    selected = _mapping(_mapping(registration, "candidate"), "selected_parameter", "candidate")
    winner = _text(selected, "parameter_id", "candidate.selected_parameter")
    reported_winner = report.get("winner_parameter_id")
    if reported_winner != winner:
        raise PlotInputError("acceptance report does not select the registered candidate")
    if analysis.get("candidate_parameter_id") != winner:
        raise PlotInputError("acceptance analysis does not select the registered candidate")
    supplied_analysis_seeds = analysis.get("optimization_seeds")
    seeds = _registered_seeds(registration)
    if (
        not isinstance(supplied_analysis_seeds, list)
        or tuple(supplied_analysis_seeds) != seeds
    ):
        raise PlotInputError("acceptance analysis top-level optimization seeds differ")
    supplied_seeds = cost_history.get("optimization_seeds")
    if not isinstance(supplied_seeds, list) or tuple(supplied_seeds) != seeds:
        raise PlotInputError("acceptance analysis optimization seed order differs")
    root = Path(campaign_root).resolve()
    execution = _mapping(registration, "execution")
    run_root = _campaign_path(root, execution.get("run_root"), "execution.run_root")
    runs: Dict[int, Path] = {}
    for seed in seeds:
        runs[seed] = _inside(
            root,
            run_root / "optimization" / winner / f"seed_{seed}",
            f"selected optimization run for seed {seed}",
        )
    # Optional stage reports, when present, must remain cryptographically bound.
    # The acceptance schema itself is the producer/consumer contract; there is
    # deliberately no pilot-selection report and no privileged first seed.
    referenced_hashes: Dict[str, str] = {}
    for path_key, digest_key, stage in (
        ("precheck_report", "precheck_report_sha256", "acceptance-precheck"),
        (
            "terminal_diagnostics_report",
            "terminal_diagnostics_report_sha256",
            "terminal-diagnostics",
        ),
    ):
        if path_key not in report and digest_key not in report:
            continue
        referenced_path, referenced = _validate_referenced_report(
            root,
            report,
            path_key=path_key,
            digest_key=digest_key,
            stage=stage,
            registration_digest=registration_digest,
            winner=winner,
        )
        stage_result = (
            referenced.get("terminal_noise_test")
            if stage == "terminal-diagnostics"
            else referenced
        )
        if not isinstance(stage_result, Mapping) or (
            stage_result.get("pass", stage_result.get("precheck_pass")) is not True
        ):
            raise PlotInputError(f"referenced {stage} report did not pass")
        referenced_hashes[str(referenced_path)] = sha256_file(referenced_path)
    return report, winner, runs, referenced_hashes


def _load_npz(path: Path, label: str) -> Dict[str, Array]:
    if not path.is_file():
        raise PlotInputError(f"{label} is missing: {path}")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {key: np.array(loaded[key], copy=True) for key in loaded.files}
    except (OSError, ValueError, KeyError) as error:
        raise PlotInputError(f"cannot read {label}: {path}") from error


def _require_array(
    record: Mapping[str, Array],
    key: str,
    label: str,
    *,
    ndim: Optional[int] = None,
    finite: bool = True,
) -> Array:
    if key not in record:
        raise PlotInputError(f"{label} lacks {key}")
    value = np.asarray(record[key])
    if ndim is not None and value.ndim != ndim:
        raise PlotInputError(f"{label}.{key} must have dimension {ndim}")
    if finite and not np.all(np.isfinite(value)):
        raise PlotInputError(f"{label}.{key} contains nonfinite values")
    return value


def _npz_scalar(
    record: Mapping[str, Array],
    key: str,
    label: str,
) -> object:
    value = _require_array(record, key, label, finite=False)
    if value.shape != ():
        raise PlotInputError(f"{label}.{key} must be a scalar")
    return value.item()


def _validate_causal_common_initial(
    root: Path,
    registration: Mapping[str, object],
    paths: Mapping[str, object],
) -> Tuple[Dict[str, Array], Dict[str, str]]:
    """Bind the comparison origin to the actual historical current-IVM state 0."""

    required_paths = {
        "common_initial": CAUSAL_COMMON_RELATIVE_PATH,
        "common_initial_provenance": CAUSAL_PROVENANCE_RELATIVE_PATH,
        "legacy_common_initial": LEGACY_COMMON_RELATIVE_PATH,
        "historical_current_ivm": HISTORICAL_CURRENT_RELATIVE_PATH,
    }
    resolved: Dict[str, Path] = {}
    hashes: Dict[str, str] = {}
    for key, expected_relative in required_paths.items():
        path = _campaign_path(root, paths.get(key), f"paths.{key}")
        observed_relative = path.relative_to(root).as_posix()
        if observed_relative != expected_relative:
            raise PlotInputError(
                f"paths.{key} must be {expected_relative}, observed {observed_relative}"
            )
        resolved[key] = path
        hashes[str(path)] = _require_registered_input(
            root, registration, path, f"paths.{key}"
        )
    if resolved["common_initial"] == resolved["legacy_common_initial"]:
        raise PlotInputError("the causal common initial cannot be the legacy common file")

    common = _load_npz(resolved["common_initial"], "causal common initial")
    legacy = _load_npz(resolved["legacy_common_initial"], "legacy common initial")
    current = _load_npz(resolved["historical_current_ivm"], "historical current IVM")
    provenance = _load_json(
        resolved["common_initial_provenance"], "causal common provenance"
    )
    if _npz_scalar(common, "schema_version", "causal common initial") != CAUSAL_COMMON_SCHEMA:
        raise PlotInputError("common initial is not the registered causal-state schema")
    if provenance.get("schema_version") != CAUSAL_COMMON_SCHEMA:
        raise PlotInputError("causal common provenance has the wrong schema")

    if int(_npz_scalar(common, "source_state_index", "causal common initial")) != 0:
        raise PlotInputError("causal common initial is not historical state 0")
    if int(_npz_scalar(common, "historical_initial_nof", "causal common initial")) != 20:
        raise PlotInputError("causal common initial must record 20 historical NoF")
    if int(_npz_scalar(common, "incremental_nof", "causal common initial")) != 0:
        raise PlotInputError("causal common initial must be incremental NoF 0")
    if int(
        _npz_scalar(common, "mean_force_vector_dimension", "causal common initial")
    ) != 2:
        raise PlotInputError("one Ala2D mean-force evaluation must return two components")

    current_required = (
        "phi_ref_iter_0",
        "M_ref_iter_0",
        "F_ref_iter_0",
        "arc_ref_iter_0",
        "path_iter_0",
        "phi_ref_iter_1",
    )
    for key in current_required:
        _require_array(current, key, "historical current IVM")
    if "g_ref_iter_0" in current and not np.array_equal(
        current["F_ref_iter_0"], current["g_ref_iter_0"]
    ):
        raise PlotInputError("historical state-0 F and g arrays differ")
    if not np.array_equal(current["path_iter_0"], current["phi_ref_iter_1"]):
        raise PlotInputError("historical path_iter_0 is not the causal state-1 path")

    C1 = np.asarray(_require_array(common, "C1", "causal common initial", ndim=1))
    C2 = np.asarray(_require_array(common, "C2", "causal common initial", ndim=1))
    phi = np.asarray(
        _require_array(common, "phi_init", "causal common initial", ndim=2)
    )
    M = np.asarray(_require_array(common, "M_init", "causal common initial", ndim=3))
    grad_F = np.asarray(
        _require_array(common, "F_init", "causal common initial", ndim=2)
    )
    arc = np.asarray(
        _require_array(common, "arc_init", "causal common initial", ndim=1)
    )
    full_path = np.asarray(
        _require_array(common, "full_path_init", "causal common initial", ndim=2)
    )
    exact_bindings = {
        "phi_init": (phi, current["phi_ref_iter_0"]),
        "M_init": (M, current["M_ref_iter_0"]),
        "F_init": (grad_F, current["F_ref_iter_0"]),
        "arc_init": (arc, current["arc_ref_iter_0"]),
    }
    for key, (observed, expected) in exact_bindings.items():
        if not np.array_equal(observed, expected):
            raise PlotInputError(
                f"causal common {key} is not exactly historical current-IVM state 0"
            )
    constructed_path = np.vstack([C1, current["phi_ref_iter_0"], C2])
    if not np.array_equal(full_path, constructed_path):
        raise PlotInputError("causal common full path is not C1/phi_ref_iter_0/C2")
    for endpoint in ("C1", "C2"):
        if not np.array_equal(common[endpoint], legacy[endpoint]):
            raise PlotInputError(f"causal common {endpoint} differs from the fixed endpoint")
    legacy_path = np.asarray(
        _require_array(legacy, "full_path_init", "legacy common initial", ndim=2)
    )
    if legacy_path.shape != full_path.shape or np.max(np.abs(legacy_path - full_path)) > 1e-12:
        raise PlotInputError("legacy geometry does not cross-check the causal state-0 path")

    recomputed = compute_article_metrics(full_path, M, grad_F)
    for key in ("I_perp_G", "I_arc", "Delta_F_barrier", "Delta_F_final"):
        stored = float(_npz_scalar(common, key, "causal common initial"))
        if not np.isclose(stored, float(recomputed[key]), rtol=1e-12, atol=1e-12):
            raise PlotInputError(f"causal common stored {key} does not recompute")
    profile = np.asarray(
        _require_array(
            common, "free_energy_profile", "causal common initial", ndim=1
        ),
        dtype=np.float64,
    )
    if not np.allclose(
        profile, recomputed["free_energy_profile"], rtol=1e-12, atol=1e-12
    ):
        raise PlotInputError("causal common free-energy profile does not recompute")

    current_sha = hashes[str(resolved["historical_current_ivm"])]
    legacy_sha = hashes[str(resolved["legacy_common_initial"])]
    common_sha = hashes[str(resolved["common_initial"])]
    if _npz_scalar(common, "source_current_ivm_sha256", "causal common initial") != current_sha:
        raise PlotInputError("causal common embeds the wrong current-IVM hash")
    if _npz_scalar(common, "source_legacy_common_sha256", "causal common initial") != legacy_sha:
        raise PlotInputError("causal common embeds the wrong legacy-common hash")
    source_hashes = _mapping(_mapping(registration, "hashes"), "source", "hashes")
    builder_sha = source_hashes.get(CAUSAL_BUILDER_RELATIVE_PATH)
    if (
        not isinstance(builder_sha, str)
        or _npz_scalar(common, "generator_sha256", "causal common initial")
        != builder_sha
    ):
        raise PlotInputError("causal common was not built by the sealed generator")

    output_provenance = provenance.get("output")
    provenance_inputs = provenance.get("inputs")
    nof_provenance = provenance.get("nof")
    if not isinstance(output_provenance, Mapping) or (
        output_provenance.get("file") != resolved["common_initial"].name
        or output_provenance.get("sha256") != common_sha
    ):
        raise PlotInputError("causal common provenance does not bind its output")
    if not isinstance(provenance_inputs, Mapping):
        raise PlotInputError("causal common provenance lacks input hashes")
    provenance_current = provenance_inputs.get("historical_current_ivm")
    provenance_legacy = provenance_inputs.get("legacy_common_initial")
    if (
        not isinstance(provenance_current, Mapping)
        or provenance_current.get("sha256") != current_sha
        or not isinstance(provenance_legacy, Mapping)
        or provenance_legacy.get("sha256") != legacy_sha
    ):
        raise PlotInputError("causal common provenance input hashes differ")
    if (
        not isinstance(nof_provenance, Mapping)
        or nof_provenance.get("name") != "number of mean-force evaluations"
        or nof_provenance.get("historical_initial_absolute_nof") != 20
        or nof_provenance.get("shared_initial_incremental_nof") != 0
        or nof_provenance.get("next_historical_state_incremental_nof") != 20
    ):
        raise PlotInputError("causal common provenance has false NoF semantics")
    return common, hashes


def _validate_v17_causal_common_initial(
    root: Path,
    registration: Mapping[str, object],
    inputs: Mapping[str, object],
) -> Tuple[Dict[str, Array], Dict[str, str]]:
    """Validate the two sealed inputs that establish the causal NoF=0 row."""

    common_path = _campaign_path(
        root, inputs.get("causal_shared_initial"), "inputs.causal_shared_initial"
    )
    current_path = _campaign_path(
        root, inputs.get("historical_current_ivm"), "inputs.historical_current_ivm"
    )
    hashes = {
        str(common_path): _require_registered_input(
            root, registration, common_path, "inputs.causal_shared_initial"
        ),
        str(current_path): _require_registered_input(
            root, registration, current_path, "inputs.historical_current_ivm"
        ),
    }
    common = _load_npz(common_path, "causal common initial")
    current = _load_npz(current_path, "historical current IVM")
    if _npz_scalar(common, "schema_version", "causal common initial") != CAUSAL_COMMON_SCHEMA:
        raise PlotInputError("common initial is not the causal-state schema")
    if int(_npz_scalar(common, "source_state_index", "causal common initial")) != 0:
        raise PlotInputError("causal common initial is not historical current-IVM state zero")
    if int(_npz_scalar(common, "incremental_nof", "causal common initial")) != 0:
        raise PlotInputError("causal common initial does not lie at incremental NoF zero")
    if int(_npz_scalar(common, "historical_initial_nof", "causal common initial")) != N_INTERNAL:
        raise PlotInputError("causal common initial has the wrong absolute historical cost")
    exact_bindings = {
        "phi_init": "phi_ref_iter_0",
        "M_init": "M_ref_iter_0",
        "F_init": "F_ref_iter_0",
        "arc_init": "arc_ref_iter_0",
    }
    for common_key, current_key in exact_bindings.items():
        observed = _require_array(common, common_key, "causal common initial")
        expected = _require_array(current, current_key, "historical current IVM")
        if not np.array_equal(observed, expected):
            raise PlotInputError(
                f"causal common {common_key} differs from historical current-IVM state zero"
            )
    full_path = np.asarray(
        _require_array(common, "full_path_init", "causal common initial", ndim=2),
        dtype=np.float64,
    )
    if full_path.shape != (N_INTERNAL + 2, 2):
        raise PlotInputError("causal common initial has the wrong Ala2D path shape")
    M = np.asarray(_require_array(common, "M_init", "causal common initial", ndim=3))
    grad = np.asarray(_require_array(common, "F_init", "causal common initial", ndim=2))
    metrics = compute_article_metrics(full_path, M, grad)
    for key in ("I_perp_G", "I_arc", "Delta_F_barrier"):
        stored = float(_npz_scalar(common, key, "causal common initial"))
        if not np.isclose(stored, float(metrics[key]), rtol=1e-12, atol=1e-12):
            raise PlotInputError(f"causal common {key} does not recompute")
    return common, hashes


def _validate_run_binding(
    run_directory: Path,
    registration_digest: str,
    winner_parameter_id: str,
    seed: int,
    registration: Mapping[str, object],
) -> Tuple[Path, Dict[str, object]]:
    if not run_directory.is_dir():
        raise PlotInputError(f"optimization run directory is missing for seed {seed}: {run_directory}")
    binding_path = run_directory / "registration_binding.json"
    binding = _load_json(binding_path, f"seed-{seed} registration binding")
    contract = _validate_algorithm_contract(registration)
    nof = _validate_nof_contract(registration)
    required = {
        "registration_sha256": registration_digest,
        "implementation_id": IMPLEMENTATION_ID,
        "oracle_kind": "restrained-md",
        "parameter_id": winner_parameter_id,
        "master_seed": seed,
        "nof_budget_mean_force_incremental": _integer(
            nof, "incremental_budget_per_optimization_run", "nof"
        ),
        "internal_images": N_INTERNAL,
        "complete_band_nof": N_INTERNAL,
        "common_initial_band_excluded_from_incremental_axis": True,
        "paid_state_kinds": list(PAID_STATE_KINDS),
        "final_path_selection": "exact public source path at incremental NoF 2000",
        "pending_next_action_is_not_committed_or_plotted": True,
        "hidden_candidate_or_recovery_evaluations": False,
        "jobs": 1,
    }
    for key, expected in required.items():
        if binding.get(key) != expected:
            raise PlotInputError(
                f"seed-{seed} registration binding field {key!r} differs: "
                f"expected {expected!r}, observed {binding.get(key)!r}"
            )
    controls = binding.get("controls")
    if not isinstance(controls, Mapping):
        raise PlotInputError(f"seed-{seed} registration binding lacks controls")
    selected = contract["selected_parameter"]
    for name in ("alpha", "r", "m_max"):
        if controls.get(name) != selected.get(name):
            raise PlotInputError(
                f"seed-{seed} control {name!r} differs from the selected candidate"
            )
    return binding_path, dict(binding)


def _validate_algorithm_contract(
    registration: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the sealed V17 candidate and its three-kind public-state contract."""

    candidate = _mapping(registration, "candidate")
    if candidate.get("implementation_id") != IMPLEMENTATION_ID:
        raise PlotInputError("candidate implementation identity differs from V17")
    selected = _mapping(candidate, "selected_parameter", "candidate")
    for name in ("parameter_id", "alpha", "r", "m_max"):
        if name not in selected:
            raise PlotInputError(f"candidate.selected_parameter lacks {name}")
    public = _mapping(registration, "public_state_contract")
    if public.get("paid_state_kinds") != list(PAID_STATE_KINDS):
        raise PlotInputError("registered paid states differ from the three V17 kinds")
    if public.get("acceptance_schema") != ACCEPTANCE_SCHEMA:
        raise PlotInputError("registration and plotter acceptance schemas differ")
    if public.get("terminal_pending_path_policy") != (
        "score_exact_public_source_path_at_2000_without_2020_commit_or_LOCF"
    ):
        raise PlotInputError("registration does not preserve the exact pending source path")
    return {"implementation_id": IMPLEMENTATION_ID, "selected_parameter": selected}


def _validate_run_manifest(
    run_directory: Path,
    binding: Mapping[str, object],
    seed: int,
    registration: Mapping[str, object],
) -> Path:
    manifest_path = run_directory / "run_manifest.json"
    manifest = _load_json(manifest_path, f"seed-{seed} run manifest")
    _validate_algorithm_contract(registration)
    if manifest.get("controls") != binding.get("controls"):
        raise PlotInputError(f"seed-{seed} manifest controls differ from its run binding")
    expected = {
        "implementation_id": IMPLEMENTATION_ID,
        "development_only": False,
        "nof_budget_mean_force_incremental": binding.get(
            "nof_budget_mean_force_incremental"
        ),
        "complete_20_image_band_cost_nof": N_INTERNAL,
        "common_initial_band_excluded_from_incremental_axis": True,
        "absolute_nof_additive_constant": N_INTERNAL,
        "internal_image_count": N_INTERNAL,
    }
    for key, expected_value in expected.items():
        if manifest.get(key) != expected_value:
            raise PlotInputError(
                f"seed-{seed} run manifest field {key!r} differs from the "
                "registered public-state semantics"
            )
    if manifest.get("oracle_formal_evidence") is not True:
        raise PlotInputError(f"seed-{seed} run is not formal restrained-MD evidence")
    if manifest.get("nof_definition") != RUN_MANIFEST_NOF_DEFINITION:
        raise PlotInputError(f"seed-{seed} run manifest has a false NoF definition")
    return manifest_path


def _load_public_plot_trajectory(
    run_directory: Path,
    manifest_path: Path,
) -> Dict[str, Array]:
    """Replay V17 public states and expose no private proposal trajectory.

    The core loader validates every durable public state and its exact replay.
    Plotting therefore never reads a candidate/proposal checkpoint and cannot
    carry an unpurchased next action forward onto the NoF=2000 point.
    """

    if not manifest_path.is_file():
        raise PlotInputError(f"run manifest is missing: {manifest_path}")
    try:
        loaded = load_public_trajectory(run_directory)
    except Exception as error:
        raise PlotInputError(f"V17 public-state replay failed: {run_directory}") from error
    return {key: value for key, value in loaded.items()}


def _validate_run_completion(
    run_directory: Path,
    manifest_path: Path,
    trajectory: Mapping[str, Array],
    seed: int,
) -> Path:
    completion_path = run_directory / "completion.json"
    completion = _load_json(completion_path, f"seed-{seed} completion report")
    manifest = _load_json(manifest_path, f"seed-{seed} run manifest")
    try:
        terminal = validate_public_trajectory(trajectory, completion=completion)
    except FormalContractError as error:
        raise PlotInputError(f"seed-{seed} completion/public prefix is invalid: {error}") from error
    state_count = int(np.asarray(trajectory["state_index"]).size)
    if completion.get("implementation_id") != manifest.get("implementation_id"):
        raise PlotInputError(f"seed-{seed} completion and manifest identities differ")
    if completion.get("run_directory") != str(run_directory.resolve()):
        raise PlotInputError(f"seed-{seed} completion binds another run directory")
    if completion.get("state_count_including_common_initial") != state_count:
        raise PlotInputError(f"seed-{seed} completion state count differs from trajectory")
    terminal_path = np.asarray(trajectory["full_path_unwrapped"])[terminal.index]
    if not np.array_equal(terminal_path, terminal.full_path_unwrapped):
        raise PlotInputError(f"seed-{seed} terminal source path changed")
    if terminal.pending_next_action and completion.get(
        "pending_validated_commit_requires_another_band"
    ) is not True:
        raise PlotInputError(f"seed-{seed} conceals the unpaid pending movement")
    return completion_path


def _validate_trajectory(
    trajectory: Mapping[str, Array],
    seed: int,
    registration: Mapping[str, object],
    common: Mapping[str, Array],
) -> Dict[str, Array]:
    nof_settings = _validate_nof_contract(registration)
    budget = _integer(
        nof_settings, "incremental_budget_per_optimization_run", "nof"
    )
    try:
        validate_public_trajectory(trajectory, expected_budget=budget)
    except FormalContractError as error:
        raise PlotInputError(f"seed-{seed} public V17 trajectory is invalid: {error}") from error
    expected_nof = np.arange(0, budget + N_INTERNAL, N_INTERNAL, dtype=np.int64)
    raw_nof = np.asarray(trajectory.get("nof_mean_force_incremental"))
    if raw_nof.ndim != 1 or not np.issubdtype(raw_nof.dtype, np.integer):
        raise PlotInputError(
            f"seed-{seed} public NoF values must be exact integer mean-force counts"
        )
    nof = raw_nof.astype(np.int64, copy=False)
    if not np.array_equal(nof, expected_nof):
        raise PlotInputError(
            f"seed-{seed} public trajectory must contain exact complete bands at "
            f"0..{budget} in steps of {N_INTERNAL} NoF"
        )
    states = np.asarray(trajectory.get("state_kind")).astype(str)
    output_residual = np.asarray(
        trajectory.get("selected_reparameterization_output_residual"),
        dtype=np.float64,
    )
    output_tolerance = _number(
        _mapping(_mapping(registration, "acceptance"), "correct_path_guard", "acceptance"),
        "selected_path_article_spring_output_residual_strictly_less_than",
        "acceptance.correct_path_guard",
    )
    if output_residual.shape != nof.shape or not np.all(np.isfinite(output_residual[1:])):
        raise PlotInputError(
            f"seed-{seed} paid-state reparameterization certificate is incomplete"
        )
    if not np.isnan(output_residual[0]):
        raise PlotInputError(
            f"seed-{seed} common initial must not invent a reparameterization certificate"
        )
    if np.any(output_residual[1:] < 0.0) or np.any(
        output_residual[1:] >= output_tolerance
    ):
        raise PlotInputError(
            f"seed-{seed} reparameterized public-state output residual must be "
            f"nonnegative and strictly below {output_tolerance:g}"
        )
    paths = np.asarray(trajectory.get("full_path_unwrapped"), dtype=np.float64)
    expected_shape = (
        nof.size,
        N_INTERNAL + 2,
        2,
    )
    if paths.shape != expected_shape or not np.all(np.isfinite(paths)):
        raise PlotInputError(
            f"seed-{seed} path history has shape {paths.shape}, expected {expected_shape}"
        )
    common_path = np.asarray(_require_array(common, "full_path_init", "common initial", ndim=2), dtype=np.float64)
    if not np.array_equal(paths[0], common_path):
        raise PlotInputError(f"seed-{seed} NoF=0 path is not the registered common initial path")
    M = np.asarray(trajectory.get("M"), dtype=np.float64)
    grad_F = np.asarray(trajectory.get("grad_F"), dtype=np.float64)
    internal_images = N_INTERNAL
    if M.shape != (nof.size, internal_images, 2, 2) or not np.all(np.isfinite(M)):
        raise PlotInputError(f"seed-{seed} metric history is incomplete or nonfinite")
    if grad_F.shape != (nof.size, internal_images, 2) or not np.all(np.isfinite(grad_F)):
        raise PlotInputError(f"seed-{seed} mean-force history is incomplete or nonfinite")
    if not np.array_equal(M[0], common["M_init"]):
        raise PlotInputError(f"seed-{seed} NoF=0 metric is not the causal common metric")
    if not np.array_equal(grad_F[0], common["F_init"]):
        raise PlotInputError(f"seed-{seed} NoF=0 mean force is not the causal common value")
    recomputed_rows = [
        compute_article_metrics(path, metric, force)
        for path, metric, force in zip(paths, M, grad_F)
    ]
    for key in ("I_perp_G", "I_arc", "Delta_F_barrier"):
        stored = np.asarray(trajectory.get(key), dtype=np.float64)
        expected = np.asarray([float(row[key]) for row in recomputed_rows])
        if stored.shape != nof.shape or not np.all(np.isfinite(stored)):
            raise PlotInputError(
                f"seed-{seed}.{key} must be one finite value per public band"
            )
        if not np.allclose(stored, expected, rtol=1e-11, atol=1e-12):
            raise PlotInputError(
                f"seed-{seed}.{key} does not recompute from its public path, "
                "metric, and mean force"
            )
    segment_lengths = np.linalg.norm(np.diff(paths, axis=1), axis=2)
    if np.any(segment_lengths <= 0.0):
        raise PlotInputError(f"seed-{seed} contains a zero-length public segment")
    expected_deviation = np.max(
        np.abs(segment_lengths / np.mean(segment_lengths, axis=1)[:, None] - 1.0),
        axis=1,
    )
    stored_deviation = np.asarray(
        trajectory.get("segment_length_relative_deviation"), dtype=np.float64
    )
    if stored_deviation.shape != nof.shape or not np.allclose(
        stored_deviation, expected_deviation, rtol=1e-11, atol=1e-12
    ):
        raise PlotInputError(
            f"seed-{seed}.segment_length_relative_deviation does not recompute"
        )
    result: Dict[str, Array] = {
        "nof_mean_force_incremental": nof,
        "state_kind": states.astype(str),
        "full_path_unwrapped": paths,
        "M": M,
        "grad_F": grad_F,
        "selected_reparameterization_output_residual": output_residual,
    }
    common_metric_keys = {
        "I_perp_G": "I_perp_G",
        "I_arc": "I_arc",
        "Delta_F_barrier": "Delta_F_barrier",
    }
    for key, common_key in common_metric_keys.items():
        values = np.asarray(trajectory.get(key), dtype=np.float64)
        initial = float(np.asarray(_require_array(common, common_key, "common initial")).item())
        if not np.isclose(values[0], initial, rtol=1e-12, atol=1e-12):
            raise PlotInputError(f"seed-{seed} NoF=0 {key} differs from the registered common initial")
        if np.any(values < 0.0):
            raise PlotInputError(f"seed-{seed}.{key} contains a negative article metric")
        result[key] = values
    return result


def _exact_indices(nof: Array, requested: Array, label: str) -> Array:
    nof = np.asarray(nof, dtype=np.int64)
    requested = np.asarray(requested, dtype=np.int64)
    positions = np.searchsorted(nof, requested)
    if np.any(positions >= nof.size):
        raise PlotInputError(f"{label} lacks requested NoF values {requested.tolist()}")
    if not np.array_equal(nof[positions], requested):
        raise PlotInputError(f"{label} does not contain exact requested NoF values")
    return positions


def _construct_feneb(
    record: Mapping[str, Array],
    common: Mapping[str, Array],
    registration: Mapping[str, object],
) -> Dict[str, Array]:
    nof_settings = _mapping(registration, "nof")
    complete_band = _integer(nof_settings, "complete_band", "nof")
    budget = _integer(
        nof_settings, "incremental_budget_per_optimization_run", "nof"
    )
    raw_counts = np.asarray(_require_array(record, "count_seq", "FENEB", ndim=1))
    if not np.issubdtype(raw_counts.dtype, np.integer):
        raise PlotInputError("FENEB count_seq must contain exact integer counts")
    counts = raw_counts.astype(np.int64, copy=False)
    if counts.size == 0 or counts[0] != 0 or np.any(np.diff(counts) <= 0):
        raise PlotInputError("FENEB count_seq must increase strictly from zero")
    nof = counts * complete_band
    keep = nof <= budget
    nof = nof[keep]
    expected_nof = np.arange(0, budget + complete_band, complete_band, dtype=np.int64)
    if not np.array_equal(nof, expected_nof):
        raise PlotInputError("FENEB does not provide the exact registered 0..budget NoF range")
    result: Dict[str, Array] = {"nof_mean_force_incremental": nof}
    # Reproduce the example's public FENEB comparison: the actual sampled mean
    # path is scored with the corresponding reference M and mean force, using
    # the same centered-tangent article implementation as IVM.  In particular,
    # ``barrier_seq`` in the raw archive is not substituted for this
    # reconstruction.
    paths = np.asarray(_require_array(record, "mean_path_unwrapped_seq", "FENEB", ndim=3), dtype=np.float64)
    if paths.shape[0] != counts.size or paths.shape[1:] != np.asarray(common["full_path_init"]).shape:
        raise PlotInputError("FENEB path history is incompatible with count_seq/common path")
    M = np.asarray(_require_array(record, "M_ref_seq", "FENEB", ndim=4), dtype=np.float64)
    grad_F = np.asarray(_require_array(record, "F_ref_seq", "FENEB", ndim=3), dtype=np.float64)
    expected_internal = N_INTERNAL
    if M.shape != (counts.size, expected_internal, 2, 2):
        raise PlotInputError("FENEB.M_ref_seq has an incompatible shape")
    if grad_F.shape != (counts.size, expected_internal, 2):
        raise PlotInputError("FENEB.F_ref_seq has an incompatible shape")
    paths = paths[keep].copy()
    M = M[keep].copy()
    grad_F = grad_F[keep].copy()
    paths[0] = np.asarray(common["full_path_init"], dtype=np.float64)
    M[0] = np.asarray(common["M_init"], dtype=np.float64)
    grad_F[0] = np.asarray(common["F_init"], dtype=np.float64)
    result["full_path_unwrapped"] = paths
    metrics = [compute_article_metrics(path, metric, force) for path, metric, force in zip(paths, M, grad_F)]
    for key in ("I_perp_G", "I_arc", "Delta_F_barrier"):
        values = np.asarray([float(np.asarray(item[key]).item()) for item in metrics])
        initial = float(np.asarray(_require_array(common, key, "common initial")).item())
        if not np.isclose(values[0], initial, rtol=1e-12, atol=1e-12):
            raise PlotInputError(f"FENEB causal initial {key} does not recompute")
        result[key] = values
    return result


def _construct_fes(record: Mapping[str, Array], registration: Mapping[str, object]) -> Dict[str, Array]:
    # The frozen Ala2D surface is stored in kJ/mol, while the article panels use
    # kcal/mol.  This physical conversion is source-controlled rather than a
    # candidate-dependent plotting knob.
    divisor = 4.184
    X = np.asarray(_require_array(record, "X", "frozen FES", ndim=2), dtype=np.float64)
    Y = np.asarray(_require_array(record, "Y", "frozen FES", ndim=2), dtype=np.float64)
    Z = np.asarray(_require_array(record, "Z", "frozen FES", ndim=2), dtype=np.float64)
    if X.shape != Y.shape or X.shape != Z.shape:
        raise PlotInputError("frozen FES X, Y, and Z shapes differ")
    Z_kcal = Z / divisor
    return {
        "phi_rad": np.deg2rad(X),
        "psi_rad": np.deg2rad(Y),
        "free_energy_kcal_per_mol": Z_kcal,
    }


def _all_run_input_hashes(
    run_directory: Path,
    binding_path: Path,
    manifest_path: Path,
    completion_path: Path,
) -> Dict[str, str]:
    paths: List[Path] = [binding_path, manifest_path, completion_path]
    state_paths = sorted((run_directory / "states").glob("state_*.npz"))
    if not state_paths:
        raise PlotInputError(f"optimization run has no state artifacts: {run_directory}")
    paths.extend(state_paths)
    checkpoint_path = run_directory / "controller_checkpoint.npz"
    if not checkpoint_path.is_file():
        raise PlotInputError(f"optimization run lacks controller checkpoint: {run_directory}")
    paths.append(checkpoint_path)
    status_path = run_directory / "status.json"
    if status_path.is_file():
        paths.append(status_path)
    attempt_directory = run_directory / "md_attempt_records"
    if not attempt_directory.is_dir() or attempt_directory.is_symlink():
        raise PlotInputError(
            f"optimization run lacks its atomic MD attempt ledger: {run_directory}"
        )
    attempt_paths = sorted(attempt_directory.glob("*.json"))
    if not attempt_paths or any(path.is_symlink() for path in attempt_paths):
        raise PlotInputError(f"optimization run has an unsafe or empty attempt ledger: {run_directory}")
    paths.extend(attempt_paths)
    return {str(path.resolve()): sha256_file(path) for path in paths}


def _audit_no_hidden_evaluations(
    run_directory: Path,
    *,
    parameter_id: str,
    master_seed: int,
    trajectory: Mapping[str, Array],
) -> Mapping[str, object]:
    """Re-audit raw attempt records; a precomputed pass flag is insufficient."""

    try:
        from run_v17_formal_campaign import audit_completed_optimization_run

        audit = audit_completed_optimization_run(
            run_directory,
            parameter_id=parameter_id,
            master_seed=master_seed,
        )
    except Exception as error:
        raise PlotInputError(
            f"seed-{master_seed} raw MD-attempt/public-state audit failed"
        ) from error
    completion = _load_json(
        run_directory / "completion.json", f"seed-{master_seed} completion report"
    )
    terminal = validate_public_trajectory(trajectory, completion=completion)
    attempt = audit.get("attempt_audit")
    if (
        audit.get("pass") is not True
        or audit.get("implementation_id") != IMPLEMENTATION_ID
        or audit.get("parameter_id") != parameter_id
        or audit.get("master_seed") != master_seed
        or audit.get("terminal_state_kind") != terminal.state_kind
        or audit.get("terminal_pending_next_action") != terminal.pending_next_action
        or not isinstance(attempt, Mapping)
        or attempt.get("pass") is not True
        or attempt.get("hidden_logical_evaluation_count") != 0
        or attempt.get("realized_mean_force_nof") != terminal.nof
    ):
        raise PlotInputError(f"seed-{master_seed} formal run audit is inconsistent")
    return audit


def build_registered_plot_data(
    registration_path: PathLike = DEFAULT_REGISTRATION,
    acceptance_report_path: PathLike = DEFAULT_ACCEPTANCE_REPORT,
) -> Dict[str, object]:
    """Validate all evidence and construct arrays for both registered figures.

    This function is pure with respect to the filesystem: it reads evidence but
    creates, edits, and replaces no files.
    """

    registration, root, input_hashes = load_sealed_registration(registration_path)
    _validate_nof_contract(registration)
    registration_path = Path(registration_path).resolve()
    acceptance_report_path = _inside(
        root, Path(acceptance_report_path).resolve(), "final acceptance report"
    )
    report, winner, run_paths, referenced_report_hashes = validate_acceptance_report(
        acceptance_report_path, registration, registration_path, root
    )
    input_hashes[str(acceptance_report_path)] = sha256_file(acceptance_report_path)
    input_hashes.update(referenced_report_hashes)
    inputs = _mapping(registration, "inputs")
    common, causal_hashes = _validate_v17_causal_common_initial(
        root, registration, inputs
    )
    input_hashes.update(causal_hashes)
    feneb_path = _campaign_path(root, inputs.get("feneb"), "inputs.feneb")
    fes_path = _campaign_path(root, inputs.get("fes"), "inputs.fes")
    input_hashes[str(feneb_path)] = _require_registered_input(
        root, registration, feneb_path, "inputs.feneb"
    )
    input_hashes[str(fes_path)] = _require_registered_input(
        root, registration, fes_path, "inputs.fes"
    )
    feneb_record = _load_npz(feneb_path, "registered FENEB trajectory")
    fes_record = _load_npz(fes_path, "registered frozen FES")
    registration_digest = sha256_file(registration_path)
    trajectories: Dict[int, Dict[str, Array]] = {}
    terminals: Dict[int, object] = {}
    for seed, run_path in run_paths.items():
        binding_path, binding = _validate_run_binding(
            run_path, registration_digest, winner, seed, registration
        )
        manifest_path = _validate_run_manifest(
            run_path, binding, seed, registration
        )
        try:
            raw_trajectory = _load_public_plot_trajectory(run_path, manifest_path)
        except Exception as error:
            raise PlotInputError(f"invalid public trajectory for seed {seed}: {run_path}") from error
        trajectories[seed] = _validate_trajectory(raw_trajectory, seed, registration, common)
        completion_path = _validate_run_completion(
            run_path, manifest_path, raw_trajectory, seed
        )
        completion = _load_json(completion_path, f"seed-{seed} completion report")
        terminals[seed] = validate_public_trajectory(
            raw_trajectory, completion=completion
        )
        _audit_no_hidden_evaluations(
            run_path,
            parameter_id=winner,
            master_seed=seed,
            trajectory=raw_trajectory,
        )
        input_hashes.update(
            _all_run_input_hashes(
                run_path, binding_path, manifest_path, completion_path
            )
        )

    analysis = report["acceptance_analysis"]
    correct_path = analysis.get("correct_path")
    correct_rows = correct_path.get("seeds") if isinstance(correct_path, Mapping) else None
    if not isinstance(correct_rows, list) or len(correct_rows) != len(terminals):
        raise PlotInputError("acceptance report lacks one terminal binding per optimization seed")
    accepted_by_seed = {
        row.get("seed"): row for row in correct_rows if isinstance(row, Mapping)
    }
    for seed, terminal in terminals.items():
        row = accepted_by_seed.get(seed)
        if (
            not isinstance(row, Mapping)
            or row.get("selected_state_index") != terminal.index
            or row.get("selected_nof") != terminal.nof
            or row.get("selected_state_kind") != terminal.state_kind
        ):
            raise PlotInputError(
                f"acceptance report does not bind seed-{seed}'s exact terminal public state"
            )

    gate = _mapping(registration, "plotting")
    quantities = gate.get(
        "threepanel_quantities", ["I_perp_G", "I_arc", "Delta_F_barrier"]
    )
    expected_quantities = ["I_perp_G", "I_arc", "Delta_F_barrier"]
    if quantities != expected_quantities:
        raise PlotInputError(
            f"plotting.threepanel_quantities must be {expected_quantities!r}"
        )
    seeds = _registered_seeds(registration)
    nof = trajectories[seeds[0]]["nof_mean_force_incremental"]
    candidate: Dict[str, Dict[str, Array]] = {}
    for metric in expected_quantities:
        rows = np.stack([trajectories[seed][metric] for seed in seeds])
        candidate[metric] = {
            "by_seed": rows,
            "median": np.median(rows, axis=0),
            "minimum": np.min(rows, axis=0),
            "maximum": np.max(rows, axis=0),
        }
    feneb = _construct_feneb(feneb_record, common, registration)
    contour_seed = _single_registered_seed(registration, "contour")
    if contour_seed not in trajectories:
        raise PlotInputError("registered contour seed is not an optimization seed")
    if gate.get("contour_seed") != contour_seed:
        raise PlotInputError("plotting.contour_seed differs from seeds.contour")
    contour_values = gate.get("contour_nof")
    if not isinstance(contour_values, list) or not contour_values or any(
        type(value) is not int for value in contour_values
    ):
        raise PlotInputError("plotting.contour_nof must be an integer list")
    contour_nof = np.asarray(contour_values, dtype=np.int64)
    if not np.array_equal(contour_nof, CONTOUR_NOF):
        raise PlotInputError("registered contour NoF must be exactly 0/200/400/2000")
    candidate_indices = _exact_indices(nof, contour_nof, f"seed-{contour_seed} trajectory")
    feneb_indices = _exact_indices(
        feneb["nof_mean_force_incremental"], contour_nof, "FENEB trajectory"
    )
    method_label = str(gate.get("method_label", "IVM"))
    horizontal_axis = _text(gate, "x_axis", "plotting")
    if method_label != "IVM":
        raise PlotInputError("the method legend must be exactly 'IVM'")
    if horizontal_axis != MEAN_FORCE_AXIS_LABEL:
        raise PlotInputError(
            "the horizontal axis must be exactly 'Number of mean-force evaluations'"
        )
    output = {
        "registration": registration,
        "registration_path": registration_path,
        "registration_sha256": registration_digest,
        "campaign_root": root,
        "acceptance_report": report,
        "acceptance_report_path": acceptance_report_path,
        "winner_parameter_id": winner,
        "optimization_run_paths": run_paths,
        "labels": {
            "method": method_label,
            "feneb": str(gate.get("feneb_label", "FENEB")),
            "horizontal_axis": horizontal_axis,
        },
        "threepanel": {
            "nof_mean_force_incremental": nof,
            "candidate": candidate,
            "feneb": feneb,
            "maximum_nof": _integer(
                _mapping(registration, "nof"),
                "incremental_budget_per_optimization_run",
                "nof",
            ),
        },
        "fourpanel": {
            "nof_mean_force_incremental": contour_nof,
            "candidate_seed": contour_seed,
            "candidate_paths": trajectories[contour_seed]["full_path_unwrapped"][candidate_indices],
            "feneb_paths": feneb["full_path_unwrapped"][feneb_indices],
            "fes": _construct_fes(fes_record, registration),
        },
        "input_hashes": dict(sorted(input_hashes.items())),
        "source_hashes": dict(_mapping(_mapping(registration, "hashes"), "source", "hashes")),
        "metric_definitions": dict(registration.get("metrics", {})),
        "nof_convention": dict(_mapping(registration, "nof")),
    }
    return output


def _atomic_figure_save(fig: object, target: Path, **kwargs: object) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        fig.savefig(temporary, **kwargs)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(target: Path, value: Mapping[str, object]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}.", suffix=".json", dir=target.parent, delete=False, mode="w", encoding="utf-8"
    )
    temporary = Path(handle.name)
    try:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, target)
    finally:
        if not handle.closed:
            handle.close()
        if temporary.exists():
            temporary.unlink()


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{target.stem}.", suffix=target.suffix, dir=target.parent, delete=False
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _render_threepanel(data: Mapping[str, object], pdf_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    threepanel = data["threepanel"]
    labels = data["labels"]
    nof = np.asarray(threepanel["nof_mean_force_incremental"])
    feneb = threepanel["feneb"]
    candidate = threepanel["candidate"]
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 1.1,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#111111",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "text.color": "#111111",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.size": 12,
        "axes.labelsize": 14,
        "legend.frameon": False,
    }
    specifications = (
        ("I_perp_G", r"$\widehat{I}_{\perp}^{\,G}$", "Riemannian normal residual", True),
        ("I_arc", r"$I_{\mathrm{arc}}$", "Equi-arclength residual", True),
        ("Delta_F_barrier", r"$\Delta F^{\dagger}\ \mathrm{(kcal\,mol^{-1})}$", "Free-energy barrier", False),
    )
    method_color = "#0072B2"
    feneb_color = "#D55E00"
    tiny = np.finfo(np.float64).tiny
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
        for axis, (metric, ylabel, title, log_scale), tag in zip(
            axes, specifications, ("a", "b", "c")
        ):
            summary = candidate[metric]
            lower = np.asarray(summary["minimum"], dtype=np.float64)
            upper = np.asarray(summary["maximum"], dtype=np.float64)
            median = np.asarray(summary["median"], dtype=np.float64)
            feneb_values = np.asarray(feneb[metric], dtype=np.float64)
            if log_scale:
                lower_plot = np.maximum(lower, tiny)
                upper_plot = np.maximum(upper, tiny)
                median_plot = np.maximum(median, tiny)
                feneb_plot = np.maximum(feneb_values, tiny)
            else:
                lower_plot, upper_plot = lower, upper
                median_plot, feneb_plot = median, feneb_values
            axis.fill_between(
                nof, lower_plot, upper_plot, color=method_color, alpha=0.16,
                linewidth=0.0, zorder=2,
            )
            markevery = max(1, nof.size // 10)
            axis.plot(
                nof, median_plot, color=method_color, lw=2.4, marker="o",
                markevery=markevery, ms=6.0, markerfacecolor="white",
                markeredgecolor=method_color, markeredgewidth=1.5,
                solid_capstyle="round", label=labels["method"], zorder=5,
            )
            feneb_nof = np.asarray(feneb["nof_mean_force_incremental"])
            axis.plot(
                feneb_nof, feneb_plot, color=feneb_color, lw=2.4, marker="s",
                markevery=max(1, feneb_nof.size // 10), ms=5.8,
                markerfacecolor="white", markeredgecolor=feneb_color,
                markeredgewidth=1.4, solid_capstyle="round",
                label=labels["feneb"], zorder=4,
            )
            axis.scatter(
                nof[-1], median_plot[-1], s=62, marker="o", facecolor=method_color,
                edgecolor="white", linewidth=1.1, zorder=10, clip_on=False,
            )
            axis.scatter(
                feneb_nof[-1], feneb_plot[-1], s=58, marker="s", facecolor=feneb_color,
                edgecolor="white", linewidth=1.1, zorder=9, clip_on=False,
            )
            axis.set_xlim(0, int(threepanel["maximum_nof"]))
            if log_scale:
                axis.set_yscale("log")
            else:
                axis.minorticks_on()
            axis.set_xlabel(labels["horizontal_axis"], labelpad=6)
            axis.set_ylabel(ylabel, labelpad=6)
            axis.set_title(title, fontsize=13, pad=12, fontweight="medium")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.tick_params(which="both", top=False, right=False)
            axis.grid(
                True, which="major", axis="y", linestyle="-", linewidth=0.6,
                alpha=0.18, color="#666666",
            )
            axis.text(
                -0.16, 1.06, tag, transform=axis.transAxes, fontsize=17,
                fontweight="bold", va="top", ha="left",
            )
        handles, legend_labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.06),
            ncol=2, fontsize=13, handlelength=2.2, columnspacing=2.4,
            borderaxespad=0.0,
        )
        fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.15, wspace=0.30)
        _atomic_figure_save(fig, pdf_path, bbox_inches="tight")
        _atomic_figure_save(fig, png_path, dpi=400, bbox_inches="tight")
        plt.close(fig)


def _path_on_fes_branch(path: Array) -> Array:
    wrapped = wrap_periodic(np.asarray(path, dtype=np.float64))
    # The C5-to-C7eq paths live in phi <= 0, psi >= 0.  Periodic wrapping is
    # deterministic, and this shift keeps a path visually continuous if a
    # value lies just across the principal-interval seam.
    result = wrapped.copy()
    for coordinate in range(2):
        for index in range(1, result.shape[0]):
            delta = result[index, coordinate] - result[index - 1, coordinate]
            if delta > np.pi:
                result[index:, coordinate] -= 2.0 * np.pi
            elif delta < -np.pi:
                result[index:, coordinate] += 2.0 * np.pi
    return result


def _render_fourpanel(data: Mapping[str, object], pdf_path: Path, png_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    fourpanel = data["fourpanel"]
    labels = data["labels"]
    fes = fourpanel["fes"]
    X = np.asarray(fes["phi_rad"])
    Y = np.asarray(fes["psi_rad"])
    Z = np.asarray(fes["free_energy_kcal_per_mol"])
    nof_values = np.asarray(fourpanel["nof_mean_force_incremental"], dtype=np.int64)
    candidate_paths = np.asarray(fourpanel["candidate_paths"])
    feneb_paths = np.asarray(fourpanel["feneb_paths"])
    if nof_values.size != 4:
        raise PlotInputError("the registered contour design must contain four panels")
    x_line = X[0, :]
    y_line = Y[:, 0]
    columns = np.flatnonzero((x_line >= -np.pi) & (x_line <= 0.0))
    rows = np.flatnonzero((y_line >= 0.0) & (y_line <= np.pi))
    if columns.size < 2 or rows.size < 2:
        raise PlotInputError("frozen FES does not cover the registered C5-to-C7eq window")
    row_slice = slice(int(rows[0]), int(rows[-1]) + 1)
    column_slice = slice(int(columns[0]), int(columns[-1]) + 1)
    X = X[row_slice, column_slice]
    Y = Y[row_slice, column_slice]
    Z = Z[row_slice, column_slice]
    Z = Z - float(np.min(Z))
    # The old visual design whitened the FES above 60 kJ/mol.  Preserve that
    # physical threshold after applying the registered kJ-to-kcal conversion.
    upper = 60.0 / 4.184
    curve = 2.0
    coordinate = np.linspace(0.0, 1.0, 256)
    mapped = np.clip(coordinate + curve * coordinate * (1.0 - coordinate), 0.0, 1.0)
    gray = 0.05 + 0.95 * mapped
    background_cmap = LinearSegmentedColormap.from_list(
        "registered_dark_to_white", list(zip(gray, gray, gray))
    )
    background_cmap.set_over("white")
    levels = np.linspace(0.0, upper, 33)
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 1.1,
        "axes.edgecolor": "#222222",
        "axes.labelcolor": "#111111",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "text.color": "#111111",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "font.size": 12,
        "axes.labelsize": 14,
        "legend.frameon": False,
    }
    method_color = "#0072B2"
    feneb_color = "#D55E00"
    special_points = (
        (-2.62593029, 2.77365591, r"$C5$", -0.05, 0.16),
        (-1.33433281, 0.99268968, r"$C7_{\mathrm{eq}}$", 0.10, -0.40),
    )
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.8), sharex=True, sharey=True)
        contour_artist = None
        for index, (axis, nof, tag) in enumerate(zip(axes, nof_values, ("a", "b", "c", "d"))):
            contour_artist = axis.contourf(
                X, Y, Z, levels=levels, cmap=background_cmap, extend="max"
            )
            for point_x, point_y, point_label, dx, dy in special_points:
                axis.plot(
                    point_x, point_y, marker="o", markersize=6,
                    markerfacecolor="white", markeredgecolor="#111111",
                    markeredgewidth=1.4, linestyle="None", zorder=9,
                )
                axis.text(
                    point_x + dx, point_y + dy, point_label, color="#111111",
                    fontsize=12, fontweight="bold", zorder=9, ha="right", va="bottom",
                    bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.75},
                )
            feneb_path = _path_on_fes_branch(feneb_paths[index])
            candidate_path = _path_on_fes_branch(candidate_paths[index])
            axis.plot(
                feneb_path[:, 0], feneb_path[:, 1], color=feneb_color, lw=2.2,
                marker="s", ms=5.0, markerfacecolor="white",
                markeredgecolor=feneb_color, markeredgewidth=1.3,
                solid_capstyle="round", zorder=5,
            )
            axis.plot(
                candidate_path[:, 0], candidate_path[:, 1], color=method_color, lw=2.2,
                marker="o", ms=5.0, markerfacecolor="white",
                markeredgecolor=method_color, markeredgewidth=1.3,
                solid_capstyle="round", zorder=6,
            )
            title = "Initialization" if int(nof) == 0 else f"Mean-force evaluations: {int(nof):,}"
            axis.set_title(title, fontsize=13, pad=10, fontweight="medium")
            axis.set_xlabel(r"$\phi$ (rad)", labelpad=5)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlim(-np.pi, 0.0)
            axis.set_ylim(0.0, np.pi)
            axis.tick_params(which="both", top=False, right=False)
            axis.text(
                -0.02, 1.11, tag, transform=axis.transAxes, fontsize=17,
                fontweight="bold", va="top", ha="left",
            )
        axes[0].set_ylabel(r"$\psi$ (rad)", labelpad=5)
        fig.subplots_adjust(left=0.05, right=0.88, top=0.82, bottom=0.14, wspace=0.08)
        position = axes[-1].get_position()
        colorbar_axis = fig.add_axes([position.x1 + 0.015, position.y0, 0.014, position.height])
        colorbar = fig.colorbar(contour_artist, cax=colorbar_axis, extend="max")
        colorbar.set_label(r"Free energy (kcal mol$^{-1}$)", labelpad=8)
        handles = [
            Line2D(
                [0], [0], color=method_color, linewidth=2.2, marker="o",
                markerfacecolor="white", markeredgecolor=method_color,
                markeredgewidth=1.3, label=labels["method"],
            ),
            Line2D(
                [0], [0], color=feneb_color, linewidth=2.2, marker="s",
                markerfacecolor="white", markeredgecolor=feneb_color,
                markeredgewidth=1.3, label=labels["feneb"],
            ),
        ]
        fig.legend(
            handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
            ncol=2, fontsize=13, handlelength=2.2, columnspacing=2.4,
            borderaxespad=0.0,
        )
        _atomic_figure_save(fig, pdf_path, bbox_inches="tight")
        _atomic_figure_save(fig, png_path, dpi=400, bbox_inches="tight")
        plt.close(fig)


def _formal_targets(data: Mapping[str, object]) -> Tuple[Path, Path]:
    plotting = _mapping(data["registration"], "plotting")
    values = plotting.get("formal_pdf_targets")
    if (
        not isinstance(values, list)
        or len(values) != 2
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise PlotInputError("plotting.formal_pdf_targets must contain two paths")
    project_root = Path(data["campaign_root"]).resolve().parents[1]
    three = _inside(project_root, project_root / values[0], "three-panel formal PDF")
    four = _inside(project_root, project_root / values[1], "four-panel formal PDF")
    if three == four:
        raise PlotInputError("registered formal figure targets are identical")
    return three, four


def _assert_formal_replacement_allowed(data: Mapping[str, object]) -> Tuple[Path, Path]:
    registration = data["registration"]
    gate = _mapping(registration, "plotting")
    if gate.get(
        "formal_pdf_replacement_requires_explicit_flag_and_passed_final_acceptance"
    ) is not True:
        raise PlotInputError("registration does not permit formal PDF replacement")
    report = data["acceptance_report"]
    analysis = report.get("acceptance_analysis")
    if (
        report.get("pass") is not True
        or report.get("decision") != "pass"
        or report.get("figure_replacement_authorized") is not True
        or report.get("registration_sha256") != data["registration_sha256"]
        or not isinstance(analysis, Mapping)
        or analysis.get("pass") is not True
        or analysis.get("decision") != "pass"
    ):
        raise PlotInputError("full-pass acceptance evidence is not bound to this registration")
    try:
        accepted_report_schema(analysis)
    except FormalContractError as error:
        raise PlotInputError("formal replacement uses the wrong acceptance schema") from error
    # Close the interval between validated reads and formal replacement: every
    # source, reference, report, binding, and public state used to construct the
    # figures must still have the digest recorded by the data-construction step.
    input_hashes = data.get("input_hashes")
    if not isinstance(input_hashes, Mapping) or not input_hashes:
        raise PlotInputError("plot data lacks the validated input-hash manifest")
    for raw_path, expected in input_hashes.items():
        path = Path(str(raw_path))
        if not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected) is None:
            raise PlotInputError(f"plot input lacks a valid SHA-256 digest: {path}")
        if not path.is_file() or sha256_file(path) != expected:
            raise PlotInputError(f"plot input changed after validation: {path}")
    targets = _formal_targets(data)
    registered_hashes = _mapping(
        gate, "formal_pdf_pre_campaign_sha256", "plotting"
    )
    for target in targets:
        expected = registered_hashes.get(str(target))
        if not isinstance(expected, str) or SHA256_PATTERN.fullmatch(expected) is None:
            raise PlotInputError(f"pre-campaign hash is not registered for {target}")
        if not target.is_file():
            raise PlotInputError(f"registered formal PDF is missing before replacement: {target}")
        observed = sha256_file(target)
        if observed != expected:
            raise PlotInputError(
                f"formal PDF changed since preregistration: {target}; "
                f"expected {expected}, observed {observed}"
            )
    return targets


def render_registered_figures(
    data: Mapping[str, object],
    output_directory: PathLike,
    *,
    formal: bool = False,
) -> Dict[str, object]:
    """Render preview PDF/PNG pairs and optionally atomically replace formal PDFs."""

    registration = data.get("registration")
    if not isinstance(registration, Mapping):
        raise PlotInputError("plot data lacks its sealed registration")
    try:
        assert_registration_is_not_executable(registration)
    except FormalContractError as error:
        # Run this before mkdir or temporary-file construction.  Direct callers
        # cannot bypass the fail-closed DRAFT gate by skipping build_plot_data.
        raise PlotInputError(str(error)) from error
    output_directory = Path(output_directory).resolve()
    formal_targets: Optional[Tuple[Path, Path]] = None
    formal_before: Dict[str, str] = {}
    if formal:
        formal_targets = _assert_formal_replacement_allowed(data)
        formal_before = {str(path): sha256_file(path) for path in formal_targets}
    # The CLI default is campaign-local; an absolute external directory is
    # possible only when the caller explicitly supplies it.
    output_directory.mkdir(parents=True, exist_ok=True)
    three_pdf = output_directory / "MFEP_IVM_vs_FENEB_bestpair_threepanel_preview.pdf"
    three_png = output_directory / "MFEP_IVM_vs_FENEB_bestpair_threepanel_preview.png"
    four_pdf = output_directory / "MFEP_IVM_vs_FENEB_contour_fourpanel_preview.pdf"
    four_png = output_directory / "MFEP_IVM_vs_FENEB_contour_fourpanel_preview.png"
    _render_threepanel(data, three_pdf, three_png)
    _render_fourpanel(data, four_pdf, four_png)
    if formal_targets is not None:
        _atomic_copy(three_pdf, formal_targets[0])
        _atomic_copy(four_pdf, formal_targets[1])
    outputs = [three_pdf, three_png, four_pdf, four_png]
    if formal_targets is not None:
        outputs.extend(formal_targets)
    output_hashes = {str(path): sha256_file(path) for path in outputs}
    provenance = {
        "schema_version": "1.0.0",
        "campaign_id": data["registration"].get("campaign_id"),
        "winner_parameter_id": data["winner_parameter_id"],
        "registration_path": str(data["registration_path"]),
        "registration_sha256": data["registration_sha256"],
        "acceptance_report_path": str(data["acceptance_report_path"]),
        "acceptance_pass": True,
        "optimization_runs": {
            str(seed): str(path) for seed, path in data["optimization_run_paths"].items()
        },
        "metric_definitions": data["metric_definitions"],
        "nof_convention": data["nof_convention"],
        "plotting_contract": data["registration"]["plotting"],
        "input_sha256": data["input_hashes"],
        "registered_source_sha256": data["source_hashes"],
        "plotting_source": str(MODULE_PATH),
        "plotting_source_sha256": sha256_file(MODULE_PATH),
        "output_sha256": output_hashes,
        "formal_replacement_requested": bool(formal),
        "formal_pre_replacement_sha256": formal_before,
    }
    provenance_path = output_directory / "plot_provenance.json"
    _atomic_json(provenance_path, provenance)
    provenance["provenance_path"] = str(provenance_path)
    provenance["provenance_sha256"] = sha256_file(provenance_path)
    return provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render fully accepted Ala2D IVM/FENEB figures from sealed evidence."
    )
    parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    parser.add_argument("--acceptance-report", type=Path, default=DEFAULT_ACCEPTANCE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PREVIEW_DIRECTORY)
    parser.add_argument(
        "--formal",
        action="store_true",
        help="atomically replace the two registered formal PDFs after all full-pass gates",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = _parser().parse_args(argv)
    data = build_registered_plot_data(arguments.registration, arguments.acceptance_report)
    result = render_registered_figures(data, arguments.output_dir, formal=arguments.formal)
    print(
        json.dumps(
            {
                "status": "rendered",
                "formal_replacement": bool(arguments.formal),
                "provenance_path": result["provenance_path"],
                "provenance_sha256": result["provenance_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PlotInputError as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        raise SystemExit(2)
