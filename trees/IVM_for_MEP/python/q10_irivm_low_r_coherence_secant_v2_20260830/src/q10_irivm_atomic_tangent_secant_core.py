#!/usr/bin/env python3
"""Low-r chord-coherent recovery-memory transport for QM10 IR-IVM.

Only the system interface changes: the fixed Q-transformed Muller potential,
the two article path resolutions, their common straight initial paths, and the
potential-specific stable proximal energy difference/preconditioner.  The
active method is otherwise the formal LEPS method traced from
``leps_all_methods_metrics.pdf``:

* Henkelman--Jonsson improved tangents everywhere;
* complete-normal proximal Armijo L-BFGS with a geometry-derived total trust
  ball and an atomic I_perp-monotonic main-stage commit;
* identical-spring reparameterization after every completed main stage;
* at most one synchronized complete-band regularized-prediction recovery;
* recovery uses only the existing alpha, r, and m_max controls;
* every uncached internal-image energy/gradient pair is one Number of Forces,
  including rejected trials and rolled-back transactions.

For the existing QM10 ASE embedding, each of the ten scalar transformed
coordinates is the x coordinate of one atom.  Therefore the formal LEPS
"max atomic norm" becomes max(abs(dx_j)), and its RMS segment length becomes
||segment||/sqrt(10).  This is the only necessary dimensional generalization
of the LEPS trust geometry and is fixed rather than tuned.

Only accepted-recovery memory publication is changed relative to the frozen
chord-coherent method (SHA-256 prefix ``0fafbb``).
The first recovery proposal at a direct anchor is its accepted-history
regularized-prediction step.  The raw proposal is purely algebraic and costs no
force call.  Direct-anchor energies drive a Henkelman--Jonsson improved-tangent
Newton correction of the adjacent-chord-length constraints until the original
equidistance ceiling is met.  Exactly that projected complete band is evaluated
once.  It is accepted atomically as a whole or rejected as a whole.  A complete
accepted recovery first certifies direct-to-projected physical secants only
when s.y > 1e-6 ||s|| ||y||.  A strong chord-coherent recovery publishes the
single certified pair with greatest normalized curvature, with the smaller
1-based internal path-image index breaking an exact tie.  Conservative and
ordinary stationarity recoveries publish every certified pair.  This ranking
uses only already evaluated recovery data and therefore costs zero forces.

If the evaluated complete band is rejected, its actual full-step displacement
is retained strictly inside recovery.  At the next bitwise-identical direct
anchor, recovery scales that stored displacement by the existing fixed Armijo
backtracking factor, one half at a time, for at most m_max completed slots.
Direction selection, chord projection, and backtracking require no force call;
each projected band is charged normally.

The private line-search state is cleared when a main stage changes the public
path.  A failed recovery evaluated after that movement may establish a new
anchor and carry to the next outer iteration only if the recomputed direct
path, energies, and gradients are bitwise identical.  Any committed
transaction, changed anchor, interruption, numerical failure, termination, or
attempt cap clears it.  The existing ``m_max`` caps the completed fixed-factor
recovery slots at one anchor, including the original full step; actual cache
misses are recorded separately.  The fixed backtracking factor
is the already existing formal-LEPS Armijo constant, not a tuned control.
Before the unchanged improved-tangent chord projection, a normal recovery is
seeded by equal-arclength polyline and improved-spring reparameterizations.
At the already evaluated highest-energy internal image, the two adjacent unit
chords define a dimensionless RMS direction mismatch
``sqrt(1-u_minus.u_plus)``.  The strong ``m_max-1`` spring correction is used
exactly when its amplified mismatch remains within one unit; otherwise the
unit correction is used.
This is a zero-NoF geometric gate, not a tangent approximation.  Every tangent
used for force projection, I_perp, spring motion, and chord projection remains
the Henkelman--Jonsson improved tangent.

When an acceptable direct band is unchanged at the existing public
stationarity tolerance, recovery is exhausted before convergence is declared.
Those stationarity probes use the ordinary normal recovery seed and the same
atomic gate and ``m_max`` line-search cap.  Only the projected band enters the
whole-band atomic recovery merge.  The main subproblem, direct
reparameterization, acceptance gate, and Number-of-Forces accounting are
otherwise unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


THIS_FILE = Path(__file__).resolve()
CAMPAIGN_ROOT = THIS_FILE.parents[1]
REPOSITORY_ROOT = THIS_FILE.parents[3]
REGISTRATION = CAMPAIGN_ROOT / "registrations/CAMPAIGN_PREREGISTRATION.json"
EXPERIMENT_ID = "q10-irivm-low-r-coherence-secant-v2-20260830"

from q10_solver_backend import load_solver_only_q10


ARTICLE_NF_BUDGET = 3000
OUTER_LIMIT = 10000
RESOLUTIONS = (10, 20)

# One compact, shared Cartesian search.  Alpha labels have integer mantissas;
# decimal-mantissa values such as 1.15e-3 are deliberately excluded.
ALPHA_LABELS = ("1e-3", "2e-3")
ALPHAS = tuple(float(label) for label in ALPHA_LABELS)
R_VALUES = (0.60, 0.65)
M_MAX_VALUES = (4,)
GRID_TRIPLES = tuple(
    (alpha, r, m_max)
    for alpha in ALPHAS
    for r in R_VALUES
    for m_max in M_MAX_VALUES
)
REGISTERED_CONTROLS = tuple(
    (n_internal, alpha, r, m_max)
    for n_internal in RESOLUTIONS
    for alpha, r, m_max in GRID_TRIPLES
)

# Fixed formal-LEPS method constants.  None is tuned or resolution-dependent.
HISTORY_SIZE = 15
ARMIJO = 0.01
BACKTRACK = 0.5
KKT_ATOL = 1.0e-14
ROUNDING_FACTOR = 64.0
REPARAMETERIZATION_PERIOD = 1
TRUST_RADIUS_FRACTION = 2.0
TRUST_RADIUS_MAXIMUM = 1.0
SPRING_DTAU = 0.05
SPRING_MAX_ITER = 500
SPRING_TOLERANCE = 1.0e-5
RECOVERY_EQUIDISTANCE_CONTRACTION = 0.2
RECOVERY_IPERP_ATOL = 1.0e-12
RECOVERY_EQUIDISTANCE_ATOL = 1.0e-14
TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT = 2
RECOVERY_BAND_EVALUATION_CAP = 1
RECOVERY_SECANT_CURVATURE = 1.0e-6

FORMAL_LEPS_OPTIMIZER = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/LEPS/"
    "leps_irivm_ala_chig_every1_sync1_20260830/src/leps_irivm_sync1.py"
)
FORMAL_LEPS_REGISTRY = Path(
    "/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/LEPS/"
    "leps_irivm_sync1_refined_grid_20260830/FIGURE_PROVENANCE.md"
)

RUNNER_FILE = CAMPAIGN_ROOT / "src/run_grid.py"
BACKEND_FILE = CAMPAIGN_ROOT / "src/q10_solver_backend.py"
AUDIT_FILE = CAMPAIGN_ROOT / "src/audit_results.py"
PROTOCOL_FILE = CAMPAIGN_ROOT / "PROTOCOL.md"
N20_GATE_MANIFEST = (
    CAMPAIGN_ROOT / "reports/n20_gate_audit/audit_manifest.json"
)
CONTRACT_TEST_FILE = (
    CAMPAIGN_ROOT / "tests/test_atomic_tangent_secant_contract.py"
)
AUDIT_EDGE_TEST_FILE = (
    CAMPAIGN_ROOT / "tests/test_independent_audit_edges.py"
)
V1_FORMAL_REGISTRATION = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "q10_irivm_low_r_coherence_secant_20260830/"
    "registrations/CAMPAIGN_PREREGISTRATION.json"
)
V1_FORMAL_CORE_SOURCE = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "q10_irivm_low_r_coherence_secant_20260830/src/"
    "q10_irivm_atomic_tangent_secant_core.py"
)
V1_FORMAL_AUDIT_SOURCE = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "q10_irivm_low_r_coherence_secant_20260830/src/audit_results.py"
)
V1_FORMAL_N20_EVIDENCE_RESULT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "q10_irivm_low_r_coherence_secant_20260830/"
    "runs/n20/685c22444caacc53.npz"
)
V1_FORMAL_N20_EVIDENCE_RECEIPT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "q10_irivm_low_r_coherence_secant_20260830/"
    "reports/receipts/n20/685c22444caacc53.receipt.json"
)
V1_FORMAL_N20_GATE_MANIFEST = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "q10_irivm_low_r_coherence_secant_20260830/"
    "reports/n20_gate_audit/audit_manifest.json"
)
V1_FORMAL_N20_GATE_REPORT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "q10_irivm_low_r_coherence_secant_20260830/"
    "reports/n20_gate_audit/AUDIT_REPORT.md"
)
FROZEN_CHORD_COHERENCE_SOURCE = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "q10_irivm_chord_coherence_20260830/src/"
    "q10_irivm_atomic_tangent_secant_core.py"
)
FROZEN_ATOMIC_SOURCE = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "q10_irivm_atomic_tangent_secant_20260830/src/"
    "q10_irivm_atomic_tangent_secant_core.py"
)
GAIN_FRACTION_SOURCE = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "q10_leps_recovery_gain_fraction_refine_20260830/src/"
    "q10_leps_recovery_gain_fraction_refine_core.py"
)
PRIOR_BACKTRACK_SOURCE = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/q10_leps_recovery_backtrack_20260830/"
    "src/q10_leps_recovery_backtrack_core.py"
)
PRIOR_BACKTRACK_REGISTRATION = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/q10_leps_recovery_backtrack_20260830/"
    "registrations/CAMPAIGN_PREREGISTRATION.json"
)
PRIOR_BACKTRACK_RESULT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/q10_leps_recovery_backtrack_20260830/"
    "runs/n20/4e85b4de460363ca.npz"
)
METRIC_SOURCE = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/unified_ir_ivm_20260827/"
    "src/q10_article_metrics.py"
)
ACCEPTANCE_AMENDMENT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/python/"
    "leps_formal_ivm_q10_transfer_mmax_ge4_refinement_20260830/"
    "registrations/ARTICLE_IVM_ACCEPTANCE_AMENDMENT.json"
)
ARTICLE_N10 = Path(
    "/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/muller_q10_10d/"
    "image12/data/frozen_runs/tuned_ivm_alpha2e3_selected.npz"
)
ARTICLE_N20 = Path(
    "/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/muller_q10_10d/"
    "data/frozen_runs/tuned_ivm_alpha1e3_selected.npz"
)
N10_RUNTIME_ROOT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/"
    "muller_q10_10d/image12"
)
N20_RUNTIME_ROOT = Path(
    "/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/muller_q10_10d"
)
REQUIRED_DEPENDENCY_PATHS = (
    RUNNER_FILE,
    BACKEND_FILE,
    AUDIT_FILE,
    PROTOCOL_FILE,
    CONTRACT_TEST_FILE,
    AUDIT_EDGE_TEST_FILE,
    V1_FORMAL_REGISTRATION,
    V1_FORMAL_CORE_SOURCE,
    V1_FORMAL_AUDIT_SOURCE,
    V1_FORMAL_N20_EVIDENCE_RESULT,
    V1_FORMAL_N20_EVIDENCE_RECEIPT,
    V1_FORMAL_N20_GATE_MANIFEST,
    V1_FORMAL_N20_GATE_REPORT,
    FROZEN_CHORD_COHERENCE_SOURCE,
    FROZEN_ATOMIC_SOURCE,
    GAIN_FRACTION_SOURCE,
    PRIOR_BACKTRACK_SOURCE,
    PRIOR_BACKTRACK_REGISTRATION,
    PRIOR_BACKTRACK_RESULT,
    METRIC_SOURCE,
    ACCEPTANCE_AMENDMENT,
    ARTICLE_N10,
    ARTICLE_N20,
    N10_RUNTIME_ROOT / "code/experiments/q10_ivm_core.py",
    N10_RUNTIME_ROOT / "code/experiments/run_q10_baselines.py",
    N10_RUNTIME_ROOT / "code/experiments/q10_potential.py",
    N10_RUNTIME_ROOT / "data/inputs/Q_data.npz",
    N20_RUNTIME_ROOT / "code/ivm/q10_ivm_core.py",
    N20_RUNTIME_ROOT / "code/baselines/run_q10_baselines.py",
    N20_RUNTIME_ROOT / "code/common/q10_potential.py",
    N20_RUNTIME_ROOT / "data/inputs/Q_data.npz",
    FORMAL_LEPS_OPTIMIZER,
    FORMAL_LEPS_REGISTRY,
)


class ContractError(RuntimeError):
    """Raised when a formal run drifts from the preregistered contract."""


class BudgetReached(RuntimeError):
    """Raised before an uncached internal-image query would exceed budget."""


class CertificationFailure(RuntimeError):
    """Raised when a proximal main solve violates its monotonic certificate."""


@dataclass(frozen=True)
class Controls:
    n_internal: int
    alpha: float
    r: float
    m_max: int

    def validate(self) -> None:
        if int(self.m_max) < 4:
            raise ContractError("m_max must be at least 4")
        key = (
            int(self.n_internal),
            float(self.alpha),
            float(self.r),
            int(self.m_max),
        )
        if key not in REGISTERED_CONTROLS:
            raise ContractError("controls are outside the shared registered grid")

    def as_dict(self) -> dict:
        return {
            "n_internal": int(self.n_internal),
            "alpha": float(self.alpha),
            "alpha_label": self.alpha_label,
            "r": float(self.r),
            "m_max": int(self.m_max),
        }

    @property
    def alpha_label(self) -> str:
        for label, value in zip(ALPHA_LABELS, ALPHAS):
            if float(self.alpha) == float(value):
                return label
        raise ContractError("alpha lacks a registered integer-mantissa label")

    @property
    def run_id(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class Ledger:
    limit: int
    total: int = 0
    blocked_misses: int = 0
    evaluation_miss_trace: list = field(default_factory=list)
    trace_outer: int = -1
    trace_phase: str = "unassigned"


class Evaluator:
    """Exact-coordinate cache with one global article-NoF ledger."""

    def __init__(
        self,
        energy_gradient,
        ledger: Ledger,
        charged: bool,
        image_index=None,
    ):
        self.energy_gradient = energy_gradient
        self.ledger = ledger
        self.charged = bool(charged)
        self.image_index = -1 if image_index is None else int(image_index)
        self.calls = 0
        self.cache: dict[bytes, tuple[float, np.ndarray]] = {}

    def query(self, value):
        point = np.ascontiguousarray(value, dtype=np.float64)
        key = point.tobytes()
        cached = self.cache.get(key)
        if cached is not None:
            return float(cached[0]), cached[1].copy()
        if self.charged and self.ledger.total >= self.ledger.limit:
            self.ledger.blocked_misses += 1
            raise BudgetReached("global Number-of-Forces budget reached")
        energy, gradient = self.energy_gradient(point)
        gradient = np.asarray(gradient, dtype=np.float64)
        self.cache[key] = (float(energy), gradient.copy())
        if self.charged:
            self.calls += 1
            self.ledger.total += 1
            self.ledger.evaluation_miss_trace.append(
                {
                    "sequence": int(len(self.ledger.evaluation_miss_trace) + 1),
                    "article_nf": int(self.ledger.total),
                    "outer": int(self.ledger.trace_outer),
                    "phase": str(self.ledger.trace_phase),
                    "image_index": int(self.image_index),
                    "coordinate_sha256": hashlib.sha256(key).hexdigest(),
                }
            )
        return float(energy), gradient.copy()


_BACKEND_CACHE = {}
_VALIDATED_REGISTRATION = None

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def validate_single_thread_environment() -> None:
    """Reject runs before NumPy work when deterministic thread gates drift."""

    for name in THREAD_ENVIRONMENT:
        if os.environ.get(name) != "1":
            raise ContractError(name + " must equal 1")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ContractError("PYTHONHASHSEED must equal 0")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_resolution_gate_token(controls, token):
    """Fail before optimization unless the staged-resolution token is current."""

    resolution = int(controls.n_internal)
    if resolution == 20:
        if token != "not_applicable":
            raise ContractError("N20 gate token must be not_applicable")
        return "not_applicable"
    if resolution != 10:
        raise ContractError("unsupported resolution for staged gate token")
    if (
        not isinstance(token, str)
        or len(token) != 64
        or any(character not in "0123456789abcdef" for character in token)
    ):
        raise ContractError("N10 requires a lowercase 64-hex N20 gate token")
    if not N20_GATE_MANIFEST.is_file():
        raise ContractError("N10 N20-gate manifest is missing")
    if _sha256(N20_GATE_MANIFEST) != token:
        raise ContractError("N10 N20-gate manifest token is stale")
    return token


def _load_backend(n_internal: int):
    key = int(n_internal)
    if key not in _BACKEND_CACHE:
        core, potential = load_solver_only_q10(REPOSITORY_ROOT, key)
        if int(core.N_INTERNAL) != key:
            raise ContractError("QM10 backend resolution drifted")
        _BACKEND_CACHE[key] = (core, potential)
    return _BACKEND_CACHE[key]


def validate_preregistration() -> dict:
    global _VALIDATED_REGISTRATION
    # Recheck on every formal member so a concurrent source or dependency
    # change cannot hide behind one process's earlier successful validation.
    payload = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    if payload.get("campaign_id") != EXPERIMENT_ID:
        raise ContractError("campaign ID drifted")
    if payload.get("registered_before_first_optimization_run") is not True:
        raise ContractError("campaign was not preregistered")
    expected_axes = {
        "alpha": list(ALPHAS),
        "alpha_labels": list(ALPHA_LABELS),
        "r": list(R_VALUES),
        "m_max": list(M_MAX_VALUES),
    }
    if payload.get("shared_grid_for_both_resolutions") != expected_axes:
        raise ContractError("registered control grid drifted")
    if payload.get("resolutions") != list(RESOLUTIONS):
        raise ContractError("registered resolutions drifted")
    if int(payload.get("registered_member_count", -1)) != len(REGISTERED_CONTROLS):
        raise ContractError("registered member count drifted")
    if int(payload.get("number_of_forces_budget_per_run", -1)) != ARTICLE_NF_BUDGET:
        raise ContractError("registered Number-of-Forces budget drifted")
    if int(payload.get("outer_iteration_safeguard", -1)) != OUTER_LIMIT:
        raise ContractError("registered outer safeguard drifted")
    if payload.get("implementation_sha256") != _sha256(THIS_FILE):
        raise ContractError("registered implementation hash drifted")
    dependencies = payload.get("dependency_sha256")
    required_dependency_names = {str(path) for path in REQUIRED_DEPENDENCY_PATHS}
    if not isinstance(dependencies, dict) or set(dependencies) != required_dependency_names:
        raise ContractError("registered dependency set drifted")
    for absolute, expected in dependencies.items():
        if _sha256(Path(absolute)) != expected:
            raise ContractError("registered dependency drifted: " + absolute)
    execution = payload.get("execution", {})
    required_execution = {
        "jobs": 1,
        "host_local_fcntl_lock": (
            "locks/host-local-q10-irivm-low-r-coherence-secant-v2.lock"
        ),
        "single_thread_environment_required": True,
        "same_source_and_control_grid_at_both_resolutions": True,
        "separate_resolution_progress_reports": True,
        "n10_core_gate_manifest": (
            "reports/n20_gate_audit/audit_manifest.json"
        ),
        "n10_gate_token_is_current_manifest_sha256": True,
        "n20_gate_token": "not_applicable",
    }
    for name, expected in required_execution.items():
        if execution.get(name) != expected:
            raise ContractError("registered execution identity drifted: " + name)
    identity = payload.get("fixed_algorithm_identity", {})
    required = {
        "tangent": "Henkelman-Jonsson improved everywhere",
        "main_subproblem": "complete-normal proximal Armijo L-BFGS",
        "main_commit": "whole-band I_perp-monotone atomic commit or rollback",
        "reparameterization": "improved-tangent identical-spring after every outer stage",
        "reparameterization_period": REPARAMETERIZATION_PERIOD,
        "recovery": "one accepted-history or same-anchor half-backtracked algebraic raw proposal per outer, adjacent-chord-coherent saddle seed, direct-energy improved-tangent chord-Newton projection, one evaluated projected band, whole-band atomic acceptance",
        "direct_acceptance_reference": "post-main I_perp",
        "recovery_acceptance_reference": "post-main I_perp or positive actual proximal merit",
        "recovery_uses_pre_main_memory_snapshot_after_main_rollback": True,
        "recovery_updates_main_memory_only_after_complete_commit": True,
        "rollback_commit_memory_transport": "publish the accepted pre-main snapshot plus coherence-conditioned curvature-certified direct-to-projected recovery secants; rejected recovery trials never alter main memory",
        "recovery_uses_warm_pairs": True,
        "rejected_recovery_trial": "private actual full-step displacement carried only across a bitwise-identical direct anchor",
        "recovery_direction_selection": "accepted-history regularized predicted decrease for the first proposal; retries multiply the stored actual full-step displacement by BACKTRACK**completed_attempts without an extra force call",
        "recovery_backtrack_factor": BACKTRACK,
        "recovery_projection": "zero-NoF adjacent-chord-coherent saddle seed followed by direct-anchor-energy improved-tangent Newton correction of adjacent chord-length constraints to the original equidistance ceiling; only the projected band is evaluated",
        "recovery_chord_coherence_rule": "(m_max-1)*sqrt(1-dot(u_minus,u_plus)) <= 1",
        "recovery_chord_coherence_strong_weight": "m_max-1",
        "recovery_chord_coherence_conservative_weight": 1,
        "stationarity_definition": "acceptable unchanged direct band converges only after ordinary normal recovery exhausts its m_max slots",
        "recovery_acceptance": "whole projected band atomically accepted when equidistant and either post-main-I_perp-monotone or positive in actual proximal merit",
        "recovery_escape_merit": "sum_internal(E_direct-E_projected-||x_projected-x_direct||^2/(2*alpha))",
        "accepted_recovery_secants": "after a complete recovery commit certify direct-to-projected physical secants satisfying s.y > 1e-6 ||s|| ||y||; strong publishes normalized-curvature top one while conservative and ordinary publish all certified pairs",
        "recovery_commit_requires_complete_band": True,
        "maximum_recovery_bands_per_outer": RECOVERY_BAND_EVALUATION_CAP,
        "same_anchor_recovery_attempt_cap": "m_max completed fixed-factor recovery slots at one direct anchor; paid misses are recorded separately",
        "recovery_reads_main_r": False,
        "recovery_reads_main_m_max": True,
        "controls": "alpha, r, m_max only",
        "new_tunable_controls": False,
        "resolution_specific_algorithm_branch": False,
        "nof_or_training_phase_specific_branch": False,
        "transaction_call_cap_per_image_including_direct": TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT,
        "optimizer_reads_v2p20_opponent": False,
        "optimizer_reads_reference_barrier": False,
        "optimizer_reads_densified_delta_e": False,
        "q10_trust_adapter": "rho=min(2*median(||segment||_2/sqrt(10)),1); max_atomic_norm=max_j|d_j|",
    }
    for name, expected in required.items():
        if identity.get(name) != expected:
            raise ContractError("registered algorithm identity drifted: " + name)
    recovery_contract = payload.get("projected_recovery_contract", {})
    required_recovery_contract = {
        "fractions_for_m_max_4": [1.0, 0.5, 0.25, 0.125],
        "slot_rule": "a complete projected fraction advances one slot even on all-cache hits; partial budget fractions do not advance",
        "maximum_recovery_band_evaluations_per_outer": RECOVERY_BAND_EVALUATION_CAP,
        "maximum_transaction_calls_per_image_including_direct": TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT,
        "raw_proposal_evaluated": False,
        "projection_operator": "zero-NoF adjacent-chord-coherent saddle seed followed by Henkelman-Jonsson improved-tangent Newton correction of adjacent chord-length constraints using fixed direct-anchor energies",
        "adjacent_chord_gate_is_not_a_tangent_operator": True,
        "stationary_probe_uses_ordinary_normal_seed": True,
        "projection_target": "original max(direct_eq,0.2*pre_eq) ceiling",
        "only_projected_candidate_is_evaluated": True,
        "whole_band_atomic_merge": True,
        "accepted_secant_curvature_threshold": RECOVERY_SECANT_CURVATURE,
        "atomic_commit_gate": "equidistant and finite and changed and (post-main I_perp monotone or positive actual proximal merit)",
        "actual_proximal_merit_formula": "sum_internal(E_direct-E_projected-||x_projected-x_direct||^2/(2*alpha))",
        "actual_proximal_merit_reads_r": False,
        "complete_projected_candidate_preserved_for_offline_audit": True,
        "normalized_curvature_formula": "(s^T y)/(||s|| ||y||)",
        "strong_secant_publication": "highest normalized-curvature certified pair only",
        "conservative_secant_publication": "all curvature-certified pairs",
        "ordinary_secant_publication": "all curvature-certified pairs",
        "secant_tie_break": "smaller 1-based internal image index",
        "secant_index_convention": "1-based internal path image index",
        "rank_omitted_is_distinct_from_curvature_rejected": True,
        "publication_only_after_complete_atomic_commit": True,
    }
    for name, expected in required_recovery_contract.items():
        if recovery_contract.get(name) != expected:
            raise ContractError("registered projected recovery contract drifted: " + name)
    nof_contract = payload.get("number_of_forces_contract", {})
    required_nof_contract = {
        "unit": "one uncached joint energy-gradient evaluation at one internal image",
        "endpoints_charged": False,
        "exact_coordinate_cache": True,
        "raw_recovery_path_charged": False,
        "projected_recovery_path_charged": True,
        "rejected_and_rolled_back_trials_charged": True,
        "partial_budget_recovery_creates_or_advances_state": False,
        "backtrack_fraction_selection_charged": False,
    }
    for name, expected in required_nof_contract.items():
        if nof_contract.get(name) != expected:
            raise ContractError("registered Number-of-Forces contract drifted: " + name)
    _VALIDATED_REGISTRATION = payload
    return payload


def make_evaluators(core, potential, ledger):
    return [
        Evaluator(
            potential.energy_gradient,
            ledger,
            charged=(0 < index < int(core.N_TOTAL_IMAGES) - 1),
            image_index=index,
        )
        for index in range(int(core.N_TOTAL_IMAGES))
    ]


def set_evaluation_trace_context(ledger, outer, phase):
    """Label subsequent miss records without changing evaluation control flow."""

    ledger.trace_outer = int(outer)
    ledger.trace_phase = str(phase)


def evaluate_path(core, path, evaluators):
    count = int(core.N_TOTAL_IMAGES)
    dimension = int(np.asarray(path).shape[1])
    energies = np.empty(count, dtype=np.float64)
    gradients = np.empty((count, dimension), dtype=np.float64)
    for index in range(count):
        energies[index], gradients[index] = evaluators[index].query(path[index])
    return energies, gradients


def _rounding_slack(left, right) -> float:
    return float(
        ROUNDING_FACTOR
        * np.finfo(np.float64).eps
        * max(1.0, abs(float(left)), abs(float(right)))
    )


def _max_atomic_norm(vector) -> float:
    # QM10's established ASE embedding has ten atoms with one active x DOF.
    return float(np.max(np.abs(np.asarray(vector, dtype=np.float64))))


def rms_segment_trust_radius(path) -> float:
    points = np.asarray(path, dtype=np.float64)
    dimension = float(points.shape[1])
    segment_rms = np.linalg.norm(np.diff(points, axis=0), axis=1) / np.sqrt(
        dimension
    )
    return min(
        TRUST_RADIUS_FRACTION * float(np.median(segment_rms)),
        TRUST_RADIUS_MAXIMUM,
    )


def internal_i_perp(core, path, energies, gradients):
    """The exact path-length quadrature used inside the formal LEPS method."""

    points = np.asarray(path, dtype=np.float64)
    tangents = core.improved_tangents(points, np.asarray(energies, dtype=np.float64))
    forces = -np.asarray(gradients, dtype=np.float64)[1:-1]
    normal = forces - np.sum(forces * tangents, axis=1)[:, None] * tangents
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    weights = 0.5 * (lengths[:-1] + lengths[1:])
    contributions = weights * np.sum(normal * normal, axis=1)
    return float(np.sum(contributions)), contributions


def equidistance_integral(path) -> float:
    points = np.asarray(path, dtype=np.float64)
    ds = 1.0 / float(points.shape[0] - 1)
    first = (points[2:] - points[:-2]) / (2.0 * ds)
    second = (points[2:] - 2.0 * points[1:-1] + points[:-2]) / (ds * ds)
    inner = np.sum(first * second, axis=1)
    return float(np.sum(inner * inner) * ds)


def _i_ok(value, reference) -> bool:
    return float(value) <= float(reference) + RECOVERY_IPERP_ATOL


def _eq_ok(value, ceiling) -> bool:
    return float(value) <= float(ceiling) + RECOVERY_EQUIDISTANCE_ATOL


def clone_physical_memories(memories):
    """Deep-copy accepted physical secants for an atomic outer rollback."""

    return [
        [
            (
                np.asarray(step, dtype=np.float64).copy(),
                np.asarray(change, dtype=np.float64).copy(),
            )
            for step, change in memory
        ]
        for memory in memories
    ]


def restore_physical_memories(memories, snapshot) -> None:
    """Publish only accepted pre-main physical pairs after rollback escape."""

    memories[:] = clone_physical_memories(snapshot)


def same_anchor_recovery_retry_decision(
    completed_attempts, completed_recovery_candidate, m_max
):
    """Bound the fixed line-search slots at one anchor by existing m_max."""

    attempts = int(completed_attempts)
    if int(m_max) < 1 or attempts < 0:
        raise ContractError("invalid rollback-recovery attempt state")
    if not bool(completed_recovery_candidate):
        return attempts, False
    attempts += 1
    retry = bool(attempts < int(m_max))
    return attempts, retry


def main_stage_changed_public_path(main_stage_rolled_back, old_path, new_path):
    """Distinguish a real main commit from a no-guard zero displacement."""

    return bool(
        not bool(main_stage_rolled_back)
        and not np.array_equal(
            np.asarray(old_path, dtype=np.float64),
            np.asarray(new_path, dtype=np.float64),
        )
    )


def recovery_commit_publishes_main_snapshot(main_stage_rolled_back, committed):
    """Keep the frozen wrapper rule: rejected recovery never rewrites main memory."""

    return bool(bool(main_stage_rolled_back) and bool(committed))


def recovery_anchor_matches(state, path, energies, gradients):
    """Require a bitwise-identical direct path and evaluated state."""

    if state is None:
        return False
    anchor = state.get("anchor")
    if anchor is None:
        return False
    return bool(
        np.array_equal(anchor["path"], path)
        and np.array_equal(anchor["energies"], energies)
        and np.array_equal(anchor["gradients"], gradients)
    )


def make_recovery_backtrack_state(
    path,
    energies,
    gradients,
    direction_candidate,
    completed_attempts,
):
    """Store only recovery's rejected full displacement and direct anchor."""

    attempts = int(completed_attempts)
    if attempts < 1:
        raise ContractError("a recovery retry state requires a completed fraction")
    directions = np.asarray(direction_candidate["directions"], dtype=np.float64)
    base_step = float(
        direction_candidate.get(
            "base_step_length", direction_candidate["step_length"]
        )
    )
    if not np.all(np.isfinite(directions)) or not np.isfinite(base_step):
        raise ContractError("non-finite recovery retry state")
    if base_step <= 0.0:
        raise ContractError("non-positive recovery retry base step")
    base_displacement = np.asarray(
        direction_candidate.get(
            "base_displacement", base_step * directions
        ),
        dtype=np.float64,
    )
    if (
        base_displacement.shape != directions.shape
        or not np.all(np.isfinite(base_displacement))
    ):
        raise ContractError("invalid recovery retry base displacement")
    return {
        "anchor": {
            "path": np.asarray(path, dtype=np.float64).copy(),
            "energies": np.asarray(energies, dtype=np.float64).copy(),
            "gradients": np.asarray(gradients, dtype=np.float64).copy(),
        },
        "directions": directions.copy(),
        "base_displacement": base_displacement.copy(),
        "base_step_length": base_step,
        "base_predicted_decrease": float(
            direction_candidate.get(
                "base_predicted_decrease",
                direction_candidate["predicted_decrease"],
            )
        ),
        "g_dot_d": float(direction_candidate["g_dot_d"]),
        "norm_squared": float(direction_candidate["norm_squared"]),
        "warm_pair_count": int(direction_candidate["warm_pair_count"]),
        "completed_attempts": attempts,
    }


def recovery_backtrack_candidate(state, alpha):
    """Reuse one physical direction at the next fixed backtracking fraction."""

    if state is None:
        raise ContractError("missing recovery backtrack state")
    completed = int(state["completed_attempts"])
    if completed < 1:
        raise ContractError("invalid completed recovery-attempt count")
    fraction = float(BACKTRACK) ** completed
    step_length = float(state["base_step_length"]) * fraction
    g_dot_d = float(state["g_dot_d"])
    norm_squared = float(state["norm_squared"])
    predicted_decrease = float(
        -step_length * g_dot_d
        - (step_length * step_length * norm_squared) / (2.0 * float(alpha))
    )
    if (
        not np.isfinite(step_length)
        or not np.isfinite(predicted_decrease)
        or step_length <= 0.0
        or predicted_decrease <= 0.0
    ):
        raise ContractError("invalid recovery backtrack candidate")
    return {
        "name": "same_anchor_backtrack",
        "directions": np.asarray(state["directions"], dtype=np.float64).copy(),
        "base_displacement": np.asarray(
            state["base_displacement"], dtype=np.float64
        ).copy(),
        "g_dot_d": g_dot_d,
        "norm_squared": norm_squared,
        "step_length": step_length,
        "predicted_decrease": predicted_decrease,
        "warm_pair_count": int(state["warm_pair_count"]),
        "backtrack_fraction": fraction,
        "base_step_length": float(state["base_step_length"]),
        "base_predicted_decrease": float(state["base_predicted_decrease"]),
    }


def recovery_direction_candidate(
    core,
    path,
    energies,
    gradients,
    memories,
    alpha,
    trust_radius,
    name,
):
    """Construct one zero-NoF direction and its fair regularized prediction."""

    tangents = core.improved_tangents(path, energies)
    directions = np.zeros_like(path)
    g_dot_d = 0.0
    norm_squared = 0.0
    warm_pair_count = 0
    for image_index in range(1, int(core.N_TOTAL_IMAGES) - 1):
        tangent = tangents[image_index - 1]
        basis = core.normal_basis(tangent)
        reduced_gradient = basis.T @ gradients[image_index]
        apply_h0 = core._harmonic_preconditioner(  # noqa: SLF001
            basis, float(alpha), 0.0, True
        )
        warm_pairs = core._usable_warm_pairs(  # noqa: SLF001
            memories[image_index - 1], basis, float(alpha), HISTORY_SIZE
        )
        warm_pair_count += int(len(warm_pairs))
        reduced_direction = core._lbfgs_direction(  # noqa: SLF001
            reduced_gradient, warm_pairs, apply_h0
        )
        if float(np.dot(reduced_gradient, reduced_direction)) >= 0.0:
            reduced_direction = -np.asarray(
                apply_h0(reduced_gradient), dtype=np.float64
            )
        direction = basis @ reduced_direction
        largest = _max_atomic_norm(direction)
        if largest > float(trust_radius):
            direction *= float(trust_radius) / largest
        directions[image_index] = direction
        g_dot_d += float(np.dot(gradients[image_index], direction))
        norm_squared += float(np.dot(direction, direction))

    step_length = 0.0
    predicted_decrease = 0.0
    if norm_squared > np.finfo(np.float64).tiny and g_dot_d < 0.0:
        step_length = float(
            np.clip(-float(alpha) * g_dot_d / norm_squared, 0.0, 1.0)
        )
        predicted_decrease = float(
            -step_length * g_dot_d
            - (step_length * step_length * norm_squared) / (2.0 * float(alpha))
        )
        if not np.isfinite(predicted_decrease) or predicted_decrease <= 0.0:
            step_length = 0.0
            predicted_decrease = 0.0
    return {
        "name": str(name),
        "directions": directions,
        "g_dot_d": float(g_dot_d),
        "norm_squared": float(norm_squared),
        "step_length": float(step_length),
        "predicted_decrease": float(predicted_decrease),
        "warm_pair_count": int(warm_pair_count),
    }


def solve_normal_image(
    core,
    evaluator,
    reference,
    tangent,
    controls,
    memory,
    *,
    trust_radius,
):
    """Formal-LEPS complete-normal solve under one total trust ball."""

    accepted_step_cap = int(controls.m_max)
    new_call_cap = max(12, 4 * int(controls.m_max))
    basis = core.normal_basis(tangent)
    displacement = np.zeros(basis.shape[1], dtype=np.float64)
    value, reduced, energy, gradient, position = core._proximal_query(  # noqa: SLF001
        evaluator, reference, basis, float(controls.alpha), displacement
    )
    zero_value = float(value)
    initial_kkt = float(np.linalg.norm(reduced))
    target = max(KKT_ATOL, float(controls.r) * initial_kkt)
    start_calls = evaluator.calls
    apply_h0 = core._harmonic_preconditioner(  # noqa: SLF001
        basis, float(controls.alpha), 0.0, True
    )
    warm_pairs = core._usable_warm_pairs(  # noqa: SLF001
        memory, basis, float(controls.alpha), HISTORY_SIZE
    )
    local_pairs = []
    accepted = 0
    backtracks = 0
    hit_boundary = False

    while accepted < accepted_step_cap:
        if np.linalg.norm(reduced) <= target and accepted >= 1:
            break
        if evaluator.calls - start_calls >= new_call_cap:
            break
        pairs = (warm_pairs + local_pairs)[-HISTORY_SIZE:]
        direction = core._lbfgs_direction(reduced, pairs, apply_h0)  # noqa: SLF001
        directional = float(np.dot(reduced, direction))
        if directional >= 0.0:
            warm_pairs = []
            local_pairs = []
            direction = -np.asarray(apply_h0(reduced), dtype=np.float64)
            directional = float(np.dot(reduced, direction))
        if directional >= 0.0:
            break

        accepted_trial = False
        step_length = 1.0
        for _ in range(64):
            if evaluator.calls - start_calls >= new_call_cap:
                break
            trial_displacement = displacement + step_length * direction
            largest = _max_atomic_norm(basis @ trial_displacement)
            if largest > float(trust_radius):
                trial_displacement *= float(trust_radius) / largest
                hit_boundary = True
            effective = trial_displacement - displacement
            effective_directional = float(np.dot(reduced, effective))
            if np.linalg.norm(effective) <= 1.0e-18 or effective_directional >= 0.0:
                step_length *= BACKTRACK
                backtracks += 1
                continue
            trial = core._proximal_query(  # noqa: SLF001
                evaluator,
                reference,
                basis,
                float(controls.alpha),
                trial_displacement,
            )
            trial_value, trial_reduced, trial_energy, trial_gradient, trial_position = trial
            slack = _rounding_slack(value, trial_value)
            if (
                trial_value <= value + ARMIJO * effective_directional + slack
                and trial_value <= zero_value + slack
            ):
                old_displacement = displacement.copy()
                old_reduced = np.asarray(reduced).copy()
                old_position = np.asarray(position).copy()
                old_gradient = np.asarray(gradient).copy()
                displacement = trial_displacement
                value = float(trial_value)
                reduced = np.asarray(trial_reduced, dtype=np.float64)
                energy = float(trial_energy)
                gradient = np.asarray(trial_gradient, dtype=np.float64)
                position = np.asarray(trial_position, dtype=np.float64)
                s = displacement - old_displacement
                y = reduced - old_reduced
                if np.dot(s, y) > 1.0e-6 * np.linalg.norm(s) * np.linalg.norm(y):
                    local_pairs.append((s.copy(), y.copy()))
                    local_pairs[:] = local_pairs[-HISTORY_SIZE:]
                physical_s = position - old_position
                physical_y = gradient - old_gradient
                if np.dot(physical_s, physical_y) > (
                    -0.99 * np.dot(physical_s, physical_s) / float(controls.alpha)
                ):
                    memory.append((physical_s.copy(), physical_y.copy()))
                    memory[:] = memory[-HISTORY_SIZE:]
                accepted += 1
                accepted_trial = True
                break
            step_length *= BACKTRACK
            backtracks += 1
            if step_length < 2.0**-40:
                break
        if not accepted_trial:
            break

    if value > zero_value + _rounding_slack(zero_value, value):
        raise CertificationFailure("normal proximal objective increased")
    return position, energy, gradient, {
        "initial_kkt": initial_kkt,
        "final_kkt": float(np.linalg.norm(reduced)),
        "accepted_steps": accepted,
        "new_calls": evaluator.calls - start_calls,
        "backtracks": backtracks,
        "target_fraction": float(controls.r),
        "total_atomic_trust_radius": float(trust_radius),
        "hit_boundary": bool(hit_boundary),
    }


def solve_main_stage(core, path, energies, gradients, evaluators, memories, controls):
    tangents = core.improved_tangents(path, energies)
    trust_radius = rms_segment_trust_radius(path)
    proposed = np.asarray(path, dtype=np.float64).copy()
    proposed_energies = np.asarray(energies, dtype=np.float64).copy()
    proposed_gradients = np.asarray(gradients, dtype=np.float64).copy()
    audits = []
    for image_index in range(1, int(core.N_TOTAL_IMAGES) - 1):
        position, energy, gradient, audit = solve_normal_image(
            core,
            evaluators[image_index],
            path[image_index],
            tangents[image_index - 1],
            controls,
            memories[image_index - 1],
            trust_radius=trust_radius,
        )
        proposed[image_index] = position
        proposed_energies[image_index] = energy
        proposed_gradients[image_index] = gradient
        audits.append(audit)

    reference_i, _ = internal_i_perp(core, path, energies, gradients)
    candidate_i, _ = internal_i_perp(
        core, proposed, proposed_energies, proposed_gradients
    )
    rolled_back = bool(candidate_i > reference_i)
    if rolled_back:
        for memory in memories:
            memory.clear()
        proposed = np.asarray(path, dtype=np.float64).copy()
        proposed_energies = np.asarray(energies, dtype=np.float64).copy()
        proposed_gradients = np.asarray(gradients, dtype=np.float64).copy()
    return proposed, proposed_energies, proposed_gradients, {
        "trust_radius": float(trust_radius),
        "pre_i_perp": float(reference_i),
        "candidate_i_perp": float(candidate_i),
        "rolled_back_by_i_perp_guard": rolled_back,
        "image_audits": audits,
    }


def improved_spring_reparameterize(core, path, energies):
    result = np.asarray(path, dtype=np.float64).copy()
    fixed_energies = np.asarray(energies, dtype=np.float64)
    residual = np.inf
    iterations = 0
    for iteration in range(SPRING_MAX_ITER):
        lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)
        tangents = core.improved_tangents(result, fixed_energies)
        spring_force = (lengths[1:] - lengths[:-1])[:, None] * tangents
        residual = float(np.max(np.linalg.norm(spring_force, axis=1)))
        iterations = iteration + 1
        if residual <= SPRING_TOLERANCE:
            break
        result[1:-1] += SPRING_DTAU * spring_force
    result[0] = path[0]
    result[-1] = path[-1]
    return result, residual, iterations


def polyline_equal_arclength(path):
    """Zero-NoF equal-arclength interpolation of the recovery raw path."""

    points = np.asarray(path, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if (
        np.any(~np.isfinite(lengths))
        or np.any(lengths <= np.finfo(np.float64).tiny)
    ):
        return None
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    targets = np.linspace(0.0, float(cumulative[-1]), points.shape[0])
    result = np.empty_like(points)
    result[0] = points[0]
    result[-1] = points[-1]
    for image, target in enumerate(targets[1:-1], start=1):
        edge = min(
            int(np.searchsorted(cumulative, target, side="right") - 1),
            lengths.size - 1,
        )
        fraction = (target - cumulative[edge]) / lengths[edge]
        result[image] = points[edge] + fraction * (
            points[edge + 1] - points[edge]
        )
    return result


def chord_coherent_recovery_seed(core, raw_path, direct_energies, controls):
    """Build the zero-NoF recovery seed from adjacent-chord coherence.

    Adjacent chords are used only as a local bend diagnostic.  They are not a
    tangent approximation; all tangent operators remain ``improved_tangents``.
    """

    raw = np.asarray(raw_path, dtype=np.float64)
    energies = np.asarray(direct_energies, dtype=np.float64)
    polyline = polyline_equal_arclength(raw)
    spring, spring_residual, spring_iterations = improved_spring_reparameterize(
        core, raw, energies
    )
    saddle = int(np.argmax(energies[1:-1])) + 1
    if polyline is None:
        return spring, {
            "mode": "spring_fallback_degenerate_polyline",
            "saddle_image": saddle,
            "adjacent_chord_cosine": None,
            "coherence_loss": None,
            "coherence_loss_limit": float(
                1.0 / float(int(controls.m_max) - 1) ** 2
            ),
            "amplified_direction_mismatch": None,
            "strong_recovery_selected": False,
            "saddle_spring_weight": 1.0,
            "spring_residual": float(spring_residual),
            "spring_iterations": int(spring_iterations),
        }

    left = np.asarray(polyline[saddle] - polyline[saddle - 1])
    right = np.asarray(polyline[saddle + 1] - polyline[saddle])
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    denominator = left_norm * right_norm
    if not np.isfinite(denominator) or denominator <= np.finfo(np.float64).tiny:
        cosine = -1.0
    else:
        cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    loss = float(1.0 - cosine)
    strong_horizon = float(int(controls.m_max) - 1)
    strong = bool(strong_horizon * np.sqrt(max(0.0, loss)) <= 1.0)
    weight = float(int(controls.m_max) - 1 if strong else 1)
    seeded = polyline.copy()
    seeded[saddle] = polyline[saddle] + weight * (
        spring[saddle] - polyline[saddle]
    )
    seeded[0], seeded[-1] = raw[0], raw[-1]
    return seeded, {
        "mode": "polyline_spring_saddle_adjacent_chord_coherence",
        "saddle_image": saddle,
        "adjacent_chord_cosine": cosine,
        "coherence_loss": loss,
        "coherence_loss_limit": float(1.0 / strong_horizon**2),
        "amplified_direction_mismatch": float(
            strong_horizon * np.sqrt(max(0.0, loss))
        ),
        "strong_recovery_selected": strong,
        "saddle_spring_weight": weight,
        "spring_residual": float(spring_residual),
        "spring_iterations": int(spring_iterations),
    }


def chord_constraints_and_jacobian(path):
    """Adjacent chord-length residuals and their internal-image Jacobian."""

    points = np.asarray(path, dtype=np.float64)
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    units = segments / np.maximum(lengths[:, None], np.finfo(np.float64).tiny)
    n_internal = points.shape[0] - 2
    dimension = points.shape[1]
    constraints = lengths[1:] - lengths[:-1]
    jacobian = np.zeros(
        (n_internal, n_internal * dimension), dtype=np.float64
    )
    for row in range(n_internal):
        image = row + 1
        previous = units[image - 1]
        following = units[image]
        if image - 1 >= 1:
            start = (image - 2) * dimension
            jacobian[row, start : start + dimension] += previous
        start = (image - 1) * dimension
        jacobian[row, start : start + dimension] += -following - previous
        if image + 1 <= n_internal:
            start = image * dimension
            jacobian[row, start : start + dimension] += following
    return constraints, jacobian


def tangent_chord_constraint_project(core, raw_path, direct_energies, ceiling):
    """Zero-NoF chord projection along HJ improved tangents.

    The energy ordering is frozen at the already evaluated direct anchor.
    Newton corrections use only geometry and that fixed ordering; no raw-path
    energy or gradient is queried.
    """

    path = np.asarray(raw_path, dtype=np.float64).copy()
    fixed_energies = np.asarray(direct_energies, dtype=np.float64).copy()
    endpoints = (path[0].copy(), path[-1].copy())
    backtracks = 0
    for iteration in range(SPRING_MAX_ITER + 1):
        constraints, jacobian = chord_constraints_and_jacobian(path)
        residual = float(np.linalg.norm(constraints))
        if _eq_ok(equidistance_integral(path), ceiling):
            return path, residual, iteration, backtracks, True
        if iteration == SPRING_MAX_ITER:
            break
        tangents = core.improved_tangents(path, fixed_energies)
        n_internal = path.shape[0] - 2
        dimension = path.shape[1]
        basis = np.zeros(
            (n_internal * dimension, n_internal), dtype=np.float64
        )
        for index, tangent in enumerate(tangents):
            start = index * dimension
            basis[start : start + dimension, index] = tangent
        reduced_jacobian = jacobian @ basis
        coefficients = np.linalg.lstsq(
            reduced_jacobian, -constraints, rcond=None
        )[0]
        correction = (basis @ coefficients).reshape(n_internal, dimension)
        if not np.all(np.isfinite(correction)):
            break
        accepted = False
        fraction = 1.0
        for _ in range(64):
            trial = path.copy()
            trial[1:-1] += fraction * correction
            trial[0], trial[-1] = endpoints
            trial_constraints, _ = chord_constraints_and_jacobian(trial)
            if np.linalg.norm(trial_constraints) < residual:
                path = trial
                accepted = True
                break
            fraction *= BACKTRACK
            backtracks += 1
        if not accepted:
            break
    constraints, _ = chord_constraints_and_jacobian(path)
    return (
        path,
        float(np.linalg.norm(constraints)),
        min(SPRING_MAX_ITER, iteration),
        backtracks,
        _eq_ok(equidistance_integral(path), ceiling),
    )


def atomic_recovery_merge(
    core,
    current_state,
    candidate_state,
    ceiling,
    alpha,
    post_main_i,
):
    """Apply one r-free atomic residual-or-proximal recovery gate."""

    current = tuple(
        np.asarray(item, dtype=np.float64).copy() for item in current_state
    )
    candidate = tuple(
        np.asarray(item, dtype=np.float64).copy() for item in candidate_state
    )
    changed = np.linalg.norm(
        candidate[0][1:-1] - current[0][1:-1], axis=1
    ) > 1.0e-14
    candidate_i, _ = internal_i_perp(core, *candidate)
    candidate_eq = equidistance_integral(candidate[0])
    displacement = candidate[0][1:-1] - current[0][1:-1]
    actual_proximal_merit = float(
        np.sum(current[1][1:-1] - candidate[1][1:-1])
        - np.sum(displacement * displacement) / (2.0 * float(alpha))
    )
    finite = bool(
        np.isfinite(candidate_i)
        and np.isfinite(candidate_eq)
        and np.isfinite(actual_proximal_merit)
        and np.all(np.isfinite(candidate[0]))
        and np.all(np.isfinite(candidate[1]))
        and np.all(np.isfinite(candidate[2]))
    )
    monotone = bool(finite and _i_ok(candidate_i, float(post_main_i)))
    merit_escape = bool(finite and actual_proximal_merit > 0.0)
    accepted_by_gate = bool(
        finite
        and np.any(changed)
        and _eq_ok(candidate_eq, ceiling)
        and (monotone or merit_escape)
    )
    audit = {
        "candidate_i_perp": float(candidate_i),
        "candidate_equidistance": float(candidate_eq),
        "actual_proximal_merit": float(actual_proximal_merit),
        "finite": finite,
        "monotone_to_post_main_i_perp": monotone,
        "positive_actual_proximal_merit": merit_escape,
        "accepted_by_atomic_gate": accepted_by_gate,
    }
    if accepted_by_gate:
        return (
            candidate,
            np.ones(current[0].shape[0] - 2, dtype=bool),
            audit,
        )
    return (
        current,
        np.zeros(current[0].shape[0] - 2, dtype=bool),
        audit,
    )


def publish_accepted_recovery_secants(
    memories,
    direct_state,
    projected_state,
    *,
    strong_recovery_selected,
    stationary_normal_probe,
):
    """Publish certified physical pairs under this candidate's explicit branch.

    The caller passes the branch selected while constructing this same recovery
    candidate.  No module-global state, path identity, resolution, stage, NoF,
    or opponent data participate in publication.
    """

    strong = bool(strong_recovery_selected)
    stationary = bool(stationary_normal_probe)
    if strong and stationary:
        raise ContractError("stationary recovery cannot select strong publication")
    if stationary:
        publication_mode = "ordinary_all"
    elif strong:
        publication_mode = "strong_top1"
    else:
        publication_mode = "conservative_all"

    direct_path, _, direct_gradients = direct_state
    projected_path, _, projected_gradients = projected_state
    candidates = []
    rejected_indices = []
    for image_index in range(1, np.asarray(direct_path).shape[0] - 1):
        step = np.asarray(projected_path[image_index], dtype=np.float64) - np.asarray(
            direct_path[image_index], dtype=np.float64
        )
        change = np.asarray(
            projected_gradients[image_index], dtype=np.float64
        ) - np.asarray(direct_gradients[image_index], dtype=np.float64)
        step_norm = float(np.linalg.norm(step))
        change_norm = float(np.linalg.norm(change))
        curvature = float(np.dot(step.ravel(), change.ravel()))
        threshold = float(
            RECOVERY_SECANT_CURVATURE * step_norm * change_norm
        )
        if (
            step_norm > 0.0
            and change_norm > 0.0
            and curvature > threshold
        ):
            candidates.append(
                {
                    "image_index": int(image_index),
                    "step": step,
                    "change": change,
                    "normalized_curvature": float(
                        curvature / (step_norm * change_norm)
                    ),
                }
            )
        else:
            rejected_indices.append(int(image_index))

    if publication_mode == "strong_top1" and candidates:
        published = [
            min(
                candidates,
                key=lambda item: (
                    -item["normalized_curvature"],
                    item["image_index"],
                ),
            )
        ]
        published_index_set = {published[0]["image_index"]}
        omitted = [
            item
            for item in candidates
            if item["image_index"] not in published_index_set
        ]
    else:
        published = list(candidates)
        omitted = []

    published.sort(key=lambda item: item["image_index"])
    omitted.sort(key=lambda item: item["image_index"])
    for item in published:
        memory = memories[item["image_index"] - 1]
        memory.append((item["step"].copy(), item["change"].copy()))
        memory[:] = memory[-HISTORY_SIZE:]

    certified_indices = sorted(item["image_index"] for item in candidates)
    published_indices = [item["image_index"] for item in published]
    omitted_indices = [item["image_index"] for item in omitted]
    rejected_indices.sort()
    audit = {
        "accepted_recovery_secant_publication_mode": publication_mode,
        "accepted_recovery_secants_certified_curvature": len(candidates),
        "accepted_recovery_secant_certified_indices": certified_indices,
        "accepted_recovery_secants_rejected_curvature": len(rejected_indices),
        "accepted_recovery_secant_rejected_indices": rejected_indices,
        "accepted_recovery_secants_omitted_by_rank": len(omitted),
        "accepted_recovery_secant_omitted_indices": omitted_indices,
        "accepted_recovery_secants_added": len(published),
        "accepted_recovery_secant_published_indices": published_indices,
        "accepted_recovery_secant_published_normalized_curvatures": [
            float(item["normalized_curvature"]) for item in published
        ],
    }
    if (
        audit["accepted_recovery_secants_certified_curvature"]
        != audit["accepted_recovery_secants_added"]
        + audit["accepted_recovery_secants_omitted_by_rank"]
        or np.asarray(direct_path).shape[0] - 2
        != audit["accepted_recovery_secants_added"]
        + audit["accepted_recovery_secants_omitted_by_rank"]
        + audit["accepted_recovery_secants_rejected_curvature"]
    ):
        raise ContractError("accepted recovery secant publication counts do not close")
    return audit


def bicriteria_merge(core, current_state, candidate_state, ceiling):
    path, energies, gradients = (
        np.asarray(item, dtype=np.float64).copy() for item in current_state
    )
    cpath, cenergies, cgradients = candidate_state
    eligible = np.linalg.norm(cpath[1:-1] - path[1:-1], axis=1) > 1.0e-14
    current_i, _ = internal_i_perp(core, path, energies, gradients)
    accepted = np.zeros(path.shape[0] - 2, dtype=bool)
    if np.any(eligible):
        trial = (path.copy(), energies.copy(), gradients.copy())
        images = np.flatnonzero(eligible) + 1
        trial[0][images] = cpath[images]
        trial[1][images] = cenergies[images]
        trial[2][images] = cgradients[images]
        trial_i, _ = internal_i_perp(core, *trial)
        if _i_ok(trial_i, current_i) and _eq_ok(equidistance_integral(trial[0]), ceiling):
            accepted[eligible] = True
            return trial, accepted

    remaining = set(np.flatnonzero(eligible).tolist())
    n_internal = path.shape[0] - 2
    while remaining:
        best = None
        best_i = np.inf
        best_eq = np.inf
        for size in range(1, min(RECOVERY_BLOCK_SIZE, n_internal) + 1):
            for start in range(0, n_internal - size + 1):
                indices = [index for index in range(start, start + size) if index in remaining]
                if not indices:
                    continue
                trial = (path.copy(), energies.copy(), gradients.copy())
                images = np.asarray(indices, dtype=int) + 1
                trial[0][images] = cpath[images]
                trial[1][images] = cenergies[images]
                trial[2][images] = cgradients[images]
                trial_i, _ = internal_i_perp(core, *trial)
                trial_eq = equidistance_integral(trial[0])
                if not _i_ok(trial_i, current_i) or not _eq_ok(trial_eq, ceiling):
                    continue
                if trial_i < best_i - 1.0e-18 or (
                    abs(trial_i - best_i) <= 1.0e-18 and trial_eq < best_eq
                ):
                    best = (indices, trial)
                    best_i = float(trial_i)
                    best_eq = float(trial_eq)
        if best is None:
            break
        indices, (path, energies, gradients) = best
        current_i = best_i
        accepted[np.asarray(indices, dtype=int)] = True
        remaining.difference_update(indices)
    return (path, energies, gradients), accepted


def single_synchronized_recovery(
    core,
    pre,
    direct,
    evaluators,
    accepted_recovery_memories,
    recovery_backtrack_state,
    controls,
    residual,
    attempt_start_calls,
    stationary_probe_requested=False,
):
    pre_path, _, _ = pre
    path, energies, gradients = (
        np.asarray(item, dtype=np.float64).copy() for item in direct
    )
    pre_i, _ = internal_i_perp(core, *pre)
    direct_i, _ = internal_i_perp(core, *direct)
    pre_eq = equidistance_integral(pre_path)
    direct_eq = equidistance_integral(path)
    ceiling = max(direct_eq, RECOVERY_EQUIDISTANCE_CONTRACTION * pre_eq)
    attempt_start_calls = np.asarray(attempt_start_calls, dtype=np.int64)
    recovery_trust = rms_segment_trust_radius(pre_path)
    transaction = {
        "mode": "outer_composite_atomic_tangent_secant_recovery",
        "spring_residual": float(residual),
        "post_main_i_perp": float(pre_i),
        "pre_i_perp": float(pre_i),
        "acceptance_envelope_i_perp": float(pre_i),
        "direct_i_perp_acceptance_reference": "post-main I_perp",
        "recovery_i_perp_acceptance_reference": (
            "post-main I_perp or positive actual proximal merit"
        ),
        "direct_i_perp": float(direct_i),
        "pre_equidistance": float(pre_eq),
        "direct_equidistance": float(direct_eq),
        "equidistance_ceiling": float(ceiling),
        "main_r_read_by_recovery": False,
        "recovery_acceptance_uses_main_r": False,
        "recovery_geometry_uses_main_r": False,
        "regularized_prediction_uses_main_r": False,
        "warm_lbfgs_pairs_enabled": True,
        "private_rejected_secant_bank_enabled": False,
        "private_recovery_backtrack_state_entering": bool(
            recovery_backtrack_state is not None
        ),
        "direction_selection_rule": (
            "accepted_history_regularized_prediction_then_same_direction_fixed_backtrack"
        ),
        "direction_candidates": [],
        "selected_direction": "none",
        "same_anchor_recovery_retry_allowed": False,
        "recovery_backtrack_carry_allowed": False,
        "recovery_backtrack_lifecycle_action": "not_resolved",
        "recovery_backtrack_factor": float(BACKTRACK),
        "recovery_backtrack_fraction": 0.0,
        "recovery_backtrack_state_used": bool(
            recovery_backtrack_state is not None
        ),
        "main_memory_updates_only_after_complete_recovery_commit": True,
        "accepted_recovery_secants_enabled": True,
        "accepted_recovery_secant_curvature_threshold": float(
            RECOVERY_SECANT_CURVATURE
        ),
        "accepted_recovery_secant_publication_mode": "not_applicable",
        "accepted_recovery_secants_certified_curvature": 0,
        "accepted_recovery_secant_certified_indices": [],
        "accepted_recovery_secants_added": 0,
        "accepted_recovery_secants_rejected_curvature": 0,
        "accepted_recovery_secant_rejected_indices": [],
        "accepted_recovery_secants_omitted_by_rank": 0,
        "accepted_recovery_secant_omitted_indices": [],
        "accepted_recovery_secant_published_indices": [],
        "accepted_recovery_secant_published_normalized_curvatures": [],
        "recovery_commit_gate": (
            "post_main_i_monotone_or_positive_actual_proximal_merit"
        ),
        "candidate_i_perp": None,
        "candidate_equidistance": None,
        "actual_proximal_merit": None,
        "recovery_candidate_finite": False,
        "recovery_monotone_to_post_main_i_perp": False,
        "recovery_positive_actual_proximal_merit": False,
        "recovery_accepted_by_atomic_gate": False,
        "regularized_prediction_alpha_eff": float(controls.alpha),
        "transaction_call_cap_per_image_including_direct": (
            TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT
        ),
        "single_sync_recovery_joint_evaluation_cap": int(
            RECOVERY_BAND_EVALUATION_CAP * core.N_INTERNAL
        ),
        "recovery_band_evaluation_cap": int(RECOVERY_BAND_EVALUATION_CAP),
        "recovery_projection_definition": (
            "zero-NoF adjacent-chord-coherent saddle seed followed by direct-anchor-energy HJ improved-tangent chord-constraint Newton to the original equidistance ceiling"
        ),
        "recovery_seed_mode": "not_constructed",
        "stationary_normal_probe": False,
        "stationary_public_movement": None,
        "saddle_image": None,
        "adjacent_chord_cosine": None,
        "coherence_loss": None,
        "coherence_loss_limit": None,
        "amplified_direction_mismatch": None,
        "strong_recovery_selected": False,
        "saddle_spring_weight": None,
        "seed_spring_residual": None,
        "seed_spring_iterations": 0,
        "recovery_raw_proposal_force_evaluations": 0,
        "recovery_merge_definition": "whole projected band atomic accept or reject",
        "recovery_commit_requires_complete_band": True,
        "total_atomic_trust_radius": float(recovery_trust),
        "i_perp_absolute_tolerance": RECOVERY_IPERP_ATOL,
        "equidistance_absolute_tolerance": RECOVERY_EQUIDISTANCE_ATOL,
        "cycles": 0,
        "merged_images": 0,
        "raw_trial_evaluated": False,
        "raw_proposal_constructed": False,
        "raw_pre_addition_direction": None,
        "raw_candidate_path": None,
        "recovery_projection_seed_path": None,
        "projected_attempt_path": None,
        "projected_trial_evaluated": False,
        "projected_candidate_path": None,
        "projected_candidate_energies": None,
        "projected_candidate_gradients": None,
        "sync_trial_evaluated": False,
        "direct_joint_evaluation_vector": [],
        "raw_recovery_joint_evaluation_vector": [],
        "projected_recovery_joint_evaluation_vector": [],
        "recovery_joint_evaluation_vector": [],
        "transaction_joint_evaluation_vector": [],
        "raw_recovery_new_image_evaluations": 0,
        "projected_recovery_new_image_evaluations": 0,
        "recovery_new_image_evaluations": 0,
        "recovery_projection_residual": None,
        "recovery_projection_iterations": 0,
        "recovery_projection_backtracks": 0,
        "recovery_projection_feasible": False,
        "regularized_step_length": 0.0,
        "predicted_regularized_decrease": 0.0,
        "warm_pair_count": 0,
        "committed": False,
        "rolled_back": False,
        "interrupted": False,
    }

    def calls_now():
        return np.asarray([item.calls for item in evaluators], dtype=np.int64)

    direct_vector = calls_now() - attempt_start_calls
    raw_vector = np.zeros_like(direct_vector)
    projected_vector = np.zeros_like(direct_vector)
    transaction["direct_joint_evaluation_vector"] = direct_vector.tolist()
    transaction["direct_joint_evaluations"] = int(np.sum(direct_vector))

    def record_projected(start_calls):
        projected_vector[:] = calls_now() - np.asarray(
            start_calls, dtype=np.int64
        )
        transaction["projected_recovery_joint_evaluation_vector"] = (
            projected_vector.tolist()
        )
        transaction["projected_recovery_new_image_evaluations"] = int(
            np.sum(projected_vector)
        )

    def finish():
        total_vector = calls_now() - attempt_start_calls
        recovery_vector = total_vector - direct_vector
        accounted_vector = raw_vector + projected_vector
        if (
            np.any(direct_vector < 0)
            or np.any(raw_vector < 0)
            or np.any(projected_vector < 0)
            or np.any(direct_vector > 1)
            or np.any(raw_vector > 1)
            or np.any(projected_vector > 1)
            or np.any(total_vector > TRANSACTION_CALL_CAP_PER_IMAGE_INCLUDING_DIRECT)
            or not np.array_equal(recovery_vector, accounted_vector)
        ):
            raise ContractError("projected recovery Number-of-Forces transaction drifted")
        transaction["raw_recovery_joint_evaluation_vector"] = raw_vector.tolist()
        transaction["projected_recovery_joint_evaluation_vector"] = (
            projected_vector.tolist()
        )
        transaction["recovery_joint_evaluation_vector"] = (
            recovery_vector.tolist()
        )
        transaction["transaction_joint_evaluation_vector"] = (
            total_vector.tolist()
        )
        transaction["transaction_joint_evaluations"] = int(np.sum(total_vector))
        transaction["recovery_joint_evaluations"] = int(
            np.sum(recovery_vector)
        )
        transaction["raw_recovery_new_image_evaluations"] = int(
            np.sum(raw_vector)
        )
        transaction["projected_recovery_new_image_evaluations"] = int(
            np.sum(projected_vector)
        )
        transaction["recovery_new_image_evaluations"] = int(
            np.sum(accounted_vector)
        )
        return transaction

    direct_is_acceptable = _i_ok(direct_i, pre_i) and _eq_ok(
        direct_eq, ceiling
    )
    transaction["direct_within_post_main_i_perp"] = bool(direct_is_acceptable)
    stationary_probe = bool(
        direct_is_acceptable and stationary_probe_requested
    )
    transaction["stationary_normal_probe"] = stationary_probe
    if direct_is_acceptable and not stationary_probe:
        transaction["committed"] = True
        return (path, energies, gradients), finish(), None
    if recovery_backtrack_state is None:
        selected = recovery_direction_candidate(
            core,
            path,
            energies,
            gradients,
            accepted_recovery_memories,
            float(controls.alpha),
            recovery_trust,
            "accepted_history",
        )
        selected["backtrack_fraction"] = 1.0
        selected["base_step_length"] = float(selected["step_length"])
        selected["base_predicted_decrease"] = float(
            selected["predicted_decrease"]
        )
    else:
        selected = recovery_backtrack_candidate(
            recovery_backtrack_state, float(controls.alpha)
        )
    transaction["direction_candidates"] = [{
        "name": selected["name"],
        "predicted_regularized_decrease": float(
            selected["predicted_decrease"]
        ),
        "regularized_step_length": float(selected["step_length"]),
        "base_regularized_step_length": float(selected["base_step_length"]),
        "backtrack_fraction": float(selected["backtrack_fraction"]),
        "g_dot_d": float(selected["g_dot_d"]),
        "norm_squared": float(selected["norm_squared"]),
        "warm_pair_count": int(selected["warm_pair_count"]),
    }]
    transaction["selected_direction"] = str(selected["name"])
    transaction["warm_pair_count"] = int(selected["warm_pair_count"])
    directions = np.asarray(selected["directions"], dtype=np.float64)
    step_length = float(selected["step_length"])
    predicted_decrease = float(selected["predicted_decrease"])
    transaction["regularized_step_length"] = step_length
    transaction["predicted_regularized_decrease"] = predicted_decrease
    transaction["recovery_backtrack_fraction"] = float(
        selected["backtrack_fraction"]
    )
    if step_length <= 0.0 or predicted_decrease <= 0.0:
        transaction["rolled_back"] = True
        return pre, finish(), None

    transaction["raw_pre_addition_direction"] = np.asarray(
        directions, dtype=np.float64
    ).tolist()
    if recovery_backtrack_state is None:
        base_displacement = step_length * directions
        selected["base_displacement"] = np.asarray(
            base_displacement, dtype=np.float64
        ).copy()
        candidate_displacement = base_displacement
    else:
        candidate_displacement = float(
            selected["backtrack_fraction"]
        ) * np.asarray(selected["base_displacement"], dtype=np.float64)
    candidate_path = path + candidate_displacement
    candidate_path[0] = path[0]
    candidate_path[-1] = path[-1]
    transaction["raw_proposal_constructed"] = True
    transaction["raw_candidate_path"] = np.asarray(
        candidate_path, dtype=np.float64
    ).tolist()
    if recovery_backtrack_state is None:
        # Persist the actual synchronized full-step displacement after floating-
        # point addition and endpoint restoration, not the later merged subset.
        selected["base_displacement"] = candidate_path - path
    if stationary_probe:
        projection_seed = candidate_path
        seed_audit = {
            "mode": "ordinary_normal_seed_for_stationarity_exhaustion",
            "saddle_image": None,
            "adjacent_chord_cosine": None,
            "coherence_loss": None,
            "coherence_loss_limit": None,
            "amplified_direction_mismatch": None,
            "strong_recovery_selected": False,
            "saddle_spring_weight": None,
            "spring_residual": None,
            "spring_iterations": 0,
        }
    else:
        projection_seed, seed_audit = chord_coherent_recovery_seed(
            core, candidate_path, energies, controls
        )
    transaction["recovery_seed_mode"] = str(seed_audit["mode"])
    transaction["saddle_image"] = seed_audit["saddle_image"]
    transaction["adjacent_chord_cosine"] = seed_audit[
        "adjacent_chord_cosine"
    ]
    transaction["coherence_loss"] = seed_audit["coherence_loss"]
    transaction["coherence_loss_limit"] = seed_audit[
        "coherence_loss_limit"
    ]
    transaction["amplified_direction_mismatch"] = seed_audit[
        "amplified_direction_mismatch"
    ]
    transaction["strong_recovery_selected"] = bool(
        seed_audit["strong_recovery_selected"]
    )
    transaction["saddle_spring_weight"] = seed_audit[
        "saddle_spring_weight"
    ]
    transaction["seed_spring_residual"] = seed_audit["spring_residual"]
    transaction["seed_spring_iterations"] = int(
        seed_audit["spring_iterations"]
    )
    transaction["recovery_projection_seed_path"] = np.asarray(
        projection_seed, dtype=np.float64
    ).tolist()
    (
        projected_path,
        projection_residual,
        projection_iterations,
        projection_backtracks,
        projection_feasible,
    ) = tangent_chord_constraint_project(
        core,
        projection_seed,
        energies,
        ceiling,
    )
    transaction["recovery_projection_residual"] = float(projection_residual)
    transaction["recovery_projection_iterations"] = int(projection_iterations)
    transaction["recovery_projection_backtracks"] = int(projection_backtracks)
    transaction["recovery_projection_feasible"] = bool(projection_feasible)
    transaction["projected_attempt_path"] = np.asarray(
        projected_path, dtype=np.float64
    ).tolist()
    if not projection_feasible:
        transaction["rolled_back"] = True
        return pre, finish(), None
    projected_start_calls = calls_now()
    try:
        projected_energies, projected_gradients = evaluate_path(
            core, projected_path, evaluators
        )
    except BudgetReached:
        record_projected(projected_start_calls)
        transaction["interrupted"] = True
        transaction["rolled_back"] = True
        return pre, finish(), None
    record_projected(projected_start_calls)
    transaction["projected_trial_evaluated"] = True
    transaction["sync_trial_evaluated"] = True
    transaction["cycles"] = 1
    transaction["projected_candidate_path"] = np.asarray(
        projected_path, dtype=np.float64
    ).tolist()
    transaction["projected_candidate_energies"] = np.asarray(
        projected_energies, dtype=np.float64
    ).tolist()
    transaction["projected_candidate_gradients"] = np.asarray(
        projected_gradients, dtype=np.float64
    ).tolist()
    projected_state = (
        projected_path,
        projected_energies,
        projected_gradients,
    )
    merged, accepted, gate_audit = atomic_recovery_merge(
        core,
        (path, energies, gradients),
        projected_state,
        ceiling,
        float(controls.alpha),
        float(pre_i),
    )
    transaction["candidate_i_perp"] = float(gate_audit["candidate_i_perp"])
    transaction["candidate_equidistance"] = float(
        gate_audit["candidate_equidistance"]
    )
    transaction["actual_proximal_merit"] = float(
        gate_audit["actual_proximal_merit"]
    )
    transaction["recovery_candidate_finite"] = bool(gate_audit["finite"])
    transaction["recovery_monotone_to_post_main_i_perp"] = bool(
        gate_audit["monotone_to_post_main_i_perp"]
    )
    transaction["recovery_positive_actual_proximal_merit"] = bool(
        gate_audit["positive_actual_proximal_merit"]
    )
    transaction["recovery_accepted_by_atomic_gate"] = bool(
        gate_audit["accepted_by_atomic_gate"]
    )
    transaction["merged_images"] = int(np.count_nonzero(accepted))
    if (
        transaction["merged_images"] == int(core.N_INTERNAL)
        and transaction["recovery_accepted_by_atomic_gate"]
    ):
        publication_audit = publish_accepted_recovery_secants(
            accepted_recovery_memories,
            (path, energies, gradients),
            projected_state,
            strong_recovery_selected=bool(
                transaction["strong_recovery_selected"]
            ),
            stationary_normal_probe=bool(
                transaction["stationary_normal_probe"]
            ),
        )
        transaction.update(publication_audit)
        transaction["committed"] = True
        return merged, finish(), None
    transaction["rolled_back"] = True
    return pre, finish(), selected


def _state_record(nf, outer, phase, path, energies, gradients):
    return {
        "article_nf": int(nf),
        "outer": int(outer),
        "state_phase": str(phase),
        "path": np.asarray(path, dtype=np.float64).copy(),
        "energies": np.asarray(energies, dtype=np.float64).copy(),
        "gradients": np.asarray(gradients, dtype=np.float64).copy(),
    }


def _append_record(records, nf, outer, phase, path, energies, gradients):
    record = _state_record(nf, outer, phase, path, energies, gradients)
    if records and int(records[-1]["article_nf"]) == int(nf):
        records[-1] = record
    else:
        records.append(record)


def outer_state_chain_log(outer_start_path, post_main_state):
    """Serialize one complete main-stage state link without force queries."""

    outer_start = np.asarray(outer_start_path, dtype=np.float64)
    post_path, post_energies, post_gradients = (
        np.asarray(item, dtype=np.float64) for item in post_main_state
    )
    if (
        outer_start.ndim != 2
        or post_path.shape != outer_start.shape
        or post_gradients.shape != outer_start.shape
        or post_energies.shape != (outer_start.shape[0],)
    ):
        raise ContractError("outer public-state chain shape drifted")
    if not all(
        np.all(np.isfinite(item))
        for item in (
            outer_start,
            post_path,
            post_energies,
            post_gradients,
        )
    ):
        raise ContractError("outer public-state chain is non-finite")
    if not np.array_equal(outer_start[[0, -1]], post_path[[0, -1]]):
        raise ContractError("outer public-state chain changed an endpoint")
    return {
        "outer_start_path": outer_start.tolist(),
        "post_main_path": post_path.tolist(),
        "post_main_energies": post_energies.tolist(),
        "post_main_gradients": post_gradients.tolist(),
    }


def transaction_state_chain_log(
    outer_start_path,
    post_main_state,
    committed_public_path,
):
    """Extend a main-state log with its atomically committed public path."""

    payload = outer_state_chain_log(outer_start_path, post_main_state)
    committed = np.asarray(committed_public_path, dtype=np.float64)
    outer_start = np.asarray(outer_start_path, dtype=np.float64)
    if committed.shape != outer_start.shape or not np.all(np.isfinite(committed)):
        raise ContractError("transaction committed-public path is invalid")
    if not np.array_equal(outer_start[[0, -1]], committed[[0, -1]]):
        raise ContractError("transaction committed-public path changed an endpoint")
    payload["committed_public_path"] = committed.tolist()
    return payload


def run(
    controls: Controls,
    *,
    budget: int = ARTICLE_NF_BUDGET,
    outer_limit: int = OUTER_LIMIT,
):
    validate_single_thread_environment()
    controls.validate()
    validate_preregistration()
    if int(budget) != ARTICLE_NF_BUDGET or int(outer_limit) != OUTER_LIMIT:
        raise ContractError("formal budget or outer safeguard drifted")
    core, potential = _load_backend(controls.n_internal)
    path = np.asarray(core.common_path(), dtype=np.float64)
    ledger = Ledger(limit=int(budget))
    set_evaluation_trace_context(ledger, 0, "initial")
    evaluators = make_evaluators(core, potential, ledger)
    energies, gradients = evaluate_path(core, path, evaluators)
    memories = [[] for _ in range(int(core.N_INTERNAL))]
    recovery_backtrack_state = None
    recovery_backtrack_states_cleared_total = 0
    records = [_state_record(ledger.total, 0, "initial", path, energies, gradients)]
    transactions = []
    main_stage_audits = []
    direct_paths = []
    direct_energies = []
    direct_gradients = []
    direct_nf = []
    stop_reason = "outer_limit"
    failure_message = ""

    for outer in range(1, outer_limit + 1):
        start_nf = ledger.total
        old_path = path.copy()
        old_energies = energies.copy()
        old_gradients = gradients.copy()
        backtrack_state_entering_outer = bool(recovery_backtrack_state is not None)
        backtrack_attempts_entering_outer = int(
            recovery_backtrack_state["completed_attempts"]
            if recovery_backtrack_state is not None
            else 0
        )
        outer_memory_snapshot = clone_physical_memories(memories)
        try:
            set_evaluation_trace_context(ledger, outer, "main")
            path, energies, gradients, stage_audit = solve_main_stage(
                core, path, energies, gradients, evaluators, memories, controls
            )
            main_stage_rolled_back = bool(
                stage_audit["rolled_back_by_i_perp_guard"]
            )
            default_accepted_recovery_memories = (
                outer_memory_snapshot if main_stage_rolled_back else memories
            )
            main_path_changed = main_stage_changed_public_path(
                main_stage_rolled_back, old_path, path
            )
            backtrack_state_cleared_by_main_accept = False
            if not main_stage_rolled_back and main_path_changed:
                backtrack_state_cleared_by_main_accept = bool(
                    recovery_backtrack_state is not None
                )
                recovery_backtrack_states_cleared_total += int(
                    backtrack_state_cleared_by_main_accept
                )
                recovery_backtrack_state = None
            post_main_i, _ = internal_i_perp(core, path, energies, gradients)
            stage_audit["pre_main_memory_pair_count"] = int(
                sum(len(memory) for memory in outer_memory_snapshot)
            )
            stage_audit["live_post_guard_memory_pair_count"] = int(
                sum(len(memory) for memory in memories)
            )
            stage_audit["pre_main_memory_snapshot_used_by_recovery"] = bool(
                main_stage_rolled_back
            )
            stage_audit["recovery_memory_pair_count"] = int(
                sum(len(memory) for memory in default_accepted_recovery_memories)
            )
            stage_audit["main_public_path_changed_bitwise"] = bool(
                main_path_changed
            )
            stage_audit["recovery_backtrack_state_entering_outer"] = bool(
                backtrack_state_entering_outer
            )
            stage_audit["recovery_backtrack_state_cleared_by_main_accept"] = bool(
                backtrack_state_cleared_by_main_accept
            )
            stage_audit["same_anchor_recovery_attempt_count_entering"] = int(
                backtrack_attempts_entering_outer
            )
            stage_audit["recovery_acceptance_envelope_i_perp"] = float(
                post_main_i
            )
            stage_audit["post_main_i_perp"] = float(post_main_i)
            stage_audit["recovery_reads_main_r"] = False
            stage_audit["outer"] = int(outer)
            stage_audit["terminal_article_nf"] = int(ledger.total)
            stage_audit.update(
                outer_state_chain_log(
                    old_path,
                    (path, energies, gradients),
                )
            )
            main_stage_audits.append(stage_audit)
            _append_record(
                records,
                ledger.total,
                outer,
                "complete_main_stage",
                path,
                energies,
                gradients,
            )

            pre = (path.copy(), energies.copy(), gradients.copy())
            attempt_start_calls = np.asarray(
                [item.calls for item in evaluators], dtype=np.int64
            )
            direct_path, residual, spring_iterations = improved_spring_reparameterize(
                core, path, energies
            )
            set_evaluation_trace_context(ledger, outer, "direct")
            direct_energy, direct_gradient = evaluate_path(
                core, direct_path, evaluators
            )
            direct_paths.append(direct_path.copy())
            direct_energies.append(direct_energy.copy())
            direct_gradients.append(direct_gradient.copy())
            direct_nf.append(int(ledger.total))
            stationary_public_movement = float(
                np.max(np.linalg.norm(direct_path - old_path, axis=1))
            )
            backtrack_state_cleared_by_anchor_mismatch = False
            backtrack_anchor_matched = True
            if recovery_backtrack_state is not None:
                backtrack_anchor_matched = bool(
                    main_stage_rolled_back
                    and recovery_anchor_matches(
                        recovery_backtrack_state,
                        direct_path,
                        direct_energy,
                        direct_gradient,
                    )
                )
                if not backtrack_anchor_matched:
                    backtrack_state_cleared_by_anchor_mismatch = True
                    recovery_backtrack_states_cleared_total += 1
                    recovery_backtrack_state = None
            same_anchor_recovery_attempt_count = int(
                recovery_backtrack_state["completed_attempts"]
                if recovery_backtrack_state is not None
                else 0
            )
            set_evaluation_trace_context(ledger, outer, "projected")
            updated, transaction, rejected_recovery_descriptor = (
                single_synchronized_recovery(
                    core,
                    pre,
                    (direct_path, direct_energy, direct_gradient),
                    evaluators,
                    default_accepted_recovery_memories,
                    recovery_backtrack_state,
                    controls,
                    residual,
                    attempt_start_calls,
                    stationary_probe_requested=bool(
                        stationary_public_movement <= KKT_ATOL
                    ),
                )
            )
            if transaction["stationary_normal_probe"]:
                transaction["stationary_public_movement"] = float(
                    stationary_public_movement
                )
            transaction.update(
                transaction_state_chain_log(old_path, pre, updated[0])
            )
            transaction["outer"] = int(outer)
            transaction["spring_iterations"] = int(spring_iterations)
            transaction["recovery_backtrack_anchor_matched"] = bool(
                backtrack_anchor_matched
            )
            transaction["recovery_backtrack_state_cleared_by_anchor_mismatch"] = bool(
                backtrack_state_cleared_by_anchor_mismatch
            )
            transaction["same_anchor_recovery_attempt_count_before"] = int(
                same_anchor_recovery_attempt_count
            )
            transactions.append(transaction)
            path, energies, gradients = updated
            committed_i, _ = internal_i_perp(core, path, energies, gradients)
            transaction["committed_i_perp"] = float(committed_i)
            transaction["commit_is_monotone_to_post_main_i_perp"] = bool(
                not transaction["committed"]
                or _i_ok(committed_i, post_main_i)
            )
            transaction["pre_main_memory_published_after_rollback_commit"] = bool(
                transaction["committed"] and main_stage_rolled_back
            )
            publish_main_snapshot = recovery_commit_publishes_main_snapshot(
                main_stage_rolled_back, transaction["committed"]
            )
            transaction["accepted_pre_main_memory_restored_after_rollback"] = bool(
                publish_main_snapshot
            )
            if publish_main_snapshot:
                restore_physical_memories(memories, outer_memory_snapshot)
            if transaction["committed"]:
                _append_record(
                    records,
                    ledger.total,
                    outer,
                    "reparameterization_committed",
                    path,
                    energies,
                    gradients,
                )
            else:
                _append_record(
                    records,
                    ledger.total,
                    outer,
                    (
                        "recovery_interrupted_public_state"
                        if transaction["interrupted"]
                        else "recovery_rolled_back_public_state"
                    ),
                    path,
                    energies,
                    gradients,
                )

            movement = float(np.max(np.linalg.norm(path - old_path, axis=1)))
            completed_recovery_candidate = bool(
                transaction["sync_trial_evaluated"]
                and not transaction["interrupted"]
            )
            paid_recovery_attempt = bool(
                int(transaction["recovery_joint_evaluations"]) > 0
            )
            same_anchor_attempt_index = (
                int(same_anchor_recovery_attempt_count) + 1
                if completed_recovery_candidate
                else 0
            )
            transaction["completed_recovery_candidate_at_current_anchor"] = bool(
                completed_recovery_candidate
            )
            transaction["paid_recovery_attempt_at_current_anchor"] = bool(
                paid_recovery_attempt
            )
            transaction["same_anchor_recovery_attempt_index"] = int(
                same_anchor_attempt_index
            )
            transaction["same_anchor_recovery_attempt_count_after"] = int(
                same_anchor_attempt_index
                if completed_recovery_candidate
                else same_anchor_recovery_attempt_count
            )
            transaction["same_anchor_recovery_attempt_cap_m_max"] = int(
                controls.m_max
            )
            transaction["public_path_movement"] = float(movement)

            if transaction["interrupted"]:
                cleared = bool(recovery_backtrack_state is not None)
                recovery_backtrack_states_cleared_total += int(cleared)
                recovery_backtrack_state = None
                transaction["recovery_backtrack_state_cleared_after_transaction"] = bool(
                    cleared
                )
                transaction["recovery_backtrack_lifecycle_action"] = (
                    "clear_on_interruption"
                )
                raise BudgetReached("synchronized recovery interrupted")

            if transaction["committed"]:
                cleared = bool(recovery_backtrack_state is not None)
                recovery_backtrack_states_cleared_total += int(cleared)
                recovery_backtrack_state = None
                transaction["recovery_backtrack_state_cleared_after_transaction"] = bool(
                    cleared
                )
                transaction["recovery_backtrack_lifecycle_action"] = (
                    "clear_on_transaction_commit"
                )
                transaction["recovery_backtrack_carry_allowed"] = False
            else:
                (
                    same_anchor_recovery_attempt_count,
                    carry_allowed,
                ) = same_anchor_recovery_retry_decision(
                    same_anchor_recovery_attempt_count,
                    completed_recovery_candidate,
                    controls.m_max,
                )
                carry_allowed = bool(
                    carry_allowed
                    and rejected_recovery_descriptor is not None
                )
                transaction["same_anchor_recovery_attempt_count_after"] = int(
                    same_anchor_recovery_attempt_count
                )
                transaction["recovery_backtrack_state_cleared_after_transaction"] = False
                transaction["recovery_backtrack_carry_allowed"] = bool(
                    carry_allowed
                )
                transaction["same_anchor_recovery_retry_allowed"] = bool(
                    carry_allowed and movement <= 1.0e-14
                )
                if carry_allowed:
                    recovery_backtrack_state = make_recovery_backtrack_state(
                        direct_path,
                        direct_energy,
                        direct_gradient,
                        rejected_recovery_descriptor,
                        same_anchor_recovery_attempt_count,
                    )
                    transaction["recovery_backtrack_lifecycle_action"] = (
                        "retain_new_line_search_after_main_movement"
                        if movement > 1.0e-14
                        else "retain_for_same_anchor_backtrack"
                    )
                    if movement <= 1.0e-14:
                        continue
                else:
                    cleared = bool(recovery_backtrack_state is not None)
                    recovery_backtrack_states_cleared_total += int(cleared)
                    recovery_backtrack_state = None
                    transaction["recovery_backtrack_state_cleared_after_transaction"] = bool(
                        cleared
                    )
                    transaction["recovery_backtrack_lifecycle_action"] = (
                        "clear_on_stationary_or_attempt_cap"
                        if movement <= 1.0e-14
                        else "clear_on_no_descriptor_or_attempt_cap_after_main_movement"
                    )

            if movement <= 1.0e-14:
                stop_reason = "stationary"
                break
        except BudgetReached:
            _append_record(
                records,
                ledger.total,
                outer,
                "budget_held_public_state",
                path,
                energies,
                gradients,
            )
            recovery_backtrack_states_cleared_total += int(
                recovery_backtrack_state is not None
            )
            recovery_backtrack_state = None
            stop_reason = "budget"
            break
        except (
            CertificationFailure,
            FloatingPointError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            _append_record(
                records,
                ledger.total,
                outer,
                "numerical_failure_held_public_state",
                old_path,
                old_energies,
                old_gradients,
            )
            recovery_backtrack_states_cleared_total += int(
                recovery_backtrack_state is not None
            )
            recovery_backtrack_state = None
            stop_reason = "numerical_failure"
            failure_message = f"{type(error).__name__}: {error}"
            break
        if ledger.total == start_nf:
            recovery_backtrack_states_cleared_total += int(
                recovery_backtrack_state is not None
            )
            recovery_backtrack_state = None
            stop_reason = "stationary"
            break

    recovery_backtrack_states_cleared_total += int(
        recovery_backtrack_state is not None
    )
    recovery_backtrack_state = None

    return {
        "controls": controls,
        "q_matrix": np.asarray(potential.Q, dtype=np.float64),
        "initial_path": np.asarray(core.common_path(), dtype=np.float64),
        "records": records,
        "transactions": transactions,
        "main_stage_audits": main_stage_audits,
        "direct_paths": direct_paths,
        "direct_energies": direct_energies,
        "direct_gradients": direct_gradients,
        "direct_nf": direct_nf,
        "stop_reason": stop_reason,
        "failure_message": failure_message,
        "terminal_article_nf": int(ledger.total),
        "last_complete_article_nf": int(records[-1]["article_nf"]),
        "blocked_misses": int(ledger.blocked_misses),
        "evaluation_miss_trace": [
            dict(item) for item in ledger.evaluation_miss_trace
        ],
        "recovery_backtrack_state_remaining": bool(
            recovery_backtrack_state is not None
        ),
        "recovery_backtrack_states_cleared_total": int(
            recovery_backtrack_states_cleared_total
        ),
        "per_image_joint_evaluations": np.asarray(
            [item.calls for item in evaluators], dtype=np.int64
        ),
    }


def save_result(
    result,
    destination: Path,
    *,
    n20_gate_manifest_token: str,
) -> dict:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = result["records"]
    controls = result["controls"]
    gate_token = validate_resolution_gate_token(
        controls, n20_gate_manifest_token
    )
    transactions = result["transactions"]
    initial_shape = tuple(np.asarray(result["initial_path"]).shape)
    per_image = np.asarray(result["per_image_joint_evaluations"], dtype=np.int64)
    if int(np.sum(per_image)) != int(result["terminal_article_nf"]):
        raise ContractError("per-image Number-of-Forces ledger does not close")
    direct_paths = np.asarray(result["direct_paths"], dtype=np.float64).reshape(
        (-1,) + initial_shape
    )
    direct_energies = np.asarray(result["direct_energies"], dtype=np.float64).reshape(
        (-1, initial_shape[0])
    )
    direct_gradients = np.asarray(
        result["direct_gradients"], dtype=np.float64
    ).reshape((-1,) + initial_shape)
    payload = {
        "experiment_id": np.asarray(EXPERIMENT_ID),
        "n20_gate_manifest_sha256_token": np.asarray(gate_token),
        "run_id": np.asarray(controls.run_id),
        "config_json": np.asarray(json.dumps(controls.as_dict(), sort_keys=True)),
        "source_sha256": np.asarray(_sha256(THIS_FILE)),
        "registration_sha256": np.asarray(_sha256(REGISTRATION)),
        "formal_leps_optimizer_sha256": np.asarray(_sha256(FORMAL_LEPS_OPTIMIZER)),
        "formal_leps_registry_sha256": np.asarray(_sha256(FORMAL_LEPS_REGISTRY)),
        "q_matrix": np.asarray(result["q_matrix"]),
        "q_sha256": np.asarray(
            hashlib.sha256(
                np.ascontiguousarray(result["q_matrix"]).tobytes()
            ).hexdigest()
        ),
        "tangent_definition": np.asarray("Henkelman-Jonsson improved everywhere"),
        "main_subproblem": np.asarray("formal LEPS complete-normal proximal Armijo L-BFGS"),
        "main_state_globalization": np.asarray("I_perp-monotone whole-band atomic commit"),
        "reparameterization_definition": np.asarray("formal LEPS improved-tangent identical-spring"),
        "reparameterization_period": np.asarray(REPARAMETERIZATION_PERIOD, dtype=np.int64),
        "recovery_definition": np.asarray(
            "zero-NoF adjacent-chord-coherent saddle seed, direct-energy improved-tangent chord-Newton projection, one evaluated projected band, unchanged atomic gate, coherence-conditioned accepted physical secant publication"
        ),
        "recovery_reads_main_r": np.asarray(False),
        "recovery_chord_coherence_rule": np.asarray(
            "(m_max-1)*sqrt(1-dot(u_minus,u_plus)) <= 1"
        ),
        "recovery_chord_coherence_strong_weight": np.asarray(
            "m_max-1"
        ),
        "recovery_chord_coherence_conservative_weight": np.asarray(1),
        "stationarity_definition": np.asarray(
            "acceptable unchanged direct band converges only after ordinary normal recovery exhausts its m_max slots"
        ),
        "recovery_acceptance_reference": np.asarray(
            "post-main I_perp or positive actual proximal merit"
        ),
        "direct_acceptance_reference": np.asarray(
            "post-main I_perp"
        ),
        "recovery_uses_pre_main_memory_snapshot_after_main_rollback": np.asarray(True),
        "recovery_updates_main_memory_only_after_complete_commit": np.asarray(True),
        "rollback_commit_memory_transport": np.asarray(
            "publish the accepted pre-main snapshot plus coherence-conditioned curvature-certified direct-to-projected secants; rejected recovery trials never alter main memory"
        ),
        "rejected_recovery_trial": np.asarray(
            "private actual full-step displacement carried only across a bitwise-identical direct anchor"
        ),
        "recovery_direction_selection": np.asarray(
            "accepted-history regularized predicted decrease for the first proposal; retries multiply the stored actual full-step displacement by BACKTRACK**completed_attempts without an extra force call"
        ),
        "recovery_backtrack_factor": np.asarray(BACKTRACK),
        "recovery_projection": np.asarray(
            "adjacent-chord-coherent saddle seed followed by direct-anchor-energy HJ improved-tangent chord-constraint Newton to the original equidistance ceiling"
        ),
        "raw_recovery_path_charged": np.asarray(False),
        "projected_recovery_path_charged": np.asarray(True),
        "recovery_merge": np.asarray("whole projected band atomic accept or reject"),
        "recovery_escape_merit": np.asarray(
            "sum_internal(E_direct-E_projected-||x_projected-x_direct||^2/(2*alpha))"
        ),
        "recovery_commit_requires_complete_band": np.asarray(True),
        "accepted_recovery_secant_curvature_threshold": np.asarray(
            RECOVERY_SECANT_CURVATURE
        ),
        "recovery_secant_normalized_curvature_formula": np.asarray(
            "(s^T y)/(||s|| ||y||)"
        ),
        "recovery_secant_strong_publication": np.asarray(
            "highest normalized-curvature certified pair only"
        ),
        "recovery_secant_conservative_publication": np.asarray(
            "all curvature-certified pairs"
        ),
        "recovery_secant_ordinary_publication": np.asarray(
            "all curvature-certified pairs"
        ),
        "recovery_secant_tie_break": np.asarray(
            "smaller 1-based internal image index"
        ),
        "maximum_recovery_bands_per_outer": np.asarray(
            RECOVERY_BAND_EVALUATION_CAP, dtype=np.int64
        ),
        "same_anchor_recovery_attempt_cap": np.asarray(
            "m_max completed fixed-factor recovery slots at one direct anchor; paid misses are recorded separately"
        ),
        "recovery_uses_warm_pairs": np.asarray(True),
        "trust_radius_fraction": np.asarray(TRUST_RADIUS_FRACTION),
        "trust_radius_maximum": np.asarray(TRUST_RADIUS_MAXIMUM),
        "q10_atomic_embedding": np.asarray("ten atoms, one active transformed x coordinate per atom"),
        "optimizer_reads_v2p20_opponent": np.asarray(False),
        "optimizer_reads_reference_barrier": np.asarray(False),
        "optimizer_reads_densified_delta_e": np.asarray(False),
        "stop_reason": np.asarray(result["stop_reason"]),
        "failure_message": np.asarray(result["failure_message"]),
        "terminal_article_nf": np.asarray(result["terminal_article_nf"], dtype=np.int64),
        "last_complete_article_nf": np.asarray(result["last_complete_article_nf"], dtype=np.int64),
        "blocked_misses": np.asarray(result["blocked_misses"], dtype=np.int64),
        "evaluation_miss_trace_json": np.asarray(
            [
                json.dumps(item, sort_keys=True)
                for item in result["evaluation_miss_trace"]
            ]
        ),
        "recovery_backtrack_state_remaining": np.asarray(
            result["recovery_backtrack_state_remaining"]
        ),
        "recovery_backtrack_states_cleared_total": np.asarray(
            result["recovery_backtrack_states_cleared_total"], dtype=np.int64
        ),
        "per_image_joint_evaluations": per_image,
        "transformed_initial_path": np.asarray(result["initial_path"]),
        "article_nf": np.asarray([item["article_nf"] for item in records], dtype=np.int64),
        "total_force_evaluations": np.asarray([item["article_nf"] for item in records], dtype=np.int64),
        "article_usable_state": np.ones(len(records), dtype=bool),
        "outer": np.asarray([item["outer"] for item in records], dtype=np.int64),
        "state_phase": np.asarray([item["state_phase"] for item in records]),
        "path_history": np.asarray([item["path"] for item in records]),
        "energy_history": np.asarray([item["energies"] for item in records]),
        "gradient_history": np.asarray([item["gradients"] for item in records]),
        "direct_trial_nf": np.asarray(result["direct_nf"], dtype=np.int64),
        "direct_trial_path_history": direct_paths,
        "direct_trial_energy_history": direct_energies,
        "direct_trial_gradient_history": direct_gradients,
        "transaction_json": np.asarray(
            [json.dumps(item, sort_keys=True) for item in transactions]
        ),
        "main_stage_audit_json": np.asarray(
            [json.dumps(item, sort_keys=True) for item in result["main_stage_audits"]]
        ),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp.npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "run_id": controls.run_id,
        "n20_gate_manifest_sha256_token": gate_token,
        "controls": controls.as_dict(),
        **controls.as_dict(),
        "output": str(destination),
        "stop_reason": result["stop_reason"],
        "failure_message": result["failure_message"],
        "terminal_article_nf": int(result["terminal_article_nf"]),
        "last_complete_article_nf": int(result["last_complete_article_nf"]),
        "accepted_states": len(records),
        "transactions": len(transactions),
        "committed_transactions": int(sum(item["committed"] for item in transactions)),
        "rolled_back_transactions": int(sum(item["rolled_back"] for item in transactions)),
        "recovery_backtrack_states_cleared_total": int(
            result["recovery_backtrack_states_cleared_total"]
        ),
    }


def run_one(controls: Controls, *, n20_gate_manifest_token: str):
    gate_token = validate_resolution_gate_token(
        controls, n20_gate_manifest_token
    )
    destination = (
        CAMPAIGN_ROOT
        / "runs"
        / f"n{int(controls.n_internal)}"
        / f"{controls.run_id}.npz"
    )
    if destination.exists():
        raise ContractError("refusing overwrite: " + str(destination))
    result = run(controls)
    receipt = save_result(
        result,
        destination,
        n20_gate_manifest_token=gate_token,
    )
    return destination, result, receipt


if __name__ == "__main__":
    raise SystemExit("use run_grid.py")
