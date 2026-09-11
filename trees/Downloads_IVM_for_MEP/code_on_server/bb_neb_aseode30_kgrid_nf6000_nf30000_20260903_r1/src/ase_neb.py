"""ASE-native NEB with one continuous official ODE invocation and paid WAL.

The only numerical NEB override is the fixed central-difference tangent.  The
integrator is ASE 3.23's official ``ase.optimize.ode.ode12r``.  This amendment
passes one additional fixed compatibility argument, ``maxtol=1_000_000``;
all other ODE numerical settings remain the ASE defaults used by
``NEBOptimizer(method='ODE')``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Protocol

import ase
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.mep.neb import ASENEBMethod, NEB, NEBOptimizer
from ase.optimize import ode as ase_ode_module
import numpy as np

from .accounting import PaidInteriorResult, array_sha256


SPRING_K_CHOICES_EV_PER_A2 = (0.1, 1.0, 10.0, 100.0)
ODE_MAXTOL_EV_PER_A = 1_000_000.0
ODE_FMAX_EV_PER_A = 0.0


class ASEPaidNEBError(RuntimeError):
    """Protocol, accounting, or optimizer failure in the ASE adapter."""


class OptimizationBudgetComplete(ASEPaidNEBError):
    """A force request was attempted after the registered exact budget."""


class PaidForceBatchError(ASEPaidNEBError):
    """One complete reserved batch contained at least one failed Force."""


class ODEInvocationFailed(ASEPaidNEBError):
    """The sole official ODE invocation errored or returned prematurely."""


class LedgerLike(Protocol):
    batch_count: int
    milestones: frozenset[int]

    def evaluate_batch(
        self,
        *,
        batch_id: int,
        path32_A: Any,
        coordinates30_A: Any,
        role: str,
        state_in_hash: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> PaidInteriorResult: ...

    def close_batch(
        self,
        *,
        batch_id: int,
        disposition: str,
        state_out_hash: str,
        path32_A: Any,
        evaluated_current: bool,
        optimizer_meta: Mapping[str, Any] | None = None,
    ) -> None: ...


Hook = Callable[..., None]


def validate_spring_k(value: float) -> float:
    spring_k = float(value)
    if not math.isfinite(spring_k) or spring_k not in SPRING_K_CHOICES_EV_PER_A2:
        raise ValueError(
            f"spring k must be exactly one of {SPRING_K_CHOICES_EV_PER_A2} eV/angstrom^2"
        )
    return spring_k


def _path32(value: Any) -> np.ndarray:
    path = np.asarray(value, dtype=np.float64)
    if path.ndim == 2:
        if path.shape[0] != 32 or path.shape[1] % 3:
            raise ValueError("path must contain 32 flattened images")
        path = path.reshape((32, -1, 3))
    if path.ndim != 3 or path.shape[0] != 32 or path.shape[2] != 3:
        raise ValueError("path must have shape (32,N,3) or (32,3N)")
    if not np.isfinite(path).all():
        raise ValueError("path contains a non-finite coordinate")
    return np.ascontiguousarray(path, dtype=np.float64)


class CentralDifferenceNEBMethod(ASENEBMethod):
    """ASE standard projection with fixed ``R[i+1]-R[i-1]`` tangent."""

    formula = "tau_i = normalize(R_(i+1) - R_(i-1))"

    def get_tangent(self, state, spring1, spring2, i):
        del state, i
        tangent = np.asarray(spring1.t + spring2.t, dtype=np.float64)
        norm = float(np.linalg.norm(tangent))
        if not math.isfinite(norm) or norm <= 1.0e-30:
            raise ASEPaidNEBError("central-difference NEB tangent is singular")
        return tangent / norm

    def add_image_force(
        self, state, tangential_force, tangent, imgforce, spring1, spring2, i
    ):
        del state, i
        imgforce -= tangential_force * tangent
        imgforce += (spring2.nt * spring2.k - spring1.nt * spring1.k) * tangent


@dataclass
class _ActivePaidCall:
    result: PaidInteriorResult
    paid_path: np.ndarray
    paid_path_hash: str
    role: str
    metadata: dict[str, Any]
    energies: np.ndarray
    forces: np.ndarray
    consumed: np.ndarray
    ase_residual: float | None = None


class PaidBatchCoordinator:
    """Turn each ``NEB.get_forces`` into one exact 30-interior transaction."""

    def __init__(
        self,
        ledger: LedgerLike,
        *,
        spring_k: float,
        snapshot_hook: Hook | None = None,
        batch_hook: Hook | None = None,
        full_state_hook: Hook | None = None,
        max_batches: int,
    ) -> None:
        self.ledger = ledger
        self.spring_k = validate_spring_k(spring_k)
        self.max_batches = int(max_batches)
        if self.max_batches not in (200, 1000):
            raise ValueError("max_batches must be exactly 200 (NF6000) or 1000 (NF30000)")
        self.snapshot_hook = snapshot_hook
        self.batch_hook = batch_hook
        self.full_state_hook = full_state_hook
        self.active: _ActivePaidCall | None = None
        self.get_forces_calls = 0
        self.role_history: list[str] = []
        self.disposition_history: list[str] = []
        self.paid_residual_history: list[dict[str, Any]] = []
        self.current_path_sha256: str | None = None

    def begin(self, path32_A: Any, *, role: str, metadata: Mapping[str, Any]) -> None:
        if self.active is not None:
            raise ASEPaidNEBError("a previous paid trial lacks an ODE disposition")
        if self.ledger.batch_count >= self.max_batches:
            raise OptimizationBudgetComplete(
                f"NF{30*self.max_batches} complete; next batch was not reserved"
            )
        path = _path32(path32_A)
        batch_id = int(self.ledger.batch_count) + 1
        expected_role = "current_initial" if batch_id == 1 else "trial"
        if role != expected_role:
            raise ASEPaidNEBError(
                f"batch {batch_id} role {role!r} violates sole-current sequence"
            )
        state_hash = array_sha256(path)
        frozen_meta = {
            "ase_version": ase.__version__,
            "ase_band_class": "ase.mep.neb.NEB",
            "ase_optimizer_class": "ase.mep.neb.NEBOptimizer",
            "ase_optimizer_method": "ODE",
            "ase_ode_integrator": "ase.optimize.ode.ode12r",
            "ase_neb_method_string": "aseneb",
            "tangent": CentralDifferenceNEBMethod.formula,
            "spring_k_eV_per_A2": self.spring_k,
            "image_count": 32,
            "interior_force_calls": 30,
            "endpoint_force_calls": 0,
            "ode_maxtol_eV_per_A": ODE_MAXTOL_EV_PER_A,
            "ode_maxtol_policy": "fixed_common_non_tuned_divergence_ceiling",
            **dict(metadata),
        }
        result = self.ledger.evaluate_batch(
            batch_id=batch_id,
            path32_A=path,
            coordinates30_A=path[1:-1],
            role=role,
            state_in_hash=state_hash,
            metadata=frozen_meta,
        )
        self.get_forces_calls += 1
        self.role_history.append(role)
        self.active = _ActivePaidCall(
            result=result,
            paid_path=np.array(path, copy=True),
            paid_path_hash=state_hash,
            role=role,
            metadata=frozen_meta,
            energies=np.asarray(result.energies, dtype=np.float64),
            forces=np.asarray(result.forces, dtype=np.float64),
            consumed=np.zeros(30, dtype=np.int8),
        )
        if not result.success:
            self.ledger.close_batch(
                batch_id=batch_id,
                disposition="failed",
                state_out_hash=state_hash,
                path32_A=path,
                evaluated_current=False,
                optimizer_meta={**frozen_meta, "failed_slots": list(result.failed_slots)},
            )
            self.active = None
            raise PaidForceBatchError(
                f"paid batch {batch_id} failed at interior slots {result.failed_slots}"
            )

    def consume(self, slot: int, positions_A: Any) -> tuple[float, np.ndarray]:
        active = self.active
        if active is None:
            raise ASEPaidNEBError("ASE calculator used outside an armed paid batch")
        slot = int(slot)
        if not 0 <= slot < 30:
            raise ASEPaidNEBError("ASE calculator slot is outside [0,30)")
        position = np.asarray(positions_A, dtype=np.float64)
        if position.shape != active.paid_path[slot + 1].shape or not np.array_equal(
            position, active.paid_path[slot + 1]
        ):
            raise ASEPaidNEBError("ASE calculator coordinate differs from its paid slot")
        if active.consumed[slot]:
            raise ASEPaidNEBError("one image consumed a paid result more than once")
        active.consumed[slot] = 1
        return float(active.energies[slot]), np.array(active.forces[slot], copy=True)

    def finish_force_construction(self, residual: float) -> None:
        active = self.active
        if active is None:
            raise ASEPaidNEBError("no paid call is available to finish")
        if not np.array_equal(active.consumed, np.ones(30, dtype=np.int8)):
            missing = np.flatnonzero(active.consumed != 1).tolist()
            raise ASEPaidNEBError(f"ASE did not consume all 30 paid interiors: {missing}")
        residual = float(residual)
        if not math.isfinite(residual):
            raise ASEPaidNEBError("ASE NEB residual is non-finite")
        active.ase_residual = residual

    def close(
        self,
        *,
        disposition: str,
        current_path32_A: Any,
        current_residual: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        active = self.active
        if active is None or active.ase_residual is None:
            raise ASEPaidNEBError("paid NEB call is not ready for disposition")
        if active.role == "current_initial":
            allowed = {"current_initial"}
        else:
            allowed = {"accepted", "rejected"}
        if disposition not in allowed:
            raise ASEPaidNEBError(
                f"paid role {active.role!r} cannot close as {disposition!r}"
            )
        current_path = _path32(current_path32_A)
        current_hash = array_sha256(current_path)
        current_residual = float(current_residual)
        if not math.isfinite(current_residual):
            raise ASEPaidNEBError("current ASE residual is non-finite")
        if disposition in {"current_initial", "accepted"}:
            if current_hash != active.paid_path_hash:
                raise ASEPaidNEBError("accepted/current state is not the paid path")
        else:
            if self.current_path_sha256 is None or current_hash != self.current_path_sha256:
                raise ASEPaidNEBError("rejected trial did not roll back to last accepted current")
            if current_hash == active.paid_path_hash:
                raise ASEPaidNEBError("rejected trial duplicates the rollback current")
        close_meta = {
            **active.metadata,
            "paid_path_sha256": active.paid_path_hash,
            "paid_path_ase_residual": float(active.ase_residual),
            "current_ase_residual": current_residual,
            **dict(metadata or {}),
        }
        batch_id = int(active.result.batch_id)
        nf = int(active.result.nf_after)
        full_payload = {
            "batch_id": batch_id,
            "nf": nf,
            "role": active.role,
            "disposition": disposition,
            "paid_path32_A": np.array(active.paid_path, copy=True),
            "energies30_eV": np.array(active.energies, copy=True),
            "forces30_eV_per_A": np.array(active.forces, copy=True),
            "current_path32_A": np.array(current_path, copy=True),
            "current_state_sha256": current_hash,
        }
        # The immutable full-state shard must be durable before the accounting
        # commit permits the optimizer to advance to another physical batch.
        archive_binding: Mapping[str, Any] = {}
        if self.full_state_hook is not None:
            observed = self.full_state_hook(**full_payload)
            if not isinstance(observed, Mapping):
                raise ASEPaidNEBError(
                    "full-state hook did not return its durable archive binding"
                )
            archive_binding = dict(observed)
            if not archive_binding:
                raise ASEPaidNEBError("full-state archive binding is empty")
            close_meta["full_state_archive"] = archive_binding
        self.ledger.close_batch(
            batch_id=batch_id,
            disposition=disposition,
            state_out_hash=current_hash,
            path32_A=current_path,
            evaluated_current=True,
            optimizer_meta=close_meta,
        )
        self.current_path_sha256 = current_hash
        self.disposition_history.append(disposition)
        self.paid_residual_history.append(
            {
                "batch_id": batch_id,
                "nf": nf,
                "paid_path_sha256": active.paid_path_hash,
                "paid_path_ase_residual": float(active.ase_residual),
                "current_path_sha256": current_hash,
                "current_ase_residual": current_residual,
                "disposition": disposition,
            }
        )
        payload = {
            "batch_id": batch_id,
            "nf": nf,
            "path32_A": np.array(current_path, copy=True),
            "state_sha256": current_hash,
            "disposition": disposition,
            "ase_residual": current_residual,
            "optimizer_metadata": dict(close_meta),
            # Output-only capture of arrays already returned by the one paid
            # joint E/F batch.  No calculator or OpenMM call occurs here.
            **full_payload,
        }
        if self.batch_hook is not None:
            self.batch_hook(**payload)
        if nf in self.ledger.milestones and self.snapshot_hook is not None:
            self.snapshot_hook(**payload)
        self.active = None


class PaidASEImageCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, coordinator: PaidBatchCoordinator, slot: int) -> None:
        super().__init__()
        self.coordinator = coordinator
        self.slot = int(slot)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ASEPaidNEBError("paid ASE calculator requires Atoms")
        energy, forces = self.coordinator.consume(self.slot, atoms.get_positions())
        self.results = {"energy": energy, "forces": forces}


def build_ase_neb(
    initial_path32_A: Any,
    template_atoms: Atoms,
    coordinator: PaidBatchCoordinator,
    *,
    spring_k: float,
) -> NEB:
    path = _path32(initial_path32_A)
    spring_k = validate_spring_k(spring_k)
    if spring_k != coordinator.spring_k:
        raise ValueError("band and coordinator spring k differ")
    if len(template_atoms) != path.shape[1]:
        raise ValueError("ASE template atom count differs from common path")
    images: list[Atoms] = []
    for positions in path:
        image = template_atoms.copy()
        image.set_positions(positions)
        image.calc = None
        images.append(image)
    # `aseneb` avoids endpoint-energy requests.  The built-in NEB owns all
    # remaining force construction; only its tangent method object is replaced.
    band = NEB(images, k=spring_k, method="aseneb")
    band.neb_method = CentralDifferenceNEBMethod(band)
    for slot, image in enumerate(images[1:-1]):
        image.calc = PaidASEImageCalculator(coordinator, slot)
    if images[0].calc is not None or images[-1].calc is not None:
        raise ASEPaidNEBError("fixed endpoints must never have calculators")
    return band


@dataclass
class _PendingTrial:
    X: np.ndarray
    path: np.ndarray
    residual: float
    batch_id: int


class PaidODENEBOptimizer(NEBOptimizer):
    """Accounting observer around one official ASE ``ode12r`` invocation."""

    ode_maxtol = ODE_MAXTOL_EV_PER_A

    def __init__(self, neb: NEB, coordinator: PaidBatchCoordinator) -> None:
        if not isinstance(neb, NEB):
            raise TypeError("optimizer requires ase.mep.neb.NEB")
        if neb.method != "aseneb" or not isinstance(
            neb.neb_method, CentralDifferenceNEBMethod
        ):
            raise ValueError("band must retain aseneb plus central tangent")
        # ASE defaults are intentional; logfile=None is output-only.
        super().__init__(neb, method="ODE", logfile=None)
        self.coordinator = coordinator
        self._accepted_X = np.asarray(neb.get_positions(), dtype=np.float64).reshape(-1).copy()
        self._initial_path_sha256 = array_sha256(self._authoritative_path())
        self._accepted_residual: float | None = None
        self._accepted_evaluated = False
        self._initial_force_seen = False
        self._pending_trial: _PendingTrial | None = None
        self._ode_invocations = 0
        self._official_ode12r_calls = 0
        self._ode_returned_value = False

    def _authoritative_path(self) -> np.ndarray:
        return np.ascontiguousarray(
            np.stack([image.get_positions() for image in self.neb.images]),
            dtype=np.float64,
        )

    def _path_from_X(self, X: Any) -> np.ndarray:
        flat = np.asarray(X, dtype=np.float64).reshape(-1)
        expected = (self.neb.nimages - 2) * self.neb.natoms * 3
        if flat.size != expected or not np.isfinite(flat).all():
            raise ASEPaidNEBError("ODE coordinate vector has invalid shape or values")
        path = np.empty((32, self.neb.natoms, 3), dtype=np.float64)
        path[0] = self.neb.images[0].get_positions()
        path[-1] = self.neb.images[-1].get_positions()
        path[1:-1] = flat.reshape((30, self.neb.natoms, 3))
        return np.ascontiguousarray(path)

    def _reset_calculators(self) -> None:
        for image in self.neb.images[1:-1]:
            if not isinstance(image.calc, PaidASEImageCalculator):
                raise ASEPaidNEBError("an interior paid calculator was replaced")
            image.calc.reset()

    def _close_rejected_trial(self, reason: str) -> None:
        pending = self._pending_trial
        if pending is None:
            return
        if not self._accepted_evaluated or self._accepted_residual is None:
            raise ASEPaidNEBError("trial exists before an evaluated current")
        current_path = self._path_from_X(self._accepted_X)
        self.coordinator.close(
            disposition="rejected",
            current_path32_A=current_path,
            current_residual=self._accepted_residual,
            metadata={
                "ode_decision": "rejected",
                "decision_observed": reason,
                "paid_trial_batch_id": pending.batch_id,
                "paid_trial_path_sha256": array_sha256(pending.path),
                "paid_trial_ase_residual": pending.residual,
            },
        )
        self._pending_trial = None

    def force_function(self, X):
        # In official ode12r, absence of a callback before the next f(Xnew)
        # proves the preceding trial was rejected.
        self._close_rejected_trial("next_official_ode_force_call_without_callback")
        # Re-check at every optimizer force call, before reserving any paid
        # batch.  ASE's NEB.get_forces() dispatches tangent construction
        # through this exact object, so no optimizer evaluation can silently
        # fall back to ASE's energy-weighted tangent.
        if (
            type(self.neb) is not NEB
            or self.neb.method != "aseneb"
            or not isinstance(self.neb.neb_method, CentralDifferenceNEBMethod)
            or self.method != "ode"
        ):
            raise ASEPaidNEBError(
                "optimizer NEB/ODE/central-difference tangent identity drifted"
            )
        if self.coordinator.ledger.batch_count >= self.coordinator.max_batches:
            raise OptimizationBudgetComplete("exact configured budget complete; next batch not reserved")
        flat = np.asarray(X, dtype=np.float64).reshape(-1)
        role = "current_initial" if not self._initial_force_seen else "trial"
        if role == "trial" and np.array_equal(flat, self._accepted_X):
            raise ASEPaidNEBError("official ODE attempted to repay the accepted current")
        self.neb.set_positions(flat.reshape((-1, 3)))
        paid_path = self._authoritative_path()
        self.coordinator.begin(
            paid_path,
            role=role,
            metadata={
                "ode_invocation": 1,
                "official_ode12r_call": 1,
                "ode_force_role": role,
                "ode_accepted_steps_before": int(self.nsteps),
            },
        )
        self._reset_calculators()
        force = super().force_function(flat)
        residual = float(self.neb.get_residual())
        self.coordinator.finish_force_construction(residual)
        authoritative_X = np.asarray(self.neb.get_positions(), dtype=np.float64).reshape(-1).copy()
        authoritative_path = self._authoritative_path()
        if role == "current_initial":
            self._initial_force_seen = True
            self._accepted_X = authoritative_X
            self._accepted_residual = residual
            self._accepted_evaluated = True
            self.coordinator.close(
                disposition="current_initial",
                current_path32_A=authoritative_path,
                current_residual=residual,
                metadata={"ode_decision": "current_initial"},
            )
        else:
            self._pending_trial = _PendingTrial(
                X=authoritative_X,
                path=authoritative_path,
                residual=residual,
                batch_id=int(self.coordinator.ledger.batch_count),
            )
        return force

    def callback(self, X, F=None):
        pending = self._pending_trial
        callback_X = np.asarray(X, dtype=np.float64).reshape(-1)
        if pending is None or not np.array_equal(callback_X, pending.X):
            raise ASEPaidNEBError("ODE accepted callback does not match paid trial")
        self._accepted_X = pending.X.copy()
        self._accepted_residual = float(pending.residual)
        self._accepted_evaluated = True
        self.coordinator.close(
            disposition="accepted",
            current_path32_A=pending.path,
            current_residual=pending.residual,
            metadata={
                "ode_decision": "accepted",
                "paid_trial_batch_id": pending.batch_id,
                "paid_trial_path_sha256": array_sha256(pending.path),
            },
        )
        self._pending_trial = None
        return super().callback(X, F)

    def run_ode(self, fmax):
        if self._ode_invocations != 0:
            raise ODEInvocationFailed("a second ODE invocation is forbidden")
        if self._pending_trial is not None or self.coordinator.ledger.batch_count != 0:
            raise ODEInvocationFailed("sole ODE invocation must begin at fresh NF0")
        trial_steps = self.coordinator.max_batches - 1
        if int(self.max_steps) != trial_steps:
            raise ODEInvocationFailed("official ODE trial-step count differs from configured fresh budget")
        if float(fmax) != ODE_FMAX_EV_PER_A:
            raise ODEInvocationFailed("exact-budget ODE fmax sentinel must remain zero")
        self._ode_invocations = 1
        self._official_ode12r_calls = 1
        returned: Any = None
        try:
            # This is the official ASE 3.23 algorithm.  Relative to inherited
            # NEBOptimizer.run_ode, only the fixed common maxtol is added.
            returned = ase_ode_module.ode12r(
                self.force_function,
                self.neb.get_positions().reshape(-1),
                fmax=fmax,
                maxtol=ODE_MAXTOL_EV_PER_A,
                rtol=self.rtol,
                C1=self.C1,
                C2=self.C2,
                steps=self.max_steps,
                verbose=self.verbose,
                callback=self.callback,
                residual=self.get_residual,
            )
            self._ode_returned_value = returned is not None
        except ase_ode_module.OptimizerConvergenceError as error:
            raise ODEInvocationFailed(
                f"official ode12r raised {type(error).__name__}: {error}"
            ) from error
        finally:
            self._close_rejected_trial("sole_official_ode_invocation_boundary")
            if self._accepted_evaluated:
                self.neb.set_positions(self._accepted_X.reshape((-1, 3)))
        if int(self.coordinator.ledger.batch_count) != self.coordinator.max_batches:
            raise ODEInvocationFailed(
                "official ode12r returned before the exact configured batch budget"
            )
        return True

    def run_to_exact_budget(self) -> dict[str, Any]:
        if self.coordinator.ledger.batch_count != 0:
            raise ODEInvocationFailed("formal exact-budget run must start at NF0")
        trial_steps = self.coordinator.max_batches - 1
        completed = self.run(fmax=ODE_FMAX_EV_PER_A, steps=trial_steps)
        if completed is not True:
            raise ODEInvocationFailed("sole official ODE invocation did not complete")
        roles = self.coordinator.role_history
        dispositions = self.coordinator.disposition_history
        accepted = dispositions.count("accepted")
        rejected = dispositions.count("rejected")
        if self._ode_invocations != 1 or self._official_ode12r_calls != 1:
            raise ODEInvocationFailed("ODE invocation count is not exactly one")
        if self.coordinator.get_forces_calls != self.coordinator.max_batches:
            raise ODEInvocationFailed("NEB.get_forces paid-call count differs from configured budget")
        if roles != ["current_initial"] + ["trial"] * trial_steps:
            raise ODEInvocationFailed("paid ODE role sequence is not 1 current plus configured trials")
        if dispositions[0:1] != ["current_initial"]:
            raise ODEInvocationFailed("batch1 is not the unique current_initial")
        if accepted + rejected != trial_steps:
            raise ODEInvocationFailed("trial disposition count differs from configured budget")
        if accepted < 1:
            raise ODEInvocationFailed("zero ODE trials were accepted")
        if self.nsteps != accepted:
            raise ODEInvocationFailed("ASE callback step count differs from accepted trials")
        if self._accepted_residual is None or not self._accepted_evaluated:
            raise ODEInvocationFailed("terminal accepted current was never evaluated")
        terminal_path = self._path_from_X(self._accepted_X)
        terminal_hash = array_sha256(terminal_path)
        if terminal_hash == self._initial_path_sha256:
            raise ODEInvocationFailed("terminal path equals the NF0 common path")
        tail_start = max(1, self.coordinator.max_batches - 33)
        tail = [
            float(record["paid_path_ase_residual"])
            for record in self.coordinator.paid_residual_history
            if tail_start <= int(record["batch_id"]) <= self.coordinator.max_batches
        ]
        if len(tail) != 34:
            raise ODEInvocationFailed("paid residual tail window is incomplete")
        terminal_paid = self.coordinator.paid_residual_history[-1]
        return {
            "ase_version": ase.__version__,
            "optimizer": "ase.mep.neb.NEBOptimizer",
            "optimizer_method": self.method,
            "ode_integrator": "ase.optimize.ode.ode12r",
            "ode_fixed_configuration": {
                "alpha_unused_by_ode": float(self.alpha),
                "fmax_eV_per_A": ODE_FMAX_EV_PER_A,
                "maxtol_eV_per_A": ODE_MAXTOL_EV_PER_A,
                "maxtol_policy": "fixed_common_non_tuned_divergence_ceiling",
                "h": "ASE_default_None",
                "rtol": float(self.rtol),
                "C1": float(self.C1),
                "C2": float(self.C2),
                "hmin": "ASE_default_1e-10",
                "extrapolate": "ASE_default_3",
            },
            "spring_k_eV_per_A2": self.coordinator.spring_k,
            "images": self.neb.nimages,
            "interior_images": self.neb.nimages - 2,
            "paid_batches": int(self.coordinator.ledger.batch_count),
            "paid_nf": 30 * int(self.coordinator.ledger.batch_count),
            "ode_invocations": self._ode_invocations,
            "official_ode12r_calls": self._official_ode12r_calls,
            "ode_trial_steps_requested": trial_steps,
            "ode_returned_value_at_exact_budget": self._ode_returned_value,
            "get_forces_calls": self.coordinator.get_forces_calls,
            "current_initial_batches": roles.count("current_initial"),
            "trial_batches": roles.count("trial"),
            "current_restart_batches": 0,
            "accepted_ode_steps": accepted,
            "rejected_ode_trials": rejected,
            "terminal_paid_ase_residual": float(terminal_paid["paid_path_ase_residual"]),
            "terminal_paid_disposition": str(terminal_paid["disposition"]),
            "terminal_current_ase_residual": float(self._accepted_residual),
            "tail_paid_residual_window_first_batch": tail_start,
            "tail_paid_residual_window_last_batch": self.coordinator.max_batches,
            "tail_paid_residual_window_count": len(tail),
            "tail_paid_residual_median": float(np.median(tail)),
            "initial_path_sha256": self._initial_path_sha256,
            "terminal_path_sha256": terminal_hash,
        }


def assert_formal_ase_configuration(
    band: NEB, optimizer: PaidODENEBOptimizer, spring_k: float
) -> dict[str, Any]:
    spring_k = validate_spring_k(spring_k)
    if type(band) is not NEB or not isinstance(optimizer, NEBOptimizer):
        raise ASEPaidNEBError("formal objects must be built-in ASE NEB/NEBOptimizer")
    if band.nimages != 32 or len(band.images[1:-1]) != 30:
        raise ASEPaidNEBError("formal band must have 2 endpoints + 30 interiors")
    if band.method != "aseneb" or not isinstance(
        band.neb_method, CentralDifferenceNEBMethod
    ):
        raise ASEPaidNEBError("formal central tangent/method drifted")
    if band.images[0].calc is not None or band.images[-1].calc is not None:
        raise ASEPaidNEBError("endpoint calculators are forbidden")
    if any(not isinstance(image.calc, PaidASEImageCalculator) for image in band.images[1:-1]):
        raise ASEPaidNEBError("interior paid calculators drifted")
    if any(float(value) != spring_k for value in band.k):
        raise ASEPaidNEBError("spring k drifted")
    defaults = {
        "alpha_unused_by_ode": 0.01,
        "rtol": 0.1,
        "C1": 0.01,
        "C2": 2.0,
    }
    actual = {
        "alpha_unused_by_ode": float(optimizer.alpha),
        "rtol": float(optimizer.rtol),
        "C1": float(optimizer.C1),
        "C2": float(optimizer.C2),
    }
    if optimizer.method != "ode" or actual != defaults:
        raise ASEPaidNEBError("ASE ODE defaults drifted")
    if optimizer.ode_maxtol != ODE_MAXTOL_EV_PER_A:
        raise ASEPaidNEBError("fixed ODE maxtol drifted")
    if ase.__version__ != "3.23.0":
        raise ASEPaidNEBError("ASE version is not exactly 3.23.0")
    return {
        "pass": True,
        "ase_version": ase.__version__,
        "band_class": "ase.mep.neb.NEB",
        "optimizer_class": "ase.mep.neb.NEBOptimizer",
        "optimizer_method": "ODE",
        "ode_integrator": "ase.optimize.ode.ode12r",
        "ode_defaults": actual,
        "ode_maxtol_eV_per_A": ODE_MAXTOL_EV_PER_A,
        "ode_maxtol_policy": "fixed_common_non_tuned_divergence_ceiling",
        "ode_single_invocation_required": True,
        "ode_trial_steps": optimizer.coordinator.max_batches - 1,
        "spring_k_eV_per_A2": spring_k,
        "images": 32,
        "interior_paid_per_get_forces": 30,
        "endpoint_paid_per_get_forces": 0,
        "tangent": CentralDifferenceNEBMethod.formula,
    }
