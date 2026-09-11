"""Frozen scientific and accounting identities for the BB ASE-ODE k grid."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any


SOURCE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = SOURCE_ROOT / "PROTOCOL_KGRID.json"
PREREGISTRATION_PATH = SOURCE_ROOT / "PREREGISTRATION_KGRID.md"
SELECTION_PROTOCOL_PATH = SOURCE_ROOT / "KGRID_SELECTION_PROTOCOL.json"
CONTRACT_PATH = SOURCE_ROOT / "contracts" / "CANONICAL_OPENMM_CONTRACT.json"
INITIAL_PATH = SOURCE_ROOT / "inputs" / "common_initial_path32.npz"
PDB_PATH = SOURCE_ROOT / "inputs" / "complex_fixed.pdb"
ENDPOINT_CACHE_PATH = SOURCE_ROOT / "inputs" / "endpoint_joint_ef_cache.npz"

K_VALUES = (0.1, 1.0, 10.0, 100.0)
K_TOKENS = {0.1: "0p1", 1.0: "1", 10.0: "10", 100.0: "100"}
BUDGETS = {6000: 200, 30000: 1000}
PHASE_BUDGETS = {"screening": 6000, "formal": 30000}
NF_PER_BATCH = 30
IMAGE_COUNT = 32
INTERIOR_COUNT = 30
ATOM_COUNT = 3132
ATOM_COORDINATES = 9396
ODE_FMAX_EV_PER_A = 0.0
ODE_MAXTOL_EV_PER_A = 1_000_000.0
ASE_VERSION = "3.23.0"
ASE_NEB_SHA256 = "80eb892feb0665ba2706cec6209010da4863f5150bc4aa60d94c1aca2243703c"
ASE_ODE_SHA256 = "e36d6287025ef3a99e3f348a773f0b0f8ba5588fb8d5cd76bb7161424515830d"
INITIAL_FILE_SHA256 = "6983365a5c39d117b5ec1ce4cb9afbaed1817c1190d27a567dea1fbce983e6cf"
INITIAL_ARRAY_SHA256 = "e6104f507303bc5e06abc557cad40ccd08924d21474e67073a3a9f6bd5823769"
ENDPOINT_ARRAY_SHA256 = "5788e22f86df91ed5500e88a8dc053f7408bd2fb39414de121a6eec36b833db1"
PDB_SHA256 = "26b58b4301c1e293c7fcd01878c3a208451621099419cbf5b41bd80eb96a0f49"
ENDPOINT_CACHE_SHA256 = "5cd4e757c99f0784d8b0781d05d8083070c8cfaad2670ead26999b3a4134e99a"
PHYSICAL_CONTRACT_SHA256 = "c45d797f9d05e68c587e1108497770c8af2e00e0cca8abf73eceec6605987ccf"
PHYSICAL_IDENTITY_SHA256 = "a095c51fb9b63252ab429d6e961defec5746304172222bf79788240cafc6423b"
SYSTEM_XML_SHA256 = "ef5d6907fe238ed54be5dcd11d900a7488fe0949a479e544e645c9d2711a9969"

# Both aliases refer to the same host-local BB capacity.  Holding both keeps
# new work mutually exclusive with campaigns launched from either checkout.
CANONICAL_LOCK_PATHS = tuple(
    sorted(
        (
            Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock"),
            Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock"),
        ),
        key=str,
    )
)

SCREENING_MILESTONES = (30, 240, 480, 990, 2040, 4080, 6000)
FORMAL_MILESTONES = (
    30, 240, 480, 990, 2040, 4080, 6150, 8160, 9990,
    12000, 15000, 18000, 21000, 24000, 27000, 30000,
)
_LABEL = re.compile(
    r"fresh-nonresume-bb-neb-aseode-k(?P<k>0p1|1|10|100)-"
    r"(?P<phase>screen-nf6000|formal-nf30000)-"
    r"(?P<host>[a-z0-9][a-z0-9-]{0,63})-20260903-r1"
)


class KGridSpecError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KGridSpecError(message)


def canonical_k(value: float) -> float:
    spring_k = float(value)
    _require(math.isfinite(spring_k) and spring_k in K_VALUES,
             f"k must be exactly one of {K_VALUES}")
    return spring_k


@dataclass(frozen=True)
class KGridRun:
    spring_k: float
    phase: str
    run_label: str

    def __post_init__(self) -> None:
        spring_k = canonical_k(self.spring_k)
        _require(self.phase in PHASE_BUDGETS, "phase must be screening or formal")
        match = _LABEL.fullmatch(self.run_label)
        _require(match is not None, "run label is not canonical fresh/nonresume k-grid syntax")
        assert match is not None
        expected_phase = "screen-nf6000" if self.phase == "screening" else "formal-nf30000"
        _require(match.group("k") == K_TOKENS[spring_k]
                 and match.group("phase") == expected_phase,
                 "run label k/phase differs from formal candidate")

    @property
    def terminal_nf(self) -> int:
        return PHASE_BUDGETS[self.phase]

    @property
    def batches(self) -> int:
        return BUDGETS[self.terminal_nf]

    @property
    def trial_steps(self) -> int:
        return self.batches - 1

    @property
    def milestones_nf(self) -> tuple[int, ...]:
        return SCREENING_MILESTONES if self.phase == "screening" else FORMAL_MILESTONES

    @property
    def candidate_id(self) -> str:
        return f"neb-aseode-k{K_TOKENS[self.spring_k]}-nf{self.terminal_nf}"

    def accounting(self) -> dict[str, Any]:
        return {
            "internal_per_batch": INTERIOR_COUNT,
            "endpoints_per_batch": 0,
            "nf_per_batch": NF_PER_BATCH,
            "batches": self.batches,
            "terminal_nf": self.terminal_nf,
            "milestones_nf": list(self.milestones_nf),
            "fresh_only": True,
            "resume_allowed": False,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "spring_k_eV_per_A2": self.spring_k,
            "phase": self.phase,
            "run_label": self.run_label,
            "terminal_optimization_N_F": self.terminal_nf,
            "paid_batches": self.batches,
            "paid_interior_per_batch": INTERIOR_COUNT,
            "trial_batches": self.trial_steps,
            "single_ode_invocation": True,
            "fresh_only": True,
            "resume": False,
            "endpoint_optimization_N_F": 0,
            "all_paid_calls_full_state_archived": True,
        }


def build_run(*, spring_k: float, phase: str, run_label: str) -> KGridRun:
    return KGridRun(canonical_k(spring_k), str(phase), str(run_label))
