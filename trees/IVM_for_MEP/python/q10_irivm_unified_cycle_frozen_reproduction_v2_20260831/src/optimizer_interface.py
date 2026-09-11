"""Narrow interface between the sealed runner and materialized optimizer.

No numerical implementation lives here. The final optimizer is supplied by a
separate source-materialization step after bitwise preformal equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Mapping


INTEGER_MANTISSA_SCIENTIFIC = re.compile(r"^[1-9][0-9]*e[+-]?[0-9]+$")


class OptimizerInterfaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrozenControls:
    n_internal: int
    alpha: float
    alpha_label: str
    r: float
    m_max: int
    budget: int

    def validate(self) -> None:
        if self.n_internal not in {10, 20}:
            raise OptimizerInterfaceError("formal reproduction has only N=10 and N=20")
        if not INTEGER_MANTISSA_SCIENTIFIC.fullmatch(self.alpha_label):
            raise OptimizerInterfaceError("alpha_label must have an integer mantissa")
        if float(self.alpha_label) != float(self.alpha) or self.alpha <= 0.0:
            raise OptimizerInterfaceError("alpha and alpha_label disagree or are non-positive")
        if not (0.0 < self.r < 0.7):
            raise OptimizerInterfaceError("registered r must lie strictly between 0 and 0.7")
        if self.m_max < 4:
            raise OptimizerInterfaceError("registered m_max must be at least 4")
        if self.budget != 3000:
            raise OptimizerInterfaceError("formal reproduction budget must equal 3000")

    def as_dict(self) -> dict[str, int | float | str]:
        self.validate()
        return {
            "n_internal": int(self.n_internal),
            "alpha": float(self.alpha),
            "alpha_label": str(self.alpha_label),
            "r": float(self.r),
            "m_max": int(self.m_max),
            "budget": int(self.budget),
        }

    @property
    def run_id(self) -> str:
        # Match the materialized optimizer's run-id definition.  Its control
        # payload deliberately excludes the common campaign budget.
        payload = {
            key: value
            for key, value in self.as_dict().items()
            if key != "budget"
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]


def controls_from_registration(member: Mapping[str, Any], budget: int) -> FrozenControls:
    controls = FrozenControls(
        n_internal=int(member["n_internal"]),
        alpha=float(member["alpha"]),
        alpha_label=str(member["alpha_label"]),
        r=float(member["r"]),
        m_max=int(member["m_max"]),
        budget=int(budget),
    )
    controls.validate()
    return controls
