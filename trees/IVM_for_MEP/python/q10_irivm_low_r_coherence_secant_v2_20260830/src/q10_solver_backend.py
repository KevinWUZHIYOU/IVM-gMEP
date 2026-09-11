#!/usr/bin/env python3
"""Reference-free loader for the frozen QM10 potential and path primitives.

This module deliberately does not import any historical IR-IVM implementation.
It loads only the article QM10 potential, baseline geometry helpers, and the
article core containing the improved-tangent/path primitives needed by the
formal LEPS transfer.  The import-time reference-barrier calculation in the
article core is replaced by a NaN sentinel so the optimizer cannot consult the
offline target barrier.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np


EXPECTED_Q_SHA256 = "372962a47504e6df28a20d8b022482a6835680765434b4385a6dba0e826974bc"


class BackendContractError(RuntimeError):
    """Raised when the frozen QM10 system interface has drifted."""


def load_solver_only_q10(root: Path, n_internal: int):
    """Load a frozen QM10 system backend without evaluating its target barrier."""

    root = Path(root).resolve()
    stage = root / "paper/ActiveExperiments/muller_q10_10d"
    if int(n_internal) == 10:
        search = [stage / "image12/code/experiments"]
        core_path = search[0] / "q10_ivm_core.py"
        potential_path = search[0] / "q10_potential.py"
        potential_logical_file = stage / "image12/data/common/q10_potential.py"
    elif int(n_internal) == 20:
        search = [
            stage / "code/ivm",
            stage / "code/baselines",
            stage / "code/common",
        ]
        core_path = search[0] / "q10_ivm_core.py"
        potential_path = stage / "code/common/q10_potential.py"
        potential_logical_file = stage / "data/common/q10_potential.py"
    else:
        raise BackendContractError("unsupported QM10 path resolution")

    for name in ("q10_potential", "run_q10_baselines"):
        sys.modules.pop(name, None)
    prior_path = list(sys.path)
    try:
        for directory in reversed(search):
            sys.path.insert(0, str(directory))
        potential_spec = importlib.util.spec_from_file_location(
            "q10_potential", potential_path
        )
        if potential_spec is None or potential_spec.loader is None:
            raise ImportError(f"cannot load frozen QM10 potential: {potential_path}")
        potential = importlib.util.module_from_spec(potential_spec)
        potential.__file__ = str(potential_logical_file)
        sys.modules["q10_potential"] = potential
        potential_spec.loader.exec_module(potential)

        q_sha = hashlib.sha256(
            np.ascontiguousarray(potential.Q, dtype=np.float64).tobytes()
        ).hexdigest()
        if q_sha != EXPECTED_Q_SHA256:
            raise BackendContractError("frozen QM10 Q matrix hash drifted")

        baseline = importlib.import_module("run_q10_baselines")

        def unavailable_reference_barrier() -> float:
            return float("nan")

        baseline.reference_barrier = unavailable_reference_barrier
        module_name = f"_formal_leps_transfer_q10_core_n{int(n_internal)}"
        sys.modules.pop(module_name, None)
        core_spec = importlib.util.spec_from_file_location(module_name, core_path)
        if core_spec is None or core_spec.loader is None:
            raise ImportError(f"cannot load frozen QM10 path core: {core_path}")
        core = importlib.util.module_from_spec(core_spec)
        sys.modules[module_name] = core
        core_spec.loader.exec_module(core)
    finally:
        sys.path[:] = prior_path

    if int(core.N_INTERNAL) != int(n_internal):
        raise BackendContractError("QM10 backend resolution drifted")
    if str(core.Q_SHA256) != EXPECTED_Q_SHA256:
        raise BackendContractError("QM10 core Q identity drifted")
    if not np.isnan(float(core.REFERENCE_BARRIER)):
        raise BackendContractError("QM10 optimizer evaluated the reference barrier")
    core.SOLVER_ONLY_REFERENCE_BARRIER = True
    return core, potential
