#!/usr/bin/env python3
"""R4 authorization-closure wrapper for the frozen R3 energy evaluator.

The R3 implementation is preserved byte-for-byte.  R4 changes only prospective
identity/authorization validation and delegates all coordinate, energy, WAL,
mask, result, and audit logic to that hash-bound implementation.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent.parent
BASE_PATH = ROOT / "scripts/evaluate_bb_irivm_nf8982_0k_section_energy.py"
BASE_SHA256 = "af17cda66bee788d23b3948f6a0cb0ebad9ab8d061fac6cbf62a96e3a0000c9c"
AUTHORIZER_PATH = ROOT / "scripts/authorize_bb_irivm_nf8982_0k_section_energy_r4.py"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if _sha256(BASE_PATH) != BASE_SHA256:
    raise RuntimeError("frozen R3 base evaluator SHA-256 mismatch")
_spec = importlib.util.spec_from_file_location("bb_nf8982_energy_r3_frozen_base", BASE_PATH)
_base = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_base)

# The delegated functions resolve their prospective identity through the
# module-global __file__, so bind that identity to this R4 wrapper.
_base.__file__ = str(THIS_FILE)
_base.CONTRACT_SCHEMA = "bb-irivm-nf8982-0k-section-energy-contract-v3"
_base.CONTRACT_STATUS = "FROZEN_R4_AUTHORIZATION_CLOSURE_BEFORE_FIRST_SECTION_ENERGY_CALL"
_base.AUTH_SCHEMA = "bb-irivm-nf8982-0k-section-energy-authorization-v3"
_base.PREAUTH_SCHEMA = "bb-irivm-nf8982-0k-section-energy-preauthorization-preflight-v2"
_base.RESULT_SCHEMA = "bb-irivm-nf8982-0k-section-energy-result-v3"
_base.AUDIT_SCHEMA = "bb-irivm-nf8982-0k-section-energy-audit-v3"
_base.WAL_SCHEMA = "bb-irivm-nf8982-0k-section-energy-wal-v3"

_original_load_contract = _base.load_contract


def load_contract(contract_path, authorization_path, host_role, output, stage):
    state = _original_load_contract(contract_path, authorization_path, host_role, output, stage)
    contract = state["contract"]
    authorization = state["authorization"]
    expected_authorization = Path(str(contract.get("execution_authorization_output", ""))).expanduser().resolve()
    _base.require(authorization_path == expected_authorization, "authorization path differs from frozen contract")
    authorizer = _base.resolve_binding(contract_path, contract.get("execution_authorizer", {}), "execution authorizer")
    _base.require(authorizer == AUTHORIZER_PATH.resolve(strict=True), "noncanonical R4 execution authorizer")
    _base.require(authorization.get("authorizer") == _base.file_binding(authorizer), "authorization authorizer binding mismatch")

    preauth = __import__("json").loads(state["preauthorization_preflight"].read_text(encoding="utf-8"))
    expected_locks = {str(path) for path in _base.CANONICAL_LOCKS}
    probe = preauth.get("canonical_two_lock_nonblocking_probe", {})
    _base.require(
        probe.get("status") == "PASS" and set(probe.get("locks", [])) == expected_locks,
        "preauthorization canonical lock-set probe mismatch",
    )
    _base.require(preauth.get("thread_policy") == {name: "1" for name in _base.THREAD_VARS}, "preauthorization thread policy mismatch")
    _base.require(int(preauth.get("physical_calls_started", -1)) == 0, "preauthorization physical-call count mismatch")
    _base.require(int(preauth.get("openmm_contexts_constructed", -1)) == 0, "preauthorization constructed an OpenMM context")
    _base.require(int(preauth.get("path_stage_fresh_attempt_maximum", -1)) == _base.PATH_INTERNAL_CALLS, "preauthorization path count mismatch")
    _base.require(int(preauth.get("surface_stage_fresh_attempt_maximum", -1)) == _base.SURFACE_FRESH_CALLS, "preauthorization surface count mismatch")
    _base.require(int(preauth.get("total_fresh_attempt_maximum", -1)) == _base.TOTAL_FRESH_CALLS, "preauthorization total count mismatch")
    return state


_base.load_contract = load_contract

# Re-export the complete tested implementation surface for freezer/tests.
for _name in dir(_base):
    if not _name.startswith("__"):
        globals().setdefault(_name, getattr(_base, _name))
globals()["load_contract"] = load_contract
CONTRACT_SCHEMA = _base.CONTRACT_SCHEMA
CONTRACT_STATUS = _base.CONTRACT_STATUS
AUTH_SCHEMA = _base.AUTH_SCHEMA
PREAUTH_SCHEMA = _base.PREAUTH_SCHEMA
RESULT_SCHEMA = _base.RESULT_SCHEMA
AUDIT_SCHEMA = _base.AUDIT_SCHEMA
WAL_SCHEMA = _base.WAL_SCHEMA


if __name__ == "__main__":
    _base.main()
