#!/usr/bin/env python3
"""Authorized R3 entry point for the unchanged R2 short-four evaluator."""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_bb_dense_ivm_short_four_joint_resume_r2 as ENGINE

CONTRACT = ROOT / "registrations/diagnostics/BB_DENSE_COST_IVM_SHORT_FOUR_JOINT_EF_RESUME_20260902_R3.json"
AUTH = ROOT / "registrations/execution_authorizations/bb_dense_ivm_short_four_joint_ef_resume_20260902_r3.json"
RESULT = ROOT / "runs/diagnostics/bb_irivm_short_four_arc500_joint_ef_20260902_r3.npz"
AUTH_SCHEMA = "bb-dense-ivm-short-four-resume-execution-authorization-v1"
AUTH_STATUS = "AUTHORIZED_AFTER_ZERO_OPENMM_PREFLIGHT"

def bind(path):
    p=Path(path).resolve(strict=True); return {"path":str(p),"sha256":ENGINE.ENGINE.sha256_file(p)}
def require(c,m): ENGINE.require(c,m)
def validate_contract(contract_path):
    require(contract_path.resolve(strict=True)==CONTRACT.resolve(),"noncanonical R3 contract")
    c=json.loads(CONTRACT.read_text())
    require(c.get("authorized_entrypoint")==bind(Path(__file__)),"R3 entrypoint binding drift")
    require(c.get("execution_authorization_output")==str(AUTH),"R3 auth path drift")
    return c
def validate(contract_path, auth_path):
    c=validate_contract(contract_path)
    require(auth_path.resolve(strict=True)==AUTH.resolve(),"noncanonical R3 authorization")
    a=json.loads(AUTH.read_text())
    require(a.get("schema")==AUTH_SCHEMA and a.get("status")==AUTH_STATUS and a.get("enabled") is True,"R3 authorization invalid")
    require(a.get("contract")==bind(CONTRACT) and a.get("authorized_entrypoint")==bind(Path(__file__)),"R3 authorization binding drift")
    require(a.get("result_output")==str(RESULT) and a.get("maximum_joint_EF_attempts")==1000,"R3 authorization scope drift")
    return c
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--contract",type=Path,required=True); p.add_argument("--authorization",type=Path,required=True); p.add_argument("--host-role",choices=("local",),required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--preflight-only",action="store_true"); a=p.parse_args(argv)
    ENGINE.ENGINE.check_thread_caps()
    if a.preflight_only:
        validate_contract(a.contract)
        require(not AUTH.exists(),"preauthorization preflight requires absent authorization")
    else: validate(a.contract,a.authorization)
    require(a.output.resolve()==RESULT.resolve(),"noncanonical R3 result")
    ENGINE.CANONICAL_RESULT=RESULT.resolve()
    forwarded=["--contract",str(CONTRACT),"--host-role","local","--output",str(RESULT)]
    if a.preflight_only: forwarded.append("--preflight-only")
    ENGINE.main(forwarded)
if __name__=="__main__": main()
