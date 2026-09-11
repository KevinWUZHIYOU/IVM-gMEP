#!/usr/bin/env python3
"""Fail-closed authorized R4B entrypoint around unchanged R2 scientific logic."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT/'scripts'))
import evaluate_bb_dense_ivm_short_four_joint_resume_r2 as ENGINE
CONTRACT=ROOT/'registrations/diagnostics/BB_DENSE_COST_IVM_SHORT_FOUR_JOINT_EF_RESUME_20260902_R4B.json'
PREAUTH=ROOT/'registrations/diagnostics/BB_DENSE_COST_IVM_SHORT_FOUR_JOINT_EF_RESUME_20260902_R4B_PREAUTH.json'
AUTH=ROOT/'registrations/execution_authorizations/bb_dense_ivm_short_four_joint_ef_resume_20260902_r4b.json'
RESULT=ROOT/'runs/diagnostics/bb_irivm_short_four_arc500_joint_ef_20260902_r4b.npz'
FREEZER=ROOT/'scripts/freeze_bb_dense_ivm_short_four_joint_resume_r4b_contract.py'; AUTHORIZER=ROOT/'scripts/authorize_bb_dense_ivm_short_four_joint_resume_r4b.py'; TEST=ROOT/'tests/test_run_bb_dense_ivm_short_four_joint_resume_r4b.py'
R3_RECEIPT=ROOT/'registrations/diagnostics/BB_DENSE_COST_IVM_SHORT_FOUR_JOINT_EF_RESUME_20260902_R3_AUDIT_FAILURE_SUPERSESSION.json'
AUTH_SCHEMA='bb-dense-ivm-short-four-resume-r4b-execution-authorization-v1'; AUTH_STATUS='AUTHORIZED_AFTER_ZERO_OPENMM_PREFLIGHT'
def require(c,m): ENGINE.require(c,m)
def bind(p): p=Path(p).resolve(strict=True); return {'path':str(p),'sha256':ENGINE.ENGINE.sha256_file(p)}
def load_contract(path):
 require(path.resolve(strict=True)==CONTRACT.resolve(),'noncanonical R4B contract'); c=json.loads(CONTRACT.read_text())
 for key,p in [('authorized_entrypoint',Path(__file__)),('contract_freezer_R4B',FREEZER),('execution_authorizer',AUTHORIZER),('test',TEST),('R3_supersession_receipt',R3_RECEIPT)]: require(c.get(key)==bind(p),f'R4B {key} binding drift')
 require(c.get('execution_authorization_output')==str(AUTH) and c.get('result_output')==str(RESULT),'R4B output/auth path drift'); require(c.get('R4B_scientific_logic')=='identical_R2_evaluator_only_output_path_and_auth_gate_added','R4B scientific declaration drift'); return c
def validate_authorization(c,path):
 require(path.resolve(strict=True)==AUTH.resolve(),'noncanonical R4B authorization'); a=json.loads(AUTH.read_text())
 require(a.get('schema')==AUTH_SCHEMA and a.get('status')==AUTH_STATUS and a.get('enabled') is True,'R4B authorization invalid')
 expected={'contract':bind(CONTRACT),'authorized_entrypoint':bind(Path(__file__)),'authorizer':bind(AUTHORIZER),'zero_openmm_preflight':bind(PREAUTH),'result_output':str(RESULT),'maximum_joint_EF_attempts':1000,'optimization_N_F_contribution':0,'OpenMM_calls_before_authorization':0,'thread_policy':{n:'1' for n in ENGINE.ENGINE.THREAD_VARS},'canonical_lock_paths':[str(p) for p in sorted(ENGINE.CANONICAL_LOCKS,key=str)]}
 for k,v in expected.items(): require(a.get(k)==v,f'R4B authorization field drift: {k}')
 return a
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--contract',type=Path,required=True); p.add_argument('--authorization',type=Path); p.add_argument('--host-role',choices=('local',),required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--preflight-only',action='store_true'); a=p.parse_args(argv)
 ENGINE.ENGINE.check_thread_caps(); c=load_contract(a.contract); require(a.output.resolve()==RESULT.resolve(),'noncanonical R4B result'); ENGINE.CANONICAL_RESULT=RESULT.resolve()
 if a.preflight_only: require(a.authorization is None and not AUTH.exists(),'preauth mode requires absent authorization')
 else: require(a.authorization is not None,'authorization required'); validate_authorization(c,a.authorization)
 args2=['--contract',str(CONTRACT),'--host-role','local','--output',str(RESULT)]+(['--preflight-only'] if a.preflight_only else []); ENGINE.main(args2)
if __name__=='__main__': main()
