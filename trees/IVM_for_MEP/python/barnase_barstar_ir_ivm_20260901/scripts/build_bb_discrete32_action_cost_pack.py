#!/usr/bin/env python3
"""Build the final zero-query IVM/ODE-NEB discrete-32 action-cost pack."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import numpy as np
import build_bb_ode8_discrete32_action_pack as O

ROOT=Path(__file__).resolve().parent.parent
CONTRACT=ROOT/'registrations/diagnostics/BB_DISCRETE32_ACTION_COST_PACK_20260902_R1.json'
OUTPUT=ROOT/'runs/diagnostics/bb_discrete32_action_cost_pack_20260902_r1.npz'
AUDIT=OUTPUT.with_suffix('.audit.json')
ODE=ROOT/'runs/diagnostics/bb_ode8_discrete32_action_pack_20260902_r1.npz'
ODE_AUDIT=ODE.with_suffix('.audit.json')
PRIMARY=ROOT/'reports/verifications/BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B_PRIMARY.json'
SUPPLEMENTAL=ROOT/'reports/verifications/BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B_SUPPLEMENTAL.json'
R5B_REGISTRATION=ROOT/'registrations/diagnostics/BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B.json'
R5B_AUTHORIZATION=ROOT/'registrations/execution_authorizations/bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b.json'
R5B_OBSERVER=ROOT/'runs/full_state_history_replay_r5b/bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b_observer'
R5B_MANIFEST=R5B_OBSERVER/'full_state_history.manifest.json'
R5B_INDEX=R5B_OBSERVER/'full_state_history.canonical_index.json'
R5B_CAPTURE_AUDIT=R5B_OBSERVER/'full_state_history.audit.json'
R5B_REFERENCE_RESULT=ROOT/'runs/production/ivm_bb_irivm_a1e0_r01_m05_inf9990_r1_central_state_ivm_cfge0354bb7178e_budget9990_storagecommitted.npz'
R5B_OUTPUT_DIR=ROOT/'runs/full_state_history_replay_r5b'
PRIMARY_SCHEMA='bb-irivm-selected-full-state-history-verification-v5b'
PRIMARY_STATUS='PASS_R5B_EXACT_PREFIX318_AND_DYNAMIC_STABLE_LAST_FULL_STATE_REPLAY'
SUPPLEMENTAL_SCHEMA='bb-irivm-full-state-r5b-supplemental-verification-v1'
SUPPLEMENTAL_STATUS='PASS_R5B_EXACT_COUNTER_SCHEMA_AND_ARTIFACT_CLOSURE'
R5B_BUDGET=15000
OLD_BUDGET=9990
OLD_COMPLETE_PREFIX_CALLS=318
OLD_COMPLETE_PREFIX_NF=8982

def lower_envelope(y):
 y=np.asarray(y,dtype=np.float64); O.require(y.ndim==1 and y.size and np.isfinite(y).all(),'invalid values')
 return np.minimum.accumulate(y)
def binding(p): p=Path(p).resolve(strict=True); return {'path':str(p),'sha256':O.sha_file(p)}
def _bound(payload,name):
 b=payload.get('bindings',{}).get(name); O.require(isinstance(b,dict),'missing verifier binding '+name)
 p=Path(b['path']).resolve(strict=True); O.require(O.sha_file(p)==b['sha256'],'verifier binding drift '+name); return p
def validate_ode_pack_and_audit(pack_path=ODE,audit_path=ODE_AUDIT,require_canonical=True):
 pack_path=Path(pack_path).resolve(strict=True); audit_path=Path(audit_path).resolve(strict=True)
 if require_canonical: O.require(pack_path==ODE.resolve() and audit_path==ODE_AUDIT.resolve(),'noncanonical ODE action pack/audit path')
 audit=json.loads(audit_path.read_text())
 O.require(audit.get('schema')=='bb-ode8-discrete32-action-pack-audit-v1' and audit.get('status')=='PASS','ODE action audit is not PASS')
 O.require(Path(audit.get('result','')).resolve()==pack_path and audit.get('result_sha256')==O.sha_file(pack_path),'ODE action audit/result binding drift')
 ode_contract=Path(audit.get('contract','')).resolve(strict=True)
 O.require(ode_contract==O.CONTRACT.resolve() and audit.get('contract_sha256')==O.sha_file(ode_contract),'ODE action audit/contract binding drift')
 O.require(audit.get('arc500_used_by_metric') is False and int(audit.get('original_images',-1))==32 and int(audit.get('segments',-1))==31 and int(audit.get('physical_calls',-1))==0 and int(audit.get('optimization_N_F_contribution',-1))==0,'ODE action audit metric/NoF policy drift')
 with np.load(pack_path,allow_pickle=False) as saved:
  required={'schema','status','optimization_N_F','total_action_eV','total_action_kcal_per_mol','path_sha256','force32_sha256','metric_images','metric_segments','metric_interpolation','optimization_N_F_contribution'}
  O.require(required.issubset(saved.files),'ODE action pack fields missing')
  O.require(str(O.scalar(saved,'schema'))=='bb-ode8-discrete32-action-pack-v1' and str(O.scalar(saved,'status'))=='PASS','ODE action pack is not PASS')
  nf=np.asarray(saved['optimization_N_F'],np.int64); action=np.asarray(saved['total_action_eV'],float); kcal=np.asarray(saved['total_action_kcal_per_mol'],float); paths=np.asarray(saved['path_sha256']).astype(str); forces=np.asarray(saved['force32_sha256']).astype(str)
  O.require(np.array_equal(nf,O.EXPECTED_NF) and np.allclose(action,O.EXPECTED_S,rtol=0,atol=1e-9),'ODE action NF/value identity drift')
  O.require(action.shape==kcal.shape==paths.shape==forces.shape==nf.shape and np.allclose(kcal,action*O.KCAL,rtol=0,atol=1e-12),'ODE action pack vector/unit drift')
  O.require(int(O.scalar(saved,'metric_images'))==32 and int(O.scalar(saved,'metric_segments'))==31 and str(O.scalar(saved,'metric_interpolation'))=='none' and int(O.scalar(saved,'optimization_N_F_contribution'))==0,'ODE action pack metric/NoF policy drift')
 return nf,action,audit
def validate_r5b_verifiers(primary_path, supplemental_path, require_canonical=True):
 primary_path=Path(primary_path).resolve(strict=True); supplemental_path=Path(supplemental_path).resolve(strict=True)
 if require_canonical:
  O.require(primary_path==PRIMARY.resolve() and supplemental_path==SUPPLEMENTAL.resolve(),'noncanonical NF15000 R5B verifier path')
 primary=json.loads(primary_path.read_text()); supplemental=json.loads(supplemental_path.read_text())
 O.require(primary.get('schema')==PRIMARY_SCHEMA and primary.get('status')==PRIMARY_STATUS,'primary verifier is not the exact NF15000 R5B PASS artifact')
 O.require(supplemental.get('schema')==SUPPLEMENTAL_SCHEMA and supplemental.get('status')==SUPPLEMENTAL_STATUS,'supplemental verifier is not the exact NF15000 R5B PASS artifact')
 O.require(primary.get('exact_old_9990_complete_prefix_call_range')==[1,OLD_COMPLETE_PREFIX_CALLS] and int(primary.get('exact_old_9990_complete_prefix_last_N_F',-1))==OLD_COMPLETE_PREFIX_NF and primary.get('old_9990_terminal_tail_required_equal') is False,'R5B exact-prefix policy drift')
 canonical_count=int(primary.get('actual_canonical_call_count',-1)); last=int(primary.get('last_complete_call',-1)); terminal_nf=int(primary.get('terminal_optimization_N_F',-1))
 O.require(OLD_COMPLETE_PREFIX_CALLS<=last<=canonical_count and OLD_BUDGET<terminal_nf<=R5B_BUDGET,'R5B dynamic terminal/last-complete bounds invalid')
 inventory=primary.get('path_inventory'); O.require(isinstance(inventory,list) and len(inventory)==canonical_count,'R5B primary inventory/count drift')
 calls=[int(r.get('call_per_image',-1)) for r in inventory]; histories=[int(r.get('history_index',-1)) for r in inventory]
 O.require(calls==list(range(1,canonical_count+1)) and histories==list(range(canonical_count)),'R5B primary dynamic call axis is not continuous')
 nfs=[int(r.get('optimization_N_F',-1)) for r in inventory]
 O.require(all(value>0 for value in nfs) and all(a<=b for a,b in zip(nfs,nfs[1:])),'R5B primary optimization N_F axis invalid')
 O.require(nfs[OLD_COMPLETE_PREFIX_CALLS-1]==OLD_COMPLETE_PREFIX_NF,'R5B inventory does not contain the sealed call318/NF8982 prefix point')
 for call,row in enumerate(inventory,1):
  reportable=call<=last
  O.require(row.get('formally_reportable') is reportable and row.get('terminal_uncommitted_charged_tail') is (not reportable),'R5B reportability/tail partition drift')
 report=inventory[:last]
 O.require(report and int(report[-1]['optimization_N_F'])==max(int(r['optimization_N_F']) for r in report),'R5B dynamic highest reportable point invalid')
 O.require(int(report[-1]['optimization_N_F'])>OLD_BUDGET and int(report[-1]['optimization_N_F'])<=R5B_BUDGET,'R5B reportable history is truncated at the old 9990 budget')
 O.require(int(supplemental.get('canonical_record_count',-1))==canonical_count,'supplemental canonical count differs from primary')
 sinventory=supplemental.get('inventory'); O.require(isinstance(sinventory,list) and len(sinventory)==canonical_count,'supplemental inventory/count drift')
 O.require([(int(r.get('call_per_image',-1)),int(r.get('optimization_N_F',-1))) for r in sinventory]==list(zip(calls,nfs)),'supplemental call/N_F inventory differs from primary')
 registration=_bound(primary,'registration'); authorization=_bound(primary,'authorization'); reference=_bound(primary,'reference_result'); replay=_bound(primary,'replay_result'); replay_audit=_bound(primary,'replay_audit'); manifest=_bound(primary,'capture_manifest'); index=_bound(primary,'canonical_index'); audit=_bound(primary,'capture_audit')
 runtime_audit=json.loads(replay_audit.read_text())
 O.require(runtime_audit.get('consistent') is True and Path(runtime_audit.get('result_file','')).resolve()==replay,'primary replay result/audit closure drift')
 if require_canonical:
  O.require(registration==R5B_REGISTRATION.resolve() and authorization==R5B_AUTHORIZATION.resolve() and reference==R5B_REFERENCE_RESULT.resolve() and replay.parent==R5B_OUTPUT_DIR.resolve() and replay.name.startswith('ivm_bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b_') and replay.suffix=='.npz' and replay_audit==replay.with_suffix('.audit.json') and manifest==R5B_MANIFEST.resolve() and index==R5B_INDEX.resolve() and audit==R5B_CAPTURE_AUDIT.resolve(),'primary verifier does not bind the canonical NF15000 R5B closure')
 sb=supplemental.get('bindings',{})
 O.require(sb.get('primary_verification')==binding(primary_path),'supplemental does not bind the exact primary verifier')
 O.require(sb.get('registration')==binding(registration) and sb.get('authorization')==binding(authorization) and sb.get('manifest')==binding(manifest) and sb.get('canonical_index')==binding(index) and sb.get('capture_audit')==binding(audit),'supplemental does not bind the same canonical final R5B closure')
 return primary,supplemental,inventory,last,manifest,index,audit
def build(contract):
 c=json.loads(contract.read_text()); O.require(c.get('schema')=='bb-discrete32-action-cost-pack-contract-v1' and c.get('status')=='FROZEN_AFTER_R5B_PRIMARY_AND_SUPPLEMENTAL_PASS','contract not final')
 for k in ('builder','freezer','plotter','test','ODE_pack','ODE_audit','R5B_primary_verifier','R5B_supplemental_verifier'):
  p=Path(c[k]['path']).resolve(strict=True); O.require(O.sha_file(p)==c[k]['sha256'],'contract binding drift '+k)
 expected={'builder':Path(__file__).resolve(),'freezer':ROOT/'scripts/freeze_bb_discrete32_action_cost_pack.py','plotter':ROOT/'scripts/plot_bb_discrete32_action_cost.py','test':ROOT/'tests/test_bb_discrete32_action_cost.py','ODE_pack':ODE,'ODE_audit':ODE_AUDIT,'R5B_primary_verifier':PRIMARY,'R5B_supplemental_verifier':SUPPLEMENTAL}
 for k,p in expected.items(): O.require(Path(c[k]['path']).resolve()==p.resolve(),'contract dependency path drift '+k)
 O.require(Path(c.get('output','')).resolve()==OUTPUT.resolve() and Path(c.get('audit','')).resolve()==AUDIT.resolve(),'contract output path drift')
 primary,supplemental,inventory,last,manifest,index,_audit=validate_r5b_verifiers(c['R5B_primary_verifier']['path'],c['R5B_supplemental_verifier']['path'])
 report=inventory[:last]
 O.require(int(c.get('R5B_budget',-1))==R5B_BUDGET and int(c.get('R5B_dynamic_canonical_call_count',-1))==len(inventory) and int(c.get('R5B_dynamic_last_complete_call',-1))==last and int(c.get('R5B_dynamic_highest_reportable_optimization_N_F',-1))==int(report[-1]['optimization_N_F']),'contract dynamic R5B inventory summary drift')
 ir_nf=[]; ir_s=[]; ir_path=[]; ir_force=[]; shards=[]
 for r in report:
  p=Path(r['shard']).resolve(strict=True); O.require(O.sha_file(p)==r['shard_sha256'],'R5B shard drift')
  with np.load(p,allow_pickle=False) as z:
   call=int(O.scalar(z,'call_per_image')); nf=int(O.scalar(z,'optimization_N_F')); x=np.asarray(z['path32_A'],float); f=np.asarray(z['force_eV_per_A'],float)
   O.require(call==int(r['call_per_image']) and nf==int(r['optimization_N_F']),'R5B identity drift')
   O.require(O.sha_array(x)==r['path_sha256'] and O.sha_array(f)==r['force_sha256'],'R5B path/force drift')
   ir_nf.append(nf); ir_s.append(O.action(x,f)); ir_path.append(r['path_sha256']); ir_force.append(r['force_sha256']); shards.append(binding(p))
 ir_nf=np.asarray(ir_nf,np.int64); ir_s=np.asarray(ir_s,float); O.require(np.all(ir_nf[1:]>=ir_nf[:-1]),'IVM NF not monotone')
 onf,os_,_ode_audit=validate_ode_pack_and_audit()
 return c,primary,supplemental,ir_nf,ir_s,np.asarray(ir_path),np.asarray(ir_force),shards,onf,os_
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--contract',type=Path,default=CONTRACT); p.add_argument('--output',type=Path,default=OUTPUT); a=p.parse_args(argv)
 cp=a.contract.resolve(strict=True); out=a.output.resolve(); O.require(cp==CONTRACT.resolve() and out==OUTPUT.resolve(),'noncanonical output')
 c,pr,su,inf,is_,iph,ifh,sh,onf,os_=build(cp)
 O.write_npz(out,schema=np.asarray('bb-discrete32-action-cost-pack-v1'),status=np.asarray('PASS'),ivm_optimization_N_F=inf,ivm_total_action_eV=is_,ivm_total_action_kcal_per_mol=is_*O.KCAL,ivm_observed_lower_envelope_eV=lower_envelope(is_),ivm_path_sha256=iph,ivm_force_sha256=ifh,ode_optimization_N_F=onf,ode_total_action_eV=os_,ode_total_action_kcal_per_mol=os_*O.KCAL,ode_observed_lower_envelope_eV=lower_envelope(os_),R5B_budget=np.asarray(R5B_BUDGET),R5B_dynamic_last_complete_call=np.asarray(int(pr['last_complete_call'])),R5B_dynamic_highest_reportable_optimization_N_F=np.asarray(int(inf[-1])),metric_images=np.asarray(32),metric_segments=np.asarray(31),metric_interpolation=np.asarray('none'),optimization_N_F_contribution=np.asarray(0))
 O.write_json(AUDIT,{'schema':'bb-discrete32-action-cost-pack-audit-v1','status':'PASS','result':binding(out),'contract':binding(cp),'R5B_primary_verifier':c['R5B_primary_verifier'],'R5B_supplemental_verifier':c['R5B_supplemental_verifier'],'R5B_budget':R5B_BUDGET,'R5B_dynamic_canonical_call_count':int(pr['actual_canonical_call_count']),'R5B_last_complete_call':int(pr['last_complete_call']),'R5B_dynamic_highest_reportable_optimization_N_F':int(inf[-1]),'R5B_terminal_tail_excluded':True,'old_NF9990_artifacts_role':'exact-prefix evidence only; never a final input','IVM_shards':sh,'ODE_pack':binding(ODE),'metric_images':32,'metric_segments':31,'metric_interpolation':'none','arc500_used_by_metric':False,'physical_calls':0,'optimization_N_F_contribution':0})
 print(out); print(O.sha_file(out)); print(AUDIT); print(O.sha_file(AUDIT))
if __name__=='__main__': main()
