#!/usr/bin/env python3
"""Canonical zero-OpenMM remote4+local2 R5B cost-pack compatibility freezer."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;sys.path.insert(0,str(HERE))
import select_bb_irivm_cost_only_late_six_r3 as S
import freeze_bb_irivm_cost_late_six_arc500_r4 as F
import run_bb_irivm_cost_late_six_arc500_r4 as R
import freeze_bb_irivm_r5b_cost_pack_compat as C
SCHEMA='bb-irivm-r5b-dual-host-cost-pack-compat-v2';STATUS='PASS_CANONICAL_R5B_REMOTE4_LOCAL2_ARC500_CLOSURE'
OUTPUT=ROOT/'registrations/diagnostics/BB_IRIVM_R5B_LATE_SIX_COST_PACK_COMPAT_20260902_V2.json'
def validate_host(role,contract,result):
 state=F.load_host_state(contract,F.BUNDLE,F.PUBLICATION_GATE,role); result=result.resolve(strict=True); C.req(result==Path(state['result_output']).resolve(),'noncanonical host result')
 manifest=result.with_suffix('.manifest.json');audit=result.with_suffix('.audit.json');m=F.read_json(manifest);a=F.read_json(audit)
 C.req(m.get('schema')==R.MANIFEST_SCHEMA and m.get('status')=='PASS' and a.get('schema')==R.AUDIT_SCHEMA and a.get('status')=='PASS','host manifest/audit identity drift');C.req(m.get('contract')==F.bind(state['contract_path']) and m.get('aggregate')==F.bind(result) and a.get('contract')==F.bind(state['contract_path']) and a.get('output')==F.bind(result) and a.get('manifest')==F.bind(manifest),'host result closure drift')
 C.req(a.get('optimization_N_F_contribution')==0 and a.get('original_discrete_new_calls')==0 and a.get('fresh_joint_EF_attempt_lifetime',{}).get('attempt_failed')==0 and a.get('fresh_joint_EF_attempt_lifetime',{}).get('attempt_open')==0,'host accounting drift')
 C.req(len(m.get('shards',[]))==len(state['rows'])==(4 if role=='remote' else 2),'host 4/2 cardinality drift');points=[]
 source=result.with_suffix('.sealed_source.wal.jsonl');densewal=result.with_suffix('.dense_joint_ef.wal.jsonl')
 C.req(a.get('sealed_source_wal')==F.bind(source) and a.get('dense_joint_ef_wal')==F.bind(densewal),'host WAL binding drift')
 with R.configured_delegate(state):
  se=R.DELEGATE._read_wal(source,R.SOURCE_WAL_SCHEMA,state);de=R.DELEGATE._read_wal(densewal,R.DENSE_WAL_SCHEMA,state);life=R.DELEGATE._lifetime(de);expected=len(state['rows'])*498;C.req(life=={'attempt_started':expected,'attempt_succeeded':expected,'attempt_failed':0,'attempt_open':0,'retry_attempts':0},'host WAL lifetime drift')
  C.req(a.get('fresh_joint_EF_attempt_lifetime')==life,'host audit/WAL lifetime drift')
  for row,rec in zip(state['rows'],m['shards']):
   p=Path(rec['path']).resolve(strict=True);C.req(int(rec['calls_per_image'])==int(row['path_identifier']['calls_per_image']) and int(rec['optimization_N_F'])==int(row['optimization_N_F']) and rec['selected_path_sha256']==row['path_sha256'],'host manifest row drift');C.req(F.bind(p)=={'path':str(p),'sha256':rec['sha256']},'host shard binding drift');z=R.DELEGATE._validate_shard(p,row,state);R.DELEGATE._validate_dense_closure(de,row,z,p);R.DELEGATE._validate_source_closure(se,row,p)
   points.append((row,z,p))
 with np.load(result,allow_pickle=False) as z:
  C.req(str(R.DELEGATE.scalar(z,'schema'))==R.RESULT_SCHEMA and str(R.DELEGATE.scalar(z,'status'))=='PASS','host aggregate identity drift');C.req(np.array_equal(np.asarray(z['optimization_N_F']),np.asarray([x['optimization_N_F'] for x in state['rows']])),'host aggregate NF drift');C.req(np.array_equal(np.asarray(z['selected_path_sha256']).astype(str),np.asarray([x['path_sha256'] for x in state['rows']])), 'host aggregate path drift');C.req(int(R.DELEGATE.scalar(z,'optimization_N_F_contribution'))==0,'host aggregate NoF drift')
 return state,points,F.bind(result),F.bind(manifest),F.bind(audit)
def build(output=OUTPUT):
 prereg=S.CANONICAL_PREREGISTRATION;template=S.CANONICAL_SPLIT_TEMPLATE
 primary,supp,manifest,index=S.CANONICAL_PRIMARY,S.CANONICAL_SUPPLEMENTAL,S.CANONICAL_MANIFEST,S.CANONICAL_INDEX
 closure=S.validate_final_closure(prereg,template,primary,supp,manifest,index,enforce_canonical=True);primary_payload=closure[2]
 prep=F.validate_preparation(F.PREPARATION);hosts=[];allp=[]
 for role in ('remote','local'):
  plan=prep['host_plans'][role];state,pts,res,man,aud=validate_host(role,Path(plan['contract_output']),Path(plan['result_output']));hosts.append({'role':role,'result':res,'manifest':man,'audit':aud,'contract':F.bind(state['contract_path'])})
  for row,z,p in pts:
   call=int(row['path_identifier']['calls_per_image']);nf=int(row['optimization_N_F']);inv=[x for x in primary_payload['path_inventory'] if int(x['call_per_image'])==call and int(x['optimization_N_F'])==nf];C.req(len(inv)==1 and inv[0]['formally_reportable'] and not inv[0]['terminal_uncommitted_charged_tail'],'selected R5B inventory mismatch');rp=Path(inv[0]['shard']).resolve(strict=True)
   C.req(F.SELECTOR.sha256_file(rp)==inv[0]['shard_sha256'],'R5B shard file binding drift')
   with np.load(rp,allow_pickle=False) as q:
    C.req(str(R.DELEGATE.scalar(q,'schema'))=='bb-irivm-full-state-history-recorder-v3' and int(R.DELEGATE.scalar(q,'call_per_image'))==call and int(R.DELEGATE.scalar(q,'optimization_N_F'))==nf,'R5B shard identity drift');path=np.asarray(q['path32_A']);force=np.asarray(q['force_eV_per_A']);val=C.iperp(path,force);C.req(C.sha_array(path)==row['path_sha256']==inv[0]['path_sha256']==str(R.DELEGATE.scalar(q,'path_sha256')) and C.sha_array(force)==inv[0]['force_sha256']==str(R.DELEGATE.scalar(q,'force_sha256')),'R5B path/force closure drift');C.req(np.asarray(val).tobytes()==np.asarray(R.DELEGATE.scalar(q,'discrete_I_perp')).tobytes()==np.asarray(row['discrete_I_perp']).tobytes()==np.asarray(R.DELEGATE.scalar(z,'discrete_I_perp')).tobytes(),'weighted Iperp closure drift')
   dense=np.asarray(z['dense_coordinates_A']);energy=np.asarray(z['dense_energy_eV']);ff=np.asarray(z['dense_force_eV_per_A']);C.req(np.array_equal(dense,C.uniform500(path)) and C.sha_array(dense)==str(R.DELEGATE.scalar(z,'dense_coordinate_sha256')) and C.sha_array(energy)==str(R.DELEGATE.scalar(z,'dense_energy_sha256')) and C.sha_array(ff)==str(R.DELEGATE.scalar(z,'dense_force_sha256')),'arc500 closure drift');C.req(int(R.DELEGATE.scalar(z,'fresh_dense_joint_EF_attempts'))==498 and int(R.DELEGATE.scalar(z,'inherited_endpoint_joint_EF_values'))==2 and int(R.DELEGATE.scalar(z,'optimization_N_F_contribution'))==0,'arc500 accounting drift')
   allp.append({'host':role,'selection_role':row['selection_role'],'call_per_image':call,'optimization_N_F':nf,'path_sha256':inv[0]['path_sha256'],'force32_sha256':inv[0]['force_sha256'],'discrete_I_perp':val,'densified_Delta_E_kcal_per_mol':float(R.DELEGATE.scalar(z,'densified_Delta_E_kcal_per_mol')),'arc500_shard':F.bind(p),'R5B_shard':F.bind(rp),'optimization_N_F_contribution':0})
 C.req(len(allp)==6 and len({x['selection_role'] for x in allp})==len({x['optimization_N_F'] for x in allp})==6,'merged six-point uniqueness drift')
 payload={'schema':SCHEMA,'status':STATUS,'point_count':6,'points':sorted(allp,key=lambda x:x['optimization_N_F']),'hosts':hosts,'bindings':{'primary':F.bind(primary),'supplemental':F.bind(supp),'manifest':F.bind(manifest),'canonical_index':F.bind(index),'preparation':F.bind(F.PREPARATION),'bundle':F.bind(F.BUNDLE),'publication_gate':F.bind(F.PUBLICATION_GATE)},'optimization_N_F_contribution':0,'OpenMM_calls':0};output=output.resolve();C.req(output==OUTPUT.resolve() and not output.exists(),'noncanonical/existing compat output');F.write_exclusive(output,payload);return payload
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);a=p.parse_args();print(json.dumps(build(a.output),indent=2))
if __name__=='__main__':main()
