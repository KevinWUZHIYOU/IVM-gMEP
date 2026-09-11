#!/usr/bin/env python3
"""Build a zero-OpenMM R5B-to-late-arc500 cost plotting pack."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np

PRIMARY_SCHEMA='bb-irivm-selected-full-state-history-verification-v5b'
PRIMARY_STATUS='PASS_R5B_EXACT_PREFIX318_AND_DYNAMIC_STABLE_LAST_FULL_STATE_REPLAY'
SUPP_SCHEMA='bb-irivm-full-state-r5b-supplemental-verification-v1'
SUPP_STATUS='PASS_R5B_EXACT_COUNTER_SCHEMA_AND_ARTIFACT_CLOSURE'
PACK_SCHEMA='bb-irivm-r5b-late-arc500-cost-pack-compat-v1'; PACK_STATUS='PASS_R5B_FULL_STATE_AND_ARC500_EXACT_CLOSURE'
R5_SHARD_SCHEMA='bb-irivm-full-state-history-recorder-v3'; LATE_MANIFEST_SCHEMA='bb-irivm-late-six-arc500-joint-ef-manifest-v1'; LATE_SHARD_SCHEMA='bb-irivm-late-path-arc500-joint-ef-shard-v1'
REG=Path('/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/registrations/diagnostics/BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF15000_20260902_R5B.json')
AUTH=Path('/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/registrations/execution_authorizations/bb_irivm_a1e0_r01_m05_inf15000_fullstate_r5b.json')
def req(x,m):
 if not x: raise RuntimeError(m)
def sha_file(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def sha_array(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def bind(p): p=Path(p).resolve(strict=True); return {'path':str(p),'sha256':sha_file(p)}
def scalar(z,k):
 a=np.asarray(z[k]); req(a.shape==(),f'{k} non-scalar'); return a.item()
def iperp(path,force):
 t=path[2:]-path[:-2]; t=t/np.linalg.norm(t,axis=1,keepdims=True); f=force[1:-1]; n=f-(f*t).sum(axis=1,keepdims=True)*t
 segment=np.linalg.norm(np.diff(path,axis=0),axis=1); weights=.5*(segment[:-1]+segment[1:])
 return float(np.sum(weights*np.linalg.norm(n,axis=1)**2))
def uniform500(path):
 d=np.linalg.norm(np.diff(path,axis=0),axis=1); c=np.r_[0.,np.cumsum(d)]; q=np.linspace(0,c[-1],500); i=np.clip(np.searchsorted(c,q,side='right')-1,0,30); w=(q-c[i])/d[i]; x=(1-w[:,None])*path[i]+w[:,None]*path[i+1]; x[0]=path[0];x[-1]=path[-1];return np.ascontiguousarray(x)
def build(primary_p,supp_p,late_manifest_p,output):
 primary=json.loads(primary_p.read_text()); supp=json.loads(supp_p.read_text()); manifest=json.loads(late_manifest_p.read_text()); req(manifest.get('schema')==LATE_MANIFEST_SCHEMA and manifest.get('status')=='PASS','late manifest not PASS')
 req(primary.get('schema')==PRIMARY_SCHEMA and primary.get('status')==PRIMARY_STATUS,'R5B primary not PASS'); req(supp.get('schema')==SUPP_SCHEMA and supp.get('status')==SUPP_STATUS,'R5B supplemental not PASS')
 req(primary.get('bindings',{}).get('registration')==bind(REG) and primary.get('bindings',{}).get('authorization')==bind(AUTH),'noncanonical NF15000 source')
 req(supp.get('bindings',{}).get('primary_verification')==bind(primary_p),'supplemental/primary drift'); last=int(primary['last_complete_call']); inv=primary['path_inventory']; req(len(inv)==primary['actual_canonical_call_count'],'R5B inventory count drift')
 by_nf={int(x['optimization_N_F']):x for x in inv if x.get('formally_reportable') and int(x['call_per_image'])<=last}; req(len(by_nf)==sum(x.get('formally_reportable') and int(x['call_per_image'])<=last for x in inv),'duplicate reportable NF')
 points=[]
 for row in manifest['shards']:
  lp=Path(row['path']).resolve(strict=True); req(sha_file(lp)==row['sha256'],'late shard binding drift')
  with np.load(lp,allow_pickle=False) as l:
   req(str(scalar(l,'schema'))==LATE_SHARD_SCHEMA and str(scalar(l,'status'))=='PASS','late shard identity drift')
   nf=int(scalar(l,'optimization_N_F')); src=by_nf.get(nf); req(src is not None,'late NF absent from reportable R5B inventory'); rp=Path(src['shard']).resolve(strict=True); req(sha_file(rp)==src['shard_sha256'],'R5B shard binding drift')
   with np.load(rp,allow_pickle=False) as r:
    req(str(scalar(r,'schema'))==R5_SHARD_SCHEMA,'R5B shard schema drift')
    path=np.asarray(r['path32_A'],dtype=np.float64); force32=np.asarray(r['force_eV_per_A'],dtype=np.float64); req(path.shape==force32.shape==(32,9396),'R5B full state shape drift'); req(sha_array(path)==src['path_sha256']==str(scalar(l,'selected_path_sha256')),'same-path closure failed'); req(sha_array(force32)==src['force_sha256'],'R5B force hash drift'); value=iperp(path,force32); req(np.asarray(value).tobytes()==np.asarray(scalar(r,'discrete_I_perp')).tobytes()==np.asarray(scalar(l,'discrete_I_perp')).tobytes(),'Iperp closure failed')
   dense=np.asarray(l['dense_coordinates_A']); energy=np.asarray(l['dense_energy_eV']); force=np.asarray(l['dense_force_eV_per_A']); req(dense.shape==force.shape==(500,9396) and energy.shape==(500,) and all(np.all(np.isfinite(x)) for x in (dense,energy,force)),'arc500 shape/finite drift'); req(np.array_equal(dense,uniform500(path)),'arc500 coordinate drift'); req(sha_array(dense)==str(scalar(l,'dense_coordinate_sha256')) and sha_array(energy)==str(scalar(l,'dense_energy_sha256')) and sha_array(force)==str(scalar(l,'dense_force_sha256')),'arc500 hash drift'); req(int(scalar(l,'optimization_N_F_contribution'))==0,'diagnostic NoF contribution nonzero')
   points.append({'optimization_N_F':nf,'call_per_image':int(src['call_per_image']),'path_sha256':src['path_sha256'],'force32_sha256':src['force_sha256'],'discrete_I_perp':value,'densified_Delta_E_kcal_per_mol':float(scalar(l,'densified_Delta_E_kcal_per_mol')),'late_shard':bind(lp),'R5B_full_state_shard':bind(rp),'optimization_N_F_contribution':0})
 req(points and len({x['optimization_N_F'] for x in points})==len(points),'empty/duplicate points')
 payload={'schema':PACK_SCHEMA,'status':PACK_STATUS,'method':'IVM','point_count':len(points),'points':sorted(points,key=lambda x:x['optimization_N_F']),'last_complete_call':last,'bindings':{'R5B_primary':bind(primary_p),'R5B_supplemental':bind(supp_p),'late_manifest':bind(late_manifest_p),'registration':bind(REG),'authorization':bind(AUTH)},'OpenMM_calls':0,'optimization_N_F_contribution':0}
 output=output.resolve(); req(not output.exists(),'refusing overwrite'); output.parent.mkdir(parents=True,exist_ok=True); tmp=output.with_name('.'+output.name+f'.{os.getpid()}'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); os.link(tmp,output);tmp.unlink();return payload
def main():
 p=argparse.ArgumentParser();p.add_argument('--primary',type=Path,required=True);p.add_argument('--supplemental',type=Path,required=True);p.add_argument('--late-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(build(a.primary.resolve(strict=True),a.supplemental.resolve(strict=True),a.late_manifest.resolve(strict=True),a.output),indent=2))
if __name__=='__main__':main()
