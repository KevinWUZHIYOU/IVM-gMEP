#!/usr/bin/env python3
"""Build the zero-query ODE-NEB original-32 geometric-action pack."""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
CONTRACT=ROOT/'registrations/diagnostics/BB_ODE8_DISCRETE32_ACTION_PACK_20260902_R1.json'
OUTPUT=ROOT/'runs/diagnostics/bb_ode8_discrete32_action_pack_20260902_r1.npz'
AUDIT=OUTPUT.with_suffix('.audit.json')
ENDPOINT=ROOT/'inputs/endpoint_joint_ef_cache.npz'
EARLY=ROOT/'runs/diagnostics/bb_ode_neb_dense_four_arc500_joint_ef_20260902_r2.shards'
LATE=ROOT/'runs/diagnostics/bb_ode_neb_late_four_arc500_joint_ef_20260902_r1.shards'
FILES=(
 EARLY/'ode_neb_nf60_joint_ef_arc500_v2.npz', EARLY/'ode_neb_nf150_joint_ef_arc500_v2.npz',
 EARLY/'ode_neb_nf300_joint_ef_arc500_v2.npz', EARLY/'ode_neb_nf1320_joint_ef_arc500_v2.npz',
 LATE/'ode_neb_nf02040_arc500_joint_ef_late_r1.npz', LATE/'ode_neb_nf04080_arc500_joint_ef_late_r1.npz',
 LATE/'ode_neb_nf08160_arc500_joint_ef_late_r1.npz', LATE/'ode_neb_nf09990_arc500_joint_ef_late_r1.npz')
EXPECTED_NF=np.array([60,150,300,1320,2040,4080,8160,9990],dtype=np.int64)
EXPECTED_S=np.array([333801.27674335,270739.71247472276,150447.38616112882,10753.393187858715,6120.582683574075,2815.0336180289596,2083.922082393588,1137.411311476047])
KCAL=23.0605

def sha_file(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def sha_array(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def require(c,m):
 if not c: raise RuntimeError(m)
def scalar(z,k): return np.asarray(z[k]).item()
def action(x,f):
 x=np.asarray(x,dtype=np.float64); f=np.asarray(f,dtype=np.float64)
 require(x.shape==f.shape==(32,9396),'original32 shape drift')
 require(np.isfinite(x).all() and np.isfinite(f).all(),'nonfinite original32')
 d=np.linalg.norm(np.diff(x,axis=0),axis=1); n=np.linalg.norm(f,axis=1)
 require(np.all(d>0),'degenerate original32 segment')
 return float(np.sum(0.5*(n[:-1]+n[1:])*d))
def write_npz(path,**a):
 require(not path.exists(),'refusing overwrite'); path.parent.mkdir(parents=True,exist_ok=True)
 t=path.with_name('.'+path.name+f'.tmp.{os.getpid()}')
 with t.open('xb') as f: np.savez(f,**a); f.flush(); os.fsync(f.fileno())
 os.link(t,path); t.unlink()
def write_json(path,x):
 require(not path.exists(),'refusing overwrite'); t=path.with_name('.'+path.name+f'.tmp.{os.getpid()}')
 with t.open('x') as f: json.dump(x,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
 os.link(t,path); t.unlink()
def compute(contract):
 c=json.loads(contract.read_text()); require(c['status']=='FROZEN_ZERO_OPENMM','bad contract')
 require(c['builder_sha256']==sha_file(__file__),'builder drift'); require(c['endpoint_sha256']==sha_file(ENDPOINT),'endpoint drift')
 require(c['source_sha256']==[sha_file(p) for p in FILES],'source drift')
 ep=np.load(ENDPOINT,allow_pickle=False); eco=np.asarray(ep['coordinates_A'],float); ef=np.asarray(ep['forces_eV_per_A'],float)
 nf=[]; ss=[]; ph=[]; fh=[]
 for p in FILES:
  with np.load(p,allow_pickle=False) as z:
   x=np.asarray(z['selected_path'],float); key='internal_diagnostic_force_eV_per_A' if 'internal_diagnostic_force_eV_per_A' in z.files else 'paid_internal_force_eV_per_A'
   fi=np.asarray(z[key],float); require(fi.shape==(30,9396),'internal force shape drift'); require(np.array_equal(x[[0,-1]],eco),'endpoint coordinate mismatch')
   f=np.concatenate((ef[:1],fi,ef[1:]),axis=0); nf.append(int(scalar(z,'optimization_N_F'))); ss.append(action(x,f)); ph.append(sha_array(x)); fh.append(sha_array(f))
 nf=np.asarray(nf,np.int64); ss=np.asarray(ss,float); require(np.array_equal(nf,EXPECTED_NF),'NF drift'); require(np.allclose(ss,EXPECTED_S,rtol=0,atol=1e-9),'action drift')
 return nf,ss,np.asarray(ph),np.asarray(fh)
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--contract',type=Path,default=CONTRACT); p.add_argument('--output',type=Path,default=OUTPUT); a=p.parse_args(argv)
 contract=a.contract.resolve(strict=True); output=a.output.resolve(); require(contract==CONTRACT.resolve() and output==OUTPUT.resolve(),'noncanonical path')
 nf,s,ph,fh=compute(contract); write_npz(output,schema=np.asarray('bb-ode8-discrete32-action-pack-v1'),status=np.asarray('PASS'),optimization_N_F=nf,total_action_eV=s,total_action_kcal_per_mol=s*KCAL,path_sha256=ph,force32_sha256=fh,metric_images=np.asarray(32),metric_segments=np.asarray(31),metric_interpolation=np.asarray('none'),optimization_N_F_contribution=np.asarray(0))
 write_json(AUDIT,{'schema':'bb-ode8-discrete32-action-pack-audit-v1','status':'PASS','result':str(output),'result_sha256':sha_file(output),'contract':str(contract),'contract_sha256':sha_file(contract),'source_shards':[{'path':str(x),'sha256':sha_file(x)} for x in FILES],'endpoint':{'path':str(ENDPOINT),'sha256':sha_file(ENDPOINT)},'physical_calls':0,'optimization_N_F_contribution':0,'arc500_used_by_metric':False,'original_images':32,'segments':31})
 print(output); print(sha_file(output)); print(AUDIT); print(sha_file(AUDIT))
if __name__=='__main__': main()
