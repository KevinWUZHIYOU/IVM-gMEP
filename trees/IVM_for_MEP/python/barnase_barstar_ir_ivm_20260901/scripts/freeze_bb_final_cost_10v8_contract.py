#!/usr/bin/env python3
"""Freeze the final BB 10-IVM/8-ODE observed-only figure contract (zero OpenMM)."""
from __future__ import annotations
import argparse,json,os,sys,tempfile
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; sys.path.insert(0,str(HERE))
import plot_bb_cost_to_precision_joint_ef_dense as P
import freeze_bb_irivm_r5b_cost_pack_compat_v2 as V
import adapt_bb_irivm_r5b_cost_pack_for_final_plot as A
import select_bb_irivm_cost_only_late_six_r3 as S

SCHEMA='bb-final-joint-ef-cost-10v8-contract-v1'; STATUS='FROZEN_R5B_IVM10_ODE8_OBSERVED_ONLY'
EARLY_IVM=ROOT/'registrations/diagnostics/BB_COST_PRECISION_PACK_IVM_EARLY_MEASURED_INTERIM_20260902_R1.json'
ODE_EARLY=ROOT/'registrations/diagnostics/BB_COST_PRECISION_PACK_ODE_EARLY_JOINT_EF_20260902_R1.json'
ODE_LATE=ROOT/'registrations/diagnostics/BB_COST_PRECISION_PACK_ODE_LATE_MEASURED_INTERIM_20260902_R1.json'
COMPAT=V.OUTPUT
CONTRACT=ROOT/'registrations/diagnostics/BB_COST_TO_PRECISION_FINAL_IVM10_ODE8_R5B_20260902_R1.json'
FIGURE=Path('/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/active_candidates/barnase_barstar_ivm_vs_ode_neb_cost_to_precision_densified.pdf')

def req(x,m):
 if not x: raise RuntimeError(m)
def bind(p): return P.file_binding(Path(p).resolve(strict=True))
def _scalar(z,k): return np.asarray(z[k]).reshape(()).item()
def validate_compat_v2(path=COMPAT):
 path=Path(path).resolve(strict=True);req(path==COMPAT.resolve(),'noncanonical compat-v2 path');x=V.F.read_json(path)
 req(x.get('schema')==V.SCHEMA and x.get('status')==V.STATUS and x.get('point_count')==6 and isinstance(x.get('points'),list) and len(x['points'])==6,'compat-v2 exact identity drift')
 expected={'primary':S.CANONICAL_PRIMARY,'supplemental':S.CANONICAL_SUPPLEMENTAL,'manifest':S.CANONICAL_MANIFEST,'canonical_index':S.CANONICAL_INDEX,'preparation':V.F.PREPARATION,'bundle':V.F.BUNDLE,'publication_gate':V.F.PUBLICATION_GATE}
 req(set(x.get('bindings',{}))==set(expected),'compat-v2 binding roles drift')
 for key,p in expected.items():req(x['bindings'][key]==bind(p),'compat-v2 canonical binding drift: '+key)
 return x

def close_early_to_r5b(rows,primary):
 inventory=primary['path_inventory']; out=[]
 for row in rows:
  call=int(row['discrete_I_perp_evidence']['calls_per_image']); nf=int(row['optimization_N_F'])
  hits=[x for x in inventory if int(x['call_per_image'])==call and int(x['optimization_N_F'])==nf]
  req(len(hits)==1,'early IVM absent/nonunique in R5B'); inv=hits[0]
  req(inv['formally_reportable'] and not inv['terminal_uncommitted_charged_tail'],'early IVM is not reportable')
  shard=Path(inv['shard']).resolve(strict=True); req(P.sha256_file(shard)==inv['shard_sha256'],'R5B early shard hash drift')
  with np.load(shard,allow_pickle=False) as z:
   path=np.asarray(z['path32_A']); force=np.asarray(z['force_eV_per_A']); val=V.C.iperp(path,force)
   req(path.shape==force.shape==(32,9396) and np.isfinite(path).all() and np.isfinite(force).all(),'early R5B path/force shape or finiteness drift')
   req(str(_scalar(z,'schema'))=='bb-irivm-full-state-history-recorder-v3','early R5B shard schema drift')
   req(int(_scalar(z,'call_per_image'))==call and int(_scalar(z,'optimization_N_F'))==nf,'early R5B shard call/N_F drift')
   req(V.C.sha_array(path)==row['selected_path_sha256']==inv['path_sha256'],'early path/R5B drift')
   req(V.C.sha_array(force)==inv['force_sha256'],'early force/R5B drift')
   req(str(_scalar(z,'path_sha256'))==inv['path_sha256'] and str(_scalar(z,'force_sha256'))==inv['force_sha256'],'early shard stored hash drift')
   req(np.asarray(val).tobytes()==np.asarray(row['discrete_I_perp']).tobytes()==np.asarray(_scalar(z,'discrete_I_perp')).tobytes(),'early Iperp/R5B drift')
  out.append({'call_per_image':call,'optimization_N_F':nf,'R5B_shard':bind(shard)})
 return out

def assemble():
 closure=S.validate_final_closure(S.CANONICAL_PREREGISTRATION,S.CANONICAL_SPLIT_TEMPLATE,S.CANONICAL_PRIMARY,S.CANONICAL_SUPPLEMENTAL,S.CANONICAL_MANIFEST,S.CANONICAL_INDEX,enforce_canonical=True);primary=closure[2]
 validate_compat_v2();ei,ir=P.validate_pack_spec(EARLY_IVM); oe,er=P.validate_pack_spec(ODE_EARLY); ol,lr=P.validate_pack_spec(ODE_LATE)
 req(ei['method']=='IVM' and len(ir)==4 and oe['method']==ol['method']=='ODE-NEB' and len(er)==len(lr)==4,'pack role/cardinality drift')
 late=A.load(COMPAT)['points']; req(len(late)==6,'late IVM cardinality drift')
 early_proof=close_early_to_r5b(ir,primary)
 ivm=ir+late; ode=er+lr
 for method,rows,n in [('IVM',ivm,10),('ODE-NEB',ode,8)]:
  nf=[int(x['optimization_N_F']) for x in rows]; req(len(rows)==n and len(set(nf))==n and nf==sorted(nf),method+' N_F axis drift')
 analysis=P.analyze_observed({'IVM':ivm,'ODE-NEB':ode})[1]
 return ir,late,er,lr,early_proof,analysis
def build(output=CONTRACT,figure=FIGURE):
 output=Path(output).resolve(); figure=Path(figure).resolve();req(output==CONTRACT.resolve() and figure==FIGURE.resolve(),'noncanonical final path');req(not output.exists() and not figure.exists() and not figure.with_suffix('.json').exists(),'final artifact already exists')
 ir,late,er,lr,early_proof,analysis=assemble()
 payload={'schema':SCHEMA,'status':STATUS,'figure_output':str(figure),'implementation':{'freezer':bind(Path(__file__)),'plotter':bind(HERE/'plot_bb_final_cost_10v8.py'),'shared_renderer':bind(Path(P.__file__)),'compat_builder':bind(Path(V.__file__)),'compat_adapter':bind(Path(A.__file__))},'inputs':{'IVM_early4':bind(EARLY_IVM),'IVM_late6_compat':bind(COMPAT),'ODE_early4':bind(ODE_EARLY),'ODE_late4':bind(ODE_LATE),'R5B_primary':bind(S.CANONICAL_PRIMARY),'R5B_supplemental':bind(S.CANONICAL_SUPPLEMENTAL)},'early4_R5B_closure':early_proof,'point_counts':{'IVM':10,'ODE-NEB':8},'point_policy':'raw_observed_points_and_observed_lower_envelope_only_no_metric_interpolation','analysis':analysis,'OpenMM_calls':0,'optimization_N_F_contribution':0}
 output.parent.mkdir(parents=True,exist_ok=True)
 fd,name=tempfile.mkstemp(prefix=output.name+'.',suffix='.tmp',dir=output.parent)
 try:
  with os.fdopen(fd,'w') as stream: stream.write(json.dumps(payload,indent=2,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
  os.link(name,output)
  descriptor=os.open(output.parent,os.O_RDONLY)
  try:os.fsync(descriptor)
  finally:os.close(descriptor)
 finally: Path(name).unlink(missing_ok=True)
 return payload
def main():
 q=argparse.ArgumentParser();q.add_argument('--output',type=Path,default=CONTRACT);q.add_argument('--figure-output',type=Path,default=FIGURE);a=q.parse_args();print(json.dumps(build(a.output,a.figure),indent=2))
if __name__=='__main__':main()
