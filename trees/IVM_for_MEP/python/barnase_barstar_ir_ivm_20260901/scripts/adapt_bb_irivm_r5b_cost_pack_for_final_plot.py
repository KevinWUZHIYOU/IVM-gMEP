#!/usr/bin/env python3
"""Strict read-only adapter from the R5B compatibility pack to plot points."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent)); import freeze_bb_irivm_r5b_cost_pack_compat as C
import freeze_bb_irivm_r5b_cost_pack_compat_v2 as V2
SCHEMA='bb-final-cost-ivm-r5b-adapter-v1'; STATUS='PASS_R5B_COMPAT_POINTS_FOR_FINAL_PLOT'
def load(path):
 path=path.resolve(strict=True);x=json.loads(path.read_text());C.req((x.get('schema'),x.get('status')) in ((C.PACK_SCHEMA,C.PACK_STATUS),(V2.SCHEMA,V2.STATUS)),'compat pack not PASS');C.req(x.get('OpenMM_calls')==0 and x.get('optimization_N_F_contribution')==0,'compat accounting drift')
 points=x.get('points'); expected=x.get('point_count',6 if x.get('schema')==V2.SCHEMA else None);C.req(isinstance(points,list) and len(points)==expected,'compat point count drift');last=-1;out=[]
 for p in points:
  nf=int(p['optimization_N_F']);C.req(nf>last,'compat NF not strictly ordered');last=nf
  keys=('arc500_shard','R5B_shard') if x.get('schema')==V2.SCHEMA else ('late_shard','R5B_full_state_shard')
  for k in keys:
   b=p[k];q=Path(b['path']).resolve(strict=True);C.req(C.sha_file(q)==b['sha256'],f'{k} binding drift')
  C.req(p.get('optimization_N_F_contribution')==0,'point diagnostic NoF nonzero');out.append({'method':'IVM','optimization_N_F':nf,'discrete_I_perp':float(p['discrete_I_perp']),'densified_Delta_E_kcal_per_mol':float(p['densified_Delta_E_kcal_per_mol']),'path_sha256':p['path_sha256'],'force32_sha256':p['force32_sha256']})
 return {'schema':SCHEMA,'status':STATUS,'source_pack':C.bind(path),'points':out,'OpenMM_calls':0,'optimization_N_F_contribution':0}
def main():
 p=argparse.ArgumentParser();p.add_argument('--pack',type=Path,required=True);a=p.parse_args();print(json.dumps(load(a.pack),indent=2,sort_keys=True))
if __name__=='__main__':main()
