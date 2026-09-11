#!/usr/bin/env python3
"""Plot raw original-32 action observations and observed lower envelopes."""
from pathlib import Path
import json,os,tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import build_bb_discrete32_action_cost_pack as B
import build_bb_ode8_discrete32_action_pack as O
FIG=Path('/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/active_candidates/barnase_barstar_ivm_vs_ode_neb_discrete32_geometric_action_cost_to_precision.pdf')
PROV=FIG.with_name(FIG.stem+'_provenance.json')
def publish_pair(render_pdf,provenance):
 O.require(not FIG.exists() and not PROV.exists(),'refusing overwrite');FIG.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix=FIG.name+'.',dir=FIG.parent) as td:
  pdf=Path(td)/FIG.name;prov=Path(td)/PROV.name;render_pdf(pdf)
  payload=dict(provenance);payload['figure']={'path':str(FIG),'sha256':O.sha_file(pdf)}
  with prov.open('x') as stream:json.dump(payload,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
  linked=False
  try:os.link(pdf,FIG);linked=True;os.link(prov,PROV)
  except BaseException:
   if linked:FIG.unlink(missing_ok=True)
   raise
  descriptor=os.open(FIG.parent,os.O_RDONLY)
  try:os.fsync(descriptor)
  finally:os.close(descriptor)
def load_verified_pack():
 c=json.loads(B.CONTRACT.read_text())
 O.require(c.get('schema')=='bb-discrete32-action-cost-pack-contract-v1' and c.get('status')=='FROZEN_AFTER_R5B_PRIMARY_AND_SUPPLEMENTAL_PASS','action-cost contract is not final')
 expected={'builder':Path(B.__file__).resolve(),'freezer':B.ROOT/'scripts/freeze_bb_discrete32_action_cost_pack.py','plotter':Path(__file__).resolve(),'test':B.ROOT/'tests/test_bb_discrete32_action_cost.py','ODE_pack':B.ODE,'ODE_audit':B.ODE_AUDIT,'R5B_primary_verifier':B.PRIMARY,'R5B_supplemental_verifier':B.SUPPLEMENTAL}
 for role in ('builder','freezer','plotter','test','ODE_pack','ODE_audit','R5B_primary_verifier','R5B_supplemental_verifier'):
  record=c.get(role); O.require(isinstance(record,dict) and Path(record.get('path','')).resolve(strict=True)==expected[role].resolve() and O.sha_file(record['path'])==record.get('sha256'),'action-cost contract binding drift '+role)
 O.require(Path(c.get('output','')).resolve()==B.OUTPUT.resolve() and Path(c.get('audit','')).resolve()==B.AUDIT.resolve(),'action-cost contract output drift')
 O.require(Path(c.get('figure_output','')).resolve()==FIG.resolve() and Path(c.get('figure_provenance','')).resolve()==PROV.resolve(),'action-cost contract figure path drift')
 O.require(c.get('x_axis')=={'label':r'Path-dependent geometric action, $\hat S$ (eV)','quantity':'original32_path_dependent_geometric_action','scale':'log','direction':'larger_to_smaller_left_to_right','inverted':True},'action-cost x-axis semantics drift')
 O.require(c.get('y_axis')=={'label':'Number of Forces','quantity':'optimization_N_F','scale':'linear','formatter':'thousands_k'},'action-cost y-axis semantics drift')
 O.require(c.get('curve_semantics')=='observed lower-envelope cost-to-precision frontier; no metric interpolation','action-cost curve semantics drift')
 primary,_supplemental,inventory,last,*_=B.validate_r5b_verifiers(c['R5B_primary_verifier']['path'],c['R5B_supplemental_verifier']['path'],require_canonical=True)
 ode_nf,ode_action,_ode_audit=B.validate_ode_pack_and_audit(c['ODE_pack']['path'],c['ODE_audit']['path'],require_canonical=True)
 highest=int(inventory[last-1]['optimization_N_F'])
 O.require(int(c.get('R5B_budget',-1))==B.R5B_BUDGET and int(c.get('R5B_dynamic_canonical_call_count',-1))==len(inventory) and int(c.get('R5B_dynamic_last_complete_call',-1))==last and int(c.get('R5B_dynamic_highest_reportable_optimization_N_F',-1))==highest,'action-cost contract dynamic R5B summary drift')
 audit=json.loads(B.AUDIT.read_text())
 O.require(audit.get('schema')=='bb-discrete32-action-cost-pack-audit-v1' and audit.get('status')=='PASS','action-cost audit is not PASS')
 O.require(audit.get('result')==B.binding(B.OUTPUT) and audit.get('contract')==B.binding(B.CONTRACT),'action-cost audit publication binding drift')
 O.require(audit.get('R5B_primary_verifier')==c['R5B_primary_verifier'] and audit.get('R5B_supplemental_verifier')==c['R5B_supplemental_verifier'],'action-cost audit R5B binding drift')
 O.require(int(audit.get('R5B_dynamic_canonical_call_count',-1))==len(inventory) and int(audit.get('R5B_last_complete_call',-1))==last and int(audit.get('R5B_dynamic_highest_reportable_optimization_N_F',-1))==highest and audit.get('R5B_terminal_tail_excluded') is True,'action-cost audit dynamic R5B summary drift')
 with np.load(B.OUTPUT,allow_pickle=False) as saved:
  required={'schema','status','ivm_optimization_N_F','ivm_total_action_eV','ivm_observed_lower_envelope_eV','ode_optimization_N_F','ode_total_action_eV','ode_observed_lower_envelope_eV','R5B_budget','R5B_dynamic_last_complete_call','R5B_dynamic_highest_reportable_optimization_N_F','metric_images','metric_segments','metric_interpolation','optimization_N_F_contribution'}
  O.require(required.issubset(saved.files),'action-cost result fields missing')
  O.require(str(O.scalar(saved,'schema'))=='bb-discrete32-action-cost-pack-v1' and str(O.scalar(saved,'status'))=='PASS','action-cost result is not PASS')
  inf=np.asarray(saved['ivm_optimization_N_F'],np.int64); is_=np.asarray(saved['ivm_total_action_eV'],float); ienv=np.asarray(saved['ivm_observed_lower_envelope_eV'],float)
  onf=np.asarray(saved['ode_optimization_N_F'],np.int64); os_=np.asarray(saved['ode_total_action_eV'],float); oenv=np.asarray(saved['ode_observed_lower_envelope_eV'],float)
  O.require(inf.shape==is_.shape==ienv.shape==(last,) and onf.shape==os_.shape==oenv.shape and onf.size>0,'action-cost result vector shape drift')
  O.require(np.array_equal(onf,ode_nf) and np.array_equal(os_,ode_action),'final action-cost ODE arrays differ from the sealed ODE pack')
  O.require(np.all(inf[1:]>=inf[:-1]) and int(inf[-1])==highest and int(inf[-1])>B.OLD_BUDGET,'action-cost IVM history is not the dynamic NF15000 history')
  O.require(np.array_equal(ienv,B.lower_envelope(is_)) and np.array_equal(oenv,B.lower_envelope(os_)),'action-cost observed envelope drift')
  O.require(int(O.scalar(saved,'R5B_budget'))==B.R5B_BUDGET and int(O.scalar(saved,'R5B_dynamic_last_complete_call'))==last and int(O.scalar(saved,'R5B_dynamic_highest_reportable_optimization_N_F'))==highest,'action-cost result dynamic R5B summary drift')
  O.require(int(O.scalar(saved,'metric_images'))==32 and int(O.scalar(saved,'metric_segments'))==31 and str(O.scalar(saved,'metric_interpolation'))=='none' and int(O.scalar(saved,'optimization_N_F_contribution'))==0,'action-cost result metric/NoF policy drift')
 return {'IVM':(inf,is_,ienv),'ODE-NEB':(onf,os_,oenv)},c,audit,primary
def main():
 O.require(not FIG.exists() and not PROV.exists(),'refusing overwrite'); data,contract,audit,primary=load_verified_pack()
 fig,ax=plt.subplots(figsize=(5.1,3.35)); colors={'IVM':'#0072B2','ODE-NEB':'#D55E00'}
 for name in ('IVM','ODE-NEB'):
  nf,action,env=data[name]
  ax.scatter(action,nf,s=15,alpha=.48,color=colors[name],label='_nolegend_',zorder=3); ax.step(env,nf,where='pre',lw=2,color=colors[name],label=name)
 ax.set_xscale('log'); ax.invert_xaxis(); ax.set_xlabel(r'Path-dependent geometric action, $\hat S$ (eV)'); ax.set_ylabel('Number of Forces'); ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_position: f'{value/1000:g}k' if abs(value)>=1000 else f'{value:g}')); ax.grid(alpha=.2); ax.legend(frameon=False,fontsize=8); fig.tight_layout()
 def render(path):fig.savefig(path,format='pdf');plt.close(fig)
 publish_pair(render,{'schema':'bb-discrete32-action-cost-figure-provenance-v1','status':'PASS','result':B.binding(B.OUTPUT),'audit':B.binding(B.AUDIT),'contract':B.binding(B.CONTRACT),'plotter':B.binding(Path(__file__)),'R5B_primary_verifier':contract['R5B_primary_verifier'],'R5B_supplemental_verifier':contract['R5B_supplemental_verifier'],'R5B_budget':B.R5B_BUDGET,'R5B_dynamic_canonical_call_count':int(primary['actual_canonical_call_count']),'R5B_dynamic_last_complete_call':int(primary['last_complete_call']),'R5B_dynamic_highest_reportable_optimization_N_F':int(data['IVM'][0][-1]),'old_NF9990_artifacts_role':'exact-prefix evidence only; never a final input','raw_markers_retained':True,'observed_lower_envelope':True,'metric_interpolation':'none','x_axis':contract['x_axis'],'y_axis':contract['y_axis'],'curve_semantics':contract['curve_semantics'],'arc500_used_by_metric':False,'physical_calls':0,'optimization_N_F_contribution':0})
 print(FIG); print(O.sha_file(FIG)); print(PROV); print(O.sha_file(PROV))
if __name__=='__main__': main()
