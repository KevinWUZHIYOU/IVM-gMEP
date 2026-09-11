#!/usr/bin/env python3
"""Freeze final action-cost inputs only after both R5B verifiers pass."""
from __future__ import annotations
import argparse
from pathlib import Path
import build_bb_discrete32_action_cost_pack as B
import build_bb_ode8_discrete32_action_pack as O
ROOT=B.ROOT; PLOT=ROOT/'scripts/plot_bb_discrete32_action_cost.py'; TEST=ROOT/'tests/test_bb_discrete32_action_cost.py'
FIG=Path('/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/active_candidates/barnase_barstar_ivm_vs_ode_neb_discrete32_geometric_action_cost_to_precision.pdf')
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--primary-verifier',type=Path,required=True); p.add_argument('--supplemental-verifier',type=Path,required=True); a=p.parse_args(argv)
 pri=a.primary_verifier.resolve(strict=True); sup=a.supplemental_verifier.resolve(strict=True)
 q,s,inventory,last,_manifest,_index,_audit=B.validate_r5b_verifiers(pri,sup,require_canonical=True)
 B.validate_ode_pack_and_audit(B.ODE,B.ODE_AUDIT,require_canonical=True)
 O.require(not B.CONTRACT.exists() and not B.OUTPUT.exists() and not B.AUDIT.exists(),'prospective final output exists')
 highest=int(inventory[last-1]['optimization_N_F'])
 payload={'schema':'bb-discrete32-action-cost-pack-contract-v1','status':'FROZEN_AFTER_R5B_PRIMARY_AND_SUPPLEMENTAL_PASS','metric':'original32 31-segment trapezoid path-dependent geometric action','metric_interpolation':'none','raw_markers':True,'observed_lower_envelope':True,'x_axis':{'label':r'Path-dependent geometric action, $\hat S$ (eV)','quantity':'original32_path_dependent_geometric_action','scale':'log','direction':'larger_to_smaller_left_to_right','inverted':True},'y_axis':{'label':'Number of Forces','quantity':'optimization_N_F','scale':'linear','formatter':'thousands_k'},'curve_semantics':'observed lower-envelope cost-to-precision frontier; no metric interpolation','methods':['IVM','ODE-NEB'],'R5B_budget':B.R5B_BUDGET,'R5B_dynamic_canonical_call_count':len(inventory),'R5B_dynamic_last_complete_call':last,'R5B_dynamic_highest_reportable_optimization_N_F':highest,'R5B_history_policy':'all and only dynamic formally-reportable stable-last calls 1..last_complete; old NF9990 artifacts are prefix evidence only','physical_calls':0,'optimization_N_F_contribution':0}
 for k,x in [('builder',Path(B.__file__)),('freezer',Path(__file__)),('plotter',PLOT),('test',TEST),('ODE_pack',B.ODE),('ODE_audit',B.ODE_AUDIT),('R5B_primary_verifier',pri),('R5B_supplemental_verifier',sup)]: payload[k]=O.binding(x) if hasattr(O,'binding') else {'path':str(x.resolve(strict=True)),'sha256':O.sha_file(x)}
 payload.update({'output':str(B.OUTPUT),'audit':str(B.AUDIT),'figure_output':str(FIG),'figure_provenance':str(FIG.with_name(FIG.stem+'_provenance.json'))}); O.write_json(B.CONTRACT,payload); print(B.CONTRACT); print(O.sha_file(B.CONTRACT))
if __name__=='__main__': main()
