#!/usr/bin/env python3
"""Render the frozen final BB IVM10/ODE8 contract without metric interpolation."""
from __future__ import annotations
import argparse,json,sys,os,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import plot_bb_cost_to_precision_joint_ef_dense as P
import freeze_bb_final_cost_10v8_contract as F
import adapt_bb_irivm_r5b_cost_pack_for_final_plot as A
def main():
 q=argparse.ArgumentParser();q.add_argument('--contract',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--validate-only',action='store_true');a=q.parse_args()
 c=json.loads(a.contract.resolve(strict=True).read_text());F.req(c.get('schema')==F.SCHEMA and c.get('status')==F.STATUS,'final contract identity drift');F.req(a.output.resolve()==Path(c['figure_output']).resolve()==F.FIGURE.resolve(),'final output drift');F.req(c.get('point_counts')=={'IVM':10,'ODE-NEB':8} and c.get('point_policy')=='raw_observed_points_and_observed_lower_envelope_only_no_metric_interpolation' and c.get('OpenMM_calls')==0 and c.get('optimization_N_F_contribution')==0,'final policy drift')
 for key,path in {'freezer':Path(F.__file__),'plotter':Path(__file__),'shared_renderer':Path(P.__file__),'compat_builder':Path(F.V.__file__),'compat_adapter':Path(A.__file__)}.items(): F.req(c['implementation'][key]==P.file_binding(path.resolve(strict=True)),'implementation binding drift: '+key)
 expected={'IVM_early4':F.EARLY_IVM,'IVM_late6_compat':F.COMPAT,'ODE_early4':F.ODE_EARLY,'ODE_late4':F.ODE_LATE,'R5B_primary':F.S.CANONICAL_PRIMARY,'R5B_supplemental':F.S.CANONICAL_SUPPLEMENTAL};F.req(set(c.get('inputs',{}))==set(expected),'final input roles drift')
 for key,path in expected.items():F.req(c['inputs'][key]==P.file_binding(Path(path).resolve(strict=True)),'canonical input binding drift: '+key)
 ir,late,oe,ol,proof,expected=F.assemble();F.req(proof==c['early4_R5B_closure'],'early4 R5B contract closure drift');points={'IVM':ir+late,'ODE-NEB':oe+ol}
 arrays,analysis=P.analyze_observed(points);F.req(analysis==c['analysis']==expected,'analysis drift')
 if not a.validate_only:
  final=a.output.resolve();F.req(not final.exists(),'final PDF exists');side=final.with_suffix('.json');F.req(not side.exists(),'final sidecar exists')
  payload={'schema':'bb-final-joint-ef-cost-10v8-provenance-v1','status':'PASS_AUDITED_JOINT_EF_OBSERVED_ONLY','figure':{'path':str(final),'sha256':'PENDING_ATOMIC_RENDER'},'contract':P.file_binding(a.contract.resolve()),'point_counts':c['point_counts'],'point_policy':c['point_policy'],'raw_points':points,'analysis':analysis,'OpenMM_calls':0,'optimization_N_F_contribution':0}
  with tempfile.TemporaryDirectory(prefix=final.name+'.',dir=final.parent) as td:
   tpdf=Path(td)/final.name; tside=Path(td)/side.name;P.render(arrays,analysis,tpdf);payload['figure']={'path':str(final),'sha256':P.sha256_file(tpdf)}
   with tside.open('x') as stream:stream.write(json.dumps(payload,indent=2,sort_keys=True)+'\n');stream.flush();os.fsync(stream.fileno())
   pdf_linked=False
   try:os.link(tpdf,final);pdf_linked=True;os.link(tside,side)
   except BaseException:
    if pdf_linked:final.unlink(missing_ok=True)
    raise
   fd=os.open(final.parent,os.O_RDONLY)
   try:os.fsync(fd)
   finally:os.close(fd)
 print(json.dumps({'status':'PASS' if a.validate_only else 'RENDERED','point_counts':c['point_counts']}))
if __name__=='__main__':main()
