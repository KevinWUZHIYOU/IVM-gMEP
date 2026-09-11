import unittest,tempfile,json,sys
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import freeze_bb_final_cost_10v8_contract as F
import plot_bb_final_cost_10v8 as Q
class T(unittest.TestCase):
 def test_compat_requires_exact_v2_and_canonical_primary_binding(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);names=('primary','supplemental','manifest','index','prep','bundle','gate');files={n:d/n for n in names}
   for n,p in files.items():p.write_text(n)
   expected={'primary':files['primary'],'supplemental':files['supplemental'],'manifest':files['manifest'],'canonical_index':files['index'],'preparation':files['prep'],'bundle':files['bundle'],'publication_gate':files['gate']}
   compat=d/'compat.json';payload={'schema':F.V.SCHEMA,'status':F.V.STATUS,'point_count':6,'points':[{}]*6,'bindings':{k:F.bind(p) for k,p in expected.items()}};compat.write_text(json.dumps(payload))
   patches=[mock.patch.object(F,'COMPAT',compat),mock.patch.object(F.S,'CANONICAL_PRIMARY',files['primary']),mock.patch.object(F.S,'CANONICAL_SUPPLEMENTAL',files['supplemental']),mock.patch.object(F.S,'CANONICAL_MANIFEST',files['manifest']),mock.patch.object(F.S,'CANONICAL_INDEX',files['index']),mock.patch.object(F.V.F,'PREPARATION',files['prep']),mock.patch.object(F.V.F,'BUNDLE',files['bundle']),mock.patch.object(F.V.F,'PUBLICATION_GATE',files['gate'])]
   for p in patches:p.start()
   try:
    self.assertEqual(F.validate_compat_v2(compat)['point_count'],6)
    payload['schema']='bb-irivm-r5b-cost-pack-compat-v1';compat.write_text(json.dumps(payload))
    with self.assertRaisesRegex(RuntimeError,'identity'):F.validate_compat_v2(compat)
    payload['schema']=F.V.SCHEMA;payload['bindings']['primary']=F.bind(files['supplemental']);compat.write_text(json.dumps(payload))
    with self.assertRaisesRegex(RuntimeError,'primary'):F.validate_compat_v2(compat)
   finally:
    for p in reversed(patches):p.stop()
 def test_assemble_routes_primary_from_nine_item_closure(self):
  primary={'sentinel':'primary'};closure=('a','b',primary,'d','e','f','g','h','i')
  ivm=[{'optimization_N_F':n,'discrete_I_perp':1.,'densified_Delta_E_kcal_per_mol':1.} for n in (1,2,3,4)]
  late=[{'optimization_N_F':n,'discrete_I_perp':1.,'densified_Delta_E_kcal_per_mol':1.} for n in (5,6,7,8,9,10)]
  ode1=[{'optimization_N_F':n,'discrete_I_perp':2.,'densified_Delta_E_kcal_per_mol':2.} for n in (1,2,3,4)];ode2=[dict(x,optimization_N_F=x['optimization_N_F']+4) for x in ode1]
  with mock.patch.object(F.S,'validate_final_closure',return_value=closure),mock.patch.object(F,'validate_compat_v2',return_value={}),mock.patch.object(F.P,'validate_pack_spec',side_effect=[({'method':'IVM'},ivm),({'method':'ODE-NEB'},ode1),({'method':'ODE-NEB'},ode2)]),mock.patch.object(F.A,'load',return_value={'points':late}),mock.patch.object(F,'close_early_to_r5b',return_value=['proof']*4) as close:
   out=F.assemble();self.assertEqual(len(out[0])+len(out[1]),10);close.assert_called_once_with(ivm,primary)
 def test_cardinality_and_early_r5b_gate(self):
  rows=[{'discrete_I_perp_evidence':{'calls_per_image':2},'optimization_N_F':60,'selected_path_sha256':'p','discrete_I_perp':1.}]
  primary={'path_inventory':[{'call_per_image':2,'optimization_N_F':60,'formally_reportable':True,'terminal_uncommitted_charged_tail':False,'shard':'x','shard_sha256':'h','path_sha256':'p','force_sha256':'f'}]}
  with mock.patch.object(F.Path,'resolve',lambda self,strict=False:self),mock.patch.object(F.P,'sha256_file',return_value='h'),mock.patch.object(F.np,'load') as load,mock.patch.object(F.V.C,'sha_array',side_effect=lambda a:'p' if a.shape[0]==32 and not a.any() else 'f'),mock.patch.object(F.V.C,'iperp',return_value=1.),mock.patch.object(F,'bind',return_value={'path':'x','sha256':'h'}):
   z={'schema':F.np.asarray('bb-irivm-full-state-history-recorder-v3'),'call_per_image':F.np.asarray(2),'optimization_N_F':F.np.asarray(60),'path32_A':F.np.zeros((32,9396)),'force_eV_per_A':F.np.ones((32,9396)),'path_sha256':F.np.asarray('p'),'force_sha256':F.np.asarray('f'),'discrete_I_perp':F.np.asarray(1.)};load.return_value.__enter__.return_value=z
   def enter():
    return z
   # Match the requested inventory row on every context entry.
   load.return_value.__enter__.side_effect=lambda: z
   self.assertEqual(len(F.close_early_to_r5b(rows,primary)),1)
   primary['path_inventory'][0]['path_sha256']='bad'
   with self.assertRaisesRegex(RuntimeError,'path'):F.close_early_to_r5b(rows,primary)
 def test_formal_outputs_absent(self):
  self.assertFalse(F.CONTRACT.exists());self.assertFalse(F.FIGURE.exists())
 def test_build_then_validate_only_and_contract_tamper(self):
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);contract=d/'contract.json';figure=d/'figure.pdf'
   def rows(method,n,offset=0):
    return [{'method':method,'optimization_N_F':offset+i+1,'discrete_I_perp':float(n-i),'densified_Delta_E_kcal_per_mol':float(2*n-i)} for i in range(n)]
   ir,late,oe,ol=rows('IVM',4),rows('IVM',6,10),rows('ODE-NEB',4),rows('ODE-NEB',4,10)
   analysis=F.P.analyze_observed({'IVM':ir+late,'ODE-NEB':oe+ol})[1];assembled=(ir,late,oe,ol,[{'call_per_image':2}],analysis)
   binding={'path':str(Path(__file__).resolve()),'sha256':'a'*64}
   ep=Path(__file__).resolve();paths=[mock.patch.object(F,k,ep) for k in ('EARLY_IVM','COMPAT','ODE_EARLY','ODE_LATE')]+[mock.patch.object(F.S,k,ep) for k in ('CANONICAL_PRIMARY','CANONICAL_SUPPLEMENTAL')]
   for p in paths:p.start()
   self.addCleanup(lambda:[p.stop() for p in reversed(paths)])
   with mock.patch.object(F,'CONTRACT',contract),mock.patch.object(F,'FIGURE',figure),mock.patch.object(F,'assemble',return_value=assembled),mock.patch.object(F,'bind',return_value=binding):F.build(contract,figure)
   with mock.patch.object(F,'CONTRACT',contract),mock.patch.object(F,'FIGURE',figure),mock.patch.object(F,'assemble',return_value=assembled),mock.patch.object(Q.P,'file_binding',return_value=binding),mock.patch.object(sys,'argv',['p','--contract',str(contract),'--output',str(figure),'--validate-only']):Q.main()
   def render(_a,_b,path):Path(path).write_bytes(b'%PDF-fixture')
   with mock.patch.object(F,'CONTRACT',contract),mock.patch.object(F,'FIGURE',figure),mock.patch.object(F,'assemble',return_value=assembled),mock.patch.object(Q.P,'file_binding',return_value=binding),mock.patch.object(Q.P,'render',side_effect=render),mock.patch.object(Q.P,'sha256_file',return_value='b'*64),mock.patch.object(sys,'argv',['p','--contract',str(contract),'--output',str(figure)]):Q.main()
   self.assertTrue(figure.exists() and figure.with_suffix('.json').exists())
   figure.unlink();figure.with_suffix('.json').unlink()
   real_link=Q.os.link;calls=[]
   def broken(src,dst):
    calls.append(dst)
    if len(calls)==2:raise OSError('fixture link failure')
    return real_link(src,dst)
   with mock.patch.object(F,'CONTRACT',contract),mock.patch.object(F,'FIGURE',figure),mock.patch.object(F,'assemble',return_value=assembled),mock.patch.object(Q.P,'file_binding',return_value=binding),mock.patch.object(Q.P,'render',side_effect=render),mock.patch.object(Q.P,'sha256_file',return_value='b'*64),mock.patch.object(Q.os,'link',side_effect=broken),mock.patch.object(sys,'argv',['p','--contract',str(contract),'--output',str(figure)]):
    with self.assertRaisesRegex(OSError,'link failure'):Q.main()
   self.assertFalse(figure.exists());self.assertFalse(figure.with_suffix('.json').exists())
   for occupied,message in ((figure,'PDF'),(figure.with_suffix('.json'),'sidecar')):
    occupied.write_bytes(b'x')
    with mock.patch.object(F,'CONTRACT',contract),mock.patch.object(F,'FIGURE',figure),mock.patch.object(F,'assemble',return_value=assembled),mock.patch.object(Q.P,'file_binding',return_value=binding),mock.patch.object(sys,'argv',['p','--contract',str(contract),'--output',str(figure)]):
     with self.assertRaisesRegex(RuntimeError,message):Q.main()
    other=figure.with_suffix('.json') if occupied==figure else figure
    self.assertFalse(other.exists());occupied.unlink()
   x=json.loads(contract.read_text());x['point_counts']['IVM']=9;contract.write_text(json.dumps(x))
   with mock.patch.object(F,'CONTRACT',contract),mock.patch.object(F,'FIGURE',figure),mock.patch.object(Q.P,'file_binding',return_value=binding),mock.patch.object(sys,'argv',['p','--contract',str(contract),'--output',str(figure),'--validate-only']):
    with self.assertRaisesRegex(RuntimeError,'policy'):Q.main()
if __name__=='__main__':unittest.main()
