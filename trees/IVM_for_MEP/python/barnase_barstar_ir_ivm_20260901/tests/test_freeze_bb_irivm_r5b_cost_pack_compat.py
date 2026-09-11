import importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; spec=importlib.util.spec_from_file_location('C',ROOT/'scripts/freeze_bb_irivm_r5b_cost_pack_compat.py'); C=spec.loader.load_module()
spec2=importlib.util.spec_from_file_location('A',ROOT/'scripts/adapt_bb_irivm_r5b_cost_pack_for_final_plot.py'); A=spec2.loader.load_module()
class Compat(unittest.TestCase):
 def test_weighted_iperp_nonuniform_manual_formula(self):
  path=np.zeros((32,9396)); path[:,0]=np.cumsum(np.r_[0.,np.linspace(.2,1.7,31)])
  force=np.zeros_like(path); force[1:-1,1]=np.linspace(.3,2.1,30)
  seg=np.diff(path[:,0]); weights=.5*(seg[:-1]+seg[1:]); expected=float(np.sum(weights*force[1:-1,1]**2))
  self.assertEqual(C.iperp(path,force),expected)
  wrong=float(np.linalg.norm(force[1:-1])/np.sqrt(force[1:-1].size)); self.assertNotEqual(expected,wrong)
 def fixture(self,d):
  path=np.zeros((32,9396));path[:,0]=np.arange(32);force32=np.zeros_like(path);energy32=np.arange(32.)
  r=d/'r5.npz';np.savez(r,schema=np.asarray(C.R5_SHARD_SCHEMA),path32_A=path,energy_eV=energy32,force_eV_per_A=force32,discrete_I_perp=np.asarray(0.))
  dense=C.uniform500(path);e=np.arange(500.);f=np.zeros_like(dense);late=d/'late.npz';np.savez(late,schema=np.asarray(C.LATE_SHARD_SCHEMA),status=np.asarray('PASS'),optimization_N_F=np.asarray(300),selected_path_sha256=np.asarray(C.sha_array(path)),discrete_I_perp=np.asarray(0.),dense_coordinates_A=dense,dense_energy_eV=e,dense_force_eV_per_A=f,dense_coordinate_sha256=np.asarray(C.sha_array(dense)),dense_energy_sha256=np.asarray(C.sha_array(e)),dense_force_sha256=np.asarray(C.sha_array(f)),densified_Delta_E_kcal_per_mol=np.asarray(1.),optimization_N_F_contribution=np.asarray(0))
  primary=d/'primary.json';p={'schema':C.PRIMARY_SCHEMA,'status':C.PRIMARY_STATUS,'actual_canonical_call_count':1,'last_complete_call':10,'bindings':{'registration':C.bind(C.REG),'authorization':C.bind(C.AUTH)},'path_inventory':[{'call_per_image':10,'optimization_N_F':300,'path_sha256':C.sha_array(path),'force_sha256':C.sha_array(force32),'shard':str(r),'shard_sha256':C.sha_file(r),'formally_reportable':True}]};primary.write_text(json.dumps(p))
  supp=d/'supp.json';supp.write_text(json.dumps({'schema':C.SUPP_SCHEMA,'status':C.SUPP_STATUS,'bindings':{'primary_verification':C.bind(primary)}}))
  man=d/'manifest.json';man.write_text(json.dumps({'schema':C.LATE_MANIFEST_SCHEMA,'status':'PASS','shards':[{'path':str(late),'sha256':C.sha_file(late)}]}));return primary,supp,man,r
 def test_pass_and_no_openmm(self):
  import sys
  with tempfile.TemporaryDirectory() as x:
   d=Path(x);a,b,c,_=self.fixture(d);out=d/'out.json';q=C.build(a,b,c,out);self.assertEqual(q['status'],C.PACK_STATUS);self.assertEqual(A.load(out)['status'],A.STATUS);self.assertNotIn('openmm',sys.modules)
 def test_r5_force_tamper_rejected(self):
  with tempfile.TemporaryDirectory() as x:
   d=Path(x);a,b,c,r=self.fixture(d)
   with np.load(r,allow_pickle=False) as z:p={k:np.asarray(z[k]) for k in z.files}
   p['force_eV_per_A']=np.ones((32,9396));np.savez(r,**p)
   with self.assertRaisesRegex(RuntimeError,'R5B shard binding drift'):C.build(a,b,c,d/'out.json')
if __name__=='__main__':unittest.main()
