import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'scripts/run_bb_dense_ivm_short_four_joint_resume_r4b.py'; s=importlib.util.spec_from_file_location('r4b',P); R=s.loader.load_module()
class R4B(unittest.TestCase):
 def test_formal_closure(self):
  c=R.load_contract(R.CONTRACT); self.assertEqual(R.validate_authorization(c,R.AUTH)['optimization_N_F_contribution'],0); self.assertNotIn('openmm',sys.modules)
 def test_each_authorization_field_tamper_rejected_after_canonical_path_gate(self):
  c=R.load_contract(R.CONTRACT); source=json.loads(R.AUTH.read_text())
  for key in ('authorizer','zero_openmm_preflight','optimization_N_F_contribution','OpenMM_calls_before_authorization','thread_policy','canonical_lock_paths','result_output','maximum_joint_EF_attempts'):
   with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
    q=Path(d)/'auth.json'; x=dict(source); x[key]=None; q.write_text(json.dumps(x))
    with mock.patch.object(R,'AUTH',q):
     with self.assertRaisesRegex(Exception,f'authorization field drift: {key}'): R.validate_authorization(c,q)
 def test_each_contract_source_binding_tamper_rejected(self):
  source=json.loads(R.CONTRACT.read_text())
  for key in ('authorized_entrypoint','contract_freezer_R4B','execution_authorizer','test'):
   with self.subTest(key=key),tempfile.TemporaryDirectory() as d:
    q=Path(d)/'contract.json'; x=dict(source); x[key]=None; x['execution_authorization_output']=str(R.AUTH); x['result_output']=str(R.RESULT); q.write_text(json.dumps(x))
    with mock.patch.object(R,'CONTRACT',q):
     with self.assertRaisesRegex(Exception,f'{key} binding drift'): R.load_contract(q)
 def test_noncanonical_paths_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   q=Path(d)/'x'; q.write_text('{}')
   with self.assertRaisesRegex(Exception,'noncanonical R4B contract'): R.load_contract(q)
if __name__=='__main__': unittest.main()
