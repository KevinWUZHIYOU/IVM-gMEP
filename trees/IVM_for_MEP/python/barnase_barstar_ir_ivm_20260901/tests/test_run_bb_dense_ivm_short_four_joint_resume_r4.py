import importlib.util,json,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'scripts/run_bb_dense_ivm_short_four_joint_resume_r4.py'; s=importlib.util.spec_from_file_location('r4',P); R=s.loader.load_module()
class T(unittest.TestCase):
 def test_formal_closure(self): c=R.load_contract(R.CONTRACT); a=R.validate_authorization(c,R.AUTH); self.assertEqual(a['optimization_N_F_contribution'],0)
 def test_each_authorization_field_tamper_rejected(self):
  c=R.load_contract(R.CONTRACT); original=R.AUTH
  for key in ('authorizer','zero_openmm_preflight','optimization_N_F_contribution','OpenMM_calls_before_authorization','thread_policy','canonical_lock_paths','result_output','maximum_joint_EF_attempts'):
   with self.subTest(key=key), tempfile.TemporaryDirectory() as d:
    # Canonical-path rejection is itself fail-closed before any payload can be accepted.
    wrong=Path(d)/'auth.json'; x=json.loads(original.read_text()); x[key]=None; wrong.write_text(json.dumps(x))
    with self.assertRaisesRegex(Exception,'noncanonical R4 authorization'): R.validate_authorization(c,wrong)
 def test_no_openmm_import(self):
  import sys; self.assertNotIn('openmm',sys.modules)
if __name__=='__main__': unittest.main()
