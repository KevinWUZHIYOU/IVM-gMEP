import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
import build_bb_ode8_discrete32_action_pack as B
class T(unittest.TestCase):
 def test_formula(self):
  x=np.zeros((32,9396)); x[:,0]=np.arange(32); f=np.zeros_like(x); f[:,0]=2
  self.assertEqual(B.action(x,f),62.0)
 def test_real_sources_exact_expected(self):
  if not B.CONTRACT.exists(): self.skipTest('freeze first')
  nf,s,_,_=B.compute(B.CONTRACT); self.assertTrue(np.array_equal(nf,B.EXPECTED_NF)); self.assertTrue(np.allclose(s,B.EXPECTED_S,rtol=0,atol=1e-9))
 def test_no_arc500_metric_and_zero_calls(self):
  text=Path(B.__file__).read_text(); self.assertNotIn("dense_force_eV_per_A']",text); self.assertNotIn('OpenMMBackend',text)
if __name__=='__main__': unittest.main()
