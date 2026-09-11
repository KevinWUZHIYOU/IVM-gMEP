import importlib.util, sys, tempfile, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r3_gate',ROOT/'scripts/run_bb_dense_ivm_short_four_joint_resume_r3.py')
R3=importlib.util.module_from_spec(spec); spec.loader.exec_module(R3)

class R3GateTest(unittest.TestCase):
    def test_formal_authorization_and_contract_pass(self):
        contract=R3.validate(R3.CONTRACT,R3.AUTH)
        self.assertEqual(contract['result_output'],str(R3.RESULT))
        self.assertNotIn('openmm',sys.modules)
    def test_noncanonical_authorization_path_fails(self):
        with tempfile.TemporaryDirectory() as d:
            wrong=Path(d)/'auth.json'; wrong.write_text(R3.AUTH.read_text())
            with self.assertRaisesRegex(Exception,'noncanonical R3 authorization'):
                R3.validate(R3.CONTRACT,wrong)
        self.assertNotIn('openmm',sys.modules)

if __name__=='__main__': unittest.main()
