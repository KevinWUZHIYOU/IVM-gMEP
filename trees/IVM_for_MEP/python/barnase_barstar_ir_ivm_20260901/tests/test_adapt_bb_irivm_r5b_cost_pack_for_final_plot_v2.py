import json,tempfile,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import adapt_bb_irivm_r5b_cost_pack_for_final_plot as A
import freeze_bb_irivm_r5b_cost_pack_compat_v2 as V
class AdapterV2(unittest.TestCase):
 def test_six_point_v2_pass_and_binding_tamper_fails(self):
  with tempfile.TemporaryDirectory() as d:
   d=Path(d); shard=d/'s';shard.write_bytes(b'x');b={'path':str(shard),'sha256':A.C.sha_file(shard)}
   points=[{'optimization_N_F':i+1,'discrete_I_perp':float(i),'densified_Delta_E_kcal_per_mol':float(6-i),'path_sha256':str(i),'force32_sha256':str(i),'optimization_N_F_contribution':0,'arc500_shard':b,'R5B_shard':b} for i in range(6)]
   p=d/'pack.json';p.write_text(json.dumps({'schema':V.SCHEMA,'status':V.STATUS,'points':points,'OpenMM_calls':0,'optimization_N_F_contribution':0}));self.assertEqual(len(A.load(p)['points']),6)
   points[0]['arc500_shard']['sha256']='0'*64;p.write_text(json.dumps({'schema':V.SCHEMA,'status':V.STATUS,'points':points,'OpenMM_calls':0,'optimization_N_F_contribution':0}))
   with self.assertRaisesRegex(RuntimeError,'binding drift'):A.load(p)
if __name__=='__main__':unittest.main()
