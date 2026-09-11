#!/usr/bin/env python3
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT/'scripts')); import run_bb_dense_ivm_short_four_joint_resume_r4b as R4
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name('.'+p.name+f'.{os.getpid()}')
 with t.open('x') as f: json.dump(x,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
 if p.exists(): t.unlink(); raise FileExistsError(p)
 os.link(t,p); t.unlink()
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument('--contract',type=Path,default=R4.CONTRACT); p.add_argument('--preflight',type=Path,required=True); p.add_argument('--output',type=Path,default=R4.AUTH); a=p.parse_args(argv); c=R4.load_contract(a.contract); pre=a.preflight.resolve(strict=True); out=a.output.resolve(); R4.require(pre==R4.PREAUTH.resolve() and out==R4.AUTH.resolve(),'noncanonical authorizer path')
 q=json.loads(pre.read_text()); R4.require(q.get('status')=='ZERO_OPENMM_PREFLIGHT_PASS' and q.get('contract_sha256')==R4.ENGINE.ENGINE.sha256_file(R4.CONTRACT),'preflight binding/status drift'); R4.require(q.get('fresh_R2_joint_EF_attempts_planned')==1000 and q.get('optimization_N_F_contribution')==0,'preflight accounting drift'); ph=q.get('physical_preflight',{}); R4.require(ph.get('canonical_two_locks_available') is True and ph.get('R2_attempt_lifetime',{}).get('attempt_started')==0,'preflight lock/lifetime drift')
 x={'schema':R4.AUTH_SCHEMA,'status':R4.AUTH_STATUS,'enabled':True,'contract':R4.bind(R4.CONTRACT),'authorized_entrypoint':R4.bind(Path(R4.__file__)),'authorizer':R4.bind(Path(__file__)),'zero_openmm_preflight':R4.bind(pre),'result_output':str(R4.RESULT),'maximum_joint_EF_attempts':1000,'optimization_N_F_contribution':0,'OpenMM_calls_before_authorization':0,'thread_policy':{n:'1' for n in R4.ENGINE.ENGINE.THREAD_VARS},'canonical_lock_paths':[str(z) for z in sorted(R4.ENGINE.CANONICAL_LOCKS,key=str)]}; write(out,x)
if __name__=='__main__': main()
