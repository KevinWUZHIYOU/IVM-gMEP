#!/usr/bin/env python3
"""Issue R3 authorization from a captured zero-OpenMM preflight report."""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT/"scripts"))
import run_bb_dense_ivm_short_four_joint_resume_r3 as R3
def write(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name('.'+path.name+f'.{os.getpid()}')
    with tmp.open('x') as f: json.dump(payload,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    if path.exists(): tmp.unlink(); raise FileExistsError(path)
    os.link(tmp,path); tmp.unlink()
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--contract',type=Path,default=R3.CONTRACT); p.add_argument('--preflight',type=Path,required=True); p.add_argument('--output',type=Path,default=R3.AUTH); a=p.parse_args(argv)
    c=a.contract.resolve(strict=True); pre=a.preflight.resolve(strict=True); out=a.output.resolve()
    R3.require(c==R3.CONTRACT.resolve() and out==R3.AUTH.resolve(),'noncanonical authorization inputs')
    report=json.loads(pre.read_text()); R3.require(report.get('status')=='ZERO_OPENMM_PREFLIGHT_PASS','preflight not PASS'); R3.require(report.get('contract_sha256')==R3.ENGINE.ENGINE.sha256_file(c),'preflight contract drift'); R3.require(report.get('fresh_R2_joint_EF_attempts_planned')==1000,'preflight scope drift')
    payload={'schema':R3.AUTH_SCHEMA,'status':R3.AUTH_STATUS,'enabled':True,'contract':R3.bind(c),'authorized_entrypoint':R3.bind(Path(R3.__file__)),'authorizer':R3.bind(Path(__file__)),'zero_openmm_preflight':R3.bind(pre),'result_output':str(R3.RESULT),'maximum_joint_EF_attempts':1000,'optimization_N_F_contribution':0,'OpenMM_calls_before_authorization':0}
    write(out,payload); print(json.dumps({'authorization':str(out),'sha256':R3.ENGINE.ENGINE.sha256_file(out),'status':R3.AUTH_STATUS},indent=2))
if __name__=='__main__': main()
