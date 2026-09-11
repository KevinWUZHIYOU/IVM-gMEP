#!/usr/bin/env python3
from pathlib import Path
from unittest import mock
import sys
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT/'scripts'))
import freeze_bb_dense_ivm_short_four_joint_resume_r2_contract as R2
import run_bb_dense_ivm_short_four_joint_resume_r4 as R4
def main():
 original=R2.atomic_json_no_overwrite
 def enriched(path,p):
  p=dict(p); p.update({'supersedes_R2_and_R3_without_reuse':True,'authorized_entrypoint':R4.bind(Path(R4.__file__)),'contract_freezer_R4':R4.bind(Path(__file__)),'execution_authorizer':R4.bind(R4.AUTHORIZER),'test':R4.bind(R4.TEST),'R3_supersession_receipt':R4.bind(R4.R3_RECEIPT),'execution_authorization_output':str(R4.AUTH),'R4_scientific_logic':'identical_R2_evaluator_only_output_path_and_auth_gate_added'}); original(path,p)
 R2.EVALUATOR.CANONICAL_RESULT=R4.RESULT.resolve()
 with mock.patch.object(R2,'atomic_json_no_overwrite',enriched): R2.main()
if __name__=='__main__': main()
