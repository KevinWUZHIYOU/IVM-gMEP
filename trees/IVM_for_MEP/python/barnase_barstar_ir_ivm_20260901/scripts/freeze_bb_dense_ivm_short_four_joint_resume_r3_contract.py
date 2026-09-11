#!/usr/bin/env python3
"""Freeze R3 by reusing R2 science and adding the authorized entrypoint."""
from __future__ import annotations
import json, sys
from pathlib import Path
from unittest import mock
ROOT=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(ROOT/"scripts"))
import freeze_bb_dense_ivm_short_four_joint_resume_r2_contract as R2
import run_bb_dense_ivm_short_four_joint_resume_r3 as R3

def main():
    original=R2.atomic_json_no_overwrite
    def enriched(path,payload):
        payload=dict(payload); payload.update({"supersedes_R2_without_physical_calls":True,"authorized_entrypoint":R3.bind(Path(R3.__file__)),"execution_authorization_output":str(R3.AUTH),"R3_scientific_logic":"identical_R2_evaluator_only_output_path_and_auth_gate_added"})
        original(path,payload)
    R2.EVALUATOR.CANONICAL_RESULT=R3.RESULT.resolve()
    with mock.patch.object(R2,"atomic_json_no_overwrite",enriched): R2.main()
if __name__=="__main__": main()
