#!/usr/bin/env python3
"""Freeze the ODE8 zero-query compact-pack inputs."""
from __future__ import annotations
import json,os
from pathlib import Path
import build_bb_ode8_discrete32_action_pack as B
TEST=B.ROOT/'tests/test_bb_ode8_discrete32_action_pack.py'
def main():
 B.require(not B.CONTRACT.exists() and not B.OUTPUT.exists() and not B.AUDIT.exists(),'prospective output exists')
 payload={'schema':'bb-ode8-discrete32-action-pack-contract-v1','status':'FROZEN_ZERO_OPENMM','metric':'sum over 31 original-image segments of 0.5*(||F_k||+||F_k+1||)*||x_k+1-x_k||','images':32,'segments':31,'interpolation':'none','arc500_used_by_metric':False,'optimization_N_F_contribution':0,'expected_NF':B.EXPECTED_NF.tolist(),'expected_action_eV':B.EXPECTED_S.tolist(),'builder':str(Path(B.__file__).resolve()),'builder_sha256':B.sha_file(B.__file__),'freezer':str(Path(__file__).resolve()),'freezer_sha256':B.sha_file(__file__),'test':str(TEST.resolve(strict=True)),'test_sha256':B.sha_file(TEST),'endpoint':str(B.ENDPOINT),'endpoint_sha256':B.sha_file(B.ENDPOINT),'sources':[str(x) for x in B.FILES],'source_sha256':[B.sha_file(x) for x in B.FILES],'output':str(B.OUTPUT),'audit':str(B.AUDIT),'physical_calls':0}
 B.write_json(B.CONTRACT,payload); print(B.CONTRACT); print(B.sha_file(B.CONTRACT))
if __name__=='__main__': main()
