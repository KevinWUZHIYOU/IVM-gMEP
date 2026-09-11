#!/usr/bin/env python3
"""Freeze the zero-OpenMM coordinate manifest for the NF8982 BB 0 K section."""
from __future__ import annotations

import argparse, hashlib, json, os
from pathlib import Path
from typing import Mapping
import numpy as np

import plot_bb_irivm_evolution_hashbound as common

SCHEMA = "bb-irivm-nf8982-0k-section-freeze-contract-v1"
STATUS = "FROZEN_BEFORE_ANY_SECTION_ENERGY_CALL"
POINTS, NORMAL_POINTS, ATOMS = 59, 28, 3132
N_GRID = np.arange(-0.10, 0.4401, 0.02, dtype=np.float64)

def array_sha(value):
    a=np.ascontiguousarray(np.asarray(value,dtype="<f8")); return hashlib.sha256(a.tobytes()).hexdigest()

def arc(path):
    f=common._as_band(path).reshape(len(path),-1); d=np.linalg.norm(np.diff(f,axis=0),axis=1)
    q=np.r_[0.0,np.cumsum(d)]; common.require(q[-1]>1e-14,"degenerate path"); return q/q[-1]

def resample(path, target):
    p=common._as_band(path); f=p.reshape(len(p),-1); s=arc(p)
    return np.stack([np.interp(target,s,f[:,j]) for j in range(f.shape[1])],1).reshape(len(target),p.shape[1],3)

def basis(center, initial):
    flat=center.reshape(POINTS,-1); tangent=np.empty_like(flat)
    tangent[1:-1]=flat[2:]-flat[:-2]; tangent[0]=flat[1]-flat[0]; tangent[-1]=flat[-1]-flat[-2]
    norm=np.linalg.norm(tangent,axis=1); common.require(np.all(norm>1e-14),"degenerate center tangent")
    tangent/=norm[:,None]; displacement=initial.reshape(POINTS,-1)-flat
    displacement-=np.sum(displacement*tangent,axis=1)[:,None]*tangent
    size=np.linalg.norm(displacement,axis=1); valid=size>1e-12
    common.require(np.count_nonzero(valid)>=3,"insufficient normal anchors")
    normal=np.zeros_like(displacement); normal[valid]=displacement[valid]/size[valid,None]
    good=np.flatnonzero(valid)
    for i in np.flatnonzero(~valid):
        candidate=normal[good[np.argmin(np.abs(good-i))]].copy()
        candidate-=np.dot(candidate,tangent[i])*tangent[i]; candidate/=np.linalg.norm(candidate)
        normal[i]=candidate
    return tangent.reshape(center.shape),normal.reshape(center.shape)

def exclusive_npz(path, **payload):
    common.require(not path.exists(),f"refusing to overwrite {path}"); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name("."+path.name+".tmp.npz"); np.savez_compressed(tmp,**payload); os.link(tmp,path); tmp.unlink()

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument("--contract",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(argv); cp=a.contract.resolve(strict=True); c=json.loads(cp.read_text())
    common.require(c.get("schema")==SCHEMA and c.get("status")==STATUS,"section contract not frozen")
    base=common.resolve_binding(cp,c["base_evolution_contract"],"base evolution contract"); bc=json.loads(base.read_text())
    gates=common.validate_json_gates_before_selected_npz(base,bc)
    common.require(int(gates["actual_nf"])==9985,"unexpected selected terminal NF")
    with np.load(gates["paths"]["selected_result"],allow_pickle=False) as z:
        selected=common.collect_selected_committed_paths(z,gates["verification"],int(gates["actual_nf"]))
    common.require(8982 in selected and 9985 not in selected,"NF8982/partial-tail policy mismatch")
    target=np.linspace(0.0,1.0,POINTS); center=resample(selected[8982]["path"],target); initial=resample(selected[-1]["path"],target)
    tangents,normals=basis(center,initial)
    coordinates=center[:,None,:,:]+N_GRID[None,:,None,None]*np.sqrt(ATOMS)*normals[:,None,:,:]
    hashes=np.asarray([[array_sha(coordinates[i,j]) for j in range(NORMAL_POINTS)] for i in range(POINTS)])
    exclusive_npz(a.output.resolve(),schema=np.asarray("bb-irivm-nf8982-0k-section-coordinate-manifest-v1"),
        status=np.asarray(STATUS),center_source_optimization_N_F=np.asarray(8982,dtype=np.int64),
        terminal_partial_tail_excluded=np.asarray(True),center_path59_A=center,center_s=target,
        common_initial_path59_A=initial,central_tangents=tangents,normal_directions=normals,
        normal_coordinate_rms_A=N_GRID,coordinates_A=coordinates,coordinate_sha256=hashes,
        coordinate_space=np.asarray("Cartesian Angstrom"),surface_kind=np.asarray("unrelaxed current-IR-IVM-centered single-normal 0 K section"),
        required_energy_evaluations=np.asarray(POINTS*NORMAL_POINTS,dtype=np.int64),required_force_evaluations=np.asarray(0,dtype=np.int64),
        optimization_N_F_contribution=np.asarray(0,dtype=np.int64),selected_result=np.asarray(str(gates["paths"]["selected_result"])),
        selected_result_sha256=np.asarray(common.sha256_file(gates["paths"]["selected_result"])),center_path_sha256=np.asarray(array_sha(center)))
    print(json.dumps({"output":str(a.output.resolve()),"sha256":common.sha256_file(a.output.resolve()),"shape":list(coordinates.shape),
      "physical_calls_started":0,"center_N_F":8982,"partial_tail_excluded":True},indent=2))
if __name__=="__main__": main()
