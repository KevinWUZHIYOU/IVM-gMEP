#!/usr/bin/env python3
"""Freeze the verified short-replay four-path joint-E/F arc500 contract."""

from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
FREEZER=Path(__file__).resolve(); EVALUATOR=ROOT/"scripts/evaluate_bb_dense_ivm_short_four_joint.py"; ENGINE=ROOT/"scripts/evaluate_bb_dense_four_paths_joint_v2.py"; BB_CORE=ROOT/"src/bb_core.py"; TOPOLOGY=ROOT/"inputs/complex_fixed.pdb"
CANONICAL_VERIFICATION=ROOT/"reports/verifications/BB_IRIVM_SELECTED_LEGACY_REPLAY_NF1500_20260902_R1.json"
CALLS=(2,5,10,50); EXPECTED_NF=(60,150,300,1329); N_DENSE=500; DIMENSION=9396; N_IMAGES=32
MINIMUM_FREE_BYTES_RESERVE=8*1024**3; ESTIMATED_SHARD_BYTES=(2*N_DENSE*DIMENSION+N_IMAGES*DIMENSION)*8+8*1024**2
CANONICAL_LOCKS={Path("/Users/wuzhiyou/IVM_for_MEP/.bb_global_heavy_task.lock").resolve(),Path("/Users/wuzhiyou/Downloads/IVM_for_MEP/.bb_global_heavy_task.lock").resolve()}
METRIC_POLICY={"coordinate_space":"unweighted_3N_cartesian","sampling":"uniform_cartesian_arclength","interpolation":"piecewise_linear","number_of_points":500,"grid_formula":"s_j=j*L/499,j=0,...,499","include_both_endpoints":True,"dense_physical_call":"one_joint_energy_full_force_getState_per_point","dense_force_use":"stored_only_not_used_by_metrics_or_optimization","discrete_I_perp_implementation":"bb_core.path_diagnostics","tangent":"normalized central difference: x[i+1] - x[i-1]","weight":"half sum of adjacent Cartesian chord lengths","original_fixed_endpoint_diagnostic_force_evaluations":0,"dense_fixed_endpoint_diagnostic_force_evaluations":2,"optimization_N_F_contribution":0,"original_discrete_state":"verified_short_replay_path_and_sealed_online_Iperp_zero_new_calls"}

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bind(path): path=Path(path).expanduser().resolve(strict=True); return {"path":str(path),"sha256":sha(path)}
def array_sha(a): return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float64).tobytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--short-replay-verification",type=Path,default=CANONICAL_VERIFICATION); p.add_argument("--topology",type=Path,default=TOPOLOGY); p.add_argument("--host-role",choices=("local","remote"),default="local"); p.add_argument("--global-lock-path",type=Path,required=True); p.add_argument("--peer-global-lock-path",type=Path,required=True); a=p.parse_args()
    destination=a.output.expanduser().resolve();
    if destination.exists(): raise FileExistsError(f"refusing to overwrite {destination}")
    topology=a.topology.expanduser().resolve(strict=True)
    if topology!=TOPOLOGY.resolve(strict=True) or sha(topology)!="26b58b4301c1e293c7fcd01878c3a208451621099419cbf5b41bd80eb96a0f49": raise ValueError("noncanonical BB topology")
    verification_path=a.short_replay_verification.expanduser().resolve(strict=True)
    if verification_path != CANONICAL_VERIFICATION.resolve(strict=True) or sha(verification_path)!="dfa12125bd68420e9c196443121b34104f83202fde64113b37856281f7fae322": raise ValueError("noncanonical short replay verification")
    verification=json.loads(verification_path.read_text());
    if verification.get("schema")!="bb-irivm-selected-legacy-replay-verification-v1" or verification.get("status")!="PASS_EXACT_OUTPUT_ONLY_PREFIX" or verification.get("exact_prefix_through_call")!=50 or verification.get("exact_prefix_through_optimization_N_F")!=1329: raise ValueError("short replay verification is not exact-prefix PASS through call50")
    closure=verification.get("bindings",{}); required={"amendment","authorization","reference_result","registration","replay_audit","replay_result"}
    if set(closure)!=required: raise ValueError("short replay closure roles mismatch")
    for role,record in closure.items():
        if bind(Path(record["path"]))!=record: raise ValueError(f"short replay {role} binding drift")
    audit=json.loads(Path(closure["replay_audit"]["path"]).read_text())
    if audit.get("consistent") is not True or audit.get("endpoint_joint_ef_cache_misses")!=0: raise ValueError("short replay NoF audit mismatch")
    verified={int(r["call_per_image"]):r for r in verification["new_reporting_rows"]}; result=Path(closure["replay_result"]["path"]).resolve(strict=True)
    with np.load(result,allow_pickle=False) as s:
        stored_calls=np.asarray(s["stored_path_calls"],dtype=np.int64); stored_paths=np.asarray(s["stored_paths"],dtype=np.float64); calls=np.asarray(s["calls_per_image"],dtype=np.int64); counters=np.asarray(s["counter_vectors"],dtype=np.int64); iperp=np.asarray(s["i_perp"],dtype=np.float64)
        paths=[]
        for call,expected_nf in zip(CALLS,EXPECTED_NF):
            si=np.flatnonzero(stored_calls==call); hi=np.flatnonzero(calls==call)
            if si.size!=1 or hi.size!=1: raise ValueError(f"call {call} absent/nonunique")
            path=np.ascontiguousarray(stored_paths[int(si[0])]); idx=int(hi[0]); row=verified[call]; nf=int(np.sum(counters[idx],dtype=np.int64)); value=float(iperp[idx]); path_sha=array_sha(path)
            if idx!=int(row["history_index"]) or nf!=expected_nf or nf!=int(row["optimization_N_F"]) or np.asarray(value).tobytes()!=np.asarray(float(row["discrete_I_perp_eV2_per_A"])).tobytes() or path_sha!=row["selected_path_sha256"] or row.get("fully_evaluated_current_path") is not True or row.get("terminal_tail") is not False: raise ValueError(f"call {call} verified row mismatch")
            paths.append({"label":f"IVM_call{call}_NF{nf}","method":"IVM","host_role":a.host_role,"optimization_N_F":nf,"path_identifier":{"calls_per_image":call,"history_index":idx,"fully_evaluated_current_path":True,"terminal_tail":False},"short_replay_source":bind(result),"path_sha256":path_sha,"online_discrete_I_perp":value,"original_discrete_policy":"verified_short_replay_path_and_sealed_online_Iperp_zero_new_calls","dense_policy":"fresh_or_complete_validated_shard_500_joint_EF"})
    locks=[a.global_lock_path.expanduser().resolve(),a.peer_global_lock_path.expanduser().resolve()]
    if len(set(locks))!=2 or set(locks)!=CANONICAL_LOCKS: raise ValueError("canonical two-lock set required")
    contract={"schema":"bb-dense-four-ivm-short-prefix-joint-ef-contract-v1","frozen":True,"method":"IVM","jobs":1,"purpose":"fresh joint-E/F arc500 for four verified exact-prefix short-replay IVM paths","contract_freezer":bind(FREEZER),"evaluator":bind(EVALUATOR),"joint_engine":bind(ENGINE),"bb_core":bind(BB_CORE),"topology":bind(topology),"short_replay_verification":bind(verification_path),"short_replay_closure":closure,"host_local_lock_paths":[str(x) for x in locks],"metric_policy":METRIC_POLICY,"paths":paths,"physical_call_policy":{"original_discrete_new_calls_per_path":0,"dense_joint_EF_per_path":500,"dense_endpoint_joint_EF_per_path":2,"dense_full_force_storage":True,"optimization_N_F_contribution":0,"jobs":1,"all_six_thread_caps":1,"two_host_local_fcntl_locks":True},"storage_policy":{"authoritative_unit":"complete_atomic_path_shard","aggregate_role":"index_metrics_and_plot_compatible_dense_energy_not_dense_coordinates_or_forces","minimum_free_bytes_reserve":MINIMUM_FREE_BYTES_RESERVE,"estimated_uncompressed_shard_bytes_per_path":ESTIMATED_SHARD_BYTES},"durability_policy":{"separate_source_and_dense_append_only_fsync_WAL":True,"atomic_complete_path_shard":True,"shard_contains_dense_coordinates_energy_full_force":True,"manifest_binds_every_shard_and_array_hash":True,"source_and_dense_WAL_bind_published_shard":True,"restart_reuse_unit":"complete_fully_validated_path_shard","reuse_individual_incomplete_WAL_successes":False,"refuse_final_output_overwrite":True}}
    destination.parent.mkdir(parents=True,exist_ok=True); tmp=destination.with_name(destination.name+f".tmp.{os.getpid()}")
    with tmp.open("x") as f: f.write(json.dumps(contract,indent=2,sort_keys=True)+"\n"); f.flush(); os.fsync(f.fileno())
    try:
        os.link(tmp,destination); fd=os.open(destination.parent,os.O_RDONLY); os.fsync(fd); os.close(fd)
    finally:
        tmp.unlink()
    print(json.dumps({"status":"FROZEN","contract":str(destination),"sha256":sha(destination),"calls":list(CALLS),"optimization_N_F":list(EXPECTED_NF)},indent=2))

if __name__=="__main__": main()
