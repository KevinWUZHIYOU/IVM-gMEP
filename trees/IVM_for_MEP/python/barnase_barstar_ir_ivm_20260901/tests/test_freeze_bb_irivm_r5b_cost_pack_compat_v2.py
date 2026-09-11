"""Zero-physical semantic tests for the canonical R5B cost-pack v2 freezer.

The arrays are deliberately tiny in the Cartesian dimension, while retaining
the production 32-image and 500-point axes.  Every negative test rewrites the
affected JSON/NPZ and recomputes its outer binding, so it reaches the semantic
check named by the test rather than failing at a stale wrapper hash.
"""
from __future__ import annotations

from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "compat_v2_under_test", SCRIPTS / "freeze_bb_irivm_r5b_cost_pack_compat_v2.py"
)
V = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(V)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def rewrite_npz(path: Path, mutate) -> None:
    with np.load(path, allow_pickle=False) as saved:
        fields = {key: np.asarray(saved[key]) for key in saved.files}
    mutate(fields)
    np.savez(path, **fields)


class Fixture:
    roles = ["near2000", "near4000", "near6000", "near8000", "near8982", "highest"]
    calls = [70, 140, 210, 280, 318, 360]
    nfs = [2000, 4000, 6000, 8000, 8982, 12000]

    def __init__(self, root: Path):
        self.root = root
        self.reg = self.file("registration.json", {})
        self.auth = self.file("authorization.json", {})
        self.capture_audit = self.file("capture.audit.json", {})
        self.capture_wal = self.file("capture.wal.jsonl", {})
        self.preparation_file = self.file("preparation.json", {})
        self.bundle_file = self.file("bundle.json", {})
        self.publication_gate_file = self.file("publication-gate.json", {})
        self.index = self.file("canonical-index.json", {"records": []})
        self.r5 = []
        self.inventory = []
        for call, nf in zip(self.calls, self.nfs):
            path = np.zeros((32, 3), np.float64)
            path[:, 0] = np.linspace(0, 31, 32)
            path[:, 1] = call * 1e-4 * np.sin(np.linspace(0, np.pi, 32))
            force = np.zeros_like(path)
            force[1:-1, 1] = np.linspace(0.1, 1.0, 30) + call * 1e-5
            energy = np.linspace(-2, 2, 32)
            p = root / f"r5-call-{call}.npz"
            np.savez(
                p,
                schema=np.asarray("bb-irivm-full-state-history-recorder-v3"),
                call_per_image=np.asarray(call), optimization_N_F=np.asarray(nf),
                path32_A=path, energy_eV=energy, force_eV_per_A=force,
                path_sha256=np.asarray(V.C.sha_array(path)),
                energy_sha256=np.asarray(V.C.sha_array(energy)),
                force_sha256=np.asarray(V.C.sha_array(force)),
                discrete_I_perp=np.asarray(V.C.iperp(path, force)),
            )
            self.r5.append(p)
            self.inventory.append({
                "history_index": call - 1, "call_per_image": call,
                "optimization_N_F": nf, "path_sha256": V.C.sha_array(path),
                "force_sha256": V.C.sha_array(force), "shard": str(p),
                "shard_sha256": V.C.sha_file(p), "formally_reportable": True,
                "terminal_uncommitted_charged_tail": False,
            })
        self.primary = self.file("primary.json", {
            "schema": V.S.PRIMARY_SCHEMA, "status": V.S.PRIMARY_STATUS,
            "actual_canonical_call_count": len(self.inventory),
            "last_complete_call": self.calls[-1], "path_inventory": self.inventory,
            "bindings": {"registration": V.F.bind(self.reg), "authorization": V.F.bind(self.auth),
                         "capture_manifest": {}, "canonical_index": V.F.bind(self.index),
                         "capture_audit": V.F.bind(self.capture_audit)},
        })
        self.manifest = self.file("capture-manifest.json", {"canonical_records": self.inventory})
        self._primary()["bindings"]["capture_manifest"] = V.F.bind(self.manifest)
        self._save_primary()
        self.supp = self.file("supplemental.json", {
            "schema": V.S.SUPPLEMENTAL_SCHEMA, "status": V.S.SUPPLEMENTAL_STATUS,
            "bindings": {"primary_verification": V.F.bind(self.primary),
                         "manifest": V.F.bind(self.manifest), "canonical_index": V.F.bind(self.index),
                         "registration": V.F.bind(self.reg), "authorization": V.F.bind(self.auth),
                         "capture_audit": V.F.bind(self.capture_audit), "wal": V.F.bind(self.capture_wal)},
        })
        self.hosts = {
            "remote": self.make_host("remote", range(4)),
            "local": self.make_host("local", range(4, 6)),
        }
        self.prep = {"host_plans": {r: {"contract_output": str(h["contract"]),
                                         "result_output": str(h["result"])}
                                      for r, h in self.hosts.items()}}
        self.output = root / "compat-v2.json"

    def file(self, name, payload):
        p = self.root / name
        write_json(p, payload)
        return p

    def _primary(self): return json.loads(self.primary.read_text())
    def _save_primary(self, payload=None):
        payload = self._primary() if payload is None else payload
        write_json(self.primary, payload)
        s = json.loads(self.supp.read_text()) if hasattr(self, "supp") else None
        if s is not None:
            s["bindings"]["primary_verification"] = V.F.bind(self.primary)
            write_json(self.supp, s)

    def make_host(self, role, indices):
        indices = list(indices); contract = self.file(f"{role}.contract.json", {})
        result = self.root / f"{role}.npz"; rows=[]; records=[]; shards=[]
        for j in indices:
            inv=self.inventory[j]; rp=self.r5[j]
            with np.load(rp,allow_pickle=False) as q:
                path=np.asarray(q["path32_A"]); force=np.asarray(q["force_eV_per_A"])
            dense=V.C.uniform500(path); energy=np.linspace(-1,1,500); ff=np.zeros_like(dense)
            shard=self.root/f"{role}-{j}.npz"
            np.savez(shard,schema=np.asarray(V.R.SHARD_SCHEMA),status=np.asarray("PASS"),
                call_per_image=np.asarray(inv["call_per_image"]),optimization_N_F=np.asarray(inv["optimization_N_F"]),
                selected_path_sha256=np.asarray(inv["path_sha256"]),discrete_I_perp=np.asarray(V.C.iperp(path,force)),
                dense_coordinates_A=dense,dense_energy_eV=energy,dense_force_eV_per_A=ff,
                dense_coordinate_sha256=np.asarray(V.C.sha_array(dense)),dense_energy_sha256=np.asarray(V.C.sha_array(energy)),dense_force_sha256=np.asarray(V.C.sha_array(ff)),
                densified_Delta_E_kcal_per_mol=np.asarray(float(j+1)),fresh_dense_joint_EF_attempts=np.asarray(498),
                inherited_endpoint_joint_EF_values=np.asarray(2),optimization_N_F_contribution=np.asarray(0))
            shards.append(shard); rows.append({"selection_role":self.roles[j],"optimization_N_F":inv["optimization_N_F"],
                "path_sha256":inv["path_sha256"],"discrete_I_perp":V.C.iperp(path,force),
                "path_identifier":{"calls_per_image":inv["call_per_image"]}})
            records.append({"path":str(shard),"sha256":V.C.sha_file(shard),"calls_per_image":inv["call_per_image"],
                            "optimization_N_F":inv["optimization_N_F"],"selected_path_sha256":inv["path_sha256"]})
        np.savez(result,schema=np.asarray(V.R.RESULT_SCHEMA),status=np.asarray("PASS"),
                 optimization_N_F=np.asarray([r["optimization_N_F"] for r in rows]),
                 selected_path_sha256=np.asarray([r["path_sha256"] for r in rows]),optimization_N_F_contribution=np.asarray(0))
        source=self.file(f"{role}.sealed_source.wal.jsonl", {"kind":"source"})
        densewal=self.file(f"{role}.dense_joint_ef.wal.jsonl", {"kind":"dense"})
        manifest=self.file(f"{role}.manifest.json", {"schema":V.R.MANIFEST_SCHEMA,"status":"PASS",
            "contract":V.F.bind(contract),"aggregate":V.F.bind(result),"shards":records})
        audit=self.file(f"{role}.audit.json", {"schema":V.R.AUDIT_SCHEMA,"status":"PASS",
            "contract":V.F.bind(contract),"output":V.F.bind(result),"manifest":V.F.bind(manifest),
            "sealed_source_wal":V.F.bind(source),"dense_joint_ef_wal":V.F.bind(densewal),
            "optimization_N_F_contribution":0,"original_discrete_new_calls":0,
            "fresh_joint_EF_attempt_lifetime":{"attempt_started":len(rows)*498,"attempt_succeeded":len(rows)*498,"attempt_failed":0,"attempt_open":0,"retry_attempts":0}})
        return {"contract":contract,"result":result,"manifest":manifest,"audit":audit,"source":source,"densewal":densewal,"rows":rows,"shards":shards}

    def closure(self, *_args, **kwargs):
        p=json.loads(self.primary.read_text()); s=json.loads(self.supp.read_text())
        if p.get("schema")!=V.S.PRIMARY_SCHEMA or p.get("status")!=V.S.PRIMARY_STATUS: raise ValueError("primary R5B verification is not PASS")
        if s.get("schema")!=V.S.SUPPLEMENTAL_SCHEMA or s.get("status")!=V.S.SUPPLEMENTAL_STATUS: raise ValueError("supplemental R5B verification is not PASS")
        if s.get("bindings",{}).get("primary_verification")!=V.F.bind(self.primary): raise ValueError("supplemental closure differs")
        inv=p.get("path_inventory",[]); calls=[x["call_per_image"] for x in inv]
        if calls!=sorted(calls) or len(calls)!=len(set(calls)): raise ValueError("call gap")
        last=p.get("last_complete_call")
        if last not in calls: raise ValueError("last-complete")
        for x in inv:
            expected=x["call_per_image"]<=last
            if x.get("formally_reportable")!=expected or x.get("terminal_uncommitted_charged_tail")==expected: raise ValueError("tail")
        return ({},{},p,s,{}, {},{},inv,{})

    def patches(self):
        def load_host_state(contract,*_):
            for role,h in self.hosts.items():
                if Path(contract).resolve()==h["contract"].resolve():
                    return {"host_role":role,"rows":h["rows"],"contract_path":h["contract"].resolve(),"result_output":h["result"].resolve()}
            raise AssertionError(contract)
        def read_wal(path,*_): return {"path":str(path)}
        def lifetime(wal):
            role="remote" if "remote" in wal["path"] else "local"; n=len(self.hosts[role]["rows"])*498
            return {"attempt_started":n,"attempt_succeeded":n,"attempt_failed":0,"attempt_open":0,"retry_attempts":0}
        def validate_shard(path,row,_state):
            with np.load(path,allow_pickle=False) as saved:
                return {key:np.asarray(saved[key]) for key in saved.files}
        stack=[mock.patch.object(V.S,"validate_final_closure",side_effect=self.closure),
               mock.patch.object(V.S,"CANONICAL_PRIMARY",self.primary),
               mock.patch.object(V.S,"CANONICAL_SUPPLEMENTAL",self.supp),
               mock.patch.object(V.S,"CANONICAL_MANIFEST",self.manifest),
               mock.patch.object(V.S,"CANONICAL_INDEX",self.index),
               mock.patch.object(V.F,"validate_preparation",return_value=self.prep),
               mock.patch.object(V.F,"PREPARATION",self.preparation_file),
               mock.patch.object(V.F,"BUNDLE",self.bundle_file),
               mock.patch.object(V.F,"PUBLICATION_GATE",self.publication_gate_file),
               mock.patch.object(V.F,"load_host_state",side_effect=load_host_state),
               mock.patch.object(V.R,"configured_delegate",side_effect=lambda _s:nullcontext()),
               mock.patch.object(V.R.DELEGATE,"_read_wal",side_effect=read_wal),
               mock.patch.object(V.R.DELEGATE,"_lifetime",side_effect=lifetime),
               mock.patch.object(V.R.DELEGATE,"_validate_shard",side_effect=validate_shard),
               mock.patch.object(V.R.DELEGATE,"_validate_dense_closure",return_value=None),
               mock.patch.object(V.R.DELEGATE,"_validate_source_closure",return_value=None),
               mock.patch.object(V,"OUTPUT",self.output)]
        return stack

    def build(self):
        stack=self.patches()
        for p in stack:p.start()
        try:return V.build(self.output)
        finally:
            for p in reversed(stack):p.stop()


class CompatV2SemanticTests(unittest.TestCase):
    def fresh(self):
        td=tempfile.TemporaryDirectory(); self.addCleanup(td.cleanup); return Fixture(Path(td.name))

    def test_positive_complete_remote4_local2_zero_physical(self):
        f=self.fresh(); out=f.build()
        self.assertEqual(out["status"],V.STATUS); self.assertEqual([h["role"] for h in out["hosts"]],["remote","local"])
        self.assertEqual(len(out["points"]),6); self.assertEqual(out["OpenMM_calls"],0); self.assertEqual(out["optimization_N_F_contribution"],0)
        self.assertNotIn("openmm",sys.modules)

    def test_real_builder_output_is_accepted_by_final_plot_adapter(self):
        import adapt_bb_irivm_r5b_cost_pack_for_final_plot as adapter
        f=self.fresh(); out=f.build(); adapted=adapter.load(f.output)
        self.assertEqual(out["point_count"],6)
        self.assertTrue(all(p["optimization_N_F_contribution"]==0 for p in out["points"]))
        self.assertEqual(len(adapted["points"]),6)

    def test_wrong_weighted_iperp_rejected_with_fresh_hash(self):
        f=self.fresh(); h=f.hosts["remote"]; rewrite_npz(h["shards"][0],lambda z:z.__setitem__("discrete_I_perp",np.asarray(9.0)))
        m=json.loads(h["manifest"].read_text());m["shards"][0]["sha256"]=V.C.sha_file(h["shards"][0]);write_json(h["manifest"],m)
        a=json.loads(h["audit"].read_text());a["manifest"]=V.F.bind(h["manifest"]);write_json(h["audit"],a)
        with self.assertRaisesRegex(RuntimeError,"weighted Iperp"):f.build()

    def test_primary_schema_status_and_supplemental_status_are_closed(self):
        for target,key in (("primary","schema"),("primary","status"),("supp","schema"),("supp","status")):
            f=self.fresh();p=json.loads(getattr(f,target).read_text());p[key]="FAKE";write_json(getattr(f,target),p)
            if target=="primary":
                s=json.loads(f.supp.read_text());s["bindings"]["primary_verification"]=V.F.bind(f.primary);write_json(f.supp,s)
            with self.subTest(target=target,key=key),self.assertRaises(ValueError):f.build()

    def test_supplemental_primary_path_binding_rejected_even_after_valid_hash(self):
        f=self.fresh();other=f.file("other-primary.json",json.loads(f.primary.read_text()));s=json.loads(f.supp.read_text());s["bindings"]["primary_verification"]=V.F.bind(other);write_json(f.supp,s)
        with self.assertRaisesRegex(ValueError,"supplemental closure"):f.build()

    def test_call_gap_last_complete_and_tail_rejected(self):
        for kind in ("gap","last","tail"):
            f=self.fresh();p=json.loads(f.primary.read_text())
            if kind=="gap":p["path_inventory"][1]["call_per_image"]=p["path_inventory"][0]["call_per_image"]
            elif kind=="last":p["last_complete_call"]=999
            else:p["path_inventory"][-1]["terminal_uncommitted_charged_tail"]=True
            write_json(f.primary,p);s=json.loads(f.supp.read_text());s["bindings"]["primary_verification"]=V.F.bind(f.primary);write_json(f.supp,s)
            with self.subTest(kind=kind),self.assertRaises(ValueError):f.build()

    def test_host_4_2_missing_and_duplicate_selection_rejected(self):
        f=self.fresh();f.hosts["remote"]["rows"].pop()
        with self.assertRaisesRegex(RuntimeError,"4/2 cardinality"):f.build()
        f=self.fresh();f.hosts["local"]["rows"][1]["selection_role"]=f.hosts["local"]["rows"][0]["selection_role"]
        with self.assertRaisesRegex(RuntimeError,"uniqueness"):f.build()

    def test_aggregate_manifest_audit_wal_and_lifetime_rejected(self):
        mutations=("aggregate","manifest","audit","wal","lifetime")
        for kind in mutations:
            f=self.fresh();h=f.hosts["remote"]
            if kind=="aggregate":rewrite_npz(h["result"],lambda z:z.__setitem__("optimization_N_F",np.asarray([1,2,3,4])))
            elif kind=="manifest":
                m=json.loads(h["manifest"].read_text());m["shards"][0]["optimization_N_F"]+=1;write_json(h["manifest"],m)
            elif kind=="audit":
                a=json.loads(h["audit"].read_text());a["optimization_N_F_contribution"]=1;write_json(h["audit"],a)
            elif kind=="wal":
                a=json.loads(h["audit"].read_text());a["dense_joint_ef_wal"]={"path":str(h["densewal"]),"sha256":"0"*64};write_json(h["audit"],a)
            else:
                a=json.loads(h["audit"].read_text());a["fresh_joint_EF_attempt_lifetime"]["attempt_open"]=1;write_json(h["audit"],a)
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):f.build()

    def test_audit_started_succeeded_retry_lifetime_must_equal_wal(self):
        """A hash-consistent PASS audit must not disagree with its bound WAL."""
        for field, value in (("attempt_started", 1991), ("attempt_succeeded", 1991), ("retry_attempts", 1)):
            f=self.fresh(); h=f.hosts["remote"]; a=json.loads(h["audit"].read_text())
            a["fresh_joint_EF_attempt_lifetime"][field]=value; write_json(h["audit"],a)
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "lifetime|accounting"):
                f.build()

    def test_r5_shard_hash_call_nf_path_and_force_rejected(self):
        for kind in ("hash","call","nf","path","force"):
            f=self.fresh();p=f.r5[0]
            if kind=="hash":p.write_bytes(p.read_bytes()+b"x")
            else:
                def mutate(z):
                    if kind in ("call","nf"):z["call_per_image" if kind=="call" else "optimization_N_F"]=np.asarray(999)
                    elif kind=="path":z["path32_A"]=np.asarray(z["path32_A"])+0.2
                    else:z["force_eV_per_A"]=np.asarray(z["force_eV_per_A"])+0.2
                rewrite_npz(p,mutate)
            if kind!="hash":
                pri=json.loads(f.primary.read_text());pri["path_inventory"][0]["shard_sha256"]=V.C.sha_file(p);write_json(f.primary,pri);s=json.loads(f.supp.read_text());s["bindings"]["primary_verification"]=V.F.bind(f.primary);write_json(f.supp,s)
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):f.build()

    def test_arc500_coordinate_energy_force_hash_and_498_2_nof_rejected(self):
        for kind in ("coord","energy","force","fresh","endpoint","nof"):
            f=self.fresh();h=f.hosts["remote"];p=h["shards"][0]
            def mutate(z):
                key={"coord":"dense_coordinates_A","energy":"dense_energy_eV","force":"dense_force_eV_per_A",
                     "fresh":"fresh_dense_joint_EF_attempts","endpoint":"inherited_endpoint_joint_EF_values","nof":"optimization_N_F_contribution"}[kind]
                z[key]=np.asarray(z[key])+1
            rewrite_npz(p,mutate);m=json.loads(h["manifest"].read_text());m["shards"][0]["sha256"]=V.C.sha_file(p);write_json(h["manifest"],m);a=json.loads(h["audit"].read_text());a["manifest"]=V.F.bind(h["manifest"]);write_json(h["audit"],a)
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):f.build()


if __name__ == "__main__":
    unittest.main()
