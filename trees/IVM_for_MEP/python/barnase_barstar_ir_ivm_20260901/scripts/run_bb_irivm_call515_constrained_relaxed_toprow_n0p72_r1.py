#!/usr/bin/env python3
"""Authorized, resumable 15-point n=0.72 A constrained-relaxation supplement.

This is a diagnostic continuation of the frozen call-515 15x9 pilot.  It uses
the identical s grid, fixed tau/nu projectors and FIRE parameters.  Each joint
energy/Force request and its resulting optimizer state is fsync'd before the
next request.  These calls contribute zero to optimization N_F.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np

SCIENCE_ROOT = Path("/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901")
SCIENCE_SCRIPTS = SCIENCE_ROOT / "scripts"
if str(SCIENCE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCIENCE_SCRIPTS))
import run_bb_irivm_call515_constrained_relaxed_pilot_r1 as base

THIS = Path(__file__).resolve()
ROOT = SCIENCE_ROOT
WORK_ROOT = THIS.parent.parent
REG = WORK_ROOT / "registrations/diagnostics"
OUT = WORK_ROOT / "runs/diagnostics"
SOURCE = base.COORDINATE
SOURCE_AUDIT = base.COORDINATE_AUDIT
SOURCE_SHA = base.COORDINATE_SHA256
SOURCE_AUDIT_SHA = base.COORDINATE_AUDIT_SHA256
CONTRACT = REG / "BB_IRIVM_CALL515_CONSTRAINED_RELAXED_TOPROW_N0P72_CONTRACT_R1_20260903.json"
PREFLIGHT = REG / "BB_IRIVM_CALL515_CONSTRAINED_RELAXED_TOPROW_N0P72_PREFLIGHT_R1_20260903.json"
AUTH = REG / "BB_IRIVM_CALL515_CONSTRAINED_RELAXED_TOPROW_N0P72_AUTHORIZATION_R1_20260903.json"
RUN_DIR = OUT / "bb_irivm_call515_nf14983_constrained_relaxed_toprow_n0p72_15s_20260903_r1"
ATTEMPTS = RUN_DIR / "attempts"
WAL = RUN_DIR / "force.wal.jsonl"
RESULT = RUN_DIR / "result.npz"
AUDIT = RUN_DIR / "audit.json"

N_S = 15
N_ATOMS = base.N_ATOMS
N_VALUE = 0.72
SOURCE_TOP_N = 0.69
MAX_UPDATES = base.MAX_UPDATE_STEPS
MAX_CALLS_PER_POINT = MAX_UPDATES + 1
MAX_CALLS = N_S * MAX_CALLS_PER_POINT
MIN_FREE = 2 * 1024**3
RUNTIME_RESERVE = 1024**3
SCHEMA = "bb-irivm-call515-constrained-relaxed-toprow-n0p72-r1-v1"


def configure_work_root(path: Path) -> None:
    """Select the staging-owned registration/output root before any action."""
    global WORK_ROOT, REG, OUT, CONTRACT, PREFLIGHT, AUTH, RUN_DIR, ATTEMPTS, WAL, RESULT, AUDIT
    WORK_ROOT = Path(path).expanduser().resolve()
    REG = WORK_ROOT / "registrations/diagnostics"
    OUT = WORK_ROOT / "runs/diagnostics"
    CONTRACT = REG / "BB_IRIVM_CALL515_CONSTRAINED_RELAXED_TOPROW_N0P72_CONTRACT_R1_20260903.json"
    PREFLIGHT = REG / "BB_IRIVM_CALL515_CONSTRAINED_RELAXED_TOPROW_N0P72_PREFLIGHT_R1_20260903.json"
    AUTH = REG / "BB_IRIVM_CALL515_CONSTRAINED_RELAXED_TOPROW_N0P72_AUTHORIZATION_R1_20260903.json"
    RUN_DIR = OUT / "bb_irivm_call515_nf14983_constrained_relaxed_toprow_n0p72_15s_20260903_r1"
    ATTEMPTS = RUN_DIR / "attempts"
    WAL = RUN_DIR / "force.wal.jsonl"
    RESULT = RUN_DIR / "result.npz"
    AUDIT = RUN_DIR / "audit.json"


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bind(path: Path) -> dict[str, Any]:
    p = path.resolve(strict=True)
    return {"path": str(p), "sha256": sha(p), "bytes": p.stat().st_size}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with tmp.open("x", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        os.link(tmp, path); base.fsync_parent(path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npz(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    try:
        np.savez(tmp, **payload)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.link(tmp, path); base.fsync_parent(path)
    finally:
        tmp.unlink(missing_ok=True)


def source_data() -> dict[str, np.ndarray]:
    require(sha(SOURCE) == SOURCE_SHA and sha(SOURCE_AUDIT) == SOURCE_AUDIT_SHA,
            "frozen 15x9 source binding mismatch")
    data = base.validate_coordinates()
    require(data["coordinates"].shape == (15, 9, N_ATOMS, 3), "source shape mismatch")
    require(np.array_equal(data["n"], np.asarray([-0.12, -0.06, 0., 0.12, 0.24,
                                                   0.36, 0.48, 0.60, 0.69])),
            "source n grid mismatch")
    origin = np.ascontiguousarray(
        data["coordinates"][:, 8] + (N_VALUE - SOURCE_TOP_N) * np.sqrt(N_ATOMS)
        * data["normals"], dtype=np.float64)
    return {**data, "origin": origin}


def contract_payload() -> dict[str, Any]:
    data = source_data()
    return {
        "schema": SCHEMA + "-contract", "status": "FROZEN_BEFORE_PHYSICS",
        "created_utc": now(), "runner": bind(THIS), "frozen_base_runner": bind(Path(base.__file__)),
        "science_root": str(SCIENCE_ROOT), "work_root": str(WORK_ROOT),
        "coordinate_manifest": bind(SOURCE), "coordinate_audit": bind(SOURCE_AUDIT),
        "center_state": {"method": "IVM", "call_per_image": 515,
                         "optimization_N_F": 14983,
                         "state_role": "terminal_uncommitted_charged_tail",
                         "formally_reportable": False},
        "grid": {"s_values": data["s"].tolist(), "n_value_rms_A": N_VALUE,
                 "points": N_S,
                 "construction": "x(s,0.69)+(0.72-0.69)*sqrt(3132)*fixed_nu(s)"},
        "constraint": {"fixed_tau_nu": True,
                       "projector": "P(v)=v-tau*(tau dot v)-nu*(nu dot v)"},
        "optimizer": {"name": "projected_FIRE_with_globally_scaled_per_atom_displacement_cap",
                      "fmax_eV_per_A": base.FMAX_EV_A,
                      "maximum_update_steps": MAX_UPDATES,
                      "maximum_force_calls_per_point": MAX_CALLS_PER_POINT,
                      "dt0": base.DT0, "dtmax": base.DTMAX, "finc": base.FINC,
                      "fdec": base.FDEC, "alpha0": base.ALPHA0,
                      "falpha": base.FALPHA, "Nmin": base.NMIN,
                      "maximum_per_atom_step_A": base.MAX_ATOM_STEP_A},
        "physical_calls": {"kind": "joint energy+Force", "maximum": MAX_CALLS,
                           "energy_only": 0, "optimization_N_F_contribution": 0},
        "durability": {"wal": str(WAL), "attempts": str(ATTEMPTS),
                       "attempt_fields": ["coordinate_A", "energy_eV", "raw_force_eV_A",
                                          "projected_force_eV_A", "next_coordinate_A",
                                          "fire_velocity_out", "fire_dt_out", "fire_alpha_out",
                                          "fire_n_positive_out"]},
        "host_role": "local", "jobs": 1,
        "host_local_fcntl_locks": [str(p) for p in base.LOCKS],
        "thread_policy": {k: "1" for k in base.THREAD_VARS},
        "outputs": {"run_dir": str(RUN_DIR), "result": str(RESULT), "audit": str(AUDIT)},
    }


def load_contract() -> dict[str, Any]:
    payload = json.loads(CONTRACT.resolve(strict=True).read_text())
    require(payload.get("schema") == SCHEMA + "-contract"
            and payload.get("status") == "FROZEN_BEFORE_PHYSICS"
            and payload.get("runner") == bind(THIS)
            and payload.get("frozen_base_runner") == bind(Path(base.__file__))
            and payload.get("science_root") == str(SCIENCE_ROOT)
            and payload.get("work_root") == str(WORK_ROOT)
            and payload.get("coordinate_manifest") == bind(SOURCE)
            and payload["physical_calls"]["maximum"] == MAX_CALLS,
            "contract binding mismatch")
    return payload


def prepare() -> None:
    require(not any(p.exists() for p in (CONTRACT, PREFLIGHT, AUTH, RUN_DIR)),
            "supplement chain already exists")
    atomic_json(CONTRACT, contract_payload())
    print(json.dumps({"status": "FROZEN_BEFORE_PHYSICS", "contract": bind(CONTRACT)}, indent=2))


def preflight() -> None:
    load_contract(); base.check_thread_caps(); source_data()
    require(not PREFLIGHT.exists() and not AUTH.exists() and not RUN_DIR.exists(),
            "preflight chain state mismatch")
    with base.acquire_locks() as locks:
        pass
    free = shutil.disk_usage(OUT).free
    require(free >= MIN_FREE, "insufficient disk")
    payload = {"schema": SCHEMA + "-preflight", "status": "PASS_ZERO_PHYSICS",
               "created_utc": now(), "contract": bind(CONTRACT), "runner": bind(THIS),
               "two_lock_probe": {"status": "PASS", "locks": locks},
               "thread_policy": {k: os.environ[k] for k in base.THREAD_VARS},
               "free_bytes": free, "maximum_force_calls": MAX_CALLS,
               "openmm_contexts_constructed": 0, "physical_force_calls_started": 0,
               "optimization_N_F_contribution": 0}
    atomic_json(PREFLIGHT, payload); print(json.dumps(payload, indent=2))


def authorize() -> None:
    load_contract(); base.check_thread_caps()
    require(PREFLIGHT.exists() and not AUTH.exists() and not RUN_DIR.exists(),
            "authorization chain state mismatch")
    pre = json.loads(PREFLIGHT.read_text())
    require(pre.get("status") == "PASS_ZERO_PHYSICS" and pre.get("contract") == bind(CONTRACT)
            and pre.get("runner") == bind(THIS) and pre.get("physical_force_calls_started") == 0,
            "preflight binding mismatch")
    with base.acquire_locks() as locks:
        pass
    payload = {"schema": SCHEMA + "-authorization", "status": "AUTHORIZED",
               "enabled": True, "created_utc": now(), "contract": bind(CONTRACT),
               "preflight": bind(PREFLIGHT), "runner": bind(THIS),
               "maximum_force_calls": MAX_CALLS, "jobs": 1,
               "two_lock_probe": {"status": "PASS", "locks": locks},
               "optimization_N_F_contribution": 0}
    atomic_json(AUTH, payload); print(json.dumps(payload, indent=2))


class Wal:
    def __init__(self, header: Mapping[str, Any]):
        WAL.parent.mkdir(parents=True, exist_ok=True)
        if WAL.exists():
            lines = WAL.read_text().splitlines()
            require(lines and json.loads(lines[0]) == dict(header), "WAL header mismatch")
            self.f = WAL.open("a", buffering=1)
        else:
            self.f = WAL.open("x", buffering=1); self.append(header); base.fsync_parent(WAL)
    def append(self, item: Mapping[str, Any]) -> None:
        self.f.write(json.dumps(item, sort_keys=True, allow_nan=False) + "\n")
        self.f.flush(); os.fsync(self.f.fileno())
    def close(self) -> None: self.f.close()


def read_wal(header: Mapping[str, Any]) -> tuple[dict[str, dict], dict[str, dict], dict[int, dict], list[dict]]:
    if not WAL.exists(): return {}, {}, {}, []
    lines = WAL.read_text().splitlines(); require(json.loads(lines[0]) == dict(header), "WAL drift")
    starts: dict[str, dict] = {}; done: dict[str, dict] = {}; points: dict[int, dict] = {}; published = []
    for line in lines[1:]:
        x = json.loads(line); kind = x.get("event")
        if kind == "started": require(x["id"] not in starts, "duplicate start"); starts[x["id"]] = x
        elif kind == "succeeded":
            require(x["id"] in starts and x["id"] not in done, "orphan/duplicate success")
            p = Path(x["shard"]["path"]); require(bind(p) == x["shard"], "shard binding mismatch")
            done[x["id"]] = x
        elif kind == "failed": raise RuntimeError("failed physical attempt requires new authorization")
        elif kind == "point_complete": require(x["point"] not in points, "duplicate completion"); points[x["point"]] = x
        elif kind == "published": published.append(x)
        else: raise RuntimeError(f"unknown WAL event {kind}")
    require(set(starts) == set(done), "open physical attempt requires new authorization")
    require(len(starts) <= MAX_CALLS, "force budget exceeded")
    require(len(published) <= 1, "duplicate result publication")
    if published:
        require(bind(Path(published[0]["result"]["path"])) == published[0]["result"],
                "published result binding mismatch")
    return starts, done, points, published


def shard_path(i: int, step: int) -> Path:
    return ATTEMPTS / f"s{i:02d}" / f"step_{step:03d}.npz"


def load_shard(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        x = {k: np.asarray(z[k]) for k in z.files}
    require(str(x["schema"].item()) == SCHEMA + "-attempt", "attempt schema mismatch")
    for key in ("coordinate_A", "raw_force_eV_A", "projected_force_eV_A", "next_coordinate_A"):
        require(base.sha256_array(x[key]) == str(x[key + "_sha256"].item()), f"{key} hash mismatch")
    return x


def run() -> None:
    load_contract(); base.check_thread_caps()
    auth = json.loads(AUTH.resolve(strict=True).read_text())
    require(auth.get("enabled") is True and auth.get("status") == "AUTHORIZED"
            and auth.get("contract") == bind(CONTRACT) and auth.get("preflight") == bind(PREFLIGHT)
            and auth.get("runner") == bind(THIS) and auth.get("maximum_force_calls") == MAX_CALLS,
            "authorization mismatch")
    data = source_data()
    header = {"schema": SCHEMA + "-wal", "status": "OPEN_APPEND_ONLY_FSYNC",
              "contract": bind(CONTRACT), "authorization": bind(AUTH),
              "maximum_force_calls": MAX_CALLS, "optimization_N_F_contribution": 0}
    require(shutil.disk_usage(OUT).free >= MIN_FREE or RUN_DIR.exists(), "insufficient disk")
    with base.acquire_locks() as held:
        wal = Wal(header); backend = None
        try:
            starts, done, completed, published = read_wal(header)
            if AUDIT.exists(): print(AUDIT.read_text()); return
            for i in range(N_S):
                if i in completed: continue
                events = sorted((v for v in done.values() if v["point"] == i), key=lambda x: x["step"])
                require([x["step"] for x in events] == list(range(len(events))), "noncontiguous history")
                origin = data["origin"][i]; tau = data["tangents"][i]; nu = data["normals"][i]
                if events:
                    last = load_shard(Path(events[-1]["shard"]["path"]))
                    if bool(last["terminal"].item()):
                        wal.append({"event": "point_complete", "utc": now(), "point": i,
                                    "reason": str(last["terminal_reason"].item()), "calls": len(events)})
                        completed[i] = {}; continue
                    position = last["next_coordinate_A"].copy(); velocity = last["fire_velocity_out"].copy()
                    dt = float(last["fire_dt_out"]); alpha = float(last["fire_alpha_out"])
                    npos = int(last["fire_n_positive_out"]); step = len(events)
                else:
                    position = origin.copy(); velocity = np.zeros_like(origin)
                    dt, alpha, npos, step = base.DT0, base.ALPHA0, 0, 0
                while True:
                    require(shutil.disk_usage(OUT).free >= RUNTIME_RESERVE, "runtime disk reserve reached")
                    require(step <= MAX_UPDATES and len(starts) < MAX_CALLS, "force budget exhausted")
                    ident = f"s{i:02d}:step{step:03d}"
                    start = {"event": "started", "utc": now(), "id": ident, "point": i,
                             "step": step, "s_value": float(data["s"][i]), "n_value_rms_A": N_VALUE,
                             "coordinate_sha256": base.sha256_array(position)}
                    if backend is None:
                        if str(base.SRC) not in sys.path: sys.path.insert(0, str(base.SRC))
                        from bb_core import OpenMMBackend
                        backend = OpenMMBackend(base.TOPOLOGY, platform_preference=("CPU",),
                                                precision="double", label="call515_toprow_n0p72")
                    wal.append(start); starts[ident] = start
                    try:
                        energy, raw = backend.evaluate_joint(position.reshape(-1))
                    except Exception as exc:
                        wal.append({"event": "failed", "utc": now(), "id": ident, "error": repr(exc)})
                        raise
                    raw = np.ascontiguousarray(raw.reshape(N_ATOMS, 3)); force = base.project(raw, tau, nu)
                    fmax = base.projected_fmax(force); terminal = fmax <= base.FMAX_EV_A or step == MAX_UPDATES
                    reason = "converged" if fmax <= base.FMAX_EV_A else ("maximum_update_steps" if terminal else "continue")
                    if terminal:
                        nxt, vel2, dt2, alpha2, npos2 = position.copy(), velocity.copy(), dt, alpha, npos
                    else:
                        nxt, vel2, dt2, alpha2, npos2 = base.fire_advance(
                            position, origin, force, tau, nu, velocity, dt, alpha, npos, step == 0)
                    payload: dict[str, np.ndarray] = {
                        "schema": np.asarray(SCHEMA + "-attempt"), "status": np.asarray("PASS"),
                        "id": np.asarray(ident), "point": np.asarray(i), "step": np.asarray(step),
                        "s_value": np.asarray(data["s"][i]), "n_value_rms_A": np.asarray(N_VALUE),
                        "coordinate_A": position, "energy_eV": np.asarray(energy),
                        "raw_force_eV_A": raw, "projected_force_eV_A": force,
                        "next_coordinate_A": nxt, "fire_velocity_out": vel2,
                        "fire_dt_out": np.asarray(dt2), "fire_alpha_out": np.asarray(alpha2),
                        "fire_n_positive_out": np.asarray(npos2), "projected_fmax_eV_A": np.asarray(fmax),
                        "terminal": np.asarray(terminal), "terminal_reason": np.asarray(reason),
                        "optimization_N_F_contribution": np.asarray(0),
                    }
                    for key in ("coordinate_A", "raw_force_eV_A", "projected_force_eV_A", "next_coordinate_A"):
                        payload[key + "_sha256"] = np.asarray(base.sha256_array(payload[key]))
                    path = shard_path(i, step); atomic_npz(path, payload)
                    success = {"event": "succeeded", "utc": now(), "id": ident, "point": i,
                               "step": step, "energy_eV": float(energy), "projected_fmax_eV_A": fmax,
                               "next_coordinate_sha256": base.sha256_array(nxt), "shard": bind(path),
                               "terminal": terminal, "terminal_reason": reason}
                    wal.append(success); done[ident] = success
                    if terminal:
                        wal.append({"event": "point_complete", "utc": now(), "point": i,
                                    "reason": reason, "calls": step + 1}); completed[i] = {}; break
                    position, velocity, dt, alpha, npos, step = nxt, vel2, dt2, alpha2, npos2, step + 1
            starts, done, completed, published = read_wal(header)
            require(len(completed) == N_S, "not all points complete")
            finals = [load_shard(Path(sorted((v for v in done.values() if v["point"] == i),
                                              key=lambda x: x["step"])[-1]["shard"]["path"]))
                      for i in range(N_S)]
            result = {"schema": np.asarray(SCHEMA + "-result"), "status": np.asarray("PASS"),
                      "s_values": data["s"], "n_value_rms_A": np.asarray(N_VALUE),
                      "initial_coordinates_A": data["origin"],
                      "final_coordinates_A": np.stack([x["coordinate_A"] for x in finals]),
                      "final_energy_eV": np.asarray([x["energy_eV"] for x in finals]),
                      "final_raw_force_eV_A": np.stack([x["raw_force_eV_A"] for x in finals]),
                      "final_projected_force_eV_A": np.stack([x["projected_force_eV_A"] for x in finals]),
                      "physical_force_calls_total": np.asarray(len(starts)),
                      "optimization_N_F_contribution": np.asarray(0),
                      "held_host_locks": np.asarray(held)}
            if RESULT.exists():
                base.validate_npz_payload(RESULT, result)
            else:
                atomic_npz(RESULT, result)
            if not published:
                wal.append({"event": "published", "utc": now(), "result": bind(RESULT)})
            audit = {"schema": SCHEMA + "-audit", "status": "PASS", "completed_utc": now(),
                     "contract": bind(CONTRACT), "preflight": bind(PREFLIGHT), "authorization": bind(AUTH),
                     "result": bind(RESULT), "wal": bind(WAL), "points": N_S,
                     "physical_force_calls_total": len(starts), "maximum_force_calls": MAX_CALLS,
                     "optimization_N_F_contribution": 0, "held_host_locks": held}
            atomic_json(AUDIT, audit); print(json.dumps(audit, indent=2))
        finally:
            wal.close()


def self_test() -> None:
    require(MAX_UPDATES == 50 and MAX_CALLS == 765 and base.FMAX_EV_A == 0.10,
            "registered constants drift")
    require(SCIENCE_ROOT == Path("/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901")
            and len(base.LOCKS) == 2, "science-root/two-lock policy drift")
    require((base.DT0, base.DTMAX, base.FINC, base.FDEC, base.ALPHA0, base.FALPHA,
             base.NMIN, base.MAX_ATOM_STEP_A) == (0.10, 1.00, 1.10, 0.50, 0.10, 0.99, 5, 0.10),
            "FIRE constants drift")
    rng = np.random.default_rng(72); tau = rng.normal(size=(N_ATOMS, 3)); tau /= np.linalg.norm(tau)
    nu = base.project(rng.normal(size=(N_ATOMS, 3)), tau, np.zeros_like(tau)); nu /= np.linalg.norm(nu)
    p = base.project(rng.normal(size=(N_ATOMS, 3)), tau, nu)
    require(abs(float(np.vdot(p, tau))) < 1e-10 and abs(float(np.vdot(p, nu))) < 1e-10,
            "projector self-test failed")
    data = source_data()
    delta = data["origin"] - data["coordinates"][:, 8]
    observed_rms = np.sqrt(np.mean(np.sum(delta * delta, axis=2), axis=1))
    require(np.allclose(observed_rms, 0.03, atol=2e-15, rtol=0.0),
            "n=0.72 top-row construction self-test failed")
    print(json.dumps({"status": "PASS_ZERO_PHYSICS_SELF_TEST", "points": 15,
                      "n_value_rms_A": 0.72, "maximum_force_calls": 765,
                      "openmm_contexts_constructed": 0, "physical_force_calls_started": 0,
                      "optimization_N_F_contribution": 0}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=THIS.parent.parent,
                        help="staging root that owns registrations/ and runs/")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-contract", action="store_true")
    modes.add_argument("--preflight", action="store_true")
    modes.add_argument("--authorize", action="store_true")
    modes.add_argument("--run", action="store_true")
    modes.add_argument("--self-test-zero-physics", action="store_true")
    args = parser.parse_args()
    configure_work_root(args.work_root)
    if args.prepare_contract: prepare()
    elif args.preflight: preflight()
    elif args.authorize: authorize()
    elif args.run: run()
    else: self_test()


if __name__ == "__main__":
    main()
