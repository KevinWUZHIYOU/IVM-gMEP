"""Registered, serial image-chain gMAM / two-stage StringNET experiments."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
LEPS = ROOT.parent
sys.path.insert(0, str(LEPS / 'src'))
from leps_core import leps_energy_gradient, physical_to_flat, flat_to_physical
from run_string_baseline import cubic_arclength_reparameterize, lbfgs_direction
from torch_leps import torch_leps_energy

NATIVE = LEPS / 'leps_irivm_ala_chig_lepsnative_20260829/src/leps_irivm_lepsnative.py'
spec = importlib.util.spec_from_file_location('_leps_action_native', NATIVE)
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
BUDGET = 20000
REG = ROOT / 'registration.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def configs():
    result = []
    for method, beta in [('gMAM', None)] + [('StringNET', b) for b in [0.1, 0.3, 1., 3.]]:
        for optimizer, rates in [('LBFGS', [0.01, 0.1, 1.]), ('Adam', [0.001, 0.01, 0.1])]:
            for rate in rates:
                for reparam in ['string', 'ivm_spring']:
                    c = dict(method=method, beta=beta, optimizer=optimizer, rate=rate,
                             reparam=reparam, budget=BUDGET, warmup=50 if beta else 0,
                             accounting='force_plus_1_per_hvp_v3')
                    c['id'] = hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]
                    result.append(c)
    return result


def primitive(x, hvp):
    t = torch.tensor(x, requires_grad=True)
    e = torch_leps_energy(t)
    g = torch.autograd.grad(e.sum(), t, create_graph=hvp)[0]
    h = torch.autograd.grad(torch.linalg.vector_norm(g, dim=1).sum(), t)[0] if hvp else None
    return e.detach().numpy(), g.detach().numpy(), None if h is None else h.detach().numpy()


class BudgetReached(Exception):
    pass


class Oracle:
    def __init__(self, limit):
        self.limit, self.nof, self.hvp, self.force_calls = limit, 0, 0, 0
        self.cache = [{} for _ in range(22)]
        self.events = []
        self.phase = 'initial'
        self.outer = 0

    def query(self, x, need_hvp=False):
        x = np.asarray(x, dtype=float)
        fresh = [i for i in range(22) if x[i].tobytes() not in self.cache[i]]
        missing_h = [i for i in range(22) if need_hvp and i not in fresh
                     and self.cache[i][x[i].tobytes()][2] is None]
        force_charge = sum(0 < i < 21 for i in fresh)
        hvp_charge = sum(0 < i < 21 for i in fresh + missing_h) if need_hvp else 0
        charge = force_charge + hvp_charge
        if self.nof + charge > self.limit:
            raise BudgetReached
        if fresh:
            e, g, h = primitive(x[fresh], need_hvp)
            for j, i in enumerate(fresh):
                self.cache[i][x[i].tobytes()] = (e[j], g[j], h[j] if need_hvp else None)
        if missing_h:
            _, _, h = primitive(x[missing_h], True)
            for j, i in enumerate(missing_h):
                key = x[i].tobytes()
                e, g, _ = self.cache[i][key]
                self.cache[i][key] = (e, g, h[j])
        self.force_calls += force_charge
        self.hvp += hvp_charge
        self.nof += charge
        if fresh or missing_h:
            self.events.append(dict(outer=self.outer, phase=self.phase, nof=charge,
                                    force_calls=force_charge, hvp=hvp_charge,
                                    total_nof=self.nof, total_hvp=self.hvp,
                                    total_force_calls=self.force_calls))
        vals = [self.cache[i][x[i].tobytes()] for i in range(22)]
        return (np.array([v[0] for v in vals]), np.array([v[1] for v in vals]),
                np.array([v[2] for v in vals]) if need_hvp else None)


def objective(x, oracle, beta=None):
    e, g, h = oracle.query(x, beta is None)
    diff = np.diff(x, axis=0)
    lengths = np.linalg.norm(diff, axis=1)
    if np.any(lengths <= 1e-15):
        raise FloatingPointError('collapsed adjacent path nodes')
    w = np.zeros(22)
    w[:-1] += 0.5 * lengths
    w[1:] += 0.5 * lengths
    if beta is None:
        a = np.linalg.norm(g, axis=1)
        value = np.dot(w, a)
        derivative = w[:, None] * h
        length_coefficient = 0.5 * (a[:-1] + a[1:])
    else:
        anchor = np.max(e)
        a = np.exp(beta * (e - anchor))
        z = np.dot(w, a)
        value = anchor + np.log(z) / beta
        derivative = (w * a / z)[:, None] * g
        length_coefficient = 0.5 * (a[:-1] + a[1:]) / (beta * z)
    edge = length_coefficient[:, None] * diff / lengths[:, None]
    derivative[:-1] -= edge
    derivative[1:] += edge
    if not np.isfinite(value) or not np.all(np.isfinite(derivative)):
        raise FloatingPointError('nonfinite objective/gradient')
    return float(value), derivative[1:-1].ravel(), e, g


def reparameterize(x, e, mode):
    if mode == 'string':
        return flat_to_physical(cubic_arclength_reparameterize(physical_to_flat(x)))
    y, _ = native.improved_spring_reparameterize(physical_to_flat(x), e)
    return flat_to_physical(y)


def run(c):
    oracle = Oracle(BUDGET)
    with np.load(LEPS / 'inputs/common_initial_path.npz') as z:
        x = flat_to_physical(z['initial_path'])
    history, costs, hvps, steps, phases, energies, gradients = [], [], [], [], [], [], []
    objectives, objective_gradnorm = [], []

    def record(step, phase, value, grad, e, g):
        history.append(x.copy()); costs.append(oracle.nof); hvps.append(oracle.hvp)
        steps.append(step); phases.append(phase); energies.append(e.copy()); gradients.append(g.copy())
        objectives.append(value); objective_gradnorm.append(float(np.linalg.norm(grad)))

    beta = c['beta'] if c['warmup'] else None
    value, grad, e, g = objective(x, oracle, beta)
    record(0, 'initial', value, grad, e, g)
    pairs, previous_x, previous_grad = [], None, None
    moment, variance, adam_step = np.zeros(40), np.zeros(40), 0
    stop, failure, entered_gmam = 'outer_limit', '', beta is None
    started = time.time()
    for step in range(1, 5001):
        oracle.outer = step
        try:
            if step == c['warmup'] + 1 and c['warmup']:
                beta = None
                pairs, previous_x, previous_grad = [], None, None
                moment[:], variance[:], adam_step = 0, 0, 0
                oracle.phase = 'objective_switch'
                value, grad, e, g = objective(x, oracle, beta)
                entered_gmam = True
            if previous_x is not None and c['optimizer'] == 'LBFGS':
                s = x[1:-1].ravel() - previous_x
                y = grad - previous_grad
                if np.dot(s, y) > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
                    pairs.append((s, y)); pairs = pairs[-10:]
            old_x = x.copy()
            previous_x, previous_grad = x[1:-1].ravel().copy(), grad.copy()
            if c['optimizer'] == 'LBFGS':
                d = lbfgs_direction(grad, pairs, initial_scale=c['rate'])
            else:
                adam_step += 1
                moment = 0.9 * moment + 0.1 * grad
                variance = 0.999 * variance + 0.001 * grad * grad
                d = -c['rate'] * (moment / (1 - 0.9**adam_step)) / (np.sqrt(variance / (1 - 0.999**adam_step)) + 1e-8)
            cap = np.max(np.linalg.norm(d.reshape(20, 2), axis=1))
            d *= min(1., 0.2 / max(cap, 1e-300))
            success = False
            for backtrack in range(24 if c['optimizer'] == 'LBFGS' else 1):
                trial = old_x.copy()
                trial[1:-1] += (0.5**backtrack * d).reshape(20, 2)
                oracle.phase = 'optimizer_trial'
                tv, tg, te, tgr = objective(trial, oracle, beta)
                if c['optimizer'] == 'Adam' or tv <= value + 1e-4 * 0.5**backtrack * np.dot(grad, d) + 1e-14:
                    success = True
                    break
            if not success:
                stop = 'line_search_stalled'
                break
            # The full optimizer update is reparameterized unconditionally.
            projected = reparameterize(trial, te, c['reparam'])
            oracle.phase = 'reparameterized_state'
            nv, ng, ne, ngr = objective(projected, oracle, beta)
            x, value, grad, e, g = projected, nv, ng, ne, ngr
            record(step, 'warmup' if beta is not None else 'gmam', value, grad, e, g)
            if np.max(np.linalg.norm(x - old_x, axis=1)) <= 1e-14 and entered_gmam:
                stop = 'stationary'
                break
        except BudgetReached:
            stop = 'budget'
            break
        except (FloatingPointError, ValueError, np.linalg.LinAlgError) as exc:
            stop, failure = 'numerical_failure', str(exc)
            break
    payload = dict(path_history=np.asarray(history), energy_history=np.asarray(energies),
                   gradient_history=np.asarray(gradients), article_nf=np.asarray(costs),
                   hvp_history=np.asarray(hvps), outer=np.asarray(steps), state_phase=np.asarray(phases),
                   objective=np.asarray(objectives), objective_gradient_norm=np.asarray(objective_gradnorm),
                   terminal_nof=oracle.nof, terminal_hvp=oracle.hvp,
                   terminal_force_calls=oracle.force_calls, config_json=json.dumps(c),
                   stop_reason=stop, failure_message=failure, entered_gmam=entered_gmam,
                   events_json=json.dumps(oracle.events), elapsed_seconds=time.time()-started,
                   registration_sha256=sha(REG), source_sha256=sha(__file__))
    assert sum(ev['nof'] for ev in oracle.events) == oracle.nof
    assert sum(ev['hvp'] for ev in oracle.events) == oracle.hvp
    assert oracle.nof == oracle.force_calls + oracle.hvp
    target = ROOT / 'runs' / (c['id'] + '.npz')
    target.parent.mkdir(exist_ok=True)
    np.savez_compressed(target, **payload)
    return dict(config=c, stop=stop, nof=oracle.nof, hvp=oracle.hvp,
                force_calls=oracle.force_calls, steps=steps[-1],
                entered_gmam=entered_gmam, failure=failure, seconds=time.time()-started)


def validate():
    with np.load(LEPS / 'inputs/common_initial_path.npz') as z:
        x = flat_to_physical(z['initial_path']).copy()
    x[1:-1] += np.random.default_rng(123).normal(size=(20,2)) * 0.03
    e, g, _ = primitive(x, True)
    er, gr = leps_energy_gradient(x[:,0], x[:,1])
    np.testing.assert_allclose(e, er, atol=2e-12, rtol=1e-12)
    np.testing.assert_allclose(g, gr, atol=2e-11, rtol=1e-11)
    reports = []
    for beta in [None, .1, .3, 1., 3.]:
        value, grad, _, _ = objective(x, Oracle(100000), beta)
        finite = []
        for i in range(40):
            plus, minus = x.copy(), x.copy()
            plus[1+i//2,i%2] += 1e-6
            minus[1+i//2,i%2] -= 1e-6
            fp = objective(plus, Oracle(100000), beta)[0]
            fm = objective(minus, Oracle(100000), beta)[0]
            finite.append((fp-fm)/2e-6)
        np.testing.assert_allclose(grad, finite, atol=2e-7, rtol=2e-5)
        reports.append(dict(beta=beta, max_gradient_error=float(np.max(np.abs(grad-finite)))))
    accounting = Oracle(100000)
    accounting.query(x)
    assert (accounting.nof, accounting.force_calls, accounting.hvp) == (20, 20, 0)
    accounting.query(x, True)
    assert (accounting.nof, accounting.force_calls, accounting.hvp) == (40, 20, 20)
    accounting.query(x, True)
    assert accounting.nof == 40
    fresh_accounting = Oracle(40)
    fresh_accounting.query(x, True)
    assert fresh_accounting.nof == 40
    limited = Oracle(39)
    try:
        limited.query(x, True)
    except BudgetReached:
        pass
    else:
        raise AssertionError('HVP budget gate did not block full batch')
    assert limited.nof == 0 and limited.hvp == 0 and limited.force_calls == 0
    write_json(ROOT / 'validation.json', dict(passed=True, derivatives=reports,
                                             accounting_checks_passed=True,
                                             source_sha256=sha(__file__)))


def registration_payload():
    dependencies = [Path(__file__).resolve(), ROOT/'PROTOCOL.md', LEPS/'src/torch_leps.py',
                    LEPS/'src/leps_core.py', LEPS/'src/run_string_baseline.py', NATIVE,
                    LEPS/'inputs/common_initial_path.npz']
    return dict(configs=configs(), registered_before_runs=True,
                inputs={str(p.relative_to(LEPS)):sha(p) for p in dependencies},
                budget=BUDGET, selection='mean normalized log I_perp + log absolute barrier error at 1000:1000:20000',
                derivative_validation=sha(ROOT/'validation.json'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['validate', 'register', 'run'])
    args = parser.parse_args()
    if args.mode == 'validate':
        validate(); print('derivative and potential checks passed', flush=True); return
    payload = registration_payload()
    if args.mode == 'register':
        if REG.exists():
            assert json.loads(REG.read_text()) == payload
        else:
            write_json(REG, payload)
        print('registered', len(configs()), 'combinations', flush=True); return
    assert json.loads(REG.read_text()) == payload
    with (LEPS/'leps_irivm_sync1_refined_grid_20260830/locks/host-local-leps-grid.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = []
        for index, c in enumerate(configs()):
            destination = ROOT/'runs'/(c['id']+'.npz')
            if destination.exists():
                with np.load(destination) as z:
                    assert z['source_sha256'].item() == sha(__file__)
                    result = dict(config=c, nof=int(z['terminal_nof']), stop=z['stop_reason'].item(),
                                  hvp=int(z['terminal_hvp']), force_calls=int(z['terminal_force_calls']),
                                  steps=int(z['outer'][-1]),
                                  entered_gmam=bool(z['entered_gmam']), failure=z['failure_message'].item())
            else:
                print('START', index+1, c, flush=True)
                result = run(c)
            status.append(result)
            write_json(ROOT/'status.json', dict(completed=len(status), total=len(configs()), runs=status))
            print('DONE', index+1, result, flush=True)


if __name__ == '__main__':
    main()
