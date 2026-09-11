#!/usr/bin/env python3
"""Offline, 22-node full-gradient action; optimizer files are read-only."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

for key in ('OPENMM_CPU_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[key] = '1'

import numpy as np
from chignolin_core import (OpenMMBackend, EV_TO_KCAL, TANGENT_DEFINITION,
                           central_tangents, local_path_weights, path_diagnostics)
import plot_chignolin_ivm_arc500 as reference

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / 'registrations/GEOMETRIC_ACTION_DISCRETE_V1.json'
OUT = ROOT / 'runs/diagnostics/chignolin_geometric_action_discrete_v1.npz'
SOURCE_CONTRACT = ROOT / 'registrations/ARC_LENGTH_500_IVM_REPLACEMENT_CONTRACT.json'
EXPECTED_SOURCE_CONTRACT_SHA = 'f351020439a1ef7d75bd0c21ae6d843722d813912542bf5cef984f3f0ab424b8'


def sha(path):
    return reference.sha256_file(Path(path))


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def discrete_action(path, forces):
    """Full-gradient trapezoid integral on the original polygon, in eV.

    Tangents are validated using the production central difference but are
    NOT used to replace the full gradient by its normal projection.
    Interior quadrature weights equal production local_path_weights exactly.
    """
    path = np.asarray(path, dtype=float)
    forces = np.asarray(forces, dtype=float)
    if path.shape != forces.shape or path.ndim != 2 or len(path) < 3:
        raise ValueError('Path/force shapes must match with at least three nodes')
    if not np.all(np.isfinite(path)) or not np.all(np.isfinite(forces)):
        raise ValueError('Non-finite geometry or force')
    central_tangents(path)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    if np.any(lengths <= 0):
        raise ValueError('Zero length segment')
    norms = np.linalg.norm(forces, axis=1)
    interior = float(np.dot(local_path_weights(path), norms[1:-1]))
    endpoint = float(0.5 * (lengths[0] * norms[0] + lengths[-1] * norms[-1]))
    segment_sum = float(np.dot(lengths, 0.5 * (norms[:-1] + norms[1:])))
    np.testing.assert_allclose(segment_sum, interior + endpoint, rtol=5e-15)
    return segment_sum, endpoint, norms, lengths


def selected_rows():
    if sha(SOURCE_CONTRACT) != EXPECTED_SOURCE_CONTRACT_SHA:
        raise ValueError('Reference source contract changed')
    contract_path, contract = reference.load_contract(SOURCE_CONTRACT)
    histories = reference.load_raw_histories(contract_path, contract)
    rows = []
    for method, history in histories.items():
        with np.load(history['source_result'], allow_pickle=False) as saved:
            if str(saved['tangent_definition'].item()) != TANGENT_DEFINITION:
                raise ValueError('Source tangent mismatch')
            paths = saved['stored_paths']
            np.testing.assert_array_equal(saved['stored_path_calls'], saved['calls_per_image'])
            for index in history['source_record_index']:
                path = paths[index]
                if path.shape != (22, 498):
                    raise ValueError('Unexpected path shape')
                central_tangents(path)
                rows.append(dict(method=method, source_result=history['source_result'],
                                 source_result_sha256=history['source_result_sha256'],
                                 source_record_index=int(index), path=path,
                                 selected_path_sha256=array_sha(path),
                                 number_of_forces=int(saved['counter_vectors'][index].sum()),
                                 counter_vector=saved['counter_vectors'][index],
                                 i_perp=float(saved['i_perp'][index])))
    return contract, rows


def register():
    if CONTRACT.exists():
        raise FileExistsError(CONTRACT)
    original, rows = selected_rows()
    sources = [Path(__file__), ROOT / 'src/chignolin_core.py',
               ROOT / 'src/plot_chignolin_ivm_arc500.py']
    data = dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Additional discrete geometric-action diagnostic; no optimizer rerun',
        formula='sum_i 0.5*(norm(gradV_i)+norm(gradV_(i+1)))*norm(x_(i+1)-x_i)',
        gradient='full physical gradient; not normal force; no extra gradV dot tangent term',
        tangent=TANGENT_DEFINITION, quadrature='polygonal arc-length trapezoid',
        interior_weights='chignolin_core.local_path_weights', endpoint_policy='include half-segment endpoint terms',
        spatial_nodes=22, densification=False, n_atoms=166, coordinate_space='unweighted 498D Cartesian, Angstrom',
        output_unit='kcal/mol', ev_to_kcal=EV_TO_KCAL,
        source_contract=str(SOURCE_CONTRACT), source_contract_sha256=sha(SOURCE_CONTRACT),
        topology=original['topology'], common_initial_path_sha256=original['common_initial_path_sha256'],
        methods={m: dict(source_result=v['source_result'], source_result_sha256=v['source_result_sha256'],
                        eligibility='initial_or_reparam_committed' if m == 'IVM' else 'all_saved_states')
                 for m, v in original['methods'].items()},
        counts={m: sum(r['method'] == m for r in rows) for m in reference.METHOD_ORDER},
        selected_path_count=len(rows), expected_diagnostic_joint_evaluations=len(rows)*22,
        selected_paths_sha256=array_sha(np.asarray([r['selected_path_sha256'] for r in rows])),
        jobs=1, openmm_cpu_threads=1, host_lock=str(ROOT / 'logs/chignolin_ir_ivm_host.lock'),
        diagnostic_cost_excluded_from_optimization_axis=True,
        optimization_cost='counter_vectors.sum(axis=1), all 22 images',
        iperp_verification_rtol=1e-5, iperp_verification_atol=1e-8,
        threshold_policy='max(method minimum) over IVM, ODE-NEB, OBSM-FIRE; only ODE-NEB arrows',
        sources={str(p): sha(p) for p in sources})
    CONTRACT.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, indent=2), flush=True)


def compute():
    config = json.loads(CONTRACT.read_text())
    for p, h in config['sources'].items():
        if sha(p) != h:
            raise ValueError(f'Source drift: {p}')
    if OUT.exists() or OUT.with_suffix('.audit.json').exists():
        raise FileExistsError('Refusing to overwrite registered diagnostic')
    _, rows = selected_rows()
    if array_sha(np.asarray([r['selected_path_sha256'] for r in rows])) != config['selected_paths_sha256']:
        raise ValueError('Path selection drift')
    topology = Path(config['topology']['path'])
    if sha(topology) != config['topology']['sha256']:
        raise ValueError('Topology drift')
    lock = Path(config['host_lock']).open('a+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    backend = OpenMMBackend(topology, platform_preference=('CPU',), precision='double',
                            label='offline_discrete_geometric_action')
    for j, row in enumerate(rows):
        path = row['path']
        before = backend.raw_joint_evaluations
        evaluated = [backend.evaluate_joint(point) for point in path]
        energy = np.array([ef[0] for ef in evaluated])
        forces = np.array([ef[1] for ef in evaluated])
        if backend.raw_joint_evaluations - before != 22:
            raise RuntimeError('Diagnostic NoF discrepancy')
        action, endpoint, norms, lengths = discrete_action(path, forces)
        check = path_diagnostics(path, energy, forces[1:-1])['i_perp']
        if not np.isclose(check, row['i_perp'], rtol=config['iperp_verification_rtol'],
                          atol=config['iperp_verification_atol']):
            raise ValueError(f"Iperp drift: {row['method']} record {row['source_record_index']}: {check} vs {row['i_perp']}")
        row.update(action_kcal=action*EV_TO_KCAL, endpoint_term_kcal=endpoint*EV_TO_KCAL,
                   recomputed_i_perp=check, gradient_norm_ev_per_angstrom=norms,
                   segment_length_angstrom=lengths, energy_ev=energy,
                   diagnostic_joint_evaluations=22)
        if j % 100 == 0 or j + 1 == len(rows):
            print(f"{j+1}/{len(rows)} paths; diagnostic NoF={backend.raw_joint_evaluations}; "
                  f"{row['method']}, NF={row['number_of_forces']}, S={row['action_kcal']:.10g}", flush=True)
    if backend.raw_joint_evaluations != config['expected_diagnostic_joint_evaluations']:
        raise ValueError('Total diagnostic count mismatch')
    keys = ['method', 'source_result', 'source_result_sha256', 'source_record_index',
            'selected_path_sha256', 'number_of_forces', 'i_perp', 'recomputed_i_perp',
            'action_kcal', 'endpoint_term_kcal', 'diagnostic_joint_evaluations',
            'gradient_norm_ev_per_angstrom', 'segment_length_angstrom', 'energy_ev', 'counter_vector']
    np.savez_compressed(OUT, **{k: np.asarray([r[k] for r in rows]) for k in keys},
                        contract_sha256=np.asarray(sha(CONTRACT)))
    csv_keys = keys[:11]
    csv_path = OUT.with_suffix('.csv')
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_keys)
        writer.writeheader()
        writer.writerows({k: r[k] for k in csv_keys} for r in rows)
    report = dict(status='validated', contract=str(CONTRACT), contract_sha256=sha(CONTRACT),
                  result=str(OUT), result_sha256=sha(OUT), csv=str(csv_path), csv_sha256=sha(csv_path),
                  selected_path_count=len(rows), counts=config['counts'],
                  diagnostic_joint_evaluations=backend.raw_joint_evaluations,
                  diagnostic_cost_excluded_from_optimization_axis=True,
                  backend_platform=backend.platform_name, backend_requested_precision='double',
                  actual_precision_note='Same production CPU backend; precision keyword configures only GPU platforms in core',
                  openmm_cpu_threads=1,
                  maximum_iperp_absolute_difference=max(abs(r['recomputed_i_perp']-r['i_perp']) for r in rows),
                  maximum_iperp_relative_difference=max(abs(r['recomputed_i_perp']-r['i_perp'])/max(abs(r['i_perp']),1e-30) for r in rows))
    OUT.with_suffix('.audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--register', action='store_true')
    parser.add_argument('--compute', action='store_true')
    args = parser.parse_args()
    if args.register == args.compute:
        parser.error('Choose exactly one of --register or --compute')
    register() if args.register else compute()
