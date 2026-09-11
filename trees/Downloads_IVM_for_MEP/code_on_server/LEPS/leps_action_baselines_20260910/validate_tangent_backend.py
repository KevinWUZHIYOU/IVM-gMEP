"""Cross-check batched tangents and the unchanged complete spring flow against ASE."""
import json
import time
import numpy as np
import run_experiments as run
from tangent_backend import improved_tangents


def main():
    with np.load(run.LEPS/'inputs/common_initial_path.npz') as z:
        initial = z['initial_path']
    rng = np.random.default_rng(720)
    paths = [initial]
    for scale in [.001, .01, .1]:
        x = initial.copy()
        x[1:-1] += scale*rng.normal(size=(20,6))
        paths.append(x)
    for result in sorted((run.ROOT/'archive/pre_batched_tangent/runs').glob('*.npz')):
        with np.load(result) as z:
            if str(z['stop_reason']) != 'numerical_failure':
                paths.append(run.physical_to_flat(z['path_history'][-1]))
    original = run.ASE_TANGENTS
    errors, flow_errors = [], []
    started = time.time()
    for x in paths:
        e, _ = run.leps_energy_gradient(x[:,0],x[:,3])
        t1 = original(x,e)
        t2 = improved_tangents(x,e)
        np.testing.assert_allclose(t1,t2,atol=1e-14,rtol=1e-13)
        errors.append(float(np.max(abs(t1-t2))))
        run.native.base.improved_tangents = original
        y1, _ = run.native.improved_spring_reparameterize(x,e)
        run.native.base.improved_tangents = improved_tangents
        y2, _ = run.native.improved_spring_reparameterize(x,e)
        run.native.base.improved_tangents = original
        np.testing.assert_allclose(y1,y2,atol=1e-12,rtol=1e-12)
        flow_errors.append(float(np.max(abs(y1-y2))))
    run.write_json(run.ROOT/'tangent_validation.json',dict(passed=True,paths=len(paths),
        maximum_tangent_error=max(errors),maximum_complete_flow_error=max(flow_errors),
        seconds=time.time()-started,backend_sha256=run.sha(run.ROOT/'tangent_backend.py'),
        reference_source_sha256=run.sha(run.LEPS/'src/leps_core.py')))
    print('ASE tangent and full spring-flow equivalence passed',len(paths),max(errors),max(flow_errors),flush=True)


if __name__=='__main__':
    main()
