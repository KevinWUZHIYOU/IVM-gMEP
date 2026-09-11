# Chignolin IVM–NEB experiment

This directory is an independent, audited Chignolin experiment. It does not
modify `IVM_on_server`.

## Fixed physical setup

- Inputs: `chignolin_clean.pdb`, `folded_min.traj`, `misfolded_min.traj`
- Path: 20 internal images, 22 images including endpoints
- Force field: `amber14-all.xml` + `implicit/gbn2.xml`
- OpenMM: CPU platform, double precision, `NoCutoff`
- No climbing image is used in this experiment
- Every NEB baseline uses `NEBOptimizer(method="ODE")`; FIRE and staged
  optimizer switching are excluded
- The NEB scan varies only
  \(k=[0.1,1,5,10,20,50,80,100,500,1000,2000]\). The stored `alpha`
  value is fixed interface metadata and is not a scan variable because ASE's
  ODE branch does not use it
- Both methods load the same elementwise-identical IDPP path from
  `inputs/common_initial_path.npz`

The only production tangent is

\[
\tau_i =
\frac{x_{i+1}-x_{i-1}}
{\lVert x_{i+1}-x_{i-1}\rVert}.
\]

`CentralDifferenceNEBMethod` replaces ASE's energy-weighted tangent while
retaining the usual NEB spring length difference. The same central difference
is used by IVM, IDPP, spring reparameterization, acceptance/rollback, all
metrics, and all plots.

## Evaluation accounting

Each of the 22 images owns a run-local exact-coordinate cache and four
counters:

- energy API requests
- force API requests
- coordinate-cache hits
- real joint OpenMM energy–force evaluations

Changing coordinates alone does not count. A request at a coordinate that is
new to that image in the current experiment causes one joint energy–force
evaluation; energy and force requests at the same coordinate reuse that
result. Returning exactly to a coordinate already evaluated by the same image
in the same run reuses its stored energy and force. Cache histories are never
shared between experiments.

The primary horizontal axis is the maximum number of real evaluations among
the 20 internal images. Every history point also stores the full 22-image
counter vector, the internal-image mean, and the internal-image sum. Dense
barrier calculations use separate diagnostic contexts and never contribute to
the optimization axis.

Before an optional IVM reparameterization is accepted, every internal image is
checked independently for its frozen-central-tangent IVP objective, energy,
and central-difference physical normal-force norm. A failed image is restored
to its pre-reparameterization coordinate and its future reparameterization
step scale is halved; the rejected real evaluation remains counted.

## Metrics

Only the unnormalized line integral is reported:

\[
I_\perp = \sum_{i=1}^{20}
\frac{\lVert x_i-x_{i-1}\rVert+\lVert x_{i+1}-x_i\rVert}{2}
\lVert F_{\perp,i}\rVert_2^2.
\]

The 0 K forward barrier is referenced to the folded endpoint. Dense barrier
postprocessing first uses 10 and 20 subdivisions per segment. If their
difference exceeds 0.05 kcal/mol, it doubles to 40 and then, if required, 80
subdivisions.

## Main programs

- `src/prepare_experiment.py`: build and audit the shared central-IDPP path
- `src/run_neb.py`: run one pure-ODE fixed-\(k\) central-difference NEB
  configuration
- `src/run_ivm.py`: run one audited IVM state policy
- `src/run_grid.py`: run the 96-config IVM screen, 12-config middle stage,
  protected late polish, or the complete NEB scan
- `src/analyze_neb_scan.py`: rank the 11 fixed-\(k\), pure-ODE 500-call runs
- `src/analyze_neb_extensions.py`: select the best fixed \(k\) from the three
  independent 2000-call reruns
- `src/postprocess_dense_barrier.py`: adaptive dense barrier with independent
  diagnostic accounting
- `src/validate_experiment.py`: fail-closed production validation
- `src/plot_production.py`: PDF/SVG/600-dpi PNG figures plus CSV/JSON exports
- `tests/run_tests.py`: dependency-light audit tests

## Tests

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python tests/run_tests.py
```

The tests cover coordinate-only changes, repeated changes before a request,
force→energy and energy→force reuse, ASE's force/energy request order, NEB
calculator reuse, rejected evaluations, backend/cache-miss equality,
checkpoint restoration, central-difference NEB runtime tangents, and central
spring reparameterization.

## Result roles

Files whose `run_label` is `production` are independent runs starting from the
shared initial path. Tuning outputs are retained as evidence but are never
concatenated into a production curve.

## Selected production results

- The 11-member, pure-ODE, \(k\)-only scan selected \(k=20\). The stored
  `alpha=1e-4` is compatibility metadata and is not used by ASE's ODE branch.
- The selected IVM is an independent 500-call run:
  `runs/ivm/ivm_state_policy_v5_hysteresis_production_central_state_ivm_cfg580da24ccdfc_budget500.npz`.
- This production run contains no force-call-index schedule. Alpha is
  controlled by the actual/predicted proximal decrease ratio, the trust
  radius scales with the median adjacent-image RMS displacement, and optional
  spring reparameterization is triggered and re-armed only by force and
  spacing states.
- At call 500, IVM has unnormalized \(I_\perp=0.00815182\), physical normal
  \(f_{\max}=0.00353810\) eV/Å, segment-length CV \(0.21263\), and adaptively
  densified forward barrier \(21.22056\) kcal/mol.
- IVM is below the 11-\(k\) pointwise NEB \(I_\perp\) envelope at every call
  from 2 through 500. The weakest, median, and call-500 advantages are
  2.49×, 76.9×, and 55.8×.
- At the finest common \(I_\perp=0.0284317\), IVM needs 211 calls and the
  extended fixed NEB needs 5000 calls: 23.70×.
- At the finest common densified barrier ceiling \(20.83376\) kcal/mol, IVM
  needs 265 calls and NEB needs 3060 calls: 11.55×.
- The IVM optimization audit has exactly 8673 backend evaluations and 8673
  summed image-evaluator cache misses. The separate 500-path dense diagnostic
  has 99,101 backend evaluations and 99,101 cache misses.
- `reports/ivm_state_policy_v5_report.md` records theory status, the
  transferable state policy, performance, equidistance tradeoff, and complete
  accounting. All 42 automated tests pass.
