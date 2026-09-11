# Audited LEPS IVM and baseline experiment

This directory is the independent LEPS sibling of `barnase_barstar` and
`chignolin`.  It does not modify `IVM_on_server`.

## Fixed problem definition

- The LEPS energy and force are algebraically identical to
  `LEPS_IVM_Images_N20_tune_7_1.ipynb` and `LEPS_neb.ipynb`.
- Endpoints are `(r_AB, r_BC) = (0.74200064, 8.0)` and
  `(8.0, 0.74200064)`.
- There are 20 internal images and 22 total images.
- Every run loads the same byte-identical ASE-IDPP path from
  `inputs/common_initial_path.npz`.
- No climbing image is used.
- Every tangent-dependent normal projection and every \(I_\perp\) diagnostic
  uses ASE 3.23.0 `ImprovedTangentMethod`, i.e. the energy-weighted
  Henkelman–Jónsson improved tangent.  Original String itself uses a
  full-force update and cubic equal-arclength reparameterization, so its
  update does not require a tangent.
- The ODE-NEB production uses `NEBOptimizer(method="ode")`; its ODE branch
  does not use `alpha`.  The separately labeled static-NEB baseline uses the
  same NEB force with a fixed Euler step.
- LEPS is evaluated analytically on the Mac CPU; CUDA is neither required nor
  used.

The analytically certified index-1 saddle is

```text
(r_AB, r_BC) = (1.1493779934, 0.8624687699)
Delta E*      = 1.3430860763 eV
```

## IVM implementation

At each outer iteration the improved tangents are frozen.  In the two
energy-relevant LEPS coordinates, each internal-image normal space is
one-dimensional.  The code solves

\[
\min_z\; U(x_i+z n_i)+\frac{z^2}{2\alpha},
\]

where \(n_i\) is perpendicular to the frozen ASE improved tangent.  A
one-dimensional projected L-BFGS/BFGS iteration is used with an Armijo test on
the genuine proximal objective.  No potential Hessian is requested or
finite-differenced.

The alpha study uses one fixed value throughout each independent run:

```text
alpha = 1, 3, 10, 20, 30, 50, 70, 100, 150, 300
```

Three subproblem stopping policies are crossed with two reparameterization
policies:

```text
inexact: relative KKT tolerance 0.80, at most 15 calls/image
tight:   relative KKT tolerance 0.05, at most 80 calls/image
near-exact:
         projected-gradient KKT rtol 1e-4, atol 1e-7,
         at most 300 calls/image, solved by SciPy L-BFGS-B

none
post_hermite: coordinate-only equal-chord projection on the frozen
              improved-tangent Hermite curve
```

`inexact` and `tight` use the inexpensive one-dimensional projected BFGS
implementation used by the production algorithm.  `near-exact` keeps the
identical frozen-normal proximal objective but uses a mature box-constrained
L-BFGS-B line search.  It is a deliberate reviewer-facing control for the
question “what happens if every subproblem is solved almost to completion?”
It is not used to improve or select the production curve.  Across the ten
reparameterized alpha runs, 4165/4180 image-subproblems (99.64%) satisfy the
near-exact KKT target.

The selected production setting is fixed `alpha=30`, tight subproblems, and
post-Hermite reparameterization.  The reparameterization is an optional
operator-splitting step after the IVM subproblem; it never calls the LEPS
energy/force backend while constructing the proposal.

## Evaluation accounting

Each of the 22 images owns an independent experiment-local evaluator.
A cache miss computes and stores energy and force jointly.  Merely modifying
coordinates does not increment a counter.  Force followed by energy, energy
followed by force, repeated property requests, and exact revisits to any
coordinate previously evaluated by that image in the same experiment are
cache hits.  Caches are never shared across runs.

Every trace stores:

- energy API request count per image
- force API request count per image
- cache-hit count per image
- real joint energy–force evaluation count per image
- internal-image max, mean, and sum

Two cost projections are retained and must not be conflated:

- `calls_per_image` is the maximum number of unique joint evaluations over the
  20 internal images.  It is used by the original scans and legacy figures.
- `Number of Forces`, as used by the registered all-method figure below, is
  the sum of uncached joint energy-force evaluations over all 20 internal
  images.  Fixed endpoints, cache hits, and repeated API requests are excluded.

Because every cache miss computes energy and force jointly, the second
quantity is the actual optimization force cost.  Dense barrier sampling is
analytic postprocessing and is excluded from both axes.  All production runs
satisfy

```text
sum(all 22 image cache misses) == raw LEPS backend calls
```

## Metrics

\[
I_\perp=\sum_{i=1}^{20}
\frac{\lVert x_i-x_{i-1}\rVert+\lVert x_{i+1}-x_i\rVert}{2}
\lVert F_{\perp,i}\rVert_2^2 .
\]

The energy barrier is evaluated on 80 linear subdivisions of every image
segment and referenced to the reactant endpoint.  The equidistance diagnostic
uses internal central differences:

\[
I_{\mathrm{arc}} =
\int_0^1
(\phi'(s)\cdot\phi''(s))^2\,|\phi'(s)|\,ds .
\]

This discrete form is exactly zero to roundoff for an equal-chord polygon and
therefore tracks the NEB spring-equilibrium condition without endpoint
finite-difference artifacts.  A second spacing metric is

\[
F_{\mathrm{spring,max}}^{(k=1)}
=\max_i\left|\lVert x_{i+1}-x_i\rVert-\lVert x_i-x_{i-1}\rVert\right|.
\]

The common unit spring constant makes this a pure spacing diagnostic, rather
than rewarding or penalizing NEB for the value of \(k\) being scanned.

## Continuous neural-path experiment

`src/run_nn_ivm_theta_al_adaptive.py` implements the fixed-endpoint
parameterization

\[
\phi(s;\theta)=\bar\phi(s)+s(s-1)\,\delta_\theta(s),
\]

where the final layer starts at zero, giving the byte-identical common straight
path.  The selected float64 network has two hidden tanh layers of width 32 and
Fourier features through order 4.

Every NN–IVM subminimization directly optimizes all trainable parameters
\(\theta\).  At 20 collocation points it minimizes the energy plus the normal
proximal displacement and uses an augmented Lagrangian to enforce zero
tangential displacement.  There is no pointwise solve followed by a neural
fit.  The selected fixed parameters are \(\alpha=100\), at most six
multiplier cycles, and at most 12 strong-Wolfe L-BFGS iterations per cycle.
L-BFGS curvature is retained while the AL penalty is unchanged.  The penalty
is multiplied by five only when the maximum constraint residual fails to
decrease by at least one half; curvature is then reset.  The accepted maximum
tangential displacement is at most \(5\times10^{-3}\).

Each objective closure makes one historically cached joint energy-force
query.  A custom autograd operation uses the returned force as
\(-\nabla U\), so a cache hit does not secretly recompute the Torch LEPS
potential and no energy Hessian is formed or finite-differenced.

Both neural methods use exactly 20 internal collocation streams.  The samples
are equal-arclength stratified and reproducibly jittered on every iteration:
NN-IVM uses at most 0.15 and gMAM uses at most 0.45 of one stratum width.
This covers the intervals over time without giving a neural method more force
samples than discrete IVM or NEB.  The tangent at every requested or
diagnostic point is \(d\phi/ds\) from autograd.

`src/run_gmam_autodiff.py` directly minimizes

\[
\int_0^1\lVert\nabla U(\phi(s))\rVert\,\lVert\phi'(s)\rVert\,ds
\]

with a pure Torch LEPS potential.  `backward()` differentiates through the
force norm and path tangent; no hand-derived gradient, explicit Hessian, or
finite-difference Hessian is used.  From the common path this gMAM run enters
a high-barrier stationary branch, which is retained as an honest
initialization-sensitive outcome.

The common primary cost axis is the maximum unique joint energy–force
evaluations per one of the 20 internal images/streams.  Dense diagnostics use
200 linear subdivisions per segment for discrete paths and up to 16,385
autograd-tangent points for continuous paths; these diagnostic evaluations
are not optimization samples.

## Added discrete baselines

`src/run_string_baseline.py` implements Original String and OBSM with either
FIRE or L-BFGS frozen-normal subminimizers.  The focused 300-call scan,
1000-call branch audit, L-BFGS reparameterization refinement, and independent
3000-call productions are reproducible through
`run_baseline_refinement.py`, `run_baseline_extensions.py`,
`run_lbfgs_reparam_refinement.py`, and `run_baseline_production.py`.

The selected production settings are static NEB `(k, dt)=(10, 0.05)`,
Original String `dt=0.07`, OBSM-FIRE
`(lambda, inner, dt)=(0.85, 4, 0.18)`, and OBSM-L-BFGS `lambda=0.95`, one
inner step, initial inverse-Hessian scale 0.1, history 7, with
reparameterization every five outer steps.  Derived final-value and
first-attainment summaries are written to
`reports/leps_all_methods_production.json` and
`tables/leps_all_methods_production.csv`.  The six production NPZ files, not
these summaries, are the authoritative trajectory data.

## Figure guide

- `leps_ivm_vs_neb`: the main \(I_\perp\), \(I_{\mathrm{arc}}\), raw barrier,
  and final-path comparison.
- `leps_cost_to_precision`: minimum calls/image needed to first attain a
  requested \(I_\perp\) or barrier-error threshold.
- `leps_alpha_sensitivity`: all three inner stopping policies, with/without
  reparameterization, compared at the common 200-call cutoff using
  \(I_\perp\), raw \(\Delta E\), \(I_{\mathrm{arc}}\), and maximum unit-\(k\)
  spring force.
- `leps_alpha_trajectories`: early 0–100-call trajectories for every alpha
  under the near-exact, reparameterized control.
- `leps_subproblem_reparam_ablation`: complete 0–200-call curves at the
  selected alpha for all six stopping/reparameterization combinations.
- `leps_path_evolution_1x4`: equal-cost snapshots at 0, 30, 200, and 500
  calls/image on the exact LEPS potential; only the common initial path is
  drawn in the first panel.
- `leps_continuous_methods_comparison`: dense \(I_\perp\), raw barrier, and
  barrier error for discrete IVM, ODE–NEB, NN–IVM, and direct-autodiff gMAM
  on the complete common max-calls/image axis.
- `leps_continuous_per_stream_comparison`: the two main precision metrics on
  the same audited per-stream axis.
- `leps_continuous_cost_to_precision`: first-attainment curves using the
  dense metrics and max calls per internal image.
- `leps_continuous_path_evolution_1x4`: common-path and 30/150/300-call
  snapshots on the exact LEPS surface.
- `leps_continuous_counter_audit`: per-evaluator real joint evaluations and
  run-local cache hits for all four production runs.
- `leps_all_methods_metrics`: the refined-grid IR-IVM, now named `IVM`, and
  five other discrete production methods versus the true total number of
  uncached internal-image force evaluations; the former article IVM is not
  plotted. Panel a is \(I_\perp\), panel b is the densified forward barrier
  \(\Delta E\), and panel c is the 22-node discrete geometric action. All
  three plotting boxes are square.
- `leps_all_methods_cost_to_precision`: first attainment for normal-force and
  densified-barrier-error thresholds across all six methods.

## Current registered provenance: `leps_all_methods_metrics.pdf`

The figure was permanently replaced on 2026-08-30 by explicit user
authorization.  Its single curve named `IVM` is now the refined-grid IR-IVM
with

\[
\alpha=100,\qquad r=3\times10^{-3},\qquad m_{\max}=5.
\]

The former article IVM trajectory is no longer plotted. The five non-IVM
comparison trajectories are unchanged. On 2026-09-06 the action panel was
added by explicit user authorization and all three panels were made square.
The output, source, and paper-working copies are byte-identical, with SHA-256
`081aad16011737c67d8d9eaded57559655f9b45f6ab18817940ca019c17f66b4`.
The IVM result contains 28 completed outer iterations plus the initial state;
all 29 accepted states are displayed with individual blue markers.

The chosen run is the fixed-rule winner of the preregistered 150-combination
refined grid and the deduplicated 186 settings across both registered stages.
It passes 19/21 strict same-NoF checks; no setting passes 21/21.  The
densified-barrier misses at NoF 150 and 278 remain disclosed in the registry.
Figure replacement does not relabel this outcome as a strict pass.

The authoritative, self-contained registry for the current figure is
`leps_irivm_sync1_refined_grid_20260830/FIGURE_PROVENANCE.md`.  It traces the
formal PDF to the plotted-point CSV, reconstruction manifest, plotting code,
selected raw NPZ, algorithm implementation, preregistration, protocol,
independent audit, selection manifests, NoF ledger, and the five unchanged
comparison NPZ files.

Rebuild the current registered figure with:

```bash
cd /Users/wuzhiyou/Downloads/IVM_for_MEP

MPLCONFIGDIR=/private/tmp/leps-three-metrics-mpl \
  /Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  code_on_server/LEPS/leps_geometric_action_20260904/src/remake_leps_all_methods_three_metrics.py
```

## Additional discrete geometric-action figure (2026-09-04)

`Figures/LEPS/leps_all_methods_geometric_action.pdf` is the earlier standalone
view of the action that is now also incorporated into panel c of the current
three-metric figure. It uses the same six trajectories and
NoF values, and evaluates the full-gradient geometric action on the original
22 nodes using segment-length trapezoidal quadrature, with no densification.
The existing improved tangent is reused for the I_perp alignment audit; the
full-gradient action itself is tangent-independent.

The calculation script, all plotted points, quadrature/tangent audit arrays,
input/output hashes, numerical results, and reproduction command are registered
in [GEOMETRIC_ACTION_PROVENANCE.md](leps_geometric_action_20260904/GEOMETRIC_ACTION_PROVENANCE.md).

## Historical six-method provenance (superseded for this PDF)

The remainder of this section records the earlier six-method rendering for
historical reproducibility.  Its hashes and redraw command do **not** identify
the current `leps_all_methods_metrics.pdf`; use the current registry above.

This section is the provenance registry for the working-copy paper figure

```text
/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/LEPS/
leps_all_methods_metrics.pdf
```

The experiment root \(R\) in all paths below is
`/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/LEPS`.  The current
formal manuscript tree under `/Users/wuzhiyou/IVM_for_MEP/paper` is separate;
this registration does not by itself declare the figure an active formal-paper
asset.

### Artifact identity

`src/analyze_full_method_comparison.py` writes the source PDF, SVG, and PNG
under `R/figures`.  The paper-working-copy PDF is then copied separately.  The
two PDFs registered here are currently byte-identical.

| Role | Path relative to \(R\) | Bytes | SHA-256 |
|---|---|---:|---|
| Source PDF | `figures/leps_all_methods_metrics.pdf` | 64,763 | `6f1a5ea305850ee031f8af0bcdbb33daaf1e74dab83c97ea37cb3ffabbad7a3d` |
| Paper working copy | `../../Figures/LEPS/leps_all_methods_metrics.pdf` | 64,763 | `6f1a5ea305850ee031f8af0bcdbb33daaf1e74dab83c97ea37cb3ffabbad7a3d` |
| Source SVG | `figures/leps_all_methods_metrics.svg` | 86,107 | `940c3adfab35155182844de33944d07a6ba9a68551a642ce0d4a464967b2f0e3` |
| Source PNG, 600 dpi | `figures/leps_all_methods_metrics.png` | 342,462 | `be6bc61b8cd0fcf54b8ae02c2e4c8e12540ec641e0536364705c3aed51bfa717` |

The direct plot generator is
`src/analyze_full_method_comparison.py`, SHA-256
`d23acd455826ea0df8d1021f7a57d46375a1fc216d6faddab658238f5425a961`.
It requires exactly one match for each of the six frozen production patterns
and fails closed otherwise.  The Muller-Q10 PDF
`/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/muller_q10_10d/figures/active_candidates/muller22_q10_article_method_comparison_1x3_linear3000.pdf`
is a visual-style reference only; it supplies no LEPS data.

### Complete data dependency chain

```text
inputs/common_initial_path.npz + src/leps_core.py
    -> production drivers and algorithm implementations
    -> six production NPZ files + six audit JSON files
    -> src/analyze_full_method_comparison.py
    -> tables/leps_all_methods_metrics_source.csv
    -> figures/leps_all_methods_metrics.{pdf,svg,png}
    -> ../../Figures/LEPS/leps_all_methods_metrics.pdf
```

The plot reads the six NPZ files directly.  It does not read the derived JSON
or CSV summaries as curve input.  The generated
`tables/leps_all_methods_metrics_source.csv` is an exact, machine-readable
projection of every point displayed in this figure; its SHA-256 is
`6ec944a509e42ab4922b485e1c3ce2d2d51fd22d0a24432c91b30a8400f3a603`.

### Six authoritative production trajectories

Every NPZ is paired with its same-basename `.audit.json`.  The audit fixes the
full configuration hash, stopping reason, per-image counter vectors, cache
hits, raw backend total, and the counter-consistency result.

| Method and frozen configuration | Production NPZ and SHA-256 | Audit JSON and SHA-256 |
|---|---|---|
| IVM: fixed \(\alpha=30\); KKT rtol `0.05`, atol `1e-10`; at most 80 inner calls; history 7; independent projected L-BFGS; post-Hermite; budget 500 | `runs/ivm/ivm_production_alpha30_tight_reparam_cfg29674d42197b_budget500.npz`<br>`2f43cd57182af04de40fd33ef4a825ec49cfe0e9c9d6f24f43a800367be6e65c` | same basename `.audit.json`<br>`59a159fd7378c7ce2b9d04f6205dc6a8dadfbb5deafb5e52a79f641bee96c15c` |
| ODE-NEB: ASE ODE; improved tangent; \(k=10\); rtol `0.1`, C1 `0.01`, C2 `2`; fmax `1e-8`; budget 5000 | `runs/neb/neb_production_k10_cfgfe491f5a6c6f_budget5000.npz`<br>`ae0c2bbf3f8fa8539bcaffaa98e0298b5f90661f145fcb5e7a5c6b61c4548b0a` | same basename `.audit.json`<br>`15e619d4b31ba4b3c070b39ac826d8066f95e4c99f0cc4331edf896d080ab087` |
| Static-NEB: improved tangent; \(k=10\); static Euler step `0.05`; fmax `1e-8`; budget 3000 | `runs/neb/production_static_k10_dt0.05_cfgba1c63497f9a_budget3000.npz`<br>`a8daa64a59d7316c4546f47591cbfcdd509c3744eb0def2390c3951c5cdcb046` | same basename `.audit.json`<br>`7a04594dec6ab14486b6d5a8580db9f2cb4a51c50825b7c480c29ce0a31bf7f8` |
| Original String: full-force Euler `dt=0.07`; cubic-arclength reparameterization every step; budget 3000 | `runs/original_string/production_original_dt0.07_r1_cfg6446cefea5fd_budget3000.npz`<br>`c480e8520ff37b493e20c8dd476655aee58d0ebbd119360eccd74f8971f1a4cd` | same basename `.audit.json`<br>`d4fdd1ae05a714eb7d57229d0b2ab05442f68e2a76d844b91db16bbffe4a627e` |
| OBSM-FIRE: \(\lambda=0.85\); 4 inner steps; FIRE `dt=0.18`, \(\beta=0.1\), mass 1; reparameterize every step; budget 3000 | `runs/obsm_fire/production_fire_stable_l0.85_n4_dt0.18_cfgc5d07bd3b321_budget3000.npz`<br>`24dc2c8882d3cab39861fb69a8630b317532f26a79bd6c7fb913a1ce23faa456` | same basename `.audit.json`<br>`eb5a03039297d7e9b499e305bfec171a0dfc087461a595dd9f50f2cd1ec28e41` |
| OBSM-L-BFGS: \(\lambda=0.95\); 1 inner step; initial inverse-Hessian scale `0.1`; history 7; Armijo C1 `1e-4`; backtrack `0.5`; reparameterize every 5 outer steps; budget 3000 | `runs/obsm_lbfgs/production_lbfgs_r5_l0.95_n1_s0.1_cfg78c1b29f7570_budget3000.npz`<br>`b5ae71ab114e8dffb90be582b327c1c6468f13123289b4358cdd3ea143be22cb` | same basename `.audit.json`<br>`abff53c7e2fef43ceee7b1fc483e90e2b9f6ad6495b10372da717abf653e2b40` |

The corresponding full configuration SHA-256 values are, in table order,
`29674d42197b8a0c0a5aa9d230ea891184da3bb5d8ba50cd0ec8c7aa54b063d2`,
`fe491f5a6c6ff77fbea58cd4b13d2555f625331c03867515a80de47c7ffba37a`,
`ba1c63497f9a503d4bc75b21e24806a2f4b467d5996aef205f99bcb75ea77d63`,
`6446cefea5fdd111ab794cd59c21aa6e319aa347b5026321d2d110c58e599e25`,
`c5d07bd3b321c4dd2e9ba9a1ade03b833af3d404bf1d0906af6f45a9f7f0a104`,
and
`78c1b29f7570265674bbfc21808c26852faedfdbdfdeea90eac035f21898bb9a`.

### Exact plotted-field transformation

- **x, `Number of Forces`:** total uncached force evaluations summed over the
  20 internal images.  For IVM and both NEB runs the plotter verifies
  `total_internal_calls == sum(counter_vector[:, 1:-1])`.  For Original String
  and both OBSM runs it uses
  `sum(joint_evaluations_history[:, 1:-1])`.  The first and last columns are
  fixed endpoints and are excluded.  `force_request_vector` is deliberately
  not used because it includes cache hits and repeated requests.
- **Panel a:** NPZ field `i_perp`, on a logarithmic y axis.
- **Panel b:** the raw densified forward barrier, on a linear y axis.  The IVM
  and NEB schemas use `dense_barrier`; Original String and OBSM use `barrier`
  for the densified value and retain the sparse-image value separately as
  `raw_sparse_barrier`.
- **Densification:** analytic LEPS energy on 80 linear subdivisions of every
  image segment, relative to the reactant endpoint.  It is diagnostic
  postprocessing and does not increment the force ledger.
- **IVM view:** retain the initial point and states satisfying
  `segment_length_cv <= 1e-10`.  This removes 21 accepted but
  pre-reparameterization trial states and leaves 180 committed states before
  the force-window mask.
- **Window:** retain only checkpoints with `Number of Forces <= 20000`; do not
  interpolate or extrapolate.  Displayed point counts and last x values are
  IVM `180/2572`, ODE-NEB `930/20000`, Static-NEB `1000/20000`, Original
  String `1000/20000`, OBSM-FIRE `200/19920`, and OBSM-L-BFGS `418/19985`.
  IVM ends at 2,572 because that frozen production run converged there; its
  shorter curve is not a plotting omission.
- **Presentation:** x range 0--20000 with ticks every 5000; both panels have no
  grid; panel letters are outside and above the axes.  Style, fonts, colors,
  line types, and white-faced markers follow the Muller-Q10 reference named
  above.

For scale, the full production trajectories end at total internal force costs
of 2,572 (IVM), 100,000 (ODE-NEB), 46,240 (Static-NEB), 22,421 (Original
String), 56,036 (OBSM-FIRE), and 43,370 (OBSM-L-BFGS).  These full trajectories
remain in the NPZ files even though the registered figure stops at 20,000.

### Common scientific inputs and implementation code

All six NPZ files contain the same byte-identical 22-by-6 initial array: 20
internal images plus two fixed endpoints, prepared by ASE IDPP.  Two distinct
hashes are intentionally registered:

- NPZ container file `inputs/common_initial_path.npz`: SHA-256
  `59ce6ed37215299177bf28b7756e8045b530ff08bd8e5d557945106fdc373da4`.
- Canonical float64 `initial_path` array bytes: SHA-256
  `6cf4f5cb5c0eca04fd8fc0d72f71b751526442f6422e4ecb7e31be0a07b063c8`.

The initialization certificate is `inputs/initialization_audit.json`, SHA-256
`d1c822d3b1fd3ed0161328991ad5b62de2d7d46415373d13c79b3367875b6d80`.
It records the ASE-IDPP source, endpoints, tangent definition, segment lengths,
and the certified index-1 saddle/barrier.  The six embedded initial arrays and
their stored canonical hashes have been checked against this registered input.

| Responsibility | Code | Current SHA-256 |
|---|---|---|
| IDPP, scans, IVM/ODE production selection | `src/run_study.py` | `75e260e297de8280b8e2d5e8c4a0193323d36ab143c80d8f4098cd0c7b851403` |
| Four added long-budget productions | `src/run_baseline_production.py` | `e7cf874bcf28cc58e80957062197a2e71bc85c751e982d43d989c4dbfec4278a` |
| IVM algorithm | `src/run_ivm.py` | `9361d463b3474222d00075f80cf366ed91c1ba820c31a05cce2109f1c2deb13a` |
| ODE/static NEB algorithms | `src/run_neb.py` | `e4651408e73b4fe4a79580aeeb4431c08258bb2c90c7cb9bc5302f4ad2e53535` |
| Original String and OBSM algorithms | `src/run_string_baseline.py` | `7b053a6110b6750f8c09a81ece8d3589f862cf6e7dff6f9cacac7546d40352a1` |
| LEPS energy/force, counters, improved tangent, diagnostics, densification | `src/leps_core.py` | `a80f0c23252a2e273f4cf77d156c9a26de91b4559f56a4fad264638fa0e57aca` |
| Plotting and exact plotted-point export | `src/analyze_full_method_comparison.py` | `d23acd455826ea0df8d1021f7a57d46375a1fc216d6faddab658238f5425a961` |

The relevant implementation anchors are `leps_core.py` lines 25--45 for the
fixed problem, 107--194 for analytic energy/force, 480--506 for the direct ASE
improved-tangent call, 567--584 for \(I_\perp\), and 690--707 for the densified
barrier.  Production orchestration is in `run_study.py` lines 289--349 and
`run_baseline_production.py` lines 41--114.

The current environment is Python 3.8.20, NumPy 1.24.4, SciPy 1.10.1,
Matplotlib 3.7.2, and ASE 3.23.0; `requirements.txt` has SHA-256
`3c0dba3bd8cd02991a677e87bb127dfde1eab3ecd359b2448f34ea3fff048dfe`.

### Derived reports, selection history, and limitations

The plotter also writes `reports/leps_all_methods_production.json` and
`tables/leps_all_methods_production.csv`.  Their current SHA-256 values are
`54631143faf9d29e196387ce78d445f79d87a7200193253b28f9e44414d2c3d9`
and
`0bf2273b55197f41492679063bd263bb08e8e852b09fc96ad47472779b2e8a65`.
They contain final/minimum/first-attainment summaries and the figure-axis
contract, but they are not sufficient to reconstruct the trajectories without
the six NPZ files.

IVM/ODE selection history is in `reports/selection_summary.json` and
`reports/results_summary.md`.  Added-baseline selection history is in
`reports/leps_baseline_refinement_selection.json` and
`reports/leps_baseline_extension_audit.json`, with the corresponding scan
drivers described in the Added discrete baselines section.

The IVM and ODE audit JSONs preserve historical absolute `leps_staging` paths
for `result` and `initial_path_file`.  Those strings record the original
generation location, which has since been migrated.  The registered local NPZ
paths, exact file hashes, embedded initial arrays, canonical array hash, and
current plotter matches establish the active lineage; the audit files have not
been silently rewritten.

The NPZ artifacts embed configuration and initial-array hashes, but not a Git
commit or generator-code hash.  This directory is not a Git repository.
Therefore the current source-code hashes above identify the code now paired
with the frozen data; they are not claimed to be cryptographic proof that every
source byte is identical to the source used at the historical run time.

### Redraw from the registered NPZ files

```bash
cd /Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/LEPS

MPLCONFIGDIR=/tmp/leps-all-methods-mpl \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/analyze_full_method_comparison.py --root .

rsync -a figures/leps_all_methods_metrics.pdf \
  ../../Figures/LEPS/leps_all_methods_metrics.pdf

cmp -s figures/leps_all_methods_metrics.pdf \
  ../../Figures/LEPS/leps_all_methods_metrics.pdf
shasum -a 256 figures/leps_all_methods_metrics.pdf \
  ../../Figures/LEPS/leps_all_methods_metrics.pdf
```

PDF creation metadata can change on a later render, so a later correct redraw
need not reproduce the registered PDF byte hash.  On a given redraw, `cmp`
must still prove that the source PDF and paper copy are identical.

## Reproduce

Use the same Python environment as the Chignolin project:

```bash
cd /Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/LEPS

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/prepare_initial_path.py --root .

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/run_study.py --root . --stage all

MPLCONFIGDIR=/tmp/leps-mpl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/analyze_study.py --root .

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/run_baseline_production.py --root . --budget 3000

MPLCONFIGDIR=/tmp/leps-all-methods-mpl \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/analyze_full_method_comparison.py --root .

rsync -a figures/leps_all_methods_metrics.pdf \
  ../../Figures/LEPS/leps_all_methods_metrics.pdf

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/run_nn_ivm_theta_al_adaptive.py --root . \
  --output runs/continuous_v3/nn_ivm_adaptive_al_a100_j015_b500.npz \
  --run-label nn_ivm_adaptive_al_a100_j015_b500 \
  --alpha 100 --width 32 --depth 2 --fourier-order 4 \
  --collocation-count 20 --collocation-jitter 0.15 --budget 500 \
  --inner-iterations 12 --inner-evaluations 18 \
  --al-cycles 6 --al-penalty 100 --al-penalty-growth 5 \
  --constraint-reduction-target 0.5 --constraint-tolerance 0.005 \
  --diagnostic-points 8193 --arclength-points 4097

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/run_gmam_autodiff.py --root . \
  --output runs/continuous_v2/gmam_autodiff20_production_lr001_b500.npz \
  --run-label gmam_autodiff20_production_lr001_b500 \
  --collocation-count 20 --collocation-jitter 0.45 \
  --lr 0.001 --lr-decay 0.005 --budget 500 \
  --diagnostic-points 8193 --arclength-points 4097

MPLCONFIGDIR=/tmp/leps-continuous-mpl \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/analyze_continuous_study.py --root .

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
tests/run_tests.py

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
src/validate_experiment.py --root .
```

`reports/results_summary.md` gives the selection rule and numerical findings.
`src/analyze_study.py` regenerates the original discrete-study figures;
`src/analyze_full_method_comparison.py` separately regenerates the two
six-method figures and their CSV/JSON derivatives; and
`src/analyze_continuous_study.py` regenerates the continuous-method figures.
Figure outputs are exported as PDF, SVG, and 600-dpi PNG.  For
`leps_all_methods_metrics`, the complete curve source remains the six NPZ
files, while `tables/leps_all_methods_metrics_source.csv` records the exact
displayed projection.
