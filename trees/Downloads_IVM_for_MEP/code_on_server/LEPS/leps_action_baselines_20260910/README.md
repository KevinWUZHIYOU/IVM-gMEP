# LEPS: image-chain gMAM and two-stage StringNET

This campaign implements the two methods requested by the user on the same
22-node path as the registered LEPS baselines: 20 movable images and two fixed
endpoints. **StringNET here names the requested two-stage image-chain method;
it does not introduce a neural-network parameterization.** The older neural
gMAM experiment in the continuous-path figure is a separate experiment.

## Source and data provenance

All paths below are relative to `code_on_server/LEPS` unless stated otherwise.
The authoritative machine-readable registration is
`leps_action_baselines_20260910/registration.json`. It freezes all 60 parameter
combinations and the SHA-256 identities of the protocol, experiment runner,
potential, initialization, and both imported reparameterizers before the
production grid. Do not change registered source files to resume a run.

1. `inputs/common_initial_path.npz`: common ASE-IDPP initialization.
2. `src/torch_leps.py`: differentiable implementation of the existing LEPS
   potential; `src/leps_core.py` supplies independent NumPy energy/force checks.
3. `leps_action_baselines_20260910/run_experiments.py`: objective quadrature,
   objective derivatives, optimizers, phase switch, cache, and cost ledger.
4. `src/run_string_baseline.py`: unchanged cubic equal-arclength redistribution.
5. `leps_irivm_ala_chig_lepsnative_20260829/src/leps_irivm_lepsnative.py`:
   unchanged improved-tangent identical-spring redistribution, without recovery.
   The registered `tangent_backend.py` batches its nonperiodic improved-tangent
   algebra to avoid repeated ASE object construction. Before the full rerun,
   14 initial, perturbed, and experimental paths were compared against ASE:
   maximum tangent difference `2.22e-16`, maximum complete-redistribution
   difference `8.88e-16`. See `tangent_validation.json`. The spring flow and
   all optimizer parameters remain unchanged. Original-backend results are
   retained under `archive/pre_batched_tangent/`, not mixed into final selection.
6. `leps_action_baselines_20260910/runs/<configuration-id>.npz`: all raw paths,
   force values, objective values, update indices, phases, HVP counts, and events.
7. `leps_action_baselines_20260910/analyze_and_plot.py`: independent metric and
   accounting audit, frozen-rule parameter selection, and figure generation.
8. `leps_action_baselines_20260910/reports/grid_summary.csv`: every tested
   configuration, including numerical failures and ineligible warmup runs.
9. `leps_action_baselines_20260910/reports/selection.json`: selected gMAM,
   selected StringNET, best run for each beta, and optimizer/reparameterizer
   comparisons. The reported winners are grid optima under the registered
   rule, not claims of global optimality.
10. `leps_action_baselines_20260910/reports/leps_all_methods_metrics_source.csv`
    and `leps_all_methods_metrics_provenance.json`: every plotted point, raw
    source hashes, selected parameters, figure hash, and plotting-code hash.

The six inherited trajectories and their prior registration remain in
`leps_irivm_sync1_refined_grid_20260830/reports/figure/`. Their data are unchanged;
only the displayed label `Original String` becomes `String Method`. The prior
PDFs are preserved in this campaign's `archive/previous_six_method_figure/`.
The prior IVM-specific scientific provenance remains in
`leps_irivm_sync1_refined_grid_20260830/FIGURE_PROVENANCE.md`.

## The two optimized functionals

Let `l_i = |x_(i+1)-x_i|`. Both objectives use the original 21 path segments,
including endpoint values, not the 500-point diagnostic densification:

- gMAM: `S_h = sum_i l_i (|grad U_i| + |grad U_(i+1)|)/2`.
- StringNET warmup:
  `I_beta,h = log(sum_i l_i (exp(beta U_i)+exp(beta U_(i+1)))/2)/beta`,
  evaluated with a stable common exponential shift.

Their derivatives include derivatives of both integrand values **and segment
lengths**. The gMAM integrand derivative is `H grad U / |grad U|`, computed by
automatic differentiation. At an exactly vanishing gradient, the norm's zero
subgradient is used; the potential and norm are not smoothed. Independent
finite-difference checks of the complete chain-objective gradient are stored
in `validation.json`. Finite differences here are a derivative test, never a
path-tangent formula. All tangent-based residuals and spring redistribution use
the Henkelman--Jonsson improved tangent.

StringNET completes exactly 50 optimizer updates with the warmup functional;
each update includes redistribution. Update 51 uses gMAM, and optimizer memory
is reset at that switch. Each subsequent update still includes redistribution.
No recovery, residual acceptance gate, saddle location, or diagnostic barrier
is fed to either optimizer. Failed attempts are retained in the cost ledger.

## Grid and equivalent force cost

The Cartesian grid has 60 combinations: gMAM or one of four StringNET betas
`{0.1, 0.3, 1, 3}`, L-BFGS initial scales `{0.01, 0.1, 1}` or Adam learning
rates `{0.001, 0.01, 0.1}`, and either of the two redistribution rules.
All other settings are fixed; see `PROTOCOL.md` for Armijo, memory, displacement
cap, and stopping details. Each run has a **20,000 equivalent-NoF budget**.
This budget includes trial states and post-redistribution states.

The final accounting convention is the user's explicit clarification:

`NoF = uncached ordinary internal-image force evaluations + internal-image HVPs`.

Thus a fresh gMAM node costs two NoF; if its ordinary gradient is already
cached, its new HVP adds one NoF. A fully cached value costs zero. StringNET
warmup has no HVP cost, and the switch adds one NoF for each cached internal
node needing an HVP. Endpoints are fixed and excluded from optimization cost.
The code records ordinary forces and HVPs separately and checks the full
equivalent cost **before** each batch. This is an equivalent-work convention,
not a claim that a central-difference HVP is actually evaluated by the code.

Two superseded accounting versions are archived and excluded from all final
selection: `archive/pre_hvp_cost_amendment/` and
`archive/two_extra_hvp_superseded/`. Their completed runs are not reused as
production results under a different budget.

## Metrics and selection

`I_perp` is the article's segment-length-weighted squared normal residual with
improved tangents. The densified forward barrier uses 500 total linearly
interpolated points. Discrete geometric action uses the 22-node trapezoid above.
Diagnostics are offline and not fed to the optimizers or added to NoF.

Selection uses 20 predetermined budgets, `1000, 2000, ..., 20000`, and the
latest complete state at or below each budget. The score is the mean log
normalized residual plus mean log normalized absolute error in the densified
barrier. Floors are `1e-12` for the residual and `1e-6 eV` for barrier error;
normalization uses the common initial path. Ties use final action, total NoF,
then configuration ID. Numerical failures and StringNET runs that do not
finish warmup and a gMAM update are ineligible. The full protocol was frozen
before the production results, and reference information is used only for
offline selection. The figures plot actual saved states, not cumulative minima.

## Rebuild

From the project root, using the registered local environment:

```sh
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python code_on_server/LEPS/leps_action_baselines_20260910/run_experiments.py run
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python code_on_server/LEPS/leps_action_baselines_20260910/analyze_and_plot.py
```

The runner resumes only source-identical registered results and holds the
existing independent host-local LEPS `fcntl` lock; execution is serial.
The analyzer refuses an incomplete grid or a source/hash/accounting mismatch.
It cross-checks potential values, forces, phase switches, fixed endpoints,
initial paths, and ASE improved tangents before selection and publication.

Final outputs are the updated `Figures/LEPS/leps_all_methods_metrics.pdf`
(also copied into `output/pdf/`, `paper0819/Figures/LEPS/`, and the LEPS source
figure directory) and `output/pdf/leps_action_baselines_tuning.pdf`, which
shows the beta-specific selected StringNET curves. Each figure retains three
square metric panels over the same 20,000-NoF window.

## Completed results (2026-09-10)

All 60 registered configurations have been executed and independently audited.
There are 49 eligible results and 11 numerical failures, all retained. Total
production equivalent NoF is **1,043,540 = 569,640 ordinary forces + 473,900
HVPs**. This excludes archived superseded runs, validation, and offline metrics.

| Selected method | Optimizer | Redistribution | Beta | Last saved NoF | Final I_perp | Final densified barrier (eV) | Final action (eV) |
|---|---|---|---:|---:|---:|---:|---:|
| gMAM | Adam, rate 0.01 | String Method cubic | -- | 19960 | 0.0819384 | 4.51932295 | 6.87585046 |
| StringNET | L-BFGS, initial scale 1 | IVM spring, no recovery | 1 | 19940 | 0.403813 | 1.35671968 | 1.64766743 |

The selected raw gMAM result is `runs/ac98a95a65d71eee.npz`; its terminal
equivalent cost is 20,000 (10,000 ordinary forces + 10,000 HVPs), including
work after the last complete saved state. The selected StringNET result is
`runs/f5637bc583dfb260.npz`; its terminal cost is 19,980 (11,000 ordinary
forces + 8,980 HVPs). It has 277 completed updates: 50 warmup updates and
227 gMAM updates. Warmup finishes at 2040 NoF; the first completed gMAM update
is saved at 2340 NoF. The selected gMAM has 249 completed updates.

| StringNET beta | Selected optimizer / scale | Redistribution | Joint score (lower is better) | Final I_perp | Final barrier (eV) |
|---:|---|---|---:|---:|---:|
| 0.1 | Adam / 0.1 | IVM spring | -5.49879 | 0.00169588 | 4.51985414 |
| 0.3 | L-BFGS / 0.1 | IVM spring | -3.80482 | 0.709633 | 1.36068749 |
| 1 | L-BFGS / 1 | IVM spring | -8.34031 | 0.403813 | 1.35671968 |
| 3 | Adam / 0.01 | String Method | -1.86916 | 0.306800 | 4.51759214 |

The reference forward barrier is 1.34308607634410 eV. Neither selected method
reaches `I_perp <= 1e-4` within the budget. Small beta does not guarantee the
correct barrier branch: beta 0.1 has a smaller residual but a substantially
higher barrier than beta 1. The selected StringNET has smaller discrete action
than the registered IVM, but a much larger normal residual; this is not evidence
of better MEP convergence. Its warmup update 33, at 1360 NoF, reaches a
densified barrier of 103.322 eV and action 458.891 eV.

At the user's explicit request, the final display omits spike values above
`I_perp=100`, `barrier=5 eV`, and `action=30 eV`. These round limits retain
the full ranges of the six inherited trajectories. Out-of-range values are
excluded from display; visible sampled points are connected by straight lines
without smoothing, as explicitly requested by the user. An upward triangle at
the true NoF of each omitted maximum reports the actual peak value. All
omitted points, values, and costs are registered in `reports/spike_display.json`.
Raw NPZ data, source CSV values, parameter selection, and NoF are unchanged.
The full-range figure version is preserved in
`archive/full_range_before_display_amendment/`. No cumulative minima are used.

The final curves use a uniform 500-NoF sampling rule for every method:
choose the latest recorded state at or below each checkpoint, retain initial
and final states, and deduplicate repeated selections. The same rule is used
in the beta comparison. `reports/sparse_sampling.json` registers every chosen
source index and actual NoF. Peak annotations still use the full data; straight
segments connect displayed points and do not reconstruct intermediate states.
Sampling does not change raw
data, full source CSVs, selection scores, or reported convergence thresholds.
The preceding display is archived in `archive/before_uniform_sparse_display/`.

`paper0819/main.tex` now describes the two definitions, phase switch, cost,
selection, and observed limitations. Its formal root PDF is not overwritten;
build only the preview with SyncTeX using the repository's prescribed command.
