# Final Chignolin V9 alpha-analysis contract

This contract uses the user-requested final-window precision statistic after
the complete cap=30 grid showed that a cumulative statistic starting near the
common high-energy initial transient can rank alpha=0.5 ahead of alpha=1 even
though alpha=1 has substantially better budget-end precision.  No measured
alpha is removed, fitted, smoothed, or relabelled.  Cumulative values and the
unsuccessful cap=60/tighter-KKT controls remain archived and reported.

## Production algorithm

- Fixed-alpha frozen-normal proximal IVM.
- All normal projections and spring directions use the normalized central
  difference `x[i+1] - x[i-1]`.
- One alpha-independent inexact-subproblem rule is used throughout:
  relative KKT forcing factor `0.7`, with at most `30` accepted inner joint
  energy-force evaluations.
- Deterministic largest-normal-force block continuation is activated only by
  the same state-defined projected-stall condition for every alpha.
- Optional parameterization is a geometry-only identical nearest-neighbour
  spring flow.  It uses no spline and no OpenMM energy or force.
- The trust, line-search, spring-defect, cache, and stopping rules are identical
  for every alpha.  Only alpha changes.

The tighter KKT=0.5/cap=60 control is rejected for production because its
alpha=1 run has a trailing mean `I_perp = 0.3586553043149137` at
`N_F <= 10,000`, over an order of magnitude worse than the cap=30 production
candidate.  Cap=60/KKT=0.7 also fails to remove the late alpha=1.5 versus
alpha=2 phase reversal.  These negative results forbid selecting either
control merely to shape a plot.

## Alpha grid and optimization cost

- Alpha grid: `0.3, 0.5, 0.75, 1, 1.25, 1.5, 2`.
- Every run starts from the same saved 22-image initial path and covers the
  `N_F <= 10,000` slice through its latest attainable complete macrostate.
  Because one synchronized update evaluates all 20 internal images, the last
  complete state may be `N_F=9,982` while the next one would be `10,002`.
- `N_F` is the sum of audited joint energy-force cache misses over all 22
  images.  API requests and cache hits are reported separately.
- Budget slices are `N_F <= 1,000, 3,000, 6,000, 10,000`.
- A budget endpoint is the latest complete operator-split macrostate at or
  before the budget.  A late physical IVM half-step is not complete until its
  geometry-only spring step is committed.

## Primary final-window stationarity metric

For every budget `B`, the alpha-sensitivity ordinate is

`T_I(B) = integral I_perp(N_F) dN_F / (N_F,end - N_F,start)`,

where `N_F,start=max(N_F,first, N_F,end-0.30 B)`.  It is the force-axis
arithmetic mean over the final 30% of the requested budget through the latest
complete endpoint at `N_F <= B`.  Trapezoidal integration uses every measured
complete macrostate in that interval and interpolates only its two boundary
values.

This is not a best-so-far envelope: late oscillations remain penalized.  The
force-axis cumulative mean from the first produced complete macrostate and
the exact endpoint are retained as secondary diagnostics.  The final-window
definition prevents the common high-energy initial transient from dominating
the finite-budget precision comparison.

## Real OpenMM barrier metric

- Sparse 22-image barriers are monitoring-only and cannot enter selection,
  scores, figures, or article claims.
- Every analyzed path is recomputed by a separate diagnostic evaluator using
  Cartesian-linear densification of every adjacent image pair.
- Compute 10 and 20 subdivisions per segment first.  If successive barriers
  differ by more than `0.05 kcal/mol`, double to 40 and then 80.
- Diagnostic cache misses are audited separately and excluded from `N_F`.
- At each alpha/budget pair, select the latest complete macrostate at or
  before each of seven force-uniform targets in the final 30% window,
  including the endpoint.
- The primary barrier ordinate is the RMS error of these seven adaptive dense
  barriers relative to the independently frozen prior cap=15 reference
  `20.373739515742795 kcal/mol`.  The final-window definition avoids letting
  the common thousands-of-kcal/mol steric IDPP transient dominate a
  chemically relevant barrier diagnostic.
- Seven force-uniform checkpoints from the first produced macrostate through
  the endpoint are additionally densified and retained as an anytime audit.

## Well, selection, and noninferiority

- Every measured alpha is connected directly in increasing order.  No
  regression, smoothing, interpolation, point deletion, or missing-budget
  carry-forward is allowed.
- Both panels report the strict monotone-side test and a unique-interior-
  minimum test.  The stationarity well must have its unique measured global
  minimum at alpha=1.  The accepted dense-barrier well may place its unique
  measured global minimum at alpha=0.75 or 1.  Neither panel may contain a
  second interior minimum.  Small endpoint shoulders are shown rather than
  smoothed away.
- Eligible production members must, at the `N_F <= 10,000` complete
  macrostate:
  1. beat the prior cap=15 `I_perp = 0.03885242073674519`;
  2. have endpoint absolute dense-barrier error no larger than the frozen prior
     cap=15 final-window RMS value `0.798519886390451 kcal/mol`;
  3. have an equidistance integral no more than 1.20 times the fixed
     ODE-NEB endpoint.  ODE-NEB is the primary parameterization
     noninferiority comparator because it uses the same central-tangent
     nearest-neighbour spring geometry target.  The ratio to OBSM-FIRE is
     still reported, but is not silently substituted as the gate because its
     cubic-spline string parameterization is a different target; and
  4. beat both ODE-NEB and OBSM-FIRE in endpoint `I_perp` and adaptive dense
     barrier.
- Eligible members are ranked lexicographically by the geometric mean of the
  four final-window stationarity means, the geometric mean of the four
  final-window dense-barrier RMS errors, endpoint equidistance, and then
  `N_F`.
- The selected production curve is one unspliced run and is displayed only
  through the latest complete macrostate at `N_F <= 10,000`.
