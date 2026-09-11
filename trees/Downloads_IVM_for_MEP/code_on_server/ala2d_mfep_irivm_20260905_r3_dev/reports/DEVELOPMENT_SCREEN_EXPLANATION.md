# Why the V15 development grid is systematic

Status: **fixed development screen; not formal MD evidence**.

The completed V14 development screen returned `NO_COMBINATION_PASSED`.  Its
calibrated-noise trajectories showed that an unresolved relative-residual
interval or direction comparison could repeatedly select exact-center HOLD,
even when forward progress had not been ruled out.  V15 therefore changes only
those controller decisions: its relative-residual stop is the one-sided
confidence test \(U(P_j)\le rL(P_0)\), and an unresolved possible L-BFGS
advantage falls back to H0 whenever H0 has a strictly positive improvement
lower confidence bound.  If that required fallback H0 lower bound is
nonpositive with reducible uncertainty, HOLD remains allowed; the nonselected
L-BFGS lower bound cannot bypass an unresolved takeover comparison.

The V14 outcome is revision rationale only.  V15 does not insert a visually
attractive parameter or tune from the V14 rankings.  It repeats the complete
V14 factorial unchanged and asks three independent control questions:

1. Is the original \(13\times10^{-5}\) normal scale best retained, reduced by
   \(3\times10^{-5}\), or increased by the same amount?
2. Does rebuilding the frozen outer problem after a 60% or a 30% statistically
   resolved residual reduction give the better path?  These correspond to
   \(r=0.4\) and \(r=0.7\).
3. Does a short or long frozen-frame solve work better?  The admissible integer
   caps are \(m_{\max}=4\) and \(8\), both above the user's lower bound of 3.

Their Cartesian product has 12 parameter combinations.  All other numerical
choices, the 2,000-query horizon, and the acceptance rules are fixed.  Each
combination is first run on the deterministic FES and then on the same three
calibrated-noise streams.  Common random numbers make differences between
parameter combinations easier to attribute to the parameters rather than to a
lucky synthetic draw.

The algorithm does not see \(I_\perp\), the free-energy barrier, the reference
path, current IVM, or FENEB during an optimization.  Afterward, every public
path is independently scored on the frozen deterministic FES.  This separates
"the controller survived realistic noise" from "a noisy measurement happened
to look good."

The unchanged full screen deliberately mirrors the requested scientific claim.  Through
1,600 query equivalents, the three-noise-stream median must beat current IVM
at every checkpoint in the normal residual, article arclength error, and
barrier error.  From 1,620 to 2,000, the two MD-sensitive quantities are judged
by win fraction, window median, normalized AUC, and two-of-three individual
repetitions; article arclength remains a deterministic geometric gate.  A
separate deterministic run cannot fail and be rescued by noisy repetitions.

If no parameter combination passes, the procedure calls that an algorithmic
failure, not a parameter winner.  The method is then revised before any new
restrained MD.  If a refinement is necessary, it must be declared as a whole
bracket/factorial before its outcomes are inspected.  The machine-readable
source of these rules is `registrations/DEVELOPMENT_SCREEN_PLAN.json`.

This screen is development-only.  Its deterministic and synthetic calls are
query equivalents, not formal mean-force evaluations, and neither the runner
nor analyzer starts restrained MD or writes a formal PDF.
