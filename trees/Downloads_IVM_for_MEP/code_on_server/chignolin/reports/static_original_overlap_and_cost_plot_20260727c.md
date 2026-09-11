# Chignolin Static-NEB / Original String audit and cost-plot revision

Date: 2026-07-27

## Conclusion

The near overlap of Static-NEB and Original String is not caused by sharing
the same update routine or by an accidental tangent projection in Original
String.

The independently tuned production configurations happen to use the same
explicit step coefficient, `0.0125`, and both keep the band close to uniform
arclength. Their normal displacement is therefore almost the same, while
their main algorithmic difference acts in the tangential/gauge direction.
The plotted normal residual and energy barrier are largely insensitive to
this reparameterization difference.

## Implementations audited

### Original String

Source:

`/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/src/run_string_family.py`

Production configuration:

- `dt = 0.0125`;
- one full physical-force update on every internal image;
- no tangent projection in the descent step;
- Appendix cubic-spline equal-arclength reparameterization after every step.

The update before reparameterization is exactly

```text
x_i <- x_i + dt * F_i
```

### Static-NEB

Sources:

- `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/src/run_neb.py`
- `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/src/chignolin_core.py`

Production configuration:

- explicit static Euler coefficient `alpha = 0.0125`;
- spring coefficient `k = 35`;
- centered-difference tangent;
- standard NEB force: physical normal force plus tangential spring force;
- no String cubic reparameterization.

The per-image update is

```text
x_i <- x_i + alpha * [
    F_perp,i + k * (|x_{i+1}-x_i| - |x_i-x_{i-1}|) * tau_i
]
```

A direct synthetic-force audit reproduced this expression exactly.

## Why the two physical curves overlap

For a nearly equal-arclength band and a small step, cubic reparameterization
removes the tangential part of the full-force String displacement:

```text
R_cubic(x + dt * F)
    = x + dt * F_perp + a tangential gauge redistribution + O(dt^2).
```

Static-NEB applies the same first-order normal displacement and uses its
spring term to control the tangential gauge. Starting from the same exactly
equal-arclength IDPP band, the two algorithms therefore trace almost the same
geometric path.

They are not numerically identical:

- RMS coordinate difference at call 2: `0.004690 Å`;
- RMS coordinate difference at call 499: `0.001063 Å`;
- final relative path L2 difference: `5.71e-5`;
- largest relative difference over the plotted raw `I_perp` curves: `0.405%`;
- largest relative difference over the sparse barrier curves: `0.488%`.

Their parameterizations are visibly different:

- Original String final segment-length CV: `2.11e-6`;
- Static-NEB final segment-length CV: `6.20e-4`;
- Original String final equidistance integral: `0.525`;
- Static-NEB final equidistance integral: `408.10`.

Thus the overlap in `I_perp` and barrier is an expected near-equivalence of
the geometric paths, not duplicated code.

## Revised cost-to-precision figure

Output:

`/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/chignolin/figures/paper_methods_total_nf_20260727c/chignolin_cost_to_precision_total_nf.pdf`

Changes:

- record-setting measured Pareto vertices are joined by straight lines;
- no artificial threshold-grid staircase is drawn;
- the vertical axis is linear;
- cost is the total whole-band Number of Force Evaluations over all 22 images;
- black double-headed arrows show same-precision IVM speedups relative to
  ODE-NEB and OBSM-FIRE;
- IVM and ODE-NEB retain the manuscript blue/orange colors;
- Static-NEB, Original String, and OBSM-FIRE use distinct line styles and the
  remaining colorblind-safe colors.

Arrow values:

| Metric | Shared threshold | IVM cost | Baseline | Baseline cost | IVM speedup |
|---|---:|---:|---|---:|---:|
| `I_perp` | 0.4642178 | 802 | ODE-NEB | 10002 | 12.5x |
| `I_perp` | 0.4642178 | 802 | OBSM-FIRE | 9482 | 11.8x |
| Densified barrier | 25.4903 kcal/mol | 902 | ODE-NEB | 10002 | 11.1x |
| Densified barrier | 25.4903 kcal/mol | 902 | OBSM-FIRE | 7062 | 7.8x |

The shared threshold in each panel is the strictest threshold attained by all
three compared methods, so neither baseline is extrapolated.

## Validation

- 53 project tests pass.
- The Original String full-force test passes.
- The central-difference NEB tangent test passes.
- The static NEB projection/spring formula passed a direct numerical audit.
- The PDF is one page with embedded fonts.
- All generated SVG files parse successfully.

Revised plotting-source SHA-256:

`eef0575af6ec1db1cfd63d216a2d8b0192e561d77b506c1dbefb3edef2008108`

Cost-to-precision PDF SHA-256:

`03192d4e3d52d7c899f85815b09308013e37d056a771b053e9f96f3bd72e05e0`
