# Barnase--Barstar IVM20k / ODE-NEB30k discrete cost-to-precision

Status: `PASS_VISUALLY_CONFIRMED_300_DPI`

## Output

- PDF: `/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm20k_ode30k_cost_to_precision_1x3.pdf`
- PDF SHA-256: `f4d132b858b9e8ba7c667a2e534861059b6be8b6723f6615324b6a059ebbd51a`
- Data: `/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm20k_ode30k_cost_to_precision_1x3.data.json`
- Data SHA-256: `4f15b7457b85a1c3fc6c014fb20ba89ef842028c128a1ef40637ae03342800a2`
- Analysis and plotting code: `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/plot_bb_irivm20k_ode30k_cost_to_precision_1x3.py`
- Code SHA-256: `138984019a4f7d54d174fe8682a576724c82e16cfac53da0c24b621cab5bc3ee`
- Provenance JSON: `/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm20k_ode30k_cost_to_precision_1x3.provenance.json`

## Formal method histories

- `IVM` is the selected BB IR-IVM algorithm (inexact update plus equal-arclength
  reparameterization; alpha=1, r=0.1, m_max=5). The figure includes every R6B
  formally reportable state with `N_F <= 20000`: calls 1--693, ending at the
  actual `N_F=19993`. Calls 1--515 use locally stored R5B shards whose path,
  energy, and Force hashes are independently checked against the R6B inventory;
  calls 516--693 use a compact 178-row ledger computed directly beside the
  remote external-disk R6B shards and synchronized through the existing
  Mutagen project session. Every compact row carries its source shard and
  path/E/Force hashes. No state after 20k and no terminal-uncommitted tail is
  included.
- `ODE-NEB` is ASE `NEBOptimizer(method="ODE")` with the selected
  `k=0.1 eV A^-2`. All 1000 paid interior-image batches are used through the
  exact `N_F=30000`. After a rejected trial, the already accepted current path
  remains the reported state while the 30 paid Forces remain counted.
- Fixed endpoints were inherited and add zero optimization Forces.

## Metrics

All three metrics use the same saved 32-image state, with no densification,
interpolation, smoothing, or mass weighting.

- `I_perp`: centered path tangent, normal component of the full stored Force,
  and local Cartesian Voronoi weights.
- `Delta E`: `max(E_0,...,E_31)-E_0`, converted with
  `1 eV = 23.0605 kcal mol^-1`. The plot label is simply
  `Delta E`, as requested.
- Discrete action `S_hat`:
  `S_hat=sum_i 0.5*(||F_i||+||F_(i+1)||)*||x_(i+1)-x_i||` over the 31 native
  Cartesian segments. Because OpenMM stores `F=-grad V`, `||F||=||grad V||`.

The curves are strict cumulative-best first-passage frontiers. Neither the
metric nor `N_F` is interpolated. This offline calculation performs zero new
energy/Force evaluations and adds zero `N_F`.

## Display geometry

All three metric axes use the same reversed logarithmic convention.  The
display windows are `I_perp: 1e4 to 5`, `Delta E: 1000 to 80 kcal mol^-1`, and
`S_hat: 1e5 to 3e3 kcal mol^-1`.  These windows focus the shared
convergence regime; the strict first-passage thresholds and ratios are still
computed from each method's complete saved frontier before display clipping.

## Numerical closure

- IVM: 693 formally reportable stored checkpoints, final `N_F=19993`.
- ODE-NEB: 1000 paid-batch states, final `N_F=30000`.
- Strictest common `I_perp`: `83.5449690445`;
  first passage IVM/ODE-NEB = `5104` /
  `29970` Forces.
- Strictest common `Delta E`: `188.691332623`
  kcal mol^-1; first passage IVM/ODE-NEB =
  `5163` /
  `30000` Forces.
- Strictest common action: `9869.25889319`
  kcal mol^-1; first passage IVM/ODE-NEB =
  `5223` /
  `29940` Forces.

Every plotted row records its path, energy, and Force SHA-256 in the data JSON.
Every IR source shard and every ODE paid transport shard is hash-checked before
its values are admitted. The ODE decoder additionally reconstructs every XOR
delta block bit-exactly and verifies all declared logical-array hashes.
