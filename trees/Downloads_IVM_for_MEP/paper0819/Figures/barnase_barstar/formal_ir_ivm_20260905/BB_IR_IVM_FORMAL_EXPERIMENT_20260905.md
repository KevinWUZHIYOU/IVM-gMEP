# Barnase--Barstar IR-IVM formal experiment registry

Status: `PASS_FORMAL_NATIVE32_IRIVM20K_ODE30K_20260905`

This directory is the provenance root for the Barnase--Barstar experiment used
in `paper0819/main.tex`.  It registers the selected inexact, reparameterized
IVM realization (IR-IVM) as the paper's formal BB IVM result and binds the
three replacement figures, their native 32-image data, experiment histories,
selection receipts, plotting code, and audits.

## Publication display policy

The manuscript and figure faces display force budgets as integer `k` labels.
Nonexact checkpoints use an approximation sign, for example
`N_F approximately 20k`; exact values remain here and in the JSON provenance.

| Displayed label | IR-IVM exact state | ODE-NEB exact state |
|---|---:|---:|
| Initial | 0 | 0 |
| approximately 2k | 1,993 | 2,040 |
| approximately 10k | 9,985 (call 352) | 9,990 |
| Final | 19,993 (call 693; displayed approximately 20k) | 30,000 |

The IR-IVM `10k*` state is fully evaluated but is not followed by a committing
final reparameterization.  The formal final IR-IVM state is call 693 and does
not belong to the terminal-uncommitted charged tail.

## Formal manuscript figures

| Role | Formal paper file | SHA-256 |
|---|---|---|
| Evolution on the constrained-relaxed local section | `../barnase_barstar_ir_ivm_evolution.pdf` | `8dff0c767361bd10f1439e1ca55556a58430652a053e5a822066f2f2fbf2f23b` |
| Native-32 cost to precision for three diagnostics | `../barnase_barstar_ir_ivm_cost_to_precision.pdf` | `f4d132b858b9e8ba7c667a2e534861059b6be8b6723f6615324b6a059ebbd51a` |
| Four IR-IVM configurations plus native-32 energy profiles | `../barnase_barstar_ir_ivm_energy_and_configurations.pdf` | `6032e6c9008b67f8813c4a8825b825d6bf4988004e930ca4c70b2a91f3aa14c9` |

Each formal file is byte-identical to its corresponding canonical source under
`/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/`.  The paper source is
`/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819/main.tex`, SHA-256
`9e714521bc5e16970b061eaa90f4f146b1115b1659b16fb5b197788beb390ce4`.
The verified 31-page preview is
`/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819/tmp/vscode-latex/main.pdf`,
SHA-256 `d5e0a078d3f283a38c135d07f3588502b5c9e6f33fc7b264f435d655423e9607`.
The formal root `paper0819/main.pdf` is not a build target for this registry.

## Common physical system and path

- 3,132 atoms and 9,396 Cartesian coordinates; 32 images, of which 30 move.
- Common repaired 1BRS-derived initial path and byte-identical fixed endpoints.
- OpenMM with Amber14 and GBN2 implicit solvent, without a nonbonded cutoff.
- Both methods and every diagnostic use the normalized central chord at an
  internal image, `tau_i proportional to x_(i+1)-x_(i-1)`.
- The two fixed endpoint energy/force values are treated as precomputed and add
  zero optimization Number of Forces.

## Formal IR-IVM algorithm and selection

The registered setting is `alpha=1`, `r=0.1`, `m_max=5`, `eta=1.1`, L-BFGS
history 7, identity metric, matrix-free normal projection, and a 1 Angstrom
maximum per-atom displacement cap.  After every outer iteration the frozen
mesh/readiness conditions are tested.  When eligible, equal-Cartesian-arclength
state-curve reparameterization is attempted; a rejected direct proposal may
enter the registered recovery process.  All fully evaluated direct and
recovery trials are charged.

The final setting is bound by the following copied selection chain:

| Receipt | Status/decision | SHA-256 |
|---|---|---|
| `experiment_receipts/BB_IRIVM_ALPHA_PRINCIPAL_SELECTION_20260901_R1.json` | two-principal-diagnostic alpha screen | `9f04553f390b5d6e6e2d1984047f2692e979e952b4d91854426c07d33abdbd69` |
| `experiment_receipts/BB_IRIVM_ALPHA_CONFIRMATION_SELECTION_20260901_R1.json` | alpha confirmation; alpha 1 advanced to interaction test | `e45fb79a37ac5ef87a86927ed0790490973c2ff4b6210a35f9f2cc4e3bd5f1a9` |
| `experiment_receipts/BB_IRIVM_R_SCAN_SELECTION_20260901_R1.json` | controlled r scan; r=0.1 advanced | `ba588e8516e297cc85a3852e337c53e6543cf394d56586c11dbfaaf923311a97` |
| `experiment_receipts/BB_IRIVM_M_MAX_SCAN_SELECTION_20260902_R1.json` | controlled m_max scan; m_max=5 advanced | `1c4cb4bc9b7a7dfc29f2e1fb798499413c57ba19e2e28ec79cfa91ebe36aa9b6` |
| `experiment_receipts/BB_IRIVM_LOCAL_INTERACTION_SELECTION_20260902_R1.json` | unique Pareto-nondominated winner alpha=1, r=0.1, m_max=5 | `8e5094fc0075847001f00f1b9b6e0ea532608eb17e77aa4da76e9411edc32b21` |
| `experiment_receipts/BB_IRIVM_LONG_BUDGET_SELECTION_20260902_R1.json` | selected setting frozen for an independent long run | `b91e30de47727cd7f6f8c1211f53d1ae4973e2bacc4f9a3996bb97c5980aec5d` |

The replay/run closure is copied as
`experiment_receipts/BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF30000_20260904_R6B.json`
(`0c0c1d9df6113bc67eedb83ea0a6a2b5abc8b3042acb6d6913dfb702a351c5b2`),
with primary verification
`00a215f4e2953a551b7d5294780f65368b3ce4545dcd28f4623dd2645a7c1345`
and supplemental verification
`2e0516dc2a2cdeb2f72f3fcdb3b5d4b9f5ba5801e3e962bebfe2d845e141a556`.
The registered hard cap was 30k, the actual sealed counter was 29,978, and
the last complete formally reportable state was call 818 at 23,512.  Calls
819--1053 are a segregated terminal-uncommitted charged tail and are excluded.
All formal paper figures stop earlier, at call 693 and 19,993.

Key IR-IVM experiment code:

| File | SHA-256 |
|---|---|
| `/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/run_barnase_barstar_ir_ivm_member.py` | `ebb9d2c52d5e3a12c477c0e735797d9aff5fee5df311b8c84f87f0b39f4cfd85` |
| `/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/scripts/run_selected_ivm_full_state_history_replay_r6_nf30000.py` | `a33ff86c937d91fdcbb16516dc25021ddae7ca1ea37ac2246aefd371bb55bd93` |
| `/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/run_ivm_v9_discrete_selected_eta_shim.py` | `a2f071ea787a008ed001fd806b8952f84d8db90335b544308478f3f6ae2facf9` |
| `/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/src/run_ivm_v9_spring_projected_production_20260730.py` | `991c11ff1f8eab014a6e54b4bf8ba610b0e52d071b202dc131b03bfabe6143fa` |

## Formal ODE-NEB algorithm and selection

ODE-NEB uses ASE `NEB(..., method="aseneb")` force assembly and
`NEBOptimizer(method="ODE")`, without a climbing image.  The frozen BB
adapter overrides the path tangent with the same normalized central chord used
by IR-IVM; this is not a claim about ASE's unmodified default tangent.
The spring scan tested `k={0.1,1,10,100} eV Angstrom^-2` to 6k.  A
prospectively frozen time-averaged-regret/Pareto rule on `I_perp` and `Delta E`
selected the unique nondominated `k=0.1`, which was restarted from the common
initial path and run fresh to exactly 30k.

- Selection receipt:
  `experiment_receipts/BB_ASEODE_KGRID_SELECTION_NF6000_20260903_R1.json`,
  SHA-256 `4da93e5f277acfa3e9f0dc3a170968abd081dba9689bac4cb60eb1edd60551fc`.
- Terminal audit:
  `experiment_receipts/fresh-nonresume-bb-neb-aseode-k0p1-formal-nf30000-remote-20260903-r1.terminal-audit.json`,
  SHA-256 `e8a3f797b5c2d0ef31b36e9222a14a2fb7f23e58e0ae220cca6af4bba648f579`.
- Production entry point:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/bb_neb_aseode30_kgrid_nf6000_nf30000_20260903_r1/scripts/run_kgrid_openmm_production.py`,
  SHA-256 `cf2d6c495e9fa19f838c0d5918da863b78b428cc89be3186a870374a65301188`.
- Central-tangent ASE adapter:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/bb_neb_aseode30_kgrid_nf6000_nf30000_20260903_r1/src/ase_neb.py`,
  SHA-256 `cccd779e8af159e24a7e5f86ca7cc479dd3b2c549a2d3438bd0dac9a4875da38`.
- Campaign runner:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/bb_neb_aseode30_kgrid_nf6000_nf30000_20260903_r1/src/kgrid_runner.py`,
  SHA-256 `2ad0c0332c767a3210022971b223baef8c6ac3e7cf1e82cff399e6cb38f8b5ea`.

## Number-of-Forces policy

One physical joint energy/force query on one moving image contributes one to
`N_F`.  Rejected IR-IVM direct/recovery trials and rejected ODE trial batches
remain charged; exact cache hits add zero.  The two inherited fixed endpoints
add zero.  Therefore IR-IVM checkpoints need not be multiples of 30, whereas a
complete ODE-NEB batch contributes 30.  Capturing, decoding, metric evaluation,
plotting, and verification made zero new physical calls and add zero `N_F`.

## Formal native-32 diagnostics

For every method/checkpoint, the three diagnostics use that checkpoint's same
saved 32-image path, energy, and force arrays.  There is no path densification,
metric interpolation, smoothing, or mass weighting.

- `I_perp = sum_(i=1)^30 w_i ||(I-tau_i tau_i^T) grad V(x_i)||^2`, with
  `w_i=(ell_(i-1)+ell_i)/2` and `ell_i=||x_(i+1)-x_i||`.
- `Delta E = max_(i=0..31) V(x_i)-V(x_0)`.
- `S_hat = sum_(i=0)^30 0.5 (||F_i||+||F_(i+1)||) ell_i`.
- Energy and action use `1 eV = 23.0605 kcal mol^-1`.

The cost curves are strict cumulative-best first-passage frontiers with no
interpolation of metric or force cost.  At the lowest recorded threshold
attained by both methods, the exact first passages and ODE-to-IVM ratios are:

| Diagnostic | Common threshold | IR-IVM `N_F` | ODE-NEB `N_F` | Ratio shown |
|---|---:|---:|---:|---:|
| `I_perp` | 83.5449690445 | 5,104 | 29,970 | 5.9x |
| `Delta E` (kcal mol^-1) | 188.691332623 | 5,163 | 30,000 | 5.8x |
| `S_hat` (kcal mol^-1) | 9,869.25889319 | 5,223 | 29,940 | 5.7x |

## Figure-specific source closure

### Cost to precision

- Frozen data: `cost_to_precision.data.json`, SHA-256
  `4f15b7457b85a1c3fc6c014fb20ba89ef842028c128a1ef40637ae03342800a2`.
- Provenance: `cost_to_precision.provenance.json`, SHA-256
  `e11be549d5f5d7528b468302439561bf06b236b3c8219e9a553001c02d473958`.
- Trace: `cost_to_precision.traceability.md`, SHA-256
  `43f47c25fb0fd1c71c494739be3b432a29658a52bc4c931a656d0d61f6833104`.
- Visual/structural audit: `cost_to_precision.audit.json`, SHA-256
  `3ff1a7c6e9d5fa489bf12bf348f7ea090aad63da608f8515bb9d351389f615ac`.
- Plot/analysis code:
  `plot_scripts/plot_bb_irivm20k_ode30k_cost_to_precision_1x3.py`, SHA-256
  `138984019a4f7d54d174fe8682a576724c82e16cfac53da0c24b621cab5bc3ee`.
- Compact call-516--693 IR ledger:
  `ir_ivm_calls0516_0693_native32_metrics.json`, SHA-256
  `0a4abdb4b1661e492749bf0798929a8d877fee085882284b93aba1ed7b4d9ccd`.
- Its zero-physics generator:
  `plot_scripts/generate_bb_irivm_r6b_20k_suffix_metrics_remote.py`, SHA-256
  `c2bff2b466680e15536d350d3a7fc0cbca5aef307b835a107c46b6dbc65095e3`.

The cost history contains 693 formally reportable IR-IVM checkpoints through
19,993 and all 1,000 paid ODE batches through 30,000.  Repeated current states
after rejected trials are valid paid checkpoints, not 693 accepted moves.
All three displayed metric axes use the same reversed logarithmic convention;
panel c writes the discrete action simply as `S_hat`, with discreteness stated
in words and in the quadrature definition rather than as a symbol subscript.
The display windows do not alter the full-history first-passage calculations.

### Configurations and energy profile

- IR-IVM call-693 state:
  `/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/reports/plot_inputs/bb_irivm_nf20k_nf30k_20260904_r1/call_0693_stable_last.npz`,
  file SHA-256 `c673536713cdcaf8040cdda0c1a5bf10786908c90fcdcf0b9d68c8e209b85f4c`;
  path/energy/force array SHA-256 values are respectively
  `854f2d6a85412b2d9a7f3c7cd0564dd11e12cdb4e7b62e304c4e76c3f090f3ee`,
  `244ea7f9275902a6746c68d9a69145cb3cc1376c80c859cfe1f64fa44b0df5d8`,
  and `3044705e23c3630b63ffbfb99fa6c1da4330f8c0a8800abab14f54acd384b2b6`.
- ODE exact-30k decoded cache:
  `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/formal_discrete32_cache_r2_audited_v2_local_mutagen/formal_current_state_nf30000.npz`,
  SHA-256 `89f48aa8ece0fdb86a3330b0449ab33b8165991fab864724622370dbf677d15b`.
- Provenance: `energy_and_configurations.provenance.json`, SHA-256
  `08b6c5339560fbd98cdb23f1d9e577d6e9c8827c0856628920ff9e6a76d23aef`.
- Trace: `energy_and_configurations.traceability.md`, SHA-256
  `0e180bde8c0be59a68c59b0869ebae99b7b055e9495ef9174c19f441bcfab323`.
- Plot code:
  `plot_scripts/plot_bb_irivm20k_ode30k_snapshots_energy.py`, SHA-256
  `b23359bdde7eab802e0db5c11ea040202628edaaba6b4f0efb7c8f405814ab93`.

The bottom panel joins the 32 stored energies directly and normalizes Cartesian
arclength separately for each path.  The four panels all come from IR call 693:
bound-to-separated zero-based indices `[7,5,3,0]`, corresponding to
separated-to-bound PDB frames `[25,27,29,32]`.  The exact derived PDB, ChimeraX
script, and four PNGs are frozen under `energy_and_configurations_assets/` and
are individually bound by the provenance JSON.  ChimeraX renders only saved
coordinates and makes no physical call.

### Evolution figure and diagnostic surface

- Provenance: `evolution_source.provenance.json`, SHA-256
  `37e6acdcbe881a1023f1526aa2397bd06b2059aa09d7e8bd969739adcc9ba144`.
- Full trace: `evolution_source.traceability.md`, SHA-256
  `67ad930f816d3e905fbb09824ea7418cb93e7e185578e478ebaae931f05c5c04`.
- Render manifest: `evolution_render.manifest.json`, SHA-256
  `4be598f15b9a2caee99a83552a4b5c09347854f87420f149951d60a63748e6c0`.
- Canonical plot code:
  `plot_scripts/plot_bb_r2_smooth_c2_gamma15_to_linear_preview.py`, SHA-256
  `83f043bee06f199c575eda3da41c8e2e3bd53a1fe2b2532a52acbbdc8489af92`.

The displayed 15-by-10 surface is a branch-local fixed-subspace diagnostic,
not a global or unconstrained PES.  Its coordinates are constructed around
the fully evaluated IR-IVM call-515 reference band at exact `N_F=14,983`.
For every nonendpoint grid point, `(s,n)` is held fixed and the remaining
9,394 Cartesian directions are relaxed by FIRE until maximum projected force
is at most 0.1 eV/Angstrom.  The merged 150-point result is
`/Users/wuzhiyou/IVM_for_MEP/python/bb_relaxed_surface_continuation_20260903_r1/r2_dispatch/partial_campaign/merged/result.npz`
(SHA-256 `750bef9a1344960f10f40b8c4a65d6c0daf7c393d5bf01d90123db6163c81a13`)
and its PASS audit is `audit.json`
(SHA-256 `60e517c3f0ea24b08aa711dab4756c423c7fd4eea1ef10a360e7e02e16d15ea1`).
The diagnostic surface used 18,879 separately recorded joint E/F calls; those
calls do not contribute to either optimizer's `N_F`.  Merge, postprocessing,
and plotting add zero calls.  Quantitative method comparison is therefore
based on the three native-32 diagnostics, not on this IR-IVM-centered display.

## Protocol supersession

Earlier BB `Arc500` and densified-barrier products remain historical diagnostic
records.  They are not inputs to these three formal figures and do not define
their metric contract.  For the 2026-09-05 formal BB result, the authoritative
contract is the native stored 32-image path with discrete `I_perp`, discrete
`Delta E`, and discrete action specified above.  This explicit supersession
prevents the historical densified and current native-32 values from being
mixed in one comparison.

## Reproduction and verification

From `/Users/wuzhiyou/Downloads/IVM_for_MEP`, using the recorded environment:

```bash
python code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/plot_bb_irivm20k_ode30k_cost_to_precision_1x3.py --mode all
python code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/plot_bb_irivm20k_ode30k_snapshots_energy.py
python code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/plot_bb_r2_smooth_c2_gamma15_to_linear_preview.py --promote-canonical --final-ir20k
```

The second command reuses the frozen molecular PNGs; add
`--render-configurations` only to regenerate those ChimeraX renders.  These
commands are offline consumers of saved data and add zero physical evaluations.
After regeneration, verify the hashes above and copy the canonical PDFs
byte-for-byte to the three formal paper paths.

Compile the manuscript only to its preview directory:

```bash
cd /Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819
/Library/TeX/texbin/latexmk -pdf -synctex=1 -interaction=nonstopmode -file-line-error -halt-on-error -outdir=tmp/vscode-latex main.tex
```

The final compile completed successfully.  The BB pages (20--22) were rendered
at 180 dpi and visually checked; the standalone PDFs were also rendered at
220--240 dpi and structurally audited.  All three are single-page,
unencrypted PDFs with embedded fonts.  Cost panels and evolution panels are
square, and no figure contains a precise noninteger NoF label on its face.
