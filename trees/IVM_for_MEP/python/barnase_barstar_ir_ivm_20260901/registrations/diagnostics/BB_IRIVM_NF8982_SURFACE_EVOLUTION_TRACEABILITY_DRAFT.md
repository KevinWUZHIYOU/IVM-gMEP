# BB IR-IVM NF8982 surface-evolution traceability

Status: **prepared, not frozen**. The final figure contract must not be frozen
until the complete R4 `surface` result, WAL, and audit all pass. This document
does not authorize an OpenMM call.

## Scientific definition

- Background: the new `59 x 28` unrelaxed Cartesian 0 K section centered on
  the last complete selected IR-IVM path at optimization `N_F = 8982`.
- Surface coordinates: frozen in
  `runs/diagnostics/bb_irivm_nf8982_0k_section_coordinates_59x28_20260902_r2.npz`.
- Normal grid: `n_j = -0.12 + 0.03 j` Angstrom, `j=0,...,27`, with index 4
  stored as exact `+0.0`.
- Surface values: 1,652 stored energies = 1,650 fresh energy-only calls + two
  fixed-endpoint energies inherited from the canonical endpoint cache.
- Diagnostic forces: zero. Optimization `N_F` contribution: zero.
- The legacy `barnase_branch_local_true0k_evolution.pdf` is used only for
  visual style. Its PES and calibrated CV are not plotted.

## Curves and optimization cost

The final panels use verified stored Cartesian paths and the exact audited
optimization Number of Forces:

| Panel | IVM `N_F` | ODE-NEB `N_F` |
|---|---:|---:|
| b | 210 | 240 |
| c | 429 | 480 |
| d | 8982 | 9990 |

Each 32-image path is independently resampled to the frozen 59-point
Cartesian arc-length grid. Its plotted ordinate is
`dot(path(s)-center(s), normal(s))/sqrt(3132)` in Angstrom. Fixed endpoints
are set to exactly zero. No energy or force is evaluated during projection.

## Prepared code

- Final plotter:
  `scripts/plot_bb_irivm_nf8982_surface_evolution.py`
- Post-surface figure-contract freezer:
  `scripts/freeze_bb_irivm_nf8982_surface_evolution_figure_contract.py`
- Verified path loader (read-only dependency):
  `scripts/plot_bb_irivm_evolution_hashbound.py`
- R4 surface evaluator/validator:
  `scripts/evaluate_bb_irivm_nf8982_0k_section_energy_r4.py`

The freezer will produce, only after the surface passes:

- `registrations/diagnostics/BB_IRIVM_NF8982_SURFACE_EVOLUTION_FIGURE_CONTRACT_R1_20260902.json`
- `/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/active_candidates/barnase_barstar_irivm_nf8982_surface_evolution.pdf`
- the same-stem `.json` provenance record.

## Required final closure

The frozen figure contract directly binds SHA-256 identities for the plotter,
contract freezer, legacy style PDF, R2 coordinate manifest, R4 physical
contract and authorization, both R4 stage result/WAL/audit triples, the
verified path-source contract, and the verified path-loader code. The plotter
then replays both R4 result validators and the IVM/ODE stored-path gates before
rendering. Output publication is atomic and refuses overwrite.

## Zero-physical synthetic QA

The prepared dry-run command is:

```bash
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
  scripts/plot_bb_irivm_nf8982_surface_evolution.py \
  --synthetic-dry-run \
  --output /Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_nf8982_surface_evolution_synthetic_preview_v3.pdf
```

This preview is explicitly marked synthetic and is not scientific data.
