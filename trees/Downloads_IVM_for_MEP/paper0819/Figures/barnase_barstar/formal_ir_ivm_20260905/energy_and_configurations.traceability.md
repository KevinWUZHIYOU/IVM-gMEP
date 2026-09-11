# BB IVM-20k / ODE-NEB-30k configurations and energy profile

Status: `PASS_FORMAL_STORED32_ZERO_PHYSICS_RENDER`

## Figure meaning

The energy-profile panel compares the accepted IR-IVM call 693 state at optimization $N_F=19,993$ with the formal ASE ODE-NEB state at $N_F=30,000$. Both curves connect their 32 stored energies directly; no densification, interpolation, new energy evaluation, or new force evaluation was used.

All four molecular panels are configurations from the accepted IR-IVM $N_F=19,993$ path. In bound-to-separated zero-based path indexing they are `[7, 5, 3, 0]`; in the derived separated-to-bound multimodel PDB they are one-based frames `[25, 27, 29, 32]`. They denote pre-barrier, barrier region, post-barrier, and the bound complex, respectively.

## Sources

- IR-IVM state: `/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/reports/plot_inputs/bb_irivm_nf20k_nf30k_20260904_r1/call_0693_stable_last.npz`
- IR-IVM primary history audit: `/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/reports/verifications/BB_IRIVM_SELECTED_FULL_STATE_HISTORY_REPLAY_NF30000_20260904_R6B_PRIMARY.json`
- ODE-NEB state cache: `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/formal_discrete32_cache_r2_audited_v2_local_mutagen/formal_current_state_nf30000.npz`
- ODE-NEB comparison pack: `/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/active_candidates/barnase_barstar_discrete32_ode_nf30000_r2_comparison.json`
- ODE-NEB terminal audit: `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/terminal_audits/fresh-nonresume-bb-neb-aseode-k0p1-formal-nf30000-remote-20260903-r1.terminal-audit.json`
- PDB topology/template: `/Users/wuzhiyou/IVM_for_MEP/python/barnase_barstar_ir_ivm_20260901/inputs/complex_fixed.pdb`
- Renderer: `/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/plot_bb_irivm20k_ode30k_snapshots_energy.py`
- Machine-readable provenance: `/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm20k_ode30k_snapshots_plus_energy.provenance.json`

## Accounting

This plotting workflow made 0 physical energy/force calls, constructed 0 OpenMM contexts, and contributed 0 to optimization $N_F$. ChimeraX was used only to render cartoons from already saved Cartesian coordinates.
