# Source record: `chignolin_ir_ivm_overlay_total_nf.pdf`

Registered: 2026-09-06 (Asia/Shanghai).

This paper-local PDF is a byte-identical copy of the registered candidate figure:

`/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/chignolin/active_candidates/chignolin_ir_ivm_cpu1/chignolin_ir_ivm_overlay_total_nf.pdf`

PDF SHA-256: `06f402095523217f04ea2847e2e9b7a9d1f7d141fa807393be68b3f04c709396`.

The complete human-readable record is:

`/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/chignolin/active_candidates/chignolin_ir_ivm_cpu1/chignolin_ir_ivm_overlay_total_nf_provenance.md`

The complete optimization-algorithm, parameter-search, production-NPZ and
force-accounting audit is:

`/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/chignolin/active_candidates/chignolin_ir_ivm_cpu1/RESULTS_AND_PROVENANCE.md`

The paper-local copies of the machine record and plotted data are:

- `chignolin_ir_ivm_overlay_total_nf_provenance.json`, SHA-256 `bc56c16d94a950da4c4d618e39558e3dff8d62144cd88bf94633c81c305a291b`;
- `chignolin_ir_ivm_overlay_total_nf_frontier.csv`, SHA-256 `3705876f7a14861b242d6f65dfa027c2e30acb9d67e48840b0f16d945294b6ab`;
- `chignolin_ir_ivm_overlay_total_nf_annotations.csv`, SHA-256 `d714baaea66997aeeefb2b768e6ca846fc99b40518d79cec0501b1a71ce3fe6c`.

## Code and immutable diagnostic inputs

All experiment-relative paths below are rooted at:

`/Users/wuzhiyou/IVM_for_MEP/python/chignolin_ir_ivm_20260826`

- combined plotting code: `src/plot_chignolin_ivm_three_metrics.py`, SHA-256 `689e1a4322305a4e1b2d79ee9b557f2fa20625d6ac648550e91b9ca15c18bf73`;
- physical arc-length-500 evaluation and standardization: `src/postprocess_arc_length_500.py`, SHA-256 `607c0ceb41f3de4f6cf40d8094f37947bb687f72d4c516be219453cb5bbe4a17`;
- arc-length-500 loading, first-passage and style code: `src/plot_chignolin_ivm_arc500.py`, SHA-256 `62d30b5b81e0ab3b42bdf080308e95c615a4ecee0b960946b1df4dd15e0743b9`;
- geometric-action computation: `src/postprocess_geometric_action.py`, SHA-256 `959167006d816b4346b9b8a569d2349632a256309340c7ff558511855b6dc1e0`;
- tangent, quadrature and OpenMM backend: `src/chignolin_core.py`, SHA-256 `ad5d6532f7acd7c7b2560fdb68e6ead796374bcf0b24d1c0790b7e51fa5e4faa`;
- arc-length-500 diagnostic: `runs/diagnostics/chignolin_ivm_arc_length_500.npz`, SHA-256 `3def8ffb3c70342d61285813511203c2970e30cb0146f37aa1d99aa3ce20a564`;
- discrete-action diagnostic: `runs/diagnostics/chignolin_geometric_action_discrete_v1.npz`, SHA-256 `1c0acd1d423be9bddd77e3a4cade5008f4cff438e747d5bd83b3849e581e5921`.

The plotting code refuses unvalidated diagnostic hashes and cross-matches every arc-length-500 row against the action diagnostic by method, source record, optimization NoF, `I_perp`, selected-path SHA, source path and source SHA. The diagnostic calls are excluded from the optimization Number-of-Forces coordinate. The current redraw changes only the displayed geometric-action symbol from `S-hat` with a discretization subscript to `S-hat` without a subscript; plotted data, thresholds, first-passage counts and speedup annotations are unchanged. The pre-2026-09-06 two-panel artifact remains recoverable under the candidate directory's `previous_2panel_overlay_20260906/`.
