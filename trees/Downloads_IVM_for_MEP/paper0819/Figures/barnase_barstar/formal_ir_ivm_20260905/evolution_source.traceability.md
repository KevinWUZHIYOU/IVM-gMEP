# Barnase--Barstar constrained-relaxation evolution: traceability

Updated: 2026-09-05 01:01 HKT (Asia/Hong_Kong).

This figure overlays the current final-panel paths---IR-IVM target 20k at actual optimization `N_F=19,993` and ODE-NEB at `N_F=30,000`---on the completely converged fixed-subspace 15 × 10 diagnostic surface. The background surface itself remains centered on the saved IR-IVM call-515 path at `N_F=14,983`; it is not an unconstrained global potential-energy surface. That background center is the fully evaluated terminal returned state rather than a subsequently committed reparameterized state, whereas the overlaid IR-IVM 20k path is the separately audited, formally reportable call-693 state with no terminal uncommitted charged tail.

## Final figure and machine-readable record

- [Final PDF](/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution.pdf): square-panel globally smooth fitted canonical, SHA-256 `8dff0c767361bd10f1439e1ca55556a58430652a053e5a822066f2f2fbf2f23b`, 114,481 bytes.
- [Machine-readable provenance](/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution.provenance.json): SHA-256 `37e6acdcbe881a1023f1526aa2397bd06b2059aa09d7e8bd969739adcc9ba144`, 24,794 bytes.
- [Globally smooth fit canonical manifest](/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution.smooth_global_fit.manifest.json): SHA-256 `4be598f15b9a2caee99a83552a4b5c09347854f87420f149951d60a63748e6c0`, 4,902 bytes.
- [Globally smooth fit builder](/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/plot_bb_r2_smooth_c2_gamma15_to_linear_preview.py): SHA-256 `83f043bee06f199c575eda3da41c8e2e3bd53a1fe2b2532a52acbbdc8489af92`, 25,292 bytes.
- [Seven-variant sweep manifest](/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution_square_all_points_curvature_sweep.manifest.json): SHA-256 `4054010f5a4edfce85367275025717004c6b979d0c18ed99dde76468cf497deb`.
- [Square/formal-BB-style sweep plotting wrapper](/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/plot_bb_r2_square_all_images_curvature_sweep.py): SHA-256 `eda539ef44635446d4bf329a0c019ae3db7f4adcd63ad914fbef4bc75347afe5`.
- [Validated R2 base plotting wrapper](/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/plot_bb_r2_merged_relaxed15x10_evolution.py): SHA-256 `040f5e996f5c23b9a8f34b37a3fc7c1540f076e17c2e9ff1f5b770f6503e69cb`.
- [Established plotting implementation](/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/bb_neb_aseode_kgrid_campaign_nf6000_nf30000_20260903_r1/analysis/plot_bb_relaxed15x9_diagnostic_evolution.py): SHA-256 `0e396771c15c11d1cb0eea0133f1b6bd0f325d0f68d3c236a4877a25352ba756`.

The JSON provenance binds the output PDF, every final data product, every plotted path snapshot, the plotting and postprocessing code, the path-style reference, and the cross-host transfer receipts. Its output SHA equals the PDF hash above. The sweep manifest binds all seven PDFs and all seven per-output provenance files.

## Surface construction

The diagnostic-surface center is the saved, fully evaluated 32-image IR-IVM path at call 515 and optimization `N_F=14,983`. This center is distinct from the final-panel IR-IVM call-693 overlay at `N_F=19,993`. The center path was resampled by Cartesian arc length to 59 positions. Interior tangents use centered differences; endpoints use the adjacent segment. At each position, the normal is the initial-path displacement after removing its tangent component and normalizing it.

The displayed grid is a deterministic subset:

- 15 longitudinal values s between 0 and 1;
- 10 path-normal RMS offsets n = [-0.12, -0.06, 0, 0.12, 0.24, 0.36, 0.48, 0.60, 0.69, 0.72] Angstrom;
- 150 configurations: 148 freshly relaxed configurations and two cached fixed endpoints at (s,n) = (0,0) and (1,0).

The original 15 × 9 coordinates and additional measured n = 0.72 row are bound through [prepare_bundle.py](/Users/wuzhiyou/IVM_for_MEP/python/bb_relaxed_surface_continuation_20260903_r1/prepare_bundle.py). The added row is evaluated data, not extrapolation.

## Constrained relaxation algorithm

Each non-endpoint configuration was relaxed independently with the frozen FIRE implementation in [run_continuation.py](/Users/wuzhiyou/IVM_for_MEP/python/bb_relaxed_surface_continuation_20260903_r1/run_continuation.py), using the same physical backend, topology, energy, force, projector, and optimizer constants throughout.

No subset of atoms is frozen. For all 3,132 atoms, force, velocity, and displacement are projected onto the complement of the fixed tangent tau(s) and normal nu(s):

`P = I - tau tau^T - nu nu^T`, and `F_relax = P F`.

The constraints `(R-R0)·tau = 0` and `(R-R0)·nu = 0` preserve the selected (s,n), while the remaining 3N-2 Cartesian directions relax. Saved states and the final merge require constraint drift at most 1e-10.

Frozen FIRE parameters are fmax = 0.10 eV/Angstrom, dt0 = 0.10, dtmax = 1.00, finc = 1.10, fdec = 0.50, alpha0 = 0.10, falpha = 0.99, nmin = 5, a maximum per-atom displacement of 0.10 Angstrom per update, and 500 total update steps per configuration. Convergence means max_i ||F_relax,i|| <= 0.10 eV/Angstrom. This is a relaxation stopping test, not the path-integrated article metric I_perp.

## Continuation, saved quantities, and validation

The original surface calculation had already made 7,023 joint energy/force calls. It left 24 of the 148 fresh configurations converged and 124 at the 50-update cap. The continuation did not restart them. It restored the exact saved coordinate, energy, raw force, projected force, velocity, time step, FIRE mixing parameter, and positive-power counter. The already saved terminal force was advanced exactly once before requesting the next energy/force value.

An earlier closed continuation checkpoint solved 43 of those 124 configurations with 4,224 additional joint E/F calls. The remaining 81 configurations were frozen into disjoint native-host assignments: 54 on `Mac-mini.local` and 27 on `wuzhiyoudeMacBook-Air-206.local`. The local share was divided into seven batches of 4, 4, 4, 4, 4, 4, and 3 points.

| Native run | Points | New joint E/F calls | Converged | Failed/open |
|---|---:|---:|---:|---:|
| Remote Mac | 54 | 5,065 | 54 | 0 |
| Local 01 | 4 | 257 | 4 | 0 |
| Local 02 | 4 | 561 | 4 | 0 |
| Local 03 | 4 | 401 | 4 | 0 |
| Local 04 | 4 | 461 | 4 | 0 |
| Local 05 | 4 | 258 | 4 | 0 |
| Local 06 | 4 | 368 | 4 | 0 |
| Local 07 | 3 | 261 | 3 | 0 |
| **R2 disjoint total** | **81** | **7,632** | **81** | **0** |

Complete diagnostic accounting is 7,023 original + 4,224 inherited continuation + 7,632 disjoint continuation = **18,879 joint E/F calls**. These calls are recorded but have `optimization_N_F_contribution=0`: they construct the diagnostic background and are not added to the IR-IVM or ODE-NEB cost-to-precision axes.

Before every new physical call, an fsync'd WAL start record is written. Each successful call stores coordinate, energy, raw force, projected force, next coordinate, and complete outgoing FIRE state in a lossless NPZ shard with array and file SHA-256 bindings, followed by an fsync'd success record. Open or failed calls cannot be retried silently.

Every batch passed full replay on its native host. The final [merged result](/Users/wuzhiyou/IVM_for_MEP/python/bb_relaxed_surface_continuation_20260903_r1/r2_dispatch/partial_campaign/merged/result.npz) has SHA-256 `750bef9a1344960f10f40b8c4a65d6c0daf7c393d5bf01d90123db6163c81a13`; its [merge audit](/Users/wuzhiyou/IVM_for_MEP/python/bb_relaxed_surface_continuation_20260903_r1/r2_dispatch/partial_campaign/merged/audit.json) has SHA-256 `60e517c3f0ea24b08aa711dab4756c423c7fd4eea1ef10a360e7e02e16d15ea1` and status `PASS_MERGED_150_POINTS`. It records 150 points, 148/148 fresh points converged, zero points at the 500-step cap, zero missing points, zero overlapping points, and repeated-projection residual exactly 0.0.

The last configuration, point 148, converged at optimizer step 233 with projected maximum force 0.09860818081098599 eV/Angstrom.

## Final surface adapter and cross-host transfer

- [Canonical R2 surface adapter](/Users/wuzhiyou/IVM_for_MEP/python/bb_relaxed_surface_continuation_20260903_r1/postprocess_r2_merged_surface.py): SHA-256 `329bf9504fa9372005027611ce6b0c52f7803622f63014f56a5d0ecfa96b2c67`.
- [Surface grid](/Users/wuzhiyou/IVM_for_MEP/python/bb_relaxed_surface_continuation_20260903_r1/postprocess/r2_merged/surface_grid.npz): SHA-256 `2043159d3d1a51d8500910aaee31662770e47da37e5ba3568b90a7b1bc0d982c`.
- [Surface-source audit](/Users/wuzhiyou/IVM_for_MEP/python/bb_relaxed_surface_continuation_20260903_r1/postprocess/r2_merged/surface_sources.json): SHA-256 `a91f0bd16bc6aa7d06a6fe88027c33f38760dd01743876ad0571d23f6ed005b7`.
- [Final aggregate transfer tool](/Users/wuzhiyou/IVM_for_MEP/python/bb_storage_archive_20260903_r1/r2_final_aggregate_transfer.py): SHA-256 `540e09db14d28403f70b638130914257eff3448654852a4554f6e26133083aee`.
- [Transfer manifest](/Users/wuzhiyou/IVM_for_MEP/python/bb_storage_archive_20260903_r1/r2_final_aggregate/transfer.manifest.json): SHA-256 `36a96b4006e1100dc93c71bf9425f197e022cc5c3b57af83e1a6cc44dbbb5f4e`.
- [Local installation receipt](/Users/wuzhiyou/IVM_for_MEP/python/bb_storage_archive_20260903_r1/r2_final_aggregate/local_install.receipt.json) records `PASS_LOCAL_INSTALLED_AND_REVALIDATED_R2_AGGREGATE` for ten files totaling 154,919,921 bytes.

The remote host produced the merge and surface. Only the minimum audited aggregate was staged and returned through the registered Mutagen session. The local installation accepted absent or byte-identical targets only. The canonical adapter was then run again locally and found the saved NPZ and JSON identical. Postprocessing and plotting made zero physical calls and changed no saved physical values.

Two preliminary postprocess launches exited before output: the first because the adapter was not yet installed on the Python 3.8 remote host; the second because the initial adapter used `Path.is_relative_to`, unavailable in Python 3.8. The final adapter uses the equivalent `relative_to` guard. Nine local tests and a real remote Python 3.8 inside/outside-path save test passed before the successful formal run. Neither preliminary launch imported the energy backend or altered the merged data.

## Path snapshots and panel labels

| Panel label | IR-IVM N_F | ODE-NEB N_F |
|---|---:|---:|
| Initial path | 0 | 0 |
| 2k | 1,993 | 2,040 |
| 10k | 9,985 | 9,990 |
| final | 19,993 | 30,000 |

The `10k*` label denotes the fully evaluated call-352 state at actual `N_F=9,985`; its saved path was not committed by a subsequent final reparameterization. The final `IR-IVM 20k` path is instead call 693 at actual `N_F=19,993`; the provenance marks it formally reportable and `terminal_uncommitted_charged_tail=false`. Exact source shards and hashes for both states are bound in the machine-readable provenance.

## Path representation, square panels, and curvature sweep

Both the path artists and their colors are copied from [the formal original BB evolution PDF](/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/barnase_branch_local_true0k_evolution.pdf), SHA-256 `26625a3f4a3a23c4d4302be6283e8a07879d7ca6fe569abec276f2e0ee8f9394`, and its [plotting source](/Users/wuzhiyou/Downloads/IVM_for_MEP/code_on_server/barnase_barstar_chigstyle/src/plot_manuscript_candidate.py), SHA-256 `aa30cd8530c67df66de511868db88711c2614193396aefc93627d59be060151c`.

- Initial path: `#7B3294`, solid, 1.55 pt, 2.15 pt circle markers, `markevery=1`, z-order 5.
- ODE-NEB: `#D55E00`, solid, 1.45 pt, 2.15 pt circle markers, `markevery=1`, z-order 5.
- IR-IVM: `#0072B2`, solid, 1.75 pt, 2.15 pt square markers, `markevery=1`, z-order 6.
- The formal source does not override marker face, marker edge, or opacity. The fresh Matplotlib defaults are therefore retained: marker face and edge follow the line color, marker-edge width is 1.0 pt, and opacity is unset (fully opaque).
- Every curve is drawn directly from its saved native `(32, 9396)` Cartesian path. Each native image is projected to its closest point on the registered 59-point center polyline, using the corresponding interpolated normal only to obtain `(s,n)`. The line connects those 32 projected native images and `markevery=1` shows all 32 markers. There is no path resampling, path interpolation, or densification.
- The shared projected native range is `s in [0,1]` and `n in [-0.0031531518,0.7213065821]` Angstrom. The y-axis upper limit is therefore minimally expanded from 0.72 to 0.73 Angstrom so that no native marker is clipped; the measured background itself remains the unchanged `[-0.12,0.72]` Angstrom surface.
- Every panel has a measured physical box width/height of exactly 1.0, and the colorbar is vertically aligned with the square panels.
- The formal canonical uses one globally defined nonlinear map: a positive mixture of a linear map and a normalized Gaussian CDF. The fitted parameters are Gaussian center `110.56` kcal/mol, width `68.33` kcal/mol, Gaussian fraction `0.6176883752`, and linear fraction `0.3823116248`. There is no join at 200 or 240 kcal/mol and therefore no change of formula at either energy.
- The fit target preserves the relative `gamma=1.5` tick spacing through 200 kcal/mol while rescaling 200 kcal/mol to 72% of the colorbar; from 200 kcal/mol to the measured maximum it targets uniform color-coordinate progression. The actual fitted coordinates at 50, 100, 150, 200, 300, 400, 500, and 600 kcal/mol are `0.08646`, `0.30689`, `0.57159`, `0.70872`, `0.78426`, `0.83982`, `0.89535`, and `0.95088`. Thus 200--600 is visually regular rather than compressed into a short top segment.
- The map is analytic and C-infinity over the full displayed interval, with strictly positive derivative everywhere. A 100,010-point check found minimum derivative `5.553179405499118e-4` per kcal/mol and inverse round-trip error at most `4.55e-13` kcal/mol. It maps exactly from `(0,0)` to `(688.4553818247,1)`; the maximum anchor error is `0.02499` below or at 200 kcal/mol and `0.00694` above 200 kcal/mol.
- The seven archived curvature variants share identical surface energy samples, native path snapshots, four checkpoints, titles, limits, and labels. Their curvature changes only the 50--240 kcal/mol color allocation. The former curvature-2 member is preserved byte-for-byte at [its explicit variant path](/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution_square_all_points_curvature_2.pdf), SHA-256 `4ace8386b4d86ea5b3471d4fc1377382ff9d4439c9f198369ae6ac0248fe7401`; the seven-variant manifest now binds that address. No saved energy or path coordinate changed, and plotting made zero physical calls.

| Color curvature | PDF | SHA-256 | Bytes |
|---:|---|---|---:|
| 0.7 | [variant](/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution_square_all_points_curvature_0p7.pdf) | `0ae6ff14b33b35d291cce00a1b20b86fb78a5f8b148d30340987c8706a4b2e52` | 108,048 |
| 1 | [variant](/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution_square_all_points_curvature_1.pdf) | `03369aa89c7aa8b37603a2f04adbbf8e0ca1c5a61eca6e56f5e1b127802452fc` | 109,995 |
| 1.25 | [variant](/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution_square_all_points_curvature_1p25.pdf) | `2393e4b549ca28e0f83500c7da96a95f53bb91fe01ec33f53145e7c7f04944b6` | 108,605 |
| 1.5 | [variant](/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution_square_all_points_curvature_1p5.pdf) | `93d66d903578b1e59c68deb1bd72111001896280ce2ce3e4b98353e7943b99fe` | 108,308 |
| 2 | [preserved variant](/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution_square_all_points_curvature_2.pdf) | `4ace8386b4d86ea5b3471d4fc1377382ff9d4439c9f198369ae6ac0248fe7401` | 108,178 |
| 3 | [variant](/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution_square_all_points_curvature_3.pdf) | `30fcbda133feb0b0263961f4a74effea0b2d61ebc303ab129b3437a443217c84` | 105,668 |
| 4 | [variant](/Users/wuzhiyou/Downloads/IVM_for_MEP/output/pdf/barnase_barstar_irivm_constrained_relaxed_evolution_square_all_points_curvature_4.pdf) | `e973d83e54912558940fc5c6add0dd3fc7a1a2d38b3d30d8d149f47acd68ad4b` | 103,460 |

## Contour style and dashed-line fix

The style reference is [the formal original BB evolution PDF](/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/barnase_barstar/barnase_branch_local_true0k_evolution.pdf), SHA-256 `26625a3f4a3a23c4d4302be6283e8a07879d7ca6fe569abec276f2e0ee8f9394`.

The apparent dashed lines beside white contours were fill-polygon rendering seams, not scientific curves. The fix rasterizes only the filled-contour background at 600 dpi with antialiasing disabled. White contours, method paths, markers, axes, and text remain vector. White lines are regenerated directly from the internal filled-contour boundaries rather than from an independent level set.

The canonical output uses the globally smooth fitted normalization above. The archived sweep retains color curvatures 0.7, 1, 1.25, 1.5, 2, 3, and 4. The former 28 filled bands were multiplied by 0.6 and rounded to 17. The resulting 18 boundaries have 16 internal white solid contours, which also equals 27 former internal contours multiplied by 0.6 and rounded. White lines remain at 0.25 pt width and opacity 0.28 with no dash pattern. The background remains the fully sampled interval [-0.12, 0.72] Angstrom with no energy extrapolation; the plot axis alone extends to 0.73 Angstrom to retain the highest native marker. All saved relative energies are nonnegative with respect to the fixed bound endpoint, so no clipping exception was used.

Final PDF QA:

- the canonical is a one-page, unencrypted PDF 1.4 file, 498.605 × 154.574 pt; each archived sweep member is a one-page, unencrypted PDF 1.4 file, 527.59 × 154.051 pt;
- four raster image XObjects, one filled-background image per panel;
- seven path artists per variant, each with 32 stored points and 32 displayed markers; all six saved snapshots and the initial path have native Cartesian shape `(32,9396)`;
- the canonical and all seven archived sweep PDFs have zero nonempty dash-pattern operators: the three ODE-NEB paths, their legend handle, IR-IVM, the initial path, and all white contours are solid, matching the formal reference;
- embedded Arial/DejaVu fonts;
- the canonical 220 dpi render and all seven archived 220 dpi renders were inspected: panels are square, all native markers and the formal-BB colors are visible, white solid contours align with the color bands, dashed-looking fill seams are absent, and no nonconvergence cross or rendering gap appears within the sampled background. Canonical colorbar labels expose 50, 100, 150, 200, 300, 400, 500, and 600 kcal/mol on one side with no overlap, plus the exact top value in scientific notation.
