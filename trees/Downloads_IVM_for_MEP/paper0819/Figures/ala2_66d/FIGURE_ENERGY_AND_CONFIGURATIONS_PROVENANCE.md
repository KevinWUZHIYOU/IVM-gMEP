# Alanine 66D energy profile and ball-and-stick configurations

## Figure and reproducible source

This figure adds a molecular interpretation of the same final reported Ala
paths already used in the registered path-evolution figure. It uses the
production IVM run with `alpha=10`, `r=0.7`, and `m_max=5`, together with the
frozen ODE--NEB comparison. The four molecular configurations belong to IVM
and explicitly include its reactant and product endpoints.

| Role | Path |
| --- | --- |
| Figure PDF | `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_energy_and_configurations.pdf` |
| Figure PNG | `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_energy_and_configurations.png` |
| All 44 native-image profile values | `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_energy_and_configurations_source.csv` |
| Machine-readable manifest | `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_energy_and_configurations_manifest.json` |
| Exact coordinates and energies | `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_energy_and_configurations_assets/selected_paths_and_energies.npz` |
| Molecular raster assets and rendering transforms | `/Users/wuzhiyou/IVM_for_MEP/output/pdf/alanine66d_energy_and_configurations_assets/` |
| Figure builder | `/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_energy_and_configurations.py` |
| Coordinate-based molecular renderer | `/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/alanine_66d/code/plotting/render_ala_ballstick.py` |
| Existing registered path selection | `/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_grid_ivm_path_evolution_1x4.py` |
| Earlier path-selection registration | `/Users/wuzhiyou/IVM_for_MEP/paper/ActiveExperiments/alanine_66d/FIGURE_GRID_IVM_PATH_EVOLUTION_1X4_PROVENANCE.md` |

The new figure builder reads and verifies the frozen path-selector source,
original optimizer results, energy diagnostic, diagnostic contract, and
dihedral helper before plotting. The generated manifest binds the output
PDF, PNG, CSV, coordinate archive, both new code files, source energy cache,
PDB topology, and every molecular raster to their SHA-256 hashes.

Final output hashes and manuscript verification are recorded at the end of
this document.

## Optimizer paths and Number of Forces

The IVM run is the existing registered `inexact_reparam` experiment:

```text
alpha = 10
residual-ratio threshold r = 0.7
maximum internal steps m_max = 5
total optimization NoF budget = 5000
registered experiment identifier = a736dd230285d61c
```

Its result is
`/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/runs/stage4_ala_discrete_inexact_reparam_grid_v1/members/ivm_stage4_ala_discrete_inexact_reparam_grid_v1_a1e1_rr70_c05_tnf5000_r1_central_state_ivm_cfgc3a418a9c878_budget5000.npz`,
SHA-256 `0da1bbab469de81ca5b9229c4ebef656b7688ccb79c822ec14f0f4c92226d161`.

The plotted IVM path is `stored_paths[235]`, the last formally complete,
mesh-eligible state, at **3587 NoF**. Its path-array SHA-256 is
`b9517af023413fe127f151d2c92b24ac1df4e2fd8c02ecf7d52c7f7e6cea750a`.
Its recorded segment-length coefficient of variation is
`0.004611990669233385` and length ratio is `1.0220948376359082`, within the
registered limits `0.0065` and `1.03`. The raw 5000-NoF tail is partial and
uncommitted and is excluded under the existing reporting rule. The selected
path is not relabeled as convergence-certified.

The ODE--NEB result is
`/Users/wuzhiyou/IVM_for_MEP/python/alanine_v10_chignolin_v9_fixed_alpha_20260730/formal_figure_inputs/ala_0to1_N22_k50_alpha0.0001_ode_results.npz`,
SHA-256 `935561709ecbdfe87af31b78b45335a66545356b3e62f331a4a9fbb8125d9a09`.
The plotted path is `images_seq[4637]`, its terminal saved path, at
**100022 NoF**, with path-array SHA-256
`c3e43a1ecb1728e5262ce885a4c8a039f9843c0876a4281dc5f7ebc88c661178`.

Both paths have 22 molecular images, each with 22 atoms and 66 Cartesian
coordinates in Angstrom. IVM NoF is the saved
`total_internal_joint_evaluations` over the 20 movable internal images.
ODE--NEB retains the previously frozen convention `20 * count_seq + 2`.
The labels preserve these existing conventions. Display alignment, energy
cache reading, dihedral projection, and rendering add **zero optimization
NoF**.

## Cached energy source and exact alignment

The original optimizer archives do not store the full per-image potential
profile. The figure reuses the already completed geometric-action
diagnostic, which saved each native image's physical potential energy:

| Source | SHA-256 |
| --- | --- |
| `/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/runs/diagnostics/ala_discrete_geometric_action_v5.npz` | `d19d8efacf4b48b31ace4a87ff6857b04f65c6be9b3a1172db9e4506f58076eb` |
| `/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/runs/diagnostics/ala_discrete_geometric_action_v5.audit.json` | `797e24ffc28bd8684fccdb8b689089057dd6dd188ac53d80b383d387343c5ee3` |
| `/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/reports/ala_discrete_geometric_action_v5_preregistered.json` | `48eb99096e6aa1e569f6ef15a3b5c86dac0affcbaea2c563df42f60dcd55ffba` |

The IVM energy vector is `energy_ev[21]`; the ODE--NEB vector is
`energy_ev[4659]`. These row numbers are verified by method, source-result
hash, source record index, optimization NoF, and selected-path hash, rather
than assumed from row position. The stored segment-length vectors also
agree with direct differences of the selected original coordinates.

Each displayed energy is

```text
(energy_ev[image] - energy_ev[first_image]) * 23.0605 kcal/mol.
```

The energy zero is each method's own saved reactant image. The horizontal
coordinate is the cumulative polygonal arclength in the original
66-dimensional Cartesian coordinates divided by that path's total
arclength. It is calculated before any display alignment. The graph connects
22 native energy samples with straight segments as visual guides; it does
not smooth, densify, or interpolate the potential energy.

**No potential or force is evaluated to make this figure.** The 44 reused
native-image energies were previously evaluated as offline diagnostics,
whose historical cost remains excluded from optimization NoF. The plotting
and rendering code constructs no physical simulation context.

## Four IVM configurations

Configuration selection is deterministic and uses only the native IVM
profile: reactant; the first rising-branch image whose relative energy is at
least half the native peak energy; the unique highest-energy native image;
and product. The final selected indices are **`[0, 8, 12, 21]` (zero-based)**,
displayed as images **1, 9, 13, and 22**. The third displayed configuration is
labelled **Transition state**.

| Configuration | Zero-based image | Normalized arclength | Relative energy (kcal/mol) | phi (degrees) | psi (degrees) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reactant | 0 | 0 | 0 | -75.2432624907 | 57.0672145134 |
| Rising branch | 8 | 0.382001614232 | 1.491867798456 | -74.3072757986 | 97.4364405663 |
| Transition state | 12 | 0.572512505754 | 2.347065049327 | -98.5744287821 | 144.4809105614 |
| Product | 21 | 1 | 1.017492401654 | -152.5488661852 | 161.7449438147 |

The half-peak selection level is `1.1735325246634416 kcal/mol`.
Image 7 has `0.6524049687617356 kcal/mol`, below this level; image 8 has
`1.4918677984560047 kcal/mol`, above it. The native peak at image 12 is unique
to an absolute energy tolerance of `1e-10 kcal/mol`. The endpoint
configurations are the exact saved endpoints, not representative nearby
images or separately minimized structures.

## Native-image peak versus densified barrier

The native IVM peak is `2.347065049326883 kcal/mol`, and the native ODE--NEB
peak is `2.3085180568722725 kcal/mol`. These describe the 22 stored samples
and are **not** the densified energy barriers used in the cost-to-precision
figure. The Transition state configuration is selected by the highest stored
energy on this path.

For example, the existing registered densified calculation for exactly the
selected IVM path gives `2.580429425532039 kcal/mol` at 160 subdivisions per
segment; the difference from 80 subdivisions is
`0.00004279528106954089 kcal/mol`, below the registered `0.001` tolerance.
This higher value demonstrates why the native-image peak must not be
presented as densified Delta E.

That metric is recorded in
`/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/runs/diagnostics/stage4_ala_discrete_grid_metric_campaign_v2_1787608620070816000/tasks/88/88e60e0ddaca5ba742b57bb6b15e5137912f33c4cbfbefe3732c7853fb59be95.json`,
SHA-256 `60a2c918c29fa12626c0688241b0f4d0fc5f8c69a44febdf9103c5d9e566aef8`.
The task's source locator names another grid experiment (`m_max=50`), but
its complete path SHA-256 is exactly the selected `m_max=5` path's hash.
Six registered state references share this content-addressed metric. The
shared locator does not change the optimizer identity used in this figure.

## Molecular topology and phi/psi labels

The topology comes from
`/Users/wuzhiyou/IVM_for_MEP/python/alanine_ivm_clean_four_level_20260801/inputs/ala2_fixed.pdb`,
SHA-256 `7ffaf924e345b6fbbba06d0388a67d18f04c07729a2ab7d65ede9a61cd27eda2`.
Its ACE--ALA--NME molecule has 22 atoms and 21 covalent connections. The
renderer uses an explicit, fixed bond list in the PDB's atom order; it does
not infer connectivity from individual optimized configurations. This
connectivity was independently compared with the standard-residue PDB
topology parser, without creating a force-field System or simulation
Context. All displayed hydrogens are retained. The sticks convey covalent
connectivity, with each chemical connection drawn as one rod rather than a
chemical bond-order diagram.

Dihedrals use the existing article helper
`/Users/wuzhiyou/IVM_for_MEP/python/alanine_v10_chignolin_v9_fixed_alpha_20260730/src/plot_v49_formal_ala_1x3.py`,
SHA-256 `4af8feee0843206f17fc6f4b1b19eb52743719a4f553821a8a041ac9d06626dd`:

| Label | Zero-based atoms | Named atoms | Torsion axis |
| --- | --- | --- | --- |
| phi | `(4, 6, 8, 14)` | C(ACE)--N(ALA)--CA(ALA)--C(ALA) | N(ALA)--CA(ALA) |
| psi | `(6, 8, 14, 16)` | N(ALA)--CA(ALA)--C(ALA)--N(NME) | CA(ALA)--C(ALA) |

Values are signed dihedral angles wrapped to `[-180, 180)` degrees, computed
on the original saved coordinates before display alignment. All four
configurations show their actual phi and psi values. Three schematic,
transparent vector triangles are overlaid directly on each original
configuration in panel a:

| Plane | Zero-based atoms | Named atoms | Color |
| --- | --- | --- | --- |
| 0 | `(4,6,8)` | C(ACE)--N--CA | Teal `#2A9D8F` |
| 1 | `(6,8,14)` | N--CA--C | Amber `#D9A234` |
| 2 | `(8,14,16)` | CA--C--N(NME) | Purple `#A16AC0` |

Phi is the dihedral between planes 0 and 1, sharing bond `(6,8)` = N--CA.
Psi is the dihedral between planes 1 and 2, sharing bond `(8,14)` = CA--C.
The middle amber plane is deliberately shared by both definitions. The
phi and psi leaders terminate at the actual projected central-bond
midpoints; they do not indicate angles measured in the two-dimensional
picture. A compact color key identifies the atom triplet for each plane.

Each triangle vertex is the exact atom-center projection recorded by the
original molecular renderer. These translucent overlays are schematic
vector annotations, not opaque physical surfaces with depth ordering
against the spheres and rods. There are no separate dihedral panels,
enlarged surrogate molecules, or additional camera views. The four original
molecular PNGs retain their original bytes, view, scale, and aspect ratio.

## Rendering and graphical conventions

Molecules are rendered directly from the stored atomic coordinates as
analytically intersected spheres and finite cylinders. Each configuration
is aligned to the reactant using the same five backbone atoms
`[4, 6, 8, 14, 16]` and a proper Kabsch rotation with determinant +1, so no
reflection changes molecular chirality. All four share an orthographic
camera and common scale. The selected view only improves bond visibility;
it does not alter distances, energies, or measured dihedrals. Every rigid
transform and rendering input-coordinate hash is recorded in the asset
manifest.

Carbon is charcoal, nitrogen blue, oxygen red, and hydrogen pale gray.
Element colors identify atoms independently of the energy-curve method
colors. IVM uses the established blue curve and hollow-circle markers;
ODE--NEB uses the established orange curve and hollow-square markers.
Numbered configuration callouts associate the four molecular panels with
their exact native points on the energy graph. Following the legibility
revision, both sets of numbers use 9.5-point bold white type on matching
dark-blue circular badges (`#005A86`) with a white outline. The configuration
badges are placed immediately before the configuration titles, vertically
aligned with the title text;
the graph badges are 16 points above their corresponding IVM samples and
connected by short leaders. Only the numbering style changed: the original
molecular rasters, triangle annotations, selected coordinates, and all
44 profile values are unchanged.

The figure is designed at 180 by 112 mm in two rows. Axes, labels, curves,
markers, and the direct triangular plane overlays remain vector elements
in the PDF; molecular renders are embedded at 600 dpi. The four source
snapshot images are 1600 by 1200 pixels, with their aspect ratios preserved.
Panel a contains the four full configurations and their direct dihedral
annotations; panel b contains the energy comparison. The initial
snapshot renderer source is archived in the asset directory as
`render_ala_ballstick_snapshot_source.py`; its SHA-256 matches the renderer
recorded when the four unchanged snapshot PNGs were created. The current
renderer identity is separately recorded. Its standalone dihedral-view
helper and associated development assets are unused by the final figure;
the final plane annotations are drawn by the figure builder on the
original molecular views.

## Rebuild and validation

Run from `/Users/wuzhiyou/IVM_for_MEP`:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/ala_energy_configurations_mpl \
XDG_CACHE_HOME=/private/tmp/ala_energy_configurations_cache \
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python \
paper/ActiveExperiments/alanine_66d/code/plotting/plot_ala_energy_and_configurations.py \
--validate-only
```

Omit `--validate-only` to rebuild the molecular renders and final figure.
The optional `--reuse-renders` flag is appropriate only when the stored
rendering inputs and molecular PNG hashes pass verification, and the
recorded renderer hash matches either the current renderer or the archived
snapshot renderer source. The builder's input audit
passed on 2026-09-10 with the expected selections `[0,8,12,21]`, IVM 3587
NoF, and ODE--NEB 100022 NoF.

The target manuscript is
`/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819/main.tex`. After insertion,
build its preview without replacing the formal root `main.pdf`:

```bash
cd /Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819
/Library/TeX/texbin/latexmk -pdf -synctex=1 -interaction=nonstopmode \
-file-line-error -halt-on-error -outdir=tmp/vscode-latex main.tex
```

The manuscript figure label is `fig:Ala2_configurations`. The paper uses
the names Reactant, Rising branch, Transition state, and Product directly.

## Final registration and visual verification (2026-09-10)

The final figure uses direct plane annotations, not the superseded separate
dihedral-definition row. All 17 file bindings checked in the output manifest,
including the four original molecular PNGs, matched their recorded hashes.

| Registered artifact | SHA-256 |
| --- | --- |
| Figure PDF | `1a9e60d20d32df9ad3d06a7c711b6e4767e3d9de8b7dfd59ae342cfc0197ba43` |
| Figure PNG | `1e4b4846f013504942b2aba4834c70e6f78057627325e6d1885c73c6097c4090` |
| Native-image source CSV | `47a2a81de5e16197a25570afb21625f9b324016fa45bc29cc4bd0f58ece1a1f4` |
| Output manifest JSON | `0171fc8fedca8cddaf0cfeb975201f84ce8e9462b314460382a0e7af783ca162` |
| Exact paths and energies NPZ | `41728e3194469a8b83837accb70da6ba56a51c69399908d3fbc55c30007fbf59` |
| Figure builder | `e2cb12a781fdd8f153cd50d3e542515ad7350d6fee1756b1c454148078a5312b` |
| Current molecular renderer | `1e440e276d99036642276dfb88967e9e9f5464a7fe92a5aa8e80656174c5ea11` |

The paper copy is
`/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819/Figures/ala2_66d/alanine66d_energy_and_configurations.pdf`
and is byte-identical to the registered output PDF. Its source table,
manifest, and this provenance document are copied alongside it. The paper's
same-stem `_assets/` directory contains the four original molecular PNGs,
rendering metadata, exact coordinate/energy NPZ, and original snapshot
renderer. The `code/` subdirectory archives the figure builder, current
renderer, and registered path selector for provenance; run the canonical
builder from its original repository location as specified above, since
its relative input locators are repository-specific.

PDF inspection confirmed one page at 510.236 by 317.480 points
(180 by 112 mm), all five fonts embedded, and four molecular image objects
at 600 dpi. The figure PDF was independently rasterized and visually
checked: no clipped labels, overlapping callouts, distorted molecule
aspect ratios, or missing glyphs. Thin outer facets in some configurations
are the true near-edge-on projections; no geometry or camera was changed
to artificially widen them.

The prescribed preview compilation completed successfully with SyncTeX:

- Preview: `/Users/wuzhiyou/Downloads/IVM_for_MEP/paper0819/tmp/vscode-latex/main.pdf`;
- Figure 7, page 21 of the 35-page preview;
- Preview PDF SHA-256: `3e68080cd07bac72704a3233bb1aaa829fa732dd3373f59fde5792e6caba0a6d`;
- `main.tex` SHA-256 observed at verification: `d5a9dd0a9be5fef5df5286c1a049e78caaf57362146a55d4ff83064150995c12`.

The final manuscript figure page was also rasterized and visually checked.
The enlarged-number revision was checked both in the standalone PDF and at
the manuscript's actual figure width: all eight badges are legible, with no
overlap against titles, molecules, triangles, or profile curves. This
numbering revision makes no manuscript-source edits. Other manuscript text
and LEPS figures changed concurrently in the shared workspace; the preview
and source hashes above capture the verification state, not an isolated
source snapshot.
The compilation has no fatal errors or unresolved-reference warnings.
Existing title-area warnings at source line 188 and generic page-output
underfull-box warnings remain; those unrelated areas were not edited.

The figure workflow targets only the preview directory for compilation and
does not issue a copy to the formal root `main.pdf`. In this shared-workspace
session, a separate change to the formal root PDF was nevertheless observed
(latest observed SHA-256
`3c4d6c6783504184e09552c3ed9ecbe6f87e8c9be8c92e276727e9434c2d0bc4`),
so preservation of that file's initial bytes is not claimed. Its writer was
not established and no rollback or replacement was attempted. The preview
listed above is the verified manuscript artifact delivered for this figure.
