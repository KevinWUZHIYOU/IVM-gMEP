# IVM–MEP manuscript: figure and experiment code

Code release for the formal figures of the IVM (Iterative Variational Method)
manuscript whose source of truth is
`trees/Downloads_IVM_for_MEP/paper0819/main.tex`.

For every figure that the compiled manuscript actually includes, this release
contains two kinds of code:

- **Plotting code** – the script or notebook that renders the PDF LaTeX
  includes.
- **Experiment code** – the optimizer, baseline, driver, metric and audit
  sources that produced the plotted data.

Frozen provenance records — preregistrations, selection manifests, traceability
documents and plotted-value tables — are kept next to the code they describe.

## 中文说明

本仓库收录论文 `paper0819/main.tex` 中**真正参与排版**的 16 张正式图所对应的
代码，每张图都给出两类来源：

1. **画图代码**：生成该 PDF 的脚本或 notebook；
2. **实验代码**：产生被画数据的优化器、baseline、运行脚本与审计代码。

完整的图—代码对应关系见 `FIGURE_CODE_MAP.md`。原始工作目录中数 GB 的
`runs/` 迭代档案与 MD 输入结构没有收录，只保留了画图直接读取的冻结数据与
登记文件。

## Start here

`FIGURE_CODE_MAP.md` maps each of the 16 active manuscript figures to its
plotting code, experiment code and provenance record.

## Layout

`trees/` mirrors the original working directories, so the relative paths
recorded inside the provenance documents stay meaningful:

| Release path | Original root |
| --- | --- |
| `trees/Downloads_IVM_for_MEP/` | `/Users/wuzhiyou/Downloads/IVM_for_MEP` |
| `trees/IVM_for_MEP/` | `/Users/wuzhiyou/IVM_for_MEP` |
| `trees/Code/` | `/Users/wuzhiyou/Code` |

The manuscript figure files live under
`trees/Downloads_IVM_for_MEP/paper0819/Figures/`, together with the per-figure
`*_manifest.json`, `*_PROVENANCE.md`, `*_provenance.json` and `*_source.csv`
records that register each PDF.

## What is included, and what is not

Included: plotting scripts, optimizer/baseline/driver/audit sources, unit
tests, preregistrations, selection manifests, traceability documents,
plotted-value tables, and the small frozen inputs that the plotting code reads.

Not included: multi-gigabyte optimization archives (`runs/`), MD input
structures and force-field files (`inputs/`), Python byte-code caches, and
figure files that `main.tex` does not use (several are retained inside
`\iffalse` blocks and are not part of the compiled paper). All of those remain
in the original working directories listed above.

## Environment

The registered interpreter for every experiment in this release is

```text
/Users/wuzhiyou/anaconda3/envs/torch-nightly/bin/python
```

The system `python3` is not the registered interpreter for these artifacts.
The molecular dynamics work uses OpenMM (`amber14-all.xml`, `implicit/gbn2.xml`,
CPU platform, double precision, no nonbonded cutoff); several campaigns also
use ASE, PyTorch and JAX. Each campaign directory keeps its own
`requirements.txt` or `PROTOCOL.md` where one was registered.

## Reproduction caveats

The sources are copied verbatim, so they still contain the absolute paths of
the original working directories. To rerun a script in this release, either
recreate that directory layout or edit the recorded input paths. The
provenance files state the exact registered command for each figure.

Several figures are *reconstructions* from frozen production results rather
than fresh optimizer runs. This is recorded for each case in
`FIGURE_CODE_MAP.md`; no optimizer was rerun for the LEPS replacement figures,
for example.

## Known gaps

- `Figures/chignolin/transition_2x5.pdf` is included by `main.tex`
  (`fig:Chignolin_transition`), but no generating script for it exists in any
  local archive. The PDF was written by Matplotlib on 2026-05-29, before the
  registered Chignolin campaigns, and the older paper asset manifests
  attribute it only to the V9 production runner rather than to a plotter. Its
  code is therefore the one documented exception in this release.
- `paper/ActiveExperiments/alanine_66d/data/` was described as
  "code/data migration pending" in the earlier
  `PAPER_ASSET_MANIFEST.md`; the plotters were subsequently located and are
  included here.

## Data availability for the plotted curves

The plotted-value tables (the `*_source.csv` and `*_frontier.csv` files next to
each PDF) are sufficient to re-render every figure. Where the underlying raw
optimization state is small enough to ship, it is included — for example the
selected LEPS runs, the QM10 frozen runs, the Chignolin IR-IVM reproduction
states and its arc-length-500 / discrete-action diagnostics, and the MFEP
summary `.npz` files.

Two input sets could not be located in the current local mirrors and are
therefore absent:

- the ODE–NEB, static-NEB, "Original String" and OBSM–FIRE Chignolin production
  archives that
  `$D/paper0819/Figures/chignolin/chignolin_ir_ivm_overlay_total_nf_provenance.json`
  records under `code_on_server/chignolin/runs/neb/` and
  `code_on_server/chignolin/runs/confirmed1000_productions_20260728c/` — those
  directories no longer hold the named `.npz` files;
- multi-gigabyte `runs/` archives for the Barnase–barstar and Alanine 66D
  campaigns, which remain in the original working directories.
