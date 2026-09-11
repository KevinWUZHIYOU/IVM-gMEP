#!/usr/bin/env python3
"""Additional Iperp/action cost-to-precision figure, preserving original PDF."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import plot_chignolin_ivm_arc500 as ref
from postprocess_geometric_action import ROOT, CONTRACT, OUT, sha, selected_rows

FIGDIR = Path('/Users/wuzhiyou/Downloads/IVM_for_MEP/Figures/chignolin/active_candidates/chignolin_ir_ivm_cpu1')
STEM = 'chignolin_ivm_geometric_action_1x2'


def validated_data():
    config = json.loads(CONTRACT.read_text())
    audit = json.loads(OUT.with_suffix('.audit.json').read_text())
    if audit['result_sha256'] != sha(OUT) or audit['contract_sha256'] != sha(CONTRACT):
        raise ValueError('Result/contract hash mismatch')
    if audit['diagnostic_joint_evaluations'] != config['expected_diagnostic_joint_evaluations']:
        raise ValueError('Diagnostic NoF mismatch')
    _, original = selected_rows()
    with np.load(OUT, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if len(data['method']) != config['selected_path_count']:
        raise ValueError('Selected row count mismatch')
    for i, r in enumerate(original):
        for k in ('method', 'source_result', 'source_result_sha256', 'source_record_index',
                  'selected_path_sha256', 'number_of_forces', 'i_perp'):
            if data[k][i] != r[k]:
                raise ValueError(f'Original record binding mismatch: {i}, {k}')
    np.testing.assert_array_equal(data['counter_vector'].sum(axis=1), data['number_of_forces'])
    np.testing.assert_array_equal(data['diagnostic_joint_evaluations'], 22)
    norms = data['gradient_norm_ev_per_angstrom']
    lengths = data['segment_length_angstrom']
    reconstructed = (lengths * (norms[:, :-1] + norms[:, 1:]) * 0.5).sum(axis=1) * config['ev_to_kcal']
    np.testing.assert_allclose(reconstructed, data['action_kcal'], rtol=5e-15)
    np.testing.assert_allclose(data['recomputed_i_perp'], data['i_perp'],
                               rtol=config['iperp_verification_rtol'], atol=config['iperp_verification_atol'])
    return data, config, audit


def report_for(data, config, audit):
    metrics = ('i_perp', 'action_kcal')
    thresholds = {q: max(float(data[q][data['method'] == m].min()) for m in ref.REFERENCE_METHODS)
                  for q in metrics}
    methods = {}
    for m in ref.METHOD_ORDER:
        take = data['method'] == m
        nf = data['number_of_forces'][take]
        methods[m] = dict(
            count=int(take.sum()), source=config['methods'][m],
            first_passage={q: ref.first_passage(nf, data[q][take], thresholds[q]) for q in metrics},
            initial_action_kcal=float(data['action_kcal'][take][0]),
            final_action_kcal=float(data['action_kcal'][take][-1]),
            best_action_kcal=float(data['action_kcal'][take].min()),
            final_publishable_nf=int(nf[-1]))
    return dict(status='validated', definition=config['formula'], unit='kcal/mol',
                thresholds=thresholds, methods=methods, diagnostic_audit=audit,
                contract=dict(path=str(CONTRACT), sha256=sha(CONTRACT)),
                code={str(p): sha(p) for p in (Path(__file__).resolve(),
                         ROOT/'src/postprocess_geometric_action.py', ROOT/'src/chignolin_core.py',
                         ROOT/'src/plot_chignolin_ivm_arc500.py')},
                interpretation='Discrete 22-node trapezoid action, not densified; full gradient; neither Iperp nor energy barrier',
                diagnostic_cost_excluded_from_optimization_axis=True)


def draw(data, report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / (STEM + '.pdf')).exists():
        raise FileExistsError('Refusing to overwrite action figure; use a new output directory')
    ref.setup_style(mpl)
    fig = plt.figure(figsize=(ref.mm_to_inch(180), ref.mm_to_inch(72)), dpi=220)
    grid = GridSpec(1, 2, figure=fig, left=.065, right=.975, bottom=.17, top=.78, wspace=.24)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    amin, amax = float(data['action_kcal'].min()), float(data['action_kcal'].max())
    action_bounds = (amin * .60, 10**np.ceil(np.log10(amax)))
    bounds = {'i_perp': (1e-2, 10**4.5), 'action_kcal': action_bounds}
    frontier_rows, annotation_rows = [], []
    for ax, q in zip(axes, ('i_perp', 'action_kcal')):
        for m in ref.METHOD_ORDER:
            take = data['method'] == m
            xx, yy = ref.pareto_frontier(data['number_of_forces'][take], data[q][take], *bounds[q])
            style = ref.METHOD_STYLE[m]
            ax.plot(xx, yy, **{k: style[k] for k in ('color','linestyle','linewidth','marker','markersize','zorder')},
                    markerfacecolor='white', markeredgewidth=.55, markevery=max(1, len(xx)//8),
                    alpha=1 if m == 'IVM' else .92)
            for x, y in zip(xx, yy):
                candidates = np.flatnonzero(take & (data[q] == x) & (data['number_of_forces'] == y))
                if not len(candidates):
                    raise ValueError('Frontier not bound to a source row')
                j = candidates[-1]
                frontier_rows.append(dict(method=m, metric=q, value=float(x), number_of_forces=int(y),
                    source_result=str(data['source_result'][j]), source_record_index=int(data['source_record_index'][j]),
                    selected_path_sha256=str(data['selected_path_sha256'][j])))
        ax.set_xscale('log')
        ax.set_xlim(bounds[q][1], bounds[q][0])
        ax.set_ylim(0, 22000)
        ax.set_yticks((0, 5000, 10000, 15000, 20000))
        ax.yaxis.set_major_formatter(FuncFormatter(ref.force_formatter))
        ax.set_ylabel('Number of Forces')
        ax.grid(True, which='major', color='#D9D9D6', linewidth=.4, alpha=.62)
        target = report['thresholds'][q]
        ivm = report['methods']['IVM']['first_passage'][q]
        ode = report['methods']['ODE-NEB']['first_passage'][q]
        if ivm is None or ode is None:
            raise ValueError('Reference method does not reach shared threshold')
        ref.add_speedup_arrow(ax, target, ivm, ode, target*.82)
        annotation_rows.append(dict(metric=q, shared_threshold=target, ivm_number_of_forces=ivm,
                                    ode_number_of_forces=ode, ode_over_ivm=ode/ivm))
    axes[0].set_xlabel(r'$I_\perp$')
    axes[1].set_xlabel(r'$\widehat{S}$ (kcal mol$^{-1}$)')
    fig.canvas.draw()
    for ax, label in zip(axes, 'ab'):
        p = ax.get_position()
        fig.text(p.x0-.027, p.y1+.012, label, fontsize=8, fontweight='bold', va='bottom')
    handles = [Line2D([0],[0], **{k:ref.METHOD_STYLE[m][k] for k in
               ('label','color','linestyle','linewidth','marker','markersize')},
               markerfacecolor='white', markeredgewidth=.6) for m in ref.METHOD_ORDER]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.49,.975), ncol=3,
               columnspacing=1.35, handlelength=2.35, handletextpad=.5, frameon=False)
    outputs = {}
    for suffix in ('pdf','png','svg'):
        p = output_dir / (STEM+'.'+suffix)
        fig.savefig(p, dpi=600)
        outputs[suffix] = dict(path=str(p), sha256=sha(p))
    plt.close(fig)
    for kind, rows in (('frontier', frontier_rows), ('annotations', annotation_rows)):
        p = output_dir / (STEM+'_'+kind+'.csv')
        ref.write_csv(p, rows)
        outputs[kind] = dict(path=str(p), sha256=sha(p))
    report.update(annotations=annotation_rows, outputs=outputs,
                  action_xlim=list(reversed(action_bounds)), action_xscale='log')
    (output_dir / (STEM+'_provenance.json')).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=FIGDIR)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    d, c, a = validated_data()
    r = report_for(d, c, a)
    if not args.dry_run:
        draw(d, r, args.output_dir.resolve())
    print(json.dumps(r, indent=2))
