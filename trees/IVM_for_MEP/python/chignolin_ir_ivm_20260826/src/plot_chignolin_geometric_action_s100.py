#!/usr/bin/env python3
"""User-requested action threshold 100; retain the original registered figure."""
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

import plot_chignolin_geometric_action as original

ref = original.ref
STEM = 'chignolin_ivm_geometric_action_1x2_s100'


def report_for(data, config, audit):
    report = original.report_for(data, config, audit)
    report['thresholds']['action_kcal'] = 100.0
    report['threshold_policy'] = {
        'i_perp': 'Unchanged: max of minima over IVM, ODE-NEB, OBSM-FIRE',
        'action_kcal': 'User-specified fixed threshold 100 kcal/mol',
    }
    report['code'][str(Path(__file__).resolve())] = original.sha(Path(__file__))
    for method, details in report['methods'].items():
        take = data['method'] == method
        details['first_passage']['action_kcal'] = ref.first_passage(
            data['number_of_forces'][take], data['action_kcal'][take], 100.0)
    report['new_force_evaluations'] = 0
    return report


def draw(data, report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob(STEM + '*')):
        raise FileExistsError('Use a new output directory; existing outputs are preserved')
    ref.setup_style(mpl)
    fig = plt.figure(figsize=(ref.mm_to_inch(180), ref.mm_to_inch(72)), dpi=220)
    grid = GridSpec(1, 2, figure=fig, left=.065, right=.975, bottom=.17, top=.78, wspace=.24)
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    bounds = {
        'i_perp': (1e-2, 10**4.5),
        'action_kcal': (float(data['action_kcal'].min())*.60,
                        10**np.ceil(np.log10(float(data['action_kcal'].max())))),
    }
    frontiers, annotations = [], []
    for ax, metric in zip(axes, ('i_perp', 'action_kcal')):
        for method in ref.METHOD_ORDER:
            take = data['method'] == method
            xx, yy = ref.pareto_frontier(data['number_of_forces'][take], data[metric][take], *bounds[metric])
            style = ref.METHOD_STYLE[method]
            ax.plot(xx, yy, **{k: style[k] for k in
                    ('color', 'linestyle', 'linewidth', 'marker', 'markersize', 'zorder')},
                    markerfacecolor='white', markeredgewidth=.55,
                    markevery=max(1, len(xx)//8), alpha=1 if method == 'IVM' else .92)
            for x, y in zip(xx, yy):
                match = np.flatnonzero(take & (data[metric] == x) & (data['number_of_forces'] == y))
                if not len(match):
                    raise ValueError('Frontier point is not bound to a source row')
                j = match[-1]
                frontiers.append(dict(method=method, metric=metric, value=float(x), number_of_forces=int(y),
                    source_result=str(data['source_result'][j]), source_record_index=int(data['source_record_index'][j]),
                    selected_path_sha256=str(data['selected_path_sha256'][j])))
        ax.set_xscale('log')
        ax.set_xlim(bounds[metric][1], bounds[metric][0])
        ax.set_ylim(0, 22000)
        ax.set_yticks((0, 5000, 10000, 15000, 20000))
        ax.yaxis.set_major_formatter(FuncFormatter(ref.force_formatter))
        ax.set_ylabel('Number of Forces')
        ax.grid(True, which='major', color='#D9D9D6', linewidth=.4, alpha=.62)
        target = report['thresholds'][metric]
        ivm = report['methods']['IVM']['first_passage'][metric]
        ode = report['methods']['ODE-NEB']['first_passage'][metric]
        if ivm is None:
            raise ValueError('IVM does not reach the requested threshold')
        ratio = None if ode is None else ode/ivm
        if ode is not None:
            ref.add_speedup_arrow(ax, target, ivm, ode, target*.82)
        else:
            ax.axvline(target, color='#767676', linewidth=.55, linestyle=':', zorder=1)
            last_nf = report['methods']['ODE-NEB']['final_publishable_nf']
            text = (r'$\widehat{S}_h\leq100$' + f'\nIVM: {ivm:,} NoF\n'
                    f'ODE-NEB: not reached ({last_nf:,} NoF)')
            ax.text(.035, .96, text, transform=ax.transAxes, ha='left', va='top',
                    fontsize=6.7, linespacing=1.4,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2))
        annotations.append(dict(metric=metric, threshold=target, ivm_number_of_forces=ivm,
            ode_number_of_forces=ode, ode_over_ivm=ratio,
            ode_status='not_reached' if ode is None else 'reached',
            ode_last_publishable_nf=report['methods']['ODE-NEB']['final_publishable_nf']))
    axes[0].set_xlabel(r'$I_\perp$')
    axes[1].set_xlabel(r'$\widehat{S}_h$ (kcal mol$^{-1}$)')
    fig.canvas.draw()
    for ax, label in zip(axes, 'ab'):
        p = ax.get_position()
        fig.text(p.x0-.027, p.y1+.012, label, fontsize=8, fontweight='bold', va='bottom')
    handles = [Line2D([0], [0], **{k: ref.METHOD_STYLE[m][k] for k in
                ('label', 'color', 'linestyle', 'linewidth', 'marker', 'markersize')},
                markerfacecolor='white', markeredgewidth=.6) for m in ref.METHOD_ORDER]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.49, .975), ncol=3,
               columnspacing=1.35, handlelength=2.35, handletextpad=.5, frameon=False)
    outputs = {}
    for suffix in ('pdf', 'png', 'svg'):
        path = output_dir/(STEM+'.'+suffix)
        fig.savefig(path, dpi=600)
        outputs[suffix] = dict(path=str(path), sha256=original.sha(path))
    plt.close(fig)
    for kind, rows in (('frontier', frontiers), ('annotations', annotations)):
        path = output_dir/(STEM+'_'+kind+'.csv')
        ref.write_csv(path, rows)
        outputs[kind] = dict(path=str(path), sha256=original.sha(path))
    report.update(outputs=outputs, annotations=annotations,
                  action_xlim=list(reversed(bounds['action_kcal'])), action_xscale='log')
    (output_dir/(STEM+'_provenance.json')).write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=original.FIGDIR)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    data, config, audit = original.validated_data()
    report = report_for(data, config, audit)
    if not args.dry_run:
        draw(data, report, args.output_dir.resolve())
    print(json.dumps(report, indent=2))
