"""Independently audit the frozen grid, select, and rebuild LEPS figures."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
LEPS = ROOT.parent
PROJECT = LEPS.parents[1]
REPORT = ROOT / 'reports'
sys.path.insert(0, str(LEPS / 'src'))
from leps_core import leps_energy_gradient, improved_tangents, physical_to_flat, reference_saddle

OLD_REPORT = LEPS / 'leps_irivm_sync1_refined_grid_20260830/reports/figure'
OLD_REG = OLD_REPORT / 'leps_all_methods_metrics_provenance.json'
OLD_CSV = OLD_REPORT / 'leps_all_methods_metrics_source.csv'
REFERENCE = float(reference_saddle()['forward_barrier'])


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def write_csv(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def metrics(paths):
    """Vectorized HJ tangents, identical to ASE's ImprovedTangentMethod."""
    e, g = leps_energy_gradient(paths[..., 0], paths[..., 1])
    edge = np.diff(paths, axis=1)
    left, right = edge[:, :-1], edge[:, 1:]
    up, mid, down = e[:, 2:], e[:, 1:-1], e[:, :-2]
    hi = np.maximum(abs(up-mid), abs(down-mid))[..., None]
    lo = np.minimum(abs(up-mid), abs(down-mid))[..., None]
    tangent = np.where((up > down)[..., None], right*hi + left*lo, right*lo + left*hi)
    tangent = np.where(((up > mid) & (mid > down))[..., None], right, tangent)
    tangent = np.where(((up < mid) & (mid < down))[..., None], left, tangent)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    normal = g[:, 1:-1] - np.sum(g[:, 1:-1]*tangent, axis=-1, keepdims=True)*tangent
    length = np.linalg.norm(edge, axis=-1)
    weights = .5*(length[:, :-1] + length[:, 1:])
    residual = np.sum(weights*np.sum(normal**2, axis=-1), axis=-1)
    action = np.sum(.5*(np.linalg.norm(g[:, :-1], axis=-1) + np.linalg.norm(g[:, 1:], axis=-1))*length, axis=-1)
    pos = np.linspace(0., 21., 500)
    index = np.minimum(pos.astype(int), 20)
    t = (pos-index)[None, :, None]
    dense = paths[:, index]*(1-t) + paths[:, index+1]*t
    de, _ = leps_energy_gradient(dense[..., 0], dense[..., 1])
    barrier = de.max(axis=1) - e[:, 0]
    # Every run: compare initial/middle/final tangent to the actual ASE code.
    for j in sorted(set([0, len(paths)//2, len(paths)-1])):
        ref = improved_tangents(physical_to_flat(paths[j]), e[j])[:, [0, 3]]
        np.testing.assert_allclose(tangent[j], ref, atol=1e-12, rtol=1e-10)
    return e, g, residual, barrier, action


def audit():
    status = json.loads((ROOT/'status.json').read_text())
    assert status['completed'] == status['total'] == 60, 'production grid is incomplete'
    reg = json.loads((ROOT/'registration.json').read_text())
    for path, expected in reg['inputs'].items():
        assert sha(LEPS/path) == expected, path
    assert sha(ROOT/'validation.json') == reg['derivative_validation']
    assert len(reg['configs']) == 60
    with np.load(LEPS/'inputs/common_initial_path.npz') as z:
        initial = z['initial_path'][:, [0, 3]]
    _, _, i0, b0, _ = metrics(initial[None])
    summary, traces = [], {}
    for c in reg['configs']:
        p = ROOT/'runs'/(c['id']+'.npz')
        with np.load(p) as saved:
            z = {k:saved[k] for k in saved.files}
        assert json.loads(str(z['config_json'])) == c
        assert str(z['registration_sha256']) == sha(ROOT/'registration.json')
        assert str(z['source_sha256']) == sha(ROOT/'run_experiments.py')
        paths, nof = z['path_history'], z['article_nf']
        np.testing.assert_array_equal(paths[0], initial)
        np.testing.assert_array_equal(paths[:, 0], np.broadcast_to(initial[0], paths[:, 0].shape))
        np.testing.assert_array_equal(paths[:, -1], np.broadcast_to(initial[-1], paths[:, -1].shape))
        assert np.all(np.diff(nof) >= 0)
        assert int(z['terminal_nof']) <= c['budget']
        events = json.loads(str(z['events_json']))
        nf = nh = ng = 0
        for ev in events:
            assert ev['nof'] == ev['force_calls'] + ev['hvp']
            nf += ev['nof']; nh += ev['hvp']; ng += ev['force_calls']
            assert (nf, nh, ng) == (ev['total_nof'], ev['total_hvp'], ev['total_force_calls'])
        assert (nf, nh, ng) == (int(z['terminal_nof']), int(z['terminal_hvp']), int(z['terminal_force_calls']))
        for cost, hvp in zip(nof, z['hvp_history']):
            assert any(ev['total_nof'] == cost and ev['total_hvp'] == hvp for ev in events)
        e, g, residual, barrier, action = metrics(paths)
        np.testing.assert_allclose(e, z['energy_history'], atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(g, z['gradient_history'], atol=1e-9, rtol=1e-9)
        length = np.linalg.norm(np.diff(paths, axis=1), axis=-1)
        for j, (outer, phase) in enumerate(zip(z['outer'], z['state_phase'])):
            warm = c['method'] == 'StringNET' and outer <= 50
            if outer:
                assert phase == ('warmup' if warm else 'gmam')
            if warm:
                a = np.exp(c['beta']*(e[j]-e[j].max()))
                val = e[j].max() + np.log(np.sum(.5*(a[:-1]+a[1:])*length[j]))/c['beta']
            else:
                val = action[j]
            np.testing.assert_allclose(val, z['objective'][j], atol=1e-9, rtol=1e-9)
        eligible = str(z['stop_reason']) != 'numerical_failure' and np.all(np.isfinite([residual, barrier, action]))
        if c['method'] == 'StringNET':
            eligible = eligible and bool(z['entered_gmam']) and np.any(z['state_phase'] == 'gmam')
        checkpoints = np.arange(1000, 20001, 1000)
        idx = np.searchsorted(nof, checkpoints, side='right')-1
        score = np.mean(np.log(np.maximum(residual[idx], 1e-12)/max(i0[0], 1e-12)))
        score += np.mean(np.log(np.maximum(abs(barrier[idx]-REFERENCE), 1e-6)/max(abs(b0[0]-REFERENCE), 1e-6)))
        row = dict(**c, eligible=bool(eligible), score=float(score) if np.isfinite(score) else None,
                   stop=str(z['stop_reason']), steps=int(z['outer'][-1]),
                   nof=nf, force_calls=ng, hvp=nh, last_saved_nof=int(nof[-1]),
                   final_i_perp=float(residual[-1]), final_barrier=float(barrier[-1]),
                   final_barrier_error=float(abs(barrier[-1]-REFERENCE)), final_action=float(action[-1]),
                   min_i_perp=float(residual.min()), min_barrier_error=float(abs(barrier-REFERENCE).min()),
                   threshold_nof=int(nof[np.flatnonzero(residual<=1e-4)[0]]) if np.any(residual<=1e-4) else None,
                   switch_nof=int(nof[z['outer']==50][0]) if np.any(z['outer']==50) and c['warmup'] else None,
                   source_sha256=sha(p))
        summary.append(row)
        traces[c['id']] = (nof, residual, barrier, action)
        out = REPORT/'metrics'/(c['id']+'.npz')
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, nof=nof, i_perp=residual, barrier=barrier, action=action,
                            source_sha256=sha(p), analyzer_sha256=sha(__file__))
        print('AUDITED', c['id'], c['method'], 'eligible', bool(eligible), flush=True)
    def best(rows):
        return min([r for r in rows if r['eligible']], key=lambda r:(r['score'], r['final_action'], r['nof'], r['id']))
    winners = {method:best([r for r in summary if r['method']==method]) for method in ['gMAM', 'StringNET']}
    by_beta = {str(b):best([r for r in summary if r['method']=='StringNET' and r['beta']==b]) for b in [.1,.3,1.,3.]}
    by_optimizer_reparam = {}
    for method in ['gMAM','StringNET']:
        for optimizer in ['LBFGS','Adam']:
            for reparam in ['string','ivm_spring']:
                candidates=[r for r in summary if r['method']==method and r['optimizer']==optimizer
                            and r['reparam']==reparam and r['eligible']]
                by_optimizer_reparam['/'.join([method,optimizer,reparam])] = best(candidates) if candidates else None
    selection = dict(winners=winners, by_beta=by_beta, by_optimizer_reparam=by_optimizer_reparam,
                     reference_barrier=REFERENCE,
                     registration_sha256=sha(ROOT/'registration.json'), analyzer_sha256=sha(__file__),
                     completed=60, eligible=sum(r['eligible'] for r in summary),
                     numerical_failures=sum(r['stop']=='numerical_failure' for r in summary),
                     cumulative_nof=sum(r['nof'] for r in summary),
                     cumulative_force_calls=sum(r['force_calls'] for r in summary),
                     cumulative_hvp=sum(r['hvp'] for r in summary))
    write_csv(REPORT/'grid_summary.csv', summary)
    write_json(REPORT/'selection.json', selection)
    write_json(REPORT/'audit.json', dict(passed=True, checked_runs=60, fixed_endpoints=True,
        common_initial_path=True, energy_force_crosscheck=True, ase_improved_tangent_crosscheck=True,
        objective_crosscheck=True, phase_switch_check=True, force_plus_one_hvp_ledger_check=True,
        registration_sha256=sha(ROOT/'registration.json'), analyzer_sha256=sha(__file__)))
    return selection, traces


def figure(traces, output):
    colors = {'IVM':'#0072B2','ODE-NEB':'#D55E00','Static-NEB':'#B279A2',
              'String Method':'#777777','OBSM-FIRE':'#5B9279','OBSM-L-BFGS':'#CC79A7',
              'gMAM':'#17BECF','StringNET':'#6A3D9A'}
    extra_colors = ['#E69F00','#6A3D9A','#CC79A7','#56B4E9']
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':7,'axes.labelsize':7.5,'xtick.labelsize':6.6,'ytick.labelsize':6.6,
        'legend.fontsize':6.4,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.75,'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes = plt.subplots(1,3,figsize=(200/25.4,88/25.4))
    ceilings = [100., 5., 30.]
    omitted, annotation_counts, sampling = [], [0,0,0], []
    for index, (name, values) in enumerate(traces.items()):
        nf, *ys = values
        budgets=np.arange(500.,min(20000.,nf[-1])+1,500.)
        chosen=np.searchsorted(nf,budgets,side='right')-1
        chosen=np.unique(np.concatenate(([0],chosen[chosen>=0],[len(nf)-1]))).astype(int)
        sampling.append(dict(method=name,source_indices=chosen.tolist(),actual_nof=nf[chosen].tolist()))
        for panel, (ax,y) in enumerate(zip(axes, ys)):
            color=colors.get(name, extra_colors[index%4])
            above=y>ceilings[panel]
            visible=chosen[~above[chosen]]
            sx,shown=nf[visible],y[visible]
            ax.plot(sx, shown, label=name, color=color,
                    lw=1.9 if name=='IVM' else 1.25,
                    ls='-' if name in ['IVM','gMAM','StringNET'] else '--',
                    marker='o' if name=='IVM' else None, ms=2.5,mfc='white',mew=.6,
                    zorder=30 if name=='IVM' else 6)
            if np.any(above):
                peak=int(np.argmax(y))
                height=1.025+.09*annotation_counts[panel]
                annotation_counts[panel]+=1
                transform=ax.get_xaxis_transform()
                ax.plot([nf[peak]],[height],marker='^',color=color,ms=4,
                        transform=transform,linestyle='none',zorder=40,clip_on=False)
                value=f'{y[peak]:.2g}' if panel==0 else f'{y[peak]:.1f}'
                ax.annotate('peak '+value,xy=(nf[peak],height),xycoords=transform,
                            xytext=(5,0),textcoords='offset points',ha='left',va='center',
                            fontsize=6,color=color)
                omitted.append(dict(method=name,metric=['i_perp','barrier','action'][panel],
                    display_ceiling=ceilings[panel],peak_value=float(y[peak]),peak_nof=float(nf[peak]),
                    omitted_points=[dict(index=int(j),nof=float(nf[j]),value=float(y[j]))
                                    for j in np.flatnonzero(above)]))
    axes[0].set_yscale('log')
    axes[0].set_ylim(1e-29,ceilings[0])
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v,_: f'1e{int(round(np.log10(v)))}' if v>0 else ''))
    axes[1].axhline(REFERENCE,ls=':',lw=.7,color='#444444')
    axes[1].text(19700,REFERENCE+.04,r'$\Delta E^*$',ha='right',va='bottom',fontsize=6.5)
    axes[1].set_ylim(0,ceilings[1])
    axes[2].set_ylim(0,ceilings[2])
    for ax,label in zip(axes,[r'$I_\perp$',r'Densified energy barrier $\Delta E$ (eV)',r'Discrete geometric action $\widehat{S}$ (eV)']):
        ax.set_xlim(0,20000);ax.set_xticks([0,5000,10000,15000,20000])
        ax.set_xlabel('Number of Forces');ax.set_ylabel(label);ax.set_box_aspect(1)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.52,.985),
               ncol=4 if len(traces)>6 else 3, columnspacing=1.25,handlelength=2.3)
    fig.subplots_adjust(left=.068,right=.970,bottom=.17,top=.76,wspace=.40)
    fig.canvas.draw()
    ratios = [ax.get_window_extent().width/ax.get_window_extent().height for ax in axes]
    np.testing.assert_allclose(ratios,1.,atol=1e-12)
    for ax,letter in zip(axes,'abc'):
        pos=ax.get_position();fig.text(pos.x0-.034,pos.y1+.012,letter,fontsize=9,fontweight='bold')
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(output.with_suffix('.png'),dpi=180)
    plt.close(fig)
    return ratios, omitted, sampling


def main():
    selection, new_traces = audit()
    old = json.loads(OLD_REG.read_text())
    for info in old['plotted_results'].values():
        assert sha(LEPS/info['path']) == info['sha256']
    assert sha(OLD_CSV) == old['source_csv_sha256']
    with OLD_CSV.open(newline='') as f:
        rows=list(csv.DictReader(f))
    traces, results = {}, {}
    for name, info in old['plotted_results'].items():
        label = 'String Method' if name=='Original String' else name
        results[label]=info
    for row in rows:
        if row['method']=='Original String':row['method']='String Method'
    for name in ['IVM','ODE-NEB','Static-NEB','String Method','OBSM-FIRE','OBSM-L-BFGS']:
        subset=[r for r in rows if r['method']==name]
        traces[name]=tuple(np.array([float(r[k]) for r in subset]) for k in
            ['number_of_forces','i_perp','densified_barrier_eV','geometric_action_eV'])
    # Verify the inherited plotted values against their actual frozen paths.
    wrapper=LEPS/'leps_irivm_sync1_refined_grid_20260830/src/remake_leps_all_methods_selected.py'
    spec=importlib.util.spec_from_file_location('_selected_old_leps',wrapper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for name in traces:
        raw=LEPS/results[name]['path']
        nof,paths=(module.plot.ir_ivm_trace(raw) if name=='IVM' else module.plot.baseline_trace(raw))
        keep=nof<=20000
        np.testing.assert_array_equal(nof[keep],traces[name][0])
        _,_,ip,bar,act=metrics(paths[keep][:,:,[0,3]])
        for recomputed,registered in zip((ip,bar,act),traces[name][1:]):
            np.testing.assert_allclose(recomputed,registered,atol=2e-12,rtol=2e-10)
    for name, winner in selection['winners'].items():
        traces[name]=new_traces[winner['id']]
        raw=ROOT/'runs'/(winner['id']+'.npz')
        info=dict(path=str(raw.relative_to(LEPS)),sha256=sha(raw),configuration=winner)
        results[name]=info
        for index, (nf,ip,bar,act) in enumerate(zip(*traces[name])):
            rows.append(dict(method=name,plotted_index=index,number_of_forces=int(nf),i_perp=float(ip),
                densified_barrier_eV=float(bar),geometric_action_eV=float(act),
                densification='500 total coordinate-wise linear interpolation points (barrier only)',
                geometric_action_quadrature='22-node segment-length trapezoidal rule; no densification',
                production_result=info['path'],production_result_sha256=info['sha256']))
    archive=ROOT/'archive/previous_six_method_figure'
    archive.mkdir(parents=True,exist_ok=True)
    outputs=[PROJECT/'output/pdf/leps_all_methods_metrics.pdf',PROJECT/'Figures/LEPS/leps_all_methods_metrics.pdf',
             LEPS/'figures/leps_all_methods_metrics.pdf',PROJECT/'paper0819/Figures/LEPS/leps_all_methods_metrics.pdf']
    for index,p in enumerate(outputs+[OLD_REG,OLD_CSV]):
        target=archive/(str(index)+'_'+p.name)
        if p.exists() and not target.exists():shutil.copyfile(p,target)
    ratios,omitted,sampling=figure(traces,outputs[0])
    for target in outputs[1:]:shutil.copyfile(outputs[0],target)
    tuning={'IVM':traces['IVM'],'gMAM':traces['gMAM']}
    for beta,winner in selection['by_beta'].items():
        tuning['StringNET '+r'$\beta='+f'{float(beta):g}'+r'$']=new_traces[winner['id']]
    _,tuning_omitted,tuning_sampling=figure(tuning,PROJECT/'output/pdf/leps_action_baselines_tuning.pdf')
    sampling_file=REPORT/'sparse_sampling.json'
    write_json(sampling_file,dict(interval_nof=500,
        rule='latest recorded state at or below each 500-NoF checkpoint; retain initial/final states; deduplicate; same rule for every method; connect visible sampled points with straight lines without smoothing; retain full-data peak annotations',
        all_methods=sampling,beta_comparison=tuning_sampling))
    display=REPORT/'spike_display.json'
    write_json(display,dict(user_authorized=True,
        policy='values above display ceilings are excluded from display only; visible sampled points are connected by straight lines without smoothing; upward triangle marks maximum omitted value at its true NoF; raw values, costs, and selection remain unchanged',
        ceilings=dict(i_perp=100.,barrier_eV=5.,action_eV=30.),
        all_methods=omitted,beta_comparison=tuning_omitted))
    source=REPORT/'leps_all_methods_metrics_source.csv'
    write_csv(source,rows)
    contract=dict(old['metric_contract'])
    contract['number_of_forces']='user-specified equivalent cost: uncached internal-image force evaluations + one per internal-image HVP; endpoints/cache hits excluded; all trials included'
    registry=dict(figure=str(outputs[1].relative_to(PROJECT)),figure_sha256=sha(outputs[1]),
        plot_script=str(Path(__file__).resolve().relative_to(LEPS)),plot_script_sha256=sha(__file__),
        source_csv=str(source.relative_to(LEPS)),source_csv_sha256=sha(source),
        plotted_results=results,metric_contract=contract,
        spike_display=str(display.relative_to(LEPS)),spike_display_sha256=sha(display),
        sparse_sampling=str(sampling_file.relative_to(LEPS)),sparse_sampling_sha256=sha(sampling_file),
        selection=str((REPORT/'selection.json').relative_to(LEPS)),selection_sha256=sha(REPORT/'selection.json'),
        previous_registry=str(OLD_REG.relative_to(LEPS)),previous_registry_sha256=sha(OLD_REG),
        figure_layout=dict(arrangement='1x3',each_axes_box_aspect=1.,verified_display_width_to_height=ratios),
        replacement_date='2026-09-10',audit_sha256=sha(REPORT/'audit.json'))
    write_json(REPORT/'leps_all_methods_metrics_provenance.json',registry)
    print(json.dumps({k:selection[k] for k in ['completed','eligible','numerical_failures','cumulative_nof','winners','by_beta']},indent=2),flush=True)


if __name__=='__main__':
    main()
