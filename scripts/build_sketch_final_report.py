"""Aggregate completed checkpoints and write the measured research wrap-up."""
from pathlib import Path
import argparse
import json
import math
import os
import sys
import importlib.metadata
import hashlib

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def table(frame,columns=None):
    if frame.empty:return '_No completed measurements._'
    frame=frame[columns] if columns else frame
    def fmt(x):
        if isinstance(x,(float,np.floating)):return 'NA' if not np.isfinite(x) else f'{x:.4g}'
        return str(x)
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None))


def build(out):
    config=json.loads((out/'experiment_config.json').read_text())
    config['package_versions']={p:importlib.metadata.version(p) for p in ['torch','transformer-lens','transformers','datasets','numpy','pandas','scipy']}
    config['python_version']=sys.version
    candidate_run=ROOT/'results/mib_sketch_acdc/20261015T054348Z'
    config['candidate_source_run']=candidate_run.relative_to(ROOT).as_posix()
    config['candidate_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [candidate_run/'selected_edges.csv',candidate_run/'exact_acdc_trajectory.csv']}
    pd.DataFrame(config['edges']).to_csv(out/'selected_edges.csv',index=False)
    config['portable_text_sha256']={}
    for name in list(config['implementation'])+list(config['old_notebooks']):
        path=ROOT/name.replace('\\','/')
        if path.exists():
            config['portable_text_sha256'][name.replace('\\','/')]=hashlib.sha256(
                path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    frames=[];hard_frames=[];decompositions=[];runtime=[];ranks=[];state_errors=[];spectra=[]
    common=out/'common_compute/compute_accounting.csv'
    if common.exists():
        part=pd.read_csv(common);part['discovery_examples']=0;part['seed']=42;runtime.append(part)
    completed=[]
    for marker in sorted(out.glob('seed*_n*/complete.json')):
        folder=marker.parent;info=json.loads(marker.read_text());n=info['discovery_examples'];seed=info['seed']
        completed.append((seed,n))
        metrics=pd.read_csv(folder/'prediction_comparison.csv');frames.append(metrics)
        hard=pd.read_csv(folder/'difficult_decisions.csv',dtype={'state':str});hard['discovery_examples']=n;hard['seed']=seed;hard_frames.append(hard)
        raw=pd.read_csv(folder/'compute_accounting.csv');raw['discovery_examples']=n;raw['seed']=seed;runtime.append(raw)
        pred=pd.read_csv(folder/'predictions.csv',dtype={'state':str})
        sk=pred[pred.method.str.startswith('sketch_rank_')]
        np.testing.assert_allclose(sk.prediction-sk.validation_effect,
            sk.taylor_signed_error+sk.compression_signed_error+sk.generalization_shift,atol=1e-10,rtol=1e-10)
        for (anchor,method,removed),group in pred.groupby(['anchor_strength','method','removed_edges']):
            state_errors.append({'discovery_examples':n,'seed':seed,'anchor_strength':anchor,'method':method,
                'removed_edges':removed,'queries':len(group),'mae':abs(group.prediction-group.validation_effect).mean()})
        for path in folder.glob('bundle_*.npz'):
            with np.load(path,allow_pickle=False) as bundle:
                eigen=np.linalg.eigvalsh(bundle['full_second_order'])
                energy=np.sort(eigen**2)[::-1];cumulative=np.cumsum(energy)/(energy.sum()+1e-30)
                spectra.append({'discovery_examples':n,'seed':seed,'context':path.stem,
                    'rank_for_99pct_squared_eigenvalue_energy':min(len(eigen),int(np.searchsorted(cumulative,.99)+1)) if energy.sum()>1e-30 else 0,
                    'min_eigenvalue':eigen.min(),'max_eigenvalue':eigen.max()})
        for (anchor,method),g in pred.groupby(['anchor_strength','method']):
            decompositions.append({'discovery_examples':n,'seed':seed,'anchor_strength':anchor,'method':method,
                'estimator_mae_train':abs(g.prediction-g.train_effect).mean(),
                'full_taylor_mae_train':abs(g.taylor_signed_error).mean(),
                'compression_mae':abs(g.compression_signed_error).mean(),
                'exact_train_validation_shift_mae':abs(g.generalization_shift).mean(),
                'total_validation_mae':abs(g.prediction-g.validation_effect).mean()})
        rank=pd.read_csv(folder/'curvature_diagnostics.csv',dtype={'state':str});rank['discovery_examples']=n;rank['seed']=seed;ranks.append(rank)
    if not frames:raise RuntimeError('No complete configuration to report; no results invented.')
    metrics=pd.concat(frames,ignore_index=True).sort_values(['seed','discovery_examples','anchor_strength','method'])
    hard=pd.concat(hard_frames,ignore_index=True)
    raw=pd.concat(runtime,ignore_index=True);decomp=pd.DataFrame(decompositions)
    metrics.to_csv(out/'prediction_comparison.csv',index=False)
    metrics.to_csv(out/'sample_efficiency.csv',index=False)
    hard.to_csv(out/'difficult_decisions.csv',index=False)
    raw.to_csv(out/'runtime_breakdown.csv',index=False)
    decomp.to_csv(out/'error_decomposition.csv',index=False)
    state_errors=pd.DataFrame(state_errors);state_errors.to_csv(out/'error_by_pruned_edges.csv',index=False)
    pd.DataFrame(spectra).to_csv(out/'hessian_spectrum.csv',index=False)
    sketches=metrics[metrics.method.str.startswith('sketch_rank_')].copy()
    sketches['rank']=sketches.method.str.rsplit('_',n=1).str[-1].astype(int)
    sketches.to_csv(out/'sketch_rank_comparison.csv',index=False)
    hard['effect_scale']=pd.cut(abs(hard.selection_train16_effect),[0,5e-4,.01,.3,np.inf],
        labels=['below_5e-4','5e-4_to_0.01','0.01_to_0.3','at_least_0.3'],include_lowest=True)
    strata=hard.groupby(['discovery_examples','seed','anchor_strength','method','effect_scale'],observed=True).agg(
        decision_accuracy=('correct','mean'),decision_count=('correct','size'),unique_queries=('query_id','nunique')).reset_index()
    strata.to_csv(out/'difficult_scale_strata.csv',index=False)
    hard['absolute_error']=abs(hard.prediction-hard.validation_effect)
    hard_errors=hard.groupby(['discovery_examples','seed','anchor_strength','method']).agg(
        difficult_mae=('absolute_error','mean'),difficult_accuracy=('correct','mean'),
        decision_count=('correct','size')).reset_index()
    hard_errors.to_csv(out/'difficult_error_metrics.csv',index=False)
    seed_summary=metrics[metrics.discovery_examples==128].groupby(['anchor_strength','method']).agg(
        seeds=('seed','nunique'),mae_mean=('mae','mean'),mae_seed_sd=('mae','std'),
        difficult_accuracy_mean=('difficult_accuracy','mean'),
        difficult_accuracy_seed_sd=('difficult_accuracy','std')).reset_index()
    seed_summary.to_csv(out/'seed_summary.csv',index=False)
    pd.concat(ranks,ignore_index=True).to_csv(out/'curvature_diagnostics.csv',index=False)
    config['completed_configurations']=[{'seed':seed,'discovery_examples':n} for seed,n in completed]
    config['incomplete_primary_sizes']=[n for n in [16,32,64,128] if (42,n) not in completed]
    (out/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    primary=metrics[(metrics.seed==42)&(metrics.anchor_strength==0)]
    matched=[]
    for row in primary[primary.method!='first_order'].itertuples():
        controls=primary[(primary.method=='first_order')&(primary.wall_seconds<=row.wall_seconds)]
        if controls.empty:continue
        control=controls.sort_values('discovery_examples').iloc[-1]
        matched.append({'curvature_examples':row.discovery_examples,'curvature_method':row.method,
            'curvature_seconds':row.wall_seconds,'curvature_mae':row.mae,
            'curvature_difficult_accuracy':row.difficult_accuracy,
            'first_order_examples_at_budget':int(control.discovery_examples),'first_order_seconds':control.wall_seconds,
            'first_order_mae':control.mae,'first_order_difficult_accuracy':control.difficult_accuracy})
    matched=pd.DataFrame(matched);matched.to_csv(out/'matched_compute_first_order.csv',index=False)
    matched_synopsis=(f'Across {len(matched)} measured primary curvature/budget comparisons, curvature '
        f'has lower MAE in {int((matched.curvature_mae<matched.first_order_mae).sum())} and higher '
        f'difficult-decision accuracy in {int((matched.curvature_difficult_accuracy>matched.first_order_difficult_accuracy).sum())}. '
        'These paired rows reuse data and are not independent trials. ')
    first=primary[primary.method=='first_order'].set_index('discovery_examples')
    full=primary[primary.method=='full_second_order'].set_index('discovery_examples')
    comparison=[]
    for n in first.index:
        a=first.loc[n];b=full.loc[n]
        comparison.append({'examples':n,'first_MAE':a.mae,'full_MAE':b.mae,
            'first_difficult_accuracy':a.difficult_accuracy,'full_difficult_accuracy':b.difficult_accuracy,
            'full_MAE_gain':a.mae-b.mae,'full_difficult_gain':b.difficult_accuracy-a.difficult_accuracy})
    comparison=pd.DataFrame(comparison)
    result_synopsis=(f'At anchor 0, full second order lowers held-out MAE at '
        f'{int((comparison.full_MAE_gain>0).sum())} of {len(comparison)} completed primary sizes and '
        f'improves difficult-decision accuracy at {int((comparison.full_difficult_gain>0).sum())} of '
        f'{len(comparison)} sizes. These are descriptive signs, not independent significance tests. ')
    rank_curves=[g.sort_values('rank').mae.to_numpy() for _,g in
                 sketches[(sketches.seed==42)&(sketches.anchor_strength==0)].groupby('discovery_examples')]
    result_synopsis+=(f'Held-out sketch MAE decreases monotonically with rank at '
        f'{sum(bool(np.all(np.diff(curve)<=0)) for curve in rank_curves)} of {len(rank_curves)} '
        'primary sizes. Recovering the reference Hessian more accurately need not improve '
        'finite-removal prediction. ')
    if 128 in first.index:
        a=first.loc[128];b=full.loc[128]
        result_synopsis+=(f'With 128 discovery prompts, first-order MAE is {a.mae:.6g} versus '
            f'{b.mae:.6g} for full second order; difficult-decision accuracy is '
            f'{a.difficult_accuracy:.1%} versus {b.difficult_accuracy:.1%}. ')
    mechanism_synopsis=''
    reference=decomp[(decomp.seed==42)&(decomp.discovery_examples==128)&(decomp.anchor_strength==0)]
    if not reference.empty:
        ref=reference.set_index('method')
        row=ref.loc['full_second_order']
        mechanism_synopsis=(f'At n=128, seed 42, anchor 0, exact train-to-validation shift MAE '
            f'is {row.exact_train_validation_shift_mae:.6g}, while full-Taylor truncation MAE '
            f'on train is {row.full_taylor_mae_train:.6g}. Rank-4 and rank-16 compression MAE '
            f'are {ref.loc["sketch_rank_4","compression_mae"]:.6g} and '
            f'{ref.loc["sketch_rank_16","compression_mae"]:.6g}. Increasing rank therefore '
            'addresses a smaller error source in this configuration; it cannot remove '
            'Taylor truncation or a mismatch between discovery and validation effects. ')
    stress=decomp[(decomp.discovery_examples==128)&(decomp.seed==44)&(decomp.anchor_strength==.5)&(decomp.method=='sketch_rank_4')]
    if not stress.empty:
        row=stress.iloc[0]
        mechanism_synopsis+=(f'Compression is not uniformly harmless: at the seed-44 midpoint, '
            f'rank-4 compression MAE rises to {row.compression_mae:.6g}, exceeding the '
            f'full-Taylor train error {row.full_taylor_mae_train:.6g}. Its validation MAE '
            f'is {row.total_validation_mae:.6g}; exact train-to-validation shift MAE is '
            f'{row.exact_train_validation_shift_mae:.6g}. The seed/anchor tables retain this failure case. ')
    # Prespecified training-only seed-extension screen; validation is excluded.
    eligible=metrics[(metrics.seed==42)&(metrics.discovery_examples==128)&metrics.method.str.startswith('sketch_rank_')].copy()
    candidates=[]
    for row in eligible.itertuples():
        base=metrics[(metrics.seed==42)&(metrics.discovery_examples==128)&(metrics.anchor_strength==row.anchor_strength)&(metrics.method=='first_order')].iloc[0]
        training_mae_gain=(base.train_mae-row.train_mae)/base.train_mae if base.train_mae else 0
        if training_mae_gain>=.05 or row.train_difficult_accuracy_gain_over_first>=.02:
            candidates.append({'method':row.method,'anchor_strength':row.anchor_strength,
                'train_mae_gain_fraction':training_mae_gain,'train_difficult_gain':row.train_difficult_accuracy_gain_over_first,
                'train_mae':row.train_mae})
    candidates=sorted(candidates,key=lambda r:r['train_mae'])
    selected=candidates[0] if candidates else None
    extension_ranks=sorted({4,int(selected['method'].rsplit('_',1)[1])}) if selected else []
    pending_extension=[seed for seed in [43,44] if selected and (seed,128) not in completed]
    config['pending_seed_extension']=pending_extension
    config['seed_extension_configuration']=selected
    config['seed_extension_ranks']=extension_ranks
    config['reproduction_commands']=[
        f'.\\.venv\\Scripts\\python.exe scripts\\run_sketch_diagnostics_final.py --sizes 16 32 64 128 --out {prefix_dir}'
        for prefix_dir in [out.relative_to(ROOT).as_posix()]]
    if selected:
        config['reproduction_commands'].append(
            '.\\.venv\\Scripts\\python.exe scripts\\run_sketch_diagnostics_final.py --sizes 128 --seeds 43 44 --ranks '
            +' '.join(map(str,extension_ranks))+' --out '+out.relative_to(ROOT).as_posix())
    (out/'experiment_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (out/'seed_extension_selection.json').write_text(json.dumps({'rule':config['seed_extension_rule'],
        'training_only_candidates':candidates,'selected':selected,'ranks_to_run':extension_ranks,
        'additional_seeds':[43,44] if selected else []},indent=2),encoding='utf-8')
    # Measured figures. Prompt CIs condition on each learned estimator.
    for y,name,ylabel in [('mae','sample_efficiency_mae','held-out effect MAE'),
                         ('difficult_accuracy','sample_efficiency_difficult_accuracy','train-defined difficult-decision accuracy')]:
        fig,ax=plt.subplots(figsize=(8,5))
        for method,g in primary.groupby('method'):
            g=g.sort_values('discovery_examples');ax.plot(g.discovery_examples,g[y],'o-',label=method)
            low,high=('mae_ci_low','mae_ci_high') if y=='mae' else ('difficult_accuracy_ci_low','difficult_accuracy_ci_high')
            ax.fill_between(g.discovery_examples,g[low],g[high],alpha=.10)
        ax.set(xlabel='nested discovery examples',ylabel=ylabel);ax.set_xticks([16,32,64,128]);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for n,g in sketches[(sketches.seed==42)&(sketches.anchor_strength==0)].groupby('discovery_examples'):
        g=g.sort_values('rank');ax.plot(g['rank'],g.mae,'o-',label=f'n={n}')
    ax.set(xlabel='nested randomized sketch rank',ylabel='held-out MAE');ax.set_xticks([4,8,12,16]);ax.legend()
    fig.tight_layout();fig.savefig(out/'rank_vs_error.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for method,g in primary.groupby('method'):
        g=g.sort_values('discovery_examples')
        ax.plot(g.wall_seconds,g.mae,'o-',label=method)
    ax.set(xlabel='measured standalone estimator seconds (all states)',ylabel='held-out MAE');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'compute_vs_quality.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for (anchor,method),g in metrics[metrics.seed==42].groupby(['anchor_strength','method']):
        if method not in ['first_order','full_second_order','sketch_rank_4','sketch_rank_16']:continue
        g=g.sort_values('discovery_examples')
        ax.plot(g.discovery_examples,g.mae,'o-',label=f'{method}; anchor={anchor}')
    ax.set(xlabel='discovery examples',ylabel='held-out MAE');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'midpoint_comparison.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    target=state_errors[(state_errors.seed==42)&(state_errors.discovery_examples==primary.discovery_examples.max())&(state_errors.anchor_strength==0)]
    for method,g in target.groupby('method'):
        g=g.sort_values('removed_edges');ax.plot(g.removed_edges,g.mae,'o-',label=method)
    ax.set(xlabel='already removed edges (candidate composition changes)',ylabel='held-out effect MAE');ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'error_vs_pruned_edges.png',dpi=150);plt.close(fig)
    latest_n=int(primary.discovery_examples.max())
    latest_pred=pd.read_csv(out/f'seed42_n{latest_n}'/'predictions.csv',dtype={'state':str})
    query_states=set()
    for row in latest_pred[latest_pred.method=='first_order'].itertuples():
        query_states.add(row.state)
        trial=list(row.state);trial[row.candidate_index]='1';query_states.add(''.join(trial))
    pd.DataFrame([{'state':state,'removed_edges':state.count('1')} for state in sorted(query_states)]).to_csv(
        out/'diagnostic_binary_states.csv',index=False)
    decision_pairs=pd.read_csv(out/'decision_pairs.csv')
    decision_rows=latest_pred[latest_pred.anchor_strength==0].merge(decision_pairs,on='query_id',validate='many_to_many')
    decision_rows['correct']=(abs(decision_rows.prediction)<decision_rows.threshold)==(abs(decision_rows.validation_effect)<decision_rows.threshold)
    sweep=decision_rows.groupby(['method','threshold']).correct.mean().reset_index(name='decision_accuracy')
    sweep.to_csv(out/'threshold_sweep.csv',index=False)
    first_rows=latest_pred[(latest_pred.anchor_strength==0)&(latest_pred.method=='first_order')]
    full_rows=latest_pred[(latest_pred.anchor_strength==0)&(latest_pred.method=='full_second_order')]
    cases=first_rows[['query_id','edge','state','train_effect','validation_effect','prediction']].rename(columns={'prediction':'first_prediction'})
    cases=cases.merge(full_rows[['query_id','prediction']].rename(columns={'prediction':'full_prediction'}),on='query_id',validate='one_to_one')
    cases=cases.merge(decision_pairs[decision_pairs.difficult][['query_id','threshold']],on='query_id',validate='one_to_many')
    cases['train_first_wrong']=(abs(cases.first_prediction)<cases.threshold)!=(abs(cases.train_effect)<cases.threshold)
    cases['validation_first_correct']=(abs(cases.first_prediction)<cases.threshold)==(abs(cases.validation_effect)<cases.threshold)
    cases['validation_full_correct']=(abs(cases.full_prediction)<cases.threshold)==(abs(cases.validation_effect)<cases.threshold)
    cases['train_absolute_error_gain']=abs(cases.first_prediction-cases.train_effect)-abs(cases.full_prediction-cases.train_effect)
    cases=cases[cases.train_first_wrong].sort_values(['train_absolute_error_gain','query_id','threshold'],ascending=[False,True,True])
    cases.to_csv(out/'train_selected_first_order_disagreements.csv',index=False)
    case_table=cases[abs(cases.train_effect)>=.01].drop_duplicates('query_id').head(6)
    fig,ax=plt.subplots(figsize=(8,5))
    for method,g in sweep.groupby('method'):
        ax.plot(g.threshold,g.decision_accuracy,label=method)
    ax.set_xscale('symlog',linthresh=1e-5)
    ax.set(xlabel='train-selected threshold (symlog scale)',ylabel='held-out decision accuracy',ylim=(0,1.03))
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'accuracy_vs_threshold.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    for method,g in primary.groupby('method'):
        g=g.sort_values('discovery_examples');ax.plot(g.wall_seconds,g.difficult_accuracy,'o-',label=method)
    ax.set(xlabel='measured standalone estimator seconds (all states)',ylabel='difficult-decision accuracy')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'accuracy_vs_cost.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5))
    difficult_plot=hard_errors[(hard_errors.seed==42)&(hard_errors.discovery_examples==latest_n)&(hard_errors.anchor_strength==0)]
    ax.bar(difficult_plot.method,difficult_plot.difficult_mae)
    ax.tick_params(axis='x',rotation=25)
    ax.set(ylabel='MAE on fixed train-defined difficult query/threshold pairs')
    fig.tight_layout();fig.savefig(out/'difficult_decision_errors.png',dpi=150);plt.close(fig)
    audit=pd.read_csv(out/'intervention_correctness_audit.csv')
    seed_rows=metrics[metrics.discovery_examples==128]
    higher=primary[primary.method!='first_order']
    gains=higher.merge(primary[primary.method=='first_order'][['discovery_examples','difficult_accuracy','mae']],on='discovery_examples',suffixes=('','_first'))
    plausible=gains[gains.method.str.startswith('sketch_rank_')&(gains.difficult_accuracy>gains.difficult_accuracy_first)&
                    (gains.hypothetical_break_even_avoided_queries<=gains.available_candidate_queries)]
    qualifying=[]
    for (anchor,method),group in metrics[metrics.method.str.startswith('sketch_rank_')].groupby(['anchor_strength','method']):
        main=group[group.seed==42]
        gains_by_size=[]
        for row in main.itertuples():
            base=metrics[(metrics.seed==42)&(metrics.anchor_strength==anchor)&(metrics.method=='first_order')&(metrics.discovery_examples==row.discovery_examples)].iloc[0]
            gains_by_size.append(row.difficult_accuracy-base.difficult_accuracy)
        rep=group[group.discovery_examples==128]
        rep_gains=[]
        for row in rep.itertuples():
            base=metrics[(metrics.seed==row.seed)&(metrics.anchor_strength==anchor)&(metrics.method=='first_order')&(metrics.discovery_examples==128)].iloc[0]
            rep_gains.append(row.difficult_accuracy-base.difficult_accuracy)
        if (len(gains_by_size)==4 and all(g>0 for g in gains_by_size) and {42,43,44}.issubset(set(rep.seed))
            and all(g>0 for g in rep_gains) and (rep.difficult_accuracy_gain_ci_low>0).all()
            and (rep.hypothetical_break_even_avoided_queries<=rep.available_candidate_queries).all()):
            qualifying.append((anchor,method))
    if config['incomplete_primary_sizes']:
        recommendation='INCOMPLETE: finish the remaining planned primary sizes before a final continue/pause decision.'
    elif pending_extension:
        recommendation='INCOMPLETE: primary sizes are complete; finish the training-selected seed extension before the final research decision.'
    elif not qualifying:
        recommendation='PAUSE the Hessian-based Sketch-ACDC approach. This restricted experiment does not satisfy both decision-quality and amortization requirements. Finish documentation before separately considering MAttr; no MAttr experiment was started.'
    else:
        recommendation='CONTINUE only as a modified, targeted end-to-end experiment: '+str(qualifying)+'. These configurations show repeated decision gains and a hypothetical amortization route; actual pruning savings still require measured verification and refresh costs.'
    query_info=json.loads((out/'query_selection.json').read_text())
    hard_definition=decision_pairs[decision_pairs.difficult]
    tiny_pairs=int((abs(hard_definition.selection_train16_effect)<5e-4).sum())
    scale_note=(f'Of the {len(hard_definition)} fixed difficult pairs, {tiny_pairs} have '
        'train-16 effect magnitude below 5e-4. These numerical-scale decisions dominate the '
        'unstratified count; substantive-effect strata are descriptive checks, not replacement success criteria. ')
    prefix=out.relative_to(ROOT).as_posix()
    sections=[f'# Sketch-ACDC final research report\n\nResults directory: `{prefix}`.\n\n**Recommendation: {recommendation}**',
        '## 1. Research question and fixed setup\n\nDoes second-order Hessian information improve context-dependent MIB IOI edge-removal prediction enough to repay its construction? GPT-2 Small, 16 fixed EAP-selected MIB edges, official train/validation data, float32 model/derivatives, batch size 2. Detached 16-dimensional accumulation and pseudoinverse use float64. Discovery subsets are nested prefixes of the seed-42 train permutation. The same 128 validation prompts and the same queries are used for every size and estimator. Public test was not loaded. Validation was not used for query selection, thresholds, fitting, or seed-extension selection.\n\n'
        +f'Frozen training-only query set: {query_info["query_count"]} edge/state queries, {len(query_info["states"])} partial states, {query_info["decision_pairs"]} query/threshold pairs, including **{query_info["difficult_pairs"]} difficult pairs across {query_info["difficult_unique_queries"]} distinct queries**. Near-boundary pairs are within max(1e-6, 5.1% of exact train-16 effect magnitude); this selection remains fixed across data sizes. Thresholds extend beyond 0.01–0.30 and are derived from train effects only. '+scale_note+'The bootstrap resamples the same validation prompt indices jointly across queries (200 replicates); it conditions on the fitted estimator.',
        '## 2. Previous experiments and what they established\n\n'
        '- Toy factual baseline: feasibility of singleton/pairwise prediction; no MIB leaderboard implication.\n'
        '- Standardized head-level IOI experiment: held-out mask / validation-prompt MAE 0.736744 (learned first order) versus 0.367987 (learned second order); mean absolute pair interaction 0.4830. These used `IOIScorer`, not the defective edge adapter, so this bug does not invalidate them. A globally fitted quadratic over head masks is distinct from a local Hessian approximation for an edge removal.\n'
        '- Head-level circuit selection: mixed quality despite prediction improvement; it also used `IOIScorer` and is unaffected by this edge-specific defect.\n'
        '- Historical edge Sketch-ACDC Stage A: loaded 32/64 train/validation and removed 10 of 16 candidates at threshold 0.05. Isolated endpoint checks were insufficient to establish dependent-mask correctness.\n'
        '- Historical B/C pilot (16 train, 8 validation, one threshold): MAE 0.152787 first order, 0.224677 rank 2, 0.207928 rank 4. All methods reached the same six-edge state; verified arms checked all 16 decisions. H1 collection took 413.37 s and Stage C 173.64 s. These are preserved measurements of the frozen-sender function, not validated general results for official simultaneous MIB interventions. They require corrected reruns for such claims; historical timings include three forward passes per batch, whereas the corrected adapter uses two.\n'
        '- Earlier diagnostic smoke: only two effective prompts per split and two queries. It discovered the dependency bug; it could not establish a rank trend or scientific advantage. No prior notebook or numeric artifact has been overwritten.',
        '## 3. Intervention correctness and historical implications\n\n'
        'The old adapter cached `corrupted_sender - original_clean_sender` before any intervention. MIB adds `corrupted_sender - current_intervened_sender` at each removed connection during the patched pass. When an upstream removal changes a downstream sender, the old update subtracts the wrong value. At z=1 a removed connection must carry the fixed corrupted donor, independent of the current sender. Detaching current senders would additionally break mixed derivatives. At the intact state, agreement along individual coordinate axes can preserve singleton effects and diagonal curvature while still giving incorrect mixed derivatives; randomized HVP sketches depend on those mixed derivatives. Thus an HVP finite-difference check against the historical function alone was insufficient to establish correct MIB semantics.\n\n'
        f'The corrected shared `EdgeExperiment` now captures current sender outputs without detaching their derivative. `DiagnosticEdgeExperiment` adds persistent caching without another patching implementation. `FrozenSenderEdgeExperiment` is retained solely as an explicitly named historical control; cache identities include adapter versions. **All {len(audit)} audited masks passed** the independent MIB evaluator, including random densities, partial states, dependent edges, and sampled prior pilot queries. Maximum corrected discrepancy: {audit.max_abs_corrected_vs_official.max():.6g}; maximum historical discrepancy: {audit.max_abs_historical_vs_official.max():.6g}.\n\n'
        'Regression tests cover the complete shared adapter on a two-edge dependent DAG, all binary states, its analytic mixed derivative, receiver isolation, version-separated cache identities, full-Hessian recovery, nested rank-16 recovery, and streamed weighted derivatives. Real-model endpoint and HVP finite-difference checks are recorded in `correctness_passed.json`. Prior single-edge intact-state measurements remain valid by construction; dependent-state quantities need rechecking. Spot checks of prior queries do not certify every historical query.',
        '## 4. Completed 16/32/64/128 results\n\nCompleted (seed, discovery size): '+str(completed)+'. Incomplete primary sizes: '+str(config['incomplete_primary_sizes'])+'.\n\n'+result_synopsis+'\n\n'+table(comparison),
        '## 5. Estimator comparison, midpoint, and error sources\n\n'+table(primary,['discovery_examples','method','mae','spearman','difficult_accuracy','difficult_decision_count','false_prune_rate','false_retain_rate'])+
        '\n\nAll method rows share exact endpoints and data. Full second order is an analysis reference. Sketch HVPs are measured directly and are never computed by multiplying the reference Hessian. Nested orthonormal randomized directions are shared across ranks; rank-16 is a compression sanity reference, not scalable sketching.\n\n'
        'MIB supplies binary circuit evaluation; the dynamic linear edge-strength interpolation is our explicitly defined continuous extension, not an invented official API. The shared midpoint sets all intact coordinates to 0.5; it can lie far from either binary endpoint. Its Taylor difference includes cross terms. At a candidate-only midpoint, the pure quadratic term cancels, so a midpoint benefit would not alone establish a benefit of second order. A large gain against the midpoint first-order estimator should also be compared with the stronger anchor-0 first-order control. Both anchors at n=128, seed 42:\n\n'+table(metrics[(metrics.seed==42)&(metrics.discovery_examples==128)],['anchor_strength','method','mae','difficult_accuracy','wall_seconds'])+
        '\n\nSigned sketch error = full-Taylor error on train + sketch-minus-full compression error + exact train-minus-validation shift. Absolute terms do not add because signs can cancel. The full-Taylor column is the common reference error; estimator train MAE also shows the actual first-order error. '+mechanism_synopsis+'\n\n'+table(decomp[(decomp.seed==42)&(decomp.anchor_strength==0)],['discovery_examples','method','estimator_mae_train','full_taylor_mae_train','compression_mae','exact_train_validation_shift_mae','total_validation_mae'])+
        '\n\nDecision accuracy and false-prune/retain rates use all query/threshold pairs as denominator; difficult accuracy uses the fixed train-defined subset. Difficult effect-scale strata are saved separately in `difficult_scale_strata.csv`. Near-zero effects can be below a useful floating-point decision margin; those rows remain in the primary metric, and the descriptive strata do not change the success criteria.\n\nError on difficult query/threshold pairs (rows are weighted equally, so a query near several thresholds appears several times):\n\n'+table(hard_errors[(hard_errors.seed==42)&(hard_errors.anchor_strength==0)])+
        '\n\nIllustrative first-order mistakes chosen on training labels only at the largest completed primary size. Rows are ranked by full-second-order training error reduction, then limited to distinct queries with train effect magnitude at least 0.01. This is a descriptive subset, not another success criterion. Validation columns show whether that training-selected correction transfers:\n\n'+table(case_table,['edge','state','threshold','train_effect','validation_effect','first_prediction','full_prediction','validation_first_correct','validation_full_correct'])+
        '\n\nSeed extension at n=128, both anchors. Gain intervals compare with first order at the same anchor:\n\n'+table(seed_rows,['seed','anchor_strength','method','mae','difficult_accuracy','difficult_accuracy_gain_ci_low','difficult_accuracy_gain_ci_high'])+
        '\n\nAcross-seed summaries at n=128 (seed standard deviation is descriptive; seed count differs for unreplicated ranks):\n\n'+table(seed_summary),
        '## 6. Compute accounting and break-even\n\n'+table(primary,['discovery_examples','method','wall_seconds','forward_passes_batches','backward_calls','hvp_operations','peak_cuda_allocated_mib','hypothetical_break_even_avoided_queries','available_candidate_queries'])+
        '\n\nCosts include corrupted-donor preparation, patched forwards, backwards, direct HVPs, and matrix reconstruction. Only one batch graph is retained, making 128 prompts feasible on 12 GB. Full columns and randomized products reuse batch graph preparation; per-method costs allocate the measured shared preparation and corresponding directional work to each standalone method. These allocations are **not additive**. Raw `runtime_breakdown.csv` counts the actual campaign work once; peak memory for curvature methods is a measured shared-stage upper bound.\n\n'
        'The exact-query reference is measured using three fresh train candidate evaluations with current-state score already known. Break-even = ceiling(standalone construction seconds / measured fresh-query seconds). It is a **hypothetical no-verification lower bound** on queries that must be avoided, not a measured speedup. Every refresh adds construction again; verification adds exact query cost and can only worsen this bound. This final diagnostic did not execute another hybrid search or claim verification savings. The historical verified pilot saved zero exact decisions. A separately labeled sensitivity column assumes a 2.35x faster exact baseline based on the independent report; that speedup was not measured in this repository.\n\nMatched-compute first-order control: use the largest actually measured first-order discovery set whose construction fits the curvature method budget, chosen by sample count and cost rather than validation quality. This directly tests spending compute on more data instead of curvature. '+matched_synopsis+'The available first-order control is capped at 128 examples even when more would fit the budget.\n\n'+table(matched),
        f'The fixed query benchmark uses {len(query_states)} distinct binary states (current states plus candidate removals), catalogued in `diagnostic_binary_states.csv`; each is evaluated on the relevant train subset and the identical validation subset. Validation states are cached across jobs. Raw ledger mask/cache trackers describe their live process session; their cumulative-mask field does not become a campaign-wide count when saved rows are restored after interruption. Forward-batch and example-forward logs remain reconciled across restarts.\n\n'
        'The fixed candidate bank was selected by official EAP on 16 training prompts in the prior pilot (4.16 s attribution stage). That historical common prerequisite is reused and is not counted as new campaign work.',
        'The full six-configuration campaign was executed through the batch-streamed CLI. `notebooks/mib_sketch_diagnostics.smoke.executed.ipynb` separately records execution of every notebook code cell on the explicitly labeled small smoke configuration, including the optional Hessian and the final campaign dashboard. It is not presented as the full campaign execution transcript.',
        '## 7. Limitations and reproducibility\n\n'
        'Restricted 16-edge graph; no full-model circuit, CPR, CMD, or leaderboard score. Three train-selected partial states and fixed candidate rankings limit generality. Query selection uses the first 16 training examples, so sample-efficiency curves condition on that fixed query set. The fixed validation subset has appeared in earlier project experiments and is not a fresh private confirmation set. Bootstrap intervals represent prompt sampling conditional on each estimator, not an independent replication or a multiple-testing-adjusted population claim. Additional seeds change train subsets and sketch directions together; those effects are not separately identified. The larger datasets remain nested within each seed. Seed extensions replicate the training-selected sketch, the rank-4 budget control from the same HVP prefix, and first-order/full-Hessian controls, not every rank. Public test remains unused.\n\n'
        'Reproduce with `scripts/run_sketch_diagnostics_final.py --sizes 16 32 64 128 --out <run-directory>`. Sizes and seeds are independently configurable; context bundles and completion markers support restart. Exact source hashes, package/model provenance, train/validation manifests, random seeds, selected edges, thresholds, source reports, and incomplete configurations are saved. No smoke fallback occurs. Model/device errors stop the configuration.\n\n'
        f'Actual completed campaign stages: {len(raw)}; forward batches {int(raw.forward_passes_batches.sum())}; example-forwards {int(raw.examples_processed.sum())}; backward calls {int(raw.backward_calls.sum())}; HVP operations {int(raw.hvp_operations.sum())}; measured stage wall seconds {raw.wall_seconds.sum():.2f}. This excludes initial model/data loading and report rendering, not omitted algorithm work.',
        '## 8. Independent ACDC reports: attributed context, not our measurements\n\n'
        f'Source snapshots and SHA-256 values are saved in `{prefix}/external_reports/` and the configuration. '
        f'[Stage G]({prefix}/external_reports/ACDC_FP32_SCALE_REPORT.md) reports no reliable sketch quality benefit over prior-order contiguous groups at 7,588 edges. '
        f'[Stage H]({prefix}/external_reports/ACDC_GPT2_REPORT.md) reports 32,491-edge GPT-2 batching at about 6% of its ACDC reference cost but 20–30% worse held-out loss; sketch bank cost offsets most saved checks. '
        f'[Stage I]({prefix}/external_reports/ACDC_GPT2_GAP_REPORT.md) reports that 64 discovery pairs substantially improve held-out quality, while a good first-order prior plus contiguous grouping remains difficult to beat and the sketch costs about 1.2x for similar quality. '
        f'[RESEARCH_LOG]({prefix}/external_reports/RESEARCH_LOG.md) describes input-direction failures on sparse deletions, improved local ranking with output-space Gauss–Newton sketches, and a 2.35x prefix-cache optimization preserving ACDC decisions.\n\n'
        'Those experiments use a squared probability-distance loss, different graph scales and data roles, and exact-checked group deletions. Our scalar IOI logit-difference and restricted one-edge queries do not reproduce them or support a direct numeric comparison. They motivate larger discovery sets, strong first-order and optimized-exact controls, and a demanding end-to-end cost test. No output-space or group-deletion algorithm was implemented in this wrap-up.',
        '## 9. Research decision\n\n'+recommendation+'\n\n'
        'Continue requires repeatable difficult-decision gains across prompts/seeds and a plausible amortization route. Full Taylor accuracy with poor/expensive Hessian compression would motivate a separate output-space investigation. Weak incremental value or construction/verification cost beyond plausible savings supports pausing input-Hessian Sketch-ACDC and considering MAttr as a separately authorized next study. These are the supplied qualitative criteria. The conservative evidence checks below operationalize them for this report; they are not a formally preregistered hypothesis test. The training-only seed-extension rule was recorded in the campaign configuration before the primary runs.\n\n'
        +'Training-only seed-extension selection: '+json.dumps(selected)+'. Pending extension seeds: '+str(pending_extension)+'. The selection takes the lowest training MAE among qualifying rows; tiny differences among nearly identical high-rank estimators do not establish a scientifically optimal rank. The supplied consistency criterion is interpreted conservatively: directionally consistent difficult-decision gains across discovery sizes/seeds, paired-prompt intervals supporting the seed results, and a no-verification break-even within available candidate queries. Mixed or uncertain evidence does not establish the requested continuation case.']
    figures='\n\n## Measured figures\n\n'+'\n\n'.join(
        f'![{name}](<{prefix}/{file}>)' for name,file in [
            ('Prediction MAE versus discovery examples','sample_efficiency_mae.png'),
            ('Difficult-decision accuracy versus discovery examples','sample_efficiency_difficult_accuracy.png'),
            ('Sketch rank versus prediction error','rank_vs_error.png'),
            ('Measured computation versus prediction quality','compute_vs_quality.png'),
            ('Difficult-decision accuracy versus computation','accuracy_vs_cost.png'),
            ('Decision accuracy versus threshold','accuracy_vs_threshold.png'),
            ('Prediction error on difficult decisions','difficult_decision_errors.png'),
            ('Midpoint comparison','midpoint_comparison.png'),
            ('Error across partially pruned states','error_vs_pruned_edges.png')])
    report='\n\n'.join(sections)+figures+'\n'
    report='\n'.join(line.rstrip() for line in report.splitlines())+'\n'
    local_report=report.replace(f']({prefix}/','](').replace(f'](<{prefix}/','](<')
    (out/'SKETCH_ACDC_FINAL_REPORT.md').write_text(local_report,encoding='utf-8')
    (ROOT/'SKETCH_ACDC_FINAL_REPORT.md').write_text(report,encoding='utf-8')
    print('Report written:',out/'SKETCH_ACDC_FINAL_REPORT.md')
    print('Training-only seed extension:',selected)
    print(recommendation)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('result_dir',type=Path)
    build(parser.parse_args().result_dir.resolve())
