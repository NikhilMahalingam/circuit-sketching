"""Post-execution descriptive summary; frozen decisions and sources unchanged.

Run from the repository root after the registered report and preservation audit.
No model evaluation, hypothesis test, tuning, or decision rule is added here.
"""
from pathlib import Path
import json,sys,pandas as pd
r=Path.cwd();o=r/'results/mattr_matched_size/20261023_gate5e'
sys.path.insert(0,str(r))
s=json.loads((o/'research_summary.json').read_text());d=s['primary'];p=json.loads((o/'gate5e_protocol.json').read_text())
cap=d['cap_T1'];nom=d['nominal_0.5pct'];ratio=d['advantage_retention_ratio']
R=(f"{100*ratio['point']:.2f}% (paired 95% interval {100*ratio['lower']:.2f}% to {100*ratio['upper']:.2f}%)" if ratio['stable'] else 'unstable/undefined under the frozen denominator rule')
matched=pd.read_csv(o/'matched_results.csv');means=matched[(matched.selection=='cap')&(matched.kind=='original')].groupby(['method','target']).accuracy.mean().unstack('target')
control=pd.read_csv(o/'paired_cap_comparisons.csv');control=control[(control.seed=='equal_seed_mean')&control.target.isin([p['T1'],p['T2']])]
old=s['nominal_original_means'];base=old['baseline'];hier=old['hierarchy']
ens=pd.read_csv(o/'ensemble_vs_components.csv');ens=ens[ens.target.isin([p['T1'],p['T2']])]
closure=pd.read_csv(o/'closure_nominal_effects.csv');clo=closure.drop_duplicates('original').copy();clo['method']=clo.original.str.replace(r'_s\d+$','',regex=True)
clomean=clo.groupby('method')[['CPR_delta','CMD_delta']].mean().reset_index();clomean.to_csv(o/'closure_area_effect_summary.csv',index=False)
compact=closure[closure.fraction==.005].copy();compact['method']=compact.original.str.replace(r'_s\d+$','',regex=True);compact.groupby('method')[['original_actual','closure_actual','original_accuracy','closure_accuracy','accuracy_delta']].mean().to_csv(o/'closure_halfpercent_summary.csv')
from src.mattr_optimization.gate2_final_analysis import table
seed_changes=', '.join(f'{100*x:+.2f}' for x in d['per_seed_cap_changes'])
t2=s['capped_T2'][0];baseline_ensemble=s['baseline_ensemble_T2'][0]
assert json.loads((o/'final_validation.json').read_text())['passed']
text=f'''## Outcome interpretation

Frozen classification: **{s['classification']}**. Primary capped-T1 success: **{s['primary_success']}**. Official pruning passed monotonicity and nestedness for every audited prefix. The fresh nominal 0.5% gain was **{100*nom['mean']:.2f} percentage points** (paired 95% interval {100*nom['lower']:.2f} to {100*nom['upper']:.2f}). At capped **T1={p['T1']}**, hierarchy averaged **{100*means.loc['hierarchy',p['T1']]:.2f}%** accuracy versus **{100*means.loc['baseline',p['T1']]:.2f}%**, a **{100*cap['mean']:.2f}-point** difference (95% interval {100*cap['lower']:.2f} to {100*cap['upper']:.2f}); seed changes 42/43/44 were **{seed_changes} points**. **{d['positive_cap_seeds']}/3** seeds improved. These are positive-margin IOI accuracies, separate from vocabulary top-1.

The **advantage-retention ratio R is {R}**. It is not a causal proportion explained by pruning. The registered primary comparison is capped: seed 42 compares 116 against 116 actual edges; seeds 43/44 compare 114/113 hierarchy edges against 116 baseline edges. Only seed 42 is exact at T1. Across all 10 targets and 3 paired seeds, **{s['exact_primary_pairs']}/{s['registered_primary_pairs']}** comparisons are exact. The exact subsets and their prompt intervals are reported below; capped success does not establish three-seed exact-size superiority.

At secondary **T2={p['T2']}**, the pooled hierarchy-minus-baseline difference was **{100*t2['mean']:.2f} points** (descriptive 95% interval {100*t2['lower']:.2f} to {100*t2['upper']:.2f}). Actual sizes at T2 are 296/295 for baseline/hierarchy seed 42, and 296/296 for seeds 43/44. This pooled gain is driven by seed 42; the exactly matched seeds 43/44 average a small -0.39-point difference (interval includes zero). At 400 actual edges the exactly matched mean gain is +1.89 points; at 640 it is zero. Dense curves therefore show a compact benefit that fades as accuracy saturates. These secondary endpoints do not change the primary decision.

Control mean positive-margin accuracy:

{table(means[[p['T1'],p['T2']]].reset_index())}

Secondary paired contrasts (hierarchy minus comparator, accuracy fractions):

{table(control[['comparator','target','mean','lower','upper','positive_seeds']])}

A gamma-zero tie or advantage means the residual penalty is not established as the cause of the compact gain. Real grouping beat shuffled grouping by 20.64 points at T1 and 3.84 points at T2 on these capped comparisons; the secondary prompt intervals above exclude zero. These findings favor a structural parameter-sharing explanation over residual regularization alone, conditional on the fixed rankings and cohort. They do not establish training-seed significance. No group-stability correlation alone establishes causal fidelity.

The baseline ensemble reached **{100*baseline_ensemble['ensemble_accuracy']:.2f}%** accuracy at capped T2 (actual **{baseline_ensemble['actual_edges']}** edges); the prespecified 90% diagnostic **{'passed' if baseline_ensemble['baseline_T2_90pct_pass'] else 'failed'}**. Its component seed 42/43/44 accuracies there were **{100*baseline_ensemble['seed42_accuracy']:.2f}%/{100*baseline_ensemble['seed43_accuracy']:.2f}%/{100*baseline_ensemble['seed44_accuracy']:.2f}%**. This reduces the poor seed-42 outcome but does not beat the strongest seed. All ensemble comparisons and their three-run training costs remain separate from marginal construction/inference costs.

Closure mean changes in official CPR/CMD:

{table(clomean)}

MIB integrates the nominal fraction coordinates, so closure changes are **nominal-grid counting effects**, with actual sizes listed in closure_nominal_effects.csv. They are not matched-size ranking-quality evidence. No matched-size closure evaluations were run. Closure audit covers all known first-appearance boundaries from original prefixes <= 1624; appended-tail nesting samples do not establish a pruning-free tail. Diagnostic rank correlations use canonical ordinal tie breaks; original ranking evaluation retains the pinned official Torch sort.

Original nominal mean CPR was **{base['CPR']:.6f} baseline versus {hier['CPR']:.6f} hierarchy**; CMD was **{base['CMD']:.6f} versus {hier['CMD']:.6f}**. Thus the compact accuracy benefit did not improve leaderboard CPR. CPR, CMD, compact decision recovery and full-model fidelity remain distinct. Gate 5D's mixed decision is preserved.

Measured inference-process wall time: **{s['inference_process_seconds']/60:.2f} minutes**, evaluation-stage wall time **{s['evaluation_seconds']/60:.2f} minutes**, CPU topology/construction **{s['graph_analysis_seconds']:.2f} seconds**. There were **{s['actual_forward_calls']:,}** real root forwards and **{s['example_forwards']:,}** example-forwards, with **{s['output_cache_hits']}** exact-output cache hits. CUDA peaks: **{s['peak_allocated_mib']/1024:.2f} GiB allocated**, **{s['peak_reserved_mib']/1024:.2f} GiB reserved**; these are not physical GPU occupancy or kernel-utilization timers. Zero optimizer updates or backward calls were executed.

Registered-target skip lists for all 18 original/ensemble rankings:

{table(pd.read_csv(o/'registered_target_match_summary.csv'))}

Recommendation: prioritize a future independent evaluation of the existing gamma-zero component parameterization and rank ensembles for compact circuits, while retaining standard MAttr as the CPR reference. The penalty has not demonstrated value here. Treat closure as a separate nominal-counting direction and charge ensembles for three trained runs. This is a recommendation only. Stop after Gate 5E; no further gate, training, new mask/refinement algorithm, or leaderboard submission has run.

'''
(o/'INTERPRETATION.md').write_text(text.rstrip()+'\n')
for path in [o/'GATE5E_MATCHED_SIZE_REPORT.md',r/'GATE5E_MATCHED_SIZE_REPORT.md']:
 body=path.read_text();heading,rest=body.split('\n',1)
 if rest.lstrip().startswith('## Outcome interpretation'):
  rest=rest[rest.index('## Actual frozen decision'):]
 body=heading+'\n\n'+text+rest.lstrip()
 if path.parent==o:
  body=body.replace(f'(results/mattr_matched_size/{o.name}/figures/','(figures/')
 path.write_text(body)
print('Added descriptive result interpretation; protocol/decisions unchanged')
