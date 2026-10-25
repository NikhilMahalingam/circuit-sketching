"""Post-execution interpretation of frozen Gate5F outputs; zero model forwards.

Run from the repository root. Does not alter methods, criteria or decisions.
"""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd

ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
from src.mattr_data_scaling.core import verified
from src.mattr_optimization.gate2_final_analysis import table

OUT=ROOT/'results/mattr_data_scaling/gate5f'
summary=json.loads((OUT/'gate5f_decision.json').read_text(encoding='utf-8'))
primary=summary['primary'];estimate=primary['primary']
assert json.loads((OUT/'final_audit.json').read_text())['passed']
assert json.loads((OUT/'completeness.json').read_text())['completed']==64
assert sum(verified(folder) for folder in (OUT/'evaluations').glob('*/*') if folder.is_dir())==64
caps=pd.read_csv(OUT/'accuracy_size_curves.csv')
metrics=pd.read_csv(OUT/'official_metrics.csv')
seed=pd.read_csv(OUT/'primary_seed_effects.csv').query("cohort=='primary'").copy()
factorial=pd.read_csv(OUT/'factorial_contrasts.csv')
factorial=factorial[(factorial.cohort=='primary')&(factorial.seed=='equal_seed_mean')].set_index('contrast')

# Independently recompute the five deciding differences from per-prompt records.
effects=[]
for r in range(42,47):
    vectors=[]
    for method in ('baseline','h2'):
        folder=OUT/'evaluations/primary'/f'{method}_n10000_s{r}_step5000'
        assert verified(folder)
        frame=pd.read_csv(folder/'cap_per_prompt.csv').query('target==116').sort_values('example')
        assert frame.example.tolist()==list(range(512))
        vectors.append(frame.correct.to_numpy(dtype=float))
    effects.append(float((vectors[1]-vectors[0]).mean()))
np.testing.assert_array_equal(effects,primary['per_seed'])
assert np.isclose(np.mean(effects),estimate['mean'])

full=caps[(caps.cohort=='primary')&(caps.dataset_size==10000)&((caps.step==5000)|(caps.method=='eap_ig'))]
budget_means=full.groupby(['method','target'])[['accuracy','actual_edges','top1','faithfulness','mean_margin']].mean()
metric_means=metrics[(metrics.cohort=='primary')&(metrics.dataset_size==10000)&((metrics.step==5000)|(metrics.method=='eap_ig'))].groupby('method')[['CPR','CMD','accuracy_log_AUC']].mean()
comparison=budget_means.xs(116,level='target').join(metric_means)
comparison.to_csv(OUT/'full_data_primary_summary.csv')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(9,5))
for method,label in [('baseline','Ordinary MAttr (5 fixed seeds)'),('h2','H2 gamma=0 (5 fixed seeds)'),('eap_ig','EAP-IG (one full-data ranking)')]:
    curve=budget_means.xs(method,level='method').sort_index()
    ax.plot(curve.actual_edges,100*curve.accuracy,'o-',label=label)
ax.set(xlabel='Actual retained edges (seed mean for MAttr/H2)',ylabel='Positive-margin IOI accuracy (%)',title='Full discovery pool: MAttr/H2 at 5,000 updates')
ax.axvline(116,linestyle=':',color='grey',alpha=.7)
ax.legend();fig.tight_layout();fig.savefig(OUT/'figures/full_data_accuracy_comparison.png',dpi=150);plt.close(fig)
actual=full[(full.step==5000)&(full.target==116)].pivot(index='seed',columns='method',values='actual_edges')
seed['baseline_actual_edges']=seed.seed.map(lambda r:int(actual.loc[str(r),'baseline']))
seed['h2_actual_edges']=seed.seed.map(lambda r:int(actual.loc[str(r),'h2']))
seed['baseline_accuracy_pct']=seed.baseline_accuracy*100
seed['h2_accuracy_pct']=seed.h2_accuracy*100
seed['difference_pp']=seed.effect*100
seed.to_csv(OUT/'primary_seed_interpretation.csv',index=False)

factor=caps[(caps.cohort=='primary')&(caps.target==116)&caps.method.isin(['baseline','h2'])&caps.seed.isin(['42','43','44'])].groupby(['dataset_size','step','method']).accuracy.mean().unstack()
factor['difference_pp']=100*(factor.h2-factor.baseline)
factor[['baseline','h2']]*=100
factor=factor.reset_index().rename(columns={'baseline':'baseline_accuracy_pct','h2':'h2_accuracy_pct'})
factor.to_csv(OUT/'factorial_interpretation.csv',index=False)

usage=[]
for path in sorted(OUT.glob('stream_n*_s*.csv')):
    name=path.stem;n=int(name.split('_')[1][1:]);r=int(name.split('_')[2][1:])
    frame=pd.read_csv(path)
    for steps in (1000,5000):
        unique=int(frame.row.iloc[:steps].nunique())
        usage.append(dict(discovery_pool=n,seed=r,updates=steps,unique_examples=unique,
                          pool_coverage_pct=100*unique/n,registered_endpoint=not(r>44 and steps==1000)))
pd.DataFrame(usage).to_csv(OUT/'actual_training_example_usage.csv',index=False)
visited=pd.DataFrame(usage).query('discovery_pool==10000 and updates==5000').unique_examples

enrichment=pd.read_csv(OUT/'enrichment_vs_topology.csv')
enrichment=enrichment[(enrichment.cohort=='primary')&(enrichment.dataset_size==10000)&(enrichment.target==116)&((enrichment.step==5000)|(enrichment.method=='eap_ig'))]
head_summary=enrichment.groupby('method')[['known_heads_observed','known_heads_control_mean','incident_known_fraction_observed','incident_known_fraction_control_mean','incident_known_fraction_enrichment']].mean()
head_summary.to_csv(OUT/'known_head_summary.csv')

cost=pd.read_csv(OUT/'compute_summary.csv')
stage_cost=cost.groupby(cost.job.str.split('_').str[0]).wall_seconds.sum()
cost_total=float(cost.wall_seconds.sum())
analysis_seconds=json.loads((OUT/'analysis_metadata.json').read_text())['seconds']
historical_seconds=pd.read_csv(OUT/'reused_training_cost.csv').historical_training_seconds.sum()
matched=[]
for r in range(43,47):
    b=cost[cost.job==f'train_baseline_n10000_s{r}'].training_seconds.sum()
    h=cost[cost.job==f'train_h2_n10000_s{r}'].training_seconds.sum()
    matched.append(dict(seed=r,baseline_training_seconds=b,h2_training_seconds=h,relative_overhead_pct=100*(h/b-1)))
pd.DataFrame(matched).to_csv(OUT/'matched_training_overhead.csv',index=False)
overhead=100*(sum(x['h2_training_seconds'] for x in matched)/sum(x['baseline_training_seconds'] for x in matched)-1)
clean=np.load(OUT/'evaluations/primary/baseline_n10000_s42_step5000/clean_reference.npz')
clean_margin=float(clean['margin'].mean())
base=comparison.loc['baseline'];hier=comparison.loc['h2'];eap=comparison.loc['eap_ig']

def pct(x):return f'{100*x:.2f}%'
def pp(x):return f'{100*x:+.2f}'

report=f'''# Gate 5F: data scaling and training duration — interpreted results

## Main finding

Unregularized receiver-component hierarchy retains a meaningful compact-circuit advantage after scaling the discovery pool and training duration. The frozen classification is **{primary['classification']}**. This is a success for positive-margin IOI accuracy under the registered capped-size rule. **It did not improve official CPR, and EAP-IG achieved better accuracy at the primary compact budget.**

At N=10,000 and5,000 updates, five-seed mean accuracy at the116-edge cap was **{pct(hier.accuracy)} for H2 versus {pct(base.accuracy)} for ordinary MAttr**. The paired gain was **{100*estimate['mean']:.2f} percentage points**, with a **95% paired prompt-bootstrap interval of {100*estimate['lower']:.2f}–{100*estimate['upper']:.2f} points**. All frozen correctness and larger-budget safeguards passed.

Three of five seeds improved; two regressed. The seed-effect SD was **{100*primary['checks']['seed_std']:.2f} points**, median **{100*primary['checks']['seed_median']:.2f} points**, and range **{100*primary['checks']['seed_min']:.2f} to {100*primary['checks']['seed_max']:.2f} points**. Thus the registered word “robust” does not mean every seed improved or that training-seed population significance was established. Confidence intervals resample prompts conditional on the five fixed trained models and exact template mix.

## Primary per-seed results

Accuracy columns below are percentages; differences are percentage points. All circuits contain at most116 actual edges.

{table(seed[['seed','baseline_accuracy_pct','h2_accuracy_pct','difference_pp','baseline_actual_edges','h2_actual_edges','corrected','broken']])}

Only seed45 is exactly size-matched at116, where H2 gains31.64 points. The other pairs differ by one to four actual edges. This supports **capped-size superiority**, rather than a five-seed exact-size claim. Seeds44 and46 already have high baseline accuracy and slightly worsen under H2; seed42 remains a weak H2 outcome at50.98% despite its large relative improvement. Across the five paired models there are533 corrected and30 broken prompt-model decisions; those are not2,560 independent evaluation prompts.

## Does ordinary MAttr catch up with longer training?

It catches up partially. Using only the registered factorial seeds42–44:

{table(factor)}

At N128, longer training reduces the hierarchy advantage from34.90 to11.39 points. At N10,000 it reduces the advantage from65.23 to23.11 points. Both methods improve in absolute accuracy, but ordinary MAttr improves more: **+45.12/+48.05 points** for N128/N10,000, compared with **+21.61/+5.92 points** for H2. The full-data65.23→23.11 comparison uses the same three seeds; the five-seed deciding gain of19.65 points is a separate estimand.

The registered **“Baseline catches up” classification does not pass**, because the full-data5,000-step advantage remains positive and its interval is well outside the practical-equivalence band. Narrowing the gap is different from eliminating it.

## Does more discovery data remove the advantage?

No, under this sampling design. Moving from N128 to N10,000 **increases** the relative H2 advantage by **{100*factorial.loc['data_effect1000','mean']:.2f} points at1,000 updates** and **{100*factorial.loc['data_effect5000','mean']:.2f} points at5,000 updates**. The data-by-duration interaction is **{100*factorial.loc['interaction','mean']:.2f} points** (95% prompt interval {100*factorial.loc['interaction','lower']:.2f} to {100*factorial.loc['interaction','upper']:.2f}). This reflects greater erosion of H2's advantage with extra optimization in the large-pool condition, not disappearance of its final advantage. The frozen data-limited pattern does not apply.

These are observed pool effects, not a clean causal estimate of dataset size: N128 seeds vary both subset and optimization draws, while N10,000 uses one fixed pool. “Full data” denotes the eligible pool, not a complete training pass. Five thousand stochastic draws visited **{int(visited.min())}–{int(visited.max())} distinct training rows** per full-data seed. EAP-IG visited all10,000 rows. Exact usage counts are in actual_training_example_usage.csv.

## Accuracy, CPR, CMD and EAP-IG

All comparisons below use the same fresh512-prompt primary cohort. MAttr/H2 are five-seed averages at N10,000/5,000 updates; EAP-IG is one full-data ranking.

| Method | Accuracy at116-edge cap | Vocabulary top-1 at cap | Official CPR | Official CMD (lower is better) | Accuracy log-AUC |
| --- | --- | --- | --- | --- | --- |
| Ordinary MAttr | {pct(base.accuracy)} | {pct(base.top1)} | {base.CPR:.6f} | {base.CMD:.6f} | {base.accuracy_log_AUC:.6f} |
| H2, gamma0 | {pct(hier.accuracy)} | {pct(hier.top1)} | {hier.CPR:.6f} | {hier.CMD:.6f} | {hier.accuracy_log_AUC:.6f} |
| Official EAP-IG | {pct(eap.accuracy)} | {pct(eap.top1)} | {eap.CPR:.6f} | {eap.CMD:.6f} | {eap.accuracy_log_AUC:.6f} |

H2 lowers mean CPR by **{base.CPR-hier.CPR:.6f}** (about{100*(base.CPR-hier.CPR)/base.CPR:.2f}%) and does not improve CPR in any of the five primary seeds. Its lower CMD and higher accuracy log-AUC are different benefits. EAP-IG is strongest at the116-edge cap, although H2 has higher accuracy log-AUC over the full official nominal grid. That grid weights extremely small nominal circuits too, so the two comparisons need not order methods identically. EAP-IG is not a five-seed significance comparison and no leaderboard/private-test evaluation was performed.

The large MAttr CPRs include pronounced margin amplification. The unmodified clean model's mean IOI margin is **{clean_margin:.3f}**; at the640-edge cap, ordinary MAttr and H2 have mean margins of **{budget_means.loc[('baseline',640),'mean_margin']:.3f}** and **{budget_means.loc[('h2',640),'mean_margin']:.3f}**, compared with **{budget_means.loc[('eap_ig',640),'mean_margin']:.3f}** for EAP-IG. Their corresponding normalized faithfulness values are about10.28,10.06 and2.21. Higher CPR here is not equivalent to closer reproduction of clean-model behavior. All values come from the official normalization/evaluator; no replacement CPR/CMD was invented.

## Larger budgets and exact subsets

The five-seed H2-minus-baseline mean gains at296,400 and640-edge caps are **{pp(budget_means.loc[('h2',296),'accuracy']-budget_means.loc[('baseline',296),'accuracy'])}**, **{pp(budget_means.loc[('h2',400),'accuracy']-budget_means.loc[('baseline',400),'accuracy'])}**, and **{pp(budget_means.loc[('h2',640),'accuracy']-budget_means.loc[('baseline',640),'accuracy'])} percentage points**. No mean regression exceeds the frozen2-point safeguard. The accuracy-size curves show the main benefit in small circuits, followed by saturation. At240 edges all five pairs are exactly size-matched and the gain is1.37 points; at400 it is0.08 points and at640 zero. Exact subsets are descriptive and sometimes contain only one seed.

## Known IOI mechanisms and topology controls

At the primary cap, ordinary MAttr represents an average of{head_summary.loc['baseline','known_heads_observed']:.1f} known heads versus{head_summary.loc['h2','known_heads_observed']:.1f} for H2, compared with topology-control expectations of{head_summary.loc['baseline','known_heads_control_mean']:.2f} and{head_summary.loc['h2','known_heads_control_mean']:.2f}. EAP-IG represents{head_summary.loc['eap_ig','known_heads_observed']:.0f} known heads and has the strongest compact accuracy despite not maximizing head count. All methods show incident-edge enrichment relative to the registered relabeling control; the full summaries retain each class separately.

The64 controls per observed mask exactly preserve actual size and connectivity by relabeling attention heads within layers, with Q/K/V coupled. They condition on layer structure and do not test whether the circuits favor the right layers. Head coverage/enrichment is a descriptive structural observation, not proof of causal mechanism fidelity or an explanation of H2's accuracy gains.

## Compute and integrity

- Recorded new process/stage wall time sums to **{cost_total/3600:.2f} hours**: training processes{stage_cost['train']/3600:.2f}h, evaluation processes{stage_cost['evaluate']/3600:.2f}h, attribution processes{stage_cost['eap']/3600:.2f}h, plus bounded preflight and CPU preparation. This is accumulated monotonic job wall time, not CUDA utilization time or elapsed calendar time including user pauses.
- CPU aggregation/reporting took{analysis_seconds:.1f}s, including54.3s of topology-control analysis; do not add the latter again. Reused historical training timers total{historical_seconds/3600:.2f}h and are separate from new work.
- The new registered training executed57,000 optimizer updates across12 trajectories. The aggregate counter is{int(cost.optimizer_updates.sum()):,}, including180 bounded smoke updates. Recorded root forwards:{int(cost.forward_batches.sum()):,}; example-forwards:{int(cost.example_forwards.sum()):,}; backwards:{int(cost.backward_calls.sum()):,}.
- Peak CUDA allocation:{cost.peak_allocated_mib.max()/1024:.2f}GiB; peak reservation:{cost.peak_reserved_mib.max()/1024:.2f}GiB; sampled CPU RSS:{cost.sampled_peak_cpu_rss_mib.max()/1024:.2f}GiB. These are allocator/RSS measurements, not total physical VRAM occupancy.
- Across matched new full-data seed43–46 training stages, H2's time overhead was approximately{overhead:.2f}%. This is a measured training-stage comparison, not an end-to-end efficiency or leaderboard improvement claim.
- The final audit passed: all64 evaluation jobs complete, paired streams verified, cohorts disjoint, model weights unchanged, counters reconciled and frozen classification reproduced. This interpretation independently recomputed the five deciding means from per-prompt outputs. The protocol and all previous gate decisions remain unchanged.

## Research decision

The narrow hypothesis survives: receiver-component parameterization improves compact circuits relative to ordinary MAttr after more data and training. Extra optimization erodes, but does not remove, the mean advantage. The result does **not** establish a CPR improvement, superiority to EAP-IG at the primary cap, uniform improvement across seeds, or exact-size superiority across all five pairs.

The next useful question is why seed failures persist and why EAP-IG achieves better compact correctness with far lower margin distortion. Retain ordinary MAttr as the CPR reference and EAP-IG as a required compact-accuracy reference. Prioritize independent validation and ranking-stability diagnosis before proposing another optimization change. No additional training, algorithm, research gate or submission is launched by this report.

## Detailed artifacts and figures

The complete automatically generated tables and original audit remain in [the registered analysis](GATE5F_DATA_SCALING_REPORT.generated.md). Supporting interpretation tables include full_data_primary_summary.csv, primary_seed_interpretation.csv, factorial_interpretation.csv, actual_training_example_usage.csv, known_head_summary.csv and matched_training_overhead.csv. The secondary Gate5E development cohort gives a20.27-point five-seed gain; it is descriptive and does not determine the classification. The primary cohort is new relative to smaller campaigns, not historically untouched across Gate1's full-validation evaluation.

![Accuracy versus actual edge count](figures/accuracy_size.png)

![Five-seed primary effects](figures/seed_effects.png)

![Full-data comparison: five-seed MAttr/H2 and one EAP-IG ranking](figures/full_data_accuracy_comparison.png)
'''

# Improve prose spacing without changing paths, identifiers, or measured values.
for word in ('and','at','from','to','seed','seeds','are','not','all','only','about','takes','took','new','the','fresh','including','with','cost','processes','versus'):
    report=re.sub(r'\b'+word+r'(?=\d)',word+' ',report)
report=re.sub(r'\bN(?=\d)','N=',report)
report=report.replace('Gate5','Gate 5').replace('updates,','updates,')
report=report.replace('296,400 and 640-edge caps','296, 400, and 640-edge caps')
report=report.replace('32491','32,491').replace('not 2,560 independent evaluation prompts','not 2,560 independent evaluation prompts')
report=re.sub(r'(?<=\d)(?=(?:corrected|broken|independent|actual|known|controls|s|h|GiB)\b)',' ',report)
report=report.replace('## Compute and integrity\n','## Compute and integrity\n\nFrozen protocol SHA-256: `'+(OUT/'gate5f_protocol.sha256').read_text().strip()+'`.\n')

generated=OUT/'GATE5F_DATA_SCALING_REPORT.generated.md'
if not generated.exists():
    original=(OUT/'GATE5F_DATA_SCALING_REPORT.md').read_text(encoding='utf-8')
    generated.write_text(original,encoding='utf-8')
(OUT/'INTERPRETATION.md').write_text(report,encoding='utf-8')
(OUT/'GATE5F_DATA_SCALING_REPORT.md').write_text(report,encoding='utf-8')
root_report=report.replace('(figures/',f'(results/mattr_data_scaling/{OUT.name}/figures/')
root_report=root_report.replace('(GATE5F_DATA_SCALING_REPORT.generated.md)',f'(results/mattr_data_scaling/{OUT.name}/GATE5F_DATA_SCALING_REPORT.generated.md)')
(ROOT/'GATE5F_DATA_SCALING_REPORT.md').write_text(root_report,encoding='utf-8')
metadata=dict(scope='Post-execution interpretation only; criteria, model outputs and decisions unchanged',
    protocol_sha256=hashlib.sha256((OUT/'gate5f_protocol.json').read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    inputs={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['gate5f_decision.json','final_audit.json','accuracy_size_curves.csv','official_metrics.csv','compute_summary.csv']},
    additional_model_forwards=0,additional_training_updates=0,primary_means_independently_verified=True)
(OUT/'interpretation_provenance.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
print('Interpreted report generated; zero additional model forwards/training updates.')
