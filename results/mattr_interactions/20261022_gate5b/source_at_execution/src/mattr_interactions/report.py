"""Actual-result diagnostic report, without new interventions or tuning."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.mattr_optimization.common import ROOT,sha,write_json
from .statistics import bootstrap_means

def table(frame):
    from src.mattr_optimization.gate2_final_analysis import table as existing_table
    return existing_table(frame)

def generate(out):
    screen=pd.read_csv(out/'screen_interactions.csv');confirmation=pd.read_csv(out/'confirmation_interactions.csv')
    examples=pd.read_csv(out/'confirmation_per_example.csv');null=pd.read_csv(out/'stability_null_comparison.csv')
    proc=json.loads((out/'process_summary.json').read_text());cost=pd.read_csv(out/'compute_accounting.csv')
    merged=confirmation.merge(screen[['pair_id','interaction','joint']],on='pair_id',suffixes=('_validation','_train'))
    merged['sign_consistent']=np.sign(merged.interaction_validation)==np.sign(merged.interaction_train)
    merged['ci_excludes_zero']=(merged.interaction_lo>0)|(merged.interaction_hi<0)
    merged['familywise_ci_excludes_zero']=(merged.interaction_familywise_lo>0)|(merged.interaction_familywise_hi<0)
    merged['reproduced_substantial']=merged.sign_consistent&merged.ci_excludes_zero&merged.substantial
    merged['useful_addition']=(merged.direction=='addition')&merged.reproduced_substantial&(merged.joint_validation>=.1)&merged.joint_exceeds_both_singles&(merged.accuracy_joint>0)&(merged.accuracy_joint_lo>=0)
    merged.to_csv(out/'heldout_consistency.csv',index=False)
    primary=merged[merged.selection_role=='primary']
    reproducibility=float(primary.reproduced_substantial.mean())
    option_a=bool(primary.reproduced_substantial.sum()>=4 and reproducibility>=.5 and primary.useful_addition.sum()>=2)
    b_groups=null[(null.component_gain>=.1)&null.above_null_95]
    # Matched comparisons aggregate correlated pairs within prompt before resampling prompts.
    controls=pd.read_csv(out/'confirmation_control_matches.csv');diffs=[];ratios=[]
    for match in controls.itertuples():
        r=merged[merged.pair_id==match.primary_pair].iloc[0]
        if r.category not in ('same_receiver','same_component'):continue
        a=examples[examples.pair_id==match.primary_pair].sort_values('example').interaction.to_numpy()
        b=examples[examples.pair_id==match.control_pair].sort_values('example').interaction.to_numpy()
        # |mean interaction| uncertainty is calculated from shared prompt resamples below.
        diffs.append((a,b));ratios.append(abs(a.mean())/max(.01,abs(b.mean())))
    concentration={'comparisons':len(diffs),'mean_ratio':None,'difference':None,'lo':None,'hi':None,'passed':False}
    if diffs:
        a=np.array([x[0] for x in diffs]);b=np.array([x[1] for x in diffs]);rng=np.random.default_rng(5242)
        draws=rng.integers(0,a.shape[1],size=(2000,a.shape[1]))
        aa=np.abs(a[:,draws].mean(2)).mean(0);bb=np.abs(b[:,draws].mean(2)).mean(0)
        difference=np.abs(a.mean(1)).mean()-np.abs(b.mean(1)).mean()
        ratio=np.abs(a.mean(1)).mean()/max(.01,np.abs(b.mean(1)).mean());lo,hi=np.quantile(aa-bb,[.025,.975])
        concentration={'comparisons':len(diffs),'mean_ratio':float(ratio),'difference':float(difference),'lo':float(lo),'hi':float(hi),'passed':bool(ratio>=1.5 and lo>0)}
    option_b=bool(len(b_groups)>=2 and concentration['passed'])
    recommendation='both as separate hypotheses' if option_a and option_b else 'Option A' if option_a else 'Option B' if option_b else 'neither under the frozen criteria'
    decision={'recommendation':recommendation,'option_A':option_a,'option_B':option_b,'primary_confirmation_pairs':len(primary),
       'reproduced_substantial_pairs':int(primary.reproduced_substantial.sum()),'reproduced_fraction':reproducibility,
       'useful_excluded_additions':int(primary.useful_addition.sum()),'component_groupings_passing':b_groups.grouping.tolist(),
       'within_component_vs_matched_control':concentration,'model_training_updates':0,'protocol_sha256':sha(out/'gate5b_protocol.json')}
    write_json(out/'research_decision.json',decision)
    groups=screen.groupby(['fraction','direction']).agg(pairs=('pair_id','size'),mean_abs_interaction=('abs_interaction','mean'),median_abs_interaction=('abs_interaction','median'),substantial_fraction=('substantial','mean'),additive_sign_disagreements=('sign_flip_vs_additive','sum')).reset_index()
    held=primary.groupby(['fraction','direction']).agg(pairs=('pair_id','size'),sign_consistency=('sign_consistent','mean'),reproduced_substantial=('reproduced_substantial','sum'),familywise_CI_nonzero=('familywise_ci_excludes_zero','sum'),useful_additions=('useful_addition','sum'),joint_accuracy_gain=('accuracy_joint','mean')).reset_index()
    groups.to_csv(out/'screen_summary.csv',index=False);held.to_csv(out/'confirmation_summary.csv',index=False)
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout();plt.savefig(figdir/(name+'.png'),dpi=150);plt.close()
    plt.figure(figsize=(7,5))
    for direction,color in [('deletion','tab:blue'),('addition','tab:orange')]:
        d=screen[screen.direction==direction];plt.scatter(d.additive,d.joint,label=direction,alpha=.6,s=20,color=color)
    ends=[min(screen.additive.min(),screen.joint.min()),max(screen.additive.max(),screen.joint.max())]
    plt.plot(ends,ends,'k--');plt.xlabel('Measured singleton sum (margin change)');plt.ylabel('Exact joint margin change');plt.legend();save('additive_vs_joint')
    plt.figure(figsize=(7,4))
    for direction in ('deletion','addition'):
        plt.hist(screen[screen.direction==direction].abs_interaction,bins=25,alpha=.55,label=direction)
    plt.xlabel('Absolute mean interaction (IOI logit margin)');plt.ylabel('Screened pairs');plt.legend();save('interaction_distribution')
    plt.figure(figsize=(9,4));cats=[];values=[]
    for (direction,cat),g in screen.groupby(['direction','category']):cats.append(direction+'\n'+cat);values.append(g.abs_interaction.values)
    plt.boxplot(values,tick_labels=cats,showfliers=False);plt.ylabel('Absolute mean interaction');plt.xticks(rotation=20,ha='right');save('interaction_by_category')
    plt.figure(figsize=(6,5))
    for direction in ('deletion','addition'):
        d=merged[(merged.direction==direction)&(merged.selection_role=='primary')]
        plt.errorbar(d.interaction_train,d.interaction_validation,yerr=[d.interaction_validation-d.interaction_lo,d.interaction_hi-d.interaction_validation],fmt='o',label=direction,alpha=.7)
    limits=[min(merged.interaction_train.min(),merged.interaction_validation.min()),max(merged.interaction_train.max(),merged.interaction_validation.max())]
    plt.plot(limits,limits,'k--');plt.xlabel('Screen interaction: 32 train prompts');plt.ylabel('Confirmation interaction: 64 new prompts');plt.legend();save('train_vs_heldout')
    plt.figure(figsize=(9,4));x=np.arange(len(null));plt.bar(x,null.component_mean_spearman,label='Component mean rank');plt.scatter(x,null.common_permutation_null_95,marker='_',s=180,color='black',label='95th common-permutation null')
    plt.axhline(null.edge_mean_spearman.iloc[0],color='tab:red',linestyle='--',label='Individual edge ranks');plt.xticks(x,null.grouping,rotation=25,ha='right');plt.ylabel('Mean paired Spearman across 3 seeds');plt.legend(fontsize=8);save('edge_vs_component_stability')
    body=f'''# Gate 5B: Edge interactions and component structure

Frozen diagnostic decision: **{recommendation}**. No MAttr training, new mask algorithm, ranking refinement, or leaderboard submission occurred.

## Design and intervention semantics

GPT-2 Small, official MIB IOI, all32,491 edges. Saved Gate2 N128 rankings at5000 updates; seed42 interventions, seeds42/43/44 offline stability. Screen uses the first32 exact discovery rows; confirmation uses64 newly selected official validation prompts disjoint by row and clean prompt from prior small Gate2–5A development/confirmation manifests and from discovery prompts. Historical Gate1 used full validation, so this is fresh relative to development cohorts, not untouched across all research. Protocol SHA-256 `{sha(out/'gate5b_protocol.json')}`.

Each base circuit is selected by official signed apply_topn and connectivity pruning. Subsequent individual/joint toggles are accepted only if official prune leaves exactly that mask. This fixes circuit context and prevents collateral cleanup from masquerading as interactions. Unsafe paths are recorded in topology_selection_audit.csv. Pruning stays enabled in the evaluator; disconnected additions are excluded. Consequently these diagnostics do not exhaust interactions requiring several new connecting edges.

The official evaluate_graph patcher is used verbatim. Safe caching stores immutable donor-only source matrices; every clean/patched source is recomputed live. Source/model/token/mask/cohort identities guard caches. Correctness compares dependent multi-edge paths against the pinned corrected MAttr live closure without running its trainer, and checks cache equivalence, repeatability and singleton batching. Per-example output is the official correct-minus-incorrect logit difference and official vocabulary top1 accuracy, not a homemade CPR/CMD. CPR/CMD are not computed for this bounded pair experiment.

## Pair selection and screening

48 candidates at each1%/2% nominal cutoff:24 retained and24 excluded, combining high/boundary/disagreement strata. Parent/head diversity and MLP connections are recorded. Up to120 unique i<j pairs per cutoff, stratified by receiver, shared component, cross-component and matched random controls. Shortfalls are reported rather than filling an unregistered sweep. Selection uses only learned train rankings/topology, including saved signed EAP-IG scores. Full all-pairs matrices are never constructed.

{table(pd.read_csv(out/'topology_selection_audit.csv'))}

{table(groups)}

Interactions are joint-minus-singleton-sum. A substantial interaction requires absolute mean>=.1 logit margin and relative magnitude>=.2, denominator max(.1,|single_i|+|single_j|). These are exploratory decision thresholds frozen before effects. A large interaction is not necessarily beneficial, and a deletion interaction does not demonstrate a missed addition.

## Held-out confirmation

Top3 absolute interactions plus1 joint-gain extremum per cutoff/direction were selected on train only; matched previously screened random controls were then selected by type/layer signature. Selection IDs and hashes were frozen before loading confirmation prompts. All uncertainty resamples prompts with shared draws across pairs. Tables provide95% and familywise Bonferroni intervals; pair rows are correlated and never treated as independent bootstrap units.

{table(held)}

Primary pairs reproduced with substantial magnitude, consistent sign and nonzero95% CI: **{int(primary.reproduced_substantial.sum())}/{len(primary)}**. Useful held-out additions meeting the frozen margin, singleton and accuracy safeguards: **{int(primary.useful_addition.sum())}**. Full values, sign consistency and familywise intervals are in heldout_consistency.csv; confirmation contains both primary and matched-control rows.

{table(primary[['pair_id','direction','interaction_train','interaction_validation','interaction_lo','interaction_hi','joint_validation','accuracy_joint','sign_consistent','useful_addition']])}

## Component structure and stability

Edges are grouped separately by source head, destination head, Q/K/V receiver, source/destination MLP and source/destination layer. Counts, retained fractions, eligibility-normalized enrichment, score distributions and rank distributions are saved for every group/seed/cutoff. The same edge can participate in different diagnostic groupings; grouping is not a partition across all tables. Edge sets use official connectivity-pruned membership; component ranks use mean percentile importance so size alone cannot dominate.

{table(null)}

The200 common edge-label permutations preserve cross-seed edge covariance, component sizes and overlapping group assignments. They test whether averaging correlated edge scores alone explains apparent stability. Top10% component/edge overlaps use comparable selection fractions but distinct units; official edge overlap at.5/1/2/5% is also saved. Three seeds give descriptive variation, not a training-randomness significance claim. Existing EAP-IG correlations are observational references, not new baseline runs.

Within-component primary versus type/layer-matched control comparison: `{json.dumps(concentration)}`. Prompt-resampled confidence bounds apply to the mean difference of absolute mean interactions, with shared draws. Unmatched structural/layer differences and train-selected extremal pairs limit causal interpretations. Control matching quality is explicitly recorded; sparse candidate strata may not permit exact matching.

## Decision and limitations

Option A passes only with at least4 substantial reproducible primary interactions, at least50% primary reproducibility and2 jointly useful excluded additions with positive held-out accuracy gain. Option B passes only if at least2 groupings exceed individual-edge Spearman by.1 and the95th size/covariance-preserving null, plus >=1.5 within-component interaction concentration with a positive bootstrap difference bound. Actual frozen decision: `{json.dumps(decision)}`.

These measurements cannot establish that a new masking algorithm would improve MAttr: its existing joint soft-mask objective already encounters interactions. Candidate selection is deliberately local around two small circuits and excludes cleanup-confounded paths. IOI prompts share templates; prompt-bootstrap intervals quantify sampling of this cohort and do not establish template-family generalization. Large margins need not improve behavioral fidelity. Fresh confirmation is small and historical full-validation exposure remains. Neither proposed algorithm is implemented or authorized automatically; the original Gate5 ranking refinement remains deferred.

## Compute and preservation

{table(cost)}

Total process wall time: **{proc['seconds']/60:.2f}min**; actual root forward batches **{proc['counts']['forward_batches']:,}**, example-forwards **{proc['counts']['example_forwards']:,}**, model backwards/optimizer updates **0/0**. Exact evaluation cache hits **{proc['exact_cache_hits']}**, donor cache hits **{proc['donor_cache_hits']}**, unique exact evaluations **{proc['unique_evaluations_this_run']}**. Peak CUDA allocation **{cost.peak_allocated_mib.max():.1f}MiB**. Stage timers synchronize CUDA and use monotonic perf_counter; CPU offline work is separately labeled. CPU donor-cache bytes are recorded, not counted as VRAM. Raw masks, cached outputs and prompt manifests stay local; small per-example effects contain indices and numerical effects only. Completed artifact checks protect every pre-existing tracked file. No model weights, tokens or private data are published.
'''
    for name in ('additive_vs_joint','interaction_distribution','interaction_by_category','train_vs_heldout','edge_vs_component_stability'):
        body+=f'\n![{name}](figures/{name}.png)\n'
    (out/'GATE5B_INTERACTION_DIAGNOSTICS_REPORT.md').write_text(body)
    root_body=body.replace('(figures/',f'({out.relative_to(ROOT).as_posix()}/figures/')
    (ROOT/'GATE5B_INTERACTION_DIAGNOSTICS_REPORT.md').write_text(root_body)
