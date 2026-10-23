"""Actual results, single-primary inference, and explicitly exploratory rankings."""
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import torch
from src.mattr_optimization.common import ROOT, sha, write_json
from src.mattr_optimization.gate2_final_analysis import table
from .design import protocol
from .statistics import bootstrap_indices, estimates, primary_analysis


def interpolate_values(lower,upper,low_count,high_count,target):
    assert low_count <= target <= high_count
    if low_count == high_count:
        assert target == low_count
        return np.asarray(lower,dtype=float),0.
    weight = (target-low_count)/(high_count-low_count)
    return (1-weight)*np.asarray(lower,dtype=float)+weight*np.asarray(upper,dtype=float),weight


def generate(out):
    start=time.perf_counter()
    p=protocol(out)
    process=json.loads((out/'inference_process_summary.json').read_text())
    if process['projection_stop']:
        body='# Gate 5E: Matched-size circuit evaluation\n\nStopped by the prespecified two-hour projected-workload gate after the first complete ranking. No remaining evaluation was launched.\n\n'+json.dumps(json.loads((out/'first_ranking_projection.json').read_text()),indent=2)+'\n'
        for path in (out/'GATE5E_MATCHED_SIZE_REPORT.md',ROOT/'GATE5E_MATCHED_SIZE_REPORT.md'):
            path.write_text(body)
        return
    assert json.loads((out/'inference_complete.json').read_text())['passed']
    cohort=pd.read_csv(out/'cohort_ids_and_hashes.csv')
    families=cohort.template_id.to_numpy()
    draws=bootstrap_indices(families,5000,5643)
    nominal_arrays={};cap_arrays={};upper_arrays={}
    nominal_rows=[];metrics=[];matched_rows=[];interpolation=[];interpolation_arrays={}
    catalog={r['key']:r for r in p['catalog']}
    for key,rec in catalog.items():
        dest=out/'nominal'/key
        saved=json.loads((dest/'official_result.json').read_text());result=saved['result']
        assert saved['per_prompt_sha256']==sha(dest/'per_prompt.csv')
        nominal=pd.read_csv(dest/'curve.csv').assign(ranking=key,kind=rec['kind'],method=rec['method'],seed=str(rec['seed']))
        nominal_rows.append(nominal)
        metrics.append({'ranking':key,'kind':rec['kind'],'method':rec['method'],'seed':str(rec['seed']),
                        'CPR':result[1],'CMD':result[2],'accuracy_log_AUC':result[6]})
        data=pd.read_csv(dest/'per_prompt.csv')
        nominal_arrays[key]=data.pivot(index='fraction',columns='example',values='correct').loc[p['grid']].to_numpy(dtype=float)
        if rec['kind']=='closure':
            continue
        dest=out/'matched'/key
        measured=pd.read_csv(dest/'measurements.csv')
        measured['seed']=measured.seed.astype(str)
        matched_rows.append(measured)
        data=pd.read_csv(dest/'per_prompt.csv')
        cap=data[data.selection=='cap'].pivot(index='target',columns='example',values='correct')
        upper=data[data.selection=='upper'].pivot(index='target',columns='example',values='correct')
        for target in p['targets']:
            cap_arrays[key,target]=cap.loc[target].to_numpy(dtype=float)
        for target in (p['T1'],p['T2']):
            upper_arrays[key,target]=upper.loc[target].to_numpy(dtype=float)
            low=measured[(measured.selection=='cap')&(measured.target==target)].iloc[0]
            high=measured[(measured.selection=='upper')&(measured.target==target)].iloc[0]
            values,weight=interpolate_values(cap_arrays[key,target],upper_arrays[key,target],int(low.actual_edges),int(high.actual_edges),target)
            interpolation_arrays[key,target]=values
            interpolation.append({'ranking':key,'method':rec['method'],'seed':str(rec['seed']),'target':target,
                                  'lower_actual':int(low.actual_edges),'upper_actual':int(high.actual_edges),
                                  'gap':int(high.actual_edges-low.actual_edges),'upper_weight':weight,
                                  'interpolated_accuracy':float(values.mean()),'directly_evaluated':low.actual_edges==high.actual_edges,
                                  'large_gap_flag':bool(high.actual_edges-low.actual_edges>max(10,.10*target))})
    nominal=pd.concat(nominal_rows,ignore_index=True);metrics=pd.DataFrame(metrics);matched=pd.concat(matched_rows,ignore_index=True)
    nominal.to_csv(out/'nominal_curves.csv',index=False);metrics.to_csv(out/'official_metrics.csv',index=False)
    matched.to_csv(out/'matched_results.csv',index=False);pd.DataFrame(interpolation).to_csv(out/'interpolated_estimates.csv',index=False)
    h=np.stack([cap_arrays[f'hierarchy_s{s}',p['T1']] for s in (42,43,44)])
    b=np.stack([cap_arrays[f'baseline_s{s}',p['T1']] for s in (42,43,44)])
    nom=np.stack([nominal_arrays[f'hierarchy_s{s}'][p['grid'].index(.005)]-nominal_arrays[f'baseline_s{s}'][p['grid'].index(.005)] for s in (42,43,44)])
    valid=all(json.loads((out/name).read_text())['passed'] for name in ('correctness_passed.json','exact_cache_correctness.json','phaseA_complete.json','model_weight_audit.json'))
    decision=primary_analysis(h-b,nom,draws,valid=valid)
    write_json(out/'gate5e_decision.json',decision)
    comparisons=[];exact_rows=[]
    for target in p['targets']:
        for comparator in ('baseline','gamma_zero','shuffled','edge_l2'):
            differences=np.stack([cap_arrays[f'hierarchy_s{s}',target]-cap_arrays[f'{comparator}_s{s}',target] for s in (42,43,44)])
            pooled,_=estimates(differences.mean(axis=0),draws)
            for i,seed in enumerate((42,43,44)):
                result,_=estimates(differences[i],draws)
                comparisons.append({'target':target,'comparator':comparator,'seed':str(seed),**result,
                                    'corrected_prompts':int(((cap_arrays[f'hierarchy_s{seed}',target]==1)&(cap_arrays[f'{comparator}_s{seed}',target]==0)).sum()),
                                    'broken_prompts':int(((cap_arrays[f'hierarchy_s{seed}',target]==0)&(cap_arrays[f'{comparator}_s{seed}',target]==1)).sum())})
            comparisons.append({'target':target,'comparator':comparator,'seed':'equal_seed_mean',**pooled,
                                'positive_seeds':int((differences.mean(axis=1)>0).sum())})
        valid_seeds=[];diff=[]
        for seed in (42,43,44):
            a=matched[(matched.ranking==f'hierarchy_s{seed}')&(matched.target==target)&(matched.selection=='cap')].iloc[0]
            z=matched[(matched.ranking==f'baseline_s{seed}')&(matched.target==target)&(matched.selection=='cap')].iloc[0]
            if a.actual_edges==z.actual_edges==target:
                valid_seeds.append(seed);diff.append(cap_arrays[f'hierarchy_s{seed}',target]-cap_arrays[f'baseline_s{seed}',target])
        if diff:
            exact,_=estimates(np.mean(diff,axis=0),draws)
            exact_rows.append({'target':target,'exact_seeds':json.dumps(valid_seeds),'n_fixed_seeds':len(valid_seeds),**exact,
                               'scope':'descriptive exact subset of already evaluated capped circuits; no separate exact forwards'})
        else:
            exact_rows.append({'target':target,'exact_seeds':'[]','n_fixed_seeds':0,'mean':np.nan,'lower':np.nan,'upper':np.nan,
                               'scope':'no exact paired comparison attained by both original rankings'})
    comparisons=pd.DataFrame(comparisons);comparisons.to_csv(out/'paired_cap_comparisons.csv',index=False)
    exact=pd.DataFrame(exact_rows);exact.to_csv(out/'exact_subset_primary_comparisons.csv',index=False)
    interp_compare=[]
    for target in (p['T1'],p['T2']):
        diff=np.stack([interpolation_arrays[f'hierarchy_s{s}',target]-interpolation_arrays[f'baseline_s{s}',target] for s in (42,43,44)])
        for i,seed in enumerate((42,43,44)):
            estimate,_=estimates(diff[i],draws);interp_compare.append({'target':target,'seed':str(seed),**estimate,'interpolated':True})
        estimate,_=estimates(diff.mean(axis=0),draws);interp_compare.append({'target':target,'seed':'equal_seed_mean',**estimate,'interpolated':True})
    pd.DataFrame(interp_compare).to_csv(out/'interpolated_paired_differences.csv',index=False)
    closure=[]
    stability=[]
    for rec in p['catalog']:
        if rec['kind']!='closure':continue
        parent=rec['parent'];key=rec['key']
        c=metrics[metrics.ranking==key].iloc[0];o=metrics[metrics.ranking==parent].iloc[0]
        for f in p['grid']:
            a=nominal[(nominal.ranking==parent)&np.isclose(nominal.fraction,f)].iloc[0]
            z=nominal[(nominal.ranking==key)&np.isclose(nominal.fraction,f)].iloc[0]
            closure.append({'original':parent,'closure':key,'fraction':f,'original_actual':int(a.actual_edges),
                            'closure_actual':int(z.actual_edges),'original_accuracy':a.accuracy,'closure_accuracy':z.accuracy,
                            'accuracy_delta':z.accuracy-a.accuracy,'CPR_delta':c.CPR-o.CPR,'CMD_delta':c.CMD-o.CMD,
                            'same_mask':a.mask_sha256==z.mask_sha256,'interpretation':'nominal-grid size/counting effect; not matched-size ranking quality'})
        from .topology import canonical_ranks
        x=torch.load(ROOT/catalog[parent]['scores'],map_location='cpu',weights_only=False).numpy()
        y=torch.load(ROOT/rec['scores'],map_location='cpu',weights_only=False).numpy()
        stability.append({'ranking':parent,'closure':key,'signed_rank_spearman':float(spearmanr(canonical_ranks(x),canonical_ranks(y)).statistic),
                          'ordering_scope':'first-appearance groups from original prefixes<=1624; unseen tail appended'})
    closure=pd.DataFrame(closure);closure.to_csv(out/'closure_nominal_effects.csv',index=False)
    pd.DataFrame(stability).to_csv(out/'closure_rank_stability.csv',index=False)
    # Before/after cross-seed stability is descriptive; it consumes no new prompts.
    from .topology import graph_new,canonical_ranks
    from src.mattr_compact_validation.inference import assign_scores
    graph=graph_new();seed_stability=[]
    for method in ('baseline','hierarchy','gamma_zero','shuffled','edge_l2'):
        for kind,suffix in [('original',''),('closure','_closure')]:
            vectors={s:torch.load(ROOT/catalog[f'{method}_s{s}{suffix}']['scores'],map_location='cpu',weights_only=False) for s in (42,43,44) if f'{method}_s{s}{suffix}' in catalog}
            masks={}
            for s,vec in vectors.items():
                assign_scores(graph,vec)
                for fraction in (.005,.01,.02,.05):
                    graph.apply_topn(int(fraction*32491),absolute=False,level='edge',prune=True)
                    masks[s,fraction]=graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()
            for a,bseed in itertools.combinations(vectors,2):
                rho=float(spearmanr(canonical_ranks(vectors[a].numpy()),canonical_ranks(vectors[bseed].numpy())).statistic)
                for fraction in (.005,.01,.02,.05):
                    x,y=masks[a,fraction],masks[bseed,fraction];union=int((x|y).sum())
                    seed_stability.append({'method':method,'kind':kind,'seed_a':a,'seed_b':bseed,'fraction':fraction,
                                           'rank_spearman':rho,'pruned_edge_jaccard':float((x&y).sum()/union) if union else np.nan})
    pd.DataFrame(seed_stability).to_csv(out/'closure_cross_seed_stability.csv',index=False)
    ensemble=[]
    for method in ('baseline','hierarchy','gamma_zero'):
        key=f'{method}_ensemble'
        for target in p['targets']:
            row=matched[(matched.ranking==key)&(matched.selection=='cap')&(matched.target==target)].iloc[0]
            seeds=[float(cap_arrays[f'{method}_s{s}',target].mean()) for s in (42,43,44)]
            ensemble.append({'method':method,'target':target,'actual_edges':int(row.actual_edges), 'ensemble_accuracy':row.accuracy,
                             'seed42_accuracy':seeds[0],'seed43_accuracy':seeds[1],'seed44_accuracy':seeds[2],
                             'component_min':min(seeds),'component_mean':float(np.mean(seeds)),
                             'ensemble_minus_component_min':float(row.accuracy-min(seeds)),
                             'baseline_T2_90pct_pass':bool(method=='baseline' and target==p['T2'] and row.accuracy>=.90)})
    ensemble=pd.DataFrame(ensemble);ensemble.to_csv(out/'ensemble_vs_components.csv',index=False)
    jumps=pd.concat([pd.read_csv(out/'topology'/f'{r["key"]}.csv').assign(ranking=r['key']) for r in p['catalog'] if r['kind']!='closure'],ignore_index=True)
    jumps.groupby(['ranking','jump']).size().rename('prefixes').reset_index().to_csv(out/'jump_distribution.csv',index=False)
    jump_summary=jumps.groupby('ranking').agg(max_jump=('jump','max'),mean_jump=('jump','mean'),
                       zero_jump_prefixes=('jump',lambda x:int((x==0).sum())),multi_edge_jump_prefixes=('jump',lambda x:int((x>1).sum()))).reset_index()
    jump_summary.to_csv(out/'jump_summary.csv',index=False)
    attain=pd.read_csv(out/'exact_attainability_1_640.csv')
    skipped=attain[~attain.exactly_attained].groupby('ranking').target.agg(list).to_dict()
    write_json(out/'skipped_exact_targets_1_640.json',skipped)
    cost=pd.read_csv(out/'compute_accounting.csv')
    nominal_cost=cost[cost.stage.str.startswith('nominal_')];matched_cost=cost[cost.stage.str.startswith('matched_')]
    eval_seconds=float(nominal_cost.seconds.sum()+matched_cost.seconds.sum())
    nominal_cost.to_csv(out/'per_ranking_nominal_compute.csv',index=False)
    matched_cost.to_csv(out/'per_ranking_matched_compute.csv',index=False)
    assert int(cost.forward_batches.sum())==process['counts']['forward_batches']
    assert int(cost.example_forwards.sum())==process['counts']['example_forwards']
    primary_pairs=pd.read_csv(out/'exact_primary_comparisons.csv')
    mean_original=metrics[metrics.kind=='original'].groupby('method').mean(numeric_only=True)
    cap_main=comparisons[(comparisons.comparator=='baseline')&comparisons.target.isin([p['T1'],p['T2']])]
    closure_summary=closure.drop_duplicates('original').groupby(closure.drop_duplicates('original').original.str.replace(r'_s\d+$','',regex=True))[['CPR_delta','CMD_delta']].mean()
    summary={'classification':decision['classification'],'primary_success':decision['primary_success'],
             'T1':p['T1'],'T2':p['T2'],'primary':decision,'nominal_original_means':mean_original.to_dict(orient='index'),
             'registered_primary_pairs':len(primary_pairs),'exact_primary_pairs':int(primary_pairs.both_exact.sum()),
             'capped_T2':cap_main[(cap_main.seed=='equal_seed_mean')&(cap_main.target==p['T2'])].to_dict(orient='records'),
             'baseline_ensemble_T2':ensemble[(ensemble.method=='baseline')&(ensemble.target==p['T2'])].to_dict(orient='records'),
             'inference_process_seconds':process['seconds'],'evaluation_seconds':eval_seconds,
             'graph_analysis_seconds':json.loads((out/'phaseA_complete.json').read_text())['graph_analysis_seconds'],
             'actual_forward_calls':process['counts']['forward_batches'],'example_forwards':process['counts']['example_forwards'],
             'output_cache_hits':process['output_cache_hits'],'unique_exact_masks':process['unique_exact_masks'],
             'peak_allocated_mib':float(cost.peak_allocated_mib.max()),'peak_reserved_mib':float(cost.peak_reserved_mib.max()),
             'sampled_peak_cpu_rss_mib':float(cost.sampled_peak_cpu_rss_mib.max()),'optimizer_updates':0}
    write_json(out/'research_summary.json',summary)
    figures(out,p,nominal,matched,comparisons,jumps,ensemble,metrics,closure)
    semantics=json.loads((out/'official_pruning_semantics.json').read_text())
    phase=json.loads((out/'phaseA_complete.json').read_text())
    reference_lines='\n'.join(f'- `{v["file"]}:{v["first_line"]}` through line{v["last_line"]} ({k}).' for k,v in semantics.items() if isinstance(v,dict))
    report=f'''# Gate 5E: Matched-size circuit evaluation and ranking robustness

## Actual frozen decision

**{decision['classification']}**. Primary capped-T1 success: **{decision['primary_success']}**. T1=**{p['T1']}**, T2=**{p['T2']}**, computed as medians over all15 saved Gate5D actual counts, not hardcoded estimates. Full decision and advantage-retention ratio:

```json
{json.dumps(decision,indent=2)}
```

Gate5D remains mixed/inconclusive; its protocol, results and failed seed-consistency criterion were not revised. This inference-only gate tests whether the nominal advantage remains when circuits are capped by actual edge count. It does not train/optimize MAttr, modify scores by gradient, fine-tune model weights or introduce differentiable masks. Closure and equal-rank ensembles are the only prespecified exploratory ranking constructions.

## Official pruning and size counting

{semantics['pruning']}

**MIB counts grid size nominally**, as `int(percentage * len(graph.edges))`, then runs official pruning. The evaluator returns actual weighted counts for reporting but integrates over the original percentage coordinates. CPR and CMD use the nominal linear grid; accuracy log-AUC uses nominal log-grid coordinates. About **{100*semantics['CPR_linear_area_weight_at_or_below1pct']:.3f}%** of the CPR quadrature weight lies at1% and below. Closure changes on that grid are legitimate training-free ranking/counting effects; they do not establish better equal-size circuits.

Pinned source references:

{reference_lines}

All15 original signed rankings and18 total original/ensemble mappings retain every canonical edge identity. Official `apply_topn(...,absolute=False,prune=True)` alone determines every circuit. Source/model/checkpoint/effective-score hashes match Gate5D. Original tie semantics follow the pinned flattened PyTorch sort; only the prespecified ensembles use canonical-index tie breaks on signed ordinal ranks.

## CPU topology and closure correctness

PhaseA verified A(n) nondecreasing and P(n) nested for every n1..1624 for15 original rankings and3 ensembles, reproducing all saved Gate5D counts AND masks through5% exactly. All real edges are audited against the canonical32491-edge graph. Complete A(n), mask hashes, packed P(n), first appearances and jump statistics were saved. Maximum jumps and flat intervals are in jump_summary.csv/jump_distribution.csv. CPU topology/construction time: **{phase['graph_analysis_seconds']:.3f}seconds**.

All **{phase['valid_closure_rankings']}** closure rankings passed their mandatory correctness tests. Every prefix through the last known first-appearance group was tested, not merely50samples: each group boundary returns exactly its originalP(n), with zero edges pruned; every nonboundary prefix obeys the lower/upper group sandwich, with consecutive nesting. Partial-group pruning is allowed. An additional50full-range samples check nesting only. Export order is deterministic and all32491 edges appear once. The initial group audit uses original n<=1624; **the appended tail is not claimed pruning-free**. Detailed scope and per-prefix checks are saved in closure_correctness.csv and topology/. No official pruning code was changed. Invalid closure rankings would have been skipped while retaining originals.

## Protocol and fresh cohort

Protocol SHA-256: `{sha(out/'gate5e_protocol.json')}`. All original/derived ranking hashes, source/dependency pins, selections, targets, interpolation rules, cohort/hash manifests, metrics, bootstrap, thresholds and stop conditions were frozen before any fresh-cohort forward pass.

Official MIB IOI validation cohort: **{len(cohort)}** prompts, selection seed5543, proportional stratification over nine exact official template categories. Exclusions cover all historical small development/confirmation cohorts, **all512 Gate5D prompts**, and discovery training prompts. Both clean and corrupted hashes and cross-role collisions are excluded/deduplicated, alongside official row IDs. Labels are verified single tokens. Raw prompts remain local; IDs/hashes and generic template labels are versioned. Gate1 historically evaluated the entire10000-example validation split: this cohort is fresh relative to smaller campaigns, **not untouched across all historical research or a private test**.

## Original nominal-size results

{table(metrics[metrics.kind=='original'])}

These use the complete official ten-point grid. CPR/CMD and accuracy log-AUC come from the unchanged official MIB evaluator. Positive-margin accuracy and separate full-vocabulary top1 are recorded from the same logits. New-cohort clean/empty references are shared safely, without mutable patched-activation caches.

## Capped comparisons and exact subset

For each target, the largest audited n with A(n)<=T is used. There is no separate exact-size evaluation. An exact comparison means **both** circuits attain T; otherwise it is a capped comparison and the remaining actual-count difference is retained. At any attained count, nestedness makes the circuit unique. There are **{phase['both_exact_pairs']}/{phase['registered_pair_comparisons']}** exact comparisons across all unordered pairs of18original/ensemble rankings and10targets (cross-seed pairs descriptive), including **{phase['primary_both_exact']}/{phase['primary_pairs']}** paired baseline/hierarchy target comparisons. The latter span3fixed seeds x10targets; they are not30independent trials.

{table(cap_main[['target','seed','mean','lower','upper']])}

All target curves/control differences are in paired_cap_comparisons.csv. Per-ranking caps, selected n, actual sizes, shortfalls and mask hashes are in matched_results.csv. Every count1..640 has an exact-attainability flag for each original and ensemble; skipped counts are explicitly listed in skipped_exact_targets_1_640.json. Pair-by-pair exact/capped labels and size differences are in exact_comparisons_all_pairs.csv; primary pairs in exact_primary_comparisons.csv.

Exact-primary subsets of the same measured cap queries:

{table(exact)}

Intervals above apply only to the indicated exact fixed-seed subset, which can contain fewer than three seeds. No broader exact-size superiority is inferred from a capped result.

## Interpolation at T1 and T2

For each ranking, lower/upper attainable circuits bracket the target. Linear interpolation of measured accuracy uses weight (T-lower)/(upper-lower); equal counts use the exact circuit. These are **interpolated estimates, not directly evaluated exact circuits**. Gaps exceeding max(10edges,10%ofT) are flagged without hiding them. Estimates, brackets/gaps and flags are in interpolated_estimates.csv.

{table(pd.DataFrame(interp_compare))}

## Paired prompt uncertainty and advantage retention

The sole primary hypothesis is the capped T1 comparison. It requires positive equally seed-weighted accuracy gain, positive changes in>=2/3 seeds, paired template-stratified95% bootstrap lower>0, and valid masks/evaluator. **5000** shared prompt resamples, seed**5643**, are used across methods, seeds, targets and the nominal/cap contrast. Training seeds are fixed, never treated as independently sampled prompt rows or resampled. T2, control and interpolation intervals are descriptive secondary/unadjusted. Prompt uncertainty conditions on the fixed official-template mix, not unseen-template generalization or training-seed significance.

R is the cappedT1 accuracy gain divided by the fresh nominal0.5% gain. Shared resamples preserve numerator/denominator covariance. A denominator with95%CI containing0 or absolute point<1/N is marked unstable/undefined; invalid zero-denominator draws are counted. R is an **advantage-retention ratio, not a causal fraction explained by pruning**. Classification follows the frozen order: non-replication if nominal lower<=0; ranking-quality effect if primary passes and stableR>=.5; mostly connectivity if nominal positive, primary fails, and capCIincludes0 or stableR<.25; mixed otherwise. None proves a causal mechanism.

## Structural controls at T1/T2

{table(comparisons[(comparisons.seed=='equal_seed_mean')&comparisons.target.isin([p['T1'],p['T2']])])}

Gamma-zero, shuffled and edge-L2 controls remain included regardless of performance. A gamma-zero tie favors a parameterization/optimization interpretation over a residual-penalty benefit. Real-versus-shuffled contrasts probe grouping, with exploratory secondary prompt intervals; smoother group scores alone do not establish causal fidelity.

## Closure ranking: nominal-grid effects

{table(closure_summary.reset_index())}

Every closure ranking was evaluated on the full official CPR/CMD grid. closure_nominal_effects.csv reports original and closure actual counts beside **every** accuracy difference, plus CPR/CMD changes and exact mask identity. These are **nominal-grid leaderboard-counting effects**, not evidence of improved equal-size ranking quality. No matched-size closure evaluations were run: at counts attained by the original, the constructed closure boundary has the same originalP(n), so that evaluation is redundant. closure_rank_stability.csv provides parent/transformed signed-order Spearman; closure_cross_seed_stability.csv reports before/after cross-seed rank correlations and official pruned-edge Jaccard. Mandatory group/tail nesting checks document scope. An improvement does not authorize another campaign.

## Equal-weight rank ensembles

{table(ensemble[ensemble.target.isin([p['T1'],p['T2']])])}

Original score magnitudes are converted to signed ordinal ranks with canonical-index ties, averaged equally across seeds, then ties broken by canonical index. All edges appear once; no validation-dependent weights. All3ensembles were evaluated on the full official grid and registered cap targets. The baseline90% diagnostic atT2 is explicitly recorded above. Compare ensemble performance against the component minimum and each seed, not only a favorable average. ensemble_training_cost.csv charges the **three preexisting1000-step runs** separately from new inference/construction. Ensembling is not a cheaper discovery method.

{table(pd.read_csv(out/'ensemble_training_cost.csv'))}

Separate CPU construction replay (values match frozen exports exactly; original construction is already charged in PhaseA):

{table(pd.read_csv(out/'ensemble_construction_cost.csv'))}

## Actual compute and correctness

Inference-process wall time **{process['seconds']/60:.3f}minutes**, of which nominal/matched evaluation stages total **{eval_seconds/60:.3f}minutes**. Root model forwards **{process['counts']['forward_batches']:,}**, individual-example forwards **{process['counts']['example_forwards']:,}**, optimizer updates/backward calls **0/0**. Unique exact full-cohort masks **{process['unique_exact_masks']}**, output-cache hits **{process['output_cache_hits']}**, reference-scalar hits **{process['reference_scalar_hits']}**. Cache hits execute no model forward; no activation caches are used. CPU topology/construction time is separate and charged above. Preparation, bootstrap/reporting and audits are additional CPU work, not GPU training.

CUDA-synchronized monotonic timers measure actual stage wall time; allocation peaks are **{cost.peak_allocated_mib.max():.2f}MiB allocated**, **{cost.peak_reserved_mib.max():.2f}MiB reserved**, sampled CPU RSS **{cost.sampled_peak_cpu_rss_mib.max():.2f}MiB**. These are allocator/RSS values, not total physical VRAM occupancy or CUDA kernel-utilization time. Per-ranking nominal and matched-stage costs are saved. First-ranking projection was checked against7200seconds before any remaining full-cohort evaluation; its measured inputs are in first_ranking_projection.json. Exact per-prompt output reuse explains differences between requested and unique circuits.

Existing official batching/repeat/reference checks and new cached-versus-uncached MIB comparisons passed, with zero repeat forwards on cache hits. Full circuit matches clean reference, all evaluated masks match frozen official masks, no gradients are stored, and complete model weight hashes match before/after. Historical artifacts and Gate5C/5D decisions remain preserved.

## Limitations and recommendation

This is a capped-size experiment when actual counts differ. Exact subsets and interpolated estimates must retain their labels. Size matching does not establish a causal explanation of the nominal advantage; circuits can differ in composition and task behavior. Closure only audits first-appearance groups built from n<=1624, and nominal closure gains compare different actual sizes. Ensembles consume3trained runs. The same three trained/subset seeds and known IOI template families remain; prompt intervals do not establish training-seed significance or unseen-template generalization. Historical full-validation exposure remains, and no public/private test or leaderboard submission was used.

Use the frozen classification and exact-size evidence above to judge whether compact-ranking quality remains plausible. Treat closure as a separate deterministic nominal-grid direction and ensemble robustness as a three-run tradeoff. Any future direction is a recommendation only. Stop after Gate5E: no new training, score optimization, differentiable masks, further gates or unregistered ranking refinement.
'''
    for name in ('accuracy_actual_size','paired_target_difference','nominal_vs_capped','retention_curves','ensemble_vs_seeds','closure_cpr_cmd'):
        report+=f'\n\n![{name}](results/mattr_matched_size/{out.name}/figures/{name}.png)'
    for path in (out/'GATE5E_MATCHED_SIZE_REPORT.md',ROOT/'GATE5E_MATCHED_SIZE_REPORT.md'):
        path.write_text(report+'\n')
    write_json(out/'analysis_metadata.json',{'seconds':time.perf_counter()-start,'additional_model_forwards':0,'bootstrap_replicates':5000,'seed':5643})
    print(json.dumps(summary,indent=2),flush=True)


def figures(out,p,nominal,matched,comparisons,jumps,ensemble,metrics,closure):
    dest=out/'figures';dest.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout();plt.savefig(dest/(name+'.png'),dpi=150);plt.close()
    plt.figure(figsize=(9,5))
    for method,group in matched[(matched.selection=='cap')&(matched.kind=='original')].groupby('method'):
        mean=group.groupby('target')[['actual_edges','accuracy']].mean()
        plt.plot(mean.actual_edges,mean.accuracy,'o-',label=method)
    plt.xlabel('Actual retained edges (mean across3 fixed seeds)');plt.ylabel('Positive-margin accuracy');plt.legend();save('accuracy_actual_size')
    plt.figure(figsize=(8,4))
    main=comparisons[(comparisons.comparator=='baseline')&(comparisons.seed=='equal_seed_mean')].sort_values('target')
    plt.plot(main.target,main['mean']*100,'o-');plt.fill_between(main.target,main.lower*100,main.upper*100,alpha=.2)
    plt.axhline(0,color='black',linestyle='--');plt.axvline(p['T1'],linestyle=':',color='grey')
    plt.xlabel('At-most target actual edge count');plt.ylabel('Hierarchy minus baseline accuracy (points)');save('paired_target_difference')
    decision=json.loads((out/'gate5e_decision.json').read_text())
    plt.figure(figsize=(7,4))
    for i,name in enumerate(('nominal_0.5pct','cap_T1')):
        row=decision[name];plt.errorbar(i,row['mean']*100,yerr=np.maximum(0,np.array([[row['mean']-row['lower']],[row['upper']-row['mean']]]))*100,fmt='o',capsize=4)
    plt.xticks([0,1],['Nominal0.5%','CappedT1']);plt.axhline(0,color='black',linestyle='--')
    plt.ylabel('Hierarchy minus baseline accuracy (points)');save('nominal_vs_capped')
    plt.figure(figsize=(9,5))
    for key,g in jumps.groupby('ranking'):
        if key.startswith(('baseline_s','hierarchy_s')):plt.plot(g.nominal_n,g.actual_edges,label=key,alpha=.8)
    plt.plot([0,1624],[0,1624],'k--',label='No pruning');plt.xlabel('Nominal selected edges n');plt.ylabel('Actual A(n) after official pruning');plt.legend(fontsize=8);save('retention_curves')
    plt.figure(figsize=(9,5))
    base=ensemble[ensemble.method=='baseline'].sort_values('target')
    for col in ('seed42_accuracy','seed43_accuracy','seed44_accuracy','ensemble_accuracy'):
        plt.plot(base.target,base[col],'o-',label=col)
    plt.axhline(.9,color='black',linestyle='--');plt.xlabel('At-most target actual edges');plt.ylabel('Baseline positive-margin accuracy');plt.legend();save('ensemble_vs_seeds')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    unique=closure.drop_duplicates('original')
    for ax,metric in zip(axes,('CPR','CMD')):
        for row in unique.itertuples():
            a=metrics[metrics.ranking==row.original].iloc[0][metric];b=metrics[metrics.ranking==row.closure].iloc[0][metric]
            ax.scatter(a,b)
        values=metrics[metric];low,high=values.min(),values.max();ax.plot([low,high],[low,high],'k--')
        ax.set_xlabel('Original nominal-grid '+metric);ax.set_ylabel('Closure nominal-grid '+metric)
    save('closure_cpr_cmd')
