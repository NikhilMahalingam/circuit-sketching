"""Analyze saved official outputs; no inference or training."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.mattr_optimization.common import ROOT, sha, write_json
from src.mattr_optimization.gate2_final_analysis import table
from .design import METHODS, SEEDS, validate_protocol
from .statistics import bootstrap_indices, compare_arrays, decision, auc_prompt_values


def generate(out):
    start = time.perf_counter()
    p = validate_protocol(out)
    cohort = pd.read_csv(out / 'cohort_ids_and_hashes.csv')
    grid = p['grid']
    summary, curves, families = [], [], []
    arrays = {}
    for method in METHODS:
        per_seed = []
        for seed in SEEDS:
            dest = out / f'evaluations/{method}_s{seed}'
            saved = json.loads((dest / 'official_result.json').read_text())
            assert sha(dest / 'per_prompt.csv') == saved['per_prompt_sha256']
            result = saved['result']
            frame = pd.read_csv(dest / 'per_prompt.csv')
            curve = pd.read_csv(dest / 'curve.csv')
            summary.append({'method': method, 'seed': seed, 'CPR': result[1], 'CMD': result[2], 'accuracy_log_AUC': result[6],
                            **{f'accuracy_{f}': result[5][grid.index(f)] for f in p['compact_fractions']},
                            **{f'top1_{f}': float(curve.loc[np.isclose(curve.fraction, f), 'top1_accuracy'].iloc[0]) for f in p['compact_fractions']}})
            curves.append(curve)
            accuracy = frame.pivot(index='fraction', columns='example', values='correct').loc[grid].to_numpy(dtype=float)
            assert accuracy.shape == (10, len(cohort))
            assert np.allclose(auc_prompt_values(accuracy[None], grid).mean(), result[6], atol=1e-7)
            per_seed.append(accuracy)
            fam = frame.groupby(['template_id', 'fraction'], as_index=False).agg(accuracy=('correct', 'mean'),
                       top1_accuracy=('top1', 'mean'), examples=('example', 'count'))
            families.append(fam.assign(method=method, seed=seed))
        arrays[method] = np.stack(per_seed)
    summary = pd.DataFrame(summary)
    curves = pd.concat(curves, ignore_index=True)
    families = pd.concat(families, ignore_index=True)
    summary.to_csv(out / 'official_metrics.csv', index=False)
    curves.to_csv(out / 'faithfulness_curves.csv', index=False)
    families.to_csv(out / 'template_family_accuracy.csv', index=False)
    compact = curves[curves.fraction.isin(p['compact_fractions'])]
    compact.to_csv(out / 'compact_accuracy_and_edge_counts.csv', index=False)
    idx = bootstrap_indices(cohort.template_id, p['bootstrap']['replicates'], p['bootstrap']['seed'])
    comparisons, aucs = [], []
    for comparator in ('baseline', 'gamma_zero', 'shuffled', 'edge_l2'):
        frame, auc = compare_arrays(arrays['hierarchy'], arrays[comparator], cohort.template_id, grid, idx, comparator)
        comparisons.append(frame)
        aucs.append(auc)
    comparisons = pd.concat(comparisons, ignore_index=True)
    aucs = pd.concat(aucs, ignore_index=True)
    comparisons.to_csv(out / 'paired_bootstrap_accuracy.csv', index=False)
    aucs.to_csv(out / 'paired_bootstrap_accuracy_auc.csv', index=False)
    pooled = comparisons[comparisons.seed == 'equal_seed_mean']
    per_template = []
    for family in sorted(set(cohort.template_id)):
        mask = cohort.template_id == family
        for method in ('hierarchy', 'gamma_zero', 'shuffled', 'edge_l2'):
            for f in p['compact_fractions']:
                j = grid.index(f)
                for si, seed in enumerate(SEEDS):
                    delta = arrays[method][si, j, mask] - arrays['baseline'][si, j, mask]
                    per_template.append({'template_id': family, 'method': method, 'seed': seed, 'fraction': f,
                                         'prompts': int(mask.sum()), 'accuracy_delta': float(delta.mean())})
    pd.DataFrame(per_template).to_csv(out / 'template_family_paired_differences.csv', index=False)
    valid = json.loads((out / 'correctness_passed.json').read_text())['passed']
    verdict = decision(comparisons, aucs, valid)
    write_json(out / 'gate5d_decision.json', verdict)
    cost = pd.read_csv(out / 'compute_accounting.csv')
    per_method = cost[cost.stage.str.startswith('evaluate_')].copy()
    per_method['method_seed'] = per_method.stage.str.removeprefix('evaluate_')
    per_method.to_csv(out / 'per_method_compute.csv', index=False)
    process = json.loads((out / 'inference_process_summary.json').read_text())
    assert int(cost.forward_batches.sum()) == process['counts']['forward_batches']
    assert int(cost.example_forwards.sum()) == process['counts']['example_forwards']
    assert process['counts']['backward_calls'] == process['counts']['optimizer_updates'] == 0
    fig = out / 'figures'
    fig.mkdir(exist_ok=True)
    def save(name):
        plt.tight_layout()
        plt.savefig(fig / (name + '.png'), dpi=150)
        plt.close()
    plt.figure(figsize=(8, 4))
    for method, group in compact.groupby('method'):
        y = group.groupby('fraction').accuracy.mean().reindex(p['compact_fractions'])
        plt.plot(np.asarray(p['compact_fractions']) * 100, y, 'o-', label=method)
    plt.xscale('log'); plt.xlabel('Nominal retained edges (%) before connectivity pruning')
    plt.ylabel('Positive-margin accuracy (3-seed mean)'); plt.legend(); save('compact_accuracy')
    plt.figure(figsize=(8, 4))
    for seed in SEEDS:
        group = comparisons[(comparisons.comparator == 'baseline') & (comparisons.seed == str(seed))].sort_values('fraction')
        plt.plot(group.fraction * 100, group.delta * 100, 'o-', label=f'seed {seed}')
    plt.axhline(0, color='black', linestyle='--'); plt.xscale('log'); plt.xlabel('Nominal retained edges (%)')
    plt.ylabel('Hierarchy minus baseline accuracy (percentage points)'); plt.legend(); save('per_seed_accuracy_difference')
    plt.figure(figsize=(9, 4))
    selected = pooled[pooled.fraction.isin([.005, .01])].reset_index(drop=True)
    for i, row in selected.iterrows():
        plt.errorbar(row.delta * 100, i, xerr=np.array([[row.delta - row.lower], [row.upper - row.delta]]) * 100, fmt='o')
    plt.yticks(range(len(selected)), [f'{r.comparator}, {r.fraction * 100:g}%' for r in selected.itertuples()])
    plt.axvline(0, color='black', linestyle='--'); plt.xlabel('Hierarchy minus comparator accuracy (percentage points)')
    plt.title('Pooled paired prompt bootstrap; Bonferroni primary/control families'); save('paired_bootstrap_intervals')
    plt.figure(figsize=(8, 4))
    counts = comparisons[(comparisons.comparator == 'baseline') & comparisons.fraction.isin([.005, .01]) & (comparisons.seed != 'equal_seed_mean')]
    x = np.arange(len(counts))
    plt.bar(x - .18, counts.corrected, .36, label='Corrected by hierarchy')
    plt.bar(x + .18, counts.broken, .36, label='Broken by hierarchy')
    plt.xticks(x, [f'{r.seed}/{r.fraction*100:g}%' for r in counts.itertuples()]); plt.ylabel('Prompt count')
    plt.legend(); save('corrected_vs_broken')
    plt.figure(figsize=(7, 4))
    for method, group in summary.groupby('method'):
        plt.scatter(group.CMD, group.CPR, label=method)
        for r in group.itertuples():
            plt.annotate(str(r.seed), (r.CMD, r.CPR), fontsize=7)
    plt.xlabel('Official CMD (lower is closer to full-model margin recovery)'); plt.ylabel('Official CPR'); plt.legend(); save('cpr_vs_cmd')
    plt.figure(figsize=(max(8, len(set(cohort.template_id)) * .28), 4))
    template_order = sorted(set(cohort.template_id))
    for fraction in (.005, .01):
        values = pd.DataFrame(per_template)
        values = values[(values.method == 'hierarchy') & (values.fraction == fraction)].groupby('template_id').accuracy_delta.mean()
        plt.plot(np.arange(len(template_order)), values.reindex(template_order) * 100, 'o-', label=f'{fraction*100:g}%')
    plt.xticks(range(len(template_order)), [x[:6] for x in template_order], rotation=90, fontsize=7)
    plt.axhline(0, color='black', linestyle='--'); plt.ylabel('Hierarchy minus baseline accuracy (points)')
    plt.xlabel('Exact official template ID (hash prefix); fixed families, no unseen-family claim'); plt.legend(); save('template_family_accuracy')
    means = summary.groupby('method', as_index=False).mean(numeric_only=True).drop(columns='seed')
    means.to_csv(out / 'method_means.csv', index=False)
    endpoint = pooled[(pooled.comparator == 'baseline') & pooled.fraction.isin([.005, .01])]
    audit = json.loads((out / 'cohort_audit.json').read_text())
    headline = {'classification': verdict['classification'], 'primary_success': verdict['primary']['passed'],
                'structural_advantage': verdict['structural_advantage'],
                'primary_endpoint_results': endpoint.to_dict(orient='records'),
                'method_means': means.to_dict(orient='records'),
                'total_evaluation_seconds': float(per_method.seconds.sum()), 'inference_process_seconds': process['seconds'],
                'actual_forward_calls': process['counts']['forward_batches'],
                'example_forwards': process['counts']['example_forwards'],
                'peak_cuda_allocated_mib': float(cost.peak_allocated_mib.max()),
                'peak_cuda_reserved_mib': float(cost.peak_reserved_mib.max()),
                'sampled_peak_cpu_rss_mib': float(cost.sampled_peak_cpu_rss_mib.max()),
                'reference_cache_hits': process['reference_or_result_cache_hits'], 'optimizer_updates': 0}
    write_json(out / 'research_summary.json', headline)
    selected_checks = comparisons[(comparisons.comparator == 'baseline') & comparisons.fraction.isin([.005, .01])]
    report = f'''# Gate 5D: Hierarchical MAttr compact-circuit generalization

## Executed decision

**{verdict['classification']}**. All15 saved rankings were evaluated at exactly1000 training updates; Gate5D performs **zero optimizer updates and zero model backward calls**. This gate evaluates a new compact-accuracy hypothesis; it does not override Gate5C's failed CPR gate or authorize5000-step continuation. Ordinary MAttr remains the CPR reference unless independently demonstrated otherwise.

{table(endpoint[['fraction','delta','lower','upper','corrected','broken']])}

Differences are accuracy fractions, not percentage points. Pooled values equally weight the three FIXED trained seeds. Corrected/broken pooled counts are means of per-seed counts, not independent3N observations. The confidence intervals are paired, exact-template-stratified prompt bootstrap with5000 shared resamples, seed5642, Bonferroni over the two primary endpoints (97.5% marginal percentile intervals, simultaneous coverage at least95%). These intervals describe prompt sampling conditional on the fixed template mix, not training-seed significance or unseen-template performance.

## Protocol, historical exposure and cohort audit

Protocol SHA-256: `{sha(out / 'gate5d_protocol.json')}`. Frozen before any model forward on the new cohort. All methods/seeds,15 checkpoint/effective-score hashes, original32491-edge mapping, exclusions, exact row IDs and UTF8 hashes, circuit grid, bootstrap and thresholds were recorded. Source pins, checkpoint metadata, group mappings, model configuration and effective exports were audited. No methods were selected or discarded using fresh-cohort performance.

Selected **{len(cohort)}** official MIB IOI validation prompts from **{audit['eligible_unique_rows']}** eligible unique strings, stratified proportionally across **{len(audit['selected_template_counts'])}** exact official `template` values using largest-remainder quotas and seed5542. Exclusions use both filtered official row IDs and exact prompt SHA256, including prior small Gate2/3/4/5A development evaluations, Gate5B held-out confirmation, Gate5C development, and all located discovery-training manifests. Clean prompt duplicates are removed even if row IDs differ. Corrupted-prompt hashes from historical manifests are also excluded conservatively. Cohort audit records every input manifest/hash. Official names/labels are verified single tokens. No public/private test data or generated substitute is used.

**Historical Gate1 evaluated the full official10000-example validation split. These prompts are fresh relative to the smaller development/confirmation cohorts, not untouched by all previous research.** Template labels come directly from the dataset's `template` field; none are invented from text. Some templates can remain correlated, and stratified prompt intervals condition on the observed family mixture. This is not a held-out-family design.

## Frozen checkpoint provenance

All methods use Gate5C1000-step scores (baseline exact-bypass scores verified bitwise against Gate2). Effective scores are evaluated for hierarchical checkpoints; raw packed h/residual parameters are never interpreted as edge scores. Per-checkpoint hashes and config/cohort provenance are in the protocol and checkpoint_provenance.csv. No model training, score adaptation, ranking refinement or mask changes occurred.

## Official per-seed and descriptive means

{table(summary)}

{table(means)}

The original official signed `apply_topn(..., absolute=False, prune=True)` selects nominal circuit fractions. Rankings are nested before pruning; actual retained-edge counts can be smaller and are not forced to nominal counts. `faithfulness_curves.csv` and `compact_accuracy_and_edge_counts.csv` retain every official grid point and mask hash. Official CPR/CMD use ratios of cohort means and the unchanged evaluator's integrals; no homemade replacement is used. Accuracy is the proportion with correct-minus-incorrect margin>0. Full-vocabulary top1 is a separate official `acc` diagnostic captured from the same logits, with no extra model pass. CPR rewards amplification and is not interchangeable with model fidelity or compact accuracy; CMD is reported separately.

## Paired uncertainty, corrections and controls

{table(selected_checks[['seed','fraction','delta','lower','upper','corrected','broken']])}

{table(pooled[['comparator','fraction','delta','lower','upper']])}

{table(aucs)}

Primary success requires positive mean differences and positive corrected pooled lower bounds at both0.5%/1%, at least2/3 positive seeds at each, mean accuracy-log-AUC change>=-.02, no mean loss>.02 at2%/5%, and valid semantics. Per-seed primary intervals correct for two endpoints within each seed, not all six seed-specific tests. Control intervals use a distinct Bonferroni family of6 pooled contrasts (3controls x2 primary cutoffs). Accuracy-log-AUC95% intervals are secondary/unadjusted; bootstrap replicates use the linear functional of per-prompt correctness with weights verified against the official point estimate. No bootstrap treats seeds as independently sampled prompts.

Frozen decision details:

```json
{json.dumps(verdict, indent=2)}
```

If gamma-zero hierarchy shows the same gain, evidence favors parameterization/optimization geometry rather than the residual penalty. Actual component grouping must beat shuffled grouping, gamma-zero hierarchy AND generic edge regularization at both primary cutoffs to support the registered structural claim. Group-score stability alone is not evidence of causal fidelity. Controls remain included regardless of outcome; no further sweep or training is authorized.

## Exact-template descriptive analysis

Full per-template/seed/method accuracy and top1 are in template_family_accuracy.csv; paired template contrasts are in template_family_paired_differences.csv. Template counts and IDs are frozen. Small per-template samples and repeated lexical patterns limit precision; the template plot is descriptive, with no post-hoc family selection or unseen-family significance claim.

## Actual retained edges

{table(compact[['method','seed','fraction','requested_edges','actual_edges','accuracy','top1_accuracy']])}

## Computation and inference correctness

Official full-grid evaluation wall time **{per_method.seconds.sum()/60:.3f}minutes**. Inference-process wall time including loading/correctness **{process['seconds']/60:.3f}minutes**. Actual model forward calls **{process['counts']['forward_batches']:,}**, individual example-forwards **{process['counts']['example_forwards']:,}**. Unique evaluation prompts **{len(cohort)}**, reused identically for each of15 rankings. CUDA timers synchronize and use monotonic perf_counter; root-forward counters count real calls, not cache hits. Official graph-independent reference scalar/result cache hits **{process['reference_or_result_cache_hits']}**; donor activation cache disabled, no mutable patched activations reused. References computed once for the same model/cohort/intervention are safe to share across rankings.

Peak allocated CUDA **{cost.peak_allocated_mib.max():.2f}MiB**, peak reserved **{cost.peak_reserved_mib.max():.2f}MiB**, sampled peak CPU RSS **{cost.sampled_peak_cpu_rss_mib.max():.2f}MiB**. These are allocator/RSS measurements, not total physical VRAM occupancy or CUDA kernel utilization time. Per-method timing and forwards are in per_method_compute.csv; the first method bears one-time reference computation. All methods use batch20; no silent resource adaptation. Model parameters have no stored gradients and complete before/after weight hashes match. Correctness checks compare repeated and singleton batching, full-circuit versus clean reference, nested/pruned masks, and official reference-cache equivalence before the full evaluation.

{table(per_method[['method_seed','seconds','forward_batches','example_forwards','peak_allocated_mib','peak_reserved_mib','sampled_peak_cpu_rss_mib']])}

## Recommendation and limitations

Use the frozen primary/control decisions above to interpret compact-accuracy generalization separately from structural attribution and CPR. A primary success warrants further independent-cohort research as a recommendation only; it does not demonstrate a leaderboard victory. A gamma-zero tie prevents attributing gains to regularization. Only three fixed paired training/subset seeds were measured, the official templates are reused, historical full-validation exposure remains, and no full-data/multi-model generalization was tested. Preserve the original MAttr and historical failed gates. Stop after Gate5D; no new training, algorithms, further gates or leaderboard submission.
'''
    for name in ('compact_accuracy','per_seed_accuracy_difference','paired_bootstrap_intervals','corrected_vs_broken','cpr_vs_cmd','template_family_accuracy'):
        report += f'\n\n![{name}](results/mattr_compact_validation/{out.name}/figures/{name}.png)'
    for path in (out / 'GATE5D_COMPACT_GENERALIZATION_REPORT.md', ROOT / 'GATE5D_COMPACT_GENERALIZATION_REPORT.md'):
        path.write_text(report + '\n')
    provenance_rows = [{k: v for k, v in rec.items() if k != 'source_provenance'} for rec in p['rankings']]
    pd.DataFrame(provenance_rows).to_csv(out / 'checkpoint_provenance.csv', index=False)
    write_json(out / 'analysis_metadata.json', {'seconds': time.perf_counter() - start, 'additional_model_forwards': 0,
               'bootstrap_unit': 'shared prompt within exact official templates', 'replicates': 5000})
    print(json.dumps(headline, indent=2), flush=True)
