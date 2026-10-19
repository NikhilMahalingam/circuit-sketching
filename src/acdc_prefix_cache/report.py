"""Report measured Stage A evidence and explicitly unexecuted performance stages."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def generate(out):
    out=Path(out)
    config=read(out/'experiment_config.json')
    passed=(out/'stage_a_passed.json').exists()
    a=read(out/'stage_a_passed.json') if passed else {}
    query_path=out/'intervention_loss_comparison.csv'
    q=pd.read_csv(query_path) if query_path.exists() else pd.DataFrame()
    decisions=pd.read_csv(out/'decision_trace_comparison.csv') if (out/'decision_trace_comparison.csv').exists() else pd.DataFrame()
    memory_all=pd.read_csv(out/'memory_benchmark.csv') if (out/'memory_benchmark.csv').exists() else pd.DataFrame()
    memory=memory_all[memory_all.stage=='A_correctness_shared_residency'] if not memory_all.empty else memory_all
    estimates=[]
    for name,columns in [('layerwise_benchmark',['stage','method','trial','earliest_affected_layer','seconds']),
                         ('full_acdc_benchmark',['stage','method','trial','discovery_examples','candidate_checks','total_seconds'])]:
        if not (out/(name+'.csv')).exists():pd.DataFrame(columns=columns).to_csv(out/(name+'.csv'),index=False)
    lines=['# Prefix-cached ACDC: independent MIB adaptation','',f'Campaign: `{out.relative_to(ROOT)}`.','',
        '## Scope and independent reference','',
        'The supplied Downloads/RESEARCH_LOG.md, Stage J, reports 2.35× faster execution, bitwise loss agreement and a later replay of all 32,491 decisions with the same 686-edge GPT-2 circuit. It describes auto-circuit single-template IOI, ABC donors, and squared full-vocabulary probability distance. The reported full replay took 2.5 hours under power capping. Those are independent author-reported measurements, not measurements from this repository. The complete Stage-J implementation and original trace are not supplied or present in the inspected clones; source-log identity and the relevant excerpt are saved in independent_source.json. Exact reconstruction of that original experiment has not been performed.','',
        'This implementation instead uses official MIB IOI train/validation data, GPT-2 Small, float32, and the official negative logit difference (get_metric(..., loss=True)). A deletion is accepted when proposed mean loss minus current mean loss is strictly below the fixed threshold. This signed rule matches the default ACDC abs_value_threshold=False; the historical restricted Sketch-ACDC absolute-change decision rule is not silently reused. The threshold, traversal and starting graph are identical between the two new arms. This is an optimization reproduction on MIB IOI, not a numerical reproduction of the independent 2.35× experiment.','',
        '## Implementation and safety','',
        'The standard executor always runs the full transformer. The cached executor uses TransformerLens.forward(start_at_layer=L) with a cloned full-model residual stream before block L, the original token IDs and the original padding mask. Residual connections and all attention/MLP operations within the suffix are executed by TransformerLens itself. A final-logit receiver uses the last block rather than bypassing its hook_resid_post. Block zero falls back to a full forward.','',
        'The prefix proof uses the actual full MIB DAG. Every removed edge changes its receiver; its descendant closure is propagated through all graph dependencies. Cached sender nodes before the proposed boundary must have empty intersection with that closure, and no removed receiver hook may be skipped. The union includes all previously accepted removals, not only the current candidate. Arbitrary out-of-order states therefore move the boundary earlier when necessary; explicitly unsafe boundaries are rejected. Unknown nodes, unsupported architectures, model/config changes and incompatible checkpoint identities fail rather than silently reusing stale state.','',
        'Both arms cache identical fixed corrupted donors. Only the optimized arm also retains clean sender tensors and residual boundaries. At a skipped, proven-unaffected prefix, donor-minus-clean is valid. At every sender in the suffix, donor-minus-CURRENT-intervened output is recomputed. The matrix update is checked against the corrected shared receiver_patch function, corrected EdgeExperiment on dependent real-model queries, and the independent official MIB evaluate_graph. The historical FrozenSenderEdgeExperiment is never used. No gradients are tracked during inference. Repeated intervention queries are executed, not memoized.','',
        'Traversal sorts by descending receiver DAG phase, descending sender index, then edge name. Attention and MLP have separate within-block phases. The compatible exact search tests each real edge independently and performs no automatic cascading deletions or node-skipping shortcuts. The official MIB oracle does its own graph.prune(); its losses are independently checked, including a disconnected-head case. Saved retained identities refer to the explicit tested edge mask, not a relabeled canonical graph. This traversal/control flow is not claimed to match an unavailable auto-circuit trace.','',
        f'Configuration: discovery {config["train_examples"]}, validation {config["validation_examples"]}, batch {config["batch_size"]}, seed {config["seed"]}, threshold {config["threshold"]}, atol {config["atol"]}, rtol {config["rtol"]}. Public test is not used. Existing installed WSL packages are reused without installing dependencies or importing MAttr experiment code. The older corrected MIB clone is selected explicitly; all previously tracked experiment artifacts are protected by SHA-256 checks.','',
        '## Stage A measurements','']
    if passed:
        errors=q.absolute_error.to_numpy()
        meaningful=q.relative_error.dropna()
        lines += [f'**Stage A passed.** {a["single_queries"]} single-edge and {a["multi_queries"]} multi-edge queries cover receiver layers 0,2,4,6,8,10,11, q/k/v inputs, MLP inputs, final residual receivers, dependent senders, randomized masks, and an early previous deletion paired with a late candidate.','',
            f'Across {len(q)} per-example train comparisons: max absolute loss error **{errors.max():.8g}**, mean absolute error **{errors.mean():.8g}**, exact matches **{int(q.exact_match.sum())}/{len(q)}**. Maximum meaningful relative error: {meaningful.max():.8g}. Maximum error against the independent official MIB oracle: **{q.official_absolute_error.max():.8g}**.','',
            f'The restricted trajectory checks {a["restricted_decisions"]} edges selected across early/middle/late layers in reverse-topological subsequence order. The longer trajectory checks the first {a["longer_partial_decisions"]} edges of the full deterministic traversal. Candidate IDs, proposed losses, keep/remove decisions, before/after state hashes, and complete retained edge lists match. Interrupting, rebuilding caches and resuming the longer trajectory also matches all {a["resume_decisions"]} decisions. In total, {len(decisions)} saved comparison records include the resume replay; these are not {len(decisions)} unique full-graph decisions.','',
            '**No full 32,491-edge search was executed.** Matching these partial traces does not establish full-trajectory correctness or full-search speedup. Stage A uses diagnostic timings, not a repeated, isolated end-to-end performance experiment.','']
        if (out/'block_counts_a.json').exists():
            counts=read(out/'block_counts_a.json')
            lines += [f'Ten real small-TransformerLens unit tests passed. The GPU gate recorded {counts["root_forward_batches"]:,} actual root forward batches, {counts["example_forwards"]:,} example-forwards and {counts["block_executions"]:,} actual block executions across both arms, independent oracles, warmups, replay and held-out checks. Every recorded inference forward disabled gradients.','']
        validation=out/'stage_a_validation.csv'
        if validation.exists():
            v=pd.read_csv(validation)
            lines += ['Held-out validation checks of the two final partial circuits:','',
                '| Trajectory | Examples | Standard loss | Cached loss | Max discrepancy |',
                '|---|---:|---:|---:|---:|']
            for row in v.itertuples():
                lines.append(f'| {row.trajectory} | {row.examples} | {row.standard_loss:.7f} | {row.cached_loss:.7f} | {row.max_absolute_error:.3g} |')
            lines += ['']
        if not memory.empty:
            lines += [f'Maximum sampled Stage-A preparation allocator peak: {memory.peak_allocated_mib.max():.1f} MiB allocated / {memory.peak_reserved_mib.max():.1f} MiB reserved. Standard donor cache: {memory.loc[memory.method=="standard","cache_bytes"].iloc[0]/2**20:.1f} MiB; optimized donor + clean-prefix cache: {memory.loc[memory.method=="cached","cache_bytes"].iloc[0]/2**20:.1f} MiB. Both executors coexist for correctness testing, so these allocator peaks are **not** isolated per-method VRAM benchmarks. Stage B measures isolated allocations.','']
        # Estimates use actual Stage-A whole-query times, including all batches.
        s=pd.read_csv(out/'standard_execution.csv');c=pd.read_csv(out/'cached_execution.csv')
        standard_per_check=float(s.wall_seconds.median())
        cached_by_layer=c.groupby('earliest_affected_layer').wall_seconds.median()
        from .runner import configure_repository
        configure_repository(ROOT)
        from eap.graph import Graph
        # Use stored graph config, without loading another model or GPU weights.
        graph=Graph.from_model(read(out/'graph_config.json'))
        from .graph import Dependencies
        deps=Dependencies(graph)
        layers=np.array([graph.edges[n].child.layer for n in deps.order])
        def cached_check(layer):
            if layer in cached_by_layer:return float(cached_by_layer.loc[layer])
            nearest=min(cached_by_layer.index,key=lambda x:abs(x-layer))
            return float(cached_by_layer.loc[nearest])
        for size in config['stage_b_sizes']:
            scale=size/config['train_examples']
            setup=float(memory.constructor_plus_setup_seconds.sum())*scale
            b_checks=sum(config['stage_b_checks'])
            late=cached_check(11)
            b=config['timing']['repeat_trials_B_C']*(setup*len(config['stage_b_checks'])+
                b_checks*(standard_per_check+late)*scale)
            full=setup+len(layers)*standard_per_check*scale+sum(cached_check(int(l)) for l in layers)*scale
            estimates.append({'discovery_examples':size,'stage_b_medium_paired_seconds_estimate':b,
                'stage_c_one_paired_full_seconds_estimate':full,
                'stage_c_requested_repeats_seconds_estimate':full*config['timing']['repeat_trials_B_C'],
                'performance_measurement':False,'method':'Stage-A diagnostic query timing, linear examples scaling; layer-nearest interpolation'})
        pd.DataFrame(estimates).to_csv(out/'runtime_estimates.csv',index=False)
        lines += ['## Planning estimates before Stage B/C','',
            'These are rough planning estimates, not measured benchmark results, speedup claims, confidence intervals, or upper bounds. They scale Stage-A diagnostic query times linearly with discovery size and interpolate missing layers from nearby measured boundaries. Growing deletion-mask bookkeeping, checkpoint I/O, validation, microbenchmark warmup, GPU clocks and caching memory pressure can change runtime substantially. Stage B is needed to calibrate them. Stage-C rows below are one standard/cached full-graph pair; repeating all sizes is potentially hours-long.','',
            '| Discovery examples | B medium pairs (configured repeats) | C one full pair | C configured repeats |',
            '|---|---:|---:|---:|']
        for row in estimates:
            lines.append(f'| {row["discovery_examples"]} | {row["stage_b_medium_paired_seconds_estimate"]/60:.1f} min | {row["stage_c_one_paired_full_seconds_estimate"]/3600:.2f} h | {row["stage_c_requested_repeats_seconds_estimate"]/3600:.2f} h |')
        lines += ['', 'Stage B additionally includes seven-layer repeated microbenchmarks and held-out evaluations. The first traversal prefixes are concentrated at late receivers and must be labeled partial trajectories. Their end-to-end speedup cannot be promoted to a full-search speedup.','']
        plt.scatter(q.earliest_affected_layer,q.absolute_error,s=10,alpha=.4)
        plt.xlabel('Earliest affected block');plt.ylabel('Absolute standard − cached loss error')
        plt.tight_layout();plt.savefig(out/'stage_a_loss_agreement.png',dpi=160);plt.close()
    else:lines += ['Stage A has not passed. No numerical or pruning-equivalence conclusion is drawn.','']
    b_done=(out/'stage_b_passed.json').exists();c_done=(out/'stage_c_passed.json').exists()
    status=('The current Stage-A report therefore does not claim a measured speedup, fastest layer, discovery-size trend, or cross-model generalization.'
            if not b_done and not c_done else 'Only completed stages support the measured comparisons below; partial trajectories do not establish full-search performance.')
    lines += ['## Performance status and next action','',
        f'Stage B: {"completed" if b_done else "NOT RUN"}. Stage C: {"completed" if c_done else "NOT RUN; disabled by default"}. '
        +status+' The small real-model unit tests are structural correctness tests, not a second standardized model/task benchmark.','',
        'Next authorized checkpoint: review the implementation, numerical agreement and estimates before executing Stage B or C. No sketching, Hessians, group deletion or MAttr algorithm has been implemented. Possible later optimization: avoid conservative full-graph dependency propagation for nodes proven unable to affect logits, but that requires a separate correctness proof and is not implemented.','']
    # Render actual B/C data only when present; resumed costs include failed attempts.
    full=pd.read_csv(out/'full_acdc_benchmark.csv')
    if not full.empty:
        pairs=full.pivot(index=['stage','discovery_examples','candidate_checks','trial'],columns='method',values='total_seconds').dropna()
        pairs['speedup']=pairs.standard/pairs.cached
        pairs.reset_index().to_csv(out/'measured_paired_speedups.csv',index=False)
        lines += ['Measured end-to-end discovery speedup includes per-arm tokenization, constructor, donor/cache construction, synchronization, mask bookkeeping, exact decisions, checkpoint I/O and previous failed attempts. Common model/dataset loading and warmups are separate; validation time and including-validation totals are also saved. Medians and interquartile ranges are reported across paired trials.','']
        for key,group in pairs.groupby(level=['stage','discovery_examples','candidate_checks']):
            lo,hi=np.quantile(group.speedup,[.25,.75])
            lines.append(f'- Stage {key[0]}, n={key[1]}, checks={key[2]}: median speedup {group.speedup.median():.3f}×, IQR [{lo:.3f}, {hi:.3f}], {len(group)} paired trials. '+('Full graph.' if key[0]=='C' else 'Partial trajectory.'))
        plot=pairs.reset_index()
        for (stage,checks),group in plot.groupby(['stage','candidate_checks']):
            stat=group.groupby('discovery_examples').speedup.median()
            plt.plot(stat.index,stat,'o-',label=f'{stage}, {checks} checks')
        plt.xlabel('Discovery examples');plt.ylabel('End-to-end discovery speedup');plt.legend()
        plt.tight_layout();plt.savefig(out/'speedup_vs_examples.png',dpi=160);plt.close()
        for (stage,size),group in plot.groupby(['stage','discovery_examples']):
            stat=group.groupby('candidate_checks').speedup.median()
            plt.plot(stat.index,stat,'o-',label=f'{stage}, n={size}')
        plt.xlabel('Candidate checks');plt.ylabel('End-to-end discovery speedup');plt.legend()
        plt.tight_layout();plt.savefig(out/'speedup_vs_checks.png',dpi=160);plt.close()
        if not memory_all.empty:
            isolated=memory_all[memory_all.isolated_memory_comparison==True]
            for method,group in isolated.groupby('method'):
                plt.scatter(group.cache_bytes/2**20,group.peak_allocated_mib,label=method)
            plt.xlabel('Cache size (MiB)');plt.ylabel('Peak allocated CUDA memory (MiB)');plt.legend()
            plt.tight_layout();plt.savefig(out/'memory_vs_cache.png',dpi=160);plt.close()
    micro=pd.read_csv(out/'layerwise_benchmark.csv')
    if not micro.empty:
        stat=micro.groupby(['earliest_affected_layer','method']).seconds.median().unstack()
        stat['speedup']=stat.standard/stat.cached;stat.to_csv(out/'layerwise_summary.csv')
        plt.plot(stat.index,stat.speedup,'o-');plt.xlabel('Earliest affected block');plt.ylabel('Median microbenchmark speedup')
        plt.tight_layout();plt.savefig(out/'speedup_vs_layer.png',dpi=160);plt.close()
        micro.groupby(['earliest_affected_layer','method']).seconds.agg(
            median='median',std='std',p25=lambda x:x.quantile(.25),p75=lambda x:x.quantile(.75),
            samples='size').to_csv(out/'layerwise_variability.csv')
    failures=sorted(out.rglob('failure_*.json'))
    if failures:lines += ['', 'Unsuccessful stages (not silently excluded): '+', '.join(str(p.relative_to(out)) for p in failures)]
    text='\n'.join(lines)+'\n'
    (ROOT/'ACDC_PREFIX_CACHE_REPORT.md').write_text(text,encoding='utf-8')
    (out/'ACDC_PREFIX_CACHE_REPORT.md').write_text(text,encoding='utf-8')
    print('Report:',ROOT/'ACDC_PREFIX_CACHE_REPORT.md',flush=True)
