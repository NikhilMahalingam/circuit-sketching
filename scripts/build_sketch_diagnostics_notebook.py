"""Build the diagnostics notebook without editing any existing notebook."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# Why did second-order Sketch-ACDC underperform?

Reuse the existing GPT-2 Small MIB IOI loader, 16 EAP-selected edges, saved ACDC trajectory, edge mapping, exact query API, HVP/Nyström routines, CUDA timing, and forward ledger. Use the existing project kernel. This notebook installs nothing and edits no earlier notebook.

**Default: lightweight smoke only.** Load the requested 16 train / 32 validation prompts, then use 2 prompts per split, one saved partial state, two candidates, and rank 4 for execution checks. Set `SMOKE=False` for the full diagnostic query set. The expensive exact-autodiff Hessian remains separately optional via `RUN_FULL_HESSIAN=True`. Conclusions are generated only from measured rows, and smoke conclusions are explicitly provisional.

An intervention caveat is audited first: MIB captures current sender outputs during the intervened pass. The historical adapter captured clean sender outputs in a separate pass. They agree for isolated removals but can differ for dependent masks. The diagnostic adapter inherits the existing mapping/API and uses MIB's dynamic sender convention with gradients preserved through the current sender. Previous notebooks and measured artifacts remain historical records.''')

code('''from pathlib import Path
import os, sys, json, hashlib, importlib.metadata
from datetime import datetime, timezone
import numpy as np, pandas as pd, torch
ROOT=Path.cwd().resolve()
if not (ROOT/'mib_ioi/runtime.py').exists(): ROOT=ROOT.parent
sys.path.insert(0,str(ROOT))
from mib_ioi.runtime import configure_repository, seed_everything, load_model, load_splits, subset, Ledger, write_json
from src.sketch_acdc.core import EdgeExperiment, FrozenSenderEdgeExperiment, exact_acdc, hessian_vector_product
from src.sketch_acdc.diagnostics import (DiagnosticEdgeExperiment,load_edges,load_states,exact_queries,
    predict_methods,threshold_grid,analyze,plot_results,automatic_summary,decision_metrics)
SOURCE_RUN=ROOT/'results/mib_sketch_acdc/20261015T054348Z'
SMOKE=os.environ.get('MIB_DIAGNOSTICS_FULL_RUN')!='1'
RUN_FULL_HESSIAN=os.environ.get('MIB_DIAGNOSTICS_FULL_HESSIAN')=='1'
RUN_NEW_TRAJECTORIES=False
SEED=42
TRAIN_N,VALIDATION_N=16,32
BATCH_SIZE=2
RANKS=[4] if SMOKE else [4,8,12,16]
ANCHORS=[0.0,0.5]
OUT=ROOT/'results/mib_sketch_diagnostics'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
os.environ.setdefault('HF_DATASETS_OFFLINE','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
seed_everything(SEED)
print('SMOKE:',SMOKE,'full Hessian:',RUN_FULL_HESSIAN,'ranks:',RANKS,'results:',OUT)''')

md('''## Reuse the selected edges and exact ACDC states

No EAP reranking is performed. The source run and source hashes are saved. Partially pruned states are selected near 1, 3, 6, and 9 accepted removals from the existing train-only trajectory. Threshold decisions will be swept from 0.01 to 0.30 on these fixed queries. Optional new train-only trajectories can extend state coverage.

Thresholds near observed effects are chosen **only from training effects**, then applied unchanged to validation effects. An absence of effects in this interval is a diagnostic result; the notebook does not manufacture difficult decisions by changing edges or labels.''')

code('''source_config=json.loads((SOURCE_RUN/'experiment_config.json').read_text())
assert source_config['model']=='gpt2' and source_config['task']=='ioi'
repo,provenance=configure_repository(ROOT)
assert provenance['mib_commit']==source_config['mib_provenance']['mib_commit']
assert provenance['eap_commit']==source_config['mib_provenance']['eap_commit']
model=load_model('cuda')
datasets,manifest,split_info=load_splits(model,{'train':TRAIN_N,'validation':VALIDATION_N},SEED)
manifest.to_csv(OUT/'example_manifest.csv',index=False)
from eap.graph import Graph
graph=Graph.from_model(model);graph.reset(empty=False)
edges=load_edges(SOURCE_RUN/'selected_edges.csv')
assert len(edges)==16
pd.DataFrame([vars(e) for e in edges]).to_csv(OUT/'selected_edges.csv',index=False)
states=load_states(SOURCE_RUN/'exact_acdc_trajectory.csv',16)
if SMOKE: states=states[:1]
effective={k:subset(ds,range(2)) if SMOKE else ds for k,ds in datasets.items()}
effective['audit']=subset(datasets['train'],range(2))
old_notebooks={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (ROOT/'notebooks').glob('*.ipynb') if 'diagnostics' not in p.name}
source_paths=[SOURCE_RUN/'selected_edges.csv',SOURCE_RUN/'exact_acdc_trajectory.csv',
              ROOT/'src/sketch_acdc/core.py',ROOT/'src/sketch_acdc/diagnostics.py']
config={'model':'gpt2','task':'ioi','seed':SEED,'batch_size':BATCH_SIZE,'smoke':SMOKE,
        'requested_prompts':{'train':TRAIN_N,'validation':VALIDATION_N},
        'effective_prompts':{k:len(v) for k,v in effective.items()},
        'splits':split_info,'effective_split_fingerprints':{k:v.dataset._fingerprint for k,v in effective.items()},
        'edges':[vars(e) for e in edges],'source_run':str(SOURCE_RUN),'mib_provenance':provenance,
        'ranks':RANKS,'anchors':ANCHORS,'run_full_hessian':RUN_FULL_HESSIAN,
        'intervention':DiagnosticEdgeExperiment.VERSION,
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        'previous_notebook_sha256':old_notebooks,
        'packages':{p:importlib.metadata.version(p) for p in ['torch','transformer-lens','numpy','pandas','scipy']}}
ledger=Ledger('cuda',OUT)
experiment=DiagnosticEdgeExperiment(model,graph,edges,effective,BATCH_SIZE,json.dumps(config,sort_keys=True),
                                   ledger,cache_dir=OUT/'exact_cache')
write_json(OUT/'experiment_config.json',config)
print('Loaded official splits:',{k:len(v) for k,v in datasets.items()},'effective:',config['effective_prompts'])
print('Partial states:', [''.join(map(str,s)) for s in states])''')

md('''## Verify MIB equivalence and derivatives before comparing approximations

Audit baseline, individual, partially pruned, and dependent masks against the independent official `evaluate_graph`. Also record the discrepancy from the historical frozen-sender adapter. A dependent stress mask includes the first six high-attribution edges; it is a mechanics test, not an ACDC-selected circuit.

Check CUDA HVPs against centered differences of gradients on two official train prompts. All model weights are frozen. Inference results and disk-cache entries contain detached scores only.''')

code('''from numpy.testing import assert_allclose
zero=[0]*16
one=zero.copy();one[0]=1
dependent=zero.copy();dependent[:6]=[1]*6
frozen=FrozenSenderEdgeExperiment(model,graph,edges,effective,BATCH_SIZE,'historical_frozen_audit',ledger)
audit=[]
with ledger.instrument(model),ledger.stage('intervention_audit',estimate='4 binary masks on 2 train prompts; official and historical adapter comparisons'):
    before=experiment.evaluate_state(zero,'audit',fresh=True)
    for name,state in [('baseline',zero),('single',one),('partial',states[0]),('dependent',dependent)]:
        actual=experiment.evaluate_state(state,'audit',fresh=True)
        official=experiment.official_binary_score(state,'audit')
        historical=frozen.evaluate_state(state,'audit',fresh=True)
        assert_allclose(actual,official,atol=5e-4,rtol=1e-4)
        audit.append({'mask':name,'state':''.join(map(str,state)),
                      'max_abs_dynamic_vs_official':float(max(abs(actual-official))),
                      'max_abs_historical_vs_official':float(max(abs(historical-official)))})
    assert_allclose(before,experiment.evaluate_state(zero,'audit',fresh=True),atol=3e-4,rtol=1e-4)
    calls_before=len(ledger.calls)
    cached=experiment.evaluate_state(one,'audit')
    assert len(ledger.calls)==calls_before
    experiment.cache.clear()
    assert_allclose(cached,experiment.evaluate_state(one,'audit'),atol=0,rtol=0)
audit=pd.DataFrame(audit);audit.to_csv(OUT/'intervention_audit.csv',index=False)
print(audit.to_string(index=False))
with ledger.instrument(model),ledger.stage('numerical_derivative_check',estimate='one HVP and two gradient finite differences on 2 prompts'):
    point=torch.zeros(16,device='cuda',dtype=model.cfg.dtype);point[:6]=0.3
    direction=torch.zeros_like(point);direction[0]=1;direction[3]=-0.5
    f=lambda z:experiment.continuous(z,'audit').mean()
    hv=hessian_vector_product(f,point,direction)
    grad=lambda z:torch.autograd.grad(f(z.requires_grad_(True)),z)[0].detach()
    eps=0.01
    finite=(grad((point+eps*direction).clone())-grad((point-eps*direction).clone()))/(2*eps)
    assert_allclose(hv.cpu(),finite.cpu(),atol=0.01,rtol=0.08)
    ledger.backward_calls+=4;ledger.hvp_calls+=1
    assert all(p.grad is None for p in model.parameters())
write_json(OUT/'numerical_checks.json',{'official_binary_equivalence':True,'cache_forward_count':True,
    'hvp_finite_difference_max_abs_error':float((hv-finite).abs().max()),'no_parameter_gradients':True})
print('Dynamic-sender HVP finite-difference max error:',float((hv-finite).abs().max()))''')

md('''## Experiment 1 — exact candidate effects and difficult thresholds

All prediction methods use exactly these endpoints and the same train prompts. Exact per-example effects are saved for both train and validation. Disk cache identity includes model/configuration, split fingerprints, source hashes, adapter version, edge order, and binary state; cache hits do not count as model forward passes.

`false_prune_rate` and `false_retain_rate` use all measured decisions as denominator; conditional rates are also saved. A prune decision is `abs(effect) < threshold`, matching the existing restricted ACDC convention.''')

code('''if RUN_NEW_TRAJECTORIES and not SMOKE:
    generated=[]
    with ledger.instrument(model),ledger.stage('new_train_trajectories',estimate='5 thresholds x 16 sequential trials on train prompts'):
        for threshold in [0.01,0.05,0.10,0.20,0.30]:
            _,rows=exact_acdc(experiment,'train',threshold)
            generated.extend(dict(row,threshold=threshold) for row in rows)
    pd.DataFrame(generated).to_csv(OUT/'recomputed_acdc_trajectories.csv',index=False)
    states=load_states(OUT/'recomputed_acdc_trajectories.csv',16)
candidate_limit=2 if SMOKE else None
n_queries=sum(min(candidate_limit or 16,sum(1-bit for bit in s)) for s in states)
exact_batches=2*sum((len(effective[k])+BATCH_SIZE-1)//BATCH_SIZE for k in ['train','validation'])*(n_queries+len(states))
print('Exact query estimate:',n_queries,'candidate removals across',len(states),'states, at most',exact_batches,'forward batches before hits')
with ledger.instrument(model),ledger.stage('exact_ground_truth',estimate=f'{n_queries} fixed candidate queries on train and validation'):
    queries,per_example=exact_queries(experiment,states,candidate_limit)
queries.to_csv(OUT/'exact_queries.csv',index=False)
per_example.to_csv(OUT/'per_example_effects.csv',index=False)
thresholds=threshold_grid(queries.train_effect)
pd.DataFrame({'threshold':thresholds}).to_csv(OUT/'thresholds.csv',index=False)
coverage=int(((abs(queries.train_effect)>=0.01)&(abs(queries.train_effect)<=0.30)).sum())
config.update(states=states,query_count=len(queries),thresholds=thresholds.tolist(),
              train_queries_in_threshold_range=coverage)
write_json(OUT/'experiment_config.json',config)
print('Train effects in [0.01,0.30]:',coverage,'/',len(queries),'; sweep thresholds:',len(thresholds))
print(queries[['query_id','train_effect','validation_effect']].head().to_string(index=False))''')

md('''## Experiments 2–3 — separate Taylor truncation, compression, and prompt shift

For endpoints `s` and `t=s+e_i` and shared expansion point `a`, predict the difference of Taylor polynomials:

`g·(t−s) + 0.5[(t−a)ᵀH(t−a) − (s−a)ᵀH(s−a)]`.

At anchor strength zero this becomes `g_i + H_ii/2`. The 0.5 anchor sets **all intact candidate coordinates** to 0.5 while retaining already removed coordinates at 1; the cross terms therefore matter. This shared midpoint can be far from either endpoint. A candidate-only midpoint would cancel the quadratic term in the endpoint difference, so a midpoint improvement alone would not establish a benefit from second order.

The optional full Hessian uses exact automatic differentiation, one basis HVP per column, and records symmetry error. Sketches reuse the existing randomized HVP implementation without constructing the full Hessian. Each construction is measured independently. Common gradient cost is allocated to each method in `method_costs.csv`; raw ledger stages count it once.

With the full Hessian enabled, the signed held-out error decomposes exactly as: **full Taylor minus exact train effect + sketch minus full Taylor + exact train minus exact validation effect**. The last term measures prompt shift. Compression cannot be isolated empirically when the full reference is disabled.''')

code('''n_contexts=len(states)*len(ANCHORS)
estimated_hvps=n_contexts*(sum(RANKS)+(16 if RUN_FULL_HESSIAN else 0))
derivative_batches=(estimated_hvps+n_contexts)*2*((len(effective['train'])+BATCH_SIZE-1)//BATCH_SIZE)
print('Derivative estimate:',n_contexts,'contexts,',estimated_hvps,'HVPs,',derivative_batches,
      'forward batches plus backward work. Full Hessian:',RUN_FULL_HESSIAN)
with ledger.instrument(model):
    predictions,costs,curvature_checks=predict_methods(experiment,queries,ANCHORS,RANKS,
        RUN_FULL_HESSIAN,SEED,ledger,OUT)
predictions.to_csv(OUT/'predictions.csv',index=False)
curvature_checks.to_csv(OUT/'curvature_diagnostics.csv',index=False)
metrics,sweep,difficult=analyze(predictions,thresholds)
# Exact interventions are an oracle reference: their labels equal the truth by
# definition. Their measured shared train+validation query cost is shown too.
oracle_metrics=[];oracle_sweep=[]
for split in ['train','validation']:
    true=queries[split+'_effect'].to_numpy()
    oracle_metrics.append({'anchor_strength':0.,'method':'exact_intervention','split':split,
        'n_queries':len(queries),'mae':0.,'spearman':1. if len(true)>1 and np.std(true)>0 else np.nan})
    for threshold in thresholds:
        oracle_sweep.append({'anchor_strength':0.,'method':'exact_intervention','split':split,
                            'threshold':threshold,**decision_metrics(true,true,threshold)})
metrics=pd.concat([metrics,pd.DataFrame(oracle_metrics)],ignore_index=True)
sweep=pd.concat([sweep,pd.DataFrame(oracle_sweep)],ignore_index=True)
exact_cost=next(s for s in ledger.stages if s['stage']=='exact_ground_truth')
cost_fields=['wall_seconds','forward_passes_batches','examples_processed','backward_calls','hvp_operations']
costs=pd.concat([costs,pd.DataFrame([{'state':'ALL','anchor_strength':0.,'method':'exact_intervention',
                                   **{k:exact_cost[k] for k in cost_fields}}])],ignore_index=True)
costs.to_csv(OUT/'method_costs.csv',index=False)
decision_summary=sweep.groupby(['anchor_strength','method','split'])[
    ['decision_accuracy','false_prune_rate','false_retain_rate']].mean().reset_index()
metrics=metrics.merge(decision_summary,on=['anchor_strength','method','split'],validate='one_to_one')
total_cost=costs.groupby(['anchor_strength','method'])[cost_fields].sum().reset_index()
metrics=metrics.merge(total_cost,on=['anchor_strength','method'],validate='many_to_one')
metrics.to_csv(OUT/'method_metrics.csv',index=False)
sweep.to_csv(OUT/'threshold_sweep.csv',index=False)
difficult.to_csv(OUT/'difficult_decisions.csv',index=False)
prediction_errors=predictions.assign(abs_taylor=abs(predictions.full_taylor_signed_error_train),
    abs_compression=abs(predictions.compression_signed_error),
    abs_prompt_shift=abs(predictions.prompt_shift_train_minus_validation))
decomposition=prediction_errors.groupby(['anchor_strength','method','removed_edges'])[
    ['abs_taylor','abs_compression','abs_prompt_shift']].mean().reset_index()
decomposition.to_csv(OUT/'error_decomposition.csv',index=False)
print(metrics.to_string(index=False))''')

md('''## Experiment 4 — measured plots, cost, and automatic research checkpoint

Difficult-decision plots use the same retrospective first-order-disagreement rows for every method, rather than letting each method select its own test subset. This subset is descriptive; thresholds were chosen on train effects. The cost plot reports estimator construction time, while the ledger separately includes exact ground truth, donor captures, official audits, numerical checks, backward passes, and HVPs. No forward-equivalent weighting or ACDC acceleration is assumed.''')

code('''plot_results(predictions,metrics,sweep,difficult,costs,OUT)
summary=automatic_summary(metrics,sweep,difficult,costs,SMOKE)
if coverage==0:
    summary+='Threshold coverage: no measured train effects lie in [0.01,0.30]; this edge/state set may not contain difficult pruning decisions in the requested range.\\n'
historical_gap=float(audit.max_abs_historical_vs_official.max())
summary+=f'Intervention audit: maximum historical-adapter discrepancy from official MIB = {historical_gap:.6g}. The diagnostic adapter passed official endpoint checks.\\n'
(OUT/'summary.md').write_text(summary,encoding='utf-8')
print(summary)
for name,digest in old_notebooks.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, f'Previous notebook changed: {name}'
pd.DataFrame(ledger.stages).to_csv(OUT/'runtime_breakdown.csv',index=False)
write_json(OUT/'completion.json',{'smoke':SMOKE,'full_hessian_measured':RUN_FULL_HESSIAN,
    'old_notebooks_unchanged':True,'successful':True,'query_count':len(queries)})
print('Results saved:',OUT)''')

md('''## Final research campaign and wrap-up

The final campaign uses the same shared adapter and diagnostics module, with streamed batch derivatives for 16/32/64/128 nested discovery examples and a fixed 128-prompt validation set. It first gates on a 24-mask official MIB audit, constructs hard decisions from train-16 effects, and completes the 16-example job before larger sizes. The full Hessian is an analysis reference. All context bundles and configurations are checkpointed under `results/mib_sketch_diagnostics/final/`.

Run from the project terminal using the existing environment:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_sketch_diagnostics_final.py --sizes 16 32 64 128
```

Use `--out <existing-run-directory> --sizes 32` to resume an individual size. Additional seeds are selected from training-only criteria recorded before validation analysis. Generate/update the report and plots with `scripts/build_sketch_final_report.py <run-directory>`. The research report preserves historical measurements and explicitly distinguishes earlier head-level regression, historical frozen-sender edge results, corrected final diagnostics, and the independently supplied Stage G–I reports. No MAttr run is launched.''')
code('''# Results dashboard only: this cell never launches another campaign.
final_root=ROOT/'results/mib_sketch_diagnostics/final'
final_runs=sorted(final_root.glob('*')) if final_root.exists() else []
if final_runs:
    FINAL_RESULTS=final_runs[-1]
    print('Final campaign:',FINAL_RESULTS)
    path=FINAL_RESULTS/'prediction_comparison.csv'
    if path.exists():
        print(pd.read_csv(path)[['discovery_examples','seed','anchor_strength','method','mae',
                                'difficult_accuracy','wall_seconds']].to_string(index=False))
    else: print('No completed aggregated results yet; inspect per-size completion markers.')
    report=FINAL_RESULTS/'SKETCH_ACDC_FINAL_REPORT.md'
    print('Report:',report,'exists:',report.exists())
else:
    print('The final campaign has not run. Smoke results above remain explicitly provisional.')''')
book=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
target=root/'notebooks/mib_sketch_diagnostics.ipynb'
nbf.write(book,target)
print(target)
