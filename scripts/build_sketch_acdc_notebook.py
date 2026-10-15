"""Generate the staged Sketch-ACDC notebook from reviewable source cells."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []
def md(source): cells.append(nbf.v4.new_markdown_cell(source))
def code(source): cells.append(nbf.v4.new_code_cell(source))

md('''# Sketch-Assisted ACDC on official MIB IOI

**Stage A is the default execution path.** Stages B and C require explicit flags after the numerical checks pass. The 16 selected items are EAP-ranked sender-to-receiver **edges** in MIB's complete GPT-2 Small graph. Results here describe restricted sequential pruning, not a full ACDC circuit or an official MIB leaderboard score.

MIB graph/patching: `external/MIB-circuit-track/EAP-IG/src/eap/{graph,utils,evaluate,attribute}.py` at the commit recorded below. Original ACDC traversal/threshold: `external/Automatic-Circuit-Discovery/acdc/TLACDCExperiment.py`, `TLACDCCorrespondence.py`. MIB exact attribution toggles each edge independently, whereas ACDC updates its current graph after each accepted removal. This notebook implements that sequential behavior on a fixed candidate set. No upstream source is modified.''')

code('''from pathlib import Path
import os, sys, json, time, subprocess, importlib.metadata
from datetime import datetime, timezone
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
ROOT = Path.cwd().resolve()
if not (ROOT / 'mib_ioi/runtime.py').exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
from mib_ioi.runtime import configure_repository, seed_everything, load_model, load_splits, Ledger, write_json, subset
from src.sketch_acdc.core import EdgeExperiment, select_edges, exact_acdc, hessian_vector_product, nystrom_sketch
SEED = 42
PILOT = os.environ.get('SKETCH_ACDC_PROFILE') == 'pilot'
TRAIN_N, VALIDATION_N = (16, 8) if PILOT else (32, 64)
BATCH_SIZE, EAP_BATCH_SIZE = 2, 2
EDGE_COUNT = 16
SMOKE_COUNT = TRAIN_N
RUN_STAGE_B = os.environ.get('SKETCH_ACDC_RUN_B') == '1'
RUN_STAGE_C = os.environ.get('SKETCH_ACDC_RUN_C') == '1'
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
OUT = ROOT / 'results/mib_sketch_acdc' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT.mkdir(parents=True, exist_ok=True)
seed_everything(SEED)
print('Results:', OUT, 'batch size:', BATCH_SIZE, 'pilot:', PILOT, 'stage B/C:', RUN_STAGE_B, RUN_STAGE_C)''')

md('''## Stage A — official data, edge rankings, intervention and derivative checks

The IOI score is MIB's per-example correct-minus-incorrect logit difference. A removed edge adds its corrupted-minus-clean sender output to precisely its receiver input. The continuous coefficient multiplies that same update; its zero and one endpoints are checked against MIB's independent graph evaluator. All model parameters remain frozen after EAP attribution.''')
code('''repo, provenance = configure_repository(ROOT)
from eap.graph import Graph
from eap.attribute import get_scores_eap
from MIB_circuit_track.metrics import get_metric
acdc_dir = ROOT / 'external/Automatic-Circuit-Discovery'
acdc_commit = subprocess.check_output(['git','-C',str(acdc_dir),'rev-parse','HEAD'], text=True).strip()
model = load_model('cuda')
datasets, manifest, split_info = load_splits(model, {'train':TRAIN_N, 'validation':VALIDATION_N}, SEED)
ledger = Ledger('cuda', OUT)
manifest.to_csv(OUT/'example_manifest.csv', index=False)
print('MIB:', provenance['mib_commit'], 'EAP:', provenance['eap_commit'], 'ACDC:', acdc_commit)
print('Splits:', {k:len(v) for k,v in datasets.items()}, 'full split overlap:', split_info['full_filtered_split_clean_prompt_overlap'])
print(manifest[['split','clean_prompt','corrupted_prompt','correct_answer']].head(3).to_string(index=False))''')

code('''graph = Graph.from_model(model)
graph.reset(empty=False)
def metric_loss(logits, clean_logits, lengths, labels):
    value = get_metric('logit_diff','ioi',model.tokenizer,model)(logits, clean_logits, lengths, labels, mean=True, loss=True)
    def count_backward(gradient):
        ledger.backward_calls += 1
        return gradient
    value.register_hook(count_backward)
    return value
print('Official EAP ranking:', TRAIN_N, 'train examples,', (TRAIN_N+EAP_BATCH_SIZE-1)//EAP_BATCH_SIZE, 'batches')
with ledger.instrument(model), ledger.stage('official_edge_eap', estimate='3 forward batches plus one backward per EAP batch'):
    try:
        model.requires_grad_(True)
        with torch.enable_grad():
            scores = get_scores_eap(model, graph, datasets['train'].to_dataloader(EAP_BATCH_SIZE), metric_loss, quiet=True)
    finally:
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
edges = select_edges(graph, scores, EDGE_COUNT)
selected = pd.DataFrame([vars(edge) for edge in edges])
selected.to_csv(OUT/'selected_edges.csv', index=False)
print(selected[['name','eap_score']].to_string(index=False))
identity = json.dumps({'model':'gpt2','task':'ioi','mib':provenance['mib_commit'],'eap':provenance['eap_commit'], 'seed':SEED, 'intervention':'patching'}, sort_keys=True)
experiment_splits = dict(datasets)
# First 16 training prompts estimate derivatives; the other 16 calibrate residuals.
experiment_splits['calibration'] = subset(datasets['train'], range(TRAIN_N//2,TRAIN_N))
experiment_splits['smoke'] = subset(datasets['train'], range(SMOKE_COUNT))
experiment = EdgeExperiment(model, graph, edges, experiment_splits, BATCH_SIZE, identity, ledger)
config = {'model':'gpt2','task':'ioi','dataset':'official MIB IOI','splits':split_info,
          'seed':SEED,'train_examples':TRAIN_N,'validation_examples':VALIDATION_N,
          'batch_size':BATCH_SIZE,'eap_batch_size':EAP_BATCH_SIZE,'selected_edges':selected.to_dict('records'),
          'intervention':'MIB corrupted-minus-clean sender activation at receiver input',
          'mib_provenance':provenance,'acdc_commit':acdc_commit,
          'profile':'pilot' if PILOT else 'full','run_stage_b':RUN_STAGE_B,'run_stage_c':RUN_STAGE_C,
          'sketch_ranks':[2,4] if PILOT else [2,4,8,16],
          'packages':{p:importlib.metadata.version(p) for p in ['torch','transformer-lens','datasets','numpy','pandas']}}
write_json(OUT/'experiment_config.json',config)''')

code('''# Compare exact graph patching and differentiable interpolation endpoints on a tiny official subset.
from mib_ioi.runtime import subset
from eap.evaluate import evaluate_baseline
from functools import partial
from numpy.testing import assert_allclose
all_zero = [0]*EDGE_COUNT
single = all_zero.copy(); single[0] = 1
double = single.copy(); double[1] = 1
small_ds = experiment_splits['smoke']
with ledger.instrument(model), ledger.stage('stage_a_edge_equivalence', estimate=f'{SMOKE_COUNT} examples; baseline, one edge, two edges, official comparisons'):
    base = experiment.evaluate_state(all_zero,'smoke',fresh=True)
    official_base = evaluate_baseline(model,small_ds.to_dataloader(BATCH_SIZE),partial(experiment.metric,mean=False),quiet=True).numpy()
    assert_allclose(base,official_base,atol=3e-4,rtol=1e-4)
    for state in [single,double]:
        actual = experiment.evaluate_state(state,'smoke',fresh=True)
        official = experiment.official_binary_score(state,'smoke')
        assert_allclose(actual,official,atol=5e-4,rtol=1e-4)
        assert_allclose(actual,experiment.evaluate_state(state,'smoke',fresh=True),atol=3e-4,rtol=1e-4)
        assert_allclose(actual,experiment.evaluate_state(state,'smoke'),atol=0,rtol=0)
    assert_allclose(base,experiment.evaluate_state(all_zero,'smoke',fresh=True),atol=3e-4,rtol=1e-4)
    print('Endpoint and restoration checks passed. Baseline score:',base.mean(), 'one-edge score:',experiment.evaluate_state(single,'smoke').mean())''')

code('''# Synthetic HVP reference and a two-edge finite-difference check on real IOI examples.
H = torch.tensor([[2.,0.3],[0.3,-1.]],device='cuda',dtype=torch.float64)
q = lambda z: 0.5*z @ H @ z
point = torch.tensor([0.2,0.4],device='cuda',dtype=torch.float64)
vector = torch.tensor([1.,-0.7],device='cuda',dtype=torch.float64)
torch.testing.assert_close(hessian_vector_product(q,point,vector),H@vector,atol=1e-10,rtol=1e-10)
with ledger.instrument(model), ledger.stage('stage_a_hvp', estimate=f'2-edge directional HVP on {SMOKE_COUNT} official train examples'):
    z0 = torch.zeros(EDGE_COUNT,device='cuda',dtype=model.cfg.dtype)
    z0[:2] = 0.35
    direction = torch.zeros_like(z0); direction[:2] = torch.tensor([1.,-0.5],device='cuda')
    f = lambda z: experiment.continuous(z,'smoke').mean()
    hv = hessian_vector_product(f,z0,direction)
    eps = 0.01
    gradient = lambda z: torch.autograd.grad(f(z.requires_grad_(True)),z)[0].detach()
    finite = (gradient((z0+eps*direction).clone())-gradient((z0-eps*direction).clone()))/(2*eps)
    assert torch.isfinite(hv).all() and torch.isfinite(finite).all()
    assert_allclose(hv[:2].detach().cpu(),finite[:2].cpu(),atol=0.04,rtol=0.25)
    ledger.backward_calls += 4
    ledger.hvp_calls += 1
    print('HVP check passed:',hv[:2].detach().cpu().numpy(),'finite difference:',finite[:2].cpu().numpy())''')

code('''# A complete fixed-order restricted ACDC smoke trajectory; validation is not used for decisions.
print('Restricted ACDC smoke:',EDGE_COUNT,'edge trials,',SMOKE_COUNT,'train examples, ~',
      EDGE_COUNT*((SMOKE_COUNT+BATCH_SIZE-1)//BATCH_SIZE)*3,'forward batches before cache hits')
with ledger.instrument(model), ledger.stage('stage_a_exact_acdc', estimate=f'{EDGE_COUNT} edge trials on {SMOKE_COUNT} train examples'):
    final_state, rows = exact_acdc(experiment,'smoke',threshold=0.05)
trajectory = pd.DataFrame(rows)
trajectory.to_csv(OUT/'exact_acdc_trajectory.csv',index=False)
with ledger.instrument(model), ledger.stage('stage_a_final_validation', estimate='Held-out validation baseline and final-state scores'):
    validation_baseline = experiment.evaluate_state(all_zero,'validation').mean()
    validation_score = experiment.evaluate_state(final_state,'validation').mean()
    # The full official MIB evaluator must agree for the terminal binary mask too.
    assert_allclose(experiment.evaluate_state(final_state,'smoke'),
                    experiment.official_binary_score(final_state,'smoke'),atol=5e-4,rtol=1e-4)
print('Pruned:',sum(final_state),'of',EDGE_COUNT,'candidate edges; validation IOI baseline:',
      validation_baseline,'final:',validation_score,'change:',validation_score-validation_baseline)
pd.DataFrame(ledger.stages).to_csv(OUT/'runtime_breakdown.csv',index=False)
write_json(OUT/'stage_a_checks.json',{'edge_endpoint_equivalence':True,'hvp_synthetic_and_finite_difference':True,
    'restricted_acdc_completed':True,'smoke_train_examples':SMOKE_COUNT,
    'validation_baseline_score':float(validation_baseline),
    'validation_score':float(validation_score)})''')

md('''## Stage B — H1 prediction experiment (opt in)

Run only after Stage A passes. At each state from several exact-ACDC trajectories, compare local gradient and randomized Hessian predictions against exact next-edge effects. Keep train examples for estimator construction and validation examples only for held-out assessment. A full rank-16 sketch requires 16 HVPs per expansion state and may cost more than exact edge trials.''')
code('''from src.sketch_acdc.research import collect_h1, rank_summary, calibrate_bound, hybrid_acdc
H1_THRESHOLDS = [0.05] if PILOT else [0.01, 0.05, 0.1]
SKETCH_RANKS = [2, 4] if PILOT else [2, 4, 8, 16]
H1_TRAIN_LIMIT = TRAIN_N//2 if PILOT else 16
if RUN_STAGE_B:
    estimated_states = EDGE_COUNT*len(H1_THRESHOLDS)
    estimated_hvps = estimated_states*sum(SKETCH_RANKS)
    print('Stage B estimate:',estimated_states,'states,',estimated_hvps,'HVP operations,',
          sum((EDGE_COUNT-i) for i in range(EDGE_COUNT))*len(H1_THRESHOLDS),
          'candidate effects on train and validation; this can be expensive on 12 GB.')
    with ledger.instrument(model), ledger.stage('stage_b_h1',estimate=f'{estimated_hvps} HVP operations plus exact candidate queries'):
        h1 = collect_h1(experiment,H1_THRESHOLDS,SKETCH_RANKS,H1_TRAIN_LIMIT,SEED,ledger)
    h1.to_csv(OUT/'edge_prediction_results.csv',index=False)
    ablation = rank_summary(h1)
    ablation.to_csv(OUT/'sketch_rank_ablation.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    subset_h1=h1[h1['rank']==max(SKETCH_RANKS)]
    axes[0].scatter(subset_h1.true_validation_effect,subset_h1.first_order_prediction,s=8,label='first')
    axes[0].scatter(subset_h1.true_validation_effect,subset_h1.second_order_prediction,s=8,label='second')
    lo=min(subset_h1.true_validation_effect.min(),subset_h1.second_order_prediction.min())
    hi=max(subset_h1.true_validation_effect.max(),subset_h1.second_order_prediction.max())
    axes[0].plot([lo,hi],[lo,hi],'k--'); axes[0].set(xlabel='true effect',ylabel='predicted effect');axes[0].legend()
    for method,group in ablation.groupby('method'):
        axes[1].plot(group['rank'],group['mae'],marker='o',label=method)
    axes[1].set(xlabel='sketch rank',ylabel='held-out MAE');axes[1].legend()
    for method,column in [('first','first_order_decision_agrees'),('second','second_order_decision_agrees')]:
        axes[2].plot(subset_h1.groupby('removed_edges')[column].mean(),marker='o',label=method)
    axes[2].set(xlabel='removed edges',ylabel='decision agreement');axes[2].legend()
    fig.tight_layout();fig.savefig(OUT/'h1_predictions.png',dpi=150);plt.close(fig)
    pd.DataFrame(ledger.stages).to_csv(OUT/'runtime_breakdown.csv',index=False)
    print(ablation.to_string(index=False))
else:
    print('Stage B disabled; no H1 results have been measured.')''')
md('''## Stage C — H2 hybrid pruning (opt in)

This stage must follow a measured H1 benefit. Calibrate error intervals on held-out train queries, verify threshold-overlapping decisions exactly, compare refresh schedules and count every forward, backward and HVP. Restricted-graph IOI scores are not CPR/CMD.''')
code('''if RUN_STAGE_C:
    if not RUN_STAGE_B:
        raise RuntimeError('Run measured Stage B first; calibration uses train residuals from that stage.')
    C_THRESHOLD=0.05
    C_RANK=4
    C_REFRESH=[None,2] if PILOT else [None,2,4,'error',1]
    bound=calibrate_bound(h1[h1['rank']==C_RANK],quantile=0.9)
    first_bound=calibrate_bound(h1[h1['rank']==C_RANK],method='first_order',quantile=0.9)
    configurations=[('first_order_only',0,0,None,False),
                    ('first_order_verified',0,first_bound,2,True),
                    ('second_order_only',C_RANK,0,None,False),
                    ('second_order_verified',C_RANK,bound,2,True)]
    configurations += [('second_order_refresh_'+str(r),C_RANK,bound,r,True) for r in C_REFRESH]
    print('Stage C estimate:',len(configurations)*EDGE_COUNT,'decisions; refresh=1 costs about',
          EDGE_COUNT*C_RANK,'HVPs per algorithm plus exact verification and final validation.')
    quality=[]; decisions=[]
    with ledger.instrument(model),ledger.stage('stage_c_hybrid',estimate='Train-only decisions; held-out validation final quality'):
        for method,rank,error_bound,refresh,conservative in configurations:
            experiment.cache.clear()  # Compare cold compute, not another method's cached masks.
            start=time.perf_counter();before=len(ledger.calls);backwards=ledger.backward_calls
            state,detail,queries=hybrid_acdc(experiment,C_THRESHOLD,rank,H1_TRAIN_LIMIT,error_bound,
                                               refresh,conservative,SEED,ledger)
            detail['method']=method;decisions.append(detail)
            score=experiment.evaluate_state(state,'validation').mean()
            quality.append({'method':method,'refresh_after_removals':refresh,'validation_score':score,
                            'deviation_from_baseline':score-experiment.evaluate_state(all_zero,'validation').mean(),
                            'remaining_candidate_edges':EDGE_COUNT-sum(state),'exact_intervention_queries':queries,
                            'forward_batches':len(ledger.calls)-before,
                            'backward_calls':ledger.backward_calls-backwards,
                            'wall_seconds':time.perf_counter()-start})
        start=time.perf_counter();before=len(ledger.calls)
        experiment.cache.clear()
        exact_state,exact_rows=exact_acdc(experiment,'train',C_THRESHOLD)
        score=experiment.evaluate_state(exact_state,'validation').mean()
        quality.append({'method':'exact_acdc','refresh_after_removals':np.nan,'validation_score':score,
                        'deviation_from_baseline':score-experiment.evaluate_state(all_zero,'validation').mean(),
                        'remaining_candidate_edges':EDGE_COUNT-sum(exact_state),
                        'exact_intervention_queries':EDGE_COUNT,'forward_batches':len(ledger.calls)-before,
                        'backward_calls':0,'wall_seconds':time.perf_counter()-start})
    pd.concat(decisions).to_csv(OUT/'hybrid_acdc_results.csv',index=False)
    quality=pd.DataFrame(quality)
    # The H1 data collection was required for rank comparison and empirical
    # bounds. Show its measured cost separately from each C algorithm's loop.
    h1_cost=next(s for s in ledger.stages if s['stage']=='stage_b_h1')
    quality['shared_h1_experiment_seconds']=np.where(quality.method=='exact_acdc',0,h1_cost['wall_seconds'])
    quality['shared_h1_forward_batches']=np.where(quality.method=='exact_acdc',0,h1_cost['forward_passes_batches'])
    quality['wall_seconds_including_h1_experiment']=quality.wall_seconds+quality.shared_h1_experiment_seconds
    quality['forward_batches_including_h1_experiment']=quality.forward_batches+quality.shared_h1_forward_batches
    quality.to_csv(OUT/'circuit_quality.csv',index=False)
    pd.DataFrame(ledger.stages).to_csv(OUT/'runtime_breakdown.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].scatter(quality.wall_seconds_including_h1_experiment,quality.validation_score)
    axes[1].scatter(quality.forward_batches_including_h1_experiment+quality.backward_calls,quality.validation_score)
    fractions=pd.concat(decisions).groupby('method').exact_verification.mean()
    hybrid_quality=quality.set_index('method').loc[fractions.index]
    axes[2].scatter(fractions,hybrid_quality.validation_score)
    axes[0].set(xlabel='wall seconds incl. measured H1 experiment',ylabel='validation IOI score')
    axes[1].set(xlabel='forward + backward calls incl. H1 (unweighted)',ylabel='validation IOI score')
    axes[2].set(xlabel='fraction exactly verified',ylabel='validation IOI score')
    fig.tight_layout();fig.savefig(OUT/'h2_quality_cost.png',dpi=150);plt.close(fig)
    refresh_quality=quality[quality.method.str.startswith('second_order_refresh_')]
    refresh_axis=refresh_quality.refresh_after_removals.map(lambda x:-1 if x=='error' else (0 if pd.isna(x) else x))
    fig,ax=plt.subplots();ax.scatter(refresh_axis,refresh_quality.wall_seconds,c=refresh_quality.validation_score)
    ax.set(xlabel='refresh interval (-1=train error trigger; 0=none/exact)',ylabel='wall seconds')
    fig.tight_layout();fig.savefig(OUT/'h2_refresh.png',dpi=150);plt.close(fig)
    print(quality.to_string(index=False))
else:
    print('Stage C disabled; no acceleration claim is made.')''')

notebook = nbf.v4.new_notebook(cells=cells, metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
target = ROOT/'notebooks/mib_sketch_acdc.ipynb'
target.parent.mkdir(exist_ok=True)
nbf.write(notebook,target)
print(target)
