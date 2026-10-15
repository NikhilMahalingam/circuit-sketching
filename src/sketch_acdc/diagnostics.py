"""Controlled Taylor/compression diagnostics using official MIB edge semantics.

Unlike the historical frozen-difference adapter, current sender activations are
captured during the intervened pass. No historical results/notebooks are edited.
"""
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch

from .core import (EdgeExperiment, EdgeSpec, FrozenSenderEdgeExperiment,
                   hessian_vector_product, nystrom_sketch, state_key, receiver_patch,
                   nystrom_from_products)


def load_edges(path):
    rows = pd.read_csv(path).to_dict('records')
    return [EdgeSpec(name=r['name'], sender_index=int(r['sender_index']),
                     receiver_index=int(r['receiver_index']), hook=r['hook'],
                     head=None if pd.isna(r['head']) else int(r['head']),
                     layer=int(r['layer']), eap_score=float(r['eap_score'])) for r in rows]


def load_states(path, n_edges=16, removed_targets=(1, 3, 6, 9)):
    trajectory = pd.read_csv(path, dtype={'state_before': str})
    strings = list(dict.fromkeys(trajectory.state_before))
    if any(len(s) != n_edges or set(s)-{'0', '1'} for s in strings):
        raise ValueError('Trajectory state length/encoding does not match selected edges.')
    available = [s for s in strings if 0 < s.count('1') < n_edges]
    if not available:
        raise ValueError('No partially pruned states in the saved trajectory.')
    selected = dict.fromkeys(min(available, key=lambda s: abs(s.count('1')-target))
                             for target in removed_targets)
    return [list(map(int, s)) for s in selected]


class DiagnosticEdgeExperiment(EdgeExperiment):
    """Shared corrected edge API with persistent exact-score caching."""
    VERSION = EdgeExperiment.VERSION

    def __init__(self, *args, cache_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_id = hashlib.sha256((self.cache_id+self.VERSION).encode()).hexdigest()
        self.cache_dir = Path(cache_dir) / self.cache_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_state(self, state, split, fresh=False):
        key = state_key(state, self.cache_id, split)
        digest = hashlib.sha256(repr(key).encode()).hexdigest()
        path = self.cache_dir / (digest+'.npz')
        if not fresh and key not in self.cache and path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if saved['identity'].item() != repr(key):
                    raise ValueError('Exact intervention cache identity mismatch.')
                self.cache[key] = saved['score'].copy()
        result = super().evaluate_state(state, split, fresh=fresh)
        if not path.exists() or fresh:
            np.savez_compressed(path, identity=repr(key), score=result)
        return result


def full_hessian(function, point, ledger=None):
    """Exact autodiff Hessian columns; bounded memory, 16 HVPs for 16 edges."""
    columns = []
    for i in range(point.numel()):
        basis = torch.zeros_like(point); basis[i] = 1
        columns.append(hessian_vector_product(function, point, basis).detach())
        if ledger:
            ledger.backward_calls += 2
            ledger.hvp_calls += 1
    raw = torch.stack(columns, dim=1)
    asymmetry = float(torch.linalg.norm(raw-raw.T)/(torch.linalg.norm(raw)+1e-12))
    return (raw+raw.T)/2, asymmetry


def taylor_effect(gradient, hessian, anchor, state, candidate):
    """Difference of two Taylor polynomials, valid for any shared anchor."""
    before = np.asarray(state, dtype=float) - anchor
    after = before.copy(); after[candidate] += 1
    result = float(gradient[candidate])
    if hessian is not None:
        result += float(0.5*(after @ hessian @ after - before @ hessian @ before))
    return result


def threshold_grid(training_effects, lower=0.01, upper=0.30, step=0.01):
    """Observed-effect thresholds use training labels only, before validation."""
    base = list(np.arange(lower, upper+step/2, step))
    observed = np.abs(np.asarray(training_effects, dtype=float))
    near = observed[(observed >= lower) & (observed <= upper)]
    for value in near:
        margin = max(1e-5, value*0.01)
        base.extend([value-margin, value, value+margin])
    return np.unique(np.round([x for x in base if lower <= x <= upper], 8))


def exact_queries(experiment, states, candidate_limit=None):
    from tqdm import tqdm
    rows, per_example = [], []
    for state in states:
        bits = ''.join(map(str, state))
        candidates = [i for i, bit in enumerate(state) if not bit]
        if candidate_limit is not None:
            candidates = candidates[:candidate_limit]
        for i in tqdm(candidates,desc=f'exact removed={sum(state)}',leave=False):
            query_id = f'{bits}:{i}'
            row = {'query_id':query_id, 'state':bits, 'removed_edges':sum(state),
                   'candidate_index':i, 'edge':experiment.edges[i].name}
            for split in ('train', 'validation'):
                effects = experiment.exact_candidate_effect(state, i, split)
                row[split+'_effect'] = float(effects.mean())
                row[split+'_effect_se'] = float(effects.std(ddof=1)/np.sqrt(len(effects))) if len(effects)>1 else np.nan
                for j, effect in enumerate(effects):
                    per_example.append({'query_id':query_id,'split':split,'example_index':j,'effect':effect})
            rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(per_example)


def _cost(ledger, gradient_row, curvature_row=None):
    keys = ['wall_seconds','forward_passes_batches','examples_processed','backward_calls','hvp_operations']
    return {key:gradient_row[key]+(curvature_row[key] if curvature_row else 0) for key in keys}


def predict_methods(experiment, queries, anchors, ranks, include_full, seed, ledger, out):
    """Same query endpoints and train prompts for all methods; separate cost stages."""
    predictions, costs, diagnostics = [], [], []
    for state_bits, group in queries.groupby('state', sort=False):
        state = np.array(list(map(int,state_bits)), dtype=float)
        for strength in anchors:
            label = f'{state_bits}_anchor{strength:g}'
            anchor = np.where(state == 1, 1., strength)
            point = torch.tensor(anchor, device=experiment.model.cfg.device, dtype=experiment.model.cfg.dtype)
            function = lambda z: experiment.continuous(z, 'train').mean()
            print(f'Derivatives {label}: {len(group)} identical queries; ranks={ranks}; full={include_full}',flush=True)
            with ledger.stage('gradient_'+label):
                variable = point.detach().requires_grad_(True)
                value = function(variable)
                gradient = torch.autograd.grad(value, variable)[0].detach().cpu().numpy()
                ledger.backward_calls += 1
                del value, variable
            gradient_cost = ledger.stages[-1]
            matrices = {'first_order':None}
            costs.append({'state':state_bits,'anchor_strength':strength,'method':'first_order',
                          **_cost(ledger,gradient_cost)})
            full = None
            if include_full:
                with ledger.stage('full_hessian_'+label):
                    full_tensor, asymmetry = full_hessian(function, point, ledger)
                full = full_tensor.detach().cpu().numpy()
                del full_tensor
                matrices['full_second_order'] = full
                costs.append({'state':state_bits,'anchor_strength':strength,'method':'full_second_order',
                              **_cost(ledger,gradient_cost,ledger.stages[-1])})
                diagnostics.append({'state':state_bits,'anchor_strength':strength,'method':'full_second_order',
                                    'hessian_relative_asymmetry':asymmetry})
            for rank in ranks:
                with ledger.stage(f'sketch{rank}_'+label):
                    approx, omega, y = nystrom_sketch(function, point, rank, seed)
                    ledger.backward_calls += 2*rank; ledger.hvp_calls += rank
                method = f'sketch_rank_{rank}'
                matrix = approx.detach().cpu().numpy()
                matrices[method] = matrix
                core = (omega.T@y).detach().cpu().numpy()
                diagnostics.append({'state':state_bits,'anchor_strength':strength,'method':method,
                                    'core_condition_number':float(np.linalg.cond(core)),
                                    'hessian_relative_compression_error':float(np.linalg.norm(matrix-full)/(np.linalg.norm(full)+1e-12)) if full is not None else np.nan})
                costs.append({'state':state_bits,'anchor_strength':strength,'method':method,
                              **_cost(ledger,gradient_cost,ledger.stages[-1])})
                del approx, omega, y
            np.savez_compressed(Path(out)/f'derivatives_{label}.npz', anchor=anchor, gradient=gradient,
                                **{key:matrix for key,matrix in matrices.items() if matrix is not None})
            for query in group.to_dict('records'):
                i = query['candidate_index']
                full_prediction = taylor_effect(gradient,full,anchor,state,i) if full is not None else np.nan
                for method,matrix in matrices.items():
                    prediction = taylor_effect(gradient,matrix,anchor,state,i)
                    predictions.append({**query,'anchor_strength':strength,'method':method,'prediction':prediction,
                        'distance_before_from_anchor':float(np.linalg.norm(state-anchor)),
                        'distance_after_from_anchor':float(np.linalg.norm(state+np.eye(len(state))[i]-anchor)),
                        'train_signed_error':prediction-query['train_effect'],
                        'validation_signed_error':prediction-query['validation_effect'],
                        'full_taylor_signed_error_train':full_prediction-query['train_effect'],
                        'compression_signed_error':prediction-full_prediction if method.startswith('sketch_') else np.nan,
                        'prompt_shift_train_minus_validation':query['train_effect']-query['validation_effect']})
    return pd.DataFrame(predictions), pd.DataFrame(costs), pd.DataFrame(diagnostics)


def decision_metrics(true, predicted, threshold):
    exact = np.abs(np.asarray(true)) < threshold
    approximate = np.abs(np.asarray(predicted)) < threshold
    false_prune = approximate & ~exact
    false_retain = ~approximate & exact
    return {'decision_accuracy':float(np.mean(approximate==exact)),
            'false_prune_rate':float(false_prune.mean()), 'false_retain_rate':float(false_retain.mean()),
            'false_prune_conditional':float(false_prune.sum()/(~exact).sum()) if (~exact).sum() else np.nan,
            'false_retain_conditional':float(false_retain.sum()/exact.sum()) if exact.sum() else np.nan,
            'n_queries':len(exact),'false_prune_count':int(false_prune.sum()),'false_retain_count':int(false_retain.sum())}


def analyze(predictions, thresholds):
    from scipy.stats import spearmanr
    metrics, sweep, difficult = [], [], []
    first = predictions[(predictions.method=='first_order') & (predictions.anchor_strength==0)]
    if first.query_id.duplicated().any():
        raise ValueError('Duplicate first-order query IDs.')
    for (anchor,method), group in predictions.groupby(['anchor_strength','method'],sort=False):
        group = group.set_index('query_id').loc[first.query_id].reset_index()
        for split in ('train','validation'):
            truth = group[split+'_effect'].to_numpy()
            pred = group.prediction.to_numpy()
            baseline = first.set_index('query_id').loc[group.query_id].prediction.to_numpy()
            correlation = float(spearmanr(truth,pred).statistic) if len(group)>1 and np.std(truth)>0 and np.std(pred)>0 else np.nan
            metrics.append({'anchor_strength':anchor,'method':method,'split':split,'n_queries':len(group),
                            'mae':float(np.mean(abs(pred-truth))),'spearman':correlation})
            for threshold in thresholds:
                mismatch = (abs(baseline)<threshold)!=(abs(truth)<threshold)
                sweep.append({'anchor_strength':anchor,'method':method,'split':split,'threshold':threshold,
                              **decision_metrics(truth,pred,threshold)})
                for index in np.where(mismatch)[0]:
                    difficult.append({'query_id':group.iloc[index].query_id,'anchor_strength':anchor,
                                      'method':method,'split':split,'threshold':threshold,
                                      'first_order_prediction':baseline[index],'prediction':pred[index],
                                      'true_effect':truth[index],'absolute_error':abs(pred[index]-truth[index]),
                                      'correct_decision':bool((abs(pred[index])<threshold)==(abs(truth[index])<threshold))})
    difficult_columns=['query_id','anchor_strength','method','split','threshold','first_order_prediction',
                       'prediction','true_effect','absolute_error','correct_decision']
    return pd.DataFrame(metrics), pd.DataFrame(sweep), pd.DataFrame(difficult,columns=difficult_columns)


def automatic_summary(metrics, sweep, difficult, costs, smoke):
    prefix = 'LIGHTWEIGHT SMOKE TEST — not a scientific conclusion.\n' if smoke else ''
    val = metrics[(metrics.split=='validation') & (metrics.anchor_strength==0)].set_index('method')
    statements = []
    if 'full_second_order' in val.index:
        first,second = val.loc['first_order','mae'],val.loc['full_second_order','mae']
        statements.append(f'1. Full second order held-out MAE {second:.6g}; first order {first:.6g}. Lower: '+('full second order.' if second<first else 'first order (or tied).'))
    else:
        statements.append('1. Full Hessian was disabled; full second order versus first order is unmeasured.')
    ranks = [(int(key.rsplit('_',1)[1]),float(row.mae)) for key,row in val.iterrows() if key.startswith('sketch_rank_')]
    ranks.sort()
    if len(ranks)>1:
        monotone=all(b[1]<=a[1] for a,b in zip(ranks,ranks[1:]))
        statements.append(f'2. Held-out MAE by sketch rank: {ranks}; monotonic improvement: {monotone}.')
    else:
        statements.append(f'2. Measured rank/MAE: {ranks}; a rank trend requires at least two ranks.')
    if difficult.empty or not (difficult.split=='validation').any():
        statements.append('3. No first-order disagreements on the measured held-out threshold sweep; difficult-decision benefit is unresolved.')
    else:
        selected=difficult[(difficult.split=='validation') & (difficult.anchor_strength==0)]
        rates=selected.groupby('method').correct_decision.mean().to_dict()
        statements.append(f'3. Decision accuracy on the retrospective common first-order-disagreement set: {rates}.')
    timing=costs[costs.anchor_strength==0].groupby('method').wall_seconds.sum().to_dict()
    approximations=val.drop(index='exact_intervention',errors='ignore')
    best=approximations.mae.idxmin()
    baseline=float(val.loc['first_order','mae'])
    gain=(baseline-float(val.loc[best,'mae']))/baseline if baseline else 0.
    ratio=timing[best]/timing['first_order'] if timing['first_order'] else np.nan
    statements.append(f'4. Lowest measured held-out MAE: {best}; improvement over first order {gain:.1%}, construction time ratio {ratio:.2f}x. Construction seconds (shared gradient allocated per method): {timing}. This tradeoff does not establish end-to-end pruning savings.')
    return prefix+'\n'.join(statements)+'\n'


def plot_results(predictions, metrics, sweep, difficult, costs, out):
    import matplotlib.pyplot as plt
    out=Path(out)
    val=metrics[(metrics.split=='validation') & (metrics.anchor_strength==0)]
    fig,ax=plt.subplots(figsize=(6,4))
    ranked=val[val.method.str.startswith('sketch_rank_')].copy()
    ranked['rank']=ranked.method.str.rsplit('_',n=1).str[-1].astype(int)
    ranked=ranked.sort_values('rank')
    ax.plot(ranked['rank'],ranked.mae,'o-',label='sketch')
    for color,method in [('C1','first_order'),('C2','full_second_order')]:
        row=val[val.method==method]
        if len(row): ax.axhline(row.mae.iloc[0],linestyle='--',color=color,label=method)
    ax.set_xticks(ranked['rank'])
    ax.set(xlabel='sketch rank',ylabel='held-out effect MAE');ax.legend();fig.tight_layout()
    fig.savefig(out/'mae_vs_rank.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for method,group in sweep[(sweep.split=='validation') & (sweep.anchor_strength==0)].groupby('method'):
        ax.plot(group.threshold,group.decision_accuracy,label=method)
    ax.set(xlabel='pruning threshold',ylabel='held-out decision accuracy',ylim=(0,1.05));ax.legend();fig.tight_layout()
    fig.savefig(out/'accuracy_vs_threshold.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    if not difficult.empty and (difficult.split=='validation').any():
        group=difficult[(difficult.split=='validation') & (difficult.anchor_strength==0)].groupby('method').absolute_error.mean()
        ax.bar(group.index,group.values);ax.tick_params(axis='x',rotation=30)
    else:
        ax.text(.5,.5,'No first-order disagreements in measured sweep',ha='center',va='center',transform=ax.transAxes)
    ax.set(ylabel='MAE on first-order-disagreement query/threshold rows');fig.tight_layout()
    fig.savefig(out/'difficult_decision_errors.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    accuracy=sweep[(sweep.split=='validation') & (sweep.anchor_strength==0)].groupby('method').decision_accuracy.mean()
    seconds=costs[costs.anchor_strength==0].groupby('method').wall_seconds.sum()
    for method in accuracy.index:
        ax.scatter(seconds[method],accuracy[method],label=method)
    ax.set(xlabel='measured construction / exact-query seconds',ylabel='mean accuracy over threshold sweep',ylim=(0,1.05))
    ax.legend(loc='lower right',fontsize=8);fig.tight_layout()
    fig.savefig(out/'accuracy_vs_cost.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for (anchor,method),group in predictions.groupby(['anchor_strength','method']):
        error=group.assign(error=abs(group.validation_signed_error)).groupby('removed_edges').error.mean()
        ax.plot(error.index,error.values,'o-',label=f'{method}, anchor={anchor:g}')
    ax.set(xlabel='already removed edges',ylabel='held-out effect MAE');ax.legend(fontsize=7);fig.tight_layout()
    fig.savefig(out/'error_vs_pruned_edges.png',dpi=150);plt.close(fig)


def streaming_curvature_bundle(experiment,state,strength,ranks,seed,ledger,out,include_full=True):
    """Batch-streamed, direct-autodiff reference and nested randomized HVPs.

    Each batch's gradient graph is reused across directions, then freed. Full
    Hessian columns and sketch HVPs are measured separately; sketches are NEVER
    computed from the reference Hessian. Shared graph preparation is allocated
    to each standalone estimator; raw ledger totals count it only once.
    """
    from tqdm import tqdm
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    bits=''.join(map(str,state));label=f'{bits}_a{strength:g}'
    saved=out/f'bundle_{label}.npz';cost_path=out/f'cost_{label}.csv'
    if saved.exists() and cost_path.exists():
        with np.load(saved,allow_pickle=False) as data:
            bundle={k:data[k].copy() for k in data.files}
        return bundle,pd.read_csv(cost_path)
    n=len(state);dataset=experiment.splits['train'];count=len(dataset)
    anchor=np.where(np.array(state)==1,1.,strength)
    base=torch.tensor(anchor,device=experiment.model.cfg.device,dtype=experiment.model.cfg.dtype)
    def batches():
        return dataset.to_dataloader(experiment.batch_size)
    gradient=np.zeros(n,dtype=np.float64)
    with ledger.stage('first_order_'+label,estimate=f'{count} prompts; {len(batches())} batches, one backward per batch'):
        for clean,corrupt,labels in tqdm(batches(),desc='gradient '+label,leave=False):
            z=base.detach().clone().requires_grad_(True)
            value=experiment._batch_score(clean,corrupt,labels,z).sum()/count
            gradient+=torch.autograd.grad(value,z)[0].detach().double().cpu().numpy()
            ledger.backward_calls+=1
            del value,z
    first_cost=ledger.stages[-1]
    gen=torch.Generator().manual_seed(seed)
    omega=torch.linalg.qr((torch.randint(0,2,(n,max(ranks)),generator=gen).double()*2-1)).Q
    directions=[]
    if include_full:directions.extend(torch.eye(n,dtype=torch.float64).T)
    directions.extend(omega.T)
    vectors=torch.stack(directions).to(base.device,base.dtype)
    products=np.zeros((n,len(directions)),dtype=np.float64)
    directional_seconds=np.zeros(len(directions));prep_seconds=0.
    with ledger.stage('curvature_bundle_'+label,estimate=f'{len(batches())} batches x {len(directions)} direct HVPs; one retained batch graph'):
        for clean,corrupt,labels in tqdm(batches(),desc='curvature '+label,leave=False):
            ledger.sync();start=time.perf_counter()
            z=base.detach().clone().requires_grad_(True)
            value=experiment._batch_score(clean,corrupt,labels,z).sum()/count
            g=torch.autograd.grad(value,z,create_graph=True)[0]
            ledger.backward_calls+=1
            ledger.sync();prep_seconds+=time.perf_counter()-start
            for j,vector in enumerate(vectors):
                ledger.sync();start=time.perf_counter()
                hv=torch.autograd.grad(g,z,grad_outputs=vector,retain_graph=j+1<len(directions))[0]
                products[:,j]+=hv.detach().double().cpu().numpy()
                ledger.backward_calls+=1;ledger.hvp_calls+=1
                ledger.sync();directional_seconds[j]+=time.perf_counter()-start
            del value,g,z,hv
    curvature_cost=ledger.stages[-1]
    # Include all batch iteration/tokenization/copy overhead in graph preparation.
    prep_seconds+=max(0.,curvature_cost['wall_seconds']-prep_seconds-directional_seconds.sum())
    bundle={'anchor':anchor,'gradient':gradient,'omega':omega.numpy()}
    rows=[{'state':bits,'anchor_strength':strength,'method':'first_order',
           **{k:first_cost[k] for k in ['wall_seconds','forward_passes_batches','examples_processed',
                                       'backward_calls','hvp_operations','peak_cuda_allocated_mib']},
           'cost_scope':'independently measured first-order estimator'}]
    offset=n if include_full else 0
    if include_full:
        raw=products[:,:n]
        bundle['full_second_order']=(raw+raw.T)/2
        bundle['hessian_relative_asymmetry']=np.array(np.linalg.norm(raw-raw.T)/(np.linalg.norm(raw)+1e-12))
    bundle['hvp_products']=products[:,offset:]
    methods=([('full_second_order',n)] if include_full else [])+[(f'sketch_rank_{r}',r) for r in ranks]
    for method,rank in methods:
        start=time.perf_counter()
        if method=='full_second_order':seconds=directional_seconds[:n].sum()
        else:
            y=torch.tensor(products[:,offset:offset+rank],dtype=torch.float64)
            bundle[method]=nystrom_from_products(omega[:,:rank],y).numpy()
            seconds=directional_seconds[offset:offset+rank].sum()
        post_seconds=time.perf_counter()-start
        rows.append({'state':bits,'anchor_strength':strength,'method':method,
                     'wall_seconds':prep_seconds+seconds+post_seconds,
                     'forward_passes_batches':curvature_cost['forward_passes_batches'],
                     'examples_processed':curvature_cost['examples_processed'],
                     'backward_calls':len(batches())*(1+rank),'hvp_operations':len(batches())*rank,
                     'peak_cuda_allocated_mib':curvature_cost['peak_cuda_allocated_mib'],
                     'cost_scope':'measured standalone allocation; shared graph and nested HVP prefixes; peak is shared upper bound'})
    costs=pd.DataFrame(rows)
    # The cost file is written last and serves as the completion marker; a
    # partially written bundle alone is never reused on restart.
    np.savez_compressed(saved,**bundle)
    costs.to_csv(cost_path,index=False)
    return bundle,costs


def final_query_metrics(predictions,query_effects,decision_pairs,bootstrap_seed=42,bootstraps=200):
    """Fixed train-defined hard pairs and paired validation-prompt bootstrap."""
    from scipy.stats import spearmanr
    metrics=[]
    rng=np.random.default_rng(bootstrap_seed)
    qids=list(query_effects)
    matrix=np.stack([query_effects[q] for q in qids])
    sampled=matrix[:,rng.integers(0,matrix.shape[1],(bootstraps,matrix.shape[1]))].mean(axis=2)
    index={q:i for i,q in enumerate(qids)}
    for (strength,method),group in predictions.groupby(['anchor_strength','method'],sort=False):
        group=group.set_index('query_id').loc[qids]
        pred=group.prediction.to_numpy();true=group.validation_effect.to_numpy()
        base=predictions[(predictions.method=='first_order')&(predictions.anchor_strength==strength)].set_index('query_id').loc[qids].prediction.to_numpy()
        pairs=decision_pairs.copy()
        qi=np.array([index[q] for q in pairs.query_id]);tau=pairs.threshold.to_numpy()
        truth_decision=abs(true[qi])<tau;pred_decision=abs(pred[qi])<tau
        hard=pairs.difficult.to_numpy(dtype=bool)
        hard_count=int(hard.sum())
        train_true=group.train_effect.to_numpy()
        train_hard_accuracy=float(((abs(pred[qi[hard]])<tau[hard])==(abs(train_true[qi[hard]])<tau[hard])).mean()) if hard_count else np.nan
        train_base_accuracy=float(((abs(base[qi[hard]])<tau[hard])==(abs(train_true[qi[hard]])<tau[hard])).mean()) if hard_count else np.nan
        acc=float((truth_decision==pred_decision).mean())
        hard_acc=float((truth_decision[hard]==pred_decision[hard]).mean()) if hard_count else np.nan
        boot_mae=abs(pred[:,None]-sampled).mean(axis=0)
        boot_base=abs(base[:,None]-sampled).mean(axis=0)
        boot_hard=((abs(pred[qi[hard],None])<tau[hard,None])==(abs(sampled[qi[hard]])<tau[hard,None])).mean(axis=0) if hard_count else np.full(bootstraps,np.nan)
        base_hard=((abs(base[qi[hard],None])<tau[hard,None])==(abs(sampled[qi[hard]])<tau[hard,None])).mean(axis=0) if hard_count else np.full(bootstraps,np.nan)
        interval=lambda a:np.quantile(a,[.025,.975]) if np.isfinite(a).all() else [np.nan,np.nan]
        mae_ci=interval(boot_mae);gain_ci=interval(boot_base-boot_mae);acc_ci=interval(boot_hard);acc_gain_ci=interval(boot_hard-base_hard)
        metrics.append({'anchor_strength':strength,'method':method,'n_queries':len(group),
            'mae':float(abs(pred-true).mean()),'mae_ci_low':mae_ci[0],'mae_ci_high':mae_ci[1],
            'mae_gain_over_first_ci_low':gain_ci[0],'mae_gain_over_first_ci_high':gain_ci[1],
            'spearman':float(spearmanr(pred,true).statistic),'decision_accuracy':acc,
            'false_prune_rate':float((pred_decision & ~truth_decision).mean()),
            'false_retain_rate':float((~pred_decision & truth_decision).mean()),
            'difficult_decision_count':hard_count,'difficult_unique_queries':int(pairs.loc[hard,'query_id'].nunique()),
            'difficult_accuracy':hard_acc,'difficult_accuracy_ci_low':acc_ci[0],'difficult_accuracy_ci_high':acc_ci[1],
            'difficult_accuracy_gain_ci_low':acc_gain_ci[0],'difficult_accuracy_gain_ci_high':acc_gain_ci[1],
            'train_difficult_accuracy':train_hard_accuracy,'train_difficult_accuracy_gain_over_first':train_hard_accuracy-train_base_accuracy,
            'train_mae':float(abs(group.prediction-group.train_effect).mean())})
    return pd.DataFrame(metrics)
