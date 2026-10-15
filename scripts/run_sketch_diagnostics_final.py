"""Final configurable campaign using the shared diagnostics implementation.

No installs, custom IOI data, or independent patching implementation. Every job
is checkpointed by size/seed/context and validation/query selection is frozen.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('HF_HOME',str(ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_DATASETS_OFFLINE','1');os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'));os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from mib_ioi.runtime import configure_repository,seed_everything,load_model,load_splits,subset,Ledger,write_json
from src.sketch_acdc.core import EdgeExperiment,FrozenSenderEdgeExperiment,exact_acdc,hessian_vector_product
from src.sketch_acdc.diagnostics import (DiagnosticEdgeExperiment,load_edges,load_states,
    streaming_curvature_bundle,taylor_effect,exact_queries,final_query_metrics)

SOURCE=ROOT/'results/mib_sketch_acdc/20261015T054348Z'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def state_string(state):return ''.join(map(str,state))


def restore_ledger(folder):
    folder.mkdir(parents=True,exist_ok=True);ledger=Ledger('cuda',folder)
    if (folder/'compute_accounting.csv').exists():
        ledger.stages=pd.read_csv(folder/'compute_accounting.csv').to_dict('records')
        ledger.calls=pd.read_csv(folder/'forward_log.csv').to_dict('records')
        ledger.backward_calls=int(sum(s['backward_calls'] for s in ledger.stages))
        ledger.hvp_calls=int(sum(s['hvp_operations'] for s in ledger.stages))
    ledger.component_count=16
    return ledger


def correctness_audit(model,graph,edges,data,out,identity,ledger):
    path=out/'intervention_correctness_audit.csv'
    if path.exists() and (out/'correctness_passed.json').exists():return
    splits={'audit':subset(data['train'],range(4))}
    fixed=DiagnosticEdgeExperiment(model,graph,edges,splits,2,identity,ledger,cache_dir=out/'cache_audit')
    historical=FrozenSenderEdgeExperiment(model,graph,edges,splits,2,identity,ledger)
    rng=np.random.default_rng(42)
    masks=[('baseline',[0]*16),('dependent_top6',[1]*6+[0]*10)]
    masks.extend(('partial_'+str(i),s) for i,s in enumerate(load_states(SOURCE/'exact_acdc_trajectory.csv')))
    for count in [2,4,8,12,16]:
        for repeat in range(2):
            state=np.zeros(16,dtype=int);state[rng.choice(16,count,replace=False)]=1
            masks.append((f'random_k{count}_{repeat}',state.tolist()))
    previous=pd.read_csv(SOURCE/'edge_prediction_results.csv',dtype={'state':str}).drop_duplicates(['state','candidate_index'])
    for idx in np.linspace(0,len(previous)-1,8).astype(int):
        row=previous.iloc[idx];state=list(map(int,row.state));state[int(row.candidate_index)]=1
        masks.append((f'previous_query_sample_{idx}',state))
    rows=[]
    with ledger.instrument(model),ledger.stage('correctness_audit',estimate=f'{len(masks)} masks x 4 train prompts x official/new/historical; no validation use'):
        before=fixed.evaluate_state([0]*16,'audit',fresh=True)
        for label,state in tqdm(masks,desc='Official multi-edge audit'):
            new=fixed.evaluate_state(state,'audit',fresh=True)
            official=fixed.official_binary_score(state,'audit')
            old=historical.evaluate_state(state,'audit',fresh=True)
            error=float(max(abs(new-official)));old_error=float(max(abs(old-official)))
            np.testing.assert_allclose(new,official,atol=5e-4,rtol=1e-4)
            rows.append({'mask':label,'state':state_string(state),'removed_edges':sum(state),
                         'max_abs_corrected_vs_official':error,'max_abs_historical_vs_official':old_error,
                         'historical_mean_signed_error':float((old-official).mean()),'corrected_pass':True})
        np.testing.assert_allclose(before,fixed.evaluate_state([0]*16,'audit',fresh=True),atol=3e-4,rtol=1e-4)
        # Both first and second derivatives must include upstream sender changes.
        z=torch.full((16,),.3,device='cuda',dtype=torch.float32)
        v=torch.zeros_like(z);v[0]=1;v[3]=-.5
        f=lambda x:fixed.continuous(x,'audit').mean()
        hv=hessian_vector_product(f,z,v)
        grad=lambda x:torch.autograd.grad(f(x.requires_grad_(True)),x)[0].detach()
        eps=.01;finite=(grad((z+eps*v).clone())-grad((z-eps*v).clone()))/(2*eps)
        np.testing.assert_allclose(hv.cpu(),finite.cpu(),atol=.01,rtol=.08)
        ledger.hvp_calls+=1;ledger.backward_calls+=4
        assert all(p.grad is None for p in model.parameters())
    pd.DataFrame(rows).to_csv(path,index=False)
    write_json(out/'correctness_passed.json',{'passed':True,'masks':len(masks),'prompts':4,
        'hvp_fd_max_abs_error':float((hv-finite).abs().max()),'adapter':EdgeExperiment.VERSION})


def freeze_queries(model,graph,edges,data,out,identity,ledger):
    path=out/'query_definitions.csv'
    if path.exists():
        return pd.read_csv(path,dtype={'state':str}),pd.read_csv(out/'decision_pairs.csv')
    discovery={'train':subset(data['train'],range(16))}
    exp=DiagnosticEdgeExperiment(model,graph,edges,discovery,2,identity,ledger,cache_dir=out/'cache_selection')
    with ledger.instrument(model),ledger.stage('train_only_query_selection',estimate='16 singleton trials and three exact train-only trajectories; 16 prompts'):
        zero=[0]*16
        singleton=np.array([exp.exact_candidate_effect(zero,i,'train').mean() for i in range(16)])
        important=abs(singleton[abs(singleton)>.01])
        trajectory_thresholds=np.unique(np.quantile(important,[.25,.5,.75]))
        trajectory=[];pool=[]
        for tau in trajectory_thresholds:
            final,rows=exact_acdc(exp,'train',float(tau))
            trajectory.extend(dict(row,threshold=tau) for row in rows)
            pool.extend(row['state_before'] for row in rows);pool.append(state_string(final))
        pd.DataFrame(trajectory).to_csv(out/'corrected_train_trajectories.csv',index=False)
        # Retain one historical partial state; select other states only from
        # corrected, accepted ACDC trajectories (never force-remove an edge).
        selected=[state_string(load_states(SOURCE/'exact_acdc_trajectory.csv',removed_targets=(1,))[0])]
        pool=sorted(set(s for s in pool if 0<s.count('1')<16))
        for target in [4,8]:
            candidates=[s for s in pool if s not in selected]
            if not candidates:raise RuntimeError('Not enough distinct corrected ACDC states.')
            selected.append(min(candidates,key=lambda s:(abs(s.count('1')-target),-s[:6].count('1'),s)))
        rows=[]
        for bits in selected:
            state=list(map(int,bits))
            for i,bit in enumerate(state):
                if bit:continue
                effect=float(exp.exact_candidate_effect(state,i,'train').mean())
                rows.append({'query_id':f'{bits}:{i}','state':bits,'removed_edges':sum(state),
                             'candidate_index':i,'edge':edges[i].name,'selection_train16_effect':effect})
    queries=pd.DataFrame(rows);queries.to_csv(path,index=False)
    magnitudes=abs(queries.selection_train16_effect.to_numpy())
    thresholds=list(np.linspace(0.01,max(.3,float(max(magnitudes))*1.2),31))
    for x in magnitudes:
        thresholds.extend(max(1e-7,x*factor) for factor in [.5,.95,.99,1.01,1.05,2.])
    thresholds=np.unique(np.round(thresholds,9))
    pairs=[]
    for row in queries.itertuples():
        mag=abs(row.selection_train16_effect)
        for tau in thresholds:
            pairs.append({'query_id':row.query_id,'threshold':tau,
                          'difficult':abs(mag-tau)<=max(1e-6,mag*.051),
                          'selection_train16_effect':row.selection_train16_effect})
    pairs=pd.DataFrame(pairs);pairs.to_csv(out/'decision_pairs.csv',index=False)
    pd.DataFrame({'threshold':thresholds}).to_csv(out/'thresholds.csv',index=False)
    write_json(out/'query_selection.json',{'training_examples':16,'seed':42,
        'trajectory_thresholds':trajectory_thresholds.tolist(),'states':selected,
        'query_count':len(queries),'decision_pairs':len(pairs),'difficult_pairs':int(pairs.difficult.sum()),
        'difficult_unique_queries':int(pairs.loc[pairs.difficult,'query_id'].nunique()),
        'validation_used':False})
    return queries,pairs


def job(model,graph,edges,train,validation,size,seed,queries,pairs,out,identity,ranks):
    folder=out/f'seed{seed}_n{size}';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():print('Already complete:',folder,flush=True);return
    ledger=restore_ledger(folder)
    datasets={'train':subset(train,range(size)),'validation':validation}
    exp=DiagnosticEdgeExperiment(model,graph,edges,datasets,2,identity+f':n{size}:seed{seed}',ledger,cache_dir=folder/'exact_cache')
    # Reuse only scores from this IDENTICAL fixed validation dataset and model.
    validation_cache=out/'fixed_validation_scores.npz'
    val_id=hashlib.sha256((identity+validation.dataset._fingerprint+EdgeExperiment.VERSION).encode()).hexdigest()
    if validation_cache.exists():
        with np.load(validation_cache,allow_pickle=False) as saved:
            if saved['identity'].item()!=val_id:raise ValueError('Fixed validation cache identity changed.')
            for key in saved.files:
                if key.startswith('state_'):exp.cache[(exp.cache_id,'validation',tuple(map(int,key[6:])))] = saved[key].copy()
    states=[list(map(int,s)) for s in queries.state.drop_duplicates()]
    query_path=folder/'exact_queries.csv'
    if not query_path.exists():
        with ledger.instrument(model),ledger.stage('exact_ground_truth',estimate=f'{len(queries)} train/validation queries; fixed validation cache reused where present'):
            measured,effects=exact_queries(exp,states)
        measured.to_csv(query_path,index=False);effects.to_csv(folder/'per_example_effects.csv',index=False)
        val_values={f'state_{state_string(key[2])}':value for key,value in exp.cache.items() if key[1]=='validation'}
        if val_values:np.savez_compressed(validation_cache,identity=val_id,**val_values)
    measured=pd.read_csv(query_path,dtype={'state':str})
    assert set(measured.query_id)==set(queries.query_id)
    effects=pd.read_csv(folder/'per_example_effects.csv')
    validation_effects={q:g.sort_values('example_index').effect.to_numpy() for q,g in effects[effects.split=='validation'].groupby('query_id')}
    # Calibrate the true cost of an uncached trial with current score reused.
    if not (folder/'exact_query_timing.json').exists():
        with ledger.instrument(model),ledger.stage('fresh_candidate_cost',estimate='3 uncached exact TRAIN candidate trials; current-score reuse assumed'):
            start=time.perf_counter()
            for row in measured.iloc[:3].itertuples():
                state=list(map(int,row.state));state[row.candidate_index]=1
                exp.evaluate_state(state,'train',fresh=True)
            ledger.sync();elapsed=time.perf_counter()-start
        write_json(folder/'exact_query_timing.json',{'queries':3,'seconds_per_query':elapsed/3,
            'scope':'uncached candidate forward evaluation; already-known current score; donor preparation included'})
    prediction_rows=[];cost_frames=[];curvature_rows=[]
    contexts=len(states)*2;hvp_est=contexts*(16+max(ranks))*math.ceil(size/2)
    print(f'FULL n={size} seed={seed}: {len(queries)} queries, validation=128, {contexts} contexts, {hvp_est} HVP operations; batch=2',flush=True)
    for state in states:
        bits=state_string(state);group=measured[measured.state==bits]
        for strength in [0.,.5]:
            with ledger.instrument(model):
                bundle,costs=streaming_curvature_bundle(exp,state,strength,ranks,seed,ledger,folder,include_full=True)
            cost_frames.append(costs)
            full=bundle['full_second_order'];g=bundle['gradient'];anchor=bundle['anchor']
            curvature_rows.append({'state':bits,'anchor_strength':strength,'method':'full_second_order',
                'relative_asymmetry':float(bundle['hessian_relative_asymmetry'])})
            matrices={'first_order':None,'full_second_order':full}
            matrices.update({f'sketch_rank_{rank}':bundle[f'sketch_rank_{rank}'] for rank in ranks})
            for method,h in matrices.items():
                if method.startswith('sketch'):
                    curvature_rows.append({'state':bits,'anchor_strength':strength,'method':method,
                        'relative_compression_error':float(np.linalg.norm(h-full)/(np.linalg.norm(full)+1e-12))})
                for q in group.to_dict('records'):
                    pred=taylor_effect(g,h,anchor,state,q['candidate_index'])
                    full_pred=taylor_effect(g,full,anchor,state,q['candidate_index'])
                    prediction_rows.append({**q,'anchor_strength':strength,'method':method,'prediction':pred,
                        'taylor_signed_error':full_pred-q['train_effect'],
                        'compression_signed_error':pred-full_pred if method.startswith('sketch') else np.nan,
                        'generalization_shift':q['train_effect']-q['validation_effect']})
            pd.DataFrame(prediction_rows).to_csv(folder/'predictions.partial.csv',index=False)
    predictions=pd.DataFrame(prediction_rows);costs=pd.concat(cost_frames,ignore_index=True)
    predictions.to_csv(folder/'predictions.csv',index=False);costs.to_csv(folder/'method_costs.csv',index=False)
    pd.DataFrame(curvature_rows).to_csv(folder/'curvature_diagnostics.csv',index=False)
    metrics=final_query_metrics(predictions,validation_effects,pairs,seed)
    total_cost=costs.groupby(['anchor_strength','method'])[['wall_seconds','forward_passes_batches','examples_processed','backward_calls','hvp_operations']].sum().reset_index()
    peak=costs.groupby(['anchor_strength','method']).peak_cuda_allocated_mib.max().reset_index()
    metrics=metrics.merge(total_cost,on=['anchor_strength','method']).merge(peak,on=['anchor_strength','method'])
    metrics['discovery_examples']=size;metrics['seed']=seed
    query_seconds=json.loads((folder/'exact_query_timing.json').read_text())['seconds_per_query']
    metrics['seconds_per_uncached_exact_query']=query_seconds
    metrics['hypothetical_break_even_avoided_queries']=np.ceil(metrics.wall_seconds/query_seconds).astype(int)
    metrics['available_candidate_queries']=len(queries)
    metrics['hypothetical_break_even_against_2_35x_faster_exact']=np.ceil(metrics.wall_seconds/(query_seconds/2.35)).astype(int)
    metrics.to_csv(folder/'prediction_comparison.csv',index=False)
    hard=predictions.merge(pairs[pairs.difficult],on='query_id',validate='many_to_many')
    hard['exact_prune']=abs(hard.validation_effect)<hard.threshold;hard['predicted_prune']=abs(hard.prediction)<hard.threshold
    hard['correct']=hard.exact_prune==hard.predicted_prune
    hard.to_csv(folder/'difficult_decisions.csv',index=False)
    write_json(folder/'complete.json',{'success':True,'discovery_examples':size,'validation_examples':128,
        'seed':seed,'ranks':ranks,'queries':len(queries),'smoke':False,'heldout_validation_fingerprint':validation.dataset._fingerprint})
    print(metrics[['anchor_strength','method','mae','difficult_accuracy','wall_seconds','hypothetical_break_even_avoided_queries']].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sizes',type=int,nargs='+',default=[16,32,64,128])
    parser.add_argument('--seeds',type=int,nargs='+',default=[42])
    parser.add_argument('--ranks',type=int,nargs='+',default=[4,8,12,16])
    parser.add_argument('--out',type=Path)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if any(n not in [16,32,64,128] for n in args.sizes):raise ValueError('Sizes must be 16/32/64/128; no smoke fallback.')
    out=args.out or ROOT/'results/mib_sketch_diagnostics/final'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=out.resolve();out.mkdir(parents=True,exist_ok=True);print('FINAL OUTPUT:',out,flush=True)
    seed_everything(42);repo,provenance=configure_repository(ROOT)
    model=load_model('cuda');data,manifest,split_info=load_splits(model,{'train':128,'validation':128},42)
    manifest.to_csv(out/'dataset_manifest.csv',index=False)
    from eap.graph import Graph
    graph=Graph.from_model(model);graph.reset(empty=False);edges=load_edges(SOURCE/'selected_edges.csv')
    paths=[ROOT/'src/sketch_acdc/core.py',ROOT/'src/sketch_acdc/diagnostics.py',Path(__file__)]
    implementation={str(p.relative_to(ROOT)):digest(p) for p in paths}
    config_path=out/'experiment_config.json'
    if config_path.exists():
        config=json.loads(config_path.read_text())
        if config['implementation']!=implementation:raise RuntimeError('Implementation changed: use a new run directory.')
    else:
        config={'model':'gpt2','task':'official MIB IOI','requested_sizes':[16,32,64,128],
            'primary_seed':42,'fixed_validation_seed':42,'validation_examples':128,'split_info':split_info,
            'edges':[vars(e) for e in edges],'batch_size':2,'anchors':[0.,.5],'ranks':args.ranks,
            'adapter':EdgeExperiment.VERSION,'implementation':implementation,'mib_provenance':provenance,
            'precision':'model/derivatives float32; detached small matrix accumulation and pseudoinverse float64',
            'bootstrap_prompts':200,'selection':'train16 seed42 only; fixed thereafter',
            'seed_extension_rule':'Additional seeds 43/44 for a sketch with >=5% training MAE gain or >=2pp training difficult-decision gain; choose before viewing validation extension results.',
            'external_reports':{},'old_notebooks':{str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'notebooks').glob('*.ipynb') if p.name!='mib_sketch_diagnostics.ipynb'}}
        sources=out/'external_reports';sources.mkdir(exist_ok=True)
        for name in ['README','RESEARCH_LOG','ACDC_FP32_SCALE_REPORT','ACDC_GPT2_REPORT','ACDC_GPT2_GAP_REPORT']:
            p=Path('C:/Users/Nikhil/Downloads')/(name+'.md')
            if p.exists():shutil.copyfile(p,sources/p.name);config['external_reports'][name]={'source':str(p),'sha256':digest(p)}
        write_json(config_path,config)
    identity=json.dumps({'implementation':implementation,'model':provenance,'edges':[e.name for e in edges]},sort_keys=True)
    common=restore_ledger(out/'common_compute')
    correctness_audit(model,graph,edges,data,out,identity,common)
    if args.audit_only:return
    queries,pairs=freeze_queries(model,graph,edges,data,out,identity,common)
    for seed in args.seeds:
        train=data['train']
        if seed!=42:
            extra,extra_manifest,extra_info=load_splits(model,{'train':128},seed)
            train=extra['train'];extra_manifest.to_csv(out/f'train_manifest_seed{seed}.csv',index=False)
            write_json(out/f'train_info_seed{seed}.json',extra_info)
        for size in args.sizes:
            job(model,graph,edges,train,data['validation'],size,seed,queries,pairs,out,identity,args.ranks)
    for name,sha in config['old_notebooks'].items():
        if digest(ROOT/name)!=sha:raise AssertionError('An old notebook changed: '+name)
    print('REQUESTED CONFIGURATIONS FINISHED:',out,flush=True)


if __name__=='__main__':main()
