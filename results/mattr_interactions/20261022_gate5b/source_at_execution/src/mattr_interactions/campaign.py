"""Bounded Gate 5B campaign; never invokes a trainer or optimizer."""
import argparse
import copy
from functools import partial
import importlib.util
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import ROOT,UPSTREAM,configure,provenance,sha,write_json,protected_snapshot
from src.mattr_optimization.gate3 import baseline
from src.mattr_optimization.gate1 import LiveAccounting
from .design import prepare,cleanup,toggled,assign,FRACTIONS,digest
from .evaluation import Evaluator,four_way,pair_effects
from .statistics import summarize,select_confirmation

def select_data(model,out,p,confirmation=False):
    from MIB_circuit_track.dataset import HFEAPDataset
    split='validation' if confirmation else 'train'
    ds=HFEAPDataset('mib-bench/ioi',model.tokenizer,split=split,task='ioi',model_name='gpt2')
    excluded_indices=set();excluded_prompts=set();inputs={}
    if confirmation:
        for parent in (ROOT/'results/mattr_optimization',ROOT/'results/mattr_expander'):
            for path in sorted(parent.rglob('*manifest.csv')):
                if 'validation' not in path.name:continue
                frame=pd.read_csv(path)
                if len(frame)>=10000:continue # historical Gate1 is full-validation, explicitly disclosed
                if 'official_filtered_index' in frame:excluded_indices.update(frame.official_filtered_index.astype(int))
                if 'clean_prompt' in frame:excluded_prompts.update(frame.clean_prompt)
                inputs[str(path.relative_to(ROOT))]=sha(path)
        for seed in (42,43,44):
            path=baseline(seed)/'train_manifest.csv';excluded_prompts.update(pd.read_csv(path).clean_prompt)
        indices=[]
        for i in np.random.default_rng(p['confirmation_seed']).permutation(len(ds)):
            if int(i) in excluded_indices or ds[int(i)][0] in excluded_prompts:continue
            indices.append(int(i))
            if len(indices)==p['confirmation_prompts']:break
        if len(indices)!=p['confirmation_prompts']:raise ValueError('Insufficient disjoint official validation rows.')
    else:
        selection=json.loads((baseline(42)/'train_selection.json').read_text());indices=selection['indices'][:32]
        assert ds.dataset._fingerprint==selection['official_fingerprint']
    ds.dataset=ds.dataset.select(indices)
    rows=[]
    for j,i in enumerate(indices):
        clean,donor,labels=ds[j];meta=ds.dataset[j]['metadata']
        for name in ('indirect_object','subject','random_c'):
            assert len(model.tokenizer(' '+meta[name],add_special_tokens=False).input_ids)==1
        rows.append({'official_filtered_index':i,'clean_prompt':clean,'corrupted_prompt':donor,
                     'correct_token':labels[0],'incorrect_token':labels[1]})
    frame=pd.DataFrame(rows);frame.to_csv(out/f'{split}_manifest.csv',index=False)
    write_json(out/f'{split}_cohort.json',{'split':split,'indices':indices,'fingerprint':ds.dataset._fingerprint,
       'manifest_sha256':sha(out/f'{split}_manifest.csv'),'excluded_index_count':len(excluded_indices),
       'excluded_prompt_count':len(excluded_prompts),'prior_development_manifests':inputs,
       'disjoint_from_prior_small_development':True if confirmation else None,
       'historical_full_Gate1_validation_exposure':confirmation})
    if not confirmation:
        expected=pd.read_csv(baseline(42)/'train_manifest.csv').head(32)
        assert frame.clean_prompt.tolist()==expected.clean_prompt.tolist()
    return ds

def live_oracle(model,dataset,evaluator,masks,out):
    """Capture the pinned live loss closure without executing learn_scores.

    No trainer, optimizer, mask relaxation, or model backward is called.
    """
    from transformer_lens import HookedTransformer
    import MIB_circuit_track.dataset as datasets
    import matryoshka_attribution as package
    class Complete(Exception):pass
    records=[]
    def inspect_loss(total,loss_fn,**kw):
        assert total==32491
        for name,mask in masks:
            # Official CLI loss returns negative correct-minus-incorrect margin.
            live=[]
            with torch.no_grad():
                for index in range(len(dataset)):
                    with patch.object(random,'randint',return_value=index):
                        live.append(-float(loss_fn(torch.tensor(mask,dtype=torch.float32,device='cuda'))))
            oracle=evaluator.evaluate(mask,dataset,'audit_live',use_cache=False,donor_cache=False)[0]
            error=float(np.max(np.abs(np.array(live)-oracle)))
            if not np.allclose(live,oracle,atol=1e-4,rtol=2e-5):raise ValueError(f'Live vs official mismatch: {name}/{error}')
            records.append({'mask':name,'live_vs_official_max_error':error,'examples':len(dataset),'passed':True})
        raise Complete()
    spec=importlib.util.spec_from_file_location('gate5b_live_oracle',UPSTREAM/'scripts/mib/eval_mib_edge.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    argv=['eval_mib_edge.py','--model','gpt2','--task','ioi','--steps','1','--mode','iso',
          '--k-schedule','uniform','--masking','topk','--seed','42','--no-wandb','--output',str(out/'oracle_cli')]
    with patch.object(HookedTransformer,'from_pretrained',return_value=model),\
         patch.object(datasets,'HFEAPDataset',return_value=dataset),\
         patch.object(package,'learn_scores',inspect_loss),patch.object(sys,'argv',argv):
        try:module.main()
        except Complete:pass
        else:raise ValueError('Oracle unexpectedly reached training/evaluation.')
    pd.DataFrame(records).to_csv(out/'live_patching_audit.csv',index=False)

def validate(evaluator,dataset,pairs,out):
    graph=evaluator.graph;small=copy.copy(dataset);small.dataset=dataset.dataset.select(range(2))
    full=np.ones(32491,dtype=bool);empty=np.zeros(32491,dtype=bool)
    masks=[('full',full),('empty',empty)]
    for f in FRACTIONS:
        base=np.load(out/f'base_{f:g}.npy');masks.append((f'base_{f}',base))
        for direction in ('deletion','addition'):
            pair=pairs[(pairs.fraction==f)&(pairs.direction==direction)].iloc[0]
            masks.append((f'{f}_{direction}_joint',toggled(base,[int(pair.i),int(pair.j)])))
    mapping=pd.read_csv(out/'edge_mapping.csv',keep_default_na=False)
    # Explicit dependent path input -> a0.h0 -> a1.h0<q>, plus a0.h0 -> logits.
    dependent=[mapping.index[mapping.edge==name][0] for name in ('input->a0.h0<q>','a0.h0->a1.h0<q>')]
    dependent_mask=toggled(full,dependent);assert np.array_equal(cleanup(graph,dependent_mask),dependent_mask)
    masks.append(('dependent_two_edge_path',dependent_mask))
    live_oracle(evaluator.model,small,evaluator,masks,out)
    records=[]
    for name,mask in masks:
        plain=evaluator.evaluate(mask,small,'audit',use_cache=False,donor_cache=False)
        cached=evaluator.evaluate(mask,small,'audit',use_cache=False)
        repeat=evaluator.evaluate(mask,small,'audit',use_cache=False)
        for metric,a,b,c in zip(('margin','accuracy'),plain,cached,repeat):
            error=float(np.max(np.abs(a-b)));repeat_error=float(np.max(np.abs(b-c)))
            assert np.allclose(a,b,atol=1e-4,rtol=2e-5) and np.array_equal(b,c),(name,metric,error,repeat_error)
            records.append({'mask':name,'metric':metric,'donor_cache_max_error':error,'repeat_error':repeat_error,'passed':True})
    # Same frozen mask, identical examples, batch2 vs singleton evaluation.
    mask=masks[-1][1];batched=evaluator.evaluate(mask,small,'batch',use_cache=False)[0];singles=[]
    for i in range(2):
        one=copy.copy(small);one.dataset=small.dataset.select([i]);singles.extend(evaluator.evaluate(mask,one,'single',use_cache=False)[0])
    assert np.allclose(batched,singles,atol=1e-4,rtol=2e-5)
    pd.DataFrame(records).to_csv(out/'donor_cache_audit.csv',index=False)
    write_json(out/'correctness_passed.json',{'passed':True,'masks':len(masks),'live_patcher':'pinned corrected eval_mib_edge.py',
      'official_oracle':'eap.evaluate.evaluate_graph, normal prune enabled','dependent_multi_edge_checked':True,
      'batch_single_max_error':float(np.max(abs(batched-np.array(singles)))),'training_updates':0,'model_backwards':0})

def measure(evaluator,pairs,dataset,out,cohort):
    rows=[];started=time.perf_counter()
    print(f'{cohort}: {len(pairs)} pairs, {len(dataset)} prompts, batch4; at most {4*len(pairs)} mask requests before exact-cache reuse.',flush=True)
    for n,pair in enumerate(pairs.itertuples(),1):
        base=np.load(out/f'base_{pair.fraction:g}.npy')
        effects=pair_effects(four_way(evaluator,base,pair,dataset,cohort))
        for i in range(len(dataset)):rows.append({'pair_id':pair.pair_id,'example':i,**{k:float(v[i]) for k,v in effects.items()}})
        if n==1 or n%20==0 or n==len(pairs):
            evaluator.meter.status(cohort=cohort,pairs_completed=n,pairs_total=len(pairs),exact_cache_hits=evaluator.cache_hits,
                                   donor_cache_hits=evaluator.donors.hits)
            print(f'{cohort}: {n}/{len(pairs)} pairs; actual forwards={evaluator.meter.counts["forward_batches"]}',flush=True)
        if time.perf_counter()-evaluator.meter.started>14400 or evaluator.meter.counts['example_forwards']>30000:
            raise RuntimeError('Frozen compute cap reached; no experiment expansion.')
    per=pd.DataFrame(rows);per.to_csv(out/f'{cohort}_per_example.csv',index=False)
    summary=summarize(pairs,per,confirmation=cohort=='confirmation');summary.to_csv(out/f'{cohort}_interactions.csv',index=False)
    return summary

def run(out,prepare_only=False,report_only=False):
    configure();torch.set_num_threads(4)
    from eap.graph import Graph
    from .stability import analyze
    from .report import generate
    out=out.resolve();out.mkdir(parents=True,exist_ok=True)
    if report_only:generate(out);return
    p=prepare(out)
    for name,h in p['ranking_inputs'].items():assert sha(ROOT/name)==h,name
    for name,h in p['selection_artifacts'].items():assert sha(out/name)==h,name
    if prepare_only:return
    if (out/'completed.json').exists():
        print('Completed campaign reused; no new model forwards.',flush=True);generate(out);return
    if (out/'process_summary.json').exists():raise ValueError('Failed attempt preserved; inspect before explicitly resuming.')
    protected=protected_snapshot();write_json(out/'protected_previous_hashes.json',protected)
    prov=provenance();expected=json.loads((baseline(42)/'experiment_config.json').read_text())['source_provenance']
    assert prov==expected,'Pinned environment changed'
    write_json(out/'source_provenance.json',prov)
    for name,h in p['sources'].items():
        path=out/'source_at_execution'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU model fallback.')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(42);random.seed(42);np.random.seed(42)
    from transformer_lens import HookedTransformer
    from src.mattr_optimization.gate4_init import edge_mapping
    meter=LiveAccounting(out);success=False;evaluator=None
    try:
        with meter.stage('offline_ranking_stability'):
            graph=Graph.from_model(dict(n_layers=12,n_heads=12,parallel_attn_mlp=False,d_model=768))
            analyze(out,pd.read_csv(out/'edge_mapping.csv',keep_default_na=False),graph)
        with meter.stage('model_loading'):
            model=meter.attach(HookedTransformer.from_pretrained('gpt2-small'));model.eval();model.requires_grad_(False)
            assert model.cfg.dtype==torch.float32
            model.cfg.use_attn_result=True;model.cfg.use_split_qkv_input=True;model.cfg.use_hook_mlp_in=True;model.cfg.ungroup_grouped_query_attention=True
            graph=Graph.from_model(model)
            actual=edge_mapping(graph,graph);saved=pd.read_csv(out/'edge_mapping.csv',keep_default_na=False)
            pd.testing.assert_frame_equal(actual,saved[actual.columns])
        with meter.stage('training_cohort_loading'):train=select_data(model,out,p)
        evaluator=Evaluator(model,graph,out,meter,sha(out/'gate5b_protocol.json'))
        pairs=pd.read_csv(out/'selected_pairs.csv')
        with meter.stage('correctness_checks'),torch.no_grad():validate(evaluator,train,pairs,out)
        with meter.stage('training_screen'),torch.no_grad():screen=measure(evaluator,pairs,train,out,'screen')
        ids=select_confirmation(screen);mapping=pd.read_csv(out/'edge_mapping.csv',keep_default_na=False)
        controls=[];matches=[]
        def signature(row):
            a,b=mapping.iloc[int(row.i)],mapping.iloc[int(row.j)]
            return (a.edge_type,b.edge_type,a.source_layer,a.destination_layer,b.source_layer,b.destination_layer)
        for primary in pairs[pairs.pair_id.isin(ids)].itertuples():
            pool=pairs[(pairs.fraction==primary.fraction)&(pairs.direction==primary.direction)&pairs.control&(~pairs.pair_id.isin(ids))]
            if pool.empty:continue
            options=sorted([(sum(x!=y for x,y in zip(signature(primary),signature(r))),r.pair_id) for r in pool.itertuples()])
            distance,control_id=options[0];controls.append(control_id)
            matches.append({'primary_pair':primary.pair_id,'control_pair':control_id,'type_layer_mismatch':distance})
        chosen=pairs[pairs.pair_id.isin(ids+controls)].copy();chosen['selection_role']=np.where(chosen.pair_id.isin(ids),'primary','matched_control')
        pd.DataFrame(matches).to_csv(out/'confirmation_control_matches.csv',index=False)
        chosen.to_csv(out/'confirmation_pairs.csv',index=False)
        write_json(out/'confirmation_selection_frozen.json',{'primary_pair_ids':ids,'pair_ids':chosen.pair_id.tolist(),'selection_sha256':sha(out/'confirmation_pairs.csv'),
                  'training_result_sha256':sha(out/'screen_interactions.csv'),'protocol_sha256':sha(out/'gate5b_protocol.json'),
                  'confirmation_loaded':False,'confirmation_model_forwards':0})
        with meter.stage('confirmation_cohort_loading'):validation=select_data(model,out,p,confirmation=True)
        with meter.stage('heldout_confirmation'),torch.no_grad():measure(evaluator,chosen,validation,out,'confirmation')
        for name,h in protected.items():assert sha(ROOT/name)==h,name
        assert meter.counts['backward_calls']==0 and meter.counts['optimizer_updates']==0
        assert all(not e['grad_enabled'] for e in meter.events)
        assert not any(parameter.grad is not None for parameter in model.parameters())
        assert len(evaluator.unique)<=p['budget']['unique_masks_max']
        success=True
    except BaseException as exc:
        import traceback
        write_json(out/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),
                   'oom':isinstance(exc,torch.cuda.OutOfMemoryError),'settings_unchanged':True,
                   'oom_advice':'Use a separately frozen lower batch-size configuration after inspecting the failure; no silent changes.'})
        raise
    finally:
        meter.stop.set();meter.worker.join();meter.save();meter.stream.close()
        write_json(out/'process_summary.json',{'success':success,'seconds':time.perf_counter()-meter.started,'counts':meter.counts,
            'unique_evaluations_this_run':len(evaluator.unique) if evaluator else 0,'exact_cache_hits':evaluator.cache_hits if evaluator else 0,
            'donor_cache_hits':evaluator.donors.hits if evaluator else 0,'donor_cache_misses':evaluator.donors.misses if evaluator else 0,
            'donor_cache_cpu_bytes':evaluator.donors.bytes if evaluator else 0})
    write_json(out/'completed.json',{'passed':success,'historical_files_unchanged':len(protected),
       'training_updates':0,'new_algorithms':False,'leaderboard_submission':False,'protocol_sha256':sha(out/'gate5b_protocol.json')})
    generate(out)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args();run(args.out,args.prepare_only,args.report_only)
