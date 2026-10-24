"""Pinned official EAP-IG-inputs, fixed batch4 and five left-endpoint samples."""
from functools import partial
import json
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import sha,write_json
from src.mattr_optimization.gate1 import LiveAccounting
from src.mattr_hierarchy.runner import weight_hash
from src.mattr_matched_size.topology import graph_new
from .core import protocol,verified,marker,attempt,configure_runtime,load_model,close_meter,runtime_guard

def project_memory(tokenizer,full,observed_peak,available):
    lengths=[np.array([len(x)+1 for x in tokenizer(full[col].tolist(),add_special_tokens=False).input_ids]) for col in ('clean_prompt','corrupted_prompt')]
    sampled_max=max(int(x[:8].max()) for x in lengths);full_max=max(int(x.max()) for x in lengths)
    upper=observed_peak*max(1.,(full_max/sampled_max)**2)
    if upper>available:raise RuntimeError('Conservative full-length EAP memory projection exceeds physical device memory; preserve registered settings and inspect before full attribution')
    return dict(sampled_max_positions=sampled_max,full_train_max_positions=full_max,conservative_full_length_allocated_mib=upper,device_total_mib=available)

def attribute(model,ds,meter):
    from eap.attribute import attribute as official
    from MIB_circuit_track.metrics import get_metric
    graph=graph_new();old=torch.Tensor.backward
    def backward(t,*args,**kwargs):
        meter.counts['backward_calls']+=1;return old(t,*args,**kwargs)
    model.zero_grad(set_to_none=True)
    with patch.object(torch.Tensor,'backward',backward):
        official(model,graph,ds.to_dataloader(4),partial(get_metric('logit_diff','ioi',model.tokenizer,model),mean=True,loss=True),
                 method='EAP-IG-inputs',intervention='patching',ig_steps=5,quiet=True)
    values=graph.scores[graph.real_edge_mask.bool()].detach().cpu().clone()
    assert values.shape==(32491,) and torch.isfinite(values).all()
    return values

def run(out,n,seed,approved=False):
    if n==10000 and seed!=42:raise ValueError('The registered full-data EAP result is one seed42 ranking, not independent seed replicates')
    p=protocol(out);rec=next((j for j in p['eap_jobs'] if j['dataset_size']==n and (n==10000 or j['seed']==seed)),None)
    if rec is None:raise ValueError('Unregistered EAP job')
    dest=out/'attribution'/rec['key'];dest.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol_sha256=sha(out/'gate5f_protocol.json'),job=rec)
    if verified(dest,identity):print('Verified cached EAP:',rec['key'],flush=True);return
    runtime_guard(out,approved);configure_runtime()
    from .core import verify_runtime
    verify_runtime(out)
    from MIB_circuit_track.dataset import HFEAPDataset
    from .core import atomic_torch
    with attempt(out,'eap_'+rec['key']) as work:
        meter=LiveAccounting(work)
        try:
            model=load_model(meter,seed=seed,freeze=True);before=weight_hash(model)
            ds=HFEAPDataset('mib-bench/ioi',model.tokenizer,split='train',task='ioi',model_name='gpt2')
            frame=pd.read_csv(out/('train_n10000.csv' if n==10000 else f'train_n128_s{seed}.csv'))
            ds.dataset=ds.dataset.select(frame.official_filtered_index.astype(int).tolist())
            assert len(ds)==n
            print(f'Official EAP-IG-inputs:{n} training examples,batch4,ig_steps5; {7*((n+3)//4)} forwards,{5*((n+3)//4)} backwards',flush=True)
            start=time.perf_counter()
            with meter.stage('eap_ig'):values=attribute(model,ds,meter)
            seconds=time.perf_counter()-start
            assert weight_hash(model)==before
            path=dest/'scores.pt'
            if path.exists():assert torch.equal(torch.load(path,weights_only=False),values)
            else:atomic_torch(path,values)
            write_json(work/'rates.json',dict(eap_examples=n,eap_seconds=seconds))
            write_json(dest/'attribution.json',dict(identity=identity,weight_hash=before,weights_unchanged=True,examples=n,
                seconds=seconds,counts=meter.counts.copy(),batch=4,ig_steps=5,convention='pinned mean-loss,batch-aligned normalization; no abs/sign flip'))
            marker(dest,identity,[path,dest/'attribution.json'])
        finally:close_meter(meter)

def preflight(out):
    """Bounded train-only4/8 throughput and equal-batch partition test."""
    import copy
    from MIB_circuit_track.dataset import HFEAPDataset
    from .evaluate import selected_prefixes
    from src.mattr_matched_size.evaluation import OfficialEngine
    from .core import attempt
    with attempt(out,'bounded_eap_eval_preflight') as work:
        meter=LiveAccounting(work)
        try:
            model=load_model(meter,freeze=True);before=weight_hash(model)
            ds=HFEAPDataset('mib-bench/ioi',model.tokenizer,split='train',task='ioi',model_name='gpt2')
            def subset(ids):
                sub=copy.copy(ds);sub.dataset=ds.dataset.select(ids);return sub
            vectors={};rows=[]
            for name,ids in [('first4',list(range(4))),('all8',list(range(8))),('last4',list(range(4,8)))]:
                start=time.perf_counter()
                with meter.stage('eap_'+name):vectors[name]=attribute(model,subset(ids),meter)
                rows.append(dict(name=name,examples=len(ids),seconds=time.perf_counter()-start))
            expected=(vectors['first4']+vectors['last4'])/2
            torch.testing.assert_close(vectors['all8'],expected,rtol=2e-4,atol=2e-5)
            rate=rows[1]['seconds']/8;ratio=rows[1]['seconds']/rows[0]['seconds']
            assert .5<ratio<5,'Throughput grossly nonlinear; inspect instead of extrapolating blindly'
            # Four training rows only; official full/empty/reference checks, no512-row evaluation.
            graph=graph_new();engine=OfficialEngine(model,graph,subset(list(range(4))),work/'eval', 'bounded-training-preflight','first4',2)
            (work/'eval').mkdir(exist_ok=True)
            start=time.perf_counter()
            with meter.stage('bounded_official_evaluation'),torch.no_grad():
                # Use normal10-point grid so the same official evaluator is tested.
                from src.mattr_optimization.common import GRID
                result,curve,per=engine.area(vectors['first4'],GRID)
            evaluation_seconds=time.perf_counter()-start
            assert weight_hash(model)==before
            table=pd.DataFrame(rows);table.to_csv(out/'eap_preflight_throughput.csv',index=False)
            full=pd.read_csv(out/'train_n10000.csv')
            available=torch.cuda.get_device_properties(0).total_memory/2**20
            memory=project_memory(model.tokenizer,full,max(r['peak_allocated_mib'] for r in meter.rows),available)
            write_json(out/'eap_preflight.json',dict(passed=True,examples=16,ig_steps=5,batch_size=4,
                partition_equivalent=True,max_partition_error=float((vectors['all8']-expected).abs().max()),
                observed_8_to_4_time_ratio=ratio,seconds_per_example=rate,full10000_projected_seconds=rate*10000,
                peak_allocated_mib=max(r['peak_allocated_mib'] for r in meter.rows),peak_reserved_mib=max(r['peak_reserved_mib'] for r in meter.rows),
                **memory,
                memory_projection='fixed batch4; entire observed allocated peak scaled by squared maximum-token-length ratio as a conservative heuristic, not a guaranteed bound. OOM stops without changing settings.',
                normalization='official mean=True loss=True plus divide by total_items. Partition test uses identical batch4 and example-weighted chunks; varying batch sizes is not equivalent raw-score scaling.',
                model_weight_sha256=before,evaluation_seconds_per_ranking_projection=evaluation_seconds*(512/4)*2))
        finally:close_meter(meter)
