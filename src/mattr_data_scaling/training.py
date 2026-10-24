"""Scoped adapter around the pinned loss, mask, optimizer and trainer; no GPT updates."""
import importlib.util
import json
from pathlib import Path
import random
import shutil
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import ROOT,UPSTREAM,MIB,sha,write_json
from src.mattr_hierarchy.adapter import Hierarchy
from src.mattr_hierarchy.checkpoint import rng_state,restore_rng
from src.mattr_hierarchy.runner import weight_hash
from src.mattr_optimization.gate1 import LiveAccounting
from .core import protocol,job_key,verified,marker,atomic_torch,configure_runtime,attempt,close_meter,runtime_guard

def checked_checkpoint(path,identity=None):
    path=Path(path)
    if sha(path)!=path.with_suffix('.sha256').read_text().strip():raise ValueError('Checkpoint hash mismatch: '+str(path))
    state=torch.load(path,map_location='cpu',weights_only=False)
    if identity is not None and state['identity']!=identity:raise ValueError('Checkpoint configuration mismatch')
    return state

def checkpoints(folder):
    """Preserve incomplete atomic-write tails; resume the last verified checkpoint."""
    paths=[]
    for path in sorted(Path(folder).glob('step_*.pt')):
        if not path.with_suffix('.sha256').exists():
            quarantine=path.parent/'incomplete';quarantine.mkdir(exist_ok=True)
            target=quarantine/(path.name+'.'+__import__('uuid').uuid4().hex)
            path.replace(target)
            print('Preserved incomplete checkpoint without checksum:',target,flush=True)
        else:checked_checkpoint(path);paths.append(path)
    return paths

def learn(original,total,loss_fn,adapter,plan,*,target,folder,identity,initial=None,meter=None,**kw):
    """Official trainer, independent recorded streams, atomic exact-state checkpoints."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    paths=checkpoints(folder)
    saved=checked_checkpoint(paths[-1],identity) if paths else initial
    offset=saved['completed'] if saved else 0
    if offset>target:raise ValueError('Requested target precedes checkpoint')
    log=list(saved['train_log']) if saved else []
    holder={};base_adam=torch.optim.Adam;cursor=[offset];last_row=[None]
    class Adam(base_adam):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);holder['optimizer']=self
            params=[x for g in self.param_groups for x in g['params']]
            assert len(params)==1 and params[0].numel()==adapter.count(total)
            if saved:
                with torch.no_grad():params[0].copy_(saved['scores'].to(params[0]))
                self.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        def step(self,*args,**kwargs):
            if meter:meter.update(float(self.param_groups[0]['params'][0].grad.detach().double().norm()))
            return super().step(*args,**kwargs)
    class Sampler:
        def sample(self):
            row=plan.iloc[cursor[0]];last_row[0]=int(row.row);cursor[0]+=1
            return float(row.k)
    def sampled(mask):
        with patch.object(random,'randint',return_value=last_row[0]):loss=loss_fn(mask)
        if loss is None:raise RuntimeError('Official length-mismatch skip; stop instead of silently changing update counts or stream alignment')
        return loss
    def callback(step,k,loss,scores):
        completed=offset+step+1
        assert cursor[0]==completed
        log.append((completed-1,float(k),float(k)/total,float(loss)))
        if completed%100==0 or completed==target:
            if scores.is_cuda:torch.cuda.synchronize()
            start=time.perf_counter();path=folder/f'step_{completed:06d}.pt'
            state=dict(identity=identity,completed=completed,scores=scores.detach().cpu().clone(),
                optimizer=holder['optimizer'].state_dict(),rng=rng_state(),train_log=log.copy(),
                stream_position=completed,stream_prefix_hash=__import__('hashlib').sha256(plan.iloc[:completed].to_csv(index=False,float_format='%.17g').encode()).hexdigest())
            with torch.no_grad():state['effective_scores']=adapter.transform(scores).detach().cpu().clone()
            atomic_torch(path,state)
            checksum=path.with_suffix('.sha256');pending=checksum.with_suffix('.sha256.tmp')
            pending.write_text(sha(path)+'\n');pending.replace(checksum)
            with (folder/'serialization.jsonl').open('a') as f:f.write(json.dumps(dict(step=completed,seconds=time.perf_counter()-start))+'\n')
            print(f'Update {completed}/{target}',flush=True)
    if offset==target:return saved
    # No reimplementation of Adam, gradient, top-k, or GPT edge patching.
    with patch.object(torch.optim,'Adam',Adam),adapter.install():
        original(total,sampled,steps=target-offset,k_sampler=Sampler(),on_step=callback,log_every=0,**kw)
    return checked_checkpoint(folder/f'step_{target:06d}.pt',identity)

def export_score(path,raw,adapter,effective_override=None):
    with torch.no_grad():effective=adapter.transform(raw.to(adapter.ids.device if adapter.ids is not None else 'cpu')).detach().cpu() if effective_override is None else effective_override.detach().cpu()
    assert effective.shape==(32491,) and torch.isfinite(effective).all()
    if path.exists():assert torch.equal(torch.load(path,map_location='cpu',weights_only=False),effective)
    else:atomic_torch(path,effective)
    return path

def verify_historical(out,checkpoint):
    inventory=pd.read_csv(out/'checkpoint_inventory.csv')
    row=inventory[inventory.checkpoint==str(checkpoint.relative_to(ROOT))].iloc[0]
    if not row.exact_reuse_valid or sha(checkpoint)!=row.checkpoint_sha256:raise ValueError('Historical checkpoint no longer matches frozen reuse inventory')
    return row

def historical_export(dest,checkpoint,step,adapter):
    row=verify_historical(dest.parent.parent.parent,checkpoint)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    original=checkpoint.parent.parent/f'effective_scores_{step}.pt'
    if adapter.kind=='hierarchy':
        assert original.exists(),'Historical H2 export required; never replace its GPU reduction with a newly rounded CPU ranking'
        assert sha(original)==row.effective_score_file_sha256
        values=torch.load(original,map_location='cpu',weights_only=False)
        torch.testing.assert_close(adapter.transform(state['scores']),values,atol=2e-6,rtol=1e-5)
        if dest.exists():assert sha(dest)==sha(original)
        else:
            temp=dest.with_suffix('.tmp');shutil.copyfile(original,temp);temp.replace(dest)
        return dest
    return export_score(dest,state['scores'],adapter)

def train(out,n,method,seed,target=5000,resume=False,approved=False,smoke=False):
    started=time.perf_counter()
    p=protocol(out) if not smoke else None
    jobs=p['training_jobs'] if p else json.loads((out/'replacement_job_manifest.json').read_text())
    rec=next((j for j in jobs if (j['dataset_size'],j['method'],j['seed'])==(n,method,seed)),None)
    if rec is None:raise ValueError('Unregistered job')
    key=job_key(method,n,seed);dest=out/('smoke' if smoke else 'training')/key;dest.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol_sha256=sha(out/'gate5f_protocol.json') if p else 'bounded-preflight',job=key,
        stream_sha256=sha(out/f'stream_n{n}_s{seed}.csv'),group_sha256=sha(out/'H2_groups.npy'),method=method,
        data_sha256=sha(out/('train_n10000.csv' if n==10000 else f'train_n128_s{seed}.csv')))
    if verified(dest,identity):print('Verified cached training:',key,flush=True);return
    existing=checkpoints(dest/'checkpoints')
    if existing and not resume:raise ValueError('Incomplete run detected. Use --resume to restore verified state; historical outputs untouched.')
    adapter=Hierarchy(np.load(out/'H2_groups.npy'),kind='hierarchy') if method=='h2' else Hierarchy(kind='bypass')
    if not smoke and target<5000 and rec['reuse_step']>=target:
        state=torch.load(ROOT/rec['endpoints'][str(target)],map_location='cpu',weights_only=False)
        historical_export(dest/f'effective_scores_{target}.pt',ROOT/rec['endpoints'][str(target)],target,adapter)
        write_json(dest/f'reuse_step{target}.json',dict(identity=identity,source=rec['endpoints'][str(target)]))
        print('Verified historical intermediate reuse:',key,target,flush=True);return
    if not smoke and rec['reuse_step']==5000:
        files=[]
        for step in (1000,5000):
            state=torch.load(ROOT/rec['endpoints'][str(step)],map_location='cpu',weights_only=False)
            files.append(historical_export(dest/f'effective_scores_{step}.pt',ROOT/rec['endpoints'][str(step)],step,adapter))
        write_json(dest/'reuse.json',dict(identity=identity,source=rec,new_training_seconds=0,new_CPU_export_seconds=time.perf_counter()-started,historical_cost_separate=True))
        marker(dest,identity,files+[dest/'reuse.json']);print('Verified historical reuse:',key,flush=True);return
    if not smoke:runtime_guard(out,approved)
    configure_runtime()
    if not smoke:
        from .core import verify_runtime
        verify_runtime(out)
    import matryoshka_attribution as package
    from transformer_lens import HookedTransformer
    from MIB_circuit_track.dataset import HFEAPDataset
    from src.mattr_optimization.gate0 import check_patching
    from src.mattr_optimization.gate4_init import edge_mapping
    from src.mattr_matched_size.topology import graph_new
    plan=pd.read_csv(out/f'stream_n{n}_s{seed}.csv',float_precision='round_trip')
    manifest=pd.read_csv(out/('train_n10000.csv' if n==10000 else f'train_n128_s{seed}.csv'))
    initial=None
    if not smoke and not existing and rec['reuse_step']:
        verify_historical(out,ROOT/rec['reuse_checkpoint'])
        initial=torch.load(ROOT/rec['reuse_checkpoint'],map_location='cpu',weights_only=False)
        assert initial['completed']==rec['reuse_step']
        historical_export(dest/f'effective_scores_{rec["reuse_step"]}.pt',ROOT/rec['reuse_checkpoint'],rec['reuse_step'],adapter)
        write_json(dest/f'reuse_step{rec["reuse_step"]}.json',dict(identity=identity,source=rec['reuse_checkpoint']))
    target=30 if smoke else target
    print(f'{key}:target={target},historical_offset={initial["completed"] if initial else 0}; checkpoints1000/5000; no GPT-weight optimization',flush=True)
    with attempt(out,('smoke_' if smoke else 'train_')+key) as work:
        write_json(work/'configuration.json',identity)
        meter=LiveAccounting(work);before=[None];dataset=[None];result=[None]
        original_model=HookedTransformer.from_pretrained;original_data=HFEAPDataset.__init__;original_train=package.learn_scores
        old_backward=torch.Tensor.backward
        def backward(t,*args,**kwargs):
            meter.counts['backward_calls']+=1;return old_backward(t,*args,**kwargs)
        def model_load(*args,**kwargs):
            with meter.stage('model_loading'):
                model=meter.attach(original_model(*args,**kwargs));model.eval();assert model.cfg.dtype==torch.float32
                before[0]=weight_hash(model);return model
        def data_load(ds,*args,**kwargs):
            assert kwargs['split']=='train','Training command must never reach evaluation'
            with meter.stage('training_data_loading'):
                original_data(ds,*args,**kwargs)
                ds.dataset=ds.dataset.select(manifest.official_filtered_index.astype(int).tolist())
                for i,row in manifest.iterrows():
                    clean,donor,labels=ds[i]
                    assert (clean,donor)==(row.clean_prompt,row.corrupted_prompt) and list(labels)==[row.correct_token,row.incorrect_token]
                dataset[0]=ds
        class Finished(Exception):pass
        def train_call(total,loss_fn,**kwargs):
            assert total==32491
            pd.testing.assert_frame_equal(edge_mapping(graph_new(),graph_new()),pd.read_csv(out/'edge_mapping.csv',keep_default_na=False))
            ad=Hierarchy(np.load(out/'H2_groups.npy'),kind='hierarchy',device='cuda') if method=='h2' else Hierarchy(kind='bypass')
            if smoke:
                with meter.stage('live_patching_oracle_checks'):check_patching(loss_fn,SimpleNamespace(datasets={'train':dataset[0]}),meter.model,work)
            for k in ('steps','on_step','log_every'):kwargs.pop(k,None)
            start=time.perf_counter()
            with meter.stage('training'):
                state=learn(original_train,total,loss_fn,ad,plan,target=target,folder=dest/'checkpoints',identity=identity,initial=initial,meter=meter,**kwargs)
            seconds=time.perf_counter()-start
            for step in (1000,5000,30):
                ckpt=dest/f'checkpoints/step_{step:06d}.pt'
                if ckpt.exists():
                    saved_state=checked_checkpoint(ckpt,identity)
                    export_score(dest/f'effective_scores_{step}.pt',saved_state['scores'],ad,saved_state['effective_scores'])
            if smoke:
                from src.mattr_hierarchy.runner import tensor_state_equal
                # Same live objective, method and fixed draws: uninterrupted repeat and15+15 resume.
                for mode in ('repeat','resume'):
                    for stop in ([30] if mode=='repeat' else [15,30]):
                        with meter.stage(f'smoke_{mode}_{stop}'):
                            replay=learn(original_train,total,loss_fn,ad,plan,target=stop,folder=work/mode,identity=identity,meter=meter,**kwargs)
                    assert torch.equal(state['scores'],replay['scores']) and tensor_state_equal(state['optimizer'],replay['optimizer'])
                    assert tensor_state_equal(state['rng'],replay['rng'])
                write_json(work/'smoke_checks.json',dict(passed=True,repeat_scores_optimizer_equal=True,resume_scores_optimizer_equal=True,rng_equal=True,steps=30))
            result[0]=state
            write_json(work/'rates.json',dict(training_seconds=seconds,completed_target=target,updates=target-(initial['completed'] if initial else (checked_checkpoint(existing[-1],identity)['completed'] if existing else 0))))
            raise Finished()
        argv=['edge','--mib-path',str(MIB),'--model','gpt2','--task','ioi','--steps',str(target),'--k-schedule','uniform',
              '--masking','topk','--mode','iso','--lr','.05','--T','.5','--n_iters','30','--optimizer','adam','--seed',str(seed),
              '--cf-cache-gb','4','--batch-size','20','--no-wandb','--output',str(work/'unused_upstream_output')]
        spec=importlib.util.spec_from_file_location('gate5f_official',UPSTREAM/'scripts/mib/eval_mib_edge.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        try:
            with patch.object(HookedTransformer,'from_pretrained',model_load),patch.object(HFEAPDataset,'__init__',data_load),patch.object(package,'learn_scores',train_call),patch.object(torch.Tensor,'backward',backward),patch.object(sys,'argv',argv):
                try:module.main()
                except Finished:pass
            assert result[0] is not None
            after=weight_hash(meter.model);assert after==before[0]
            write_json(work/'model_weight_audit.json',dict(passed=True,before=before[0],after=after,model_optimizer_parameters=0))
        finally:close_meter(meter)
        if target==5000 or smoke:
            files=list(dest.glob('effective_scores_*.pt'))+[work/'model_weight_audit.json']
            # Marker hashes must be relative to their job root.
            write_json(dest/'model_weight_audit.json',json.loads((work/'model_weight_audit.json').read_text()))
            marker(dest,identity,list(dest.glob('effective_scores_*.pt'))+[dest/'model_weight_audit.json'])
        print('Completed',key,'updates',target,flush=True)
