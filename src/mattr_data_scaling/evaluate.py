"""Official frozen-ranking evaluation; signed capped masks and immutable outputs."""
import json
import time
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import ROOT,GRID,sha,write_json
from src.mattr_hierarchy.runner import weight_hash
from src.mattr_optimization.gate1 import LiveAccounting
from src.mattr_compact_validation.inference import assign_scores
from src.mattr_matched_size.topology import graph_new,digest
from src.mattr_matched_size.evaluation import OfficialEngine,ExactCache
from .core import protocol,endpoint_key,job_key,marker,verified,attempt,close_meter,configure_runtime,load_model,runtime_guard,CAPS

class HashedCache(ExactCache):
    """Identity plus byte integrity on every reused immutable per-mask output."""
    def get(self,mask_hash,compute):
        import hashlib
        identity={**self.identity,'mask_sha256':mask_hash}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        path=self.out/(key+'.npz');checksum=self.out/(key+'.npz.sha256')
        if path.exists() and not checksum.exists():
            import uuid
            incomplete=self.out/'incomplete';incomplete.mkdir(exist_ok=True)
            tag=uuid.uuid4().hex
            for old in (path,path.with_suffix('.json')):
                if old.exists():old.replace(incomplete/(old.name+'.'+tag))
            print('Preserved interrupted cache write; recomputing verified output',flush=True)
        if path.exists():
            if sha(path)!=checksum.read_text().strip():raise ValueError('Exact output cache hash mismatch; no unverified reuse')
            return super().get(mask_hash,compute)
        margin,top1=map(lambda x:np.asarray(x,dtype=np.float32),compute())
        assert margin.shape==top1.shape and np.isfinite(margin).all() and np.isfinite(top1).all()
        temporary=path.with_suffix('.npz.tmp')
        with temporary.open('wb') as handle:np.savez_compressed(handle,margin=margin,top1=top1)
        temporary.replace(path);write_json(path.with_suffix('.json'),identity)
        pending=checksum.with_suffix('.tmp');pending.write_text(sha(path)+'\n');pending.replace(checksum)
        self.misses+=1
        return margin.copy(),top1.copy()

def selected_prefixes(graph,scores,caps=CAPS):
    assign_scores(graph,scores);previous=np.zeros(32491,dtype=bool);counts=[0];hashes=[digest(previous)];masks=[previous]
    for n in range(1,32492):
        graph.apply_topn(n,absolute=False,prune=True,level='edge')
        current=graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()
        assert not np.any(previous&~current),'Pruning nestedness failure; stop'
        counts.append(int(current.sum()));hashes.append(digest(current));masks.append(current)
        previous=current
        if counts[-1]>max(caps):break
    else:raise ValueError('No prefix exceeds largest registered cap')
    rows=[];chosen={}
    for cap in caps:
        n=int(np.where(np.array(counts)<=cap)[0][-1]);mask=masks[n]
        rows.append(dict(target=cap,nominal_n=n,actual_edges=counts[n],shortfall=cap-counts[n],mask_sha256=hashes[n]))
        chosen[cap]=mask
    return pd.DataFrame(rows),chosen,pd.DataFrame(dict(nominal_n=range(len(counts)),actual_edges=counts,mask_sha256=hashes))

def ranking(out,method,n,seed,step):
    if method=='eap_ig':
        key=f'eap_ig_n128_s{seed}' if n==128 else 'eap_ig_n10000'
        folder=out/'attribution'/key
    else:
        key=endpoint_key(dict(method=method,dataset_size=n,seed=seed,step=step));folder=out/'training'/job_key(method,n,seed)
    path=folder/('scores.pt' if method=='eap_ig' else f'effective_scores_{step}.pt')
    if not path.exists():raise FileNotFoundError(f'Missing registered ranking: {path}')
    # Training at1000 can be an intermediate of an incomplete5000 job; validate checkpoint/export directly.
    complete=verified(folder)
    if complete:
        saved=json.loads((folder/'completed.json').read_text())
        assert saved['identity']['protocol_sha256']==sha(out/'gate5f_protocol.json'),'Ranking belongs to a different protocol'
    if not complete:
        receipt=folder/f'reuse_step{step}.json'
        if receipt.exists():
            r=json.loads(receipt.read_text());assert r['identity']['protocol_sha256']==sha(out/'gate5f_protocol.json')
            from .training import verify_historical
            record=verify_historical(out,ROOT/r['source'])
            if method=='h2':assert sha(path)==record.effective_score_file_sha256
            else:
                original=torch.load(ROOT/r['source'],map_location='cpu',weights_only=False)
                assert torch.equal(original['scores'],torch.load(path,map_location='cpu',weights_only=False))
            return key,path
        from .training import checked_checkpoint
        state=checked_checkpoint(folder/f'checkpoints/step_{step:06d}.pt')
        assert state['identity']['protocol_sha256']==sha(out/'gate5f_protocol.json')
        from src.mattr_hierarchy.adapter import Hierarchy
        adapter=Hierarchy(np.load(out/'H2_groups.npy')) if method=='h2' else Hierarchy(kind='bypass')
        torch.testing.assert_close(state['effective_scores'],torch.load(path,weights_only=False),rtol=0,atol=0)
    return key,path

def evaluate(out,method,n,seed,step,cohort,approved=False):
    if method=='eap_ig' and (step!=0 or (n==10000 and seed!=42)):raise ValueError('EAP uses step0 and a single full-data seed42 ranking')
    p=protocol(out)
    if method!='eap_ig' and dict(method=method,dataset_size=n,seed=seed,step=step) not in p['endpoints']:raise ValueError('Unregistered evaluation endpoint')
    key,path=ranking(out,method,n,seed,step)
    dest=out/'evaluations'/cohort/key;dest.mkdir(parents=True,exist_ok=True)
    identity=dict(protocol_sha256=sha(out/'gate5f_protocol.json'),scores_sha256=sha(path),cohort_sha256=sha(out/f'cohort_{cohort}.csv'),key=key)
    if verified(dest,identity):print('Verified cached evaluation:',cohort,key,flush=True);return
    runtime_guard(out,approved);configure_runtime()
    from .core import verify_runtime
    verify_runtime(out)
    from MIB_circuit_track.dataset import HFEAPDataset
    with attempt(out,'evaluate_'+cohort+'_'+key) as work:
        write_json(work/'configuration.json',identity);meter=LiveAccounting(work)
        try:
            model=load_model(meter,freeze=True);before=weight_hash(model)
            assert before==json.loads((out/'preflight_gpu.json').read_text())['model_weight_sha256']
            ds=HFEAPDataset('mib-bench/ioi',model.tokenizer,split='validation',task='ioi',model_name='gpt2')
            manifest=pd.read_csv(out/f'cohort_{cohort}.csv',keep_default_na=False)
            ds.dataset=ds.dataset.select(manifest.official_filtered_index.astype(int).tolist())
            for i,row in manifest.iterrows():
                clean,donor,labels=ds[i];assert (clean,donor)==(row.clean_prompt,row.corrupted_prompt)
                assert list(labels)==[row.correct_token,row.incorrect_token]
            scores=torch.load(path,map_location='cpu',weights_only=False);graph=graph_new()
            assert scores.shape==(32491,) and torch.isfinite(scores).all()
            start=time.perf_counter();caps,masks,topology=selected_prefixes(graph,scores,p['caps'])
            topology.to_csv(dest/'topology.csv',index=False)
            write_json(dest/'topology_time.json',dict(seconds=time.perf_counter()-start))
            engine=OfficialEngine(model,graph,ds,dest,sha(out/'gate5f_protocol.json'),identity['cohort_sha256'],20)
            engine.cache=HashedCache(dest/'exact_outputs',engine.cache.identity)
            start=time.perf_counter()
            print(f'{key}/{cohort}:10 nominal queries +10 capped queries,512 prompts,batch20; maximum1040 root forwards plus2 references before cache reuse',flush=True)
            with meter.stage('official_evaluation'),torch.no_grad():
                area,curve,per=engine.area(scores,GRID)
                measured=[];prompt=[]
                for row in caps.to_dict('records'):
                    result,margin,top1=engine.selected(scores,row['nominal_n'],row['mask_sha256'])
                    measured.append({**row,**result})
                    for i in range(len(ds)):prompt.append(dict(target=row['target'],example=i,template_id=manifest.template_id.iloc[i],
                        correct=int(margin[i]>0),margin=float(margin[i]),top1=int(top1[i])))
            seconds=time.perf_counter()-start
            assert weight_hash(model)==before and all(x.grad is None for x in model.parameters())
            assert meter.counts['backward_calls']==meter.counts['optimizer_updates']==0
            curve.to_csv(dest/'nominal.csv',index=False);per.to_csv(dest/'nominal_per_prompt.csv',index=False)
            pd.DataFrame(measured).to_csv(dest/'caps.csv',index=False);pd.DataFrame(prompt).to_csv(dest/'cap_per_prompt.csv',index=False)
            np.savez_compressed(dest/'cap_masks.npz',**{str(k):v for k,v in masks.items()})
            write_json(dest/'official_result.json',dict(CPR=area[1],CMD=area[2],accuracy_log_AUC=area[6],official_return=area,
                method=method,dataset_size=n,seed=seed,step=step,cohort=cohort,weight_hash=before,correctness_passed=True,
                cache_hits=engine.cache.hits,mask_cache_misses=engine.cache.misses,counts=meter.counts.copy()))
            write_json(work/'rates.json',dict(evaluation_seconds=seconds))
            files=[dest/x for x in ('nominal.csv','nominal_per_prompt.csv','caps.csv','cap_per_prompt.csv','cap_masks.npz','official_result.json','topology.csv')]
            marker(dest,identity,files)
        finally:close_meter(meter)
