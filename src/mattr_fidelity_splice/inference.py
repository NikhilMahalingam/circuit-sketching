"""Read-only recording at the pinned official evaluator's exact logit position."""
import copy
import json
from functools import partial
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch

from src.mattr_optimization.common import ROOT, GRID, configure, sha, write_json
from src.mattr_hierarchy.runner import weight_hash
from src.mattr_optimization.gate1 import LiveAccounting
from src.mattr_data_scaling.core import configure_runtime, load_model, close_meter
from src.mattr_matched_size.topology import graph_new, digest as mask_digest
from .core import HISTORY, ArrayCache, attempt, read, digest, protocol, identity, marker, verified, runtime_guard
from .design import selected, load_scores


def full_kl(clean_logp, circuit_logits):
    """Forward KL over every vocabulary item, float32, per example."""
    lp = torch.log_softmax(circuit_logits.float(),dim=-1)
    ref = torch.as_tensor(clean_logp,device=lp.device,dtype=torch.float32)
    assert ref.shape == lp.shape and torch.isfinite(ref).all() and torch.isfinite(lp).all()
    value = (ref.exp()*(ref-lp)).sum(-1)
    if not torch.isfinite(value).all() or (value < -1e-5).any():
        raise ValueError('Invalid full-vocabulary KL')
    return value


class Recorder:
    def __init__(self,model,manifest,reference=None,save_logp=False):
        from MIB_circuit_track.metrics import get_metric
        self.metric=partial(get_metric('logit_diff','ioi',model.tokenizer,model),mean=False,loss=False)
        self.model,self.manifest,self.reference,self.save_logp=model,manifest,reference,save_logp
        self.rows=[];self.logps=[];self.cursor=0

    def __call__(self,logits,clean_logits,lengths,labels):
        from MIB_circuit_track.metrics import get_logit_positions
        assert not torch.is_grad_enabled()
        final=get_logit_positions(logits,lengths).float()
        size=len(final);part=self.manifest.iloc[self.cursor:self.cursor+size]
        labs=torch.as_tensor(labels,device=final.device,dtype=torch.long)
        expected=torch.as_tensor(part[['correct_token','incorrect_token']].to_numpy(),device=final.device)
        assert torch.equal(labs,expected) and labs.min()>=0 and labs.max()<final.shape[-1]
        margin=self.metric(logits,clean_logits,lengths,labels)
        own=final.gather(1,labs);torch.testing.assert_close(margin,own[:,0]-own[:,1],rtol=0,atol=0)
        top=final.argmax(-1)
        arrays=dict(margin=margin.detach().cpu().numpy(),top1_correct=(top==labs[:,0]).cpu().numpy(),
            top1_id=top.cpu().numpy(),position=(lengths-1).cpu().numpy())
        if self.reference is not None:
            sl=slice(self.cursor,self.cursor+size);ref=self.reference
            arrays.update(kl=full_kl(ref['logp'][sl],final).cpu().numpy(),
                margin_error=np.abs(arrays['margin']-ref['margin'][sl]),
                margin_difference=arrays['margin']-ref['margin'][sl],
                sign_agreement=(np.sign(arrays['margin'])==np.sign(ref['margin'][sl])),
                top1_agreement=(arrays['top1_id']==ref['top1_id'][sl]))
            assert np.array_equal(arrays['position'],ref['position'][sl])
        if self.save_logp:self.logps.append(torch.log_softmax(final,-1).cpu().numpy())
        self.rows.append(arrays);self.cursor+=size
        return margin

    def arrays(self):
        assert self.cursor==len(self.manifest)
        result={k:np.concatenate([r[k] for r in self.rows]) for k in self.rows[0]}
        if self.save_logp:result['logp']=np.concatenate(self.logps)
        assert all(np.isfinite(v).all() for v in result.values())
        return result


def model_identity(model):
    tok=model.tokenizer
    config=json.loads(json.dumps(model.cfg.to_dict(),default=str))
    return dict(name='gpt2-small',weight_sha256=weight_hash(model),config_sha256=digest(config),
        config=config,tokenizer_class=type(tok).__name__,vocab_sha256=digest(tok.get_vocab()),
        tokenizer_sha256=digest(dict(vocab=tok.get_vocab(),special=tok.special_tokens_map,
            backend=tok.backend_tokenizer.to_str() if hasattr(tok,'backend_tokenizer') else tok.init_kwargs)),
        vocab_size=len(tok),position='official input_length-1',precision='float32')


def dataset(model,manifest):
    from MIB_circuit_track.dataset import HFEAPDataset
    ds=HFEAPDataset('mib-bench/ioi',model.tokenizer,split='validation',task='ioi',model_name='gpt2')
    ds.dataset=ds.dataset.select(manifest.official_filtered_index.astype(int).tolist())
    for i,row in manifest.iterrows():
        clean,donor,labels=ds[i]
        assert (clean,donor)==(row.clean_prompt,row.corrupted_prompt)
        assert list(labels)==[row.correct_token,row.incorrect_token]
    return ds


def position_identity(model,ds,batch):
    from eap.utils import tokenize_plus
    positions=[]
    for clean,donor,labels in ds.to_dataloader(batch):
        _,_,lengths,_=tokenize_plus(model,clean)
        _,_,corrupted_lengths,_=tokenize_plus(model,donor)
        assert torch.equal(lengths,corrupted_lengths)
        positions.extend((lengths-1).cpu().tolist())
    return dict(rule='official input_length-1',exact_position_sha256=digest(positions))


def clean_reference(model,ds,frame,cache,batch=20,validate=True):
    from eap.evaluate import evaluate_baseline
    def compute():
        rec=Recorder(model,frame,save_logp=True)
        with torch.no_grad():
            value=evaluate_baseline(model,ds.to_dataloader(batch),rec,quiet=True)
        arrays=rec.arrays();np.testing.assert_array_equal(value.numpy(),arrays['margin'])
        return arrays
    result=cache.get('clean_logprob',compute)
    if validate:
        small=copy.copy(ds);small.dataset=ds.dataset.select(range(min(2,len(ds))))
        rec=Recorder(model,frame.iloc[:len(small)].reset_index(drop=True),save_logp=True)
        with torch.no_grad():evaluate_baseline(model,small.to_dataloader(2),rec,quiet=True)
        fresh=rec.arrays()
        for k,values in fresh.items():np.testing.assert_allclose(values,result[k][:len(small)],rtol=2e-5,atol=1e-4)
    return result


class Engine:
    def __init__(self,model,graph,ds,frame,reference,cache,batch):
        from eap.evaluate import evaluate_graph
        self.model,self.graph,self.ds,self.frame=model,graph,ds,frame
        self.reference,self.cache,self.batch=reference,cache,batch
        self.raw=evaluate_graph
        self.loader=ds.to_dataloader(batch)
        from MIB_circuit_track.metrics import get_metric
        self.metric=partial(get_metric('logit_diff','ioi',model.tokenizer,model),mean=False,loss=False)

    def current(self):
        mask=self.graph.in_graph[self.graph.real_edge_mask.bool()].numpy().copy()
        self.graph.prune()
        assert np.array_equal(mask,self.graph.in_graph[self.graph.real_edge_mask.bool()].numpy())
        key=mask_digest(mask)
        def compute():
            rec=Recorder(self.model,self.frame,self.reference)
            with torch.no_grad():value=self.raw(self.model,self.graph,self.loader,rec,quiet=True,intervention='patching')
            arrays=rec.arrays();np.testing.assert_array_equal(value.numpy(),arrays['margin'])
            assert mask_digest(self.graph.in_graph[self.graph.real_edge_mask.bool()].numpy())==key
            return arrays
        return self.cache.get(key,compute)

    def area(self,scores):
        import MIB_circuit_track.evaluation as official
        from src.mattr_compact_validation.inference import assign_scores
        assign_scores(self.graph,scores)
        self.graph.apply_topn(0,absolute=False,prune=True)
        empty=self.current()
        self.empty=empty
        refs=dict(baseline=float(torch.from_numpy(self.reference['margin']).mean()),
                  corrupted=float(torch.from_numpy(empty['margin']).mean()))
        assert abs(refs['baseline']-refs['corrupted'])>1e-8
        def graph_call(model,graph,loader,metric,**kw):
            assert model is self.model and graph is self.graph and loader is self.loader and metric is self.metric
            assert kw.get('intervention')=='patching'
            return torch.from_numpy(self.current()['margin'].copy())
        with patch.object(official,'evaluate_graph',graph_call),torch.no_grad():
            result=official.evaluate_area_under_curve(self.model,self.graph,self.loader,self.metric,
                absolute=False,quiet=True,level='edge',intervention='patching',refs=refs)
        return result,refs


def summarize(arrays,reference):
    from scipy.stats import pearsonr,spearmanr
    m,clean=arrays['margin'],reference['margin']
    valid=np.abs(clean)>=.5;ratios=m[valid]/clean[valid]
    incorrect=clean<=0
    incorrect_vocab=~reference['top1_correct'].astype(bool)
    def correlation(fn):
        return float(fn(m,clean).statistic) if np.std(m)>0 and np.std(clean)>0 else None
    return dict(kl=float(arrays['kl'].mean()),accuracy=float((m>0).mean()),top1_accuracy=float(arrays['top1_correct'].mean()),
        margin=float(m.mean()),margin_error_mean=float(arrays['margin_error'].mean()),margin_error_median=float(np.median(arrays['margin_error'])),
        margin_difference=float((m-clean).mean()),amplification_median=float(np.median(ratios)) if len(ratios) else None,
        amplification_q25=float(np.quantile(ratios,.25)) if len(ratios) else None,amplification_q75=float(np.quantile(ratios,.75)) if len(ratios) else None,
        amplification_above2=float((ratios>2).mean()) if len(ratios) else None,amplification_above5=float((ratios>5).mean()) if len(ratios) else None,
        amplification_excluded=int((~valid).sum()),pearson=correlation(pearsonr),spearman=correlation(spearmanr),
        sign_agreement=float(arrays['sign_agreement'].mean()),top1_agreement=float(arrays['top1_agreement'].mean()),
        full_incorrect_n=int(incorrect.sum()),full_incorrect_accuracy=float((m[incorrect]>0).mean()) if incorrect.any() else None,
        full_incorrect_sign_agreement=float(arrays['sign_agreement'][incorrect].mean()) if incorrect.any() else None,
        full_vocab_incorrect_n=int(incorrect_vocab.sum()),
        full_vocab_incorrect_IOI_accuracy=float((m[incorrect_vocab]>0).mean()) if incorrect_vocab.any() else None,
        full_vocab_incorrect_sign_agreement=float(arrays['sign_agreement'][incorrect_vocab].mean()) if incorrect_vocab.any() else None)


def gpu_preflight(out):
    """Two historical prompts only. No cohort_5G model output is requested here."""
    configure_runtime()
    frame=pd.read_csv(HISTORY/'cohort_primary.csv',keep_default_na=False).iloc[:2].reset_index(drop=True)
    fresh=pd.read_csv(out/'cohort_hashes.csv')
    seen=set(frame.clean_sha256)|set(frame.corrupted_sha256)
    assert not seen & (set(fresh.clean_sha256)|set(fresh.corrupted_sha256))
    # Record prompt identities BEFORE the first forward, including interrupted attempts.
    frame.drop(columns=['clean_prompt','corrupted_prompt','official_template']).to_csv(out/'pre_freeze_test_prompts.csv',index=False)
    with attempt(out,'bounded_historical_correctness') as work:
        meter=LiveAccounting(work)
        try:
            model=load_model(meter,freeze=True);before=weight_hash(model);ident=model_identity(model)
            assert before==read(HISTORY/'preflight_gpu.json')['model_weight_sha256'],'Model differs from historical trained-ranking reference'
            ds=dataset(model,frame);graph=graph_new()
            context=dict(model=ident,cohort_sha256=sha(out/'pre_freeze_test_prompts.csv'),precision='float32',
                position=position_identity(model,ds,2),implementation='historical-correctness-v1')
            cache=ArrayCache(work/'cache',context)
            with meter.stage('bounded_clean_cache'),torch.no_grad():
                ref=clean_reference(model,ds,frame,cache,batch=2)
                again=clean_reference(model,ds,frame,cache,batch=2,validate=False)
                for k in ref:np.testing.assert_array_equal(ref[k],again[k])
            engine=Engine(model,graph,ds,frame,ref,ArrayCache(work/'masks',context),2)
            rec=next(r for r in read(out/'ranking_catalog.json') if r['kind']=='core' and r['method']=='baseline')
            scores=load_scores(rec)
            with meter.stage('bounded_official_grid'),torch.no_grad():
                torch.cuda.synchronize();start=time.perf_counter()
                area,refs=engine.area(scores)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
            full=selected(graph,scores,32491);assert full.all()
            with meter.stage('bounded_batch_cache_checks'),torch.no_grad():
                full_output=engine.current()
                np.testing.assert_allclose(full_output['margin'],ref['margin'],atol=1e-4,rtol=2e-5)
                np.testing.assert_allclose(full_output['kl'],0,atol=1e-6)
                count=meter.counts['forward_batches'];engine.current()
                assert meter.counts['forward_batches']==count
                query=pd.read_csv(out/'queries.csv')
                q=query[(query.ranking==rec['key'])&(query.selection=='cap')&(query.target==116)].iloc[0]
                selected(graph,scores,q.nominal_n)
                paired=engine.current()
                singleton=Engine(model,graph,ds,frame,ref,ArrayCache(work/'batch1',context),1).current()
                for k in paired:np.testing.assert_allclose(paired[k],singleton[k],rtol=3e-5,atol=2e-4)
                # Independent unwrapped official evaluator verifies original CPR/CMD and interventions.
                from MIB_circuit_track.evaluation import evaluate_area_under_curve
                from src.mattr_compact_validation.inference import assign_scores
                assign_scores(graph,scores)
                raw=evaluate_area_under_curve(model,graph,ds.to_dataloader(2),engine.metric,absolute=False,quiet=True,refs={})
                np.testing.assert_allclose(np.array(area[1:4]),np.array(raw[1:4]),rtol=2e-5,atol=1e-5)
                independent=Recorder(model,frame)
                selected(graph,scores,q.nominal_n)
                engine.raw(model,graph,ds.to_dataloader(2),independent,quiet=True,intervention='patching')
                np.testing.assert_array_equal(paired['margin'],independent.arrays()['margin'])
            assert before==weight_hash(model) and all(x.grad is None and not x.requires_grad for x in model.parameters())
            assert meter.counts['optimizer_updates']==meter.counts['backward_calls']==0
            # Numerical independent KL oracle, with real CPU float32 tensors and probabilities.
            a=torch.tensor([[.2,-.7,1.8],[4.,-2.,.1]])
            b=torch.tensor([[-.4,.8,1.2],[2.,1.,-3.]])
            lp=torch.log_softmax(a,-1)
            actual=full_kl(lp,b)
            expected=torch.nn.functional.kl_div(torch.log_softmax(b,-1),lp,reduction='none',log_target=True).sum(-1)
            torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)
            write_json(out/'preflight_gpu.json',dict(passed=True,model=ident,historical_prompt_count=2,fresh_cohort_forwards=0,
                full_circuit_clean=True,official_CPR_CMD_equivalent=True,unchanged_patching=True,batch_consistent=True,
                clean_cache_validated=True,cache_hits_no_forwards=True,weights_unchanged=True,no_gradients=True,
                KL_max_synthetic_error=float((actual-expected).abs().max()),
                seconds_per_query_projection=elapsed/10*(512/2)*1.25,
                bounded_counts=meter.counts.copy(),pre_freeze_prompt_hash=sha(out/'pre_freeze_test_prompts.csv')))
            record=read(out/'preflight_gpu.json')
            record['tested_sources']={str(path.relative_to(ROOT)).replace('\\','/'):sha(path)
                for path in [ROOT/'src/mattr_fidelity_splice/inference.py',ROOT/'src/mattr_fidelity_splice/core.py']}
            write_json(out/'preflight_gpu.json',record)
        finally:close_meter(meter)


def evaluate(out,job_id,approved=False):
    p=protocol(out,live=True)
    job=next((j for j in p['jobs'] if j['id']==job_id),None)
    if job is None:raise ValueError('Unregistered job')
    ident=identity(out,p,job);dest=out/'evaluations'/job_id;dest.mkdir(parents=True,exist_ok=True)
    if verified(dest,ident):print('Verified completed:',job_id,flush=True);return
    runtime_guard(out,p,approved);configure_runtime()
    rec=next(r for r in p['rankings'] if r['key']==job['ranking'])
    scores=load_scores(rec)
    queries=pd.read_csv(out/'queries.csv');queries=queries[queries.ranking==rec['key']]
    assert digest(queries.to_dict('records'))==job['queries_sha256']
    frame=pd.read_csv(out/'cohort_manifest.csv',keep_default_na=False)
    print(f'{job_id}: {len(queries)} queries, {len(frame)} prompts, batch20. Distinct mask outputs reused by verified identity; zero training.',flush=True)
    with attempt(out,job_id) as work:
        write_json(work/'configuration.json',ident)
        meter=LiveAccounting(work)
        try:
            model=load_model(meter,freeze=True);before=weight_hash(model)
            actual=model_identity(model)
            assert actual==p['model'],'Model/tokenizer/config differs from freeze'
            ds=dataset(model,frame);graph=graph_new()
            context=dict(model=actual,cohort_sha256=p['cohort_sha256'],batch=20,precision='float32',
                implementation_commit=p['implementation_commit'],protocol_sha256=ident['protocol_sha256'],
                intervention='pinned official patching',position=position_identity(model,ds,20))
            clean_cache=ArrayCache(out/'cache/clean',context)
            with meter.stage('clean_reference_cache_validation'),torch.no_grad():
                reference=clean_reference(model,ds,frame,clean_cache,batch=20)
            cache=ArrayCache(out/'cache/circuits',context)
            engine=Engine(model,graph,ds,frame,reference,cache,20)
            rows=[];per=[]
            torch.cuda.synchronize();start=time.perf_counter()
            with meter.stage('registered_evaluation'),torch.no_grad():
                area,refs=engine.area(scores) if job['part']!='B' else (None,None)
                for query in queries.to_dict('records'):
                    mask=selected(graph,scores,query['nominal_n'])
                    assert mask_digest(mask)==query['mask_sha256']
                    arrays=engine.current()
                    if query['selection']=='nominal' and query['target']==1:
                        np.testing.assert_allclose(arrays['margin'],reference['margin'],atol=1e-4,rtol=2e-5)
                        np.testing.assert_allclose(arrays['kl'],0,atol=1e-5)
                    rows.append({**query,**summarize(arrays,reference)})
                    for i in range(len(frame)):
                        per.append({**query,'example':i,'template_id':frame.template_id.iloc[i],
                            **{k:v[i].item() for k,v in arrays.items()},'clean_margin':reference['margin'][i].item(),
                            'clean_top1_id':int(reference['top1_id'][i]),'incorrect_token':int(frame.incorrect_token.iloc[i]),
                            'correct_token':int(frame.correct_token.iloc[i]),
                            'corrupted_margin':float(engine.empty['margin'][i]) if area is not None else None})
            torch.cuda.synchronize();seconds=time.perf_counter()-start
            if area is not None:
                from .statistics import official_areas
                curve=pd.DataFrame(rows);margins=np.array([next(r for r in rows if r['selection']=='nominal' and r['target']==f)['margin'] for f in GRID])
                check=official_areas(margins,refs['baseline'],refs['corrupted'])
                np.testing.assert_allclose(check[:2],area[1:3],atol=2e-5,rtol=2e-5)
            assert before==weight_hash(model) and all(x.grad is None and not x.requires_grad for x in model.parameters())
            assert meter.counts['backward_calls']==meter.counts['optimizer_updates']==0
            pd.DataFrame(rows).to_csv(dest/'metrics.csv',index=False)
            pd.DataFrame(per).to_csv(dest/'per_prompt.csv',index=False)
            write_json(dest/'summary.json',dict(identity=ident,official_result=area,refs=refs,
                evaluation_seconds=seconds,requested_masks=len(queries),cache_hits=cache.hits,cache_misses=cache.misses,
                clean_cache_hits=clean_cache.hits,counts=meter.counts.copy(),weight_before=before,weight_after=before,
                correctness_passed=True,model_gradients_stored=False))
            marker(dest,ident,[dest/'metrics.csv',dest/'per_prompt.csv',dest/'summary.json'])
        finally:close_meter(meter)
