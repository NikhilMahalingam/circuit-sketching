"""Pinned official MAttr loop; only score parameterization and penalty change."""
import argparse
import hashlib
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
from src.mattr_optimization.common import ROOT,UPSTREAM,MIB,configure,provenance,sha,write_json
from src.mattr_optimization.gate2 import NestedData,archive_attempt
from src.mattr_optimization.gate3 import baseline
from src.mattr_optimization.gate1 import LiveAccounting
from src.mattr_optimization.gate0 import check_patching
from src.mattr_optimization.gate4_init import edge_mapping
from src.mattr_optimization.gate4_train import evaluate_scores
from src.mattr_optimization.checkpoint import rng_state,restore_rng
from .adapter import Hierarchy,group_sum,adam_update_contribution
from .checkpoint import checkpointed

def weight_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def tensor_state_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tensor_state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(tensor_state_equal(x,y) for x,y in zip(a,b))
    return a==b

def run(a):
    configure()
    from transformer_lens import HookedTransformer
    from MIB_circuit_track.dataset import HFEAPDataset
    import MIB_circuit_track.evaluation as evaluation
    import matryoshka_attribution as package
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no automatic CPU/OOM fallback.')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);campaign=a.protocol.parent
    proto=json.loads(a.protocol.read_text())
    for name,h in proto['source_sha256'].items():assert sha(ROOT/name)==h,name
    prov=provenance();assert prov==json.loads((baseline(a.seed)/'experiment_config.json').read_text())['source_provenance']
    ids_path=campaign/f'{a.family}{"_shuffled" if a.shuffled else ""}_groups.npy'
    ids=np.load(ids_path) if a.family in ('H1','H2') else None
    kind='hierarchy' if ids is not None else a.family
    config={'family':a.family,'gamma':a.gamma,'scale':a.scale,'seed':a.seed,'shuffled':a.shuffled,
            'phase':a.phase,'pool':128,'protocol_sha256':sha(a.protocol),'group_hash':sha(ids_path) if ids is not None else None,
            'source_provenance':prov,'model_weight_updates':False,'model_gradient_flags':'unchanged official baseline; only attribution parameters enter Adam'}
    if (out/'experiment_config.json').exists():assert json.loads((out/'experiment_config.json').read_text())==config
    archive_attempt(out);write_json(out/'experiment_config.json',config);identity=json.dumps(config,sort_keys=True)
    meter=LiveAccounting(out);data=NestedData(out,128,a.seed,128);success=False
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original_load,original_data=HookedTransformer.from_pretrained,HFEAPDataset.__init__
    original_train,official_eval=package.learn_scores,evaluation.evaluate_area_under_curve
    old_backward=torch.Tensor.backward
    with meter.stage('group_loading'):
        adapter=Hierarchy(ids,a.gamma,a.scale,kind=kind,device='cuda')
    before_hash=None
    def load(*args,**kw):
        nonlocal before_hash
        with meter.stage('model_loading'):
            model=meter.attach(original_load(*args,**kw));assert model.cfg.dtype==torch.float32
            from eap.graph import Graph
            actual=edge_mapping(Graph.from_model(model),Graph.from_model(model))
            pd.testing.assert_frame_equal(actual,pd.read_csv(campaign/'edge_mapping.csv',keep_default_na=False))
            before_hash=weight_hash(model);return model
    def dataset_init(ds,*args,**kw):
        with meter.stage('data_loading_'+kw['split']):
            original_data(ds,*args,**kw);data.select(ds,kw['split'])
            pd.testing.assert_frame_equal(pd.read_csv(out/f'{kw["split"]}_manifest.csv'),pd.read_csv(baseline(a.seed)/f'{kw["split"]}_manifest.csv'))
    class PilotComplete(Exception):pass
    def pilot(loss_fn):
        from matryoshka_attribution.masks import build_mask
        with meter.stage('official_patching_checks'):check_patching(loss_fn,data,meter.model,out)
        score=torch.load(baseline(42)/'checkpoints/step_000500.pt',map_location='cuda',weights_only=False)['scores'].requires_grad_()
        rows=[];transforms={family:Hierarchy(np.load(campaign/f'{family}_groups.npy'),kind='hierarchy',device='cuda') for family in ('H1','H2')}
        for fraction in (.005,.01,.02,.05,.5):
            for index in range(4):
                mask=build_mask(score,float(int(32491*fraction)),'topk',T=.5,n_iters=30).mask
                with patch.object(random,'randint',return_value=index):loss=loss_fn(mask)
                grad=torch.autograd.grad(loss,score)[0];meter.counts['backward_calls']+=1
                for family in ('H1','H2'):
                    transform=transforms[family]
                    with torch.no_grad():
                        means=group_sum(score,transform.padded,transform.valid)/transform.sizes
                        centered=score-means[transform.ids]
                        packed=torch.cat([means,centered]);eff,r=transform.parts(packed)
                        gm=torch.cat([group_sum(grad,transform.padded,transform.valid),grad-(group_sum(grad,transform.padded,transform.valid)/transform.sizes)[transform.ids]])
                        for gamma in (.001,.01):
                            penalty=gamma*r.square().mean();greg=2*gamma*r/32491
                            rows.append({'family':family,'gamma':gamma,'fraction':fraction,'example':index,
                               'mattr_loss':float(loss),'mattr_abs_loss':abs(float(loss)),'penalty':float(penalty),
                               'mattr_gradient_norm':float(gm.double().norm()),'regularizer_gradient_norm':float(greg.double().norm()),
                               'gradient_ratio':float(greg.double().norm()/gm.double().norm().clamp_min(1e-30)),
                               'loss_ratio':float(penalty/abs(loss).clamp_min(1e-12)),'residual_variance':float(r.var())})
        pd.DataFrame(rows).to_csv(out/'magnitude_probe.csv',index=False)
        raise PilotComplete()
    def train(total,loss_fn,**kw):
        assert total==32491 and kw['optimizer']=='adam' and kw['k_schedule']=='uniform' and 'k_sampler' not in kw
        if a.phase=='pilot':
            with meter.stage('training_only_magnitude_probe'):pilot(loss_fn)
        checkpoint_paths=sorted((out/'checkpoints').glob('step_*.pt'))
        offset=torch.load(checkpoint_paths[-1],map_location='cpu',weights_only=False)['completed'] if checkpoint_paths else 0
        if a.phase=='smoke':
            with meter.stage('official_patching_checks'):check_patching(loss_fn,data,meter.model,out)
            state=rng_state();options={**kw,'on_step':None,'log_every':0,'steps':30}
            with meter.stage('original_reference_30'):
                reference=original_train(total,loss_fn,**options);meter.counts['optimizer_updates']+=30
            expected_rng=rng_state();restore_rng(state)
            with meter.stage('dedicated_bypass_30'),Hierarchy(kind='bypass').install():
                replay=original_train(total,loss_fn,**options);meter.counts['optimizer_updates']+=30
            assert torch.equal(reference.scores,replay.scores) and reference.train_log==replay.train_log
            assert tensor_state_equal(expected_rng,rng_state())
            write_json(out/'bypass_equivalence.json',{'passed':True,'raw_scores_bitwise_equal':True,'rng_and_logs_equal':True,
              'scope':'dedicated bypass vs original; gamma-zero hierarchy is deliberately NOT claimed identical'})
            restore_rng(state)
        randint=random.randint;draws=[];diagnostics=[];last_losses={}
        def sampled(mask):
            def counted(lo,hi):
                i=randint(lo,hi);draws.append(i);return i
            with patch.object(random,'randint',counted):loss=loss_fn(mask)
            if loss is None:return None
            if kind=='bypass':
                last_losses.update(mattr=float(loss.detach()),penalty=0.);return loss
            penalty=adapter.penalty();last_losses.update(mattr=float(loss.detach()),penalty=float(penalty.detach()))
            return loss+penalty
        def inspect_step(parameter,optimizer):
            step=offset+len(meter.grad_norms)
            assert id(parameter) not in {id(p) for p in meter.model.parameters()}
            if step==1 or step%50==0:
                with torch.no_grad():
                    reg=torch.zeros_like(parameter) if kind=='bypass' else adapter.penalty_gradient()
                    gm=parameter.grad.detach()-reg
                    effective=parameter if kind=='bypass' else adapter.effective
                    residual=parameter if kind=='bypass' else adapter.centered
                    h=parameter[:adapter.group_count] if adapter.group_count else torch.zeros(1,device=parameter.device)
                    diagnostics.append({'step':step,'mattr_loss':last_losses['mattr'],'regularization_loss':last_losses['penalty'],
                        'mattr_gradient_norm':float(gm.double().norm()),'regularizer_gradient_norm':float(reg.double().norm()),
                        'gradient_ratio':float(reg.double().norm()/gm.double().norm().clamp_min(1e-30)),
                        'adam_update_regularizer_relative_contribution':adam_update_contribution(parameter,optimizer,reg),
                        'residual_variance':float(residual.var()),'component_score_variance':float(h.var(unbiased=False)),
                        'effective_score_variance':float(effective.var()),'total_gradient_norm':float(parameter.grad.detach().double().norm())})
                    assert torch.isfinite(parameter.grad).all()
        external=kw.pop('on_step',None)
        def completed(step,k,loss,scores):
            if external:external(step,k,loss,scores)
            if step==0 or (step+1)%100==0 or step+1==a.target:
                print(f'{a.family} shuffled={a.shuffled} gamma={a.gamma} seed={a.seed} {step+1}/{a.target}',flush=True)
                meter.status(completed_iteration=step+1,target=a.target,family=a.family,gamma=a.gamma,seed=a.seed)
        initial_rng=rng_state()
        with meter.stage('training'),adapter.install():
            result=checkpointed(original_train,total,sampled,folder=out/'checkpoints',identity=identity,
                 interval=100 if a.target>30 else 10,on_update=meter.update,parameter_count=adapter.count(total),
                 inspect_step=inspect_step,on_step=completed,**kw)
        if a.phase=='smoke':
            final_rng=rng_state();target_result=result.scores.clone()
            restore_rng(initial_rng)
            options={**kw,'log_every':0}
            with meter.stage('exact_resume_15_plus_15'),adapter.install():
                for target in (15,30):
                    options['steps']=target
                    resumed=checkpointed(original_train,total,loss_fn if kind=='bypass' else lambda mask:loss_fn(mask)+adapter.penalty(),
                        folder=out/'resume_test',identity=identity,interval=15,on_update=meter.update,
                        parameter_count=adapter.count(total),**options)
            astate=torch.load(out/'checkpoints/step_000030.pt',map_location='cpu',weights_only=False)
            bstate=torch.load(out/'resume_test/step_000030.pt',map_location='cpu',weights_only=False)
            assert torch.equal(target_result,resumed.scores)
            assert tensor_state_equal(astate['optimizer'],bstate['optimizer']) and tensor_state_equal(final_rng,rng_state())
            write_json(out/'exact_resume.json',{'passed':True,'scores_optimizer_rng_bitwise_equal':True,'steps':30,'split':15})
        assert len(result.train_log)==a.target
        old=pd.read_csv(baseline(a.seed)/'training_example_draws.csv').subset_index.iloc[offset:a.target].tolist()
        assert draws==old,'Paired row stream changed'
        prior=torch.load(baseline(a.seed)/f'checkpoints/step_{100 if a.target==30 else a.target:06d}.pt',map_location='cpu',weights_only=False)
        assert result.k_log==[row[1] for row in prior['train_log'][:a.target]],'Uniform k stream changed'
        if kind=='bypass' and a.target==1000:assert torch.equal(result.scores,prior['scores']),'Dedicated C0 differs from Gate2'
        pd.DataFrame({'step':range(offset+1,a.target+1),'subset_index':draws}).to_csv(out/f'example_draws_{a.target}.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(out/f'gradient_diagnostics_{a.target}.csv',index=False)
        pd.DataFrame(result.train_log,columns=['step','k','fraction','loss']).to_csv(out/f'training_metrics_{a.target}.csv',index=False)
        torch.save(result.scores,out/f'original_parameters_{a.target}.pt')
        with torch.no_grad():result.scores=adapter.transform(result.scores.to('cuda')).detach().cpu()
        torch.save(result.scores,out/f'effective_scores_{a.target}.pt');return result
    def evaluate(model,graph,loader,metric,**kw):
        results={}
        for step in (30,500,1000,2500,5000):
            if step>a.target or (step==30 and a.phase!='smoke'):continue
            state=torch.load(out/f'checkpoints/step_{step:06d}.pt',map_location='cpu',weights_only=False)
            with torch.no_grad():effective=adapter.transform(state['scores'].to('cuda')).cpu()
            ident={'config':identity,'checkpoint_sha256':sha(out/f'checkpoints/step_{step:06d}.pt'),
              'validation_manifest_sha256':sha(out/'validation_manifest.csv'),'ranking':'effective centered scores'}
            r=evaluate_scores(model,graph,data.datasets['validation'],metric,effective,out,step,meter,ident,official_eval)
            results[step]=r;write_json(out/f'quality_{step}.json',{'family':a.family,'shuffled':a.shuffled,'gamma':a.gamma,'seed':a.seed,'step':step,
              'cpr':r[1],'cmd':r[2],'accuracy_auc':r[6]})
        return results[a.target]
    def backward(tensor,*args,**kw):
        result=old_backward(tensor,*args,**kw);meter.counts['backward_calls']+=1;return result
    argv=['eval_mib_edge.py','--mib-path',str(MIB),'--model','gpt2','--task','ioi','--steps',str(a.target),
      '--k-schedule','uniform','--masking','topk','--mode','iso','--lr','.05','--optimizer','adam',
      '--split','validation','--train-split','train','--batch-size','20','--seed',str(a.seed),'--cf-cache-gb','4','--no-wandb','--output',str(out/f'upstream_{a.target}')]
    write_json(out/'upstream_command.json',{'argv':argv,'official_loss':'negative one-example IOI logit difference; no batch/edge mean'})
    spec=importlib.util.spec_from_file_location('hierarchy_official_edge',UPSTREAM/'scripts/mib/eval_mib_edge.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        with patch.object(HookedTransformer,'from_pretrained',load),patch.object(HFEAPDataset,'__init__',dataset_init),\
             patch.object(package,'learn_scores',train),patch.object(evaluation,'evaluate_area_under_curve',evaluate),\
             patch.object(torch.Tensor,'backward',backward):
            with patch.object(sys,'argv',argv):
                try:module.main()
                except PilotComplete:
                    if a.phase!='pilot':raise
        after=weight_hash(meter.model);assert after==before_hash
        write_json(out/'model_weight_audit.json',{'passed':True,'before_sha256':before_hash,'after_sha256':after,
            'in_optimizer':False,'updated_model_parameters':0,'gradient_flags':'official unchanged; gradients are not weight updates'})
        success=True
    except BaseException as exc:
        import traceback
        write_json(out/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'oom':isinstance(exc,torch.cuda.OutOfMemoryError),
          'advice':'Settings unchanged. Inspect failure; a lower batch size requires a new explicit configuration.'});raise
    finally:
        meter.stop.set();meter.worker.join();meter.save();meter.stream.close()
        write_json(out/'process_summary.json',{'success':success,'target':a.target,'seconds':time.perf_counter()-meter.started,'counts':meter.counts})
    write_json(out/f'completed_{a.target}.json',{'passed':True,'phase':a.phase,'config_sha256':sha(out/'experiment_config.json'),
       'effective_sha256':sha(out/f'effective_scores_{a.target}.pt') if a.phase!='pilot' else None})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--family',choices=('H1','H2','bypass','edge'),required=True);p.add_argument('--gamma',type=float,default=0.)
    p.add_argument('--scale',type=float,default=1.);p.add_argument('--seed',type=int,choices=(42,43,44),default=42)
    p.add_argument('--target',type=int,choices=(30,1000,5000),default=1000);p.add_argument('--shuffled',action='store_true')
    p.add_argument('--phase',choices=('pilot','smoke','train'),default='train');run(p.parse_args())
