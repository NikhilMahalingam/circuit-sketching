"""Official edge CLI adapter: raw checkpoints, effective masks/rankings, same RNG."""
import argparse
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
from scipy.stats import spearmanr
from src.mattr_optimization.common import ROOT,UPSTREAM,MIB,configure,provenance,sha,write_json
from src.mattr_optimization.gate2 import NestedData,archive_attempt
from src.mattr_optimization.gate3 import baseline
from src.mattr_optimization.gate1 import LiveAccounting
from src.mattr_optimization.gate0 import check_patching
from src.mattr_optimization.checkpoint import checkpointed,rng_state,restore_rng
from src.mattr_optimization.gate4_train import evaluate_scores
from .coupling import Coupling


def run(a):
    configure()
    from transformer_lens import HookedTransformer
    from MIB_circuit_track.dataset import HFEAPDataset
    import MIB_circuit_track.evaluation as evaluation
    import matryoshka_attribution as package
    if not torch.cuda.is_available():raise RuntimeError('RTX/CUDA required; settings will not change silently.')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    proto=json.loads(a.protocol.read_text());campaign=a.protocol.parent
    if sha(a.protocol)!=(campaign/'gate5a_protocol.sha256').read_text().strip():raise ValueError('Frozen protocol changed.')
    for name,h in proto['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Implementation changed after protocol freeze: '+name)
    prov=provenance()
    if prov!=json.loads((baseline(a.seed)/'experiment_config.json').read_text())['source_provenance']:
        raise ValueError('Pinned baseline runtime/source differs.')
    graph_path=campaign/'graphs'/f'{a.family}_neighbors.npy'
    graph_sha=sha(graph_path) if a.family not in ('zero','scale') else None
    if graph_sha and graph_sha!=proto['graph_files'][a.family]:raise ValueError('Graph changed after freeze.')
    if sha(campaign/'edge_mapping.csv')!=proto['mapping_sha256']:raise ValueError('Canonical real-edge mapping changed.')
    config={'gate':'5A','family':a.family,'beta':a.beta,'seed':a.seed,'pool':128,
            'protocol_sha256':sha(a.protocol),'graph_file_sha256':graph_sha,
            'mapping_sha256':proto['mapping_sha256'],'source_provenance':prov}
    if (out/'experiment_config.json').exists() and json.loads((out/'experiment_config.json').read_text())!=config:
        raise ValueError('Resume identity/graph differs.')
    archive_attempt(out);write_json(out/'experiment_config.json',config);identity=json.dumps(config,sort_keys=True)
    meter=LiveAccounting(out);data=NestedData(out,128,a.seed,128);success=False
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original_load,original_data=HookedTransformer.from_pretrained,HFEAPDataset.__init__
    original_train,official_eval=package.learn_scores,evaluation.evaluate_area_under_curve
    original_backward=torch.Tensor.backward
    with meter.stage('graph_loading'):
        indices=torch.tensor(np.load(graph_path),dtype=torch.long,device='cuda') if graph_sha else None
    mixer=Coupling(indices,a.beta,kind='scale' if a.family=='scale' else 'zero' if a.family=='zero' else 'graph')
    old=sorted((out/'checkpoints').glob('step_*.pt'))
    offset=torch.load(old[-1],weights_only=False,map_location='cpu')['completed'] if old else 0
    def load(*args,**kw):
        with meter.stage('model_loading'):
            model=meter.attach(original_load(*args,**kw))
            if model.cfg.dtype!=torch.float32:raise ValueError('Float32 recipe required.')
            from eap.graph import Graph
            from src.mattr_optimization.gate4_init import edge_mapping
            actual=edge_mapping(Graph.from_model(model),Graph.from_model(model))
            pd.testing.assert_frame_equal(actual,pd.read_csv(campaign/'edge_mapping.csv',keep_default_na=False))
            return model
    def dataset_init(ds,*args,**kw):
        with meter.stage('data_loading_'+kw['split']):
            original_data(ds,*args,**kw);data.select(ds,kw['split'])
            pd.testing.assert_frame_equal(pd.read_csv(out/f'{kw["split"]}_manifest.csv'),pd.read_csv(baseline(a.seed)/f'{kw["split"]}_manifest.csv'))
    def train(total,loss_fn,**kw):
        if total!=32491 or kw['k_schedule']!='uniform' or kw['optimizer']!='adam' or 'k_sampler' in kw:
            raise ValueError('All real edges and unchanged official optimizer/sampling required.')
        if a.smoke:
            model=meter.model
            with meter.stage('official_patching_checks'):check_patching(loss_fn,data,model,out)
            state=rng_state();options={**kw,'on_step':None,'log_every':0}
            with meter.stage('beta0_reference_30'):
                reference=original_train(total,loss_fn,**options)
                meter.counts['optimizer_updates']+=len(reference.train_log)
            expected_rng=rng_state();restore_rng(state)
            with meter.stage('beta0_identity_30'),Coupling(indices,0).install():
                identity_run=original_train(total,loss_fn,**options)
                meter.counts['optimizer_updates']+=len(identity_run.train_log)
            actual_rng=rng_state()
            if not torch.equal(reference.scores,identity_run.scores) or reference.train_log!=identity_run.train_log:
                raise ValueError('Beta-zero GPU score/stream equivalence failed.')
            if not torch.equal(expected_rng['torch'],actual_rng['torch']) or expected_rng['python']!=actual_rng['python']:
                raise ValueError('Beta-zero RNG stream differs.')
            if len(expected_rng['cuda'])!=len(actual_rng['cuda']) or not all(torch.equal(x,y) for x,y in zip(expected_rng['cuda'],actual_rng['cuda'])):
                raise ValueError('Beta-zero CUDA RNG differs.')
            if not np.array_equal(expected_rng['numpy'][1],actual_rng['numpy'][1]):raise ValueError('Beta-zero NumPy RNG differs.')
            write_json(out/'beta_zero_equivalence.json',{'passed':True,'steps':30,'scores_bitwise_equal':True,
                       'loss_k_logs_equal':True,'torch_python_rng_equal':True,'cuda_numpy_rng_equal':True,
                       'timing_caveat':'reference uses colder donor cache; not an overhead benchmark'})
            restore_rng(state)
        draws=[];randint=random.randint;diagnostics=[];previous_scores=None
        def sampled_loss(mask):
            def counted(lo,hi):
                i=randint(lo,hi);draws.append(i);return i
            with patch.object(random,'randint',counted):return loss_fn(mask)
        def update(norm):
            meter.update(norm);step=offset+len(meter.grad_norms)
            if step==1 or step%50==0:
                raw=mixer.raw;eff=mixer.effective
                if raw.grad is None or eff.grad is None or not torch.isfinite(raw.grad).all() or not torch.isfinite(eff.grad).all():
                    raise ValueError('Missing/nonfinite score gradient.')
                top=torch.topk(eff.detach(),max(1,int(.01*total))).indices
                diagnostics.append({'step':step,'raw_gradient_norm':norm,'effective_gradient_norm':float(eff.grad.detach().double().norm()),
                    'raw_score_variance':float(raw.detach().var()),'effective_score_variance':float(eff.detach().var()),
                    'raw_gradient_top_score_fraction':float(raw.grad[top].abs().sum()/raw.grad.abs().sum().clamp_min(1e-30)),
                    'effective_gradient_top_score_fraction':float(eff.grad[top].abs().sum()/eff.grad.abs().sum().clamp_min(1e-30))})
        external=kw.pop('on_step',None)
        def completed(step,k,loss,scores):
            nonlocal previous_scores
            if external:external(step,k,loss,scores)
            if step==0 or (step+1)%50==0:
                with torch.no_grad():effective=mixer(scores).detach().cpu()
                if previous_scores is not None:
                    diagnostics[-1]['rank_spearman_since_previous']=float(spearmanr(effective.numpy(),previous_scores.numpy()).statistic)
                    x=set(torch.topk(effective,324).indices.tolist());y=set(torch.topk(previous_scores,324).indices.tolist())
                    diagnostics[-1]['top1pct_jaccard_since_previous']=len(x&y)/len(x|y)
                previous_scores=effective
            if step==0 or (step+1)%100==0 or step+1==a.target:
                meter.status(completed_iteration=step+1,target=a.target,family=a.family,beta=a.beta,seed=a.seed)
                print(f'{a.family} beta={a.beta} seed={a.seed}: {step+1}/{a.target}',flush=True)
        with meter.stage('training'),mixer.install():
            result=checkpointed(original_train,total,sampled_loss,folder=out/'checkpoints',identity=identity,
                                interval=100,on_update=update,on_step=completed,**kw)
        if len(result.train_log)!=a.target:raise ValueError('Optimizer updates skipped.')
        expected=pd.read_csv(baseline(a.seed)/'training_example_draws.csv').subset_index.iloc[offset:a.target].tolist()
        if draws!=expected:raise ValueError('Paired example stream changed.')
        checkpoint_step=a.target if a.target!=30 else 100
        prior=torch.load(baseline(a.seed)/f'checkpoints/step_{checkpoint_step:06d}.pt',weights_only=False,map_location='cpu')
        if result.k_log!=[r[1] for r in prior['train_log'][:a.target]]:raise ValueError('Paired uniform-k stream changed.')
        if a.family=='zero' and a.target==1000 and not torch.equal(result.scores,prior['scores']):
            raise ValueError('Beta-zero 1,000-step scores differ from Gate2.')
        pd.DataFrame({'step':range(offset+1,a.target+1),'subset_index':draws}).to_csv(out/f'example_draws_{a.target}.csv',index=False)
        pd.DataFrame(diagnostics).to_csv(out/f'gradient_diagnostics_{a.target}.csv',index=False)
        pd.DataFrame(result.train_log,columns=['step','k','retained_fraction','loss']).to_csv(out/f'training_metrics_{a.target}.csv',index=False)
        torch.save(result.scores,out/f'raw_scores_{a.target}.pt')
        with torch.no_grad():result.scores=mixer(result.scores.to('cuda')).detach().cpu()
        torch.save(result.scores,out/f'effective_scores_{a.target}.pt')
        return result
    def evaluate(model,graph,loader,metric,**kw):
        answers={}
        for step in (30,500,1000,2500,5000):
            if step>a.target or (step==30 and not a.smoke):continue
            state=torch.load(out/f'checkpoints/step_{step:06d}.pt',weights_only=False,map_location='cpu')
            with torch.no_grad():effective=mixer(state['scores'].to('cuda')).detach().cpu()
            ident={'configuration':identity,'raw_checkpoint_sha256':sha(out/f'checkpoints/step_{step:06d}.pt'),
                   'validation_manifest_sha256':sha(out/'validation_manifest.csv'),'export':'effective scores'}
            r=evaluate_scores(model,graph,data.datasets['validation'],metric,effective,out,step,meter,ident,official_eval)
            answers[step]=r;write_json(out/f'quality_{step}.json',{'family':a.family,'beta':a.beta,'seed':a.seed,
                'step':step,'cpr':r[1],'cmd':r[2],'accuracy_auc':r[6]})
        return answers[a.target]
    def backward(t,*args,**kw):
        result=original_backward(t,*args,**kw);meter.counts['backward_calls']+=1;return result
    argv=['eval_mib_edge.py','--mib-path',str(MIB),'--model','gpt2','--task','ioi','--steps',str(a.target),
          '--k-schedule','uniform','--masking','topk','--mode','iso','--lr','.05','--optimizer','adam',
          '--split','validation','--train-split','train','--batch-size','20','--seed',str(a.seed),
          '--cf-cache-gb','4','--no-wandb','--output',str(out/f'upstream_{a.target}')]
    write_json(out/f'upstream_command_{a.target}.json',{'argv':argv})
    spec=importlib.util.spec_from_file_location('expander_official_edge',UPSTREAM/'scripts/mib/eval_mib_edge.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        with patch.object(HookedTransformer,'from_pretrained',load),patch.object(HFEAPDataset,'__init__',dataset_init),\
             patch.object(package,'learn_scores',train),patch.object(evaluation,'evaluate_area_under_curve',evaluate),\
             patch.object(torch.Tensor,'backward',backward),patch.object(sys,'argv',argv):module.main()
        success=True
    except BaseException as exc:
        import traceback
        write_json(out/f'failure_{a.target}.json',{'error':str(exc),'traceback':traceback.format_exc(),
                                                'oom':isinstance(exc,torch.cuda.OutOfMemoryError),'settings_unchanged':True})
        raise
    finally:
        meter.stop.set();meter.worker.join();meter.save();meter.stream.close()
        write_json(out/'process_summary.json',{'success':success,'target':a.target,'counts':meter.counts,
                                              'seconds':time.perf_counter()-meter.started})
    names=['experiment_config.json',f'quality_{a.target}.json',f'training_metrics_{a.target}.csv',
           f'raw_scores_{a.target}.pt',f'effective_scores_{a.target}.pt','train_manifest.csv','validation_manifest.csv']
    write_json(out/f'completed_{a.target}.json',{'passed':True,'sha256':{n:sha(out/n) for n in names}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--family',choices=('zero','scale','G0','G1','G2','G2perm'),required=True)
    p.add_argument('--beta',type=float,choices=(0.,.1,.25),required=True);p.add_argument('--seed',type=int,choices=(42,43,44),required=True)
    p.add_argument('--target',type=int,choices=(30,1000,5000),required=True);p.add_argument('--smoke',action='store_true')
    run(p.parse_args())
