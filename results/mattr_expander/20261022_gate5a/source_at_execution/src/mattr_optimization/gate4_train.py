"""Run the unchanged official MAttr CLI with one scoped score initialization."""
import argparse
from functools import partial
import importlib.util
import inspect
import json
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from .common import ROOT,UPSTREAM,MIB,GRID,configure,provenance,sha,write_json
from .gate3 import baseline
from .gate2 import NestedData,archive_attempt
from .gate1 import LiveAccounting
from .checkpoint import checkpointed
from .gate4_init import initialized_trainer


def membership(graph,folder,step):
    rows=[]
    for f in (.005,.01,.02,.05):
        graph.apply_topn(int(f*len(graph.edges)),absolute=False,prune=True)
        for edge in graph.edges.values():
            if edge.in_graph:rows.append({'step':step,'fraction':f,'edge':edge.name})
    pd.DataFrame(rows,columns=['step','fraction','edge']).to_csv(folder/f'membership_{step}.csv',index=False)


def evaluate_scores(model,graph,ds,metric,scores,folder,step,meter,identity,official_eval=None):
    dest=folder/f'evaluation_step{step}';dest.mkdir(exist_ok=True)
    path=dest/'official_result.json'
    import MIB_circuit_track.evaluation as evaluation
    if official_eval is None:official_eval=evaluation.evaluate_area_under_curve
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=identity:raise ValueError('Frozen evaluation identity changed.')
        result=saved['result']
    else:
        graph.scores[:]=-torch.inf;graph.scores[graph.real_edge_mask.bool()]=scores.cpu()
        with meter.stage(f'validation_evaluation_step{step}'),torch.no_grad():
            result=official_eval(model,graph,ds.to_dataloader(20),metric,
                                                       level='edge',absolute=False,quiet=True,refs={})
        write_json(path,{'identity':identity,'result':result})
    graph.scores[:]=-torch.inf;graph.scores[graph.real_edge_mask.bool()]=scores.cpu()
    pd.DataFrame({'fraction':GRID,'weighted_edge_count':result[0],'faithfulness':result[4],'accuracy':result[5]}).to_csv(dest/'faithfulness_curve.csv',index=False)
    pd.DataFrame([{'edge':e.name,'score':float(graph.scores[e.matrix_index])} for e in graph.edges.values()]).sort_values('score',ascending=False).to_csv(folder/f'ranking_{step}.csv',index=False)
    membership(graph,folder,step)
    return result


def run(args):
    configure()
    from transformer_lens import HookedTransformer
    from MIB_circuit_track.dataset import HFEAPDataset
    import MIB_circuit_track.evaluation as evaluation
    import matryoshka_attribution as package
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; lower batch size requires an explicit new protocol.')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(args.protocol.read_text())
    for name,h in protocol['adapter_sha256'].items():
        if sha(ROOT/'src/mattr_optimization'/name)!=h:raise ValueError('Source changed after freeze: '+name)
    prov=provenance()
    if prov!=json.loads((baseline(args.seed)/'experiment_config.json').read_text())['source_provenance']:
        raise ValueError('Pinned Gate2 runtime/source differs.')
    attr=args.protocol.parent/f'attribution_seed{args.seed}'
    initial=torch.load(attr/'rank_normalized.pt',weights_only=True)*args.scale
    config={'gate':4,'seed':args.seed,'scale':args.scale,'pool':128,'initial_sha256':sha(attr/'rank_normalized.pt'),
            'protocol_sha256':sha(args.protocol),'source_provenance':prov}
    if (out/'experiment_config.json').exists() and json.loads((out/'experiment_config.json').read_text())!=config:
        raise ValueError('Checkpoint initialization/configuration identity changed.')
    archive_attempt(out);write_json(out/'experiment_config.json',config);identity=json.dumps(config,sort_keys=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    original_load,original_data=HookedTransformer.from_pretrained,HFEAPDataset.__init__
    official_train=package.learn_scores;official_eval=evaluation.evaluate_area_under_curve;original_backward=torch.Tensor.backward
    data=NestedData(out,128,args.seed,128);meter=LiveAccounting(out);success=False
    old_checkpoints=sorted((out/'checkpoints').glob('step_*.pt'))
    offset=torch.load(old_checkpoints[-1],weights_only=False,map_location='cpu')['completed'] if old_checkpoints else 0
    def load(*a,**kw):
        with meter.stage('model_loading'):
            model=meter.attach(original_load(*a,**kw))
            if model.cfg.dtype!=torch.float32:raise ValueError('Float32 required.')
            return model
    def dataset_init(ds,*a,**kw):
        with meter.stage('data_loading_'+kw['split']):
            original_data(ds,*a,**kw);data.select(ds,kw['split'])
            pd.testing.assert_frame_equal(pd.read_csv(out/f'{kw["split"]}_manifest.csv'),pd.read_csv(baseline(args.seed)/f'{kw["split"]}_manifest.csv'))
    def train(total,loss_fn,**kw):
        if total!=32491 or kw['k_schedule']!='uniform' or 'k_sampler' in kw:raise ValueError('Official uniform trainer required.')
        randint=random.randint;drawn=[]
        def sampled_loss(mask):
            def counted(a,b):
                i=randint(a,b);drawn.append(i);return i
            with patch.object(random,'randint',counted):return loss_fn(mask)
        callback=kw.pop('on_step',None)
        def completed(step,k,loss,scores):
            if callback:callback(step,k,loss,scores)
            if (step+1)%100==0 or step==0:
                meter.status(completed_iteration=step+1,target=args.target,scale=args.scale,seed=args.seed)
                print(f'scale={args.scale} seed={args.seed} {step+1}/{args.target}',flush=True)
        with meter.stage('training'),initialized_trainer(initial):
            result=checkpointed(official_train,total,sampled_loss,folder=out/'checkpoints',identity=identity,
                                interval=100,on_update=meter.update,on_step=completed,**kw)
        if len(result.train_log)!=args.target:raise ValueError('Optimizer updates skipped.')
        pd.DataFrame({'step':range(offset+1,args.target+1),'subset_index':drawn}).to_csv(out/f'example_draws_{args.target}.csv',index=False)
        old=pd.read_csv(baseline(args.seed)/'training_example_draws.csv').subset_index.iloc[offset:args.target].tolist()
        if drawn!=old:raise ValueError('Paired example stream changed.')
        state=torch.load(baseline(args.seed)/f'checkpoints/step_{args.target:06d}.pt',map_location='cpu',weights_only=False)
        if result.k_log!=[r[1] for r in state['train_log']]:raise ValueError('Official uniform k stream changed.')
        pd.DataFrame(result.train_log,columns=['step','k','retained_fraction','loss']).to_csv(out/f'training_metrics_{args.target}.csv',index=False)
        torch.save(result.scores,out/f'final_scores_{args.target}.pt')
        return result
    def evaluate(model,graph,loader,metric,**kw):
        ds=data.datasets['validation'];answers={}
        # Read saved checkpoints after training; evaluating never consumes the train RNG.
        for step in (500,1000,2500,5000):
            if step>args.target:continue
            path=out/f'checkpoints/step_{step:06d}.pt'
            state=torch.load(path,map_location='cpu',weights_only=False)
            ident={'config':identity,'checkpoint_sha256':sha(path),'validation_manifest_sha256':sha(out/'validation_manifest.csv')}
            r=evaluate_scores(model,graph,ds,metric,state['scores'],out,step,meter,ident,official_eval)
            answers[step]=r
            write_json(out/f'quality_{step}.json',{'seed':args.seed,'scale':args.scale,'step':step,
                                                'cpr':r[1],'cmd':r[2],'accuracy_auc':r[6]})
        return answers[args.target]
    def backward(t,*a,**kw):
        value=original_backward(t,*a,**kw);meter.counts['backward_calls']+=1;return value
    argv=['eval_mib_edge.py','--mib-path',str(MIB),'--model','gpt2','--task','ioi','--steps',str(args.target),
          '--k-schedule','uniform','--masking','topk','--mode','iso','--lr','.05','--optimizer','adam',
          '--split','validation','--train-split','train','--batch-size','20','--seed',str(args.seed),
          '--cf-cache-gb','4','--no-wandb','--output',str(out/f'upstream_{args.target}')]
    write_json(out/f'upstream_command_{args.target}.json',{'argv':argv})
    spec=importlib.util.spec_from_file_location('gate4_official_edge',UPSTREAM/'scripts/mib/eval_mib_edge.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    try:
        with patch.object(HookedTransformer,'from_pretrained',load),patch.object(HFEAPDataset,'__init__',dataset_init),\
             patch.object(package,'learn_scores',train),patch.object(evaluation,'evaluate_area_under_curve',evaluate),\
             patch.object(torch.Tensor,'backward',backward),patch.object(sys,'argv',argv):
            # evaluate_scores calls the saved original below, avoiding wrapper recursion.
            module.main()
        success=True
    except BaseException as exc:
        import traceback
        write_json(out/f'failure_{args.target}.json',{'error':str(exc),'traceback':traceback.format_exc(),
            'oom':isinstance(exc,torch.cuda.OutOfMemoryError),'settings_unchanged':True})
        raise
    finally:
        meter.stop.set();meter.worker.join();meter.save();meter.stream.close()
        write_json(out/'process_summary.json',{'success':success,'target':args.target,'counts':meter.counts,
                                              'seconds':time.perf_counter()-meter.started})
    names=['experiment_config.json',f'final_scores_{args.target}.pt',f'training_metrics_{args.target}.csv',
           'train_manifest.csv','validation_manifest.csv']
    names += [str(p.relative_to(out)) for p in out.glob('quality_*.json')]
    write_json(out/f'completed_{args.target}.json',{'passed':True,'sha256':{n:sha(out/n) for n in names}})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--scale',type=float,choices=(.1,.5,1.,2.),required=True)
    p.add_argument('--seed',type=int,choices=(42,43,44),required=True);p.add_argument('--target',type=int,choices=(1000,5000),required=True)
    run(p.parse_args())
