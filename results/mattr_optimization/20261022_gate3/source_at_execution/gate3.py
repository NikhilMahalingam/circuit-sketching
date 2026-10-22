"""Gate 3: frozen two-mixture screening; at most one 5k confirmation, no Gate 4."""
import argparse
from datetime import datetime,timezone
import importlib.util
import inspect
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from .common import ROOT,UPSTREAM,MIB,GRID,configure,provenance,sha,write_json
from .gate2 import NestedData,archive_attempt

CRITERIA=ROOT/'results/mattr_optimization/20261021_gate2_final/gate3_success_criteria.json'
CRITERIA_SHA='f200fc163bfb0e70b4790bde28ed3a6748bf9051c7d574ed7e23bda69cf0dbde'
SEEDS=(42,43,44)


def baseline(seed):
    folder='20261021T172035Z_gate2' if seed==42 else '20261021_gate2_seeds43_44'
    return ROOT/'results/mattr_optimization'/folder/f'seed{seed}_n128'


def verify_criteria():
    if sha(CRITERIA)!=CRITERIA_SHA:
        raise ValueError('Frozen Gate 2 criteria SHA changed; refusing Gate 3.')
    return json.loads(CRITERIA.read_text())


def run_job(args):
    import pandas as pd
    import torch
    from .checkpoint import checkpointed
    from .gate1 import LiveAccounting
    from .gate3_sampling import CPRMixture
    configure();verify_criteria()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; settings will not change after OOM.')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(args.protocol.read_text())
    for n,h in protocol['adapter_sha256'].items():
        if sha(ROOT/'src/mattr_optimization'/n)!=h:raise ValueError('Gate 3 source changed after freeze.')
    config={'gate':3,'seed':args.seed,'subset_seed':args.seed,'pool':128,'lambda':args.mixture_lambda,
            'protocol_sha256':sha(args.protocol),'source_provenance':provenance()}
    expected=json.loads((baseline(args.seed)/'experiment_config.json').read_text())['source_provenance']
    if config['source_provenance']!=expected:raise ValueError('Environment/source differs from Gate 2.')
    if (out/'experiment_config.json').exists() and json.loads((out/'experiment_config.json').read_text())!=config:
        raise ValueError('Job identity changed.')
    archive_attempt(out);write_json(out/'experiment_config.json',config)
    identity=json.dumps(config,sort_keys=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from transformer_lens import HookedTransformer
    from MIB_circuit_track.dataset import HFEAPDataset
    import MIB_circuit_track.evaluation as evaluation
    import matryoshka_attribution as package
    original_load,original_data=HookedTransformer.from_pretrained,HFEAPDataset.__init__
    official_train,official_eval=package.learn_scores,evaluation.evaluate_area_under_curve
    original_backward=torch.Tensor.backward
    data=NestedData(out,128,args.seed,128)
    meter=LiveAccounting(out)
    marker=out/f'completed_{args.target}.json'

    def load(*a,**kw):
        with meter.stage('model_loading'):
            model=original_load(*a,**kw)
            if model.cfg.dtype!=torch.float32:raise ValueError('Float32 required.')
            return meter.attach(model)

    def dataset_init(ds,*a,**kw):
        with meter.stage('data_loading_'+kw['split']):
            original_data(ds,*a,**kw);data.select(ds,kw['split'])
            pd.testing.assert_frame_equal(pd.read_csv(out/f'{kw["split"]}_manifest.csv'),
                                         pd.read_csv(baseline(args.seed)/f'{kw["split"]}_manifest.csv'))

    def train(total,loss_fn,**kw):
        if total!=32491 or kw['steps']!=args.target:raise ValueError('Unexpected score count/budget.')
        draw=random.randint
        with (out/'training_example_draws.csv').open('w',buffering=1) as draws, \
             (out/'gradient_norms.csv').open('w',buffering=1) as norms, \
             (out/f'sampler_draws_target{args.target}.csv').open('w',buffering=1) as samples:
            draws.write('presentation_this_attempt,subset_index\n');norms.write('update_this_attempt,gradient_norm\n')
            samples.write('draw_this_attempt,u,branch,grid_index,k\n')
            presentations=0
            def sampled_loss(mask):
                def counted(a,b):
                    nonlocal presentations
                    i=draw(a,b);presentations+=1;draws.write(f'{presentations},{i}\n');return i
                with patch.object(random,'randint',counted):return loss_fn(mask)
            def update(norm):
                meter.update(norm);norms.write(f'{len(meter.grad_norms)},{norm}\n')
            callback=kw.pop('on_step',None)
            def completed(step,k,loss,scores):
                if callback:callback(step,k,loss,scores)
                if (step+1)%100==0 or step==0:
                    meter.status(completed_iteration=step+1,target=args.target,seed=args.seed,mixture_lambda=args.mixture_lambda)
                    print(f'lambda={args.mixture_lambda}, seed={args.seed}: {step+1}/{args.target}',flush=True)
            if args.mixture_lambda!=1.:
                kw['k_sampler']=CPRMixture(total,args.mixture_lambda,samples)
            with meter.stage('training'):
                result=checkpointed(official_train,total,sampled_loss,folder=out/'checkpoints',identity=identity,
                                    interval=100,on_update=update,on_step=completed,**kw)
        if len(result.train_log)!=args.target:raise ValueError('Skipped optimizer updates; no result may be claimed.')
        pd.DataFrame(result.train_log,columns=['step','k','retained_fraction','loss']).to_csv(out/f'training_metrics_{args.target}.csv',index=False)
        torch.save(result.scores,out/f'final_scores_{args.target}.pt')
        closure=inspect.getclosurevars(loss_fn).nonlocals
        write_json(out/f'donor_cache_{args.target}.json',{'stats':closure['cc_stats'],'bytes':closure['cc_bytes']})
        # Vanilla replay must reproduce the actual Gate-2 checkpoint/row and k draws.
        if args.mixture_lambda==1.:
            saved=torch.load(baseline(args.seed)/'checkpoints/step_001000.pt',map_location='cpu',weights_only=False)
            if not torch.equal(result.scores,saved['scores']):raise ValueError('Uniform replay does not exactly match Gate 2 scores.')
            old=pd.read_csv(baseline(args.seed)/'training_example_draws.csv').iloc[:1000].subset_index.tolist()
            if pd.read_csv(out/'training_example_draws.csv').subset_index.tolist()!=old:raise ValueError('Uniform data draw mismatch.')
            if result.k_log!=[r[1] for r in saved['train_log']]:raise ValueError('Uniform k replay mismatch.')
            write_json(out/'uniform_replay_validation.json',{'scores_bitwise_equal':True,'data_draws_equal':True,'k_draws_equal':True})
        return result

    def evaluate(model,graph,loader,metric,**kw):
        if kw.get('level')!='edge' or kw.get('absolute') is not False:raise ValueError('Signed edge evaluation required.')
        folder=out/f'evaluation_step{args.target}';folder.mkdir(exist_ok=True)
        cache_id={'config':identity,'checkpoint_sha256':sha(out/f'checkpoints/step_{args.target:06d}.pt'),
                  'validation_manifest_sha256':sha(out/'validation_manifest.csv')}
        target=folder/'official_result.json'
        refs={}
        if target.exists():
            saved=json.loads(target.read_text())
            if saved['identity']!=cache_id:raise ValueError('Evaluation cache identity changed.')
            result=saved['result']
            write_json(folder/'cache_hit.json',{'model_forwards':0})
        else:
            # The official CLI already installs the final score vector in graph.
            with meter.stage('validation_evaluation'),torch.no_grad():
                meter.status(evaluation_step=args.target)
                result=official_eval(model,graph,loader,metric,level='edge',absolute=False,quiet=True,refs=refs)
            write_json(target,{'identity':cache_id,'result':result,'references':refs})
        pd.DataFrame({'fraction':GRID,'weighted_edge_count':result[0],'faithfulness':result[4],'accuracy':result[5]}).to_csv(folder/'faithfulness_curve.csv',index=False)
        rows=[{'edge':e.name,'score':float(graph.scores[e.matrix_index])} for e in graph.edges.values()]
        pd.DataFrame(rows).sort_values('score',ascending=False).to_csv(out/f'ranking_{args.target}.csv',index=False)
        write_json(out/f'quality_{args.target}.json',{'seed':args.seed,'lambda':args.mixture_lambda,'step':args.target,
                                                    'cpr':result[1],'cmd':result[2],'accuracy_auc':result[6]})
        if args.mixture_lambda==1.:
            old=json.loads((baseline(args.seed)/'evaluation/validation_step1000/official_result.json').read_text())['result']
            import numpy as np
            if not np.allclose(result[4],old[4],rtol=1e-6,atol=1e-6) or result[5]!=old[5]:
                raise ValueError('Uniform metrics disagree with Gate 2.')
        return result

    def backward(tensor,*a,**kw):
        result=original_backward(tensor,*a,**kw);meter.counts['backward_calls']+=1;return result

    argv=['eval_mib_edge.py','--mib-path',str(MIB),'--model','gpt2','--task','ioi','--steps',str(args.target),
          '--k-schedule','uniform','--masking','topk','--mode','iso','--lr','.05','--optimizer','adam',
          '--split','validation','--train-split','train','--batch-size','20','--seed',str(args.seed),
          '--cf-cache-gb','4','--no-wandb','--output',str(out/f'upstream_{args.target}')]
    write_json(out/f'upstream_command_{args.target}.json',{'argv':argv,'only_algorithm_change':'k_sampler hook' if args.mixture_lambda!=1 else 'none'})
    spec=importlib.util.spec_from_file_location('official_edge_gate3',UPSTREAM/'scripts/mib/eval_mib_edge.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    success=False
    try:
        with patch.object(HookedTransformer,'from_pretrained',side_effect=load),patch.object(HFEAPDataset,'__init__',dataset_init), \
             patch.object(package,'learn_scores',train),patch.object(evaluation,'evaluate_area_under_curve',evaluate), \
             patch.object(torch.Tensor,'backward',backward),patch.object(sys,'argv',argv):
            module.main()
        success=True
    except BaseException as exc:
        import traceback
        write_json(out/f'failure_{args.target}.json',{'error':str(exc),'type':type(exc).__name__,
                   'traceback':traceback.format_exc(),'settings_unchanged':True,'oom':isinstance(exc,torch.cuda.OutOfMemoryError)})
        raise
    finally:
        meter.status(success=success);meter.stop.set();meter.worker.join();meter.save();meter.stream.close()
        write_json(out/'process_summary.json',{'success':success,'counts':meter.counts,'process_seconds':time.perf_counter()-meter.started,'target':args.target})
    names=['experiment_config.json',f'final_scores_{args.target}.pt',f'ranking_{args.target}.csv',f'quality_{args.target}.json',
           'train_manifest.csv','validation_manifest.csv',f'training_metrics_{args.target}.csv']
    write_json(marker,{'passed':True,'sha256':{n:sha(out/n) for n in names}})


def freeze(out):
    from .gate3_sampling import distribution
    verify_criteria();out.mkdir(parents=True,exist_ok=True)
    names=('gate3.py','gate3_sampling.py','gate3_report.py','common.py','checkpoint.py','gate2.py','gate1.py','gate0.py')
    protocol={'gate':3,'criteria_path':str(CRITERIA.relative_to(ROOT)),'criteria_sha256':CRITERIA_SHA,
      'seeds':list(SEEDS),'paired_subset_training_seeds':[[s,s] for s in SEEDS],'pool':128,'validation_size':128,'validation_seed':42,
      'lambdas':[.5,.25],'exponent':1.,'screen_steps':1000,'confirmation_steps':5000,'uniform_timing_replays':True,
      'total_edges':32491,'grid':list(GRID),'distribution':distribution(),
      'sampling':'lambda Uniform[1,N) + (1-lambda) sum_j q_j Uniform[L_j,H_j); q_j=w_j^1/sum(w_except_full); weights are linear CPR trapezoids.',
      'rounding':'c_j=int(p_j*N), positive floor matching the official evaluator; k remains floating, never rounded after jitter.',
      'jitter':'L=max(1,0.95*c); H=min(N-1,1.05*c); uniform over truncated interval, no clipping point masses.',
      'endpoint':'Full graph p=1 has fixed faithfulness 1, constant CPR contribution .25 and no ranking objective; exclude ONLY from grid training, renormalize .749 remaining mass. Evaluation grid unchanged. Uniform branch retains original near-full k<N behavior.',
      'rng':'Exactly one global CPU float32 torch.rand(1) per step, same consumption as official uniform. Inverse mixture CDF and category residual provide branch, grid index, jitter. Python row RNG untouched. No feedback/private RNG. Original checkpoint CPU/CUDA/Python/NumPy RNG restoration suffices.',
      'screen_rule':'Apply frozen primary numerical gates provisionally at 1000 vs matching 1000 baselines, OR frozen efficiency alternative vs 5000 baseline. This does not establish 5000 success. If none eligible stop. Rank eligible by mean paired relative CPR gain vs 1000; exact tie chooses lambda .5. No extra variants.',
      'confirmation':'Resume the one winner from step 1000 to 5000 for all three seeds; charge screen, restart/cache rebuilding and all other trials. Match frozen 5000 criteria without changes.',
      'memory':'Peak torch.cuda.max_memory_allocated, maximum across actual stages/attempts. Paired ratio of mean peaks, limit from original criteria.',
      'time':'CUDA-synchronized perf_counter. Screen training time/peak guards compare fresh uniform timing controls verified against Gate2; quality always Gate2. Confirmation training cost sums screen+continuation+donor-cache rebuilds against Gate2 measured 5000 training times; total costs additionally reported.',
      'adapter_sha256':{n:sha(ROOT/'src/mattr_optimization'/n) for n in names}}
    path=out/'gate3_protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=protocol:raise ValueError('Frozen Gate 3 protocol differs; refusing to change it.')
    else:
        import shutil
        protected={}
        for folder in (ROOT/'results/mattr_optimization').iterdir():
            if folder.is_dir() and folder.resolve()!=out.resolve():
                for f in folder.rglob('*'):
                    if f.is_file():protected[str(f.relative_to(ROOT))]=sha(f)
        protected['GATE2_FINAL_ANALYSIS.md']=sha(ROOT/'GATE2_FINAL_ANALYSIS.md')
        write_json(out/'protected_gate2_hashes.json',protected)
        shutil.copyfile(CRITERIA,out/'gate3_success_criteria.json')
        write_json(path,protocol)
        (out/'gate3_protocol.sha256').write_text(sha(path)+'\n')
        (out/'GATE3_SAMPLING_PROTOCOL.md').write_text('# Gate 3 sampling protocol — frozen before training\n\n'+
          f'Frozen criteria SHA-256: `{CRITERIA_SHA}`. Protocol SHA-256: `{sha(path)}`.\n\n'+
          '\n\n'.join(f'**{k}**: {protocol[k]}' for k in ('sampling','rounding','jitter','endpoint','rng','screen_rule','confirmation','memory','time'))+
          '\n\nStatic CPR-aware mixture; no online loss/validation adaptation. No edge-patching, optimizer, gradient or evaluator changes.\n')
        write_json(out/'protocol_frozen_at.json',{'utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(path),'model_training_started':False})
    return path


def launch(out,protocol,lam,seed,target):
    name='uniform' if lam==1 else f'lambda{lam:g}'
    folder=out/f'{name}_seed{seed}';folder.mkdir(exist_ok=True)
    marker=folder/f'completed_{target}.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if not saved['passed'] or any(sha(folder/n)!=h for n,h in saved['sha256'].items()):raise ValueError('Completed artifacts changed.')
        print(f'Cached completed {name} seed={seed} target={target}; zero forwards.',flush=True);return
    command=[sys.executable,'-u','-m','src.mattr_optimization.gate3','--job','--out',str(folder),'--protocol',str(protocol),
             '--mixture-lambda',str(lam),'--seed',str(seed),'--target',str(target)]
    started=time.perf_counter()
    with (folder/'console.log').open('a',buffering=1) as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in child.stdout:log.write(line);print(line,end='',flush=True)
        if child.wait()!=0:raise RuntimeError(f'Gate3 job failed: {folder}. Settings unchanged; inspect failure JSON.')
    write_json(folder/f'launch_elapsed_{target}.json',{'seconds':time.perf_counter()-started,'target':target})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--freeze-only',action='store_true');p.add_argument('--report-only',action='store_true')
    p.add_argument('--job',action='store_true');p.add_argument('--protocol',type=Path);p.add_argument('--mixture-lambda',type=float,choices=(.25,.5,1.))
    p.add_argument('--seed',type=int,choices=SEEDS);p.add_argument('--target',type=int,choices=(1000,5000))
    args=p.parse_args();args.out=args.out.resolve()
    if args.job:run_job(args);return
    if args.report_only:
        from .gate3_report import generate
        generate(args.out);return
    import fcntl
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=freeze(args.out)
        if args.freeze_only:print('Protocol frozen; no model loaded or trained.');return
        print('Gate3: three exact uniform timing replays + six prespecified 1000-step screens; at most three 4000-update continuations.',flush=True)
        for lam in (1.,.5,.25):
            for seed in SEEDS:launch(args.out,protocol,lam,seed,1000)
        from .gate3_report import generate
        decision=generate(args.out)
        if decision['winner'] is not None:
            for seed in SEEDS:launch(args.out,protocol,decision['winner'],seed,5000)
            generate(args.out)
        else:print('Both screens failed frozen gates. No expansion, confirmation or Gate4.',flush=True)
        protected=json.loads((args.out/'protected_gate2_hashes.json').read_text())
        if any(sha(ROOT/n)!=h for n,h in protected.items()):raise ValueError('Protected Gate 2 artifact changed.')
        verify_criteria()
        write_json(args.out/'gate3_completed.json',{'passed_execution':True,'winner':decision['winner'],
                   'protected_files_unchanged':len(protected),'criteria_sha256':CRITERIA_SHA,'gate4_executed':False,'leaderboard_submission':False})


if __name__=='__main__':main()
