"""Independent, gated correctness and benchmark entry points. Stage A is default."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch

from mib_ioi.runtime import (configure_repository,seed_everything,load_model,load_splits,
                             write_json)
from .engine import ExactExecutor,ExecutionMeter,prepare_batches,sync
from .graph import Dependencies,digest
from .search import atomic_json,exact_acdc

ROOT=Path(__file__).resolve().parents[2]
LAYERS=(0,2,4,6,8,10,11)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def telemetry():
    command=['nvidia-smi','--query-gpu=name,driver_version,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,memory.used',
             '--format=csv']
    try: return subprocess.check_output(command,text=True).strip()
    except (OSError,subprocess.CalledProcessError) as error: return 'Unavailable: '+str(error)


def single_edges(graph):
    names=[]
    for layer in LAYERS:
        sender='input' if layer==0 else f'm{layer-1}'
        names += [f'{sender}->a{layer}.h0<{letter}>' for letter in 'qkv']
        names.append(f'a{layer}.h0->m{layer}')
    names += ['input->logits','m11->logits']
    assert all(n in graph.edges for n in names)
    return names


def restricted_edges(deps,count=32):
    chosen=set(single_edges(deps.graph))
    chosen.update(['input->a0.h0<v>','a0.h0->a1.h0<v>','a5.h5->a8.h6<v>'])
    return [n for n in deps.order if n in chosen][:count]


def multi_states(deps,seed):
    states=[('dependent_early',{'input->a0.h0<v>','a0.h0->a1.h0<v>'}),
        ('dependent_middle',{'m3->a4.h0<v>','a4.h0->m4','m4->a6.h0<q>'}),
        ('previous_early_removal_plus_late_candidate',{'input->a0.h0<v>','a9.h9->a11.h10<q>'}),
        ('disconnected_head',{'input->a0.h0<q>','input->a0.h0<k>','input->a0.h0<v>'})]
    rng=np.random.default_rng(seed)
    for count in (4,16,48):
        states.append((f'random_{count}',set(rng.choice(deps.order,count,replace=False).tolist())))
    late=[n for n in deps.order if deps.graph.edges[n].child.layer>=8]
    states.append(('random_late_64',set(rng.choice(late,64,replace=False).tolist())))
    return states


def official_loss(model,dataset,removed,batch_size):
    from eap.graph import Graph
    from eap.evaluate import evaluate_graph
    from MIB_circuit_track.metrics import get_metric
    graph=Graph.from_model(model);graph.reset(empty=False)
    for name in removed: graph.edges[name].in_graph=False
    metric=get_metric('logit_diff','ioi',model.tokenizer,model)
    with torch.no_grad():
        values=evaluate_graph(model,graph,dataset.to_dataloader(batch_size),
            lambda *args:metric(*args,mean=False,loss=True),quiet=True,intervention='patching',skip_clean=True)
    return values.cpu().numpy()


def corrected_shared_loss(model,graph,dataset,removed,batch_size):
    from src.sketch_acdc.core import EdgeSpec,EdgeExperiment
    edges=[]
    for name in sorted(removed):
        e=graph.edges[name]
        edges.append(EdgeSpec(name,*e.matrix_index,e.hook,getattr(e.child,'head',None),e.child.layer,0.))
    exp=EdgeExperiment(model,graph,edges,{'train':dataset},batch_size,'prefix-stage-a-shared-oracle')
    # Existing dynamic-sender implementation, never FrozenSenderEdgeExperiment.
    return -exp.evaluate_state([1]*len(edges),'train',fresh=True)


def protected_snapshot():
    names=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    return {n:sha(ROOT/n) for n in names if n and (ROOT/n).is_file()
            and n not in ('README.md','.gitattributes') and
            not n.startswith(('src/acdc_prefix_cache/','notebooks/acdc_prefix_caching','tests/test_acdc_prefix_cache'))
            and n!='ACDC_PREFIX_CACHE_REPORT.md' and not n.startswith('results/acdc_prefix_cache/')}


def compare_trace(label,standard,cached,rows,atol,rtol):
    a,b=standard['trace'],cached['trace']
    if len(a)!=len(b): raise AssertionError('Trace lengths differ.')
    for x,y in zip(a,b):
        row={'trajectory':label,'step':x['step'],'edge':x['edge'],
            'standard_loss':x['loss'],'cached_loss':y['loss'],
            'absolute_error':abs(x['loss']-y['loss']),
            'standard_removed':x['removed'],'cached_removed':y['removed'],
            'candidate_matches':x['edge']==y['edge'],
            'decision_matches':x['removed']==y['removed'],
            'state_before_matches':x['state_before_hash']==y['state_before_hash'],
            'state_after_matches':x['state_after_hash']==y['state_after_hash'],
            'exact_loss_match':x['loss']==y['loss']}
        rows.append(row)
        if not all(row[k] for k in ('candidate_matches','decision_matches','state_before_matches','state_after_matches')):
            raise AssertionError(f'Decision/state disagreement at {label} {x["step"]}')
        np.testing.assert_allclose(x['loss'],y['loss'],atol=atol,rtol=rtol)
    if standard['retained']!=cached['retained']: raise AssertionError('Retained edge identities differ.')


def stage_a(model,graph,data,out,args,meter):
    results=[];queries=[];traces=[];trajectories=[];memory=[];validation_targets={}
    batches=prepare_batches(model,data['train'],args.batch_size)
    print(f'Stage A: {len(single_edges(graph))} single-edge and 8 multi-edge queries; '
          f'32 restricted + {args.long_checks} deterministic-prefix decisions per arm; '
          f'{len(batches)} batches/query. Official oracle adds 2 full forwards/batch/query. '
          'No performance or full-graph benchmark will run.',flush=True)
    meter.phase='cuda_warmup'
    with torch.no_grad():
        for _ in range(args.warmup): model(batches[0].tokens,attention_mask=batches[0].attention_mask)
    sync(model)
    executors={}
    for label,cached in [('standard',False),('cached',True)]:
        meter.phase='stage_a_'+label+'_cache_preparation'
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        executors[label]=ExactExecutor(model,graph,batches,cached=cached).prepare()
        sync(model)
        memory.append({'stage':'A_correctness_shared_residency','method':label,
            'cache_bytes':executors[label].cache_bytes,
            'cache_setup_seconds':executors[label].setup_seconds,
            'constructor_plus_setup_seconds':time.perf_counter()-started,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
            'isolated_memory_comparison':False})
    standard,cached=executors['standard'],executors['cached']
    deps=cached.dependencies
    cases=[('single',name,{name}) for name in single_edges(graph)]
    cases += [('multi',name,state) for name,state in multi_states(deps,args.seed)]
    try:
        for index,(category,label,state) in enumerate(cases):
            values={}
            for method in (('standard','cached') if index%2==0 else ('cached','standard')):
                meter.phase=f'stage_a_{category}_{method}'
                values[method]=executors[method].evaluate(state)
            meter.phase='stage_a_official_mib_oracle'
            oracle=official_loss(model,data['train'],state,args.batch_size)
            error=np.abs(values['standard']-values['cached'])
            oracle_error=np.abs(values['standard']-oracle)
            plan=deps.plan(state)
            for example,(a,b,c) in enumerate(zip(values['standard'],values['cached'],oracle)):
                queries.append({'category':category,'query':label,'example':example,
                    'split':'train',
                    'removed_edges':json.dumps(sorted(state)),'earliest_affected_layer':plan.start_layer,
                    'standard_loss':float(a),'cached_loss':float(b),'official_mib_loss':float(c),
                    'absolute_error':float(abs(a-b)),
                    'relative_error':float(abs(a-b)/abs(a)) if abs(a)>1e-6 else np.nan,
                    'exact_match':bool(a==b),'official_absolute_error':float(abs(a-c))})
            pd.DataFrame(queries).to_csv(out/'intervention_loss_comparison.csv',index=False)
            np.testing.assert_allclose(values['cached'],values['standard'],atol=args.atol,rtol=args.rtol)
            np.testing.assert_allclose(values['standard'],oracle,atol=args.atol,rtol=args.rtol)
            if category=='multi' and label.startswith('dependent'):
                meter.phase='stage_a_corrected_shared_adapter_oracle'
                shared=corrected_shared_loss(model,graph,data['train'],state,args.batch_size)
                np.testing.assert_allclose(values['standard'],shared,atol=args.atol,rtol=args.rtol)
            results.append({'test':category,'case':label,'examples':len(error),
                'max_absolute_error':float(error.max()),'mean_absolute_error':float(error.mean()),
                'exact_matches':int((error==0).sum()),'official_max_absolute_error':float(oracle_error.max()),'passed':True})
            print(f'{category} {index+1}/{len(cases)} {label}: max error {error.max():.3g}; '
                  f'official {oracle_error.max():.3g}; safe boundary {plan.start_layer}',flush=True)
        # Explicitly reject a late-only cache plan after an early accepted removal.
        unsafe={'input->a0.h0<v>','a9.h9->a11.h10<q>'}
        try: cached.evaluate(unsafe,requested_start=10)
        except ValueError: results.append({'test':'guard','case':'unsafe_prefix_rejected','passed':True})
        else: raise AssertionError('Unsafe prefix request was accepted.')
        meter.phase='stage_a_hook_removal'
        np.testing.assert_array_equal(standard.baseline,standard.evaluate(set()))
        np.testing.assert_array_equal(cached.baseline,cached.evaluate(set()))
        for label,order in [('restricted_32',restricted_edges(deps)),
                            (f'partial_prefix_{args.long_checks}',list(deps.order[:args.long_checks]))]:
            pair={}
            for method in ('standard','cached'):
                meter.phase=f'stage_a_trajectory_{label}_{method}'
                pair[method]=exact_acdc(executors[method],order,args.threshold)
                atomic_json(out/f'{label}_{method}.json',pair[method])
            compare_trace(label,pair['standard'],pair['cached'],traces,args.atol,args.rtol)
            validation_targets[label]=pair['cached']['removed']
            for method,result in pair.items():
                trajectories.append({'stage':'A_correctness_not_performance','trajectory':label,
                    'method':method,'candidate_checks':len(order),'accepted_removals':len(result['removed']),
                    'final_edge_count':len(result['retained']),'circuit_hash':result['circuit_hash'],
                    'search_seconds_diagnostic_only':result['search_seconds'],
                    'full_graph':not result['partial_graph_trajectory']})
            print(f'{label}: {len(order)} identical decisions; retained circuit hash '
                  f'{pair["cached"]["circuit_hash"][:16]}',flush=True)
        # Interrupt the longer prefix, rebuild caches, and replay the saved state.
        order=list(deps.order[:args.long_checks]);checkpoint=out/'resume_checkpoint.json'
        meter.phase='stage_a_resume_interrupted'
        exact_acdc(cached,order,args.threshold,checkpoint=checkpoint,stop_after=17)
        cached.close()
        meter.phase='stage_a_resume_cache_rebuild'
        resumed_executor=ExactExecutor(model,graph,batches,cached=True).prepare()
        meter.phase='stage_a_resume_continue'
        resumed=exact_acdc(resumed_executor,order,args.threshold,checkpoint=checkpoint)
        compare_trace('interrupt_rebuild_resume',pair['cached'],resumed,traces,args.atol,args.rtol)
        atomic_json(out/'resumed_trajectory.json',resumed)
        # Cache-boundary arithmetic may skip blocks, but no loss evaluations
        # are memoized. Save every actual model invocation and block count.
        pd.DataFrame(standard.evaluations).assign(method='standard').to_csv(out/'standard_execution.csv',index=False)
        pd.DataFrame(cached.evaluations).assign(method='cached').to_csv(out/'cached_execution.csv',index=False)
        resumed_executor.close()
        standard.close()
        validation_rows=[]
        meter.phase='stage_a_validation_preparation'
        val_batches=prepare_batches(model,data['validation'],args.batch_size)
        val_standard=ExactExecutor(model,graph,val_batches).prepare()
        val_cached=ExactExecutor(model,graph,val_batches,cached=True).prepare()
        try:
            for label,state in validation_targets.items():
                meter.phase='stage_a_validation_standard';a=val_standard.evaluate(state)
                meter.phase='stage_a_validation_cached';b=val_cached.evaluate(state)
                meter.phase='stage_a_validation_official';c=official_loss(model,data['validation'],state,args.batch_size)
                np.testing.assert_allclose(a,b,atol=args.atol,rtol=args.rtol)
                np.testing.assert_allclose(a,c,atol=args.atol,rtol=args.rtol)
                validation_rows.append({'trajectory':label,'examples':len(a),'standard_loss':float(a.mean()),
                    'cached_loss':float(b.mean()),'official_loss':float(c.mean()),
                    'max_absolute_error':float(np.max(abs(a-b))),'official_max_absolute_error':float(np.max(abs(a-c))),
                    'retained_identity_match':True})
            pd.DataFrame(validation_rows).to_csv(out/'stage_a_validation.csv',index=False)
        finally:
            val_standard.close();val_cached.close()
        if any(p.grad is not None for p in model.parameters()): raise AssertionError('Inference retained parameter gradients.')
        if any(e['grad_enabled'] for e in meter.events): raise AssertionError('Inference forward tracked gradients.')
        if any(not e['success'] for e in meter.events): raise AssertionError('A model forward failed.')
        for name,expected in json.loads((out/'protected_previous_artifacts.json').read_text()).items():
            if sha(ROOT/name)!=expected: raise AssertionError(f'Previous artifact changed: {name}')
        atomic_json(out/'stage_a_passed.json',{'passed':True,'single_queries':len(single_edges(graph)),
            'multi_queries':8,'restricted_decisions':len(restricted_edges(deps)),
            'longer_partial_decisions':args.long_checks,'resume_decisions':len(resumed['trace']),
            'all_retained_edge_identities_match':True,'full_graph_run':False,
            'previous_artifacts_unchanged':True,'atol':args.atol,'rtol':args.rtol,
            'graph_edges':len(deps.order),'graph_identity':deps.identity,
            'core_source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')},
            'batch_size':args.batch_size,'train_examples':args.train_examples,'threshold':args.threshold})
    finally:
        pd.DataFrame(results).to_csv(out/'correctness_results.csv',index=False)
        pd.DataFrame(queries).to_csv(out/'intervention_loss_comparison.csv',index=False)
        pd.DataFrame(traces).to_csv(out/'decision_trace_comparison.csv',index=False)
        pd.DataFrame(trajectories).to_csv(out/'stage_a_trajectories.csv',index=False)
        pd.DataFrame(memory).to_csv(out/'memory_benchmark.csv',index=False)
        for executor in executors.values(): executor.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['a','b','c','report'],default='a')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--train-examples',type=int,default=16)
    parser.add_argument('--validation-examples',type=int,default=16)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threshold',type=float,default=.05)
    parser.add_argument('--long-checks',type=int,default=64)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--atol',type=float,default=5e-5)
    parser.add_argument('--rtol',type=float,default=1e-5)
    parser.add_argument('--allow-full-graph',action='store_true')
    parser.add_argument('--sizes',type=int,nargs='+',default=[16,32,64,128])
    parser.add_argument('--checks',type=int,nargs='+',default=[64,128])
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args()
    if not 33<=args.long_checks<=32491: parser.error('--long-checks must be 33..32491; default lightweight trajectory is 64.')
    out=(args.out or ROOT/'results/acdc_prefix_cache'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.stage=='report':
        from .report import generate
        generate(out);return
    if args.stage=='c' and not args.allow_full_graph:
        parser.error('Stage C is disabled by default; explicitly pass --allow-full-graph after A and B pass.')
    if args.stage in ('b','c') and not (out/'stage_a_passed.json').exists():
        parser.error('First run and pass Stage A in this output directory.')
    if args.stage=='c' and not (out/'stage_b_passed.json').exists():
        parser.error('Stage C additionally requires a completed, passing Stage B.')
    if args.stage=='a' and (out/'stage_a_passed.json').exists():
        print('Stage A already passed; not rerunning model work.',flush=True)
        from .report import generate
        generate(out);return
    protection=out/'protected_previous_artifacts.json'
    if not protection.exists(): atomic_json(protection,protected_snapshot())
    else:
        for relative,expected in json.loads(protection.read_text()).items():
            if sha(ROOT/relative)!=expected: raise ValueError(f'Protected prior file changed: {relative}')
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    if not torch.cuda.is_available(): raise RuntimeError('Stage execution requires WSL CUDA; no CPU fallback.')
    repo,source=configure_repository(ROOT)
    versions={p:importlib.metadata.version(p) for p in ('torch','transformer-lens','transformers','datasets','numpy','pandas','matplotlib')}
    config={'model':'gpt2-small','mib_model_id':'gpt2','task':'ioi','seed':args.seed,
        'train_examples':args.train_examples,'validation_examples':args.validation_examples,
        'batch_size':args.batch_size,'precision':'float32','threshold':args.threshold,
        'task_loss':'negative official MIB IOI logit difference; loss=True',
        'decision_rule':'proposed_mean_loss - current_mean_loss < threshold (signed)',
        'traversal':'descending receiver DAG phase, descending sender index, edge name tie break',
        'structural_cleanup':'no automatic cascade/skip in the search; official pruned MIB is an independent loss oracle',
        'long_checks':args.long_checks,'atol':args.atol,'rtol':args.rtol,'warmup':args.warmup,
        'cpu_threads':args.cpu_threads,'source':source,'packages':versions,
        'python':platform.python_version(),'cuda_runtime':torch.version.cuda,
        'wsl':platform.platform(),'git_parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'no_environment_install_or_mattr_import':True,'public_test_accessed':False,
        'independent_original_implementation_available':False,
        'independent_result':'Research log Stage J: 2.35x, auto-circuit IOI, squared probability distance; not this configuration',
        'timing':{'cuda_synchronization':True,'repeat_trials_B_C':args.repeats,'alternate_arm_order':True},
        'stage_c_default':False,'stage_b_sizes':args.sizes,'stage_b_checks':args.checks}
    target=out/'experiment_config.json'
    if target.exists():
        previous=json.loads(target.read_text())
        phase_fields={'git_parent_commit','stage_b_sizes','stage_b_checks','timing'}
        if any(previous.get(k)!=v for k,v in config.items() if k not in phase_fields):
            raise ValueError('Core protocol differs from this campaign. Use a new directory; no silent changes.')
    else: atomic_json(target,config)
    atomic_json(out/f'stage_{args.stage}_configuration.json',{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    log=Path('C:/Users/Nikhil/Downloads/RESEARCH_LOG.md') if os.name=='nt' else Path('/mnt/c/Users/Nikhil/Downloads/RESEARCH_LOG.md')
    if log.exists():
        atomic_json(out/'independent_source.json',{'path':str(log),'sha256':sha(log),
            'stage_j':log.read_text(encoding='utf-8').split('### Stage J',1)[1].split('## Cross-stage',1)[0]})
    meter=None
    try:
        if args.stage=='a':
            command=[sys.executable,'-m','unittest','discover','-s','tests','-p','test_acdc_prefix_cache.py','-v']
            with (out/'unit_tests.log').open('w') as log:
                subprocess.run(command,cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(out/'unit_tests_passed.json',{'passed':True,'tests':10,'command':command,
                'test_source_sha256':sha(ROOT/'tests/test_acdc_prefix_cache.py')})
        atomic_json(out/f'telemetry_{args.stage}_before.json',{'sample':telemetry()})
        model=load_model('cuda')
        from transformer_lens import HookedTransformer
        atomic_json(out/'transformerlens_resume_api.json',{
            'file':inspect.getfile(HookedTransformer.forward),
            'signature':str(inspect.signature(HookedTransformer.forward)),
            'forward_source_sha256':hashlib.sha256(inspect.getsource(HookedTransformer.forward).encode()).hexdigest()})
        data,manifest,info=load_splits(model,{'train':max(args.sizes) if args.stage in ('b','c') else args.train_examples,
                                           'validation':args.validation_examples},args.seed)
        manifest.to_csv(out/f'data_manifest_{args.stage}.csv',index=False)
        atomic_json(out/f'data_selection_{args.stage}.json',info)
        from eap.graph import Graph
        graph=Graph.from_model(model)
        if len(graph.edges)!=32491: raise ValueError('Unexpected GPT-2 MIB graph; inspect before continuing.')
        atomic_json(out/'graph_config.json',dict(graph.cfg))
        deps=Dependencies(graph)
        pd.DataFrame([{'ordinal':i,'edge':n,'receiver_layer':graph.edges[n].child.layer,
            'receiver_phase':deps.phase[graph.edges[n].child.name],
            'sender':graph.edges[n].parent.name,'receiver':graph.edges[n].child.name}
            for i,n in enumerate(deps.order)]).to_csv(out/'candidate_traversal.csv',index=False)
        meter=ExecutionMeter(model)
        with meter.instrument():
            if args.stage=='a': stage_a(model,graph,data,out,args,meter)
            else:
                from .benchmark import run
                run(model,graph,data,out,args,meter)
        atomic_json(out/f'telemetry_{args.stage}_after.json',{'sample':telemetry()})
    except Exception as error:
        atomic_json(out/f'failure_{args.stage}.json',{'error':str(error),'traceback':traceback.format_exc(),
            'settings_unchanged':True,'oom':isinstance(error,torch.cuda.OutOfMemoryError)})
        if isinstance(error,torch.cuda.OutOfMemoryError):
            raise RuntimeError('CUDA OOM: choose an explicit lower batch/cache configuration in a new directory. No settings were changed.') from error
        raise
    finally:
        if meter is not None:
            pd.DataFrame(meter.events).to_csv(out/f'forward_events_{args.stage}.csv',index=False)
            atomic_json(out/f'block_counts_{args.stage}.json',{'counts':meter.block_counts,
                'root_forward_batches':len(meter.events),'example_forwards':sum(x['batch_size'] for x in meter.events),
                'block_executions':sum(meter.block_counts)})
    from .report import generate
    generate(out)
    print('Saved:',out,flush=True)


if __name__=='__main__': main()
