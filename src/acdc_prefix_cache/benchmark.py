"""Opt-in Stage B/C trials. Construction, warmup, search and validation are distinct."""
import gc
import json
from datetime import datetime,timezone
import shutil
import subprocess
import threading
import time

import numpy as np
import pandas as pd
import torch

from mib_ioi.runtime import subset
from .engine import ExactExecutor,prepare_batches,sync
from .graph import Dependencies,digest
from .search import exact_acdc,atomic_json


class GPUTrace:
    """Sample clocks/power during the method, without changing power settings."""
    def __init__(self):
        self.rows=[];self.stop=threading.Event()
        self.worker=threading.Thread(target=self.collect,daemon=True)

    def collect(self):
        while not self.stop.is_set():
            try:
                raw=subprocess.check_output(['nvidia-smi',
                    '--query-gpu=clocks.sm,power.draw,temperature.gpu,memory.used',
                    '--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()[0]
                clock,power,temp,memory=map(float,raw.split(','))
                self.rows.append({'monotonic_seconds':time.perf_counter(),'sm_clock_mhz':clock,
                    'power_watts':power,'temperature_c':temp,'device_memory_mib':memory})
            except (OSError,ValueError,subprocess.SubprocessError):pass
            self.stop.wait(1.)

    def begin(self):self.worker.start()
    def end(self):self.stop.set();self.worker.join(timeout=6)


def check_gate(out):
    from .runner import sha,ROOT
    gate=json.loads((out/'stage_a_passed.json').read_text())
    if not gate['passed']: raise ValueError('Stage A did not pass.')
    for name in ('engine.py','graph.py','search.py','runner.py'):
        relative='src/acdc_prefix_cache/'+name
        if sha(ROOT/relative)!=gate['core_source_sha256'][relative]:
            raise ValueError(f'Correctness-tested core changed: {name}; rerun Stage A in a new campaign.')


def release(executor):
    executor.close();gc.collect();torch.cuda.empty_cache()


def layerwise(model,graph,data,out,args,meter):
    from .runner import LAYERS,single_edges
    path=out/'layerwise_benchmark.csv'
    if (out/'layerwise_complete.json').exists(): return
    rows=[];batches=prepare_batches(model,subset(data['train'],range(args.train_examples)),args.batch_size)
    print(f'Layerwise: {len(LAYERS)} layers x 3 q/k/v queries x {args.repeats} trials x 2 methods; '
          f'{len(batches)} batches/query, {args.warmup} warmup queries/layer/method. '
          'Micro timings exclude preparation, which is recorded separately.',flush=True)
    expected={}
    for repeat in range(args.repeats):
        for method in (('standard','cached') if repeat%2==0 else ('cached','standard')):
            meter.phase=f'b_micro_{repeat}_{method}_preparation'
            e=ExactExecutor(model,graph,batches,cached=method=='cached').prepare()
            try:
                for layer in LAYERS:
                    queries=[n for n in single_edges(graph) if graph.edges[n].child.layer==layer
                             and graph.edges[n].child.name.startswith('a')][:3]
                    meter.phase=f'b_micro_{repeat}_{method}_layer{layer}_warmup'
                    for _ in range(args.warmup):e.evaluate({queries[0]})
                    torch.cuda.reset_peak_memory_stats()
                    for name in queries:
                        meter.phase=f'b_micro_{repeat}_{method}_layer{layer}'
                        values=e.evaluate({name});key=(repeat,name)
                        error=np.nan
                        if key in expected:
                            np.testing.assert_allclose(values,expected[key],atol=args.atol,rtol=args.rtol)
                            error=float(np.max(abs(values-expected[key])))
                            for prior in rows:
                                if prior['trial']==repeat and prior['edge']==name:prior['max_absolute_error']=error
                        else:expected[key]=values.copy()
                        rows.append({'stage':'B_micro','method':method,'trial':repeat,'edge':name,
                            'earliest_affected_layer':layer,'seconds':e.evaluations[-1]['wall_seconds'],
                            'mean_loss':float(values.mean()),'max_absolute_error':error,
                            'cache_setup_seconds':e.setup_seconds,'cache_bytes':e.cache_bytes,
                            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
                            'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
                    pd.DataFrame(rows).to_csv(path,index=False)
            finally:release(e)
    atomic_json(out/'layerwise_complete.json',{'passed':True,'rows':len(rows),'repeats':args.repeats})


def trial(model,graph,train,validation,order,out,args,meter,method,interval):
    from .runner import official_loss,telemetry
    completion=out/'result.json'
    if completion.exists():return json.loads(completion.read_text())
    out.mkdir(parents=True,exist_ok=True)
    before_telemetry=telemetry();e=None;start=time.perf_counter();success=False;result=None
    gpu_trace=GPUTrace();gpu_trace.begin()
    event_start=len(meter.events)
    try:
        meter.phase='benchmark_'+method+'_preparation'
        # Each arm pays its own tokenization, donor construction, baseline and
        # (only cached) clean-prefix capture. No prebuilt cache is donated.
        sync(model);start=time.perf_counter()
        batches=prepare_batches(model,train,args.batch_size)
        e=ExactExecutor(model,graph,batches,cached=method=='cached').prepare()
        sync(model);preparation=time.perf_counter()-start
        meter.phase='benchmark_'+method+'_warmup'
        warm_start=time.perf_counter()
        for _ in range(args.warmup):e.evaluate({order[0]})
        sync(model);warm_seconds=time.perf_counter()-warm_start
        query_start=len(e.evaluations)
        meter.phase='benchmark_'+method+'_search'
        torch.cuda.reset_peak_memory_stats();search_start=time.perf_counter()
        result=exact_acdc(e,order,args.threshold,checkpoint=out/'checkpoint.json',checkpoint_every=interval)
        sync(model);search_elapsed=time.perf_counter()-search_start
        peak_allocated=torch.cuda.max_memory_allocated()/2**20
        peak_reserved=torch.cuda.max_memory_reserved()/2**20
        meter.phase='benchmark_'+method+'_validation'
        t0=time.perf_counter()
        heldout=official_loss(model,validation,result['removed'],args.batch_size)
        sync(model);val_seconds=time.perf_counter()-t0
        records=e.evaluations[query_start:]
        row={'method':method,'discovery_examples':len(train),'candidate_checks':len(order),
            'actual_checks_this_attempt':len(records),'full_graph':len(order)==len(graph.edges),
            'cache_preparation_seconds':preparation,'warmup_seconds_excluded':warm_seconds,
            'search_seconds':search_elapsed,'total_seconds':preparation+search_elapsed,
            'validation_seconds':val_seconds,'including_validation_seconds':preparation+search_elapsed+val_seconds,
            'suffix_and_metric_seconds':sum(x['suffix_and_metric_seconds'] for x in records),
            'intervention_bookkeeping_seconds':sum(x['bookkeeping_seconds'] for x in records),
            'peak_allocated_mib':peak_allocated,'peak_reserved_mib':peak_reserved,
            'cache_bytes':e.cache_bytes,'accepted_removals':len(result['removed']),
            'final_edge_count':len(result['retained']),'circuit_hash':result['circuit_hash'],
            'validation_loss':float(heldout.mean()),'seconds_per_check':search_elapsed/max(1,len(records)),
            'resumed':(out/'attempts.json').exists(),'success':True}
        prior_attempts=json.loads((out/'attempts.json').read_text()) if (out/'attempts.json').exists() else []
        prior_seconds=sum(x['actual_wall_seconds_including_warmup'] for x in prior_attempts)
        row['previous_attempt_wall_seconds']=prior_seconds
        row['total_seconds']+=prior_seconds
        row['including_validation_seconds']+=prior_seconds
        atomic_json(out/'trajectory.json',result)
        atomic_json(out/'validation_losses.json',heldout.tolist())
        atomic_json(completion,row);success=True
        return row
    finally:
        sync(model)
        gpu_trace.end()
        attempts_path=out/'attempts.json'
        attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
        attempts.append({'success':success,'actual_wall_seconds_including_warmup':time.perf_counter()-start,
            'actual_forward_batches':len(meter.events)-event_start,'telemetry_before':before_telemetry,
            'telemetry_after':telemetry()})
        pd.DataFrame(meter.events[event_start:]).to_csv(out/f'forward_events_attempt{len(attempts)}.csv',index=False)
        pd.DataFrame(gpu_trace.rows).to_csv(out/f'gpu_telemetry_attempt{len(attempts)}.csv',index=False)
        atomic_json(attempts_path,attempts)
        if e is not None:release(e)


def run(model,graph,data,out,args,meter):
    from .runner import compare_trace,sha
    check_gate(out)
    # Keep raw evidence from earlier unsuccessful process invocations.
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for name in (f'forward_events_{args.stage}.csv',f'block_counts_{args.stage}.json'):
        source=out/name
        if source.exists():shutil.copyfile(source,out/(source.stem+'_previous_'+stamp+source.suffix))
    deps=Dependencies(graph)
    protocol={'stage':args.stage,'sizes':args.sizes,'checks':args.checks if args.stage=='b' else [len(deps.order)],
        'repeats':args.repeats,'warmup':args.warmup,'batch_size':args.batch_size,'threshold':args.threshold,
        'graph_identity':deps.identity,'traversal_sha256':digest(deps.order),
        'checkpoint_interval':100 if args.stage=='b' else 1000,
        'benchmark_source_sha256':sha(__file__)}
    target=out/f'benchmark_{args.stage}_protocol.json'
    if target.exists() and json.loads(target.read_text())!=protocol:
        raise ValueError('Benchmark protocol changed. Use a new campaign with a matching Stage A.')
    atomic_json(target,protocol)
    if args.stage=='b':layerwise(model,graph,data,out,args,meter)
    rows=[];comparison=[]
    full_path=out/'full_acdc_benchmark.csv'
    other_stages=pd.read_csv(full_path) if full_path.exists() else pd.DataFrame()
    if not other_stages.empty:other_stages=other_stages[other_stages.stage!=args.stage.upper()]
    sizes=args.sizes
    counts=args.checks if args.stage=='b' else [len(deps.order)]
    print(f'Stage {args.stage.upper()}: sizes {sizes}, checks {counts}, repeats {args.repeats}; '
          f'{2*len(sizes)*len(counts)*args.repeats} independent method runs, including setup. '
          'Full-graph searches can take hours; Stage C required explicit enablement.',flush=True)
    for size in sizes:
        train=subset(data['train'],range(size))
        for checks in counts:
            if not 1<=checks<=len(deps.order):raise ValueError('Invalid candidate-check count.')
            order=list(deps.order[:checks]);pair_root=out/f'stage_{args.stage}/n{size}_checks{checks}'
            for repeat in range(args.repeats):
                pair={}
                for method in (('standard','cached') if repeat%2==0 else ('cached','standard')):
                    folder=pair_root/f'trial{repeat}'/method
                    row=trial(model,graph,train,data['validation'],order,folder,args,meter,method,protocol['checkpoint_interval'])
                    row.update({'stage':args.stage.upper(),'trial':repeat})
                    rows.append(row)
                    pair[method]=json.loads((folder/'trajectory.json').read_text())
                    pd.concat([other_stages,pd.DataFrame(rows)],ignore_index=True).to_csv(full_path,index=False)
                compare_trace(f'{args.stage}_n{size}_k{checks}_r{repeat}',pair['standard'],pair['cached'],comparison,args.atol,args.rtol)
                va=json.loads((pair_root/f'trial{repeat}/standard/validation_losses.json').read_text())
                vb=json.loads((pair_root/f'trial{repeat}/cached/validation_losses.json').read_text())
                np.testing.assert_allclose(va,vb,atol=args.atol,rtol=args.rtol)
                pd.DataFrame(comparison).to_csv(out/f'benchmark_decision_comparison_{args.stage}.csv',index=False)
                a,b=rows[-2:]
                print(f'n={size}, checks={checks}, trial={repeat}: identical retained edges and decisions; '
                      f'{a["method"]} {a["total_seconds"]:.1f}s / {b["method"]} {b["total_seconds"]:.1f}s',flush=True)
    memory_path=out/'memory_benchmark.csv'
    previous=pd.read_csv(memory_path) if memory_path.exists() else pd.DataFrame()
    if not previous.empty:previous=previous[previous.stage!=args.stage.upper()]
    memory_rows=pd.DataFrame(rows)[['stage','method','trial','discovery_examples','candidate_checks',
        'cache_bytes','cache_preparation_seconds','peak_allocated_mib','peak_reserved_mib']]
    memory_rows['isolated_memory_comparison']=True
    pd.concat([previous,memory_rows],ignore_index=True).to_csv(memory_path,index=False)
    atomic_json(out/f'stage_{args.stage}_passed.json',{'passed':True,'runs':len(rows),'protocol':protocol,
        'partial_trajectories_only':args.stage=='b','all_decisions_and_circuits_match':True})
