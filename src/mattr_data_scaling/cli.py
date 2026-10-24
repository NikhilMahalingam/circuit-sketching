"""Standalone Gate5F stages; never needs Codex or an LLM API."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from src.mattr_optimization.common import ROOT,configure,sha,write_json
from .core import DEFAULT,protocol,verified,endpoint_key,job_key

def audit(out):
    from .analysis import expected
    from .training import checked_checkpoint
    p=protocol(out)
    assert json.loads((out/'completeness.json').read_text())['complete']
    for cohort,key,rec in expected(p):
        folder=out/'evaluations'/cohort/key;assert verified(folder),str(folder)
        result=json.loads((folder/'official_result.json').read_text());assert result['correctness_passed']
        assert result['counts']['optimizer_updates']==result['counts']['backward_calls']==0
        assert result['weight_hash']==json.loads((out/'preflight_gpu.json').read_text())['model_weight_sha256']
    for j in p['training_jobs']:
        folder=out/'training'/j['key'];assert verified(folder),j['key']
        for path in (folder/'checkpoints').glob('step_*.pt'):
            state=checked_checkpoint(path)
            import pandas as pd
            plan=pd.read_csv(out/f'stream_n{j["dataset_size"]}_s{j["seed"]}.csv',float_precision='round_trip')
            assert [r[1] for r in state['train_log']]==plan.k.iloc[:state['completed']].tolist()
            prefix=plan.iloc[:state['completed']].to_csv(index=False,float_format='%.17g').encode()
            assert __import__('hashlib').sha256(prefix).hexdigest()==state['stream_prefix_hash']
    for path in (out/'attempts').rglob('model_weight_audit.json'):
        a=json.loads(path.read_text());assert a['passed'] and a['before']==a['after']
    previous=json.loads((out/'protected_previous_hashes.json').read_text())
    for name,h in previous.items():assert sha(ROOT/name)==h,name
    import pandas as pd
    from src.mattr_compact_validation.design import text_hash
    primary=pd.read_csv(out/'cohort_primary.csv');development=pd.read_csv(out/'cohort_development.csv');train=pd.read_csv(out/'train_n10000.csv')
    seen=set(primary.clean_sha256)|set(primary.corrupted_sha256)
    assert len(seen)==1024 and not seen&(set(development.clean_sha256)|set(development.corrupted_sha256)|set(train.clean_sha256)|set(train.corrupted_sha256))
    for entry in json.loads((out/'cohort_audit.json').read_text())['exclusions']:
        path=ROOT/entry['path'];assert sha(path)==entry['sha256']
        if 'historical_full' in entry['role']:continue
        frame=pd.read_csv(path,keep_default_na=False)
        for col in ('clean_prompt','corrupted_prompt'):
            if col in frame:assert not seen&{text_hash(x) for x in frame[col]}
    from .statistics import contrasts
    arrays={}
    for e in p['endpoints']:
        frame=pd.read_csv(out/'evaluations/primary'/endpoint_key(e)/'cap_per_prompt.csv')
        for target,g in frame.groupby('target'):arrays[e['method'],e['dataset_size'],e['step'],e['seed'],target]=g.sort_values('example').correct.to_numpy(dtype=float)
    result,_,_=contrasts(arrays,primary.template_id.to_numpy())
    assert result==json.loads((out/'gate5f_decision.json').read_text())['primary']
    # CPU aggregation requires no forwards; actual counters reconcile to raw per-attempt root events.
    for path in (out/'attempts').rglob('compute_accounting.csv'):
        frame=pd.read_csv(path);events=pd.read_csv(path.parent/'forward_events.csv')
        assert int(frame.forward_batches.sum())==len(events)
        assert int(frame.example_forwards.sum())==int(events.batch_size.sum())
    write_json(out/'final_audit.json',dict(passed=True,protocol_sha256=sha(out/'gate5f_protocol.json'),historical_files_unchanged=len(previous),
        paired_streams_verified=True,cohorts_disjoint=True,model_weights_unchanged=True,classification_reproduced=True,counts_reconciled=True))
    report=out/'GATE5F_DATA_SCALING_REPORT.md'
    if report.exists():
        body=report.read_text().split('\n## Final audit')[0]+'\n## Final audit\n\nAll protocol, history, stream, cohort, model, counter and decision checks passed. See final_audit.json.\n'
        report.write_text(body)
        (ROOT/'GATE5F_DATA_SCALING_REPORT.md').write_text(body.replace('(figures/',f'(results/mattr_data_scaling/{out.name}/figures/'))
    print('Final audit passed',flush=True)

def package(out,implementation_only=False):
    dest=out/'package';dest.mkdir(exist_ok=True);paths=[]
    names=['gate5f_protocol.json','gate5f_protocol.sha256','checkpoint_inventory.csv','replacement_job_manifest.json','TRAINING_SCHEDULE_AUDIT.md',
        'schedule_audit.json','preflight_cpu.json','preflight_gpu.json','eap_preflight.json','eap_preflight_throughput.csv','known_ioi_heads.json',
        'command_manifest.json','command_manifest.sha256','cohort_primary_hashes.csv','cohort_development_hashes.csv','cohort_audit.json',
        'stream_convention.json','historical_evidence.json','historical_ranking_references.json','mapping_audit.json','group_sizes.csv',
        'bounded_smoke_summary.json','cli_help_tests.json','implementation_tests.json','runtime_projection.json','protocol_frozen_at.json']
    if not implementation_only:
        assert json.loads((out/'final_audit.json').read_text())['passed']
        names+=['GATE5F_DATA_SCALING_REPORT.md','gate5f_decision.json','official_metrics.csv','accuracy_size_curves.csv','factorial_contrasts.csv',
            'primary_seed_effects.csv','paired_seed_effects.csv','absolute_method_changes.csv','exact_size_subsets.csv','known_head_overlap.csv','enrichment_vs_topology.csv',
            'topology_control_audit.json','compute_summary.csv','checkpoint_serialization_cost.csv','reused_training_cost.csv','final_audit.json','completeness.json','analysis_metadata.json','runtime_projection.json']
    for name in names:
        source=out/name
        if not source.exists():raise FileNotFoundError(source)
        shutil.copyfile(source,dest/name);paths.append(str((dest/name).relative_to(ROOT)))
    if (out/'figures').exists() and not implementation_only:
        for source in (out/'figures').glob('*.png'):
            target=dest/'figures'/source.name;target.parent.mkdir(exist_ok=True);shutil.copyfile(source,target);paths.append(str(target.relative_to(ROOT)))
    write_json(dest/'package_manifest.json',dict(implementation_only=implementation_only,sha256={str(Path(n).relative_to(dest.relative_to(ROOT))):sha(ROOT/n) for n in paths},
        excluded='raw prompts, model weights, tensor checkpoints, mutable caches, per-prompt tables and large control replicates'))
    print('Small deliverables:',dest,flush=True)

def run_all(out,approved=False):
    p=protocol(out);manifest=out/'command_manifest.json'
    assert sha(manifest)==(out/'command_manifest.sha256').read_text().strip()
    commands=json.loads(manifest.read_text());assert commands['protocol_sha256']==sha(out/'gate5f_protocol.json')
    assert commands['commands']==p['commands'],'Command manifest no longer matches frozen protocol'
    print('Sequential foreground execution only. Ctrl+C preserves atomic checkpoints; rerun this command to resume.',flush=True)
    for i,rec in enumerate(commands['commands'],1):
        command=[sys.executable,'-u','-m','src.mattr_data_scaling.cli',rec['stage'],'--out',str(out),*rec['args']]
        if approved:command+=['--approve-over-12h']
        print(f'Job{i}/{len(commands["commands"])}:', ' '.join(command),flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    for stage in ('analyze','audit','package'):
        subprocess.run([sys.executable,'-u','-m','src.mattr_data_scaling.cli',stage,'--out',str(out)],cwd=ROOT,check=True)

class Tee:
    def __init__(self,console,log):self.console,self.log=console,log
    def write(self,value):self.console.write(value);self.log.write(value);self.log.flush()
    def flush(self):self.console.flush();self.log.flush()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','freeze','train','eap','evaluate','analyze','audit','package','run-all'])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--dataset-size',type=int,choices=[128,10000],default=128)
    parser.add_argument('--method',choices=['baseline','h2','eap_ig'],default='baseline')
    parser.add_argument('--seed',type=int,choices=range(42,47),default=42)
    parser.add_argument('--max-steps',type=int,choices=[1000,5000],default=5000)
    parser.add_argument('--step',type=int,choices=[0,1000,5000],default=5000)
    parser.add_argument('--cohort',choices=['primary','development'],default='primary')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--cpu-only',action='store_true',help='preflight CPU preparation only; cannot freeze until GPU correctness preflight passes')
    parser.add_argument('--approve-over-12h',action='store_true',help='Explicit manual approval to continue the unchanged campaign beyond projected12hours')
    parser.add_argument('--allow-incomplete',action='store_true',help='write missing-job status only; no classification when incomplete')
    parser.add_argument('--implementation-only',action='store_true',help='package preflight/protocol only, before registered experiments')
    a=parser.parse_args();configure();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    logs=out/'logs';logs.mkdir(exist_ok=True)
    path=logs/(time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'_'+a.stage+'_'+str(os.getpid())+'.log')
    def failure_log(kind,value,tb):
        import traceback
        with path.open('a') as handle:traceback.print_exception(kind,value,tb,file=handle)
        sys.__excepthook__(kind,value,tb)
    sys.excepthook=failure_log
    with path.open('w',buffering=1) as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        print('Stage',a.stage,'artifact directory',out,flush=True)
        if a.approve_over_12h:
            write_json(out/'manual_runtime_approval.json',dict(approved=True,protocol_sha256=sha(out/'gate5f_protocol.json'),command=sys.argv,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        if a.stage=='preflight':
            from .design import cpu_preflight
            from .checks import gpu_preflight
            if (out/'gate5f_protocol.json').exists():protocol(out);print('Frozen preflight verified; no rerun.');return
            if not (out/'preflight_cpu.json').exists():cpu_preflight(out)
            else:assert json.loads((out/'preflight_cpu.json').read_text())['passed']
            if not a.cpu_only and not (out/'preflight_gpu.json').exists():gpu_preflight(out)
        elif a.stage=='freeze':
            from .design import freeze
            freeze(out)
            from .core import runtime_guard
            runtime_guard(out,a.approve_over_12h)
        elif a.stage=='train':
            from .training import train
            train(out,a.dataset_size,a.method,a.seed,a.max_steps,a.resume,a.approve_over_12h)
        elif a.stage=='eap':
            from .attribution import run
            run(out,a.dataset_size,a.seed,a.approve_over_12h)
        elif a.stage=='evaluate':
            from .evaluate import evaluate
            evaluate(out,a.method,a.dataset_size,a.seed,a.step,a.cohort,a.approve_over_12h)
        elif a.stage=='analyze':
            from .analysis import analyze
            analyze(out,a.allow_incomplete)
        elif a.stage=='audit':audit(out)
        elif a.stage=='package':package(out,a.implementation_only)
        elif a.stage=='run-all':run_all(out,a.approve_over_12h)

if __name__=='__main__':main()
