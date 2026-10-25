"""Foreground standalone Gate5G CLI. No registered work starts on import."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from src.mattr_optimization.common import ROOT,configure,sha,write_json,git
from .core import DEFAULT,read,protocol,identity,verified,check_files,digest


def audit(out):
    from .analysis import completeness
    p=protocol(out);completeness(out,p)
    assert read(out/'preflight_gpu.json')['passed']
    fresh=pd.read_csv(out/'cohort_hashes.csv');historical=pd.read_csv(out/'pre_freeze_test_prompts.csv')
    seen=set(fresh.clean_sha256)|set(fresh.corrupted_sha256)
    assert len(seen)==2*len(fresh) and len(fresh)>=256
    assert not seen & (set(historical.clean_sha256)|set(historical.corrupted_sha256))
    check_files(read(out/'protected_history.json'))
    for rec in p['cohort']['exclusions']:
        path=ROOT/rec['path'];assert sha(path)==rec['sha256']
        if 'historical_full' in rec['role']:continue
        old=pd.read_csv(path,keep_default_na=False)
        from src.mattr_compact_validation.design import text_hash
        for col in ('clean_prompt','corrupted_prompt'):
            if col in old:assert not seen & {text_hash(x) for x in old[col]}
    # Verify each output's recorded execution identity, rather than current HEAD (results commit allowed).
    events=examples=0
    for job in p['jobs']:
        folder=out/'evaluations'/job['id']
        assert verified(folder,identity(out,p,job))
        s=read(folder/'summary.json')
        assert s['identity']['implementation_commit']==p['implementation_commit']
        assert s['weight_before']==s['weight_after']==p['model']['weight_sha256']
        assert s['counts']['optimizer_updates']==s['counts']['backward_calls']==0
        assert s['correctness_passed'] and not s['model_gradients_stored']
        q=pd.read_csv(out/'queries.csv');q=q[q.ranking==job['ranking']]
        result=pd.read_csv(folder/'metrics.csv')
        assert result[['selection','target','nominal_n','mask_sha256']].to_dict('records')==q[['selection','target','nominal_n','mask_sha256']].to_dict('records')
        for row in q.itertuples():
            from src.mattr_matched_size.topology import digest as mask_digest
            assert mask_digest(np.load(out/'masks'/f'{row.mask_sha256}.npy'))==row.mask_sha256
    for path in (out/'cache').rglob('*.json'):
        meta=read(path)
        assert sha(path.with_suffix('.npz'))==meta['sha256']
        ctx=meta['identity']['context']
        assert ctx['protocol_sha256']==sha(out/'gate5g_protocol.json')
        assert ctx['cohort_sha256']==p['cohort_sha256'] and ctx['model']==p['model']
    for path in (out/'attempts').rglob('forward_events.csv'):
        if path.stat().st_size<=1:continue
        f=pd.read_csv(path)
        if f.empty:continue
        assert not f.grad_enabled.any()
        # Failed forwards are preserved and charged; completed jobs themselves must be correct.
        accounting=pd.read_csv(path.parent/'compute_accounting.csv')
        assert len(f)==int(accounting.forward_batches.sum())
        assert int(f.batch_size.sum())==int(accounting.example_forwards.sum())
        assert int(accounting.backward_calls.sum())==int(accounting.optimizer_updates.sum())==0
        events+=len(f);examples+=int(f.batch_size.sum())
    from .statistics import resamples
    assert np.array_equal(np.load(out/'bootstrap_indices.npy'),resamples(fresh.template_id.to_numpy()))
    assert sha(out/'bootstrap_indices.npy')==read(out/'analysis_metadata.json')['bootstrap_indices_sha256']
    decision=read(out/'gate5g_decision.json')
    assert decision['part_a']!='Invalid/incomplete' and decision['part_c']['classification']!='Invalid/incomplete'
    # Independently reconstruct the two registered decisions from saved prompt outputs.
    from .statistics import weights,classify_a,classify_c,cpr_bootstrap
    w=weights(np.load(out/'bootstrap_indices.npy'),len(fresh))
    from src.mattr_optimization.common import GRID
    data={}
    def vector(key,selection,target,col):
        if key not in data:data[key]=pd.read_csv(out/'evaluations'/key/'per_prompt.csv')
        f=data[key];rows=f[(f.selection==selection)&np.isclose(f.target,target)].sort_values('example')
        assert rows.example.tolist()==list(range(len(fresh)))
        return rows[col].to_numpy()
    parents=sorted([r for r in p['rankings'] if r['kind']=='core' and r['method']=='baseline'],key=lambda r:r['seed'])
    eap=next(r for r in p['rankings'] if r['method']=='eap_ig')
    label,rows=classify_a(np.stack([vector(eap['key'],'nominal',f,'kl') for f in (.1,.2,.5)]),
        np.stack([[vector(r['key'],'nominal',f,'kl') for f in (.1,.2,.5)] for r in parents]),w)
    assert label==decision['part_a']
    np.testing.assert_allclose([[r['difference'],r['lower'],r['upper']] for r in rows],
        [[r['difference'],r['lower'],r['upper']] for r in decision['part_a_endpoints']],atol=1e-12)
    cpr_points=[];cpr_draws=[];acc=[]
    for parent in parents:
        child=next(r for r in p['rankings'] if r.get('parent')==parent['key'] and r.get('k')==324)
        clean=vector(parent['key'],'nominal',.001,'clean_margin')
        empty=vector(parent['key'],'nominal',.001,'corrupted_margin')
        a,_=cpr_bootstrap(np.stack([vector(parent['key'],'nominal',f,'margin') for f in GRID]),clean,empty,w)
        b,_=cpr_bootstrap(np.stack([vector(child['key'],'nominal',f,'margin') for f in GRID]),clean,empty,w)
        cpr_draws.append(b-a)
        a0=read(out/'evaluations'/parent['key']/'summary.json')['official_result'][1]
        b0=read(out/'evaluations'/child['key']/'summary.json')['official_result'][1]
        cpr_points.append(b0-a0)
        acc.append((vector(child['key'],'cap',116,'margin')>0).astype(float)-(vector(parent['key'],'cap',116,'margin')>0))
    result=classify_c(cpr_points,np.stack(cpr_draws),np.stack(acc),w)
    assert result['classification']==decision['part_c']['classification']
    np.testing.assert_allclose(result['cpr_interval'],decision['part_c']['cpr_interval'],atol=1e-12)
    np.testing.assert_allclose(result['accuracy_interval'],decision['part_c']['accuracy_interval'],atol=1e-12)
    frozen=read(out/'frozen_at.json')
    for path in (out/'attempts').rglob('configuration.json'):
        saved=read(path);attempt_record=read(path.parent/'attempt.json')
        assert saved['implementation_commit']==p['implementation_commit']
        assert saved['protocol_sha256']==sha(out/'gate5g_protocol.json')
        assert attempt_record['started_utc']>=frozen['utc'],'Fresh evaluation was attempted before freeze'
    write_json(out/'final_audit.json',dict(passed=True,protocol_sha256=sha(out/'gate5g_protocol.json'),
        implementation_commit=p['implementation_commit'],fresh_prefreeze_forwards=0,history_preserved=True,
        jobs=len(p['jobs']),root_forwards=events,example_forwards=examples,backwards=0,optimizer_updates=0,
        cached_artifacts_hashed=True,job_commits_verified=True,registered_decisions_reproduced=True,interpretation_required=False))
    print('Final audit passed. Empty researcher interpretation does not invalidate the audit.',flush=True)


def package(out,commit=False,dry_run=False):
    from .analysis import REPORT,INTERPRETATION
    p=protocol(out);audit_record=read(out/'final_audit.json')
    assert audit_record['passed'] and audit_record['protocol_sha256']==sha(out/'gate5g_protocol.json')
    # Explicit allowlist: never stage prompts, full-vocabulary references, tensor scores, checkpoints or caches.
    names=['gate5g_protocol.json','gate5g_protocol.sha256','frozen_at.json','ranking_inventory.csv','excluded_rankings.json',
        'eap_ordering.json','splice_topology.csv','topology_audit.json','cohort_hashes.csv','cohort_audit.json',
        'preflight_cpu.json','preflight_gpu.json','pre_freeze_test_prompts.csv','historical_evidence.json',
        'gate5g_decision.json','behavioral_metrics.csv','part_a_registered_comparisons.csv','splice_paired_differences.csv',
        'cap116_fidelity_per_seed.csv','cap116_fidelity_summary.csv','ranking_features.csv','seed_diagnostics.csv',
        'known_head_reference.csv','feature_associations.csv','paired_feature_changes.csv','seed42_edge_case_study.csv',
        'seed42_template_errors.csv','historical_training_diagnostics.csv','historical_discovery_cost.csv',
        'graph_analysis_compute.csv','compute_accounting.csv','official_metrics.csv','analysis_metadata.json',
        'completeness.json','final_audit.json',REPORT]
    files=[out/name for name in names]+sorted((out/'figures').glob('*.png'))+[ROOT/REPORT]
    for path in files:
        if not path.exists():raise FileNotFoundError(path)
        if path.stat().st_size>10_000_000:raise ValueError('Package file exceeds10MB: '+str(path))
    body=(out/REPORT).read_text(encoding='utf-8')
    if not body.split(INTERPRETATION,1)[-1].strip():
        print('WARNING: researcher-written interpretation is empty. Review actual results before committing.',flush=True)
    paths=[path.relative_to(ROOT).as_posix() for path in files]
    print('Reviewed small result files:\n'+'\n'.join(paths),flush=True)
    if dry_run:return
    write_json(out/'results_package_manifest.json',{path:sha(ROOT/path) for path in paths})
    paths.append((out/'results_package_manifest.json').relative_to(ROOT).as_posix())
    # Preserve unrelated staging; refuse mixing it into a Gate5G results commit.
    staged=set(git(ROOT,'diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))-{''}
    if staged-set(paths):raise ValueError('Unrelated staged files exist; package refuses to mix them')
    subprocess.run(['git','add','-f','--',*paths],cwd=ROOT,check=True)
    for path in git(ROOT,'diff','--cached','--name-only','-z').decode().split('\0'):
        if path and int(git(ROOT,'cat-file','-s',':'+path))>10_000_000:raise ValueError('Staged blob exceeds10MB')
    print(git(ROOT,'diff','--cached','--stat').decode(),flush=True)
    if commit:
        subprocess.run(['git','commit','-m','research: Gate 5G fidelity and splice results'],cwd=ROOT,check=True)
    print('No automatic push. After review/commit:\ngit push origin research/mattr-fidelity-splice',flush=True)


def run_jobs(out,stage,args):
    p=protocol(out,live=True)
    jobs=[j for j in p['jobs'] if (stage=='run-all' or j['part']=={'fidelity':'A','diagnose':'B','splice':'C'}[stage])]
    # Parts A,B,C fixed sequence; resume skips only integrity-checked completed jobs.
    jobs=sorted(jobs,key=lambda j:({'A':0,'B':1,'C':2}[j['part']],list(p['jobs']).index(j)))
    if args.job:jobs=[j for j in jobs if j['id']==args.job]
    if args.method:jobs=[j for j in jobs if next(r for r in p['rankings'] if r['key']==j['ranking'])['method']==args.method]
    if args.seed is not None:jobs=[j for j in jobs if next(r for r in p['rankings'] if r['key']==j['ranking'])['seed']==args.seed]
    if args.k is not None:jobs=[j for j in jobs if next(r for r in p['rankings'] if r['key']==j['ranking']).get('k')==args.k]
    if not jobs:raise ValueError('No registered jobs match selection')
    if args.dry_run:
        print(json.dumps(jobs,indent=2));return
    for j in jobs:
        command=[sys.executable,'-u','-m','src.mattr_fidelity_splice.cli','evaluate','--out',str(out),'--job',j['id']]
        if args.approve_over_5h:command.append('--approve-over-5h')
        print('Foreground:', ' '.join(command),flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    if stage=='diagnose':
        from .structure import features
        features(out,p)


class Tee:
    def __init__(self,console,log):self.console,self.log=console,log
    def write(self,value):self.console.write(value);self.log.write(value);self.log.flush()
    def flush(self):self.console.flush();self.log.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['preflight','freeze','fidelity','diagnose','splice','evaluate','run-all','analyze','audit','package'])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--cpu-only',action='store_true',help='preflight topology/data/provenance only; GPU correctness still required before freeze')
    parser.add_argument('--dry-run',action='store_true',help='list registered jobs/packaged files without model forwards or staging')
    parser.add_argument('--job',help='exact registered job/ranking ID')
    parser.add_argument('--method',choices=['baseline','h2','h2_regularized','eap_ig'])
    parser.add_argument('--seed',type=int,choices=range(42,47))
    parser.add_argument('--k',type=int,choices=[324,1624])
    parser.add_argument('--approve-over-5h',action='store_true',help='explicit manual approval after reviewing cumulative projection')
    parser.add_argument('--commit',action='store_true',help='package only: create reviewed results commit; never pushes')
    a=parser.parse_args();configure();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    logs=out/'logs';logs.mkdir(exist_ok=True)
    logfile=logs/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+a.stage+'_'+str(os.getpid())+'.log')
    with logfile.open('w',encoding='utf-8',buffering=1) as log,contextlib.redirect_stdout(Tee(sys.stdout,log)),contextlib.redirect_stderr(Tee(sys.stderr,log)):
        write_json(out/'status.json',dict(stage=a.stage,status='running',pid=os.getpid()))
        print('Gate5G',a.stage,'out:',out,flush=True)
        try:
            if a.stage in ('fidelity','diagnose','splice','run-all'):run_jobs(out,a.stage,a)
            elif a.stage=='evaluate':
                if not a.job:raise ValueError('--job is required')
                if a.dry_run:
                    p=protocol(out,live=True)
                    if a.job not in [j['id'] for j in p['jobs']]:raise ValueError('Unregistered job')
                    print(a.job);return
                from .inference import evaluate
                evaluate(out,a.job,a.approve_over_5h)
            elif a.stage=='preflight':
                if a.dry_run:print('CPU: historical hashes, topology/splices, official cohort. GPU: two historical prompts ONLY. Zero registered evaluations.');return
                if (out/'gate5g_protocol.json').exists():
                    protocol(out)
                    print('Frozen preflight verified; no design or correctness artifacts rewritten.',flush=True)
                    return
                from .design import preflight_cpu
                from .inference import gpu_preflight
                sources=list((ROOT/'src/mattr_fidelity_splice').glob('*.py'))+[ROOT/'tests/test_mattr_fidelity_splice.py']
                tested={path.relative_to(ROOT).as_posix():sha(path) for path in sources}
                result=subprocess.run([sys.executable,'-m','pytest','tests/test_mattr_fidelity_splice.py','-q'],cwd=ROOT,check=False)
                write_json(out/'unit_tests.json',dict(passed=result.returncode==0,tested_sources=tested))
                if result.returncode:raise RuntimeError('CPU correctness tests failed; stop preflight')
                preflight_cpu(out)
                if not a.cpu_only:gpu_preflight(out)
            elif a.stage=='freeze':
                if a.dry_run:print('Freeze immutable protocol after clean implementation commit and preflight. Zero forwards.');return
                from .design import freeze
                freeze(out)
            elif a.stage=='analyze':
                from .analysis import analyze
                analyze(out)
            elif a.stage=='audit':audit(out)
            elif a.stage=='package':package(out,a.commit,a.dry_run)
            write_json(out/'status.json',dict(stage=a.stage,status='complete'))
        except BaseException as exc:
            import traceback
            traceback.print_exc()
            write_json(out/'status.json',dict(stage=a.stage,status='failed',error=repr(exc)))
            raise


if __name__=='__main__':main()
