"""Pure experiment design, integrity, paired streams and execution records."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import time
import uuid
import numpy as np
from src.mattr_optimization.common import ROOT, sha, write_json

DEFAULT = ROOT/'results/mattr_data_scaling/gate5f'
CAPS = [60,80,100,116,140,180,240,296,400,640]
SEEDS = [42,43,44,45,46]
H2 = ROOT/'results/mattr_hierarchy/20261022_gate5c'
G5E = ROOT/'results/mattr_matched_size/20261023_gate5e'
G5D = ROOT/'results/mattr_compact_validation/20261023_gate5d'

def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def matrix():
    return [dict(method=m,dataset_size=n,seed=r,step=s)
            for n in (128,10000) for r in (SEEDS if n==10000 else SEEDS[:3])
            for m in ('baseline','h2') for s in (1000,5000)
            if not (r>44 and s==1000)]

def job_key(method,dataset_size,seed):
    return f'{method}_n{dataset_size}_s{seed}'

def endpoint_key(e):
    return job_key(e['method'],e['dataset_size'],e['seed'])+f'_step{e["step"]}'

def atomic_torch(path,obj):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(f'Immutable checkpoint already exists: {path}')
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    torch.save(obj,tmp);os.replace(tmp,path)

def marker(folder,identity,files,**extra):
    write_json(Path(folder)/'completed.json',{'passed':True,'identity':identity,
               'sha256':{str(Path(p).relative_to(folder)):sha(p) for p in files},**extra})

def verified(folder,identity=None):
    path=Path(folder)/'completed.json'
    if not path.exists():return False
    data=json.loads(path.read_text())
    if not data['passed'] or (identity is not None and data['identity']!=identity):
        raise ValueError(f'Completed artifact identity mismatch: {folder}')
    for name,h in data['sha256'].items():
        if sha(Path(folder)/name)!=h:raise ValueError(f'Corrupt completed artifact: {folder}/{name}')
    return True

def protocol(out):
    path=out/'gate5f_protocol.json'
    if sha(path)!=(out/'gate5f_protocol.sha256').read_text().strip():raise ValueError('Protocol hash changed')
    p=json.loads(path.read_text())
    for name,h in p['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Frozen implementation changed: '+name)
    for name,h in p['artifact_sha256'].items():
        if sha(out/name)!=h:raise ValueError('Frozen design artifact changed: '+name)
    return p

@contextlib.contextmanager
def attempt(out,key):
    """Foreground process; Linux lock, append-only unique attempt directory."""
    import fcntl
    folder=out/'attempts'/key/(time.strftime('%Y%m%dT%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    locks=out/'locks';locks.mkdir(exist_ok=True)
    with (locks/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        record={'job_id':folder.name,'key':key,'pid':os.getpid(),'host':socket.gethostname(),
                'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'running'}
        write_json(folder/'attempt.json',record);start=time.perf_counter()
        try:
            yield folder
            record['status']='complete'
        except BaseException as exc:
            record.update(status='interrupted_or_failed',error=repr(exc));raise
        finally:
            record.update(seconds=time.perf_counter()-start,ended_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            write_json(folder/'attempt.json',record)

def configure_runtime():
    from src.mattr_optimization.common import configure
    configure()
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no automatic fallback or settings changes')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

def verify_runtime(out):
    from src.mattr_optimization.common import provenance
    p=protocol(out)
    if provenance()!=p['source_provenance']:raise ValueError('Pinned environment/upstream/runtime differs from frozen protocol; no silent dependency changes')
    return p

def close_meter(meter):
    meter.stop.set();meter.worker.join();meter.save();meter.stream.close()

def load_model(meter,seed=42,freeze=False):
    import torch
    from transformer_lens import HookedTransformer
    random.seed(seed);torch.manual_seed(seed)
    with meter.stage('model_loading'):
        model=meter.attach(HookedTransformer.from_pretrained('gpt2-small'))
        assert model.cfg.dtype==torch.float32
        model.eval()
        for name in ('use_attn_result','use_split_qkv_input','use_hook_mlp_in','ungroup_grouped_query_attention'):
            setattr(model.cfg,name,True)
        if freeze:model.requires_grad_(False)
    return model

def runtime_guard(out,approved=False):
    """Checked before EVERY costly job, not only by the run-all wrapper."""
    p=protocol(out);pre=json.loads((out/'preflight_gpu.json').read_text())
    done=[];elapsed=json.loads((out/'preflight_cpu.json').read_text()).get('seconds',0.) if (out/'preflight_cpu.json').exists() else 0.
    training_rates=[];evaluation_rates=[];eap_rates=[];first_full=False
    for path in (out/'training').rglob('reuse.json') if (out/'training').exists() else []:
        elapsed+=json.loads(path.read_text()).get('new_CPU_export_seconds',0.)
    for path in (out/'attempts').rglob('attempt.json') if (out/'attempts').exists() else []:
        a=json.loads(path.read_text());elapsed+=a.get('seconds',0.)
    remaining_updates=0
    for j in p['training_jobs']:
        folder=out/'training'/j['key']
        if verified(folder):continue
        checkpoints=sorted(x for x in folder.glob('checkpoints/step_*.pt') if x.with_suffix('.sha256').exists())
        import torch
        state=torch.load(checkpoints[-1],map_location='cpu',weights_only=False) if checkpoints else None
        offset=state['completed'] if state else j['reuse_step']
        remaining_updates+=max(0,5000-offset)
    for path in (out/'attempts').rglob('rates.json') if (out/'attempts').exists() else []:
        r=json.loads(path.read_text())
        if r.get('updates',0)>=1000:training_rates.append(r['training_seconds']/r['updates'])
        if r.get('completed_target')==5000:first_full=True
        if r.get('evaluation_seconds'):evaluation_rates.append(r['evaluation_seconds'])
        if r.get('eap_examples'):eap_rates.append(r['eap_seconds']/r['eap_examples'])
    missing_eval=sum(not verified(out/'evaluations'/c/endpoint_key(e)) for c in ('primary','development') for e in p['endpoints'])
    missing_eap=[e for e in p['eap_jobs'] if not verified(out/'attribution'/e['key'])]
    missing_eap_eval=sum(not verified(out/'evaluations'/c/e['key']) for e in p['eap_jobs'] for c in ('primary','development'))
    update_seconds=max(training_rates or [pre['training_seconds_per_update']])
    eval_seconds=max(evaluation_rates or [pre['evaluation_seconds_per_ranking_projection']])
    eap_seconds=max(eap_rates or [pre['eap_seconds_per_example']])
    projection=elapsed+remaining_updates*update_seconds+(missing_eval+missing_eap_eval)*eval_seconds+sum(e['dataset_size'] for e in missing_eap)*eap_seconds
    data={'observed_new_job_wall_seconds':elapsed,'remaining_updates':remaining_updates,
          'seconds_per_update':update_seconds,'remaining_evaluations':missing_eval+missing_eap_eval,
          'projected_cumulative_seconds':projection,'limit_seconds':43200,'approval':approved,
          'uses_completed_5000_run_rate':first_full,'decision':'manual approval required' if projection>43200 and first_full and not approved else 'first new5000 run allowed for calibration, within budget, or explicit approval'}
    write_json(out/'runtime_projection.json',data)
    if projection>43200 and first_full and not approved:raise RuntimeError('Projected cumulative runtime exceeds 12 hours. Review runtime_projection.json; explicitly pass --approve-over-12h to continue. No settings reduced.')
    return data
