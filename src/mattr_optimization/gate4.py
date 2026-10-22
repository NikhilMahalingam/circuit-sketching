"""Frozen Gate4 pilot, paired screens and at most one confirmation. No Gate5."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from .common import ROOT,sha,write_json,git
from .gate3 import baseline,verify_criteria,CRITERIA_SHA

SEEDS=(42,43,44)
SCALES=(.1,.5,1.,2.)


def freeze(out):
    verify_criteria();out.mkdir(parents=True,exist_ok=True)
    from .gate4_attribution import attribution_sources
    from .gate3 import CRITERIA
    names=('gate4.py','gate4_init.py','gate4_attribution.py','gate4_train.py','gate4_evaluate.py','gate4_report.py',
           'common.py','checkpoint.py','gate2.py','gate1.py','gate0.py','gate3.py','gate3_report.py')
    proto={'gate':4,'parent_commit':git(ROOT,'rev-parse','HEAD').decode().strip(),
      'historical_gate3_criteria_sha256':CRITERIA_SHA,'seeds':list(SEEDS),'pool':128,'scales':list(SCALES),
      'model':'gpt2-small','task':'official MIB IOI edge','edges':32491,
      'attribution':{'method':'EAP-IG-inputs','intervention':'patching','ig_steps':5,'alpha_grid':[0,.2,.4,.6,.8],
        'batch_size':4,'aggregation':'sum','metric':'official logit_diff mean=True loss=True','orientation':'signed, no abs/sign reversal',
        'source_hashes':attribution_sources(),'data':'exact paired Gate2 N128 train subsets; no validation attribution'},
      'initialization':'scale * (2*(ascending average rank-1)/(32491-1)-1); ties equal; centered [-1,1]; float32; no RNG draw',
      'training':{'optimizer':'official Adam','lr':.05,'T':.5,'n_iters':30,'k_schedule':'uniform','mode':'iso','steps_max':5000,
        'batch_size':20,'cf_cache_gb':4,'all_other_settings':'unchanged pinned CLI; init tensor only'},
      'development_cohort':{'source':'exact Gate2 fixed seed42 validation128','manifest_sha256':sha(baseline(42)/'validation_manifest.csv')},
      'confirmation_cohort':{'source':'official filtered validation permutation seed42 positions128:256; disjoint rows/prompts from development and train',
                             'usage':'only after a single final development winner is fixed; no selection feedback'},
      'pilot':{'seed':42,'steps':1000,'selection':'exploratory top two by CPR gain vs paired1000; ties lower scale. No pilot success claims.'},
      'paired_screen':{'seeds':list(SEEDS),'steps':1000,'selection':'require same primary numerical quality gates provisionally vs1000 OR efficiency vs5000; seed42 pilot reused and labeled exploratory; select at most one eligible scale by mean paired relative CPR gain; tie lower scale; none means stop'},
      'evaluation_steps':[500,1000,2500,5000],
      'success':{'primary':{'mean_paired_relative_CPR_gain_min':.02,'positive_seeds_min':2,
        'mean_compact_accuracy_drop_max':.02,'mean_accuracy_log_auc_drop_max':.02,'compact_fractions':[.005,.01,.02,.05],
        'mean_CMD_increase_max':'max(0, mean paired CPR increase) + 0.05','total_adaptation_time_ratio_max':1.10,'peak_allocated_ratio_max':1.10},
        'efficiency':{'CPR_relative_loss_max':.01,'CMD_increase_max':.10,'accuracy_drop_max':.02,'accuracy_auc_drop_max':.02,
                      'total_adaptation_compute_reduction_min':.20},
        'confirmation':'primary uses paired5000; efficiency uses candidate1000 vs baseline5000. Disjoint confirmation repeats the applicable QUALITY guards before any generalization claim.'},
      'compute':'EAPIG + all required model/data/setup/training and restart cost; evaluation separate. EAP attribution charged once per deployable seed, all pilot/failed trials charged to research. CUDA synchronized; root forward/example/backward/updates, CPU RSS and allocation peaks measured.',
      'compute_matched':'Use existing ordinary MAttr100-step checkpoints nearest initialized1000 total adaptation cost by Gate2 checkpoint timer, if <=5000; evaluate frozen ranking only. Report actual time mismatch. No unregistered optimizer budgets beyond5000.',
      'resume':'original score/Adam/Python/NumPy/CPU/CUDA RNG checkpoint restoration; exact CPU equivalence test; pair actual row and uniform-k streams with Gate2',
      'stability':'official signed apply_topn(floor(fraction*32491),prune=True), paired edge-set Jaccard at .005/.01/.02/.05; diagnostic only',
      'no_gate5':True,'no_full_data_followup':True,'no_leaderboard':True,
      'adapter_sha256':{n:sha(ROOT/'src/mattr_optimization'/n) for n in names}}
    path=out/'gate4_protocol.json'
    if path.exists():
        existing=json.loads(path.read_text())
        # Commit hash records implementation parent, not subsequent archival commits.
        proto['parent_commit']=existing['parent_commit']
        if existing!=proto:raise ValueError('Frozen Gate4 protocol/source changed; refusing to rewrite it.')
    else:
        protected={}
        for folder in (ROOT/'results/mattr_optimization').iterdir():
            if folder.is_dir() and folder.resolve()!=out.resolve():
                for p in folder.rglob('*'):
                    if p.is_file():protected[str(p.relative_to(ROOT))]=sha(p)
        for n in ('GATE2_FINAL_ANALYSIS.md','GATE3_ADAPTIVE_SAMPLING_REPORT.md'):protected[n]=sha(ROOT/n)
        write_json(out/'protected_previous_hashes.json',protected)
        shutil.copyfile(CRITERIA,out/'historical_gate3_success_criteria.json')
        write_json(path,proto);(out/'gate4_protocol.sha256').write_text(sha(path)+'\n')
        write_json(out/'protocol_frozen_at.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':sha(path),'modified_training_started':False})
        snap=out/'source_at_execution';snap.mkdir(exist_ok=True)
        for n in names:shutil.copyfile(ROOT/'src/mattr_optimization'/n,snap/n)
    return path


def launch(out,label,module,args,marker):
    folder=out/label;folder.mkdir(parents=True,exist_ok=True)
    if (folder/marker).exists():
        saved=json.loads((folder/marker).read_text())
        if not saved['passed']:raise ValueError('Failed cached job.')
        for n,h in saved.get('sha256',{}).items():
            if sha(folder/n)!=h:raise ValueError('Cached job artifact changed: '+n)
        print('Cached:',label,marker,'; zero forwards.',flush=True);return
    command=[sys.executable,'-u','-m','src.mattr_optimization.'+module,'--out',str(folder),*map(str,args)]
    start=time.perf_counter()
    with (folder/'console.log').open('a',buffering=1) as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in child.stdout:stream.write(line);print(line,end='',flush=True)
        if child.wait()!=0:raise RuntimeError(f'Gate4 job failed: {folder}; no settings silently changed.')
    write_json(folder/f'launch_elapsed_{marker}.json',{'seconds':time.perf_counter()-start,'argv':command})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--freeze-only',action='store_true');p.add_argument('--report-only',action='store_true')
    a=p.parse_args();out=a.out.resolve()
    from .gate4_report import report,select_pilot,select_screen,confirm
    if a.report_only:report(out);return
    import fcntl
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=freeze(out)
        if a.freeze_only:print('Gate4 protocol frozen; zero modified training forwards.');return
        smoke=out/'smoke_seed42/completed.json'
        if not smoke.exists() or not json.loads(smoke.read_text())['passed']:raise ValueError('Passing measured EAPIG smoke required.')
        # Every paired training subset is attributed exactly once; zero-step controls
        # share those vectors. These are not additional scale variants.
        for seed in SEEDS:
            launch(out,f'attribution_seed{seed}','gate4_attribution',['--seed',seed],'completed.json')
            launch(out,f'eap_only_seed{seed}','gate4_evaluate',['--seed',seed,'--step',0,'--scores',out/f'attribution_seed{seed}/eap_ig_scores.pt'],'completed.json')
        def train(scale,seed,target):
            launch(out,f'scale{scale:g}_seed{seed}','gate4_train',
                   ['--protocol',protocol,'--scale',scale,'--seed',seed,'--target',target],f'completed_{target}.json')
        for scale in SCALES:train(scale,42,1000)
        scales=select_pilot(out)
        for scale in scales:
            for seed in (43,44):train(scale,seed,1000)
        winner=select_screen(out);report(out)
        if winner is not None:
            for seed in SEEDS:
                train(winner,seed,5000)
                launch(out,f'baseline2500_seed{seed}','gate4_evaluate',['--seed',seed,'--step',2500,
                       '--scores',baseline(seed)/'checkpoints/step_002500.pt'],'completed.json')
            result=confirm(out,winner)
            if result['development_passed']:
                budget=result['selected_budget']
                for seed in SEEDS:
                    for kind,path in [('initialized',out/f'scale{winner:g}_seed{seed}/checkpoints/step_{budget:06d}.pt'),
                                      ('baseline',baseline(seed)/'checkpoints/step_005000.pt')]:
                        launch(out,f'confirmation_{kind}_seed{seed}','gate4_evaluate',
                               ['--seed',seed,'--step',budget if kind=='initialized' else 5000,'--scores',path,'--cohort','confirmation'],'completed.json')
            report(out)
        # Exact no-training compute-matched ordinary ranking reference, nearest100
        # checkpoint by its actual recorded training timer. Never extend training.
        from .gate4_report import compute_matches
        for seed,step in compute_matches(out):
            launch(out,f'compute_matched_seed{seed}','gate4_evaluate',['--seed',seed,'--step',step,
                   '--scores',baseline(seed)/f'checkpoints/step_{step:06d}.pt'],'completed.json')
        report(out)
        protected=json.loads((out/'protected_previous_hashes.json').read_text())
        if any(sha(ROOT/n)!=h for n,h in protected.items()):raise ValueError('Historical Gate1–3 artifacts changed.')
        verify_criteria()
        write_json(out/'gate4_completed.json',{'passed_execution':True,'screen_winner':winner,'historical_files_unchanged':len(protected),
                                              'protocol_sha256':sha(protocol),'gate5':False,'full_data_followup':False,'leaderboard':False})


if __name__=='__main__':main()
