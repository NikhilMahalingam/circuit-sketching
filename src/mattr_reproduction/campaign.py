"""Independently resumable stages; one CUDA worker, official algorithms only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from .runtime import ROOT, EXPECTED_COMMIT, digest, write_json

SIZES = (16,32,64,128)
SEEDS = (42,43,44)


def launch(out, script, args):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        print('Complete, reused:',out,flush=True); return
    command=['bash',str(ROOT/'scripts/mattr'/script),*map(str,args),'--out',str(out)]
    print('Launching',command,flush=True)
    started=time.perf_counter();success=False
    with (out/'console.log').open('a',buffering=1) as log:
        log.write('\nATTEMPT '+datetime.now(timezone.utc).isoformat()+'\n'+json.dumps(command)+'\n')
        try:
            subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            success=True
        finally:
            path=out/'process_accounting.json'
            attempts=json.loads(path.read_text()) if path.exists() else []
            attempts.append({'wall_seconds':time.perf_counter()-started,'success':success,
                             'includes_interpreter_and_imports':True})
            write_json(path,attempts)
    print('Completed:',out,flush=True)


def require(out, stage):
    if not (out/stage/'complete.json').exists():
        raise RuntimeError(f'First complete and verify {stage}: {out/stage}')


def initialize(out):
    out.mkdir(parents=True,exist_ok=True)
    target=out/'experiment_config.json'
    config={'mattr_commit':EXPECTED_COMMIT,'model':'gpt2','task':'ioi','level':'node',
        'seeds':SEEDS,'discovery_sizes':SIZES,'steps':500,'lr':.05,
        'masking':'topk','mode':'iso','k_schedule':'uniform','include_input':True,
        'validation_comparison_size':128,'validation_seed':42,
        'reference_train_size':'all','reference_validation_size':'all',
        'reference_and_comparisons_have_distinct_cohorts':True,
        'evaluation_batch_size':20,'train_batch_size':1,'cf_cache_gb':4.,
        'intervention':'patching','public_test_accessed':False,
        'intermediate_evaluation_steps':[100,250,500],
        'baseline_ig_steps':5,'baseline_attribution_batch_size':1,
        'equal_data_is_not_equal_compute':True}
    if target.exists():
        previous=json.loads(target.read_text())
        # Access status is derived runtime metadata; every experimental setting stays fixed.
        config['public_test_accessed']=previous['public_test_accessed']
        if previous != json.loads(json.dumps(config)):
            raise ValueError('Campaign configuration changed; use a new timestamp.')
    else:
        write_json(target,config)
        protected={str(p.relative_to(ROOT)):digest(p) for p in ROOT.glob('notebooks/*sketch*') if p.is_file()}
        protected.update({str(p.relative_to(ROOT)):digest(p) for p in ROOT.glob('*SKETCH*REPORT*') if p.is_file()})
        write_json(out/'protected_previous_artifacts.json',protected)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['tests','smoke','reference','scaling','baselines','checkpoints','confirmation','report','all'],required=True)
    parser.add_argument('--require-complete',action='store_true',help='Fail report generation unless every campaign job is verified.')
    parser.add_argument('--wait-reference-pid',type=int,help='Wait for an existing reference process to exit; require its completion marker before continuing.')
    args=parser.parse_args();out=args.out.resolve();initialize(out)
    if args.wait_reference_pid:
        print(f'Waiting for existing reference process {args.wait_reference_pid}; no concurrent GPU job will start.',flush=True)
        process=Path(f'/proc/{args.wait_reference_pid}')
        while process.exists():time.sleep(1)
        require(out,'reference')
    stages=['tests','smoke','reference','scaling','baselines','checkpoints','confirmation'] if args.stage=='all' else [args.stage]
    for stage in stages:
        if stage=='tests':
            with (out/'tests.log').open('w') as log:
                subprocess.run(['bash',str(ROOT/'scripts/mattr/test_wsl.sh')],check=True,stdout=log,stderr=subprocess.STDOUT)
            write_json(out/'tests_passed.json',{'official_tests':6,'adapter_tests':5,'passed':True})
        elif stage=='smoke':
            if not (out/'tests_passed.json').exists():raise RuntimeError('Run tests stage first.')
            launch(out/'smoke','run_wsl.sh',['--kind','smoke','--train-size',16,'--validation-size',8,
                '--steps',5,'--eval-batch-size',2,'--cf-cache-gb',.25,'--checkpoint-every',1])
        elif stage=='reference':
            require(out,'smoke')
            launch(out/'reference','run_wsl.sh',['--kind','reference','--train-size','all',
                '--validation-size','all','--steps',500,'--seed',42])
        elif stage=='scaling':
            require(out,'reference')
            from .verify import verify_job
            verify_job(out/'reference')
            # Increasing pool sizes, with nested subsets independently within each seed.
            for n in SIZES:
                for seed in SEEDS:
                    launch(out/f'scaling/n{n}_seed{seed}','run_wsl.sh',
                        ['--kind','scaling','--train-size',n,'--validation-size',128,'--seed',seed])
        elif stage=='baselines':
            require(out,'reference')
            for n in SIZES:
                for seed in SEEDS:
                    require(out,f'scaling/n{n}_seed{seed}')
                    for method in ('EAP','EAP-IG'):
                        launch(out/f'baselines/{method}/n{n}_seed{seed}','evaluate_wsl.sh',
                            ['baseline','--method',method,'--train-size',n,'--validation-size',128,'--seed',seed])
        elif stage=='checkpoints':
            require(out,'reference')
            source=out/'reference'
            launch(out/'reference_common_validation','evaluate_wsl.sh',['reevaluate',
                '--source',source,'--checkpoint',source/'checkpoints/step_000500.pt','--validation-size',128])
            for n in SIZES:
                for seed in SEEDS:
                    source=out/f'scaling/n{n}_seed{seed}';require(out,f'scaling/n{n}_seed{seed}')
                    for step in (100,250,500):
                        reuse=['--reuse-final-validation'] if step==500 else []
                        launch(out/f'checkpoint_evaluations/n{n}_seed{seed}/step{step}','evaluate_wsl.sh',
                            ['reevaluate','--source',source,'--checkpoint',source/f'checkpoints/step_{step:06d}.pt',
                             '--validation-size',128,'--include-train',*reuse])
        elif stage=='report':
            from .verify import verify
            verify(out,require_complete=args.require_complete or args.stage=='all')
            from .report import generate
            generate(out)
        elif stage=='confirmation':
            from .confirm import run_campaign
            run_campaign(out)


if __name__=='__main__':main()
