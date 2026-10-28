"""Standalone foreground CLI; Stage2 evaluation is always explicit/manual."""
import argparse
import subprocess
import sys
from .common import *

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['stage1','search','freeze','displacement','confirm','analyze','audit','package','smoke'])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--seed',type=int,choices=range(42,47))
    parser.add_argument('--fraction',type=float,choices=FRACTIONS)
    parser.add_argument('--approve-over-2h',action='store_true')
    parser.add_argument('--commit',action='store_true')
    parser.add_argument('--stage1-only',action='store_true')
    args=parser.parse_args();configure();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    write_json(out/'status.json',dict(stage=args.stage,status='running',HEAD=git(ROOT,'rev-parse','HEAD').decode().strip(),source_config_sha256=digest(sources())))
    try:
        if args.stage=='stage1':
            if args.dry_run:print('CPU-only: verify all saved Gate5G inputs, execute committed Stage1 plan, prepare confirmation cohort. Zero forwards.');return
            from .stage1 import run
            from .stage2 import prepare
            run(out);prepare(out)
        elif args.stage=='search':
            from .stage2 import search
            search(out,args.dry_run)
        elif args.stage=='freeze':
            if args.dry_run:print('Freeze after candidate search. No confirmation forwards.');return
            from .stage2 import freeze
            freeze(out)
        elif args.stage=='displacement':
            if args.seed is None:
                for seed in range(42,47):
                    cmd=[sys.executable,'-u','-m','src.mattr_gate5g_addendum.cli','displacement','--out',str(out),'--seed',str(seed)]
                    if args.fraction is not None:cmd+=['--fraction',str(args.fraction)]
                    if args.approve_over_2h:cmd+=['--approve-over-2h']
                    if args.dry_run:cmd+=['--dry-run']
                    subprocess.run(cmd,cwd=ROOT,check=True)
            else:
                from .stage2 import evaluate
                evaluate(out,'displacement',args.seed,args.fraction,args.approve_over_2h,args.dry_run)
        elif args.stage=='confirm':
            from .stage2 import evaluate
            evaluate(out,'confirmation',42,None,args.approve_over_2h,args.dry_run)
        elif args.stage in ['analyze','audit','package']:
            from . import analysis
            if args.dry_run and args.stage!='package':
                stage2_protocol(out);print('Offline '+args.stage+'; zero model forwards.');return
            if args.stage=='analyze':analysis.run(out)
            elif args.stage=='audit':analysis.audit(out)
            else:analysis.package(out,args.commit,args.dry_run,args.stage1_only)
        elif args.stage=='smoke':
            if args.dry_run:print('Two historical Gate5F prompts, never fresh confirmation. One custom circuit, official one-point evaluator check.');return
            from .smoke import run
            run(out)
        write_json(out/'status.json',dict(stage=args.stage,status='complete',HEAD=git(ROOT,'rev-parse','HEAD').decode().strip()))
    except BaseException as exc:
        import traceback
        logs=out/'logs';logs.mkdir(exist_ok=True)
        log=logs/(args.stage+'_'+str(time.time_ns())+'.log');log.write_text(traceback.format_exc())
        write_json(out/'status.json',dict(stage=args.stage,status='failed',error=repr(exc),log=str(log)))
        raise

if __name__=='__main__':main()
