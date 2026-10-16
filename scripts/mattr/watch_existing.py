"""Recover end-to-end monotonic timing for an already-started Linux process."""
import argparse
import json
import os
from pathlib import Path
import time

parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True)
parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
stat=Path(f'/proc/{args.pid}/stat')
start=float(stat.read_text().split(') ',1)[1].split()[19])/os.sysconf('SC_CLK_TCK')
while stat.exists():time.sleep(.5)
elapsed=time.monotonic()-start
(args.out/'process_accounting.json').write_text(json.dumps([{'wall_seconds':elapsed,
    'success':(args.out/'complete.json').exists(),'includes_interpreter_and_imports':True,
    'measurement':'Linux process start tick to exit observation; <=0.5s observation overhead'}],indent=2)+'\n')
print('Recorded process wall seconds:',elapsed)
