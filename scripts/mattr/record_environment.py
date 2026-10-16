"""Record the isolated interpreter, lock, GPU driver and official dependencies."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

from src.mattr_reproduction.runtime import ROOT, OFFICIAL, provenance, write_json

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
data={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
      'uv':subprocess.check_output([str(ROOT/'.cache/mattr-bootstrap/bin/uv'),'--version'],text=True).strip(),
      'gpu_driver':subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True).strip(),
      'official':provenance()}
hub=ROOT/'.cache/huggingface/hub'
data['cached_hub_revisions']={str(p.relative_to(hub)):p.read_text().strip()
    for pattern in ('models--gpt2/refs/main','models--openai-community--gpt2/refs/main','datasets--mib-bench--ioi/refs/main')
    for p in hub.glob(pattern)}
write_json(args.out/'environment.json',data)
freeze=subprocess.check_output([str(ROOT/'.cache/mattr-bootstrap/bin/uv'),'pip','freeze','--python',sys.executable],text=True)
(args.out/'environment_freeze.txt').write_text(freeze)
(args.out/'official_uv.lock').write_bytes((OFFICIAL/'uv.lock').read_bytes())
print(json.dumps(data,indent=2))
