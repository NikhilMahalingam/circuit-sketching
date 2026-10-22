"""Source pins and utilities, without importing older research modules."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / 'external/matryoshka-attribution'
MIB = UPSTREAM / 'deps/MIB-circuit-track'
MATTR_SHA = 'af82b2eef145dd3e66c793a413352d6931a152d7'
MIB_SHA = '582fdbc8a2b0f7063a265c8f94851fd2a3aa5480'
GRID = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, default=str) + '\n', encoding='utf-8')
    temp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def configure():
    sys.path.insert(0, str(UPSTREAM / 'src'))
    sys.path.insert(0, str(MIB))
    sys.path.insert(0, str(MIB / 'EAP-IG/src'))


def provenance():
    sources = {}
    checks = [(UPSTREAM, MATTR_SHA, ['README.md', 'pyproject.toml', 'uv.lock',
        'scripts/mib/eval_mib.py', 'scripts/mib/eval_mib_edge.py',
        *[str(p.relative_to(UPSTREAM)).replace('\\', '/') for p in (UPSTREAM/'src').rglob('*.py')]]),
        (MIB, MIB_SHA, ['MIB_circuit_track/evaluation.py', 'MIB_circuit_track/dataset.py',
                       'MIB_circuit_track/metrics.py']),
        (MIB/'EAP-IG', None, ['src/eap/graph.py', 'src/eap/evaluate.py'])]
    commits = {}
    for repo, expected, names in checks:
        actual = git(repo, 'rev-parse', 'HEAD').decode().strip()
        if expected and expected != actual:
            raise RuntimeError(f'Unexpected pinned source: {repo}: {actual}')
        commits[str(repo.relative_to(ROOT))] = actual
        for name in names:
            path = repo/name
            # The Windows checkout may use CRLF. Permit line endings only; never rewrite it.
            raw = path.read_bytes()
            original = git(repo, 'show', f'HEAD:{name}')
            if raw.replace(b'\r\n', b'\n') != original.replace(b'\r\n', b'\n'):
                raise RuntimeError(f'Upstream source differs from pin: {path}')
            sources[str(path.relative_to(ROOT))] = sha(path)
    import torch
    return {'commits': commits, 'source_sha256': sources,
        'python': sys.version, 'executable': sys.executable,
        'versions': {p: importlib.metadata.version(p) for p in
            ['torch', 'transformers', 'transformer-lens', 'datasets', 'numpy', 'pandas',
             'scipy', 'matplotlib', 'psutil', 'pytest']},
        'torch_cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else None}


def protected_snapshot():
    names = git(ROOT, 'ls-files', '-z').decode().split('\0')
    return {name: sha(ROOT/name) for name in names if name and (ROOT/name).is_file()}


def integration_weights(grid=GRID):
    """Analysis of official linear trapezoids; benchmark metrics still call MIB."""
    return [(grid[1]-grid[0])/2 if i == 0 else
            (grid[-1]-grid[-2])/2 if i == len(grid)-1 else
            (grid[i+1]-grid[i-1])/2 for i in range(len(grid))]
