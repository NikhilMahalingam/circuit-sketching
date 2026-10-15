"""Execute the IOI notebook, preserving the clean source and all real outputs.

--smoke reduces prompt counts only, labels outputs explicitly, and still executes
every experimental section. Default runs the unmodified notebook configuration.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
for variable, relative in {'HF_HOME': '.cache/huggingface', 'MPLCONFIGDIR': '.cache/matplotlib',
                           'IPYTHONDIR': '.cache/ipython', 'JUPYTER_RUNTIME_DIR': '.cache/jupyter'}.items():
    directory = ROOT / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)

import nbformat
from nbclient import NotebookClient


def progress(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Cell {cell_index+1}: {cell.source.splitlines()[0][:100]}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume-run', type=Path)
    args = parser.parse_args()
    notebook = nbformat.read(ROOT / 'mib_ioi_experiments.ipynb', as_version=4)
    if args.resume_run:
        notebook.cells[1].source = notebook.cells[1].source.replace(
            'RESUME_DIR = None', f'RESUME_DIR = {str(args.resume_run.resolve())!r}')
    if args.smoke:
        cell = notebook.cells[1]
        substitutions = {
            'N_TRAIN, N_VALIDATION, N_PUBLIC_TEST = 128, 128, 128':
                'N_TRAIN, N_VALIDATION, N_PUBLIC_TEST = 16, 16, 16',
            'EAP_BATCH_SIZE, N_EAP_EXAMPLES = 4, 32': 'EAP_BATCH_SIZE, N_EAP_EXAMPLES = 4, 16',
            'MIB_EVAL_BATCH_SIZE, N_MIB_EVAL_EXAMPLES = 4, 32':
                'MIB_EVAL_BATCH_SIZE, N_MIB_EVAL_EXAMPLES = 4, 8',
            'BOOTSTRAP_REPEATS = 500': 'BOOTSTRAP_REPEATS = 100',
            'SAMPLE_SIZES, SAMPLE_REPEATS = [16, 32, 64, 128], 5':
                'SAMPLE_SIZES, SAMPLE_REPEATS = [4, 8, 16], 2',
        }
        for old, new in substitutions.items():
            if old not in cell.source:
                raise ValueError(f'Smoke override missing: {old}')
            cell.source = cell.source.replace(old, new)
        notebook.cells[0].source += '\n\n**Execution profile: reduced 16-example integration smoke test.**'
    name = 'mib_ioi_experiments.smoke.executed.ipynb' if args.smoke else 'mib_ioi_experiments.executed.ipynb'
    before = set((ROOT/'results').glob('mib_ioi_*'))
    started = time.perf_counter()
    client = NotebookClient(notebook, timeout=7200, kernel_name='mib-sketching', allow_errors=False,
                            resources={'metadata': {'path': str(ROOT)}}, on_cell_execute=progress)
    try:
        client.execute()
    except Exception:
        nbformat.write(notebook, ROOT / name.replace('.executed.', '.failed.'))
        raise
    nbformat.validate(notebook)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type == 'error' for c in code for o in c.outputs)
    nbformat.write(notebook, ROOT / name)
    new_runs = set((ROOT/'results').glob('mib_ioi_*')) - before
    if args.resume_run:
        new_runs = {args.resume_run.resolve()}
    report = {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'smoke': args.smoke, 'resumed': bool(args.resume_run), 'executed_code_cells': len(code),
              'wall_seconds': time.perf_counter()-started, 'notebook': name,
              'results_directories': [str(p) for p in new_runs]}
    for path in new_runs:
        (path/'notebook_execution.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
