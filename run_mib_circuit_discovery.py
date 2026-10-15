"""Run the controlled circuit notebook; exhaustive oracle requires --oracle."""
import argparse
import base64
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
for variable, relative in {'HF_HOME': '.cache/huggingface',
                           'MPLCONFIGDIR': '.cache/matplotlib',
                           'IPYTHONDIR': '.cache/ipython',
                           'JUPYTER_RUNTIME_DIR': '.cache/jupyter'}.items():
    directory = ROOT / relative
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)

import nbformat
from nbclient import NotebookClient


def progress(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Cell {cell_index+1}: {cell.source.splitlines()[0][:100]}', flush=True)


def embed_saved_outputs(notebook, run_directory):
    """Make direct execution's saved figures/checkpoint visible in the notebook."""
    run_directory = Path(run_directory)
    plot_cell = next(c for c in notebook.cells
                     if c.cell_type == 'code' and "circuit_faithfulness.png" in c.source)
    for name in ('circuit_faithfulness.png', 'circuit_scores.png',
                 'compute_comparison.png', 'oracle_gap.png'):
        plot_cell.outputs.append(nbformat.v4.new_output(
            'display_data', data={'image/png': base64.b64encode(
                (run_directory / name).read_bytes()).decode('ascii'),
                'text/plain': name}, metadata={}))
    summary_cell = next(c for c in notebook.cells
                        if c.cell_type == 'code' and "research_checkpoint.md" in c.source)
    summary_cell.outputs.append(nbformat.v4.new_output(
        'display_data', data={'text/markdown':
            (run_directory / 'research_checkpoint.md').read_text(encoding='utf-8')},
        metadata={}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--oracle', action='store_true',
                        help='Explicitly run complete train-only enumeration for eligible sizes.')
    parser.add_argument('--max-oracle-candidates-per-size', type=int, default=1000)
    parser.add_argument('--direct', action='store_true',
                        help='Execute Python cells in one process when Jupyter kernel startup is unavailable.')
    args = parser.parse_args()
    source = ROOT / 'notebooks/mib_circuit_discovery.ipynb'
    notebook = nbformat.read(source, as_version=4)
    if args.oracle:
        cell = notebook.cells[1]
        old = 'RUN_ORACLE = False'
        if old not in cell.source:
            raise ValueError('Notebook oracle flag changed; inspect before overriding.')
        cell.source = cell.source.replace(old, 'RUN_ORACLE = True')
        old_cap = 'MAX_ORACLE_CANDIDATES_PER_SIZE = 1000'
        if old_cap not in cell.source or args.max_oracle_candidates_per_size < 1:
            raise ValueError('Invalid oracle candidate cap.')
        cell.source = cell.source.replace(
            old_cap, f'MAX_ORACLE_CANDIDATES_PER_SIZE = {args.max_oracle_candidates_per_size}')
    before = set((ROOT / 'results').glob('mib_circuit_discovery_*'))
    started = time.perf_counter()
    if args.direct:
        os.environ['MPLBACKEND'] = 'Agg'
        scope = {'__name__': '__main__'}
        execution_count = 0
        try:
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type != 'code':
                    continue
                progress(cell, index)
                execution_count += 1
                stream = io.StringIO()
                try:
                    with redirect_stdout(stream), redirect_stderr(stream):
                        exec(compile(cell.source, f'notebook_cell_{index}', 'exec'), scope)
                except Exception:
                    cell.execution_count = execution_count
                    cell.outputs = [nbformat.v4.new_output(
                        'stream', name='stdout', text=stream.getvalue()),
                        nbformat.v4.new_output('error', ename='CellError',
                                               evalue='Direct cell execution failed.',
                                               traceback=traceback.format_exc().splitlines())]
                    raise
                cell.execution_count = execution_count
                cell.outputs = [nbformat.v4.new_output(
                    'stream', name='stdout', text=stream.getvalue())]
                print(stream.getvalue()[-1500:], flush=True)
        except Exception:
            nbformat.write(notebook, ROOT / 'notebooks/mib_circuit_discovery.failed.ipynb')
            raise
    else:
        client = NotebookClient(notebook, timeout=7200, kernel_name='mib-sketching',
                                allow_errors=False, resources={'metadata': {'path': str(ROOT)}},
                                on_cell_execute=progress)
        try:
            client.execute()
        except Exception:
            nbformat.write(notebook, ROOT / 'notebooks/mib_circuit_discovery.failed.ipynb')
            raise
    nbformat.validate(notebook)
    code_cells = [c for c in notebook.cells if c.cell_type == 'code']
    if not all(c.execution_count is not None for c in code_cells):
        raise RuntimeError('Execution did not complete every code cell.')
    if any(o.output_type == 'error' for c in code_cells for o in c.outputs):
        raise RuntimeError('Execution included a cell error.')
    new_runs = set((ROOT / 'results').glob('mib_circuit_discovery_*')) - before
    if len(new_runs) != 1:
        raise RuntimeError(f'Expected exactly one new results directory, got {new_runs}.')
    run_directory = new_runs.pop()
    if args.direct:
        embed_saved_outputs(notebook, run_directory)
    output = ROOT / 'notebooks/mib_circuit_discovery.executed.ipynb'
    nbformat.write(notebook, output)
    report = {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'oracle_enabled': args.oracle, 'execution_backend': 'direct_python' if args.direct else 'jupyter_kernel',
              'executed_code_cells': len(code_cells),
              'wall_seconds': time.perf_counter()-started,
              'notebook': str(output), 'results_directory': str(run_directory)}
    (Path(report['results_directory']) / 'notebook_execution.json').write_text(
        json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
