"""Execute saved-results cells only; cannot start GPU training."""
import argparse
import asyncio
import os
from pathlib import Path
import shutil

import nbformat
from nbclient import NotebookClient
from .common import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for name, folder in [('MPLCONFIGDIR','matplotlib'), ('IPYTHONDIR','ipython'), ('JUPYTER_RUNTIME_DIR','jupyter')]:
        path = ROOT/'.cache'/folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    book = nbformat.read(ROOT/'notebooks/mattr_optimization.ipynb', as_version=4)
    cell = book.cells[1]
    assert 'RUN_GATE_0=False' in cell.source
    assert 'RUN_GATE_1=False' in cell.source
    cell.source = cell.source.replace('RESULTS=None  # Set a specific timestamped directory to analyze a particular run.',
                                      'RESULTS=Path('+repr(str(out))+')')
    NotebookClient(book, timeout=120, startup_timeout=60, kernel_name='mib-sketching',
        allow_errors=False, resources={'metadata': {'path': str(ROOT)}}).execute()
    assert not any(o.output_type == 'error' for c in book.cells if c.cell_type == 'code' for o in c.outputs)
    target = out/'mattr_optimization.executed.ipynb'
    nbformat.write(book, target)
    shutil.copyfile(target, ROOT/'notebooks/mattr_optimization.executed.ipynb')
    print('Executed saved-artifact analysis:', target)


if __name__ == '__main__':
    main()
