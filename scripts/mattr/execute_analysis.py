"""Execute artifact-reading cells only; GPU experiments always run through WSL."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--kernel',default='mib-sketching');args=parser.parse_args()
    out=args.out.resolve()
    if os.name=='nt':asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    for name,folder in [('MPLCONFIGDIR','matplotlib'),('IPYTHONDIR','ipython'),('JUPYTER_RUNTIME_DIR','jupyter')]:
        path=ROOT/'.cache'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    book=nbformat.read(ROOT/'notebooks/mib_mattr_reproduction.ipynb',as_version=4)
    # Pin the selected campaign rather than silently reading whichever folder is newest.
    book.cells[1].source='\n'.join('RESULTS=Path('+repr(str(out))+')' if line.startswith('RESULTS=') else line
                                    for line in book.cells[1].source.splitlines())
    nbformat.validate(book)
    assert all('=True' not in line for line in book.cells[1].source.splitlines() if line.startswith('RUN_'))
    def progress(cell,cell_index,**kwargs):
        if cell.cell_type=='code':print('Executing analysis cell',cell_index+1,flush=True)
    client=NotebookClient(book,timeout=300,startup_timeout=60,kernel_name=args.kernel,allow_errors=False,
                          on_cell_execute=progress,
                          resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    assert all(c.execution_count is not None for c in book.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in book.cells if c.cell_type=='code' for o in c.outputs)
    target=out/'mib_mattr_reproduction.executed.ipynb';nbformat.write(book,target)
    shutil.copyfile(target,ROOT/'notebooks/mib_mattr_reproduction.executed.ipynb')
    (out/'notebook_execution.json').write_text(json.dumps({'success':True,
        'analysis_kernel':args.kernel,'model_execution_environment':'isolated WSL Ubuntu CUDA',
        'campaign':str(out),'executed_code_cells':sum(c.cell_type=='code' for c in book.cells)},indent=2)+'\n')
    print(target)


if __name__=='__main__':main()
