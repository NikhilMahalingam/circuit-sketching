"""Execute only the saved-artifact analysis interface; all RUN flags remain false."""
import argparse
import asyncio
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
        p=ROOT/'.cache'/folder;p.mkdir(exist_ok=True,parents=True);os.environ[name]=str(p)
    book=nbformat.read(ROOT/'notebooks/acdc_prefix_caching.ipynb',as_version=4)
    book.cells[1].source='\n'.join('RESULTS=Path('+repr(str(out))+')' if x.startswith('RESULTS=') else x
                                    for x in book.cells[1].source.splitlines())
    assert all('=True' not in x for x in book.cells[1].source.splitlines() if x.startswith(('RUN_','ALLOW_')))
    client=NotebookClient(book,timeout=300,startup_timeout=60,kernel_name=args.kernel,allow_errors=False,
        resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    assert all(c.execution_count is not None for c in book.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for c in book.cells if c.cell_type=='code' for o in c.outputs)
    target=out/'acdc_prefix_caching.executed.ipynb';nbformat.write(book,target)
    shutil.copyfile(target,ROOT/'notebooks/acdc_prefix_caching.executed.ipynb')
    print('Executed analysis notebook:',target)


if __name__=='__main__':main()
