"""Execute diagnostics with existing kernel dependencies; default smoke only."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import io
import os
import sys
import traceback

import nbformat

root=Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME',str(root/'.cache/huggingface'))
os.environ.setdefault('MPLCONFIGDIR',str(root/'.cache/matplotlib'))
os.environ.setdefault('HF_DATASETS_OFFLINE','1')
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('MPLBACKEND','Agg')
book=nbformat.read(root/'notebooks/mib_sketch_diagnostics.ipynb',as_version=4)
scope={'__name__':'__main__'}
class Tee:
    def __init__(self,*streams): self.streams=streams
    def write(self,value):
        for stream in self.streams: stream.write(value);stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()
count=0
for index,cell in enumerate(book.cells):
    if cell.cell_type!='code':continue
    print(f'CELL {index}',flush=True)
    capture=io.StringIO()
    count+=1
    try:
        with redirect_stdout(Tee(capture,sys.stdout)),redirect_stderr(Tee(capture,sys.stderr)):
            exec(compile(cell.source,f'diagnostics_cell_{index}','exec'),scope)
    except Exception as error:
        cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=capture.getvalue()),
                      nbformat.v4.new_output('error',ename=type(error).__name__,evalue=str(error),traceback=traceback.format_exc().splitlines())]
        out=scope.get('OUT',root/'results/mib_sketch_diagnostics')
        out.mkdir(parents=True,exist_ok=True)
        nbformat.write(book,out/'mib_sketch_diagnostics.failed.ipynb')
        raise
    cell.execution_count=count
    cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=capture.getvalue())]
    if 'OUT' in scope:
        nbformat.write(book,scope['OUT']/'mib_sketch_diagnostics.executed.ipynb')
print('Saved executed diagnostics:',scope['OUT'],flush=True)
