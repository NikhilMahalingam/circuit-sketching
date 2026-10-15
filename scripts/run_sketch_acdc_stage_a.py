"""Execute only Stage A cells, retaining outputs in the timestamped results folder."""
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import os
import sys
import traceback
import nbformat

root = Path(__file__).resolve().parents[1]
book = nbformat.read(root/'notebooks/mib_sketch_acdc.ipynb', as_version=4)
namespace = {'__name__':'__main__'}
run_all = os.environ.get('SKETCH_ACDC_RUN_B') == '1'
if os.environ.get('SKETCH_ACDC_RUN_C') == '1' and not run_all:
    raise ValueError('Stage C requires SKETCH_ACDC_RUN_B=1 for training-only calibration.')
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, value):
        for stream in self.streams: stream.write(value); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()
for index, cell in enumerate(book.cells):
    if cell.cell_type != 'code':
        continue
    if not run_all and 'if RUN_STAGE_B:' in cell.source:
        break
    capture = io.StringIO()
    print(f'CELL {index}', flush=True)
    try:
        with redirect_stdout(Tee(capture,sys.stdout)), redirect_stderr(Tee(capture,sys.stderr)):
            exec(compile(cell.source, f'cell_{index}', 'exec'), namespace)
    except Exception:
        capture.write(traceback.format_exc())
        cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=capture.getvalue())]
        out = namespace.get('OUT',root/'results/mib_sketch_acdc')
        out.mkdir(parents=True,exist_ok=True)
        nbformat.write(book,out/'mib_sketch_acdc.executed.ipynb')
        print(capture.getvalue(), flush=True)
        raise
    cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=capture.getvalue())]
    # The Tee already streamed output; the saved notebook retains all text.
out = namespace['OUT']
nbformat.write(book,out/'mib_sketch_acdc.executed.ipynb')
print('Saved',out/'mib_sketch_acdc.executed.ipynb',flush=True)
