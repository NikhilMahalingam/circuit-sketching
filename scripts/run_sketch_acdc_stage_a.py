"""Execute only Stage A cells, retaining outputs in the timestamped results folder."""
from contextlib import redirect_stdout, redirect_stderr
import io
from pathlib import Path
import traceback
import nbformat

root = Path(__file__).resolve().parents[1]
book = nbformat.read(root/'notebooks/mib_sketch_acdc.ipynb', as_version=4)
namespace = {'__name__':'__main__'}
for index, cell in enumerate(book.cells):
    if cell.cell_type != 'code':
        continue
    if 'if RUN_STAGE_B:' in cell.source:
        break
    capture = io.StringIO()
    print(f'CELL {index}', flush=True)
    try:
        with redirect_stdout(capture), redirect_stderr(capture):
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
    print(capture.getvalue()[-3000:], flush=True)
out = namespace['OUT']
nbformat.write(book,out/'mib_sketch_acdc.executed.ipynb')
print('Saved',out/'mib_sketch_acdc.executed.ipynb',flush=True)
