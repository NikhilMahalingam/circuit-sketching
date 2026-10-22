"""Build a lightweight notebook that dispatches Gate 0 and reads actual artifacts."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]


def build():
    book = nbf.v4.new_notebook()
    book.metadata.kernelspec = {'display_name': 'MIB Sketching', 'language': 'python', 'name': 'mib-sketching'}
    book.cells = [nbf.v4.new_markdown_cell('''# MAttr optimization — official edge baseline

Separate **edge-level** research, using pinned upstream MAttr and official MIB IOI.
The prior node reproduction is not the edge baseline. Read `MATTR_FINDINGS.md` for
source inspection and metric definitions. All training executes in a dedicated WSL
environment. This notebook defaults to reading saved results. Gate 0 smoke and
the 5,000-step Gate 1 baseline have separate switches, both off by default.
It never automatically starts scaling sweeps, variants or submissions.
'''), nbf.v4.new_code_cell('''from pathlib import Path
import json
import os
import subprocess
import pandas as pd
from IPython.display import display, Markdown, Image

ROOT=Path.cwd()
if not (ROOT/'src/mattr_optimization').exists():
    ROOT=ROOT.parent
assert (ROOT/'src/mattr_optimization').exists()
RUN_GATE_0=False
RUN_GATE_1=False
RESULTS=None  # Set a specific timestamped directory to analyze a particular run.
if RESULTS is None:
    completed=sorted(p for p in (ROOT/'results/mattr_optimization').glob('*')
                     if (p/'gate0_passed.json').exists() or (p/'gate1_passed.json').exists())
    RESULTS=completed[-1] if completed else None

def dispatch(out, gate):
    # No automatic next gate. Gate 1 may run for tens of minutes or longer.
    relative=Path(out).resolve().relative_to(ROOT.resolve()).as_posix()
    args=['bash','src/mattr_optimization/run_wsl.sh']
    if gate==1:
        args+=['gate1']
    args+=['--out',relative]
    if os.name=='nt':
        args=['wsl','-d','Ubuntu','--']+args
    subprocess.run(args,cwd=ROOT,check=True)

if RUN_GATE_0 and RUN_GATE_1:
    raise ValueError('Run only one gate at a time.')
if RUN_GATE_0 or RUN_GATE_1:
    from datetime import datetime, timezone
    gate=1 if RUN_GATE_1 else 0
    RESULTS=ROOT/'results/mattr_optimization'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'_gate{gate}')
    dispatch(RESULTS,gate)
print('Campaign:',RESULTS)
'''), nbf.v4.new_markdown_cell('''## Official integration and scope

CPR integrates normalized logit-difference faithfulness on a **linear** retained
fraction grid. CMD integrates absolute distance from one. Accuracy log-AUC is
separate. CPR is not clipped at one. Grid weights sum to .999, with no added zero
endpoint. Benchmark numbers below come only from the official evaluator.
'''), nbf.v4.new_code_cell('''if RESULTS is not None:
    config=json.loads((RESULTS/'experiment_config.json').read_text())
    display(pd.Series({k:v for k,v in config.items() if k not in ('training_source','source_provenance')}))
    display(pd.read_csv(RESULTS/'cpr_grid_weights.csv'))
    display(pd.read_csv(RESULTS/'patching_checks.csv'))
'''), nbf.v4.new_code_cell('''if RESULTS is not None:
    display(pd.Series(json.loads((RESULTS/'official_metrics.json').read_text())))
    display(pd.read_csv(RESULTS/'faithfulness_curve.csv'))
    accounting=RESULTS/'complete_compute_accounting.csv'
    display(pd.read_csv(accounting if accounting.exists() else RESULTS/'compute_accounting.csv'))
    gate=1 if (RESULTS/'gate1_passed.json').exists() else 0
    display(Image(filename=str(RESULTS/f'gate{gate}_diagnostics.png')))
'''), nbf.v4.new_markdown_cell('''## Execution gates

1. **Gate 0:** source inspection, metric/resume tests, short CUDA edge smoke.
2. **Gate 1:** one official 5,000-step edge baseline; explicit independent dispatch.
3. **Gate 2:** unchanged MAttr on nested 16/32/64/128 pools, fixed validation.
4. **Gate 3:** sampling ablations only after baseline noise and data scaling.
5. **Gate 4:** initialization with full EAP-IG cost; then optional Gate 5 refinement.
6. **Gate 6:** multi-model validation and submissions require explicit approval.

No hypothesis about algorithm improvements has been tested by the smoke run.
'''), nbf.v4.new_code_cell('''if RESULTS is not None:
    display(Markdown((RESULTS/'MATTR_BASELINE_REPORT.md').read_text()))
else:
    print('No completed Gate-0 experiment. Follow src/mattr_optimization/README.md.')
''')]
    nbf.write(book, ROOT/'notebooks/mattr_optimization.ipynb')


if __name__ == '__main__':
    build()
