"""Generate a lightweight notebook interface; numerical code remains importable."""
from pathlib import Path
import nbformat as nb

ROOT=Path(__file__).resolve().parents[2]


def build():
    cells=[]
    md=lambda x:cells.append(nb.v4.new_markdown_cell(x))
    code=lambda x:cells.append(nb.v4.new_code_cell(x))
    md('''# Prefix-cached ACDC on standardized MIB IOI

Independent implementation of the computational optimization described in the supplied research log, Stage J. Its reported **2.35×** speedup used auto-circuit IOI and squared probability distance; the original implementation and trace are unavailable here. Our MIB adaptation uses the official negative IOI logit difference and a documented compatible exact edge-pruning traversal. It is not an exact numerical reproduction of that independent configuration.

The corrected dynamic sender semantics are retained: donor minus **current intervened sender**. No sketching, Hessians, group deletion or MAttr methods are used. Previous experiments and installed environments are preserved.''')
    code('''from pathlib import Path
from datetime import datetime, timezone
import os, json, subprocess
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT=next(p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'src/acdc_prefix_cache').is_dir())
campaigns=sorted((ROOT/'results/acdc_prefix_cache').glob('*')) if (ROOT/'results/acdc_prefix_cache').exists() else []
RESULTS=campaigns[-1] if campaigns else ROOT/'results/acdc_prefix_cache'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN_STAGE_A=False
RUN_STAGE_B=False
RUN_STAGE_C=False
ALLOW_FULL_GRAPH=False
print('Campaign:',RESULTS)
print('Defaults: GPT-2 Small, official MIB IOI, float32, train 16, validation 16, batch 2, seed 42, threshold 0.05.')

def linux_path(path):
    path=Path(path).resolve()
    return '/mnt/'+path.drive[0].lower()+path.as_posix()[2:] if os.name=='nt' else str(path)

def run(stage):
    argv=['bash',linux_path(ROOT/'src/acdc_prefix_cache/run_wsl.sh'),'--stage',stage,'--out',linux_path(RESULTS)]
    if stage=='c':
        assert ALLOW_FULL_GRAPH, 'Stage C requires explicit full-graph authorization; it can run for hours.'
        argv+=['--allow-full-graph']
    if os.name=='nt':argv=['wsl','-d','Ubuntu','--']+argv
    print('Running:',argv)
    subprocess.run(argv,cwd=ROOT,check=True)
''')
    md('''## Stage A — correctness

The standard executor runs the complete transformer. The cached executor checks the descendant closure of **all** removed receivers against every skipped prefix sender, restores a clean residual boundary, seeds only proven-unaffected current senders, and executes the remaining native blocks. An earlier accepted deletion moves the start boundary earlier, even for a late candidate. Final residual/logit receivers conservatively retain the final block. Cache buffers are cloned before possible modification.

Stage A runs ten real small-TransformerLens unit tests, thirty single-edge queries across layers 0/2/4/6/8/10/11, eight dependent/random multi-edge states, a 32-edge restricted reverse-topological subsequence, a longer 64-edge prefix, checkpoint interruption/rebuild/replay, and held-out final-circuit checks. Official MIB and the corrected shared adapter supply independent oracles. Actual root forwards and block executions are counted; inference uses no gradients. Stage-A timings are diagnostic only.''')
    code('''if RUN_STAGE_A: run('a')
for name in ['correctness_results','stage_a_trajectories','stage_a_validation','runtime_estimates']:
    path=RESULTS/(name+'.csv')
    if path.exists():
        print(name);display(pd.read_csv(path))
    else:print(name+': not measured')
path=RESULTS/'decision_trace_comparison.csv'
if path.exists():
    trace=pd.read_csv(path)
    print('Saved decision comparison records:',len(trace))
    display(trace.groupby('trajectory')[['decision_matches','state_before_matches','state_after_matches','exact_loss_match']].all())
''')
    md('''## Stage B — performance (not automatic)

Review the Stage-A report and runtime estimates before enabling this flag. This stage repeats seven-layer microbenchmarks and matched medium partial trajectories at 16/32/64/128 discovery examples. Both arms pay their own donor construction, tokenization and baseline/cache preparation. Timing synchronizes CUDA, alternates arm order, and records power/clock telemetry, cache bytes, isolated allocator peaks, total elapsed time, and held-out loss. Warmup, micro timings, search time, setup time, and validation are separate. Partial prefixes are late-layer-heavy and cannot establish full-search speedup.''')
    code('''if RUN_STAGE_B: run('b')
for name in ['layerwise_benchmark','full_acdc_benchmark','measured_paired_speedups']:
    path=RESULTS/(name+'.csv')
    if path.exists():
        frame=pd.read_csv(path);print(name+': '+str(len(frame))+' rows');display(frame.head(24))
    else:print(name+': not run')
''')
    md('''## Stage C — full graph (disabled by default)

This stage requires passing A and B, plus explicit `ALLOW_FULL_GRAPH=True` and `RUN_STAGE_C=True`. It visits all 32,491 real edges with identical seeds, dataset, loss, threshold and traversal in both arms. Checkpoints save the actual retained identities and trace; rebuilding caches and replay overhead count toward cost. Do not start this automatically. The independent log's 2.35× speedup and 686-edge circuit are not targets for a different task loss/data implementation.''')
    code('''if RUN_STAGE_C: run('c')
for path in sorted(RESULTS.glob('*.png')):
    print(path.name);display(Image(filename=str(path)))
report=RESULTS/'ACDC_PREFIX_CACHE_REPORT.md'
if report.exists(): display(Markdown(report.read_text(encoding='utf-8')))
else: print('No conclusions: Stage A has not produced a report yet.')
''')
    book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python analysis; WSL model execution','language':'python'}})
    target=ROOT/'notebooks/acdc_prefix_caching.ipynb';nb.write(book,target);return target


if __name__=='__main__':print(build())
