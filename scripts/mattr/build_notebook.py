"""Notebook interface; all model execution dispatches to isolated WSL scripts."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[2]
cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))

md('''# MAttr reproduction on standardized MIB IOI

This interface launches the **official** node-level MAttr recipe in the isolated Ubuntu environment, then reads saved artifacts. It does not import the Windows Sketch-ACDC model environment into training. Previous experiments and reports remain unchanged.

Inspected MAttr commit: `af82b2eef145dd3e66c793a413352d6931a152d7`; its required MIB fork is pinned to `582fdbc`, with EAP-IG submodule `3fb341e`. The README's Python 3.12/default `lm` dependency group is separate from the older project's stack. See `scripts/mattr/setup_wsl.sh` for the inspected installation and official tests.

Source: [official repository](https://github.com/aryamanarora/matryoshka-attribution). Measurements below appear only after execution.''')
code('''from pathlib import Path
import json, os, subprocess, sys
from datetime import datetime, timezone
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=next((p for p in [Path.cwd().resolve(),*Path.cwd().resolve().parents] if (p/'scripts/mattr').is_dir()),None)
assert ROOT is not None, 'Open this notebook from inside the project checkout.'
# Set to an existing campaign to resume; each job has its own completion marker.
campaigns=sorted(p for p in (ROOT/'results/mattr_reproduction').glob('*') if p.is_dir()) if (ROOT/'results/mattr_reproduction').exists() else []
RESULTS=campaigns[-1] if campaigns else ROOT/'results/mattr_reproduction'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
SEEDS=[42,43,44]
DISCOVERY_SIZES=[16,32,64,128]
VALIDATION_N=128
STEPS=500
RUN_TESTS=False
RUN_SMOKE=False
RUN_REFERENCE=False
RUN_SCALING=False
RUN_BASELINES=False
RUN_CHECKPOINT_EVALUATIONS=False
RUN_PUBLIC_CONFIRMATION=False
RUN_INTERACTIONS=False  # exploratory; not part of the reference reproduction
print('Results:',RESULTS)
print('Fixed seeds:',SEEDS,'discovery pools:',DISCOVERY_SIZES,'validation:',VALIDATION_N)
assert SEEDS==[42,43,44] and DISCOVERY_SIZES==[16,32,64,128] and VALIDATION_N==128 and STEPS==500, 'These constants describe the fixed protocol; use a new script configuration/output directory to change it.'
def linux_path(path):
    path=Path(path).resolve()
    return '/mnt/'+path.drive[0].lower()+path.as_posix()[2:] if os.name=='nt' else str(path)
def run_stage(stage):
    script=ROOT/'scripts/mattr/campaign_wsl.sh'
    argv=['bash',linux_path(script),'--out',linux_path(RESULTS),'--stage',stage]
    if os.name=='nt': argv=['wsl','-d','Ubuntu','--']+argv
    print('Running:',argv)
    subprocess.run(argv,check=True,cwd=ROOT)
''')
md('''## Gates and reference

The campaign first requires the official unit tests, wrapper/resume tests, and a short real GPT-2 training/evaluation smoke. The reference keeps Adam lr 0.05, 500 steps, soft sigmoid top-k, uniform sparsity, iso/denoising, and the learned input node. The uncapped reference follows the documented full-data command. Comparisons subsequently use one fixed 128-prompt validation cohort; the reference ranking is also evaluated on that cohort so unlike cohorts are not mixed.

Training samples with replacement. Configured unique pool size, actually sampled unique prompts, and total presentations are separate fields. Safe resumption restores official Adam state, scores, and Python/Torch RNGs; source/configuration identities must match.''')
code('''if RUN_TESTS: run_stage('tests')
if RUN_SMOKE: run_stage('smoke')
if RUN_REFERENCE: run_stage('reference')
if RUN_SCALING: run_stage('scaling')
if RUN_BASELINES: run_stage('baselines')
if RUN_CHECKPOINT_EVALUATIONS: run_stage('checkpoints')
if RUN_PUBLIC_CONFIRMATION: run_stage('confirmation')
if RUN_INTERACTIONS:
    raise NotImplementedError('Exploratory interactions remain gated until the main reproduction is reviewed.')
if any([RUN_SMOKE,RUN_REFERENCE,RUN_SCALING,RUN_BASELINES,RUN_CHECKPOINT_EVALUATIONS,RUN_PUBLIC_CONFIRMATION]):
    run_stage('report')
''')
md('''## Circuit quality and compute

All compared methods use the official MIB node evaluator, signed ranking (`absolute=False`), counterfactual patching, and its ten retained-fraction points. `area_under` is the linear-fraction AUC of normalized **logit difference** (CPR); `area_from_1` integrates absolute deviation from one (CMD label in the official CLI); `acc_auc` separately integrates positive-logit-difference accuracy over log sparsity. Weighted edge count is not retained-node count. These are local public-split measurements, not a private leaderboard score.

Equal eligible discovery data is **not** equal compute. Attribution construction and evaluation are timed separately, including forwards, backwards, optimization, preparation, CPU memory, and synchronized CUDA timing. EAP/EAP-IG attribution uses batch size 1 so the official mean-loss accumulation weights examples uniformly; all methods retain evaluation batch size 20.''')
code('''tables={}
views={
    'training_results':['kind','discovery_pool','seed','sampled_unique','presentations','cpr_auc','cmd_auc','accuracy_auc','construction_wall_seconds','validation_wall_seconds'],
    'dataset_scaling':['method','discovery_pool','runs','validation_cpr_mean','validation_cpr_std','cmd_mean','accuracy_auc_mean','construction_seconds_mean','gap_mean'],
    'mattr_vs_eap':['method','discovery_pool','seed','cpr_auc','cmd_auc','accuracy_auc','input_node_rank','construction_wall_seconds'],
    'quality_vs_compute':['method','discovery_pool','seed','cpr_auc','accuracy_auc','matched_protocol_wall_seconds','attribution_forward_batches','attribution_backward_calls'],
    'checkpoint_quality':['discovery_pool','seed','step','train_cpr_auc','validation_cpr_auc','train_validation_gap','training_seconds_to_checkpoint'],
    'public_test_confirmation':['method','discovery_pool','seed','test_examples','cpr_auc','cmd_auc','accuracy_auc','process_wall_seconds']}
for name in ['training_results','dataset_scaling','mattr_vs_eap','quality_vs_compute','checkpoint_quality','stage_summary','training_curves','public_test_confirmation']:
    path=RESULTS/(name+'.csv')
    if path.exists():
        tables[name]=pd.read_csv(path)
        print('\\n'+name+': '+str(len(tables[name]))+' rows; full table: '+str(path))
        columns=[c for c in views.get(name,tables[name].columns) if c in tables[name].columns]
        display(tables[name][columns].head(24) if name=='training_curves' else tables[name][columns])
    else: print(name+': not measured yet')
''')
md('''## Saved figures and research checkpoint

Curves condition on the fixed validation subset and the original node basis. Training-quality curves show the selected training pool; validation quality uses the unchanged held-out cohort. Intermediate checkpoints are diagnostics, not extra optimization steps. Raw results and rankings are retained for reproducibility. The official README permits local public-test evaluation. Final confirmation freezes the validation-selected discovery pool, all nine ranking hashes, the three preset seeds, and a new 128-example public-test cohort before test data is loaded. Test scores never change the selected settings. Its actual completion status appears in the table and report.

The final report distinguishes our measurements from the authors' published results and identifies incomplete jobs. No sketching improvements or edge-level 5,000-step experiments are launched.''')
code('''from IPython.display import display, Image
for path in sorted(RESULTS.glob('*.png')):
    print(path.name)
    display(Image(filename=str(path)))
report=ROOT/'MATTR_REPRODUCTION_REPORT.md'
if report.exists(): print(report.read_text(encoding='utf-8'))
else: print('No research conclusions have been generated yet.')
''')
book=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (analysis; WSL execution)','language':'python','name':'python3'},'language_info':{'name':'python'}})
nbf.write(book,ROOT/'notebooks/mib_mattr_reproduction.ipynb')
print(ROOT/'notebooks/mib_mattr_reproduction.ipynb')
