"""Build the separate standardized IOI experiment notebook; never edit baseline files."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


md('''
# MIB IOI: exact interactions and prediction across prompts

GPT-2 Small, official MIB IOI data, counterfactual attention-head patching.
This notebook uses the existing project CUDA environment. It does not change the
baseline notebook. Helper code is in `mib_ioi/`; official implementations are
imported from the local MIB checkout. Run cells incrementally; flags below control
expensive stages. No conclusions are populated before results exist.
''')
code('''
from pathlib import Path
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import math
import os
import sys
import uuid

ROOT = next((p for p in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
             if (p / 'mib_ioi/runtime.py').is_file()), None)
if ROOT is None:
    raise RuntimeError('Start this notebook from the circuit-sketching repository.')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from IPython.display import display, Markdown
from mib_ioi import runtime as rt, analysis as an

SEED = 42
DEVICE = 'cuda'
MIB_REPO = Path(os.environ.get('MIB_REPO', ROOT / 'external/MIB-circuit-track'))
N_TRAIN, N_VALIDATION, N_PUBLIC_TEST = 128, 128, 128
N_COMPONENTS = 12                        # supported range: 12--16
BATCH_SIZE = 8                          # reduce to 4 or 2 explicitly on OOM
EAP_BATCH_SIZE, N_EAP_EXAMPLES = 4, 32
MIB_EVAL_BATCH_SIZE, N_MIB_EVAL_EXAMPLES = 4, 32
TRAIN_MASKS_PER_DENSITY, EVAL_MASKS_PER_DENSITY = 64, 16
BOOTSTRAP_REPEATS = 500
SAMPLE_SIZES, SAMPLE_REPEATS = [16, 32, 64, 128], 5
MEANINGFUL_INTERACTION_RATIO = 0.05      # prespecified exploratory effect-size criterion
RANK_STABILITY_THRESHOLD = 0.90         # descriptive, conditional on selected heads

RUN_EAP_SELECTION = True
RUN_SINGLETONS = True
RUN_PAIRS = True
RUN_RANDOM_MASKS = True
RUN_FITS = True
RUN_SAMPLE_EFFICIENCY = True
RUN_MIB_EVALUATION = True                # small official node-level validation run
RUN_PUBLIC_TEST = False                 # opt in only after fixing experimental choices
COMPONENTS_FILE = None                  # prior selected_components.csv if skipping EAP
RESUME_DIR = None                       # prior run directory; cache signature must match

rt.seed_everything(SEED)
assert 12 <= N_COMPONENTS <= 16
assert min(BATCH_SIZE, EAP_BATCH_SIZE, N_TRAIN, N_VALIDATION) > 0
assert BOOTSTRAP_REPEATS >= 100
RUN_DIR = (Path(RESUME_DIR).resolve() if RESUME_DIR else
           ROOT / 'results' / ('mib_ioi_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                              + '_' + uuid.uuid4().hex[:8]))
RUN_DIR.mkdir(parents=True, exist_ok=True)
ledger = rt.Ledger(DEVICE, RUN_DIR)
print('Output:', RUN_DIR)
print('Flags:', {k: v for k, v in globals().copy().items() if k.startswith('RUN_') and isinstance(v, bool)})
''')

md('''
## 1. Verified repository conventions

Inspected checkout: MIB `b759df34433c9e31043ba9e02908ce0bf20e894f`, EAP submodule
`7af394a5662de8b23ad6154716a0cd3993d447a3`. Current paths, commits, signatures,
and source hashes are recorded below; a different installation is never substituted.

- `MIB_circuit_track/utils.py`: model key `gpt2` maps to TransformerLens
  `gpt2-small`; task key `ioi` maps to `mib-bench/ioi`.
- `HFEAPDataset` calls Hugging Face `load_dataset(..., split=...)` for
  `train`, `validation`, or public `test`, then applies its official name-token-length filter.
- Each loaded item is `(clean_prompt, corrupted_prompt, [correct_id, incorrect_id])`.
  IOI uses `s2_io_flip_counterfactual`; labels are the leading-space indirect-object
  and subject name tokens. The downloaded rows also expose `template` and `metadata`.
- `get_metric('logit_diff', 'ioi', ...)` subtracts subject from indirect-object
  logits at the final **unpadded** token. `get_metric('acc', ...)` supplies vocabulary
  top-1 accuracy. Corrupted scores below retain the clean labels, as MIB does.
- `eap.utils.tokenize_plus` prepends BOS and right-pads. Graph attention nodes
  expose `hook_result` with dimensions `[batch, position, head, d_model]`.
- `run_attribution.py` exposes a TransformerLens model and calls `attribute_node`
  for node attribution. The API requires uppercase **`EAP`**, despite the README's
  lowercase spelling. Gradients are enabled only during attribution.
- MIB exact node patching removes the node's outgoing edges and uses activations
  from the matched corrupted input. The direct output-replacement adapter below
  is checked against `eap.evaluate.evaluate_graph` on an exact single-head intervention.
- `evaluate_area_under_curve` supplies official node-level validation scores;
  `area_under` / `area_from_1` are reported under their actual API field names.
  Public test is available locally; private leaderboard evaluation requires submission.
- This checkout's `Graph.to_json` tries to index a node score for the sink
  `logits` node and raises `IndexError`. Use official `Graph.to_pt` / `from_pt`
  instead and verify the score round-trip; the upstream checkout is unchanged.
''')
code('''
repo, provenance = rt.configure_repository(ROOT, MIB_REPO)
provenance['local_source_sha256'] = {
    p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    for p in ('mib_ioi/runtime.py', 'mib_ioi/analysis.py', 'notebooks/mib_ioi_experiments.ipynb')}
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME, TASKS_TO_HF_NAMES
from MIB_circuit_track.evaluation import evaluate_area_under_curve
from eap.graph import Graph
from eap.attribute_node import attribute_node
from eap.utils import tokenize_plus

print('MIB:', provenance['mib_commit'], '| EAP:', provenance['eap_commit'])
print('Model:', MODEL_NAME_TO_FULLNAME['gpt2'], '| Dataset:', 'mib-bench/' + TASKS_TO_HF_NAMES['ioi'])
for function in (HFEAPDataset, get_metric, tokenize_plus, attribute_node, evaluate_area_under_curve):
    print(function.__name__, inspect.signature(function))
rt.write_json(RUN_DIR / 'repository_provenance.json', provenance)
with ledger.stage('model_loading', 'one GPT-2 Small model; no task forwards expected'):
    model = rt.load_model(DEVICE)
print(torch.cuda.get_device_name(0), '| dtype:', model.cfg.dtype)
''')

md('''
## 2. Official split loading and reproducible subsets

Subsample **within** each official filtered split using seeded permutations; never
create new train/validation boundaries. Check exact clean-prompt overlap across
the full filtered splits, record source indices/fingerprints, and retain actual
metadata. Public test is loaded/evaluated only if its flag is enabled.
''')
code('''
sizes = {'train': N_TRAIN, 'validation': N_VALIDATION}
if RUN_PUBLIC_TEST:
    sizes['test'] = N_PUBLIC_TEST
with ledger.stage('data_loading', f'official splits: {sizes}; no model forwards'):
    datasets, examples, dataset_info = rt.load_splits(model, sizes, SEED)
examples.to_csv(RUN_DIR / 'ioi_examples.csv', index=False)
rt.write_json(RUN_DIR / 'dataset_manifest.json', dataset_info)
print('Full filtered split overlaps:', dataset_info['full_filtered_split_clean_prompt_overlap'])
display(examples.groupby('split', sort=False).head(2)[[
    'split', 'clean_prompt', 'corrupted_prompt', 'correct_answer', 'incorrect_answer', 'metadata']])
print('Available raw fields:', datasets['train'].dataset.column_names)
display(pd.DataFrame([datasets['train'].dataset[i] for i in range(2)])[['template', 'metadata']])
''')

md('''
## 3. Reproduce the official raw IOI score

`evaluate_ioi_batch` is a thin wrapper around the official tokenizer/metrics.
Raw logit difference, intervention effect (patched minus clean), and official
CPR/CMD-style circuit evaluations are separate outputs. A positive clean
logit difference means the correct name beats the distractor, not necessarily
every vocabulary token. Top-1 accuracy is also reported.
''')
code('''
# One graph-defined head is used only for the correctness probe; selection comes next.
probe_node = next(n for n in Graph.from_model(model).nodes.values() if hasattr(n, 'head'))
probe_scorer = rt.IOIScorer(model, datasets, [(probe_node.layer, probe_node.head)],
                           BATCH_SIZE, ledger, RUN_DIR / 'probe_cache')
baseline_rows = []
with ledger.stage('raw_ioi_scores', f'{2*sum(math.ceil(len(d)/BATCH_SIZE) for d in datasets.values())} '
                  'forward batches for clean/corrupted scores'), ledger.instrument(model):
    for split in datasets:
        for corrupted in (False, True):
            score, accuracy = probe_scorer.score(split, corrupted=corrupted, use_cache=False)
            baseline_rows.append({'prompt_split': split, 'input': 'corrupted' if corrupted else 'clean',
                                  'mean_logit_diff': score.mean(), 'top1_accuracy': accuracy.mean(),
                                  'correct_over_distractor_rate': (score > 0).mean(),
                                  'n_examples': len(score), 'batch_size': BATCH_SIZE})
baseline_results = pd.DataFrame(baseline_rows)
baseline_results.to_csv(RUN_DIR / 'raw_ioi_scores.csv', index=False)
display(baseline_results)
if (baseline_results.query("input == 'clean'").mean_logit_diff <= 0).any():
    raise RuntimeError('Clean mean IOI logit difference is nonpositive; investigate model/data before proceeding.')
display(pd.DataFrame(ledger.stages).tail(1))
''')

md('''
## 4. One exact causal intervention and recovery checks

Fresh forwards verify hook removal, deterministic repeat, batched versus single-example
evaluation, the raw MIB baseline, and equivalence to official graph patching.
The probe head is a correctness fixture, not the research component set.
''')
code('''
with ledger.stage('exact_intervention_checks', '4 prompts, one head, approximately 25 forward batches'), ledger.instrument(model):
    intervention_checks = rt.verify_intervention(probe_scorer, count=4)
rt.write_json(RUN_DIR / 'intervention_checks.json', intervention_checks)
display(pd.DataFrame([intervention_checks]))
probe_scorer.donors.clear()
del probe_scorer
''')

md('''
## 5. Select heads using the official EAP baseline

Run node EAP on a small **training-only** subset and select the top 12--16 attention
heads by absolute official attribution. The graph includes MLP/input nodes, but the
controlled interaction experiment selects attention heads only. The full graph is
saved unchanged for the official baseline evaluation later. Selection is fixed
before examining validation/test effects. A saved component CSV can be supplied
to skip EAP when running only exact experiments.
''')
code('''
eap_graph, ranking = None, None
if RUN_EAP_SELECTION:
    if N_EAP_EXAMPLES > len(datasets['train']):
        raise ValueError('N_EAP_EXAMPLES exceeds the selected training subset.')
    eap_data = rt.subset(datasets['train'], range(N_EAP_EXAMPLES))
    with ledger.stage('official_eap_selection',
                      f'{math.ceil(N_EAP_EXAMPLES/EAP_BATCH_SIZE)} attribution batches; '
                      f'{3*math.ceil(N_EAP_EXAMPLES/EAP_BATCH_SIZE)} forwards plus '
                      f'{math.ceil(N_EAP_EXAMPLES/EAP_BATCH_SIZE)} backwards'), ledger.instrument(model):
        eap_graph, ranking = rt.official_eap(model, eap_data, EAP_BATCH_SIZE, ledger)
    ranking.to_csv(RUN_DIR / 'official_eap_head_ranking.csv', index=False)
    selected = ranking.head(N_COMPONENTS).copy()
    selected['selection_source'] = 'official node EAP on train'
    eap_path = RUN_DIR / 'official_eap_node_importances.pt'
    eap_graph.to_pt(str(eap_path))
    reloaded = Graph.from_pt(str(eap_path))
    torch.testing.assert_close(reloaded.nodes_scores, eap_graph.nodes_scores, equal_nan=True)
    eap_config = {'api': 'eap.attribute_node.attribute_node', 'method': 'EAP', 'intervention': 'patching',
                  'level': 'node', 'aggregation': 'sum', 'metric': 'logit_diff', 'mean': True, 'loss': True,
                  'model': 'gpt2', 'task': 'ioi', 'split': 'train', 'n_examples': N_EAP_EXAMPLES,
                  'batch_size': EAP_BATCH_SIZE, 'seed': SEED,
                  'selected_filtered_indices': dataset_info['train']['selected_filtered_indices'][:N_EAP_EXAMPLES],
                  'artifact': str(eap_path), 'mib_commit': provenance['mib_commit'],
                  'eap_commit': provenance['eap_commit']}
    rt.write_json(RUN_DIR / 'official_eap_config.json', eap_config)
elif COMPONENTS_FILE:
    selected = pd.read_csv(COMPONENTS_FILE).head(N_COMPONENTS)
    selected['selection_source'] = str(Path(COMPONENTS_FILE).resolve())
else:
    raise RuntimeError('Enable RUN_EAP_SELECTION or supply a prior EAP-derived COMPONENTS_FILE; no arbitrary fallback.')
if len(selected) != N_COMPONENTS:
    raise ValueError('The selected component file/ranking has too few heads.')
selected.insert(0, 'component_index', np.arange(N_COMPONENTS)) if 'component_index' not in selected else None
selected.to_csv(RUN_DIR / 'selected_components.csv', index=False)
components = list(zip(selected.layer.astype(int), selected['head'].astype(int)))
display(selected)

cache_identity = {'model': MODEL_NAME_TO_FULLNAME['gpt2'], 'task': 'ioi', 'seed': SEED,
                  'components': components, 'dataset': dataset_info, 'batch_size': BATCH_SIZE,
                  'dtype': str(model.cfg.dtype), 'intervention': 's2_io_flip_counterfactual head-result replacement',
                  'source_sha256': provenance['source_sha256'],
                  'adapter_sha256': hashlib.sha256((ROOT/'mib_ioi/runtime.py').read_bytes()).hexdigest()}
signature = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
cache_root = RUN_DIR / 'exact_cache' / signature
scorer = rt.IOIScorer(model, datasets, components, BATCH_SIZE, ledger, cache_root)
pairs = list(itertools.combinations(range(N_COMPONENTS), 2))
single_masks = np.eye(N_COMPONENTS, dtype=np.int8)
pair_masks = np.stack([single_masks[i] + single_masks[j] for i, j in pairs])
singletons, pair_effects, pair_tables, pair_summaries = {}, {}, {}, []
mask_arrays, masks, all_results, efficiency = {}, None, None, None
comparison_ci = None

def save_experiment_config():
    package_names = ['torch', 'transformer-lens', 'transformers', 'datasets', 'numpy', 'pandas', 'scipy', 'matplotlib']
    config = {'model_key': 'gpt2', 'model': MODEL_NAME_TO_FULLNAME['gpt2'], 'task': 'ioi',
              'dataset': 'mib-bench/' + TASKS_TO_HF_NAMES['ioi'], 'dataset_splits': sizes,
              'seed': SEED, 'components': components, 'component_count': N_COMPONENTS,
              'intervention_type': 'matched s2_io_flip_counterfactual head-result replacement at all positions',
              'batch_size': BATCH_SIZE, 'eap_batch_size': EAP_BATCH_SIZE, 'eap_examples': N_EAP_EXAMPLES,
              'mask_train_per_density': TRAIN_MASKS_PER_DENSITY, 'mask_eval_per_density': EVAL_MASKS_PER_DENSITY,
              'sample_sizes': SAMPLE_SIZES, 'sample_repeats': SAMPLE_REPEATS,
              'bootstrap_repeats': BOOTSTRAP_REPEATS, 'meaningful_interaction_ratio': MEANINGFUL_INTERACTION_RATIO,
              'rank_stability_threshold': RANK_STABILITY_THRESHOLD, 'device': DEVICE,
              'gpu': torch.cuda.get_device_name(0), 'dtype': str(model.cfg.dtype),
              'packages': {p: importlib.metadata.version(p) for p in package_names},
              'provenance': provenance, 'cache_signature': signature, 'dataset_info': dataset_info,
              'run_flags': {k: v for k, v in globals().copy().items() if k.startswith('RUN_') and isinstance(v, bool)},
              'score_semantics': 'raw IOI correct-minus-incorrect logit difference; effects are patched minus clean',
              'mib_evaluation_scope': 'official evaluator on local split subset, not private leaderboard scores'}
    rt.write_json(RUN_DIR / 'experiment_config.json', config)
    return config

config = save_experiment_config()  # available even when stopping after singleton/pair acquisition
''')

md('''
## 6. Exact singleton effects

For each prompt, retain the matched difference `F(e_i) - F(0)`; summarize its mean,
sample standard deviation, and standard error. All positions of each selected head
are replaced with matched counterfactual activations. CPU score caches are reused;
only selected donor slices are retained, and no inference graph is stored.
''')
code('''
if RUN_SINGLETONS:
    singleton_frames = []
    with ledger.stage('singletons', f'{N_COMPONENTS} masks per prompt split; at most '
                      f'{(N_COMPONENTS+2)*sum(math.ceil(len(d)/BATCH_SIZE) for d in datasets.values())} '
                      'forward batches including clean baselines and donors'), ledger.instrument(model):
        for split in datasets:
            singletons[split] = scorer.effects(split, single_masks)
            singleton_frames.append(an.singleton_table(scorer, split, singletons[split]))
    singleton_results = pd.concat(singleton_frames, ignore_index=True)
    singleton_results.to_csv(RUN_DIR / 'singleton_effects.csv', index=False)
    np.savez_compressed(RUN_DIR / 'singleton_effects_per_example.npz', **singletons)
    display(singleton_results)
else:
    print('Singleton experiment skipped. Enable and run this cell before pair/model analysis.')
''')

md('''
## 7. Exact pairwise interactions

Construct each pair once (`i < j`). The signed interaction is
`Delta_ij - Delta_i - Delta_j`; it is not the total pair effect.
Relative size uses `|interaction| / (|Delta_i| + |Delta_j|)` and returns NaN
for a near-zero denominator. Arrays retain all per-example effects.
''')
code('''
if RUN_PAIRS:
    if not singletons:
        raise RuntimeError('Run the singleton cell first.')
    with ledger.stage('exact_pairs', f'{len(pairs)} unique pairs per prompt split; '
                      f'{len(pairs)*sum(math.ceil(len(d)/BATCH_SIZE) for d in datasets.values())} '
                      'forward batches if donors/baselines are cached'), ledger.instrument(model):
        for split in datasets:
            pair_effects[split] = scorer.effects(split, pair_masks)
    np.savez_compressed(RUN_DIR / 'pair_effects_per_example.npz', pairs=np.array(pairs), **pair_effects)
else:
    print('Pair experiment skipped; random-mask regressions can still run without it.')
''')

md('''
## 8. Do interactions matter statistically?

Bootstrap **examples jointly across all pairs** (not independent pairs), preserving
the matching of clean, singleton, and pair scores. Confidence intervals describe
the selected heads conditional on training-based selection. Pair-level signed
one-sample t tests are exploratory and use Benjamini–Yekutieli FDR correction.
Mean absolute interaction equals additive-pair MAE by construction. A nonzero
absolute statistic alone is not evidence of practical importance: compare its
relative scale with the prespecified criterion. These are diagnostic statistics,
not circuit faithfulness metrics. Validation-prompt results provide the main
interaction evidence; training-prompt statistics are selection-conditioned.
''')
code('''
if pair_effects:
    interaction_arrays = {}
    with ledger.stage('interaction_statistics', f'{BOOTSTRAP_REPEATS} CPU prompt bootstraps per split; 0 forwards'):
        for split in pair_effects:
            table, summary, per_example, draws = an.pair_statistics(
                singletons[split], pair_effects[split], pairs, components, split, BOOTSTRAP_REPEATS, SEED)
            pair_tables[split] = table
            pair_summaries.append(summary)
            interaction_arrays[split] = per_example
            draws.to_csv(RUN_DIR / f'pair_bootstrap_{split}.csv', index=False)
    pd.concat(pair_tables.values(), ignore_index=True).to_csv(RUN_DIR / 'pairwise_interactions.csv', index=False)
    np.savez_compressed(RUN_DIR / 'pairwise_interactions_per_example.npz', pairs=np.array(pairs), **interaction_arrays)
    pd.DataFrame(pair_summaries).to_csv(RUN_DIR / 'pairwise_statistics.csv', index=False)
    display(pd.DataFrame(pair_summaries))
    display(pair_tables['validation'].nlargest(10, 'absolute_interaction'))
    an.pair_plots(pair_tables['validation'], N_COMPONENTS, RUN_DIR)
    plt.show()
else:
    print('No pair results; statistics and plots skipped.')
''')

md('''
## 9. Multi-component masks with distinct mask splits

Sample uniformly without replacement at sizes 1, 2, approximately 25%, 50%, and
75% of heads. Finite singleton/pair pools limit the allocation; actual counts
are printed. No mask duplicates occur across train/validation/test mask splits.
The **official prompt splits remain separate**. Learned models see only training
masks on training prompts. Singletons are directly measured for the additive
baseline, and pair diagnostics include size-two test masks; those measurements
are not supplied to learned fits or used to choose hyperparameters. This makes
singleton additive performance on size-one training-prompt masks a construction
check, not an independent discovery result.
''')
code('''
if RUN_RANDOM_MASKS:
    masks, allocation = an.mask_splits(N_COMPONENTS, SEED, TRAIN_MASKS_PER_DENSITY, EVAL_MASKS_PER_DENSITY)
    allocation.to_csv(RUN_DIR / 'mask_allocation.csv', index=False)
    display(allocation.pivot(index='mask_size', columns='mask_split', values='count'))
    print('Quadratic training rank:', np.linalg.matrix_rank(an.features(masks['train'], 2)),
          '/', an.features(masks['train'], 2).shape[1])
    total_masks = sum(map(len, masks.values()))
    estimate = total_masks*sum(math.ceil(len(d)/BATCH_SIZE) for d in datasets.values())
    with ledger.stage('random_masks', f'{total_masks} unique masks; <= {estimate} forward batches '
                      '(previous singleton/pair scores are cached)'), ledger.instrument(model):
        for prompt_split in datasets:
            mask_arrays[prompt_split] = {}
            for mask_split, z in masks.items():
                print(f'{prompt_split} prompts / {mask_split} masks: {len(z)} masks', flush=True)
                mask_arrays[prompt_split][mask_split] = scorer.effects(prompt_split, z)
    an.export_masks(RUN_DIR, masks, mask_arrays)
else:
    print('Random-mask acquisition skipped.')
''')

md('''
## 10. Two first-order predictors

The additive predictor uses exact training-prompt singleton effects. The learned
linear predictor fits random training masks on the same training prompts. Both
have zero intercept because the unmodified intervention effect is exactly zero.
Metrics are reported separately for held-out validation and test **masks**, and
for same-prompt-split versus cross-prompt-split evaluation. No test-mask target
is passed to fitting. NaN correlations/R² flag insufficient or constant strata.
''')
code('''
predictions, coefficient_arrays, result_rows = {}, {}, []
if RUN_FITS:
    if not mask_arrays or not singletons:
        raise RuntimeError('Acquire random masks and singletons before fitting.')
    with ledger.stage('first_order_fits', 'CPU fits/evaluation on cached arrays; 0 forwards'):
        beta1 = an.fit(masks['train'], mask_arrays['train']['train'].mean(1), 1)
        for mask_split in ('validation', 'test'):
            z = masks[mask_split]
            predictions[mask_split] = {'additive singleton': z @ singletons['train'].mean(1),
                                       'learned first order': an.features(z, 1) @ beta1}
            for prompt_split in datasets:
                result_rows += an.prediction_rows(mask_arrays[prompt_split][mask_split].mean(1),
                                                  predictions[mask_split], z, prompt_split, mask_split)
    first_order_results = pd.DataFrame(result_rows)
    first_order_results.to_csv(RUN_DIR / 'first_order_results.csv', index=False)
    display(first_order_results.query("eval_prompt_split == 'validation' and mask_split == 'test'"))
else:
    print('First-order fitting skipped.')
''')

md('''
## 11. Explicit second-order regression

Fit the same targets with singleton features plus every `z_i*z_j` for `i < j`.
Use ordinary least squares with a verified full-column-rank training design;
no regularization tuning uses held-out masks. Compare all methods on identical
held-out masks. A paired prompt bootstrap measures second-order MAE improvement
over learned first order, conditional on the fitted models.
''')
code('''
if RUN_FITS:
    with ledger.stage('second_order_fit', 'CPU fit + paired prompt bootstrap; 0 forwards'):
        beta2 = an.fit(masks['train'], mask_arrays['train']['train'].mean(1), 2)
        second_rows = []
        for mask_split in ('validation', 'test'):
            prediction = an.features(masks[mask_split], 2) @ beta2
            predictions[mask_split]['learned second order'] = prediction
            for prompt_split in datasets:
                second_rows += an.prediction_rows(mask_arrays[prompt_split][mask_split].mean(1),
                                                   {'learned second order': prediction}, masks[mask_split],
                                                   prompt_split, mask_split)
        second_order_results = pd.DataFrame(second_rows)
        all_results = pd.concat([first_order_results, second_order_results], ignore_index=True)
        comparison_ci = an.bootstrap_improvement(mask_arrays['validation']['test'],
                         predictions['test']['learned first order'], predictions['test']['learned second order'],
                         BOOTSTRAP_REPEATS, SEED)
    second_order_results.to_csv(RUN_DIR / 'second_order_results.csv', index=False)
    all_results.to_csv(RUN_DIR / 'prediction_summary.csv', index=False)
    np.savez_compressed(RUN_DIR / 'fitted_coefficients.npz', linear=beta1, quadratic=beta2, pairs=np.array(pairs))
    rt.write_json(RUN_DIR / 'second_order_improvement_bootstrap.json', comparison_ci)
    prediction_export = []
    for prompt_split in datasets:
        for mask_split in ('validation', 'test'):
            for index, mask in enumerate(masks[mask_split]):
                prediction_export.append({'prompt_split': prompt_split, 'mask_split': mask_split,
                                          'mask_index': index, 'mask_size': int(mask.sum()),
                                          'truth': mask_arrays[prompt_split][mask_split][index].mean(),
                                          **{name: p[index] for name, p in predictions[mask_split].items()}})
    pd.DataFrame(prediction_export).to_csv(RUN_DIR / 'heldout_predictions.csv', index=False)
    display(all_results.query("eval_prompt_split == 'validation' and mask_split == 'test' and mask_size == 'all'"))
    display(pd.DataFrame([comparison_ci]))
    an.error_plot(all_results, RUN_DIR)
    plt.show()
''')

md('''
## 12. Generalization from training to validation IOI prompts

Keep coefficients and held-out masks fixed; change only the prompt set used for
the true effects. Positive `MAE_change` means worse cross-prompt prediction.
An optional exact-interaction expansion estimated on training prompts is reported
as an additional diagnostic, separately from the learned quadratic regression.
''')
code('''
if all_results is not None:
    overall = all_results.query("mask_split == 'test' and mask_size == 'all'")
    same = overall.query("eval_prompt_split == 'train'").set_index('method')
    cross = overall.query("eval_prompt_split == 'validation'").set_index('method')
    generalization = pd.DataFrame({'same_split_MAE': same.MAE, 'cross_split_MAE': cross.MAE,
                                   'MAE_change': cross.MAE - same.MAE,
                                   'same_split_Spearman': same.Spearman, 'cross_split_Spearman': cross.Spearman})
    generalization.to_csv(RUN_DIR / 'prompt_generalization.csv')
    display(generalization)
    if pair_effects:
        interaction_train = pair_effects['train'].mean(1) - np.array([
            singletons['train'][i].mean() + singletons['train'][j].mean() for i, j in pairs])
        exact_prediction = (masks['test'] @ singletons['train'].mean(1) +
                            an.features(masks['test'], 2)[:, N_COMPONENTS:] @ interaction_train)
        diagnostic = pd.DataFrame([{'method': 'exact train singleton + pair expansion', 'eval_prompt_split': split,
                                     **an.metrics(mask_arrays[split]['test'].mean(1), exact_prediction)}
                                    for split in ('train', 'validation')])
        diagnostic.to_csv(RUN_DIR / 'exact_pair_expansion_generalization.csv', index=False)
        display(diagnostic)
else:
    print('No fitted predictions; generalization comparison skipped.')
''')

md('''
## 13. Sample efficiency on a fixed validation set

Re-fit using 16/32/64/128 training examples with repeated nested random subsets.
Targets are always the same validation prompts and validation masks. No new
forward passes are needed. Error bars show variability across training subsets,
not independent benchmark replicates. Singleton **magnitude** ranking stability
is measured against all selected training examples. Head selection remains fixed
from the separate EAP pass: this measures effect-estimation sample efficiency,
not end-to-end sample efficiency including selection.
''')
code('''
if RUN_SAMPLE_EFFICIENCY:
    if not mask_arrays or not singletons:
        raise RuntimeError('Acquire masks/singletons before the sample-efficiency cell.')
    with ledger.stage('sample_efficiency', f'{SAMPLE_REPEATS} repeats at {SAMPLE_SIZES} example counts; 0 forwards'):
        efficiency = an.sample_efficiency(masks, mask_arrays, singletons, SAMPLE_SIZES, SAMPLE_REPEATS, SEED)
    efficiency.to_csv(RUN_DIR / 'sample_efficiency.csv', index=False)
    display(efficiency.groupby(['n_train_examples', 'method'])[['MAE', 'Spearman']].mean())
    an.efficiency_plot(efficiency, RUN_DIR)
    plt.show()
else:
    print('Sample-efficiency experiment skipped.')
''')

md('''
## 14. Compute accounting

One forward call is one batch, not one example and not one mask. Report example
forwards separately, including repeated donor/baseline passes. Every official
library forward is instrumented too. Cache hits never increment forward counts.
CUDA synchronizes around each measured forward and stage; stage wall time also
includes backwards, data transfer, and CPU overhead. Per-stage memory peaks
include the resident model. Acquisition costs shared by fits are not charged
again to each regression. The final export refreshes this table after MIB evaluation.
''')
code('''
ledger.save()
display(pd.DataFrame(ledger.stages))
print('Actual forward batches:', len(ledger.calls))
print('Example-forwards:', sum(c['batch_size'] for c in ledger.calls))
print('Unique intervention masks:', len(ledger.mask_keys))
''')

md('''
## 15. Official baseline artifact and exact configuration

The training-only node EAP pass in section 5 is the official baseline; reuse its
graph rather than performing the attribution twice. It scores all graph source
nodes, whereas our interaction study uses the selected attention heads. The saved
configuration records the actual function, case-sensitive method, dataset indices,
batch size, code revisions, and graph path. It is a small-subset reproduction,
not a full leaderboard submission. Edge-level attribution remains future work.
''')
code('''
if eap_graph is not None:
    display(pd.DataFrame([eap_config]))
    display(pd.DataFrame(ledger.stages).query("stage == 'official_eap_selection'"))
    print('Official graph:', eap_path)
else:
    print('EAP was skipped; component-only experiments remain valid, but no official baseline graph exists in this run.')
''')

md('''
## 16. Official MIB evaluation, with no homemade CPR/CMD

Call `MIB_circuit_track.evaluation.evaluate_area_under_curve` unchanged on the
saved EAP graph. It determines circuit sizes, pruning, normalization, and integration.
Save the exact returned fields (`weighted_edge_counts`, `area_under`,
`area_from_1`, `average`, `faithfulnesses`). These are **local subset** results,
not private-test leaderboard scores. Public-test evaluation requires the explicit
flag and uses fixed components/settings; never tune on it. The requested granularity
here is node-level, matching the official EAP graph.
''')
code('''
mib_rows = []
if RUN_MIB_EVALUATION:
    if eap_graph is None:
        raise RuntimeError('Official evaluation requires RUN_EAP_SELECTION; it is independent of exact-pair acquisition.')
    for split in ['validation'] + (['test'] if RUN_PUBLIC_TEST else []):
        n_eval = min(N_MIB_EVAL_EXAMPLES, len(datasets[split]))
        eval_data = rt.subset(datasets[split], range(n_eval))
        with ledger.stage(f'official_mib_{split}', f'10 circuit sizes + clean/corrupted normalization; '
                          f'about {35*math.ceil(n_eval/MIB_EVAL_BATCH_SIZE)} forward batches'), ledger.instrument(model):
            # Re-load through the official serializer: graph node sets do not support deepcopy.
            official = rt.official_evaluation(model, Graph.from_pt(str(eap_path)), eval_data, MIB_EVAL_BATCH_SIZE)
        rt.write_json(RUN_DIR / f'mib_official_raw_{split}.json', official)
        mib_rows.append({'method': 'official EAP', 'model': 'gpt2', 'task': 'ioi', 'level': 'node',
                         'split': split, 'n_examples': n_eval, 'batch_size': MIB_EVAL_BATCH_SIZE,
                         'area_under': official['area_under'], 'area_from_1': official['area_from_1'],
                         'average': official['average'], 'scope': 'local official split subset'})
    pd.DataFrame(mib_rows).to_csv(RUN_DIR / 'mib_baseline_results.csv', index=False)
    display(pd.DataFrame(mib_rows))
else:
    print('Official MIB evaluation skipped by RUN_MIB_EVALUATION; no CPR/CMD values are generated.')
''')

md('''
## 17. Research decision checkpoint — generated from results only

The interaction effect-size threshold and ranking threshold were fixed in the
configuration cell. This is an exploratory checkpoint for selected heads and
sampled prompts. Missing stages produce “not measured,” never inferred outcomes.
''')
code('''
answers = []
validation_pair_summary = next((r for r in pair_summaries if r['prompt_split'] == 'validation'), None)
if validation_pair_summary:
    r = validation_pair_summary
    verdict = ('above' if r['relative_mean_interaction_ci_low'] > MEANINGFUL_INTERACTION_RATIO else
               'below' if r['relative_mean_interaction_ci_high'] < MEANINGFUL_INTERACTION_RATIO else 'uncertain against')
    answers.append(f"A. Validation mean |interaction| = {r['mean_absolute_interaction']:.4f}; "
                   f"relative scale = {r['relative_mean_interaction']:.1%} "
                   f"(95% CI {r['relative_mean_interaction_ci_low']:.1%}--{r['relative_mean_interaction_ci_high']:.1%}), "
                   f"{verdict} the prespecified {MEANINGFUL_INTERACTION_RATIO:.0%} criterion. "
                   f"{r['BY_significant_pairs_0.05']}/{r['n_pairs']} signed-pair tests pass BY FDR 0.05.")
else:
    answers.append('A. Pair interactions: not measured.')
if comparison_ci is not None:
    r = comparison_ci
    verdict = 'supported' if r['ci_low'] > 0 else 'not established'
    answers.append(f"B. Second-order MAE improvement over learned first order = {r['MAE_improvement']:.4f} "
                   f"(95% prompt CI {r['ci_low']:.4f}--{r['ci_high']:.4f}); improvement {verdict}.")
    size_rows = all_results.query("eval_prompt_split == 'validation' and mask_split == 'test' and mask_size != 'all'")
    size_table = size_rows.pivot(index='mask_size', columns='method', values='MAE').sort_index()
    gain = size_table['learned first order'] - size_table['learned second order']
    answers.append('C. MAE improvement by mask size: ' + ', '.join(f'{int(k)} heads: {v:.4f}' for k, v in gain.items()) +
                   '. This descriptive pattern does not establish a monotonic trend.')
    drop = generalization.loc['learned second order']
    answers.append(f"D. Quadratic same-prompt-split MAE {drop.same_split_MAE:.4f}; validation-prompt MAE "
                   f"{drop.cross_split_MAE:.4f}; change {drop.MAE_change:+.4f}. Coefficients were fitted on train prompts only.")
else:
    answers += ['B. Held-out model comparison: not measured.', 'C. Density trend: not measured.',
                'D. Cross-prompt prediction: not measured.']
if efficiency is not None and not efficiency.empty:
    rank_table = efficiency.query("method == 'additive singleton'").groupby('n_train_examples')[
        'singleton_magnitude_rank_vs_full_train'].agg(['mean', 'min'])
    # Exclude the full-data self-comparison, which necessarily has rank correlation 1.
    eligible = rank_table[(rank_table.index < N_TRAIN) & (rank_table['min'] >= RANK_STABILITY_THRESHOLD)]
    finding = (f'{int(eligible.index.min())} examples is the first tested count meeting the '
               f'{RANK_STABILITY_THRESHOLD:.2f} rank threshold in every repeat' if len(eligible) else
               'No tested proper subset meets the ranking threshold in every repeat')
    answers.append('E. ' + finding + '; conditional on fixed EAP-selected heads and a full-train reference.')
else:
    answers.append('E. Sample efficiency/ranking stability: not measured.')
pair_stages = [s for s in ledger.stages if s['stage'] == 'exact_pairs']
if pair_stages:
    s = pair_stages[-1]
    answers.append(f"F. Exact pairs: {len(pairs)} unique pairs across {len(pair_effects)} prompt splits, "
                   f"{s['forward_passes_batches']} actual forward batches, {s['examples_processed']} example-forwards, "
                   f"{s['wall_seconds']:.2f}s wall; peak allocated CUDA {s['peak_cuda_allocated_mib']:.1f} MiB. "
                   'Baseline/donor acquisition is separately charged in compute_accounting.csv.')
else:
    answers.append('F. Exact pair cost: not measured.')
checkpoint = '\\n\\n'.join(answers)
(RUN_DIR / 'research_checkpoint.md').write_text(checkpoint, encoding='utf-8')
display(Markdown(checkpoint))
''')

md('''
## 18. Export configuration, provenance, and all completed stages

CSV files contain aggregate results; NPZ files preserve mask-by-example arrays.
Their example columns align with `ioi_examples.csv` within each prompt split;
pair rows align with the saved `pairs` array, and component indices with
`selected_components.csv`. Score caches use a provenance-dependent directory.
Skipped stages do not produce fabricated CSV rows.
''')
code('''
config = save_experiment_config()
ledger.save()
display(pd.DataFrame(ledger.stages))
print('Saved:', RUN_DIR)
print('Forward batches:', len(ledger.calls), '| example-forwards:', sum(c['batch_size'] for c in ledger.calls),
      '| cache hits:', ledger.hits, '| unique masks:', len(ledger.mask_keys))
display(pd.DataFrame({'artifact': [p.name for p in sorted(RUN_DIR.iterdir()) if p.is_file()]}))
''')

md('''
## 19. Interpretation boundaries and incremental execution

- `RUN_SINGLETONS` and `RUN_PAIRS` do not require random-mask fitting or official
  circuit evaluation. Set those later flags false for a focused exact experiment.
- For incremental kernel sessions, rerun a stage using its existing scorer to use
  caches. `RESUME_DIR` can reuse exact scores only when the cache signature matches;
  newly measured costs then describe the resumed run, not the original acquisition.
- OOM raises an explicit error; lower the relevant batch size and restart/reconfigure
  deliberately. Batch size enters the cache identity. No automatic sample reduction.
- These experiments establish prediction behavior over a selected head set.
  Head selection, bootstrap conditioning, finite density coverage, and subset-size
  uncertainty limit generalization. Public test must remain outside model selection.
- The official baseline evaluates all graph nodes; our 12--16-head surrogate is
  not yet a submitted MIB circuit-discovery method. Edge-level integration and
  efficient higher-order sketching are follow-on experiments.
''')

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata['kernelspec'] = {'display_name': 'Python (mib-sketching CUDA)',
                                  'language': 'python', 'name': 'mib-sketching'}
notebook.metadata['language_info'] = {'name': 'python', 'version': '3.11'}
nbf.validate(notebook)
for i, cell in enumerate(cells):
    if cell.cell_type == 'code':
        compile(cell.source, f'ioi_cell_{i}', 'exec')
nbf.write(notebook, ROOT / 'notebooks' / 'mib_ioi_experiments.ipynb')
print(f'Created notebooks/mib_ioi_experiments.ipynb: {len(cells)} cells, '
      f'{sum(c.cell_type == "code" for c in cells)} code cells')
