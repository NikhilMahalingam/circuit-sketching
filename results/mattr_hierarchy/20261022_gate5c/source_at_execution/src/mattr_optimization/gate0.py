"""Gate 0: invoke upstream edge CLI with explicit smoke subsets and accounting.

No algorithm variants, full reference training, scaling sweeps or test splits.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
import importlib.util
import inspect
import json
import random
import shutil
import sys
import threading
import time
import traceback
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import torch

from .common import ROOT, UPSTREAM, MIB, GRID, configure, provenance, protected_snapshot, sha, write_json, integration_weights
from .checkpoint import checkpointed, rng_state, restore_rng


class Accounting:
    def __init__(self, out):
        self.out = out
        self.phase = None
        self.rows, self.events = [], []
        self.counts = dict(forward_batches=0, example_forwards=0, backward_calls=0, optimizer_updates=0)
        self.grad_norms = []
        self.process = psutil.Process()
        self.peak_rss = self.process.memory_info().rss
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self.sample, daemon=True)
        self.worker.start()

    def sample(self):
        while not self.stop.wait(.05):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)

    @contextmanager
    def stage(self, name):
        torch.cuda.synchronize()
        before, self.phase = self.counts.copy(), name
        self.peak_rss = self.process.memory_info().rss
        torch.cuda.reset_peak_memory_stats()
        start, success = time.perf_counter(), False
        try:
            yield
            success = True
        finally:
            torch.cuda.synchronize()
            self.rows.append({'stage': name, 'success': success,
                'seconds': time.perf_counter()-start,
                **{k: self.counts[k]-before[k] for k in before},
                'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                'peak_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
                'sampled_peak_cpu_rss_mib': self.peak_rss/2**20})
            self.phase = None
            self.save()

    def attach(self, model):
        self.model = model
        original = model.forward
        def forward(*args, **kw):
            tokens = args[0] if args else kw.get('input', kw.get('input_ids'))
            success = False
            try:
                result = original(*args, **kw)
                success = True
                return result
            finally:
                self.counts['forward_batches'] += 1
                self.counts['example_forwards'] += len(tokens)
                self.events.append({'stage': self.phase, 'batch_size': len(tokens),
                    'grad_enabled': torch.is_grad_enabled(), 'success': success})
        model.forward = forward
        return model

    def update(self, norm):
        self.counts['optimizer_updates'] += 1
        self.grad_norms.append(norm)

    def save(self):
        pd.DataFrame(self.rows).to_csv(self.out/'compute_accounting.csv', index=False)
        pd.DataFrame(self.events).to_csv(self.out/'forward_events.csv', index=False)


class Selection:
    def __init__(self, out, seed, train_size, validation_size):
        self.out, self.seed = out, seed
        self.sizes = {'train': train_size, 'validation': validation_size}
        self.datasets, self.sampled = {}, []

    def select(self, dataset, split):
        if split not in self.sizes:
            raise ValueError('Gate 0 only loads official train and validation.')
        size = self.sizes[split]
        seed = self.seed if split == 'train' else 42
        indices = np.random.default_rng(seed).permutation(len(dataset))[:size].tolist()
        if len(indices) != size:
            raise ValueError('Insufficient official examples.')
        fingerprint = dataset.dataset._fingerprint
        dataset.dataset = dataset.dataset.select(indices)
        rows = []
        for i, source in enumerate(indices):
            clean, corrupted, labels = dataset[i]
            meta = dataset.dataset[i]['metadata']
            names = [meta['indirect_object'], meta['subject'], meta['random_c']]
            ids = [dataset.tokenizer(' '+n, add_special_tokens=False).input_ids for n in names]
            if any(len(x) != 1 for x in ids) or [x[0] for x in ids[:2]] != list(labels):
                raise ValueError('Official IOI labels are not single tokens; inspect metric before running.')
            rows.append({'split': split, 'subset_index': i, 'official_filtered_index': source,
                'clean_prompt': clean, 'corrupted_prompt': corrupted,
                'correct_token': labels[0], 'incorrect_token': labels[1],
                'metadata': json.dumps(meta, sort_keys=True)})
        pd.DataFrame(rows).to_csv(self.out/f'{split}_manifest.csv', index=False)
        write_json(self.out/f'{split}_selection.json', {'seed': seed, 'indices': indices,
            'official_fingerprint': fingerprint, 'selected_fingerprint': dataset.dataset._fingerprint})
        self.datasets[split] = dataset
        if len(self.datasets) == 2:
            a, b = [set(d.dataset['prompt']) for d in self.datasets.values()]
            if a & b:
                raise ValueError('Train and validation prompts overlap.')


def check_patching(loss_fn, data, model, out):
    """Use the upstream live edge closure and the official pruned hard-graph oracle."""
    from eap.evaluate import evaluate_graph
    from MIB_circuit_track.metrics import get_metric
    graph = inspect.getclosurevars(loss_fn).nonlocals['graph']
    state = rng_state()
    records = []
    # Prune first so both evaluators receive exactly the same canonical edge mask.
    gen = torch.Generator().manual_seed(731)
    random_scores = torch.rand(graph.scores.shape, generator=gen)
    graph.scores[graph.real_edge_mask.bool()] = random_scores[graph.real_edge_mask.bool()]
    dataset = data.datasets['train']
    one = torch.utils.data.Subset(dataset, [0])
    from MIB_circuit_track.dataset import collate_EAP
    loader = torch.utils.data.DataLoader(one, batch_size=1, collate_fn=collate_EAP)
    metric = partial(get_metric('logit_diff', 'ioi', model.tokenizer, model), mean=False, loss=True)
    try:
        for fraction in (0., .1, .5, 1.):
            graph.apply_topn(int(fraction*len(graph.edges)), absolute=False, level='edge', prune=True)
            mask = graph.in_graph[graph.real_edge_mask.bool()].float().to(model.cfg.device)
            with torch.no_grad(), patch.object(random, 'randint', return_value=0):
                live = float(loss_fn(mask))
                oracle = float(evaluate_graph(model, graph, loader, metric, quiet=True).mean())
            error = abs(live-oracle)
            if not np.isclose(live, oracle, atol=5e-5, rtol=1e-5):
                raise AssertionError(f'Official hard-mask patching mismatch: {fraction}: {live}, {oracle}')
            records.append({'requested_fraction': fraction, 'retained_edges': int(mask.sum()),
                'upstream_live_loss': live, 'official_mib_loss': oracle, 'absolute_error': error})
        pd.DataFrame(records).to_csv(out/'patching_checks.csv', index=False)
    finally:
        graph.reset()
        restore_rng(state)


def run(args):
    configure()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; no CPU fallback for the smoke experiment.')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'gate0_passed.json').exists():
        raise ValueError('Completed run directory: use a new timestamp for a new run.')
    protected = protected_snapshot()
    prov = provenance()
    config = {'gate': 0, 'granularity': 'edge', 'model': 'gpt2', 'task': 'ioi',
        'seed': 42, 'smoke_steps': args.steps, 'train_examples': 16, 'validation_examples': 32,
        'batch_size': 2, 'training_examples_per_update': 1, 'lr': .05, 'optimizer': 'adam',
        'adam_eps': 1e-8, 'adam_betas': [.9, .999], 'T': .5, 'n_iters': 30,
        'masking': 'topk', 'k_schedule': 'uniform', 'mode': 'iso', 'cf_cache_gb': 1.,
        'tf32': False, 'validation_seed': 42, 'training_source': prov,
        'scope': 'smoke only, not the uncapped 5000-step reference'}
    config['adapter_sha256'] = {name: sha(ROOT/'src/mattr_optimization'/name)
        for name in ('common.py', 'checkpoint.py', 'gate0.py')}
    identity = json.dumps(config, sort_keys=True)
    target = out/'experiment_config.json'
    if target.exists() and json.loads(target.read_text()) != config:
        raise ValueError('Run configuration changed; choose a new directory.')
    # Keep every interrupted attempt's measured cost. Donor caches are rebuilt on resume.
    if target.exists():
        previous = out/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        previous.mkdir(parents=True, exist_ok=False)
        for name in ('compute_accounting.csv', 'forward_events.csv', 'process_summary.json',
                     'training_metrics.csv', 'training_example_draws.csv', 'gradient_norms.csv',
                     'training_diagnostics.json', 'failure.json', 'protected_previous_artifacts.json'):
            if (out/name).exists():
                shutil.copyfile(out/name, previous/name)
    write_json(out/'protected_previous_artifacts.json', protected)
    write_json(out/'source_provenance.json', prov)
    write_json(target, config)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('Gate 0 only: seed=42; 16 train / 32 validation; '
          f'{args.steps} one-example updates; 10 official edge-grid points; eval batch=2.', flush=True)
    selection = Selection(out, 42, 16, 32)
    meter = Accounting(out)
    from transformer_lens import HookedTransformer
    from MIB_circuit_track.dataset import HFEAPDataset
    import MIB_circuit_track.evaluation as evaluation
    import matryoshka_attribution as package
    original_model = HookedTransformer.from_pretrained
    original_data = HFEAPDataset.__init__
    original_train = package.learn_scores
    original_eval = evaluation.evaluate_area_under_curve
    original_backward = torch.Tensor.backward
    losses = []

    def model_load(*a, **kw):
        with meter.stage('model_loading'):
            model = original_model(*a, **kw)
            if model.cfg.dtype != torch.float32:
                raise ValueError('Expected GPT-2 float32.')
            # Preserve upstream parameter flags and eval behavior. No new patching code.
            return meter.attach(model)

    def data_load(dataset, *a, **kw):
        with meter.stage('data_loading_'+kw['split']):
            original_data(dataset, *a, **kw)
            selection.select(dataset, kw['split'])

    def train(total, loss_fn, **kw):
        if total != 32491:
            raise ValueError(f'Unexpected edge score count {total}; node substitution forbidden.')
        with meter.stage('patching_validation'):
            check_patching(loss_fn, selection, meter.model, out)
        original_draw = random.randint
        def sampled_loss(mask):
            def draw(a, b):
                idx = original_draw(a, b)
                selection.sampled.append(idx)
                return idx
            with patch.object(random, 'randint', draw):
                return loss_fn(mask)
        with meter.stage('training'):
            result = checkpointed(original_train, total, sampled_loss,
                folder=out/'checkpoints', identity=identity, on_update=meter.update, **kw)
        losses.extend(result.train_log)
        pd.DataFrame(losses, columns=['step', 'k', 'retained_fraction', 'loss']).to_csv(out/'training_metrics.csv', index=False)
        pd.DataFrame({'subset_index': selection.sampled}).to_csv(out/'training_example_draws.csv', index=False)
        pd.DataFrame({'gradient_norm': meter.grad_norms}).to_csv(out/'gradient_norms.csv', index=False)
        closure = inspect.getclosurevars(loss_fn).nonlocals
        write_json(out/'training_diagnostics.json', {'counterfactual_cache_stats': closure['cc_stats'],
            'cached_bytes': closure['cc_bytes'], 'actually_sampled_unique': len(set(selection.sampled)),
            'sampled_presentations_this_attempt': len(selection.sampled),
            'model_parameters_requiring_grad': sum(p.numel() for p in meter.model.parameters() if p.requires_grad),
            'model_parameters_with_grad': sum(p.numel() for p in meter.model.parameters() if p.grad is not None)})
        return result

    def evaluate(*a, **kw):
        if kw.get('level') != 'edge' or kw.get('absolute') is not False:
            raise ValueError('Official signed edge evaluation required.')
        with meter.stage('official_validation_evaluation'), torch.no_grad():
            result = original_eval(*a, **kw)
        model, graph = a[:2]
        rows = [{'edge': e.name, 'score': float(graph.scores[e.matrix_index])} for e in graph.edges.values()]
        pd.DataFrame(rows).sort_values('score', ascending=False).to_csv(out/'ranking.csv', index=False)
        pd.DataFrame({'fraction': GRID, 'requested_edges': [int(x*len(rows)) for x in GRID],
            'weighted_edge_count': result[0], 'faithfulness': result[4],
            'accuracy': result[5], 'linear_cpr_weight': integration_weights()}).to_csv(out/'faithfulness_curve.csv', index=False)
        write_json(out/'official_metrics.json', {'cpr': result[1], 'cmd': result[2],
            'mean_faithfulness': result[3], 'accuracy_auc': result[6], 'validation_examples': 32,
            'baseline_scope': 'Gate-0 smoke, not full reference'})
        return result

    def backward(tensor, *a, **kw):
        result = original_backward(tensor, *a, **kw)
        meter.counts['backward_calls'] += 1
        return result

    argv = ['eval_mib_edge.py', '--mib-path', str(MIB), '--model', 'gpt2', '--task', 'ioi',
        '--steps', str(args.steps), '--k-schedule', 'uniform', '--masking', 'topk', '--mode', 'iso',
        '--lr', '0.05', '--optimizer', 'adam', '--split', 'validation', '--train-split', 'train',
        '--batch-size', '2', '--seed', '42', '--cf-cache-gb', '1', '--no-wandb', '--output', str(out/'upstream')]
    write_json(out/'upstream_command.json', {'argv': argv, 'local_adapters':
        ['16/32 deterministic subsets', 'source guards', 'RNG/Adam checkpoints',
         'accounting', 'hard-mask oracle validation', 'no-grad held-out evaluation'],
        'upstream_hyperparameters_unchanged_except_smoke_steps': True})
    spec = importlib.util.spec_from_file_location('official_edge_gate0', UPSTREAM/'scripts/mib/eval_mib_edge.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    success = False
    try:
        with patch.object(HookedTransformer, 'from_pretrained', side_effect=model_load), \
             patch.object(HFEAPDataset, '__init__', data_load), \
             patch.object(package, 'learn_scores', train), \
             patch.object(evaluation, 'evaluate_area_under_curve', evaluate), \
             patch.object(torch.Tensor, 'backward', backward), patch.object(sys, 'argv', argv):
            module.main()
        for name, digest in protected.items():
            if sha(ROOT/name) != digest:
                raise AssertionError(f'Previous artifact changed: {name}')
        write_json(out/'gate0_passed.json', {'passed': True, 'steps': args.steps,
            'official_edge_count': 32491, 'previous_artifacts_unchanged': True,
            'gate1_executed': False, 'test_split_accessed': False})
        success = True
    except Exception as error:
        write_json(out/'failure.json', {'type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'settings_unchanged': True})
        if isinstance(error, torch.cuda.OutOfMemoryError):
            raise RuntimeError('CUDA OOM: settings unchanged; inspect memory before explicitly configuring a new run.') from error
        raise
    finally:
        meter.stop.set()
        meter.worker.join()
        meter.save()
        write_json(out/'process_summary.json', {'success': success, 'counts_this_attempt': meter.counts})
    from .report import generate
    generate(out)
    print('Gate 0 complete; Gate 1 and all sweeps remain unrun:', out, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=__import__('pathlib').Path,
        default=ROOT/'results/mattr_optimization'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        from .report import generate
        generate(args.out)
    else:
        if not 1 <= args.steps <= 100:
            parser.error('Gate 0 permits only 1–100 smoke steps; full training requires Gate 1.')
        run(args)


if __name__ == '__main__':
    main()
