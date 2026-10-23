"""Gate 1 only: uncapped, official 5,000-step GPT-2/IOI edge baseline."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import inspect
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import pandas as pd
import numpy as np
import torch

from .common import ROOT, UPSTREAM, MIB, GRID, configure, provenance, protected_snapshot, sha, write_json, integration_weights
from .checkpoint import checkpointed
from .gate0 import Accounting, check_patching


class LiveAccounting(Accounting):
    """Append real root forwards immediately; keep cold-start and interrupted costs."""
    def __init__(self, out):
        super().__init__(out)
        self.started = time.perf_counter()
        self.stream = (out/'forward_events.jsonl').open('a', buffering=1)

    def attach(self, model):
        super().attach(model)
        original = model.forward
        def forward(*a, **kw):
            try:
                return original(*a, **kw)
            finally:
                self.stream.write(json.dumps(self.events[-1])+'\n')
        model.forward = forward
        return model

    def status(self, **extra):
        write_json(self.out/'live_status.json', {'elapsed_seconds': time.perf_counter()-self.started,
            'stage': self.phase, **self.counts, **extra})


class FullData:
    """Audit every row but preserve the official loader's order and full size."""
    def __init__(self, out):
        self.out = out
        self.datasets = {}
        self.sampled = []

    def audit(self, dataset, split):
        if split not in ('train', 'validation'):
            raise ValueError('Gate 1 never accesses public/private test splits.')
        rows = []
        for i in range(len(dataset)):
            clean, donor, labels = dataset[i]
            raw = dataset.dataset[i]
            meta = raw['metadata']
            names = [meta['indirect_object'], meta['subject'], meta['random_c']]
            ids = [dataset.tokenizer(' '+n, add_special_tokens=False).input_ids for n in names]
            if any(len(x) != 1 for x in ids) or [x[0] for x in ids[:2]] != list(labels):
                raise ValueError(f'Non-single-token official IOI answer at {split}:{i}.')
            rows.append({'split': split, 'official_filtered_index': i, 'clean_prompt': clean,
                'corrupted_prompt': donor, 'correct_token': labels[0], 'incorrect_token': labels[1],
                'metadata': json.dumps(meta, sort_keys=True)})
        pd.DataFrame(rows).to_csv(self.out/f'{split}_manifest.csv', index=False)
        write_json(self.out/f'{split}_selection.json', {'split': split, 'uncapped': True,
            'preserves_official_order': True, 'examples': len(dataset),
            'official_fingerprint': dataset.dataset._fingerprint,
            'official_cache_files': dataset.dataset.cache_files})
        self.datasets[split] = dataset
        if len(self.datasets) == 2:
            clean_sets = [set(d.dataset['prompt']) for d in self.datasets.values()]
            if clean_sets[0] & clean_sets[1]:
                raise ValueError('Official train/validation clean prompts overlap.')


def gate0_evidence():
    evidence = sorted((ROOT/'results/mattr_optimization').glob('*/gate0_passed.json'))
    for path in reversed(evidence):
        marker = json.loads(path.read_text())
        if marker.get('passed') and marker.get('official_edge_count') == 32491:
            return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}
    raise RuntimeError('A passing official edge Gate 0 campaign is required.')


def config_for(out, prov, evidence, batch_size, cache_gb):
    return {'gate': 1, 'model': 'gpt2', 'task': 'ioi', 'granularity': 'edge', 'seed': 42,
        'steps': 5000, 'train_split': 'train', 'evaluation_split': 'validation',
        'train_subset_cap': None, 'validation_subset_cap': None, 'preserve_official_row_order': True,
        'optimizer': 'adam', 'lr': .05, 'adam_eps': 1e-8, 'adam_betas': [.9, .999],
        'masking': 'topk', 'k_schedule': 'uniform', 'mode': 'iso', 'T': .5, 'n_iters': 30,
        'evaluation_batch_size': batch_size, 'training_examples_per_update': 1,
        'counterfactual_cache_gb': cache_gb, 'tf32': False,
        'checkpoint_interval': 100, 'gate0_evidence': evidence,
        'source_provenance': prov,
        'adapter_sha256': {name: sha(ROOT/'src/mattr_optimization'/name)
            for name in ('gate1.py', 'gate0.py', 'checkpoint.py', 'common.py')},
        'resource_overrides': {'evaluation_batch_size': batch_size != 20,
                               'counterfactual_cache_gb': cache_gb != 4.}}


def run(args):
    configure()
    if not torch.cuda.is_available():
        raise RuntimeError('Gate 1 requires CUDA; CPU fallback is disabled.')
    out = args.out.resolve()
    if (out/'gate1_passed.json').exists():
        raise ValueError('Completed Gate 1 directory; do not overwrite it.')
    prov = provenance()
    config = config_for(out, prov, gate0_evidence(), args.batch_size, args.cf_cache_gb)
    target = out/'experiment_config.json'
    if target.exists() and json.loads(target.read_text()) != config:
        raise ValueError('Resume configuration/source identity differs; use a new directory.')
    out.mkdir(parents=True, exist_ok=True)
    if target.exists():
        archive = out/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archive.mkdir(parents=True)
        for name in ('compute_accounting.csv', 'forward_events.csv', 'forward_events.jsonl',
                     'process_summary.json', 'training_example_draws.csv', 'gradient_norms.csv',
                     'live_status.json', 'failure.json'):
            if (out/name).exists():
                shutil.move(str(out/name), str(archive/name))
    protected = protected_snapshot()
    # Gate-0 artifacts can still be untracked in the working tree; protect those too.
    for marker in (ROOT/'results/mattr_optimization').glob('*/gate0_passed.json'):
        for p in marker.parent.rglob('*'):
            if p.is_file():
                protected[str(p.relative_to(ROOT))] = sha(p)
    write_json(out/'protected_previous_artifacts.json', protected)
    write_json(target, config)
    write_json(out/'source_provenance.json', prov)
    snapshot = out/'implementation_at_execution'
    snapshot.mkdir(exist_ok=True)
    for name in config['adapter_sha256']:
        shutil.copyfile(ROOT/'src/mattr_optimization'/name, snapshot/name)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print('Gate 1: seed 42; full official train/validation in original order; 5000 edge-score '
          f'updates; eval batch {args.batch_size}; donor cache {args.cf_cache_gb} GB. No sweeps.', flush=True)
    data = FullData(out)
    meter = LiveAccounting(out)
    from transformer_lens import HookedTransformer
    from MIB_circuit_track.dataset import HFEAPDataset
    import MIB_circuit_track.evaluation as evaluation
    import matryoshka_attribution as package
    original_model = HookedTransformer.from_pretrained
    original_data = HFEAPDataset.__init__
    original_train = package.learn_scores
    original_eval = evaluation.evaluate_area_under_curve
    original_backward = torch.Tensor.backward
    identity = json.dumps(config, sort_keys=True)

    def model_load(*a, **kw):
        with meter.stage('model_loading'):
            model = original_model(*a, **kw)
            if model.cfg.dtype != torch.float32:
                raise ValueError('Expected float32 GPT-2.')
            # Keep upstream weight-gradient behavior for the official baseline.
            return meter.attach(model)

    def data_load(dataset, *a, **kw):
        with meter.stage('data_loading_'+kw['split']):
            original_data(dataset, *a, **kw)
            data.audit(dataset, kw['split'])
        print(f'Official {kw["split"]}: {len(dataset)} examples; uncapped, original row order.', flush=True)

    def train(total, loss_fn, **kw):
        if total != 32491 or kw['steps'] != 5000:
            raise ValueError('Gate 1 requires all 32491 edge scores and 5000 iterations.')
        from matryoshka_attribution.trainer import learn_scores
        defaults = inspect.signature(learn_scores).parameters
        assert defaults['adam_eps'].default == 1e-8
        assert defaults['adam_betas'].default == (.9, .999)
        torch.cuda.synchronize()
        original_draw = random.randint
        draws = (out/'training_example_draws.csv').open('w', buffering=1)
        draws.write('presentation_this_attempt,official_filtered_index\n')
        gradients = (out/'gradient_norms.csv').open('w', buffering=1)
        gradients.write('update_this_attempt,gradient_norm\n')
        def sampled_loss(mask):
            def draw(a, b):
                idx = original_draw(a, b)
                data.sampled.append(idx)
                draws.write(f'{len(data.sampled)},{idx}\n')
                return idx
            with patch.object(random, 'randint', draw):
                return loss_fn(mask)
        def update(norm):
            meter.update(norm)
            gradients.write(f'{len(meter.grad_norms)},{norm}\n')
        previous = kw.pop('on_step', None)
        def step_done(step, k, loss, scores):
            if previous:
                previous(step, k, loss, scores)
            if (step+1) % 50 == 0 or step == 0:
                meter.status(completed_iteration=step+1, current_loss=loss, k=k)
                print(f'Gate 1 progress: {step+1}/5000; actual forwards={meter.counts["forward_batches"]}; '
                      f'elapsed={time.perf_counter()-meter.started:.1f}s', flush=True)
        try:
            with meter.stage('training'):
                result = checkpointed(original_train, total, sampled_loss, folder=out/'checkpoints',
                    identity=identity, interval=100, on_update=update, on_step=step_done, **kw)
        finally:
            draws.close()
            gradients.close()
        pd.DataFrame(result.train_log, columns=['step', 'k', 'retained_fraction', 'loss']).to_csv(out/'training_metrics.csv', index=False)
        torch.save(result.scores, out/'final_scores.pt')
        closure = inspect.getclosurevars(loss_fn).nonlocals
        write_json(out/'training_diagnostics.json', {'counterfactual_cache_stats': closure['cc_stats'],
            'cached_bytes': closure['cc_bytes'], 'sampled_unique_this_attempt': len(set(data.sampled)),
            'presentations_this_attempt': len(data.sampled), 'logged_updates_total': len(result.train_log),
            'model_parameters_with_grad': sum(p.numel() for p in meter.model.parameters() if p.grad is not None)})
        # Correctness work follows training, so no warmed donor cache changes its cost.
        with meter.stage('post_training_patching_validation'):
            check_patching(loss_fn, data, meter.model, out)
        return result

    def evaluate(*a, **kw):
        if kw.get('level') != 'edge' or kw.get('absolute') is not False:
            raise ValueError('Expected official signed edge-level evaluation.')
        model, graph = a[:2]
        rows = [{'edge': e.name, 'score': float(graph.scores[e.matrix_index])} for e in graph.edges.values()]
        pd.DataFrame(rows).sort_values('score', ascending=False).to_csv(out/'ranking.csv', index=False)
        folder = out/'raw_validation'
        folder.mkdir(exist_ok=True)
        cache_identity = {'ranking_sha256': sha(out/'ranking.csv'), 'config_identity': identity,
            'validation_fingerprint': data.datasets['validation'].dataset._fingerprint,
            'validation_size': len(data.datasets['validation'])}
        manifest_path = folder/'manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['identity'] != cache_identity:
                raise ValueError('Cached validation identity mismatch.')
        else:
            manifest = {'identity': cache_identity, 'vectors': {}}
        original_baseline = evaluation.evaluate_baseline
        original_graph = evaluation.evaluate_graph
        graph_calls = 0
        def vector(name, fn, *aa, **kk):
            path = folder/(name+'.npy')
            cached = manifest['vectors'].get(name)
            if cached and path.exists() and sha(path) == cached['sha256']:
                values = np.load(path, allow_pickle=False)
                print(f'Reused completed official validation vector: {name}; no forwards.', flush=True)
            else:
                result = fn(*aa, **kk)
                values = result.detach().cpu().numpy()
                if values.shape != (len(data.datasets['validation']),):
                    raise ValueError(f'Unexpected official validation vector shape {values.shape}.')
                np.save(path, values, allow_pickle=False)
                manifest['vectors'][name] = {'sha256': sha(path), 'examples': len(values)}
                write_json(manifest_path, manifest)
                print(f'Saved official validation vector: {name}; {len(values)} examples.', flush=True)
            return torch.from_numpy(values)
        def baseline(*aa, **kk):
            return vector('clean_reference', original_baseline, *aa, **kk)
        def graph_evaluation(*aa, **kk):
            nonlocal graph_calls
            name = 'corrupted_reference' if graph_calls == 0 else f'grid_{graph_calls-1:02d}'
            graph_calls += 1
            return vector(name, original_graph, *aa, **kk)
        # Empty reference cache changes no forwards, and exposes the official normalization.
        refs = {}
        kw['refs'] = refs
        with meter.stage('official_validation_evaluation'), torch.no_grad(), \
             patch.object(evaluation, 'evaluate_baseline', baseline), \
             patch.object(evaluation, 'evaluate_graph', graph_evaluation):
            result = original_eval(*a, **kw)
        if graph_calls != 11:
            raise ValueError('Unexpected official evaluation call sequence; inspect before using cached vectors.')
        pd.DataFrame({'fraction': GRID, 'requested_edges': [int(x*len(rows)) for x in GRID],
            'weighted_edge_count': result[0], 'faithfulness': result[4],
            'accuracy': result[5], 'linear_cpr_weight': integration_weights()}).to_csv(out/'faithfulness_curve.csv', index=False)
        write_json(out/'official_metrics.json', {'cpr': result[1], 'cmd': result[2],
            'mean_faithfulness': result[3], 'accuracy_auc': result[6],
            'validation_examples': len(data.datasets['validation']), 'official_references': refs,
            'scope': 'Gate 1 uncapped official 5000-step edge baseline'})
        return result

    def backward(tensor, *a, **kw):
        result = original_backward(tensor, *a, **kw)
        meter.counts['backward_calls'] += 1
        return result

    argv = ['eval_mib_edge.py', '--mib-path', str(MIB), '--model', 'gpt2', '--task', 'ioi',
        '--steps', '5000', '--k-schedule', 'uniform', '--masking', 'topk', '--mode', 'iso',
        '--lr', '0.05', '--optimizer', 'adam', '--split', 'validation', '--train-split', 'train',
        '--batch-size', str(args.batch_size), '--seed', '42', '--cf-cache-gb', str(args.cf_cache_gb),
        '--no-wandb', '--output', str(out/'upstream')]
    write_json(out/'upstream_command.json', {'argv': argv, 'uncapped': True, 'official_row_order': True,
        'evaluation_no_grad': True, 'upstream_weight_gradients_preserved': True})
    spec = importlib.util.spec_from_file_location('official_edge_gate1', UPSTREAM/'scripts/mib/eval_mib_edge.py')
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
        write_json(out/'gate1_passed.json', {'passed': True, 'iterations': 5000,
            'train_examples': len(data.datasets['train']), 'validation_examples': len(data.datasets['validation']),
            'edge_count': 32491, 'previous_artifacts_unchanged': True,
            'gate2_executed': False, 'test_split_accessed': False})
        success = True
    except BaseException as error:
        write_json(out/'failure.json', {'type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'settings_unchanged': True,
            'oom': isinstance(error, torch.cuda.OutOfMemoryError)})
        if isinstance(error, torch.cuda.OutOfMemoryError):
            raise RuntimeError('CUDA OOM: no settings changed. The saved scores/checkpoints remain available; '
                'any smaller evaluation batch or donor cache must be explicitly documented.') from error
        raise
    finally:
        meter.status(success=success)
        meter.stop.set()
        meter.worker.join()
        meter.save()
        meter.stream.close()
        write_json(out/'process_summary.json', {'success': success, 'counts_this_attempt': meter.counts,
            'process_seconds': time.perf_counter()-meter.started})
    from .gate1_report import generate
    generate(out)
    print('Gate 1 complete. Gate 2 and all algorithm sweeps remain unrun:', out, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--cf-cache-gb', type=float, default=4.)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.cf_cache_gb < 0:
        parser.error('Invalid explicit resource settings.')
    if args.report_only:
        from .gate1_report import generate
        generate(args.out)
    else:
        run(args)


if __name__ == '__main__':
    main()
