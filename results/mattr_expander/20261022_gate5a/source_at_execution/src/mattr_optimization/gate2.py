"""User-launched discovery-data scaling; unchanged official edge MAttr only.

Parent orchestration is CPU-only. Each pool/seed runs in a fresh subprocess.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import os

from .common import ROOT, sha, write_json


def subset_indices(length, size, seed):
    import numpy as np
    if not 1 <= size <= length:
        raise ValueError(f'Cannot select {size} from {length} official rows.')
    return np.random.default_rng(seed).permutation(length)[:size].tolist()


class NestedData:
    def __init__(self, out, pool, seed, validation_size):
        self.out, self.pool, self.seed = Path(out), pool, seed
        self.validation_size = validation_size
        self.datasets = {}

    def select(self, dataset, split):
        import pandas as pd
        if split not in ('train', 'validation'):
            raise ValueError('Gate 2 may load only official train and validation.')
        size, seed = (self.pool, self.seed) if split == 'train' else (self.validation_size, 42)
        indices = subset_indices(len(dataset), size, seed)
        fingerprint = dataset.dataset._fingerprint
        dataset.dataset = dataset.dataset.select(indices)
        records = []
        for i, source in enumerate(indices):
            clean, donor, labels = dataset[i]
            meta = dataset.dataset[i]['metadata']
            names = [meta['indirect_object'], meta['subject'], meta['random_c']]
            ids = [dataset.tokenizer(' '+n, add_special_tokens=False).input_ids for n in names]
            if any(len(x) != 1 for x in ids) or [x[0] for x in ids[:2]] != list(labels):
                raise ValueError('Official IOI labels must be single tokens for this metric.')
            records.append({'split': split, 'subset_index': i, 'official_filtered_index': source,
                'clean_prompt': clean, 'corrupted_prompt': donor, 'correct_token': labels[0],
                'incorrect_token': labels[1], 'metadata': json.dumps(meta, sort_keys=True)})
        frame = pd.DataFrame(records)
        path = self.out/f'{split}_manifest.csv'
        meta_path = self.out/f'{split}_selection.json'
        selection = {'split': split, 'size': size, 'seed': seed, 'indices': indices,
            'official_fingerprint': fingerprint, 'subset_fingerprint': dataset.dataset._fingerprint}
        # Do not overwrite a previous manifest until data identity has been checked.
        if meta_path.exists() and json.loads(meta_path.read_text()) != selection:
            raise ValueError('Official dataset identity/subset changed since checkpoint.')
        if path.exists():
            pd.testing.assert_frame_equal(pd.read_csv(path), frame, check_dtype=False)
        else:
            frame.to_csv(path, index=False)
        write_json(meta_path, selection)
        self.datasets[split] = dataset
        if len(self.datasets) == 2:
            sets = [set(d.dataset['prompt']) for d in self.datasets.values()]
            if sets[0] & sets[1]:
                raise ValueError('Selected train and validation prompts overlap.')


def gate1_evidence(explicit=None):
    paths = [Path(explicit)] if explicit else sorted(
        (ROOT/'results/mattr_optimization').glob('*/gate1_passed.json'), reverse=True)
    for path in paths:
        if path.exists():
            marker = json.loads(path.read_text())
            if marker.get('passed') and marker.get('iterations') == 5000 and marker.get('edge_count') == 32491:
                audit = path.parent/'final_validation.json'
                if audit.exists() and json.loads(audit.read_text()).get('passed'):
                    return {'marker': str(path.resolve()), 'marker_sha256': sha(path), 'audit_sha256': sha(audit)}
    raise ValueError('An audited completed Gate 1 campaign is required (or pass --gate1-evidence).')


def protocol_for(args):
    return {'gate': 2, 'model': 'gpt2', 'task': 'ioi', 'granularity': 'edge',
        'sizes': args.sizes, 'seeds': args.seeds, 'steps': 5000,
        'validation_size': args.validation_size, 'validation_seed': 42,
        'checkpoint_evaluations': [500, 1000, 5000], 'batch_size': args.batch_size,
        'cf_cache_gb': 4., 'gate1_evidence': gate1_evidence(args.gate1_evidence),
        'adapter_sha256': {n: sha(ROOT/'src/mattr_optimization'/n) for n in
            ('gate2.py', 'common.py', 'checkpoint.py', 'gate0.py', 'gate1.py')}}


def archive_attempt(out):
    import shutil
    if (out/'process_summary.json').exists() or (out/'forward_events.jsonl').exists():
        folder = out/'attempts'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        folder.mkdir(parents=True)
        for name in ('process_summary.json', 'compute_accounting.csv', 'forward_events.csv',
                     'forward_events.jsonl', 'live_status.json', 'training_example_draws.csv',
                     'gradient_norms.csv', 'failure.json'):
            if (out/name).exists():
                shutil.move(str(out/name), str(folder/name))


def run_job(args):
    import importlib.util
    import inspect
    import random
    import time
    import traceback
    from unittest.mock import patch
    import pandas as pd
    import torch
    from .common import configure, provenance, UPSTREAM, MIB, GRID, integration_weights
    from .gate1 import LiveAccounting
    from .checkpoint import checkpointed
    configure()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback and no automatic OOM changes.')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())
    config = {'campaign_protocol': protocol, 'seed': args.seed, 'pool': args.pool,
              'source_provenance': provenance()}
    config_path = out/'experiment_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Job configuration/source changed; use a new campaign.')
    archive_attempt(out)
    write_json(config_path, config)
    identity = json.dumps(config, sort_keys=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from transformer_lens import HookedTransformer
    from MIB_circuit_track.dataset import HFEAPDataset
    import MIB_circuit_track.evaluation as evaluation
    import matryoshka_attribution as package
    original_load, original_data = HookedTransformer.from_pretrained, HFEAPDataset.__init__
    original_train, original_eval = package.learn_scores, evaluation.evaluate_area_under_curve
    original_backward = torch.Tensor.backward
    data = NestedData(out, args.pool, args.seed, protocol['validation_size'])
    meter = LiveAccounting(out)
    refs_by_split = {}
    curves = []

    def load(*a, **kw):
        with meter.stage('model_loading'):
            model = original_load(*a, **kw)
            if model.cfg.dtype != torch.float32:
                raise ValueError('Expected official GPT-2 float32 recipe.')
            return meter.attach(model)

    def dataset_init(dataset, *a, **kw):
        with meter.stage('data_loading_'+kw['split']):
            original_data(dataset, *a, **kw)
            data.select(dataset, kw['split'])
        print(f'Gate 2: seed={args.seed}, pool={args.pool}, {kw["split"]}={len(dataset)}', flush=True)

    def train(total, loss_fn, **kw):
        if total != 32491 or kw['steps'] != 5000:
            raise ValueError('Gate 2 must preserve all edge scores and the 5000-iteration budget.')
        original_draw = random.randint
        draw_path = out/'training_example_draws.csv'
        with draw_path.open('w', buffering=1) as draw_file, (out/'gradient_norms.csv').open('w', buffering=1) as grad_file:
            draw_file.write('presentation_this_attempt,subset_index\n')
            grad_file.write('update_this_attempt,gradient_norm\n')
            presentations = 0
            def sampled_loss(mask):
                def draw(a, b):
                    nonlocal presentations
                    i = original_draw(a, b)
                    presentations += 1
                    draw_file.write(f'{presentations},{i}\n')
                    return i
                with patch.object(random, 'randint', draw):
                    return loss_fn(mask)
            def update(norm):
                meter.update(norm)
                grad_file.write(f'{len(meter.grad_norms)},{norm}\n')
            external = kw.pop('on_step', None)
            def completed(step, k, loss, scores):
                if external:
                    external(step, k, loss, scores)
                if (step+1) % 100 == 0 or step == 0:
                    meter.status(completed_iteration=step+1, seed=args.seed, pool=args.pool)
                    print(f'seed={args.seed}, N={args.pool}: {step+1}/5000; forwards={meter.counts["forward_batches"]}', flush=True)
            with meter.stage('training'):
                result = checkpointed(original_train, total, sampled_loss, folder=out/'checkpoints',
                    identity=identity, interval=100, on_update=update, on_step=completed, **kw)
        if len(result.train_log) != 5000:
            raise ValueError('Official loop skipped iterations; inspect before claiming 5000 updates.')
        pd.DataFrame(result.train_log, columns=['step', 'k', 'retained_fraction', 'loss']).to_csv(out/'training_metrics.csv', index=False)
        torch.save(result.scores, out/'final_scores.pt')
        closure = inspect.getclosurevars(loss_fn).nonlocals
        write_json(out/'training_diagnostics.json', {'cache_stats': closure['cc_stats'],
            'cache_bytes': closure['cc_bytes'], 'presentations_this_attempt': presentations})
        return result

    def curve(model, graph, loader, metric, split, step):
        folder = out/f'evaluation/{split}_step{step}'
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = out/f'checkpoints/step_{step:06d}.pt'
        graph.scores[:] = -torch.inf
        scores = torch.load(checkpoint, map_location='cpu', weights_only=False)['scores']
        graph.scores[graph.real_edge_mask.bool()] = scores
        cache_id = {'config': identity, 'checkpoint_sha256': sha(checkpoint),
                    'data_manifest_sha256': sha(out/f'{split}_manifest.csv')}
        target = folder/'official_result.json'
        if target.exists():
            saved = json.loads(target.read_text())
            if saved['identity'] != cache_id:
                raise ValueError('Evaluation-cache identity differs; refusing to reuse it.')
            result = tuple(saved['result'])
            refs_by_split.setdefault(split, {}).update(saved['references'])
            with (out/'evaluation_cache_hits.jsonl').open('a') as log:
                log.write(json.dumps({'split': split, 'step': step, 'model_forwards': 0})+'\n')
        else:
            refs = refs_by_split.setdefault(split, {})
            with meter.stage(f'{split}_evaluation_step{step}'), torch.no_grad():
                result = original_eval(model, graph, loader, metric, level='edge', absolute=False, quiet=True, refs=refs)
            write_json(target, {'identity': cache_id, 'result': result, 'references': refs})
        pd.DataFrame({'fraction': GRID, 'requested_edges': [int(p*32491) for p in GRID],
            'weighted_edge_count': result[0], 'faithfulness': result[4], 'accuracy': result[5],
            'linear_cpr_weight': integration_weights()}).to_csv(folder/'faithfulness_curve.csv', index=False)
        curves.append({'seed': args.seed, 'pool': args.pool, 'split': split, 'step': step,
            'cpr': result[1], 'cmd': result[2], 'accuracy_auc': result[6]})
        pd.DataFrame(curves).to_csv(out/'quality.csv', index=False)
        return result

    def evaluate(model, graph, loader, metric, **kw):
        if kw.get('level') != 'edge' or kw.get('absolute') is not False:
            raise ValueError('Signed official edge evaluation required.')
        for step in (500, 1000, 5000):
            result = curve(model, graph, loader, metric, 'validation', step)
        # Training quality is diagnostic only; it never updates scores or chooses swaps.
        curve(model, graph, data.datasets['train'].to_dataloader(protocol['batch_size']), metric, 'train', 5000)
        rows = [{'edge': e.name, 'score': float(graph.scores[e.matrix_index])} for e in graph.edges.values()]
        pd.DataFrame(rows).sort_values('score', ascending=False).to_csv(out/'ranking.csv', index=False)
        return result  # Upstream CLI saves the final validation result, not training metrics.

    def backward(tensor, *a, **kw):
        result = original_backward(tensor, *a, **kw)
        meter.counts['backward_calls'] += 1
        return result

    argv = ['eval_mib_edge.py', '--mib-path', str(MIB), '--model', 'gpt2', '--task', 'ioi',
        '--steps', '5000', '--k-schedule', 'uniform', '--masking', 'topk', '--mode', 'iso',
        '--lr', '.05', '--optimizer', 'adam', '--split', 'validation', '--train-split', 'train',
        '--batch-size', str(protocol['batch_size']), '--seed', str(args.seed), '--cf-cache-gb', '4',
        '--no-wandb', '--output', str(out/'upstream')]
    write_json(out/'upstream_command.json', {'argv': argv, 'unchanged_algorithm': True})
    spec = importlib.util.spec_from_file_location('official_edge_gate2', UPSTREAM/'scripts/mib/eval_mib_edge.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    success = False
    try:
        with patch.object(HookedTransformer, 'from_pretrained', side_effect=load), \
             patch.object(HFEAPDataset, '__init__', dataset_init), \
             patch.object(package, 'learn_scores', train), \
             patch.object(evaluation, 'evaluate_area_under_curve', evaluate), \
             patch.object(torch.Tensor, 'backward', backward), patch.object(sys, 'argv', argv):
            module.main()
        success = True
    except BaseException as error:
        write_json(out/'failure.json', {'type': type(error).__name__, 'error': str(error),
            'settings_unchanged': True, 'oom': isinstance(error, torch.cuda.OutOfMemoryError),
            'traceback': traceback.format_exc()})
        raise
    finally:
        meter.status(success=success)
        meter.stop.set(); meter.worker.join(); meter.save(); meter.stream.close()
        write_json(out/'process_summary.json', {'success': success, 'counts': meter.counts,
            'process_seconds': time.perf_counter()-meter.started})
    names = ['experiment_config.json', 'final_scores.pt', 'ranking.csv', 'quality.csv',
             'train_manifest.csv', 'validation_manifest.csv', 'training_metrics.csv']
    write_json(out/'completed.json', {'passed': True, 'sha256': {n: sha(out/n) for n in names}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[16, 32, 64, 128])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--validation-size', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--gate1-evidence', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--job', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--pool', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--protocol', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.job:
        run_job(args)
        return
    if args.report_only:
        from .gate2_report import generate
        generate(args.out)
        return
    if (sorted(set(args.sizes)) != args.sizes or any(n not in (16,32,64,128) for n in args.sizes)
        or len(set(args.seeds)) != len(args.seeds) or any(s not in (42,43,44) for s in args.seeds)
        or args.validation_size < 1 or args.batch_size < 1):
        parser.error('Use unique increasing sizes from 16/32/64/128 and unique seeds from 42/43/44; positive resources.')
    protocol = protocol_for(args)
    jobs = [(seed, n) for seed in args.seeds for n in args.sizes]
    print(f'Gate 2: {len(jobs)} jobs x 5000 updates; N={args.sizes}, seeds={args.seeds}; '
          f'fixed validation={args.validation_size}, seed=42; checkpoints 500/1000/5000.', flush=True)
    if args.dry_run:
        print(json.dumps(protocol, indent=2))
        print('Dry run only: no output directory, model loading or training.')
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (out/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = out/'campaign_protocol.json'
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError('Campaign protocol changed. Reuse identical arguments or start a new directory.')
        write_json(path, protocol)
        for seed, n in jobs:
            folder = out/f'seed{seed}_n{n}'
            marker = folder/'completed.json'
            if marker.exists():
                saved = json.loads(marker.read_text())
                if not saved['passed'] or any(sha(folder/name) != h for name,h in saved['sha256'].items()):
                    raise ValueError('Completed artifacts changed; refusing to overwrite them.')
                print(f'Reused completed seed={seed}, N={n}; zero model forwards.', flush=True)
                continue
            folder.mkdir(exist_ok=True)
            command = [sys.executable, '-u', '-m', 'src.mattr_optimization.gate2', '--job',
                '--out', str(folder), '--pool', str(n), '--seed', str(seed), '--protocol', str(path)]
            with (folder/'console.log').open('a', buffering=1) as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                for line in child.stdout:
                    log.write(line)
                    print(line, end='', flush=True)
                if child.wait() != 0:
                    raise RuntimeError(f'Job seed={seed}, N={n} failed. Settings unchanged; inspect {folder}/failure.json.')
            from .gate2_report import generate
            generate(out)
        write_json(out/'gate2_passed.json', {'passed': True, 'jobs': len(jobs), 'seeds': args.seeds,
            'scope': 'Only the configured discovery-data scaling runs; no algorithm variants.', 'gate3_executed': False})
        print('Configured Gate 2 campaign complete. Gate 3 is not dispatched.', flush=True)


if __name__ == '__main__':
    main()
