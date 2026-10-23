"""Read-only metric recording around the pinned official MIB evaluator."""
from functools import partial
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from src.mattr_optimization.common import ROOT, configure, provenance, sha, write_json
from src.mattr_optimization.gate1 import LiveAccounting
from src.mattr_optimization.gate4_init import edge_mapping
from src.mattr_hierarchy.runner import weight_hash
from .design import PRIOR, validate_protocol


class Recorder:
    """Return exactly official logit_diff; copy only O(batch) outputs, never logits."""
    def __init__(self, model):
        from MIB_circuit_track.metrics import get_metric
        self.margin = partial(get_metric('logit_diff', 'ioi', model.tokenizer, model), mean=False, loss=False)
        self.top1 = partial(get_metric('acc', 'ioi', model.tokenizer, model), mean=False, loss=False)
        self.margins = []
        self.accuracies = []

    def __call__(self, *args, **kwargs):
        assert not torch.is_grad_enabled()
        m = self.margin(*args, **kwargs)
        a = self.top1(*args, **kwargs)
        self.margins.append(m.detach().cpu().numpy().copy())
        self.accuracies.append(a.detach().cpu().numpy().copy())
        return m

    def arrays(self, size):
        margins = np.concatenate(self.margins).reshape(-1, size)
        top1 = np.concatenate(self.accuracies).reshape(-1, size)
        assert np.isfinite(margins).all() and np.isfinite(top1).all()
        return margins, top1


def assign_scores(graph, scores):
    assert scores.shape == (32491,) and torch.isfinite(scores).all()
    graph.scores[:] = -torch.inf
    graph.scores[graph.real_edge_mask.bool()] = scores.cpu()


def circuit_audit(graph, scores, grid):
    assign_scores(graph, scores)
    previous = None
    rows = []
    for fraction in grid:
        count = int(fraction * len(graph.edges))
        graph.apply_topn(count, absolute=False, level='edge', prune=False)
        unpruned = graph.in_graph[graph.real_edge_mask.bool()].clone()
        assert int(unpruned.sum()) == count
        if previous is not None:
            assert not (previous & ~unpruned).any(), 'Ranking not nested before pruning'
        previous = unpruned
        graph.prune()
        pruned = graph.in_graph[graph.real_edge_mask.bool()].clone()
        assert not (pruned & ~unpruned).any()
        packed = np.packbits(pruned.numpy())
        import hashlib
        rows.append({'fraction': fraction, 'requested_edges': count, 'actual_edges': int(pruned.sum()),
                     'weighted_edges': float(graph.weighted_edge_count()),
                     'mask_sha256': hashlib.sha256(packed.tobytes()).hexdigest(),
                     'nested_before_pruning': True})
    return rows


def correctness(model, graph, ds, p, out, meter):
    import copy
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    small = copy.copy(ds)
    small.dataset = ds.dataset.select(range(2))
    scores = torch.load(ROOT / p['rankings'][0]['scores'], map_location='cpu', weights_only=False)
    rows = circuit_audit(graph, scores, p['grid'])
    records = []
    arrays = []
    # Same official patcher/metric with batch2, repeated batch2, and singleton batching.
    for batch in (2, 2, 1):
        assign_scores(graph, scores)
        recorder = Recorder(model)
        with meter.stage(f'correctness_batch{batch}_{len(arrays)}'), torch.no_grad():
            result = evaluate_area_under_curve(model, graph, small.to_dataloader(batch), recorder,
                        level='edge', absolute=False, quiet=True, refs={}, percentages=(.005, .01, 1.))
        margin, top1 = recorder.arrays(2)
        assert np.allclose(margin[-1], margin[0], atol=1e-4, rtol=2e-5), 'Full circuit != clean reference'
        assert np.array_equal(top1[-1], top1[0])
        arrays.append((margin, top1))
        if len(arrays) > 1:
            err = float(np.max(np.abs(arrays[0][0] - margin)))
            assert np.allclose(arrays[0][0], margin, atol=1e-4, rtol=2e-5)
            assert np.array_equal(arrays[0][1], top1)
            records.append({'batch_size': batch, 'max_margin_error': err, 'top1_equal': True, 'passed': True})
    # Official reference-cache reuse must preserve point estimates/per-example values.
    refs = {'baseline': float(torch.tensor(arrays[0][0][0]).mean()),
            'corrupted': float(torch.tensor(arrays[0][0][1]).mean())}
    assign_scores(graph, scores)
    recorder = Recorder(model)
    with meter.stage('correctness_reference_cache'), torch.no_grad():
        evaluate_area_under_curve(model, graph, small.to_dataloader(2), recorder, level='edge',
                                  absolute=False, quiet=True, refs=refs, percentages=(.005, .01, 1.))
    cached = recorder.arrays(2)
    np.testing.assert_array_equal(cached[0], arrays[0][0][2:])
    np.testing.assert_array_equal(cached[1], arrays[0][1][2:])
    pd.DataFrame(records).to_csv(out / 'correctness_checks.csv', index=False)
    write_json(out / 'correctness_passed.json', {'passed': True, 'official_patcher_unchanged': True,
               'batch_repeat_singleton': True, 'reference_cache_equivalent': True,
               'full_returns_clean': True, 'nested_masks': True, 'mask_audit_points': len(rows),
               'optimizer_updates': 0, 'backward_calls': 0})


def run(out):
    configure()
    from transformer_lens import HookedTransformer
    from transformers import AutoTokenizer
    from eap.graph import Graph
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    p = validate_protocol(out)
    if (out / 'inference_complete.json').exists():
        raise ValueError('Completed inference directory; use report-only, never overwrite results.')
    assert torch.cuda.is_available(), 'RTX CUDA required; no automatic fallback.'
    assert provenance() == p['source_provenance']
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meter = LiveAccounting(out)
    started = time.perf_counter()
    success = False
    cache_hits = 0
    try:
        with meter.stage('model_loading'), torch.no_grad():
            model = meter.attach(HookedTransformer.from_pretrained('gpt2-small'))
            for key, value in p['model_config'].items():
                if key.startswith('n_') or key == 'd_model':
                    assert getattr(model.cfg, key) == value
                else:
                    setattr(model.cfg, key, value)
            model.eval()
            assert model.cfg.dtype == torch.float32
            before = weight_hash(model)
            assert all(r['model_weights_sha256'] == before for r in p['rankings'])
        graph = Graph.from_model(model)
        pd.testing.assert_frame_equal(edge_mapping(graph, graph), pd.read_csv(out / 'edge_mapping.csv', keep_default_na=False))
        with meter.stage('cohort_loading'):
            ds = HFEAPDataset('mib-bench/ioi', AutoTokenizer.from_pretrained('gpt2'), split='validation', task='ioi', model_name='gpt2')
            manifest = pd.read_csv(out / 'cohort_manifest.csv', keep_default_na=False)
            ds.dataset = ds.dataset.select(manifest.official_filtered_index.astype(int).tolist())
            for index, row in manifest.iterrows():
                clean, donor, labels = ds[index]
                assert clean == row.clean_prompt and donor == row.corrupted_prompt
                assert labels == [row.correct_token, row.incorrect_token]
        correctness(model, graph, ds, p, out, meter)
        refs = {}
        all_counts = []
        n = len(ds)
        batches = int(np.ceil(n / p['batch_size']))
        print(f'Gate5D:15 frozen rankings x10 grid points x{n} prompts; batch{p["batch_size"]}; '
              f'at most {2 * batches * (150 + 2)} full-cohort forwards plus correctness; zero training.', flush=True)
        for number, rec in enumerate(p['rankings'], 1):
            key = f'{rec["method"]}_s{rec["seed"]}'
            dest = out / 'evaluations' / key
            dest.mkdir(parents=True, exist_ok=True)
            identity = {'protocol_sha256': sha(out / 'gate5d_protocol.json'), 'scores_sha256': rec['scores_sha256'],
                        'checkpoint_sha256': rec['checkpoint_sha256'], 'cohort_sha256': p['cohort']['manifest_sha256']}
            if (dest / 'official_result.json').exists():
                saved = json.loads((dest / 'official_result.json').read_text())
                assert saved['identity'] == identity
                assert sha(dest / 'per_prompt.csv') == saved['per_prompt_sha256']
                cache_hits += 1
                continue
            scores = torch.load(ROOT / rec['scores'], map_location='cpu', weights_only=False)
            mask_rows = circuit_audit(graph, scores, p['grid'])
            assign_scores(graph, scores)
            recorder = Recorder(model)
            had_refs = bool(refs)
            with meter.stage('evaluate_' + key), torch.no_grad():
                result = evaluate_area_under_curve(model, graph, ds.to_dataloader(p['batch_size']), recorder,
                           level='edge', absolute=False, intervention='patching', quiet=True, refs=refs)
            margin, top1 = recorder.arrays(n)
            if had_refs:
                cache_hits += 2  # graph-independent reference values, not model forwards
                assert margin.shape == (10, n)
            else:
                assert margin.shape == (12, n)
                np.savez_compressed(out / 'reference_per_prompt.npz', clean_margin=margin[0], empty_margin=margin[1],
                                    clean_top1=top1[0], empty_top1=top1[1])
                write_json(out / 'reference_scores.json', refs)
                margin, top1 = margin[2:], top1[2:]
            assert abs(refs['baseline'] - refs['corrupted']) > 1e-8
            point_rows = []
            for j, fraction in enumerate(p['grid']):
                acc = (margin[j] > 0).mean()
                # All point estimates and areas remain the official evaluator's return.
                assert np.isclose(acc, result[5][j], atol=1e-7)
                assert np.isclose(mask_rows[j]['weighted_edges'], result[0][j])
                point_rows.append({**mask_rows[j], 'method': rec['method'], 'seed': rec['seed'],
                                   'faithfulness': result[4][j], 'accuracy': result[5][j],
                                   'margin': float(margin[j].mean()), 'top1_accuracy': float(top1[j].mean())})
            pd.DataFrame(point_rows).to_csv(dest / 'curve.csv', index=False)
            prompt_rows = []
            for j, fraction in enumerate(p['grid']):
                for i in range(n):
                    prompt_rows.append({'example': i, 'template_id': manifest.template_id.iloc[i],
                                        'fraction': fraction, 'margin': float(margin[j, i]),
                                        'correct': int(margin[j, i] > 0), 'top1': int(top1[j, i])})
            pd.DataFrame(prompt_rows).to_csv(dest / 'per_prompt.csv', index=False)
            write_json(dest / 'official_result.json', {'identity': identity, 'result': result,
                       'per_prompt_sha256': sha(dest / 'per_prompt.csv'), 'method': rec['method'], 'seed': rec['seed']})
            all_counts.extend(point_rows)
            meter.status(rankings_completed=number, rankings_total=15, method=rec['method'], seed=rec['seed'], cache_hits=cache_hits)
            print(f'{number}/15 {key}: CPR={result[1]:.6f} CMD={result[2]:.6f} accAUC={result[6]:.6f}', flush=True)
            if time.perf_counter() - started > p['budget']['evaluation_seconds_max'] or meter.counts['forward_batches'] > p['budget']['actual_model_forward_calls_max']:
                raise RuntimeError('Frozen compute limit reached; no expansion.')
        with torch.no_grad():
            after = weight_hash(model)
        assert before == after
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(not event['grad_enabled'] and event['success'] for event in meter.events)
        write_json(out / 'model_weight_audit.json', {'passed': True, 'before_sha256': before, 'after_sha256': after,
                   'optimizer_updates': 0, 'model_backwards': 0, 'stored_model_gradients': False})
        success = True
    except BaseException as exc:
        import traceback
        write_json(out / 'inference_failure.json', {'error': str(exc), 'traceback': traceback.format_exc(),
                   'OOM': isinstance(exc, torch.cuda.OutOfMemoryError),
                   'advice': 'Stop. Batch size and evaluator unchanged. A smaller batch requires a documented resource amendment before rerun.'})
        raise
    finally:
        meter.stop.set()
        meter.worker.join()
        meter.save()
        meter.stream.close()
        write_json(out / 'inference_process_summary.json', {'success': success, 'seconds': time.perf_counter() - started,
                   'counts': meter.counts, 'reference_or_result_cache_hits': cache_hits,
                   'donor_activation_cache_hits': 0, 'activation_cache': 'disabled; official immutable reference scalar cache enabled',
                   'unique_evaluation_prompts': p['cohort']['size'], 'fixed_rankings': 15})
    write_json(out / 'inference_complete.json', {'passed': True, 'rankings': 15, 'training_updates': 0})
