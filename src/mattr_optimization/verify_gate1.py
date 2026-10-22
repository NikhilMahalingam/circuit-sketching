"""Audit completed Gate-1 artifacts without model loading or extra forwards."""
import argparse
import csv
import json
import random
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch

from .common import ROOT, GRID, configure, sha, write_json


def verify(out):
    configure()
    out = Path(out)
    config = json.loads((out/'experiment_config.json').read_text())
    marker = json.loads((out/'gate1_passed.json').read_text())
    assert config['gate'] == 1 and config['steps'] == 5000 and marker['passed']
    assert marker['gate2_executed'] is False and marker['test_split_accessed'] is False
    assert config['train_subset_cap'] is None and config['validation_subset_cap'] is None
    assert config['preserve_official_row_order'] is True
    for name, h in config['adapter_sha256'].items():
        assert sha(out/'implementation_at_execution'/name) == h
        assert sha(ROOT/'src/mattr_optimization'/name) == h, f'Executed core source later changed: {name}'
    for name, h in config['source_provenance']['source_sha256'].items():
        assert sha(ROOT/name) == h, name
    protected = json.loads((out/'protected_previous_artifacts.json').read_text())
    # The root baseline report is intentionally updated after the protection check.
    intentionally_updated = {'MATTR_BASELINE_REPORT.md'}
    for name, h in protected.items():
        if name not in intentionally_updated:
            assert sha(ROOT/name) == h, name
    train_manifest = pd.read_csv(out/'train_manifest.csv')
    val_manifest = pd.read_csv(out/'validation_manifest.csv')
    assert len(train_manifest) == marker['train_examples']
    assert len(val_manifest) == marker['validation_examples']
    assert train_manifest.official_filtered_index.tolist() == list(range(len(train_manifest)))
    assert val_manifest.official_filtered_index.tolist() == list(range(len(val_manifest)))
    assert not set(train_manifest.clean_prompt) & set(val_manifest.clean_prompt)
    train = pd.read_csv(out/'training_metrics.csv')
    assert train.step.tolist() == list(range(5000)), 'Skipped/duplicate iterations; inspect before claiming 5000 updates.'
    assert np.isfinite(train[['k', 'loss']].to_numpy()).all()
    assert ((train.k >= 1) & (train.k <= 32491)).all()
    scores = torch.load(out/'final_scores.pt', map_location='cpu', weights_only=True)
    checkpoint = torch.load(out/'checkpoints/step_005000.pt', map_location='cpu', weights_only=False)
    upstream = torch.load(out/'upstream/ioi_gpt2_scores.pt', map_location='cpu', weights_only=False)
    assert scores.shape == (32491,) and torch.isfinite(scores).all()
    assert torch.equal(scores, checkpoint['scores']) and torch.equal(scores, upstream['scores'])
    assert checkpoint['completed'] == 5000
    assert int(next(iter(checkpoint['optimizer']['state'].values()))['step']) == 5000
    rank = pd.read_csv(out/'ranking.csv')
    assert len(rank) == 32491 and rank.edge.nunique() == 32491
    assert rank.score.is_monotonic_decreasing
    # Reconstruct only the official graph from the cached pretrained configuration.
    from transformer_lens.loading_from_pretrained import get_pretrained_model_config
    from eap.graph import Graph
    cfg = get_pretrained_model_config('gpt2-small', device='cpu')
    graph = Graph.from_model(cfg)
    graph.scores[graph.real_edge_mask.bool()] = scores
    expected = {e.name: float(graph.scores[e.matrix_index]) for e in graph.edges.values()}
    assert set(rank.edge) == set(expected)
    assert all(np.isclose(row.score, expected[row.edge], rtol=1e-12, atol=1e-12) for row in rank.itertuples())
    masked_scores = graph.scores.clone()
    masked_scores[~graph.real_edge_mask] = -torch.inf
    sorted_flat = torch.argsort(masked_scores.view(-1), descending=True)[:32491].tolist()
    # Official logits receivers use backward index -1: normalize valid Python
    # negative indices before translating the matrix position into a flat index.
    flat_names = {(e.matrix_index[0] % graph.scores.shape[0])*graph.scores.shape[1]
                  +(e.matrix_index[1] % graph.scores.shape[1]): e.name for e in graph.edges.values()}
    assert len(flat_names) == 32491
    official_order = [flat_names[i] for i in sorted_flat]
    pd.DataFrame({'rank': range(1, 32492), 'edge': official_order,
        'score': [expected[name] for name in official_order],
        'flat_matrix_index': sorted_flat}).to_csv(out/'official_nested_ranking.csv', index=False)
    curve = pd.read_csv(out/'faithfulness_curve.csv')
    metrics = json.loads((out/'official_metrics.json').read_text())
    assert curve.fraction.tolist() == list(GRID)
    assert curve.requested_edges.tolist() == [int(p*32491) for p in GRID]
    membership = []
    for row in curve.itertuples():
        graph.apply_topn(row.requested_edges, absolute=False, level='edge', prune=False)
        actual_flat = graph.in_graph.view(-1).nonzero().flatten().tolist()
        assert set(actual_flat) == set(sorted_flat[:row.requested_edges])
        graph.prune()
        retained = sorted(flat_names[i] for i in graph.in_graph.view(-1).nonzero().flatten().tolist())
        assert float(graph.weighted_edge_count()) == row.weighted_edge_count
        membership.append({'fraction': row.fraction, 'requested_edges': row.requested_edges,
            'retained_edges_after_pruning': len(retained), 'weighted_edge_count': row.weighted_edge_count,
            'retained_edge_sha256': hashlib.sha256(json.dumps(retained).encode()).hexdigest()})
    pd.DataFrame(membership).to_csv(out/'official_circuit_membership_summary.csv', index=False)
    manifest = json.loads((out/'raw_validation/manifest.json').read_text())
    assert manifest['identity']['ranking_sha256'] == sha(out/'ranking.csv')
    assert len(manifest['vectors']) == 12
    raw = {}
    for name, info in manifest['vectors'].items():
        file = out/'raw_validation'/(name+'.npy')
        assert sha(file) == info['sha256']
        raw[name] = torch.from_numpy(np.load(file, allow_pickle=False))
        assert raw[name].shape == (len(val_manifest),) and torch.isfinite(raw[name]).all()
    baseline = raw['clean_reference'].mean().item()
    corrupted = raw['corrupted_reference'].mean().item()
    for i, row in enumerate(curve.itertuples()):
        values = raw[f'grid_{i:02d}']
        faith = (values.mean().item()-corrupted)/(baseline-corrupted)
        assert np.isclose(faith, row.faithfulness, atol=1e-7, rtol=1e-7)
        assert np.isclose((values > 0).float().mean().item(), row.accuracy, atol=1e-7)
    assert np.isclose(np.trapezoid(curve.faithfulness, curve.fraction), metrics['cpr'], atol=1e-7)
    assert np.isclose(np.trapezoid(np.abs(1-curve.faithfulness), curve.fraction), metrics['cmd'], atol=1e-7)
    event_files = sorted((out/'attempts').glob('*/forward_events.jsonl'))+[out/'forward_events.jsonl']
    events = [json.loads(line) for path in event_files for line in path.read_text().splitlines() if line.strip()]
    assert all(e['success'] for e in events), 'Failures must remain accounted and investigated.'
    assert all(not e['grad_enabled'] for e in events if e['stage'] in
               ('official_validation_evaluation', 'post_training_patching_validation'))
    costs = pd.read_csv(out/'complete_compute_accounting.csv')
    assert int(costs.forward_batches.sum()) == len(events)
    assert int(costs.example_forwards.sum()) == sum(e['batch_size'] for e in events)
    assert int(costs.backward_calls.sum()) >= 5000 and int(costs.optimizer_updates.sum()) >= 5000
    draw_frames = [pd.read_csv(p) for p in sorted((out/'attempts').glob('*/training_example_draws.csv'))]
    draw_frames.append(pd.read_csv(out/'training_example_draws.csv'))
    draws = pd.concat(draw_frames, ignore_index=True)
    if len(draw_frames) == 1:
        expected_rng = random.Random(42)
        assert draws.official_filtered_index.tolist() == [expected_rng.randint(0, len(train_manifest)-1) for _ in range(5000)]
    result = {'passed': True, 'logged_updates': 5000, 'ranking_edges': 32491,
        'raw_vectors': 12, 'raw_examples_per_vector': len(val_manifest),
        'official_ranking_tie_order_exported': True,
        'presentation_csv_order_matches_official': rank.edge.tolist() == official_order,
        'forward_batches': len(events), 'example_forwards': sum(e['batch_size'] for e in events),
        'published_cpr_rounding_match': round(metrics['cpr'], 2) == 10.59,
        'official_cpr': metrics['cpr'], 'official_cmd': metrics['cmd'],
        'prior_artifacts_preserved': True, 'additional_model_forwards': 0}
    write_json(out/'final_validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    verify(parser.parse_args().out)
