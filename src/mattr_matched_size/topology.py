"""Only official Graph.apply_topn/prune determine circuit membership."""
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import ROOT, configure, sha, write_json
from src.mattr_compact_validation.inference import assign_scores

PREVIOUS = ROOT / 'results/mattr_compact_validation/20261023_gate5d'
MAX_PREFIX = 1624
N_EDGES = 32491


def digest(mask):
    return hashlib.sha256(np.packbits(np.asarray(mask, dtype=bool)).tobytes()).hexdigest()


def graph_new():
    from eap.graph import Graph
    return Graph.from_model(dict(n_layers=12, n_heads=12, parallel_attn_mlp=False, d_model=768))


def official_order(graph, scores):
    assign_scores(graph, scores)
    flat = graph.real_edge_mask.flatten().bool()
    indices = torch.full(flat.shape, -1, dtype=torch.long)
    indices[flat] = torch.arange(int(flat.sum()))
    order = indices[torch.argsort(graph.scores.flatten(), descending=True)]
    return order[order >= 0].numpy()


def canonical_ranks(scores):
    """Ordinal signed ranks, descending score, canonical-index tie break."""
    score = np.asarray(scores)
    order = np.lexsort((np.arange(len(score)), -score))
    ranks = np.empty(len(score), dtype=np.int64)
    ranks[order] = np.arange(1, len(score) + 1)
    return ranks


def score_order(order):
    order = np.asarray(order, dtype=np.int64)
    assert np.array_equal(np.sort(order), np.arange(len(order)))
    values = np.empty(len(order), dtype=np.float32)
    values[order] = -np.arange(1, len(order) + 1, dtype=np.float32)
    return torch.from_numpy(values)


def ensemble_scores(vectors):
    mean_rank = np.mean([canonical_ranks(x) for x in vectors], axis=0)
    order = np.lexsort((np.arange(len(mean_rank)), mean_rank))
    return score_order(order), mean_rank


def curve(graph, scores, maximum=MAX_PREFIX):
    assign_scores(graph, scores)
    n_edges = int(graph.real_edge_mask.sum())
    packed = np.zeros((maximum + 1, (n_edges + 7) // 8), dtype=np.uint8)
    counts = np.zeros(maximum + 1, dtype=np.int64)
    first = np.full(n_edges, maximum + 1, dtype=np.int64)
    previous = np.zeros(n_edges, dtype=bool)
    rows = []
    for n in range(1, maximum + 1):
        graph.apply_topn(n, absolute=False, level='edge', prune=True)
        current = graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()
        assert not np.any(previous & ~current), f'Official pruning is not nested at{n}'
        counts[n] = int(current.sum())
        assert counts[n] >= counts[n - 1], f'Official A(n) decreases at{n}'
        first[current & ~previous] = n
        packed[n] = np.packbits(current)
        rows.append({'nominal_n': n, 'actual_edges': int(counts[n]),
                     'pruned_edges': n - int(counts[n]), 'jump': int(counts[n] - counts[n - 1]),
                     'mask_sha256': digest(current)})
        previous = current
    return pd.DataFrame(rows), counts, packed, first


def unpack(packed, n, size=N_EDGES):
    return np.unpackbits(packed[n], count=size).astype(bool)


def capped_selection(counts, target):
    n = int(np.searchsorted(counts, target, side='right') - 1)
    if n == len(counts) - 1 and counts[n] < target:
        raise ValueError('Prefix audit range does not reach target; do not extrapolate.')
    return n, int(counts[n])


def upper_selection(counts, target):
    n = int(np.searchsorted(counts, target, side='left'))
    if n >= len(counts):
        raise ValueError('No audited upper-size circuit for interpolation.')
    return n, int(counts[n])


def closure_order(first, original_order, maximum=MAX_PREFIX):
    inverse = np.empty(len(first), dtype=np.int64)
    inverse[original_order] = np.arange(len(first))
    return np.lexsort((inverse, first)), inverse


def validate_closure(graph, scores, counts, packed, first, original_order):
    order, inverse = closure_order(first, original_order, len(counts) - 1)
    exported = score_order(order)
    assert np.array_equal(official_order(graph, exported), order)
    assert np.array_equal(closure_order(first, original_order, len(counts) - 1)[0], order)
    events = np.flatnonzero(np.diff(counts) > 0) + 1
    boundaries = counts[events]
    previous = np.zeros(len(order), dtype=bool)
    rows = []
    for m in range(1, int(counts[-1]) + 1):
        graph.apply_topn(m, absolute=False, level='edge', prune=True)
        actual = graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()
        assert not np.any(previous & ~actual), f'Closure nesting failure at{m}'
        next_idx = int(np.searchsorted(boundaries, m, side='left'))
        upper = unpack(packed, int(events[next_idx]), len(order))
        if m == boundaries[next_idx]:
            assert np.array_equal(actual, upper) and int(actual.sum()) == m, f'Closure boundary failure at{m}'
            boundary = True
        else:
            lower = unpack(packed, int(events[next_idx - 1]), len(order)) if next_idx else np.zeros(len(order), dtype=bool)
            assert not np.any(lower & ~actual) and not np.any(actual & ~upper), f'Closure sandwich failure at{m}'
            boundary = False
        rows.append({'closure_prefix': m, 'group_boundary': boundary, 'actual_edges': int(actual.sum()),
                     'boundary_or_sandwich_passed': True})
        previous = actual
    # Additional full-range samples test nesting only; appended tail is NOT claimed pruning-free.
    samples = sorted(set(np.linspace(max(1, int(counts[-1])), len(order), 50, dtype=int).tolist()))
    for m in samples:
        graph.apply_topn(int(m), absolute=False, level='edge', prune=True)
        actual = graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()
        assert not np.any(previous & ~actual), f'Closure tail sampled nesting failure at{m}'
        previous = actual
    assert len(rows) >= 50
    return exported, pd.DataFrame(rows), {'passed': True, 'all_boundaries_checked': len(events),
              'every_prefix_in_known_groups_checked': len(rows), 'additional_tail_nesting_samples': len(samples),
              'group_boundary_scope_max_original_prefix': len(counts) - 1,
              'known_closure_scope_max': int(counts[-1]), 'tail_pruning_free_claim': False}


def run(out):
    configure()
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'phaseA_complete.json').exists():
        return json.loads((out / 'phaseA_complete.json').read_text())
    start = time.perf_counter()
    p = json.loads((PREVIOUS / 'gate5d_protocol.json').read_text())
    assert sha(PREVIOUS / 'gate5d_protocol.json') == (PREVIOUS / 'gate5d_protocol.sha256').read_text().strip()
    records = []
    old_counts = pd.read_csv(PREVIOUS / 'faithfulness_curves.csv')
    t1 = int(old_counts[np.isclose(old_counts.fraction, .005)].actual_edges.median())
    t2 = int(old_counts[np.isclose(old_counts.fraction, .01)].actual_edges.median())
    targets = list(dict.fromkeys([60, 80, 100, t1, 140, 180, 240, t2, 400, 640]))
    inputs = {}
    for rec in p['rankings']:
        for field in ('checkpoint', 'scores', 'config'):
            assert sha(ROOT / rec[field]) == rec[field + '_sha256']
        state = torch.load(ROOT / rec['checkpoint'], map_location='cpu', weights_only=False)
        assert state['completed'] == 1000 and all(int(x['step']) == 1000 for x in state['optimizer']['state'].values())
        key = f'{rec["method"]}_s{rec["seed"]}'
        inputs[key] = torch.load(ROOT / rec['scores'], map_location='cpu', weights_only=False)
        records.append({'key': key, 'kind': 'original', 'method': rec['method'], 'seed': rec['seed'],
                        'scores': rec['scores'], 'scores_sha256': rec['scores_sha256']})
    derived = out / 'derived_rankings'
    derived.mkdir(exist_ok=True)
    for method in ('baseline', 'hierarchy', 'gamma_zero'):
        scores, ranks = ensemble_scores([inputs[f'{method}_s{seed}'].numpy() for seed in (42, 43, 44)])
        key = f'{method}_ensemble'
        path = derived / (key + '.pt')
        torch.save(scores, path)
        inputs[key] = scores
        records.append({'key': key, 'kind': 'ensemble', 'method': method, 'seed': 'ensemble',
                        'scores': str(path.relative_to(ROOT)), 'scores_sha256': sha(path)})
        pd.DataFrame({'canonical_edge_index': np.arange(N_EDGES), 'mean_signed_rank': ranks}).to_csv(derived / (key + '.csv'), index=False)
    graph = graph_new()
    from src.mattr_optimization.gate4_init import edge_mapping
    mapping = edge_mapping(graph, graph)
    pd.testing.assert_frame_equal(mapping, pd.read_csv(ROOT / 'results/mattr_hierarchy/20261022_gate5c/edge_mapping.csv', keep_default_na=False))
    mapping.to_csv(out / 'edge_mapping.csv', index=False)
    topology = out / 'topology'
    topology.mkdir(exist_ok=True)
    queries, attain, closures, audits = [], [], [], []
    for number, rec in enumerate(records, 1):
        key = rec['key']
        scores = inputs[key]
        order = official_order(graph, scores)
        frame, counts, packed, first = curve(graph, scores)
        frame.to_csv(topology / (key + '.csv'), index=False)
        np.savez_compressed(topology / (key + '_sets.npz'), packed=packed, counts=counts, first=first, original_order=order)
        if rec['kind'] == 'original':
            for fraction in (.001, .002, .005, .01, .02, .05):
                previous = old_counts[(old_counts.method == rec['method']) & (old_counts.seed == rec['seed']) & np.isclose(old_counts.fraction, fraction)].iloc[0]
                n = int(fraction * N_EDGES)
                assert int(counts[n]) == int(previous.actual_edges)
                assert digest(unpack(packed, n)) == previous.mask_sha256
        for target in range(1, 641):
            attain.append({'ranking': key, 'kind': rec['kind'], 'target': target,
                           'exactly_attained': bool(np.any(counts == target))})
        for target in targets:
            for which, (n, actual) in [('cap', capped_selection(counts, target))] + (
                [('upper', upper_selection(counts, target))] if target in (t1, t2) else []):
                queries.append({'ranking': key, 'method': rec['method'], 'seed': rec['seed'], 'kind': rec['kind'],
                                'selection': which, 'target': target, 'nominal_n': n, 'actual_edges': actual,
                                'shortfall': target - actual, 'exact': actual == target,
                                'mask_sha256': digest(unpack(packed, n))})
        if rec['kind'] == 'original':
            try:
                exported, checked, audit = validate_closure(graph, scores, counts, packed, first, order)
                closure_key = key + '_closure'
                path = derived / (closure_key + '.pt')
                torch.save(exported, path)
                checked.to_csv(topology / (closure_key + '_checks.csv'), index=False)
                closures.append({**rec, 'key': closure_key, 'kind': 'closure', 'parent': key,
                                 'scores': str(path.relative_to(ROOT)), 'scores_sha256': sha(path)})
            except AssertionError as exc:
                audit = {'passed': False, 'error': str(exc), 'action': 'retain original; skip transformed ranking'}
            audits.append({'ranking': key, **audit})
        print(f'CPU topology {number}/18 {key}; A(1624)={counts[-1]}; elapsed={time.perf_counter()-start:.1f}s', flush=True)
    q = pd.DataFrame(queries)
    q.to_csv(out / 'circuit_queries.csv', index=False)
    pd.DataFrame(attain).to_csv(out / 'exact_attainability_1_640.csv', index=False)
    pd.DataFrame(audits).to_csv(out / 'closure_correctness.csv', index=False)
    pairs = []
    for target in targets:
        cap = q[(q.target == target) & (q.selection == 'cap')].set_index('ranking')
        for a, b in itertools.combinations(cap.index, 2):
            x, y = cap.loc[a], cap.loc[b]
            primary = x['kind'] == y['kind'] == 'original' and {x.method, y.method} == {'baseline', 'hierarchy'} and x.seed == y.seed
            pairs.append({'target': target, 'ranking_a': a, 'ranking_b': b,
                          'actual_a': int(x.actual_edges), 'actual_b': int(y.actual_edges),
                          'count_difference': int(x.actual_edges - y.actual_edges),
                          'both_exact': bool(x.exact and y.exact), 'primary_paired_comparison': primary})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(out / 'exact_comparisons_all_pairs.csv', index=False)
    pairs[pairs.primary_paired_comparison].to_csv(out / 'exact_primary_comparisons.csv', index=False)
    write_json(out / 'ranking_catalog.json', records + closures)
    result = {'passed': True, 'original_rankings': 15, 'ensemble_rankings': 3, 'valid_closure_rankings': len(closures),
              'monotonic': True, 'nested': True, 'Gate5D_counts_and_masks_reproduced': True,
              'prefix_range': [1, MAX_PREFIX], 'T1': t1, 'T2': t2, 'targets': targets,
              'graph_analysis_seconds': time.perf_counter() - start,
              'registered_pair_comparisons': len(pairs), 'both_exact_pairs': int(pairs.both_exact.sum()),
              'primary_pairs': int(pairs.primary_paired_comparison.sum()),
              'primary_both_exact': int(pairs[pairs.primary_paired_comparison].both_exact.sum()), 'model_forwards': 0}
    write_json(out / 'phaseA_complete.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out.resolve())
