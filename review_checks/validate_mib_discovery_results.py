"""Independently validate one completed head-circuit discovery run."""
from itertools import combinations
from math import comb
from pathlib import Path
import json
import sys

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mib_ioi import discovery as d


def validate(directory):
    run = Path(directory).resolve()
    config = json.loads((run / 'experiment_config.json').read_text())
    prior = Path(config['prior_run'])
    selection = pd.read_csv(run / 'circuit_selection_results.csv', dtype={'removal_mask': str})
    oracle = pd.read_csv(run / 'oracle_search_results.csv', dtype={'removal_mask': str})
    costs = pd.read_csv(run / 'runtime_comparison.csv')
    prior_costs = pd.read_csv(prior / 'compute_accounting.csv').set_index('stage')
    manifest = pd.read_csv(run / 'dataset_manifest.csv')
    forward = pd.read_csv(run / 'forward_log.csv')
    accounting = pd.read_csv(run / 'compute_accounting.csv')
    with np.load(run / 'train_coefficients.npz') as fitted:
        beta1, beta2 = fitted['beta1'], fitted['beta2']
    n = len(config['component_list'])
    sizes = list(config['circuit_sizes'])
    assert manifest.groupby('split').size().to_dict() == {
        'train': config['train_examples'], 'validation': config['validation_examples']}
    assert set(manifest.query('split == "train"').example_hash).isdisjoint(
        set(manifest.query('split == "validation"').example_hash))
    assert len(selection) == 2 * (3*len(sizes) + len(config['oracle_completed_sizes']))
    prior_single = pd.read_csv(prior / 'singleton_effects.csv').query(
        'prompt_split == "train"').sort_values('component_index').iloc[:n]
    prior_pairs = pd.read_csv(prior / 'pairwise_interactions.csv').query(
        'prompt_split == "train" and i < @n and j < @n').sort_values(['i', 'j'])
    pair_indices = list(combinations(range(n), 2))
    np.testing.assert_allclose(beta1, d.fit_first_order(prior_single.effect.to_numpy()), atol=1e-9)
    np.testing.assert_allclose(beta2, d.fit_second_order(
        prior_single.effect.to_numpy(), prior_pairs.true_pair_effect.to_numpy(), pair_indices), atol=1e-9)
    components = pd.read_csv(prior / 'selected_components.csv').sort_values('component_index')
    component_names = components.component.iloc[:n].tolist()
    ranks = pd.read_csv(prior / 'official_eap_head_ranking.csv')
    eap_order = [component_names.index(name) for name in ranks.component if name in component_names]
    for row in selection.itertuples():
        mask = np.array([int(bit) for bit in row.removal_mask], dtype=np.int8)
        assert len(mask) == n and row.n_retained == row.circuit_size
        assert row.n_removed == int(mask.sum()) == n-row.circuit_size
        assert json.loads(row.retained_heads) == [component_names[i] for i in d.retained(mask)]
        assert json.loads(row.removed_heads) == [component_names[i] for i in np.flatnonzero(mask)]
        np.testing.assert_allclose(row.change_from_full,
                                   row.actual_task_score-row.full_task_score, atol=1e-8)
        np.testing.assert_allclose(row.absolute_score_error,
                                   abs(row.change_from_full), atol=1e-8)
        np.testing.assert_allclose(row.faithfulness, d.faithfulness(
            row.actual_task_score, row.full_task_score, row.all_removed_task_score), atol=1e-8)
        if row.method == 'EAP':
            np.testing.assert_array_equal(mask, d.select_eap(n, row.circuit_size, eap_order))
        if row.method in ('first order', 'second order'):
            order = 1 if row.method == 'first order' else 2
            beta = beta1 if order == 1 else beta2
            expected, prediction, _ = d.select_surrogate(n, row.circuit_size, beta, order)
            np.testing.assert_array_equal(mask, expected)
            np.testing.assert_allclose(row.predicted_train_effect, prediction, atol=1e-8)
        matching = selection.query(
            'method == @row.method and circuit_size == @row.circuit_size')
        assert matching.removal_mask.nunique() == 1  # validation could not choose a new mask
    if config['run_oracle']:
        for k in config['oracle_completed_sizes']:
            rows = oracle.query('circuit_size == @k')
            assert len(rows) == comb(n, k)
            assert rows.removal_mask.nunique() == len(rows)
            winner = rows.query('is_train_winner == True')
            assert len(winner) == 1
            assert winner.train_absolute_score_error.iloc[0] == rows.train_absolute_score_error.min()
            selected = selection.query('method == "oracle" and circuit_size == @k')
            assert set(selected.removal_mask) == {winner.removal_mask.iloc[0]}
    else:
        assert oracle.empty and selection.query('method == "oracle"').empty
    assert len(forward) == int(accounting.forward_passes_batches.sum())
    assert int(forward.batch_size.sum()) == int(accounting.examples_processed.sum())
    assert forward.success.all()
    assert not forward.grad_enabled.any()
    assert {'circuit_faithfulness.png', 'circuit_scores.png', 'oracle_gap.png',
            'compute_comparison.png'}.issubset({p.name for p in run.iterdir()})
    for row in costs.itertuples():
        assert row.total_forward_batches == (row.historical_acquisition_forward_batches
                                             + row.final_exact_forward_batches
                                             + row.oracle_search_forward_batches)
        assert row.total_example_forwards == (row.historical_acquisition_example_forwards
                                              + row.final_exact_example_forwards
                                              + row.oracle_search_example_forwards)
        if row.method == 'second order':
            assert row.historical_pair_seconds > 0
        elif row.method in ('EAP', 'first order'):
            assert row.historical_pair_seconds == 0
        stages = (['raw_ioi_scores', 'official_eap_selection'] if row.method == 'EAP'
                  else ['raw_ioi_scores', 'singletons'] if row.method == 'first order'
                  else ['raw_ioi_scores', 'singletons', 'exact_pairs']
                  if row.method == 'second order' else ['raw_ioi_scores'])
        assert row.peak_cuda_allocated_mib >= max(
            prior_costs.loc[s, 'peak_cuda_allocated_mib'] for s in stages)
        expected_seconds = (row.historical_EAP_seconds + row.historical_singleton_seconds
                            + row.historical_pair_seconds + row.historical_raw_baseline_seconds
                            + row.fit_seconds + row.surrogate_or_EAP_selection_seconds
                            + row.final_exact_evaluation_seconds + row.oracle_exhaustive_search_seconds)
        np.testing.assert_allclose(row.total_discovery_seconds, expected_seconds, atol=1e-6)
    notebook = nbformat.read(ROOT / 'notebooks/mib_circuit_discovery.executed.ipynb',
                             as_version=4)
    cells = [c for c in notebook.cells if c.cell_type == 'code']
    assert len(cells) == 9 and all(c.execution_count is not None for c in cells)
    assert not any(o.output_type == 'error' for c in cells for o in c.outputs)
    return {'passed': True, 'components': n, 'circuit_sizes': sizes,
            'methods': sorted(selection.method.unique().tolist()),
            'current_run_forward_batches': len(forward),
            'current_run_example_forwards': int(forward.batch_size.sum()),
            'oracle_enabled': bool(config['run_oracle'])}


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python validate_mib_discovery_results.py RESULTS_DIRECTORY')
    result = validate(sys.argv[1])
    (Path(sys.argv[1]) / 'independent_validation.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
