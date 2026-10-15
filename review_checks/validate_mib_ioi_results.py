"""Independently validate saved real IOI results; no model inference required."""
import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, r2_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    root = args.run_dir
    config = json.loads((root/'experiment_config.json').read_text())
    assert config['model_key'] == 'gpt2' and config['task'] == 'ioi'
    assert config['dataset'] == 'mib-bench/ioi' and config['device'] == 'cuda'
    n = config['component_count']
    components = pd.read_csv(root/'selected_components.csv')
    assert len(components) == n and not components[['layer', 'head']].duplicated().any()
    examples = pd.read_csv(root/'ioi_examples.csv')
    for a, b in itertools.combinations(examples.split.unique(), 2):
        assert set(examples[examples.split == a].example_hash).isdisjoint(
            set(examples[examples.split == b].example_hash))
    assert all(v == 0 for v in config['dataset_info']['full_filtered_split_clean_prompt_overlap'].values())
    forwards = pd.read_csv(root/'forward_log.csv')
    accounting = pd.read_csv(root/'compute_accounting.csv')
    assert forwards.success.all() and accounting.success.all()
    assert len(forwards) == accounting.forward_passes_batches.sum()
    assert forwards.batch_size.sum() == accounting.examples_processed.sum()
    assert set(forwards.loc[forwards.grad_enabled, 'stage']) <= {'official_eap_selection'}
    assert accounting.peak_cuda_allocated_mib.max() > 0
    checks = json.loads((root/'intervention_checks.json').read_text())
    for key in ('hook_removal_max_error', 'determinism_max_error', 'batching_max_error', 'official_graph_max_error'):
        assert checks[key] < checks['tolerance']
    with np.load(root/'singleton_effects_per_example.npz') as singles, \
            np.load(root/'pair_effects_per_example.npz') as pairs, \
            np.load(root/'pairwise_interactions_per_example.npz') as interactions:
        identities = pairs['pairs']
        assert set(map(tuple, identities)) == set(itertools.combinations(range(n), 2))
        for split in examples.split.unique():
            expected = pairs[split] - np.stack([singles[split][i]+singles[split][j] for i, j in identities])
            np.testing.assert_allclose(interactions[split], expected)
    with np.load(root/'mask_effects_per_example.npz') as data:
        mask_sets = {s: set(map(tuple, data[f'masks_{s}'])) for s in ('train', 'validation', 'test')}
        assert all(a.isdisjoint(b) for a, b in itertools.combinations(mask_sets.values(), 2))
        for split, keys in mask_sets.items():
            assert len(keys) == len(data[f'masks_{split}'])
        prediction_rows = pd.read_csv(root/'heldout_predictions.csv')
        summary = pd.read_csv(root/'prediction_summary.csv')
        for row in summary.itertuples():
            records = prediction_rows[(prediction_rows.prompt_split == row.eval_prompt_split) &
                                      (prediction_rows.mask_split == row.mask_split)]
            if row.mask_size != 'all':
                records = records[records.mask_size == int(row.mask_size)]
            truth, pred = records.truth.to_numpy(), records[row.method].to_numpy()
            expected = [mean_absolute_error(truth, pred), pearsonr(truth, pred).statistic,
                        spearmanr(truth, pred).statistic, r2_score(truth, pred)]
            np.testing.assert_allclose([row.MAE, row.Pearson, row.Spearman, row.R2], expected,
                                       rtol=1e-7, atol=1e-9, equal_nan=True)
        # Independently re-fit training-only coefficients from saved arrays.
        z = data['masks_train'].astype(float)
        design = np.column_stack([z] + [z[:, i]*z[:, j] for i, j in itertools.combinations(range(n), 2)])
        y = data['effects_train_train'].mean(1)
        with np.load(root/'fitted_coefficients.npz') as fitted:
            np.testing.assert_allclose(fitted['linear'], np.linalg.lstsq(z, y, rcond=None)[0], atol=1e-10)
            np.testing.assert_allclose(fitted['quadratic'], np.linalg.lstsq(design, y, rcond=None)[0], atol=1e-10)
        execution = json.loads((root/'notebook_execution.json').read_text())
        if not execution.get('resumed', True):
            batches = sum(math.ceil(count/config['batch_size']) for count in config['dataset_splits'].values())
            unique = set().union(*mask_sets.values())
            single_keys = set(map(tuple, np.eye(n, dtype=int)))
            pair_keys = set()
            for i, j in itertools.combinations(range(n), 2):
                key = [0]*n
                key[i] = key[j] = 1
                pair_keys.add(tuple(key))
            expected = {'raw_ioi_scores': 2*batches, 'singletons': (n+2)*batches,
                        'exact_pairs': n*(n-1)//2*batches,
                        'random_masks': len(unique-single_keys-pair_keys)*batches,
                        'official_eap_selection': 3*math.ceil(config['eap_examples']/config['eap_batch_size'])}
            for stage, count in expected.items():
                actual = accounting.loc[accounting.stage == stage, 'forward_passes_batches'].sum()
                assert actual == count, (stage, actual, count)
    mib = pd.read_csv(root/'mib_baseline_results.csv')
    for row in mib.itertuples():
        raw = json.loads((root/f'mib_official_raw_{row.split}.json').read_text())
        assert len(raw['faithfulnesses']) == len(raw['weighted_edge_counts']) == 10
        np.testing.assert_allclose([row.area_under, row.area_from_1, row.average],
                                   [raw['area_under'], raw['area_from_1'], raw['average']])
    result = {'passed': True, 'real_forward_batches': len(forwards),
              'example_forwards': int(forwards.batch_size.sum()), 'components': n,
              'pairs': n*(n-1)//2, 'mask_counts': {s: len(v) for s, v in mask_sets.items()},
              'prompt_counts': examples.groupby('split').size().to_dict(),
              'peak_cuda_allocated_mib': float(accounting.peak_cuda_allocated_mib.max()),
              'heldout_metrics_recomputed': True, 'training_only_fits_recomputed': True,
              'official_evaluator_outputs_preserved': True}
    (root/'independent_validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
