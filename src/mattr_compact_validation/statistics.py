"""Shared, template-stratified paired prompt bootstrap; fixed trained seeds."""
import numpy as np
import pandas as pd


def bootstrap_indices(families, replicates=5000, seed=5642):
    families = np.asarray(families)
    rng = np.random.default_rng(seed)
    columns = []
    for family in sorted(set(families)):
        members = np.flatnonzero(families == family)
        columns.append(rng.choice(members, size=(replicates, len(members)), replace=True))
    return np.concatenate(columns, axis=1)


def interval(values, indices, endpoints=1):
    values = np.asarray(values, dtype=np.float64)
    draws = values[indices].mean(axis=1)
    alpha = .05 / endpoints
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(low), float(high)


def auc_prompt_values(correct, grid):
    """Linear functional for BOOTSTRAP only; point estimate uses official MIB."""
    x = np.log(np.asarray(grid, dtype=float))
    # [seed, grid, prompt] -> [seed, prompt]
    return np.sum(np.diff(x)[None, :, None] * (correct[:, :-1] + correct[:, 1:]) / 2, axis=1) / (x[-1] - x[0])


def compare_arrays(a, b, families, grid, indices, comparator):
    """a,b are [seed, grid, prompt] float arrays of positive-margin correctness."""
    assert a.shape == b.shape and a.shape[0] == 3 and a.shape[2] == len(families)
    diff = a - b
    rows = []
    for fraction in (.005, .01, .02, .05):
        j = list(grid).index(fraction)
        for seed_index, seed in enumerate((42, 43, 44)):
            delta, lo, hi = interval(diff[seed_index, j], indices, 2)
            rows.append({'comparator': comparator, 'seed': str(seed), 'fraction': fraction,
                         'delta': delta, 'lower': lo, 'upper': hi, 'family_endpoints': 2,
                         'corrected': int(((a[seed_index, j] == 1) & (b[seed_index, j] == 0)).sum()),
                         'broken': int(((a[seed_index, j] == 0) & (b[seed_index, j] == 1)).sum()),
                         'interval_scope': '2 cutoffs within this seed; no correction across seeds'})
        k = 2 if comparator == 'baseline' else 6
        delta, lo, hi = interval(diff[:, j].mean(axis=0), indices, k)
        rows.append({'comparator': comparator, 'seed': 'equal_seed_mean', 'fraction': fraction,
                     'delta': delta, 'lower': lo, 'upper': hi, 'family_endpoints': k,
                     'corrected': float(np.mean(((a[:, j] == 1) & (b[:, j] == 0)).sum(axis=1))),
                     'broken': float(np.mean(((a[:, j] == 0) & (b[:, j] == 1)).sum(axis=1))),
                     'interval_scope': 'shared prompt draws; equally weighted FIXED seeds'})
    auc_delta = auc_prompt_values(a, grid) - auc_prompt_values(b, grid)
    auc = []
    for i, seed in enumerate((42, 43, 44)):
        d, low, high = interval(auc_delta[i], indices)
        auc.append({'comparator': comparator, 'seed': str(seed), 'delta': d, 'lower': low, 'upper': high})
    d, low, high = interval(auc_delta.mean(axis=0), indices)
    auc.append({'comparator': comparator, 'seed': 'equal_seed_mean', 'delta': d, 'lower': low, 'upper': high})
    return pd.DataFrame(rows), pd.DataFrame(auc)


def decision(comparisons, auc, valid=True):
    def test(comparator):
        rows = comparisons[comparisons.comparator == comparator]
        pooled = rows[rows.seed == 'equal_seed_mean'].set_index('fraction')
        per_seed = rows[rows.seed != 'equal_seed_mean']
        checks = {'valid_conventions': bool(valid)}
        for f in (.005, .01):
            checks[f'positive_mean_{f}'] = bool(pooled.loc[f, 'delta'] > 0)
            checks[f'corrected_lower_{f}'] = bool(pooled.loc[f, 'lower'] > 0)
            checks[f'positive_2seeds_{f}'] = bool((per_seed[per_seed.fraction == f].delta > 0).sum() >= 2)
        if comparator == 'baseline':
            checks['accuracy_auc'] = bool(auc[(auc.comparator == comparator) & (auc.seed == 'equal_seed_mean')].delta.iloc[0] >= -.02)
            checks['compact_2_5_safeguard'] = bool(all(pooled.loc[f, 'delta'] >= -.02 for f in (.02, .05)))
        return {'passed': all(checks.values()), 'checks': checks}
    primary = test('baseline')
    controls = {key: test(key) for key in ('gamma_zero', 'shuffled', 'edge_l2')}
    structural = primary['passed'] and all(c['passed'] for c in controls.values())
    if structural:
        classification = 'Compact-accuracy generalization success and structural-hierarchy advantage'
    elif primary['passed']:
        classification = 'Compact-accuracy generalization success; structural-hierarchy advantage not established'
    elif (comparisons[(comparisons.comparator == 'baseline') &
                      (comparisons.seed == 'equal_seed_mean') & comparisons.fraction.isin([.005, .01])].delta > 0).all():
        classification = 'Mixed or inconclusive'
    else:
        classification = 'Negative result'
    return {'primary': primary, 'controls': controls, 'structural_advantage': bool(structural),
            'classification': classification, 'training_seed_significance_claim': False,
            'CPR_improvement_is_separate': True, 'further_training_authorized': False}
