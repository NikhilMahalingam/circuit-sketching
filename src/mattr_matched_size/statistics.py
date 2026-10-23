"""One primary cap endpoint, fixed-seed paired prompt bootstrap and ratio."""
import numpy as np
from src.mattr_compact_validation.statistics import bootstrap_indices


def estimates(values, draws):
    values = np.asarray(values, dtype=np.float64)
    boot = values[draws].mean(axis=1)
    lower, upper = np.quantile(boot, [.025, .975])
    return {'mean': float(values.mean()), 'lower': float(lower), 'upper': float(upper)}, boot


def primary_analysis(cap_difference, nominal_difference, draws, valid=True):
    cap_difference = np.asarray(cap_difference, dtype=float)
    nominal_difference = np.asarray(nominal_difference, dtype=float)
    assert cap_difference.shape == nominal_difference.shape and cap_difference.shape[0] == 3
    cap, cap_draws = estimates(cap_difference.mean(axis=0), draws)
    nominal, nom_draws = estimates(nominal_difference.mean(axis=0), draws)
    positive = int((cap_difference.mean(axis=1) > 0).sum())
    checks = {'positive_mean': cap['mean'] > 0, 'positive_at_least2_seeds': positive >= 2,
              'paired_prompt_lower_positive': cap['lower'] > 0, 'correct_evaluator_and_masks': bool(valid)}
    passed = bool(all(checks.values()))
    unstable = nominal['lower'] <= 0 <= nominal['upper'] or abs(nominal['mean']) < 1 / cap_difference.shape[1]
    ratio = {'stable': not unstable, 'point': None, 'lower': None, 'upper': None,
             'denominator_nearzero_threshold': 1 / cap_difference.shape[1],
             'invalid_bootstrap_denominators': int((np.abs(nom_draws) < 1e-15).sum()),
             'causal_explained_fraction': False}
    if not unstable:
        samples = cap_draws[np.abs(nom_draws) >= 1e-15] / nom_draws[np.abs(nom_draws) >= 1e-15]
        ratio.update(point=cap['mean'] / nominal['mean'], lower=float(np.quantile(samples,.025)), upper=float(np.quantile(samples,.975)))
    if nominal['lower'] <= 0:
        label = 'Non-replication'
    elif passed and ratio['stable'] and ratio['point'] >= .5:
        label = 'Ranking-quality effect'
    elif not passed and ((cap['lower'] <= 0 <= cap['upper']) or (ratio['stable'] and ratio['point'] < .25)):
        label = 'Mostly connectivity'
    else:
        label = 'Mixed'
    return {'primary_success': passed, 'checks': {k:bool(v) for k,v in checks.items()},
            'cap_T1': cap, 'nominal_0.5pct': nominal, 'positive_cap_seeds': positive,
            'per_seed_cap_changes': cap_difference.mean(axis=1).tolist(),
            'per_seed_nominal_changes': nominal_difference.mean(axis=1).tolist(),
            'advantage_retention_ratio': ratio, 'classification': label,
            'scope': 'capped-size comparison; no causal fraction or training-seed significance claim',
            'new_training_authorized': False}
