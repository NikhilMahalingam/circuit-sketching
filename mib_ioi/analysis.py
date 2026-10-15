"""Prompt-level statistics and disjoint mask experiments (not MIB CPR/CMD)."""
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def metrics(truth, prediction):
    y, p = np.asarray(truth, float), np.asarray(prediction, float)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Metric inputs must be aligned and finite.')
    varied = len(y) >= 2 and np.ptp(y) > 1e-12 and np.ptp(p) > 1e-12
    denom = ((y - y.mean()) ** 2).sum()
    return {'MAE': float(abs(y-p).mean()),
            'Pearson': float(stats.pearsonr(y, p).statistic) if varied else np.nan,
            'Spearman': float(stats.spearmanr(y, p).statistic) if varied else np.nan,
            'R2': float(1 - ((y-p)**2).sum()/denom) if len(y) >= 2 and denom > 1e-12 else np.nan}


def features(masks, order=1):
    z = np.asarray(masks, float)
    if z.ndim != 2 or order not in (1, 2):
        raise ValueError('Expected a mask matrix and order 1 or 2.')
    if order == 1:
        return z
    pairs = list(itertools.combinations(range(z.shape[1]), 2))
    return np.column_stack([z] + [z[:, i] * z[:, j] for i, j in pairs])


def fit(masks, targets, order):
    design = features(masks, order)
    rank = np.linalg.matrix_rank(design)
    if rank != design.shape[1]:
        raise ValueError(f'Order-{order} training design rank {rank}/{design.shape[1]}; increase training masks.')
    return np.linalg.lstsq(design, np.asarray(targets), rcond=None)[0]


def mask_splits(n, seed, train_per_density=64, eval_per_density=16):
    """Uniform sampling without replacement within fixed intervention sizes.

    Small strata cannot satisfy arbitrary counts (only n singleton masks exist).
    Divide those finite pools approximately equally across the three splits.
    """
    if not 12 <= n <= 16:
        raise ValueError('This controlled experiment supports 12--16 selected heads.')
    rng = np.random.default_rng(seed)
    sizes = sorted({1, 2, round(n*.25), round(n*.5), round(n*.75)})
    result = {s: [] for s in ('train', 'validation', 'test')}
    allocation = []
    for k in sizes:
        pool = list(itertools.combinations(range(n), k))
        rng.shuffle(pool)
        target = train_per_density + 2*eval_per_density
        if len(pool) < target:
            nv = nt = min(eval_per_density, len(pool)//3)
            nr = min(train_per_density, len(pool) - nv - nt)
        else:
            nr, nv, nt = train_per_density, eval_per_density, eval_per_density
        cursor = 0
        for split, count in zip(result, (nr, nv, nt)):
            for indices in pool[cursor:cursor+count]:
                mask = np.zeros(n, dtype=np.int8)
                mask[list(indices)] = 1
                result[split].append(mask)
            allocation.append({'mask_size': k, 'mask_split': split, 'count': count})
            cursor += count
    result = {k: np.stack(v) for k, v in result.items()}
    sets = {k: set(map(tuple, v)) for k, v in result.items()}
    assert all(len(sets[k]) == len(result[k]) for k in result)
    assert all(sets[a].isdisjoint(sets[b]) for a, b in itertools.combinations(sets, 2))
    design = features(result['train'], 2)
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('Quadratic mask design is rank deficient; increase TRAIN_MASKS_PER_DENSITY.')
    return result, pd.DataFrame(allocation)


def singleton_table(scorer, split, effects):
    baseline = scorer.score(split)[0]
    rows = []
    for i, ((layer, head), values) in enumerate(zip(scorer.components, effects)):
        rows.append({'prompt_split': split, 'component_index': i, 'component': f'a{layer}.h{head}',
                     'layer': layer, 'head': head, 'baseline_score': baseline.mean(),
                     'intervened_score': (baseline + values).mean(), 'effect': values.mean(),
                     'absolute_effect': abs(values.mean()), 'std_across_examples': values.std(ddof=1),
                     'standard_error': values.std(ddof=1)/np.sqrt(len(values)), 'n_examples': len(values)})
    return pd.DataFrame(rows)


def pair_statistics(singletons, pairs_effects, pairs, components, split, bootstrap, seed):
    additive = np.stack([singletons[i] + singletons[j] for i, j in pairs])
    interaction = pairs_effects - additive
    true, pred = pairs_effects.mean(1), additive.mean(1)
    means = interaction.mean(1)
    scale = np.array([abs(singletons[i].mean()) + abs(singletons[j].mean()) for i, j in pairs])
    rng = np.random.default_rng(seed)
    draws, signed_means = [], []
    for _ in range(bootstrap):
        indices = rng.integers(0, singletons.shape[1], singletons.shape[1])
        y, p = pairs_effects[:, indices].mean(1), additive[:, indices].mean(1)
        m = metrics(y, p)
        m['mean_absolute_interaction'] = float(abs(y-p).mean())
        draw_scale = np.mean([abs(singletons[i, indices].mean()) + abs(singletons[j, indices].mean())
                              for i, j in pairs])
        m['relative_mean_interaction'] = float(abs(y-p).mean()/draw_scale) if draw_scale > 1e-12 else np.nan
        draws.append(m)
        signed_means.append(y-p)
    boot = pd.DataFrame(draws)
    signed_ci = np.quantile(np.stack(signed_means), [.025, .975], axis=0)
    # Signed interactions per example: exploratory two-sided one-sample t tests.
    # BY controls FDR under arbitrary dependence among the many head pairs.
    pvalues = np.asarray(stats.ttest_1samp(interaction, 0, axis=1).pvalue)
    pvalues = np.where(np.isfinite(pvalues), pvalues, 1.)
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.minimum.accumulate((pvalues[order] * m * np.sum(1/np.arange(1, m+1)) /
                                      np.arange(1, m+1))[::-1])[::-1]
    qvalues = np.empty(m)
    qvalues[order] = np.minimum(adjusted, 1.)
    rows = []
    for p, (i, j) in enumerate(pairs):
        rows.append({'prompt_split': split, 'i': i, 'j': j,
                     'component_i': f'a{components[i][0]}.h{components[i][1]}',
                     'component_j': f'a{components[j][0]}.h{components[j][1]}',
                     'true_pair_effect': true[p], 'first_order_prediction': pred[p],
                     'interaction': means[p], 'absolute_interaction': abs(means[p]),
                     'singleton_magnitude_scale': scale[p],
                     'relative_interaction_size': abs(means[p])/scale[p] if scale[p] > 1e-12 else np.nan,
                     'standard_error': interaction[p].std(ddof=1)/np.sqrt(interaction.shape[1]),
                     'signed_ci_low': signed_ci[0, p], 'signed_ci_high': signed_ci[1, p],
                     'signed_ttest_p': pvalues[p], 'BY_q': qvalues[p]})
    summary = {'prompt_split': split, 'n_pairs': len(pairs), 'n_examples': singletons.shape[1],
               **metrics(true, pred), 'mean_absolute_interaction': float(abs(means).mean()),
               'median_absolute_interaction': float(np.median(abs(means))),
               'relative_mean_interaction': float(abs(means).mean()/scale.mean()) if scale.mean() > 1e-12 else np.nan,
               'BY_significant_pairs_0.05': int((qvalues < .05).sum())}
    for metric in ('MAE', 'Spearman', 'mean_absolute_interaction', 'relative_mean_interaction'):
        values = boot[metric].dropna()
        summary[metric + '_ci_low'] = float(values.quantile(.025)) if len(values) else np.nan
        summary[metric + '_ci_high'] = float(values.quantile(.975)) if len(values) else np.nan
    return pd.DataFrame(rows), summary, interaction, boot


def pair_plots(table, n, output):
    matrix = np.full((n, n), np.nan)
    for r in table.itertuples():
        matrix[r.i, r.j] = matrix[r.j, r.i] = r.interaction
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    limit = max(float(np.nanmax(abs(matrix))), 1e-8)
    im = axes[0].imshow(matrix, cmap='coolwarm', vmin=-limit, vmax=limit)
    axes[0].set(title='Signed pair interaction', xlabel='Component index', ylabel='Component index')
    fig.colorbar(im, ax=axes[0], label='IOI logit-difference units')
    axes[1].hist(table.absolute_interaction, bins=15)
    axes[1].set(title='Interaction magnitudes', xlabel='Absolute mean interaction', ylabel='Pairs')
    x, y = table.first_order_prediction, table.true_pair_effect
    axes[2].scatter(x, y, alpha=.7)
    bounds = [min(x.min(), y.min()), max(x.max(), y.max())]
    axes[2].plot(bounds, bounds, '--', color='black')
    axes[2].set(title='Additivity on pairs', xlabel='Singleton additive prediction', ylabel='Exact pair effect')
    fig.tight_layout()
    fig.savefig(Path(output) / 'pairwise_interactions.png', dpi=160)
    return fig


def prediction_rows(truth, predictions, masks, prompt_split, mask_split):
    rows = []
    sizes = masks.sum(1)
    for method, prediction in predictions.items():
        for size in ['all'] + list(np.unique(sizes)):
            take = np.ones(len(masks), dtype=bool) if size == 'all' else sizes == size
            rows.append({'method': method, 'fit_prompt_split': 'train', 'eval_prompt_split': prompt_split,
                         'mask_split': mask_split, 'mask_size': size, 'n_masks': int(take.sum()),
                         **metrics(truth[take], prediction[take])})
    return rows


def fit_predictions(train_masks, train_effects, test_masks, train_singletons):
    y = train_effects.mean(1)
    beta1, beta2 = fit(train_masks, y, 1), fit(train_masks, y, 2)
    return {'additive singleton': test_masks @ train_singletons.mean(1),
            'learned first order': features(test_masks, 1) @ beta1,
            'learned second order': features(test_masks, 2) @ beta2}, (beta1, beta2)


def export_masks(out_dir, masks, arrays):
    """Persist averaged rows plus arrays aligned by split/mask/example index."""
    for mask_split, z in masks.items():
        rows = []
        for prompt_split, by_mask_split in arrays.items():
            for index, (mask, values) in enumerate(zip(z, by_mask_split[mask_split])):
                rows.append({'prompt_split': prompt_split, 'mask_split': mask_split, 'mask_index': index,
                             'mask': json.dumps(mask.tolist()), 'mask_size': int(mask.sum()),
                             'true_score_change': values.mean(), 'n_examples': len(values),
                             'standard_error': values.std(ddof=1)/np.sqrt(len(values))})
        pd.DataFrame(rows).to_csv(Path(out_dir) / f'random_mask_{mask_split}.csv', index=False)
    np.savez_compressed(Path(out_dir) / 'mask_effects_per_example.npz',
                        **{f'masks_{s}': z for s, z in masks.items()},
                        **{f'effects_{p}_{s}': a for p, by_s in arrays.items() for s, a in by_s.items()})


def sample_efficiency(masks, arrays, singletons, sizes, repeats, seed):
    """Re-fit from cached per-example effects; no extra model forwards.

    Nested randomized training-prompt subsets; fixed validation prompts and
    validation masks. Test masks are not used to choose the sample size.
    """
    rows = []
    available = arrays['train']['train'].shape[1]
    target = arrays['validation']['validation'].mean(1)
    reference = singletons['train'].mean(1)
    for repeat in range(repeats):
        permutation = np.random.default_rng(seed + repeat).permutation(available)
        for size in sizes:
            if size > available:
                continue
            indices = permutation[:size]
            preds, _ = fit_predictions(masks['train'], arrays['train']['train'][:, indices],
                                       masks['validation'], singletons['train'][:, indices])
            rank = metrics(abs(reference), abs(singletons['train'][:, indices].mean(1)))['Spearman']
            for method, pred in preds.items():
                rows.append({'n_train_examples': size, 'repeat': repeat, 'method': method,
                             'eval_prompt_split': 'validation', 'mask_split': 'validation',
                             'singleton_magnitude_rank_vs_full_train': rank,
                             **metrics(target, pred)})
    return pd.DataFrame(rows)


def error_plot(results, output):
    table = results[(results.eval_prompt_split == 'validation') & (results.mask_split == 'test') &
                    (results.mask_size != 'all')]
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, group in table.groupby('method', sort=False):
        group = group.assign(mask_size=group.mask_size.astype(int)).sort_values('mask_size')
        ax.plot(group.mask_size, group.MAE, 'o-', label=method)
    ax.set(xlabel='Number of heads patched', ylabel='MAE (IOI logit-difference units)',
           title='Held-out masks on validation prompts')
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / 'prediction_error_by_mask_size.png', dpi=160)
    return fig


def efficiency_plot(table, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, group in table.groupby('method', sort=False):
        for ax, column in zip(axes, ['MAE', 'Spearman']):
            agg = group.groupby('n_train_examples')[column].agg(['mean', 'std'])
            ax.errorbar(agg.index, agg['mean'], yerr=agg['std'].fillna(0), marker='o', label=method)
            ax.set(xlabel='IOI training examples', ylabel=f'Held-out validation {column}')
    axes[0].legend()
    fig.suptitle('Nested prompt subsamples; error bars are across-subset SD')
    fig.tight_layout()
    fig.savefig(Path(output) / 'sample_efficiency.png', dpi=160)
    return fig


def bootstrap_improvement(truth_per_example, linear, quadratic, repeats, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        indices = rng.integers(0, truth_per_example.shape[1], truth_per_example.shape[1])
        truth = truth_per_example[:, indices].mean(1)
        values.append(abs(truth-linear).mean() - abs(truth-quadratic).mean())
    return {'MAE_improvement': float(abs(truth_per_example.mean(1)-linear).mean() -
                                   abs(truth_per_example.mean(1)-quadratic).mean()),
            'ci_low': float(np.quantile(values, .025)), 'ci_high': float(np.quantile(values, .975)),
            'scope': 'Prompt bootstrap conditional on fixed training fit, components, and test masks'}
