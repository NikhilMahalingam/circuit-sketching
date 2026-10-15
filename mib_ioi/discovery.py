"""Head-circuit selection over a fixed MIB IOI component set.

All masks use 1 = remove/patch from the matched corrupted prompt.
Selection helpers consume training measurements only; model inference stays in
the notebook through IOIScorer.
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from time import perf_counter

import numpy as np

from mib_ioi.analysis import features


def removal_mask(n: int, kept) -> np.ndarray:
    kept = tuple(int(i) for i in kept)
    if n < 1 or len(set(kept)) != len(kept) or any(i < 0 or i >= n for i in kept):
        raise ValueError("Kept heads must be distinct indices within the selected set.")
    mask = np.ones(n, dtype=np.int8)
    mask[list(kept)] = 0
    return mask


def retained(mask) -> tuple[int, ...]:
    z = np.asarray(mask)
    if z.ndim != 1 or not np.isin(z, [0, 1]).all():
        raise ValueError("Expected a one-dimensional binary removal mask.")
    return tuple(np.flatnonzero(z == 0).tolist())


def candidates(n: int, k: int):
    if not 0 <= k <= n:
        raise ValueError("Circuit size outside component set.")
    for kept in combinations(range(n), k):
        yield removal_mask(n, kept)


def candidate_count(n: int, k: int) -> int:
    return comb(n, k)


def fit_first_order(singleton_effects):
    singleton = np.asarray(singleton_effects, dtype=float)
    if singleton.ndim != 1 or not np.isfinite(singleton).all():
        raise ValueError("Expected finite mean train singleton effects.")
    n = len(singleton)
    single_masks = np.eye(n, dtype=np.int8)
    result = np.linalg.lstsq(features(single_masks, 1), singleton, rcond=None)[0]
    np.testing.assert_allclose(features(single_masks, 1) @ result, singleton, atol=1e-9)
    return result


def fit_second_order(singleton_effects, pair_effects, pairs):
    """Fit no-intercept quadratic model on exact train singleton/pair masks."""
    singleton = np.asarray(singleton_effects, dtype=float)
    if singleton.ndim != 1 or not np.isfinite(singleton).all():
        raise ValueError("Expected finite mean train singleton effects.")
    n = len(singleton)
    pairs = [tuple(map(int, pair)) for pair in pairs]
    if pairs != list(combinations(range(n), 2)):
        raise ValueError("Pair rows must contain each i<j exactly once in order.")
    pair_effects = np.asarray(pair_effects, dtype=float)
    if pair_effects.shape != (len(pairs),) or not np.isfinite(pair_effects).all():
        raise ValueError("Expected one finite mean train effect per pair.")
    single_masks = np.eye(n, dtype=np.int8)
    pair_masks = np.stack([removal_mask(n, set(range(n)) - {i, j}) for i, j in pairs])
    design2 = features(np.concatenate([single_masks, pair_masks]), 2)
    targets2 = np.concatenate([singleton, pair_effects])
    if np.linalg.matrix_rank(design2) != design2.shape[1]:
        raise ValueError("Singleton/pair design is not full rank.")
    order2 = np.linalg.lstsq(design2, targets2, rcond=None)[0]
    np.testing.assert_allclose(design2 @ order2, targets2, atol=1e-9)
    return order2


def predicted_effect(mask, coefficients, order: int) -> float:
    z = np.asarray(mask, dtype=np.int8).reshape(1, -1)
    x = features(z, order)
    beta = np.asarray(coefficients, dtype=float)
    if x.shape[1] != len(beta) or not np.isfinite(beta).all():
        raise ValueError("Coefficient shape/value does not match circuit features.")
    return float(x[0] @ beta)


def select_surrogate(n: int, k: int, coefficients, order: int):
    """Minimize |predicted task-score change|; ties use combination order."""
    start = perf_counter()
    best, best_loss, best_prediction = None, float("inf"), None
    for mask in candidates(n, k):
        prediction = predicted_effect(mask, coefficients, order)
        loss = abs(prediction)
        if loss < best_loss:
            best, best_loss, best_prediction = mask, loss, prediction
    return best, best_prediction, perf_counter() - start


def select_eap(n: int, k: int, ranked_indices):
    ranked = tuple(map(int, ranked_indices))
    if len(ranked) != n or set(ranked) != set(range(n)):
        raise ValueError("EAP ranking must cover the exact selected component set.")
    return removal_mask(n, ranked[:k])


def faithfulness(score: float, full_score: float, empty_score: float) -> float:
    """1 - absolute score error / error of the all-removed circuit.

    This custom head-subset score is not official MIB CPR or CMD. It may be
    negative, and is undefined if all-removal does not change the mean score.
    """
    scale = abs(empty_score - full_score)
    return float(1 - abs(score - full_score) / scale) if scale > 1e-12 else float("nan")


def oracle_search(n: int, k: int, evaluate_train, full_score: float,
                  max_candidates: int, progress=None):
    """Exhaustive train-only search; fail closed before a partial enumeration."""
    count = candidate_count(n, k)
    if count > max_candidates:
        raise ValueError(f"{count} candidates exceed MAX_ORACLE_CANDIDATES={max_candidates}.")
    rows = []
    best = None
    best_loss = float("inf")
    for index, mask in enumerate(candidates(n, k), 1):
        score = float(evaluate_train(mask))
        loss = abs(score - full_score)
        rows.append((mask.copy(), score, loss))
        if loss < best_loss:
            best, best_loss = mask.copy(), loss
        if progress is not None:
            progress(index, count)
    if len(rows) != count:
        raise AssertionError("Oracle search did not visit every candidate.")
    return best, rows


def bootstrap_faithfulness(circuit_scores, full_scores, empty_scores,
                           repeats: int, seed: int):
    circuit = np.asarray(circuit_scores, dtype=float)
    full = np.asarray(full_scores, dtype=float)
    empty = np.asarray(empty_scores, dtype=float)
    if circuit.shape != full.shape or circuit.shape != empty.shape or circuit.ndim != 1:
        raise ValueError("Bootstrap inputs must align by prompt.")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        ix = rng.integers(0, len(full), len(full))
        value = faithfulness(circuit[ix].mean(), full[ix].mean(), empty[ix].mean())
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(values, [.025, .975])))


def interaction_terms(mask, beta2):
    z = np.asarray(mask, dtype=np.int8)
    pairs = list(combinations(range(len(z)), 2))
    beta = np.asarray(beta2, dtype=float)
    if beta.shape != (len(z) + len(pairs),):
        raise ValueError("Second-order coefficient vector has the wrong size.")
    return {(i, j): float(beta[len(z) + p]) for p, (i, j) in enumerate(pairs)
            if z[i] and z[j]}
