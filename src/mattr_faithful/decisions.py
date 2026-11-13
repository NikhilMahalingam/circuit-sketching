"""Registered decisions and paired prompt bootstrap, without model inference."""

import numpy as np

from src.mattr_fidelity_splice.statistics import official_areas, resamples, weights

ARMS = {"M0": ("margin", False), "M1": ("margin", True),
        "K0": ("kl", False), "K1": ("kl", True), "R0": ("repo_cmd", False)}
SEEDS = tuple(range(42, 47))
REFERENCES = ("eapig_signed", "eapig_abs")
BASELINES = ("B0", "B1")
CAPS = (60, 116, 180, 296, 640)
N = 32491
EPS = 1e-8


def ranking_key(arm, seed, step=5000):
    return f"{arm}_s{seed}_{step}"


def stage1_keys():
    return (*REFERENCES, *(ranking_key("mattr_abs", seed) for seed in SEEDS))


def development_keys():
    return (*stage1_keys(), *(ranking_key(arm, seed, step)
             for arm in (*BASELINES, *ARMS)
             for seed in (SEEDS if arm in BASELINES else SEEDS[:3])
             for step in (1000, 5000)))


def confirmation_keys(winner):
    return (*stage1_keys(), *(ranking_key(arm, seed)
             for arm in (*BASELINES, *ARMS)
             for seed in (SEEDS if arm in (*BASELINES, winner) else SEEDS[:3])))


def reference_choice(cmd):
    if set(cmd) != set(REFERENCES) or not all(np.isfinite(list(cmd.values()))):
        raise ValueError("Both finite EAP-IG CMD scores are required")
    if abs(cmd["eapig_abs"] - cmd["eapig_signed"]) <= EPS:
        return "eapig_abs"
    return min(cmd, key=cmd.get)


def pilot_decision(losses, cmd, reference, correctness):
    losses = np.asarray(losses, dtype=float)
    finite = losses.shape == (5000,) and bool(np.isfinite(losses).all())
    first = float(losses[:500].mean()) if finite else None
    last = float(losses[-500:].mean()) if finite else None
    limit = 3 * reference
    checks = dict(all_5000_losses_finite=finite,
                  last_500_mean_strictly_lower=finite and last < first,
                  cmd_within_three_times_reference=cmd is not None and bool(np.isfinite(cmd) and cmd <= limit),
                  correctness_passed=bool(correctness))
    return dict(decision="continue" if all(checks.values()) else "stop", checks=checks,
                failures=[name for name, passed in checks.items() if not passed],
                first_500_mean=first, last_500_mean=last, cmd=cmd, limit=limit)


def choose_arm(results):
    if set(results) != set(ARMS):
        raise ValueError("Every registered arm is required")
    values = np.asarray(list(results.values()))
    if values.shape != (len(ARMS), 3) or not np.isfinite(values).all():
        raise ValueError("Every arm needs three finite CMD values")
    means = {arm: float(np.mean(v)) for arm, v in results.items()}
    best = min(means.values())
    tied = [arm for arm in ARMS if means[arm] - best < .001]
    preference = {"margin": 0, "repo_cmd": 1, "kl": 2}
    winner = min(tied, key=lambda a: (ARMS[a][1], preference[ARMS[a][0]], a))
    return dict(winner=winner, means=means, tied=tied, per_seed=results)


def classify(delta, individual, interval, valid=True):
    if not valid or len(individual) != 5 or not np.isfinite([delta, *individual, *interval]).all():
        return "Invalid"
    lower, upper = interval
    if delta < 0 and sum(x < 0 for x in individual) >= 4 and upper < 0:
        return "Lower CMD"
    if lower > 0:
        return "Higher CMD"
    if -.005 < lower <= upper < .005:
        return "Comparable"
    return "Inconclusive"


def relative_difference(delta, reference):
    return 100 * delta / reference if abs(reference) > EPS else None


def loss_bins(k, losses):
    k, losses = np.asarray(k), np.asarray(losses)
    if k.shape != losses.shape or len(k) == 0 or not ((0 <= k) & (k <= N)).all():
        raise ValueError("Invalid sparsity/loss log")
    edges = np.searchsorted([.01 * N, .10 * N], k, side="right")
    return [dict(bin=name, count=len(subset), fraction=len(subset) / len(k),
                 mean=float(subset.mean()) if len(subset) else None,
                 median=float(np.median(subset)) if len(subset) else None,
                 overall_mean=float(losses.mean()))
            for i, name in enumerate(("small", "medium", "large"))
            for subset in [losses[edges == i]]]


def paired_cmd(selected, comparator, clean, empty, templates, expected, draws=5000, seed=5647):
    """All seeds and comparators share prompt draws and recompute cohort means."""
    indices = resamples(templates, draws, seed)
    w = weights(indices, len(clean))
    clean, empty = np.asarray(clean, dtype=float), np.asarray(empty, dtype=float)
    actual = []
    bootstrap = []
    for matrices, point in zip((selected, comparator), expected, strict=True):
        matrices = np.asarray(matrices, dtype=float)
        if matrices.ndim != 3 or matrices.shape[-1] != len(clean):
            raise ValueError("Expected rankings × official grid × aligned prompts")
        points = official_areas(matrices.mean(-1).T, clean.mean(), empty.mean())[1]
        np.testing.assert_allclose(points, point, atol=2e-6, rtol=2e-6)
        actual.append(np.asarray(point, dtype=float))
        scores = [official_areas(m @ w.T, w @ clean, w @ empty)[1] for m in matrices]
        bootstrap.append(np.asarray(scores))
    per_seed = actual[0] - actual[1]
    delta = float(per_seed.mean())
    changes = (bootstrap[0] - bootstrap[1]).mean(axis=0)
    good = np.isfinite(changes)
    ci = np.quantile(changes[good], [.025, .975]).tolist() if good.any() else [None, None]
    return dict(delta=delta, relative=relative_difference(delta, float(actual[1].mean())),
                per_seed=per_seed.tolist(), interval=ci, unstable=int((~good).sum()),
                selected_mean=float(actual[0].mean()), comparator_mean=float(actual[1].mean())), indices
