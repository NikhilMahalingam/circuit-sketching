"""Small, reproducible intervention experiments with prompt-held-out evaluation.

All candidates learn shared coefficients from mean training-prompt score changes.
Validation prompts select hyperparameters. Test prompts are evaluated separately,
using the frozen selection. These are prediction experiments, not MIB scores.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Callable
import uuid

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .core import HeadScorer, PromptSpec


@dataclass(frozen=True)
class ExperimentConfig:
    prompts: list[PromptSpec]
    prompt_splits: dict[str, list[str]]
    head_sets: dict[str, list[tuple[int, int]]]
    seeds: tuple[int, ...] = (42, 43)
    n_train_masks: int = 64
    n_eval_masks: int = 24
    ablation_probability: float = 0.3
    direction_budgets: tuple[int, ...] = (8, 16, 32)
    epsilons: tuple[float, ...] = (0.05, 0.25)
    projection_dims: tuple[int, ...] = (8, 16)
    ridge_alpha: float = 1e-3


def _config_dict(config):
    return json.loads(json.dumps(asdict(config)))


def _digest(config):
    return hashlib.sha256(json.dumps(_config_dict(config), sort_keys=True).encode()).hexdigest()


def validate_config(config):
    ids = [p.id for p in config.prompts]
    if len(set(ids)) != len(ids) or any(not name for name in ids):
        raise ValueError("Prompt IDs must be nonempty and unique.")
    texts = [" ".join(p.text.split()).casefold() for p in config.prompts]
    if len(set(texts)) != len(texts) or any(not text for text in texts):
        raise ValueError("Prompt texts must be nonempty and unique across all splits.")
    if set(config.prompt_splits) != {"train", "validation", "test"}:
        raise ValueError("Provide exactly train, validation, and test prompt splits.")
    assigned = [p for split in config.prompt_splits.values() for p in split]
    if (any(not split for split in config.prompt_splits.values()) or
            len(set(assigned)) != len(assigned) or set(assigned) != set(ids)):
        raise ValueError("Every prompt must belong to exactly one nonempty split.")
    if not config.head_sets or not config.seeds or len(set(config.seeds)) != len(config.seeds):
        raise ValueError("Supply head sets and distinct seeds.")
    if any(isinstance(s, bool) or not isinstance(s, (int, np.integer)) or s < 0 for s in config.seeds):
        raise ValueError("Seeds must be nonnegative integers.")
    for value in (config.n_train_masks, config.n_eval_masks):
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError("Training and evaluation mask counts must be integers >= 2.")
    if not 0 < config.ablation_probability < 1:
        raise ValueError("Ablation probability must be strictly between zero and one.")
    if not np.isfinite(config.ridge_alpha) or config.ridge_alpha < 0:
        raise ValueError("ridge_alpha must be finite and nonnegative.")
    if (not config.direction_budgets or any(isinstance(k, bool) or not isinstance(k, int) or k < 1
                                          for k in config.direction_budgets)):
        raise ValueError("Direction budgets must be positive integers.")
    if not config.epsilons or any(not np.isfinite(e) or e <= 0 for e in config.epsilons):
        raise ValueError("Finite-difference epsilons must be positive and finite.")
    if not config.projection_dims or any(isinstance(k, bool) or not isinstance(k, int) or k < 1
                                        for k in config.projection_dims):
        raise ValueError("Projection dimensions must be positive integers.")
    for name, heads in config.head_sets.items():
        n = len(heads)
        if not name or n < 2 or len(set(map(tuple, heads))) != n:
            raise ValueError("Each named head set needs at least two distinct components.")
        if max(config.projection_dims) > n * (n - 1) // 2:
            raise ValueError(f"Projection dimension exceeds the number of pairs in {name}.")
        if config.n_train_masks + 2 * config.n_eval_masks > 2 ** n:
            raise ValueError(f"Too many distinct masks requested for {n} heads in {name}.")


def pair_features(masks):
    masks = np.asarray(masks, dtype=float)
    if masks.ndim != 2:
        raise ValueError("Masks must have shape [samples, heads].")
    pairs = list(itertools.combinations(range(masks.shape[1]), 2))
    return np.column_stack([masks[:, i] * masks[:, j] for i, j in pairs])


def sample_mask_splits(n, n_train, n_eval, probability, seed):
    """Seeded unique masks; zero is training-only, with no split overlap.

    For small systems, sample without replacement using Bernoulli weights.
    For large systems use bounded rejection sampling; never loop indefinitely.
    """
    total = n_train + 2 * n_eval
    if n < 2 or n_train < 2 or n_eval < 2 or total > 2 ** n or not 0 < probability < 1:
        raise ValueError("Invalid mask sampling configuration or exhausted mask space.")
    rng = np.random.default_rng(seed)
    if n <= 16:
        integers = np.arange(1, 2 ** n, dtype=np.uint64)
        pool = ((integers[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.int8)
        counts = pool.sum(axis=1)
        log_weights = counts * np.log(probability) + (n - counts) * np.log1p(-probability)
        weights = np.exp(log_weights - log_weights.max())
        weights /= weights.sum()
        chosen = rng.choice(len(pool), size=total - 1, replace=False, p=weights)
        all_masks = np.vstack([np.zeros((1, n), dtype=np.int8), pool[chosen]])
    else:
        rows = [(0,) * n]
        seen = set(rows)
        for _ in range(max(10000, 1000 * total)):
            row = tuple((rng.random(n) < probability).astype(np.int8))
            if row not in seen:
                rows.append(row)
                seen.add(row)
                if len(rows) == total:
                    break
        if len(rows) != total:
            raise ValueError("Could not draw enough unique masks; reduce counts or change probability.")
        all_masks = np.asarray(rows, dtype=np.int8)
    return {"train": all_masks[:n_train],
            "validation": all_masks[n_train:n_train + n_eval],
            "test": all_masks[n_train + n_eval:]}


def _metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Cannot report metrics for nonfinite scores.")
    residual = truth - prediction
    centered = truth - truth.mean()
    denominator = float(centered @ centered)
    varying = np.ptp(truth) > 0 and np.ptp(prediction) > 0
    return {"MAE": float(np.abs(residual).mean()),
            "R2": float(1 - residual @ residual / denominator) if denominator else np.nan,
            "Pearson": float(pearsonr(truth, prediction).statistic) if varying else np.nan,
            "Spearman": float(spearmanr(truth, prediction).statistic) if varying else np.nan}


def _fit(design, targets, alpha):
    # No intercept: the zero-ablation score change is exactly zero.
    if alpha:
        design = np.vstack([design, np.sqrt(alpha) * np.eye(design.shape[1])])
        targets = np.concatenate([targets, np.zeros(design.shape[1])])
    return np.linalg.lstsq(design, targets, rcond=None)[0]


@dataclass
class _Candidate:
    method: str
    name: str
    coefficients: np.ndarray
    acquisition_forwards: int
    projection: np.ndarray | None = None

    def predict(self, masks):
        design = np.asarray(masks, dtype=float)
        if self.method == "quadratic":
            design = np.column_stack([design, pair_features(design)])
        elif self.method == "projected_quadratic":
            design = np.column_stack([design, pair_features(design) @ self.projection])
        return design @ self.coefficients


@dataclass
class _Job:
    head_set: str
    seed: int
    scorer: HeadScorer
    masks: dict
    test_ids: tuple
    selected: list[_Candidate]


@dataclass
class ExperimentRun:
    config: ExperimentConfig
    tables: dict[str, pd.DataFrame]
    _jobs: list[_Job] = field(repr=False)
    _config_digest: str = field(repr=False)
    test_evaluated: bool = False
    metadata: dict = field(default_factory=dict)


def _stage_cost(scorer, before, head_set, seed, stage):
    after = scorer.cost_snapshot()
    row = {"head_set": head_set, "seed": seed, "stage": stage}
    for name in ("forward_count", "failed_forwards", "cache_hits", "forward_seconds"):
        row[name] = after[name] - before[name]
    for name in ("peak_allocated_mib", "peak_reserved_mib"):
        row[name] = after[name]
    return row


def _evaluate(scorer, ids, masks):
    return np.asarray([scorer.delta(prompt_id, masks) for prompt_id in ids])


def _evaluation_rows(head_set, seed, split, ids, masks, truth, candidates):
    metrics, predictions = [], []
    for candidate in candidates:
        pred = candidate.predict(masks)
        combined = np.tile(pred, len(ids))
        common = {"head_set": head_set, "seed": seed, "method": candidate.method,
                  "candidate": candidate.name}
        metrics.append({**common, **_metrics(truth.ravel(), combined), "split": split,
                        "n_prompts": len(ids), "n_masks": len(masks),
                        "standalone_acquisition_forwards": candidate.acquisition_forwards})
        for i, prompt_id in enumerate(ids):
            for j, mask in enumerate(masks):
                predictions.append({**common, "split": split, "prompt_id": prompt_id,
                                    "mask": json.dumps(mask.tolist()), "truth": truth[i, j],
                                    "prediction": pred[j]})
    return metrics, predictions


def _refresh_logs(run):
    rows = []
    for job in run._jobs:
        rows.extend({**record, "head_set": job.head_set, "seed": job.seed}
                    for record in job.scorer.records)
    run.tables["forward_log"] = pd.DataFrame(rows)


def run_development(model, config: ExperimentConfig, progress: Callable = print):
    """Fit on training prompts; select on validation; never score test prompts."""
    import copy
    config = copy.deepcopy(config)
    validate_config(config)
    tables = {name: [] for name in (
        "validation_metrics", "validation_predictions", "selections",
        "acquisition_diagnostics", "costs", "split_manifest", "masks",
        "training_targets", "model_parameters")}
    for split, ids in config.prompt_splits.items():
        tables["split_manifest"].extend({"split": split, "prompt_id": p} for p in ids)
    jobs = []
    train_ids = config.prompt_splits["train"]
    val_ids = config.prompt_splits["validation"]
    for head_set, heads in config.head_sets.items():
        n = len(heads)
        for seed in config.seeds:
            progress(f"{head_set}, seed {seed}: training on {len(train_ids)} prompts")
            scorer = HeadScorer(model, heads, config.prompts)
            masks = sample_mask_splits(n, config.n_train_masks, config.n_eval_masks,
                                       config.ablation_probability, seed)
            for split, rows in masks.items():
                tables["masks"].extend({"head_set": head_set, "seed": seed, "split": split,
                                        "mask": json.dumps(row.tolist())} for row in rows)
            before = scorer.cost_snapshot()
            scorer.stage = "binary training acquisition"
            y = _evaluate(scorer, train_ids, masks["train"])
            tables["costs"].append(_stage_cost(scorer, before, head_set, seed, scorer.stage))
            target = y.mean(axis=0)
            for i, mask in enumerate(masks["train"]):
                tables["training_targets"].append({"head_set": head_set, "seed": seed,
                                                   "mask": json.dumps(mask.tolist()),
                                                   "mean_score_change": target[i]})
            X = masks["train"].astype(float)
            Q = pair_features(X)
            # Each training prompt's zero mask supplies the baseline, once.
            binary_budget = len(train_ids) * len(X)
            candidates = []
            designs = [("linear", "linear", X, None),
                       ("quadratic", "quadratic", np.column_stack([X, Q]), None)]
            projection_rng = np.random.default_rng(np.random.SeedSequence([seed, 100]))
            for size in sorted(set(config.projection_dims)):
                projection = projection_rng.normal(size=(Q.shape[1], size)) / np.sqrt(size)
                designs.append(("projected_quadratic", f"projection={size}",
                                np.column_stack([X, Q @ projection]), projection))
            for method, name, design, projection in designs:
                candidate = _Candidate(method, name, _fit(design, target, config.ridge_alpha),
                                       binary_budget, projection)
                candidates.append(candidate)
                tables["acquisition_diagnostics"].append({"head_set": head_set, "seed": seed,
                    "method": method, "candidate": name, "rank": int(np.linalg.matrix_rank(design)),
                    "features": design.shape[1], "condition_number": float(np.linalg.cond(design)),
                    "standalone_acquisition_forwards": binary_budget,
                    "ridge_alpha": config.ridge_alpha})

            direction_rng = np.random.default_rng(np.random.SeedSequence([seed, 200]))
            directions = direction_rng.choice([-1.0, 1.0], size=(max(config.direction_budgets), n))
            for epsilon in sorted(set(config.epsilons)):
                progress(f"{head_set}, seed {seed}: directional measurements, epsilon={epsilon:g}")
                before = scorer.cost_snapshot()
                scorer.stage = f"directional acquisition epsilon={epsilon:g}"
                measurements = []
                for direction in directions:
                    slopes = [(scorer.score(pid, 1 + epsilon * direction) -
                               scorer.score(pid, 1 - epsilon * direction)) / (2 * epsilon)
                              for pid in train_ids]
                    measurements.append(float(np.mean(slopes)))
                tables["costs"].append(_stage_cost(scorer, before, head_set, seed, scorer.stage))
                for k in sorted(set(config.direction_budgets)):
                    gradient, _, rank, _ = np.linalg.lstsq(directions[:k], measurements[:k], rcond=None)
                    points = {tuple(np.ones(n))}
                    for direction in directions[:k]:
                        points.add(tuple(1 + epsilon * direction))
                        points.add(tuple(1 - epsilon * direction))
                    budget = len(train_ids) * len(points)
                    name = f"epsilon={epsilon:g}, directions={k}"
                    # Masks indicate removal, the negative of increasing head scale.
                    candidates.append(_Candidate("directional", name, -gradient, budget))
                    tables["acquisition_diagnostics"].append({"head_set": head_set, "seed": seed,
                        "method": "directional", "candidate": name, "rank": int(rank),
                        "features": n, "condition_number": float(np.linalg.cond(directions[:k])),
                        "standalone_acquisition_forwards": budget, "epsilon": epsilon,
                        "directions": k, "nominal_endpoint_forwards": 2 * k * len(train_ids)})

            progress(f"{head_set}, seed {seed}: evaluating validation prompts")
            before = scorer.cost_snapshot()
            scorer.stage = "validation evaluation"
            truth = _evaluate(scorer, val_ids, masks["validation"])
            tables["costs"].append(_stage_cost(scorer, before, head_set, seed, scorer.stage))
            rows, predictions = _evaluation_rows(head_set, seed, "validation", val_ids,
                                                 masks["validation"], truth, candidates)
            selected = []
            for method in sorted({c.method for c in candidates}):
                best = min((row for row in rows if row["method"] == method),
                           key=lambda row: (row["MAE"], row["standalone_acquisition_forwards"], row["candidate"]))
                selected.append(next(c for c in candidates
                                     if c.method == method and c.name == best["candidate"]))
                tables["selections"].append({"head_set": head_set, "seed": seed, "method": method,
                    "candidate": best["candidate"], "validation_MAE": best["MAE"]})
            selected_keys = {(c.method, c.name) for c in selected}
            for row in rows + predictions:
                row["selected"] = (row["method"], row["candidate"]) in selected_keys
            tables["validation_metrics"].extend(rows)
            tables["validation_predictions"].extend(predictions)
            for candidate in candidates:
                tables["model_parameters"].append({"head_set": head_set, "seed": seed,
                    "method": candidate.method, "candidate": candidate.name,
                    "coefficients": json.dumps(candidate.coefficients.tolist()),
                    "projection": json.dumps(candidate.projection.tolist()) if candidate.projection is not None else "null",
                    "selected": (candidate.method, candidate.name) in selected_keys})
            jobs.append(_Job(head_set, seed, scorer, masks,
                             tuple(config.prompt_splits["test"]), selected))
    run = ExperimentRun(config, {name: pd.DataFrame(rows) for name, rows in tables.items()},
                        jobs, _digest(config))
    run.metadata.update({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                         "target": "prompt-held-out binary-mask score changes",
                         "selection": "validation MAE per method, head set, and seed",
                         "test_evaluated": False,
                         "cost_scope": "model forwards only; excludes loading, fitting, plotting",
                         "peak_memory_scope": "process CUDA peak, including model weights",
                         "standalone_cost": "distinct training-prompt endpoints including baseline; excludes evaluation and hyperparameter search"})
    _refresh_logs(run)
    return run


def evaluate_test(run: ExperimentRun, progress: Callable = print):
    """Evaluate the frozen candidates once, without additional fitting or selection."""
    if run.test_evaluated:
        raise RuntimeError("Test data were already evaluated for this run. Keep them out of tuning.")
    if _digest(run.config) != run._config_digest:
        raise ValueError("Configuration changed after selection; start a new development run.")
    # Mark before accessing test scores, so a partial failure cannot silently repeat it.
    run.test_evaluated = True
    metrics, predictions, costs = [], [], []
    for job in run._jobs:
        progress(f"{job.head_set}, seed {job.seed}: final held-out test prompts")
        before = job.scorer.cost_snapshot()
        job.scorer.stage = "final test evaluation"
        truth = _evaluate(job.scorer, job.test_ids, job.masks["test"])
        costs.append(_stage_cost(job.scorer, before, job.head_set, job.seed, job.scorer.stage))
        rows, preds = _evaluation_rows(job.head_set, job.seed, "test", job.test_ids,
                                      job.masks["test"], truth, job.selected)
        metrics.extend(rows)
        predictions.extend(preds)
    run.tables["test_metrics"] = pd.DataFrame(metrics)
    run.tables["test_predictions"] = pd.DataFrame(predictions)
    run.tables["costs"] = pd.concat([run.tables["costs"], pd.DataFrame(costs)], ignore_index=True)
    run.metadata["test_evaluated"] = True
    _refresh_logs(run)
    return run


def export_run(run: ExperimentRun, output_root=Path("results") / "experiments"):
    """Export actual tables and full configuration to a fresh directory."""
    if _digest(run.config) != run._config_digest:
        raise ValueError("Configuration changed after the experiment; cannot export misleading metadata.")
    directory = Path(output_root) / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    for name, frame in run.tables.items():
        frame.to_csv(directory / f"{name}.csv", index=False)
    payload = {"configuration": _config_dict(run.config), "metadata": run.metadata,
               "configuration_sha256": run._config_digest}
    (directory / "configuration.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return directory
