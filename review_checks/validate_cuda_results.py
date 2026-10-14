"""Check real notebook exports after run_notebook.py succeeds."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
execution = json.loads((ROOT / "review_checks/cuda_execution.json").read_text())
run_dir = Path(execution["results_directory"])
config = json.loads((run_dir / "configuration.json").read_text())
assert config["device"] == "cuda"
assert execution["executed_code_cells"] == 23
train = pd.read_csv(run_dir / "training_masks.csv")
test = pd.read_csv(run_dir / "heldout_predictions.csv")
singles = pd.read_csv(run_dir / "single_effects.csv")
pairs = pd.read_csv(run_dir / "pairwise_interactions.csv")
summary = pd.read_csv(run_dir / "experiment_summary.csv")
stages = pd.read_csv(run_dir / "stage_costs.csv")
totals = pd.read_csv(run_dir / "total_costs.csv").iloc[0]
forwards = pd.read_csv(run_dir / "forward_log.csv")
directions = pd.read_csv(run_dir / "directional_measurements.csv")

train_keys = {tuple(json.loads(value)) for value in train["mask"]}
test_keys = {tuple(json.loads(value)) for value in test["mask"]}
assert len(train_keys) == len(train) == config["n_train"]
assert len(test_keys) == len(test) == config["n_test"]
assert train_keys.isdisjoint(test_keys)
n = len(config["components"])
assert len(singles) == n
assert set(zip(pairs["i"], pairs["j"])) == set(itertools.combinations(range(n), 2))
assert len(pairs) == n * (n - 1) // 2

metrics_columns = {
    "first-order regression": "first_order_prediction",
    "explicit second-order regression": "second_order_prediction",
    "toy directional sketch": "toy_sketch_prediction",
}
truth = test["score_change"].to_numpy()
for row in summary.itertuples():
    prediction = test[metrics_columns[row.method]].to_numpy()
    np.testing.assert_allclose(
        [row.MAE, row.Pearson, row.Spearman, row.R2],
        [mean_absolute_error(truth, prediction), stats.pearsonr(truth, prediction)[0],
         stats.spearmanr(truth, prediction)[0], r2_score(truth, prediction)],
        rtol=1e-8, atol=1e-10,
    )

# Derive forward costs independently from unique intervention points.
binary_keys = {(0,) * n} | train_keys | test_keys
for indices in [(i,) for i in range(n)] + list(itertools.combinations(range(n), 2)):
    mask = [0] * n
    for index in indices:
        mask[index] = 1
    binary_keys.add(tuple(mask))
scaling_keys = {(0.0,) * n}
for encoded in directions["direction"]:
    direction = np.asarray(json.loads(encoded))
    scaling_keys.add(tuple(config["epsilon"] * direction))
    scaling_keys.add(tuple(-config["epsilon"] * direction))
if config["local_gradient_check"]:
    for index in range(n):
        offset = np.zeros(n)
        offset[index] = config["epsilon"]
        scaling_keys.add(tuple(offset))
        scaling_keys.add(tuple(-offset))
expected_forwards = len(binary_keys) + len(scaling_keys) - 1 + 3
assert len(binary_keys) == totals["distinct_binary_masks_evaluated"]
assert len(scaling_keys) == totals["distinct_scaling_points_evaluated"]
assert expected_forwards == totals["model_evaluations"] == len(forwards) == stages.model_evaluations.sum()
assert forwards["success"].all()
assert totals["peak_allocated_mib"] > 0
assert len(list(run_dir.glob("*.png"))) == 5
report = {
    "device": config["device"], "actual_forwards": int(totals["model_evaluations"]),
    "independently_expected_forwards": expected_forwards,
    "unique_binary_masks": len(binary_keys), "unique_scaling_points": len(scaling_keys),
    "train_test_overlap": len(train_keys & test_keys), "heldout_metrics_recomputed": True,
    "peak_allocated_mib": float(totals["peak_allocated_mib"]),
    "peak_reserved_mib": float(totals["peak_reserved_mib"]),
    "forward_seconds": float(totals["forward_seconds"]), "png_plots": 5,
}
(ROOT / "review_checks/cuda_results_validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
print(summary.to_string(index=False))
