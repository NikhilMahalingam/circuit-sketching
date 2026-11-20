"""CPU-only interpretation of completed, frozen Gate 6A artifacts."""

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CAMPAIGN = Path(__file__).resolve().parent
ROOT = CAMPAIGN.parents[2]


def read(name):
    return json.loads((CAMPAIGN / name).read_text())


def markdown(frame):
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def main():
    metrics = pd.read_csv(CAMPAIGN / "metrics.csv")
    curves = pd.read_csv(CAMPAIGN / "faithfulness_curves.csv")
    costs = pd.read_csv(CAMPAIGN / "compute_accounting.csv")
    decision = read("decision.json")
    audit = read("final_audit.json")
    selection = read("selection.json")
    assert audit["passed"] and decision["classification"] == "Inconclusive" and selection["winner"] == "M1"
    for name, expected in audit["artifact_sha256"].items():
        assert hashlib.sha256((CAMPAIGN / name).read_bytes()).hexdigest() == expected, name
    confirmation = metrics.query("role == 'confirmation'")
    selected = confirmation.query("arm == 'M1'").sort_values("seed")
    reference = confirmation.query("arm == 'eapig_abs'").iloc[0]
    effects = selected.cmd.to_numpy() - reference.cmd
    np.testing.assert_allclose(effects, decision["primary"]["per_seed"], rtol=1e-12, atol=1e-12)

    # Independently replay the frozen prompt draws and linear CMD quadrature.
    indices = np.load(CAMPAIGN / "bootstrap_indices.npy", allow_pickle=False)
    weights = np.zeros((len(indices), indices.shape[1]), dtype=float)
    for i, row in enumerate(indices):
        weights[i] = np.bincount(row, minlength=indices.shape[1]) / indices.shape[1]
    grid = np.array([.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.])
    dx = np.diff(grid)

    def bootstrap(key):
        with np.load(CAMPAIGN / "evaluations/confirmation" / key / "vectors.npz") as data:
            clean, empty = weights @ data["clean"].astype(float), weights @ data["empty"].astype(float)
            assert (np.abs(clean - empty) >= 1e-8).all()
            f = (data["grid_margins"].astype(float) @ weights.T - empty) / (clean - empty)
        gaps = np.abs(1 - f)
        return (dx[:, None] * (gaps[:-1] + gaps[1:]) / 2).sum(axis=0)

    samples = np.mean([bootstrap(key) for key in selected.ranking], axis=0) - bootstrap("eapig_abs")
    np.testing.assert_allclose(np.quantile(samples, [.025, .975]), decision["primary"]["interval"], atol=1e-12)
    assert int(costs.forward_batches.sum()) == audit["forward_batches"]
    assert int(costs.example_forwards.sum()) == audit["example_forwards"]

    labels = {"B0": "Ordinary MAttr, signed", "B1": "H2 logit-difference, signed", "M0": "Squared margin, ordinary",
              "M1": "Squared margin, H2 (selected)", "K0": "KL, ordinary", "K1": "KL, H2", "R0": "Upstream CMD surrogate",
              "eapig_abs": "EAP-IG absolute (reference)", "eapig_signed": "EAP-IG signed", "mattr_abs": "Raw MAttr absolute"}
    order = list(labels)
    summary = confirmation.groupby("arm").agg(seeds=("cmd", "size"), cmd=("cmd", "mean"), cpr=("cpr", "mean"), accuracy_log_auc=("accuracy_log_auc", "mean")).reindex(order).reset_index()
    summary.to_csv(CAMPAIGN / "review_method_summary.csv", index=False)
    display = summary.copy()
    display["arm"] = display.arm.map(labels)
    for col in ("cmd", "cpr", "accuracy_log_auc"):
        display[col] = display[col].map(lambda x: f"{x:.6f}")
    seed_table = selected[["seed", "cmd", "cpr", "accuracy_log_auc"]].copy()
    seed_table["delta_cmd_vs_reference"] = effects
    seed_table.to_csv(CAMPAIGN / "review_selected_seeds.csv", index=False)
    seed_table["seed"] = seed_table.seed.astype(int)
    for col in seed_table.columns[1:]:
        seed_table[col] = seed_table[col].map(lambda x: f"{x:.6f}")
    compact = curves.query("role == 'confirmation' and selection == 'cap'").groupby(["arm", "target"], as_index=False)[["actual_edges", "accuracy", "kl"]].mean()
    compact.to_csv(CAMPAIGN / "review_compact_summary.csv", index=False)
    fidelity = curves.query("role == 'confirmation' and selection == 'grid' and target in [0.1, 0.2, 0.5]").groupby(["arm", "target"], as_index=False)[["kl", "accuracy", "sign_agreement", "top1_agreement", "actual_edges"]].mean()
    fidelity.to_csv(CAMPAIGN / "review_distribution_fidelity.csv", index=False)
    cap116 = compact[compact.target == 116].set_index("arm")
    cap_table = pd.DataFrame({"Method": [labels[k] for k in ("B0", "B1", "M1", "eapig_abs", "eapig_signed")],
                              "Mean actual edges": [f"{cap116.loc[k, 'actual_edges']:.1f}" for k in ("B0", "B1", "M1", "eapig_abs", "eapig_signed")],
                              "IOI accuracy": [f"{100*cap116.loc[k, 'accuracy']:.2f}%" for k in ("B0", "B1", "M1", "eapig_abs", "eapig_signed")]})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    axes[0].plot(selected.seed, selected.cmd, "o-", label="Margin matching + H2")
    axes[0].axhline(reference.cmd, color="black", linestyle="--", label="EAP-IG absolute")
    axes[0].set(xlabel="Training seed", ylabel="Official CMD (lower is better)", title="Frozen confirmation comparison", xticks=range(42, 47))
    axes[0].legend(fontsize=8)
    for arm in ("B0", "B1", "M1", "eapig_abs", "eapig_signed"):
        rows = compact[compact.arm == arm].sort_values("target")
        axes[1].plot(rows.target, 100*rows.accuracy, "o-", label=labels[arm])
    axes[1].set(xlabel="Actual-edge cap", ylabel="IOI accuracy (%)", xscale="log", title="Compact-circuit accuracy")
    axes[1].legend(fontsize=7)
    fig.savefig(CAMPAIGN / "figures/review_confirmation.png", dpi=170)
    plt.close(fig)
    training = costs.job.str.match(r"(?:M[01]|K[01]|R0)_s\d+_5000$")
    evaluation = costs.job.str.startswith(("development_", "confirmation_"))
    hours = costs.seconds.sum() / 3600
    peaks = costs[["peak_allocated_mib", "peak_reserved_mib", "sampled_peak_cpu_rss_mib"]].max()
    dev = pd.DataFrame({"Arm": list(selection["means"]), "Development mean CMD": [f"{v:.6f}" for v in selection["means"].values()]})
    text = f'''# Gate 6A results review

The registered outcome is **Inconclusive**. The selected method, squared margin matching with the H2 destination hierarchy (M1), reached mean confirmation CMD **0.027343**, compared with **0.028605** for the frozen absolute-order EAP-IG reference. The difference is **−0.001262 (−4.41%)**, with a paired prompt-bootstrap 95% interval **[−0.005151, +0.002936]**. Three of five seeds improved. The registered success rule requires four improving seeds and an interval entirely below zero. The interval also extends beyond the frozen ±0.005 equivalence band, so the classification remains Inconclusive.

Changing the training objective substantially reduced ordinary MAttr's margin distortion. Whether it reliably beats the strongest EAP-IG CMD reference remains unresolved.

## Completed experiment and verification

The campaign trained all five registered arms on seeds 42–44, then extended M1 to seeds 45–46. That is **17 trajectories and 85,000 registered optimizer updates**. Existing ordinary and H2 logit-difference rankings and full-data EAP-IG scores were reused. Screening used the historical 512-prompt development cohort. Confirmation used a separate 512-prompt cohort across nine exact templates. All evaluated metrics below are from confirmation unless marked development.

The pilot passed all four frozen conditions. Mean margin loss decreased from 0.086699 over the first 500 updates to 0.007048 over the last 500. Its development CMD was 0.018047, below the frozen 0.088879 limit. The cached discovery normalizer was D = 5.560939789.

The official audit passed. This review independently checked its artifact hashes, recomputed all five selected-seed differences, replayed the saved paired bootstrap draws and CMD quadrature, and reconciled the recorded root forwards and example-forwards. The signed/absolute ranking distinction is material. On development, EAP-IG CMD was 1.032934 under signed ordering and 0.029626 under absolute ordering. The preregistered rule froze the latter before new training results.

Training protocol SHA-256 `{audit['training_sha256']}`.

Confirmation protocol SHA-256 `{audit['confirmation_sha256']}`.

## Development selection

{markdown(dev)}

M1 won development screening with CMD 0.023928. Its five-seed confirmation mean rose to 0.027343. This change combines a new prompt cohort with two additional training seeds. It cannot be assigned entirely to prompt generalization. M0 and K0 have slightly lower descriptive confirmation means on their three trained seeds, but the frozen winner remains M1.

## Confirmation metrics

{markdown(display)}

The reported CPR uses each row's stated circuit ordering. Ordinal exports preserve signed learned scores, while the two EAP-IG conventions and raw-MAttr absolute diagnostic produce different circuits. Ordinary MAttr's high CPR accompanies CMD near 9.8 and large distributional distortion. Margin matching moves normalized faithfulness much closer to 1. H2 margin training gives accuracy log-AUC 0.873868 versus 0.750556 for the CMD-selected EAP-IG reference and 0.898611 for signed EAP-IG.

## Selected method by seed

{markdown(seed_table)}

![Confirmation CMD and compact accuracy](figures/review_confirmation.png)

Seeds 43, 44 and 45 beat the fixed EAP-IG reference. Seeds 42 and 46 regress. Seed 45 has accuracy log-AUC 0.733580 despite a competitive CMD. A low aggregate CMD therefore leaves meaningful compact-circuit differences.

## Compact circuits and full-distribution fidelity

{markdown(cap_table)}

At the 116-edge cap, M1 improves positive-margin IOI accuracy by **11.91 percentage points** over ordinary MAttr, falls **7.07 points** below historical H2 logit-difference training, and falls **17.85 points** below signed EAP-IG. It exceeds absolute-order EAP-IG by **43.09 points**. Actual retained sizes vary as shown, so these are capped-budget comparisons.

At nominal 10%, 20%, and 50% retention, M1's full-vocabulary KL is **0.051842, 0.033516, and 0.011776**, versus **24.414367, 25.206347, and 25.218686** for ordinary signed MAttr. Absolute-order EAP-IG remains lower at **0.042166, 0.029401, and 0.006766**. Ordinary KL training gives **0.026321, 0.014974, and 0.002846** across its three seeds. Those are descriptive comparisons across unequal numbers of trained seeds.

M1's CMD improves by **93.31%** relative to the five original raw-score absolute MAttr rankings (0.408547 mean CMD). The paired interval for that difference is [−0.388737, −0.373197]. Raw-score absolute ordering depends on score centering, and these archived checkpoints do not independently reproduce a private leaderboard submission.

Cross-seed full-edge rank correlation averaged **0.6884** for M1, compared with **0.4699** for ordinary MAttr and **0.4597** for H2 logit-difference training. Greater ranking agreement accompanies lower distortion here. It does not establish that the selected components better match a ground-truth causal circuit.

## Compute and limitations

Recorded attempt wall time totals **{hours:.2f} hours**, comprising **{costs.loc[training, 'seconds'].sum()/3600:.2f} hours** for new training processes and **{costs.loc[evaluation, 'seconds'].sum()/3600:.2f} hours** for evaluation processes, plus discovery targets and bounded checks. The records contain **234,815 root forward passes**, **1,490,103 example-forwards**, and **85,510 backward calls/optimizer updates**. The extra 510 updates belong to bounded checks. Peak CUDA allocated memory was **{peaks.peak_allocated_mib/1024:.2f} GiB**, reserved memory **{peaks.peak_reserved_mib/1024:.2f} GiB**, and sampled CPU RSS **{peaks.sampled_peak_cpu_rss_mib/1024:.2f} GiB**. These are process timers and allocator/RSS measurements. Reused baseline training and original EAP-IG attribution costs are outside the new campaign total.

The bootstrap conditions on the fixed trained rankings and template mixture. Its interval captures prompt sampling, with zero unstable denominator draws. It does not estimate training-seed population uncertainty. The model/task scope is GPT-2 Small IOI. The new losses use the original optimizer settings, development was historically inspected, and Gate 1 previously evaluated the full validation split. Confirmation is fresh relative to recent gates. Public-test and leaderboard comparisons remain outside these results.

## Interpretation and recommendation (researcher-written)

Faithfulness training addresses the margin-amplification behavior that motivated Gate 6A and produces a useful tradeoff between CMD, distributional KL and compact accuracy. This campaign does not satisfy its registered EAP-IG superiority criterion. The distinction between absolute and signed rankings explains a substantial part of the apparent advantage over the earlier signed baselines.

The next research decision should preserve absolute-order EAP-IG as the CMD comparator and signed EAP-IG as a compact-accuracy comparator. KL training merits consideration when full-distribution fidelity is the objective. Any new hypothesis or expanded seed comparison should be frozen separately. No follow-up experiments were launched by this review.

The original generated report, registered classification, protocols and audit remain preserved. This separate report supplies the interpretation requested after execution.
'''
    (CAMPAIGN / "GATE6A_RESULTS_REPORT.md").write_text(text, encoding="utf-8")
    root_text = text.replace("](figures/", "](results/mattr_faithful/gate6a/figures/")
    (ROOT / "GATE6A_RESULTS_REPORT.md").write_text(root_text, encoding="utf-8")
    provenance = {"original_audit_sha256": hashlib.sha256((CAMPAIGN / "final_audit.json").read_bytes()).hexdigest(),
                  "independent_bootstrap_replay": True, "model_forwards": 0,
                  "registered_classification": decision["classification"]}
    (CAMPAIGN / "review_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(text)


if __name__ == "__main__":
    main()
