"""Deterministic CPU analysis, report generation, and final artifact audit."""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.mattr_fidelity_splice.analysis import table
from src.mattr_fidelity_splice.core import read, verified
from src.mattr_gate5g_addendum.common import checked_json, freeze_json
from src.mattr_optimization.common import ROOT, sha, write_json
from src.mattr_matched_size.topology import graph_new
from .core import CONFIRMATION, HISTORY, TRAINING, protocol, reference
from .decisions import ARMS, SEEDS, classify, loss_bins, paired_cmd, ranking_key
from .evaluate import check_result, rank
from .pipeline import confirmation, linked, stage1_reuse
from .preflight import order

REPORT = "GATE6A_FAITHFUL_MATTR_REPORT.md"
LIMITATIONS = """## Limitations

- All registered experiments use GPT-2 Small on IOI. Generalization to other tasks and models remains untested.
- New losses reuse the original optimizer configuration. It may be suboptimal for margin matching or KL.
- The selected EAP-IG reference is one fixed ranking. The experiment measures no variation across attribution runs.
- The development cohort was previously inspected. Confirmation is disjoint from the recent cohorts, with historical exposure through Gate 1's full validation.
- Published and leaderboard CMD values come from different splits and may use different score-selection settings.
- Prompt-bootstrap intervals condition on the fixed rankings, training seeds, and template mixture.
- The margin objective matches per-example margins during soft-mask training. Official CMD measures aggregate margins on discrete, connectivity-pruned circuits.
"""


def comparison(out, c, scores):
    cohort = pd.read_csv(out / "confirmation.csv")

    def arrays(key):
        with np.load(out / "evaluations/confirmation" / key / "vectors.npz") as data:
            return tuple(data[k] for k in ("grid_margins", "clean", "empty"))

    winner = [ranking_key(c["selected"], seed) for seed in SEEDS]
    absolute = [ranking_key("mattr_abs", seed) for seed in SEEDS]
    _, clean, empty = arrays(c["reference"])
    for key in [*winner, *absolute]:
        _, a, b = arrays(key)
        np.testing.assert_array_equal(clean, a)
        np.testing.assert_array_equal(empty, b)

    def compare(comparator):
        expected = [[scores[key]["cmd"] for key in keys] for keys in (winner, comparator)]
        a, b = ([arrays(key)[0] for key in keys] for keys in (winner, comparator))
        return paired_cmd(a, b, clean, empty, cohort.template_id, expected, **{
            key: c["bootstrap"][key] for key in ("draws", "seed")})

    primary, indices = compare([c["reference"]])
    diagnostic, second = compare(absolute)
    np.testing.assert_array_equal(indices, second)
    label = classify(primary["delta"], primary["per_seed"], primary["interval"],
                     valid=primary["interval"][0] is not None)
    diagnostic["mean_absolute_per_seed_difference"] = float(np.abs(diagnostic["per_seed"]).mean())
    comparator_cmd = np.array([scores[key]["cmd"] for key in absolute])
    if np.all(np.abs(comparator_cmd) > 1e-8):
        diagnostic["mean_relative_per_seed_difference"] = float(np.mean(100 * np.array(diagnostic["per_seed"]) / comparator_cmd))
    else:
        diagnostic["mean_relative_per_seed_difference"] = None
    return dict(classification=label, primary=primary, mattr_abs=diagnostic), indices


def compute(out):
    rows = []
    for path in sorted((out / "attempts").glob("*/*/attempt.json")):
        attempt = read(path)
        if attempt["key"].startswith("command_"):
            continue
        events = path.parent / "forward_events.jsonl"
        import json
        data = [json.loads(line) for line in events.read_text().splitlines()] if events.exists() else []
        ratefile = path.parent / "rates.json"
        rate = read(ratefile) if ratefile.exists() else {}
        meter = path.parent / "compute_accounting.csv"
        counts = pd.read_csv(meter) if meter.exists() and meter.stat().st_size > 3 else pd.DataFrame()
        row = dict(job=attempt["key"], attempt=path.parent.name, status=attempt["status"],
                   seconds=attempt.get("seconds", 0.), forward_batches=len(data),
                   example_forwards=sum(r["batch_size"] for r in data),
                   cache_hits=rate.get("cache_hits", 0), cache_misses=rate.get("cache_misses", 0))
        for key in ("backward_calls", "optimizer_updates"):
            row[key] = int(counts[key].sum()) if key in counts else 0
        for key in ("peak_allocated_mib", "peak_reserved_mib", "sampled_peak_cpu_rss_mib"):
            row[key] = float(counts[key].max()) if key in counts else None
        if len(counts):
            assert row["forward_batches"] == counts.forward_batches.sum()
            assert row["example_forwards"] == counts.example_forwards.sum()
        rows.append(row)
    return pd.DataFrame(rows)


def structural(out, p, c):
    graph = graph_new()
    mapping = pd.read_csv(out / "edge_mapping.csv", keep_default_na=False)
    rankings = {}
    for key in c["rankings"]:
        scores, _ = rank(out, key, p)
        rankings[key] = order(graph, scores, key.startswith("mattr_abs_"))
    known = read(HISTORY / "known_ioi_heads.json")
    known_heads = {f"a{layer}.h{head}" for nodes in known["classes"].values() for layer, head in nodes}
    comp, stable = [], []
    for key, ranked in rankings.items():
        if key.split("_")[0] not in ARMS:
            continue
        seed = key.split("_")[1][1:]
        for size in (324, 3249):
            top = set(ranked[:size])
            rows = mapping.iloc[ranked[:size]]
            source = rows.parent.astype(str)
            dest = rows.child.astype(str)
            heads = set(source) | set(dest)
            receivers = dest + rows.qkv.map(lambda port: f"<{port}>" if port else "")
            for comparison_key in (c["reference"], f"B0_s{seed}_5000"):
                comp_size = 324 if comparison_key == c["reference"] else size
                other = set(rankings[comparison_key][:comp_size])
                comp.append(dict(ranking=key, comparator=comparison_key, top=size,
                                 jaccard=len(top & other)/len(top | other),
                                 known_head_count=len(heads & known_heads),
                                 destination_counts=receivers.value_counts().to_json()))
    for arm in (*ARMS, "B0", "B1"):
        keys = [key for key in rankings if key.startswith(arm + "_")]
        for a, b in combinations(keys, 2):
            ra, rb = np.argsort(rankings[a]), np.argsort(rankings[b])
            stable.append(dict(arm=arm, first=a, second=b, spearman=float(np.corrcoef(ra, rb)[0, 1])))
    return pd.DataFrame(comp), pd.DataFrame(stable)


def save_plot(figures, name, frame, x, y, group=None, *, log_x=False, scatter=False):
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    for label, rows in frame.groupby(group) if group else [(None, frame)]:
        rows = rows.sort_values(x)
        if scatter:
            ax.scatter(rows[x], rows[y], label=label, alpha=.7)
        else:
            ax.plot(rows[x], rows[y], ".-", label=label)
    if group and frame[group].nunique() <= 12:
        ax.legend(fontsize=8)
    if log_x:
        ax.set_xscale("log")
    ax.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "))
    fig.savefig(figures / f"{name}.png", dpi=150)
    plt.close(fig)


def analyze(out):
    p = protocol(out)
    if (out / "final_audit.json").exists():
        from src.mattr_fidelity_splice.core import check_files
        check_files(checked_json(out / "final_audit.json")["artifact_sha256"], out)
        return read(out / "decision.json")
    pilot = linked(out, "pilot_decision")
    if pilot["decision"] == "stop":
        report_text(out, p, None, pilot, None, None)
        return {"decision": "stop", **pilot}
    c = confirmation(out, p)
    stage1_reuse(out, p)
    summary = {}
    metrics, curves, ratios, logs, bins = [], [], [], [], []
    for role, keys in (("development", p["development_jobs"]), ("confirmation", c["rankings"])):
        for key in keys:
            s = check_result(out, p, role, key)
            if s is None:
                report_text(out, p, c, pilot, {"classification": "Invalid", "missing": f"{role}/{key}"}, None)
                raise ValueError(f"Missing registered result {role}/{key}")
            if role == "confirmation":
                summary[key] = s
            parts = key.rsplit("_", 2)
            if len(parts) == 3 and parts[-2][1:].isdigit() and parts[-1].isdigit():
                arm, seed, step = parts[0], int(parts[-2][1:]), int(parts[-1])
            else:
                arm, seed, step = key, None, 0
            meta = dict(role=role, ranking=key, arm=arm, step=step, seed=seed)
            metrics.append(dict(**meta, **{k:s[k] for k in ("cmd", "cpr", "accuracy_log_auc", "faithfulness_above_one")},
                                raw_score_stats=s["raw_score_stats"]))
            curves.extend(dict(**meta, **point) for point in s["points"])
            with np.load(out / "evaluations" / role / key / "vectors.npz") as vectors:
                for point in s["points"]:
                    name = f"{point['selection']}_{point['target']}_ratio"
                    ratios.extend(dict(**meta, selection=point["selection"], target=point["target"], ratio=v)
                                  for v in vectors[name])
    decision, indices = comparison(out, c, summary)
    np.save(out / "bootstrap_indices.npy", indices, allow_pickle=False)
    write_json(out / "decision.json", decision)
    comp, stable = structural(out, p, c)
    for folder in sorted((out / "training").iterdir()):
        frame = pd.read_csv(folder / "steps.csv", float_precision="round_trip")
        arm, seed, _ = folder.name.split("_")
        logs.append(frame.assign(arm=arm, seed=int(seed[1:])))
        for label, block in (("1000", frame.iloc[:1000]), ("5000", frame),
                             ("first_500", frame.iloc[:500]), ("last_500", frame.iloc[-500:])):
            bins.extend(dict(arm=arm, seed=int(seed[1:]), checkpoint=label, **row)
                        for row in loss_bins(block.k, block.loss))
    for key, rec in checked_json(out / "graph_preflight.json")["pairing"].items():
        import torch
        arm, seed, step = key.split("_")
        if step != "5000":
            continue
        log = torch.load(ROOT / rec["checkpoint"], weights_only=False, map_location="cpu")["train_log"]
        frame = pd.DataFrame(log, columns=["step", "k", "fraction", "loss"])
        for label, block in (("1000", frame.iloc[:1000]), ("5000", frame),
                             ("first_500", frame.iloc[:500]), ("last_500", frame.iloc[-500:])):
            bins.extend(dict(arm=arm, seed=int(seed[1:]), checkpoint=label, **row)
                        for row in loss_bins(block.k, block.loss))
    m, f, b, cost = pd.DataFrame(metrics), pd.DataFrame(curves), pd.DataFrame(bins), compute(out)
    losses = pd.concat(logs, ignore_index=True)
    paired = pd.DataFrame(dict(seed=SEEDS, selected_vs_eapig=decision["primary"]["per_seed"],
                               selected_vs_raw_absolute=decision["mattr_abs"]["per_seed"]))
    hierarchy = []
    for (role, step, seed), block in m.dropna(subset=["seed"]).groupby(["role", "step", "seed"]):
        by_arm = block.set_index("arm")
        for ordinary, h2 in (("B0", "B1"), ("M0", "M1"), ("K0", "K1")):
            if ordinary not in by_arm.index or h2 not in by_arm.index:
                continue
            a, b_name = by_arm.loc[ordinary, "ranking"], by_arm.loc[h2, "ranking"]
            a_points, b_points = (f[(f.role == role) & (f.ranking == key)].set_index(["selection", "target"])
                                  for key in (a, b_name))
            hierarchy.extend(dict(role=role, step=step, seed=int(seed), ordinary=ordinary, hierarchy=h2,
                                  selection=selection, target=target, delta_kl=float(row.kl-a_points.loc[(selection,target),"kl"]),
                                  delta_accuracy=float(row.accuracy-a_points.loc[(selection,target),"accuracy"]),
                                  delta_cmd=float(by_arm.loc[h2,"cmd"]-by_arm.loc[ordinary,"cmd"]))
                             for (selection, target), row in b_points.iterrows())
    fidelity = f[(f.selection == "grid") & f.target.isin([.1,.2,.5])].groupby(
        ["role","arm","step","target"])[["kl","accuracy"]].agg(["mean","std"]).reset_index()
    fidelity.columns = ["_".join(x for x in pair if x) for pair in fidelity.columns]
    outputs = dict(metrics=m, faithfulness_curves=f, sparsity_loss_bins=b, training_steps=losses,
                   edge_composition=comp, ranking_stability=stable, compute_accounting=cost,
                   paired_cmd_differences=paired, margin_ratios=pd.DataFrame(ratios),
                   hierarchy_comparisons=pd.DataFrame(hierarchy), seed_fidelity_variability=fidelity)
    for name, data in outputs.items():
        data.to_csv(out / f"{name}.csv", index=False)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    cf = f[f.role == "confirmation"]
    cm = m[m.role == "confirmation"]
    save_plot(figures, "faithfulness", cf[cf.selection == "grid"].groupby(["arm", "target"], as_index=False).faithfulness.mean(),
              "target", "faithfulness", "arm", log_x=True)
    save_plot(figures, "cmd_by_seed", cm.dropna(subset=["seed"]), "seed", "cmd", "arm", scatter=True)
    save_plot(figures, "cpr_cmd", cm, "cmd", "cpr", "arm", scatter=True)
    kl = cf[(cf.selection == "grid") & cf.target.isin([.1,.2,.5])]
    save_plot(figures, "kl", kl.groupby(["arm","target"], as_index=False).kl.mean(), "target", "kl", "arm", log_x=True)
    caps = cf[cf.selection == "cap"]
    save_plot(figures, "compact_accuracy", caps, "actual_edges", "accuracy", "arm", scatter=True)
    save_plot(figures, "eapig_overlap", comp[comp.comparator == c["reference"]], "top", "jaccard", "ranking", scatter=True)
    save_plot(figures, "rank_stability", stable.assign(pair=np.arange(len(stable))), "pair", "spearman", "arm", scatter=True)
    save_plot(figures, "training_loss", losses.assign(block=(losses.step-1)//100).groupby(["arm","block"], as_index=False).loss.mean(),
              "block", "loss", "arm")
    save_plot(figures, "sparsity_loss", b.groupby(["arm","checkpoint","bin"], as_index=False)["mean"].mean().assign(
              series=lambda d:d.arm+" "+d.checkpoint), "bin", "mean", "series", scatter=True)
    save_plot(figures, "selected_vs_raw_absolute", cm[cm.arm.isin([c["selected"],"mattr_abs"])],
              "seed", "cmd", "arm")
    fig, ax = plt.subplots(figsize=(7.5, 4.5), layout="constrained")
    rf = outputs["margin_ratios"].query("role == 'confirmation' and selection == 'grid' and target == 0.1")
    for arm, group in rf.groupby("arm"):
        x = np.sort(group.ratio.to_numpy())
        ax.plot(x, np.arange(1, len(x)+1)/len(x), label=arm)
    ax.set(xlabel="Signed circuit / clean margin ratio", ylabel="Empirical cumulative probability", xscale="symlog")
    ax.legend(fontsize=8)
    fig.savefig(figures / "margin_ratios.png", dpi=150)
    plt.close(fig)
    report_text(out, p, c, pilot, decision, outputs)
    return decision


def report_text(out, p, c, pilot, decision, outputs):
    pre = checked_json(out / "preflight.json")
    ref = reference(out)
    text = ["# Gate 6A faithfulness-trained MAttr", "## Protocol and objective",
            f"Training SHA-256 `{sha(out / TRAINING)}`.",
            "Margin matching uses squared per-example clean-margin error divided by D². KL uses the complete vocabulary and the clean distribution as target. R0 reuses the upstream per-example floored-gap CMD surrogate.",
            f"Discovery D = {pre['targets']['D']:.10g}. Strategy: {pre['targets']['strategy']}.",
            f"Reference {ref['name']}, development CMD values {ref['cmd']}. Positive ordinal score export preserves the learned signed order. The raw-absolute diagnostic uses untransformed scores.",
            "Prior-work audit: node-level KL and CMD-shaped training appear in MAttr Appendix G.1, Table 8. The published loss ablation reports CPR and Compactness.",
            "## Pilot", f"Decision **{pilot['decision']}**. Checks {pilot['checks']}.",
            f"First and last 500-step means {pilot['first_500_mean']} and {pilot['last_500_mean']}. CMD {pilot['cmd']}, limit {pilot['limit']}."]
    if c:
        text += [f"Confirmation SHA-256 `{sha(out / CONFIRMATION)}`.",
                 f"Selected arm **{c['selected']}**. Development 5,000-step arm means {linked(out, 'selection')['means']}."]
    if decision:
        text += ["## Registered CMD decision", f"**{decision['classification']}**"]
        for name in ("primary", "mattr_abs"):
            if name in decision:
                d = decision[name]
                rel = f"{d['relative']:+.2f}%" if d["relative"] is not None else "undefined"
                text.append(f"{name} selected CMD {d['selected_mean']:.8f}, comparator CMD {d['comparator_mean']:.8f}, paired difference {d['delta']:+.8f} ({rel}), 95% interval {d['interval']}, unstable draws {d['unstable']}.")
        if "missing" in decision:
            text.append(f"Missing {decision['missing']}.")
    if outputs:
        m = outputs["metrics"]
        text += ["## Official per-ranking metrics", table(m.drop(columns="raw_score_stats").round(6)),
                 "## Paired confirmation differences", table(outputs["paired_cmd_differences"].round(6)),
                 "## Development raw-absolute diagnostic",
                 table(m[(m.role == "development") & (m.arm == "mattr_abs")][["ranking","cmd","raw_score_stats"]]),
                 "## Runtime and fidelity artifacts",
                 "See `compute_accounting.csv` for actual process wall time, root forwards, example-forwards, backward calls, optimizer updates, CUDA allocator peaks, and sampled CPU RSS. Gate 5F baseline and attribution rankings were reused.",
                 "See `faithfulness_curves.csv` for all pruned edge counts, per-grid CMD contributions, compact accuracy, margin errors, KL, and agreement metrics.",
                 "See `edge_composition.csv`, `ranking_stability.csv`, `hierarchy_comparisons.csv`, `seed_fidelity_variability.csv`, and `sparsity_loss_bins.csv` for the descriptive analyses."]
        text += [f"![{path.stem.replace('_',' ')}](figures/{path.name})" for path in sorted((out / "figures").glob("*.png"))]
    text += [LIMITATIONS, "## Interpretation and recommendation (researcher-written)"]
    (out / REPORT).write_text("\n\n".join(text) + "\n", encoding="utf-8")


def audit(out):
    p = protocol(out)
    c = confirmation(out, p)
    stage1_reuse(out, p)
    preflight = checked_json(out / "preflight.json")
    assert preflight["passed"] and preflight["bounded"]["passed"]
    from .targets import verified_targets
    assert verified_targets(out / "targets")
    selection = linked(out, "selection")
    for key, value in selection["evaluation_sha256"].items():
        assert sha(out / "evaluations/development" / key / "completed.json") == value
    scores = {}
    for role, keys in (("development", p["development_jobs"]), ("confirmation", c["rankings"])):
        for key in keys:
            data = check_result(out, p, role, key, verify_masks=True)
            assert data is not None and data["model_weights_unchanged"]
            assert data["accounting"]["optimizer_updates"] == data["accounting"]["backward_calls"] == 0
            if role == "confirmation":
                scores[key] = data
    assert all(rank(out, key, p)[1] == value for key, value in c["rankings"].items())
    from src.mattr_data_scaling.training import checked_checkpoint
    expected_training = {ranking_key(arm, seed) for arm, seed in p["training_jobs"]}
    expected_training.update(ranking_key(c["selected"], seed) for seed in SEEDS[-2:])
    assert {path.name for path in (out / "training").iterdir() if path.is_dir()} == expected_training
    for key in sorted(expected_training):
        path = out / "training" / key
        assert verified(path)
        receipt = read(path / "completed.json")["identity"]
        assert receipt["implementation_commit"] == p["implementation_commit"] and not receipt["source_status"]
        evidence = read(path / "weight_audit.json")
        assert evidence["passed"] and evidence["before"] == evidence["after"]
        for step in (1000, 5000):
            checkpoint = path / "checkpoints" / f"step_{step:06d}.pt"
            state = checked_checkpoint(checkpoint, receipt)
            assert state["completed"] == step
            rkey = key.rsplit("_", 1)[0] + f"_{step}"
            assert sha(checkpoint) == read(path / "rankings.json")[rkey]["checkpoint_sha256"]
            rank(out, rkey, p)
    stamp = (out / CONFIRMATION).stat().st_mtime_ns
    assert all(path.stat().st_mtime_ns >= stamp for path in (out / "attempts").glob("confirmation_*/*/forward_events.jsonl"))
    assert checked_json(out / "graph_preflight.json")["cohort"]["pre_freeze_confirmation_forwards"] == 0
    dev = pd.read_csv(out / "development_hashes.csv")
    confirm = pd.read_csv(out / "confirmation_hashes.csv")
    train = pd.read_csv(out / "discovery.csv", usecols=["clean_sha256", "corrupted_sha256"])
    previous = set(dev.clean_sha256) | set(dev.corrupted_sha256) | set(train.clean_sha256) | set(train.corrupted_sha256)
    hashes = set(confirm.clean_sha256) | set(confirm.corrupted_sha256)
    assert not previous & hashes and len(hashes) == 2*len(confirm) and confirm.template_id.nunique() == 9
    result, draws = comparison(out, c, scores)
    assert result == read(out / "decision.json")
    np.testing.assert_array_equal(draws, np.load(out / "bootstrap_indices.npy"))
    assert (out / REPORT).exists()
    costs = compute(out)
    record = dict(passed=True, classification=result["classification"], training_sha256=sha(out / TRAINING),
                  confirmation_sha256=sha(out / CONFIRMATION), ranking_count=len(scores),
                  forward_batches=int(costs.forward_batches.sum()), example_forwards=int(costs.example_forwards.sum()),
                  backward_calls=int(costs.backward_calls.sum()), updates=int(costs.optimizer_updates.sum()),
                  artifact_sha256={path.relative_to(out).as_posix(): sha(path) for path in
                      [out / REPORT, out / "decision.json", out / "metrics.csv", out / "faithfulness_curves.csv",
                       out / "sparsity_loss_bins.csv", out / "edge_composition.csv", out / "ranking_stability.csv",
                       out / "hierarchy_comparisons.csv", out / "seed_fidelity_variability.csv",
                       out / "development_hashes.csv", out / "confirmation_hashes.csv",
                       out / "paired_cmd_differences.csv", out / "compute_accounting.csv", *sorted((out / "figures").glob("*.png"))]})
    dest = out / "final_audit.json"
    if dest.exists():
        assert checked_json(dest) == record
    else:
        freeze_json(dest, record)
    return record
