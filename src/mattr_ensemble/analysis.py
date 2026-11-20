"""Shared stratified bootstrap and CPU-only analysis of the frozen rankings."""

from itertools import combinations

import numpy as np
import pandas as pd
import torch

from src.mattr_compact_validation.inference import assign_scores
from src.mattr_faithful.decisions import relative_difference
from src.mattr_faithful.evaluate import capped, mask_at
from src.mattr_fidelity_splice.analysis import table
from src.mattr_fidelity_splice.core import read
from src.mattr_fidelity_splice.statistics import official_areas, resamples, weights
from src.mattr_gate5g_addendum.common import checked_json, freeze_json
from src.mattr_matched_size.topology import digest, graph_new
from src.mattr_optimization.common import GRID, sha, write_json
from .core import FILE, MEMBERS, protocol
from .evaluation import result

REPORT = "GATE6B_ENSEMBLE_REPORT.md"


def classify(cmd_upper, accuracy_lower, valid=True):
    if not valid or not np.isfinite([cmd_upper, accuracy_lower]).all():
        return "Invalid"
    cmd, compact = cmd_upper < .005, accuracy_lower > 0
    if cmd and compact:
        return "Two-goal success"
    if cmd:
        return "CMD non-inferior only"
    if compact:
        return "Compact gain only"
    return "Neither"


def paired(first, others, first_draws, other_draws):
    values = first_draws - np.mean(other_draws, axis=0)
    valid = np.isfinite(values)
    ci = np.quantile(values[valid], [.025, .975]).tolist() if valid.any() else [None, None]
    return dict(first=float(first), comparator=float(np.mean(others)), difference=float(first - np.mean(others)),
                interval=ci, unstable=int((~valid).sum()))


def auc(accuracy):
    dx = np.diff(np.log(GRID))
    return dx @ ((accuracy[:-1] + accuracy[1:]) * .5) / dx.sum()


def compute(out):
    rows = []
    stages = []
    for receipt in sorted((out / "attempts").glob("rank_*/*/attempt.json")):
        rec = read(receipt)
        path = receipt.parent / "compute_accounting.csv"
        if path.exists() and path.stat().st_size > 3:
            frame = pd.read_csv(path)
            stages.append(frame.assign(job=rec["key"], attempt=receipt.parent.name))
            events = pd.read_json(receipt.parent / "forward_events.jsonl", lines=True)
            assert len(events) == frame.forward_batches.sum()
            if len(events):
                assert events.batch_size.sum() == frame.example_forwards.sum()
                assert not events.grad_enabled.any()
        rows.append(dict(**rec, attempt=receipt.parent.name))
    totals = pd.concat(stages, ignore_index=True) if stages else pd.DataFrame()
    assert totals.backward_calls.sum() == totals.optimizer_updates.sum() == 0
    return pd.DataFrame(rows), totals


def evaluate_statistics(out, p):
    results = {key: result(out, p, key) for key in p["jobs"]}
    if any(value is None for value in results.values()):
        raise ValueError("Missing required ranking outputs")
    manifest = pd.read_csv(out / "cohort_hashes.csv")
    indices = resamples(manifest.template_id, draws=p["bootstrap"]["draws"], seed=p["bootstrap"]["seed"])
    w = weights(indices, len(manifest))
    graph = graph_new()
    data, curves, masks, full_ranks = {}, [], {}, {}
    anchors = None
    for key, rec in results.items():
        assert rec["model_weights_unchanged"]
        assert rec["counts"]["backward_calls"] == rec["counts"]["optimizer_updates"] == 0
        scores = torch.load(out / "rankings" / f"{key}.pt", weights_only=True)
        assign_scores(graph, scores)
        full_ranks[key] = scores.numpy()
        with np.load(out / "evaluations" / key / "vectors.npz", allow_pickle=False) as file:
            v = {name:file[name] for name in file.files}
        if anchors is None:
            anchors = v["clean"], v["empty"]
        for a, b in zip(anchors, (v["clean"], v["empty"]), strict=True):
            np.testing.assert_array_equal(a, b)
        matrix = v["grid_margins"].astype(float)
        clean, empty = (a.astype(float) for a in anchors)
        normalized = official_areas(matrix.mean(1), clean.mean(), empty.mean())
        np.testing.assert_allclose(normalized, [rec["cpr"], rec["cmd"]], atol=2e-6, rtol=2e-6)
        _, cmd_draws = official_areas(matrix @ w.T, w @ clean, w @ empty)
        accuracy_grid = (matrix > 0).astype(float)
        np.testing.assert_allclose(auc(accuracy_grid.mean(1)), rec["accuracy_log_auc"], atol=2e-7)
        calc = dict(cmd=cmd_draws, auc=auc(accuracy_grid @ w.T), caps={})
        assert len(rec["points"]) == len(GRID) + len(p["caps"])
        assert {(x["selection"], x["target"]) for x in rec["points"]} == (
            {("grid", x) for x in GRID} | {("cap", x) for x in p["caps"]})
        for point in rec["points"]:
            target = point["target"]
            if point["selection"] == "cap":
                n, mask = capped(graph, target, False)
                assert n == point["nominal_edges"]
            else:
                assert point["nominal_edges"] == int(target * len(graph.edges))
                mask = mask_at(graph, point["nominal_edges"], False)
            assert int(mask.sum()) == point["actual_edges"] and digest(mask) == point["mask_sha256"]
            prefix = f"{point['selection']}_{target}"
            correct = v[prefix + "_correct"]
            np.testing.assert_array_equal(correct, v[prefix + "_margin"] > 0)
            assert correct.shape == (len(manifest),)
            for metric, field in (("accuracy","correct"), ("top1_accuracy","top1_correct"),
                                  ("sign_agreement","sign_agreement"), ("kl","kl")):
                arr = v[f"{prefix}_{field}"]
                assert arr.shape == correct.shape and np.isfinite(arr).all()
                np.testing.assert_allclose(arr.mean(), point[metric], atol=1e-8, rtol=1e-8)
            if point["selection"] == "cap":
                calc["caps"][target] = (correct.mean(), w @ correct)
                if target == 116:
                    masks[key] = mask
            curves.append(dict(ranking=key, **point))
        data[key] = calc

    f = pd.DataFrame(curves)
    m = pd.DataFrame([dict(ranking=key, **{k: rec[k] for k in ("cmd", "cpr", "accuracy_log_auc")})
                      for key, rec in results.items()])
    caps = f[f.selection == "cap"].set_index(["ranking", "target"])
    ensemble, reference = "m1_ens5", "eapig_abs"
    seeds = MEMBERS[ensemble]
    cmd = paired(results[ensemble]["cmd"], [results[reference]["cmd"]], data[ensemble]["cmd"], [data[reference]["cmd"]])

    def cap_comparison(others, size):
        point, boot = data[ensemble]["caps"][size]
        return paired(point, [data[key]["caps"][size][0] for key in others], boot,
                      [data[key]["caps"][size][1] for key in others])

    compact = cap_comparison([reference], 116)
    valid = cmd["interval"][1] is not None and compact["interval"][0] is not None
    upper, lower = (cmd["interval"][1], compact["interval"][0]) if valid else (np.nan, np.nan)
    decision = dict(classification=classify(upper, lower, valid), cmd=cmd, compact=compact,
                    relative_cmd=relative_difference(cmd["difference"], cmd["comparator"]),
                    numerically_lower_cmd=cmd["difference"] < 0, cmd_superior=bool(upper < 0),
                    cmd_noninferior=bool(upper < .005), compact_superior=bool(lower > 0))
    seed_rows = []
    for metric, key, ascending in (("CMD","cmd",True), ("Capped-116 accuracy","compact",False),
                                   ("Accuracy log-AUC","auc",False)):
        if key == "compact":
            comparison = cap_comparison(seeds, 116)
            points = [data[name]["caps"][116][0] for name in seeds]
        else:
            column = "cmd" if key == "cmd" else "accuracy_log_auc"
            points = [results[name][column] for name in seeds]
            comparison = paired(results[ensemble][column], points, data[ensemble][key],
                                [data[name][key] for name in seeds])
        best, worst = (min(points), max(points)) if ascending else (max(points), min(points))
        better = np.less if ascending else np.greater
        seed_rows.append(dict(metric=metric, **comparison, median=float(np.median(points)), best=best, worst=worst,
                              beats_mean=bool(better(comparison["first"], comparison["comparator"])),
                              beats_best=bool(better(comparison["first"], best)),
                              beats_worst=bool(better(comparison["first"], worst))))
    signed = []
    for size in p["caps"]:
        comparator = "eapig_signed"
        a, b = (caps.loc[(key, size)] for key in (ensemble, comparator))
        signed.append(dict(cap=size, **cap_comparison([comparator], size),
                           ensemble_edges=a.actual_edges, signed_edges=b.actual_edges,
                           ensemble_top1=a.top1_accuracy, signed_top1=b.top1_accuracy))
    correlations = [dict(first=a, second=b, spearman=float(np.corrcoef(full_ranks[a], full_ranks[b])[0,1]))
                    for a,b in combinations(seeds, 2)]
    mapping = pd.read_csv(out / "edge_mapping.csv", keep_default_na=False)
    known = {f"a{l}.h{h}" for group in read(out / "known_ioi_heads.json")["classes"].values() for l,h in group}
    overlaps = []
    for key in [ensemble, *seeds]:
        edges = mapping[masks[key]]
        receivers = edges.child + edges.qkv.map(lambda port: f"<{port}>" if port else "")
        overlaps.append(dict(ranking=key, jaccard=float((masks[key] & masks[ensemble]).sum() / (masks[key] | masks[ensemble]).sum()),
                             known_heads=len((set(edges.parent) | set(edges.child)) & known),
                             destination_counts=receivers.value_counts().to_json()))
    top = f[(f.selection == "cap") & (f.target == 116)][["ranking","actual_edges","accuracy","top1_accuracy","kl"]].merge(m, on="ranking")
    loss = top[top.ranking.isin(MEMBERS)].copy()
    for point in (.1,.2,.5):
        loss = loss.merge(f[(f.selection=="grid") & (f.target==point)][["ranking","kl","actual_edges"]].rename(
            columns={"kl":f"KL {point:.0%}", "actual_edges":f"edges {point:.0%}"}), on="ranking")
    return decision, indices, dict(metrics=m, faithfulness=f, capped_116=top, seed_comparison=pd.DataFrame(seed_rows),
                                  signed_comparison=pd.DataFrame(signed), loss_ensembles=loss,
                                  rank_agreement=pd.DataFrame(correlations), overlaps=pd.DataFrame(overlaps))


def figures(out, tables):
    import matplotlib.pyplot as plt

    folder = out / "figures"
    folder.mkdir(exist_ok=True)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(folder / f"{name}.png", dpi=160)
        plt.close(fig)

    for name, frame, y in (("cmd_auc", tables["metrics"], "accuracy_log_auc"),
                           ("cmd_capped_accuracy", tables["capped_116"], "accuracy")):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for row in frame.itertuples():
            individual = row.ranking in MEMBERS["m1_ens5"]
            reference = row.ranking.startswith("eapig")
            color = "tab:blue" if individual else "tab:orange" if reference else "tab:green"
            ax.scatter(row.cmd, getattr(row, y), color=color, marker="o" if individual else "s" if reference else "D")
            ax.annotate(row.ranking.replace("_5000", ""), (row.cmd, getattr(row, y)), xytext=(4, 3),
                        textcoords="offset points", fontsize=7)
        ax.set(xlabel="Official CMD (lower is better)", ylabel=y.replace("_", " "), xscale="log")
        ax.margins(x=.18, y=.12)
        save(fig, name)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    for ax, row in zip(axes, tables["seed_comparison"].itertuples(), strict=True):
        ax.bar(["Ensemble", "Seed mean"], [row.first, row.comparator], color=["tab:green", "tab:blue"])
        ax.set_title(row.metric)
    save(fig, "ensemble_vs_seed_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    f = tables["faithfulness"]
    for name in ["m1_ens5", *MEMBERS["m1_ens5"], "eapig_abs", "eapig_signed"]:
        rows = f[(f.selection == "cap") & (f.ranking == name)].sort_values("target")
        ax.plot(rows.actual_edges, rows.accuracy, ".-", label=name, alpha=.5 if name in MEMBERS["m1_ens5"] else 1)
    ax.set(xlabel="Actual retained edges", ylabel="IOI accuracy", xscale="log")
    ax.legend(fontsize=7)
    save(fig, "compact_accuracy")
    fig, ax = plt.subplots(figsize=(7, 4))
    kl = tables["capped_116"].set_index("ranking").kl
    ax.bar(["M1 ensemble", "M1 seed mean", "EAP-IG abs", "EAP-IG signed"],
           [kl["m1_ens5"], kl.loc[MEMBERS["m1_ens5"]].mean(), kl["eapig_abs"], kl["eapig_signed"]])
    ax.set_ylabel("Full-vocabulary KL at cap 116")
    save(fig, "cap116_kl")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("m1_ens5", "eapig_abs"):
        rows = f[(f.selection == "grid") & (f.ranking == name)]
        ax.plot(rows.target, rows.faithfulness, ".-", label=name)
    ax.axhline(1, color="gray", linestyle="--")
    ax.set(xlabel="Nominal edge fraction", ylabel="Official faithfulness", xscale="log")
    ax.legend()
    save(fig, "faithfulness")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    agreement, overlap = tables["rank_agreement"], tables["overlaps"].iloc[1:]
    axes[0].scatter(range(1, len(agreement)+1), agreement.spearman)
    axes[0].set(xlabel="Seed pair index", ylabel="Full-rank Spearman")
    axes[1].bar([key.split("_")[1] for key in overlap.ranking], overlap.jaccard)
    axes[1].set(xlabel="M1 seed", ylabel="Cap-116 Jaccard with ensemble")
    save(fig, "rank_agreement")


def report(out, p, decision, tables, attempts, stages):
    cmd, acc = decision["cmd"], decision["compact"]
    relative = f"{decision['relative_cmd']:+.3f}%" if decision["relative_cmd"] is not None else "undefined"
    rho = tables["rank_agreement"].spearman
    text = [
        "# Gate 6B seed ensemble results",
        "Gate 6B was designed after Gate 6A, whose registered outcome was **Inconclusive**. The primary ranking averages canonical positions across five H2 squared-margin seeds with equal weights and canonical-index tie-breaking.",
        f"Protocol SHA-256 `{sha(out / FILE)}`. Selection seed 5548 retained {p['selected']} of {p['eligible']} eligible validation prompts across nine exact template strata. The preflight used excluded discovery prompts.",
        f"## Registered result — {decision['classification']}",
        f"Ensemble CMD {cmd['first']:.6g}, absolute EAP-IG CMD {cmd['comparator']:.6g}, difference {cmd['difference']:+.6g} ({relative}), paired 95% interval {cmd['interval']}. Numerically lower CMD: {decision['numerically_lower_cmd']}. Interval entirely below zero: {decision['cmd_superior']}. CMD non-inferiority at +0.005: {decision['cmd_noninferior']}. Unstable draws: {cmd['unstable']}.",
        f"Capped-116 IOI accuracy {acc['first']:.2%} versus {acc['comparator']:.2%}, paired difference {acc['difference']:+.2%}, paired 95% interval {acc['interval']}. Compact superiority: {decision['compact_superior']}.",
        "CMD superiority requires an interval wholly below zero. The registered non-inferiority rule permits a CMD increase within the 0.005 margin.",
        "![CMD versus accuracy log-AUC](figures/cmd_auc.png)\n\n![CMD versus capped-116 accuracy](figures/cmd_capped_accuracy.png)",
        "## Capped-116 results",
        "Actual edges, IOI accuracy, top-1 accuracy and full-vocabulary KL refer to the capped circuit. CMD and CPR use the full official grid.",
        table(tables["capped_116"]),
        "## Ensemble versus the individual M1 seeds", table(tables["seed_comparison"]),
        "## Compact accuracy versus signed EAP-IG", table(tables["signed_comparison"]),
        "## Descriptive comparison of the three ensembles", table(tables["loss_ensembles"]),
        "M0 and K0 each use three trained rankings. M1 uses five.",
        "## Rank and circuit agreement",
        f"Pairwise full-edge Spearman mean {rho.mean():.5f}, median {rho.median():.5f}, minimum {rho.min():.5f}, maximum {rho.max():.5f}.",
        table(tables["overlaps"].drop(columns="destination_counts")),
        "The CSV tables provide all official grid points, mask hashes, retained counts, accuracy, CMD contributions, destination counts and pairwise correlations.",
        "## Runtime",
        f"The five historical M1 training processes took {p['historical_training_seconds']/3600:.3f} hours. CPU ranking verification and construction took {p['preparation_seconds']:.2f} seconds. Gate 6B evaluation attempts took {attempts.seconds.sum()/60:.2f} minutes and recorded {stages.forward_batches.sum()} root forwards, {stages.example_forwards.sum()} example-forwards, zero backwards and zero optimizer updates. Peak allocation {stages.peak_allocated_mib.max():.1f} MiB, reservation {stages.peak_reserved_mib.max():.1f} MiB, and sampled CPU RSS {stages.sampled_peak_cpu_rss_mib.max():.1f} MiB. Attempts without completion: {(attempts.status != 'complete').sum()}.",
        "## Ensemble members and hashes",
    ]
    for name, details in p["ensembles"].items():
        text.append(f"`{name}` score SHA-256 `{details['score_sha256']}`. Ordering SHA-256 `{details['ordering_sha256']}`.\n\n" +
                    "\n".join(f"- `{member}` `{checksum}`" for member, checksum in details["members"].items()))
    text += [*(f"![{name.replace('_',' ')}](figures/{name}.png)" for name in
               ("ensemble_vs_seed_mean", "compact_accuracy", "cap116_kl", "faithfulness", "rank_agreement")),
             "## Limitations",
             "This study covers GPT-2 Small IOI. The faithfulness losses used the original untuned MAttr hyperparameters. EAP-IG uses fixed attribution rankings. Gate 6A selected M1 on a historically inspected development cohort, and Gate 6B followed those results. Gate 1 previously evaluated all official validation rows. The five related ensemble members are fixed rankings on the same prompts. The bootstrap conditions on their rankings and exact template mixture, so it measures prompt sampling uncertainty. Ensembling requires all five training trajectories. The CMD margin allows a modest increase. Capped budgets can contain different actual edge counts. Evaluation was limited to validation.",
             "## Interpretation and recommendation (researcher-written)"]
    (out / REPORT).write_text("\n\n".join(text) + "\n", encoding="utf-8")


def analyze(out):
    p = protocol(out)
    if (out / "final_audit.json").exists():
        from src.mattr_fidelity_splice.core import check_files
        check_files(checked_json(out / "final_audit.json")["outputs"], out)
        return read(out / "decision.json")
    try:
        decision, indices, tables = evaluate_statistics(out, p)
    except (AssertionError, ValueError, FileNotFoundError) as exc:
        write_json(out / "decision.json", dict(classification="Invalid", error=str(exc)))
        raise
    attempts, stages = compute(out)
    summaries = [result(out, p, key) for key in p["jobs"]]
    tables["cache_accounting"] = pd.DataFrame([
        dict(ranking=key, circuit_hits=summary["cache_hits"], circuit_misses=summary["cache_misses"])
        for key, summary in zip(p["jobs"], summaries, strict=True)])
    for name, frame in {**tables, "attempts": attempts, "compute_accounting": stages}.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    np.save(out / "bootstrap_indices.npy", indices, allow_pickle=False)
    write_json(out / "decision.json", decision)
    figures(out, tables)
    report(out, p, decision, tables, attempts, stages)
    return decision


def audit(out):
    p = protocol(out)
    decision, indices, tables = evaluate_statistics(out, p)
    assert decision["classification"] != "Invalid" and decision == read(out / "decision.json")
    np.testing.assert_array_equal(indices, np.load(out / "bootstrap_indices.npy", allow_pickle=False))
    smoke = checked_json(out / "smoke.json")
    assert smoke["passed"] and smoke["fresh_forwards"] == 0
    hashes = pd.read_csv(out / "cohort_hashes.csv")
    all_hashes = set(hashes.clean_sha256) | set(hashes.corrupted_sha256)
    assert len(all_hashes) == 2*len(hashes) and len(hashes) == p["selected"] and hashes.template_id.nunique() == 9
    assert not all_hashes & {row[col] for row in smoke["prompts"] for col in ("clean_sha256", "corrupted_sha256")}
    from src.mattr_faithful.core import DEFAULT as old
    historical = checked_json(old / "graph_preflight.json")["cohort"]["excluded"]
    from src.mattr_optimization.common import ROOT
    prior_val = set()
    exclusions = [(ROOT / r["path"], "train" in r["role"] or "train" in r["path"].split("/")[-1])
                  for r in historical if "historical_full" not in r["role"]] + [
                      (old/"confirmation.csv", False), (old/"discovery.csv", True)]
    for path, train in exclusions:
        frame = pd.read_csv(path, keep_default_na=False)
        from src.mattr_compact_validation.design import text_hash
        assert not all_hashes & {text_hash(s) for col in ("clean_prompt","corrupted_prompt") for s in frame[col]}
        if not train and "official_filtered_index" in frame:
            prior_val.update(frame.official_filtered_index)
    assert not set(hashes.official_filtered_index) & prior_val
    stamp = (out / FILE).stat().st_mtime_ns
    assert all(file.stat().st_mtime_ns >= stamp for file in (out / "attempts").glob("rank_*/*/forward_events.jsonl"))
    attempts, stages = compute(out)
    for key in p["jobs"]:
        rows = attempts[(attempts.key == "rank_" + key) & (attempts.status == "complete")]
        assert len(rows) == 1
        logged = stages[stages.attempt == rows.iloc[0].attempt]
        counts = result(out, p, key)["counts"]
        for col, value in counts.items():
            assert int(logged[col].sum()) == value
    tables["cache_accounting"] = pd.DataFrame([
        dict(ranking=key, circuit_hits=summary["cache_hits"], circuit_misses=summary["cache_misses"])
        for key in p["jobs"] for summary in [result(out, p, key)]])
    for name, frame in {**tables, "attempts": attempts, "compute_accounting": stages}.items():
        assert (out / f"{name}.csv").read_text(encoding="utf-8") == frame.to_csv(index=False), name
    names = [f"{name}.csv" for name in (*tables, "attempts","compute_accounting")]
    files = [out / name for name in [*names, "decision.json", REPORT]] + sorted((out/"figures").glob("*.png"))
    assert len(list((out/"figures").glob("*.png"))) == 7
    record = dict(passed=True, protocol_sha256=sha(out/FILE), classification=decision["classification"],
                  outputs={file.relative_to(out).as_posix():sha(file) for file in files})
    path = out / "final_audit.json"
    if path.exists():
        assert checked_json(path) == record
    else:
        freeze_json(path, record)
    return record
