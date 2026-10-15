"""Regenerate the clean experiment notebook; scientific outputs are never embedded."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def cell(kind: str, identity: str, source: str) -> dict:
    result = {
        "cell_type": kind,
        "id": identity,
        "metadata": {},
        "source": textwrap.dedent(source).strip() + "\n",
    }
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


cells = [
    cell("markdown", "experiment-overview", r'''
    # Circuit sketching experiments

    Use this notebook to compare intervention predictors across **unseen prompts**,
    head selections, and random seeds. The original `mib_sketching_baseline.ipynb`
    remains the reference experiment. Shared implementation lives in
    `circuit_sketching/` so that methods can be changed and tested independently.

    This is a small, synthetic capital-city experiment. Its prediction accuracy
    measures cross-prompt transfer; it is **not a MIB leaderboard score** or evidence
    that a discovered circuit is faithful. An official MIB task adapter and evaluator
    are future work.

    **Start here:** select **Python (mib-sketching CUDA)**, edit the configuration,
    then **Run All**. The default run uses training and validation data only.
    `RUN_FINAL_TEST = False` protects the final test set during development.

    If the environment needs installing, run these commands from the repository
    in PowerShell, then select the registered kernel:

    ```powershell
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name mib-sketching --display-name "Python (mib-sketching CUDA)"
    .\.venv\Scripts\python.exe -m jupyterlab
    ```
    '''),
    cell("code", "setup-environment", r'''
    import importlib.metadata
    import json
    import os
    import sys
    from datetime import datetime, timezone
    from pathlib import Path
    from time import perf_counter

    # Resolve the project before importing libraries that initialize their caches.
    ROOT = next(
        (p for p in (Path.cwd(), *Path.cwd().parents)
         if (p / "requirements.txt").is_file()
         and (p / "circuit_sketching").is_dir()),
        None,
    )
    if ROOT is None:
        raise RuntimeError("Start Jupyter from the circuit-sketching repository.")
    for variable, relative in {
        "HF_HOME": ".cache/huggingface",
        "MPLCONFIGDIR": ".cache/matplotlib",
        "IPYTHONDIR": ".cache/ipython",
    }.items():
        cache_directory = ROOT / relative
        cache_directory.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache_directory)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch
    from IPython.display import display
    from transformer_lens import HookedTransformer

    from circuit_sketching.core import PromptSpec
    from circuit_sketching.experiments import (
        ExperimentConfig, evaluate_test, export_run, run_development,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is unavailable in {sys.executable}. Select Python (mib-sketching CUDA) "
            "and install requirements.txt with the project's .venv interpreter."
        )
    runtime_metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "device": "cuda",
        "gpu": torch.cuda.get_device_name(0),
        "torch_cuda": torch.version.cuda,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformer-lens", "transformers", "numpy", "scikit-learn")
        },
    }
    display(pd.Series(runtime_metadata).to_frame("value"))
    '''),
    cell("markdown", "configuration-guide", r'''
    ## Experiment configuration

    Four prompts supply training labels. Two different prompts select hyperparameters
    by validation MAE. Two further prompts are reserved for the final test.
    Each candidate continuation must be **one GPT-2 token**, including its leading
    space; the evaluator checks this and rejects incompatible prompts.

    A mask selects heads to remove across all token positions. Training, validation,
    and test mask sets are unique and disjoint for each seed. With eight heads there
    are only 256 possible masks; increasing mask counts must respect that limit.
    A seed changes mask sampling, projections, and random directions. The default
    head set is inherited from the baseline, not selected for task relevance.

    Change these values before running development. Add a head set or more seeds to
    test robustness; acquisition work grows with prompts, seeds, and head sets.
    '''),
    cell("code", "experiment-configuration", r'''
    RUN_FINAL_TEST = False  # Keep False while choosing prompts, heads, methods, or budgets.

    prompts = [
        PromptSpec("france", "The capital of France is", " Paris", " London"),
        PromptSpec("germany", "The capital of Germany is", " Berlin", " Rome"),
        PromptSpec("italy", "The capital of Italy is", " Rome", " Berlin"),
        PromptSpec("japan", "The capital of Japan is", " Tokyo", " Paris"),
        PromptSpec("spain", "The capital of Spain is", " Madrid", " Rome"),
        PromptSpec("england", "The capital of England is", " London", " Paris"),
        PromptSpec("china", "The capital of China is", " Beijing", " London"),
        PromptSpec("russia", "The capital of Russia is", " Moscow", " Berlin"),
    ]
    config = ExperimentConfig(
        prompts=prompts,
        prompt_splits={
            "train": ["france", "germany", "italy", "japan"],
            "validation": ["spain", "england"],
            "test": ["china", "russia"],
        },
        head_sets={
            "baseline_eight": [(layer, layer) for layer in range(8)],
            # "alternate_eight": [(layer, (layer + 1) % 12) for layer in range(8)],
        },
        seeds=(42, 43),
        n_train_masks=64,
        n_eval_masks=24,
        ablation_probability=0.30,
        direction_budgets=(8, 16, 32),
        epsilons=(0.05, 0.25),
        projection_dims=(8, 16),
        ridge_alpha=1e-3,
    )
    display(pd.DataFrame([
        {"split": split, "prompt_ids": ", ".join(ids), "prompt_count": len(ids)}
        for split, ids in config.prompt_splits.items()
    ]))
    '''),
    cell("markdown", "model-loading-guide", r'''
    ## Load the model

    GPT-2 weights are reused from the project cache when present. The model and
    forward inputs stay on CUDA. Evaluation uses `eval()` and disables gradients;
    the forward tracker synchronizes CUDA around timing and counts real forwards.
    Cached requests are recorded separately.
    '''),
    cell("code", "load-gpt2", r'''
    model = HookedTransformer.from_pretrained("gpt2", device="cuda")
    model.eval()
    model.requires_grad_(False)
    runtime_metadata["model"] = {
        "name": "gpt2",
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
        "d_head": model.cfg.d_head,
    }
    print(f"Loaded GPT-2 on {next(model.parameters()).device}; gradients disabled.")
    '''),
    cell("markdown", "development-methods", r'''
    ## Run development

    The target is the **change from the unablated correct-minus-incorrect logit
    difference** for each prompt. Training-prompt mean effects determine one
    predictor per head set, seed, and candidate. Its coefficients transfer unchanged
    to every validation prompt. The evaluator selects one candidate per method using
    validation MAE, without reading final-test labels or refitting on validation.

    | Method | What it learns from training prompts |
    |---|---|
    | `linear` | One coefficient per removed head |
    | `quadratic` | Head effects plus each pair once, for `i < j` |
    | `projected_quadratic` | Head effects and a random projection of pair features, fitted to full-removal mask labels |
    | `directional` | A local gradient estimated from central finite differences around the unablated model |

    Linear, quadratic, and projected quadratic share their binary training labels.
    The projection reduces feature dimension, not label acquisition at this fixed
    mask budget. Directional candidates sweep epsilon and direction count. Their
    perturbations can scale a head above its original value and do not train directly
    on complete head removal. This mismatch is an explicit experimental question.
    '''),
    cell("code", "run-development", r'''
    development_started = perf_counter()
    run = run_development(model, config, progress=print)
    runtime_metadata["development_wall_seconds"] = perf_counter() - development_started
    display(run.tables["split_manifest"])
    print(f"Development completed in {runtime_metadata['development_wall_seconds']:.1f} seconds.")
    print("The run stores a frozen configuration; later edits to config do not change it.")
    '''),
    cell("markdown", "validation-reading-guide", r'''
    ## Inspect validation accuracy and cost

    Lower MAE and higher R²/correlations are better. Negative R² means a predictor
    performs worse than the constant mean on the evaluated data. These validation
    scores are used for model selection, so they are not final-test performance.
    With only two validation prompts, results can be sensitive to prompt choice.

    `standalone_acquisition_forwards` is the training-forward budget to obtain one
    candidate's data independently, including the needed unablated references.
    The `costs` table records actual forwards executed in this shared, cached run;
    evaluation forwards are also recorded there. Do not add standalone budgets to
    estimate actual shared-run cost. Repeated directional endpoints can be cached,
    so nominal directions do not necessarily require two new forwards each.
    '''),
    cell("code", "validation-tables", r'''
    validation = run.tables["validation_metrics"]
    selected_validation = validation.loc[validation["selected"]].copy()
    display(validation.sort_values(["head_set", "seed", "method", "MAE"]))
    print("Selected candidates (chosen using validation MAE):")
    display(run.tables["selections"])
    print("Selected validation scores across configured head sets and seeds; std is descriptive, not a confidence interval:")
    display(selected_validation.groupby("method")[["MAE", "R2", "Pearson", "Spearman"]].agg(["mean", "std"]))
    print("Actual measured acquisition and evaluation costs:")
    display(run.tables["costs"])
    print("Acquisition diagnostics:")
    display(run.tables["acquisition_diagnostics"])
    '''),
    cell("code", "validation-plots", r'''
    figures = {}
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, rows in validation.groupby("method", sort=False):
        ax.scatter(rows["standalone_acquisition_forwards"], rows["MAE"], label=method, alpha=0.65)
    ax.scatter(
        selected_validation["standalone_acquisition_forwards"], selected_validation["MAE"],
        facecolors="none", edgecolors="black", s=110, linewidths=1,
        label="selected per head set / seed",
    )
    ax.set(xlabel="Standalone training acquisition forwards", ylabel="Validation MAE",
           title="Validation accuracy versus acquisition cost")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures["validation_accuracy_cost"] = fig
    plt.show()

    selected_keys = run.tables["selections"][["head_set", "seed", "method", "candidate"]]
    selected_predictions = run.tables["validation_predictions"].merge(
        selected_keys, on=["head_set", "seed", "method", "candidate"],
        how="inner", validate="many_to_one",
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    for method, rows in selected_predictions.groupby("method", sort=False):
        ax.scatter(rows["truth"], rows["prediction"], label=method, alpha=0.4, s=20)
    bounds = selected_predictions[["truth", "prediction"]].to_numpy(dtype=float)
    lower, upper = float(np.nanmin(bounds)), float(np.nanmax(bounds))
    padding = max((upper - lower) * 0.05, 0.01)
    ax.plot([lower - padding, upper + padding], [lower - padding, upper + padding],
            linestyle="--", color="black", linewidth=1)
    ax.set(xlabel="Measured validation logit-difference change", ylabel="Predicted change",
           title="Selected candidates on unseen validation prompts")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    figures["validation_selected_predictions"] = fig
    plt.show()
    '''),
    cell("markdown", "final-test-gate-guide", r'''
    ## Optional final test — keep sealed during development

    When the method choices and configuration are frozen, set `RUN_FINAL_TEST = True`
    in the configuration cell and run the notebook once. The evaluator scores only
    the already-selected candidates on held-out prompts and masks; it does not refit
    or select candidates using test results. A run object can be tested only once.

    The flag is a workflow guard, not an access-control mechanism. Repeatedly changing
    methods after viewing test results makes those prompts part of development;
    reserve new test prompts before claiming a fresh final evaluation.
    '''),
    cell("code", "optional-final-test", r'''
    if RUN_FINAL_TEST:
        if "test_metrics" not in run.tables:
            final_test_started = perf_counter()
            run = evaluate_test(run, progress=print)
            runtime_metadata["final_test_wall_seconds"] = perf_counter() - final_test_started
        else:
            print("This run already has final-test results; displaying the existing evaluation.")
        display(run.tables["test_metrics"])
    else:
        print("Final test remains sealed. Only training and validation have been evaluated.")
    '''),
    cell("code", "optional-test-plot", r'''
    if "test_predictions" in run.tables and not run.tables["test_predictions"].empty:
        test_predictions = run.tables["test_predictions"]
        fig, ax = plt.subplots(figsize=(7, 6))
        for method, rows in test_predictions.groupby("method", sort=False):
            ax.scatter(rows["truth"], rows["prediction"], label=method, alpha=0.4, s=20)
        bounds = test_predictions[["truth", "prediction"]].to_numpy(dtype=float)
        lower, upper = float(np.nanmin(bounds)), float(np.nanmax(bounds))
        padding = max((upper - lower) * 0.05, 0.01)
        ax.plot([lower - padding, upper + padding], [lower - padding, upper + padding],
                linestyle="--", color="black", linewidth=1)
        ax.set(xlabel="Measured final-test logit-difference change", ylabel="Predicted change",
               title="Frozen candidates on final-test prompts")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        figures["final_test_selected_predictions"] = fig
        plt.show()
    else:
        print("No final-test plot: the default development run does not evaluate test prompts.")
    '''),
    cell("markdown", "export-guide", r'''
    ## Save this experiment

    Export the frozen configuration, split and mask manifests, selections, predictions,
    metrics, diagnostics, real-forward log, and costs to a new directory. Figures and
    runtime versions are saved alongside them. Exporting again creates another
    directory rather than overwriting a previous run. Generated results stay local
    under the repository's existing `.gitignore` rules.
    '''),
    cell("code", "export-experiment", r'''
    output_directory = export_run(run, ROOT / "results" / "experiments")
    for name, figure in figures.items():
        figure.savefig(output_directory / f"{name}.png", dpi=160, bbox_inches="tight")
    runtime_metadata["final_test_evaluated"] = "test_metrics" in run.tables
    runtime_metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    (output_directory / "runtime_metadata.json").write_text(
        json.dumps(runtime_metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved experiment to: {output_directory}")
    display(run.tables.get("test_metrics", selected_validation))
    '''),
    cell("markdown", "experiment-next-steps", r'''
    ## What to try next

    1. Change one factor at a time: training-mask budget, projection dimension, finite-difference
       epsilon, or direction budget. Use validation performance and measured acquisition cost.
    2. Add prompt templates and head sets, then more seeds. Two final-test prompts are only
       a small transfer diagnostic; a larger prespecified test set is needed for conclusions.
    3. Implement a new method in `circuit_sketching/`, add correctness checks, restart the
       kernel, and compare under a common compute budget. Keep the baseline notebook intact.
    4. For leaderboard comparisons, add an official MIB task adapter and circuit evaluation.
       Prediction MAE and R² here are not circuit-discovery benchmark metrics.
    '''),
]


def main() -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (mib-sketching CUDA)",
                "language": "python",
                "name": "mib-sketching",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.5",
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for entry in cells:
        if entry["cell_type"] == "code":
            compile(entry["source"], f"<cell {entry['id']}>", "exec")
    destination = ROOT / "mib_sketching_experiments.ipynb"
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    code_count = sum(entry["cell_type"] == "code" for entry in cells)
    print(f"Wrote {destination.name}: {len(cells)} cells, {code_count} code cells")


if __name__ == "__main__":
    main()
