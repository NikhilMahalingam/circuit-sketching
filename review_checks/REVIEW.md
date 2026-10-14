# Notebook correctness review

Reviewed and updated `mib_sketching_baseline.ipynb` on October 14, 2026.
The notebook contains 49 cells, including 23 code cells and all 23 requested sections.
Every cell was statically inspected and every code cell compiled successfully.
The initial review used static inspection and CPU contract tests. A subsequent full GPT-2
execution completed successfully on CUDA; its outputs are saved in
`../mib_sketching_baseline.executed.ipynb`. The source notebook retains clean outputs.

| Requested check | Finding and evidence |
| --- | --- |
| 1. Only requested heads are edited | Passed CPU tensor tests for multiple heads in one layer and across layers. The helper now clones the input and assigns only `[:, :, head, :]` slices. Unselected slices compare exactly equal to their inputs. The input tensor remains unchanged. Downstream heads can still change causally following an earlier intervention. |
| 2. Correct TransformerLens dimensions | Checked against versioned TransformerLens 2.16.1 source: `hook_z` is `[batch, position, n_heads, d_head]`; `hook_result` is `[batch, position, n_heads, d_model]`. Added width and batch/position checks, plus validation that overrides belong to the requested layer and supported output hook type. Invalid layouts are rejected by executable tests. |
| 3. Paris/London single tokens | Verified canonical GPT-2 byte encoding and ranked BPE merges: `" Paris" -> [6342]`, `" London" -> [3576]`. The scalar next-token logit difference remains appropriate. Runtime ID discovery, single-token assertions, and exact decoding checks remain in the notebook. |
| 4. CUDA synchronization for timing | Synchronization occurs before and after forward and stage timing, and before the overall timer starts. CPU tests verify the wrapper's synchronization order and the CUDA helper's device guard. The full CUDA run also exercised these synchronization calls successfully. |
| 5. No unnecessary inference gradients | All model forwards route through a `torch.no_grad()` wrapper. Intervention scoring also uses `no_grad`, and model parameters have `requires_grad_(False)`. CPU tests use inputs requiring gradients and confirm the hook outputs do not retain a graph. |
| 6. Disjoint train/test masks | Executed the notebook's sampler: 64 unique training masks, 32 unique test masks, zero overlap. The zero mask appears only in training. Repeat sampling with the same seed is identical. |
| 7. Unique pairwise features | Exactly 28 unique pairs for eight components, all satisfying `i < j`. Tested feature values against logical conjunction for all 256 possible masks. The quadratic training matrix is `(64, 36)` with rank 36 and condition number approximately 18.98. |
| 8. Real-forward counter | Executable tests verify that new binary masks/scaling endpoints increment the counter, while repeated points and cached baselines do not. Static inspection confirms all forward calls pass through `run_model`. The summary now separates actual acquisition forwards (`model_evaluations`) from estimated standalone acquisition budgets. Failed forward attempts remain counted and are marked in `forward_log`. |
| 9. Held-out performance metrics | Fixed section 19: sketch-budget MAE/Pearson/Spearman/R2 now use `y_test` and predictions on `X_test`, matching both regressions. All six top-level accuracy/ranking calls use held-out labels. Singleton and coordinate-gradient reference values remain explicitly labeled diagnostics and plots, without reference-set accuracy metrics. Regression fits use training labels only; sketch budget selection remains prespecified. |
| 10. Every cell and likely runtime errors | All 23 code cells compile and subsequently executed on real GPT-2/CUDA without errors. Static review additionally covered optional branches. The reporting issues and hook-validation gaps above were fixed. Compatible NumPy, PyArrow, Datasets, TransformerLens, and Transformers versions are pinned in requirements.txt. |

Validation commands, all passed:

```powershell
py -3.11 -m unittest discover -s review_checks -p test_notebook_dataflow.py -v
py -3.11 review_checks\test_hook_contracts.py
python review_checks\verify_gpt2_candidates.py
```

The first command runs 12 tests against the actual notebook definitions and syntax tree.
The second executes the actual notebook hook/cache/timing implementations using real PyTorch
CPU tensors and a small fake model. It is a contract test, not GPT-2 inference.
The third downloads only canonical tokenizer vocabulary/merge files and verifies the two
candidate encodings without installing a tokenizer package or downloading model weights.
Its recorded source URLs, checksums, and token IDs are in `gpt2_candidate_tokenization.json`.

## Full CUDA validation

After the CUDA PyTorch installation, the project Python 3.11 environment at
`.venv/Scripts/python.exe` was verified to use PyTorch `2.14.1+cu126`, CUDA 12.6, and an
NVIDIA GeForce RTX 3060. The remaining notebook dependencies were installed from the
new `requirements.txt`. Select the registered **Python (mib-sketching CUDA)** kernel
for interactive execution; the default system Python is a different environment.

```powershell
.\.venv\Scripts\python.exe run_notebook.py
.\.venv\Scripts\python.exe review_checks\validate_cuda_results.py
```

Both commands passed. All 23 code cells executed without errors, including actual GPT-2
loading, tokenization guards, attention interventions, regressions, sketches, and exports.
Independent validation recomputed held-out metrics from exported test predictions,
confirmed zero train/test overlap, and matched 218 logged forwards to 218 expected real
forward passes. Five plots were exported. Peak allocated GPU memory was 646.64 MiB
(690 MiB reserved); synchronized forward time totaled 10.20 seconds. End-to-end execution
took 251.16 seconds including kernel startup and the initial model download.

Execution evidence is in `cuda_execution.json` and `cuda_results_validation.json`.
Real experiment outputs are under `../results/20261014T213306_084641Z_6da48ba0/`.
This run validates the notebook's default configuration; optional configurations were
statically inspected and covered by targeted contract tests where applicable.

Sources:

- [TransformerLens 2.16.1 attention implementation](https://raw.githubusercontent.com/TransformerLensOrg/TransformerLens/v2.16.1/transformer_lens/components/abstract_attention.py)
- [Canonical GPT-2 vocabulary](https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json)
- [Canonical GPT-2 BPE merges](https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt)
