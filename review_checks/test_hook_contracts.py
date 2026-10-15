"""CPU contract checks for the notebook's hooks and forward wrapper; no model download.

Uses real PyTorch tensors and extracts the notebook implementations without running
model-loading or experiment cells. The fake model tests hook indexing, not GPT-2.
"""
import ast
from contextlib import contextmanager
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
notebook = json.loads((ROOT / "notebooks" / "mib_sketching_baseline.ipynb").read_text(encoding="utf-8"))
definitions = {}
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                definitions[node.name] = node

events = []
namespace = {
    "torch": torch, "np": np, "re": re, "time": time,
    "contextmanager": contextmanager, "device": "cpu",
    "synchronize_cuda": lambda: events.append("sync"),
    "package_versions": {"transformer-lens": "static-contract-test"},
    "HEAD_HOOK_OVERRIDES": {},
    "tl_utils": SimpleNamespace(get_act_name=lambda kind, layer: f"blocks.{layer}.attn.hook_{kind}"),
}
names = [
    "CostTracker", "run_model", "print_hook_diagnostics", "hook_is_in_layer",
    "resolve_head_hook", "validate_head_activation", "validate_components",
    "scale_selected_head_outputs", "scaled_head_score", "ablate_attention_heads",
    "single_token_id", "mask_key", "evaluate_mask", "evaluate_scaling",
]
for name in names:
    exec(compile(ast.Module(body=[definitions[name]], type_ignores=[]), "<notebook>", "exec"), namespace)
namespace["costs"] = namespace["CostTracker"]()


class FakeModel:
    def __init__(self, kind):
        self.cfg = SimpleNamespace(n_layers=3, n_heads=4, d_head=5, d_model=7)
        self.kind = kind
        self.hook_dict = {f"blocks.{i}.attn.hook_{kind}": None for i in range(3)}
        self.use_attn_result = False
        width = self.cfg.d_head if kind == "z" else self.cfg.d_model
        self.original = torch.arange(1, 1 + 2 * 6 * 4 * width, dtype=torch.float32).reshape(2, 6, 4, width)
        self.original.requires_grad_(True)  # wrapper must disable autograd even for this input.
        self.outputs = {}
        self.calls = 0

    def set_use_attn_result(self, enabled):
        self.use_attn_result = enabled

    def run_with_hooks(self, tokens, fwd_hooks, reset_hooks_end):
        assert not torch.is_grad_enabled()
        assert reset_hooks_end is True
        self.calls += 1
        events.append("forward")
        for name, hook in fwd_hooks:
            before = self.original.detach().clone()
            result = hook(self.original, SimpleNamespace(name=name))
            assert torch.equal(self.original, before), "Hook mutated its input activation"
            assert not result.requires_grad
            assert result.data_ptr() != self.original.data_ptr()
            self.outputs[name] = result
        return torch.zeros(2, 6, 11)


for kind in ("z", "result"):
    model = FakeModel(kind)
    chosen = [(0, 1), (0, 3), (2, 0)]
    scales = [0.0, 0.5, -1.0]
    tokens = torch.zeros(2, 6, dtype=torch.long)
    start = namespace["costs"].model_evaluations
    events.clear()
    score = namespace["scaled_head_score"](model, tokens, chosen, scales, lambda logits: logits.sum())
    assert score == 0.0
    assert model.calls == 1
    assert namespace["costs"].model_evaluations == start + 1
    assert events == ["sync", "forward", "sync"], events
    assert set(model.outputs) == {f"blocks.{i}.attn.hook_{kind}" for i in (0, 2)}
    selected = {0: {1: 0.0, 3: 0.5}, 2: {0: -1.0}}
    for layer, heads in selected.items():
        result = model.outputs[f"blocks.{layer}.attn.hook_{kind}"]
        for head in range(model.cfg.n_heads):
            expected = model.original[:, :, head, :].detach() * heads.get(head, 1.0)
            assert torch.equal(result[:, :, head, :], expected)
    if kind == "result":
        assert model.use_attn_result


def rejects(exception_type, operation):
    with redirect_stdout(io.StringIO()):
        try:
            operation()
        except exception_type:
            return
    raise AssertionError(f"Expected {exception_type.__name__}")


# Strict shape and semantic guards reject a merged output and the wrong width.
model = FakeModel("z")
for shape in [(2, 6, 20), (2, 6, 4, 7), (2, 6, 5, 5)]:
    rejects(RuntimeError, lambda shape=shape: namespace["validate_head_activation"](
        model, "blocks.0.attn.hook_z", torch.zeros(shape)))
rejects(ValueError, lambda: namespace["validate_head_activation"](
    model, "blocks.0.hook_attn_out", torch.zeros(2, 6, 20)))
namespace["HEAD_HOOK_OVERRIDES"][0] = "blocks.2.attn.hook_z"
rejects(ValueError, lambda: namespace["resolve_head_hook"](model, 0))
namespace["HEAD_HOOK_OVERRIDES"].clear()
rejects(RuntimeError, lambda: namespace["scaled_head_score"](
    model, torch.zeros(1, 6, dtype=torch.long), [(0, 1)], [0.0], lambda logits: logits.sum()))
rejects(ValueError, lambda: namespace["validate_components"](model, [(0, 1), (0, 1)]))
rejects(ValueError, lambda: namespace["validate_components"](model, [(0, 4)]))

# CPU guard confirms the actual CUDA helper invokes synchronization only on CUDA.
cuda_sync = []
cuda_namespace = {"device": "cpu", "torch": SimpleNamespace(cuda=SimpleNamespace(
    synchronize=lambda: cuda_sync.append("cuda.synchronize")))}
exec(compile(ast.Module(body=[definitions["synchronize_cuda"]], type_ignores=[]),
             "<notebook>", "exec"), cuda_namespace)
cuda_namespace["synchronize_cuda"]()
assert cuda_sync == []
cuda_namespace["device"] = "cuda"
cuda_namespace["synchronize_cuda"]()
assert cuda_sync == ["cuda.synchronize"]

# Both binary-mask and continuous endpoint cache hits must skip real forwards.
namespace.update({"model": model, "tokens": tokens, "components": [(0, 1), (0, 3), (2, 0)],
                  "n": 3, "mask_cache": {(0, 0, 0): 0.0},
                  "scaling_cache": {(0.0, 0.0, 0.0): 0.0},
                  "logit_difference": lambda logits: logits.sum()})
start = namespace["costs"].model_evaluations
namespace["evaluate_mask"]([0, 0, 0])
namespace["evaluate_mask"]([1, 0, 0])
namespace["evaluate_mask"]([1, 0, 0])
assert namespace["costs"].model_evaluations == start + 1
assert namespace["costs"].mask_requests == 3
assert namespace["costs"].mask_cache_hits == 2
namespace["evaluate_scaling"]([0, 0, 0])
namespace["evaluate_scaling"]([0.05, 0, 0])
namespace["evaluate_scaling"]([0.05, 0, 0])
assert namespace["costs"].model_evaluations == start + 2
rejects(ValueError, lambda: namespace["mask_key"]([1, 0.5, 0]))
rejects(ValueError, lambda: namespace["evaluate_scaling"]([float("nan"), 0, 0]))

# Guards prevent silent use of the first token or inconsistent decode helpers.
class FakeTokenizerModel:
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: " Paris")
    def to_tokens(self, text, prepend_bos):
        assert prepend_bos is False
        return torch.tensor([[6342]])
    def to_single_token(self, text):
        return 6342

tokenizer_model = FakeTokenizerModel()
assert namespace["single_token_id"](tokenizer_model, " Paris") == 6342
rejects(ValueError, lambda: namespace["single_token_id"](tokenizer_model, "Paris"))
tokenizer_model.to_tokens = lambda text, prepend_bos: torch.tensor([[1, 2]])
rejects(ValueError, lambda: namespace["single_token_id"](tokenizer_model, " Paris London"))

print("PASS: hook_z and hook_result layouts; same-layer/multilayer head selection;")
print("untouched heads exactly preserved; input not mutated; no-grad forwards;")
print("one real forward counted; synchronization before and after forward;")
print("shape/semantic guards; mask and scaling cache counts; tokenizer guard failures.")
