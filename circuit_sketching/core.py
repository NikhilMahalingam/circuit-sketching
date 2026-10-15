"""Verified, cached attention-head interventions for TransformerLens.

No model is loaded at import. Scales multiply selected head outputs at every token
position. Binary masks instead use 1=ablated, and ``delta`` is the intervened
correct-minus-incorrect logit difference minus its unmodified baseline.
"""

from dataclasses import dataclass
from numbers import Integral
import re
import time

import numpy as np
import torch


@dataclass(frozen=True)
class PromptSpec:
    id: str
    text: str
    correct: str
    incorrect: str


def _single_token_id(model, text):
    ids = model.to_tokens(text, prepend_bos=False).reshape(-1)
    if ids.numel() != 1:
        raise ValueError(
            f"Candidate {text!r} produces {ids.numel()} tokens. This experiment "
            "supports single-token next-token candidates only; choose another "
            "candidate (including its leading space), or implement and separately "
            "validate a sequence log-probability metric before using it."
        )
    token_id = int(ids.item())
    if int(model.to_single_token(text)) != token_id:
        raise RuntimeError("Tokenizer helpers disagree about the candidate token ID.")
    decoded = model.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
    if decoded != text:
        raise ValueError(f"Candidate token round-trip differs: {text!r} -> {decoded!r}.")
    return token_id


def _validate_components(model, components):
    normalized = []
    for pair in components:
        if not isinstance(pair, (tuple, list, np.ndarray)) or len(pair) != 2:
            raise ValueError(f"Expected an integer (layer, head) pair, got {pair!r}.")
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
               for value in pair):
            raise ValueError(f"Layer and head indices must be integers: {pair!r}.")
        layer, head = map(int, pair)
        if not (0 <= layer < model.cfg.n_layers and 0 <= head < model.cfg.n_heads):
            raise ValueError(f"Head {(layer, head)} is outside the model configuration.")
        normalized.append((layer, head))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("Select at least one head, with no duplicate (layer, head) pairs.")
    return tuple(normalized)


def _resolve_head_hook(model, layer):
    for kind in ("z", "result"):
        expected = f"blocks.{layer}.attn.hook_{kind}"
        matches = [name for name in model.hook_dict
                   if re.search(rf"(?:^|\.)(?:blocks|layers|h)\.{layer}\.", name)
                   and name.endswith(f"hook_{kind}")]
        name = expected if expected in model.hook_dict else (
            matches[0] if len(matches) == 1 else None)
        if name is None:
            continue
        if kind == "result":
            if not hasattr(model, "set_use_attn_result"):
                continue
            model.set_use_attn_result(True)
        return name
    raise RuntimeError(
        f"No supported per-head output hook for layer {layer}. Inspect model.hook_dict; "
        "a hook_z or enabled hook_result is required, never merged hook_attn_out."
    )


class HeadScorer:
    """Cached single-prompt forwards with verified head edits and honest costs.

    ``score(prompt_id, scales)`` returns a Python float. ``baseline(prompt_id)``
    uses all-one scales. ``delta(prompt_id, masks)`` accepts one binary mask or a
    matrix of masks and returns intervened score minus baseline, so a negative
    delta means removal reduced the correct-minus-incorrect logit difference.

    Set ``stage`` before a group of requests to label subsequent forward records.
    Cache keys are ``(prompt_id, tuple(scales))`` and values contain no tensors.
    ``forward_count`` counts real forward attempts, including failed forwards;
    ``failed_forwards`` records those failures separately. Successful cache hits
    never increment either. Do not change model weights, tokenizer, selected
    heads, or prompt definitions while reusing a scorer; create a new instance.

    CUDA memory is the process-wide device peak (including model loading and
    earlier work); this scorer never resets global CUDA memory counters. A scorer
    supports one model device; sharded models are rejected.
    """

    def __init__(self, model, components, prompts):
        self.model = model
        self.components = _validate_components(model, components)
        self.n = len(self.components)
        parameters = list(model.parameters())
        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1:
            raise ValueError("HeadScorer requires a model with parameters on one device.")
        self.device = next(iter(devices))
        model.eval()
        model.requires_grad_(False)
        self.prompts = {}
        self.tokens = {}
        self.candidate_ids = {}
        for prompt in prompts:
            if not isinstance(prompt, PromptSpec):
                raise TypeError("Each prompt must be a PromptSpec.")
            if any(not isinstance(value, str) or not value
                   for value in (prompt.id, prompt.text, prompt.correct, prompt.incorrect)):
                raise ValueError("Prompt IDs, text, and both candidates must be nonempty strings.")
            if prompt.id in self.prompts:
                raise ValueError(f"Duplicate prompt ID: {prompt.id!r}.")
            correct = _single_token_id(model, prompt.correct)
            incorrect = _single_token_id(model, prompt.incorrect)
            if correct == incorrect:
                raise ValueError(f"Prompt {prompt.id!r} has identical candidate token IDs.")
            tokens = model.to_tokens(prompt.text, prepend_bos=True).to(self.device)
            if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
                raise ValueError(f"Prompt {prompt.id!r} must tokenize to [1, positive_length].")
            self.prompts[prompt.id] = prompt
            self.tokens[prompt.id] = tokens
            self.candidate_ids[prompt.id] = (correct, incorrect)
        if not self.prompts:
            raise ValueError("At least one prompt is required.")
        self.hook_names = {
            layer: _resolve_head_hook(model, layer) for layer, _ in self.components
        }
        self.cache = {}
        self.stage = "unscoped"
        self.forward_count = 0
        self.failed_forwards = 0
        self.cache_hits = 0
        self.requests = 0
        self.forward_seconds = 0.0
        self.records = []
        self._peak_allocated_mib = None
        self._peak_reserved_mib = None
        self._synchronize()
        self._capture_peak()

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _capture_peak(self):
        if self.device.type == "cuda":
            self._peak_allocated_mib = max(
                self._peak_allocated_mib or 0.0,
                torch.cuda.max_memory_allocated(self.device) / 2**20)
            self._peak_reserved_mib = max(
                self._peak_reserved_mib or 0.0,
                torch.cuda.max_memory_reserved(self.device) / 2**20)

    def _hook(self, name, selected, tokens, fired):
        width = self.model.cfg.d_head if name.endswith("hook_z") else self.model.cfg.d_model

        def scale_hook(activation, hook):
            expected = (*tokens.shape, self.model.cfg.n_heads, width)
            if hook.name != name or tuple(activation.shape) != expected:
                raise RuntimeError(
                    f"Unexpected layout at {hook.name}: {tuple(activation.shape)}; "
                    f"expected {expected} [batch, position, head, head_width]."
                )
            fired.add(name)
            edited = activation.clone()
            for head, scale in selected.items():
                edited[:, :, head, :] = activation[:, :, head, :] * scale
            return edited

        return scale_hook

    def score(self, prompt_id, scales):
        """Return correct-minus-incorrect next-token logits at given head scales."""
        if prompt_id not in self.prompts:
            raise KeyError(f"Unknown prompt ID: {prompt_id!r}.")
        scales = np.asarray(scales, dtype=np.float64)
        if scales.shape != (self.n,) or not np.isfinite(scales).all():
            raise ValueError(f"Scales must be {self.n} finite numbers in a one-dimensional vector.")
        key = (prompt_id, tuple(map(float, scales)))
        self.requests += 1
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        tokens = self.tokens[prompt_id]
        by_layer = {}
        for (layer, head), scale in zip(self.components, scales):
            by_layer.setdefault(layer, {})[head] = float(scale)
        fired = set()
        hooks = [(self.hook_names[layer], self._hook(
            self.hook_names[layer], selected, tokens, fired))
            for layer, selected in by_layer.items()]
        self._synchronize()
        started = time.perf_counter()
        self.forward_count += 1
        success = False
        try:
            with torch.no_grad():
                logits = self.model.run_with_hooks(tokens, fwd_hooks=hooks, reset_hooks_end=True)
                if fired != {name for name, _ in hooks}:
                    raise RuntimeError("Some registered head hooks did not fire.")
                if logits.ndim != 3 or tuple(logits.shape[:2]) != tuple(tokens.shape):
                    raise RuntimeError("Model must return [batch, position, vocabulary] logits.")
                correct, incorrect = self.candidate_ids[prompt_id]
                if max(correct, incorrect) >= logits.shape[-1]:
                    raise RuntimeError("Candidate token ID is outside the model output vocabulary.")
                value = float((logits[0, -1, correct] - logits[0, -1, incorrect]).item())
                if not np.isfinite(value):
                    raise ValueError("Model returned a nonfinite logit difference.")
            self.cache[key] = value
            success = True
            return value
        finally:
            if not success:
                self.failed_forwards += 1
            self._synchronize()
            seconds = time.perf_counter() - started
            self.forward_seconds += seconds
            self._capture_peak()
            self.records.append({
                "stage": self.stage, "prompt_id": prompt_id, "scales": list(key[1]),
                "seconds": seconds, "success": success,
            })

    def baseline(self, prompt_id):
        """Return/cache the unmodified score (all head scales equal one)."""
        return self.score(prompt_id, np.ones(self.n, dtype=np.float64))

    def delta(self, prompt_id, masks):
        """Intervened minus baseline score; binary masks use 1=ablated."""
        masks = np.asarray(masks)
        if masks.ndim not in (1, 2) or masks.shape[-1] != self.n:
            raise ValueError(f"Masks must have shape ({self.n},) or (n_masks, {self.n}).")
        if not np.isin(masks, (0, 1)).all():
            raise ValueError("Mask entries must be binary: 1=ablated, 0=unchanged.")
        baseline = self.baseline(prompt_id)
        if masks.ndim == 1:
            return self.score(prompt_id, 1.0 - masks) - baseline
        return np.asarray([self.score(prompt_id, 1.0 - mask) - baseline for mask in masks])

    def cost_snapshot(self):
        """Return cumulative costs; cache entries and cache hits are not forwards."""
        self._synchronize()
        self._capture_peak()
        return {
            "forward_count": self.forward_count,
            "failed_forwards": self.failed_forwards,
            "cache_hits": self.cache_hits,
            "requests": self.requests,
            "forward_seconds": self.forward_seconds,
            "peak_allocated_mib": self._peak_allocated_mib,
            "peak_reserved_mib": self._peak_reserved_mib,
        }
