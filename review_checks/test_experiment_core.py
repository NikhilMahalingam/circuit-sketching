"""Contract tests with real CPU tensors; no model weights or downloads needed."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from circuit_sketching.core import HeadScorer, PromptSpec


PROMPTS = [PromptSpec("france", "The capital of France is", " Paris", " London")]
COMPONENTS = [(0, 1), (0, 3), (2, 0)]


class FakeModel(torch.nn.Module):
    def __init__(self, kind="z"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.cfg = SimpleNamespace(n_layers=3, n_heads=4, d_head=5, d_model=7)
        self.kind = kind
        self.hook_dict = {f"blocks.{i}.attn.hook_{kind}": None for i in range(3)}
        self.use_attn_result = False
        width = self.cfg.d_head if kind == "z" else self.cfg.d_model
        self.original = torch.arange(1, 1 + 4 * 4 * width, dtype=torch.float32).reshape(1, 4, 4, width)
        self.original.requires_grad_(True)
        self.outputs = {}
        self.calls = 0
        self.events = []
        self.fail = False
        self.skip_hooks = False
        self.candidates = {" Paris": 2, " London": 3, " Berlin": 4}
        self.tokenizer = SimpleNamespace(decode=self.decode)

    def decode(self, ids, **kwargs):
        return next(text for text, number in self.candidates.items() if number == ids[0])

    def to_tokens(self, text, prepend_bos):
        if prepend_bos:
            return torch.tensor([[1, 9, 8, 7]])
        return torch.tensor([[self.candidates[text]]]) if text in self.candidates else torch.tensor([[5, 6]])

    def to_single_token(self, text):
        return self.candidates[text]

    def set_use_attn_result(self, enabled):
        self.use_attn_result = enabled

    def run_with_hooks(self, tokens, fwd_hooks, reset_hooks_end):
        assert not torch.is_grad_enabled()
        assert not self.training
        assert not self.weight.requires_grad
        assert reset_hooks_end
        assert tokens.device == self.weight.device
        self.calls += 1
        self.events.append("forward")
        if self.fail:
            raise RuntimeError("deliberate forward failure")
        total = 0.0
        if not self.skip_hooks:
            for name, hook in fwd_hooks:
                before = self.original.detach().clone()
                edited = hook(self.original, SimpleNamespace(name=name))
                assert torch.equal(before, self.original)
                assert edited.data_ptr() != self.original.data_ptr()
                assert not edited.requires_grad
                self.outputs[name] = edited
                total = total + edited.sum()
        logits = torch.zeros(1, 4, 10)
        logits[0, -1, 2] = total
        return logits


class HeadScorerContracts(unittest.TestCase):
    def test_initialization_does_not_evaluate_any_prompts(self):
        model = FakeModel()
        scorer = HeadScorer(model, COMPONENTS, PROMPTS)
        self.assertEqual(model.calls, 0)
        self.assertEqual(scorer.forward_count, 0)
        self.assertEqual(scorer.requests, 0)
        self.assertFalse(scorer.cache)
        self.assertFalse(scorer.records)

    def test_only_selected_heads_are_edited_for_both_hook_layouts(self):
        for kind in ("z", "result"):
            with self.subTest(kind=kind):
                model = FakeModel(kind)
                scorer = HeadScorer(model, COMPONENTS, PROMPTS)
                value = scorer.score("france", [0.0, 0.5, -1.0])
                self.assertIsInstance(value, float)
                self.assertEqual(set(model.outputs), {f"blocks.{i}.attn.hook_{kind}" for i in (0, 2)})
                for layer, selections in {0: {1: 0.0, 3: 0.5}, 2: {0: -1.0}}.items():
                    edited = model.outputs[f"blocks.{layer}.attn.hook_{kind}"]
                    for head in range(4):
                        self.assertTrue(torch.equal(
                            edited[:, :, head, :],
                            model.original.detach()[:, :, head, :] * selections.get(head, 1.0)))
                if kind == "result":
                    self.assertTrue(model.use_attn_result)

    def test_wrong_head_width_rank_or_batch_position_fails_and_is_counted(self):
        for shape in ((1, 4, 20), (1, 4, 4, 7), (1, 4, 5, 5), (2, 4, 4, 5), (1, 3, 4, 5)):
            with self.subTest(shape=shape):
                model = FakeModel()
                scorer = HeadScorer(model, COMPONENTS, PROMPTS)
                model.original = torch.zeros(shape)
                with self.assertRaisesRegex(RuntimeError, "Unexpected layout"):
                    scorer.baseline("france")
                self.assertEqual((scorer.forward_count, scorer.failed_forwards), (1, 1))
                self.assertFalse(scorer.cache)
                self.assertFalse(scorer.records[0]["success"])

    def test_component_guards(self):
        for components in ([], [(0, 1), (0, 1)], [(0, 4)], [(3, 0)], [(-1, 0)],
                           [(0, 1.0)], [(True, 0)], [(0,)], [1]):
            with self.subTest(components=components), self.assertRaises(ValueError):
                HeadScorer(FakeModel(), components, PROMPTS)

    def test_multitoken_roundtrip_and_distinct_token_guards(self):
        with self.assertRaisesRegex(ValueError, "single-token"):
            HeadScorer(FakeModel(), COMPONENTS,
                       [PromptSpec("bad", "Prompt", " Paris London", " London")])
        model = FakeModel()
        model.tokenizer.decode = lambda *args, **kwargs: "mismatch"
        with self.assertRaisesRegex(ValueError, "round-trip"):
            HeadScorer(model, COMPONENTS, PROMPTS)
        with self.assertRaisesRegex(ValueError, "identical"):
            HeadScorer(FakeModel(), COMPONENTS,
                       [PromptSpec("bad", "Prompt", " Paris", " Paris")])
        model = FakeModel()
        model.to_single_token = lambda text: 9
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            HeadScorer(model, COMPONENTS, PROMPTS)

    def test_duplicate_prompt_ids_and_empty_strings_fail(self):
        for prompts in ([], PROMPTS * 2, [PromptSpec("", "text", " Paris", " London")]):
            with self.subTest(prompts=prompts), self.assertRaises(ValueError):
                HeadScorer(FakeModel(), COMPONENTS, prompts)

    def test_cache_exact_endpoint_identity_and_prompt_separation(self):
        model = FakeModel()
        prompts = PROMPTS + [PromptSpec("second", "A second prompt", " Paris", " Berlin")]
        scorer = HeadScorer(model, COMPONENTS, prompts)
        scorer.stage = "acquisition"
        first = scorer.score("france", [0.0, 0.5, 1.0])
        self.assertEqual(first, scorer.score("france", np.array([0, .5, 1])))
        scorer.score("second", [0.0, 0.5, 1.0])
        scorer.score("france", [0.0, 0.5 + 1e-12, 1.0])
        self.assertEqual(model.calls, 3)
        self.assertEqual(scorer.forward_count, 3)
        self.assertEqual(scorer.cache_hits, 1)
        self.assertEqual(len(scorer.cache), 3)
        self.assertTrue(all(type(value) is float for value in scorer.cache.values()))
        self.assertTrue(all(row["stage"] == "acquisition" for row in scorer.records))
        self.assertTrue(all(row["success"] and row["seconds"] >= 0 for row in scorer.records))
        self.assertAlmostEqual(scorer.forward_seconds, sum(row["seconds"] for row in scorer.records))

    def test_failed_forwards_are_counted_but_never_cached(self):
        model = FakeModel()
        scorer = HeadScorer(model, COMPONENTS, PROMPTS)
        model.fail = True
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                scorer.baseline("france")
        self.assertEqual((model.calls, scorer.forward_count, scorer.failed_forwards), (2, 2, 2))
        self.assertFalse(scorer.cache)
        model.fail = False
        scorer.baseline("france")
        scorer.baseline("france")
        self.assertEqual((scorer.forward_count, scorer.failed_forwards, scorer.cache_hits), (3, 2, 1))

    def test_hooks_must_fire(self):
        model = FakeModel()
        scorer = HeadScorer(model, COMPONENTS, PROMPTS)
        model.skip_hooks = True
        with self.assertRaisesRegex(RuntimeError, "did not fire"):
            scorer.baseline("france")
        self.assertEqual(scorer.failed_forwards, 1)

    def test_invalid_scales_and_masks_do_not_call_model(self):
        model = FakeModel()
        scorer = HeadScorer(model, COMPONENTS, PROMPTS)
        for scales in ([1, 1], [[1, 1, 1]], [np.nan, 0, 0], [np.inf, 0, 0]):
            with self.subTest(scales=scales), self.assertRaises(ValueError):
                scorer.score("france", scales)
        for masks in ([0, .5, 1], [0, 1], [[0, 1, np.nan]], [[[1, 1, 1]]]):
            with self.subTest(masks=masks), self.assertRaises(ValueError):
                scorer.delta("france", masks)
        with self.assertRaises(KeyError):
            scorer.baseline("missing")
        self.assertEqual(model.calls, 0)

    def test_mask_delta_sign_and_shared_baseline_cache(self):
        scorer = HeadScorer(FakeModel(), COMPONENTS, PROMPTS)
        np.testing.assert_array_equal(scorer.delta("france", [[0, 0, 0], [0, 0, 0]]), [0, 0])
        self.assertEqual(scorer.forward_count, 1)
        delta = scorer.delta("france", [1, 0, 0])
        expected = -float(scorer.model.original[:, :, 1, :].sum().item())
        self.assertEqual(delta, expected)
        self.assertEqual(scorer.forward_count, 2)

    def test_cuda_sync_surrounds_actual_forwards_and_skips_cached_calls(self):
        model = FakeModel()
        scorer = HeadScorer(model, COMPONENTS, PROMPTS)
        # Intercept CUDA methods to exercise ordering without requiring a GPU in CI.
        scorer.device = torch.device("cuda:0")
        with patch("torch.cuda.synchronize", side_effect=lambda device: model.events.append("sync")), \
             patch.object(scorer, "_capture_peak"):
            scorer.baseline("france")
            scorer.baseline("france")
        self.assertEqual(model.events, ["sync", "forward", "sync"])

    def test_cpu_snapshot_has_honest_counts_and_no_gpu_memory_claim(self):
        scorer = HeadScorer(FakeModel(), COMPONENTS, PROMPTS)
        scorer.baseline("france")
        scorer.baseline("france")
        snapshot = scorer.cost_snapshot()
        self.assertEqual(snapshot["forward_count"], 1)
        self.assertEqual(snapshot["failed_forwards"], 0)
        self.assertEqual(snapshot["cache_hits"], 1)
        self.assertEqual(snapshot["requests"], 2)
        self.assertIsNone(snapshot["peak_allocated_mib"])
        self.assertIsNone(snapshot["peak_reserved_mib"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
