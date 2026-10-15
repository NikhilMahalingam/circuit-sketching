"""Offline scientific-dataflow tests for the experiment environment.

These tests use a known polynomial scorer, not GPT-2, so they can independently
verify leakage, fitting, feature construction, and cost accounting without model
downloads. They do not report synthetic scores as language-model results.
"""

import itertools
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from circuit_sketching import experiments as exp
from circuit_sketching.core import PromptSpec


class PolynomialScorer:
    """Implements the public scorer contract with observable real-call caching."""

    instances = []
    validation_multiplier = 1.0

    def __init__(self, model, components, prompts):
        self.model = model
        self.components = tuple(components)
        self.n = len(components)
        self.prompts = {prompt.id: prompt for prompt in prompts}
        self.device = SimpleNamespace(type="cpu")
        self.stage = "unscoped"
        self.cache = {}
        self.records = []
        self.requests = self.cache_hits = self.forward_count = self.failed_forwards = 0
        self.forward_seconds = 0.0
        type(self).instances.append(self)

    def score(self, prompt_id, scales):
        assert prompt_id in self.prompts
        scales = np.asarray(scales, dtype=float)
        assert scales.shape == (self.n,) and np.isfinite(scales).all()
        self.requests += 1
        key = (prompt_id, tuple(scales.tolist()))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        prompt_number = int(prompt_id[-1])
        ablated = 1.0 - scales
        linear = np.array([(-1) ** i * (0.2 + 0.1 * i) for i in range(self.n)])
        # Distinct prompt effects make prompt leakage and wrong mean fitting visible.
        linear += prompt_number * np.linspace(0.01, 0.06, self.n)
        interaction = 0.3 * ablated[0] * ablated[1] - 0.2 * ablated[2] * ablated[4]
        interaction += 0.15 * ablated[3] * ablated[5]
        delta = float(ablated @ linear + interaction)
        if prompt_id.startswith("validation"):
            delta *= type(self).validation_multiplier
        value = 12.0 + prompt_number + delta
        self.cache[key] = value
        self.forward_count += 1
        self.forward_seconds += 0.001
        self.records.append({
            "stage": self.stage, "prompt_id": prompt_id, "scales": scales.tolist(),
            "seconds": 0.001, "success": True,
        })
        return value

    def baseline(self, prompt_id):
        return self.score(prompt_id, np.ones(self.n))

    def delta(self, prompt_id, masks):
        masks = np.asarray(masks, dtype=float)
        baseline = self.baseline(prompt_id)
        if masks.ndim == 1:
            return self.score(prompt_id, 1 - masks) - baseline
        return np.array([self.score(prompt_id, 1 - mask) - baseline for mask in masks])

    def cost_snapshot(self):
        return {
            "forward_count": self.forward_count, "failed_forwards": 0,
            "cache_hits": self.cache_hits, "requests": self.requests,
            "forward_seconds": self.forward_seconds,
            "peak_allocated_mib": None, "peak_reserved_mib": None,
        }

