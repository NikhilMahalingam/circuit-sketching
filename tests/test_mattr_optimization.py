"""Unit-only artificial curves verify the actual official metric implementation."""
from contextlib import redirect_stdout
import io
import csv
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.mattr_optimization.common import configure, GRID, integration_weights
configure()
from MIB_circuit_track import evaluation
from matryoshka_attribution import learn_scores
from matryoshka_attribution.sigmoid_topk import sigmoid_topk
from src.mattr_optimization.checkpoint import checkpointed


class MetricTests(unittest.TestCase):
    def official(self, fractions=None, log=False):
        grid = GRID if fractions is None else fractions
        ys = [0., .2, .3, .4, .6, 1.2, 1.4, 1.3, 1.1, 1.]
        class Graph:
            edges = dict.fromkeys(range(32491))
            n = 0
            def apply_topn(self, n, *a, **kw):
                self.n = n
            def weighted_edge_count(self):
                return self.n
        # References have mean clean=5, mean corrupted=1. Per-example ratios would differ.
        outputs = [torch.tensor([0., 2.])] + [torch.tensor([1.+4*y, 1.+4*y]) for y in ys[:len(grid)]]
        graph = Graph()
        with patch.object(evaluation, 'evaluate_baseline', return_value=torch.tensor([2., 8.])), \
             patch.object(evaluation, 'evaluate_graph', side_effect=outputs), redirect_stdout(io.StringIO()):
            result = evaluation.evaluate_area_under_curve(None, graph, None, None,
                level='edge', absolute=False, log_scale=log, percentages=fractions)
        return result, ys[:len(grid)]

    def test_manual_trapezoid_matches_official_linear_cpr_and_cmd(self):
        result, y = self.official()
        expected = sum((b-a)*(u+v)/2 for a,b,u,v in zip(GRID, GRID[1:], y, y[1:]))
        cmd = sum((b-a)*(abs(1-u)+abs(1-v))/2 for a,b,u,v in zip(GRID, GRID[1:], y, y[1:]))
        self.assertAlmostEqual(result[1], expected, places=6)
        self.assertAlmostEqual(result[2], cmd, places=6)
        self.assertEqual(result[0], [int(p*32491) for p in GRID])
        self.assertGreater(max(result[4]), 1.)
        self.assertAlmostEqual(sum(integration_weights()), .999)
        self.assertAlmostEqual(sum(w*v for w,v in zip(integration_weights(), y)), expected)

    def test_endpoints_not_extended_to_zero_or_rescaled(self):
        class Graph:
            edges = dict.fromkeys(range(32491))
            def apply_topn(self, *a, **kw): pass
            def weighted_edge_count(self): return 0
        with patch.object(evaluation, 'evaluate_baseline', return_value=torch.tensor([5.])), \
             patch.object(evaluation, 'evaluate_graph', side_effect=[torch.tensor([1.])]+[torch.tensor([5.])]*10), \
             redirect_stdout(io.StringIO()):
            result = evaluation.evaluate_area_under_curve(None, Graph(), None, None)
        self.assertAlmostEqual(result[1], .999)
        self.assertEqual(result[2], 0.)

    def test_mask_sum_and_implicit_gradient(self):
        scores = torch.linspace(-1, 1, 100, requires_grad=True)
        mask = sigmoid_topk(scores, 27.5, T=.5, n_iters=30)
        self.assertAlmostEqual(float(mask.detach().sum()), 27.5, places=4)
        (mask*torch.arange(100.)).sum().backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertAlmostEqual(float(scores.grad.sum()), 0., places=3)

    def test_checkpoint_resume_matches_upstream_bitwise(self):
        def loss(mask):
            return (mask * torch.rand(12)).sum() * random.uniform(.1, 1.)
        kw = dict(total=12, loss_fn=loss, variant='topk', optimizer='adam', k_schedule='uniform',
                  T=.5, n_iters=30, lr=.05)
        torch.manual_seed(42); random.seed(42)
        reference = learn_scores(steps=20, **kw)
        with tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(42); random.seed(42)
            checkpointed(learn_scores, steps=10, folder=folder, identity='same', interval=10, **kw)
            torch.manual_seed(123); random.seed(123)
            resumed = checkpointed(learn_scores, steps=20, folder=folder, identity='same', interval=10, **kw)
            self.assertTrue(torch.equal(reference.scores, resumed.scores))
            self.assertEqual(reference.train_log, resumed.train_log)
            with self.assertRaises(ValueError):
                checkpointed(learn_scores, steps=20, folder=folder, identity='different', **kw)

    def test_gate1_audit_preserves_full_dataset_and_order(self):
        # Synthetic loader fixture only; never used as benchmark data.
        from types import SimpleNamespace
        from src.mattr_optimization.gate1 import FullData
        class Rows(list):
            _fingerprint = 'unit-only'
            cache_files = []
            def __getitem__(self, i):
                if isinstance(i, str):
                    return [row[i] for row in self]
                return super().__getitem__(i)
        class Dataset:
            def __init__(self, prefix):
                self.dataset = Rows({'prompt': prefix+str(i), 'metadata':
                    {'indirect_object': 'A', 'subject': 'B', 'random_c': 'C'}} for i in range(5))
            def __len__(self): return len(self.dataset)
            def __getitem__(self, i): return self.dataset[i]['prompt'], 'donor', [1, 2]
            def tokenizer(self, text, **kw):
                return SimpleNamespace(input_ids=[{' A': 1, ' B': 2, ' C': 3}[text]])
        with tempfile.TemporaryDirectory() as folder:
            data = FullData(Path(folder))
            train, validation = Dataset('train-'), Dataset('validation-')
            original = train.dataset
            data.audit(train, 'train'); data.audit(validation, 'validation')
            self.assertIs(train.dataset, original)
            with (Path(folder)/'train_manifest.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([r['clean_prompt'] for r in rows], ['train-'+str(i) for i in range(5)])
            self.assertEqual(len(rows), len(original))
            from src.mattr_optimization.resume_data_check import compare
            compare(train, Path(folder)/'train_manifest.csv')
            train.dataset[0]['prompt'] = 'changed-after-checkpoint'
            with self.assertRaises(ValueError): compare(train, Path(folder)/'train_manifest.csv')
            with self.assertRaises(ValueError): data.audit(validation, 'test')


if __name__ == '__main__':
    unittest.main()
