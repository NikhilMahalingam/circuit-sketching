"""Focused invariant tests; real MIB/CUDA integration is checked by execution."""
from pathlib import Path
import sys
import unittest
import itertools

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbformat
import numpy as np
import torch
from mib_ioi.runtime import patch_heads
from mib_ioi import analysis as an


class IOIInvariants(unittest.TestCase):
    def test_only_selected_heads_are_replaced(self):
        original = torch.randn(2, 5, 12, 8)
        saved = original.clone()
        donor = torch.randn(2, 5, 2, 8)
        result = patch_heads(original, donor, [2, 7], 12, 8)
        torch.testing.assert_close(original, saved, rtol=0, atol=0)
        torch.testing.assert_close(result[:, :, [2, 7]], donor, rtol=0, atol=0)
        other = [h for h in range(12) if h not in [2, 7]]
        torch.testing.assert_close(result[:, :, other], original[:, :, other], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            patch_heads(original, donor[:, :4], [2, 7], 12, 8)

    def test_disjoint_multi_density_design(self):
        for n in (12, 16):
            splits, allocation = an.mask_splits(n, 42)
            sets = [set(map(tuple, a)) for a in splits.values()]
            self.assertTrue(all(a.isdisjoint(b) for a, b in itertools.combinations(sets, 2)))
            for mask in splits.values():
                self.assertEqual(len(set(map(tuple, mask))), len(mask))
                self.assertEqual(set(mask.sum(1)), {1, 2, round(n*.25), round(n*.5), round(n*.75)})
            design = an.features(splits['train'], 2)
            self.assertEqual(design.shape[1], n+n*(n-1)//2)
            self.assertEqual(np.linalg.matrix_rank(design), design.shape[1])

    def test_quadratic_recovers_known_interactions(self):
        splits, _ = an.mask_splits(12, 91)
        rng = np.random.default_rng(1)
        beta = rng.normal(size=78)
        train = an.features(splits['train'], 2) @ beta
        fitted = an.fit(splits['train'], train, 2)
        np.testing.assert_allclose(an.features(splits['test'], 2) @ fitted,
                                   an.features(splits['test'], 2) @ beta, atol=1e-10)

    def test_bootstrap_resamples_examples(self):
        rng = np.random.default_rng(2)
        singles = rng.normal(size=(3, 30))
        pairs = list(itertools.combinations(range(3), 2))
        exact = np.stack([singles[i]+singles[j]+.2 for i, j in pairs])
        table, summary, interactions, draws = an.pair_statistics(
            singles, exact, pairs, [(0, 0), (1, 1), (2, 2)], 'validation', 100, 42)
        np.testing.assert_allclose(interactions, .2)
        self.assertAlmostEqual(summary['MAE'], .2)
        self.assertAlmostEqual(summary['mean_absolute_interaction'], .2)
        self.assertEqual(len(draws), 100)

    def test_notebook_schema_and_every_code_cell(self):
        notebook = nbformat.read(ROOT / 'mib_ioi_experiments.ipynb', as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                compile(cell.source, f'cell_{index}', 'exec')
                self.assertEqual(cell.outputs, [])
        sections = [c.source for c in notebook.cells if c.cell_type == 'markdown']
        for number in range(1, 20):
            self.assertTrue(any(s.startswith(f'## {number}.') for s in sections))


if __name__ == '__main__':
    unittest.main(verbosity=2)
