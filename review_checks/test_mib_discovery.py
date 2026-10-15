"""Fast circuit-search invariants without a model download."""
from itertools import combinations
from math import comb
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mib_ioi import discovery as d


class DiscoveryContracts(unittest.TestCase):
    def test_mask_means_remove_only_unkept_heads(self):
        mask = d.removal_mask(5, (0, 3))
        np.testing.assert_array_equal(mask, [0, 1, 1, 0, 1])
        self.assertEqual(d.retained(mask), (0, 3))
        self.assertEqual(len(list(d.candidates(5, 2))), comb(5, 2))
        self.assertEqual({tuple(x) for x in d.candidates(5, 2)},
                         {tuple(d.removal_mask(5, c)) for c in combinations(range(5), 2)})

    def test_exact_train_effects_fit_once_per_unordered_pair(self):
        single = np.array([-1.0, 2.0, 3.0, -4.0])
        pairs = list(combinations(range(4), 2))
        interactions = np.array([.2, -.3, .4, -.5, .6, -.7])
        pair_effects = np.array([single[i]+single[j]+interactions[p]
                                 for p, (i, j) in enumerate(pairs)])
        b1 = d.fit_first_order(single)
        b2 = d.fit_second_order(single, pair_effects, pairs)
        np.testing.assert_allclose(b1, single)
        np.testing.assert_allclose(b2, np.r_[single, interactions])
        for p, (i, j) in enumerate(pairs):
            z = np.zeros(4, dtype=int)
            z[[i, j]] = 1
            self.assertAlmostEqual(d.predicted_effect(z, b2, 2), pair_effects[p])
        with self.assertRaises(ValueError):
            d.fit_second_order(single, pair_effects, pairs[::-1])

    def test_all_methods_share_absolute_score_objective(self):
        n, k = 4, 2
        beta1 = np.array([-.5, .2, 1., -1.])
        chosen, prediction, _ = d.select_surrogate(n, k, beta1, 1)
        losses = [abs(d.predicted_effect(z, beta1, 1)) for z in d.candidates(n, k)]
        self.assertAlmostEqual(abs(prediction), min(losses))
        self.assertAlmostEqual(d.faithfulness(9.8, 10., 6.), .95)
        calls = []
        def train_only(mask):
            calls.append(tuple(mask))
            return 10 + d.predicted_effect(mask, beta1, 1)
        best, rows = d.oracle_search(n, k, train_only, 10., max_candidates=6)
        self.assertEqual(len(calls), 6)
        self.assertEqual(len(rows), 6)
        self.assertEqual(tuple(best), tuple(chosen))
        with self.assertRaises(ValueError):
            d.oracle_search(n, k, train_only, 10., max_candidates=5)
        self.assertEqual(len(calls), 6)  # cap fails before any extra evaluation

    def test_bootstrap_and_interaction_terms(self):
        scores = np.array([5., 6., 7., 8.])
        full = scores + 1.
        empty = scores - 2.
        lo, hi = d.bootstrap_faithfulness(scores, full, empty, 50, 42)
        self.assertAlmostEqual(lo, 2/3)
        self.assertAlmostEqual(hi, 2/3)
        terms = d.interaction_terms([1, 0, 1], [1, 2, 3, .4, .5, .6])
        self.assertEqual(terms, {(0, 2): .5})


if __name__ == '__main__':
    unittest.main()
