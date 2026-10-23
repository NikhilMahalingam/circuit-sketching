import unittest
import numpy as np
import pandas as pd
from src.mattr_compact_validation.design import text_hash, select_rows, group_quotas
from src.mattr_compact_validation.statistics import bootstrap_indices, interval, auc_prompt_values, compare_arrays, decision


class CompactTests(unittest.TestCase):
    def test_official_signed_rankings_nested_and_pruned(self):
        from src.mattr_optimization.common import configure
        configure()
        from eap.graph import Graph
        from src.mattr_compact_validation.inference import circuit_audit
        import torch
        graph = Graph.from_model(dict(n_layers=12, n_heads=12, parallel_attn_mlp=False, d_model=768))
        scores = torch.linspace(-2, 3, 32491)
        records = circuit_audit(graph, scores, [.005, .01, .02, .05, 1.])
        self.assertTrue(all(r['actual_edges'] <= r['requested_edges'] for r in records))
        self.assertEqual(records[-1]['actual_edges'], 32491)
        # Sign is material: this test calls only the official signed sorter.
        self.assertTrue(all(r['nested_before_pruning'] for r in records))

    def test_exclusions_deduplicate_strings_not_just_ids(self):
        rows = [{'official_filtered_index': i, 'clean_sha256': text_hash(str(i // 2)),
                 'template_id': str(i % 3)} for i in range(1200)]
        frame = pd.DataFrame(rows)
        selected, _, _ = select_rows(frame, {text_hash('1')}, {4}, target=512)
        self.assertEqual(len(selected), 512)
        self.assertTrue(selected.clean_sha256.is_unique)
        self.assertNotIn(text_hash('1'), selected.clean_sha256.tolist())
        self.assertNotIn(4, selected.official_filtered_index.tolist())
        pd.testing.assert_frame_equal(selected, select_rows(frame, {text_hash('1')}, {4}, target=512)[0])

    def test_inadequate_data_never_fabricated(self):
        frame = pd.DataFrame([{'official_filtered_index': i, 'clean_sha256': str(i), 'template_id': 'x'} for i in range(255)])
        with self.assertRaises(ValueError):
            select_rows(frame, set(), set())
        self.assertEqual(sum(group_quotas({'a': 100, 'b': 400}, 256).values()), 256)

    def test_stratified_paired_bootstrap_shared_across_seeds(self):
        families = np.array(['a'] * 10 + ['b'] * 20)
        indices = bootstrap_indices(families)
        np.testing.assert_array_equal(indices, bootstrap_indices(families))
        self.assertEqual(indices.shape, (5000, 30))
        self.assertTrue(np.all((indices < 10).sum(1) == 10))
        a = np.tile(np.arange(30) % 2, (3, 10, 1)).astype(float)
        b = np.zeros_like(a)
        grid = [.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.]
        result, auc = compare_arrays(a, b, families, grid, indices, 'baseline')
        pooled = result[result.seed == 'equal_seed_mean'].iloc[0]
        marginal = interval(a[0, 0], indices, 2)
        self.assertEqual(pooled.lower, marginal[1])  # repeating identical seeds cannot shrink prompt CI
        self.assertAlmostEqual(auc.delta.iloc[-1], .5)

    def test_auc_weights_and_simultaneous_interval(self):
        grid = [.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.]
        a = np.ones((3, 10, 12))
        np.testing.assert_allclose(auc_prompt_values(a, grid), 1.)
        rng = np.random.default_rng(5)
        values = rng.normal(size=40)
        idx = bootstrap_indices(['a'] * 40)
        ordinary = interval(values, idx, 1)
        simultaneous = interval(values, idx, 2)
        self.assertLessEqual(simultaneous[1], ordinary[1])
        self.assertGreaterEqual(simultaneous[2], ordinary[2])

    def test_gamma_zero_equivalence_blocks_structural_claim(self):
        f = ['a'] * 32
        idx = bootstrap_indices(f)
        a = np.ones((3, 10, 32)); zero = np.zeros_like(a)
        grid = [.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.]
        tables = []; aucs = []
        for name, b in [('baseline', zero), ('gamma_zero', a), ('shuffled', zero), ('edge_l2', zero)]:
            t, u = compare_arrays(a, b, f, grid, idx, name); tables.append(t); aucs.append(u)
        dec = decision(pd.concat(tables), pd.concat(aucs))
        self.assertTrue(dec['primary']['passed'])
        self.assertFalse(dec['structural_advantage'])
        self.assertFalse(dec['further_training_authorized'])


if __name__ == '__main__':
    unittest.main()
