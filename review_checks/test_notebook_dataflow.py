"""Offline checks of the actual notebook code; no GPT-2 results are produced.

Run with: py -3.11 -m unittest discover -s review_checks -p test_notebook_dataflow.py -v
Only the standard library and NumPy are required. TransformerLens/CUDA execution
is deliberately not simulated by these dataflow checks.
"""
import ast
import itertools
import json
from pathlib import Path
import re
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = json.loads((ROOT / "notebooks" / "mib_sketching_baseline.ipynb").read_text(encoding="utf-8"))
CODE_CELLS = [cell for cell in NOTEBOOK["cells"] if cell["cell_type"] == "code"]
PARSED = [ast.parse("".join(cell["source"]), filename=cell["id"]) for cell in CODE_CELLS]
NODES = [node for module in PARSED for node in module.body]
FUNCTIONS = {node.name: node for node in NODES if isinstance(node, ast.FunctionDef)}


def make_namespace():
    names = {"sample_unique_masks", "mask_key", "second_order_features"}
    definitions = [FUNCTIONS[name] for name in names]
    namespace = {"np": np, "n": 8, "pair_indices": list(itertools.combinations(range(8), 2))}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "<notebook helpers>", "exec"), namespace)
    return namespace


class NotebookDataflowTests(unittest.TestCase):
    def setUp(self):
        self.ns = make_namespace()

    def test_every_code_cell_compiles_and_schema_has_unique_ids(self):
        self.assertEqual(NOTEBOOK["nbformat"], 4)
        self.assertEqual(NOTEBOOK["nbformat_minor"], 5)
        ids = [cell["id"] for cell in NOTEBOOK["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in CODE_CELLS:
            with self.subTest(cell=cell["id"]):
                compile("".join(cell["source"]), cell["id"], "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_all_23_requested_sections_exist_in_order(self):
        matches = [re.match(r"## (\d+)\. ", "".join(cell["source"]))
                   for cell in NOTEBOOK["cells"] if cell["cell_type"] == "markdown"]
        self.assertEqual([int(match[1]) for match in matches if match], list(range(1, 24)))

    def test_train_test_are_unique_disjoint_deterministic(self):
        sample = self.ns["sample_unique_masks"]
        train = sample(64, 8, 0.3, 42, include_zero=True)
        test = sample(32, 8, 0.3, 43, excluded=train)
        repeated = sample(64, 8, 0.3, 42, include_zero=True)
        np.testing.assert_array_equal(train, repeated)
        train_keys, test_keys = set(map(tuple, train)), set(map(tuple, test))
        self.assertEqual(len(train_keys), 64)
        self.assertEqual(len(test_keys), 32)
        self.assertFalse(train_keys & test_keys)
        self.assertIn((0,) * 8, train_keys)
        self.assertNotIn((0,) * 8, test_keys)
        self.assertTrue(np.isin(train, [0, 1]).all())
        self.assertTrue(np.isin(test, [0, 1]).all())

    def test_impossible_sampling_request_fails_before_drawing(self):
        with self.assertRaises(ValueError):
            self.ns["sample_unique_masks"](9, 3, 0.3, 42)
        with self.assertRaises(ValueError):
            self.ns["sample_unique_masks"](8, 3, 0.3, 42, excluded=[(0, 0, 0)])
        with self.assertRaises(ValueError):
            self.ns["sample_unique_masks"](2, 3, 0.0, 42)

    def test_bad_binary_mask_is_not_silently_cast(self):
        key = self.ns["mask_key"]
        self.assertEqual(key(np.zeros(8)), (0,) * 8)
        self.assertEqual(key(np.ones(8, dtype=bool)), (1,) * 8)
        for mask in ([0] * 7, [0] * 7 + [0.5], [0] * 7 + [2], [0] * 7 + [np.nan], [[0] * 8]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                key(mask)

    def test_exactly_one_feature_for_each_unordered_pair(self):
        pairs = self.ns["pair_indices"]
        self.assertEqual(len(pairs), 28)
        self.assertEqual(len(set(pairs)), 28)
        self.assertTrue(all(i < j for i, j in pairs))
        all_masks = np.array(list(itertools.product([0, 1], repeat=8)), dtype=float)
        features = self.ns["second_order_features"](all_masks)
        self.assertEqual(features.shape, (256, 36))
        np.testing.assert_array_equal(features[:, :8], all_masks)
        for column, (i, j) in enumerate(pairs, start=8):
            expected = np.logical_and(all_masks[:, i], all_masks[:, j]).astype(float)
            np.testing.assert_array_equal(features[:, column], expected)

    def test_training_designs_have_full_column_rank(self):
        train = self.ns["sample_unique_masks"](64, 8, 0.3, 42, include_zero=True)
        quadratic = self.ns["second_order_features"](train)
        self.assertEqual(np.linalg.matrix_rank(train), 8)
        self.assertEqual(np.linalg.matrix_rank(quadratic), 36)
        self.assertLess(np.linalg.cond(quadratic), 100)

    def test_accuracy_and_ranking_calls_use_heldout_labels(self):
        # Inspect all calls outside function definitions. The only internal metrics
        # forwarding occurs in ranking_metrics, whose callers must also use y_test.
        calls = []
        for node in NODES:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                continue
            calls.extend(call for call in ast.walk(node) if isinstance(call, ast.Call)
                         and isinstance(call.func, ast.Name)
                         and call.func.id in {"regression_metrics", "ranking_metrics"})
        self.assertEqual(len(calls), 6)
        for call in calls:
            self.assertIsInstance(call.args[0], ast.Name)
            self.assertEqual(call.args[0].id, "y_test")
        internal = [node for node in ast.walk(FUNCTIONS["ranking_metrics"])
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "regression_metrics"]
        self.assertEqual(len(internal), 1)
        self.assertEqual(ast.unparse(internal[0].args[0]), "truth")

    def test_regressions_fit_only_training_data_without_intercepts(self):
        constructors, fits = [], []
        for module in PARSED:
            for node in ast.walk(module):
                if not isinstance(node, ast.Call):
                    continue
                if isinstance(node.func, ast.Name) and node.func.id == "LinearRegression":
                    constructors.append(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "fit":
                    fits.append(node)
        self.assertEqual(len(constructors), 2)
        for call in constructors:
            keywords = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
            self.assertIs(keywords["fit_intercept"], False)
        self.assertEqual({tuple(ast.unparse(arg) for arg in call.args) for call in fits},
                         {("X_train", "y_train"), ("X2_train", "y_train")})

    def test_summary_separates_actual_calls_from_standalone_budgets(self):
        summary = next(node for node in NODES if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == "experiment_summary"
                               for target in node.targets))
        rows = summary.value.args[0].elts
        self.assertEqual(len(rows), 3)
        for row in rows:
            columns = {key.value: value for key, value in zip(row.keys, row.values) if key is not None}
            actual = columns["model_evaluations"]
            self.assertIsInstance(actual, ast.Call)
            self.assertEqual(actual.func.id, "stage_evaluations")
            self.assertIn("standalone_acquisition_evaluations", columns)
            self.assertEqual(ast.literal_eval(columns["evaluation_split"]), "held-out test masks")

    def test_sketch_budget_selection_does_not_use_test_performance(self):
        selection = next(node for node in NODES if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "chosen_k"
                                 for target in node.targets))
        self.assertEqual(ast.unparse(selection.value), "max(sketch_budgets)")

    def test_every_model_forward_routes_through_counter(self):
        direct = []
        for node in NODES:
            if isinstance(node, ast.FunctionDef) and node.name == "run_model":
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                if isinstance(call.func, ast.Name) and call.func.id == "model":
                    direct.append(call)
                if (isinstance(call.func, ast.Attribute) and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == "model"
                        and call.func.attr in {"forward", "run_with_hooks", "run_with_cache"}):
                    direct.append(call)
        self.assertEqual(direct, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
