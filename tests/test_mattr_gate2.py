"""CPU-only subset/manifest tests; no model or training is launched."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from src.mattr_optimization.gate2 import subset_indices, NestedData


class Rows(list):
    _fingerprint = 'unit-only-full'
    def select(self, indices):
        result = Rows(super(Rows, self).__getitem__(i) for i in indices)
        result._fingerprint = json.dumps(indices)
        return result
    def __getitem__(self, key):
        if isinstance(key, str): return [r[key] for r in self]
        return super().__getitem__(key)


class Dataset:
    def __init__(self, split):
        self.dataset = Rows({'prompt': f'{split}-{i}', 'metadata':
            {'indirect_object': 'A', 'subject': 'B', 'random_c': 'C'}} for i in range(1000))
    def __len__(self): return len(self.dataset)
    def __getitem__(self, index): return self.dataset[index]['prompt'], 'donor', [1,2]
    def tokenizer(self, text, **kw): return SimpleNamespace(input_ids=[{' A':1, ' B':2, ' C':3}[text]])


class Gate2Tests(unittest.TestCase):
    def test_nested_subsets_for_every_planned_seed(self):
        for seed in (42,43,44):
            largest = subset_indices(1000,128,seed)
            for n in (16,32,64,128):
                self.assertEqual(subset_indices(1000,n,seed), largest[:n])
                self.assertEqual(len(set(largest[:n])),n)

    def test_validation_cohort_fixed_across_pools_and_seeds(self):
        expected = None
        with tempfile.TemporaryDirectory() as folder:
            for seed,n in ((42,16),(42,128),(43,32),(44,64)):
                out=Path(folder)/f'{seed}_{n}';out.mkdir()
                data=NestedData(out,n,seed,128)
                data.select(Dataset('train'),'train')
                validation=Dataset('validation');data.select(validation,'validation')
                actual=list(validation.dataset['prompt'])
                if expected is None: expected=actual
                self.assertEqual(actual,expected)
                self.assertFalse(set(actual)&set(data.datasets['train'].dataset['prompt']))

    def test_changed_data_is_rejected_without_overwriting_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);data=NestedData(out,16,42,128)
            data.select(Dataset('train'),'train')
            path=out/'train_manifest.csv';before=path.read_bytes()
            changed=Dataset('train')
            for row in changed.dataset:row['prompt']='changed-'+row['prompt']
            with self.assertRaises(AssertionError):data.select(changed,'train')
            self.assertEqual(path.read_bytes(),before)


if __name__=='__main__':unittest.main()
