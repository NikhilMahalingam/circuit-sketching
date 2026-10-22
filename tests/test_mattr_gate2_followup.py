"""CPU-only tests of seed isolation; no real model or synthetic benchmark results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.mattr_optimization import gate2, gate2_controls


class SeedIsolationTests(unittest.TestCase):
    def check_seeds(self, training_seed, subset_seed):
        with tempfile.TemporaryDirectory() as folder:
            calls=[]
            def fake_run(args):
                dataset=gate2.NestedData(args.out,args.pool,args.seed,128)
                calls.append((args.seed,dataset.seed,dataset.validation_size,args.pool))
                saved=json.loads(args.protocol.read_text())
                self.assertEqual(saved['training_seed'],training_seed)
                self.assertEqual(saved['subset_seed'],subset_seed)
            argv=['controls','--out',folder,'--training-seed',str(training_seed),'--subset-seed',str(subset_seed)]
            with patch('sys.argv',argv),patch.object(gate2,'protocol_for',return_value={}),patch.object(gate2,'run_job',fake_run):
                gate2_controls.main()
            self.assertEqual(calls,[(training_seed,subset_seed,128,128)])

    def test_only_training_seed_varies(self):
        self.check_seeds(43,42)

    def test_only_subset_seed_varies(self):
        self.check_seeds(42,44)


if __name__=='__main__':
    unittest.main()
