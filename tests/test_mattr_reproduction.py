"""Run with the isolated WSL environment and the inspected official MAttr source."""
from pathlib import Path
import random
import tempfile
import unittest
import sys
import json

if sys.platform != 'linux':
    raise unittest.SkipTest('MAttr reproduction tests use the isolated WSL Ubuntu environment.')

import numpy as np
import pandas as pd
import torch

from src.mattr_reproduction.runtime import FixedData
from src.mattr_reproduction.checkpoint import learn_checkpointed
from matryoshka_attribution.trainer import learn_scores


def seed(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def loss(mask):
    # Both RNG streams matter: official k uses torch; data sampling uses Python.
    index = random.randint(0, mask.numel()-1)
    target = torch.arange(mask.numel(), dtype=mask.dtype)/mask.numel()
    return ((mask-target)**2).sum() + .1*mask[index]


KWARGS = dict(steps=50, variant='topk', k_schedule='uniform', optimizer='adam',
              lr=.05, T=.5, n_iters=30, adam_eps=1e-8, device='cpu')


class MAttrReproductionTests(unittest.TestCase):
    def test_verifier_checks_the_actual_test_artifacts(self):
        from src.mattr_reproduction.verify import verify_job
        from src.mattr_reproduction.runtime import PERCENTAGES
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'experiment_config.json').write_text(json.dumps({'method':'fixture'}))
            (out/'complete.json').write_text('{}')
            pd.DataFrame([{'stage':'validation_evaluation','forward_batches':1,
                           'example_forwards':2}]).to_csv(out/'compute_accounting.csv',index=False)
            (out/'forward_events.jsonl').write_text(json.dumps({'stage':'validation_evaluation',
                'success':True,'batch_size':2,'grad_enabled':False})+'\n')
            pd.DataFrame({'component':[f'node{i}' for i in range(157)],
                          'forward_index':range(157),'score':np.ones(157)}).to_csv(out/'node_ranking.csv',index=False)
            for split in ('train','validation','test'):
                name='official_evaluation.json' if split=='validation' else f'official_{split}_evaluation.json'
                (out/name).write_text(json.dumps({'examples':2,'intervention':'patching','absolute_ranking':False}))
                pd.DataFrame({'prompt_sha256':[split]}).to_csv(out/f'{split}_manifest.csv',index=False)
                counts=[int(x*157) for x in PERCENTAGES]
                pd.DataFrame({'retained_fraction':PERCENTAGES,'requested_nodes':counts,
                    'retained_nodes':counts,'official_normalized_logit_diff_cpr':[float(n>0) for n in counts]}
                    ).to_csv(out/f'{split}_curve.csv',index=False)
                raw={'baseline':np.array([2.,2.]),'corrupted':np.array([-1.,-1.]),
                     **{f'p{i}':np.full(2,2. if n else -1.) for i,n in enumerate(counts)}}
                np.savez(out/f'{split}_per_example.npz',**raw)
            self.assertTrue(verify_job(out)['passed'])
            raw['p9']=np.zeros(2)
            np.savez(out/'test_per_example.npz',**raw)
            with self.assertRaises(AssertionError):verify_job(out)

    def test_public_confirmation_requires_every_validation_stage(self):
        from src.mattr_reproduction.confirm import lock_selection
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError,'all 75 main-stage jobs'):
                lock_selection(Path(folder))
            self.assertFalse((Path(folder)/'confirmation_selection.json').exists())

    def test_fresh_wrapper_preserves_official_updates_exactly(self):
        seed(42); expected = learn_scores(12, loss, **KWARGS)
        with tempfile.TemporaryDirectory() as folder:
            seed(42)
            actual = learn_checkpointed(learn_scores, 12, loss, checkpoint_dir=folder,
                                        identity='test', interval=10, **KWARGS)
        torch.testing.assert_close(actual.scores, expected.scores, atol=0, rtol=0)
        self.assertEqual(actual.train_log, expected.train_log)

    def test_resumed_wrapper_preserves_adam_and_both_rng_streams(self):
        seed(42); expected = learn_scores(12, loss, **KWARGS)
        class Interrupt(Exception):
            pass
        def interrupt(path, state):
            if state['completed_steps']==20:
                raise Interrupt()
        with tempfile.TemporaryDirectory() as folder:
            seed(42)
            with self.assertRaises(Interrupt):
                learn_checkpointed(learn_scores, 12, loss, checkpoint_dir=folder,
                    identity='test', interval=10, on_checkpoint=interrupt, **KWARGS)
            seed(999)
            actual = learn_checkpointed(learn_scores, 12, loss, checkpoint_dir=folder,
                                        identity='test', interval=10, **KWARGS)
            torch.testing.assert_close(actual.scores, expected.scores, atol=0, rtol=0)
            self.assertEqual(actual.train_log, expected.train_log)
            calls=[]
            cached = learn_checkpointed(lambda *a, **k: calls.append(True), 12, loss,
                checkpoint_dir=folder, identity='test', interval=10, **KWARGS)
            self.assertFalse(calls)
            torch.testing.assert_close(cached.scores, actual.scores, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                learn_checkpointed(learn_scores, 12, loss, checkpoint_dir=folder,
                    identity='different', interval=10, **KWARGS)

    def test_subsets_are_nested_and_do_not_perturb_global_rngs(self):
        seed(42); before=np.random.get_state(); py=random.getstate(); ts=torch.get_rng_state()
        for s in [42,43,44]:
            largest=FixedData.indices(1000,128,s)
            for n in [16,32,64,128]:
                selected=FixedData.indices(1000,n,s)
                self.assertEqual(selected,largest[:n]);self.assertEqual(len(set(selected)),n)
        np.testing.assert_equal(np.random.get_state(),before)
        self.assertEqual(random.getstate(),py);torch.testing.assert_close(torch.get_rng_state(),ts)
        self.assertNotEqual(FixedData.indices(1000,128,42),FixedData.indices(1000,128,43))


if __name__=='__main__':
    unittest.main()
