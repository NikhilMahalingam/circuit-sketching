"""CPU tests of the new sampler, checkpoint RNG and frozen decision guards."""
import random
import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import configure
configure()
from matryoshka_attribution.schedules import sample_k
from matryoshka_attribution import learn_scores
from src.mattr_optimization.checkpoint import checkpointed
from src.mattr_optimization.gate3_sampling import CPRMixture,distribution
from src.mattr_optimization.gate3 import verify_criteria
from src.mattr_optimization.gate3_report import passes


class Gate3Tests(unittest.TestCase):
    def test_uniform_exact_rng_consumption_and_values(self):
        torch.manual_seed(42)
        expected=[sample_k(32491,'uniform') for _ in range(100)]
        expected_state=torch.get_rng_state()
        torch.manual_seed(42)
        sampler=CPRMixture(32491,1.)
        self.assertEqual([sampler.sample() for _ in range(100)],expected)
        self.assertTrue(torch.equal(torch.get_rng_state(),expected_state))

    def test_grid_weights_rounding_and_endpoint(self):
        rows=distribution()
        self.assertEqual([r['rounded_k'] for r in rows],[32,64,162,324,649,1624,3249,6498,16245])
        self.assertAlmostEqual(sum(r['quadrature_weight'] for r in rows),.749)
        self.assertAlmostEqual(sum(r['conditional_grid_probability'] for r in rows),1.)
        self.assertTrue(all(1<=r['jitter_low']<r['jitter_high']<32491 for r in rows))

    def test_mixture_distribution_and_python_rng_untouched(self):
        torch.set_num_threads(4)
        for lam in (.5,.25):
            torch.manual_seed(44);random.seed(44);before=random.getstate()
            sampler=CPRMixture(32491,lam)
            values=[sampler.transform(u) for u in torch.rand(50000).tolist()]
            self.assertEqual(random.getstate(),before)
            uniform=sum(branch=='uniform' for _,branch,_ in values)
            self.assertAlmostEqual(uniform/len(values),lam,delta=.01)
            grid=[j for _,branch,j in values if branch=='grid']
            hist=np.bincount(grid,minlength=9)/len(grid)
            np.testing.assert_allclose(hist,sampler.weights,atol=.012,rtol=0)
            self.assertTrue(all(1<=k<32491 for k,_,_ in values))
            # One RNG draw per step even when selecting grid points/jitter.
            torch.manual_seed(45)
            for _ in range(100):sampler.sample()
            state=torch.get_rng_state()
            torch.manual_seed(45)
            for _ in range(100):torch.rand(1)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_resume_matches_uninterrupted_official_trainer(self):
        torch.set_num_threads(4)
        total=32491
        x=torch.linspace(-1,1,total)
        def loss(mask):
            scale=1+random.randint(0,127)/128
            return (mask*x).mean()*scale
        kwargs=dict(variant='topk',k_schedule='uniform',lr=.05,n_iters=8,device='cpu')
        torch.manual_seed(42);random.seed(42)
        full=learn_scores(total,loss,steps=12,k_sampler=CPRMixture(total,.25),**kwargs)
        with tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(42);random.seed(42)
            checkpointed(learn_scores,total,loss,folder=Path(folder),identity='fixture',interval=4,
                         steps=4,k_sampler=CPRMixture(total,.25),**kwargs)
            torch.manual_seed(99);random.seed(99)
            resumed=checkpointed(learn_scores,total,loss,folder=Path(folder),identity='fixture',interval=4,
                                 steps=12,k_sampler=CPRMixture(total,.25),**kwargs)
        self.assertTrue(torch.equal(full.scores,resumed.scores))
        self.assertEqual(full.k_log,resumed.k_log)
        self.assertEqual(full.loss_log,resumed.loss_log)

    def test_frozen_thresholds_fail_bad_compact_accuracy(self):
        criteria=verify_criteria()
        data={'relative_cpr_gain':[.03]*3,'cpr_delta':[.3]*3,'cmd_delta':[.3]*3,
              'training_seconds':[100]*3,'baseline_training_seconds':[100]*3,
              'peak_allocated_mib':[100]*3,'baseline_peak_allocated_mib':[100]*3,'accuracy_auc_delta':[0]*3}
        for f in (.005,.01,.02,.05):data[f'accuracy_delta_{f}']=[0]*3
        rows=pd.DataFrame(data)
        self.assertTrue(passes(rows,criteria)['passed'])
        rows['accuracy_delta_0.005']=-.021
        self.assertFalse(passes(rows,criteria)['passed'])
        rows['accuracy_delta_0.005']=0;rows.training_seconds=111
        self.assertFalse(passes(rows,criteria)['passed'])


if __name__=='__main__':unittest.main()
