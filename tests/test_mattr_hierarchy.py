import random
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from src.mattr_optimization.common import configure
configure()
from src.mattr_hierarchy.adapter import Hierarchy,group_sum
from src.mattr_hierarchy.checkpoint import checkpointed,rng_state,restore_rng

class HierarchyTests(unittest.TestCase):
    def test_centering_singleton_and_full_score_space(self):
        h=Hierarchy([0,0,1,2,2],dtype=torch.double)
        x=torch.arange(8,dtype=torch.double,requires_grad=True)
        eff,r=h.parts(x)
        torch.testing.assert_close(group_sum(r,h.padded,h.valid),torch.zeros(3,dtype=torch.double))
        self.assertEqual(r[2],0)
        jac=torch.autograd.functional.jacobian(h.transform,x)
        self.assertEqual(torch.linalg.matrix_rank(jac),5)
    def test_finite_difference_and_cpu_reference(self):
        h=Hierarchy([0,0,1,2,2],dtype=torch.double)
        x=torch.linspace(-.5,.6,8,dtype=torch.double,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda a:h.parts(a),x,eps=1e-6,atol=1e-6))
        eff,r=h.parts(x);ref=x[3:].clone()
        for g in range(3):ref[h.ids==g]-=x[3:][h.ids==g].mean()
        torch.testing.assert_close(eff,x[:3][h.ids]+ref)
    def test_penalty_only_centered_residuals(self):
        h=Hierarchy([0,0,1],gamma=.01,dtype=torch.double)
        x=torch.tensor([3.,4.,2.,5.,6.],dtype=torch.double,requires_grad=True)
        h.raw=x;h.effective,h.centered=h.parts(x)
        gradient=torch.autograd.grad(h.penalty(),x)[0]
        torch.testing.assert_close(gradient,h.penalty_gradient())
        torch.testing.assert_close(gradient[:2],torch.zeros(2,dtype=torch.double))
        self.assertEqual(gradient[-1],0)
    def test_mapping_and_fixed_shuffle_sizes(self):
        from eap.graph import Graph
        from src.mattr_hierarchy.groups import construct
        graph=Graph.from_model(dict(n_layers=12,n_heads=12,parallel_attn_mlp=False,d_model=768))
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);result=construct(graph,out)
            for name in ('H1','H2'):
                a=np.load(out/f'{name}_groups.npy');b=np.load(out/f'{name}_shuffled_groups.npy')
                self.assertEqual(len(a),32491);np.testing.assert_array_equal(np.bincount(a),np.bincount(b))
            self.assertEqual(result['hierarchies']['H1']['groups'],157)
            self.assertEqual(result['hierarchies']['H2']['groups'],445)
    def test_dedicated_bypass_and_exact_resume(self):
        import matryoshka_attribution.trainer as trainer
        torch.manual_seed(42);random.seed(42);np.random.seed(42)
        initial=rng_state()
        def loss(mask):return ((mask-torch.linspace(0,1,6))**2).mean()+mask[random.randint(0,5)]*.02
        options=dict(steps=12,variant='topk',k_schedule='uniform',n_iters=30,T=.5,lr=.05,optimizer='adam',device='cpu')
        original=trainer.learn_scores(6,loss,**options);expected=rng_state();restore_rng(initial)
        with Hierarchy(kind='bypass').install():bypass=trainer.learn_scores(6,loss,**options)
        self.assertTrue(torch.equal(original.scores,bypass.scores));self.assertEqual(original.train_log,bypass.train_log)
        self.assertTrue(torch.equal(expected['torch'],rng_state()['torch']))
        h=Hierarchy([0,0,1,1,2,2],gamma=.01)
        def regular(mask):return loss(mask)+h.penalty()
        with tempfile.TemporaryDirectory() as d:
            restore_rng(initial)
            with h.install():full=checkpointed(trainer.learn_scores,6,regular,folder=Path(d)/'full',identity='groups-fixed',parameter_count=9,interval=6,**options)
            restore_rng(initial)
            with h.install():
                for steps in (6,12):
                    resumed=checkpointed(trainer.learn_scores,6,regular,folder=Path(d)/'resume',identity='groups-fixed',parameter_count=9,interval=6,**{**options,'steps':steps})
            self.assertTrue(torch.equal(full.scores,resumed.scores));self.assertEqual(full.train_log,resumed.train_log)
            with h.install(),self.assertRaises(ValueError):
                checkpointed(trainer.learn_scores,6,regular,folder=Path(d)/'resume',identity='different-group-hash',parameter_count=9,**options)
    def test_official_mask_mass_and_order(self):
        from matryoshka_attribution.masks import build_mask
        h=Hierarchy([0,0,1,1,2,2]);raw=torch.linspace(-.1,.2,9,requires_grad=True)
        score=h.transform(raw)
        for k in (1.,2.5,5.):
            mask=build_mask(score,k,'topk',T=.5,n_iters=30).mask
            self.assertAlmostEqual(float(mask.detach().sum()),k,places=5)
            self.assertTrue(bool((torch.diff(mask[score.argsort()])>=0).all()))

if __name__=='__main__':unittest.main()
