"""Graph invariants, sparse spectra, forward/gradient and official optimizer tests."""
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from src.mattr_optimization.common import configure
configure()
from matryoshka_attribution import learn_scores,sigmoid_topk
from src.mattr_optimization.checkpoint import checkpointed
from src.mattr_expander.graphs import ring,generate,audit,adjacency,spectrum,permute
from src.mattr_expander.coupling import Coupling
from src.mattr_expander.analysis import guards
import pandas as pd


class ExpanderTests(unittest.TestCase):
    def test_graphs_degree_reproducibility_and_rng_isolation(self):
        torch.manual_seed(42);state=torch.get_rng_state();random.seed(42);python_state=random.getstate()
        for family in ('G0','G1','G2'):
            a,_=generate(family,257);b,_=generate(family,257)
            np.testing.assert_array_equal(a,b);self.assertEqual(audit(a)['edges'],1028)
            self.assertEqual(audit(permute(a))['edges'],1028)
        self.assertTrue(torch.equal(state,torch.get_rng_state()));self.assertEqual(python_state,random.getstate())

    def test_sparse_spectrum_against_known_small_graph(self):
        for n,d in ((21,2),(21,8),(9,8)):
            a=ring(n,d);dense=adjacency(a).toarray()/d # small reference only
            vals=np.linalg.eigvalsh(dense)
            exact=spectrum(a,'G0');numeric=spectrum(a,'G1')
            for result in (exact,numeric):
                self.assertAlmostEqual(result['laplacian_gap'],1-vals[-2],places=6)
                self.assertAlmostEqual(result['nontrivial_adjacency_magnitude'],max(abs(vals[0]),abs(vals[-2])),places=6)

    def test_values_gradients_and_finite_difference(self):
        a=ring(21);idx=torch.tensor(a,dtype=torch.long);p=torch.tensor(adjacency(a).toarray()/8,dtype=torch.double)
        x=torch.linspace(-1,1,21,dtype=torch.double,requires_grad=True);v=torch.cos(x.detach())
        mix=Coupling(idx,.25);out=mix(x);expected=.75*x+.25*p@x
        torch.testing.assert_close(out,expected)
        actual=torch.autograd.grad((out*v).sum(),x)[0]
        torch.testing.assert_close(actual,.75*v+.25*p.T@v)
        self.assertTrue(torch.autograd.gradcheck(mix,(x,),eps=1e-6,atol=1e-5))

    def test_beta_zero_and_checkpoint_resume_official_trainer(self):
        torch.set_num_threads(4);n=257;idx=torch.tensor(ring(n),dtype=torch.long)
        def loss(mask):return (mask*torch.linspace(-1,1,n)).mean()*(1+random.randint(0,127)/128)
        kw=dict(variant='topk',k_schedule='uniform',T=.5,n_iters=30,lr=.05,device='cpu')
        torch.manual_seed(42);random.seed(42);original=learn_scores(n,loss,steps=12,**kw)
        torch.manual_seed(42);random.seed(42)
        with Coupling(idx,0).install():zero=learn_scores(n,loss,steps=12,**kw)
        self.assertTrue(torch.equal(original.scores,zero.scores));self.assertEqual(original.train_log,zero.train_log)
        torch.manual_seed(42);random.seed(42)
        with Coupling(idx,.25).install():full=learn_scores(n,loss,steps=12,**kw)
        with tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(42);random.seed(42)
            with Coupling(idx,.25).install():checkpointed(learn_scores,n,loss,steps=4,folder=Path(folder),identity='fixed_graph',interval=4,**kw)
            torch.manual_seed(99);random.seed(99)
            with Coupling(idx,.25).install():resumed=checkpointed(learn_scores,n,loss,steps=12,folder=Path(folder),identity='fixed_graph',interval=4,**kw)
        self.assertTrue(torch.equal(full.scores,resumed.scores));self.assertEqual(full.train_log,resumed.train_log)

    def test_effective_export_and_soft_mass(self):
        n=257;idx=torch.tensor(ring(n),dtype=torch.long);s=torch.linspace(-1,1,n,requires_grad=True)
        for beta in (0,.1,.25):
            effective=Coupling(idx,beta)(s)
            for k in (1.,12.,128.,256.):
                mask=sigmoid_topk(effective,k,T=.5,n_iters=30)
                self.assertAlmostEqual(float(mask.detach().sum()),k,delta=.001)
        self.assertFalse(torch.equal(Coupling(idx,.25)(s),s))
        # Production-size official grid cutoffs, not only a small fixture.
        n=32491;idx=torch.tensor(ring(n),dtype=torch.long);s=torch.linspace(-1,1,n,requires_grad=True)
        for beta in (0,.1,.25):
            for fraction in (.001,.005,.01,.5,.99):
                k=float(int(n*fraction));effective=Coupling(idx,beta)(s)
                mask=sigmoid_topk(effective,k,T=.5,n_iters=30)
                self.assertAlmostEqual(float(mask.detach().sum()),k,delta=max(.01,k*1e-5))

    def test_registered_quality_guards(self):
        p={'criteria':{'primary_CPR_gain':.02,'positive_seeds':2,'accuracy_auc_drop':.02,'compact_drop':.02,
            'adaptation_time_ratio_max':1.15,'allocated_memory_ratio_max':1.1,
            'efficiency':{'CPR_loss_max':.01,'CMD_increase_max':.1,'compute_saving_min':.2}}}
        rows=pd.DataFrame({'relative_cpr_gain':[.03]*3,'cpr_delta':[.3]*3,'cmd_delta':[0.]*3,
            'adaptation_seconds':[100.]*3,'baseline_adaptation_seconds':[100.]*3,'memory_ratio':[1.]*3,
            'accuracy_auc_delta':[0.]*3,**{f'accuracy_delta_{f}':[0.]*3 for f in (.005,.01,.02,.05)}})
        self.assertTrue(guards(rows,p)['passed'])
        rows['accuracy_delta_0.005']=-.021;self.assertFalse(guards(rows,p)['passed'])
        rows['accuracy_delta_0.005']=0.;rows['adaptation_seconds']=79.
        self.assertTrue(guards(rows,p,efficiency=True)['passed'])
        rows['relative_cpr_gain']=-.011;self.assertFalse(guards(rows,p,efficiency=True)['passed'])


if __name__=='__main__':unittest.main()
