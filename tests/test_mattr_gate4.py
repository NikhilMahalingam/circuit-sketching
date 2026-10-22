"""Official CPU trainer tests; GPU attribution is separately measured, never mocked."""
import tempfile
from pathlib import Path
import unittest
import random
import torch
from src.mattr_optimization.common import configure
configure()
from matryoshka_attribution import learn_scores
from eap.graph import Graph
from src.mattr_optimization.gate4_init import rank_normalize,initialized_trainer,edge_mapping,mask_diagnostics
from src.mattr_optimization.checkpoint import checkpointed
from src.mattr_optimization.gate4_report import gates
import pandas as pd


class InitializationTests(unittest.TestCase):
    def test_signed_midranks_and_ties(self):
        self.assertTrue(torch.equal(rank_normalize(torch.tensor([-2.,0.,0.,4.])),torch.tensor([-1.,0.,0.,1.])))
        self.assertTrue(torch.equal(rank_normalize(torch.zeros(4)),torch.zeros(4)))
        with self.assertRaises(ValueError):rank_normalize(torch.tensor([0.,float('nan')]))

    def test_all_edges_and_mismatch(self):
        cfg={'n_layers':12,'n_heads':12,'parallel_attn_mlp':False,'d_model':768}
        a,b=Graph.from_model(cfg),Graph.from_model(cfg)
        frame=edge_mapping(a,b)
        self.assertEqual(len(frame),32491);self.assertEqual(frame.mattr_index.tolist(),list(range(32491)))
        self.assertEqual(set(frame.qkv),{'','q','k','v'})
        next(iter(b.edges.values())).matrix_index=(0,1)
        with self.assertRaises(ValueError):edge_mapping(a,b)

    def test_init_zero_identity_and_exact_resume(self):
        torch.set_num_threads(4)
        n=64;initial=torch.linspace(-.5,.5,n)
        def loss(mask):return (mask*torch.linspace(-1,1,n)).mean()*(1+random.randint(0,127)/128)
        kw=dict(variant='topk',k_schedule='uniform',T=.5,n_iters=30,lr=.05,device='cpu')
        random.seed(42);torch.manual_seed(42)
        zero=learn_scores(n,loss,steps=12,**kw)
        random.seed(42);torch.manual_seed(42)
        with initialized_trainer(torch.zeros(n)):check=learn_scores(n,loss,steps=12,**kw)
        self.assertTrue(torch.equal(zero.scores,check.scores));self.assertEqual(zero.train_log,check.train_log)
        random.seed(42);torch.manual_seed(42)
        with initialized_trainer(initial):full=learn_scores(n,loss,steps=12,**kw)
        with tempfile.TemporaryDirectory() as folder:
            random.seed(42);torch.manual_seed(42)
            with initialized_trainer(initial):checkpointed(learn_scores,n,loss,folder=Path(folder),identity='fixture',interval=4,steps=4,**kw)
            random.seed(99);torch.manual_seed(99)
            with initialized_trainer(initial):resumed=checkpointed(learn_scores,n,loss,folder=Path(folder),identity='fixture',interval=4,steps=12,**kw)
        self.assertTrue(torch.equal(full.scores,resumed.scores));self.assertEqual(full.train_log,resumed.train_log)

    def test_mask_mass_and_all_gradient_entries(self):
        table=mask_diagnostics(torch.linspace(-1,1,32491))
        self.assertEqual(len(table),28);self.assertTrue(table.finite_gradient.all())

    def test_primary_and_efficiency_guards(self):
        p={'success':{'primary':{'mean_paired_relative_CPR_gain_min':.02,'positive_seeds_min':2,
          'total_adaptation_time_ratio_max':1.1,'peak_allocated_ratio_max':1.1,
          'mean_compact_accuracy_drop_max':.02,'mean_accuracy_log_auc_drop_max':.02},
          'efficiency':{'CPR_relative_loss_max':.01,'CMD_increase_max':.1,
          'total_adaptation_compute_reduction_min':.2,'accuracy_drop_max':.02,'accuracy_auc_drop_max':.02}}}
        rows=pd.DataFrame({'relative_cpr_gain':[.03]*3,'cpr_delta':[.3]*3,'cmd_delta':[0]*3,
          'adaptation_seconds':[100]*3,'baseline_adaptation_seconds':[100]*3,'memory_ratio':[1.]*3,
          'accuracy_auc_delta':[0.]*3,**{f'accuracy_delta_{f}':[0.]*3 for f in (.005,.01,.02,.05)}})
        self.assertTrue(gates(rows,p)['passed'])
        rows['adaptation_seconds']=111;self.assertFalse(gates(rows,p)['passed'])
        rows['adaptation_seconds']=79;self.assertTrue(gates(rows,p,True)['passed'])
        rows['accuracy_delta_0.005']=-.021;self.assertFalse(gates(rows,p,True)['passed'])


if __name__=='__main__':unittest.main()
