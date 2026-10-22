import unittest
import numpy as np
import pandas as pd
from src.mattr_optimization.common import configure
configure()
from src.mattr_interactions.design import toggled,digest,assign,cleanup,category,protocol
from src.mattr_interactions.statistics import bootstrap_means,select_confirmation
from src.mattr_interactions.evaluation import pair_effects

class DiagnosticsTests(unittest.TestCase):
    def test_exact_toggles_and_digest(self):
        a=np.array([True,False,True]);b=toggled(a,[0,1])
        np.testing.assert_array_equal(a,[True,False,True]);np.testing.assert_array_equal(b,[False,True,True])
        self.assertNotEqual(digest(a),digest(b))
    def test_interaction_algebra(self):
        z=np.array([0.,0.]);one=np.array([1.,2.]);two=np.array([3.,4.])
        values=[(z,z),(one,z),(two,z),(one+two,z)]
        np.testing.assert_allclose(pair_effects(values)['interaction'],0)
        values[-1]=(one+two+.7,z);np.testing.assert_allclose(pair_effects(values)['interaction'],.7)
    def test_prompt_bootstrap(self):
        x=np.array([[1.,2.,3.,4.],[2.,4.,6.,8.]])
        ci=bootstrap_means(x,repetitions=1000)
        np.testing.assert_allclose(ci[1],2*ci[0]);np.testing.assert_array_equal(ci,bootstrap_means(x,repetitions=1000))
    def test_confirmation_bound_and_ties(self):
        rows=[]
        for f in (.01,.02):
            for d in ('addition','deletion'):
                rows.extend({'fraction':f,'direction':d,'pair_id':f'{f}_{d}_{i}','abs_interaction':i,'joint':i} for i in range(10))
        ids=select_confirmation(pd.DataFrame(rows));self.assertEqual(len(ids),16);self.assertEqual(len(set(ids)),16)
    def test_official_cleanup_detects_collateral(self):
        from eap.graph import Graph
        graph=Graph.from_model(dict(n_layers=2,n_heads=2,parallel_attn_mlp=False,d_model=8))
        mask=graph.real_edge_mask.bool();rows=mask.nonzero().tolist();positions={tuple(x):i for i,x in enumerate(rows)}
        base=np.zeros(len(rows),dtype=bool)
        for name in ('input->a0.h0<q>','a0.h0->logits'):
            edge=graph.edges[name];index=tuple(int(v)%dim for v,dim in zip(edge.matrix_index,mask.shape));base[positions[index]]=True
        np.testing.assert_array_equal(cleanup(graph,base),base)
        broken=base.copy();broken[np.flatnonzero(base)[0]]=False
        self.assertFalse(np.array_equal(cleanup(graph,broken),broken))
    def test_no_algorithm_or_training_in_protocol(self):
        p=protocol();self.assertTrue(p['no_retraining']);self.assertEqual(p['budget']['training_updates'],0)
    def test_donor_cache_recomputes_live_clean_sources(self):
        import torch
        from unittest.mock import patch
        import eap.evaluate as official
        from src.mattr_interactions.evaluation import DonorCache
        class Model:
            def __init__(self):self.matrix=None;self.calls=0;self.clean_value=2.
            def forward(self,tokens,**kwargs):
                self.calls+=1
                if tokens.item()==1:self.matrix.add_(5.)
                else:self.matrix.sub_(self.clean_value)
                return self.matrix.clone()
        model=Model();cache=DonorCache(model)
        def builder(*args,**kwargs):
            model.matrix=torch.zeros(1,1,1,1)
            return ([],[],[]),model.matrix
        with patch.object(official,'make_hooks_and_matrices',builder),cache.install():
            official.make_hooks_and_matrices();model.forward(torch.tensor([[1]]))
            first=model.forward(torch.tensor([[2]]));model.clean_value=3.
            official.make_hooks_and_matrices();model.forward(torch.tensor([[1]]))
            second=model.forward(torch.tensor([[2]]))
        self.assertEqual(first.item(),3.);self.assertEqual(second.item(),2.)
        self.assertEqual(cache.hits,1);self.assertEqual(cache.misses,1);self.assertEqual(model.calls,3)

if __name__=='__main__':unittest.main()
