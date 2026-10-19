"""Real small TransformerLens models; synthetic tokens are UNIT TESTS only."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from mib_ioi.runtime import configure_repository
configure_repository(Path(__file__).resolve().parents[1])
from eap.graph import Graph
from transformer_lens import HookedTransformer, HookedTransformerConfig
from src.sketch_acdc.core import receiver_patch
from src.acdc_prefix_cache.engine import Batch, ExactExecutor, ExecutionMeter, matrix_patch
from src.acdc_prefix_cache.graph import Dependencies
from src.acdc_prefix_cache.search import exact_acdc


class PrefixCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)
        cfg=HookedTransformerConfig(n_layers=4,n_heads=2,d_model=16,d_head=8,
            d_mlp=32,n_ctx=12,d_vocab=31,act_fn='gelu',normalization_type='LN',
            device='cpu',dtype=torch.float32,use_attn_result=True,
            use_split_qkv_input=True,use_hook_mlp_in=True)
        self.model=HookedTransformer(cfg).eval().requires_grad_(False)
        self.graph=Graph.from_model(self.model)
        self.batch=Batch(torch.randint(0,31,(2,7)),torch.randint(0,31,(2,7)),
            torch.ones((2,7),dtype=torch.long),torch.tensor([7,7]),[[1,2],[3,4]])
        self.standard=ExactExecutor(self.model,self.graph,[self.batch]).prepare()
        self.cached=ExactExecutor(self.model,self.graph,[self.batch],cached=True).prepare()

    def test_all_layers_and_output_receiver(self):
        names=['input->a0.h0<q>','a0.h0->m0','m0->a1.h1<k>',
               'a1.h1->m2','m2->a3.h0<v>','a3.h1->m3','input->logits']
        for name in names:
            np.testing.assert_array_equal(self.standard.evaluate({name}),self.cached.evaluate({name}))

    def test_random_multi_edge_states(self):
        rng=np.random.default_rng(42);order=list(self.cached.dependencies.order)
        for count in (2,8,32,64):
            for repeat in range(2):
                removed=set(rng.choice(order,count,replace=False))
                np.testing.assert_array_equal(self.standard.evaluate(removed),self.cached.evaluate(removed))

    def test_dependent_current_senders(self):
        state={'input->a0.h0<v>','a0.h0->a1.h0<v>','a1.h0->m2'}
        np.testing.assert_array_equal(self.standard.evaluate(state),self.cached.evaluate(state))
        # A late candidate cannot justify skipping a previously changed layer.
        plan=self.cached.dependencies.plan(state|{'m2->a3.h0<q>'})
        self.assertEqual(plan.start_layer,0)
        with self.assertRaisesRegex(ValueError,'Unsafe'):
            self.cached.evaluate(state,requested_start=3)

    def test_actual_blocks_are_skipped_and_evaluations_not_memoized(self):
        meter=ExecutionMeter(self.model)
        with meter.instrument():
            self.cached.evaluate({'m2->a3.h0<q>'})
            self.cached.evaluate({'m2->a3.h0<q>'})
        self.assertEqual(meter.block_counts,[0,0,0,2])
        self.assertEqual(len(meter.events),2)
        self.assertTrue(all(not e['grad_enabled'] for e in meter.events))

    def test_cached_boundaries_are_not_modified(self):
        before={i:t.clone() for i,t in self.cached.prepared[0]['boundaries'].items()}
        for state in ({'m2->a3.h0<q>'},{'a1.h0->m2'},{'input->logits'}):
            self.cached.evaluate(state)
        for layer,value in before.items():
            torch.testing.assert_close(value,self.cached.prepared[0]['boundaries'][layer],atol=0,rtol=0)
        np.testing.assert_array_equal(self.standard.baseline,self.cached.evaluate(set()))

    def test_model_mutation_invalidates_cache(self):
        with torch.no_grad(): next(self.model.parameters()).add_(.01)
        with self.assertRaisesRegex(ValueError,'changed'): self.cached.evaluate(set())

    def test_padding_mask_and_unequal_lengths(self):
        mask=torch.tensor([[1,1,1,1,1,1,1],[1,1,1,1,0,0,0]])
        batch=Batch(self.batch.tokens,self.batch.donors,mask,torch.tensor([7,4]),self.batch.labels)
        a=ExactExecutor(self.model,self.graph,[batch]).prepare()
        b=ExactExecutor(self.model,self.graph,[batch],cached=True).prepare()
        np.testing.assert_array_equal(a.evaluate({'m2->a3.h0<v>'}),b.evaluate({'m2->a3.h0<v>'}))

    def test_matrix_patching_matches_corrected_shared_receiver(self):
        donor=torch.randn(2,5,3,16);current=torch.randn_like(donor)
        value=torch.randn(2,5,2,16);weights=torch.tensor([[1.,0.],[1.,1.],[0.,1.]])
        groups=[(0,[(0,0),(1,1)]),(1,[(1,1),(2,2)])]
        expected=receiver_patch(value,groups,{i:donor[:,:,i] for i in range(3)},
            {i:current[:,:,i] for i in range(3)},torch.ones(3))
        torch.testing.assert_close(matrix_patch(value,donor-current,weights),expected,atol=1e-6,rtol=1e-6)

    def test_sequential_decisions_and_resume_are_identical(self):
        deps=self.cached.dependencies
        names=[n for n in deps.order if self.graph.edges[n].child.name!='logits'][::5][:32]
        expected=exact_acdc(self.standard,names,.05)
        actual=exact_acdc(self.cached,names,.05)
        self.assertEqual(expected['trace'],actual['trace'])
        self.assertEqual(expected['retained'],actual['retained'])
        with tempfile.TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'checkpoint.json'
            exact_acdc(self.cached,names,.05,checkpoint=checkpoint,stop_after=7)
            resumed=exact_acdc(self.cached,names,.05,checkpoint=checkpoint)
            self.assertEqual(resumed['trace'],actual['trace'])
            with self.assertRaisesRegex(ValueError,'differs'):
                exact_acdc(self.cached,names,.07,checkpoint=checkpoint)

    def test_dependency_proof_for_nonreverse_states(self):
        deps=Dependencies(self.graph)
        plan=deps.plan({'input->a1.h0<q>','a3.h0->logits'})
        self.assertEqual(plan.start_layer,1)
        self.assertFalse(plan.affected_nodes & plan.reusable_nodes)
        self.assertIn('a3.h0',plan.affected_nodes)
        with self.assertRaises(ValueError): deps.plan({'invented'})


if __name__=='__main__': unittest.main()
