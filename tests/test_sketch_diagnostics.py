import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import torch
from src.sketch_acdc.core import nystrom_sketch, EdgeExperiment, FrozenSenderEdgeExperiment, EdgeSpec

from src.sketch_acdc.diagnostics import (receiver_patch,full_hessian,taylor_effect,
    threshold_grid,load_states,analyze,decision_metrics,streaming_curvature_bundle)


class DiagnosticsTests(unittest.TestCase):
    def test_streamed_curvature_equals_weighted_full_function(self):
        import time
        class Data:
            def __len__(self):return 3
            def to_dataloader(self,batch):return [([1.,2.],None,None),([3.],None,None)]
        class Ledger:
            def __init__(self):self.stages=[];self.backward_calls=0;self.hvp_calls=0
            def sync(self):pass
            @contextmanager
            def stage(self,name,estimate=None):
                start=time.perf_counter();b=self.backward_calls;h=self.hvp_calls
                yield
                self.stages.append({'wall_seconds':time.perf_counter()-start,'forward_passes_batches':0,
                    'examples_processed':0,'backward_calls':self.backward_calls-b,'hvp_operations':self.hvp_calls-h,
                    'peak_cuda_allocated_mib':0})
        h=torch.tensor([[2.,.3],[.3,-1.]],dtype=torch.float64)
        class Experiment:
            model=SimpleNamespace(cfg=SimpleNamespace(device='cpu',dtype=torch.float64))
            batch_size=2;splits={'train':Data()}
            def _batch_score(self,clean,corrupt,labels,z):
                return torch.tensor(clean,dtype=z.dtype)*(.5*z@h@z+z.sum())
        with tempfile.TemporaryDirectory() as folder:
            result,costs=streaming_curvature_bundle(Experiment(),[0,1],.5,[2],42,Ledger(),folder)
        np.testing.assert_allclose(result['full_second_order'],2*h.numpy(),atol=1e-12)
        np.testing.assert_allclose(result['sketch_rank_2'],2*h.numpy(),atol=1e-12)
        np.testing.assert_allclose(result['gradient'],2*(h.numpy()@np.array([.5,1.])+1),atol=1e-12)

    def test_core_simultaneous_dependent_edges_and_mixed_derivative(self):
        # Tiny exact DAG: input -> double -> output. Two removed connections
        # must output the corrupted donor 2, never the historical value -2.
        class Model:
            cfg=SimpleNamespace(device='cpu',dtype=torch.float64)
            tokenizer=None
            def __init__(self):self.active=[]
            @contextmanager
            def hooks(self,fwd_hooks=()):
                old=self.active;self.active=old+list(fwd_hooks)
                try:yield
                finally:self.active=old
            def hook(self,name,value):
                for target,fn in self.active:
                    if target==name:
                        update=fn(value,SimpleNamespace(name=name))
                        if update is not None:value=update
                return value
            def __call__(self,tokens,**kwargs):
                x=self.hook('embed',tokens[:,:,None])
                x=self.hook('m0in',x)
                x=self.hook('m0out',2*x)
                return self.hook('output',x)
        metric_module=ModuleType('MIB_circuit_track.metrics')
        metric_module.get_metric=lambda *args:lambda logits,*a,**kw:logits[:,0,0]
        util_module=ModuleType('eap.utils')
        util_module.tokenize_plus=lambda model,rows:(torch.tensor([float(x) for x in rows],dtype=torch.float64).reshape(-1,1),None,None,1)
        def capture_hooks(model,graph,batch,n_pos,scores):
            donor=torch.zeros(batch,1,2,1,dtype=torch.float64)
            def cap(index,sign=1):
                def save(value,hook):donor[:,:,index,:]+=sign*value.detach()
                return save
            return ([('embed',cap(0)),('m0out',cap(1))],
                    [('embed',cap(0,-1)),('m0out',cap(1,-1))],[]),donor
        util_module.make_hooks_and_matrices=capture_hooks
        edges=[EdgeSpec('input->m0',0,0,'m0in',None,0,1),EdgeSpec('m0->logits',1,1,'output',None,0,1)]
        graph=SimpleNamespace(real_edge_mask=torch.ones(2,2,dtype=torch.bool),edges={})
        for spec,out in zip(edges,['embed','m0out']):
            graph.edges[spec.name]=SimpleNamespace(matrix_index=(spec.sender_index,spec.receiver_index),
                hook=spec.hook,parent=SimpleNamespace(out_hook=out,index=slice(None)))
        with patch.dict('sys.modules',{'MIB_circuit_track.metrics':metric_module,'eap.utils':util_module}):
            model=Model();experiment=EdgeExperiment(model,graph,edges,{},1,'test')
            historical=FrozenSenderEdgeExperiment(model,graph,edges,{},1,'test')
            self.assertNotEqual(experiment.cache_id,historical.cache_id)
            for a in [0.,1.]:
                for b in [0.,1.]:
                    z=torch.tensor([a,b],dtype=torch.float64)
                    self.assertAlmostEqual(float(experiment._batch_score(['3'],['1'],None,z)),(1-b)*(6-4*a)+2*b)
            self.assertAlmostEqual(float(historical._batch_score(['3'],['1'],None,torch.ones(2))),-2.)
            z=torch.tensor([.3,.4],dtype=torch.float64,requires_grad=True)
            score=experiment._batch_score(['3'],['1'],None,z).sum()
            gradient=torch.autograd.grad(score,z,create_graph=True)[0]
            mixed=torch.autograd.grad(gradient[0],z)[0][1]
            self.assertAlmostEqual(float(mixed),4.)

    def test_rank16_sketch_recovers_stable_indefinite_hessian(self):
        h=torch.diag(torch.cat([torch.linspace(-3,-1,8),torch.linspace(1,3,8)]).double())
        function=lambda z:.5*z@h@z
        approximation,_,_=nystrom_sketch(function,torch.zeros(16,dtype=torch.float64),16,seed=42)
        torch.testing.assert_close(approximation,h,atol=1e-10,rtol=1e-10)

    def test_full_hessian_and_shifted_taylor_difference(self):
        h=torch.tensor([[2.,.7],[.7,-1.]],dtype=torch.float64)
        b=torch.tensor([.3,-.2],dtype=torch.float64)
        function=lambda z:.5*z@h@z+b@z
        anchor=torch.tensor([.5,.5],dtype=torch.float64)
        matrix,asymmetry=full_hessian(function,anchor)
        torch.testing.assert_close(matrix,h)
        self.assertLess(asymmetry,1e-12)
        variable=anchor.clone().requires_grad_(True)
        gradient=torch.autograd.grad(function(variable),variable)[0].numpy()
        state=np.array([0.,1.])
        prediction=taylor_effect(gradient,matrix.numpy(),anchor.numpy(),state,0)
        trial=torch.tensor([1.,1.],dtype=torch.float64)
        actual=float(function(trial)-function(torch.tensor(state)))
        self.assertAlmostEqual(prediction,actual,places=12)

    def test_receiver_patch_preserves_other_heads_and_sender_gradients(self):
        value=torch.randn(1,2,3,4,dtype=torch.float64)
        current={5:torch.randn(1,2,4,dtype=torch.float64,requires_grad=True)}
        donor={5:torch.randn(1,2,4,dtype=torch.float64)}
        z=torch.tensor([.4],dtype=torch.float64,requires_grad=True)
        result=receiver_patch(value,[(1,[(0,5)])],donor,current,z)
        torch.testing.assert_close(result[:,:,[0,2]],value[:,:,[0,2]])
        torch.testing.assert_close(result[:,:,1],value[:,:,1]+z[0]*(donor[5]-current[5]))
        g=torch.autograd.grad(result.sum(),current[5])[0]
        torch.testing.assert_close(g,torch.full_like(g,-.4))
        torch.testing.assert_close(value[:,:,[0,2]],result[:,:,[0,2]])

    def test_midpoint_candidate_only_quadratic_term_cancels(self):
        anchor=np.array([.5,1.])
        state=np.array([0.,1.])
        gradient=np.array([2.,4.])
        h=np.array([[3.,.5],[.5,-1.]])
        self.assertAlmostEqual(taylor_effect(gradient,h,anchor,state,0),2.)

    def test_threshold_grid_bounds_and_training_adaptive_points(self):
        thresholds=threshold_grid([.123,-.9,.001])
        self.assertTrue(np.all((thresholds>=.01)&(thresholds<=.30)))
        self.assertIn(.123,thresholds)
        self.assertTrue(any(abs(t-.123)<.002 and t<.123 for t in thresholds))

    def test_trajectory_state_leading_zeros(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'trajectory.csv'
            pd.DataFrame({'state_before':['0000','0010','0011']}).to_csv(path,index=False)
            self.assertEqual(load_states(path,4,removed_targets=(1,2)),[[0,0,1,0],[0,0,1,1]])

    def test_common_difficult_subset_and_error_rates(self):
        rows=[]
        for method,pred in [('first_order',[.05,.2]),('full_second_order',[.15,.05])]:
            for i in range(2):
                rows.append({'method':method,'anchor_strength':0.,'query_id':str(i),
                             'prediction':pred[i],'train_effect':[.15,.05][i],
                             'validation_effect':[.15,.05][i]})
        metrics,sweep,difficult=analyze(pd.DataFrame(rows),[.1])
        first=sweep[(sweep.method=='first_order')&(sweep.split=='validation')].iloc[0]
        self.assertEqual(first.decision_accuracy,0)
        self.assertEqual(first.false_prune_rate,.5)
        self.assertEqual(first.false_retain_rate,.5)
        self.assertEqual(difficult.groupby(['method','split']).size().tolist(),[2,2,2,2])
        self.assertTrue(difficult[difficult.method=='full_second_order'].correct_decision.all())


if __name__=='__main__':
    unittest.main()
