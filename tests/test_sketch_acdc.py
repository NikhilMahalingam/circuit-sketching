import numpy as np
import unittest
import torch

from src.sketch_acdc.core import hessian_vector_product, nystrom_sketch, state_key, exact_acdc
from src.sketch_acdc.research import calibrate_bound, hybrid_acdc
from unittest.mock import patch
import pandas as pd


class SketchACDCTests(unittest.TestCase):
 def test_hvp_matches_known_symmetric_quadratic(self):
    h = torch.tensor([[2.0, -0.7, 0.2], [-0.7, -1.0, 0.3], [0.2, 0.3, 4.0]], dtype=torch.float64)
    f = lambda z: 0.5 * z @ h @ z + (z**3).sum()
    z = torch.tensor([0.2, -0.4, 0.1], dtype=torch.float64)
    v = torch.tensor([1.1, -0.3, 0.8], dtype=torch.float64)
    expected = (h + torch.diag(6*z)) @ v
    torch.testing.assert_close(hessian_vector_product(f, z, v), expected, atol=1e-12, rtol=1e-12)
    eps = 1e-5
    grad = lambda x: torch.autograd.grad(f(x.requires_grad_(True)), x)[0]
    finite = (grad((z + eps*v).clone()) - grad((z - eps*v).clone())) / (2*eps)
    torch.testing.assert_close(hessian_vector_product(f, z, v), finite, atol=1e-9, rtol=1e-9)


 def test_full_rank_sketch_recovers_hessian(self):
    h = torch.tensor([[3.0, 0.4], [0.4, 1.0]], dtype=torch.float64)
    f = lambda z: 0.5*z @ h @ z
    approx, _, _ = nystrom_sketch(f, torch.zeros(2, dtype=torch.float64), 2, seed=17)
    torch.testing.assert_close(approx, h, atol=1e-12, rtol=1e-12)


 def test_cache_identity_and_binary_validation(self):
    assert state_key([1, 0], 'configuration-a', 'train') != state_key([1, 0], 'configuration-b', 'train')
    assert state_key([1, 0], 'configuration-a', 'train') != state_key([1, 0], 'configuration-a', 'validation')
    with self.assertRaises(ValueError):
        state_key([0, 0.5], 'a', 'train')


 def test_restricted_acdc_is_sequential_and_restorable(self):
    class Edge:
        def __init__(self, name): self.name = name
    class Toy:
        edges = [Edge('a'), Edge('b')]
        def __init__(self): self.calls = []
        def evaluate_state(self, state, split):
            self.calls.append(tuple(state))
            z = np.asarray(state)
            return np.array([2 - 0.1*z[0] - 2*z[1] - 0.2*z[0]*z[1]])
    toy = Toy()
    state, rows = exact_acdc(toy, 'train', threshold=0.5)
    assert state == [1, 0]
    assert [row['pruned'] for row in rows] == [True, False]
    assert toy.calls == [(0, 0), (1, 0), (1, 1)]

 def test_calibration_uses_withheld_training_effects(self):
    frame = pd.DataFrame({'second_order_prediction':[0.0,0.0],
                          'true_calibration_effect':[0.1,0.3],
                          'true_validation_effect':[100.0,100.0]})
    self.assertAlmostEqual(calibrate_bound(frame,quantile=1.0),0.3)

 def test_hybrid_verifies_uncertain_decision(self):
    class Toy:
        edges=[type('Edge',(),{'name':'a'})()]
        model=type('Model',(),{'cfg':type('Cfg',(),{'device':'cpu','dtype':torch.float32})()})()
        def exact_candidate_effect(self,state,index,split):
            self.checked=(tuple(state),index,split)
            return np.array([0.15])
    toy=Toy()
    estimator={'gradient':np.array([0.09]),'curvature':np.array([0.0])}
    with patch('src.sketch_acdc.research.local_estimator',return_value=estimator):
        state,rows,queries=hybrid_acdc(toy,threshold=0.1,rank=0,train_limit=2,
                                       bound=0.05,refresh_after=None)
    self.assertEqual(state,[0])
    self.assertEqual(queries,1)
    self.assertEqual(toy.checked,((0,),0,'train'))


if __name__ == '__main__':
    unittest.main()
