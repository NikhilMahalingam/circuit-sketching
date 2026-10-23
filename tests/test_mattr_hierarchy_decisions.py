"""Guardrails for the already-frozen decisions; no parameter tuning."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
from src.mattr_optimization.common import configure
configure()
from src.mattr_hierarchy.analysis import gates,select_c,FRACTIONS

def row():
    return {'relative_cpr_gain':.025,'cpr_delta':.25,'cmd_delta':.25,'accuracy_auc_delta':0.,
            'accuracy_auc':.8,'cpr':10.25,'time_ratio':1.02,'memory_ratio':1.01,
            **{f'accuracy_delta_{f}':0. for f in FRACTIONS},**{f'accuracy_{f}':.8 for f in FRACTIONS}}

class DecisionTests(unittest.TestCase):
    def test_CPR_alone_cannot_override_compact_regression(self):
        rows=pd.DataFrame([row() for _ in range(3)])
        self.assertTrue(gates(rows)['passed'])
        rows['accuracy_delta_0.005']=-.03
        self.assertFalse(gates(rows)['passed'])
    def test_compute_guard_and_positive_seed_requirement(self):
        rows=pd.DataFrame([row() for _ in range(3)]);rows['time_ratio']=1.11
        self.assertFalse(gates(rows)['passed'])
        rows['time_ratio']=1.;rows['cpr_delta']=[1.,-.01,-.01]
        self.assertFalse(gates(rows)['passed'])
    def test_control_equivalence_prevents_structural_claim_and_confirmation(self):
        with tempfile.TemporaryDirectory() as directory,patch('src.mattr_hierarchy.analysis.compare',return_value=row()):
            self.assertFalse(select_c(Path(directory),{'family':'H1','gamma':.001}))

if __name__=='__main__':unittest.main()
