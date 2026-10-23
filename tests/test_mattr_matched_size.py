import tempfile
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
from src.mattr_matched_size.topology import capped_selection,upper_selection,canonical_ranks,ensemble_scores,score_order,closure_order
from src.mattr_matched_size.statistics import bootstrap_indices,primary_analysis
from src.mattr_matched_size.evaluation import ExactCache
from src.mattr_matched_size.analysis import interpolate_values
from src.mattr_matched_size.design import select_cohort


class MatchedTests(unittest.TestCase):
    def test_cap_uses_largest_plateau_prefix_and_labels_gaps(self):
        counts=np.array([0,0,2,2,5,5,7])
        self.assertEqual(capped_selection(counts,4),(3,2))
        self.assertEqual(upper_selection(counts,4),(4,5))
        self.assertEqual(capped_selection(counts,5),(5,5))
        self.assertEqual(upper_selection(counts,5),(4,5))
        with self.assertRaises(ValueError):capped_selection(counts,8)

    def test_canonical_signed_ties_and_equal_weight_ensembles(self):
        np.testing.assert_array_equal(canonical_ranks([-5,2,2,-1]),[4,1,2,3])
        vectors=[np.array([1,2,3]),np.array([3,2,1]),np.array([2,2,2])]
        scores,mean=ensemble_scores(vectors)
        again,_=ensemble_scores(vectors[::-1])
        np.testing.assert_array_equal(scores.numpy(),again.numpy())
        self.assertEqual(len(np.unique(scores.numpy())),3)
        self.assertTrue((scores.numpy()<0).all())
        with self.assertRaises(AssertionError):score_order([0,0,1])

    def test_closure_group_ties_and_unseen_tail(self):
        first=np.array([5,2,5,11,11,3])
        original=np.array([4,2,1,0,5,3])
        order,_=closure_order(first,original,10)
        np.testing.assert_array_equal(order,[1,5,2,0,4,3])
        np.testing.assert_array_equal(closure_order(first,original,10)[0],order)

    def test_interpolation_is_fractional_not_a_measured_binary_circuit(self):
        values,w=interpolate_values([0,1],[1,1],100,120,116)
        np.testing.assert_allclose(values,[.8,1]);self.assertEqual(w,.8)
        values,w=interpolate_values([0,1],[0,1],116,116,116)
        np.testing.assert_array_equal(values,[0,1]);self.assertEqual(w,0)

    def test_exact_cache_does_not_count_or_recompute_hits(self):
        calls=[]
        def compute():calls.append(1);return np.array([1.,-1.]),np.array([1.,0.])
        with tempfile.TemporaryDirectory() as tmp:
            cache=ExactCache(tmp,{'cohort':'a','model':'x'})
            a=cache.get('mask1',compute);a[0][0]=999
            b=cache.get('mask1',compute)
            self.assertEqual(len(calls),1);self.assertEqual(cache.hits,1);self.assertEqual(b[0][0],1.)
            ExactCache(tmp,{'cohort':'b','model':'x'}).get('mask1',compute)
            self.assertEqual(len(calls),2)

    def test_fixed_seed_ratio_and_classification_thresholds(self):
        idx=bootstrap_indices(['a']*64,5000,5643)
        nominal=np.ones((3,64))*.4
        result=primary_analysis(np.ones((3,64))*.3,nominal,idx)
        self.assertEqual(result['classification'],'Ranking-quality effect')
        self.assertAlmostEqual(result['advantage_retention_ratio']['point'],.75)
        zero=primary_analysis(np.zeros((3,64)),nominal,idx)
        self.assertEqual(zero['classification'],'Mostly connectivity')
        undefined=primary_analysis(np.zeros((3,64)),np.zeros((3,64)),idx)
        self.assertEqual(undefined['classification'],'Non-replication')
        self.assertFalse(undefined['advantage_retention_ratio']['stable'])
        one_seed=np.zeros((3,64));one_seed[0]=.9
        self.assertFalse(primary_analysis(one_seed,nominal,idx)['primary_success'])

    def test_clean_and_donor_exclusions_and_stratified_deduplication(self):
        rows=[{'official_filtered_index':i,'clean_sha256':'c'+str(i),'corrupted_sha256':'d'+str(i),
               'template_id':str(i%9)} for i in range(900)]
        rows[2]['corrupted_sha256']='c1' # cross-role duplicate, later row must be excluded
        selected,_,_=select_cohort(pd.DataFrame(rows),{'c5','d6'},{7},seed=5543)
        self.assertEqual(len(selected),512)
        self.assertFalse(set(selected.official_filtered_index)&{2,5,6,7})
        self.assertEqual(len(set(selected.clean_sha256)|set(selected.corrupted_sha256)),1024)


if __name__=='__main__':unittest.main()
