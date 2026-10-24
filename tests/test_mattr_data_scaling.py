"""Synthetic CPU tests only; no registered IOI run or reported performance."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import configure,sha,write_json
configure()
from src.mattr_data_scaling.core import matrix,marker,verified,runtime_guard
from src.mattr_data_scaling.statistics import contrasts,classify
from src.mattr_data_scaling.structure import permutation_maps,overlap
from src.mattr_matched_size.topology import graph_new
from src.mattr_optimization.gate4_init import edge_mapping

class ScalingTests(unittest.TestCase):
    def test_exact_output_cache_requires_byte_hash(self):
        from src.mattr_data_scaling.evaluate import HashedCache
        with tempfile.TemporaryDirectory() as tmp:
            cache=HashedCache(tmp,{'identity':'synthetic'});calls=[]
            def compute():calls.append(1);return np.arange(4,dtype=np.float32),np.ones(4,dtype=np.float32)
            cache.get('mask',compute);cache.get('mask',compute)
            self.assertEqual(len(calls),1)
            checksum=next(Path(tmp).glob('*.npz.sha256'));checksum.unlink()
            cache.get('mask',compute)
            self.assertEqual(len(calls),2);self.assertTrue(list((Path(tmp)/'incomplete').glob('*')))
            p=next(Path(tmp).glob('*.npz'));p.write_bytes(p.read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):cache.get('mask',compute)

    def test_registered_matrix_has28_endpoints_and_no_extra_seed_factorial(self):
        jobs=matrix();self.assertEqual(len(jobs),28)
        self.assertEqual(len({tuple(sorted(j.items())) for j in jobs}),28)
        self.assertTrue(all(j['dataset_size']==10000 and j['step']==5000 for j in jobs if j['seed']>44))

    def test_hash_verified_cache_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'x').write_text('a');marker(p,{'source':'frozen'},[p/'x'])
            self.assertTrue(verified(p,{'source':'frozen'}))
            with self.assertRaises(ValueError):verified(p,{'source':'changed'})
            (p/'x').write_text('b')
            with self.assertRaises(ValueError):verified(p)

    def test_primary_seeds45_46_never_change_factorial(self):
        def arrays(extra):
            a={}
            for n in (128,10000):
                for s in (1000,5000):
                    for r in range(42,47):
                        for cap in (116,296,400,640):
                            a['baseline',n,s,r,cap]=np.zeros(16)
                            a['h2',n,s,r,cap]=np.ones(16)*(extra if r>44 else .2)
            return a
        a,fa,_=contrasts(arrays(.4),['a']*8+['b']*8,replicates=50)
        b,fb,_=contrasts(arrays(.8),['a']*8+['b']*8,replicates=50)
        self.assertEqual(fa,fb);self.assertNotEqual(a['primary']['mean'],b['primary']['mean'])
        self.assertEqual(a['classification'],'Robust compact advantage')

    def test_correctness_and_larger_caps_block_positive_claim(self):
        primary=np.ones((5,10))*.2;small=np.ones((3,2,10))*.3
        big=dict(factorial=np.ones((3,2,10))*.2,primary_lower=.1,primary_upper=.3)
        base=np.ones((3,2,10))*.5
        self.assertEqual(classify(primary,small,big,base,base,{296:np.ones(10)*-.1},True)[0],'Data-limited advantage')
        self.assertEqual(classify(primary,small,big,base,base,{},False)[0],'Invalid correctness')

    def test_zero_crossing_is_not_equivalence(self):
        primary=np.ones((5,10))*.01
        small=np.stack([np.stack([np.ones(10)*.4,np.ones(10)*.2])]*3)
        big=dict(factorial=np.stack([np.stack([np.ones(10)*.3,np.ones(10)*.1])]*3),primary_lower=-.1,primary_upper=.1)
        base=np.stack([np.stack([np.ones(10)*.4,np.ones(10)*.7])]*3)
        label,checks=classify(primary,small,big,base,base,{},True)
        self.assertFalse(checks['catches_up']);self.assertEqual(label,'Data-limited advantage')
        big.update(primary_lower=-.01,primary_upper=.015)
        label,checks=classify(primary,small,big,base,base,{},True)
        self.assertTrue(checks['catches_up']);self.assertEqual(label,'Baseline catches up and data-limited')

    def test_topology_control_preserves_official_connectivity_and_size(self):
        graph=graph_new();mapping=edge_mapping(graph,graph)
        gen=torch.Generator().manual_seed(6);graph.scores[graph.real_edge_mask.bool()]=torch.rand(32491,generator=gen)
        graph.apply_topn(200,absolute=False,prune=True);mask=graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()
        for perm in permutation_maps(mapping.fillna(''),replicates=3):
            other=np.zeros(32491,dtype=bool);other[perm[np.flatnonzero(mask)]]=True
            graph.in_graph[:]=False;graph.in_graph[graph.real_edge_mask.bool()]=torch.from_numpy(other);graph.prune()
            np.testing.assert_array_equal(other,graph.in_graph[graph.real_edge_mask.bool()].numpy())
            self.assertEqual(mask.sum(),other.sum())

    def test_runtime_gate_requires_explicit_approval_after_first_full_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);write_json(out/'preflight_gpu.json',dict(training_seconds_per_update=1,evaluation_seconds_per_ranking_projection=1e6,eap_seconds_per_example=1))
            p=dict(training_jobs=[],endpoints=matrix(),eap_jobs=[])
            with patch('src.mattr_data_scaling.core.protocol',return_value=p):
                runtime_guard(out)  # first calibration run permitted
                write_json(out/'attempts/calibration/rates.json',dict(updates=5000,completed_target=5000,training_seconds=100))
                with self.assertRaises(RuntimeError):runtime_guard(out)
                self.assertTrue(runtime_guard(out,True)['approval'])

    def test_group_centering_gradient_and_all_edge_directions(self):
        from src.mattr_hierarchy.adapter import Hierarchy
        ad=Hierarchy([0,0,1,1,2]);raw=torch.arange(8,dtype=torch.float64,requires_grad=True)
        ad=Hierarchy([0,0,1,1,2],dtype=torch.float64)
        effective,centered=ad.parts(raw)
        for g in range(3):self.assertAlmostEqual(float(centered[ad.ids==g].detach().mean()),0)
        torch.autograd.gradcheck(lambda p:ad.transform(p),(raw,))
        self.assertEqual(len(effective),5)

    def test_checkpoint_baseline_matches_unwrapped_official_trainer(self):
        from matryoshka_attribution.trainer import learn_scores
        from src.mattr_hierarchy.adapter import Hierarchy
        from src.mattr_data_scaling.training import learn,checkpoints
        plan=pd.DataFrame(dict(row=[0]*30,k=np.linspace(1,63,30)))
        def loss(mask):return (mask-torch.linspace(0,1,64)).square().sum()
        class Sampler:
            def __init__(self):self.i=0
            def sample(self):
                k=float(plan.k.iloc[self.i]);self.i+=1;return k
        kw=dict(variant='topk',k_schedule='uniform',T=.5,n_iters=30,lr=.05,optimizer='adam',device='cpu')
        direct=learn_scores(64,loss,steps=30,k_sampler=Sampler(),**kw)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            saved=learn(learn_scores,64,loss,Hierarchy(kind='bypass'),plan,target=30,folder=path,identity='cpu',**kw)
            self.assertTrue(torch.equal(direct.scores,saved['scores']))
            torch.save({'incomplete':True},path/'step_000040.pt')
            self.assertEqual(len(checkpoints(path)),1)
            self.assertTrue(list((path/'incomplete').glob('*')))

    def test_offline_report_complete_synthetic_pipeline(self):
        """Test every aggregate/plot branch in a temporary synthetic-only directory."""
        from src.mattr_data_scaling.analysis import analyze,expected
        from src.mattr_data_scaling.core import CAPS
        p=dict(endpoints=matrix(),eap_jobs=[dict(key=f'eap_ig_n128_s{s}',dataset_size=128,seed=s) for s in (42,43,44)]+[dict(key='eap_ig_n10000',dataset_size=10000,seed=42)],caps=CAPS)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'gate5f_protocol.json').write_text('{}')
            pd.DataFrame({'synthetic_fixture':[True]}).to_csv(out/'checkpoint_inventory.csv',index=False)
            for name in ('preflight_cpu.json','preflight_gpu.json','schedule_audit.json'):write_json(out/name,dict(passed=True))
            for cohort in ('primary','development'):pd.DataFrame({'template_id':['a']*8+['b']*8}).to_csv(out/f'cohort_{cohort}.csv',index=False)
            for cohort,key,rec in expected(p):
                folder=out/'evaluations'/cohort/key;folder.mkdir(parents=True)
                write_json(folder/'official_result.json',dict(correctness_passed=True,CPR=.5,CMD=.2,accuracy_log_AUC=.5))
                rows=[];per=[]
                for cap in CAPS:
                    cutoff=8 if rec['method']=='h2' else 4
                    if cap>=296:cutoff=16
                    values=(np.arange(16)<cutoff).astype(int)
                    rows.append(dict(target=cap,accuracy=values.mean(),actual_edges=cap,nominal_n=cap,shortfall=0,mask_sha256='synthetic'))
                    for i,value in enumerate(values):per.append(dict(target=cap,example=i,template_id='a' if i<8 else 'b',correct=value,margin=float(value),top1=value))
                pd.DataFrame(rows).to_csv(folder/'caps.csv',index=False);pd.DataFrame(per).to_csv(folder/'cap_per_prompt.csv',index=False)
                marker(folder,'synthetic',[folder/x for x in ('official_result.json','caps.csv','cap_per_prompt.csv')])
            def fake_structure(dest,records):
                rows=[{**rec,'target':116,'actual_edges':116,'known_heads':2,'incident_known_fraction':.5,'name mover_recall':.5} for rec,_ in records]
                pd.DataFrame(rows).to_csv(dest/'known_head_overlap.csv',index=False)
                pd.DataFrame(rows).to_csv(dest/'topology_random_controls.csv',index=False)
            with patch('src.mattr_data_scaling.analysis.protocol',return_value=p),patch('src.mattr_data_scaling.analysis.ROOT',out),patch('src.mattr_data_scaling.structure.analyze',side_effect=fake_structure),patch('src.mattr_data_scaling.analysis.compute',return_value=pd.DataFrame(dict(job=['synthetic'],wall_seconds=[1.]))):
                analyze(out)
            self.assertEqual(len(list((out/'figures').glob('*.png'))),8)
            decision=json.loads((out/'gate5f_decision.json').read_text())
            self.assertEqual(decision['primary']['classification'],'Robust compact advantage')
            self.assertNotIn('classification',decision['development'])

if __name__=='__main__':unittest.main()
