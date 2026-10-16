"""Independent artifact consistency checks; no model evaluation or metric replacement."""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from .runtime import ROOT, PERCENTAGES, digest, write_json


def read(path):return json.loads(Path(path).read_text())


def verify_job(path):
    config=read(path/'experiment_config.json');complete=read(path/'complete.json')
    accounting=pd.read_csv(path/'compute_accounting.csv')
    events=pd.read_json(path/'forward_events.jsonl',lines=True)
    assert events.success.all(),f'Failed forward in {path}'
    assert int(accounting.forward_batches.sum())==len(events),path
    assert int(accounting.example_forwards.sum())==int(events.batch_size.sum()),path
    inference=events[events.stage.str.endswith('_evaluation')]
    assert not inference.grad_enabled.any(),path
    for split in ('train','validation','test'):
        metric_path=path/({'validation':'official_evaluation.json',
                          'train':'official_train_evaluation.json',
                          'test':'official_test_evaluation.json'}[split])
        if not metric_path.exists():continue
        metric=read(metric_path);curve=pd.read_csv(path/(split+'_curve.csv'))
        raw=np.load(path/(split+'_per_example.npz'))
        assert len(curve)==10 and curve.retained_fraction.tolist()==list(PERCENTAGES)
        assert curve.requested_nodes.tolist()==[int(x*157) for x in PERCENTAGES]
        assert int(curve.iloc[-1].retained_nodes)==157
        assert len(raw['baseline'])==metric['examples']
        assert np.isfinite(raw['baseline']).all() and np.isfinite(raw['corrupted']).all()
        np.testing.assert_allclose(raw['p9'],raw['baseline'],atol=5e-4,rtol=1e-4)
        denominator=float(raw['baseline'].mean()-raw['corrupted'].mean())
        assert denominator>0
        for i in range(10):
            assert np.isfinite(raw[f'p{i}']).all()
            expected=(float(raw[f'p{i}'].mean())-float(raw['corrupted'].mean()))/denominator
            np.testing.assert_allclose(expected,curve.iloc[i].official_normalized_logit_diff_cpr,atol=2e-6)
            if curve.iloc[i].requested_nodes==0:
                np.testing.assert_allclose(raw[f'p{i}'],raw['corrupted'],atol=1e-5)
        assert metric['intervention']=='patching' and metric['absolute_ranking'] is False
    ranking=pd.read_csv(path/'node_ranking.csv')
    assert len(ranking)==157 and ranking.component.nunique()==157
    assert ranking.forward_index.nunique()==157 and np.isfinite(ranking.score).all()
    if 'attribution_batch_size' in config and config['method'] in ('EAP','EAP-IG'):
        assert config['attribution_batch_size']==1
        n=complete['eligible_unique_train_examples']
        steps=1 if config['method']=='EAP' else config['ig_steps']
        attribution=accounting[accounting.stage=='attribution_construction']
        assert int(attribution.backward_calls.sum())==n*steps
        assert int(attribution.forward_batches.sum())==n*(steps+2)
        assert int(accounting.optimizer_steps.sum())==0
    if config.get('method')=='MAttr' and 'steps' in config:
        n=config['steps'];assert int(accounting.optimizer_steps.sum())>=n
        assert int(accounting.backward_calls.sum())>=n
        training_events=events[events.stage=='attribution_training']
        assert int(training_events.grad_enabled.sum())>=n
        curve=pd.read_csv(path/'training_curve.csv');assert curve.step.tolist()==list(range(n))
        assert np.isfinite(curve[['k','loss']].to_numpy()).all()
        sampled=pd.read_json(path/'training_presentations.jsonl',lines=True)
        assert len(sampled)==complete['total_training_presentations']
        assert sampled.subset_index.nunique()==complete['sampled_unique_train_examples']
        assert sampled.subset_index.between(0,complete['eligible_unique_train_examples']-1).all()
        checkpoint=torch.load(path/f'checkpoints/step_{n:06d}.pt',map_location='cpu',weights_only=False)
        assert checkpoint['scores'].shape==(157,) and not checkpoint['scores'].requires_grad
        from types import SimpleNamespace
        from .runtime import add_official_paths
        from .evaluate import map_mattr_scores
        add_official_paths()
        from eap.graph import Graph
        graph=Graph.from_pt(str(path/'node_ranking.pt'))
        original=graph.nodes_scores.clone()
        map_mattr_scores(graph,checkpoint['scores'],SimpleNamespace(cfg=SimpleNamespace(n_layers=12,n_heads=12)))
        torch.testing.assert_close(graph.nodes_scores,original,atol=0,rtol=0,equal_nan=True)
        for state in checkpoint['optimizer']['state'].values():assert int(state['step'])==n
        with (path/'official/ioi_gpt2_validation.pkl').open('rb') as handle:official=pickle.load(handle)
        metric=read(path/'official_evaluation.json')
        assert official['area_under']==metric['cpr_auc_area_under']
        assert official['area_from_1']==metric['cmd_auc_area_from_1']
    if config.get('stage')=='final_confirmation':
        campaign=path.parents[2]
        lock=read(campaign/'confirmation_selection.json')
        assert lock['locked_before_test'] and digest(campaign/'confirmation_selection.json')==config['selection_sha256']
        assert digest(campaign/config['source']/'node_ranking.pt')==config['ranking_sha256']
        test=pd.read_csv(path/'test_manifest.csv')
        for split in ('train','validation'):
            prior=pd.read_csv(campaign/f'reference/{split}_manifest.csv')
            assert not set(test.prompt_sha256)&set(prior.prompt_sha256)
        assert int(accounting.backward_calls.sum())==0 and int(accounting.optimizer_steps.sum())==0
    else:
        train=pd.read_csv(path/'train_manifest.csv');val=pd.read_csv(path/'validation_manifest.csv')
        assert not set(train.prompt_sha256)&set(val.prompt_sha256)
    return {'job':str(path),'forward_batches':len(events),'example_forwards':int(events.batch_size.sum()),
            'inference_forwards_without_grad':len(inference),'passed':True}


def verify(out,require_complete=False):
    jobs=[p.parent for p in sorted(out.rglob('complete.json'))]
    rows=[verify_job(path) for path in jobs]
    fixed_hash=None
    for n in (16,32,64,128):
        for seed in (42,43,44):
            source=out/f'scaling/n{n}_seed{seed}'
            if not (source/'complete.json').exists():
                if require_complete:raise AssertionError(f'Missing {source}')
                continue
            train=pd.read_csv(source/'train_manifest.csv');val=pd.read_csv(source/'validation_manifest.csv')
            val_hash=digest(source/'validation_manifest.csv')
            fixed_hash=fixed_hash or val_hash;assert val_hash==fixed_hash
            if n>16:
                prefix=pd.read_csv(out/f'scaling/n{n//2}_seed{seed}/train_manifest.csv')
                pd.testing.assert_frame_equal(train.iloc[:len(prefix)].reset_index(drop=True),prefix)
            for method in ('EAP','EAP-IG'):
                baseline=out/f'baselines/{method}/n{n}_seed{seed}'
                if not (baseline/'complete.json').exists():
                    if require_complete:raise AssertionError(f'Missing {baseline}')
                    continue
                for split in ('train','validation'):
                    pd.testing.assert_frame_equal(pd.read_csv(source/f'{split}_manifest.csv'),pd.read_csv(baseline/f'{split}_manifest.csv'))
            for step in (100,250,500):
                path=out/f'checkpoint_evaluations/n{n}_seed{seed}/step{step}'
                if require_complete:assert (path/'complete.json').exists(),path
                if step==500 and (path/'complete.json').exists():
                    first=read(source/'official_evaluation.json')['cpr_auc_area_under']
                    last=read(path/'official_evaluation.json')['cpr_auc_area_under']
                    np.testing.assert_allclose(first,last,atol=2e-5)
    for name,value in read(out/'protected_previous_artifacts.json').items():assert digest(ROOT/name)==value,name
    if require_complete:
        assert (out/'reference/complete.json').exists()
        assert (out/'reference_common_validation/complete.json').exists()
        confirmation=[p for p in jobs if 'confirmation' in p.relative_to(out).parts]
        if (out/'confirmation_selection.json').exists():
            assert len(confirmation)==9,len(confirmation)
        assert len(jobs)==75+len(confirmation),len(jobs)
    test_manifests=list((out/'confirmation').glob('*/*/test_manifest.csv'))
    if test_manifests:
        first=digest(test_manifests[0])
        assert all(digest(p)==first for p in test_manifests)
    result={'passed':True,'complete_campaign_required':require_complete,'verified_jobs':rows,
            'previous_artifacts_unchanged':True,
            'public_test_accessed':bool(list((out/'confirmation').glob('*/*/test_access_started.json')))}
    write_json(out/'artifact_verification.json',result)
    print(f'Verified {len(jobs)} completed jobs.',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    verify(args.out,args.require_complete)


if __name__=='__main__':main()
