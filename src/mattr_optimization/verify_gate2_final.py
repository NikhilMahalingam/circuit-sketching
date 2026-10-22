"""CPU-only final artifact audit; never instantiate a model or trainer."""
import json
from pathlib import Path
import pandas as pd
import torch
from .common import ROOT, sha, write_json


def main():
    first=ROOT/'results/mattr_optimization/20261021T172035Z_gate2'
    second=ROOT/'results/mattr_optimization/20261021_gate2_seeds43_44'
    final=ROOT/'results/mattr_optimization/20261021_gate2_final'
    protected=json.loads((ROOT/'.cache/gate2_followup_protected_hashes.json').read_text())
    # Snapshot paths were written on Windows. Normalize separators for WSL.
    for name,digest in protected.items():
        assert sha(ROOT/Path(name.replace('\\','/')))==digest, name
    jobs=sorted(first.glob('seed*_n*'))+sorted(second.glob('seed*_n*'))+sorted((final/'controls').glob('train*_subset*'))
    assert len(jobs)==16
    validation=None
    totals=dict(forward_batches=0,example_forwards=0,backward_calls=0,optimizer_updates=0)
    costs=[]
    for job in jobs+[final/'frozen']:
        marker=json.loads((job/'completed.json').read_text())
        assert marker['passed']
        if 'sha256' in marker:
            assert all(sha(job/name)==digest for name,digest in marker['sha256'].items())
        process=json.loads((job/'process_summary.json').read_text())
        assert process['success']
        counts=process['counts']
        events=pd.read_csv(job/'forward_events.csv')
        stages=pd.read_csv(job/'compute_accounting.csv')
        assert events.success.all() and stages.success.all()
        assert len(events)==counts['forward_batches']
        assert int(events.batch_size.sum())==counts['example_forwards']
        assert all(int(stages[key].sum())==value for key,value in counts.items())
        assert not stages.seconds.lt(0).any()
        if job==final/'frozen':
            assert not events.grad_enabled.any()
            assert counts['optimizer_updates']==counts['backward_calls']==0
        else:
            assert counts['optimizer_updates']==counts['backward_calls']==5000
            val=pd.read_csv(job/'validation_manifest.csv')
            train=pd.read_csv(job/'train_manifest.csv')
            assert len(val)==128 and not (set(val.clean_prompt)&set(train.clean_prompt))
            assert train.official_filtered_index.nunique()==len(train)
            if validation is None: validation=val
            else: pd.testing.assert_frame_equal(val,validation)
        for key in totals: totals[key]+=counts[key]
        costs.append({'job':str(job.relative_to(ROOT)),**counts,'measured_stage_seconds':stages.seconds.sum(),
                      'peak_allocated_mib':stages.peak_allocated_mib.max()})
    vectors=0
    for path in (final/'frozen').glob('*/official_result.json'):
        saved=json.loads(path.read_text())
        assert len(saved['raw_sha256'])==12
        assert all(sha(path.parent/name)==digest for name,digest in saved['raw_sha256'].items())
        for name in saved['raw_sha256']:
            vector=torch.load(path.parent/name,map_location='cpu',weights_only=False)
            assert vector.ndim==1 and torch.isfinite(vector).all()
            vectors+=1
    assert vectors==300
    q=pd.read_csv(final/'quality_all_steps.csv')
    assert len(q)==36 and not q.duplicated(['seed','pool','step']).any()
    assert q.actual_unique_visited.between(1,q.pool).all()
    assert q[q.step==5000].actual_unique_visited.eq(q[q.step==5000].pool).all()
    assert q.actual_presentations.eq(q.step).all()
    for seed in (42,43,44):
        sets=[]
        for n in (16,32,64,128):
            campaign=first if seed==42 else second
            sets.append(set(pd.read_csv(campaign/f'seed{seed}_n{n}/train_manifest.csv').official_filtered_index))
        assert all(a<=b for a,b in zip(sets,sets[1:]))
    provenance=json.loads((final/'analysis_provenance.json').read_text())
    criteria=final/'gate3_success_criteria.json'
    assert sha(criteria)==provenance['criteria_sha256']
    assert not json.loads(criteria.read_text())['gate3_executed']
    reference=json.loads((final/'frozen/gate1_on_gate2_validation/official_result.json').read_text())
    assert reference['identity']['score_sha256']==sha(ROOT/'results/mattr_optimization/20261021T134352Z_gate1/final_scores.pt')
    pd.DataFrame(costs).to_csv(final/'complete_study_accounting.csv',index=False)
    result={'passed':True,'main_jobs':12,'conditional_controls':4,'frozen_vector_files':vectors,
            'protected_files_unchanged':len(protected),'measured_validation_checkpoint_rows':len(q),
            'all_validation_cohorts_identical':True,'train_validation_disjoint':True,'nested_pools_verified':True,
            'all_selected_rows_visited_by_step500':bool(q[q.step==500].actual_unique_visited.eq(q[q.step==500].pool).all()),
            'all_selected_rows_visited_by_step1000':bool(q[q.step==1000].actual_unique_visited.eq(q[q.step==1000].pool).all()),
            'study_totals_including_preexisting_seed42':totals,
            'additional_model_forwards_for_audit':0,'frozen_evaluation_gradients_disabled':True,
            'criteria_sha256':sha(criteria),'gate3_executed':False}
    write_json(final/'final_validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
