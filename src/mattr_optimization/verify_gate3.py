"""CPU-only integrity audit of completed Gate 3; no model construction."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import ROOT,sha,write_json
from .gate3 import verify_criteria,CRITERIA_SHA,baseline
from .gate3_sampling import CPRMixture
from .gate3_report import costs
from .gate2_final_analysis import table


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();verify_criteria()
    assert sha(out/'gate3_success_criteria.json')==CRITERIA_SHA
    proto=json.loads((out/'gate3_protocol.json').read_text())
    assert sha(out/'gate3_protocol.json')==(out/'gate3_protocol.sha256').read_text().strip()
    for n,h in proto['adapter_sha256'].items():
        assert sha(ROOT/'src/mattr_optimization'/n)==h
        assert sha(out/'source_at_execution'/n)==h
    protected=json.loads((out/'protected_gate2_hashes.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    completed=json.loads((out/'gate3_completed.json').read_text())
    assert completed['passed_execution'] and not completed['gate4_executed'] and not completed['leaderboard_submission']
    assessment=json.loads((out/'success_assessment.json').read_text())
    winner=assessment['screen']['winner']
    validation=None;count=0;confirmed=0;totals={k:0 for k in ('forward_batches','example_forwards','optimizer_updates','backward_calls')}
    sampling_counts=[]
    for job in sorted(out.glob('*_seed*')):
        if not job.is_dir():continue
        cfg=json.loads((job/'experiment_config.json').read_text());seed=cfg['seed'];lam=cfg['lambda']
        assert cfg['pool']==128 and cfg['subset_seed']==seed and seed in (42,43,44) and lam in (1.,.5,.25)
        for split in ('train','validation'):
            current=pd.read_csv(job/f'{split}_manifest.csv')
            pd.testing.assert_frame_equal(current,pd.read_csv(baseline(seed)/f'{split}_manifest.csv'))
            if split=='validation':
                if validation is None:validation=current
                else:pd.testing.assert_frame_equal(validation,current)
        step=5000 if (job/'completed_5000.json').exists() else 1000
        if step==5000:assert lam==winner;confirmed+=1
        for target in ((1000,5000) if step==5000 else (1000,)):
            saved=json.loads((job/f'completed_{target}.json').read_text())
            assert saved['passed'] and all(sha(job/n)==h for n,h in saved['sha256'].items())
        usage_paths=sorted((job/'attempts').glob('*/training_example_draws.csv'))+[job/'training_example_draws.csv']
        draws=pd.concat([pd.read_csv(p) for p in usage_paths],ignore_index=True)
        old=pd.read_csv(baseline(seed)/'training_example_draws.csv').iloc[:step]
        assert draws.subset_index.tolist()==old.subset_index.tolist()
        log=pd.read_csv(job/f'training_metrics_{step}.csv');assert len(log)==step
        baseline_state=torch.load(baseline(seed)/f'checkpoints/step_{step:06d}.pt',map_location='cpu',weights_only=False)
        if lam==1.:
            assert step==1000
            assert all(json.loads((job/'uniform_replay_validation.json').read_text()).values())
            assert torch.equal(torch.load(job/'final_scores_1000.pt',map_location='cpu',weights_only=False),baseline_state['scores'])
        else:
            samples=pd.concat([pd.read_csv(job/f'sampler_draws_target{t}.csv') for t in ((1000,5000) if step==5000 else (1000,))],ignore_index=True)
            assert len(samples)==step
            old_u=np.array([(r[1]-1)/32490 for r in baseline_state['train_log']])
            np.testing.assert_allclose(samples.u,old_u,rtol=0,atol=1e-12)
            sampler=CPRMixture(32491,lam)
            transformed=[sampler.transform(u) for u in samples.u]
            np.testing.assert_allclose([x[0] for x in transformed],samples.k,rtol=0,atol=1e-8)
            np.testing.assert_allclose(log.k,samples.k,rtol=0,atol=1e-8)
            assert [x[1] for x in transformed]==samples.branch.tolist()
            assert [x[2] for x in transformed]==samples.grid_index.tolist()
            for (branch,index),group in samples.groupby(['branch','grid_index']):
                sampling_counts.append({'lambda':lam,'seed':seed,'updates':step,'branch':branch,'grid_index':index,
                                        'actual_draws':len(group),'fraction_of_updates':len(group)/step})
        for events_path in sorted((job/'attempts').glob('*/forward_events.csv'))+[job/'forward_events.csv']:
            events=pd.read_csv(events_path);assert events.success.all()
            assert not events[events.stage=='validation_evaluation'].grad_enabled.any()
        resource=costs(job)
        assert resource['optimizer_updates']==resource['backward_calls']==step
        for k in totals:totals[k]+=resource[k]
        count+=1
    assert count==9 and confirmed==(3 if winner is not None else 0)
    assert totals['optimizer_updates']==9000+(12000 if winner is not None else 0)
    pd.DataFrame(sampling_counts).to_csv(out/'empirical_sampling_counts.csv',index=False)
    distribution=[]
    for lam in proto['lambdas']:
        for row in proto['distribution']:
            distribution.append({**row,'lambda':lam,'unconditional_grid_mass':(1-lam)*row['conditional_grid_probability']})
    pd.DataFrame(distribution).to_csv(out/'sampling_distribution.csv',index=False)
    full_path=ROOT/'results/mattr_optimization/20261021_gate2_final/frozen/gate1_on_gate2_validation/official_result.json'
    full=json.loads(full_path.read_text())['result']
    references=[]
    for seed in (42,43,44):
        values=json.loads((baseline(seed)/'evaluation/validation_step5000/official_result.json').read_text())['result']
        references.append({'discovery_pool':128,'seed':seed,'updates':5000,'cpr':values[1],'cmd':values[2],'accuracy_log_auc':values[6]})
    references.append({'discovery_pool':10000,'seed':42,'updates':5000,'cpr':full[1],'cmd':full[2],'accuracy_log_auc':full[6]})
    reference_frame=pd.DataFrame(references)
    reference_frame.to_csv(out/'frozen_reference_context.csv',index=False)
    test_log=(ROOT/'.cache/gate3_unit_tests.log').read_text()
    assert 'Ran 6 tests' in test_log and '\nOK' in test_log
    shutil.copyfile(ROOT/'.cache/gate3_unit_tests.log',out/'cpu_unit_tests.log')
    result={'passed':True,'screen_jobs':9,'confirmation_jobs':confirmed,'protected_files_unchanged':len(protected),
            'criteria_sha256':CRITERIA_SHA,'protocol_sha256':sha(out/'gate3_protocol.json'),'totals':totals,
            'uniform_scores_bitwise_equal_to_gate2':True,'all_example_draws_equal_to_gate2':True,
            'one_torch_rng_draw_per_step_verified_against_gate2':True,'sampler_transform_matches_logged_k':True,
            'all_validation_cohorts_identical':True,'inference_gradients_disabled':True,
            'cpu_tests_passed':6,'additional_model_forwards_for_audit':0,'gate4_executed':False,'leaderboard_submission':False}
    write_json(out/'final_validation.json',result);print(json.dumps(result,indent=2))
    appendix='\n\n## Post-run artifact audit\n\n'+(
      f'Passed: {count} screen jobs, {confirmed} confirmation jobs, six CPU tests, '
      f'{len(protected)} protected Gate 2 files unchanged, original criteria SHA `{CRITERIA_SHA}`. '
      'All paired example draws and actual one-draw Torch RNG streams match Gate 2; logged k values match the frozen inverse CDF. '
      'Uniform scores are bitwise identical to Gate 2. All validation cohorts match; inference gradients are disabled. '
      f'Real forwards/batches: {totals["forward_batches"]:,}; example-forwards: {totals["example_forwards"]:,}; '
      f'updates: {totals["optimizer_updates"]:,}. This audit performed zero model forwards. '
      'See final_validation.json, sampling_distribution.csv and empirical_sampling_counts.csv.\n\n')
    appendix+='### Frozen reference context on the identical 128-prompt cohort\n\n'+table(reference_frame)+'\n\n'
    appendix+=('The full-data Gate 1 ranking was already evaluated on this cohort in Gate 2; these values are reused, without retraining '
               'or new model calls. Gate 3 success gates compare with the paired N=128 baselines; this table preserves the official full-data '
               'reference as additional context. A development-cohort CPR gain is not a leaderboard submission or proof of better model fidelity.\n')
    for path in (out/'GATE3_ADAPTIVE_SAMPLING_REPORT.md',ROOT/'GATE3_ADAPTIVE_SAMPLING_REPORT.md'):
        body=path.read_text().split('\n\n## Post-run artifact audit')[0]
        path.write_text(body.rstrip()+appendix,encoding='utf-8')


if __name__=='__main__':main()
