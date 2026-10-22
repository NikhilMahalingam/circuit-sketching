"""CPU-only completed-campaign audit. Raw vectors remain local, never Git staged."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import ROOT,sha,write_json
from src.mattr_optimization.gate3 import baseline,verify_criteria,CRITERIA_SHA
from src.mattr_optimization.gate2_final_analysis import table
from src.mattr_optimization.gate4_report import job_cost
from .analysis import protocol,effective_scores
from .graphs import audit as graph_audit


def main():
    started=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();proto=protocol(out);verify_criteria()
    complete=json.loads((out/'completed.json').read_text());assert complete['passed_execution']
    for name,h in proto['source_sha256'].items():
        assert sha(ROOT/name)==h,name
        assert sha(out/'source_at_execution'/name)==h,name
    previous=json.loads((out/'protected_previous_hashes.json').read_text())
    for name,h in previous.items():assert sha(ROOT/name)==h,name
    assert sha(out/'edge_mapping.csv')==proto['mapping_sha256']
    assert gzip.decompress((out/'edge_mapping.csv.gz').read_bytes())==(out/'edge_mapping.csv').read_bytes()
    mapping=pd.read_csv(out/'edge_mapping.csv',keep_default_na=False)
    assert len(mapping)==32491 and mapping.mattr_index.tolist()==list(range(32491)) and mapping.edge.nunique()==32491
    for family,h in proto['graph_files'].items():
        path=out/'graphs'/f'{family}_neighbors.npy';assert sha(path)==h
        structure=graph_audit(np.load(path));assert structure['degree_min']==structure['degree_max']==8
        assert structure['vertices']==32491 and structure['edges']==129964 and structure['components']==1
    spectral=pd.read_csv(out/'graph_spectral_statistics.csv').set_index('family')
    assert 0<spectral.loc['G0','laplacian_gap']<spectral.loc['G1','laplacian_gap']<spectral.loc['G2','laplacian_gap']
    assert spectral.spectral_residual.max()<1e-5
    assert json.loads((out/'phaseA_gpu_passed.json').read_text())['passed']
    checks=pd.read_csv(out/'gpu_mixing_checks.csv');assert checks.cuda_gradcheck.all()
    smoke=out/'phaseA_smoke';assert json.loads((smoke/'beta_zero_equivalence.json').read_text())['passed']
    assert len(pd.read_csv(smoke/'patching_checks.csv'))==4
    totals={k:0 for k in ('forward_batches','example_forwards','backward_calls','optimizer_updates')}
    jobs=[]
    for job in [smoke,*sorted((out/'jobs').glob('*')),*out.glob('baseline_2500_*')]:
        if not (job/'process_summary.json').exists():continue
        resource=job_cost(job)
        for k in totals:totals[k]+=resource[k]
        for f in sorted((job/'attempts').glob('*/forward_events.csv'))+[job/'forward_events.csv']:
            events=pd.read_csv(f);assert events.success.all()
            inference=events.stage.str.contains('evaluation|patching_checks')
            assert not events[inference].grad_enabled.any()
        if job==smoke:
            assert resource['optimizer_updates']==resource['backward_calls']==90
            continue
        if job.name.startswith('baseline_2500'):continue
        target=5000 if (job/'completed_5000.json').exists() else 1000
        assert resource['optimizer_updates']==resource['backward_calls']==target
        config=json.loads((job/'experiment_config.json').read_text());seed=config['seed']
        for split in ('train','validation'):
            pd.testing.assert_frame_equal(pd.read_csv(job/f'{split}_manifest.csv'),pd.read_csv(baseline(seed)/f'{split}_manifest.csv'))
        state=torch.load(job/f'checkpoints/step_{target:06d}.pt',weights_only=False,map_location='cpu')
        old=torch.load(baseline(seed)/f'checkpoints/step_{target:06d}.pt',weights_only=False,map_location='cpu')
        assert [r[1] for r in state['train_log']]==[r[1] for r in old['train_log']]
        draws=pd.concat([pd.read_csv(f) for f in job.glob('example_draws_*.csv')]).sort_values('step')
        assert draws.step.tolist()==list(range(1,target+1))
        assert draws.subset_index.tolist()==pd.read_csv(baseline(seed)/'training_example_draws.csv').subset_index.iloc[:target].tolist()
        raw=torch.load(job/f'raw_scores_{target}.pt',weights_only=True)
        exported=torch.load(job/f'effective_scores_{target}.pt',weights_only=True)
        assert torch.equal(raw,state['scores']) and torch.isfinite(exported).all()
        torch.testing.assert_close(exported,effective_scores(out,job,target),atol=2e-6,rtol=2e-6)
        ranking=pd.read_csv(job/f'ranking_{target}.csv').set_index('edge')
        np.testing.assert_allclose(ranking.loc[mapping.edge,'score'].to_numpy(),exported.numpy(),atol=1e-7,rtol=1e-7)
        if config['family']=='zero':assert torch.equal(raw,old['scores']) and torch.equal(exported,raw)
        for f in job.glob('gradient_diagnostics_*.csv'):
            gradient=pd.read_csv(f)
            assert np.isfinite(gradient[['raw_gradient_norm','effective_gradient_norm','raw_score_variance','effective_score_variance']]).all().all()
        jobs.append({'job':job.name,'family':config['family'],'beta':config['beta'],'seed':seed,'steps':target,**resource})
    assert totals['optimizer_updates']<=proto['budget']['optimizer_updates_max']
    test=(ROOT/'.cache/mattr_expander_tests.log').read_text();assert 'Ran 6 tests' in test and '\nOK' in test
    shutil.copyfile(ROOT/'.cache/mattr_expander_tests.log',out/'cpu_unit_tests.log')
    b=json.loads((out/'phaseB_decision.json').read_text())
    if not b['selected']:
        assert len(jobs)==9 and all(r['steps']==1000 and r['seed']==42 for r in jobs)
        assert not (out/'phaseC_decision.json').exists() and not (out/'final_decision.json').exists()
    cost=pd.DataFrame(jobs);cost.to_csv(out/'audited_training_compute.csv',index=False)
    pairs=pd.read_csv(out/'paired_results.csv');screen=pairs[pairs.step==1000]
    compact_cols=[f'accuracy_delta_{f}' for f in (.005,.01,.02,.05)]
    summary=screen.groupby(['family','beta']).agg(seeds=('seed','count'),relative_CPR_change=('relative_cpr_gain','mean'),
        CMD_change=('cmd_delta','mean'),accuracy_log_AUC_change=('accuracy_auc_delta','mean'),time_ratio=('time_ratio','mean')).reset_index()
    summary.to_csv(out/'screen_summary.csv',index=False)
    compact=screen.groupby(['family','beta'])[compact_cols].mean().reset_index();compact.to_csv(out/'compact_accuracy_differences.csv',index=False)
    launch=sum(json.loads(f.read_text())['seconds'] for f in out.rglob('launch_elapsed_*.json'))
    outcome={'passed':True,'historical_files_unchanged':len(previous),'historical_Gate3_criteria_sha256':CRITERIA_SHA,
      'protocol_sha256':sha(out/'gate5a_protocol.json'),'cpu_tests':6,'all_graphs_degree8_connected':True,
      'canonical_mapping_bijective':True,'beta_zero_bitwise_matches_Gate2':True,'raw_checkpoints_effective_rankings_verified':True,
      'paired_example_and_k_streams_equal':True,'inference_no_grad_verified':True,'totals':totals,
      'actual_child_launch_seconds':launch,'phaseB_selected':b['selected'],
      'phaseD_executed':(out/'final_decision.json').exists(),'leaderboard_submission':False,
      'additional_model_forwards_for_audit':0,'audit_seconds':time.perf_counter()-started}
    write_json(out/'final_validation.json',outcome)
    appendix='\n\n## Final artifact audit\n\n'
    appendix+=f'Passed six CPU tests and real CUDA graph/model checks. All {len(previous)} protected historical files are unchanged; historical Gate3 criteria SHA remains `{CRITERIA_SHA}`. All32,491 edge identities align, degree8 graphs are simple/undirected/connected, raw checkpoint state and effective ranking export agree, and actual training rows/k draws match Gate2. Fresh beta0 calibration scores match Gate2 bitwise. No inference gradients or model forwards were used by this audit.\n\n'
    appendix+='### Screen summary (single-seed rows are exploratory)\n\n'+table(summary)+'\n\n'+table(compact)+'\n\n'
    appendix+=f'Real root forwards {totals["forward_batches"]:,}; example-forwards {totals["example_forwards"]:,}; backwards {totals["backward_calls"]:,}; optimizer updates {totals["optimizer_updates"]:,}. Actual child-launch timers total {launch/60:.3f}min. See final_validation.json, all saved curves and local raw/effective vectors.\n\n'
    if not b['selected']:
        appendix+='PhaseB was the only research screen run. No candidate met its preregistered advancement gate; PhaseC, optional permutation training and PhaseD were skipped. Consequently final5000-step improvement and expander-specific reproducibility are **untested**, rather than disproven. One-seed comparisons cannot establish ranking stability of modified methods or generalization. Any compact/CPR gains that fail the joint safeguards remain secondary observations. No expansion of the sweep is authorized.\n'
    for path in (ROOT/'GATE5A_EXPANDER_MATTR_REPORT.md',out/'GATE5A_EXPANDER_MATTR_REPORT.md'):
        body=path.read_text().split('\n\n## Final artifact audit')[0]
        path.write_text(body.rstrip()+appendix,encoding='utf-8')
    print(json.dumps(outcome,indent=2))


if __name__=='__main__':main()
