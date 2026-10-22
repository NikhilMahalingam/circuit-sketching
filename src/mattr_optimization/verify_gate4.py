"""Post-run Gate4 audit and interpretation appendix; no model forwards."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from .common import ROOT,sha,write_json
from .gate3 import baseline,verify_criteria,CRITERIA_SHA
from .gate4_report import protocol,job_cost,metrics,base_metrics,base_cost,gates
from .gate2_final_analysis import table


def main():
    audit_started=time.perf_counter()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();proto=protocol(out);verify_criteria()
    assert sha(out/'historical_gate3_success_criteria.json')==CRITERIA_SHA
    for name,h in proto['adapter_sha256'].items():
        assert sha(ROOT/'src/mattr_optimization'/name)==h,name
        assert sha(out/'source_at_execution'/name)==h,name
    for name,h in proto['attribution']['source_hashes'].items():assert sha(ROOT/name)==h,name
    protected=json.loads((out/'protected_previous_hashes.json').read_text())
    for name,h in protected.items():assert sha(ROOT/name)==h,name
    marker=json.loads((out/'gate4_completed.json').read_text());assert marker['passed_execution']
    decision=json.loads((out/'screen_decision.json').read_text())
    pilot=json.loads((out/'pilot_decision.json').read_text());assert len(pilot['scales'])==2
    totals={k:0 for k in ('forward_batches','example_forwards','backward_calls','optimizer_updates')}
    attr_cost=[];init_cost=[];versus_eap=[];compact=[]
    for folder in sorted(out.iterdir()):
        if not folder.is_dir() or not (folder/'process_summary.json').exists():continue
        resource=job_cost(folder)
        for k in totals:totals[k]+=resource[k]
        for path in list((folder/'attempts').glob('*/forward_events.csv'))+[folder/'forward_events.csv']:
            events=pd.read_csv(path);assert events.success.all()
            inference=events.stage.str.contains('validation_evaluation|training_only_sign_sanity',regex=True)
            assert not events[inference].grad_enabled.any()
        if folder.name.startswith('attribution_'):
            seed=int(folder.name.split('seed')[1]);config=json.loads((folder/'experiment_config.json').read_text())
            assert config['ig_steps']==5 and config['batch_size']==4 and config['used_examples']==128 and not config['smoke']
            assert resource['forward_batches']==224 and resource['backward_calls']==160
            assert resource['example_forwards']==896 and resource['optimizer_updates']==0
            frame=pd.read_csv(folder/'edge_mapping.csv',keep_default_na=False)
            assert len(frame)==32491 and frame.mattr_index.tolist()==list(range(32491)) and frame.edge.nunique()==32491
            assert frame[['eap_row','eap_column']].drop_duplicates().shape[0]==32491
            pd.testing.assert_frame_equal(pd.read_csv(folder/'train_manifest.csv'),pd.read_csv(baseline(seed)/'train_manifest.csv'))
            raw=torch.load(folder/'eap_ig_scores.pt',weights_only=True)
            assert raw.shape==(32491,) and torch.isfinite(raw).all()
            stats=pd.read_csv(folder/'initialization_statistics.csv')
            assert stats.finite_gradient.all() and (stats.nonzero_gradients==32491).all()
            assert (stats.soft_mass-stats.requested_k).abs().max()<.1
            attr_cost.append({'seed':seed,**resource})
        if folder.name.startswith('scale'):
            seed=int(folder.name.split('seed')[1]);scale=float(folder.name.split('_')[0][5:])
            target=5000 if (folder/'completed_5000.json').exists() else 1000
            assert resource['optimizer_updates']==resource['backward_calls']==target
            pd.testing.assert_frame_equal(pd.read_csv(folder/'train_manifest.csv'),pd.read_csv(baseline(seed)/'train_manifest.csv'))
            pd.testing.assert_frame_equal(pd.read_csv(folder/'validation_manifest.csv'),pd.read_csv(baseline(seed)/'validation_manifest.csv'))
            states=torch.load(folder/f'checkpoints/step_{target:06d}.pt',weights_only=False,map_location='cpu')
            original=torch.load(baseline(seed)/f'checkpoints/step_{target:06d}.pt',weights_only=False,map_location='cpu')
            assert [r[1] for r in states['train_log']]==[r[1] for r in original['train_log']]
            draws=pd.concat([pd.read_csv(f) for f in sorted(folder.glob('example_draws_*.csv'))]).sort_values('step')
            assert draws.step.tolist()==list(range(1,target+1))
            assert draws.subset_index.tolist()==pd.read_csv(baseline(seed)/'training_example_draws.csv').subset_index.iloc[:target].tolist()
            init_cost.append({'scale':scale,'seed':seed,'step':target,**resource})
            eap=metrics(out/f'eap_only_seed{seed}',0,out/f'eap_only_seed{seed}/quality.json')
            for f in sorted(folder.glob('quality_*.json')):
                step=int(f.stem.split('_')[1]);q=metrics(folder,step)
                versus_eap.append({'scale':scale,'seed':seed,'step':step,'cpr':q['cpr'],'eap_only_cpr':eap['cpr'],
                                   'cpr_delta_vs_eap':q['cpr']-eap['cpr'],'cmd_delta_vs_eap':q['cmd']-eap['cmd'],
                                   'accuracy_auc_delta_vs_eap':q['accuracy_auc']-eap['accuracy_auc']})
                for fraction in (.005,.01,.02,.05):
                    compact.append({'scale':scale,'seed':seed,'step':step,'fraction':fraction,
                                    'accuracy':q[f'accuracy_{fraction}'],'eap_only_accuracy':eap[f'accuracy_{fraction}']})
    assert len(init_cost)==8  # four pilot seeds + two additional seeds for two scales
    if decision['winner'] is None:assert all(r['step']==1000 for r in init_cost)
    else:assert sum(r['step']==5000 for r in init_cost)==3
    test=(ROOT/'.cache/gate4_unit_tests.log').read_text()
    assert 'Ran 5 tests' in test and '\nOK' in test
    shutil.copyfile(ROOT/'.cache/gate4_unit_tests.log',out/'cpu_unit_tests.log')
    pd.DataFrame(attr_cost).to_csv(out/'attribution_compute.csv',index=False)
    pd.DataFrame(versus_eap).to_csv(out/'initialized_vs_eap_only.csv',index=False)
    pd.DataFrame(compact).to_csv(out/'compact_initialized_vs_eap.csv',index=False)
    result={'passed':True,'protected_historical_files_unchanged':len(protected),
      'historical_criteria_sha256':CRITERIA_SHA,'gate4_protocol_sha256':sha(out/'gate4_protocol.json'),
      'cpu_tests_passed':5,'training_jobs':len(init_cost),'screen_winner':decision['winner'],
      'totals':totals,'attribution_data_train_only':True,'all_32491_edges_bijective':True,
      'signed_orientation_verified':True,'paired_k_and_example_streams_equal':True,
      'inference_gradients_disabled':True,'additional_model_forwards_for_audit':0,
      'gate5':False,'full_data_followup':False,'leaderboard_submission':False}
    launch_seconds=sum(json.loads(f.read_text())['seconds'] for f in out.glob('*/launch_elapsed_*.json'))
    smoke_seconds=json.loads((out/'smoke_seed42/process_summary.json').read_text())['seconds']
    result['summed_actual_child_launch_seconds']=launch_seconds
    result['smoke_process_seconds_excluding_interpreter_startup']=smoke_seconds
    result['audit_seconds']=time.perf_counter()-audit_started
    changes=pd.DataFrame(versus_eap)
    summary=changes.groupby(['scale','step'])[['cpr_delta_vs_eap','cmd_delta_vs_eap','accuracy_auc_delta_vs_eap']].mean().reset_index()
    appendix='\n\n## Post-run artifact audit and attribution-only comparison\n\n'
    appendix+=f'Passed five CPU tests and real CUDA artifact audit. {len(protected)} historical files are unchanged, including Gate3 criteria SHA `{CRITERIA_SHA}`. All 32,491 edges align, signed orientation agrees with the training-only patch test, the selected train/development manifests match Gate2, and every actual training row and uniform-k draw matches its paired baseline. No inference gradients were retained. The audit ran zero model forwards.\n\n'
    appendix+='### Measured full128 EAP-IG cost (one per paired subset)\n\n'+table(pd.DataFrame(attr_cost))+'\n\n'
    appendix+='### MAttr learning beyond its attribution-only starting ranking\n\n'+table(summary)+'\n\n'
    appendix+='These are changes relative to EAP-IG alone, not ordinary zero-init MAttr; negative CPR differences cannot support attributing a gain to MAttr optimization. Cross-seed means for pilot-only scales contain only seed42; consult the per-seed table before comparing them. Compact accuracy is signed IOI margin>0, matching the official evaluator, rather than full-vocabulary top1 accuracy.\n'
    appendix+=f'\nSummed actual research child-launch timers: **{launch_seconds/60:.3f}min**, including interpreter/source checks and all registered attribution, training and frozen evaluations. The separately launched smoke process used **{smoke_seconds:.3f}s** after its source checks (its interpreter startup was not timed). CPU unit-test, implementation/protocol/report analysis time is additional. Official progress logs use calendar time and show WSL clock jumps; only synchronized monotonic perf_counter timers enter compute conclusions.\n'
    paired=pd.read_csv(out/'paired_results.csv')
    unchanged_screens=paired[paired.step==1000].copy()
    # Screening/confirmation used complete1,000/5,000 jobs and are unaffected.
    # The frozen initial reporter assigned the complete1,000 job time to500-step
    # display points too. Reconstruct ONLY those early display points using their
    # measured checkpoint prefix plus the same job's fixed setup; preserve source
    # snapshots, protocol bytes, decisions and all quality measurements.
    columns=['scale','seed','step','cpr','baseline_cpr','cpr_delta','relative_cpr_gain','cmd_delta','accuracy_auc_delta','time_ratio']
    old_table=table(paired[columns])
    for index,row in paired[paired.step==500].iterrows():
        folder=out/f'scale{row.scale:g}_seed{int(row.seed)}'
        resource=job_cost(folder,1000)
        state=torch.load(folder/'checkpoints/step_000500.pt',map_location='cpu',weights_only=False)
        prefix=resource['adaptation_seconds']-resource['training_seconds']+state['training_seconds']
        total=prefix+job_cost(out/f'attribution_seed{int(row.seed)}')['adaptation_seconds']
        paired.loc[index,'adaptation_seconds']=total
        paired.loc[index,'time_ratio']=total/base_cost(int(row.seed),500)['adaptation_seconds']
        full=paired[(paired.scale==row.scale)&(paired.seed==row.seed)&(paired.step==1000)].iloc[0]
        assert total<full.adaptation_seconds
    pd.testing.assert_frame_equal(unchanged_screens,paired[paired.step==1000])
    paired.to_csv(out/'paired_results.csv',index=False)
    new_table=table(paired[columns])
    # Re-render the requested compute plot using corrected early display points,
    # explicit ordinary-MAttr/EAP controls, and measured full-budget costs.
    plt.figure(figsize=(8,4))
    for scale,group in paired.groupby('scale'):
        plt.scatter(group.adaptation_seconds,group.cpr,label=f'Initialized scale {scale:g}')
    for seed in (42,43,44):
        for step in (500,1000,5000):
            plt.scatter(base_cost(seed,step)['adaptation_seconds'],base_metrics(out,seed,step)['cpr'],
                        marker='x',color='black',label='Ordinary MAttr' if seed==42 and step==500 else None)
        eap=json.loads((out/f'eap_only_seed{seed}/quality.json').read_text())
        plt.scatter(job_cost(out/f'attribution_seed{seed}')['adaptation_seconds'],eap['cpr'],marker='s',color='gray',
                    label='EAP-IG only' if seed==42 else None)
    plt.xlabel('Total adaptation seconds (EAP-IG + MAttr + setup)');plt.ylabel('Official development CPR')
    plt.title('Exploratory; 500-step time reconstructed from checkpoint prefix',fontsize=9)
    plt.legend(fontsize=7);plt.tight_layout();plt.savefig(out/'figures/cpr_vs_compute.png',dpi=150);plt.close()
    selected=paired[(paired.step==1000)&paired.scale.isin(pilot['scales'])]
    means=selected.groupby('scale')[['relative_cpr_gain','cmd_delta','accuracy_auc_delta','time_ratio',
      'accuracy_delta_0.005','accuracy_delta_0.01','accuracy_delta_0.02','accuracy_delta_0.05']].mean().reset_index()
    means.to_csv(out/'paired_screen_summary.csv',index=False)
    appendix+='\n### Research decision\n\n'+table(means)+'\n\n'
    appendix+='Timing-display correction: 500-step adaptation points use the measured checkpoint-prefix training timer plus the same run\'s fixed setup and EAP-IG cost. They were not separately launched 500-step processes; fixed setup includes the full run\'s postprocessing overhead. The frozen initial reporter used the complete1,000-step job time for these early display points; the published paired CSV/table and compute figure are corrected here. All 1,000-step screen times, metrics, frozen decisions and protocol/source snapshots are unchanged.\n\n'
    if decision['winner'] is None:
        appendix+='Neither registered screen passes leaderboard-quality or efficiency gates. Final initialized CPR at5,000 steps remains **untested**, because the stopping rule forbids confirmation after failure. Lower CMD indicates closer full-model margins at1,000 steps, while the compact-accuracy regressions prevent calling this an overall fidelity improvement. No disjoint confirmation cohort, extra scale, full-data follow-up, Gate5 or leaderboard was run. The small scale0.5 log-AUC gain, where present, is a secondary descriptive result and does not override the failed CPR/compact safeguards.\n'
    for path in (out/'GATE4_EAPIG_INITIALIZATION_REPORT.md',ROOT/'GATE4_EAPIG_INITIALIZATION_REPORT.md'):
        body=path.read_text().split('\n\n## Post-run artifact audit')[0]
        body=body.replace(old_table,new_table)
        if path.parent==out:body=body.replace(f'results/mattr_optimization/{out.name}/figures/','figures/')
        path.write_text(body.rstrip()+appendix,encoding='utf-8')
    result['audit_seconds']=time.perf_counter()-audit_started
    result['early_checkpoint_display_cost_corrected']=True
    result['frozen_screen_costs_and_decisions_unchanged']=True
    write_json(out/'final_validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
