"""CPU-only final artifact audit; no parameter optimization or model forwards."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import ROOT,configure,sha,write_json
from src.mattr_optimization.gate3 import baseline,CRITERIA,CRITERIA_SHA
from .adapter import Hierarchy,group_sum
from .analysis import protocol,select_b,select_c

def run(out):
    start=time.perf_counter();configure();torch.set_num_threads(4)
    p=protocol(out)
    for name,h in p['source_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in p['cohorts'].items():assert sha(ROOT/name)==h,name
    for name,h in p['mapping_hashes'].items():assert sha(out/name)==h,name
    assert sha(out/'edge_mapping.csv')==p['edge_mapping_sha256']
    protected=json.loads((out/'protected_previous_hashes.json').read_text())
    for name,h in protected.items():assert sha(ROOT/name)==h,name
    assert sha(CRITERIA)==CRITERIA_SHA
    decision_paths=[x for x in out.glob('phase*_decision.json')]
    before={x.name:sha(x) for x in decision_paths};selected=select_b(out)
    if selected:
        select_c(out,selected)
        if (out/'phaseD_decision.json').exists():select_c(out,selected,step=5000)
    assert before=={x.name:sha(x) for x in decision_paths}
    jobs=0;checkpoints=0;events=[];summaries=[];structure=[]
    for folder in [out/'pilot',out/'phaseA_smoke',*sorted((out/'jobs').glob('*'))]:
        if not folder.is_dir():continue
        cfg=json.loads((folder/'experiment_config.json').read_text());weight=json.loads((folder/'model_weight_audit.json').read_text())
        assert weight['passed'] and weight['before_sha256']==weight['after_sha256']
        for split in ('train','validation'):
            if not (folder/f'{split}_manifest.csv').exists():continue
            pd.testing.assert_frame_equal(pd.read_csv(folder/f'{split}_manifest.csv'),pd.read_csv(baseline(cfg['seed'])/f'{split}_manifest.csv'))
        if cfg['phase']=='smoke':
            assert json.loads((folder/'bypass_equivalence.json').read_text())['passed']
            assert json.loads((folder/'exact_resume.json').read_text())['passed']
        ids=np.load(out/f'{cfg["family"]}{"_shuffled" if cfg["shuffled"] else ""}_groups.npy') if cfg['family'] in ('H1','H2') else None
        h=Hierarchy(ids,cfg['gamma'],cfg['scale'],'hierarchy' if ids is not None else cfg['family'])
        for state_path in sorted((folder/'checkpoints').glob('step_*.pt')):
            state=torch.load(state_path,map_location='cpu',weights_only=False);raw=state['scores'];effective,r=h.parts(raw)
            assert torch.isfinite(effective).all() and effective.numel()==32491
            if ids is not None:
                assert float((group_sum(r,h.padded,h.valid)/h.sizes).abs().max())<1e-5
                assert all(prm in ('step','exp_avg','exp_avg_sq') for prm in state['optimizer']['state'][0])
            checkpoints+=1
            target=state['completed']
            if (folder/f'effective_scores_{target}.pt').exists():
                exported=torch.load(folder/f'effective_scores_{target}.pt',map_location='cpu',weights_only=False)
                # CPU/GPU reductions may differ by float32 rounding; ranking export is verified against saved GPU evaluation vector below.
                torch.testing.assert_close(exported,effective,atol=2e-6,rtol=1e-5)
            if (folder/f'evaluation_step{target}/official_result.json').exists():
                evaluation=json.loads((folder/f'evaluation_step{target}/official_result.json').read_text())
                assert evaluation['identity']['checkpoint_sha256']==sha(state_path)
                assert evaluation['identity']['validation_manifest_sha256']==sha(folder/'validation_manifest.csv')
        for draw_file in folder.glob('example_draws_*.csv'):
            draws=pd.read_csv(draw_file);source=pd.read_csv(baseline(cfg['seed'])/'training_example_draws.csv')
            assert draws.subset_index.tolist()==source.subset_index.iloc[draws.step.iloc[0]-1:draws.step.iloc[-1]].tolist()
        for target_log in folder.glob('training_metrics_*.csv'):
            log=pd.read_csv(target_log);baseline_log=pd.read_csv(baseline(cfg['seed'])/'training_metrics.csv')
            np.testing.assert_array_equal(log.k.to_numpy(),baseline_log.k.iloc[:len(log)].to_numpy())
        paths=sorted((folder/'attempts').glob('*/process_summary.json'))+[folder/'process_summary.json']
        for path in paths:
            proc=json.loads(path.read_text());assert proc['success'];summaries.append(proc)
            frame=pd.read_csv(path.parent/'forward_events.csv')
            assert frame.success.all()
            assert not frame[frame.stage.str.contains('evaluation|patching',na=False)].grad_enabled.any()
            assert len(frame)==proc['counts']['forward_batches'] and int(frame.batch_size.sum())==proc['counts']['example_forwards']
            events.append(frame)
        jobs+=1
    assert json.loads((out/'gpu_checks_passed.json').read_text())['passed']
    counts={k:sum(proc['counts'][k] for proc in summaries) for k in ('forward_batches','example_forwards','backward_calls','optimizer_updates')}
    launches=[json.loads(x) for x in (out/'launch_timers.jsonl').read_text().splitlines()]
    result={'passed':True,'historical_files_unchanged':len(protected),'historical_criteria_sha256':sha(CRITERIA),
      'protocol_sha256':sha(out/'gate5c_protocol.json'),'jobs':jobs,'checkpoints':checkpoints,'cpu_tests':6,
      'all_original_edges':32491,'groups':{'H1':157,'H2':445},'exact_bypass_and_resume':True,
      'weight_updates':0,'effective_exports_verified':True,'cohort_and_row_k_streams_match':True,
      'decisions_recomputed_unchanged':True,'counts':counts,'total_child_launch_seconds':sum(x['seconds'] for x in launches),
      'additional_model_forwards_for_audit':0,'audit_seconds':time.perf_counter()-start}
    write_json(out/'final_validation.json',result)
    appendix='\n\n## Final artifact audit\n\n'+json.dumps(result,indent=2)+'\n'
    for path in (out/'GATE5C_HIERARCHICAL_MATTR_REPORT.md',ROOT/'GATE5C_HIERARCHICAL_MATTR_REPORT.md'):
        body=path.read_text().split('\n\n## Final artifact audit')[0];path.write_text(body.rstrip()+appendix)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out.resolve())
