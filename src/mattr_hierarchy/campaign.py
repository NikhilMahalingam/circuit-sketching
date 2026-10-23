"""Frozen Gate 5C phases: training-only scale preflight, bounded screens, gated 5k."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from src.mattr_optimization.common import ROOT,configure,sha,write_json,git,protected_snapshot
from src.mattr_optimization.gate3 import baseline,CRITERIA,CRITERIA_SHA
from .analysis import job,select_b,select_c,controls_for,generate

REPORTS=['results/mattr_optimization/20261021T134352Z_gate1/MATTR_BASELINE_REPORT.md',
 'GATE2_FINAL_ANALYSIS.md','GATE3_ADAPTIVE_SAMPLING_REPORT.md','GATE4_EAPIG_INITIALIZATION_REPORT.md',
 'GATE5A_EXPANDER_MATTR_REPORT.md','GATE5B_INTERACTION_DIAGNOSTICS_REPORT.md']

def source_hashes():
    paths=[*sorted((ROOT/'src/mattr_hierarchy').glob('*.py')),ROOT/'tests/test_mattr_hierarchy.py']
    paths += [ROOT/'src/mattr_optimization'/n for n in ('common.py','gate0.py','gate1.py','gate2.py','gate3.py','gate4_train.py','gate4_init.py','gate4_report.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'gate5c_preflight.json').exists():
        p=json.loads((out/'gate5c_preflight.json').read_text())
        for name,h in p['source_sha256'].items():assert sha(ROOT/name)==h,name
        return p
    configure()
    import torch
    from eap.graph import Graph
    from .groups import construct
    torch.set_num_threads(4)
    reports={}
    for name in REPORTS:
        content=(ROOT/name).read_text();assert content.strip()
        reports[name]={'sha256':sha(ROOT/name),'characters_read':len(content)}
    write_json(out/'prior_reports_read.json',reports)
    protected=protected_snapshot()
    # Also protect untracked historical result vectors/checkpoints, not just Git files.
    for path in (ROOT/'results').rglob('*'):
        if path.is_file() and not path.is_relative_to(out) and not path.name.startswith('.'):
            protected[path.relative_to(ROOT).as_posix()]=sha(path)
    write_json(out/'protected_previous_hashes.json',protected)
    graph=Graph.from_model(dict(n_layers=12,n_heads=12,parallel_attn_mlp=False,d_model=768))
    construct(graph,out)
    from .gpu_checks import run
    run(out)
    tests=(ROOT/'.cache/mattr_hierarchy_tests.log').read_text();assert 'Ran 6 tests' in tests and '\nOK\n' in tests
    (out/'cpu_unit_tests.log').write_text(tests)
    p={'gate':'5C preflight','git_commit':git(ROOT,'rev-parse','HEAD').decode().strip(),
       'source_sha256':source_hashes(),'model':'gpt2-small','task':'official MIB IOI',
       'pilot':'20 gradient-only probes at saved Gate2 seed42 step500 scores; first4 training rows; k floor N*[.005,.01,.02,.05,.5]. No optimizer or validation.',
       'scale_rule':{'negligible':'both hierarchy high-gamma median gradient ratios<1e-4 AND median loss ratios<1e-6',
          'dominant':'both hierarchy low-gamma median gradient ratios>1 AND median loss ratios>.5',
          'replacement':'if either extreme condition holds, one common scale=.01/geometric_mean(high-gamma median gradient ratios); target1% regularizer/task gradient ratio for gamma.01. Otherwise scale1. No development-metric choice; gammas stay decade-separated.',
          'initial_gammas':[.001,.01]},
       'mapping_hashes':{p.name:sha(p) for p in out.glob('*groups.npy')},'edge_mapping_sha256':sha(out/'edge_mapping.csv'),
       'model_weights':'never optimizer participants; preserve official requires-grad behavior and verify complete SHA before/after'}
    write_json(out/'gate5c_preflight.json',p);(out/'gate5c_preflight.sha256').write_text(sha(out/'gate5c_preflight.json')+'\n')
    write_json(out/'preflight_frozen_at.json',{'utc':datetime.now(timezone.utc).isoformat(),'modified_CPR_seen':False})
    return p

def freeze(out,preflight):
    target=out/'gate5c_protocol.json'
    if target.exists():
        assert sha(target)==(out/'gate5c_protocol.sha256').read_text().strip()
        return json.loads(target.read_text())
    probe=pd.read_csv(out/'pilot/magnitude_probe.csv')
    stats=probe.groupby(['family','gamma'])[['gradient_ratio','loss_ratio','mattr_abs_loss','penalty']].median()
    high=stats.xs(.01,level='gamma');low=stats.xs(.001,level='gamma')
    weak=bool((high.gradient_ratio<1e-4).all() and (high.loss_ratio<1e-6).all())
    dominant=bool((low.gradient_ratio>1).all() and (low.loss_ratio>.5).all())
    scale=.01/float(np.exp(np.log(high.gradient_ratio.clip(lower=1e-12)).mean())) if weak or dominant else 1.
    assert np.isfinite(scale) and scale>0
    p={**preflight,'gate':'5C','regularization_scale':{'scale':scale,'replaced':weak or dominant,'negligible':weak,'dominant':dominant,
          'rule':preflight['scale_rule'],'training_only_medians':stats.reset_index().to_dict(orient='records'),'probe_sha256':sha(out/'pilot/magnitude_probe.csv')},
       'dependency_provenance':json.loads((out/'pilot/experiment_config.json').read_text())['source_provenance'],
       'cohorts':{str((baseline(s)/f'{split}_manifest.csv').relative_to(ROOT)):sha(baseline(s)/f'{split}_manifest.csv') for s in (42,43,44) for split in ('train','validation')},
       'seed_pairing':'N128 pool seed matches training seed42/43/44. Fixed dev128 seed42. Identical actual Python row draws and CPU Torch uniform-k draws to Gate2; no RNG consumed by groups/regularizer. Group shuffle fixed5701.',
       'objective':'L = original negative single-example logit_diff + gamma*scale*mean(centered_r^2); no penalty on h. H1/H2 gamma .001,.01; all32,491 edges.',
       'optimizer':'unchanged official Adam lr.05 eps1e-8 betas(.9,.999), float32, T.5 n_iters30, iso, uniform-k, one example/update',
       'phaseA':'six CPU tests, GPU references/gradcheck/mass,20 training-only gradient probes;30 hierarchical smoke plus original30/bypass30 and exact15+15 resumed30 checks (120 model optimizer updates). Fail correctness stops.',
       'phaseB':{'seed':42,'updates':1000,'evaluations':[500,1000],'candidates':[['H1',.001],['H1',.01],['H2',.001],['H2',.01]],
          'baseline':'fresh dedicated bypass1000 must bitwise match saved Gate2; quality uses saved Gate2; fresh control is timing/memory calibration',
          'criteria':{'CPR_relative_gain_min':.005,'accuracy_auc_drop_max':.02,'compact_accuracy_drop_max':.02,
          'CMD_guard':'positive paired CPR increase +.05','adaptation_time_ratio_max':1.10,'memory_ratio_max':1.10},
          'selection':'at most1 passing candidate, descending CPR gain then mean compact gain then smaller gamma then H1. None passes: stop, no controls/seeds beyond42.'},
       'phaseC':{'seeds':[42,43,44],'updates':1000,'controls':'only selected hierarchy gamma0, fixed size-preserving shuffle at selected gamma, ordinary edge mean-square at selected gamma; dedicated baseline calibrations43/44',
          'criteria':{'mean_paired_CPR_relative_gain_min':.02,'positive_seeds_min':2,'mean_accuracy_auc_drop_max':.02,'mean_compact_drop_max':.02,
            'CMD_guard':'mean increase <= max(0,mean absolute CPR increase)+.05','adaptation_time_ratio_max':1.10,'memory_ratio_max':1.10,
            'structural_claim':'selected real hierarchy must beat EACH gamma0/shuffle/edge-L2 control by>=.5% mean relative CPR and at least2 positive seeds, with mean compact/logAUC loss<=.02. All needed to advance; otherwise parameterization/regularization evidence only.'}},
       'phaseD':'only full C pass: exact-resume selected and all3 matched controls to5000 across3 seeds; evaluate500/1000/2500/5000, compare5000 to saved Gate2 and controls using same criteria. Restart only if necessary, charge repeated work. No full-data follow-up automatically.',
       'compact_fractions':[.005,.01,.02,.05],
       'diagnostics':'every50 updates plus first: task/reg losses, analytic penalty gradient and task gradient, read-only same-state Adam update comparison, h/residual/effective variance; log all task+regularized losses and exact raw/effective vectors. No extra model backward for diagnostics.',
       'compute':'synchronized monotonic stage/process clocks; all pilot/rejected/control/restarter work charged. Construction cost charged to adaptation. Fresh bypass calibration has same instrumentation. No physical-VRAM claims from allocator peaks.',
       'budget':{'optimizer_updates_max':70000,'child_launch_seconds_max':43200,'phaseB_updates':5000,'phaseC_additional_updates_max':13000,'phaseD_additional_updates_max':48000},
       'historical_criteria_sha256':CRITERIA_SHA,'no_leaderboard':True,'no_refinement':True,'no_expander':True,'no_interaction_training':True}
    assert sha(CRITERIA)==CRITERIA_SHA
    write_json(target,p);(out/'gate5c_protocol.sha256').write_text(sha(target)+'\n')
    write_json(out/'protocol_frozen_at.json',{'utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(target),'modified_CPR_seen':False})
    for name in p['source_sha256']:
        path=out/'source_at_execution'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    return p

def launch(out,family,gamma,seed,target=1000,shuffled=False,phase='train',scale=1.):
    dest=out/('pilot' if phase=='pilot' else 'phaseA_smoke') if phase!='train' else job(out,family,gamma,seed,shuffled)
    marker=dest/f'completed_{target}.json'
    if marker.exists():
        saved=json.loads(marker.read_text());assert saved['passed']
        assert saved['config_sha256']==sha(dest/'experiment_config.json')
        if saved['effective_sha256']:assert saved['effective_sha256']==sha(dest/f'effective_scores_{target}.pt')
        return
    dest.mkdir(parents=True,exist_ok=True)
    proto=out/('gate5c_preflight.json' if phase=='pilot' else 'gate5c_protocol.json')
    command=[sys.executable,'-u','-m','src.mattr_hierarchy.runner','--out',str(dest),'--protocol',str(proto),
       '--family',family,'--gamma',str(gamma),'--scale',str(scale),'--seed',str(seed),'--target',str(target),'--phase',phase]
    if shuffled:command.append('--shuffled')
    start=time.perf_counter();print('Launching',dest.name,phase,target,flush=True)
    with (dest/f'console_{target}.log').open('a',buffering=1) as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        code=process.wait()
    timing=out/'launch_timers.jsonl'
    with timing.open('a') as stream:stream.write(json.dumps({'job':dest.name,'phase':phase,'target':target,'seconds':time.perf_counter()-start,'returncode':code})+'\n')
    if code:raise RuntimeError(f'Gate5C job failed without setting changes: {dest}/failure.json')
    launch_seconds=sum(json.loads(line)['seconds'] for line in timing.read_text().splitlines())
    updates=0
    for summary in out.rglob('process_summary.json'):
        updates+=json.loads(summary.read_text())['counts']['optimizer_updates']
    assert launch_seconds<=43200 and updates<=70000,'Registered compute cap reached'

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--report-only',action='store_true')
    a=parser.parse_args();out=a.out.resolve()
    if a.report_only:configure();generate(out);return
    preflight=prepare(out)
    if a.prepare_only:return
    if (out/'completed.json').exists():configure();generate(out);return
    launch(out,'bypass',0,42,target=30,phase='pilot')
    p=freeze(out,preflight);scale=p['regularization_scale']['scale']
    launch(out,'H1',.01,42,target=30,phase='smoke',scale=scale)
    launch(out,'bypass',0,42,scale=scale)
    for family in ('H1','H2'):
        for gamma in (.001,.01):launch(out,family,gamma,42,scale=scale)
    selected=select_b(out)
    if selected:
        for seed in (43,44):
            launch(out,'bypass',0,seed,scale=scale)
            launch(out,selected['family'],selected['gamma'],seed,scale=scale)
        for family,gamma,shuffled in controls_for(selected):
            for seed in (42,43,44):launch(out,family,gamma,seed,shuffled=shuffled,scale=scale)
        if select_c(out,selected):
            for family,gamma,shuffled in [(selected['family'],selected['gamma'],False),*controls_for(selected)]:
                for seed in (42,43,44):launch(out,family,gamma,seed,target=5000,shuffled=shuffled,scale=scale)
            select_c(out,selected,step=5000)
    generate(out)
    for name,h in json.loads((out/'protected_previous_hashes.json').read_text()).items():assert sha(ROOT/name)==h,name
    write_json(out/'completed.json',{'passed':True,'protocol_sha256':sha(out/'gate5c_protocol.json'),
       'phaseB_selected':selected,'phaseC_executed':bool(selected),'phaseD_executed':(out/'phaseD_decision.json').exists(),
       'model_weight_updates':0,'leaderboard_submission':False})
    print('Gate5C finished; no later experiment dispatched.',flush=True)

if __name__=='__main__':main()
