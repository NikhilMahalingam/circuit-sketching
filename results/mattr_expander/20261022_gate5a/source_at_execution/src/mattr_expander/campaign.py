"""Bounded, preregistered Gate5A phases. No refinement, new sweeps or submission."""
import argparse
from datetime import datetime,timezone
import gzip
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
from src.mattr_optimization.common import ROOT,configure,sha,write_json,git
from src.mattr_optimization.gate3 import verify_criteria,CRITERIA_SHA,baseline
from .graphs import generate,audit,spectrum,permute

SEEDS=(42,43,44);BETAS=(.1,.25);FAMILIES=('G0','G1','G2')


def prepare_graphs(out):
    configure();out.mkdir(parents=True,exist_ok=True);folder=out/'graphs';folder.mkdir(exist_ok=True)
    registration={'vertices':32491,'degree':8,'seed':1701,'G0':'circular +/-1..4',
      'G1':'ring; E//4 successful uniform-edge-pair degree-preserving swaps, random endpoint orientation, reject loops/duplicates; max100*budget attempts; final connectivity assertion, no regeneration',
      'G2':'networkx.random_regular_graph(8,32491,seed=1701); sorted neighbor rows; final degree/connectivity audit',
      'permutation':'G2 independently relabeled by numpy.default_rng(2701).permutation(N); real score order never changes',
      'spectral':'G0 analytic all frequencies; others sparse eigsh LA3/SA1 tol1e-7 ncv40 maxiter6000; residual<=1e-5; no dense N-square matrices',
      'versions':{p:importlib.metadata.version(p) for p in ('networkx','numpy','scipy','torch','pandas')}}
    reg=out/'graph_generation_protocol.json'
    if reg.exists() and json.loads(reg.read_text())!=registration:raise ValueError('Graph algorithm registry differs.')
    if not reg.exists():
        write_json(reg,registration);write_json(out/'graph_algorithms_frozen_at.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':sha(reg),'modified_CPR_seen':False})
    if (out/'graphs_metadata.json').exists():
        rows=json.loads((out/'graphs_metadata.json').read_text())
        for r in rows:
            if sha(folder/f'{r["family"]}_neighbors.npy')!=r['file_sha256']:raise ValueError('Cached graph changed.')
        return rows
    from eap.graph import Graph
    from src.mattr_optimization.gate4_init import edge_mapping
    cfg={'n_layers':12,'n_heads':12,'d_model':768,'parallel_attn_mlp':False}
    mapping=edge_mapping(Graph.from_model(cfg),Graph.from_model(cfg))
    mapping.to_csv(out/'edge_mapping.csv',index=False)
    (out/'edge_mapping.csv.gz').write_bytes(gzip.compress((out/'edge_mapping.csv').read_bytes(),mtime=0))
    write_json(out/'mapping_audit.json',{'vertices':32491,'all_edges_bijective':True,
        'order':'row-major true entries of official Graph.real_edge_mask; negative logits column normalized',
        'csv_sha256':sha(out/'edge_mapping.csv'),'gzip_sha256':sha(out/'edge_mapping.csv.gz')})
    rows=[];graphs={}
    for family in FAMILIES:
        begin=time.perf_counter();neighbors,meta=generate(family);struct=audit(neighbors)
        path=folder/f'{family}_neighbors.npy';np.save(path,neighbors)
        construction=time.perf_counter()-begin
        spectral=spectrum(neighbors,family)
        row={**meta,**struct,**spectral,'required_construction_seconds':construction,'file_sha256':sha(path)}
        rows.append(row);graphs[family]=neighbors
        write_json(out/f'{family}_graph_metadata.json',row)
        print(f'{family}: gap={row["laplacian_gap"]:.9g}, |lambda_nontrivial|={row["nontrivial_adjacency_magnitude"]:.6g}, graph setup={construction:.3f}s; spectral={row["spectral_seconds"]:.3f}s',flush=True)
    begin=time.perf_counter();p=permute(graphs['G2']);np.save(folder/'G2perm_neighbors.npy',p)
    row={**rows[-1],**audit(p),'family':'G2perm','permutation_seed':2701,
         'required_construction_seconds':rows[-1]['required_construction_seconds']+time.perf_counter()-begin,
         'file_sha256':sha(folder/'G2perm_neighbors.npy'),'spectral_method':'exact relabeling of G2, spectrum identical'}
    rows.append(row);write_json(out/'graphs_metadata.json',rows);pd.DataFrame(rows).to_csv(out/'graph_spectral_statistics.csv',index=False)
    return rows


def freeze(out):
    verify_criteria();graphs=prepare_graphs(out)
    names=list((ROOT/'src/mattr_expander').glob('*.py'))
    names += [ROOT/'src/mattr_optimization'/n for n in ('checkpoint.py','common.py','gate0.py','gate1.py','gate2.py','gate3.py','gate4_init.py','gate4_train.py','gate4_evaluate.py')]
    names.append(ROOT/'tests/test_mattr_expander.py')
    protocol={'gate':'5A','parent_commit':git(ROOT,'rev-parse','HEAD').decode().strip(),'graph_seed':1701,
      'graph_registry_sha256':sha(out/'graph_generation_protocol.json'),'graph_files':{r['family']:r['file_sha256'] for r in graphs},
      'mapping_sha256':sha(out/'edge_mapping.csv'),'seeds':list(SEEDS),'betas':list(BETAS),'families':list(FAMILIES),
      'model':'gpt2-small','task':'official MIB IOI edge','pool':128,'edges':32491,
      'data':'exact Gate2 nested N128 train subset for each seed; identical fixed dev128 validation cohort',
      'development_manifest_sha256':sha(baseline(42)/'validation_manifest.csv'),
      'training':{'lr':.05,'optimizer':'official Adam unchanged','T':.5,'n_iters':30,'k_schedule':'uniform','mode':'iso','dtype':'float32','batch_size':20,'cf_cache_gb':4},
      'transformation':'B=(1-beta)I+beta P; P symmetric row-normalized degree8 adjacency; all32491 raw zero-init scores retained; original trainer.build_mask receives Bs; evaluator/export receives Bs; raw score/Adam/RNG saved for resume',
      'theory':'beta<.5 makes B positive definite and invertible for undirected regular P; reparameterization/conditioning, not enlarged hypothesis class',
      'scale_control':'s_effective=(1-beta)s; no neighbors; both registered betas run in PhaseB',
      'RNG':'local graph RNG1701 and optional permutation2701; no training RNG consumption by coupling; preserve paired Python row and CPU Torch uniform-k streams; restore Python/NumPy/CPU/CUDA checkpoint RNG',
      'phaseA':'unit/spectral/GPU checks then90 total model optimizer updates: official30, identity30, G2 beta.25 smoke30; official hard-mask patching checks at0/.1/.5/1; frozen validation ranking; fail correctness means stop',
      'phaseB':{'seed':42,'updates':1000,'jobs':'fresh zero calibration + six graph/beta variants + two scale controls',
        'rule':'eligible graph: >=.5% CPR gain vs paired Gate2_1000 OR efficiency-quality/compute gate vs5000; all dev compact/AUC loss<=.02, CMD<=positive CPR gain+.05, total adaptation overhead<=15%, peak<=10%; sort by CPR gain, then mean compact gain, then lower beta, then family name; at most2; no eligible means stop'},
      'phaseC':{'updates':1000,'candidates_max':2,'rule':'repeat selected candidates on43/44; for each selected beta also repeat G0/G1/G2/scale controls on43/44, so topology comparisons are fully paired; zero calibrations43/44; at most one winner by mean relative CPR gain then lower beta/family'},
      'criteria':{'primary_CPR_gain':.02,'positive_seeds':2,'compact_drop':.02,'accuracy_auc_drop':.02,
        'compact_fractions':[.005,.01,.02,.05],'CMD_guard':'max(0, mean paired CPR increase)+.05',
        'adaptation_time_ratio_max':1.15,'allocated_memory_ratio_max':1.10,
        'efficiency':{'CPR_loss_max':.01,'CMD_increase_max':.10,'accuracy_drop_max':.02,'accuracy_auc_drop_max':.02,'compute_saving_min':.20},
        'expander_specific':'G2 must beat G0, G1 AND scale-only by>=.5% mean paired relative CPR and>=2 positive seeds, with compact/AUC safeguards, at the same beta/budget; final claim additionally requires permutation control to pass baseline quality guards; one graph instance gives no causal spectral-gap claim'},
      'phaseD':'only after C primary/efficiency pass: continue one winning beta/configuration AND registered G0/G1/G2/scale controls at that beta across3 seeds to5000, evaluate500/1000/2500/5000; optional G2perm at1000 across3 seeds only when G2 is selected winner; no parameter reselection from permutation',
      'confirmation_generalization':'any development success remains exploratory; no generalization claim from reused dev cohort. A future disjoint validation/full-data study requires separate authorization.',
      'budget':{'optimizer_updates_max':78090,'child_wall_seconds_max':43200,'phaseB_updates_max':9000,'phaseC_additional_updates_max':18000,'phaseD_additional_updates_max':48000,'permutation_updates_max':3000},
      'timing':'CUDA synchronized perf_counter. Graph construction/validation charged to deployable adaptation, graph operation and same diagnostic cadence charged inside every training/control. Spectral diagnostics separate. All failed pilots/replays/evaluation/graph work charged in research costs. Partial500/2500 cost from measured checkpoint prefix plus job fixed setup; not separate process timing.',
      'diagnostics':'every50 updates plus first: raw/effective gradient norm, variance, top1%-score-edge gradient L1 share, successive effective rank Spearman and top1% Jaccard. All controls same cadence; finite gradients required.',
      'permutation_control':'only registered G2 relabeling2701; same real mapping; beta unchanged; no additional graph seeds',
      'historical_gate3_criteria_sha256':CRITERIA_SHA,'no_refinement':True,'no_leaderboard':True,'no_multi_model':True,
      'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in names}}
    path=out/'gate5a_protocol.json'
    if path.exists():
        old=json.loads(path.read_text());protocol['parent_commit']=old['parent_commit']
        if old!=protocol:raise ValueError('Frozen protocol/source changed; never rewrite it.')
    else:
        protected={}
        for top in ('results/mattr_optimization','results/acdc_prefix_cache','results/mib_sketch_diagnostics'):
            for f in (ROOT/top).rglob('*'):
                if f.is_file():protected[str(f.relative_to(ROOT))]=sha(f)
        for f in ROOT.glob('*REPORT.md'):protected[str(f.relative_to(ROOT))]=sha(f)
        protected['GATE2_FINAL_ANALYSIS.md']=sha(ROOT/'GATE2_FINAL_ANALYSIS.md')
        write_json(out/'protected_previous_hashes.json',protected)
        write_json(path,protocol);(out/'gate5a_protocol.sha256').write_text(sha(path)+'\n')
        write_json(out/'protocol_frozen_at.json',{'utc':datetime.now(timezone.utc).isoformat(),'sha256':sha(path),'modified_CPR_seen':False})
        snap=out/'source_at_execution'
        for p in names:
            dest=snap/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    return path


def launch(out,protocol,family,beta,seed,target,smoke=False):
    folder=out/('phaseA_smoke' if smoke else f'jobs/{family}_b{beta:g}_s{seed}');folder.mkdir(parents=True,exist_ok=True)
    marker=folder/f'completed_{target}.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if not saved['passed'] or any(sha(folder/n)!=h for n,h in saved['sha256'].items()):raise ValueError('Cached outputs changed.')
        print('Cached:',folder.name,target,'; zero model forwards.',flush=True);return
    protocol_data=json.loads(protocol.read_text());budget=protocol_data['budget']
    prior_updates=sum(json.loads(f.read_text())['counts']['optimizer_updates'] for f in out.rglob('process_summary.json'))
    previous=1000 if target==5000 and (folder/'completed_1000.json').exists() else 0
    planned=90 if smoke else target-previous
    elapsed=sum(json.loads(f.read_text())['seconds'] for f in out.rglob('launch_elapsed_*.json'))
    if prior_updates+planned>budget['optimizer_updates_max'] or elapsed>=budget['child_wall_seconds_max']:
        raise RuntimeError('Registered compute budget exhausted; no automatic expansion.')
    command=[sys.executable,'-u','-m','src.mattr_expander.runner','--out',str(folder),'--protocol',str(protocol),
             '--family',family,'--beta',str(beta),'--seed',str(seed),'--target',str(target)]
    if smoke:command.append('--smoke')
    start=time.perf_counter()
    with (folder/'console.log').open('a',buffering=1) as log:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in child.stdout:log.write(line);print(line,end='',flush=True)
        if child.wait()!=0:raise RuntimeError(f'Correctness/runtime failure in {folder}; stop without changing settings.')
    write_json(folder/f'launch_elapsed_{target}.json',{'seconds':time.perf_counter()-start,'argv':command})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--report-only',action='store_true')
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    from .analysis import report,phase_b,phase_c,final_assessment
    if a.report_only:report(out);return
    import fcntl
    with (out/'.campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol=freeze(out)
        if a.prepare_only:print('All graph algorithms and Gate5A criteria frozen, no modified CPR.');return
        if not (out/'phaseA_gpu_passed.json').exists():
            from .gpu_checks import run
            graphs={f:np.load(out/'graphs'/f'{f}_neighbors.npy') for f in FAMILIES}
            checks=run(graphs,out);write_json(out/'phaseA_gpu_passed.json',{'passed':True,'checks':checks})
        launch(out,protocol,'G2',.25,42,30,True)
        launch(out,protocol,'zero',0,42,1000)
        for family in FAMILIES+('scale',):
            for beta in BETAS:launch(out,protocol,family,beta,42,1000)
        selected=phase_b(out);report(out)
        winner=None
        if selected:
            betas=sorted(set(r['beta'] for r in selected))
            for seed in (43,44):
                launch(out,protocol,'zero',0,seed,1000)
                for beta in betas:
                    for family in FAMILIES+('scale',):launch(out,protocol,family,beta,seed,1000)
            winner=phase_c(out);report(out)
            if winner:
                if winner['family']=='G2':
                    for seed in SEEDS:launch(out,protocol,'G2perm',winner['beta'],seed,1000)
                for family in FAMILIES+('scale',):
                    for seed in SEEDS:launch(out,protocol,family,winner['beta'],seed,5000)
                for seed in SEEDS:
                    folder=out/f'baseline_2500_s{seed}';folder.mkdir(exist_ok=True)
                    if not (folder/'completed.json').exists():
                        command=[sys.executable,'-u','-m','src.mattr_optimization.gate4_evaluate','--out',str(folder),
                                 '--scores',str(baseline(seed)/'checkpoints/step_002500.pt'),'--step','2500','--seed',str(seed)]
                        start=time.perf_counter()
                        with (folder/'console.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                        write_json(folder/'launch_elapsed_2500.json',{'seconds':time.perf_counter()-start,'argv':command})
                final_assessment(out,winner);report(out)
        protected=json.loads((out/'protected_previous_hashes.json').read_text())
        if any(sha(ROOT/n)!=h for n,h in protected.items()):raise ValueError('Historical artifact changed.')
        verify_criteria()
        write_json(out/'completed.json',{'passed_execution':True,'phaseB_selected':selected,'phaseC_winner':winner,
          'historical_files_unchanged':len(protected),'protocol_sha256':sha(protocol),'refinement_run':False,'leaderboard_submission':False})
        report(out)


if __name__=='__main__':main()
