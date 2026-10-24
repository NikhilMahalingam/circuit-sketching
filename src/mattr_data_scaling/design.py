"""Data, historical compatibility inventory and immutable pre-training protocol."""
import ast
import inspect
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import ROOT, UPSTREAM, MIB, configure, provenance, sha, write_json, git
from src.mattr_optimization.gate3 import baseline
from src.mattr_optimization.gate4_init import edge_mapping
from src.mattr_compact_validation.design import exclusions, text_hash
from src.mattr_matched_size.design import select_cohort
from src.mattr_matched_size.topology import graph_new
from .core import H2,G5D,G5E,SEEDS,CAPS,matrix,job_key,digest

def historical_candidates():
    rows=[]
    for seed in SEEDS[:3]:
        for step in (1000,5000):
            rows.append(('baseline',128,seed,step,baseline(seed)/f'checkpoints/step_{step:06d}.pt'))
            rows.append(('h2',128,seed,step,H2/f'jobs/H2_g0_s{seed}/checkpoints/step_{step:06d}.pt'))
    gate1=ROOT/'results/mattr_optimization/20261021T134352Z_gate1'
    for step in (1000,5000):rows.append(('baseline',10000,42,step,gate1/f'checkpoints/step_{step:06d}.pt'))
    return rows

def streams(out):
    for seed in SEEDS:
        if seed<=44:
            state=torch.load(baseline(seed)/'checkpoints/step_000100.pt',map_location='cpu',weights_only=False)
            ks=[row[1] for row in state['train_log']]
            gen=torch.Generator();gen.set_state(state['rng']['torch'])
        else:
            gen=torch.Generator().manual_seed(seed);ks=[]
        ks += [1.+32490.*torch.rand(1,generator=gen).item() for _ in range(5000-len(ks))]
        if seed<=44:
            old=torch.load(baseline(seed)/'checkpoints/step_005000.pt',map_location='cpu',weights_only=False)
            assert ks==[r[1] for r in old['train_log']], 'Historical Torch stream has intervening RNG draws'
        for n in ((128,10000) if seed<=44 else (10000,)):
            rng=random.Random(seed)
            rows=[rng.randint(0,n-1) for _ in range(5000)]
            pd.DataFrame({'step':np.arange(1,5001),'row':rows,'k':ks}).to_csv(out/f'stream_n{n}_s{seed}.csv',index=False,float_format='%.17g')
    write_json(out/'stream_convention.json',{
        'example':'isolated random.Random(training_seed).randint(0,N-1), one draw per attempted update',
        'k':'isolated frozen float64 replay of official 1+(32491-1)*torch.rand(1).item(); zero initialization consumes no RNG',
        'seeds42_44':'first100 official recorded k values, then generator restored from step100 CPU state; all5000 values checked against historical baseline',
        'seeds45_46':'independent CPU torch.Generator.manual_seed(seed); official uniform distribution, separate from row and model RNG',
        'initialization':'exact deterministic zero; model RNG never consumes either isolated stream',
        'resume':'stream position equals completed updates; no data-length skips permitted (fail if present)'})

def official_rows(ds):
    rows=[]
    for i in range(len(ds)):
        clean,donor,labels=ds[i];raw=ds.dataset[i]
        ids=[ds.tokenizer(' '+raw['metadata'][k],add_special_tokens=False).input_ids for k in ('indirect_object','subject','random_c')]
        assert all(len(x)==1 for x in ids) and list(labels)==[x[0] for x in ids[:2]]
        rows.append(dict(official_filtered_index=i,clean_prompt=clean,corrupted_prompt=donor,
                         clean_sha256=text_hash(clean),corrupted_sha256=text_hash(donor),
                         correct_token=labels[0],incorrect_token=labels[1],
                         template_id=text_hash(raw['template']),official_template=raw['template']))
    return pd.DataFrame(rows)

def prepare_data(out):
    from transformers import AutoTokenizer
    from MIB_circuit_track.dataset import HFEAPDataset
    tok=AutoTokenizer.from_pretrained('gpt2')
    ds={s:HFEAPDataset('mib-bench/ioi',tok,split=s,task='ioi',model_name='gpt2') for s in ('train','validation')}
    assert len(ds['train'])==10000
    train=official_rows(ds['train']);val=official_rows(ds['validation'])
    train.to_csv(out/'train_n10000.csv',index=False)
    for seed in SEEDS[:3]:
        old=pd.read_csv(baseline(seed)/'train_manifest.csv')
        selected=train.iloc[old.official_filtered_index.astype(int)].reset_index(drop=True)
        assert selected.clean_prompt.tolist()==old.clean_prompt.tolist() and selected.corrupted_prompt.tolist()==old.corrupted_prompt.tolist()
        selected.to_csv(out/f'train_n128_s{seed}.csv',index=False)
    hashes,indices,records=exclusions()
    for root in (G5D,G5E):
        path=root/'cohort_manifest.csv';frame=pd.read_csv(path,keep_default_na=False)
        hashes.update(frame.clean_sha256);hashes.update(frame.corrupted_sha256)
        indices.update(frame.official_filtered_index.astype(int))
        records.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),role='excluded_Gate5D_or_E'))
    hashes.update(train.clean_sha256);hashes.update(train.corrupted_sha256)
    chosen,eligible,quotas=select_cohort(val,hashes,indices,size=512,seed=5544)
    assert len(chosen)==512, 'Gate5F requires the registered512 prompts; stop rather than change cohort size'
    chosen.insert(0,'example',range(len(chosen)));chosen.to_csv(out/'cohort_primary.csv',index=False)
    dev=pd.read_csv(G5E/'cohort_manifest.csv',keep_default_na=False)
    pd.testing.assert_frame_equal(val.iloc[dev.official_filtered_index].reset_index(drop=True)[['clean_prompt','corrupted_prompt']],dev[['clean_prompt','corrupted_prompt']])
    dev.to_csv(out/'cohort_development.csv',index=False)
    for name,frame in [('primary',chosen),('development',dev)]:
        frame.drop(columns=['clean_prompt','corrupted_prompt','official_template']).to_csv(out/f'cohort_{name}_hashes.csv',index=False)
    write_json(out/'cohort_audit.json',dict(passed=True,seed=5544,selected=len(chosen),eligible=len(eligible),quotas=quotas,
        exclusions=records,clean_donor_crossrole_overlap=0,labels_single_token=True,Gate1_full_validation_exposure=True,
        fingerprints={s:d.dataset._fingerprint for s,d in ds.items()},full_train_sha256=sha(out/'train_n10000.csv')))

def known_heads(out):
    source=ROOT/'external/Automatic-Circuit-Discovery/acdc/ioi/utils.py'
    tree=ast.parse(source.read_text());value=None
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='IOI_CIRCUIT' for t in node.targets):value=ast.literal_eval(node.value)
    assert value and sum(map(len,value.values()))==26
    write_json(out/'known_ioi_heads.json',dict(classes=value,paper='https://arxiv.org/html/2211.00593v1',
        reference='https://github.com/ArthurConmy/Automatic-Circuit-Discovery/blob/bc99ace817974b5584b7ee203d596a8e2bbcd399/acdc/ioi/utils.py#L161-L197',
        reference_commit=git(source.parents[2],'rev-parse','HEAD').decode().strip(),source_sha256=sha(source),
        convention='Published expanded26-head circuit, including parenthesized backup induction/duplicate heads; zero-based layer/head; overlap descriptive only'))

def inventory(out):
    records=[];reuses={}
    pins=provenance()
    for method,n,seed,step,path in historical_candidates():
        folder=path.parent.parent;row=dict(method=method,gamma=0,discovery_pool=n,training_seed=seed,
            subset_seed=seed if n==128 else 'fixed_full_pool',training_step=step,checkpoint=str(path.relative_to(ROOT)),
            exact_reuse_valid=False,reason='checkpoint absent')
        if path.exists():
            state=torch.load(path,map_location='cpu',weights_only=False)
            cfg=json.loads((folder/'experiment_config.json').read_text())
            table=pd.read_csv(out/f'stream_n{n}_s{seed}.csv',float_precision='round_trip')
            old=pd.read_csv(folder/('training_example_draws.csv' if method=='baseline' else 'example_draws_1000.csv'))
            rowcol='subset_index' if 'subset_index' in old else 'official_filtered_index'
            # Full Gate1 records canonical indices; N128 records subset-relative indices.
            draws=old[rowcol].astype(int).tolist()
            manifest=pd.read_csv(folder/'train_manifest.csv')
            current=pd.read_csv(out/('train_n10000.csv' if n==10000 else f'train_n128_s{seed}.csv'))
            data_ok=all(manifest[c].tolist()==current[c].tolist() for c in ('official_filtered_index','clean_prompt','corrupted_prompt','correct_token','incorrect_token'))
            stream_ok=draws[:step]==table.row.iloc[:step].tolist() and [x[1] for x in state['train_log']]==table.k.iloc[:step].tolist()
            source_ok=cfg['source_provenance']==pins
            audit_path=folder/'model_weight_audit.json'
            if audit_path.exists():
                model_ok=json.loads(audit_path.read_text())['passed'];weight_evidence='historical before/after weight-hash audit'
            elif (folder/'completed.json').exists():
                completion=json.loads((folder/'completed.json').read_text());model_ok=completion['passed']
                for name,h in completion['sha256'].items():assert sha(folder/name)==h
                groups=state['optimizer']['param_groups']
                model_ok=model_ok and len(groups)==1 and len(groups[0]['params'])==1 and state['scores'].numel()==32491
                weight_evidence='completed artifact hashes and pinned score-only Adam/source audit; historical Gate2 has no before/after model-weight hash'
            else:
                model_ok=bool((folder/'final_validation.json').exists());weight_evidence='historical full-data final validation audit'
            scores=state['scores'];effective=scores
            if method=='h2':
                from src.mattr_hierarchy.adapter import Hierarchy
                effective=Hierarchy(np.load(out/'H2_groups.npy')).transform(scores)
            row.update(dataset_manifest_sha256=sha(folder/'train_manifest.csv'),source_dependency_commits=json.dumps(cfg['source_provenance']['commits'],sort_keys=True),
                checkpoint_sha256=sha(path),effective_score_tensor_sha256=hashlib_tensor(effective),optimizer_available='optimizer' in state,
                rng_available='rng' in state,manifest_matches=data_ok,streams_match=stream_ok,source_matches=source_ok,model_unchanged=model_ok,
                model_weight_evidence=weight_evidence,exact_reuse_valid=bool(data_ok and stream_ok and source_ok and model_ok and state['completed']==step and 'optimizer' in state and 'rng' in state),
                reason='verified data/source/step/row/k/optimizer/RNG/model' if data_ok and stream_ok and source_ok and model_ok else 'data/source/stream/model incompatibility; replacement required')
            exported=folder/f'effective_scores_{step}.pt'
            row['effective_score_file_sha256']=sha(exported) if exported.exists() else None
            if row['exact_reuse_valid']:reuses[method,n,seed,step]=str(path.relative_to(ROOT))
        records.append(row)
    # Gates5D/E reference the same15 frozen Gate5C checkpoints, rather than training new ones.
    previous=json.loads((G5D/'gate5d_protocol.json').read_text())
    references=[]
    for rec in previous['rankings']:
        path=ROOT/rec['checkpoint'];assert sha(path)==rec['checkpoint_sha256']
        assert sha(ROOT/rec['scores'])==rec['scores_sha256']
        references.append({**rec,'Gate5E_reference':True,'registered_Gate5F_method':rec['method'] in ('baseline','gamma_zero')})
        if rec['method'] not in ('baseline','gamma_zero'):
            records.append(dict(method=rec['method'],gamma=rec['gamma'],discovery_pool=128,training_seed=rec['seed'],subset_seed=rec['seed'],
                training_step=1000,checkpoint=rec['checkpoint'],checkpoint_sha256=rec['checkpoint_sha256'],
                effective_score_sha256=rec['scores_sha256'],dataset_manifest_sha256=rec['training_manifest_sha256'],
                source_dependency_commits=json.dumps(rec['source_provenance']['commits'],sort_keys=True),
                optimizer_available=True,rng_available=True,exact_reuse_valid=False,reason='method/gamma outside registered baseline vs unregularized H2 design'))
    write_json(out/'historical_ranking_references.json',references)
    pd.DataFrame(records).to_csv(out/'checkpoint_inventory.csv',index=False)
    jobs=[]
    for n in (128,10000):
        for seed in (SEEDS if n==10000 else SEEDS[:3]):
            for method in ('baseline','h2'):
                step=5000 if (method,n,seed,5000) in reuses else 1000 if (method,n,seed,1000) in reuses else 0
                jobs.append(dict(key=job_key(method,n,seed),method=method,dataset_size=n,seed=seed,
                    reuse_step=step,reuse_checkpoint=reuses.get((method,n,seed,step)),
                    endpoints={str(s):reuses.get((method,n,seed,s)) for s in (1000,5000)},requires_training=step<5000))
    write_json(out/'replacement_job_manifest.json',jobs);return jobs

def hashlib_tensor(x):
    import hashlib
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def cpu_preflight(out):
    configure();out.mkdir(parents=True,exist_ok=True)
    if (out/'gate5f_protocol.json').exists():raise ValueError('Already frozen; use audit, never rewrite preflight inputs')
    start=time.perf_counter()
    # Read complete historical narrative/protocol/metadata; save exact evidence hashes.
    read={}
    for pattern in ('GATE1*REPORT.md','GATE2_FINAL_ANALYSIS.md','GATE5C*REPORT.md','GATE5D*REPORT.md','GATE5E*REPORT.md'):
        for path in ROOT.glob(pattern):path.read_text();read[str(path.relative_to(ROOT))]=sha(path)
    for root in (H2,G5D,G5E):
        for path in root.glob('*protocol*.json'):json.loads(path.read_text());read[str(path.relative_to(ROOT))]=sha(path)
    write_json(out/'historical_evidence.json',read)
    from src.mattr_hierarchy.groups import construct
    graph=graph_new();construct(graph,out)
    assert sha(out/'H2_groups.npy')==sha(H2/'H2_groups.npy')
    pd.testing.assert_frame_equal(pd.read_csv(out/'edge_mapping.csv'),pd.read_csv(H2/'edge_mapping.csv'))
    streams(out);prepare_data(out);known_heads(out);jobs=inventory(out)
    from .checks import schedule_audit
    schedule_audit(out)
    # Historical results remain protected, including intentionally untracked checkpoints.
    protected={}
    for name in git(ROOT,'ls-files','-z').decode().split('\0'):
        if name and not name.startswith(('src/mattr_data_scaling/','scripts/gate5f/')) and name not in ('.gitattributes','GATE5F_RUNBOOK.md') and (ROOT/name).is_file():protected[name]=sha(ROOT/name)
    for method,n,seed,step,path in historical_candidates():
        if path.exists():protected[str(path.relative_to(ROOT))]=sha(path)
    write_json(out/'protected_previous_hashes.json',protected)
    write_json(out/'preflight_cpu.json',dict(passed=True,seconds=time.perf_counter()-start,
        missing_training_jobs=sum(j['requires_training'] for j in jobs),registered_endpoints=len(matrix()),
        source_provenance=provenance(),group_sha256=sha(out/'H2_groups.npy')))

def freeze(out):
    if (out/'gate5f_protocol.json').exists():
        from .core import protocol
        return protocol(out)
    for name in ('preflight_cpu.json','preflight_gpu.json','schedule_audit.json'):
        assert json.loads((out/name).read_text())['passed'],name
    jobs=json.loads((out/'replacement_job_manifest.json').read_text())
    files=[p for p in (ROOT/'src/mattr_data_scaling').glob('*.py')]+list((ROOT/'scripts/gate5f').glob('*.sh'))
    files += [ROOT/'configs/gate5f.json',ROOT/'tests/test_mattr_data_scaling.py']
    # Include every reused integration adapter in the source freeze.
    for dirname in ('mattr_hierarchy','mattr_matched_size','mattr_compact_validation','mattr_optimization'):
        files+=list((ROOT/'src'/dirname).glob('*.py'))
    artifact_names=[p for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.npy','.md')]
    proto=dict(gate='5F',parent_commit=git(ROOT,'rev-parse','HEAD').decode().strip(),endpoints=matrix(),training_jobs=jobs,
        eap_jobs=[dict(key=f'eap_ig_n128_s{s}',dataset_size=128,seed=s) for s in SEEDS[:3]]+[dict(key='eap_ig_n10000',dataset_size=10000,seed=42)],
        primary_seeds=SEEDS,factorial_seeds=SEEDS[:3],caps=CAPS,primary_cap=116,batch_size=20,precision='float32',
        training=dict(optimizer='official Adam',lr=.05,T=.5,n_iters=30,mode='iso',gamma=0,k_schedule='uniform',cf_cache_gb=4.,steps=5000),
        evaluation=dict(metric='official MIB CPR/CMD/log-AUC; positive-margin and full-vocabulary top1 separately',signed=True,prune=True,cohorts=['primary','development'],classification_cohort='primary',prefix_search='all original prefixes until first actual count>640; require monotone/nested; caps use largest eligible prefix'),
        bootstrap=dict(replicates=5000,seed=5644,paired=True,stratify='exact template_id',seeds_fixed=True),
        criteria=dict(meaningful=.10,equivalence=.02,safeguard_caps=[296,400,640],safeguard_regression_max=.02,positive_primary_seeds=3,
            precedence=['Robust compact advantage','Small compact advantage','Baseline catches up and data-limited','Baseline catches up','Data-limited advantage','Mixed'],
            catches_up='delta128_1000>0; step_effect128<0 and step_effect10000<0; baseline accuracy increases with steps at both N; primary<=0 or entire95%CI within[-.02,.02]',
            data_limited='delta128 positive at both budgets; data_effect1000<0 and data_effect5000<0. At both matched durations reduction remains, operationalizing not fully explained by longer optimization.',
            invalid_correctness='Invalid correctness; no positive claim'),
        eap=dict(method='EAP-IG-inputs',ig_steps=5,alpha_grid=[0,.2,.4,.6,.8],batch_size=4,metric='official mean=True loss=True',
            partition='fixed batch4; divisible sizes128/10000; chunked batch-aligned outputs weight by examples; do not change batch size or mean reduction'),
        topology_control=dict(seed=5801,replicates=64,method='independent within-layer attention-head relabeling automorphisms; Q/K/V coupled; input/MLP/logits fixed',
            constraints='exact actual counts and source/destination type counts under official prune; conditions on layer/head-count structure; no enrichment claim about layer selection'),
        runtime_limit_seconds=43200,manual_override='--approve-over-12h; checked before each costly command; approval saved with protocol hash',
        source_provenance=provenance(),source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        artifact_sha256={p.name:sha(p) for p in artifact_names},new_model_weight_training=False,leaderboard=False)
    assert len(proto['endpoints'])==28
    commands=[]
    for j in jobs:commands.append(dict(stage='train',args=['--dataset-size',str(j['dataset_size']),'--method',j['method'],'--seed',str(j['seed']),'--resume']))
    for j in proto['eap_jobs']:commands.append(dict(stage='eap',args=['--dataset-size',str(j['dataset_size']),'--seed',str(j['seed'])]))
    for cohort in ('primary','development'):
        for e in matrix():commands.append(dict(stage='evaluate',args=['--method',e['method'],'--dataset-size',str(e['dataset_size']),'--seed',str(e['seed']),'--step',str(e['step']),'--cohort',cohort]))
        for j in proto['eap_jobs']:commands.append(dict(stage='evaluate',args=['--method','eap_ig','--dataset-size',str(j['dataset_size']),'--seed',str(j['seed']),'--step','0','--cohort',cohort]))
    proto['commands']=commands
    write_json(out/'gate5f_protocol.json',proto);(out/'gate5f_protocol.sha256').write_text(sha(out/'gate5f_protocol.json')+'\n')
    write_json(out/'command_manifest.json',dict(protocol_sha256=sha(out/'gate5f_protocol.json'),commands=commands))
    (out/'command_manifest.sha256').write_text(sha(out/'command_manifest.json')+'\n')
    write_json(out/'protocol_frozen_at.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        protocol_sha256=sha(out/'gate5f_protocol.json'),registered_training_started=False,registered_evaluation_started=False))
    print(f'Frozen protocol {sha(out/"gate5f_protocol.json")};28 endpoints; missing training jobs={sum(j["requires_training"] for j in jobs)}',flush=True)
    return proto
