"""One frozen public-test confirmation after all validation experiments finish."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch

from .runtime import (ROOT, OFFICIAL, FixedData, Meter, add_official_paths, provenance,
                      digest, write_json)
from .evaluate import model_and_graph, evaluate


def read(path):return json.loads(Path(path).read_text())


def lock_selection(out):
    """Choose only from validation means; test data has not been loaded here."""
    from .verify import verify
    from .report import generate
    # Every main-stage result is required before any test query is allowed.
    sources=[out/'smoke',out/'reference',out/'reference_common_validation']
    for n in (16,32,64,128):
        for seed in (42,43,44):
            sources += [out/f'scaling/n{n}_seed{seed}']
            sources += [out/f'baselines/{m}/n{n}_seed{seed}' for m in ('EAP','EAP-IG')]
            sources += [out/f'checkpoint_evaluations/n{n}_seed{seed}/step{s}' for s in (100,250,500)]
    if any(not (p/'complete.json').exists() for p in sources):
        raise RuntimeError('Complete all 75 main-stage jobs before locking public-test confirmation.')
    verify(out,require_complete=False)
    generate(out)
    chosen=read(out/'validation_selected_configuration.json')
    n=chosen['discovery_pool']
    jobs=[]
    for method in ('MAttr','EAP','EAP-IG'):
        for seed in (42,43,44):
            source=out/(f'scaling/n{n}_seed{seed}' if method=='MAttr' else f'baselines/{method}/n{n}_seed{seed}')
            jobs.append({'key':f'{method}_seed{seed}','method':method,'seed':seed,
                'source':str(source.relative_to(out)),
                'ranking_sha256':digest(source/'node_ranking.pt'),
                'validation_metric_sha256':digest(source/'official_evaluation.json')})
    policy=OFFICIAL/'deps/MIB-circuit-track/README.md'
    text=policy.read_text()
    if 'To evaluate on the train or (public) test set, use `--split train` / `--split test`.' not in text:
        raise RuntimeError('Reinspect benchmark permission and split name before public-test access.')
    spec={'discovery_pool':n,'mattr_optimization_steps':500,'seeds':[42,43,44],
        'selection':'highest mean MAttr validation CPR AUC over the three preset seeds; same pool for all methods',
        'validation_selection_sha256':digest(out/'validation_selected_configuration.json'),
        'test_split':'test','test_examples':128,'test_subset_seed':42,'evaluation_batch_size':20,
        'test_results_may_not_change_selection':True,'jobs':jobs,
        'permission_source':str(policy.relative_to(ROOT)),'permission_source_sha256':digest(policy),
        'confirmation_code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
            [Path(__file__),Path(__file__).with_name('evaluate.py'),Path(__file__).with_name('runner.py'),Path(__file__).with_name('runtime.py')]}}
    target=out/'confirmation_selection.json'
    if target.exists():
        if read(target)['selection']!=spec:
            raise ValueError('Frozen confirmation selection differs. Do not reselect using test outcomes.')
    else:
        write_json(target,{'selection':spec,'locked_before_test':True,
            'locked_at_utc':datetime.now(timezone.utc).isoformat()})
    return read(target)


def test_data(out,model,campaign,spec):
    from MIB_circuit_track.dataset import HFEAPDataset
    # The user authorized confirmation if permitted; the pinned README explicitly permits test.
    write_json(out/'test_access_started.json',{'split':'test','selection_sha256':digest(campaign/'confirmation_selection.json')})
    config=read(campaign/'experiment_config.json');config['public_test_accessed']=True
    write_json(campaign/'experiment_config.json',config)
    data=HFEAPDataset('mib-bench/ioi',model.tokenizer,split='test',task='ioi',model_name='gpt2')
    prior=set()
    for split in ('train','validation'):
        prior.update(pd.read_csv(campaign/f'reference/{split}_manifest.csv').clean_prompt)
    if set(data.dataset['prompt']) & prior:
        raise ValueError('Official public test clean prompts overlap the official train/validation splits.')
    full_fingerprint=data.dataset._fingerprint
    indices=FixedData.indices(len(data),spec['test_examples'],spec['test_subset_seed'])
    data.dataset=data.dataset.select(indices)
    rows=[]
    for i,index in enumerate(indices):
        clean,corrupt,labels=data[i];raw=data.dataset[i]
        names=[raw['metadata']['indirect_object'],raw['metadata']['subject']]
        ids=[model.tokenizer(' '+x,add_special_tokens=False).input_ids for x in names]
        if any(len(x)!=1 for x in ids) or [x[0] for x in ids]!=list(labels):
            raise ValueError('Public IOI label tokenization differs from the inspected official metric.')
        rows.append({'split':'test','subset_index':i,'official_filtered_index':index,
            'clean_prompt':clean,'corrupted_prompt':corrupt,'correct_token_id':labels[0],
            'incorrect_token_id':labels[1],'prompt_sha256':hashlib.sha256(clean.encode()).hexdigest()})
    frame=pd.DataFrame(rows)
    if frame.prompt_sha256.duplicated().any():raise ValueError('Duplicate selected test prompts.')
    frame.to_csv(out/'test_manifest.csv',index=False)
    write_json(out/'test_selection.json',{'split':'test','indices':indices,
        'seed':spec['test_subset_seed'],'full_official_fingerprint':full_fingerprint,
        'manifest_sha256':digest(out/'test_manifest.csv'),'train_validation_disjoint':True})
    return data


def run_job(campaign,key,out):
    if (out/'complete.json').exists():print('Already complete:',out,flush=True);return
    out.mkdir(parents=True,exist_ok=True)
    lock=read(campaign/'confirmation_selection.json');spec=lock['selection']
    if not lock['locked_before_test']:raise ValueError('Selection was not locked.')
    for relative,expected in spec['confirmation_code_sha256'].items():
        if digest(ROOT/relative)!=expected:
            raise ValueError(f'Confirmation code changed after the protocol was locked: {relative}')
    job=next(x for x in spec['jobs'] if x['key']==key)
    source=campaign/job['source']
    if digest(source/'node_ranking.pt')!=job['ranking_sha256']:
        raise ValueError('Ranking changed after validation selection.')
    if digest(source/'official_evaluation.json')!=job['validation_metric_sha256']:
        raise ValueError('Validation result changed after selection.')
    add_official_paths();info=provenance()
    random.seed(job['seed']);np.random.seed(job['seed']);torch.manual_seed(job['seed'])
    if read(source/'experiment_config.json')['provenance']!=info:
        raise ValueError('Confirmation software/model provenance differs.')
    write_json(out/'experiment_config.json',{'stage':'final_confirmation','method':job['method'],
        'model':'gpt2','task':'ioi','level':'node','seed':job['seed'],
        'discovery_pool':spec['discovery_pool'],'test_size':spec['test_examples'],
        'test_seed':spec['test_subset_seed'],'selection_sha256':digest(campaign/'confirmation_selection.json'),
        'source':job['source'],'ranking_sha256':job['ranking_sha256'],
        'public_test_accessed':True,'provenance':info})
    meter=Meter(out);meter.switch('confirmation_model_and_data_preparation');success=False
    try:
        model,_=model_and_graph(meter)
        from eap.graph import Graph
        graph=Graph.from_pt(str(source/'node_ranking.pt'))
        model.requires_grad_(False)
        data=test_data(out,model,campaign,spec)
        print(f'Frozen {key}: official public-test split; {len(data)} fixed examples; 10 circuit sizes; no attribution training.',flush=True)
        evaluate(model,graph,data,meter,out,'test',spec['evaluation_batch_size'])
        success=True
    except torch.cuda.OutOfMemoryError as error:
        write_json(out/'failure.json',{'oom':True,'error':str(error),'settings_unchanged':True})
        raise RuntimeError('CUDA OOM: stop and choose an explicit new confirmation protocol; do not tune from test scores.') from error
    finally:meter.close(success=success)
    write_json(out/'complete.json',{'success':True,'stage':'final_confirmation','method':job['method'],
        'seed':job['seed'],'test_examples':len(data),'public_test_accessed':True})


def run_campaign(out):
    from .campaign import launch
    lock=lock_selection(out)
    for job in lock['selection']['jobs']:
        launch(out/f"confirmation/{job['method']}/seed{job['seed']}",'confirm_wsl.sh',
            ['--campaign',out,'--job',job['key']])
    from .verify import verify
    from .report import generate
    verify(out,require_complete=True);generate(out)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--job');parser.add_argument('--out',type=Path)
    args=parser.parse_args();campaign=args.campaign.resolve()
    if args.job:
        if args.out is None:parser.error('--out is required for a confirmation job')
        run_job(campaign,args.job,args.out.resolve())
    else:run_campaign(campaign)


if __name__=='__main__':main()
