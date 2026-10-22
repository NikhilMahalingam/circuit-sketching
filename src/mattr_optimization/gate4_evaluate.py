"""Frozen-score official evaluation only: EAP control, missing baseline or confirmation."""
import argparse
from functools import partial
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .common import configure,provenance,sha,write_json
from .gate3 import baseline
from .gate2 import NestedData,subset_indices
from .gate1 import LiveAccounting
from .gate4_train import evaluate_scores


def run(a):
    configure()
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.metrics import get_metric
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'completed.json').exists():raise ValueError('Completed frozen evaluation must be reused.')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required.')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    meter=LiveAccounting(a.out);success=False
    try:
        with meter.stage('model_loading'):
            model=meter.attach(HookedTransformer.from_pretrained('gpt2-small'));model.eval();model.requires_grad_(False)
            model.cfg.use_attn_result=True;model.cfg.use_split_qkv_input=True;model.cfg.use_hook_mlp_in=True
            model.cfg.ungroup_grouped_query_attention=True
        with meter.stage('data_loading_validation'):
            ds=HFEAPDataset('mib-bench/ioi',model.tokenizer,split='validation',task='ioi',model_name='gpt2')
            if a.cohort=='development':
                NestedData(a.out,128,42,128).select(ds,'validation')
                pd.testing.assert_frame_equal(pd.read_csv(a.out/'validation_manifest.csv'),pd.read_csv(baseline(42)/'validation_manifest.csv'))
            else:
                indices=subset_indices(len(ds),256,42)[128:]
                ds.dataset=ds.dataset.select(indices)
                rows=[{'official_filtered_index':i,'clean_prompt':ds[j][0],'corrupted_prompt':ds[j][1],
                       'correct_token':ds[j][2][0],'incorrect_token':ds[j][2][1]} for j,i in enumerate(indices)]
                frame=pd.DataFrame(rows);dev=pd.read_csv(baseline(42)/'validation_manifest.csv')
                if set(indices)&set(dev.official_filtered_index) or set(frame.clean_prompt)&set(dev.clean_prompt):
                    raise ValueError('Confirmation/development overlap.')
                for seed in (42,43,44):
                    train=pd.read_csv(baseline(seed)/'train_manifest.csv')
                    if set(frame.clean_prompt)&set(train.clean_prompt):raise ValueError('Confirmation/train overlap.')
                frame.to_csv(a.out/'validation_manifest.csv',index=False)
        state=torch.load(a.scores,map_location='cpu',weights_only=False)
        scores=state['scores'] if isinstance(state,dict) else state
        ident={'score_sha256':sha(a.scores),'manifest_sha256':sha(a.out/'validation_manifest.csv'),
               'cohort':a.cohort,'provenance':provenance()}
        metric=partial(get_metric('logit_diff','ioi',model.tokenizer,model),mean=False,loss=False)
        result=evaluate_scores(model,Graph.from_model(model),ds,metric,scores,a.out,a.step,meter,ident)
        write_json(a.out/'quality.json',{'seed':a.seed,'step':a.step,'cpr':result[1],'cmd':result[2],
                                       'accuracy_auc':result[6],'cohort':a.cohort})
        success=True
    finally:
        meter.stop.set();meter.worker.join();meter.save();meter.stream.close()
        write_json(a.out/'process_summary.json',{'success':success,'counts':meter.counts,'seconds':time.perf_counter()-meter.started})
    write_json(a.out/'completed.json',{'passed':True,'score_sha256':sha(a.scores),'quality_sha256':sha(a.out/'quality.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--scores',type=Path,required=True);p.add_argument('--step',type=int,required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--cohort',choices=('development','confirmation'),default='development')
    run(p.parse_args())
