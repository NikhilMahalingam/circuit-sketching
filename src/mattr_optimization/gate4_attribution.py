"""Official EAP-IG inputs, canonical edge audit and training-only sign smoke test."""
import argparse
from functools import partial
import json
from pathlib import Path
import random
import time
from unittest.mock import patch
import pandas as pd
import torch
from .common import ROOT,MIB,configure,provenance,sha,git,write_json
from .gate3 import baseline
from .gate2 import NestedData
from .gate1 import LiveAccounting
from .gate4_init import edge_mapping,mask_diagnostics,rank_normalize


def attribution_sources():
    files=[(MIB,'run_attribution.py'),(MIB,'MIB_circuit_track/metrics.py'),
           (MIB/'EAP-IG','src/eap/attribute.py'),(MIB/'EAP-IG','src/eap/utils.py')]
    result={}
    for repo,name in files:
        p=repo/name
        if p.read_bytes().replace(b'\r\n',b'\n')!=git(repo,'show','HEAD:'+name).replace(b'\r\n',b'\n'):
            raise ValueError('Attribution source differs from pin: '+str(p))
        result[str(p.relative_to(ROOT))]=sha(p)
    return result


def run(out,seed,smoke=False):
    configure()
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from eap.attribute import attribute
    from eap.evaluate import evaluate_baseline,evaluate_graph
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.metrics import get_metric
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU fallback.')
    if (out/'completed.json').exists():raise ValueError('Attribution already complete; use cached artifact.')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'process_summary.json').exists():raise ValueError('Use a new explicit attempt directory after failure.')
    prov=provenance();sources=attribution_sources()
    expected=json.loads((baseline(seed)/'experiment_config.json').read_text())['source_provenance']
    if prov!=expected:raise ValueError('Runtime differs from paired Gate2 environment.')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    random.seed(seed);torch.manual_seed(seed)
    write_json(out/'experiment_config.json',{'seed':seed,'smoke':smoke,'pool':128,'used_examples':4 if smoke else 128,
      'ig_steps':5,'method':'EAP-IG-inputs','intervention':'patching','aggregation':'sum','batch_size':4,
      'metric':'official logit_diff mean=True loss=True','source_provenance':prov,'attribution_sources':sources,
      'alpha_grid':[0,.2,.4,.6,.8],'orientation':'signed score retained as-is; corrupted-minus-clean times negative-logit-difference gradient'})
    meter=LiveAccounting(out);success=False;old_backward=torch.Tensor.backward
    def backward(t,*a,**kw):
        answer=old_backward(t,*a,**kw);meter.counts['backward_calls']+=1;return answer
    try:
        with meter.stage('model_loading'):
            model=meter.attach(HookedTransformer.from_pretrained('gpt2-small'))
            if model.cfg.dtype!=torch.float32:raise ValueError('Pinned float32 model required.')
            model.eval();model.requires_grad_(False)
            model.cfg.use_attn_result=True;model.cfg.use_split_qkv_input=True;model.cfg.use_hook_mlp_in=True
            model.cfg.ungroup_grouped_query_attention=True
        with meter.stage('data_loading_train'):
            ds=HFEAPDataset('mib-bench/ioi',model.tokenizer,split='train',task='ioi',model_name='gpt2')
            NestedData(out,128,seed,128).select(ds,'train')
            pd.testing.assert_frame_equal(pd.read_csv(out/'train_manifest.csv'),pd.read_csv(baseline(seed)/'train_manifest.csv'))
            if smoke:ds.dataset=ds.dataset.select(range(4))
        graph=Graph.from_model(model)
        edge_mapping(Graph.from_model(model),graph).to_csv(out/'edge_mapping.csv',index=False)
        metric=get_metric('logit_diff','ioi',model.tokenizer,model)
        loader=ds.to_dataloader(4)
        print(f'Official EAP-IG: seed {seed}, examples={len(ds)}, ig_steps=5; expected {7*((len(ds)+3)//4)} forward batches and {5*((len(ds)+3)//4)} backwards.',flush=True)
        with meter.stage('eap_ig'),patch.object(torch.Tensor,'backward',backward):
            attribute(model,graph,loader,partial(metric,mean=True,loss=True),method='EAP-IG-inputs',
                      intervention='patching',ig_steps=5,quiet=True)
        raw=graph.scores[graph.real_edge_mask.bool()].detach().cpu().clone()
        if not torch.isfinite(raw).all():raise ValueError('Nonfinite EAP scores.')
        torch.save(raw,out/'eap_ig_scores.pt');torch.save(rank_normalize(raw),out/'rank_normalized.pt')
        mask_diagnostics(raw).to_csv(out/'initialization_statistics.csv',index=False)
        pd.DataFrame({'mattr_index':range(len(raw)),'signed_eap_score':raw.numpy(),'rank_normalized':rank_normalize(raw).numpy()}).to_csv(out/'attribution_scores.csv',index=False)
        if smoke:
            # Validate semantic orientation on training only: MIB exact removal loss is
            # loss(patched)-loss(clean), same direction as (corrupt-clean)*grad(loss).
            behavior=partial(metric,mean=False,loss=False)
            graph.in_graph[:]=graph.real_edge_mask
            with meter.stage('training_only_sign_sanity'),torch.no_grad():
                clean=float(evaluate_baseline(model,loader,behavior).mean())
                picks=torch.topk(raw,3).indices.tolist()+torch.topk(-raw,3).indices.tolist()
                mapping=pd.read_csv(out/'edge_mapping.csv');rows=[]
                for i in picks:
                    edge=graph.edges[mapping.iloc[i].edge];graph.in_graph[:]=graph.real_edge_mask
                    edge.in_graph=False
                    score=float(evaluate_graph(model,graph,loader,behavior,quiet=True).mean())
                    rows.append({'mattr_index':i,'edge':edge.name,'signed_eap':float(raw[i]),
                                 'exact_loss_increase':clean-score,'clean_margin':clean,'patched_margin':score})
                pd.DataFrame(rows).to_csv(out/'training_only_sign_sanity.csv',index=False)
            # Approximation can err per edge; source sign establishes orientation. Strong
            # aggregate contradiction among the six extrema triggers manual inspection.
            a=pd.DataFrame(rows)
            if float((a.signed_eap*a.exact_loss_increase).sum())<=0:
                raise ValueError('Training-only sign sanity contradicts source convention; stop, never flip signs.')
        success=True
    finally:
        meter.stop.set();meter.worker.join();meter.save();meter.stream.close()
        write_json(out/'process_summary.json',{'success':success,'counts':meter.counts,'seconds':time.perf_counter()-meter.started})
    write_json(out/'completed.json',{'passed':True,'sha256':{n:sha(out/n) for n in
      ('experiment_config.json','eap_ig_scores.pt','rank_normalized.pt','edge_mapping.csv','initialization_statistics.csv','train_manifest.csv')},'attribution_sources':sources})
    print(pd.read_csv(out/'compute_accounting.csv').to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=(42,43,44),default=42);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.out.resolve(),a.seed,a.smoke)
