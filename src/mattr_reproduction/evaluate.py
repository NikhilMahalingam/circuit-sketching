"""Official node baselines and reevaluation of saved rankings/checkpoints."""
import argparse
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from .runtime import (ROOT, FixedData, Meter, add_official_paths, provenance,
                      write_json, export_ranking, digest)
from .runner import install_evaluation_capture


def model_and_graph(meter):
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    model = HookedTransformer.from_pretrained('gpt2-small')
    model.eval()
    model.cfg.use_attn_result = True; model.cfg.use_split_qkv_input = True
    model.cfg.use_hook_mlp_in = True; model.cfg.ungroup_grouped_query_attention = True
    meter.attach(model, 'transformerlens')
    return model, Graph.from_model(model, node_scores=True)


def dataset(selection, model, split):
    from MIB_circuit_track.dataset import HFEAPDataset
    return selection.select(HFEAPDataset('mib-bench/ioi', model.tokenizer,
                            split=split, task='ioi', model_name='gpt2'), split)


def evaluate(model, graph, data, meter, out, split, batch_size):
    import MIB_circuit_track.evaluation as official
    from MIB_circuit_track.metrics import get_metric
    metric = partial(get_metric('logit_diff', 'ioi', model.tokenizer, model), mean=False, loss=False)
    with ExitStack() as stack:
        install_evaluation_capture(stack, meter, out, split=split)
        return official.evaluate_area_under_curve(model, graph, data.to_dataloader(batch_size), metric,
            level='node', absolute=False, intervention='patching', quiet=True)


def map_mattr_scores(graph, scores, model):
    """The official node-vector layout: input, layer-major heads, then MLPs."""
    if scores.numel() != 1 + model.cfg.n_layers*(model.cfg.n_heads+1):
        raise ValueError('Unexpected official MAttr node-score layout.')
    mapped = torch.full((graph.n_forward,), float('nan'))
    for name, node in graph.nodes.items():
        if name == 'logits':
            continue
        if name == 'input':
            value = scores[0]
        elif name.startswith('a'):
            value = scores[1+node.layer*model.cfg.n_heads+node.head]
        elif name.startswith('m'):
            value = scores[1+model.cfg.n_layers*model.cfg.n_heads+node.layer]
        else:
            raise ValueError(name)
        mapped[graph.forward_index(node, attn_slice=False)] = value
    graph.nodes_scores = mapped
    return graph


def run_baseline(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        print('Already complete:',out);return
    add_official_paths();info=provenance()
    seed=args.seed;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    config={'method':args.method,'model':'gpt2','task':'ioi','level':'node',
        'train_size':args.train_size,'validation_size':args.validation_size,'seed':seed,
        'validation_seed':42,'ig_steps':args.ig_steps,'batch_size':args.batch_size,
        'attribution_batch_size':args.attribution_batch_size,
        'attribution_batch_reason':'Uniform per-example weighting with the official mean=True loss; avoid partial-batch weighting differences.',
        'intervention':'patching','absolute_ranking':False,'provenance':info,
        'equal_data_is_not_equal_compute':True,'public_test_accessed':False}
    if (out/'experiment_config.json').exists():
        if json.loads((out/'experiment_config.json').read_text()) != config:
            raise ValueError('Baseline configuration changed; use a new directory.')
    write_json(out/'experiment_config.json',config)
    meter=Meter(out);meter.switch('model_and_data_preparation');success=False
    try:
        model,graph=model_and_graph(meter)
        selection=FixedData(out,args.train_size,seed,args.validation_size)
        train=dataset(selection,model,'train');validation=dataset(selection,model,'validation')
        from eap.attribute_node import attribute_node
        from MIB_circuit_track.metrics import get_metric
        metric=partial(get_metric('logit_diff','ioi',model.tokenizer,model),mean=True,loss=True)
        official_method='EAP' if args.method=='EAP' else 'EAP-IG-inputs'
        print(f'{args.method}: {len(train)} unique discovery examples; official {official_method}, IG steps={args.ig_steps}; attribution batch={args.attribution_batch_size}; evaluation batch={args.batch_size}; node evaluation grid=10',flush=True)
        meter.switch('attribution_construction')
        model.requires_grad_(True)
        with meter.backwards():
            attribute_node(model,graph,train.to_dataloader(args.attribution_batch_size),metric,
                method=official_method,intervention='patching',ig_steps=args.ig_steps,quiet=True)
        meter.switch('ranking_construction')
        export_ranking(graph,out,'node_ranking')
        model.zero_grad(set_to_none=True);model.requires_grad_(False)
        evaluate(model,graph,train,meter,out,'train',args.batch_size)
        evaluate(model,graph,validation,meter,out,'validation',args.batch_size)
        success=True
    except torch.cuda.OutOfMemoryError as error:
        write_json(out/'failure.json',{'oom':True,'error':str(error),'settings_unchanged':True})
        raise RuntimeError('CUDA OOM: choose an explicitly smaller batch size in a new configuration.') from error
    finally:
        meter.close(success=success)
    write_json(out/'complete.json',{'success':True,'method':args.method,
        'eligible_unique_train_examples':len(train),'sampled_unique_train_examples':len(train),
        'total_attribution_example_presentations':len(train),
        'backward_example_presentations':len(train)*(1 if args.method=='EAP' else args.ig_steps),
        'validation_examples':len(validation),'public_test_accessed':False})


def run_reevaluation(args):
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        print('Already complete:',out);return
    add_official_paths();info=provenance()
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    source_config=json.loads((args.source/'experiment_config.json').read_text())
    write_json(out/'experiment_config.json',{'method':'MAttr','stage':'ranking_reevaluation',
        'source':str(args.source),'checkpoint':str(args.checkpoint),
        'step':checkpoint['completed_steps'],'train_size':source_config['train_size'],
        'seed':source_config['seed'],'validation_size':args.validation_size,
        'validation_seed':42,'provenance':info,
        'reuse_final_validation':args.reuse_final_validation})
    meter=Meter(out);meter.switch('reevaluation_model_and_data_preparation');success=False
    try:
        model,graph=model_and_graph(meter)
        map_mattr_scores(graph,checkpoint['scores'],model)
        if checkpoint['completed_steps']==500 and (args.source/'node_ranking.pt').exists():
            from eap.graph import Graph
            original=Graph.from_pt(str(args.source/'node_ranking.pt'))
            torch.testing.assert_close(graph.nodes_scores,original.nodes_scores,atol=0,rtol=0,equal_nan=True)
        selection=FixedData(out,source_config['train_size'],source_config['seed'],args.validation_size)
        train=dataset(selection,model,'train');validation=dataset(selection,model,'validation')
        model.requires_grad_(False)
        if args.include_train:
            evaluate(model,graph,train,meter,out,'train',args.batch_size)
        if args.reuse_final_validation:
            if checkpoint['completed_steps']!=500 or not (args.source/'complete.json').exists():
                raise ValueError('Validation reuse requires a completed final 500-step source run.')
            if digest(out/'validation_manifest.csv')!=digest(args.source/'validation_manifest.csv'):
                raise ValueError('Cached validation cohort differs.')
            if args.batch_size!=source_config['evaluation_batch_size']:
                raise ValueError('Cached evaluation batch size differs.')
            if source_config['provenance']!=info:
                raise ValueError('Cached evaluator/model software provenance differs.')
            meter.switch('validation_artifact_cache_reuse')
            for name in ['validation_curve.csv','validation_per_example.npz','official_evaluation.json']:
                shutil.copyfile(args.source/name,out/name)
            write_json(out/'evaluation_cache.json',{'hits':1,'source':str(args.source),
                'guard':'identical official node scores, validation manifest, batch size and evaluator provenance',
                'additional_model_forwards':0})
        else:
            evaluate(model,graph,validation,meter,out,'validation',args.batch_size)
        success=True
    finally:
        meter.close(success=success)
    write_json(out/'complete.json',{'success':True,'step':checkpoint['completed_steps'],
        'train_examples':len(train),'validation_examples':len(validation),'public_test_accessed':False})


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest='action',required=True)
    baseline=sub.add_parser('baseline')
    baseline.add_argument('--method',choices=['EAP','EAP-IG'],required=True)
    baseline.add_argument('--train-size',type=int,required=True)
    baseline.add_argument('--seed',type=int,default=42)
    baseline.add_argument('--ig-steps',type=int,default=5)
    baseline.add_argument('--attribution-batch-size',type=int,default=1)
    reevaluate=sub.add_parser('reevaluate')
    reevaluate.add_argument('--source',type=Path,required=True)
    reevaluate.add_argument('--checkpoint',type=Path,required=True)
    reevaluate.add_argument('--include-train',action='store_true')
    reevaluate.add_argument('--reuse-final-validation',action='store_true')
    for p in [baseline,reevaluate]:
        p.add_argument('--out',type=Path,required=True)
        p.add_argument('--validation-size',type=int,default=128)
        p.add_argument('--batch-size',type=int,default=20)
    args=parser.parse_args()
    (run_baseline if args.action=='baseline' else run_reevaluation)(args)


if __name__=='__main__':
    main()
