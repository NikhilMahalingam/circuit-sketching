"""Evaluate existing score tensors only, with the official MIB evaluator.

Gate 1 is not retrained. Optional train-checkpoint evaluations fill the missing
500/1000-step generalization gaps; no validation feedback reaches an optimizer.
"""
import argparse
import json
from pathlib import Path
from functools import partial
from unittest.mock import patch
import time

from .common import ROOT, GRID, configure, provenance, sha, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--campaigns', nargs='+', type=Path, required=True)
    p.add_argument('--gate1', type=Path, required=True)
    a = p.parse_args()
    a.out = a.out.resolve()
    a.gate1 = a.gate1.resolve()
    a.campaigns = [path.resolve() for path in a.campaigns]
    configure()
    import pandas as pd
    import torch
    from transformer_lens import HookedTransformer
    from eap.graph import Graph
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.metrics import get_metric
    import MIB_circuit_track.evaluation as evaluation
    from .gate1 import LiveAccounting
    from .gate2 import NestedData, archive_attempt
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no fallback or settings changes.')
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prov = provenance()
    source_hashes = {n: sha(a.gate1/n) for n in ('final_scores.pt','final_validation.json','experiment_config.json')}
    if not json.loads((a.gate1/'final_validation.json').read_text())['passed']:
        raise ValueError('Gate 1 must have a passing audit.')
    archive_attempt(a.out)
    meter = LiveAccounting(a.out)
    success = False
    try:
        with meter.stage('model_loading'):
            model = meter.attach(HookedTransformer.from_pretrained('gpt2-small'))
            if model.cfg.dtype != torch.float32:
                raise ValueError('Frozen reference evaluation must preserve the official float32 model.')
            model.cfg.use_split_qkv_input = True
            model.cfg.use_attn_result = True
            model.cfg.use_hook_mlp_in = True
            model.cfg.ungroup_grouped_query_attention = True
            model.eval()
        graph = Graph.from_model(model)
        if int(graph.real_edge_mask.sum()) != 32491:
            raise ValueError('Unexpected edge graph.')
        metric = partial(get_metric('logit_diff', 'ioi', model.tokenizer, model), mean=False, loss=False)
        expected = next(a.campaigns[0].glob('seed*_n*/validation_manifest.csv'))
        val_dir = a.out/'fixed_validation'
        val_dir.mkdir(exist_ok=True)
        with meter.stage('data_loading_validation'):
            val = HFEAPDataset('mib-bench/ioi', model.tokenizer, split='validation', task='ioi', model_name='gpt2')
            NestedData(val_dir, 128, 42, 128).select(val, 'validation')
        pd.testing.assert_frame_equal(pd.read_csv(expected), pd.read_csv(val_dir/'validation_manifest.csv'))
        records = []

        def evaluate(label, scores_path, ds, manifest, metadata):
            folder = a.out/label
            folder.mkdir(exist_ok=True)
            identity = {'score_sha256':sha(scores_path), 'manifest_sha256':sha(manifest),
                        'provenance':prov, 'adapter_sha256':sha(Path(__file__)), 'batch_size':20}
            target = folder/'official_result.json'
            if target.exists():
                saved = json.loads(target.read_text())
                if saved['identity'] != identity or any(sha(folder/n)!=h for n,h in saved['raw_sha256'].items()):
                    raise ValueError('Frozen evaluation identity or outputs changed.')
                result = saved['result']
                print('Cached frozen evaluation:', label, flush=True)
            else:
                state = torch.load(scores_path, map_location='cpu', weights_only=False)
                scores = state['scores'] if isinstance(state, dict) else state
                scores = scores.detach().cpu()
                if scores.ndim != 1 or scores.numel() != 32491 or not torch.isfinite(scores).all():
                    raise ValueError('Expected 32,491 finite frozen edge scores.')
                with torch.no_grad():
                    graph.scores[:] = -torch.inf
                    graph.scores[graph.real_edge_mask.bool()] = scores
                names, vectors = [], []
                original_baseline, original_graph = evaluation.evaluate_baseline, evaluation.evaluate_graph
                def capture(fn, prefix):
                    def wrapped(*args, **kw):
                        vector = fn(*args, **kw)
                        name = f'{len(names):02d}_{prefix}.pt'
                        torch.save(vector.detach().cpu(), folder/name)
                        names.append(name)
                        vectors.append(len(vector))
                        return vector
                    return wrapped
                refs = {}
                print('Frozen evaluation:', label, '; 12 dataset passes, batch=20, rows=',len(ds), flush=True)
                with meter.stage(label), torch.no_grad(), \
                     patch.object(evaluation,'evaluate_baseline',capture(original_baseline,'baseline')), \
                     patch.object(evaluation,'evaluate_graph',capture(original_graph,'graph')):
                    meter.status(evaluation=label, examples=len(ds), expected_dataset_passes=12)
                    result = evaluation.evaluate_area_under_curve(model, graph, ds.to_dataloader(20), metric,
                                                                 level='edge', absolute=False, quiet=True, refs=refs)
                if len(names)!=12 or any(n!=len(ds) for n in vectors):
                    raise ValueError('Expected 12 complete per-example vectors.')
                write_json(target, {'identity':identity, 'result':result, 'references':refs,
                                    'raw_sha256':{n:sha(folder/n) for n in names}})
                meter.status(completed_evaluation=label)
            pd.DataFrame({'fraction':GRID, 'weighted_edge_count':result[0],
                          'faithfulness':result[4], 'accuracy':result[5]}).to_csv(folder/'faithfulness_curve.csv',index=False)
            records.append({**metadata,'cpr':result[1],'cmd':result[2],'accuracy_auc':result[6],'artifact':str(folder.relative_to(ROOT))})
            pd.DataFrame(records).to_csv(a.out/'quality.csv',index=False)

        evaluate('gate1_on_gate2_validation',a.gate1/'final_scores.pt',val,val_dir/'validation_manifest.csv',
                 {'kind':'full_data_reference','seed':42,'pool':10000,'split':'validation','step':5000})
        for campaign in a.campaigns:
            for marker in sorted(campaign.glob('seed*_n*/completed.json')):
                job = marker.parent
                config = json.loads((job/'experiment_config.json').read_text())
                seed,pool = config['seed'],config['pool']
                directory = a.out/f'data_seed{seed}_n{pool}'
                directory.mkdir(exist_ok=True)
                with meter.stage(f'data_loading_train_seed{seed}_n{pool}'):
                    ds = HFEAPDataset('mib-bench/ioi',model.tokenizer,split='train',task='ioi',model_name='gpt2')
                    NestedData(directory,pool,seed,128).select(ds,'train')
                pd.testing.assert_frame_equal(pd.read_csv(job/'train_manifest.csv'),pd.read_csv(directory/'train_manifest.csv'))
                for step in (500,1000):
                    evaluate(f'train_seed{seed}_n{pool}_step{step}', job/f'checkpoints/step_{step:06d}.pt',
                             ds,directory/'train_manifest.csv',{'kind':'checkpoint_train_diagnostic',
                             'seed':seed,'pool':pool,'split':'train','step':step})
                del ds
        if source_hashes != {n:sha(a.gate1/n) for n in source_hashes}:
            raise ValueError('Frozen Gate 1 artifacts changed.')
        write_json(a.out/'completed.json',{'passed':True,'gate1_unchanged':True,'source_hashes':source_hashes,
                                          'training_updates':0,'gate3_executed':False})
        success = True
    finally:
        meter.status(success=success)
        meter.stop.set(); meter.worker.join(); meter.save(); meter.stream.close()
        write_json(a.out/'process_summary.json',{'success':success,'counts':meter.counts,
                                                'process_seconds':time.perf_counter()-meter.started})


if __name__ == '__main__':
    main()
