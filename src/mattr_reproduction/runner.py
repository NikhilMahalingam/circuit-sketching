"""Execute upstream eval_mib.main with bounded official data and instrumentation."""
import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from .runtime import (ROOT, OFFICIAL, PERCENTAGES, FixedData, Meter, add_official_paths,
                      provenance, digest, write_json, export_ranking)
from .checkpoint import learn_checkpointed


def load_official_script():
    path = OFFICIAL/'scripts/mib/eval_mib.py'
    spec = importlib.util.spec_from_file_location('_official_mattr_eval', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def install_evaluation_capture(stack, meter, out, split='validation'):
    import MIB_circuit_track.evaluation as evaluation
    original_auc = evaluation.evaluate_area_under_curve
    original_base, original_graph = evaluation.evaluate_baseline, evaluation.evaluate_graph

    def auc(model, graph, dataloader, metrics, **kwargs):
        meter.switch(split+'_evaluation')
        export_ranking(graph, out, 'node_ranking')
        baseline_values = []; graph_values = []; retained = []
        apply = graph.apply_topn
        def topn(*args, **kw):
            result = apply(*args, **kw)
            if kw.get('level', 'edge') == 'node':
                retained.append({'requested_nodes': int(args[0]),
                                 'retained_nodes': int(graph.nodes_in_graph.sum().item())})
            return result
        def base(*args, **kw):
            values = original_base(*args, **kw)
            baseline_values.append(values.detach().cpu().numpy().copy())
            return values
        def circuit(*args, **kw):
            values = original_graph(*args, **kw)
            graph_values.append(values.detach().cpu().numpy().copy())
            return values
        with patch.object(graph, 'apply_topn', topn), \
             patch.object(evaluation, 'evaluate_baseline', base), \
             patch.object(evaluation, 'evaluate_graph', circuit), torch.no_grad():
            result = original_auc(model, graph, dataloader, metrics, **kwargs)
        if len(result) != 7 or len(graph_values) != 11 or len(retained) != 10:
            raise RuntimeError('Pinned official evaluator API/grid changed; inspect before adapting.')
        weighted, area_under, area_from_1, average, faith, accuracies, acc_auc = result
        rows = []
        for i, pct in enumerate(PERCENTAGES):
            rows.append({'retained_fraction': pct, **retained[i],
                         'weighted_edge_count': weighted[i],
                         'official_normalized_logit_diff_cpr': faith[i],
                         'official_accuracy': accuracies[i],
                         'raw_logit_diff_mean': float(graph_values[i+1].mean())})
        pd.DataFrame(rows).to_csv(out/(split+'_curve.csv'), index=False)
        np.savez_compressed(out/(split+'_per_example.npz'), baseline=baseline_values[0],
                            corrupted=graph_values[0], **{f'p{i}': x for i,x in enumerate(graph_values[1:])})
        result_name = 'official_evaluation.json' if split=='validation' else 'official_'+split+'_evaluation.json'
        write_json(out/result_name, {'metric': 'logit_diff', 'level': 'node', 'split': split,
            'intervention': 'patching', 'absolute_ranking': False, 'log_scale': False,
            'percentages': PERCENTAGES, 'cpr_auc_area_under': area_under,
            'cmd_auc_area_from_1': area_from_1, 'average_cpr': average,
            'accuracy_auc_logspace': acc_auc, 'examples': len(dataloader.dataset),
            'clean_logit_diff': float(baseline_values[0].mean()),
            'corrupted_logit_diff': float(graph_values[0].mean()),
            'definition_source': 'MIB_circuit_track.evaluation.evaluate_area_under_curve'})
        meter.switch('saving_and_ranking')
        return result
    stack.enter_context(patch.object(evaluation, 'evaluate_area_under_curve', auc))


def run(args):
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        print('Already complete:', out, flush=True); return
    if not torch.cuda.is_available():
        raise RuntimeError('MAttr reproduction requires WSL Ubuntu CUDA; no CPU fallback.')
    add_official_paths()
    info = provenance()
    local_files = [Path(__file__), Path(__file__).with_name('runtime.py'), Path(__file__).with_name('checkpoint.py')]
    config = {'method': 'MAttr', 'kind': args.kind, 'model': 'gpt2', 'task': 'ioi', 'level': 'node',
        'train_size': args.train_size, 'validation_size': args.validation_size,
        'seed': args.seed, 'validation_seed': 42, 'steps': args.steps,
        'optimizer': 'adam', 'lr': .05, 'masking': 'topk', 'mode': 'iso',
        'k_schedule': 'uniform', 'T': .5, 'n_iters': 30, 'adam_eps': 1e-8,
        'include_input': True, 'train_batch_size': 1,
        'evaluation_batch_size': args.eval_batch_size, 'cf_cache_gb': args.cf_cache_gb,
        'checkpoint_every': args.checkpoint_every, 'provenance': info,
        'local_source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in local_files},
        'unique_pool_is_distinct_from_sampled_unique_examples': True,
        'public_test_accessed': False}
    identity = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    config['identity'] = identity
    if (out/'experiment_config.json').exists():
        previous = json.loads((out/'experiment_config.json').read_text())
        if previous['identity'] != identity:
            raise ValueError('Run configuration/source changed. Use another output directory.')
    write_json(out/'experiment_config.json', config)
    selection = FixedData(out, args.train_size, args.seed, args.validation_size)
    module = load_official_script()
    original_learn = module.learn_scores
    meter = Meter(out); meter.switch('model_and_data_preparation')
    presentations = (out/'training_presentations.jsonl').open('a', buffering=1)
    step_state = {'step': 0}
    succeeded = False
    overall_start = time.perf_counter()

    def learn(total, loss_fn, **kwargs):
        meter.switch('attribution_training')
        closure = inspect.getclosurevars(loss_fn).nonlocals
        pretok = closure['pretok']; n = closure['n_examples']
        if len(pretok) != len(selection.datasets['train']) or n != len(pretok):
            raise AssertionError('Training closure escaped selected official subset.')
        if any(x[0].shape[0] != x[1].shape[0] for x in pretok):
            raise ValueError('Length-rejected examples require explicit skipped-step resume handling.')
        checkpoints = sorted((out/'checkpoints').glob('step_*.pt'))
        step_state['step'] = (torch.load(checkpoints[-1], map_location='cpu', weights_only=False)['completed_steps']
                              if checkpoints else 0)
        original_randint = random.randint
        def bounded_loss(mask):
            picked = []
            def randint(low, high):
                index = original_randint(low, high)
                if low == 0 and high == n-1:
                    if not 0 <= index < len(pretok):
                        raise AssertionError('Training sampler escaped configured subset.')
                    picked.append(index)
                return index
            with patch.object(random, 'randint', randint):
                value = loss_fn(mask)
            if value is not None:
                for index in picked:
                    presentations.write(json.dumps({'step': step_state['step'], 'subset_index': index,
                        'official_filtered_index': selection.datasets['train']._mattr_official_indices[index]})+'\n')
            return value
        original_callback = kwargs.pop('on_step', None)
        def on_step(step, k, loss, scores):
            step_state['step'] = step+1
            if original_callback:
                original_callback(step, k, loss, scores)
        result = learn_checkpointed(original_learn, total, bounded_loss,
            checkpoint_dir=out/'checkpoints', identity=identity, interval=args.checkpoint_every,
            meter=meter, on_step=on_step, **kwargs)
        if len(result.train_log) != args.steps:
            raise AssertionError('Requested optimization steps were not all completed.')
        pd.DataFrame(result.train_log, columns=['step','k','k_fraction','loss']).to_csv(out/'training_curve.csv', index=False)
        cf = closure['cf_cache']
        cache_path = out/'counterfactual_cache.json'
        cache_attempts = json.loads(cache_path.read_text()) if cache_path.exists() else []
        cache_attempts.append({'seen_examples_this_attempt': cf.seen_examples,
            'forward_examples_this_attempt': cf.fwd_examples,
            'hits_this_attempt': cf.seen_examples-cf.fwd_examples,
            'entries': len(cf.store), 'bytes': cf.bytes,
            'cache_restored_after_restart': False, 'official_summary': cf.stats()})
        write_json(cache_path, cache_attempts)
        meter.switch('ranking_and_model_transition')
        return result

    from transformers import AutoModelForCausalLM
    from transformer_lens import HookedTransformer
    hf_load = AutoModelForCausalLM.from_pretrained
    tl_load = HookedTransformer.from_pretrained
    official_args = [str(OFFICIAL/'scripts/mib/eval_mib.py'), '--model','gpt2','--task','ioi',
        '--steps',str(args.steps),'--k-schedule','uniform','--masking','topk','--mode','iso',
        '--lr','0.05','--optimizer','adam','--seed',str(args.seed),'--split','validation',
        '--train-split','train','--include-input','--batch-size',str(args.eval_batch_size),
        '--train-batch-size','1','--cf-cache-gb',str(args.cf_cache_gb),'--no-wandb',
        '--output',str(out/'official')]
    write_json(out/'official_command.json', {'argv': official_args,
        'subset_adapter': {'train': args.train_size, 'validation': args.validation_size, 'seed': args.seed}})
    try:
        with ExitStack() as stack:
            stack.enter_context(selection.patch_loader())
            stack.enter_context(patch.object(module, 'learn_scores', learn))
            stack.enter_context(patch.object(AutoModelForCausalLM, 'from_pretrained',
                lambda *a, **kw: meter.attach(hf_load(*a, **kw), 'huggingface')))
            stack.enter_context(patch.object(HookedTransformer, 'from_pretrained',
                lambda *a, **kw: meter.attach(tl_load(*a, **kw), 'transformerlens')))
            stack.enter_context(patch.object(sys, 'argv', official_args))
            stack.enter_context(meter.backwards())
            install_evaluation_capture(stack, meter, out)
            module.main()
        succeeded = True
    except torch.cuda.OutOfMemoryError as error:
        write_json(out/'failure.json', {'error': str(error), 'oom': True,
            'message': 'Lower an explicit evaluation batch size or CF cache budget in a NEW configuration. No examples or steps were silently changed.'})
        raise
    finally:
        presentations.close(); meter.close(success=succeeded)
    sampled = pd.read_json(out/'training_presentations.jsonl', lines=True)
    if not sampled.subset_index.between(0,len(selection.datasets['train'])-1).all():
        raise AssertionError('Out-of-subset training presentation detected.')
    write_json(out/'complete.json', {'success': True, 'kind': args.kind, 'optimization_steps': args.steps,
        'eligible_unique_train_examples': len(selection.datasets['train']),
        'sampled_unique_train_examples': int(sampled.subset_index.nunique()),
        'total_training_presentations': len(sampled), 'validation_examples': len(selection.datasets['validation']),
        'wall_seconds_this_attempt': time.perf_counter()-overall_start,
        'public_test_accessed': False})
    print('COMPLETED:', out, flush=True)


def size(value):
    return None if value == 'all' else int(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--kind', choices=['smoke','reference','scaling'], required=True)
    parser.add_argument('--train-size', type=size, default=None)
    parser.add_argument('--validation-size', type=size, default=128)
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-batch-size', type=int, default=20)
    parser.add_argument('--cf-cache-gb', type=float, default=4.)
    parser.add_argument('--checkpoint-every', type=int, default=50)
    args = parser.parse_args()
    if args.kind != 'smoke' and args.steps != 500:
        parser.error('Reference/scaling recipes keep 500 steps; use smoke for runtime checks.')
    run(args)


if __name__ == '__main__':
    main()
