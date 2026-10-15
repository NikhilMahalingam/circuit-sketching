"""Thin MIB adapters and measured, cached counterfactual head interventions.

Official modules are imported only after configure_repository() resolves the
local checkout. No custom IOI data, tokenizer, EAP, or CPR/CMD implementation.
"""
from contextlib import contextmanager
from copy import copy
from datetime import datetime, timezone
from functools import partial
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def configure_repository(root, repo=None):
    root = Path(root).resolve()
    repo = Path(repo or os.environ.get('MIB_REPO', root / 'external/MIB-circuit-track')).resolve()
    required = ['README.md', 'run_attribution.py', 'run_evaluation.py',
                'MIB_circuit_track/dataset.py', 'MIB_circuit_track/metrics.py',
                'MIB_circuit_track/utils.py', 'MIB_circuit_track/evaluation.py',
                'EAP-IG/src/eap/attribute_node.py', 'EAP-IG/src/eap/evaluate.py',
                'EAP-IG/src/eap/utils.py', 'EAP-IG/src/eap/graph.py']
    missing = [p for p in required if not (repo / p).is_file()]
    if missing:
        raise FileNotFoundError(f'MIB checkout/submodule incomplete at {repo}: {missing}. '
                                'Set MIB_REPO to the official clone; no dataset fallback is used.')
    sys.path[:0] = [str(repo), str(repo / 'EAP-IG/src')]
    import MIB_circuit_track.dataset as official_dataset
    import eap.attribute_node as official_attribution
    for module in (official_dataset, official_attribution):
        if not Path(inspect.getfile(module)).resolve().is_relative_to(repo):
            raise RuntimeError('Another MIB/EAP installation is already imported. Restart the kernel.')
    def revision(path):
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    return repo, {'repository': str(repo), 'mib_commit': revision(repo),
                  'eap_commit': revision(repo / 'EAP-IG'),
                  'source_sha256': {p: hashlib.sha256((repo / p).read_bytes()).hexdigest() for p in required}}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    print(f'Seed: {seed}; TF32 disabled; model.eval() used.')


def load_model(device='cuda'):
    from transformer_lens import HookedTransformer
    from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME
    if device != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('This experiment requires CUDA: the official node EAP source allocates on CUDA. '
                           'Select the existing project CUDA kernel; no CPU fallback is applied.')
    model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME['gpt2'], device=device)
    model.eval()
    model.set_use_attn_result(True)
    model.set_use_split_qkv_input(True)
    model.set_use_hook_mlp_in(True)
    model.cfg.ungroup_grouped_query_attention = True
    model.requires_grad_(False)
    return model


def subset(dataset, indices):
    result = copy(dataset)
    result.dataset = dataset.dataset.select(list(map(int, indices)))
    return result


def load_splits(model, sizes, seed):
    from MIB_circuit_track.dataset import HFEAPDataset
    from MIB_circuit_track.utils import TASKS_TO_HF_NAMES
    datasets, manifests, info, full_prompts = {}, [], {}, {}
    for split, size in sizes.items():
        official = HFEAPDataset('mib-bench/' + TASKS_TO_HF_NAMES['ioi'], model.tokenizer,
                               split=split, task='ioi', model_name='gpt2')
        if size < 1 or size > len(official):
            raise ValueError(f'{split}: requested {size}; official filtered split has {len(official)}.')
        full_prompts[split] = set(official.dataset['prompt'])
        indices = np.random.default_rng(seed).permutation(len(official))[:size]
        ds = subset(official, indices)
        datasets[split] = ds
        info[split] = {'official_filtered_size': len(official), 'selected_size': len(ds),
                       'official_fingerprint': official.dataset._fingerprint,
                       'selected_fingerprint': ds.dataset._fingerprint,
                       'selected_filtered_indices': indices.tolist(),
                       'columns': official.dataset.column_names}
        for i, source_index in enumerate(indices):
            raw = ds.dataset[i]
            clean, corrupted, labels = ds[i]
            names = [raw['metadata']['indirect_object'], raw['metadata']['subject']]
            ids = [model.tokenizer(' ' + n, add_special_tokens=False).input_ids for n in names]
            # MIB takes the first token; fail loudly if that is ambiguous here.
            if any(len(tokens) != 1 for tokens in ids) or [t[0] for t in ids] != labels:
                raise ValueError('Selected IOI labels are not single tokens. Inspect official metric semantics.')
            manifests.append({'split': split, 'example_index': i, 'official_filtered_index': int(source_index),
                              'clean_prompt': clean, 'corrupted_prompt': corrupted,
                              'correct_token_id': labels[0], 'incorrect_token_id': labels[1],
                              'correct_answer': names[0], 'incorrect_answer': names[1],
                              'metadata': json.dumps(raw['metadata']),
                              'template': json.dumps(raw['template']),
                              'example_hash': hashlib.sha256(clean.encode()).hexdigest()})
    overlap = {}
    names = list(datasets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap[f'{a}__{b}'] = len(full_prompts[a] & full_prompts[b])
            if overlap[f'{a}__{b}']:
                raise ValueError(f'Official clean-prompt overlap between {a} and {b}; investigate before continuing.')
    info['full_filtered_split_clean_prompt_overlap'] = overlap
    return datasets, pd.DataFrame(manifests), info


class Ledger:
    """Count actual model.forward calls, including official library calls.

    Each call is one batch; examples_processed counts repeated example-forwards.
    Stage wall time includes backwards and CPU work, unlike forward_seconds.
    """
    def __init__(self, device, out_dir):
        self.device, self.out_dir = torch.device(device), Path(out_dir)
        self.calls, self.stages = [], []
        self.stage_name = 'outside_stage'
        self.hits = 0
        self.backward_calls = 0
        self.component_count = 0
        self.mask_keys = set()

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @contextmanager
    def stage(self, name, estimate=None):
        if estimate is not None:
            print(f'{name}: {estimate}', flush=True)
        old = self.stage_name
        self.stage_name = name
        start_index, hits, backwards = len(self.calls), self.hits, self.backward_calls
        masks_before = set(self.mask_keys)
        self.sync()
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        success = False
        try:
            yield
            success = True
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError(f'CUDA OOM in {name}. Lower BATCH_SIZE/EAP_BATCH_SIZE/MIB_EVAL_BATCH_SIZE '
                               'and rerun explicitly; experimental settings were not changed.') from error
        finally:
            self.sync()
            calls = self.calls[start_index:]
            self.stages.append({'stage': name, 'success': success,
                                'forward_passes_batches': len(calls),
                                'successful_forward_passes': sum(c['success'] for c in calls),
                                'examples_processed': sum(c['batch_size'] for c in calls),
                                'forward_seconds': sum(c['seconds'] for c in calls),
                                'wall_seconds': time.perf_counter() - start,
                                'backward_calls': self.backward_calls - backwards,
                                'cache_hits': self.hits - hits,
                                'new_unique_masks': len(self.mask_keys - masks_before),
                                'unique_masks_cumulative': len(self.mask_keys),
                                'component_count': self.component_count,
                                'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated(self.device) / 2**20,
                                'peak_cuda_reserved_mib': torch.cuda.max_memory_reserved(self.device) / 2**20})
            self.stage_name = old
            self.save()
            print(f'{name}: {len(calls)} actual forward batches, '
                  f'{self.stages[-1]["wall_seconds"]:.2f}s wall', flush=True)

    @contextmanager
    def instrument(self, model):
        original = model.forward
        def measured(*args, **kwargs):
            tokens = args[0] if args else kwargs['input']
            count = len(tokens) if not isinstance(tokens, str) else 1
            # Official EAP accumulates unused parameter grads; clear them before
            # each new gradient forward without changing activation attribution.
            if torch.is_grad_enabled():
                model.zero_grad(set_to_none=True)
            self.sync()
            start, success = time.perf_counter(), False
            try:
                output = original(*args, **kwargs)
                success = True
                return output
            finally:
                self.sync()
                self.calls.append({'stage': self.stage_name, 'batch_size': count,
                                   'seconds': time.perf_counter() - start, 'success': success,
                                   'grad_enabled': torch.is_grad_enabled()})
        model.forward = measured
        try:
            yield
        finally:
            model.forward = original

    def save(self):
        pd.DataFrame(self.stages).to_csv(self.out_dir / 'compute_accounting.csv', index=False)
        pd.DataFrame(self.calls).to_csv(self.out_dir / 'forward_log.csv', index=False)


def evaluate_ioi_batch(model, tokens, attention_mask, lengths, labels, hooks=()):
    """Official raw IOI logit difference and top-1 accuracy, per example."""
    from MIB_circuit_track.metrics import get_metric
    with torch.no_grad(), model.hooks(fwd_hooks=list(hooks)):
        logits = model(tokens, attention_mask=attention_mask)
        score = get_metric('logit_diff', 'ioi', model.tokenizer, model)(
            logits, None, lengths, labels, mean=False, loss=False)
        accuracy = get_metric('acc', 'ioi', model.tokenizer, model)(
            logits, None, lengths, tuple(labels), mean=False, loss=False)
    return score.detach().cpu().numpy().astype(float), accuracy.detach().cpu().numpy().astype(float)


def patch_heads(activations, donor, heads, n_heads, d_model):
    """Replace exactly requested head-result slices, leaving the input intact."""
    if activations.ndim != 4 or activations.shape[2:] != (n_heads, d_model):
        raise ValueError('Expected TransformerLens hook_result [batch, position, head, d_model].')
    if donor.shape != (*activations.shape[:2], len(heads), d_model):
        raise ValueError('Counterfactual and clean activation shapes do not align.')
    edited = activations.clone()
    edited[:, :, heads, :] = donor.to(device=activations.device, dtype=activations.dtype)
    return edited


class IOIScorer:
    """Full counterfactual replacements at selected head outputs, all positions.

    Caches detached CPU donor slices and per-example scores only. Cache identity
    includes model/code/data/config provenance supplied by the notebook.
    """
    def __init__(self, model, datasets, components, batch_size, ledger, cache_dir):
        self.model, self.datasets, self.batch_size, self.ledger = model, datasets, batch_size, ledger
        self.components = tuple(tuple(map(int, c)) for c in components)
        if len(set(self.components)) != len(self.components) or not self.components:
            raise ValueError('Components must be nonempty and unique.')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache, self.donors = {}, {}
        self.groups = {}
        for i, (layer, head) in enumerate(self.components):
            if not (0 <= layer < model.cfg.n_layers and 0 <= head < model.cfg.n_heads):
                raise ValueError('Invalid component index.')
            self.groups.setdefault(layer, []).append((i, head))
        self.model.eval().requires_grad_(False)
        self.ledger.component_count = len(components)

    def _batch(self, ds, start, size):
        from MIB_circuit_track.dataset import collate_EAP
        from eap.utils import tokenize_plus
        clean, corrupted, labels = collate_EAP([ds[i] for i in range(start, min(start + size, len(ds)))])
        ct, am, lengths, _ = tokenize_plus(self.model, clean)
        dt, dam, dlengths, _ = tokenize_plus(self.model, corrupted)
        if ct.shape != dt.shape or not torch.equal(lengths, dlengths) or not torch.equal(am, dam):
            raise ValueError('Official clean/counterfactual tokens do not align for positionwise patching.')
        return ct, dt, am, lengths, labels

    def _donor(self, split, start, size, tokens, attention_mask, lengths, labels, cache=True):
        key = (split, start, size)
        if cache and key in self.donors:
            self.ledger.hits += 1
            return self.donors[key]
        saved = {}
        hooks = []
        for layer, pairs in self.groups.items():
            heads = [h for _, h in pairs]
            def capture(activation, hook, layer=layer, heads=heads):
                if activation.ndim != 4 or activation.shape[2:] != (self.model.cfg.n_heads, self.model.cfg.d_model):
                    raise ValueError('Unexpected head-result dimensions.')
                saved[layer] = activation[:, :, heads, :].detach().cpu().clone()
            hooks.append((f'blocks.{layer}.attn.hook_result', capture))
        score, accuracy = evaluate_ioi_batch(self.model, tokens, attention_mask, lengths, labels, hooks)
        result = saved, score, accuracy
        if cache:
            self.donors[key] = result
        return result

    def score(self, split, mask=None, *, corrupted=False, use_cache=True, batch_size=None, indices=None):
        n = len(self.components)
        mask = np.zeros(n, dtype=int) if mask is None else np.asarray(mask)
        if mask.shape != (n,) or not np.isin(mask, [0, 1]).all():
            raise ValueError('Expected a binary mask over the selected heads.')
        if corrupted and mask.any():
            raise ValueError('Raw corrupted evaluation cannot also intervene.')
        mask = tuple(map(int, mask))
        batch_size = self.batch_size if batch_size is None else batch_size
        cacheable = use_cache and batch_size == self.batch_size and indices is None
        key = (split, 'corrupted' if corrupted else ''.join(map(str, mask)))
        path = self.cache_dir / f'{key[0]}_{key[1]}.npz'
        if cacheable and key in self.cache:
            self.ledger.hits += 1
            return self.cache[key]
        if cacheable and path.exists():
            with np.load(path) as stored:
                result = stored['score'], stored['accuracy']
            self.cache[key] = result
            self.ledger.hits += 1
            return result
        ds = self.datasets[split] if indices is None else subset(self.datasets[split], indices)
        scores, accuracies = [], []
        if not corrupted:
            self.ledger.mask_keys.add(mask)
        for start in range(0, len(ds), batch_size):
            ct, dt, am, lengths, labels = self._batch(ds, start, batch_size)
            hooks = []
            if corrupted or any(mask):
                donors, donor_score, donor_acc = self._donor(
                    split, start, batch_size, dt, am, lengths, labels, cache=cacheable)
                if corrupted:
                    scores.append(donor_score)
                    accuracies.append(donor_acc)
                    continue
                for layer, pairs in self.groups.items():
                    offsets = [j for j, (i, _) in enumerate(pairs) if mask[i]]
                    heads = [head for i, head in pairs if mask[i]]
                    if not heads:
                        continue
                    donor = donors[layer][:, :, offsets, :]
                    def replace(activation, hook, donor=donor, heads=heads):
                        return patch_heads(activation, donor, heads, self.model.cfg.n_heads, self.model.cfg.d_model)
                    hooks.append((f'blocks.{layer}.attn.hook_result', replace))
            score, accuracy = evaluate_ioi_batch(self.model, ct, am, lengths, labels, hooks)
            scores.append(score)
            accuracies.append(accuracy)
        result = np.concatenate(scores), np.concatenate(accuracies)
        if not all(np.isfinite(a).all() for a in result):
            raise ValueError('Nonfinite IOI metric.')
        if cacheable:
            self.cache[key] = result
            np.savez_compressed(path, score=result[0], accuracy=result[1])
        return result

    def effects(self, split, masks):
        baseline = self.score(split)[0]
        results = []
        for index, mask in enumerate(masks):
            results.append(self.score(split, mask)[0] - baseline)
            if (index + 1) % 16 == 0 or index + 1 == len(masks):
                print(f'  {split}: {index+1}/{len(masks)} masks requested (cache hits included)', flush=True)
        return np.stack(results)


def official_eap(model, dataset, batch_size, ledger):
    from eap.graph import Graph
    from eap.attribute_node import attribute_node
    from MIB_circuit_track.metrics import get_metric
    graph = Graph.from_model(model, node_scores=True)
    metric = get_metric('logit_diff', 'ioi', model.tokenizer, model)
    def loss(*args):
        result = metric(*args, mean=True, loss=True)
        def count_backward(gradient):
            ledger.backward_calls += 1
            return gradient
        result.register_hook(count_backward)
        return result
    try:
        model.requires_grad_(True)
        with torch.enable_grad():
            attribute_node(model, graph, dataset.to_dataloader(batch_size), loss,
                           'EAP', 'patching', quiet=True)
    finally:
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
    ranks = pd.DataFrame([{'component': node.name, 'layer': node.layer, 'head': node.head,
                           'eap_score': float(node.score), 'absolute_eap_score': abs(float(node.score))}
                          for node in graph.nodes.values() if hasattr(node, 'head')])
    if not np.isfinite(ranks.eap_score).all():
        raise ValueError('Official EAP returned nonfinite head attributions.')
    ranks = ranks.sort_values('absolute_eap_score', ascending=False).reset_index(drop=True)
    ranks['rank'] = np.arange(1, len(ranks) + 1)
    return graph, ranks


def verify_intervention(scorer, split='train', count=4, tolerance=3e-4):
    """Fresh passes: hook removal, determinism, batching, official equivalence."""
    from eap.graph import Graph
    from eap.evaluate import evaluate_graph, evaluate_baseline
    from MIB_circuit_track.metrics import get_metric
    indices = list(range(min(count, len(scorer.datasets[split]))))
    mask = np.zeros(len(scorer.components), dtype=int)
    mask[0] = 1
    options = dict(use_cache=False, indices=indices)
    before = scorer.score(split, **options)[0]
    patched = scorer.score(split, mask, **options)[0]
    again = scorer.score(split, mask, **options)[0]
    after = scorer.score(split, **options)[0]
    individual = scorer.score(split, mask, batch_size=1, **options)[0]
    for a, b in ((before, after), (patched, again), (patched, individual)):
        np.testing.assert_allclose(a, b, atol=tolerance, rtol=1e-4)
    # MIB's exact node intervention disables every outgoing edge of that head.
    graph = Graph.from_model(scorer.model, node_scores=True)
    graph.reset(empty=False)
    layer, head = scorer.components[0]
    for edge in graph.nodes[f'a{layer}.h{head}'].child_edges:
        edge.in_graph = False
    ds = subset(scorer.datasets[split], indices)
    metric = partial(get_metric('logit_diff', 'ioi', scorer.model.tokenizer, scorer.model),
                     mean=False, loss=False)
    exact = evaluate_graph(scorer.model, graph, ds.to_dataloader(scorer.batch_size), metric,
                           intervention='patching', quiet=True, skip_clean=True).numpy()
    official_clean = evaluate_baseline(scorer.model, ds.to_dataloader(scorer.batch_size), metric,
                                       quiet=True).numpy()
    np.testing.assert_allclose(exact, patched, atol=tolerance, rtol=1e-4)
    np.testing.assert_allclose(official_clean, before, atol=tolerance, rtol=1e-4)
    return {'n_examples': len(indices), 'head': [layer, head], 'mean_effect': float((patched-before).mean()),
            'hook_removal_max_error': float(np.max(abs(before-after))),
            'determinism_max_error': float(np.max(abs(patched-again))),
            'batching_max_error': float(np.max(abs(patched-individual))),
            'official_graph_max_error': float(np.max(abs(exact-patched))), 'tolerance': tolerance}


def official_evaluation(model, graph, dataset, batch_size):
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    from MIB_circuit_track.metrics import get_metric
    metric = partial(get_metric('logit_diff', 'ioi', model.tokenizer, model), mean=False, loss=False)
    with torch.no_grad():
        counts, area_under, area_from_1, average, faithfulnesses = evaluate_area_under_curve(
            model, graph, dataset.to_dataloader(batch_size), metric, quiet=True,
            level='node', absolute=True, intervention='patching')
    return {'weighted_edge_counts': counts, 'area_under': area_under, 'area_from_1': area_from_1,
            'average': average, 'faithfulnesses': faithfulnesses}
