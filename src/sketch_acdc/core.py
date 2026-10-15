"""MIB edge patching with differentiable interpolation and measured sequential pruning.

The source/receiver indexing and donor capture come from the official EAP graph
and make_hooks_and_matrices. Only selected edges are varied; every other edge
remains in the complete model. This is restricted ACDC, not full graph ACDC.
"""
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import time

import numpy as np
import torch


def state_key(state, identity, split):
    if any(x not in (0, 1) for x in state):
        raise ValueError('A cached pruning state must be binary.')
    bits = tuple(int(x) for x in state)
    return (identity, split, bits)


def hessian_vector_product(function, point, vector):
    """Differentiate a scalar function twice, without forming its Hessian."""
    point = point.detach().requires_grad_(True)
    value = function(point)
    if value.ndim != 0:
        raise ValueError('HVP function must return one scalar.')
    gradient = torch.autograd.grad(value, point, create_graph=True)[0]
    return torch.autograd.grad(gradient, point, grad_outputs=vector)[0]


def nystrom_sketch(function, point, rank, seed=0, ridge=1e-5):
    """Symmetric randomized Hessian sketch; every column costs one HVP."""
    n = point.numel()
    if not 1 <= rank <= n:
        raise ValueError('Rank must be between 1 and the number of edges.')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    omega = torch.randint(0, 2, (n, rank), generator=generator).to(point.device, point.dtype) * 2 - 1
    omega = torch.linalg.qr(omega).Q
    columns = [hessian_vector_product(function, point, omega[:, i]) for i in range(rank)]
    y = torch.stack(columns, dim=1)
    return nystrom_from_products(omega,y,ridge), omega, y


def nystrom_from_products(omega,y,ridge=1e-5):
    """Shared symmetric indefinite Nyström reconstruction from measured HVPs."""
    middle = omega.T @ y
    # Symmetric indefinite Hessians require a pseudoinverse, not Cholesky.
    middle = (middle + middle.T) / 2
    approximation = y @ torch.linalg.pinv(middle, rtol=ridge) @ y.T
    return (approximation + approximation.T) / 2


@dataclass(frozen=True)
class EdgeSpec:
    name: str
    sender_index: int
    receiver_index: int
    hook: str
    head: int | None
    layer: int
    eap_score: float


def select_edges(graph, scores, count=16):
    """Deterministic high/medium/low absolute-EAP mixture of real edges."""
    ranked = sorted(graph.edges.values(), key=lambda e: (-abs(float(scores[e.matrix_index])), e.name))
    if len(ranked) < count:
        raise ValueError('Not enough real MIB edges.')
    # Disjoint quantile windows make the sampling policy explicit.
    windows = [ranked[: max(count, len(ranked)//10)],
               ranked[len(ranked)//3:len(ranked)//3 + max(count, len(ranked)//10)],
               ranked[2*len(ranked)//3:2*len(ranked)//3 + max(count, len(ranked)//10)]]
    quotas = [count//3 + (count % 3 > 0), count//3 + (count % 3 > 1), count//3]
    chosen = []
    for window, quota in zip(windows, quotas):
        chosen.extend(window[:quota])
    if len({e.name for e in chosen}) != count:
        raise AssertionError('EAP quantile windows overlapped.')
    return [EdgeSpec(e.name, int(e.matrix_index[0]), int(e.matrix_index[1]), e.hook,
                     e.child.head if hasattr(e.child, 'head') else None, e.child.layer,
                     float(scores[e.matrix_index])) for e in chosen]


class EdgeExperiment:
    """Evaluate selected MIB edges, with exact binary and differentiable z masks."""
    VERSION = 'mib_dynamic_sender_v2'
    def __init__(self, model, graph, edges, splits, batch_size, identity, ledger=None):
        from MIB_circuit_track.metrics import get_metric
        self.model, self.graph, self.edges = model, graph, list(edges)
        self.splits, self.batch_size, self.identity = splits, batch_size, identity
        self.ledger = ledger
        self.metric = get_metric('logit_diff', 'ioi', model.tokenizer, model)
        self.cache = {}
        self.grouped = defaultdict(list)
        for i, edge in enumerate(self.edges):
            official = graph.edges[edge.name]
            if (official.matrix_index != (edge.sender_index, edge.receiver_index)
                    or official.hook != edge.hook or not bool(graph.real_edge_mask[official.matrix_index])):
                raise ValueError(f'Invalid edge identity: {edge.name}')
            self.grouped[(edge.hook, edge.head)].append((i, edge.sender_index))
        payload = {'identity': identity, 'adapter_version':self.VERSION,
                   'edges': [e.name for e in edges], 'batch_size': batch_size,
                   'splits': {k: v.dataset._fingerprint for k, v in splits.items()}}
        self.cache_id = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _frozen_batch_score(self, clean, corrupted, labels, z):
        """Historical implementation retained solely for auditing old results."""
        from eap.utils import tokenize_plus, make_hooks_and_matrices
        tokens, attention_mask, lengths, n_pos = tokenize_plus(self.model, clean)
        donor_tokens, _, _, donor_n_pos = tokenize_plus(self.model, corrupted)
        if donor_n_pos != n_pos:
            raise ValueError('MIB clean and corrupted token lengths differ.')
        (donor_hooks, clean_hooks, _), differences = make_hooks_and_matrices(
            self.model, self.graph, len(clean), n_pos, None)
        # Capture the same corrupted-minus-clean sender activations as MIB.
        with torch.no_grad():
            with self.model.hooks(fwd_hooks=donor_hooks):
                self.model(donor_tokens, attention_mask=attention_mask)
            with self.model.hooks(fwd_hooks=clean_hooks):
                self.model(tokens, attention_mask=attention_mask)
        hooks = []
        by_hook = defaultdict(list)
        for (name, head), members in self.grouped.items():
            by_hook[name].append((head, members))
        for hook_name, groups in by_hook.items():
            def patch(value, hook, groups=groups):
                result = value.clone()
                for head, members in groups:
                    delta = sum(differences[:, :, sender, :] * z[i]
                                for i, sender in members)
                    if head is None:
                        result = result + delta
                    else:
                        # Clone-and-assign preserves the gradient to z and only edits
                        # this q/k/v receiver head slice.
                        result[:, :, head, :] = result[:, :, head, :] + delta
                return result
            hooks.append((hook_name, patch))
        with self.model.hooks(fwd_hooks=hooks):
            logits = self.model(tokens, attention_mask=attention_mask)
        return self.metric(logits, None, lengths, labels, mean=False, loss=False)

    def _batch_score(self, clean, corrupted, labels, z):
        from eap.utils import tokenize_plus, make_hooks_and_matrices
        tokens, mask, lengths, n_pos = tokenize_plus(self.model, clean)
        donor_tokens, _, _, donor_n_pos = tokenize_plus(self.model, corrupted)
        if donor_n_pos != n_pos:
            raise ValueError('MIB clean/corrupted sequence lengths do not match.')
        (donor_hooks, _, _), donor_matrix = make_hooks_and_matrices(
            self.model, self.graph, len(clean), n_pos, None)
        with torch.no_grad(), self.model.hooks(fwd_hooks=donor_hooks):
            self.model(donor_tokens, attention_mask=mask)
        donor = {e.sender_index:donor_matrix[:,:,e.sender_index,:].clone() for e in self.edges}
        del donor_matrix, donor_hooks
        current, hooks, seen = {}, [], set()
        for edge in self.edges:
            if edge.sender_index in seen: continue
            seen.add(edge.sender_index)
            parent = self.graph.edges[edge.name].parent
            def capture(value, hook, sender=edge.sender_index, index=parent.index):
                current[sender] = value[index]  # Preserve upstream derivatives.
            hooks.append((parent.out_hook,capture))
        by_hook = defaultdict(list)
        for (hook,head),members in self.grouped.items():
            by_hook[hook].append((head,members))
        for name,groups in by_hook.items():
            def patch(value, hook, groups=groups):
                return receiver_patch(value,groups,donor,current,z)
            hooks.append((name,patch))
        with self.model.hooks(fwd_hooks=hooks):
            logits = self.model(tokens,attention_mask=mask)
        return self.metric(logits,None,lengths,labels,mean=False,loss=False)

    def continuous(self, z, split, limit=None):
        if z.shape != (len(self.edges),):
            raise ValueError('Intervention vector has wrong size.')
        if split not in self.splits:
            raise KeyError(split)
        dataset = self.splits[split]
        count = min(limit or len(dataset), len(dataset))
        pieces = []
        for start in range(0, count, self.batch_size):
            items = [dataset[i] for i in range(start, min(count, start+self.batch_size))]
            clean = [item[0] for item in items]
            corrupt = [item[1] for item in items]
            labels = [item[2] for item in items]
            pieces.append(self._batch_score(clean, corrupt, labels, z))
        return torch.cat(pieces)

    def evaluate_state(self, state, split, fresh=False):
        key = state_key(state, self.cache_id, split)
        if len(key[2]) != len(self.edges):
            raise ValueError('State length differs from selected edge count.')
        if not fresh and key in self.cache:
            if self.ledger is not None:
                self.ledger.hits += 1
            return self.cache[key].copy()
        z = torch.tensor(key[2], device=self.model.cfg.device, dtype=self.model.cfg.dtype)
        with torch.no_grad():
            result = self.continuous(z, split).detach().cpu().numpy().astype(float)
        self.cache[key] = result.copy()
        if self.ledger is not None:
            self.ledger.mask_keys.add(str(key))
        return result

    def exact_candidate_effect(self, state, edge_index, split):
        if state[edge_index] != 0:
            raise ValueError('Candidate edge is already removed.')
        next_state = list(state)
        next_state[edge_index] = 1
        return self.evaluate_state(next_state, split) - self.evaluate_state(state, split)

    def official_binary_score(self, state, split):
        """Independent comparison against MIB evaluate_graph, with fresh graph."""
        from eap.graph import Graph
        from eap.evaluate import evaluate_graph
        graph = Graph.from_model(self.model)
        graph.reset(empty=False)
        for removed, edge in zip(state, self.edges):
            if removed:
                graph.edges[edge.name].in_graph = False
        expected = graph.in_graph.clone()
        graph.prune()
        if not torch.equal(expected, graph.in_graph):
            raise ValueError('Official graph pruning removes additional dependencies for this state.')
        values = evaluate_graph(self.model, graph, self.splits[split].to_dataloader(self.batch_size),
                                lambda *args: self.metric(*args, mean=False),
                                quiet=True, intervention='patching', skip_clean=True)
        return values.detach().cpu().numpy().astype(float)


def exact_acdc(experiment, split, threshold, order=None):
    """Fixed-order restricted ACDC, using absolute score change and strict threshold."""
    n = len(experiment.edges)
    order = list(range(n)) if order is None else list(order)
    if sorted(order) != list(range(n)):
        raise ValueError('Order must contain each selected edge once.')
    state = [0] * n
    rows = []
    current = experiment.evaluate_state(state, split).mean()
    for step, i in enumerate(order):
        before = ''.join(map(str, state))
        t0 = time.perf_counter()
        trial = state.copy()
        trial[i] = 1
        proposed = experiment.evaluate_state(trial, split).mean()
        effect = proposed - current
        accepted = abs(effect) < threshold
        if accepted:
            state, current = trial, proposed
        rows.append({'step': step, 'state_before': before, 'edge': experiment.edges[i].name,
                     'exact_effect': effect, 'pruned': accepted, 'current_ioi_score': current,
                     'remaining_candidate_edges': n-sum(state), 'decision_seconds': time.perf_counter()-t0})
    return state, rows


def receiver_patch(value, groups, donor, current, z):
    """Official edge replacement: donor minus CURRENT sender, receiver slice only."""
    result=value.clone()
    for head,members in groups:
        delta=sum(z[i]*(donor[sender]-current[sender]) for i,sender in members)
        if head is None: result=result+delta
        else: result[:,:,head,:]=result[:,:,head,:]+delta
    return result


class FrozenSenderEdgeExperiment(EdgeExperiment):
    """Historical control. Do not use for new circuit-discovery measurements."""
    VERSION='historical_frozen_sender_v1'
    _batch_score=EdgeExperiment._frozen_batch_score
