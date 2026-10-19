"""Same dynamic donor-minus-current edge replacement, full or suffix execution.

Both arms cache fixed corrupted donors. Only the optimized arm caches clean
sender outputs and residual boundaries. No frozen clean sender is substituted
for a sender computed in the affected suffix. Matrix patching is the vectorized
form of sketch_acdc.core.receiver_patch and is checked against it and MIB.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import hashlib
import time

import numpy as np
import torch

from .graph import Dependencies, digest


def sync(model):
    if torch.device(model.cfg.device).type == 'cuda': torch.cuda.synchronize()


@dataclass
class Batch:
    tokens: torch.Tensor
    donors: torch.Tensor
    attention_mask: torch.Tensor
    lengths: torch.Tensor
    labels: object


def prepare_batches(model, dataset, batch_size):
    from eap.utils import tokenize_plus
    batches = []
    for clean, corrupted, labels in dataset.to_dataloader(batch_size):
        tokens, mask, lengths, positions = tokenize_plus(model, clean)
        donors, donor_mask, _, donor_positions = tokenize_plus(model, corrupted)
        if positions != donor_positions or not torch.equal(mask, donor_mask):
            raise ValueError('Clean/corrupted padding or lengths differ. No silent alignment change.')
        batches.append(Batch(tokens, donors, mask, lengths, labels))
    return batches


class ExecutionMeter:
    """Count actual root forwards AND actual transformer-block executions."""
    def __init__(self, model):
        self.model = model
        self.events = []
        self.phase = 'outside_stage'
        self.block_counts = [0]*model.cfg.n_layers

    @contextmanager
    def instrument(self):
        original = self.model.forward
        handles = []
        for layer, block in enumerate(self.model.blocks):
            def count(module, args, layer=layer): self.block_counts[layer] += 1
            handles.append(block.register_forward_pre_hook(count))
        def forward(*args, **kwargs):
            tokens = args[0] if args else kwargs['input']
            before = sum(self.block_counts); success = False
            try:
                result = original(*args, **kwargs); success = True; return result
            finally:
                self.events.append({'phase':self.phase,'batch_size':len(tokens),
                    'start_at_layer':kwargs.get('start_at_layer'),
                    'block_executions':sum(self.block_counts)-before,
                    'grad_enabled':torch.is_grad_enabled(),'success':success})
        self.model.forward = forward
        try: yield self
        finally:
            self.model.forward = original
            for handle in handles: handle.remove()


def matrix_patch(value, differences, weights):
    if weights.ndim == 1:
        return value + torch.einsum('bpsd,s->bpd', differences, weights)
    return value + torch.einsum('bpsd,sh->bphd', differences, weights)


class ExactExecutor:
    VERSION = 'dynamic_mib_prefix_v1'

    def __init__(self, model, graph, batches, cached=False):
        self.model, self.graph, self.batches, self.cached = model, graph, batches, bool(cached)
        self.dependencies = Dependencies(graph)
        if model.cfg.dtype != torch.float32 or model.training:
            raise ValueError('This reproduction requires eval-mode float32.')
        if model.cfg.parallel_attn_mlp or model.cfg.use_normalization_before_and_after:
            raise ValueError('Only serial attention/MLP GPT-2-style blocks are supported.')
        if model.cfg.positional_embedding_type != 'standard':
            raise ValueError('Only standard learned positional embeddings are supported.')
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Freeze model weights before preparing inference caches.')
        from MIB_circuit_track.metrics import get_metric
        self.metric = get_metric('logit_diff','ioi',model.tokenizer,model)
        self.versions = tuple(p._version for p in model.parameters())
        self.model_config = self._config()
        self.prepared = []
        self.evaluations = []; self.root_forwards = 0
        self.capture_layout = self._capture_layout()
        self.setup_seconds = 0.
        self.identity = digest({'version':self.VERSION,'graph':self.dependencies.identity,
            'config':self.model_config,'batch_tokens':[
                hashlib.sha256(b.tokens.cpu().numpy().tobytes()+b.donors.cpu().numpy().tobytes()+
                    b.attention_mask.cpu().numpy().tobytes()+torch.as_tensor(b.labels).cpu().numpy().tobytes()).hexdigest()
                for b in batches]})

    def _config(self):
        fields = ('n_layers','n_heads','d_model','dtype','positional_embedding_type',
                  'use_attn_result','use_split_qkv_input','use_hook_mlp_in')
        return {name:str(getattr(self.model.cfg,name)) for name in fields}

    def _check(self):
        if self.model.training or self._config()!=self.model_config or \
                tuple(p._version for p in self.model.parameters())!=self.versions:
            raise ValueError('Model/configuration changed after cache preparation; create a new executor.')

    def _capture_layout(self):
        result = [('hook_embed',0)]
        for layer in range(self.model.cfg.n_layers):
            a = self.graph.nodes[f'a{layer}.h0']; m = self.graph.nodes[f'm{layer}']
            result += [(a.out_hook,self.graph.forward_index(a)),
                       (m.out_hook,self.graph.forward_index(m))]
        return result

    def _capture(self, target, subtract=False):
        hooks = []
        for name,index in self.capture_layout:
            def capture(value, hook, index=index):
                if subtract: target[:,:,index] -= value.detach()
                else: target[:,:,index] = value.detach()
            hooks.append((name,capture))
        return hooks

    def _call(self, *args, **kwargs):
        self.root_forwards += 1
        return self.model(*args, **kwargs)

    def prepare(self):
        self._check()
        if self.prepared: raise ValueError('Caches already prepared; construction must be counted once.')
        sync(self.model); started = time.perf_counter()
        with torch.no_grad():
            for batch in self.batches:
                shape = (*batch.tokens.shape,self.graph.n_forward,self.model.cfg.d_model)
                donor = torch.empty(shape,device=self.model.cfg.device,dtype=torch.float32)
                with self.model.hooks(fwd_hooks=self._capture(donor)):
                    self._call(batch.donors,attention_mask=batch.attention_mask)
                clean = torch.empty_like(donor) if self.cached else None
                boundaries = {}
                hooks = self._capture(clean) if self.cached else []
                if self.cached:
                    for layer in range(self.model.cfg.n_layers):
                        def boundary(value, hook, layer=layer): boundaries[layer]=value.detach().clone()
                        hooks.append((f'blocks.{layer}.hook_resid_pre',boundary))
                with self.model.hooks(fwd_hooks=hooks):
                    logits = self._call(batch.tokens,attention_mask=batch.attention_mask)
                baseline = self._loss(logits,batch).cpu().numpy().copy()
                self.prepared.append({'donor':donor,'clean':clean,'boundaries':boundaries,'baseline':baseline})
                del logits
        sync(self.model); self.setup_seconds = time.perf_counter()-started
        return self

    def _loss(self, logits, batch):
        # Official task metric, negative logit difference so smaller is better.
        return self.metric(logits,None,batch.lengths,batch.labels,mean=False,loss=True)

    @property
    def baseline(self): return np.concatenate([p['baseline'] for p in self.prepared])

    @property
    def cache_bytes(self):
        return sum(t.numel()*t.element_size() for p in self.prepared
            for t in [p['donor'],*([p['clean']] if p['clean'] is not None else []),*p['boundaries'].values()])

    def _weights(self, removed):
        weights = torch.zeros((self.graph.n_forward,self.graph.n_backward),
                              dtype=torch.float32,device=self.model.cfg.device)
        active = {}
        rows, columns = [], []
        for name in removed:
            edge = self.graph.edges[name]
            rows.append(edge.matrix_index[0]); columns.append(edge.matrix_index[1])
            child = edge.child
            if child.name.startswith('a'):
                receiver = self.graph.backward_index(child,qkv=edge.qkv,attn_slice=True)
            else: receiver = self.graph.backward_index(child)
            active[edge.hook] = (self.graph.prev_index(child),receiver)
        if rows:
            weights[torch.tensor(rows,device=weights.device),torch.tensor(columns,device=weights.device)] = 1.
        return [(hook,prev,weights[:prev,receiver]) for hook,(prev,receiver) in sorted(active.items())]

    def evaluate(self, removed, requested_start=None):
        self._check()
        if not self.prepared: raise ValueError('Prepare and account for caches first.')
        removed = frozenset(removed)
        plan = self.dependencies.plan(removed,requested_start)
        start = plan.start_layer if self.cached else 0
        sync(self.model); began = time.perf_counter(); fwd_before = self.root_forwards
        patches = self._weights(removed)
        sync(self.model); bookkeeping = time.perf_counter()-began
        pieces = []; suffix_seconds = 0.
        with torch.no_grad():
            for batch,prepared in zip(self.batches,self.prepared):
                # Begin with corrupted values. Each executed sender subtracts
                # its CURRENT value; only proven-unaffected prefix senders use
                # the clean full-model cache.
                differences = prepared['donor'].clone()
                if self.cached and start>0:
                    prefix_end = self.graph.prev_index(self.graph.nodes[f'a{start}.h0'])
                    differences[:,:,:prefix_end] -= prepared['clean'][:,:,:prefix_end]
                hooks = []
                for name,index in self.capture_layout:
                    layer = -1 if name=='hook_embed' else int(name.split('.')[1])
                    if layer < start and self.cached and start>0: continue
                    def capture(value, hook, index=index): differences[:,:,index] -= value.detach()
                    hooks.append((name,capture))
                for name,prev,weights in patches:
                    def patch(value,hook,prev=prev,weights=weights):
                        return matrix_patch(value,differences[:,:,:prev],weights)
                    hooks.append((name,patch))
                sync(self.model); t0 = time.perf_counter()
                with self.model.hooks(fwd_hooks=hooks):
                    if self.cached and start>0:
                        logits = self._call(prepared['boundaries'][start].clone(),
                            start_at_layer=start,tokens=batch.tokens,attention_mask=batch.attention_mask)
                    else:
                        logits = self._call(batch.tokens,attention_mask=batch.attention_mask)
                pieces.append(self._loss(logits,batch).cpu().numpy().copy())
                del logits, differences
                sync(self.model); suffix_seconds += time.perf_counter()-t0
        sync(self.model); elapsed = time.perf_counter()-began
        self.evaluations.append({'removed_edges':len(removed),'earliest_affected_layer':plan.start_layer,
            'executed_start_layer':start,'safe_prefix_nodes':len(plan.reusable_nodes),
            'forward_batches':self.root_forwards-fwd_before,'wall_seconds':elapsed,
            'bookkeeping_seconds':bookkeeping,'suffix_and_metric_seconds':suffix_seconds,
            'other_allocation_and_transfer_seconds':max(0.,elapsed-bookkeeping-suffix_seconds)})
        return np.concatenate(pieces)

    def close(self):
        self.prepared.clear()
