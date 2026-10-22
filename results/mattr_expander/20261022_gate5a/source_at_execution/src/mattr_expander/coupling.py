"""Sparse score transformation installed only at the official mask-builder call."""
from contextlib import contextmanager
from unittest.mock import patch
import torch


class _SymmetricMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx,s,neighbors,beta):
        ctx.save_for_backward(neighbors);ctx.beta=beta
        return (1-beta)*s+beta*s[neighbors].mean(dim=1)

    @staticmethod
    def backward(ctx,g):
        (neighbors,)=ctx.saved_tensors;beta=ctx.beta
        # For an undirected regular graph P^T=P. Gather+fixed reduction avoids
        # nondeterministic CUDA scatter accumulation into repeated source indices.
        return (1-beta)*g+beta*g[neighbors].mean(dim=1),None,None


class Coupling:
    def __init__(self,neighbors,beta,kind='graph'):
        if not 0<=beta<.5:raise ValueError('Registered beta must be in [0,.5).')
        if kind not in ('graph','scale','zero'):raise ValueError('Unknown transformation.')
        self.neighbors=neighbors;self.beta=float(beta);self.kind=kind
        self.raw=None;self.effective=None

    def __call__(self,s):
        if self.beta==0 or self.kind=='zero':return s
        if self.kind=='scale':return (1-self.beta)*s
        if self.neighbors is None or len(self.neighbors)!=len(s):raise ValueError('Neighbor/score count mismatch.')
        return _SymmetricMix.apply(s,self.neighbors,self.beta)

    @contextmanager
    def install(self):
        import matryoshka_attribution.trainer as trainer
        original=trainer.build_mask
        def build(scores,k,variant='topk',**kw):
            if variant!='topk':raise ValueError('Only unchanged official sigmoid top-k is registered.')
            self.raw=scores;self.effective=self(scores)
            if self.effective.requires_grad:self.effective.retain_grad()
            return original(self.effective,k,variant,**kw)
        with patch.object(trainer,'build_mask',build):yield
