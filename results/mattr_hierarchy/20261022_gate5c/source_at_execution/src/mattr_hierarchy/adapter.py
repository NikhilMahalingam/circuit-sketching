"""Scoped official-trainer bridge; deterministic indexed group reduction."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import torch

def group_sum(x,padded,valid):
    return (x[padded]*valid).sum(dim=1)

class _Hierarchy(torch.autograd.Function):
    @staticmethod
    def forward(ctx,packed,ids,padded,valid,sizes):
        groups=len(sizes);h=packed[:groups];r=packed[groups:]
        centered=r-(group_sum(r,padded,valid)/sizes)[ids]
        ctx.save_for_backward(ids,padded,valid,sizes)
        return h[ids]+centered,centered
    @staticmethod
    def backward(ctx,ge,gr):
        ids,padded,valid,sizes=ctx.saved_tensors
        if ge is None:ge=torch.zeros(len(ids),device=sizes.device,dtype=sizes.dtype)
        if gr is None:gr=torch.zeros_like(ge)
        combined=ge+gr
        residual=combined-(group_sum(combined,padded,valid)/sizes)[ids]
        return torch.cat([group_sum(ge,padded,valid),residual]),None,None,None,None

class Hierarchy:
    def __init__(self,ids=None,gamma=0.,scale=1.,kind='hierarchy',device='cpu',dtype=torch.float32):
        if kind not in ('hierarchy','edge','bypass'):raise ValueError(kind)
        self.kind=kind;self.gamma=gamma;self.scale=scale;self.raw=None;self.effective=None;self.centered=None
        self.ids=None;self.group_count=0
        if kind=='hierarchy':
            self.ids=torch.as_tensor(ids,dtype=torch.long,device=device)
            sizes=torch.bincount(self.ids);assert bool((sizes>0).all())
            self.group_count=len(sizes);max_size=int(sizes.max())
            padded=torch.zeros((len(sizes),max_size),dtype=torch.long)
            valid=torch.zeros(padded.shape,dtype=dtype)
            # Mapping-time group loop only; no per-edge Python loop during training.
            for i in range(len(sizes)):
                members=(self.ids.cpu()==i).nonzero().flatten();padded[i,:len(members)]=members;valid[i,:len(members)]=1
            self.padded=padded.to(device);self.valid=valid.to(device);self.sizes=sizes.to(device=device,dtype=dtype)
    def count(self,n):return n+self.group_count
    def parts(self,raw):
        if self.kind!='hierarchy':return raw,raw
        return _Hierarchy.apply(raw,self.ids,self.padded,self.valid,self.sizes)
    def transform(self,raw):return self.parts(raw)[0]
    def penalty(self):
        return self.gamma*self.scale*self.centered.square().mean()
    def penalty_gradient(self):
        if self.kind=='bypass':return torch.zeros_like(self.raw)
        # Projection is symmetric/idempotent; centered residual has zero group mean.
        grad=2*self.gamma*self.scale*self.centered.detach()/len(self.centered)
        if self.kind=='hierarchy':return torch.cat([torch.zeros(self.group_count,device=grad.device,dtype=grad.dtype),grad])
        return grad
    @contextmanager
    def install(self):
        import matryoshka_attribution.trainer as trainer
        if self.kind=='bypass':
            yield;return
        parameter=trainer.nn.Parameter;build=trainer.build_mask
        def create(value,*args,**kwargs):
            if value.ndim!=1 or not torch.equal(value,torch.zeros_like(value)):raise ValueError('Official zero initialization changed.')
            return parameter(torch.zeros(self.count(len(value)),device=value.device,dtype=value.dtype),*args,**kwargs)
        def mask(raw,k,*args,**kwargs):
            self.raw=raw;self.effective,self.centered=self.parts(raw)
            if self.effective.requires_grad:self.effective.retain_grad()
            return build(self.effective,k,*args,**kwargs)
        with patch.object(trainer,'nn',SimpleNamespace(Parameter=create)),patch.object(trainer,'build_mask',mask):yield

def adam_update_contribution(parameter,optimizer,reg_gradient):
    """Counterfactual same-state Adam update, read-only: total vs no regularizer.

    Does not run or alter an optimizer; actual training still calls official Adam.
    """
    group=optimizer.param_groups[0];state=optimizer.state.get(parameter,{})
    b1,b2=group['betas'];t=int(state.get('step',0))+1
    m=state.get('exp_avg',torch.zeros_like(parameter));v=state.get('exp_avg_sq',torch.zeros_like(parameter))
    def update(g):
        mm=b1*m+(1-b1)*g;vv=b2*v+(1-b2)*g.square()
        return group['lr']*(mm/(1-b1**t))/(torch.sqrt(vv/(1-b2**t))+group['eps'])
    total=update(parameter.grad.detach());without=update(parameter.grad.detach()-reg_gradient)
    return float((total-without).double().norm()/total.double().norm().clamp_min(1e-30))
