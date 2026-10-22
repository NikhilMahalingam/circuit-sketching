"""Initialization-only bridge to the pinned official trainer and EAP graph."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata


def rank_normalize(scores):
    """Signed ascending midranks; ties get equal scores, no arbitrary edge tie breaks."""
    x=torch.as_tensor(scores).detach().cpu()
    if x.ndim!=1 or x.numel()<2 or not torch.isfinite(x).all():
        raise ValueError('A finite real-edge vector is required.')
    z=2*(rankdata(x.double().numpy(),method='average')-1)/(len(x)-1)-1
    return torch.from_numpy(z).float()


@contextmanager
def initialized_trainer(initial):
    """Scope only trainer.nn.Parameter; no mutation of torch, Adam, or upstream files.

    The existing checkpoint adapter may restore saved scores/Adam/RNG afterwards.
    No RNG is consumed here; the official loop still creates the Parameter once.
    """
    import matryoshka_attribution.trainer as trainer
    initial=torch.as_tensor(initial).detach()
    parameter=trainer.nn.Parameter
    calls=[]
    def create(value,*args,**kwargs):
        if calls or value.shape!=initial.shape or not torch.equal(value,torch.zeros_like(value)):
            raise ValueError('Official score construction changed; initialization refused.')
        calls.append(True)
        return parameter(initial.to(value).clone(),*args,**kwargs)
    with patch.object(trainer,'nn',SimpleNamespace(Parameter=create)):
        yield
    if len(calls)!=1:raise ValueError('Expected exactly one official score construction.')


def edge_mapping(mattr_graph,eap_graph):
    """Canonical row-major real-mask order used by eval_mib_edge.py scatter."""
    mask=mattr_graph.real_edge_mask.bool()
    if len(mattr_graph.edges)!=32491 or int(mask.sum())!=32491:
        raise ValueError('Expected GPT-2 Small real graph with 32,491 edges.')
    if set(mattr_graph.edges)!=set(eap_graph.edges):raise ValueError('Missing EAP edges.')
    locations={tuple(idx):i for i,idx in enumerate(mask.nonzero().tolist())}
    rows=[];seen=set()
    for name,edge in mattr_graph.edges.items():
        other=eap_graph.edges[name]
        # Logits use column -1 in official Edge.matrix_index; the flattened
        # real mask uses its equivalent positive coordinate.
        idx=tuple(int(v)%dim for v,dim in zip(edge.matrix_index,mask.shape))
        if idx in seen or idx not in locations:raise ValueError('Duplicate/invalid edge coordinate.')
        seen.add(idx)
        other_idx=tuple(int(v)%dim for v,dim in zip(other.matrix_index,mask.shape))
        if idx!=other_idx or edge.hook!=other.hook or edge.qkv!=other.qkv:
            raise ValueError(f'Edge alignment differs: {name}')
        computed=(mattr_graph.forward_index(edge.parent,attn_slice=False),
                  mattr_graph.backward_index(edge.child,edge.qkv,attn_slice=False))
        if idx!=tuple(int(v)%dim for v,dim in zip(computed,mask.shape)):
            raise ValueError('Source/destination indexing differs.')
        rows.append({'mattr_index':locations[idx],'edge':name,'eap_row':idx[0],'eap_column':idx[1],
                     'parent':edge.parent.name,'child':edge.child.name,'qkv':edge.qkv or '',
                     'destination_hook':edge.hook})
    frame=pd.DataFrame(rows).sort_values('mattr_index').reset_index(drop=True)
    if seen!=set(locations) or frame.edge.duplicated().any():raise ValueError('Mapping is not bijective.')
    # Sentinel matrix tests all coordinates, not only sample attention or MLP edges.
    sentinel=torch.arange(mask.numel()).reshape(mask.shape)
    assert sentinel[mask].tolist()==[int(sentinel[r.eap_row,r.eap_column]) for r in frame.itertuples()]
    return frame


def mask_diagnostics(scores,scales=(.1,.5,1.,2.)):
    from matryoshka_attribution import sigmoid_topk
    z=rank_normalize(scores)
    records=[]
    for scale in scales:
        for fraction in (.001,.005,.01,.02,.05,.5,.99):
            s=(z*scale).requires_grad_();k=float(int(len(s)*fraction))
            mask=sigmoid_topk(s,k,T=.5,n_iters=30)
            # Nonconstant deterministic probe avoids the sum(mask)=k null gradient.
            probe1=torch.autograd.grad((mask*torch.linspace(-1,1,len(mask))).sum(),s,retain_graph=True)[0]
            probe2=torch.autograd.grad((mask*torch.sin(torch.arange(len(mask))*.123)).sum(),s)[0]
            row={'scale':scale,'fraction':fraction,'requested_k':k,'soft_mass':float(mask.detach().sum()),
                 'mass_error':float(mask.detach().sum()-k),'score_mean':float(s.detach().mean()),
                 'score_std':float(s.detach().std()),'score_min':float(s.detach().min()),'score_max':float(s.detach().max()),
                 'mask_min':float(mask.detach().min()),'mask_max':float(mask.detach().max()),
                 'saturated_fraction':float(((mask<1e-7)|(mask>1-1e-7)).float().mean()),
                 'finite_gradient':bool(torch.isfinite(probe1).all() and torch.isfinite(probe2).all()),
                 'nonzero_gradients':int(((probe1!=0)|(probe2!=0)).sum())}
            if abs(row['mass_error'])>max(.01,k*1e-5) or not row['finite_gradient'] or row['nonzero_gradients']!=len(s):
                raise ValueError(f'Initialization/mask diagnostic failed: {row}')
            records.append(row)
    return pd.DataFrame(records)
