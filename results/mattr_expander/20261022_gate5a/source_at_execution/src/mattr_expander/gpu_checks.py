"""Small-reference CUDA correctness and measured full-size sparse operation cost."""
import time
import numpy as np
import torch
from .coupling import Coupling
from .graphs import ring


def run(graphs,out):
    if not torch.cuda.is_available():raise RuntimeError('CUDA required.')
    records=[]
    from src.mattr_optimization.common import configure
    configure()
    from matryoshka_attribution import sigmoid_topk
    for family,neighbors in graphs.items():
        small=ring(21);idx=torch.tensor(small,dtype=torch.long,device='cuda')
        x=torch.linspace(-1,1,21,device='cuda',dtype=torch.double,requires_grad=True)
        v=torch.cos(x.detach());mix=Coupling(idx,.25)
        cpu_x=x.detach().cpu().requires_grad_();cpu_idx=torch.tensor(small,dtype=torch.long)
        cpu_y=.75*cpu_x+.25*cpu_x[cpu_idx].mean(1)
        cpu_g=torch.autograd.grad((cpu_y*v.cpu()).sum(),cpu_x)[0]
        y=mix(x);g=torch.autograd.grad((y*v).sum(),x)[0]
        torch.testing.assert_close(y.cpu(),cpu_y);torch.testing.assert_close(g.cpu(),cpu_g)
        if not torch.autograd.gradcheck(mix,(x,),eps=1e-6,atol=1e-5):raise ValueError('CUDA finite difference failed.')
        full_idx=torch.tensor(neighbors,dtype=torch.long,device='cuda')
        full=torch.linspace(-1,1,len(neighbors),device='cuda',requires_grad=True)
        full_mix=Coupling(full_idx,.25)
        # Compare production topology's forward AND exact symmetric backward
        # against CPU gather autograd, beyond the small finite-difference test.
        cpu_full=full.detach().cpu().requires_grad_();cpu_n=torch.tensor(neighbors,dtype=torch.long)
        expected=.75*cpu_full+.25*cpu_full[cpu_n].mean(1)
        weights=torch.sin(torch.arange(len(full),dtype=torch.float32)*.123)
        ref_grad=torch.autograd.grad((expected*weights).sum(),cpu_full)[0]
        actual=full_mix(full);actual_grad=torch.autograd.grad((actual*weights.cuda()).sum(),full)[0]
        torch.testing.assert_close(actual.cpu(),expected,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(actual_grad.cpu(),ref_grad,atol=2e-6,rtol=2e-6)
        assert torch.equal(Coupling(full_idx,0)(full),full)
        for beta in (0,.1,.25):
            for fraction in (.001,.005,.01,.5,.99):
                k=float(int(len(full)*fraction));mask=sigmoid_topk(Coupling(full_idx,beta)(full),k,T=.5,n_iters=30)
                if abs(float(mask.detach().sum())-k)>max(.01,k*1e-5):raise ValueError('Production GPU soft mask mass differs.')
        for _ in range(20):torch.autograd.grad(full_mix(full).square().sum(),full)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        for _ in range(200):torch.autograd.grad(full_mix(full).square().sum(),full)
        torch.cuda.synchronize()
        records.append({'family':family,'cpu_gpu_forward_max_error':float((actual.cpu()-expected).abs().max().detach()),
                        'cpu_gpu_gradient_max_error':float((actual_grad.cpu()-ref_grad).abs().max()),
                        'cuda_gradcheck':True,'forward_backward_repeats':200,
                        'seconds':time.perf_counter()-started,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20})
    import pandas as pd
    pd.DataFrame(records).to_csv(out/'gpu_mixing_checks.csv',index=False)
    return records
