"""GPU values/gradients and score-operation timing, without model forwards."""
import time
import numpy as np
import pandas as pd
import torch
from .adapter import Hierarchy,group_sum
from src.mattr_optimization.common import write_json

def run(out):
    from matryoshka_attribution.masks import build_mask
    rows=[];bench=[];generator=torch.Generator().manual_seed(5742)
    for family in ('H1','H2'):
        ids=np.load(out/f'{family}_groups.npy');cpu=Hierarchy(ids,dtype=torch.float64)
        gpu=Hierarchy(ids,device='cuda',dtype=torch.float64)
        raw=torch.randn(cpu.count(32491),generator=generator,dtype=torch.float64)*.05
        a=raw.clone().requires_grad_();b=raw.cuda().requires_grad_();probe=torch.linspace(-1,1,32491,dtype=torch.float64)
        x,r=cpu.parts(a);y,s=gpu.parts(b)
        gc=torch.autograd.grad((x*probe).sum()+.01*r.square().mean(),a)[0]
        gg=torch.autograd.grad((y*probe.cuda()).sum()+.01*s.square().mean(),b)[0]
        error=float((x-y.cpu()).abs().max());gradient_error=float((gc-gg.cpu()).abs().max())
        assert error<1e-10 and gradient_error<1e-10
        assert float((group_sum(s,gpu.padded,gpu.valid)/gpu.sizes).abs().max())<1e-10
        assert torch.equal(gpu.transform(b),gpu.transform(b))
        single=Hierarchy([0,0,1],device='cuda',dtype=torch.double)
        small=torch.linspace(-.1,.2,5,device='cuda',dtype=torch.double,requires_grad=True)
        assert torch.autograd.gradcheck(single.parts,small,atol=1e-5)
        fp32=Hierarchy(ids,device='cuda');score=fp32.transform(raw.float().cuda())
        for fraction in (.001,.005,.01,.02,.05,.5,.99):
            k=int(32491*fraction);mask=build_mask(score,k,'topk',T=.5,n_iters=30).mask
            mass=float(mask.sum());assert abs(mass-k)<=max(.02,k*1e-5)
            assert bool((torch.diff(mask[score.argsort()])>=-1e-7).all())
            rows.append({'family':family,'fraction':fraction,'requested_mass':k,'actual_mass':mass,
                         'CPU_GPU_value_error':error,'CPU_GPU_gradient_error':gradient_error,'passed':True})
    for kind in ('bypass','edge','H1','H2'):
        ids=np.load(out/f'{kind}_groups.npy') if kind in ('H1','H2') else None
        h=Hierarchy(ids,gamma=.01,kind='hierarchy' if ids is not None else kind,device='cuda')
        x=torch.zeros(h.count(32491),device='cuda',requires_grad=True)
        probe=torch.linspace(-1,1,32491,device='cuda')
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        for _ in range(200):
            effective,r=h.parts(x);loss=(effective*probe).mean()+(.01*r.square().mean() if kind!='bypass' else 0)
            loss.backward();x.grad=None
        torch.cuda.synchronize()
        bench.append({'kind':kind,'repetitions':200,'seconds':time.perf_counter()-start,
                      'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'scope':'score transform, MSE and backward only; no model/topk/Adam'})
    pd.DataFrame(rows).to_csv(out/'gpu_correctness.csv',index=False)
    pd.DataFrame(bench).to_csv(out/'score_operation_microbenchmark.csv',index=False)
    write_json(out/'gpu_checks_passed.json',{'passed':True,'model_forwards':0,'GPU_finite_difference':True,
                'deterministic_export':True,'CPU_GPU_values_and_gradients':True,'all_original_scores_eligible':32491})
