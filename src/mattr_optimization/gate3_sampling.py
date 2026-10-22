"""Static CPR quadrature mixture for the official trainer's k_sampler hook."""
import bisect
import math
import torch
from .common import GRID, integration_weights


class CPRMixture:
    """One CPU float32 Torch draw per update; no private RNG or feedback state.

    A single inverse CDF gives an independent conditional uniform variate within
    either the continuous uniform component or a categorical grid interval.
    The official CPU Torch RNG checkpoint therefore resumes this sampler exactly.
    """
    def __init__(self, total, mixture_lambda, log=None):
        if mixture_lambda not in (.25,.5,1.):
            raise ValueError('Only prespecified lambdas .25/.5 and uniform control 1 are allowed.')
        self.total,self.lam,self.log=int(total),float(mixture_lambda),log
        weights=integration_weights()
        # Full graph has constant faithfulness 1: retain it in official evaluation,
        # exclude its constant quadrature contribution from gradient training.
        self.fractions=list(GRID[:-1])
        self.centers=[int(p*total) for p in self.fractions]
        self.weights=[w/sum(weights[:-1]) for w in weights[:-1]]
        self.cdf=[]
        cumulative=0.
        for w in self.weights:
            cumulative+=w;self.cdf.append(cumulative)
        self.cdf[-1]=1.
        self.bounds=[(max(1.,c*.95),min(float(total-1),c*1.05)) for c in self.centers]
        self.draws=0

    def transform(self,u):
        if not 0<=u<1:
            raise ValueError('u must lie in [0,1).')
        if u<self.lam:
            v=u/self.lam
            return 1.+(self.total-1.)*v, 'uniform',-1
        r=(u-self.lam)/(1.-self.lam)
        j=min(bisect.bisect_right(self.cdf,r),len(self.cdf)-1)
        left=0. if j==0 else self.cdf[j-1]
        width=self.cdf[j]-left
        v=min(max((r-left)/width,0.),math.nextafter(1.,0.))
        low,high=self.bounds[j]
        return low+(high-low)*v,'grid',j

    def sample(self):
        u=torch.rand(1).item()  # exactly the official uniform sampler's RNG consumption
        k,branch,j=self.transform(u)
        self.draws+=1
        if self.log:
            self.log.write(f'{self.draws},{u:.17g},{branch},{j},{k:.17g}\n')
        return k

    def observe(self,*args):
        pass  # static, not loss/validation-adaptive


def distribution(total=32491):
    s=CPRMixture(total,.5)
    return [{'fraction':p,'quadrature_weight':w,'rounded_k':c,'conditional_grid_probability':prob,
             'jitter_low':lo,'jitter_high':hi}
            for p,w,c,prob,(lo,hi) in zip(GRID[:-1],integration_weights()[:-1],s.centers,s.weights,s.bounds)]
