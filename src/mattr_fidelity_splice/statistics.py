"""Shared paired template bootstrap; cohort normalization before official quadrature."""
import numpy as np
from src.mattr_optimization.common import GRID


def resamples(templates,draws=5000,seed=5645):
    templates=np.asarray(templates)
    rng=np.random.default_rng(seed)
    parts=[]
    for name in sorted(set(templates)):
        ids=np.flatnonzero(templates==name)
        parts.append(rng.choice(ids,size=(draws,len(ids)),replace=True))
    return np.concatenate(parts,axis=1)


def weights(indices,n):
    result=np.zeros((len(indices),n),dtype=np.float64)
    for i,row in enumerate(indices):result[i]=np.bincount(row,minlength=n)/n
    return result


def interval(values,alpha=.05):
    values=np.asarray(values)
    good=np.isfinite(values)
    if not good.any():return [None,None]
    return np.quantile(values[good],[alpha/2,1-alpha/2]).tolist()


def official_areas(margins,clean,corrupted):
    """Vectorized replay of pinned evaluation.py lines checked against its live point result.

    First axis is the unchanged nominal grid. Remaining axes may be bootstrap draws.
    This routine never averages per-prompt ratios or per-prompt CPR values.
    """
    m=np.asarray(margins,dtype=np.float64)
    den=np.asarray(clean)-np.asarray(corrupted)
    with np.errstate(divide='ignore',invalid='ignore'):
        f=(m-corrupted)/den
    f=np.where(np.abs(den)<1e-8,np.nan,f)
    cpr=0.;cmd=0.
    for i in range(len(GRID)-1):
        dx=GRID[i+1]-GRID[i]
        cpr=cpr+dx*(f[i]+f[i+1])/2
        cmd=cmd+dx*(np.abs(1-f[i])+np.abs(1-f[i+1]))/2
    return cpr,cmd


def cpr_bootstrap(margins,clean,corrupted,w):
    clean_mean=w@np.asarray(clean,dtype=np.float64)
    corrupt_mean=w@np.asarray(corrupted,dtype=np.float64)
    unstable=np.abs(clean_mean-corrupt_mean)<1e-8
    values,_=official_areas(np.asarray(margins,dtype=np.float64)@w.T,clean_mean,corrupt_mean)
    values=np.asarray(values);values[unstable]=np.nan
    return values,unstable


def classify_a(eap,baselines,w):
    """Arrays [3,N] and [5,3,N]; fixed seed average before prompt resampling."""
    diff=np.asarray(eap)-np.asarray(baselines).mean(0)
    draws=diff@w.T
    rows=[]
    eap_pass=mattr_pass=True
    for j,f in enumerate((.1,.2,.5)):
        lo,hi=interval(draws[j],.05/3)
        wins=int((np.asarray(eap)[j].mean()<np.asarray(baselines)[:,j].mean(-1)).sum())
        reverse=int((np.asarray(eap)[j].mean()>np.asarray(baselines)[:,j].mean(-1)).sum())
        eap_pass &= hi<0 and wins>=4
        mattr_pass &= lo>0 and reverse>=4
        rows.append(dict(fraction=f,difference=float(diff[j].mean()),lower=lo,upper=hi,eap_seed_wins=wins,mattr_seed_wins=reverse))
    return ('EAP-IG closer' if eap_pass else 'MAttr closer' if mattr_pass else 'Mixed'),rows


def classify_c(cpr_point,cpr_draws,accuracy_diff,w):
    # Equal seed-weighting; the same prompts are paired across seeds and parents/splices.
    pooled=np.asarray(accuracy_diff).mean(0)
    acc_ci=interval(pooled@w.T)
    cpr_ci=interval(np.asarray(cpr_draws).mean(0))
    gain=float(pooled.mean())
    compact=gain>=.10 and acc_ci[0]>0
    # Undefined replicate normalization prevents a confirmatory claim. Count/report
    # such replicates; never replace their denominator or silently call them stable.
    valid=cpr_ci[0] is not None and np.isfinite(cpr_draws).all()
    cpr_pass=valid and cpr_ci[0]>-.05
    return dict(classification='Invalid/incomplete' if not valid else 'Best of both' if compact and cpr_pass else
        'Compact gain with CPR cost' if compact else 'No useful compact gain',
        cpr_difference=float(np.mean(cpr_point)),cpr_interval=cpr_ci,accuracy_difference=gain,
        accuracy_interval=acc_ci,compact_pass=bool(compact),cpr_noninferiority_pass=bool(cpr_pass),
        unstable_replicates=int((~np.isfinite(np.asarray(cpr_draws).mean(0))).sum()))
