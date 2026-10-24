"""Fixed-model prompt uncertainty and preregistered five-seed decision."""
import numpy as np
from src.mattr_compact_validation.statistics import bootstrap_indices

def interval(x,draws):
    x=np.asarray(x,dtype=float);boot=x[draws].mean(axis=1)
    lo,hi=np.quantile(boot,[.025,.975])
    return dict(mean=float(x.mean()),lower=float(lo),upper=float(hi))

def classify(primary,small,large,baseline_small,baseline_large,safeguards,correctness):
    """small/large: [3seeds,2stepbudgets,prompts]; primary:[5seeds,prompts]."""
    if not correctness:return 'Invalid correctness',{}
    mean=float(primary.mean());positive=int((primary.mean(axis=1)>0).sum())
    safe=all(float(v.mean())>=-.02 for v in safeguards.values())
    robust=mean>=.10 and positive>=3 and large['primary_lower']>0 and safe
    modest=0<mean<.10 and positive>=3 and large['primary_lower']>0 and safe
    d128=np.asarray(small).mean(axis=(0,2));d10000=np.asarray(large['factorial']).mean(axis=(0,2))
    b128=np.asarray(baseline_small).mean(axis=(0,2));b10000=np.asarray(baseline_large).mean(axis=(0,2))
    catches=bool(d128[0]>0 and d128[1]<d128[0] and d10000[1]<d10000[0] and b128[1]>b128[0] and b10000[1]>b10000[0]
                 and (mean<=0 or (large['primary_lower']>=-.02 and large['primary_upper']<=.02)))
    limited=bool((d128>0).all() and (d10000-d128<0).all())
    label=('Robust compact advantage' if robust else 'Small compact advantage' if modest else
           'Baseline catches up and data-limited' if catches and limited else 'Baseline catches up' if catches else
           'Data-limited advantage' if limited else 'Mixed')
    return label,dict(robust=robust,small=modest,catches_up=catches,data_limited=limited,larger_budget_safe=safe,
        positive_seeds=positive,seed_std=float(primary.mean(axis=1).std(ddof=1)),seed_min=float(primary.mean(axis=1).min()),
        seed_max=float(primary.mean(axis=1).max()),seed_median=float(np.median(primary.mean(axis=1))))

def contrasts(arrays,templates,correctness=True,replicates=5000,seed=5644):
    """arrays[(method,N,steps,seed,cap)] -> paired per-prompt correctness."""
    draws=bootstrap_indices(np.asarray(templates),replicates,seed)
    delta=lambda n,s,r,cap=116:arrays['h2',n,s,r,cap]-arrays['baseline',n,s,r,cap]
    primary=np.stack([delta(10000,5000,r) for r in range(42,47)])
    estimate=interval(primary.mean(axis=0),draws)
    small=np.stack([[delta(128,s,r) for s in (1000,5000)] for r in (42,43,44)])
    big=np.stack([[delta(10000,s,r) for s in (1000,5000)] for r in (42,43,44)])
    bsmall=np.stack([[arrays['baseline',128,s,r,116] for s in (1000,5000)] for r in (42,43,44)])
    bbig=np.stack([[arrays['baseline',10000,s,r,116] for s in (1000,5000)] for r in (42,43,44)])
    safeguards={cap:np.stack([delta(10000,5000,r,cap) for r in range(42,47)]) for cap in (296,400,640)}
    label,checks=classify(primary,small,dict(factorial=big,primary_lower=estimate['lower'],primary_upper=estimate['upper']),bsmall,bbig,safeguards,correctness)
    quantities={'delta128_1000':small[:,0],'delta128_5000':small[:,1],'delta10000_1000':big[:,0],'delta10000_5000':big[:,1],
        'data_effect1000':big[:,0]-small[:,0],'data_effect5000':big[:,1]-small[:,1],
        'step_effect128':small[:,1]-small[:,0],'step_effect10000':big[:,1]-big[:,0],
        'interaction':(big[:,1]-big[:,0])-(small[:,1]-small[:,0])}
    rows=[]
    for name,values in quantities.items():
        for i,r in enumerate((42,43,44)):rows.append(dict(contrast=name,seed=str(r),**interval(values[i],draws)))
        rows.append(dict(contrast=name,seed='equal_seed_mean',positive_seeds=int((values.mean(axis=1)>0).sum()),**interval(values.mean(axis=0),draws)))
    return dict(classification=label,primary=estimate,per_seed=primary.mean(axis=1).tolist(),checks=checks,
                prompt_scope='conditional on fixed trained models and nine-template mixture; no training-seed significance claim'),rows,draws
