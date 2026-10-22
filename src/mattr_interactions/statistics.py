"""Prompt-resampled uncertainty; no pair-row pseudo-replication."""
import numpy as np
import pandas as pd

def bootstrap_means(values,seed=5242,repetitions=2000,alpha=.05):
    x=np.asarray(values,dtype=float)
    if x.ndim==1:x=x[None,:]
    draws=np.random.default_rng(seed).integers(0,x.shape[1],size=(repetitions,x.shape[1]))
    means=x[:,draws].mean(axis=2)
    return np.quantile(means,[alpha/2,1-alpha/2],axis=1).T

def summarize(pairs,per_example,confirmation=False):
    rows=[];m=len(pairs)
    for pair in pairs.itertuples():
        x=per_example[per_example.pair_id==pair.pair_id]
        row=pair._asdict();row.pop('Index',None)
        for field in ('single_i','single_j','joint','interaction','accuracy_joint','accuracy_i','accuracy_j'):
            v=x[field].to_numpy();lo,hi=bootstrap_means(v)[0]
            row[field]=float(v.mean());row[field+'_lo']=float(lo);row[field+'_hi']=float(hi)
        ci=bootstrap_means(x.interaction,alpha=.05/max(1,m))[0]
        row['interaction_familywise_lo'],row['interaction_familywise_hi']=ci
        row['additive']=row['single_i']+row['single_j']
        row['abs_interaction']=abs(row['interaction'])
        row['relative_interaction']=abs(row['interaction'])/max(.1,abs(row['single_i'])+abs(row['single_j']))
        row['sign_flip_vs_additive']=np.sign(row['joint'])!=np.sign(row['additive'])
        row['joint_exceeds_both_singles']=row['joint']>max(row['single_i'],row['single_j'])
        row['substantial']=row['abs_interaction']>=.1 and row['relative_interaction']>=.2
        row['confirmation']=confirmation;rows.append(row)
    return pd.DataFrame(rows)

def select_confirmation(screen):
    selected=[]
    for (_,direction),group in screen.groupby(['fraction','direction'],sort=True):
        main=group.sort_values(['abs_interaction','pair_id'],ascending=[False,True]).head(3)
        remaining=group[~group.pair_id.isin(main.pair_id)]
        extra=remaining.sort_values(['joint','pair_id'],ascending=[False,True]).head(1) if direction=='addition' else remaining.assign(magnitude=remaining.joint.abs()).sort_values(['magnitude','pair_id'],ascending=[False,True]).head(1)
        selected.extend(main.pair_id);selected.extend(extra.pair_id)
    return selected
