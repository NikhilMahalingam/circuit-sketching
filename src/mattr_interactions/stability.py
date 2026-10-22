"""Offline component stability, including a covariance-preserving size null."""
from itertools import combinations
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata,spearmanr
from src.mattr_optimization.gate3 import baseline
from .design import SEEDS,layer

def jaccard(a,b):
    return len(a&b)/len(a|b) if a|b else 1.

def analyze(out,mapping,graph):
    ranks=[];masks={};summary=[];overlaps=[]
    for seed in SEEDS:
        scores=torch.load(baseline(seed)/'final_scores.pt',map_location='cpu',weights_only=False)
        ranks.append((32491-rankdata(-scores.numpy(),method='average'))/32490)
        for fraction in (.005,.01,.02,.05,.1):
            graph.scores[:]=-torch.inf;graph.scores[graph.real_edge_mask.bool()]=scores
            graph.apply_topn(int(fraction*32491),absolute=False,level='edge',prune=True)
            masks[seed,fraction]=graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()
    ranks=np.array(ranks)
    groups={
      'source_head':mapping.parent.where(mapping.parent.str.startswith('a')),
      'destination_head':mapping.child.where(mapping.child.str.startswith('a')),
      'qkv_receiver':(mapping.child+'<'+mapping.qkv.fillna('')+'>').where(mapping.child.str.startswith('a')),
      'source_MLP':mapping.parent.where(mapping.parent.str.startswith('m')),
      'destination_MLP':mapping.child.where(mapping.child.str.startswith('m')),
      'source_layer':mapping.parent.map(layer).astype(str).where(mapping.parent.map(layer)>=0),
      'destination_layer':mapping.child.map(layer).astype(str).where(mapping.child.map(layer)>=0)}
    pair_indices=list(combinations(range(3),2));records=[];null_records=[]
    edge_rho=float(np.mean([spearmanr(ranks[a],ranks[b]).statistic for a,b in pair_indices]))
    for a,b in pair_indices:
        overlaps.append({'grouping':'edge','seed_a':SEEDS[a],'seed_b':SEEDS[b],'rank_spearman':spearmanr(ranks[a],ranks[b]).statistic,
                         'top10pct_jaccard':jaccard(set(np.argsort(-ranks[a])[:3249]),set(np.argsort(-ranks[b])[:3249]))})
        for f in (.005,.01,.02,.05):
            records.append({'grouping':'edge','fraction':f,'seed_a':SEEDS[a],'seed_b':SEEDS[b],
                            'jaccard':jaccard(set(np.flatnonzero(masks[SEEDS[a],f])),set(np.flatnonzero(masks[SEEDS[b],f])))})
    permutations=[np.random.default_rng(5342+i).permutation(32491) for i in range(200)]
    for name,labels in groups.items():
        codes,names=pd.factorize(labels,sort=True);valid=codes>=0;size=np.bincount(codes[valid]);n=len(names)
        def aggregate(values):return np.array([np.bincount(codes[valid],weights=x[valid],minlength=n)/size for x in values])
        means=aggregate(ranks)
        observed=float(np.mean([spearmanr(means[a],means[b]).statistic for a,b in pair_indices]))
        null=[]
        for perm in permutations:
            mixed=aggregate(ranks[:,perm])
            null.append(np.mean([spearmanr(mixed[a],mixed[b]).statistic for a,b in pair_indices]))
        null_records.append({'grouping':name,'components':n,'edge_mean_spearman':edge_rho,'component_mean_spearman':observed,
                             'component_gain':observed-edge_rho,'common_permutation_null_mean':np.mean(null),
                             'common_permutation_null_95':np.quantile(null,.95),'above_null_95':observed>np.quantile(null,.95)})
        for a,b in pair_indices:
            top=max(1,int(np.ceil(.1*n)))
            overlaps.append({'grouping':name,'seed_a':SEEDS[a],'seed_b':SEEDS[b],
                             'rank_spearman':spearmanr(means[a],means[b]).statistic,
                             'top10pct_jaccard':jaccard(set(np.argsort(-means[a])[:top]),set(np.argsort(-means[b])[:top]))})
        for si,seed in enumerate(SEEDS):
            scores=torch.load(baseline(seed)/'final_scores.pt',map_location='cpu',weights_only=False).numpy()
            frame=pd.DataFrame({'group':labels,'score':scores,'percentile':ranks[si]})
            dist=frame.groupby('group').agg(score_mean=('score','mean'),score_std=('score','std'),score_min=('score','min'),score_max=('score','max'),rank_percentile_median=('percentile','median'),rank_percentile_std=('percentile','std'))
            for f in (.005,.01,.02,.05):
                counts=np.bincount(codes[valid],weights=masks[seed,f][valid],minlength=n)
                rate=counts/size;global_rate=masks[seed,f].mean()
                for k,label in enumerate(names):
                    summary.append({'grouping':name,'component':label,'seed':seed,'fraction':f,'eligible_edges':int(size[k]),
                        'retained_edges':int(counts[k]),'retained_fraction':rate[k],'enrichment':rate[k]/max(global_rate,1e-9),
                        'mean_rank_percentile':means[si,k],**dist.loc[label].to_dict()})
    pd.DataFrame(summary).to_csv(out/'component_statistics.csv',index=False)
    pd.DataFrame(overlaps).to_csv(out/'ranking_stability.csv',index=False)
    pd.DataFrame(records).to_csv(out/'edge_membership_stability.csv',index=False)
    pd.DataFrame(null_records).to_csv(out/'stability_null_comparison.csv',index=False)
    # Historical EAP-IG is observational, using only already-existing scores.
    from src.mattr_optimization.common import ROOT
    refs=[]
    for si,seed in enumerate(SEEDS):
        p=ROOT/f'results/mattr_optimization/20261022_gate4/attribution_seed{seed}/eap_ig_scores.pt'
        if p.exists():
            e=torch.load(p,map_location='cpu',weights_only=False).numpy()
            refs.append({'seed':seed,'reference':'saved official EAP-IG','signed_rank_spearman':spearmanr(ranks[si],e).statistic,
                         'top1pct_raw_ranking_jaccard':jaccard(set(np.argsort(-ranks[si])[:324]),set(np.argsort(-e)[:324]))})
    pd.DataFrame(refs).to_csv(out/'historical_attribution_reference.csv',index=False)
