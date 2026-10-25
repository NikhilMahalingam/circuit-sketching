"""Exploratory graph diagnostics only; no fitting or outcome-dependent tuning."""
import re
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.mattr_data_scaling.structure import overlap
from src.mattr_matched_size.topology import graph_new,official_order
from .core import HISTORY, read
from .design import load_scores


def complete_path(mapping,mask,known):
    rows=mapping.iloc[np.flatnonzero(mask)]
    parents={};children={}
    for row in rows.itertuples():
        children.setdefault(row.parent,set()).add(row.child)
        parents.setdefault(row.child,set()).add(row.parent)
    def reachable(start,edges):
        seen={start};queue=[start]
        while queue:
            for v in edges.get(queue.pop(),()):
                if v not in seen:seen.add(v);queue.append(v)
        return seen
    forward=reachable('input',children);backward=reachable('logits',parents)
    return bool(set(known)&forward&backward)


def features(out,p):
    start=time.perf_counter();mapping=pd.read_csv(out/'edge_mapping.csv',keep_default_na=False)
    queries=pd.read_csv(out/'queries.csv')
    classes=read(HISTORY/'known_ioi_heads.json')['classes']
    eap=next(r for r in p['rankings'] if r['method']=='eap_ig')
    graph=graph_new();eo=official_order(graph,load_scores(eap));er=np.argsort(eo)
    names=[f'a{l}.h{h}' for l,h in classes['name mover']]
    rows=[];edges=[]
    for rec in p['rankings']:
        if rec['kind']=='splice':continue
        scores=load_scores(rec).numpy();order=official_order(graph,load_scores(rec))
        rank=np.argsort(order)
        rho=float(spearmanr(rank,er).statistic)
        scale=max(float(np.std(scores)),1e-12)
        gaps={f'normalized_score_gap_{n}':float((scores[order[n-1]]-scores[order[n]])/scale) for n in (116,324)}
        for target in (116,296):
            q=queries[(queries.ranking==rec['key'])&(queries.selection=='cap')&(queries.target==target)].iloc[0]
            mask=np.load(out/'masks'/f'{q.mask_sha256}.npy')
            eq=queries[(queries.ranking==eap['key'])&(queries.selection=='cap')&(queries.target==target)].iloc[0]
            em=np.load(out/'masks'/f'{eq.mask_sha256}.npy')
            selected=mapping.iloc[np.flatnonzero(mask)]
            groups=selected.child.astype(str)+'<'+selected.qkv.astype(str)+'>'
            dist=groups.value_counts(normalize=True).to_numpy()
            data=dict(ranking=rec['key'],method=rec['method'],seed=rec['seed'],dataset_size=rec['dataset_size'],step=rec['step'],target=target,
                canonical_order_sha256=__import__('hashlib').sha256(order.tobytes()).hexdigest(),
                nominal_prefix=int(q.nominal_n),actual_edges=int(q.actual_edges),pruned_edges=int(q.nominal_n-q.actual_edges),
                pruned_fraction=float(1-q.actual_edges/max(1,q.nominal_n)),destination_count=len(dist),
                destination_concentration=float((dist**2).sum()),destination_entropy=float(-(dist*np.log(dist)).sum()),
                complete_name_mover_path=complete_path(mapping,mask,names),
                eap_jaccard=float((mask&em).sum()/max(1,(mask|em).sum())),full_signed_spearman_eap=rho,**gaps,
                **overlap(mapping,mask,classes))
            for layer in range(12):
                data[f'layer_{layer}_fraction']=float(selected.child.str.match(fr'(a|m){layer}(\.|$)').mean()) if len(selected) else 0.
            rows.append(data)
            if rec['kind']=='core' and rec['method']=='baseline' and rec['seed'] in (42,44,46):
                for row in selected.itertuples():edges.append(dict(ranking=rec['key'],target=target,edge=row.edge,parent=row.parent,child=row.child,qkv=row.qkv))
    frame=pd.DataFrame(rows);frame.to_csv(out/'ranking_features.csv',index=False)
    pd.DataFrame(edges).to_csv(out/'seed42_edge_case_study.csv',index=False)
    pd.DataFrame([dict(key=k,known_head_class=k,heads=str(v)) for k,v in classes.items()]).to_csv(out/'known_head_reference.csv',index=False)
    pd.DataFrame([dict(seconds=time.perf_counter()-start,model_forwards=0,scope='topology features; exploratory and dependent checkpoints')]).to_csv(out/'graph_analysis_compute.csv',index=False)
    return frame
