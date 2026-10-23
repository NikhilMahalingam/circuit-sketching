"""Deterministic disjoint groups from the official canonical edge schema."""
import time
import numpy as np
import pandas as pd
from src.mattr_optimization.common import sha,write_json
from src.mattr_optimization.gate4_init import edge_mapping

def construct(graph,out):
    started=time.perf_counter();mapping=edge_mapping(graph,graph)
    mapping.to_csv(out/'edge_mapping.csv',index=False)
    summary=[];audit={}
    for kind in ('H1','H2'):
        if kind=='H1':labels=mapping.parent.astype(str)
        else:
            labels=mapping.child.astype(str).copy()
            attention=labels.str.startswith('a')
            assert mapping.loc[attention,'qkv'].isin(['q','k','v']).all()
            labels.loc[attention]=labels.loc[attention]+'<'+mapping.loc[attention,'qkv']+'>'
        ids,names=pd.factorize(labels,sort=True);ids=ids.astype(np.int64)
        counts=np.bincount(ids)
        assert len(ids)==32491 and len(counts)==len(names) and counts.min()>0
        np.save(out/f'{kind}_groups.npy',ids)
        shuffled=np.random.default_rng(5701).permutation(ids)
        np.save(out/f'{kind}_shuffled_groups.npy',shuffled)
        assert np.array_equal(np.bincount(shuffled),counts)
        frame=mapping.copy();frame['group_id']=ids;frame['group_name']=labels
        frame.to_csv(out/f'{kind}_edge_groups.csv',index=False)
        for i,name in enumerate(names):
            summary.append({'hierarchy':kind,'group_id':i,'component':name,'edges':int(counts[i]),
              'type':'attention' if name.startswith('a') else 'MLP' if name.startswith('m') else 'schema:'+name})
        audit[kind]={'edges':len(ids),'groups':len(counts),'smallest':int(counts.min()),'largest':int(counts.max()),
           'singletons':int((counts==1).sum()),'ids_sha256':sha(out/f'{kind}_groups.npy'),
           'shuffled_sha256':sha(out/f'{kind}_shuffled_groups.npy'),'shuffle_seed':5701,
           'fallback':'H1 input and H2 logits are explicit official schema components, not a shared unmatched bucket'}
    pd.DataFrame(summary).to_csv(out/'group_sizes.csv',index=False)
    result={'passed':True,'canonical_mapping_sha256':sha(out/'edge_mapping.csv'),'hierarchies':audit,
            'construction_seconds':time.perf_counter()-started,'source':'pinned official Graph parent/child/qkv and canonical real-mask order'}
    write_json(out/'mapping_audit.json',result);return result
