"""Known-head overlap with layer-preserving graph-automorphism controls."""
import json
import re
import time
import numpy as np
import pandas as pd
from src.mattr_optimization.common import write_json
from src.mattr_matched_size.topology import graph_new

def head(x):
    match=re.fullmatch(r'a(\d+)\.h(\d+)',str(x))
    return tuple(map(int,match.groups())) if match else None

def permutation_maps(mapping,seed=5801,replicates=64):
    rng=np.random.default_rng(seed);keys=list(zip(mapping.parent,mapping.child,mapping.qkv))
    lookup={x:i for i,x in enumerate(keys)};result=[]
    for _ in range(replicates):
        perm=[rng.permutation(12) for l in range(12)]
        def rename(x):
            h=head(x);return f'a{h[0]}.h{perm[h[0]][h[1]]}' if h else x
        indices=np.array([lookup[rename(a),rename(b),c] for a,b,c in keys])
        assert np.array_equal(np.sort(indices),np.arange(len(keys)))
        result.append(indices)
    return result

def overlap(mapping,mask,classes):
    rows=mapping.iloc[np.flatnonzero(mask)]
    pairs=[(head(a),head(b)) for a,b in zip(rows.parent,rows.child)]
    represented={x for pair in pairs for x in pair if x is not None}
    known={tuple(x) for values in classes.values() for x in values}
    data=dict(actual_edges=int(mask.sum()),known_heads=len(represented&known),other_heads=len(represented-known),
        incident_known_fraction=sum(a in known or b in known for a,b in pairs)/max(1,len(pairs)))
    for role,values in classes.items():
        found=represented&set(map(tuple,values));data[role+'_count']=len(found);data[role+'_recall']=len(found)/len(values)
    return data

def analyze(out,records):
    start=time.perf_counter();mapping=pd.read_csv(out/'edge_mapping.csv',keep_default_na=False)
    classes=json.loads((out/'known_ioi_heads.json').read_text())['classes']
    maps=permutation_maps(mapping);graph=graph_new();real=graph.real_edge_mask.bool()
    observed=[];control=[];cache={}
    for rec,folder in records:
        with np.load(folder/'cap_masks.npz') as saved:
            for target in saved.files:
                mask=saved[target];key=np.packbits(mask).tobytes()
                original=overlap(mapping,mask,classes)
                observed.append({**rec,'target':int(target),**original})
                if key not in cache:
                    values=[]
                    for rep,perm in enumerate(maps):
                        transformed=np.zeros(32491,dtype=bool);transformed[perm[np.flatnonzero(mask)]]=True
                        graph.in_graph[:]=False;graph.in_graph[real]=__import__('torch').from_numpy(transformed)
                        graph.prune();actual=graph.in_graph[real].numpy().copy()
                        assert np.array_equal(actual,transformed) and actual.sum()==mask.sum()
                        # Whole-node relabeling preserves source/destination types, Q/K/V and connectivity exactly.
                        values.append(dict(replicate=rep,**overlap(mapping,actual,classes)))
                    cache[key]=values
                for value in cache[key]:control.append({**rec,'target':int(target),**value,'count_shortfall':0})
    pd.DataFrame(observed).to_csv(out/'known_head_overlap.csv',index=False)
    pd.DataFrame(control).to_csv(out/'topology_random_controls.csv',index=False)
    write_json(out/'topology_control_audit.json',dict(passed=True,seconds=time.perf_counter()-start,unique_masks=len(cache),
        exact_size_match=True,replicates=64,matching='within-layer head relabeling automorphism; same layer/head degree pattern, QKV roles and source/destination edge types',
        limitation='conditions on layer structure and head counts. Enrichment tests identity within layers, not preference for known layers. No model forwards; descriptive only.'))
