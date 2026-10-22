"""Official evaluator with immutable donor-only caching and exact result identities."""
from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import sha,write_json
from .design import assign,cleanup,digest,toggled

class DonorCache:
    """Cache only the official matrix after a completed corrupted forward.

    The untouched official clean hooks subtract LIVE patched source activations
    from a new matrix on every query. No frozen-clean/sender cache is used.
    """
    def __init__(self,model,limit=2*2**30):
        self.model=model;self.limit=limit;self.values=OrderedDict();self.bytes=0
        self.hits=0;self.misses=0;self.pending=None;self.next_is_donor=True

    @contextmanager
    def install(self):
        import eap.evaluate as official
        builder=official.make_hooks_and_matrices;forward=self.model.forward
        def make(*args,**kw):
            hooks,matrix=builder(*args,**kw)
            self.pending=matrix;self.next_is_donor=True
            return hooks,matrix
        def run(tokens,*args,**kw):
            if not self.next_is_donor:return forward(tokens,*args,**kw)
            self.next_is_donor=False
            am=kw.get('attention_mask')
            key=(id(self.model),tuple(self.pending.shape),str(tokens.dtype),
                 hashlib.sha256(tokens.detach().cpu().numpy().tobytes()).hexdigest(),
                 None if am is None else hashlib.sha256(am.detach().cpu().numpy().tobytes()).hexdigest())
            if key in self.values:
                self.hits+=1;cached=self.values.pop(key);self.values[key]=cached
                self.pending.copy_(cached);return None # corrupted_logits is unused by evaluate_graph
            self.misses+=1
            result=forward(tokens,*args,**kw)
            saved=self.pending.detach().cpu().clone();size=saved.numel()*saved.element_size()
            if size<=self.limit:
                while self.bytes+size>self.limit:
                    _,old=self.values.popitem(last=False);self.bytes-=old.numel()*old.element_size()
                self.values[key]=saved;self.bytes+=size
            return result
        with patch.object(official,'make_hooks_and_matrices',make),patch.object(self.model,'forward',run):
            yield

class Evaluator:
    def __init__(self,model,graph,out,meter,protocol_hash):
        from MIB_circuit_track.metrics import get_metric
        self.model=model;self.graph=graph;self.out=Path(out);self.meter=meter
        self.identity={'protocol_sha256':protocol_hash,'model':'pinned GPT2 float32','batch_size':4}
        self.metrics=[partial(get_metric(name,'ioi',model.tokenizer,model),mean=False,loss=False) for name in ('logit_diff','acc')]
        self.donors=DonorCache(model);self.cache_hits=0;self.unique=set()

    def evaluate(self,mask,dataset,cohort,use_cache=True,donor_cache=True):
        from eap.evaluate import evaluate_graph
        desired=np.asarray(mask,dtype=bool)
        actual=cleanup(self.graph,desired)
        if not np.array_equal(desired,actual):raise ValueError('Automatic cleanup would change extra edges; excluded intervention.')
        identity={**self.identity,'mask_sha256':digest(desired),'cohort':cohort,'dataset_fingerprint':dataset.dataset._fingerprint}
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        path=self.out/'exact_cache'/f'{key}.npz';meta=path.with_suffix('.json')
        if use_cache and path.exists():
            assert json.loads(meta.read_text())==identity
            self.cache_hits+=1
            with np.load(path) as saved:return saved['margin'].copy(),saved['accuracy'].copy()
        assign(self.graph,desired)
        with torch.no_grad():
            if donor_cache:
                with self.donors.install():values=evaluate_graph(self.model,self.graph,dataset.to_dataloader(4),self.metrics,quiet=True)
            else:values=evaluate_graph(self.model,self.graph,dataset.to_dataloader(4),self.metrics,quiet=True)
        result=tuple(x.detach().cpu().numpy().copy() for x in values)
        if any(not np.isfinite(x).all() for x in result):raise ValueError('Nonfinite official metric.')
        assert np.array_equal(self.graph.in_graph[self.graph.real_edge_mask.bool()].numpy(),desired)
        if use_cache:
            path.parent.mkdir(exist_ok=True);np.savez_compressed(path,margin=result[0],accuracy=result[1]);write_json(meta,identity)
        self.unique.add(key)
        return result

def four_way(evaluator,base,pair,dataset,cohort):
    return [evaluator.evaluate(toggled(base,indices),dataset,cohort) for indices in ([],[int(pair.i)],[int(pair.j)],[int(pair.i),int(pair.j)])]

def pair_effects(values):
    (base,_),(a,aa),(b,ab),(joint,aj)=values
    return {'single_i':a-base,'single_j':b-base,'joint':joint-base,
            'interaction':joint-a-b+base,'accuracy_joint':aj-values[0][1],
            'accuracy_i':aa-values[0][1],'accuracy_j':ab-values[0][1]}
