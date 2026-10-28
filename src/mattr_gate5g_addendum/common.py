import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import ROOT,GRID,MIB,configure,git,sha,write_json,integration_weights,provenance
from src.mattr_fidelity_splice.core import read,digest,marker,verified,check_files,ArrayCache,attempt
from src.mattr_fidelity_splice.core import protocol as original_protocol
from src.mattr_fidelity_splice.analysis import completeness,table
from src.mattr_fidelity_splice.design import load_scores,positive_scores
from src.mattr_matched_size.topology import graph_new,official_order,digest as mask_digest

OLD=ROOT/'results/mattr_fidelity_splice/gate5g'
DEFAULT=ROOT/'results/mattr_gate5g_addendum/gate5g_addendum'
FRACTIONS=[.1,.2,.5]
EPS=1e-8

def sources():
    paths=list((ROOT/'src/mattr_gate5g_addendum').glob('*.py'))+list((ROOT/'scripts/gate5g_addendum').glob('*.sh'))+[ROOT/'tests/test_mattr_gate5g_addendum.py']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}

def original():
    p=original_protocol(OLD);completeness(OLD,p)
    decision=read(OLD/'gate5g_decision.json')
    assert decision['part_a']=='EAP-IG closer' and decision['part_c']['classification']=='Compact gain with CPR cost'
    assert read(OLD/'final_audit.json')['passed'] and read(OLD/'analysis_repair.json')['status']=='passed'
    assert p['grid']==list(GRID)
    return p

def record(p,key):return next(r for r in p['rankings'] if r['key']==key)
def core(p,method,seed):return next(r for r in p['rankings'] if r['kind']=='core' and r['method']==method and r['seed']==seed)
def order(rec):return official_order(graph_new(),load_scores(rec))
def summary(key):return read(OLD/'evaluations'/key/'summary.json')
def prompts(key):return pd.read_csv(OLD/'evaluations'/key/'per_prompt.csv')
def vector(frame,selection,target,col):
    return frame[(frame.selection==selection)&np.isclose(frame.target,target)].sort_values('example')[col].to_numpy()
def query(key,selection,target):
    f=pd.read_csv(OLD/'queries.csv');return f[(f.ranking==key)&(f.selection==selection)&np.isclose(f.target,target)].iloc[0]
def oldmask(q):
    m=np.load(OLD/'masks'/f'{q.mask_sha256}.npy');assert mask_digest(m)==q.mask_sha256 and m.sum()==q.actual_edges;return m

def explicit(selected):
    """Unique positive order, exact unpruned set, then unchanged official prune."""
    selected=np.asarray(selected,dtype=bool)
    assert selected.shape==(32491,)
    indices=np.concatenate([np.flatnonzero(selected),np.flatnonzero(~selected)])
    scores=positive_scores(indices);graph=graph_new()
    from src.mattr_compact_validation.inference import assign_scores
    assign_scores(graph,scores);n=int(selected.sum())
    graph.apply_topn(n,absolute=False,prune=False,level='edge')
    np.testing.assert_array_equal(graph.in_graph[graph.real_edge_mask.bool()].numpy(),selected)
    graph.prune();mask=graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()
    assert not np.any(mask&~selected)
    return scores,mask

def save_set(out,key,selected,**meta):
    scores,mask=explicit(selected);folder=out/'circuits';folder.mkdir(exist_ok=True)
    path=folder/(key+'.npz');np.savez_compressed(path,selected=selected,mask=mask,scores=scores.numpy())
    return dict(id=key,path=path.relative_to(out).as_posix(),sha256=sha(path),requested_edges=int(selected.sum()),
        actual_edges=int(mask.sum()),mask_sha256=mask_digest(mask),**meta)

def check_set(out,rec):
    assert sha(out/rec['path'])==rec['sha256']
    with np.load(out/rec['path']) as data:
        scores,mask=explicit(data['selected']);np.testing.assert_array_equal(scores.numpy(),data['scores'])
        np.testing.assert_array_equal(mask,data['mask'])
    assert mask_digest(mask)==rec['mask_sha256'];return scores,mask

def freeze_json(path,obj):
    path=Path(path)
    if path.exists():raise FileExistsError('Immutable protocol already exists')
    path.write_text(json.dumps(obj,sort_keys=True,separators=(',',':')),encoding='utf-8')
    path.with_suffix('.sha256').write_text(sha(path)+'\n')

def checked_json(path):
    assert sha(path)==Path(path).with_suffix('.sha256').read_text().strip()
    return read(path)

def stage2_protocol(out,live=False):
    p=checked_json(out/'stage2_protocol.json');check_files(p['sources']);check_files(p['artifacts'],out)
    original_protocol(OLD)
    assert sha(OLD/'gate5g_protocol.json')==p['original_protocol_sha256']
    for rec in p['circuits']:check_set(out,rec)
    if live:
        assert git(ROOT,'rev-parse','HEAD').decode().strip()==p['implementation_commit'],'Frozen implementation HEAD mismatch'
        assert not git(ROOT,'status','--porcelain','--',*p['sources']).strip(),'Tracked experiment source dirty'
        assert provenance()==p['dependencies'],'Pinned runtime differs'
    return p
