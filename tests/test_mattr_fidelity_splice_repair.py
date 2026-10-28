"""Regression tests exercise the frozen real structural extractor, not a mock."""
import json
import numpy as np
import pandas as pd
import pytest
import torch

from src.mattr_fidelity_splice import structure
from src.mattr_fidelity_splice.repair_analysis import corrected_overlap_counts, assert_same_decision


def test_real_features_duplicate_count_repair(tmp_path, monkeypatch):
    size = 400
    mapping = pd.DataFrame(dict(parent=['input','a9.h9','input','a5.h5']*(size//4),
        child=['a9.h9','logits','a5.h5','logits']*(size//4), qkv=['q','','v','']*(size//4),
        edge=[f'edge{i}' for i in range(size)]))
    mapping.to_csv(tmp_path/'edge_mapping.csv', index=False)
    history=tmp_path/'history';history.mkdir()
    (history/'known_ioi_heads.json').write_text(json.dumps({'classes':{'name mover':[[9,9]],'induction':[[5,5]]}}))
    masks=tmp_path/'masks';masks.mkdir()
    records=[dict(key=k,method=m,kind='core',seed=42,dataset_size=10000,step=5000 if m=='baseline' else 0)
        for k,m in [('baseline','baseline'),('eap','eap_ig')]]
    queries=[]
    for target in (116,296):
        mask=np.arange(size)<target;np.save(masks/f'mask{target}.npy',mask)
        for r in records:queries.append(dict(ranking=r['key'],selection='cap',target=target,
            mask_sha256=f'mask{target}',nominal_n=target+10,actual_edges=target))
    pd.DataFrame(queries).to_csv(tmp_path/'queries.csv',index=False)
    monkeypatch.setattr(structure,'HISTORY',history)
    monkeypatch.setattr(structure,'graph_new',lambda:object())
    monkeypatch.setattr(structure,'load_scores',lambda rec:torch.arange(size,0,-1,dtype=torch.float32))
    monkeypatch.setattr(structure,'official_order',lambda graph,scores:np.argsort(-scores.numpy()))
    original=structure.overlap
    # The underlying frozen code remains reproducibly identifiable, including its bug.
    with pytest.raises(TypeError,match='actual_edges'):structure.features(tmp_path,{'rankings':records})
    with corrected_overlap_counts():
        frame=structure.features(tmp_path,{'rankings':records})
    assert structure.overlap is original
    assert frame.actual_edges.tolist()==[116,296,116,296]
    assert frame.complete_name_mover_path.all()
    assert frame.known_heads.tolist()==[2,2,2,2]
    assert frame.eap_jaccard.eq(1).all()
    assert (tmp_path/'ranking_features.csv').exists()


def test_bad_overlap_count_is_not_silently_removed(monkeypatch):
    def bad(mapping, mask, classes):return {'actual_edges':999}
    monkeypatch.setattr(structure,'overlap',bad)
    with pytest.raises(ValueError,match='edge count'):
        with corrected_overlap_counts():structure.overlap(None,np.ones(2,dtype=bool),None)
    assert structure.overlap is bad


def test_registered_decision_replay_is_strict():
    assert_same_decision({'label':'same','gain':[.1,True]}, {'label':'same','gain':[.1+1e-14,True]})
    with pytest.raises(AssertionError):assert_same_decision({'label':'same'}, {'label':'changed'})
    with pytest.raises(AssertionError):assert_same_decision({'gain':.1}, {'gain':.11})
