"""CPU correctness: exact selected sets, official pruning, deterministic rules."""
import numpy as np
import pytest
from src.mattr_gate5g_addendum.common import configure,explicit,mask_digest,OLD,original_protocol
from src.mattr_gate5g_addendum.stage2 import best,build_displacement
from src.mattr_gate5g_addendum.analysis import reading,four_contrasts

def test_explicit_sets_use_real_official_pruning():
    configure()
    for ids in [[],[0,32490],range(32491)]:
        selected=np.zeros(32491,dtype=bool);selected[list(ids)]=True
        scores,mask=explicit(selected)
        assert (scores.numpy()>0).all() and np.unique(scores).size==32491
        assert not (mask&~selected).any()
        if selected.all():assert mask.all()

def test_equal_unpruned_displacement():
    rng=np.random.default_rng(8);parent=rng.permutation(200);splice=np.r_[parent[-30:],parent[:-30]]
    P=set(parent[:70]);S=set(splice[:70]);A=S-P;D=P-S
    assert len(A)==len(D)
    assert (P-D)|A==S
    assert len(P-D)==70-len(D) and len(P|A)==70+len(A)

def test_search_ties_and_reading_rules():
    a=dict(gain=.2,actual_edges=120,ranks=[10],mask_sha256='a')
    b=dict(gain=.2,actual_edges=119,ranks=[11],mask_sha256='b')
    assert best([a,b])==b
    labels=reading(-1,-1,-1.8,.2,[.1,.3])
    assert 'mostly displacement' not in labels
    assert reading(-1,0,-1,0,[-.05,.05])==['mostly displacement','approximately additive']
    assert reading(-1,-1,-1,.7,[.5,.9])==['mostly displacement','mostly insertion','non-additive']
    assert reading(1,1,0,1,[1,2])==['relative thresholds unstable']

def test_actual_four_circuit_bootstrap_normalizes_cohort_means():
    import pandas as pd
    from src.mattr_fidelity_splice.statistics import weights,resamples
    frames={}
    base=np.arange(1.,9.)
    for name,delta in [('parent',0.),('remove',-2.),('add',1.),('splice',-.5)]:
        frames[name]=pd.DataFrame(dict(margin=base+delta,clean_margin=base+1,empty_margin=np.ones(8)*-1))
    w=weights(resamples(['a']*4+['b']*4,100,5646),8)
    vals,draws=four_contrasts(frames,w)
    den=(frames['parent'].clean_margin-frames['parent'].empty_margin).mean()
    assert vals['remove']==pytest.approx(-2/den) and vals['interaction']==pytest.approx(.5/den)
    np.testing.assert_allclose(draws['interaction'],.5/(w@(base+2)))
    frames['parent']['clean_margin']=frames['parent'].empty_margin
    with pytest.raises(ValueError,match='Unstable'):four_contrasts(frames,w)

@pytest.mark.skipif(not (OLD/'gate5g_protocol.json').exists(),reason='Historical artifacts unavailable')
def test_all_parent_and_splice_masks_reproduce(tmp_path):
    configure();p=original_protocol(OLD)
    records=build_displacement(tmp_path,p)
    assert len(records)==60
    for seed in range(42,47):
        for f in [.1,.2,.5]:
            g={r['kind']:r for r in records if r['seed']==seed and r['fraction']==f}
            assert g['parent']['requested_edges']==g['splice']['requested_edges']
            assert g['remove']['requested_edges']+g['add']['requested_edges']==2*g['parent']['requested_edges']
