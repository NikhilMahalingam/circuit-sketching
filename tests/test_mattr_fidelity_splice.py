"""Gate5G unit tests are synthetic/CPU; no downloaded model or fresh-prompt forwards."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from src.mattr_fidelity_splice.core import ArrayCache, digest, marker, verified, check_files
from src.mattr_fidelity_splice.design import positive_scores, splice_order
from src.mattr_fidelity_splice.inference import full_kl
from src.mattr_fidelity_splice.statistics import resamples,weights,official_areas,cpr_bootstrap,classify_a,classify_c
from src.mattr_fidelity_splice.structure import complete_path


def test_kl_independent_double_oracle():
    rng=np.random.default_rng(42)
    a=rng.normal(size=(7,91));b=rng.normal(size=(7,91))
    clean=torch.log_softmax(torch.tensor(a,dtype=torch.float32),-1)
    actual=full_kl(clean,torch.tensor(b,dtype=torch.float32)).numpy()
    from scipy.special import log_softmax
    lp=log_softmax(a,axis=-1);lq=log_softmax(b,axis=-1)
    np.testing.assert_allclose(actual,(np.exp(lp)*(lp-lq)).sum(-1),atol=5e-7,rtol=2e-6)
    np.testing.assert_allclose(full_kl(clean,torch.tensor(a)),0,atol=1e-7)


def test_splice_order_and_positive_export():
    e=np.array([4,2,0,1,3]);p=np.array([3,2,1,4,0])
    order=splice_order(e,p,2)
    np.testing.assert_array_equal(order,[4,2,3,1,0])
    scores=positive_scores(order)
    assert bool((scores>0).all()) and len(torch.unique(scores))==5
    np.testing.assert_array_equal(torch.argsort(scores,descending=True),order)
    np.testing.assert_array_equal(torch.argsort(scores.abs(),descending=True),order)


def test_bootstrap_pairing_and_strata():
    labels=np.array(['a']*5+['b']*3+['c']*2)
    ids=resamples(labels,100,5645)
    np.testing.assert_array_equal(ids,resamples(labels,100,5645))
    for row in ids:assert list(np.unique(labels[row],return_counts=True)[1])==[5,3,2]
    w=weights(ids,len(labels));np.testing.assert_allclose(w.sum(-1),1)
    values=np.arange(10.)
    np.testing.assert_allclose(w@values,values[ids].mean(-1))


def test_cpr_normalization_after_cohort_means():
    clean=np.array([1.,4.,8.]);empty=np.array([-1.,1.,2.])
    margins=np.arange(30.).reshape(10,3)
    w=weights(resamples(['a']*3,20),3)
    val,unstable=cpr_bootstrap(margins,clean,empty,w)
    for i in range(20):
        expected=official_areas(margins@w[i],clean@w[i],empty@w[i])[0]
        assert val[i]==pytest.approx(expected)
    assert not unstable.any()
    val,unstable=cpr_bootstrap(margins,clean,clean,w)
    assert unstable.all() and np.isnan(val).all()


def test_registered_decisions():
    w=weights(resamples(['a']*10,50),10)
    baseline=np.ones((5,3,10))*2
    assert classify_a(np.ones((3,10)),baseline,w)[0]=='EAP-IG closer'
    assert classify_a(np.ones((3,10))*3,baseline,w)[0]=='MAttr closer'
    assert classify_a(np.ones((3,10))*2,baseline,w)[0]=='Mixed'
    good=classify_c([0]*5,np.zeros((5,50)),np.ones((5,10))*.2,w)
    assert good['classification']=='Best of both'
    assert classify_c([-.1]*5,np.ones((5,50))*-.1,np.ones((5,10))*.2,w)['classification']=='Compact gain with CPR cost'
    assert classify_c([0]*5,np.zeros((5,50)),np.ones((5,10))*.05,w)['classification']=='No useful compact gain'


def test_cache_identity_integrity_and_interrupted_resume(tmp_path):
    count=[0]
    def compute():count[0]+=1;return dict(kl=np.array([1.,2.]))
    cache=ArrayCache(tmp_path,dict(model='a',cohort='x',position=-1))
    cache.get('mask',compute);cache.get('mask',compute)
    assert count[0]==1 and cache.hits==1
    ArrayCache(tmp_path,dict(model='b',cohort='x',position=-1)).get('mask',compute)
    assert count[0]==2
    path=next(tmp_path.glob('*.npz'));path.write_bytes(b'corrupt')
    with pytest.raises(ValueError):cache.get('mask',compute)
    for p in tmp_path.glob('*.json'):p.unlink()
    cache.get('mask',compute)
    assert (tmp_path/'incomplete').is_dir()


def test_completed_output_hash_enforcement(tmp_path):
    path=tmp_path/'metrics.csv';path.write_text('test')
    marker(tmp_path,{'implementation_commit':'a'},[path])
    assert verified(tmp_path,{'implementation_commit':'a'})
    with pytest.raises(ValueError):verified(tmp_path,{'implementation_commit':'b'})
    path.write_text('changed')
    with pytest.raises(ValueError):verified(tmp_path)


def test_complete_name_mover_path():
    import pandas as pd
    frame=pd.DataFrame(dict(parent=['input','a9.h9','input'],child=['a9.h9','logits','m0']))
    assert complete_path(frame,[True,True,False],['a9.h9'])
    assert not complete_path(frame,[True,False,True],['a9.h9'])


def test_official_positive_masks_small_prefixes():
    from src.mattr_optimization.common import configure
    configure()
    from src.mattr_matched_size.topology import graph_new,official_order
    from src.mattr_fidelity_splice.design import selected
    graph=graph_new();scores=torch.arange(32491,dtype=torch.float32).sin()
    order=official_order(graph,scores);export=positive_scores(order)
    for n in (32,64,162,324,1624,32491):
        a=selected(graph,scores,n);b=selected(graph,export,n);c=selected(graph,export,n,True)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(b,c)


def test_code_lock_and_post_results_analysis(tmp_path,monkeypatch):
    from src.mattr_fidelity_splice import core
    from src.mattr_optimization.common import sha
    code=tmp_path/'code.py';code.write_text('original')
    p=dict(implementation_commit='implementation',sources={'code.py':sha(code)},artifacts={},rankings=[],dependencies={})
    path=tmp_path/'gate5g_protocol.json';path.write_text(json.dumps(p,sort_keys=True,separators=(',',':')))
    (tmp_path/'gate5g_protocol.sha256').write_text(sha(path))
    monkeypatch.setattr(core,'check_files',lambda mapping,base=None: [None for k,v in mapping.items() if sha(tmp_path/k)==v] if all(sha(tmp_path/k)==v for k,v in mapping.items()) else (_ for _ in ()).throw(ValueError('changed')))
    head=['implementation']
    monkeypatch.setattr(core,'git',lambda *a:head[0].encode() if a[1]=='rev-parse' else b'')
    monkeypatch.setattr(core,'provenance',lambda:{})
    assert core.protocol(tmp_path,live=True)==p
    head[0]='results_commit'
    assert core.protocol(tmp_path)==p
    with pytest.raises(ValueError,match='HEAD'):core.protocol(tmp_path,live=True)
    code.write_text('changed')
    with pytest.raises(ValueError):core.protocol(tmp_path)


def test_complete_offline_analysis_and_figures(tmp_path,monkeypatch):
    """Synthetic end-to-end report path; not a Gate5G evaluation or scientific result."""
    import pandas as pd
    from src.mattr_fidelity_splice import analysis,structure
    from src.mattr_optimization.common import GRID,write_json,sha
    out=tmp_path/'campaign';out.mkdir()
    (out/'gate5g_protocol.json').write_text('synthetic fixture')
    n=256
    frame=pd.DataFrame(dict(example=range(n),template_id=['a']*(n//2)+['b']*(n//2)))
    frame.to_csv(out/'cohort_manifest.csv',index=False)
    cp=out/'checkpoint.pt';torch.save({'train_log':[(0,100,.01,1.),(1,200,.02,.9)]},cp)
    rankings=[]
    for method in ('baseline','h2'):
        for seed in range(42,47):
            rankings.append(dict(key=f'{method}_{seed}',kind='core',method=method,seed=seed,dataset_size=10000,step=5000,checkpoint=str(cp),checkpoint_sha256=sha(cp)))
    rankings.append(dict(key='eap',kind='core',method='eap_ig',seed=42,dataset_size=10000,step=0))
    for parent in list(rankings[:-1]):
        for k in (324,1624):rankings.append({**parent,'key':f'{parent["key"]}_splice{k}','parent':parent['key'],'kind':'splice','k':k})
    jobs=[dict(id=r['key'],ranking=r['key'],part='A' if r['kind']=='core' else 'C') for r in rankings]
    p=dict(implementation_commit='synthetic',rankings=rankings,jobs=jobs,cohort={'eligible':1000})
    monkeypatch.setattr(analysis,'protocol',lambda out:p)
    monkeypatch.setattr(analysis,'identity',lambda *a:{})
    monkeypatch.setattr(analysis,'verified',lambda *a:True)
    monkeypatch.setattr(analysis,'ROOT',tmp_path)
    monkeypatch.setattr(analysis,'HISTORY',tmp_path/'no_history')
    masks=[]
    for r in rankings:
        folder=out/'evaluations'/r['key'];folder.mkdir(parents=True)
        metrics=[];per=[]
        queries=[('nominal',f,int(f*32491)) for f in GRID]+[('cap',c,c) for c in (60,80,100,116,140,180,240,296,400,640)]
        if r['kind']=='splice':queries += [('boundary',n,n) for n in (r['k']-1,r['k'],r['k']+1)]
        for selection,target,prefix in queries:
            clean=np.tile(np.arange(1.,5.),n//4);empty=np.ones(n)*-1
            margin=clean*.8 if r['kind']=='splice' or r['method']=='eap_ig' else clean*.5
            if selection=='cap' and target==116 and r['method']=='baseline' and r['kind']=='core':margin=np.ones(n)*-1
            kl=.1 if r['method']=='eap_ig' else .2
            metric=dict(ranking=r['key'],selection=selection,target=target,nominal_n=prefix,actual_edges=prefix,mask_sha256='synthetic',
                kl=kl,accuracy=float((margin>0).mean()),top1_accuracy=.5,sign_agreement=.75,top1_agreement=.5,margin_error_mean=.5,
                margin=float(margin.mean()))
            metrics.append(metric)
            for i in range(n):per.append({**metric,'example':i,'template_id':frame.template_id.iloc[i],
                'clean_margin':clean[i],'margin':margin[i],'corrupted_margin':empty[i],
                'top1_id':1,'incorrect_token':2,'correct_token':1})
        pd.DataFrame(metrics).to_csv(folder/'metrics.csv',index=False)
        pd.DataFrame(per).to_csv(folder/'per_prompt.csv',index=False)
        margins=np.stack([clean*(.8 if r['kind']=='splice' or r['method']=='eap_ig' else .5) for f in GRID])
        cpr,cmd=official_areas(margins.mean(-1),clean.mean(),empty.mean())
        area=[[],float(cpr),float(cmd),0,[],[],.8]
        write_json(folder/'summary.json',dict(official_result=area))
        if r['kind']=='splice':masks.append(dict(ranking=r['key'],capped116_identical=True))
    pd.DataFrame(masks).to_csv(out/'splice_topology.csv',index=False)
    def fake_features(out,p):
        rows=[]
        for r in rankings:
            if r['kind']=='splice':continue
            for target in (116,296):rows.append(dict(ranking=r['key'],method=r['method'],seed=r['seed'],dataset_size=r['dataset_size'],step=r['step'],target=target,
                actual_edges=target,nominal_prefix=target+10,destination_entropy=2.,complete_name_mover_path=True,
                eap_jaccard=.5,full_signed_spearman_eap=.7,known_heads=10))
        return pd.DataFrame(rows)
    monkeypatch.setattr(structure,'features',fake_features)
    analysis.analyze(out)
    assert (out/'gate5g_decision.json').exists()
    assert len(list((out/'figures').glob('*.png')))==11
    report=(out/analysis.REPORT).read_text()
    assert 'Secondary endpoint: KL and accuracy at capped116' in report
    assert not report.split(analysis.INTERPRETATION)[1].strip()
    assert json.loads((out/'analysis_metadata.json').read_text())['model_forwards']==0
    # Exercise the zero-forward final audit and package allowlist, including the
    # post-results-commit contract, without invoking Git staging or real model data.
    from src.mattr_fidelity_splice import cli
    from src.mattr_matched_size.topology import digest as mask_digest
    from src.mattr_fidelity_splice.core import identity
    p['cohort']['exclusions']=[];p['model']={'weight_sha256':'synthetic-model'}
    p['cohort_sha256']=sha(out/'cohort_manifest.csv')
    monkeypatch.setattr(cli,'protocol',lambda out:p)
    monkeypatch.setattr(cli,'verified',lambda *a:True)
    monkeypatch.setattr(cli,'ROOT',tmp_path)
    monkeypatch.setattr(cli,'check_files',lambda *a:None)
    frame['clean_sha256']=['clean'+str(i) for i in range(n)]
    frame['corrupted_sha256']=['donor'+str(i) for i in range(n)]
    frame.to_csv(out/'cohort_hashes.csv',index=False)
    pd.DataFrame(dict(clean_sha256=['historical-clean'],corrupted_sha256=['historical-donor'])).to_csv(out/'pre_freeze_test_prompts.csv',index=False)
    write_json(out/'preflight_gpu.json',dict(passed=True))
    write_json(out/'protected_history.json',{})
    write_json(out/'frozen_at.json',dict(utc='2020-01-01T00:00:00Z'))
    mask=np.ones(4,dtype=bool);mh=mask_digest(mask)
    (out/'masks').mkdir();np.save(out/'masks'/f'{mh}.npy',mask)
    queries=[]
    for job in jobs:
        folder=out/'evaluations'/job['id']
        f=pd.read_csv(folder/'metrics.csv');f['mask_sha256']=mh;f.to_csv(folder/'metrics.csv',index=False)
        queries.extend(f.to_dict('records'))
        s=json.loads((folder/'summary.json').read_text())
        s.update(identity=identity(out,p,job),weight_before='synthetic-model',weight_after='synthetic-model',
            counts={'optimizer_updates':0,'backward_calls':0},correctness_passed=True,model_gradients_stored=False)
        write_json(folder/'summary.json',s)
    pd.DataFrame(queries).to_csv(out/'queries.csv',index=False)
    cli.audit(out)
    assert json.loads((out/'final_audit.json').read_text())['registered_decisions_reproduced']
    # Missing manifest/metadata artifacts are deliberately tiny synthetic fixtures.
    for name in ['gate5g_protocol.sha256','ranking_inventory.csv','excluded_rankings.json','eap_ordering.json','topology_audit.json',
        'cohort_audit.json','preflight_cpu.json','historical_evidence.json','ranking_features.csv','known_head_reference.csv',
        'seed42_edge_case_study.csv','graph_analysis_compute.csv']:
        if not (out/name).exists():(out/name).write_text('{}\n')
    cli.package(out,dry_run=True)
    assert not (out/'results_package_manifest.json').exists()
