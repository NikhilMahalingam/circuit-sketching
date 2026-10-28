"""Bounded correctness only on two excluded historical Gate5F prompts."""
from .common import *
from .stage2 import prepare,session,measure

def run(out):
    prepare(out);p=original()
    frame=pd.read_csv(ROOT/'results/mattr_data_scaling/gate5f/cohort_primary.csv',keep_default_na=False).iloc[:2].reset_index(drop=True)
    fresh=pd.read_csv(out/'confirmation_hashes.csv')
    assert not (set(frame.clean_sha256)|set(frame.corrupted_sha256))&(set(fresh.clean_sha256)|set(fresh.corrupted_sha256))
    frame.drop(columns=['clean_prompt','corrupted_prompt','official_template']).to_csv(out/'smoke_prompt_hashes.csv',index=False)
    q=query(core(p,'baseline',42)['key'],'nominal',.1);mask=oldmask(q)
    rec=save_set(out,'smoke_historical',mask,stage='smoke')
    assert rec['mask_sha256']==q.mask_sha256
    context=dict(role='bounded_historical_only',implementation_commit=git(ROOT,'rev-parse','HEAD').decode().strip(),
        smoke_prompt_sha256=sha(out/'smoke_prompt_hashes.csv'))
    with session(out,'bounded_smoke',frame,p['model'],context) as (engine,meter,work):
        metric,arrays=measure(out,engine,meter,rec)
        from MIB_circuit_track.evaluation import evaluate_area_under_curve
        from src.mattr_compact_validation.inference import assign_scores
        scores,_=check_set(out,rec);assign_scores(engine.graph,scores)
        with meter.stage('official_point_oracle'),torch.no_grad():
            oracle=evaluate_area_under_curve(engine.model,engine.graph,engine.loader,engine.metric,quiet=True,
                level='edge',absolute=False,percentages=((rec['requested_edges']+.25)/32491,),refs={})
        np.testing.assert_allclose(metric['faithfulness'],oracle[4][0],rtol=2e-5,atol=2e-5)
        before=meter.counts['forward_batches'];measure(out,engine,meter,rec);assert meter.counts['forward_batches']==before
        write_json(out/'smoke_passed.json',dict(passed=True,prompts=2,fresh_forwards=0,official_faithfulness_equal=True,
            KL='unchanged validated full-vocabulary Gate5G implementation',counts=meter.counts,model_hash=p['model']['weight_sha256']))
