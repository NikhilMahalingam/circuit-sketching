"""Post-run CPU artifact audit and descriptive conditional-effect tables.

This adds no interventions, selection rules or decision criteria.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from src.mattr_optimization.common import ROOT,sha,write_json,configure
from .design import cleanup,toggled,digest
from .evaluation import pair_effects
from .statistics import select_confirmation

def run(out):
    started=time.perf_counter();configure()
    import torch
    from eap.graph import Graph
    torch.set_num_threads(4)
    p=json.loads((out/'gate5b_protocol.json').read_text())
    assert sha(out/'gate5b_protocol.json')==(out/'gate5b_protocol.sha256').read_text().strip()
    for name,h in p['sources'].items():assert sha(ROOT/name)==h,name
    for name,h in p['ranking_inputs'].items():assert sha(ROOT/name)==h,name
    for name,h in p['selection_artifacts'].items():assert sha(out/name)==h,name
    protected=json.loads((out/'protected_previous_hashes.json').read_text())
    for name,h in protected.items():assert sha(ROOT/name)==h,name
    graph=Graph.from_model(dict(n_layers=12,n_heads=12,parallel_attn_mlp=False,d_model=768))
    candidates=pd.read_csv(out/'selected_candidates.csv');pairs=pd.read_csv(out/'selected_pairs.csv')
    assert len(candidates)==96 and len(pairs)<=240 and not pairs.pair_id.duplicated().any()
    assert (pairs.i<pairs.j).all()
    for (f,d),g in candidates.groupby(['fraction','direction']):
        assert len(g)==24 and g.edge.nunique()==24
        assert ((g.parent_type=='MLP')|(g.child_type=='MLP')).any()
        assert ((g.parent_type=='attention')|(g.child_type=='attention')).any()
    # Every four-way context has exactly zero, one or two changes and no cleanup collateral.
    for pair in pairs.itertuples():
        base=np.load(out/f'base_{pair.fraction:g}.npy')
        assert base[int(pair.i)]==base[int(pair.j)]==(pair.direction=='deletion')
        for indices in ([],[int(pair.i)],[int(pair.j)],[int(pair.i),int(pair.j)]):
            desired=toggled(base,indices);assert np.sum(desired!=base)==len(indices)
            assert np.array_equal(cleanup(graph,desired),desired)
    selected=pd.read_csv(out/'confirmation_pairs.csv');screen=pd.read_csv(out/'screen_interactions.csv')
    primary_ids=select_confirmation(screen)
    assert set(selected[selected.selection_role=='primary'].pair_id)==set(primary_ids)
    assert len(selected)<=32
    frozen=json.loads((out/'confirmation_selection_frozen.json').read_text())
    assert frozen['confirmation_loaded'] is False and frozen['confirmation_model_forwards']==0
    assert frozen['selection_sha256']==sha(out/'confirmation_pairs.csv')
    cohort=json.loads((out/'validation_cohort.json').read_text())
    frame=pd.read_csv(out/'validation_manifest.csv');seen=set()
    for name,h in cohort['prior_development_manifests'].items():
        assert sha(ROOT/name)==h,name
        prior=pd.read_csv(ROOT/name)
        if 'official_filtered_index' in prior:assert not set(frame.official_filtered_index)&set(prior.official_filtered_index)
        if 'clean_prompt' in prior:assert not set(frame.clean_prompt)&set(prior.clean_prompt)
        seen.add(name)
    assert len(frame)==64 and frame.clean_prompt.nunique()==64
    # Independently reconstruct numerical four-way effects from the exact metric cache.
    cache={}
    for meta in (out/'exact_cache').glob('*.json'):
        ident=json.loads(meta.read_text())
        if ident['cohort'] not in ('screen','confirmation'):continue
        assert ident['protocol_sha256']==sha(out/'gate5b_protocol.json')
        assert meta.stem==hashlib.sha256(json.dumps(ident,sort_keys=True).encode()).hexdigest()
        cache[ident['cohort'],ident['mask_sha256']]=meta.with_suffix('.npz')
    checked=0;conditionals=[];absolute_scores=[]
    for name,selection,n in [('screen',pairs,32),('confirmation',selected,64)]:
        per=pd.read_csv(out/f'{name}_per_example.csv');summary=pd.read_csv(out/f'{name}_interactions.csv')
        assert per.groupby('pair_id').size().eq(n).all()
        for pair in selection.itertuples():
            base=np.load(out/f'base_{pair.fraction:g}.npy');values=[]
            for indices in ([],[int(pair.i)],[int(pair.j)],[int(pair.i),int(pair.j)]):
                path=cache[name,digest(toggled(base,indices))]
                with np.load(path) as saved:values.append((saved['margin'].copy(),saved['accuracy'].copy()))
            actual=per[per.pair_id==pair.pair_id].sort_values('example')
            expected=pair_effects(values)
            absolute_scores.append({'cohort':name,'pair_id':pair.pair_id,'fraction':pair.fraction,'direction':pair.direction,
              **{label:float(value[0].mean()) for label,value in zip(('F_base','F_i','F_j','F_ij'),values)},
              **{label:float(value[1].mean()) for label,value in zip(('accuracy_base','accuracy_i','accuracy_j','accuracy_ij'),values)}})
            for field,v in expected.items():np.testing.assert_allclose(actual[field],v,atol=1e-6,rtol=1e-6)
            row=summary[summary.pair_id==pair.pair_id].iloc[0]
            for field,v in expected.items():assert np.isclose(row[field],np.mean(v),atol=1e-6)
            ci=row.joint-row.single_j;cj=row.joint-row.single_i
            conditionals.append({'cohort':name,'pair_id':pair.pair_id,'fraction':pair.fraction,'direction':pair.direction,
                'single_i':row.single_i,'single_j':row.single_j,'joint':row.joint,'conditional_i_given_j':ci,'conditional_j_given_i':cj,
                'i_effect_sign_changes':bool(np.sign(ci)!=np.sign(row.single_i)),
                'j_effect_sign_changes':bool(np.sign(cj)!=np.sign(row.single_j)),
                'joint_improves_margin':row.joint>0,'joint_accuracy_gain':row.accuracy_joint})
            checked+=1
    pd.DataFrame(conditionals).to_csv(out/'conditional_edge_effects.csv',index=False)
    pd.DataFrame(absolute_scores).to_csv(out/'four_way_scores.csv',index=False)
    events=pd.read_csv(out/'forward_events.csv');proc=json.loads((out/'process_summary.json').read_text())
    assert proc['success'] and events.success.all() and not events.grad_enabled.any()
    assert len(events)==proc['counts']['forward_batches']
    assert int(events.batch_size.sum())==proc['counts']['example_forwards']
    assert proc['counts']['backward_calls']==proc['counts']['optimizer_updates']==0
    assert proc['unique_evaluations_this_run']<=p['budget']['unique_masks_max']
    assert json.loads((out/'correctness_passed.json').read_text())['passed']
    tests=(ROOT/'.cache/mattr_interactions_tests.log').read_text()
    assert 'Ran 7 tests' in tests and '\nOK\n' in tests
    (out/'cpu_unit_tests.log').write_text(tests)
    compare=screen.groupby(['fraction','direction','control']).agg(pairs=('pair_id','size'),mean_abs_interaction=('abs_interaction','mean'),mean_joint_effect=('joint','mean'),mean_accuracy_effect=('accuracy_joint','mean')).reset_index()
    compare.to_csv(out/'screen_control_comparison.csv',index=False)
    strata=screen.groupby(['direction','stratum_i','stratum_j']).agg(pairs=('pair_id','size'),mean_abs_interaction=('abs_interaction','mean'),sign_disagreements=('sign_flip_vs_additive','mean')).reset_index()
    strata.to_csv(out/'ranking_strata_comparison.csv',index=False)
    # Additional descriptive grouping diagnostics; these never enter the frozen decision.
    from scipy.stats import rankdata,spearmanr
    from itertools import combinations
    from src.mattr_optimization.gate3 import baseline
    mapping=pd.read_csv(out/'edge_mapping.csv',keep_default_na=False)
    from .design import layer
    labels_by_group={
      'source_head':mapping.parent.where(mapping.parent.str.startswith('a')),
      'destination_head':mapping.child.where(mapping.child.str.startswith('a')),
      'qkv_receiver':(mapping.child+'<'+mapping.qkv+'>').where(mapping.child.str.startswith('a')),
      'source_MLP':mapping.parent.where(mapping.parent.str.startswith('m')),
      'destination_MLP':mapping.child.where(mapping.child.str.startswith('m')),
      'source_layer':mapping.parent.map(layer).astype(str).where(mapping.parent.map(layer)>=0),
      'destination_layer':mapping.child.map(layer).astype(str).where(mapping.child.map(layer)>=0)}
    importance=[];masks={};memberships=[];explanation=[]
    for seed in (42,43,44):
        s=torch.load(baseline(seed)/'final_scores.pt',map_location='cpu',weights_only=False)
        importance.append((32491-rankdata(-s.numpy(),method='average'))/32490)
        for f in (.01,.02):
            graph.scores[:]=-torch.inf;graph.scores[graph.real_edge_mask.bool()]=s
            graph.apply_topn(int(f*32491),absolute=False,level='edge',prune=True)
            masks[seed,f]=graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()
    importance=np.array(importance)
    for grouping,labels in labels_by_group.items():
        codes,names=pd.factorize(labels,sort=True);valid=codes>=0;size=np.bincount(codes[valid]);n=len(names)
        means=np.array([np.bincount(codes[valid],weights=x[valid],minlength=n)/size for x in importance])
        for a,b in combinations(range(3),2):
            prediction=means[a,codes[valid]];target=importance[b,valid]
            r2=1-np.square(target-prediction).sum()/np.square(target-target.mean()).sum()
            within=(np.square(importance[a,valid]-means[a,codes[valid]])).sum()
            total=np.square(importance[a,valid]-importance[a,valid].mean()).sum()
            explanation.append({'grouping':grouping,'source_seed':(42,43,44)[a],'target_seed':(42,43,44)[b],
              'within_source_seed_eta_squared':1-within/total,'cross_seed_component_only_R2':r2,
              'cross_seed_component_prediction_spearman':spearmanr(prediction,target).statistic,
              'cross_seed_individual_edge_spearman_same_edges':spearmanr(importance[a,valid],target).statistic})
            for f in (.01,.02):
                ma,mb=masks[(42,43,44)[a],f],masks[(42,43,44)[b],f]
                for k,name in enumerate(names):
                    ids=(codes==k);inter=int((ma&mb&ids).sum());union=int(((ma|mb)&ids).sum())
                    memberships.append({'grouping':grouping,'component':name,'fraction':f,'seed_a':(42,43,44)[a],'seed_b':(42,43,44)[b],
                      'eligible_edges':int(size[k]),'retained_a':int((ma&ids).sum()),'retained_b':int((mb&ids).sum()),
                      'intersection':inter,'union':union,'within_component_edge_jaccard':inter/union if union else np.nan})
    pd.DataFrame(explanation).to_csv(out/'component_identity_explanation.csv',index=False)
    pd.DataFrame(memberships).to_csv(out/'within_component_edge_stability.csv',index=False)
    from .report import table
    notes='\n\n## Additional artifact audit and conditional effects\n\n'
    notes+='All three saved Gate2 rankings vary both discovery subset and training RNG; their stability analysis cannot separate these sources. This diagnostic does not infer a causal benefit of hierarchy or interaction-aware training. Candidate and matched-control strata are limited and train-selected; observed rates are not prevalence estimates for the full graph.\n\n'
    notes+=table(compare)+'\n\n'+table(strata)
    notes+='\n\nWithin-component retained-edge overlaps are saved in within_component_edge_stability.csv (empty unions are undefined, not perfect stability). Component-only cross-seed prediction and descriptive within-seed variance explained are in component_identity_explanation.csv. These are additional diagnostics, not new selection objectives or modified success criteria.\n\n'
    notes+=table(pd.DataFrame(explanation).groupby('grouping')[['within_source_seed_eta_squared','cross_seed_component_only_R2','cross_seed_component_prediction_spearman','cross_seed_individual_edge_spearman_same_edges']].mean().reset_index())
    matches=pd.read_csv(out/'confirmation_control_matches.csv')
    exact_matches=int((matches.type_layer_mismatch==0).sum())
    notes+=f'\n\nImportant control limitation: **{exact_matches}/{len(matches)} confirmation control matches were exact on all source/destination type and layer fields**. They are the closest available controls in the frozen candidate set. The large within-component/control contrast therefore remains vulnerable to layer/connectivity confounding; it is not a causal component-structure result. No extra controls were added after outcomes.\n\n'
    r2=pd.DataFrame(explanation).groupby('grouping').cross_seed_component_only_R2.mean()
    notes+=f'Component means can rank consistently while explaining little individual-edge variation. Source-head component-only cross-seed R2 averages {r2["source_head"]:.4f} and source-MLP R2 {r2["source_MLP"]:.4f} here; a coarse component constraint could discard important edge-specific information. The formal frozen gate favors OptionB as a research hypothesis, not a validated replacement for edge-level masking.\n\n'
    consistency=pd.read_csv(out/'heldout_consistency.csv');primary=consistency[consistency.selection_role=='primary']
    validated_conditionals=pd.DataFrame(conditionals).query("cohort == 'confirmation'")
    sign_changes=int(validated_conditionals.i_effect_sign_changes.sum()+validated_conditionals.j_effect_sign_changes.sum())
    decision=json.loads((out/'research_decision.json').read_text())
    notes+=f'Final diagnostic conclusion: {int(primary.sign_consistent.sum())}/{len(primary)} primary interaction signs agreed between train and confirmation, but only {int(primary.reproduced_substantial.sum())}/{len(primary)} met the full substantial/reproducible criterion and {int(primary.useful_addition.sum())} excluded additions passed the usefulness/accuracy rule. Held-out conditional edge-effect sign reversals: {sign_changes}. Frozen recommendation: {decision["recommendation"]}. These diagnostics do not demonstrate that a proposed algorithm improves CPR; do not begin either algorithm automatically.\n'
    notes+='\n\nConditional edge effects `joint - other_single` and their sign changes are in conditional_edge_effects.csv. These expose context-dependent usefulness without interpreting every nonadditive pair as a missed useful circuit. Canonical mappings, exact mask cleanup, cache algebra, selection hashes, disjoint cohorts, forward counters and unchanged historical files were independently audited with zero additional model forwards.\n'
    for path in (out/'GATE5B_INTERACTION_DIAGNOSTICS_REPORT.md',ROOT/'GATE5B_INTERACTION_DIAGNOSTICS_REPORT.md'):
        body=path.read_text().split('\n\n## Additional artifact audit')[0];path.write_text(body.rstrip()+notes)
    result={'passed':True,'protocol_sha256':sha(out/'gate5b_protocol.json'),'frozen_source_hashes':len(p['sources']),
      'protected_historical_files_unchanged':len(protected),'candidate_edges':len(candidates),'screen_pairs':len(pairs),
      'confirmation_pairs':len(selected),'four_way_cache_reconstructions':checked,'confirmation_prompts':64,
      'prior_small_validation_manifests_checked':len(seen),'cpu_tests':7,'counts':proc['counts'],
      'additional_model_forwards':0,'audit_source_sha256':sha(Path(__file__)),'seconds':time.perf_counter()-started}
    write_json(out/'final_validation.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out.resolve())
