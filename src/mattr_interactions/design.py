"""Topology/ranking-only selection and a priori diagnostic decisions."""
from itertools import combinations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from src.mattr_optimization.common import ROOT, sha, write_json
from src.mattr_optimization.gate3 import baseline
from src.mattr_optimization.gate4_init import edge_mapping

SEEDS=(42,43,44)
FRACTIONS=(.01,.02)

def digest(mask):
    return hashlib.sha256(np.packbits(np.asarray(mask,dtype=bool)).tobytes()).hexdigest()

def assign(graph,mask):
    graph.reset(empty=False)
    graph.in_graph[:]=False
    graph.in_graph[graph.real_edge_mask.bool()]=torch.as_tensor(mask,dtype=torch.bool)

def cleanup(graph,mask):
    assign(graph,mask);graph.prune()
    return graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()

def toggled(base,indices):
    result=base.copy();result[list(indices)]=~result[list(indices)]
    return result

def components(row):
    return {str(x) for x in (row.parent,row.child) if str(x).startswith(('a','m'))}

def category(a,b):
    if a.child==b.child and a.qkv==b.qkv:return 'same_receiver'
    if components(a)&components(b):return 'same_component'
    return 'different_components'

def layer(node):
    import re
    found=re.match(r'[am](\d+)',str(node))
    return int(found[1]) if found else -1

def edge_type(node):
    return 'attention' if str(node).startswith('a') else 'MLP' if str(node).startswith('m') else str(node)

def protocol():
    return {'gate':'5B','model':'gpt2-small','task':'official MIB IOI','edge_count':32491,
      'ranking_steps':5000,'discovery_pool':128,'ranking_seeds':list(SEEDS),'intervention_seed':42,
      'fractions':list(FRACTIONS),'train_prompts':32,'confirmation_prompts':64,'batch_size':4,
      'selection_seed':5142,'confirmation_seed':5042,'bootstrap_seed':5242,'bootstrap_replicates':2000,
      'candidates_per_cutoff':48,'pairs_per_cutoff_max':120,
      'candidate_rule':'24 retained: 8 highest, 8 nearest retained boundary, 8 largest signed-rank disagreement with saved EAP-IG; 24 excluded:12 nearest below requested cutoff,12 disagreement. Distinct endpoint/type round-robin for diversity. Select only individual toggles unchanged by official prune. Excluded disagreement pool limited to next4096 ranks. No prompt effects used.',
      'pair_rule':'Each direction: up to15 same receiver,15 shared head/MLP,15 different components,15 unused random controls matched to first45 pairs by source/destination types and layers where possible. Joint prune must equal exact two-edge toggle. i<j, unique within cutoff/direction. Shortfalls reported, never expand beyond120.',
      'confirmation_rule':'Per cutoff/direction: top3 absolute training interaction; plus1 largest joint addition gain (or deletion magnitude), excluding those3. At most16 primary pairs. For each primary select one previously screened random control in that cutoff/direction, minimizing source/destination type and layer mismatch (tie by pair_id), reusing identical controls. At most32 total. Ties by pair_id. Freeze IDs/hashes before loading confirmation data/effects.',
      'cohort_rule':'Screen first32 exact Gate2 seed42 N128 rows. Confirm seeded shuffle of official filtered validation, excluding every small (<10000 rows) prior Gate2-5A validation manifest by index AND prompt, and all three discovery-pool prompts. Gate1 full validation is historical exposure, not independent of all research.',
      'intervention':'official evaluate_graph patching, signed ranking topn prune=True for base; exact binary toggles only when subsequent official prune changes no additional edge; no new mask/patch implementation; original live closure used only as correctness oracle',
      'cache':'Exact scores keyed by cohort fingerprint, protocol/source/ranking/mask hashes. CPU immutable donor-only activation matrix <=2GiB LRU, keyed by exact tokens, attention mask, source shape and model identity. Never cache clean/patched sources. Cache hits do not count as model forwards.',
      'criteria':{'substantial_absolute_margin':.1,'substantial_relative_interaction':.2,
        'reproducible_selected_fraction_min':.5,'option_A_confirmed_pairs_min':4,
        'option_A_useful_additions_min':2,'useful_joint_margin_gain_min':.1,
        'useful_addition':'positive held-out joint effect >=.1, interaction CI excludes0, signed interaction agrees with train, joint > max(single_i,single_j), accuracy gain positive and bootstrap95 lower>=0',
        'option_B_component_spearman_gain_min':.1,'option_B_null_percentile':.95,
        'option_B_groupings_min':2,'option_B_interaction_concentration_ratio_min':1.5,
        'option_B_concentration':'same receiver/component mean absolute confirmed interaction >=1.5 times type/layer matched random controls, with prompt-bootstrap CI for the difference excluding0; insufficient controls means inconclusive, not success',
        'uncertainty':'Shared prompt bootstrap indices across pairs; per-pair95% and familywise Bonferroni intervals. No bootstrap over correlated pair/threshold rows; no significance claims from3 training seeds.',
        'direction_decision':'A only if reproducibility and >=4 substantial/sign-consistent confirmed pairs and >=2 useful excluded additions; B only if >=2 component groupings exceed edge Spearman by.1 and95th common-permutation null and within-component concentration criterion. Both allowed; otherwise neither. Exploratory evidence, never proof of algorithm improvement.'},
      'stability_null':'200 common edge-label permutations across all3 seeds, preserving cross-seed edge covariance, overlapping group memberships and exact component sizes; compare normalized mean percentile rank. Also random size-matched edge membership controls.',
      'budget':{'unique_masks_max':450,'example_forwards_max':30000,'seconds_max':14400,'training_updates':0},
      'no_retraining':True,'no_new_algorithm':True,'no_leaderboard':True}

def diversified(frame,count):
    chosen=[];seen=set()
    for _,r in frame.iterrows():
        key=(r.parent,r.child_type)
        if key not in seen:chosen.append(int(r.mattr_index));seen.add(key)
        if len(chosen)==count:return chosen
    for i in frame.mattr_index:
        if int(i) not in chosen:chosen.append(int(i))
        if len(chosen)==count:break
    return chosen

def prepare(out):
    from eap.graph import Graph
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    target=out/'gate5b_protocol.json'
    if target.exists():
        p=json.loads(target.read_text())
        assert sha(target)==(out/'gate5b_protocol.sha256').read_text().strip()
        for name,h in p['sources'].items():assert sha(ROOT/name)==h,name
        return p
    graph=Graph.from_model(dict(n_layers=12,n_heads=12,parallel_attn_mlp=False,d_model=768))
    mapping=edge_mapping(graph,graph)
    mapping['parent_type']=mapping.parent.map(edge_type);mapping['child_type']=mapping.child.map(edge_type)
    mapping['source_layer']=mapping.parent.map(layer);mapping['destination_layer']=mapping.child.map(layer)
    mapping['edge_type']=mapping.parent_type+'->'+mapping.child_type
    mapping.to_csv(out/'edge_mapping.csv',index=False)
    scores=torch.load(baseline(42)/'final_scores.pt',map_location='cpu',weights_only=False)
    eap_path=ROOT/'results/mattr_optimization/20261022_gate4/attribution_seed42/eap_ig_scores.pt'
    eap=torch.load(eap_path,map_location='cpu',weights_only=False).numpy() if eap_path.exists() else None
    mapping['rank']=rankdata(-scores.numpy(),method='ordinal').astype(int)
    mapping['eap_rank']=rankdata(-eap,method='average') if eap is not None else np.nan
    mapping['disagreement']=abs(mapping['rank']-mapping.eap_rank)
    candidates=[];pairs=[];audits=[];contexts={};rng=np.random.default_rng(5142)
    for fraction in FRACTIONS:
        key=f'{fraction:g}';graph.scores[:]=-torch.inf;graph.scores[graph.real_edge_mask.bool()]=scores
        graph.apply_topn(int(32491*fraction),absolute=False,level='edge',prune=True)
        base=graph.in_graph[graph.real_edge_mask.bool()].numpy().copy();contexts[key]=base
        np.save(out/f'base_{key}.npy',base)
        safe=[]
        # Connected endpoints guarantee safe additions; explicit prune audit still checks selected candidates.
        active={name for name,node in graph.nodes.items() if node.in_graph};active.add('logits')
        for row in mapping.itertuples():
            i=row.mattr_index
            if base[i]:
                valid=np.array_equal(cleanup(graph,toggled(base,[i])),toggled(base,[i]))
            else:valid=row.rank>int(fraction*32491) and row.parent in active and row.child in active
            if valid:safe.append(i)
        eligible=mapping[mapping.mattr_index.isin(safe)]
        selected=[]
        for direction,specs in [('deletion',[('high',8),('boundary',8),('disagreement',8)]),('addition',[('boundary',12),('disagreement',12)])]:
            pool=eligible[base[eligible.mattr_index.to_numpy()]==(direction=='deletion')]
            for stratum,n in specs:
                available=pool[~pool.mattr_index.isin(selected)]
                if stratum=='high':available=available.sort_values('rank')
                elif stratum=='boundary':available=available.sort_values('rank',ascending=direction=='addition')
                else:
                    if direction=='addition':available=available[available['rank']<=int(fraction*32491)+4096]
                    available=available.sort_values(['disagreement','rank'],ascending=[False,True])
                picks=diversified(available,n)
                if len(picks)!=n:raise ValueError(f'Insufficient safe candidates {fraction}/{direction}/{stratum}: {len(picks)}')
                for i in picks:
                    desired=toggled(base,[i]);assert np.array_equal(cleanup(graph,desired),desired)
                    candidates.append({**mapping.iloc[i].to_dict(),'fraction':fraction,'direction':direction,'stratum':stratum})
                selected.extend(picks)
        local=pd.DataFrame(candidates).query('fraction == @fraction')
        for direction in ('deletion','addition'):
            records=[];rejected=0
            for i,j in combinations(sorted(local.query('direction == @direction').mattr_index),2):
                desired=toggled(base,[i,j]);actual=cleanup(graph,desired)
                if not np.array_equal(desired,actual):rejected+=1;continue
                a,b=mapping.iloc[i],mapping.iloc[j]
                records.append({'i':i,'j':j,'category':category(a,b),'signature':(a.edge_type,b.edge_type,int(a.source_layer),int(a.destination_layer),int(b.source_layer),int(b.destination_layer))})
            chosen=[]
            for cat in ('same_receiver','same_component','different_components'):
                pool=[r for r in records if r['category']==cat]
                order=rng.permutation(len(pool))[:15]
                chosen.extend([{**pool[k],'control':False,'matched_pair':None,'match_distance':None} for k in order])
            used={(r['i'],r['j']) for r in chosen}
            for pos in range(min(15,len(records)-len(chosen))):
                target_pair=chosen[pos%len(chosen)] if chosen else records[0]
                options=[r for r in records if (r['i'],r['j']) not in used]
                distances=[sum(x!=y for x,y in zip(r['signature'],target_pair['signature'])) for r in options]
                best=np.flatnonzero(np.array(distances)==min(distances));r=options[int(rng.choice(best))]
                chosen.append({**r,'control':True,'matched_pair':f"{key}_{direction}_{target_pair['i']}_{target_pair['j']}",'match_distance':min(distances)})
                used.add((r['i'],r['j']))
            for r in chosen:
                r={k:v for k,v in r.items() if k!='signature'}
                pairs.append({**r,'fraction':fraction,'direction':direction,'pair_id':f"{key}_{direction}_{r['i']}_{r['j']}",
                  'edge_i':mapping.iloc[r['i']].edge,'edge_j':mapping.iloc[r['j']].edge,
                  'stratum_i':local.set_index('mattr_index').loc[r['i'],'stratum'],
                  'stratum_j':local.set_index('mattr_index').loc[r['j'],'stratum']})
            audits.append({'fraction':fraction,'direction':direction,'candidates':24,'valid_pairs':len(records),
                'joint_cleanup_rejections':rejected,'selected_pairs':len(chosen),'safe_singles_total':len(eligible),
                'base_edges':int(base.sum()),'requested_edges':int(fraction*32491)})
    pd.DataFrame(candidates).to_csv(out/'selected_candidates.csv',index=False)
    pd.DataFrame(pairs).to_csv(out/'selected_pairs.csv',index=False)
    pd.DataFrame(audits).to_csv(out/'topology_selection_audit.csv',index=False)
    p=protocol();p['sources']={str(x.relative_to(ROOT)).replace('\\','/'):sha(x) for x in [*sorted((ROOT/'src/mattr_interactions').glob('*.py')),ROOT/'tests/test_mattr_interactions.py']}
    p['ranking_inputs']={str(baseline(s).relative_to(ROOT))+'/final_scores.pt':sha(baseline(s)/'final_scores.pt') for s in SEEDS}
    if eap is not None:p['eap_reference']={'path':str(eap_path.relative_to(ROOT)),'sha256':sha(eap_path),'orientation':'signed official EAP-IG, train-only128; observational reference'}
    p['selection_artifacts']={x.name:sha(x) for x in [out/'edge_mapping.csv',out/'selected_candidates.csv',out/'selected_pairs.csv',*out.glob('base_*.npy')]}
    write_json(target,p);(out/'gate5b_protocol.sha256').write_text(sha(target)+'\n')
    write_json(out/'selection_frozen.json',{'protocol_sha256':sha(target),'training_or_confirmation_model_forwards':0,'pairs':len(pairs),'candidate_count':len(candidates)})
    return p
