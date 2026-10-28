"""Prespecified post hoc diagnostics. Only saved arrays, graphs and CPU arithmetic."""
import inspect
import math
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .common import *
from src.mattr_data_scaling.structure import head,overlap
from src.mattr_fidelity_splice.statistics import weights,interval

def band(f):return '<=1%' if f<=.01 else '2%-5%' if f<=.05 else '10%-50%' if f<1 else '100%'
def counts(x):return dict(positive=int((x>0).sum()),zero=int((x==0).sum()),negative=int((x<0).sum()))

def run(out):
    configure();torch.set_num_threads(4);out.mkdir(parents=True,exist_ok=True)
    plan=ROOT/'stage1_analysis_plan.md'
    plan_commit=git(ROOT,'log','-1','--format=%H','--','stage1_analysis_plan.md').decode().strip()
    assert git(ROOT,'show',plan_commit+':stage1_analysis_plan.md').replace(b'\r\n',b'\n')==plan.read_bytes().replace(b'\r\n',b'\n')
    ident=dict(plan_sha256=sha(plan),original_protocol_sha256=sha(OLD/'gate5g_protocol.json'),source_sha256=sources())
    if verified(out/'stage1',ident):
        check_files(read(out/'stage1/outputs.json'),out)
        print('Verified Stage1 already complete');return
    start=time.perf_counter();p=original();protected={}
    for root in [OLD]:
        for path in root.rglob('*'):
            if path.is_file() and 'cache' not in path.parts and 'logs' not in path.parts and 'locks' not in path.parts:
                protected[path.relative_to(ROOT).as_posix()]=sha(path)
    write_json(out/'protected_gate5g.json',protected)
    mapping=pd.read_csv(OLD/'edge_mapping.csv',keep_default_na=False)
    classes=read(ROOT/'results/mattr_data_scaling/gate5f/known_ioi_heads.json')['classes']
    ea=next(r for r in p['rankings'] if r['method']=='eap_ig');evalues=load_scores(ea).numpy();eo=order(ea)
    signed=[]
    negative_heads=set(map(tuple,classes['negative']))
    incident=np.array([head(a) in negative_heads or head(b) in negative_heads for a,b in zip(mapping.parent,mapping.child)])
    for n in [324,1624,32491]:
        ids=eo[:n];supp=incident[ids]
        signed.append(dict(prefix=n,**counts(evalues[ids]),suppressor_incident_edges=int(supp.sum()),
            suppressor_incident_fraction=float(supp.mean()),negative_among_incident=float((evalues[ids][supp]<0).mean()) if supp.any() else None))
    pd.DataFrame(signed).to_csv(out/'eap_sign_counts.csv',index=False)
    first=np.flatnonzero(evalues[eo]<=0)
    first_nonpositive=int(first[0]+1) if len(first) else None
    pd.DataFrame(dict(rank=np.arange(1,1625),canonical_edge=eo[:1624],score=evalues[eo[:1624]])).to_csv(out/'eap_prefix_scores.csv',index=False)
    # Sign semantics from the actual pinned implementation and objective.
    import eap.attribute as attr
    import eap.utils as utils
    import MIB_circuit_track.metrics as metrics
    import MIB_circuit_track.evaluation as evaluator
    import src.mattr_data_scaling.attribution as historical_attr
    references=[]
    for fn in [historical_attr.attribute,attr.get_scores_eap_ig,utils.make_hooks_and_matrices,metrics.logit_diff,evaluator.evaluate_area_under_curve]:
        path=Path(inspect.getsourcefile(fn));lines,line=inspect.getsourcelines(fn)
        references.append(dict(path=path.relative_to(ROOT).as_posix(),function=fn.__name__,first_line=line,last_line=line+len(lines)-1,sha256=sha(path)))
    write_json(out/'source_references.json',references)
    quadrature=np.array(integration_weights());assert len(quadrature)==10
    pd.DataFrame(dict(fraction=GRID,weight=quadrature,band=[band(f) for f in GRID])).to_csv(out/'official_weights.csv',index=False)
    indices=np.load(OLD/'bootstrap_indices.npy');w=weights(indices,len(indices[0]));assert indices.shape[0]==5000
    assert sha(OLD/'bootstrap_indices.npy')==read(OLD/'analysis_metadata.json')['bootstrap_indices_sha256']
    curves=[];bands=[];boot={};inventory=[];composition=[];margins=[];template_rows=[];changes=[];per_change=[];verification=[]
    for rec in [r for r in p['rankings'] if r['kind']=='splice']:
        parent=record(p,rec['parent']);po=order(parent);so=order(rec)
        parent_rank=np.argsort(po)+1;pr=summary(parent['key']);sr=summary(rec['key'])
        pp=prompts(parent['key']);sp=prompts(rec['key']);deltaf=np.array(sr['official_result'][4])-np.array(pr['official_result'][4])
        deltacmd=np.abs(1-np.array(sr['official_result'][4]))-np.abs(1-np.array(pr['official_result'][4]))
        cpr=quadrature*deltaf;cmd=quadrature*deltacmd
        for name,values,index in [('CPR',cpr,1),('CMD',cmd,2)]:
            expected=sr['official_result'][index]-pr['official_result'][index]
            np.testing.assert_allclose(values.sum(),expected,rtol=1e-9,atol=1e-10)
            verification.append(dict(ranking=rec['key'],metric=name,sum=float(values.sum()),saved_difference=expected,error=float(values.sum()-expected)))
        assert abs(cpr[-1])<1e-10 and abs(cmd[-1])<1e-10
        clean=vector(pp,'nominal',.001,'clean_margin');empty=vector(pp,'nominal',.001,'corrupted_margin')
        den=w@(clean-empty);assert np.all(np.abs(den)>EPS)
        pf=np.stack([vector(pp,'nominal',f,'margin') for f in GRID]);sf=np.stack([vector(sp,'nominal',f,'margin') for f in GRID])
        fbp=(pf@w.T-w@empty)/den;fbs=(sf@w.T-w@empty)/den
        for metric,contrib,bd in [('CPR',cpr,quadrature[:,None]*(fbs-fbp)),('CMD',cmd,quadrature[:,None]*(np.abs(1-fbs)-np.abs(1-fbp)))]:
            for b in ['<=1%','2%-5%','10%-50%','100%']:
                select=np.array([band(f)==b for f in GRID]);value=float(contrib[select].sum())
                bands.append(dict(ranking=rec['key'],method=rec['method'],seed=rec['seed'],k=rec['k'],metric=metric,band=b,
                    contribution=value,signed_accounting_share=value/contrib.sum() if abs(contrib.sum())>=EPS else None))
                boot[rec['method'],rec['k'],metric,b,rec['seed']]=bd[select].sum(0)
        for j,f in enumerate(GRID):
            n=int(f*32491);P=np.zeros(32491,dtype=bool);P[po[:n]]=True;S=np.zeros_like(P);S[so[:n]]=True
            add=S&~P;drop=P&~S;assert add.sum()==drop.sum()
            pq=query(parent['key'],'nominal',f);sq=query(rec['key'],'nominal',f);pm=oldmask(pq);sm=oldmask(sq)
            curves.append(dict(ranking=rec['key'],parent=parent['key'],method=rec['method'],seed=rec['seed'],k=rec['k'],fraction=f,band=band(f),
                delta_cpr=cpr[j],absolute_cpr_contribution=abs(cpr[j]),delta_cmd=cmd[j],absolute_cmd_contribution=abs(cmd[j]),
                added=int(add.sum()),displaced=int(drop.sum()),fraction_changed=float(add.sum()/n),eap_prefix_parent_overlap=int(P[eo[:rec['k']]].sum()),
                parent_actual=int(pm.sum()),splice_actual=int(sm.sum()),post_added=int((sm&~pm).sum()),post_displaced=int((pm&~sm).sum())))
            for which,mask in [('added',add),('displaced',drop),('post_added',sm&~pm),('post_displaced',pm&~sm)]:
                for edge in np.flatnonzero(mask):
                    inventory.append(dict(ranking=rec['key'],fraction=f,kind=which,edge_index=int(edge),edge=mapping.edge.iloc[edge]))
                if f in FRACTIONS and which in ('added','displaced'):
                    ids=np.flatnonzero(mask);values=evalues[ids]
                    base=dict(ranking=rec['key'],method=rec['method'],seed=rec['seed'],k=rec['k'],fraction=f,kind=which,count=len(ids),**counts(values),
                        eap_mean=float(values.mean()) if len(ids) else None,eap_median=float(np.median(values)) if len(ids) else None,
                        eap_q05=float(np.quantile(values,.05)) if len(ids) else None,eap_q95=float(np.quantile(values,.95)) if len(ids) else None,
                        parent_rank_min=int(parent_rank[ids].min()) if len(ids) else None,parent_rank_max=int(parent_rank[ids].max()) if len(ids) else None,
                        parent_rank_median=float(np.median(parent_rank[ids])) if len(ids) else None,negative_head_incidence=float(incident[ids].mean()) if len(ids) else None)
                    base.update({key:value for key,value in overlap(mapping,mask,classes).items() if key.endswith('_recall') or key.endswith('_count')})
                    composition.append(base)
                    for edge in ids:
                        changes.append(dict(ranking=rec['key'],fraction=f,kind=which,edge_index=int(edge),parent_rank=int(parent_rank[edge]),eap_score=float(evalues[edge]),
                            parent=mapping.parent.iloc[edge],child=mapping.child.iloc[edge],qkv=mapping.qkv.iloc[edge],negative_head_incident=bool(incident[edge])))
            if f in FRACTIONS:
                dm=vector(sp,'nominal',f,'margin')-vector(pp,'nominal',f,'margin');dk=vector(sp,'nominal',f,'kl')-vector(pp,'nominal',f,'kl')
                worst=np.sort(dm)[:math.ceil(.1*len(dm))];negative_mass=float(-dm[dm<0].sum());net=float(-dm.sum())
                margins.append(dict(ranking=rec['key'],method=rec['method'],seed=rec['seed'],k=rec['k'],fraction=f,mean=float(dm.mean()),median=float(np.median(dm)),
                    q05=float(np.quantile(dm,.05)),q25=float(np.quantile(dm,.25)),q75=float(np.quantile(dm,.75)),q95=float(np.quantile(dm,.95)),reduced_fraction=float((dm<0).mean()),
                    negative_loss_mass=negative_mass,top10_signed_net_share=float(-worst.sum()/net) if abs(dm.mean())>=EPS else None,
                    top10_negative_mass_share=float(-np.minimum(worst,0).sum()/negative_mass) if negative_mass>EPS else None,
                    kl_lower_fraction=float((dk<0).mean()),kl_change_mean=float(dk.mean()),kl_change_median=float(np.median(dk)),
                    margin_kl_spearman=float(spearmanr(dm,dk).statistic) if np.std(dm)>0 and np.std(dk)>0 else None))
                ids=pp[(pp.selection=='nominal')&np.isclose(pp.target,f)].sort_values('example').template_id.to_numpy()
                tmp=pd.DataFrame(dict(template_id=ids,dm=dm,dk=dk))
                for template,g in tmp.groupby('template_id'):template_rows.append(dict(ranking=rec['key'],fraction=f,template_id=template,count=len(g),margin_mean=g.dm.mean(),kl_mean=g.dk.mean(),reduced_fraction=(g.dm<0).mean()))
                for i in range(len(dm)):per_change.append(dict(ranking=rec['key'],fraction=f,example=i,dm=dm[i],dk=dk[i]))
    pooled=[]
    bframe=pd.DataFrame(bands)
    for (method,k,metric,b),g in bframe.groupby(['method','k','metric','band']):
        lo,hi=interval(np.stack([boot[method,k,metric,b,s] for s in range(42,47)]).mean(0))
        pooled.append(dict(method=method,k=k,metric=metric,band=b,mean=g.contribution.mean(),lower=lo,upper=hi))
    tables={'grid_contributions':curves,'band_contributions':bands,'band_bootstrap':pooled,'decomposition_verification':verification,
        'edge_inventory':inventory,'edge_composition':composition,'edge_attributes':changes,'margin_changes':margins,'margin_by_template':template_rows,'prompt_changes':per_change}
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(out/(name+'.csv'),index=False)
    pooledframe=pd.DataFrame(pooled)
    totals=pooledframe.groupby(['method','k','metric'])['mean'].transform('sum')
    pooledframe['signed_accounting_share']=pooledframe['mean']/totals.where(totals.abs()>=EPS)
    pooledframe.to_csv(out/'band_bootstrap.csv',index=False)
    attrs=pd.DataFrame(changes)
    attrs['source_layer']=attrs.parent.str.extract(r'(\d+)',expand=False).fillna(attrs.parent)
    attrs['destination_layer']=attrs.child.str.extract(r'(\d+)',expand=False).fillna(attrs.child)
    attrs['receiver']=attrs.child+'<'+attrs.qkv+'>'
    for column in ['source_layer','destination_layer','receiver']:
        grouped=attrs.groupby(['ranking','fraction','kind',column]).size().rename('edges').reset_index()
        grouped['fraction_of_set']=grouped.edges/grouped.groupby(['ranking','fraction','kind']).edges.transform('sum')
        grouped.to_csv(out/('edge_'+column+'_composition.csv'),index=False)
    s42=core(p,'baseline',42);per=prompts(s42['key']);seed=[];failures=[];components=[]
    saved_features=pd.read_csv(OLD/'ranking_features.csv')
    masks={t:oldmask(query(s42['key'],'cap',t)) for t in [116,296]}
    for t in [116,296]:
        g=per[(per.selection=='cap')&(per.target==t)];bad=g[g.margin<=0]
        feat=saved_features[(saved_features.ranking==s42['key'])&(saved_features.target==t)].iloc[0].to_dict()
        seed.append(dict(target=t,accuracy=float((g.margin>0).mean()),top1_accuracy=float(g.top1_correct.mean()),failures=len(bad),
            wrong_name_top1=int((bad.top1_id==bad.incorrect_token).sum()),other_top1=int(((bad.top1_id!=bad.incorrect_token)&(bad.top1_id!=bad.correct_token)).sum()),**{k:v for k,v in feat.items() if k not in ('target',)}))
        for template,group in g.groupby('template_id'):failures.append(dict(target=t,template_id=template,examples=len(group),failures=int((group.margin<=0).sum())))
    newmask=masks[296]&~masks[116]
    for edge in np.flatnonzero(newmask):components.append(mapping.iloc[edge].to_dict())
    pd.DataFrame(seed).to_csv(out/'seed42_diagnostics.csv',index=False);pd.DataFrame(failures).to_csv(out/'seed42_template_failures.csv',index=False)
    pd.DataFrame(components).to_csv(out/'seed42_added_edges.csv',index=False)
    # Public edge identities/numeric diagnostics only; no prompt text or logits.
    import gzip
    import shutil
    for name in ['edge_inventory','edge_attributes','prompt_changes']:
        with (out/(name+'.csv')).open('rb') as source, (out/(name+'.csv.gz')).open('wb') as target:
            with gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) as compressed:
                shutil.copyfileobj(source,compressed)
    oldcomponents=set(mapping.loc[masks[116],'parent'])|set(mapping.loc[masks[116],'child'])
    newcomponents=(set(mapping.loc[masks[296],'parent'])|set(mapping.loc[masks[296],'child']))-oldcomponents
    write_json(out/'seed42_new_components.json',sorted(newcomponents))
    plot(out,pd.DataFrame(curves),pd.DataFrame(composition),pd.DataFrame(per_change),evalues[eo[:1624]])
    check_files(protected)
    audit=dict(passed=True,original_protocol_sha256=ident['original_protocol_sha256'],plan_commit=plan_commit,plan_sha256=sha(plan),
        first_nonpositive_eap_rank=first_nonpositive,original_implementation=p['implementation_commit'],model_forwards=0,backwards=0,optimizer_updates=0,
        bootstrap_indices_sha256=sha(OLD/'bootstrap_indices.npy'),CPR_CMD_tolerance='rtol1e-9,atol1e-10; original unrounded JSON faithfulness used',
        frozen_decisions_unchanged=True,seconds=time.perf_counter()-start,source_sha256=sources())
    write_json(out/'stage1_audit.json',audit)
    report(out,audit,pd.DataFrame(signed),pooledframe,pd.DataFrame(curves),pd.DataFrame(margins),pd.DataFrame(seed))
    folder=out/'stage1';folder.mkdir(exist_ok=True)
    # Hash Stage1 products; subsequent Stage2 outputs are independent.
    # Status changes after this function returns; the combined report is extended
    # by Stage2. Neither is an immutable Stage1 measurement artifact.
    paths=[x for x in out.iterdir() if x.is_file() and x.suffix in ('.csv','.json','.md','.gz')
        and x.name not in ['status.json','GATE5G_ADDENDUM_REPORT.md','runtime_projection.json','runtime_approval.json']]+list((out/'figures').glob('stage1*.png'))
    write_json(folder/'outputs.json',{x.relative_to(out).as_posix():sha(x) for x in paths})
    marker(folder,ident,[folder/'outputs.json'])
    print('Stage1 complete; zero model forwards. First nonpositive EAP rank:',first_nonpositive,flush=True)

def plot(out,curves,composition,prompts_,scores):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir(exist_ok=True)
    def save(name):plt.tight_layout();plt.savefig(folder/(name+'.png'),dpi=150);plt.close()
    plt.figure();plt.plot(np.arange(1,len(scores)+1),scores);plt.axhline(0,color='black');plt.xlabel('Original signed EAP rank');plt.ylabel('Attribution');save('stage1_eap_rank')
    plt.figure();plt.hist(scores,bins=50);plt.xlabel('EAP score, top1624');save('stage1_eap_histogram')
    for metric in ['cpr','cmd']:
        plt.figure(figsize=(8,4))
        for (method,k),g in curves.groupby(['method','k']):
            avg=g.groupby('fraction')['delta_'+metric].mean();plt.plot(avg.index,avg.values,'o-',label=f'{method} K{k}')
        plt.xscale('log');plt.axhline(0,color='black');plt.legend();plt.ylabel('Weighted Δ'+metric.upper());plt.xlabel('Nominal fraction');save('stage1_'+metric+'_contributions')
    plt.figure(figsize=(8,4))
    for (kind,),g in composition.groupby(['kind']):
        avg=g.groupby('fraction').negative_head_incidence.mean();plt.plot(avg.index,avg.values,'o-',label=kind)
    plt.legend();plt.ylabel('Negative-name-mover incidence (not attribution sign)');save('stage1_head_composition')
    cols=[c for c in composition if c.endswith('_count')]
    grouped=composition.groupby(['kind','fraction'])[cols].mean()
    plt.figure(figsize=(10,5));plt.imshow(grouped.to_numpy(),aspect='auto');plt.colorbar(label='Mean represented heads per class')
    plt.xticks(range(len(cols)),[c.replace('_count','') for c in cols],rotation=30,ha='right');plt.yticks(range(len(grouped)),[str(x) for x in grouped.index]);save('stage1_all_head_classes')
    plt.figure(figsize=(8,4))
    for f,g in prompts_.groupby('fraction'):plt.hist(g.dm,bins=60,histtype='step',label=str(f),density=True)
    plt.xlabel('Splice − parent prompt margin');plt.legend();save('stage1_margin_distribution')
    plt.figure(figsize=(6,5));sample=prompts_.iloc[::max(1,len(prompts_)//5000)];plt.scatter(sample.dm,sample.dk,s=3,alpha=.2);plt.xlabel('Δ margin');plt.ylabel('Δ KL');save('stage1_margin_kl')

def report(out,audit,signs,bands,curves,margins,seed):
    body='# Gate 5G addendum: post hoc diagnostic measurements\n\nAll analyses are exploratory and post hoc. Original decisions remain **EAP-IG closer** and **Compact gain with CPR cost**. No new model outputs were produced.\n\n'
    body+='## Source identity and attribution semantics\n\n'+str(audit)+'\n\n'
    body+='Gate5F uses negative mean IOI logit difference as its attribution loss. Official hooks form corrupted-minus-clean activations and contract them with loss gradients, giving the corresponding clean-minus-corrupted margin-gradient convention. Positive scores are local/IG attribution under this objective, not exact causal effects. No abs or negation is applied to the saved scores. Source line references and SHA256 are in source_references.json. MAttr signs are not interpreted causally.\n\n'+table(signs)+'\n\n'
    body+=f'First non-positive signed EAP rank: {audit["first_nonpositive_eap_rank"]}.\n\n'
    if int(signs.loc[signs.prefix==324,'negative'].iloc[0])==0:
        body+='The K=324 splice does not introduce negatively attributed edges through its EAP-IG prefix. Negative EAP-IG attribution is therefore not a direct explanation for the observed CPR decrease.\n\n'
    body+='Suppressor-associated incidence and negative attribution remain distinct. Neither head membership nor attribution proves causal function, and other negative causal effects remain possible.\n\n'
    body+='## Exact official CPR/CMD accounting\n\nOriginal unrounded official JSON faithfulness, linear nominal trapezoid weights and absolute-distance CMD integrand reproduce all20 differences at rtol1e-9/atol1e-10. The full100% contribution is zero. Band intervals below use saved shared5000 paired exact-template bootstrap draws and normalize replicate cohort means. They condition on fixed trained rankings, not training-seed population uncertainty.\n\n'+table(bands)+'\n\nSigned shares in band_contributions.csv can exceed100% or be negative when gains/losses cancel; they are accounting shares, not probabilities. Mean-margin normalization is algebra, not an independent discovery.\n\n'
    body+='## Insertions and displacements\n\n'+table(curves[curves.fraction.isin(FRACTIONS)].groupby(['method','k','fraction'])[['added','displaced','fraction_changed','parent_actual','splice_actual','post_added','post_displaced']].mean().reset_index())+'\n\nFull pre/post-pruning identities are in edge_inventory.csv. edge_composition.csv and edge_attributes.csv retain signed EAP scores, score quantiles, parent ranks, every known-head class, suppressor incidence and source/destination components separately.\n\n'
    body+='## Prompt margin/KL changes\n\n'+table(margins.groupby(['method','k','fraction'])[['mean','median','reduced_fraction','top10_signed_net_share','top10_negative_mass_share','kl_lower_fraction','kl_change_mean','margin_kl_spearman']].mean().reset_index())+'\n\nPer-comparison quantiles, negative mass and template distributions remain in margin_changes.csv and margin_by_template.csv. Near-zero net changes get no signed concentration ratio. CPR loss, prompt margin loss and KL improvement/regression are distinct.\n\n'
    body+='## Seed42 saved-data diagnostics\n\n'+table(seed[['target','accuracy','top1_accuracy','failures','wrong_name_top1','other_top1','known_heads','complete_name_mover_path','eap_jaccard','nominal_prefix','pruned_edges']])+'\n\nNew components are in seed42_new_components.json, edges in seed42_added_edges.csv, and template failures in seed42_template_failures.csv. These descriptors do not identify a causally missing edge.\n\n'
    body+='## Stage2 — not executed\n\nDisplacement/insertion/non-additivity, adaptive search and fresh confirmation remain unmeasured. The user must run the bounded, frozen manual pipeline. Remove/Add vary budgets; pruning is nonlinear. Circuit-level non-additivity does not isolate individual-edge neural interactions. Candidate restoration also changes size and is not an equal-size superiority claim.\n\n'
    body+=f'## Resources\n\nCPU wall seconds: {audit["seconds"]:.3f}; model forwards/backwards/optimizer updates:0/0/0. No original artifacts or classifications changed.\n\n'
    for path in sorted((out/'figures').glob('stage1*.png')):body+=f'![{path.stem}](figures/{path.name})\n\n'
    body+='## Interpretation and recommendation (researcher-written)\n'
    (out/'GATE5G_ADDENDUM_REPORT.md').write_text(body,encoding='utf-8')
    (ROOT/'GATE5G_ADDENDUM_REPORT.md').write_text(body.replace('(figures/',f'(results/mattr_gate5g_addendum/{out.name}/figures/'),encoding='utf-8')
