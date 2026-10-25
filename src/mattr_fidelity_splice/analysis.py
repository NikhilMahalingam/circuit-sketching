"""Offline analysis, original registered decisions and ten reproducible figures."""
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.mattr_optimization.common import ROOT, GRID, sha, write_json
from .core import protocol, identity, verified, read, HISTORY
from .statistics import resamples,weights,interval,classify_a,classify_c,cpr_bootstrap,official_areas

INTERPRETATION = '## Interpretation and recommendation (researcher-written)'
REPORT = 'GATE5G_FIDELITY_SPLICE_REPORT.md'


def table(frame):
    # Avoid the optional tabulate dependency.
    if frame.empty:return 'No applicable observations.'
    f=frame.copy()
    for col in f:
        if pd.api.types.is_float_dtype(f[col]):f[col]=f[col].map(lambda v:f'{v:.6g}')
    return '| '+' | '.join(map(str,f.columns))+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in f.itertuples(index=False,name=None))


def completeness(out,p):
    missing=[]
    for j in p['jobs']:
        if not verified(out/'evaluations'/j['id'],identity(out,p,j)):missing.append(j['id'])
    write_json(out/'completeness.json',dict(complete=not missing,missing=missing,registered_jobs=len(p['jobs'])))
    if missing:
        write_json(out/'gate5g_decision.json',dict(part_a='Invalid/incomplete',part_c='Invalid/incomplete',missing=missing))
        raise ValueError(f'{len(missing)} registered jobs incomplete; no valid scientific classification')


def analyze(out):
    start=time.perf_counter();p=protocol(out);completeness(out,p)
    frame=pd.read_csv(out/'cohort_manifest.csv',keep_default_na=False)
    all_metrics=[];prompts={};summaries={}
    for j in p['jobs']:
        folder=out/'evaluations'/j['id']
        metrics=pd.read_csv(folder/'metrics.csv');per=pd.read_csv(folder/'per_prompt.csv')
        assert len(per)==len(metrics)*len(frame)
        meta=next(r for r in p['rankings'] if r['key']==j['ranking'])
        for k in ('method','seed','dataset_size','step','kind'):metrics[k]=meta[k]
        metrics['k']=meta.get('k',0)
        all_metrics.append(metrics);prompts[j['id']]=per;summaries[j['id']]=read(folder/'summary.json')
    metrics=pd.concat(all_metrics,ignore_index=True)
    metrics.to_csv(out/'behavioral_metrics.csv',index=False)
    def vector(key,selection,target,col):
        per=prompts[key]
        sub=per[(per.selection==selection)&np.isclose(per.target,target)].sort_values('example')
        assert sub.example.tolist()==list(range(len(frame)))
        return sub[col].to_numpy()
    idx=resamples(frame.template_id.to_numpy());np.save(out/'bootstrap_indices.npy',idx)
    w=weights(idx,len(frame))
    baseline=[r for r in p['rankings'] if r['kind']=='core' and r['method']=='baseline']
    baseline=sorted(baseline,key=lambda r:r['seed'])
    eap=next(r for r in p['rankings'] if r['method']=='eap_ig')
    e=np.stack([vector(eap['key'],'nominal',f,'kl') for f in (.1,.2,.5)])
    b=np.stack([[vector(r['key'],'nominal',f,'kl') for f in (.1,.2,.5)] for r in baseline])
    a_label,a_rows=classify_a(e,b,w)
    pd.DataFrame(a_rows).to_csv(out/'part_a_registered_comparisons.csv',index=False)
    all_c=[];c_boot={};point={}
    for rec in [r for r in p['rankings'] if r['kind']=='splice']:
        parent=rec['parent'];key=rec['key']
        base_per=prompts[parent];base_nom=base_per[(base_per.selection=='nominal')&np.isclose(base_per.target,.001)].sort_values('example')
        clean=base_nom.clean_margin.to_numpy();empty=base_nom.corrupted_margin.to_numpy()
        distributions=[]
        for name in (parent,key):
            margins=np.stack([vector(name,'nominal',f,'margin') for f in GRID])
            values,bad=cpr_bootstrap(margins,clean,empty,w)
            area=official_areas(margins.mean(-1),clean.mean(),empty.mean())
            np.testing.assert_allclose(area[:2],summaries[name]['official_result'][1:3],rtol=2e-5,atol=2e-5)
            distributions.append(values)
        c_boot[key]=distributions[1]-distributions[0]
        delta=vector(key,'cap',116,'margin')>0
        delta=delta.astype(float)-(vector(parent,'cap',116,'margin')>0)
        point[key]=summaries[key]['official_result'][1]-summaries[parent]['official_result'][1]
        ci=interval(c_boot[key]);aci=interval(delta@w.T)
        all_c.append(dict(ranking=key,parent=parent,method=rec['method'],seed=rec['seed'],k=rec['k'],
            cpr_difference=point[key],cpr_lower=ci[0],cpr_upper=ci[1],accuracy_difference=float(delta.mean()),
            accuracy_lower=aci[0],accuracy_upper=aci[1],unstable_replicates=int(bad.sum())))
    c_rows=pd.DataFrame(all_c);c_rows.to_csv(out/'splice_paired_differences.csv',index=False)
    primary=[r for r in p['rankings'] if r['kind']=='splice' and r['method']=='baseline' and r['k']==324]
    differences=np.stack([(vector(r['key'],'cap',116,'margin')>0).astype(float)-(vector(r['parent'],'cap',116,'margin')>0) for r in primary])
    c_decision=classify_c([point[r['key']] for r in primary],np.stack([c_boot[r['key']] for r in primary]),differences,w)
    topo=pd.read_csv(out/'splice_topology.csv')
    c_decision['eap_identical_primary_masks']=topo[topo.ranking.isin([r['key'] for r in primary])].capped116_identical.tolist()
    write_json(out/'gate5g_decision.json',dict(part_a=a_label,part_a_endpoints=a_rows,part_c=c_decision,
        prompt_intervals_conditional_on_fixed_models=True,seed_population_significance_claim=False))
    # Prominent cap116 endpoint, separately by seed and with fixed seed means.
    cap=metrics[(metrics.selection=='cap')&(metrics.target==116)&(metrics.kind=='core')]
    cap.to_csv(out/'cap116_fidelity_per_seed.csv',index=False)
    cap.groupby('method')[['kl','accuracy','top1_accuracy','sign_agreement','top1_agreement','margin_error_mean']].mean().to_csv(out/'cap116_fidelity_summary.csv')
    from .structure import features
    feat=features(out,p)
    compact=metrics[(metrics.selection=='cap')&metrics.target.isin([116,296])]
    diag=feat.merge(compact[['ranking','target','accuracy','top1_accuracy']],on=['ranking','target'],validate='one_to_one')
    diag.to_csv(out/'seed_diagnostics.csv',index=False)
    associations=[]
    exclude={'seed','dataset_size','step','target','actual_edges','accuracy','top1_accuracy'}
    for target,g in diag.groupby('target'):
        raw_count=len(g)
        if 'canonical_order_sha256' in g:g=g.drop_duplicates('canonical_order_sha256')
        for col in g.select_dtypes(include=['number','bool']).columns:
            if col in exclude:continue
            valid=g[[col,'accuracy']].dropna()
            rho=float(spearmanr(valid[col],valid.accuracy).statistic) if valid[col].nunique()>1 and valid.accuracy.nunique()>1 else None
            associations.append(dict(target=target,feature=col,spearman=rho,n=len(valid),archival_rows=raw_count,
                inference='exact duplicate complete orders collapsed; related checkpoints not independent; no significance test'))
    associations=pd.DataFrame(associations)
    associations.to_csv(out/'feature_associations.csv',index=False)
    pairs=[]
    for (n,step,seed,target),g in diag.groupby(['dataset_size','step','seed','target']):
        a=g[g.method=='baseline'];h=g[g.method=='h2']
        # Duplicate archival aliases remain documented, but do not create extra paired units.
        if a.empty or h.empty:continue
        a=a.iloc[0];h=h.iloc[0]
        data=dict(dataset_size=n,step=step,seed=seed,target=target,accuracy_difference=h.accuracy-a.accuracy)
        for col in ['destination_entropy','complete_name_mover_path','eap_jaccard','full_signed_spearman_eap','known_heads']:
            data[col+'_difference']=float(h[col])-float(a[col])
        pairs.append(data)
    pd.DataFrame(pairs).to_csv(out/'paired_feature_changes.csv',index=False)
    cases=[]
    for seed in (42,44,46):
        r=next(r for r in baseline if r['seed']==seed)
        sub=prompts[r['key']];sub=sub[(sub.selection=='cap')&sub.target.isin([116,296])].copy()
        sub['incorrect_name_top1']=sub.top1_id==sub.incorrect_token
        sub['other_token_top1']=(sub.top1_id!=sub.incorrect_token)&(sub.top1_id!=sub.correct_token)
        sub['positive_margin']=sub.margin>0
        sub['seed']=seed
        cases.append(sub.groupby(['seed','target','template_id'])[['positive_margin','incorrect_name_top1','other_token_top1','sign_agreement']].mean().reset_index())
    pd.concat(cases).to_csv(out/'seed42_template_errors.csv',index=False)
    # Training logs are read-only historical diagnostics; no training/stream-swap experiment.
    training=[]
    for r in [r for r in p['rankings'] if r['kind']=='core' and r['method']!='eap_ig']:
        import torch
        state=torch.load(ROOT/r['checkpoint'],map_location='cpu',weights_only=False)
        log=np.asarray(state['train_log'],dtype=float)
        training.append(dict(ranking=r['key'],seed=r['seed'],method=r['method'],updates=len(log),
            loss_mean=float(log[:,-1].mean()),loss_last100_mean=float(log[-100:,-1].mean()),
            k_mean=float(log[:,1].mean()),checkpoint_sha256=r['checkpoint_sha256']))
    pd.DataFrame(training).to_csv(out/'historical_training_diagnostics.csv',index=False)
    compute=[]
    for path in (out/'attempts').rglob('compute_accounting.csv'):
        if path.stat().st_size<=1:continue
        f=pd.read_csv(path);f['attempt']=str(path.parent.relative_to(out));compute.append(f)
    compute=pd.concat(compute,ignore_index=True) if compute else pd.DataFrame()
    compute.to_csv(out/'compute_accounting.csv',index=False)
    costs=[]
    eap_file=HISTORY/'attribution/eap_ig_n10000/attribution.json'
    eap_seconds=read(eap_file)['seconds'] if eap_file.exists() else 0.
    reuse_file=HISTORY/'reused_training_cost.csv'
    reuses=pd.read_csv(reuse_file) if reuse_file.exists() else pd.DataFrame()
    for rec in [r for r in p['rankings'] if r['kind']=='core']:
        seconds=0.
        if rec['method']!='eap_ig':
            job=f'{rec["method"]}_n10000_s{rec["seed"]}'
            for path in (HISTORY/'attempts').glob('train_'+job+'/*/rates.json'):
                seconds+=read(path).get('training_seconds',0.)
            if not reuses.empty:seconds+=reuses.loc[reuses.job==job,'historical_training_seconds'].sum()
        costs.append(dict(ranking=rec['key'],method=rec['method'],seed=rec['seed'],
            parent_training_stage_seconds=seconds,eap_attribution_seconds=eap_seconds,
            combined_splice_discovery_seconds=seconds+eap_seconds,
            scope='reused discovery timers, not new inference; one common EAP attribution is shared; CPU construction/evaluation additionally charged; not end-to-end training-process timers'))
    pd.DataFrame(costs).to_csv(out/'historical_discovery_cost.csv',index=False)
    figures(out,metrics,prompts,c_rows,diag)
    old=(out/REPORT).read_text(encoding='utf-8') if (out/REPORT).exists() else ''
    interpretation=old.split(INTERPRETATION,1)[1] if INTERPRETATION in old else '\n\n'
    nominal=[]
    for r in p['rankings']:
        result=summaries[r['key']]['official_result']
        if result:nominal.append(dict(ranking=r['key'],method=r['method'],seed=r['seed'],k=r.get('k',0),CPR=result[1],CMD=result[2],accuracy_log_AUC=result[6]))
    nominal=pd.DataFrame(nominal);nominal.to_csv(out/'official_metrics.csv',index=False)
    body='# Gate 5G: behavioral fidelity, seed diagnosis and ranking splicing\n\n'
    body+=f'Frozen implementation: `{p["implementation_commit"]}`. Protocol SHA-256: `{sha(out/"gate5g_protocol.json")}`.\n\n'
    body+='Historical Gate5F capped116 accuracy: ordinary66.17%, H285.82%, EAP99.02%; CPR10.536,10.429,2.022. These are context only; all new comparisons below use the fresh Gate5G cohort.\n\n'
    body+=f'## Cohort and ranking audits\n\n{len(frame)} official validation prompts; seed5545; nine exact-template strata; {p["cohort"]["eligible"]} eligible after clean/donor/cross-role exclusions. Gate1 evaluated the entire validation split historically. No private test accessed. Rankings, checkpoint/effective exports, model/tokenizer identity and official masks were hash-verified. All jobs record the frozen implementation; later results commits are allowed.\n\n'
    body+='## EAP convention and splice correctness\n\nSigned descending CPU torch.argsort; absolute=False; no score negation or transformation in Gate5F. Ties follow the pinned CPU implementation. Canonical orders are frozen; strictly positive unique ordinal scores preserve all tested top-n sets and official masks in signed/absolute selection. See eap_ordering.json for source line references, splice_topology.csv for cap116 identity and required prefix. K=324 and1624 were fixed before outcomes.\n\n'
    body+=f'## Part A — registered KL comparison\n\nClassification: **{a_label}**. Full-vocabulary KL(p_full||p_circuit); lower is closer.\n\n'+table(pd.DataFrame(a_rows))+'\n\n98.333333% marginal intervals provide Bonferroni simultaneous95% coverage over these three endpoints. H2 is descriptive. Prompt uncertainty conditions on fixed trained seeds and template mix; no training-seed significance claim.\n\n'
    body+='## Secondary endpoint: KL and accuracy at capped116 actual edges\n\n'+table(cap[['method','seed','actual_edges','kl','accuracy','top1_accuracy','sign_agreement','top1_agreement','margin_error_mean']])+'\n\n'+table(cap.groupby('method')[['kl','accuracy','top1_accuracy']].mean().reset_index())+'\n\nCapped budgets can differ in actual retained counts; this is not universally exact-size matching.\n\n'
    body+='## Margin fidelity and amplification\n\nFull metrics in behavioral_metrics.csv include mean/median absolute error, signed differences, signed ratios only abs(clean)>=.5, excluded counts, IQR, fractions above2/5, Pearson/Spearman, full-model incorrect subset accuracy, sign and vocabulary-top1 agreement. Ratios never use near-zero denominators. Official aggregate faithfulness is not replaced by prompt-ratio averages.\n\n'
    margin_cols=['method','seed','margin_error_mean','amplification_median','amplification_q25','amplification_q75','amplification_above2','amplification_above5','amplification_excluded']
    shown=metrics[(metrics.kind=='core')&(metrics.selection=='nominal')&np.isclose(metrics.target,.1)]
    body+='Per-ranking descriptive margin diagnostics at nominal10% (remaining endpoints remain in the full CSV):\n\n'+table(shown[[c for c in margin_cols if c in shown]])+'\n\n'
    body+='## Part B — exploratory seed diagnosis\n\n'+table(diag[(diag.dataset_size==10000)&(diag.step==5000)&(diag.method=='baseline')&diag.seed.isin([42,44,46])][['seed','target','accuracy','actual_edges','nominal_prefix','complete_name_mover_path','known_heads','eap_jaccard','destination_entropy']])+'\n\nSee feature_associations.csv (sample sizes, descriptive correlations only), paired_feature_changes.csv, seed42_edge_case_study.csv, seed42_template_errors.csv and historical_training_diagnostics.csv. All positive, weak and contradictory correlations are retained. Same-seed checkpoints are dependent. N128 additionally varies subsets; full-pool runs vary row/k streams. These observations do not isolate causal RNG-stream effects. No stream-swap runs occurred. Rank-scale differences between parameterizations preclude interpreting raw score gaps as comparable calibrated importance. No outcome-selected predictor was used.\n\n'
    body+='Prespecified feature associations (including weak/undefined observations; duplicate complete orders collapsed):\n\n'+table(associations[['target','feature','spearman','n','archival_rows']])+'\n\n'
    body+=f'## Part C — registered splice comparison\n\nClassification: **{c_decision["classification"]}**. Equally seed-weighted CPR difference {c_decision["cpr_difference"]:.6f},95% interval {c_decision["cpr_interval"]}; capped116 accuracy difference {c_decision["accuracy_difference"]:.6f},95% interval {c_decision["accuracy_interval"]}. Unstable normalization draws: {c_decision["unstable_replicates"]}.\n\n'
    body+=f'EAP-identical cap116 masks by construction (primary seeds42–46): {c_decision["eap_identical_primary_masks"]}. Identical-prefix compact gains are expected by construction; preservation of full-grid CPR is the additional test.\n\n'+table(c_rows)+'\n\nCPR bootstrap normalizes replicate cohort-mean margins then applies the exact official grid/quadrature; point results were verified against the evaluator. Joint PartC criteria use separate unadjusted95% intervals, not a global A/C familywise guarantee. K1624 and H2 splices remain secondary.\n\n'
    body+='## Official metrics and boundary behavior\n\n'+table(nominal)+'\n\nBoundary +/-1 queries and full nominal curves are saved, including actual retained counts. Splicing preserves EAP prefixes up toK, but can change circuits far aboveK. This is a deterministic ranking transformation, not learned attribution.\n\n'
    body+='## Compute and limitations\n\n'
    if not compute.empty:
        body+=f'Recorded stages: {compute.seconds.sum()/3600:.3f}h; actual root forwards {int(compute.forward_batches.sum())}; example-forwards {int(compute.example_forwards.sum())}; backward calls/optimizer updates0/0. CUDA allocated peak {compute.peak_allocated_mib.max():.1f}MiB; reserved peak {compute.peak_reserved_mib.max():.1f}MiB; sampled CPU RSS {compute.sampled_peak_cpu_rss_mib.max():.1f}MiB. These are allocator/RSS figures, not physical VRAM occupancy or GPU utilization. Failed/resumed attempt costs are included.\n\n'
    body+='Graph-only time is in graph_analysis_compute.csv; CPU preparation and analysis are separate. Cache hits execute zero circuit forwards, while each job charges clean sample validation. Splice method cost includes existing EAP attribution plus parent MAttr training and CPU construction, not inference alone. Fixed seed and template limitations remain; no leaderboard submission or further research phase occurred.\n\n'
    for path in sorted((out/'figures').glob('*.png')):body+=f'![{path.stem}](figures/{path.name})\n\n'
    body+=INTERPRETATION+interpretation
    (out/REPORT).write_text(body,encoding='utf-8')
    (ROOT/REPORT).write_text(body.replace('(figures/',f'(results/mattr_fidelity_splice/{out.name}/figures/'),encoding='utf-8')
    write_json(out/'analysis_metadata.json',dict(protocol_sha256=sha(out/'gate5g_protocol.json'),implementation_commit=p['implementation_commit'],
        bootstrap_indices_sha256=sha(out/'bootstrap_indices.npy'),seconds=time.perf_counter()-start,model_forwards=0,
        caveat='prompt intervals conditional on fixed rankings; PartsA/C not one global familywise95%claim'))
    print('Analysis complete:',a_label,'/',c_decision['classification'],flush=True)


def figures(out,metrics,prompts,c_rows,diag):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=out/'figures';dest.mkdir(exist_ok=True)
    core=metrics[metrics.kind=='core']
    def save(name):
        plt.tight_layout();plt.savefig(dest/(name+'.png'),dpi=160);plt.close()
    for selection,name,x in [('nominal','01_kl_nominal','target'),('cap','01_kl_actual','actual_edges')]:
        plt.figure(figsize=(7,4))
        for method,g in core[core.selection==selection].groupby('method'):
            avg=g.groupby('target')[[x,'kl']].mean() if x!='target' else g.groupby('target').kl.mean().reset_index()
            plt.plot(avg[x],avg.kl,'o-',label=method)
        plt.xscale('log');plt.xlabel('Nominal fraction' if selection=='nominal' else 'Actual edges (seed mean)');plt.ylabel('Full-vocabulary forward KL');plt.legend();save(name)
    cap=core[(core.selection=='cap')&(core.target==116)]
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,col in zip(axs,['kl','accuracy']):
        cap.groupby('method')[col].mean().plot.bar(ax=ax);ax.set_title(f'Cap116: {col}');ax.tick_params(axis='x',rotation=20)
    save('02_cap116_kl_accuracy')
    plt.figure(figsize=(7,4))
    for method,g in core[(core.selection=='nominal')&np.isclose(core.target,.1)].groupby('method'):
        ratios=[]
        for key in g.ranking:
            per=prompts[key];sub=per[(per.selection=='nominal')&np.isclose(per.target,.1)]
            sub=sub[sub.clean_margin.abs()>=.5];ratios.extend((sub.margin/sub.clean_margin).tolist())
        if ratios:
            low,high=min(ratios),max(ratios)
            if high-low<1e-8:low-=.5;high+=.5
            plt.hist(ratios,bins=np.linspace(low,high,41),histtype='step',label=method,density=True)
    plt.xlabel('Signed margin amplification, |clean|>=0.5 (nominal10%)');plt.ylabel('Density');plt.legend();save('03_margin_amplification')
    plt.figure(figsize=(6,5))
    for method,g in core[(core.selection=='cap')&(core.target==116)].groupby('method'):
        key=g.ranking.iloc[0];s=prompts[key];s=s[(s.selection=='cap')&(s.target==116)]
        plt.scatter(s.clean_margin,s.margin,s=7,alpha=.4,label=method+' (seed42)')
    plt.xlabel('Full clean margin');plt.ylabel('Circuit margin (cap116)');plt.legend();save('04_margin_scatter')
    plt.figure(figsize=(7,4))
    for method,g in diag[diag.target==116].groupby('method'):plt.scatter(g.eap_jaccard,g.accuracy,label=method)
    plt.xlabel('EAP pruned-mask Jaccard, cap116');plt.ylabel('Positive-margin accuracy');plt.legend(fontsize=7);save('05_accuracy_features')
    plt.figure(figsize=(7,4))
    selected=diag[(diag.target==116)&(diag.dataset_size==10000)&(diag.step==5000)&diag.seed.isin([42,44,46])]
    for method,g in selected.groupby('method'):plt.plot(g.seed,g.known_heads,'o',label=method)
    plt.xlabel('Training seed');plt.ylabel('Known heads incident to cap116 circuit');plt.legend();save('06_head_coverage')
    plt.figure(figsize=(7,4))
    for (method,k),g in c_rows.groupby(['method','k']):plt.plot(g.seed,g.accuracy_difference,'o-',label=f'{method} K{k}')
    plt.axhline(0,color='black',lw=1);plt.xlabel('Training seed');plt.ylabel('Splice − parent accuracy at cap116');plt.legend(fontsize=7);save('07_splice_accuracy')
    plt.figure(figsize=(7,4))
    for (method,k),g in c_rows.groupby(['method','k']):
        plt.errorbar(g.seed,g.cpr_difference,yerr=[g.cpr_difference-g.cpr_lower,g.cpr_upper-g.cpr_difference],fmt='o',label=f'{method} K{k}')
    plt.axhline(-.05,color='red',ls='--',label='−0.05 CPR');plt.xlabel('Seed');plt.ylabel('Splice − parent CPR (prompt95%CI)');plt.legend(fontsize=7);save('08_splice_cpr')
    boundary=metrics[metrics.selection=='boundary']
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for k,g in boundary.groupby('k'):
        g=g.copy();g['offset']=g.nominal_n-k
        avg=g.groupby('offset')[['accuracy','actual_edges']].mean()
        axs[0].plot(avg.index,avg.accuracy,'o-',label=f'K{k}');axs[1].plot(avg.index,avg.actual_edges,'o-',label=f'K{k}')
    for ax in axs:ax.set_xlabel('Nominal prefix − K');ax.legend()
    axs[0].set_ylabel('Accuracy (10-parent mean)');axs[1].set_ylabel('Actual edges (mean)');save('09_boundary')
    plt.figure(figsize=(7,4))
    for method,g in metrics[(metrics.selection=='nominal')&np.isclose(metrics.target,.1)].groupby('method'):
        x=[];y=[]
        from .core import read
        for r in g.itertuples():
            x.append(read(out/'evaluations'/r.ranking/'summary.json')['official_result'][1]);y.append(r.kl)
        plt.scatter(x,y,label=method)
    plt.xlabel('Official CPR');plt.ylabel('Forward KL at nominal10%');plt.legend();save('10_cpr_kl')
