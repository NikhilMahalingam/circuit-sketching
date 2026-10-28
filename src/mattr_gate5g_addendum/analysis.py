"""Offline Stage2 contrasts, conditional paired bootstrap, audits and packaging."""
import shutil
from .common import *
from src.mattr_fidelity_splice.statistics import resamples,weights,interval

def reading(remove,add,splice,interaction,ci):
    if abs(splice)<EPS:return ['relative thresholds unstable']
    labels=[]
    if np.sign(remove)==np.sign(splice) and abs(remove)>=.75*abs(splice):labels.append('mostly displacement')
    if np.sign(add)==np.sign(splice) and abs(add)>=.75*abs(splice):labels.append('mostly insertion')
    if ci[0]>=-.1*abs(splice) and ci[1]<=.1*abs(splice):labels.append('approximately additive')
    if (ci[0]>0 or ci[1]<0) and abs(interaction)>.25*abs(splice):labels.append('non-additive')
    return labels or ['no descriptive label']

def four_contrasts(frames,w):
    ref=frames['parent'];den=(ref.clean_margin-ref.empty_margin).mean()
    bden=w@(ref.clean_margin-ref.empty_margin).to_numpy()
    if abs(den)<EPS or (np.abs(bden)<EPS).any():raise ValueError('Unstable bootstrap normalization; no repaired denominator')
    point={k:(g.margin.mean()-ref.empty_margin.mean())/den for k,g in frames.items()}
    rep={k:(w@g.margin.to_numpy()-w@ref.empty_margin.to_numpy())/bden for k,g in frames.items()}
    vals={k:point[k]-point['parent'] for k in ['remove','add','splice']}
    draws={k:rep[k]-rep['parent'] for k in ['remove','add','splice']}
    vals['interaction']=vals['splice']-vals['remove']-vals['add']
    draws['interaction']=draws['splice']-draws['remove']-draws['add']
    return vals,draws

def completed(out,p):
    context=dict(protocol_sha256=sha(out/'stage2_protocol.json'),implementation_commit=p['implementation_commit'],role='fresh_confirmation',cohort_sha256=p['cohort_sha256'])
    for rec in p['circuits']:
        if not verified(out/'evaluations'/rec['id'],dict(**context,circuit=rec)):raise ValueError('Incomplete job: '+rec['id'])

def run(out):
    p=stage2_protocol(out);completed(out,p)
    frame=pd.read_csv(out/'confirmation_cohort.csv');indices=resamples(frame.template_id.to_numpy(),5000,5646);w=weights(indices,len(frame))
    np.save(out/'stage2_bootstrap_indices.npy',indices)
    arrays={};results=[]
    for rec in p['circuits']:
        dest=out/'evaluations'/rec['id'];result=read(dest/'result.json');f=pd.read_csv(dest/'per_prompt.csv')
        assert len(f)==len(frame);arrays[rec['id']]=f
        results.append({k:v for k,v in rec.items() if k not in ['selected_edges']}|result['metrics'])
    pd.DataFrame(results).to_csv(out/'stage2_circuit_results.csv',index=False)
    contrasts=[];boots={}
    for seed in range(42,47):
        for fraction in FRACTIONS:
            group={r['kind']:r for r in p['circuits'] if r['stage']=='displacement' and r['seed']==seed and r['fraction']==fraction}
            ff={kind:arrays[r['id']] for kind,r in group.items()}
            vals,draws=four_contrasts(ff,w)
            for key in vals:
                lo,hi=interval(draws[key]);contrasts.append(dict(seed=seed,fraction=fraction,quantity=key,value=float(vals[key]),lower=lo,upper=hi))
                boots[seed,fraction,key]=draws[key]
    per=pd.DataFrame(contrasts);per.to_csv(out/'nonadditivity_per_seed.csv',index=False)
    pooled=[];labels=[]
    for fraction in FRACTIONS:
        values={};cis={}
        for key in ['remove','add','splice','interaction']:
            values[key]=float(per[(per.fraction==fraction)&(per.quantity==key)].value.mean())
            lo,hi=interval(np.stack([boots[s,fraction,key] for s in range(42,47)]).mean(0));cis[key]=[lo,hi]
            pooled.append(dict(fraction=fraction,quantity=key,value=values[key],lower=lo,upper=hi))
        labels.append(dict(fraction=fraction,labels=reading(values['remove'],values['add'],values['splice'],values['interaction'],cis['interaction'])))
    pd.DataFrame(pooled).to_csv(out/'nonadditivity_pooled.csv',index=False);write_json(out/'descriptive_reading.json',labels)
    c={r['kind']:r for r in p['circuits'] if r['stage']=='confirmation'}
    base=(arrays[c['base116']['id']].margin.to_numpy()>0).astype(float)
    aug=(arrays[c['candidate']['id']].margin.to_numpy()>0).astype(float)
    big=(arrays[c['base296']['id']].margin.to_numpy()>0).astype(float)
    gain=aug-base;gap=big-base;dg=w@gap;valid=np.abs(dg)>=EPS
    ratio=np.divide(w@gain,dg,out=np.full(len(w),np.nan),where=valid)
    gci=interval(w@gain);rci=interval(ratio)
    confirmation=dict(accuracy_gain=float(gain.mean()),gain_interval=gci,
        gap_recovered=float(gain.mean()/gap.mean()) if abs(gap.mean())>=EPS else None,ratio_interval=rci,
        unstable_ratio_replicates=int((~valid).sum()),size_caveat='budgets differ; no equal-size superiority claim',
        confirmation='positive held-out gain' if gain.mean()>0 and gci[0]>0 else 'failed or inconclusive confirmation')
    write_json(out/'confirmation_results.json',confirmation)
    candidate=checked_json(out/'candidate.json');mapping=pd.read_csv(OLD/'edge_mapping.csv',keep_default_na=False)
    old=original();ea=next(r for r in old['rankings'] if r['method']=='eap_ig');eo=order(ea);er=np.argsort(eo)+1;es=load_scores(ea).numpy()
    profile=mapping.iloc[candidate['edges']].copy();profile['mattr_rank']=candidate['ranks'];profile['eap_rank']=er[candidate['edges']];profile['eap_score']=es[candidate['edges']]
    from src.mattr_data_scaling.structure import head
    classes=read(ROOT/'results/mattr_data_scaling/gate5f/known_ioi_heads.json')['classes']
    profile['known_classes']=[';'.join(k for k,v in classes.items() if head(a) in set(map(tuple,v)) or head(b) in set(map(tuple,v))) for a,b in zip(profile.parent,profile.child)]
    profile.to_csv(out/'confirmed_candidate_profile.csv',index=False)
    compute=[]
    for file in (out/'attempts').rglob('compute_accounting.csv'):
        if file.stat().st_size>1:
            f=pd.read_csv(file);f['attempt']=str(file.parent.relative_to(out));compute.append(f)
    costs=pd.concat(compute,ignore_index=True) if compute else pd.DataFrame();costs.to_csv(out/'stage2_compute.csv',index=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,4))
    for key,g in pd.DataFrame(pooled).groupby('quantity'):plt.errorbar(g.fraction,g.value,yerr=[g.value-g.lower,g.upper-g.value],fmt='o-',label=key)
    plt.axhline(0,color='black');plt.legend();plt.ylabel('Normalized faithfulness change');plt.savefig(out/'figures/stage2_nonadditivity.png',dpi=150);plt.close()
    report=out/'GATE5G_ADDENDUM_REPORT.md';body=report.read_text(encoding='utf-8')
    body=body.split('## Stage2 results')[0].split('## Interpretation and recommendation (researcher-written)')[0]
    body+='## Stage2 results\n\nPost hoc exploratory interventions; no new full-grid CPR calculated. Remove/Add change sizes and official pruning can change connectivity.\n\n'+table(pd.DataFrame(pooled))+'\n\n'+str(labels)+'\n\n'+str(confirmation)+'\n\n'
    body+='Search trace and candidate_set_sha256 are frozen in candidate.json before any fresh-cohort forward. Seed-specific estimates and circuit sizes are saved separately. Conditional prompt-bootstrap intervals do not estimate training-seed population uncertainty.\n\n'
    body+='![Stage2 non-additivity](figures/stage2_nonadditivity.png)\n\n## Interpretation and recommendation (researcher-written)\n\n'
    report.write_text(body,encoding='utf-8');(ROOT/'GATE5G_ADDENDUM_REPORT.md').write_text(body.replace('(figures/',f'(results/mattr_gate5g_addendum/{out.name}/figures/'),encoding='utf-8')
    write_json(out/'stage2_analysis.json',dict(passed=True,protocol_sha256=sha(out/'stage2_protocol.json'),bootstrap_sha256=sha(out/'stage2_bootstrap_indices.npy'),model_forwards=0))

def audit(out):
    p=stage2_protocol(out);completed(out,p);original()
    assert read(out/'stage2_analysis.json')['passed']
    candidate=checked_json(out/'candidate.json');assert sha(out/'candidate.json')==p['candidate_sha256']
    assert candidate['candidate_set_sha256']==digest(sorted(candidate['edges'])) and not candidate['confirmation_used']
    fresh=pd.read_csv(out/'confirmation_cohort.csv');oldframe=pd.read_csv(OLD/'cohort_manifest.csv')
    assert len(fresh)>=256 and len(set(fresh.clean_sha256)|set(fresh.corrupted_sha256))==2*len(fresh)
    assert not (set(fresh.clean_sha256)|set(fresh.corrupted_sha256))&(set(oldframe.clean_sha256)|set(oldframe.corrupted_sha256))
    for rec in read(out/'confirmation_audit.json')['exclusions']:
        assert sha(ROOT/rec['path'])==rec['sha256']
        oldrows=pd.read_csv(ROOT/rec['path'],keep_default_na=False)
        from src.mattr_compact_validation.design import text_hash
        for col in ['clean_prompt','corrupted_prompt']:
            if col in oldrows:assert not (set(fresh.clean_sha256)|set(fresh.corrupted_sha256))&{text_hash(x) for x in oldrows[col]}
    if (out/'smoke_prompt_hashes.csv').exists():
        smoke=pd.read_csv(out/'smoke_prompt_hashes.csv')
        assert not (set(fresh.clean_sha256)|set(fresh.corrupted_sha256))&(set(smoke.clean_sha256)|set(smoke.corrupted_sha256))
    for cfg in (out/'attempts').rglob('configuration.json'):
        ctx=read(cfg)
        if ctx.get('role')=='fresh_confirmation':
            assert ctx['implementation_commit']==p['implementation_commit'] and ctx['protocol_sha256']==sha(out/'stage2_protocol.json')
        elif ctx.get('role')=='adaptive_old_Gate5G_only':
            assert ctx['implementation_commit']==p['implementation_commit'] and ctx['search_protocol_sha256']==sha(out/'search_protocol.json')
    for attempt_ in (out/'attempts').rglob('weight_audit.json'):
        r=read(attempt_);assert r['passed'] and r['before']==r['after']==p['model']['weight_sha256']
    forward=examples=0
    for file in (out/'attempts').rglob('forward_events.csv'):
        if file.stat().st_size<=1:continue
        events=pd.read_csv(file);cost=pd.read_csv(file.parent/'compute_accounting.csv')
        assert not events.grad_enabled.any() and len(events)==cost.forward_batches.sum() and events.batch_size.sum()==cost.example_forwards.sum()
        assert cost.backward_calls.sum()==cost.optimizer_updates.sum()==0
        forward+=len(events);examples+=int(events.batch_size.sum())
    for rec in p['circuits']:
        r=read(out/'evaluations'/rec['id']/'result.json');assert r['identity']['implementation_commit']==p['implementation_commit']
        assert r['weight_sha256']==p['model']['weight_sha256']
    for file in (out/'cache').rglob('*.json'):
        meta=read(file);assert meta['sha256']==sha(file.with_suffix('.npz'))
    check_files(read(out/'protected_gate5g.json'))
    write_json(out/'final_audit.json',dict(passed=True,protocol_sha256=sha(out/'stage2_protocol.json'),
        forward_batches=forward,example_forwards=examples,backwards=0,optimizer_updates=0,original_decisions_unchanged=True,
        candidate_frozen_before_fresh=True,all_jobs_complete=True))
    print('Addendum audit passed')

def package(out,commit=False,dry_run=False,stage1_only=False):
    if stage1_only:
        assert read(out/'stage1_audit.json')['passed']
        check_files(read(out/'stage1/outputs.json'),out)
    else:
        p=stage2_protocol(out);completed(out,p)
        audit_record=read(out/'final_audit.json')
        assert audit_record['passed'] and audit_record['protocol_sha256']==sha(out/'stage2_protocol.json')
    names=['stage1_audit.json','eap_sign_counts.csv','eap_prefix_scores.csv','source_references.json','official_weights.csv',
        'grid_contributions.csv','band_contributions.csv','band_bootstrap.csv','decomposition_verification.csv','edge_composition.csv',
        'margin_changes.csv','margin_by_template.csv','seed42_diagnostics.csv','seed42_template_failures.csv','seed42_new_components.json',
        'seed42_added_edges.csv','confirmation_hashes.csv','confirmation_audit.json','GATE5G_ADDENDUM_REPORT.md']
    names+=['edge_source_layer_composition.csv','edge_destination_layer_composition.csv','edge_receiver_composition.csv',
        'edge_inventory.csv.gz','edge_attributes.csv.gz','prompt_changes.csv.gz']
    if not stage1_only:names+=['stage2_protocol.json','stage2_protocol.sha256','candidate.json','candidate.sha256','stage2_circuit_results.csv',
        'nonadditivity_per_seed.csv','nonadditivity_pooled.csv','descriptive_reading.json','confirmation_results.json','confirmed_candidate_profile.csv','stage2_compute.csv','stage2_analysis.json','final_audit.json']
    paths=[out/name for name in names]+list((out/'figures').glob('*.png'))+[ROOT/'GATE5G_ADDENDUM_REPORT.md']
    for path in paths:
        assert path.exists(),path
        if path.stat().st_size>10_000_000:raise ValueError('Small package exceeds10MB per file: '+str(path))
    print('\n'.join(str(p.relative_to(ROOT)) for p in paths))
    if not (out/'GATE5G_ADDENDUM_REPORT.md').read_text(encoding='utf-8').split('## Interpretation and recommendation (researcher-written)')[-1].strip():
        print('WARNING: researcher interpretation is empty; review actual results.')
    if dry_run:return
    allowed=[p.relative_to(ROOT).as_posix() for p in paths]
    staged=set(git(ROOT,'diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))-{''}
    assert not staged-set(allowed),'Unrelated staged changes; refusing mixed results commit'
    import subprocess
    subprocess.run(['git','add','-f','--',*allowed],cwd=ROOT,check=True)
    print(git(ROOT,'diff','--cached','--stat').decode())
    if commit:subprocess.run(['git','commit','-m','research: Gate 5G addendum targeted intervention results'],cwd=ROOT,check=True)
    print('No automatic push. Run: git push origin research/mattr-gate5g-addendum')
