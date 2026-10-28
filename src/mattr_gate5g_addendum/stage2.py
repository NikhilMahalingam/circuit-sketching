"""Manual-only targeted inference and old-cohort bounded adaptive search."""
import copy
from contextlib import contextmanager
from .common import *
from src.mattr_data_scaling.core import configure_runtime,load_model,close_meter
from src.mattr_optimization.gate1 import LiveAccounting
from src.mattr_hierarchy.runner import weight_hash
from src.mattr_fidelity_splice.inference import dataset,model_identity,position_identity,clean_reference,Engine,summarize

def prepare(out):
    if (out/'confirmation_cohort.csv').exists():
        a=read(out/'confirmation_audit.json');assert sha(out/'confirmation_cohort.csv')==a['sha256'];return
    from transformers import AutoTokenizer
    from MIB_circuit_track.dataset import HFEAPDataset
    from src.mattr_data_scaling.design import official_rows
    from src.mattr_matched_size.design import select_cohort
    old=original();seen=set();indices=set();exclusions=[]
    records=old['cohort']['exclusions']+[dict(path=(OLD/'cohort_manifest.csv').relative_to(ROOT).as_posix(),sha256=sha(OLD/'cohort_manifest.csv'),role='excluded_Gate5G')]
    for rec in records:
        path=ROOT/rec['path'];assert sha(path)==rec['sha256']
        if 'historical_full' in rec['role']:continue
        frame=pd.read_csv(path,keep_default_na=False)
        from src.mattr_compact_validation.design import text_hash
        for col in ['clean_prompt','corrupted_prompt']:
            if col in frame:seen.update(text_hash(x) for x in frame[col])
        if 'train' not in rec['role'] and 'train' not in path.name and 'official_filtered_index' in frame:indices.update(frame.official_filtered_index.astype(int))
        exclusions.append(rec)
    tok=AutoTokenizer.from_pretrained('gpt2');ds=HFEAPDataset('mib-bench/ioi',tok,split='validation',task='ioi',model_name='gpt2')
    rows=official_rows(ds);selected,eligible,quotas=select_cohort(rows,seen,indices,size=512,seed=5546)
    for r in selected.itertuples():assert len(tok(r.clean_prompt).input_ids)==len(tok(r.corrupted_prompt).input_ids)
    selected.insert(0,'example',range(len(selected)));selected.to_csv(out/'confirmation_cohort.csv',index=False)
    selected.drop(columns=['clean_prompt','corrupted_prompt','official_template']).to_csv(out/'confirmation_hashes.csv',index=False)
    write_json(out/'confirmation_audit.json',dict(passed=True,sha256=sha(out/'confirmation_cohort.csv'),eligible=len(eligible),selected=len(selected),
        seed=5546,quotas=quotas,exclusions=exclusions,pre_freeze_model_forwards=0,private_test=False))

def build_displacement(out,p):
    records=[]
    for seed in range(42,47):
        parent=core(p,'baseline',seed);splice=next(r for r in p['rankings'] if r.get('parent')==parent['key'] and r.get('k')==324)
        po=order(parent);so=order(splice)
        for f in FRACTIONS:
            n=int(f*32491);P=np.zeros(32491,dtype=bool);P[po[:n]]=True;S=np.zeros_like(P);S[so[:n]]=True
            A=S&~P;D=P&~S;assert A.sum()==D.sum()
            for kind,selected in [('parent',P),('remove',P&~D),('add',P|A),('splice',S)]:
                rec=save_set(out,f's{seed}_f{f}_{kind}',selected,stage='displacement',seed=seed,fraction=f,kind=kind,
                    added=int(A.sum()),displaced=int(D.sum()),selected_edges=np.flatnonzero(selected).tolist())
                if kind in ['parent','splice']:assert rec['mask_sha256']==query((parent if kind=='parent' else splice)['key'],'nominal',f).mask_sha256
                records.append(rec)
    return records

def search_plan(out):
    path=out/'search_protocol.json'
    if path.exists():return checked_json(path)
    assert not git(ROOT,'status','--porcelain','--',*sources()).strip(),'Commit tested implementation before starting adaptive search'
    prepare(out);p=original();parent=core(p,'baseline',42);po=order(parent)
    q1=query(parent['key'],'cap',116);q2=query(parent['key'],'cap',296)
    plan=dict(implementation_commit=git(ROOT,'rev-parse','HEAD').decode().strip(),sources=sources(),dependencies=provenance(),
        original_protocol_sha256=sha(OLD/'gate5g_protocol.json'),cohort_sha256=sha(OLD/'cohort_manifest.csv'),
        confirmation_sha256=sha(out/'confirmation_cohort.csv'),model=p['model'],batch=20,
        candidate_ranks=list(range(int(q1.nominal_n)+1,int(q2.nominal_n)+1)),candidate_edges=po[int(q1.nominal_n):int(q2.nominal_n)].tolist(),
        C116=save_set(out,'search_C116',oldmask(q1),stage='search',kind='base116'),
        C296=save_set(out,'search_C296',oldmask(q2),stage='search',kind='base296'),
        rules=dict(level1_blocks=8,next_blocks=4,levels=3,maximum_unique_masks=60,
            tie='maximum old-cohort accuracy gain; then fewer actual edges; then lowest original rank; then mask hash',
            rule='at level1 evaluate both C116+block and C296-block; refine only the winning addition block into4 at each next level; final candidate is best addition over ALL evaluated levels using the same tie rule; stop at singleton',
            precision='float32',k_sampling='none',optimizer_updates=0,backwards=0))
    freeze_json(path,plan);return plan

def locked_search(out):
    p=search_plan(out);check_files(p['sources'])
    assert git(ROOT,'rev-parse','HEAD').decode().strip()==p['implementation_commit']
    assert not git(ROOT,'status','--porcelain','--',*p['sources']).strip()
    assert provenance()==p['dependencies']
    assert sha(OLD/'cohort_manifest.csv')==p['cohort_sha256'] and sha(out/'confirmation_cohort.csv')==p['confirmation_sha256']
    return p

@contextmanager
def session(out,key,frame,model_info,context):
    """The only path to model construction; used by manual Stage2 commands."""
    configure_runtime()
    with attempt(out,key) as work:
        write_json(work/'configuration.json',context);meter=LiveAccounting(work)
        try:
            model=load_model(meter,freeze=True);before=weight_hash(model)
            assert model_identity(model)==model_info
            ds=dataset(model,frame);ctx=dict(**context,model=model_info,cohort_identity=digest(frame[['clean_sha256','corrupted_sha256']].to_dict('records')),
                positions=position_identity(model,ds,20),precision='float32',batch=20)
            with meter.stage('clean_reference'),torch.no_grad():
                ref=clean_reference(model,ds,frame,ArrayCache(out/'cache/clean',ctx),20)
            engine=Engine(model,graph_new(),ds,frame,ref,ArrayCache(out/'cache/circuits',ctx),20)
            from src.mattr_compact_validation.inference import assign_scores
            assign_scores(engine.graph,torch.arange(32491,dtype=torch.float32));engine.graph.apply_topn(0,absolute=False,prune=True)
            with meter.stage('empty_reference'),torch.no_grad():empty=engine.current()
            engine.empty=empty
            engine.refs=dict(baseline=float(torch.from_numpy(ref['margin']).mean()),corrupted=float(torch.from_numpy(empty['margin']).mean()))
            assert abs(engine.refs['baseline']-engine.refs['corrupted'])>EPS
            yield engine,meter,work
            assert before==weight_hash(model) and all(x.grad is None and not x.requires_grad for x in model.parameters())
            assert meter.counts['optimizer_updates']==meter.counts['backward_calls']==0
            write_json(work/'weight_audit.json',dict(passed=True,before=before,after=weight_hash(model),counts=meter.counts))
        finally:close_meter(meter)

def measure(out,engine,meter,rec):
    scores,mask=check_set(out,rec)
    from src.mattr_compact_validation.inference import assign_scores
    assign_scores(engine.graph,scores);engine.graph.apply_topn(rec['requested_edges'],absolute=False,prune=True,level='edge')
    assert mask_digest(engine.graph.in_graph[engine.graph.real_edge_mask.bool()].numpy())==rec['mask_sha256']
    with meter.stage('circuit_'+rec['id']),torch.no_grad():arrays=engine.current()
    summary_=summarize(arrays,engine.reference)
    summary_['faithfulness']=(float(torch.from_numpy(arrays['margin']).mean())-engine.refs['corrupted'])/(engine.refs['baseline']-engine.refs['corrupted'])
    return summary_,arrays

def best(candidates):
    return sorted(candidates,key=lambda r:(-r['gain'],r['actual_edges'],min(r['ranks']),r['mask_sha256']))[0]

def search(out,dry_run=False):
    if dry_run:
        original();print('Old Gate5G cohort ONLY;8 additions+8 removals then4 then4; at most60 unique masks;3levels. No search or confirmation executed.');return
    plan=locked_search(out)
    if (out/'candidate.json').exists():
        receipt=checked_json(out/'candidate.json');assert receipt['search_protocol_sha256']==sha(out/'search_protocol.json')
        for row in receipt['trace']:check_set(out,row['circuit'])
        assert verified(out/'search_complete',receipt['identity']);print('Verified completed search');return
    frame=pd.read_csv(OLD/'cohort_manifest.csv',keep_default_na=False)
    fresh=pd.read_csv(out/'confirmation_hashes.csv');assert not (set(frame.clean_sha256)|set(frame.corrupted_sha256))&(set(fresh.clean_sha256)|set(fresh.corrupted_sha256))
    context=dict(implementation_commit=plan['implementation_commit'],search_protocol_sha256=sha(out/'search_protocol.json'),role='adaptive_old_Gate5G_only')
    trace=[]
    with session(out,'adaptive_search',frame,plan['model'],context) as (engine,meter,work):
        base,ba=measure(out,engine,meter,plan['C116']);measure(out,engine,meter,plan['C296'])
        _,C116=check_set(out,plan['C116']);_,C296=check_set(out,plan['C296'])
        active=np.array(plan['candidate_edges']);rank_lookup=dict(zip(plan['candidate_edges'],plan['candidate_ranks']))
        unique={plan['C116']['mask_sha256'],plan['C296']['mask_sha256']}
        for level in range(1,4):
            contenders=[]
            for number,ids in enumerate(np.array_split(active,8 if level==1 else 4)):
                if not len(ids):continue
                for kind in (['addition','removal'] if level==1 else ['addition']):
                    selected=C116.copy() if kind=='addition' else C296.copy();selected[ids]=kind=='addition'
                    rec=save_set(out,f'search_L{level}_b{number}_{kind}',selected,stage='search',kind=kind)
                    unique.add(rec['mask_sha256']);assert len(unique)<=60
                    metric,_=measure(out,engine,meter,rec)
                    row=dict(level=level,block=number,kind=kind,edges=ids.tolist(),ranks=[rank_lookup[int(e)] for e in ids],
                        gain=metric['accuracy']-base['accuracy'],**metric,actual_edges=rec['actual_edges'],mask_sha256=rec['mask_sha256'],circuit=rec,
                        complete_name_mover_path=None,connectivity_note='official pruning; see selected/pruned masks and structural candidate metadata')
                    trace.append(row)
                    if kind=='addition':contenders.append(row)
                    write_json(out/'search_trace_partial.json',trace)
            winner=best(contenders);active=np.array(winner['edges'])
            if len(active)==1:break
        from src.mattr_fidelity_splice.structure import complete_path
        mapping=pd.read_csv(OLD/'edge_mapping.csv',keep_default_na=False)
        classes=read(ROOT/'results/mattr_data_scaling/gate5f/known_ioi_heads.json')['classes']
        names=[f'a{l}.h{h}' for l,h in classes['name mover']]
        for row in trace:
            _,mask=check_set(out,row['circuit']);row['complete_name_mover_path']=complete_path(mapping,mask,names)
            row['connectivity_changed']=mask_digest(mask)!=plan['C116']['mask_sha256']
            row['adds_complete_input_logits_path']=bool((mask&~C116).any())
            row['path_definition']='new retained edge in the officially pruned connected graph, relative to C116; topological only, not causal function'
        winner=best([row for row in trace if row['kind']=='addition'])
        candidate=dict(identity=context,search_protocol_sha256=sha(out/'search_protocol.json'),edges=winner['edges'],ranks=winner['ranks'],
            candidate_set_sha256=digest(sorted(winner['edges'])),circuit=winner['circuit'],trace=trace,source_cohort_sha256=sha(OLD/'cohort_manifest.csv'),
            confirmation_used=False,unique_circuit_masks=len(unique))
        assert weight_hash(engine.model)==plan['model']['weight_sha256']
        assert meter.counts['backward_calls']==meter.counts['optimizer_updates']==0
        freeze_json(out/'candidate.json',candidate)
        folder=out/'search_complete';folder.mkdir(exist_ok=True);marker(folder,context,[])
    print('Old-cohort search complete. Candidate fixed; run freeze before confirmation.',flush=True)

def freeze(out):
    path=out/'stage2_protocol.json'
    if path.exists():return stage2_protocol(out)
    original_p=original();prepare(out);search_p=locked_search(out);candidate=checked_json(out/'candidate.json')
    assert candidate['confirmation_used'] is False
    assert candidate['search_protocol_sha256']==sha(out/'search_protocol.json')
    assert verified(out/'search_complete',candidate['identity'])
    records=build_displacement(out,original_p)
    for name,old in [('base116',search_p['C116']),('candidate',candidate['circuit']),('base296',search_p['C296'])]:
        _,mask=check_set(out,old)
        # Candidate is already a custom augmented selection; preserve its unpruned set.
        with np.load(out/old['path']) as data:selection=data['selected'].copy()
        records.append(save_set(out,'confirm_'+name,selection,stage='confirmation',kind=name,seed=42,fraction=None))
    artifact_paths=[out/n for n in ['confirmation_cohort.csv','confirmation_hashes.csv','confirmation_audit.json','candidate.json','candidate.sha256','search_protocol.json','search_protocol.sha256']]
    if (out/'smoke_passed.json').exists():
        assert read(out/'smoke_passed.json')['passed']
        smoke=pd.read_csv(out/'smoke_prompt_hashes.csv');fresh=pd.read_csv(out/'confirmation_hashes.csv')
        assert not (set(smoke.clean_sha256)|set(smoke.corrupted_sha256))&(set(fresh.clean_sha256)|set(fresh.corrupted_sha256))
        artifact_paths += [out/'smoke_passed.json',out/'smoke_prompt_hashes.csv']
    p=dict(implementation_commit=git(ROOT,'rev-parse','HEAD').decode().strip(),sources=sources(),dependencies=provenance(),model=search_p['model'],
        original_protocol_sha256=sha(OLD/'gate5g_protocol.json'),candidate_sha256=sha(out/'candidate.json'),circuits=records,jobs=[r['id'] for r in records],
        artifacts={x.relative_to(out).as_posix():sha(x) for x in artifact_paths},cohort_sha256=sha(out/'confirmation_cohort.csv'),
        definitions='P, P-D, P|A, (P-D)|A unpruned selected sets; official prune per query. Remove/Add budgets vary; no full-grid CPR estimate.',
        bootstrap=dict(draws=5000,seed=5646,paired=True,strata='exact official template',fixed_seeds=True),
        reading_rules=dict(displacement=.75,insertion=.75,additive_band=.1,nonadditive_fraction=.25,near_zero=EPS,
            caveat='exploratory circuit-level non-additivity includes pruning/size/intervention effects; no direct edge-interaction inference'),
        runtime=dict(limit_seconds=7200,approval='--approve-over-2h',calibration='after first complete displacement seed; all attempts including search charged',
            historical_seconds_per_request=historical_rate(),initial_request_upper_bound=123,projected_upper_seconds=123*historical_rate(),
            projection_basis='largest observed original Gate5G baseline per-query process/evaluation wall;123requests=60displacement+60search+3confirmation before cache reuse'),
        stops='OOM/correctness/hash/weights/cohort failure stops; no settings changes; zero backwards and optimizer updates')
    freeze_json(path,p);stage2_protocol(out,live=True)
    print('Stage2 frozen:',sha(path),'Jobs:',len(records),flush=True)

def historical_rate():
    p=original_protocol(OLD);rates=[]
    for seed in range(42,47):
        rec=core(p,'baseline',seed);s=summary(rec['key'])
        times=[read(x)['seconds'] for x in (OLD/'attempts'/rec['key']).glob('*/attempt.json') if read(x)['status']=='complete']
        rates.append(max(times or [s['evaluation_seconds']])/s['requested_masks'])
    return max(rates)

def guard(out,p,approved):
    done=[]
    for rec in p['circuits']:
        folder=out/'evaluations'/rec['id']
        if verified(folder):done.append(rec)
    complete_seeds=[s for s in range(42,47) if sum(r.get('stage')=='displacement' and r.get('seed')==s for r in done)==12]
    wall=sum(read(x).get('seconds',0.) for x in (out/'attempts').rglob('attempt.json'))
    rates=[read(out/'evaluations'/r['id']/'result.json')['seconds'] for r in done]
    projection=wall+(len(p['circuits'])-len(done))*max(rates or [p['runtime']['historical_seconds_per_request']])
    write_json(out/'runtime_projection.json',dict(projected_seconds=projection,limit_seconds=7200,completed_seeds=complete_seeds,explicit_approval=approved))
    if approved:write_json(out/'runtime_approval.json',dict(protocol_sha256=sha(out/'stage2_protocol.json'),approved=True))
    if complete_seeds and projection>7200 and not approved:raise RuntimeError('Projected cumulative GPU-job wall exceeds2hours; explicitly pass --approve-over-2h after reviewing projection')

def evaluate(out,stage,seed=None,fraction=None,approved=False,dry_run=False):
    p=stage2_protocol(out,live=True)
    recs=[r for r in p['circuits'] if r['stage']==stage and (seed is None or r['seed']==seed) and (fraction is None or r['fraction']==fraction)]
    assert recs
    if dry_run:print(json.dumps(recs,indent=2));return
    frame=pd.read_csv(out/'confirmation_cohort.csv',keep_default_na=False)
    context=dict(protocol_sha256=sha(out/'stage2_protocol.json'),implementation_commit=p['implementation_commit'],role='fresh_confirmation',cohort_sha256=p['cohort_sha256'])
    pending=[]
    for rec in recs:
        identity=dict(**context,circuit=rec)
        if not verified(out/'evaluations'/rec['id'],identity):pending.append(rec)
    if not pending:print('Verified completed selected jobs');return
    # One model per seed/group command. Guard again after every circuit (including a completed seed).
    guard(out,p,approved)
    print(f'{len(pending)} circuit requests x {len(frame)} prompts, batch20;2 forwards per batch per uncached circuit plus references.',flush=True)
    with session(out,stage+'_s'+str(seed),frame,p['model'],context) as (engine,meter,work):
        for rec in pending:
            guard(out,p,approved);torch.cuda.synchronize();start=time.perf_counter()
            metrics,arrays=measure(out,engine,meter,rec);torch.cuda.synchronize()
            dest=out/'evaluations'/rec['id'];dest.mkdir(parents=True,exist_ok=True)
            table_=pd.DataFrame(arrays);table_['clean_margin']=engine.reference['margin'];table_['empty_margin']=engine.empty['margin'];table_['template_id']=frame.template_id
            table_.to_csv(dest/'per_prompt.csv',index=False)
            write_json(dest/'result.json',dict(identity=dict(**context,circuit=rec),metrics=metrics,circuit=rec,counts=meter.counts.copy(),
                seconds=time.perf_counter()-start,weight_sha256=weight_hash(engine.model),cache_hits=engine.cache.hits,cache_misses=engine.cache.misses))
            # The before/after weight check occurs before each completion marker.
            assert weight_hash(engine.model)==p['model']['weight_sha256']
            marker(dest,dict(**context,circuit=rec),[dest/'per_prompt.csv',dest/'result.json'])
