"""Historical provenance, topology-only splices and a genuinely disjoint cohort."""
import inspect
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from src.mattr_optimization.common import ROOT, GRID, configure, git, provenance, sha, write_json
from src.mattr_matched_size.topology import graph_new, official_order, digest as mask_digest
from src.mattr_compact_validation.inference import assign_scores
from .core import HISTORY, N, CAPS, SIZES, SEEDS, read, relative, digest, check_files, pinned_extra_sources


def load_scores(rec):
    path = ROOT / rec['scores']
    if sha(path) != rec['scores_sha256']:
        raise ValueError('Ranking export changed: ' + str(path))
    values = torch.load(path, map_location='cpu', weights_only=False)
    if values.shape != (N,) or not torch.isfinite(values).all():
        raise ValueError('Invalid canonical score vector')
    return values


def inventory(out):
    from src.mattr_data_scaling.core import verified, protocol as prior_protocol
    from src.mattr_data_scaling.evaluate import ranking
    from src.mattr_data_scaling.design import historical_candidates, hashlib_tensor
    from src.mattr_hierarchy.adapter import Hierarchy
    from src.mattr_compact_validation.design import ranking_records
    previous = prior_protocol(HISTORY)
    assert sha(HISTORY / 'gate5f_protocol.json') == (HISTORY / 'gate5f_protocol.sha256').read_text().strip()
    assert read(HISTORY / 'final_audit.json')['passed']
    # Gate5F already verified compatibility; independently check every endpoint/export and checkpoint again.
    catalog, exclusions = [], []
    groups = np.load(HISTORY / 'H2_groups.npy')

    def add(key, method, n, seed, step, scores, checkpoint, gate, inputs, core=False, gamma=0):
        values = torch.load(scores, map_location='cpu', weights_only=False)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False) if checkpoint else None
        reconstruction_error = 0.
        if state is not None:
            assert state['completed'] == step
            effective = state.get('effective_scores')
            if effective is None:
                effective = Hierarchy(groups).transform(state['scores']) if method.startswith('h2') else state['scores']
                # Original Gate5C exports were reduced on CUDA; CPU reconstruction has
                # float32 reduction-order roundoff. Preserve and hash the ORIGINAL export.
                # The same historical Gate5D adapter audit uses these explicit tolerances.
                if method.startswith('h2'):
                    torch.testing.assert_close(values,effective,atol=2e-6,rtol=1e-5)
                else:torch.testing.assert_close(values,effective,rtol=0,atol=0)
            else:torch.testing.assert_close(values,effective,rtol=0,atol=0)
            reconstruction_error=float((values-effective).abs().max())
        rec = dict(key=key, method=method, dataset_size=n, seed=seed, step=step, source_gate=gate,
            subset_seed=seed if n == 128 else 'fixed_full_pool', gamma=gamma, kind='core' if core else 'extended',
            scores=relative(scores), scores_sha256=sha(scores), effective_tensor_sha256=hashlib_tensor(values),
            checkpoint=relative(checkpoint) if checkpoint else None, checkpoint_sha256=sha(checkpoint) if checkpoint else None,
            compatibility_status='verified', input_hashes={relative(p): sha(p) for p in inputs + [scores] + ([checkpoint] if checkpoint else [])})
        rec['CPU_effective_reconstruction_max_error']=reconstruction_error
        load_scores(rec)
        catalog.append(rec)

    for e in previous['endpoints']:
        key, scores = ranking(HISTORY, e['method'], e['dataset_size'], e['seed'], e['step'])
        folder = scores.parent
        job = next(j for j in previous['training_jobs'] if j['key'] == f'{e["method"]}_n{e["dataset_size"]}_s{e["seed"]}')
        cp = folder / f'checkpoints/step_{e["step"]:06d}.pt'
        inputs = [HISTORY / 'gate5f_protocol.json', folder / 'completed.json']
        if cp.exists():
            assert sha(cp) == cp.with_suffix('.sha256').read_text().strip()
            inputs.append(cp.with_suffix('.sha256'))
        else:
            cp = ROOT / job['endpoints'][str(e['step'])]
            # Verify historical checkpoint against the original frozen inventory.
            inv = pd.read_csv(HISTORY / 'checkpoint_inventory.csv')
            match = inv[inv.checkpoint == relative(cp)].iloc[0]
            assert sha(cp) == match.checkpoint_sha256
        add(key, e['method'], e['dataset_size'], e['seed'], e['step'], scores, cp, '5F', inputs,
            core=e['dataset_size'] == 10000 and e['step'] == 5000)
    ef = HISTORY / 'attribution/eap_ig_n10000'
    assert verified(ef)
    attr = read(ef / 'attribution.json')
    assert attr['weights_unchanged'] and attr['convention'].endswith('no abs/sign flip')
    add('eap_ig_n10000', 'eap_ig', 10000, 42, 0, ef / 'scores.pt', None, '5F',
        [ef / 'completed.json', ef / 'attribution.json'], core=True)
    refs = ranking_records()  # Independent Gate5C checkpoint/adapter/config/optimizer/effective-score audit.
    for rec in refs:
        cp, sp = ROOT / rec['checkpoint'], ROOT / rec['scores']
        key = f'gate5c_{rec["method"]}_s{rec["seed"]}_step1000'
        method = 'h2' if rec['method'] in ('hierarchy', 'gamma_zero') else rec['method']
        # ranking_records has already re-derived all nonzero/shuffled controls using its correct adapter.
        if rec['method'] in ('baseline', 'gamma_zero', 'hierarchy'):
            if rec['method'] == 'hierarchy':method = 'h2_regularized'
            add(key, method, 128, rec['seed'], 1000, sp, cp, '5C', [ROOT / rec['config']], gamma=rec['gamma'])
        else:
            exclusions.append(dict(key=key, checkpoint=relative(cp), reason='Verified historical control outside ordinary/H2-gamma-zero comparison; no silent relabeling', verified=True))
    # Gate2/5C historical exports not represented by Gate5F are retained only with frozen evidence.
    inv = pd.read_csv(HISTORY / 'checkpoint_inventory.csv', keep_default_na=False)
    for method, n, seed, step, cp in historical_candidates():
        if not cp.exists():
            exclusions.append(dict(checkpoint=relative(cp), reason='checkpoint unavailable')); continue
        row = inv[inv.checkpoint == relative(cp)]
        if row.empty or not bool(row.iloc[0].exact_reuse_valid):
            exclusions.append(dict(checkpoint=relative(cp), reason='no compatible frozen provenance')); continue
        assert sha(cp) == row.iloc[0].checkpoint_sha256
        # Already represented by a Gate5F export with exact tensor equality; preserve source alias.
        match = next(r for r in catalog if (r['method'],r['dataset_size'],r['seed'],r['step']) == (method,n,seed,step))
        match.setdefault('source_aliases', []).append(dict(source_gate='2' if method == 'baseline' and n == 128 else '5C' if method == 'h2' else '1', checkpoint=relative(cp), sha256=sha(cp)))
        match['input_hashes'][relative(cp)] = sha(cp)
    assert sum(r['kind'] == 'core' for r in catalog) == 11
    write_json(out / 'ranking_inventory.json', catalog)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('input_hashes','source_aliases')} for r in catalog]).to_csv(out / 'ranking_inventory.csv', index=False)
    write_json(out / 'excluded_rankings.json', exclusions)
    return catalog


def positive_scores(order):
    order = np.asarray(order, dtype=np.int64)
    assert np.array_equal(np.sort(order), np.arange(len(order)))
    values = np.empty(len(order), dtype=np.float32)
    values[order] = np.arange(len(order), 0, -1, dtype=np.float32)
    return torch.from_numpy(values)


def splice_order(eap, parent, k):
    assert len(eap) == len(parent) == len(np.unique(parent)) == len(np.unique(eap))
    prefix = np.asarray(eap[:k])
    return np.concatenate([prefix, np.asarray(parent)[~np.isin(parent, prefix)]])


def selected(graph, scores, n, absolute=False):
    assign_scores(graph, scores)
    graph.apply_topn(int(n), absolute=absolute, level='edge', prune=True)
    return graph.in_graph[graph.real_edge_mask.bool()].numpy().copy()


def topology(out, catalog):
    from src.mattr_data_scaling.evaluate import selected_prefixes
    graph = graph_new()
    assert len(graph.edges) == N
    from src.mattr_optimization.gate4_init import edge_mapping
    mapping = edge_mapping(graph, graph)
    pd.testing.assert_frame_equal(mapping, pd.read_csv(HISTORY / 'edge_mapping.csv', keep_default_na=False))
    mapping.to_csv(out / 'edge_mapping.csv', index=False)
    folder = out / 'rankings'; folder.mkdir(exist_ok=True)
    eap = next(r for r in catalog if r['method'] == 'eap_ig')
    eap_values = load_scores(eap)
    order = official_order(graph, eap_values)
    export = positive_scores(order)
    assert np.array_equal(official_order(graph, export), order)
    np.save(folder / 'eap_order.npy', order)
    torch.save(export, folder / 'eap_positive.pt')
    # Source-line evidence reflects actual local pinned files, not an assumed API.
    import eap.graph as graph_module
    from src.mattr_data_scaling import attribution, evaluate
    convention = dict(absolute=False, sign_flip=False, score_transform='none in Gate5F',
        ordering='CPU torch.argsort of full flattened graph.scores descending; nonreal masked to -inf; ties use pinned torch argsort (not stable). Freeze resulting full order; derived scores are distinct positive float32 integers N..1.',
        sources={relative(Path(m.__file__)): dict(sha256=sha(m.__file__),
            references={name: inspect.getsourcelines(getattr(m, name))[1] for name in names})
            for m,names in [(attribution,['attribute']), (evaluate,['evaluate']), (graph_module,[])]},
        graph_apply_topn_line=inspect.getsourcelines(graph_module.Graph.apply_topn)[1],
        graph_prune_line=inspect.getsourcelines(graph_module.Graph.prune)[1],
        score_ties=int(N - len(torch.unique(eap_values))), signed_vs_absolute_same=bool(np.array_equal(order,official_order(graph,eap_values.abs()))))
    queries, audits, derived, topology_seconds = [], [], [], time.perf_counter()
    cores = [r for r in catalog if r['kind'] == 'core' and r['method'] != 'eap_ig']
    for parent in cores:
        parent_order = official_order(graph, load_scores(parent))
        for k in SIZES:
            key = f'{parent["key"]}_splice{k}'
            o = splice_order(order, parent_order, k); scores = positive_scores(o)
            assert np.array_equal(official_order(graph, scores), o)
            path = folder / (key + '.pt'); op = folder / (key + '_order.npy')
            torch.save(scores,path); np.save(op,o)
            derived.append(dict(key=key, parent=parent['key'], kind='splice', method=parent['method'],
                seed=parent['seed'], dataset_size=10000, step=5000, k=k, scores=relative(path), scores_sha256=sha(path),
                order_sha256=sha(op), input_hashes={relative(path):sha(path),relative(op):sha(op)}))
    mask_dir = out / 'masks'; mask_dir.mkdir(exist_ok=True)
    all_records = catalog + derived
    for i, rec in enumerate(all_records):
        scores = load_scores(rec)
        # A uses complete caps/grid; B only four caps; C caps/grid plus fixed boundary +/-1 diagnostics.
        caps = CAPS if rec['kind'] in ('core','splice') else [60,116,180,296]
        table, _, _ = selected_prefixes(graph, scores, caps)
        local = [dict(selection='cap', target=int(r.target), nominal_n=int(r.nominal_n), mask_sha256=r.mask_sha256,
                      actual_edges=int(r.actual_edges)) for r in table.itertuples()]
        if rec['kind'] in ('core','splice'):
            local += [dict(selection='nominal',target=float(f),nominal_n=int(f*N)) for f in GRID]
        if rec['kind'] == 'splice':
            local += [dict(selection='boundary',target=n,nominal_n=n) for n in (rec['k']-1,rec['k'],rec['k']+1)]
        previous = np.zeros(N,dtype=bool)
        for n in sorted(set(q['nominal_n'] for q in local)):
            mask = selected(graph,scores,n)
            assert not np.any(previous & ~mask)
            previous = mask
            h = mask_digest(mask)
            path = mask_dir / (h + '.npy')
            if path.exists():assert np.array_equal(np.load(path),mask)
            else:np.save(path,mask)
            for q in [q for q in local if q['nominal_n'] == n]:
                if 'mask_sha256' in q:assert q['mask_sha256'] == h
                q.update(ranking=rec['key'],actual_edges=int(mask.sum()),mask_sha256=h)
            if rec['kind'] == 'splice':
                assert np.array_equal(mask, selected(graph,scores,n,True))
                if n <= rec['k']:assert np.array_equal(mask,selected(graph,eap_values,n))
            if rec['method'] == 'eap_ig':
                assert np.array_equal(mask,selected(graph,export,n))
                assert np.array_equal(mask,selected(graph,export,n,True))
        if rec['kind'] == 'core':
            historical = HISTORY / 'evaluations/primary' / rec['key']
            for name,kind in [('nominal.csv','nominal'),('caps.csv','cap')]:
                old = pd.read_csv(historical / name)
                for q in [q for q in local if q['selection'] == kind]:
                    row = old[np.isclose(old['fraction' if kind == 'nominal' else 'target'],q['target'])].iloc[0]
                    assert q['mask_sha256'] == row.mask_sha256 and q['actual_edges'] == row.actual_edges
        queries.extend(local)
        print(f'Topology {i+1}/{len(all_records)}: {rec["key"]}; zero forwards',flush=True)
    q = pd.DataFrame(queries)
    ea = q[(q.ranking == eap['key']) & (q.selection == 'cap') & (q.target == 116)].iloc[0]
    for rec in derived:
        row = q[(q.ranking == rec['key']) & (q.selection == 'cap') & (q.target == 116)].iloc[0]
        audits.append(dict(ranking=rec['key'],k=rec['k'],eap_n324_actual=int(selected(graph,eap_values,324).sum()),
            eap_capped116_nominal_n=int(ea.nominal_n),splice_capped116_nominal_n=int(row.nominal_n),
            capped116_identical=row.mask_sha256 == ea.mask_sha256, prefix_suffices=ea.nominal_n <= rec['k']))
    q.to_csv(out / 'queries.csv',index=False)
    pd.DataFrame(audits).to_csv(out / 'splice_topology.csv',index=False)
    write_json(out / 'eap_ordering.json',convention)
    write_json(out / 'ranking_catalog.json',all_records)
    write_json(out / 'topology_audit.json',dict(passed=True,canonical_edges=N,core_rankings=11,splices=20,
        signed_and_absolute_derived_equal=True,historical_masks_equal=True,nesting=True,
        seconds=time.perf_counter()-topology_seconds,model_forwards=0))


def cohort(out):
    from transformers import AutoTokenizer
    from MIB_circuit_track.dataset import HFEAPDataset
    from src.mattr_compact_validation.design import exclusions
    from src.mattr_data_scaling.design import official_rows
    from src.mattr_matched_size.design import select_cohort
    hashes, indices, records = exclusions()
    extras = [ROOT/'results/mattr_compact_validation/20261023_gate5d/cohort_manifest.csv',
        ROOT/'results/mattr_matched_size/20261023_gate5e/cohort_manifest.csv',HISTORY/'cohort_primary.csv',
        HISTORY/'cohort_development.csv',HISTORY/'train_n10000.csv']
    for path in extras:
        frame = pd.read_csv(path,keep_default_na=False)
        hashes.update(frame.clean_sha256);hashes.update(frame.corrupted_sha256)
        if 'train_' not in path.name:indices.update(frame.official_filtered_index.astype(int))
        records.append(dict(path=relative(path),sha256=sha(path),role='excluded_train' if 'train_' in path.name else 'excluded_evaluation'))
    tok = AutoTokenizer.from_pretrained('gpt2')
    ds = HFEAPDataset('mib-bench/ioi',tok,split='validation',task='ioi',model_name='gpt2')
    all_rows = official_rows(ds)
    chosen, eligible, quotas = select_cohort(all_rows,hashes,indices,size=512,seed=5545)
    # Tokenization only: no model construction or forwards. Official patcher expects
    # aligned clean/donor tokenizer lengths. Actual evaluation positions are obtained
    # from the official evaluator, including its BOS/padding convention.
    lengths=[]
    for row in chosen.itertuples():
        a=tok(row.clean_prompt,add_special_tokens=True).input_ids
        b=tok(row.corrupted_prompt,add_special_tokens=True).input_ids
        assert len(a)==len(b),'Clean/corrupted token alignment failure'
        lengths.append(len(a))
    chosen['tokenizer_prompt_length']=lengths
    chosen.insert(0,'example',range(len(chosen)))
    chosen.to_csv(out/'cohort_manifest.csv',index=False)
    chosen.drop(columns=['clean_prompt','corrupted_prompt','official_template']).to_csv(out/'cohort_hashes.csv',index=False)
    write_json(out/'cohort_audit.json',dict(passed=True,eligible=len(eligible),selected=len(chosen),seed=5545,
        quotas=quotas,exclusions=records,token_labels_single=True,crossrole_overlap=0,
        official_fingerprint=ds.dataset._fingerprint,Gate1_full_validation_exposure=True,private_test_access=False))


def preflight_cpu(out):
    configure();torch.set_num_threads(4)
    if (out/'gate5g_protocol.json').exists():raise ValueError('Already frozen: preflight cannot rewrite design')
    cfg=read(ROOT/'configs/gate5g.json')
    assert cfg['caps']==CAPS and cfg['splice_sizes']==SIZES and cfg['nominal_grid']==list(GRID)
    assert cfg['batch_size']==20 and cfg['cohort_seed']==5545 and cfg['bootstrap_seed']==5645
    assert cfg['bootstrap_draws']==5000 and cfg['precision']=='float32'
    assert cfg['optimizer_updates']==cfg['backward_calls']==0
    start = time.perf_counter()
    evidence = {}
    for pattern in ['GATE4*REPORT.md','GATE5B*REPORT.md','GATE5C*REPORT.md','GATE5D*REPORT.md','GATE5E*REPORT.md','GATE5F*REPORT.md']:
        for path in ROOT.glob(pattern):
            path.read_text(encoding='utf-8');evidence[relative(path)] = sha(path)
    for path in ROOT.glob('results/**/gate5*_protocol.json'):
        if 'mattr_fidelity_splice' not in str(path):read(path);evidence[relative(path)]=sha(path)
    catalog = inventory(out);topology(out,catalog);cohort(out)
    # Preserve historical tracked artifacts plus all provenance inputs and exclusion manifests.
    protected = dict(evidence)
    for name in git(ROOT,'ls-files','-z','results').decode().split('\0'):
        if name and (ROOT/name).is_file():protected[name]=sha(ROOT/name)
    for rec in catalog:protected.update(rec['input_hashes'])
    for rec in read(out/'cohort_audit.json')['exclusions']:protected[rec['path']]=rec['sha256']
    write_json(out/'protected_history.json',protected)
    write_json(out/'historical_evidence.json',evidence)
    write_json(out/'preflight_cpu.json',dict(passed=True,seconds=time.perf_counter()-start,dependencies=provenance(),model_forwards=0))


def freeze(out):
    from .core import protocol
    if (out/'gate5g_protocol.json').exists():return protocol(out)
    assert read(out/'preflight_cpu.json')['passed'] and read(out/'preflight_gpu.json')['passed']
    tests=read(out/'unit_tests.json')
    assert tests['passed']
    check_files(tests['tested_sources'])
    check_files(read(out/'preflight_gpu.json')['tested_sources'])
    if git(ROOT,'status','--porcelain','--untracked-files=no').strip():
        raise ValueError('Freeze requires a clean tested implementation commit')
    catalog = read(out/'ranking_catalog.json')
    queries = pd.read_csv(out/'queries.csv')
    jobs = []
    for rec in catalog:
        part = 'A' if rec['kind']=='core' else 'C' if rec['kind']=='splice' else 'B'
        jobs.append(dict(id=rec['key'],ranking=rec['key'],part=part,query_count=int((queries.ranking==rec['key']).sum()),
            scores_sha256=rec['scores_sha256'],queries_sha256=digest(queries[queries.ranking==rec['key']].to_dict('records'))))
    sources = {}
    for directory in ['mattr_fidelity_splice','mattr_data_scaling','mattr_matched_size','mattr_compact_validation','mattr_hierarchy','mattr_optimization']:
        for path in (ROOT/'src'/directory).glob('*.py'):sources[relative(path)]=sha(path)
    for path in list((ROOT/'scripts/gate5g').glob('*.sh'))+[ROOT/'tests/test_mattr_fidelity_splice.py',ROOT/'configs/gate5g.json']:
        sources[relative(path)] = sha(path)
    sources.update(pinned_extra_sources())
    artifacts = {relative(p)[len(relative(out))+1:]:sha(p) for p in out.rglob('*') if p.is_file() and
        (p.parent==out and p.suffix in ('.json','.csv') or p.parent.name in ('rankings','masks')) and p.name not in ('status.json',)}
    gpu = read(out/'preflight_gpu.json')
    p = dict(gate='5G',implementation_commit=git(ROOT,'rev-parse','HEAD').decode().strip(),
        dependencies=provenance(),sources=sources,artifacts=artifacts,rankings=catalog,jobs=jobs,
        model=gpu['model'],batch_size=20,precision='float32',cohort_sha256=sha(out/'cohort_manifest.csv'),
        cohort=read(out/'cohort_audit.json'),eap_ordering=read(out/'eap_ordering.json'),
        grid=list(GRID),caps=CAPS,splice_sizes=SIZES,
        bootstrap=dict(draws=5000,seed=5645,strata='exact template; preserve counts; same prompt draws for all endpoints/methods/seeds'),
        part_a=dict(primary_nominal=[.1,.2,.5],secondary_cap=116,alpha_each=.05/3,seed_wins=4,
            criteria='all three: EAP-minus-equally-weighted baseline mean 98.333333% marginal CI upper<0 and EAP lower than >=4/5 seeds; reversed equivalent for MAttr closer; H2 descriptive',
            classifications=['EAP-IG closer','MAttr closer','Mixed','Invalid/incomplete']),
        part_c=dict(primary_method='baseline',primary_k=324,cpr_noninferiority=-.05,compact_min_gain=.10,
            criteria='paired95% CPR lower>-.05 AND cap116 accuracy mean>=.10 and lower>0, correctness passes; joint intersection rule; no global A/C95%guarantee',
            classifications=['Best of both','Compact gain with CPR cost','No useful compact gain','Invalid/incomplete']),
        features=['known_head_class_recall','complete_name_mover_path','nominal_prefix','pruned_fraction',
            'destination_count','destination_concentration','destination_entropy','layer_distribution',
            'normalized_score_gap_116_324','eap_mask_jaccard','full_signed_ranking_spearman'],
        metrics=dict(KL='full-vocabulary sum p_full*(logp_full-logp_circuit);float32 logsoftmax; no truncation',
            positions='official get_logit_positions(input_length-1); full vocabulary argmax',
            amplification='signed circuit/full ratio only abs(full)>=.5; excluded count; signed differences all prompts',
            task='official positive IOI margin distinct from full-vocabulary top1 correctness/agreement',
            CPR_CMD='unmodified MIB evaluator; bootstrap cohort means then official normalization/quadrature; never prompt CPR averages'),
        unstable_bootstrap='count |mean(clean)-mean(empty)|<1e-8 draws; undefined values not repaired; PartC confirmatory classification Invalid/incomplete if any draw is unstable',
        compute_projection=dict(seconds_per_query=gpu['seconds_per_query_projection'],
            total_seconds=sum(j['query_count'] for j in jobs)*gpu['seconds_per_query_projection'],
            rule='after first complete A ranking, check cumulative five-hour projection before every costly job; explicit --approve-over-5h'),
        artifacts_schema=dict(per_prompt='example,margin,top1_correct,top1_id,kl,margin_error,sign_agreement,top1_agreement',
            completed='frozen implementation commit + protocol/cohort/ranking/job identity; output byte hashes',cache='model/tokenizer/position/cohort/precision/code/protocol/mask identity+byte hash'),
        stopping='essential failure/OOM stops; no changed settings, training, new attribution, private split, leaderboard or further gates')
    check_files(read(out/'protected_history.json'))
    for rec in catalog:check_files(rec['input_hashes'])
    path=out/'gate5g_protocol.json'
    path.write_text(json.dumps(p,sort_keys=True,separators=(',',':')),encoding='utf-8')
    (out/'gate5g_protocol.sha256').write_text(sha(path)+'\n')
    write_json(out/'frozen_at.json',dict(protocol_sha256=sha(path),implementation_commit=p['implementation_commit'],
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),fresh_cohort_forwards=0))
    print('Protocol frozen:',sha(path),'implementation:',p['implementation_commit'],flush=True)
    return protocol(out,live=True)
