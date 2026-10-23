"""Freeze topology-only selections and a strictly disjoint official cohort."""
import inspect
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import pandas as pd
from src.mattr_optimization.common import ROOT, GRID, configure, provenance, sha, write_json, integration_weights
from src.mattr_compact_validation.design import exclusions, group_quotas, text_hash
from .topology import PREVIOUS, N_EDGES, ensemble_scores


def protocol(out):
    path = out / 'gate5e_protocol.json'
    assert sha(path) == (out / 'gate5e_protocol.sha256').read_text().strip()
    p = json.loads(path.read_text())
    for name, value in p['sources'].items():
        assert sha(ROOT / name) == value, name
    for rec in p['catalog']:
        assert sha(ROOT / rec['scores']) == rec['scores_sha256']
    for name, value in p['frozen_artifacts'].items():
        assert sha(out / name) == value, name
    return p


def select_cohort(frame, previous_hashes, previous_indices, size=512, seed=5543):
    eligible = frame[~frame.clean_sha256.isin(previous_hashes) & ~frame.corrupted_sha256.isin(previous_hashes) &
                     ~frame.official_filtered_index.isin(previous_indices)].sort_values('official_filtered_index')
    # Greedy canonical-row deduplication covers clean/clean, donor/donor and cross-role collisions.
    seen, keep = set(), []
    for row in eligible.itertuples():
        both = {row.clean_sha256, row.corrupted_sha256}
        if len(both) != 2 or both & seen:
            continue
        seen.update(both)
        keep.append(row.Index)
    eligible = eligible.loc[keep]
    target = min(size, len(eligible))
    if target < 256:
        raise ValueError(f'Only{target} eligible official rows; minimum256; no generated fallback.')
    counts = eligible.groupby('template_id').size().to_dict()
    assert len(counts) == 9, 'Pinned official cohort should expose nine exact templates.'
    quotas = group_quotas(counts, target)
    rng = np.random.default_rng(seed)
    selected = []
    for family in sorted(quotas):
        group = eligible[eligible.template_id == family]
        selected.extend(group.iloc[rng.permutation(len(group))[:quotas[family]]].index)
    selected = eligible.loc[selected].sort_values(['template_id', 'official_filtered_index']).reset_index(drop=True)
    assert len(set(selected.clean_sha256) | set(selected.corrupted_sha256)) == 2 * len(selected)
    assert not (set(selected.clean_sha256) | set(selected.corrupted_sha256)) & previous_hashes
    assert not set(selected.official_filtered_index) & previous_indices
    return selected, eligible, quotas


def prepare(out):
    configure()
    if (out / 'gate5e_protocol.json').exists():
        return protocol(out)
    start = time.perf_counter()
    phase = json.loads((out / 'phaseA_complete.json').read_text())
    assert phase['passed'] and phase['monotonic'] and phase['nested']
    prev = json.loads((PREVIOUS / 'gate5d_protocol.json').read_text())
    assert sha(PREVIOUS / 'gate5d_protocol.json') == (PREVIOUS / 'gate5d_protocol.sha256').read_text().strip()
    prov = provenance()
    assert prov == prev['source_provenance']
    from transformers import AutoTokenizer
    from MIB_circuit_track.dataset import HFEAPDataset
    from eap.graph import Graph
    from MIB_circuit_track.evaluation import evaluate_area_under_curve
    ds = HFEAPDataset('mib-bench/ioi', AutoTokenizer.from_pretrained('gpt2'), split='validation', task='ioi', model_name='gpt2')
    prior_hashes, prior_indices, manifests = exclusions()
    # Gate5D is excluded as an entire cohort, both clean and corrupted text.
    prior_path = PREVIOUS / 'cohort_manifest.csv'
    prior = pd.read_csv(prior_path, keep_default_na=False)
    prior_hashes.update(text_hash(x) for col in ('clean_prompt', 'corrupted_prompt') for x in prior[col])
    prior_indices.update(prior.official_filtered_index.astype(int))
    manifests.append({'path': str(prior_path.relative_to(ROOT)), 'sha256': sha(prior_path),
                      'rows': len(prior), 'role': 'excluded_entire_Gate5D_cohort'})
    rows = []
    for i in range(len(ds)):
        clean, donor, labels = ds[i]
        raw = ds.dataset[i]
        ids = [ds.tokenizer(' ' + raw['metadata'][k], add_special_tokens=False).input_ids for k in ('indirect_object','subject','random_c')]
        assert all(len(x) == 1 for x in ids) and labels == [x[0] for x in ids[:2]]
        rows.append({'official_filtered_index': i, 'clean_prompt': clean, 'corrupted_prompt': donor,
                     'clean_sha256': text_hash(clean), 'corrupted_sha256': text_hash(donor),
                     'correct_token': labels[0], 'incorrect_token': labels[1],
                     'official_template': raw['template'], 'template_id': text_hash(raw['template'])})
    chosen, eligible, quotas = select_cohort(pd.DataFrame(rows), prior_hashes, prior_indices)
    chosen.insert(0, 'example', range(len(chosen)))
    chosen.to_csv(out / 'cohort_manifest.csv', index=False)
    chosen.drop(columns=['clean_prompt','corrupted_prompt','official_template']).to_csv(out / 'cohort_ids_and_hashes.csv', index=False)
    chosen[['template_id','official_template']].drop_duplicates().to_csv(out / 'official_template_labels.csv', index=False)
    audit = {'selected': len(chosen), 'eligible_unique_rows': len(eligible), 'selection_seed': 5543,
             'official_rows': len(ds), 'official_fingerprint': ds.dataset._fingerprint,
             'selected_template_counts': quotas, 'exclusion_manifests': manifests,
             'excluded_prompt_hashes': len(prior_hashes), 'excluded_validation_indices': len(prior_indices),
             'clean_overlap': 0, 'corrupted_overlap': 0, 'cross_role_overlap': 0,
             'clean_and_corrupted_unique': True, 'labels_single_tokens': True,
             'historical_Gate1_full_validation_exposure': True, 'private_test': False}
    write_json(out / 'cohort_audit.json', audit)
    catalog = json.loads((out / 'ranking_catalog.json').read_text())
    # A timed deterministic replay isolates construction from three-run training.
    import torch
    ensemble_construction=[]
    for method in ('baseline','hierarchy','gamma_zero'):
        originals=[r for r in catalog if r['kind']=='original' and r['method']==method]
        vectors=[torch.load(ROOT/r['scores'],map_location='cpu',weights_only=False).numpy() for r in originals]
        timer=time.perf_counter();rebuilt,_=ensemble_scores(vectors);seconds=time.perf_counter()-timer
        saved=next(r for r in catalog if r['kind']=='ensemble' and r['method']==method)
        assert torch.equal(rebuilt,torch.load(ROOT/saved['scores'],map_location='cpu',weights_only=False))
        ensemble_construction.append({'method':method,'CPU_construction_seconds_replay':seconds,
                                      'reproduced_export':True,'model_forwards':0,
                                      'scope':'timed deterministic replay; original construction included in PhaseA total; file loading excluded'})
    pd.DataFrame(ensemble_construction).to_csv(out/'ensemble_construction_cost.csv',index=False)
    cost = pd.read_csv(ROOT / 'results/mattr_hierarchy/20261022_gate5c/compute_accounting.csv')
    ensemble_cost = []
    for method, prefix in [('baseline','bypass_g0'), ('hierarchy','H2_g0.01'), ('gamma_zero','H2_g0')]:
        sub = cost[cost.job.isin([f'{prefix}_s{x}' for x in (42,43,44)])]
        assert len(sub) == 3
        ensemble_cost.append({'method': method, 'separate_training_runs': 3, 'optimizer_updates_total': 3000,
                              'existing_training_seconds': float(sub.training_seconds.sum()),
                              'existing_adaptation_seconds': float(sub.adaptation_seconds.sum()),
                              'scope': 'reused three1000-step runs; not newGate5E compute or a cheaper discovery claim'})
    pd.DataFrame(ensemble_cost).to_csv(out / 'ensemble_training_cost.csv', index=False)
    semantics = {}
    for name, fn in [('apply_topn',Graph.apply_topn), ('prune',Graph.prune), ('evaluate_area_under_curve',evaluate_area_under_curve)]:
        lines, first = inspect.getsourcelines(fn)
        file = Path(inspect.getsourcefile(fn))
        semantics[name] = {'file': str(file.relative_to(ROOT)), 'first_line': first,
                           'last_line': first + len(lines) - 1, 'sha256': sha(file)}
    semantics['grid_counting'] = 'nominal selection fraction of len(graph.edges) before pruning; integrals use percentages, not weighted_edge_counts'
    semantics['pruning'] = 'iteratively remove nodes lacking incoming or outgoing edges, then edges with absent parents/children, until both masks stabilize; input treated as having incoming edges, logits receiver forced present. Q/K/V map to their shared head node for node-presence checks; no new edges are added.'
    semantics['CPR_linear_area_weight_at_or_below1pct'] = sum(w for f,w in zip(GRID,integration_weights()) if f<=.01) / sum(integration_weights())
    write_json(out / 'official_pruning_semantics.json', semantics)
    reports = ['GATE5C_HIERARCHICAL_MATTR_REPORT.md','GATE5D_COMPACT_GENERALIZATION_REPORT.md']
    write_json(out / 'historical_reports_read.json', {x: {'sha256': sha(ROOT / x), 'characters': len((ROOT / x).read_text())} for x in reports})
    protected = {}
    names = subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    for name in names:
        if name and name != '.gitattributes' and (ROOT / name).is_file():
            protected[name] = sha(ROOT / name)
    for dirname in ('mattr_optimization','mattr_expander','mattr_interactions','mattr_hierarchy','mattr_compact_validation'):
        for path in (ROOT / 'results' / dirname).rglob('*'):
            if path.is_file():
                protected[str(path.relative_to(ROOT))] = sha(path)
    write_json(out / 'protected_previous_hashes.json', protected)
    frozen_names = ['phaseA_complete.json','ranking_catalog.json','edge_mapping.csv','circuit_queries.csv',
                    'exact_attainability_1_640.csv','exact_comparisons_all_pairs.csv','exact_primary_comparisons.csv',
                    'closure_correctness.csv','cohort_manifest.csv','cohort_ids_and_hashes.csv','cohort_audit.json',
                    'official_pruning_semantics.json','ensemble_training_cost.csv','ensemble_construction_cost.csv']
    source_paths = list((ROOT / 'src/mattr_matched_size').glob('*.py')) + [ROOT / 'src/mattr_matched_size/run_wsl.sh',
             ROOT / 'tests/test_mattr_matched_size.py', ROOT / 'src/mattr_compact_validation/inference.py',
             ROOT / 'src/mattr_compact_validation/design.py', ROOT / 'src/mattr_compact_validation/statistics.py',
             ROOT / 'src/mattr_optimization/common.py', ROOT / 'src/mattr_optimization/gate0.py', ROOT / 'src/mattr_optimization/gate1.py']
    total_requests = 10 * len(catalog) + len(pd.read_csv(out / 'circuit_queries.csv'))
    p = {'gate': '5E', 'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
         'source_provenance': prov, 'Gate5D_protocol_sha256': sha(PREVIOUS / 'gate5d_protocol.json'),
         'Gate5D_decision_sha256': sha(PREVIOUS / 'gate5d_decision.json'), 'rankings': prev['rankings'], 'catalog': catalog,
         'model': prev['model'], 'model_config': prev['model_config'], 'task': prev['task'], 'grid': list(GRID),
         'batch_size': 20, 'precision': 'float32', 'seeds': [42,43,44],
         'T1': phase['T1'], 'T2': phase['T2'], 'targets': phase['targets'],
         'algorithms': {'topology': 'official signed apply_topn, prune=True for every n1..1624; original score/tie convention unchanged',
                        'cap': 'largest audited nominal n with A(n)<=T, including n0; exact-size evaluations are the both-exact labeled subset, never extra forwards',
                        'interpolation': 'at T1/T2, linear accuracy interpolation between largest attainable<=T and smallest attainable>=T; flag gap>max(10,0.10*T); no claim of directly evaluated circuit',
                        'closure': 'first appearance in officialP(n), n1..1624; ties by original official rank; unseen edges appended in original order; unique negative ordinal scores; verify all known-group prefixes and50 full-range nesting samples; no tail pruning-free claim; full-grid NOMINAL effects only, no matched-size closure evaluations',
                        'ensemble': 'within each source vector, signed descending ordinal ranks tied by canonical edge index; equal average of3 ranks; final ties by canonical index; unique negative ordinal scores; no validation weights'},
         'cohort': {'seed': 5543, 'size': len(chosen), 'manifest_sha256': sha(out / 'cohort_manifest.csv'),
                    'rule': 'exclude prior official indices plus exact clean and donor hashes from all small prior campaigns includingGate5D and training; deduplicate both/cross-role hashes greedily by canonical row; proportional nine-template largest-remainder quotas then sorted-template seeded permutation; no private test'},
         'metrics': {'official': 'pinned MIB evaluate_area_under_curve; CPR/CMD/accuracy-logAUC unchanged; official eap.evaluate_graph for exact masks; official positive-margin accuracy, logit_diff, and separate vocabulary top1',
                     'normalization': 'fresh-cohort clean/empty means; official refs shared across allrankings; matched faithfulness uses same measured references; no homemade CPR/CMD',
                     'cache': 'immutable exact per-prompt output cache by mask+model+cohort+protocol; no activation caches, no mutable patched activations'},
         'bootstrap': {'resamples': 5000, 'seed': 5643, 'unit': 'shared prompts within exact template strata, fixedcounts; same resample all methods/seeds/targets; fixed trained seeds equally weighted, never3N independent rows',
                       'primary': 'single T1 endpoint, two-sided95% percentile interval; success lower>0',
                       'secondary': 'T2, controls, exact subsets and interpolation use descriptive unadjusted95% intervals',
                       'R': 'pooled cap-difference / pooled nominal0.5%-difference; shared-draw percentile ratioCI; unstable if denominator95%intervalcontains0 or abs(point)<1/cohortN; report invalid draw count'},
         'criteria': {'primary': 'T1 cap mean>0, >=2of3 positive seeds, paired prompt95%lower>0, all correctness valid',
                      'T2': 'secondary only; never changes T1decision',
                      'classification': ['Non-replication if nominal lower<=0','Ranking-quality effect if primary passes and stableR>=0.5',
                                         'Mostly connectivity if nominal lower>0, primary fails, and (capCIcontains0 or stableR<0.25)', 'Mixed otherwise'],
                      'interpretation': 'descriptive; R is advantage-retention ratio, never a causal proportion; capped success not exact-size superiority',
                      'ensemble_diagnostic': 'baselineensemble cap accuracy>=.90 at T2; compare3components without assuming3-run training is cheap'},
         'budget': {'planned_circuit_requests': total_requests, 'optimizer_updates': 0, 'model_backwards': 0,
                    'projection': 'first complete original full-grid wall seconds /10 * plannedrequests (conservatively includes one-time refs); if >7200seconds STOP before remaining evaluations',
                    'OOM': 'stop and report; do not silently change batch or protocol'},
         'stopping': 'stop matched experiment if original A monotonicity/nesting fails; invalid closure skipped, originals retained; checkpoint/cohort/evaluator failure stops inference; no newtraining/furthergates/leaderboard',
         'frozen_artifacts': {x: sha(out / x) for x in frozen_names},
         'topology_sha256': {str(x.relative_to(out)): sha(x) for x in (out / 'topology').glob('*') if x.is_file()},
         'sources': {str(x.relative_to(ROOT)): sha(x) for x in source_paths}}
    write_json(out / 'gate5e_protocol.json', p)
    (out / 'gate5e_protocol.sha256').write_text(sha(out / 'gate5e_protocol.json') + '\n')
    write_json(out / 'protocol_frozen_at.json', {'protocol_sha256': sha(out / 'gate5e_protocol.json'),
               'fresh_cohort_model_forwards': 0, 'preparation_seconds': time.perf_counter() - start})
    print(json.dumps({'frozen': True, 'SHA256': sha(out / 'gate5e_protocol.json'), 'T1': p['T1'], 'T2': p['T2'],
                      'cohort': len(chosen), 'planned_requests': total_requests},indent=2),flush=True)
    return p
