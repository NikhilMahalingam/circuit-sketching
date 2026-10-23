"""Checkpoint provenance, official-data exclusions and pre-evaluation freeze."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from src.mattr_optimization.common import ROOT, GRID, configure, provenance, sha, write_json
from src.mattr_optimization.gate3 import baseline
from src.mattr_optimization.gate4_init import edge_mapping
from src.mattr_hierarchy.adapter import Hierarchy

PRIOR = ROOT / 'results/mattr_hierarchy/20261022_gate5c'
SEEDS = (42, 43, 44)
METHODS = {'baseline': 'bypass_g0', 'hierarchy': 'H2_g0.01',
           'gamma_zero': 'H2_g0', 'shuffled': 'H2_g0.01', 'edge_l2': 'edge_g0.01'}
REPORTS = ['GATE2_FINAL_ANALYSIS.md', 'GATE5B_INTERACTION_DIAGNOSTICS_REPORT.md',
           'GATE5C_HIERARCHICAL_MATTR_REPORT.md']


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def group_quotas(counts, target):
    """Largest-remainder proportional allocation, ties by exact template hash."""
    keys = sorted(counts)
    if target > sum(counts.values()) or target < 1:
        raise ValueError('Insufficient eligible official prompts.')
    exact = {k: target * counts[k] / sum(counts.values()) for k in keys}
    result = {k: int(np.floor(exact[k])) for k in keys}
    for k in sorted(keys, key=lambda k: (-(exact[k] - result[k]), k))[:target - sum(result.values())]:
        result[k] += 1
    assert sum(result.values()) == target and all(result[k] <= counts[k] for k in keys)
    return result


def select_rows(frame, excluded_hashes, excluded_indices, target=512, seed=5542):
    eligible = frame.loc[~frame.clean_sha256.isin(excluded_hashes) &
                         ~frame.official_filtered_index.isin(excluded_indices)].copy()
    # Official validation can contain duplicated strings at distinct IDs.
    eligible = eligible.sort_values('official_filtered_index').drop_duplicates('clean_sha256')
    size = target if len(eligible) >= target else len(eligible)
    if size < 256:
        raise ValueError(f'Only {size} disjoint unique prompts; need at least256. No synthetic fallback.')
    rng = np.random.default_rng(seed)
    counts = eligible.groupby('template_id').size().to_dict()
    quotas = group_quotas(counts, size)
    selected = []
    for family in sorted(quotas):
        group = eligible[eligible.template_id == family].sort_values('official_filtered_index')
        selected.extend(group.iloc[rng.permutation(len(group))[:quotas[family]]].index.tolist())
    chosen = eligible.loc[selected].sort_values(['template_id', 'official_filtered_index']).reset_index(drop=True)
    assert len(chosen) == size and chosen.clean_sha256.is_unique
    return chosen, eligible, quotas


def exclusions():
    hashes, indices, records = set(), set(), []
    roots = [ROOT / ('results/' + x) for x in
             ('mattr_optimization', 'mattr_expander', 'mattr_interactions', 'mattr_hierarchy')]
    for root in roots:
        for path in sorted(root.rglob('*manifest.csv')):
            frame = pd.read_csv(path, keep_default_na=False)
            if 'clean_prompt' not in frame:
                continue
            is_train = 'train' in path.name
            # Historical full Gate1 validation is disclosed, rather than excluding the whole split.
            if not is_train and len(frame) >= 10000:
                role = 'historical_full_validation_exposure_only'
            else:
                role = 'excluded_training' if is_train else 'excluded_small_evaluation'
                for col in ('clean_prompt', 'corrupted_prompt'):
                    if col in frame:
                        hashes.update(text_hash(x) for x in frame[col] if isinstance(x, str))
                if not is_train and 'official_filtered_index' in frame:
                    indices.update(frame.official_filtered_index.astype(int))
            records.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                            'rows': len(frame), 'role': role})
    if not records or not indices:
        raise ValueError('Missing historical cohort exclusions.')
    return hashes, indices, records


def ranking_records():
    records = []
    for method, prefix in METHODS.items():
        for seed in SEEDS:
            folder = PRIOR / f'jobs/{prefix}_s{seed}{"_shuffled" if method == "shuffled" else ""}'
            cfg = json.loads((folder / 'experiment_config.json').read_text())
            state_path = folder / 'checkpoints/step_001000.pt'
            state = torch.load(state_path, map_location='cpu', weights_only=False)
            scores_path = folder / 'effective_scores_1000.pt'
            scores = torch.load(scores_path, map_location='cpu', weights_only=False)
            assert state['completed'] == 1000 and len(state['train_log']) == 1000
            assert cfg['seed'] == seed and cfg['pool'] == 128 and cfg['phase'] == 'train'
            assert scores.shape == (32491,) and torch.isfinite(scores).all()
            old_eval = json.loads((folder / 'evaluation_step1000/official_result.json').read_text())
            complete = json.loads((folder / 'completed_1000.json').read_text())
            assert old_eval['identity']['checkpoint_sha256'] == sha(state_path)
            assert complete['effective_sha256'] == sha(scores_path)
            assert complete['config_sha256'] == sha(folder / 'experiment_config.json')
            group_path = PRIOR / f'H2{"_shuffled" if method == "shuffled" else ""}_groups.npy'
            ids = np.load(group_path) if cfg['family'] == 'H2' else None
            adapter = Hierarchy(ids, cfg['gamma'], cfg['scale'], 'hierarchy' if ids is not None else cfg['family'])
            with torch.no_grad():
                rebuilt = adapter.transform(state['scores'])
            torch.testing.assert_close(rebuilt, scores, atol=2e-6, rtol=1e-5)
            if method == 'baseline':
                base = torch.load(baseline(seed) / 'checkpoints/step_001000.pt', map_location='cpu', weights_only=False)
                assert torch.equal(scores, base['scores'])
            weight = json.loads((folder / 'model_weight_audit.json').read_text())
            assert weight['passed'] and weight['before_sha256'] == weight['after_sha256']
            rec = {'method': method, 'seed': seed, 'step': 1000, 'family': cfg['family'],
                   'gamma': cfg['gamma'], 'shuffled': cfg['shuffled'], 'edge_count': 32491,
                   'checkpoint': str(state_path.relative_to(ROOT)), 'checkpoint_sha256': sha(state_path),
                   'scores': str(scores_path.relative_to(ROOT)), 'scores_sha256': sha(scores_path),
                   'config': str((folder / 'experiment_config.json').relative_to(ROOT)),
                   'config_sha256': sha(folder / 'experiment_config.json'),
                   'training_manifest_sha256': sha(folder / 'train_manifest.csv'),
                   'development_manifest_sha256': sha(folder / 'validation_manifest.csv'),
                   'model_weights_sha256': weight['after_sha256'], 'source_provenance': cfg['source_provenance']}
            records.append(rec)
    return records


def validate_protocol(out):
    path = out / 'gate5d_protocol.json'
    assert sha(path) == (out / 'gate5d_protocol.sha256').read_text().strip()
    p = json.loads(path.read_text())
    for name, expected in p['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    for rec in p['rankings']:
        for field in ('checkpoint', 'scores', 'config'):
            assert sha(ROOT / rec[field]) == rec[field + '_sha256']
    assert sha(out / 'cohort_manifest.csv') == p['cohort']['manifest_sha256']
    return p


def prepare(out):
    configure()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'gate5d_protocol.json').exists():
        return validate_protocol(out)
    start = time.perf_counter()
    from transformers import AutoTokenizer
    from MIB_circuit_track.dataset import HFEAPDataset
    from eap.graph import Graph
    reports = {x: {'sha256': sha(ROOT / x), 'characters': len((ROOT / x).read_text())} for x in REPORTS}
    write_json(out / 'prior_reports_read.json', reports)
    documents = ['results/mattr_optimization/20261021T172035Z_gate2/campaign_protocol.json',
                 'results/mattr_optimization/20261021_gate2_seeds43_44/campaign_protocol.json',
                 'results/mattr_interactions/20261022_gate5b/gate5b_protocol.json',
                 'results/mattr_hierarchy/20261022_gate5c/gate5c_protocol.json',
                 'results/mattr_hierarchy/20261022_gate5c/final_validation.json']
    write_json(out / 'historical_protocols_read.json', {name: {'sha256': sha(ROOT / name),
               'top_level_fields': list(json.loads((ROOT / name).read_text()))} for name in documents})
    rankings = ranking_records()
    prov = provenance()
    assert all(r['source_provenance'] == prov for r in rankings)
    graph = Graph.from_model(dict(n_layers=12, n_heads=12, parallel_attn_mlp=False, d_model=768))
    mapping = edge_mapping(graph, graph)
    pd.testing.assert_frame_equal(mapping, pd.read_csv(PRIOR / 'edge_mapping.csv', keep_default_na=False))
    assert len(mapping) == 32491
    mapping.to_csv(out / 'edge_mapping.csv', index=False)
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    ds = HFEAPDataset('mib-bench/ioi', tokenizer, split='validation', task='ioi', model_name='gpt2')
    assert len(ds) == 10000
    # This is an explicit official field, not a heuristic inferred from prompt text.
    has_template = 'template' in ds.dataset.column_names
    rows = []
    for index in range(len(ds)):
        clean, donor, labels = ds[index]
        raw = ds.dataset[index]
        if any(len(tokenizer(' ' + raw['metadata'][key], add_special_tokens=False).input_ids) != 1
               for key in ('indirect_object', 'subject', 'random_c')):
            raise ValueError('Official IOI answer is not a single token; stop, do not change labels.')
        template = raw['template'] if has_template else None
        rows.append({'official_filtered_index': index, 'clean_sha256': text_hash(clean),
                     'corrupted_sha256': text_hash(donor), 'clean_prompt': clean, 'corrupted_prompt': donor,
                     'correct_token': labels[0], 'incorrect_token': labels[1],
                     'template_id': text_hash(template) if template is not None else 'unlabelled',
                     'official_template': template})
    hashes, indices, records = exclusions()
    selected, eligible, quotas = select_rows(pd.DataFrame(rows), hashes, indices)
    selected.insert(0, 'example', range(len(selected)))
    selected.to_csv(out / 'cohort_manifest.csv', index=False)
    public = selected.drop(columns=['clean_prompt', 'corrupted_prompt', 'official_template'])
    public.to_csv(out / 'cohort_ids_and_hashes.csv', index=False)
    audit = {'official_rows': len(ds), 'eligible_unique_rows': len(eligible), 'selected': len(selected),
             'selection_seed': 5542, 'excluded_unique_prompt_hashes': len(hashes),
             'excluded_validation_indices': len(indices), 'overlap_hashes': 0, 'overlap_indices': 0,
             'duplicates': 0, 'official_fingerprint': ds.dataset._fingerprint,
             'id_fields': [x for x in ds.dataset.column_names if 'id' in x.lower()],
             'ID_convention': 'official filtered validation row index; dataset exposes no separate example ID unless listed',
             'template_field': 'template' if has_template else None,
             'template_rule': 'exact official template string; SHA256 as stable family ID, no text heuristics',
             'eligible_template_counts': eligible.groupby('template_id').size().to_dict(),
             'selected_template_counts': quotas, 'excluded_manifests': records,
             'historical_full_Gate1_validation_exposure': True,
             'freshness': 'disjoint from smaller development/confirmation cohorts and discovery prompts; NOT untouched across all historical research'}
    write_json(out / 'cohort_audit.json', audit)
    protected = {}
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for name in tracked:
        # This file intentionally receives the new campaign's archival Git rules.
        if name and name != '.gitattributes' and (ROOT / name).is_file():
            protected[name] = sha(ROOT / name)
    # Include raw local historical checkpoints as well as tracked summaries.
    for root in (ROOT / 'results/mattr_optimization', ROOT / 'results/mattr_expander',
                 ROOT / 'results/mattr_interactions', ROOT / 'results/mattr_hierarchy'):
        for path in root.rglob('*'):
            if path.is_file():
                protected[str(path.relative_to(ROOT))] = sha(path)
    write_json(out / 'protected_previous_hashes.json', protected)
    source_paths = list((ROOT / 'src/mattr_compact_validation').glob('*.py'))
    source_paths += [ROOT / 'tests/test_mattr_compact_validation.py',
                     ROOT / 'src/mattr_compact_validation/run_wsl.sh',
                     ROOT / 'src/mattr_optimization/common.py', ROOT / 'src/mattr_optimization/gate0.py',
                     ROOT / 'src/mattr_optimization/gate1.py']
    p = {'gate': '5D', 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
         'model': 'gpt2-small', 'task': 'official mib-bench/ioi', 'precision': 'float32',
         'model_config': {'n_layers': 12, 'n_heads': 12, 'd_model': 768, 'use_split_qkv_input': True,
                          'use_attn_result': True, 'use_hook_mlp_in': True, 'ungroup_grouped_query_attention': True},
         'rankings': rankings, 'seeds': list(SEEDS), 'source_provenance': prov,
         'mapping_sha256': sha(out / 'edge_mapping.csv'), 'canonical_Gate5C_mapping_sha256': sha(PRIOR / 'edge_mapping.csv'),
         'cohort': {'size': len(selected), 'split': 'validation', 'manifest_sha256': sha(out / 'cohort_manifest.csv'),
                    'ids_hashes_sha256': sha(out / 'cohort_ids_and_hashes.csv'), 'audit_sha256': sha(out / 'cohort_audit.json'),
                    'fingerprint': ds.dataset._fingerprint, 'selection_seed': 5542,
                    'algorithm': 'exclude historical indices and exact UTF8 clean/donor hashes; deduplicate clean hashes by first official index; proportional largest-remainder exact-template quotas, hash tie breaks; sorted-template seeded permutation5542; final order template/index'},
         'grid': list(GRID), 'compact_fractions': [.005, .01, .02, .05], 'batch_size': 20,
         'evaluation': {'official': 'MIB_circuit_track.evaluation.evaluate_area_under_curve', 'absolute': False,
                        'intervention': 'patching', 'prune': True, 'level': 'edge',
                        'metrics': ['official logit_diff', 'official positive-margin accuracy', 'official CPR/CMD',
                                    'official accuracy log-AUC', 'official acc (vocabulary top1) diagnostic'],
                        'recorder': 'read-only metric wrapper captures per-example outputs from unchanged official forwards',
                        'cache': 'official graph-independent reference scalars reused across all methods; no activation caching, no mutable patched-activation reuse; frozen completed results may be reused after full identity verification'},
         'bootstrap': {'replicates': 5000, 'seed': 5642, 'unit': 'prompt, shared across all methods/seeds/grid points',
                       'stratification': 'resample within exact official template, fixed observed counts',
                       'seed_pooling': 'equal mean across3 fixed trained seeds for each prompt; never pool3N independent rows or resample seeds',
                       'primary_familywise': 'Bonferroni two endpoints: equal-tail97.5% marginal intervals (quantiles.0125,.9875) give simultaneous>=95%; per-seed intervals control2 endpoints within that seed only',
                       'control_familywise': 'Bonferroni6 pooled control-endpoint contrasts (3controls*2cutoffs); descriptive per-seed contrasts',
                       'secondary_auc': 'paired95% percentile interval, unadjusted and explicitly secondary'},
         'criteria': {'positive_mean_both_primary': True, 'positive_seeds_each_min': 2,
                      'familywise_pooled_lower_each_gt': 0., 'mean_auc_delta_min': -.02,
                      'mean_compact_2_5_delta_min': -.02, 'invalid_masking_allowed': False,
                      'structural': 'real hierarchy must additionally beat every3 control at both primary endpoints: pooled corrected lower>0 and positive in>=2 seeds each; gamma-zero equivalence prevents residual-penalty attribution',
                      'classification': 'primary pass=compact-accuracy generalization; primary+allcontrols=structural advantage; positive pooled effects but failed safeguards/uncertainty=mixed/inconclusive; otherwise negative'},
         'budget': {'optimization_steps': 0, 'model_backwards': 0, 'methods': 5, 'rankings': 15,
                    'evaluation_seconds_max': 14400, 'actual_model_forward_calls_max': 25000,
                    'OOM': 'stop with message; no silent batch/semantics changes; independent explicit resource amendment needed before rerun'},
         'stopping': 'stop on provenance/cohort/masking/correctness/nonfinite failure or resource cap; no retraining/new methods/further gates/leaderboard',
         'source_sha256': {str(x.relative_to(ROOT)): sha(x) for x in source_paths}}
    write_json(out / 'gate5d_protocol.json', p)
    (out / 'gate5d_protocol.sha256').write_text(sha(out / 'gate5d_protocol.json') + '\n')
    write_json(out / 'protocol_frozen_at.json', {'protocol_sha256': sha(out / 'gate5d_protocol.json'),
               'prior_new_cohort_model_forwards': 0, 'preparation_seconds': time.perf_counter() - start})
    print(json.dumps({'frozen': True, 'protocol_sha256': sha(out / 'gate5d_protocol.json'),
                      'cohort': len(selected), 'templates': len(quotas), 'rankings': len(rankings)}, indent=2), flush=True)
    return p
