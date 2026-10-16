"""Local accounting and data selection; attribution/evaluation stay upstream."""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import functools
import hashlib
import importlib.metadata
import json
import os
import resource
import subprocess
import threading
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / 'external/matryoshka-attribution'
EXPECTED_COMMIT = 'af82b2eef145dd3e66c793a413352d6931a152d7'
PERCENTAGES = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def provenance():
    def revision(path):
        return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    mib = OFFICIAL / 'deps/MIB-circuit-track'
    actual = revision(OFFICIAL)
    if actual != EXPECTED_COMMIT:
        raise RuntimeError(f'Expected MAttr {EXPECTED_COMMIT}, found {actual}')
    if subprocess.check_output(['git', '-C', str(OFFICIAL), 'diff', '--name-only'], text=True).strip():
        raise RuntimeError('Official MAttr tracked sources changed; inspect before reproducing.')
    packages = ['torch', 'transformers', 'transformer-lens', 'datasets', 'numpy', 'pandas',
                'scipy', 'matplotlib', 'pytest', 'accelerate', 'psutil']
    files = [OFFICIAL/'scripts/mib/eval_mib.py', OFFICIAL/'src/matryoshka_attribution/trainer.py',
             OFFICIAL/'uv.lock', mib/'MIB_circuit_track/evaluation.py', mib/'MIB_circuit_track/dataset.py',
             mib/'EAP-IG/src/eap/attribute_node.py']
    return {'mattr_commit': actual, 'mib_commit': revision(mib),
            'eap_commit': revision(mib/'EAP-IG'),
            'versions': {p: importlib.metadata.version(p) for p in packages},
            'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in files},
            'cuda_runtime': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
            'cuda_total_memory_mib': torch.cuda.get_device_properties(0).total_memory / 2**20}


def add_official_paths():
    import sys
    sys.path.insert(0, str(OFFICIAL/'src'))
    from matryoshka_attribution.deps import add_mib_to_sys_path
    return add_mib_to_sys_path(OFFICIAL/'deps/MIB-circuit-track')


class Meter:
    """Actual root-model forwards/backwards; contiguous stages never double count."""
    def __init__(self, out):
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        self.rows = []
        if (self.out/'compute_accounting.csv').exists():
            self.rows = pd.read_csv(self.out/'compute_accounting.csv').to_dict('records')
        self.counts = dict(forward_batches=0, example_forwards=0, backward_calls=0, optimizer_steps=0)
        self.name = None; self.process = psutil.Process(); self.peak_rss = 0
        self.stop = threading.Event()
        self.worker = threading.Thread(target=self._sample_memory, daemon=True); self.worker.start()
        self.events = (self.out/'forward_events.jsonl').open('a', buffering=1)

    def _sample_memory(self):
        while not self.stop.wait(.05):
            self.peak_rss = max(self.peak_rss, self.process.memory_info().rss / 2**20)

    @staticmethod
    def sync():
        torch.cuda.synchronize()

    def switch(self, name, success=True):
        self.sync()
        if self.name is not None:
            row = {'stage': self.name, 'success': success,
                   'wall_seconds': time.perf_counter()-self.start,
                   **{k: self.counts[k]-self.before[k] for k in self.counts},
                   'peak_cuda_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
                   'peak_cuda_reserved_mib': torch.cuda.max_memory_reserved()/2**20,
                   'peak_cpu_rss_mib': self.peak_rss,
                   'lifetime_cpu_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
            self.rows.append(row)
            pd.DataFrame(self.rows).to_csv(self.out/'compute_accounting.csv', index=False)
        self.name = name
        if name is not None:
            self.before = self.counts.copy(); self.peak_rss = self.process.memory_info().rss/2**20
            torch.cuda.reset_peak_memory_stats(); self.start = time.perf_counter()

    def attach(self, model, backend):
        if getattr(model, '_mattr_meter_attached', False):
            return model
        original = model.forward
        @functools.wraps(original)
        def forward(*args, **kwargs):
            tokens = args[0] if args else kwargs.get('input_ids', kwargs.get('input'))
            batch = len(tokens) if not isinstance(tokens, str) else 1
            self.sync(); start = time.perf_counter(); success = False
            try:
                output = original(*args, **kwargs); success = True
                return output
            finally:
                self.sync()
                self.counts['forward_batches'] += 1; self.counts['example_forwards'] += batch
                self.events.write(json.dumps({'stage': self.name, 'backend': backend, 'batch_size': batch,
                    'success': success, 'grad_enabled': torch.is_grad_enabled(),
                    'seconds': time.perf_counter()-start})+'\n')
        model.forward = forward; model._mattr_meter_attached = True
        return model

    @contextmanager
    def backwards(self):
        original = torch.Tensor.backward
        meter = self
        def backward(tensor, *args, **kwargs):
            result = original(tensor, *args, **kwargs)
            meter.counts['backward_calls'] += 1
            return result
        with patch.object(torch.Tensor, 'backward', backward):
            yield

    def close(self, success=True):
        self.switch(None, success=success); self.stop.set(); self.worker.join(); self.events.close()


class FixedData:
    """Select official filtered rows, with a local RNG and immutable prompt manifests."""
    def __init__(self, out, train_size, seed, validation_size=128, validation_seed=42):
        self.out = Path(out); self.train_size = train_size; self.seed = seed
        self.validation_size = validation_size; self.validation_seed = validation_seed
        self.datasets = {}; self.manifests = {}; self.full_prompt_sets = {}

    @staticmethod
    def indices(length, size, seed):
        if size is None:
            return list(range(length))
        if not 1 <= size <= length:
            raise ValueError(f'Requested {size} examples from {length} official rows.')
        return np.random.default_rng(seed).permutation(length)[:size].tolist()

    def select(self, dataset, split):
        if split not in ('train', 'validation'):
            raise ValueError('Public test is not accessed by reproduction/scaling runs.')
        self.full_prompt_sets[split] = set(dataset.dataset['prompt'])
        size = self.train_size if split == 'train' else self.validation_size
        seed = self.seed if split == 'train' else self.validation_seed
        indices = self.indices(len(dataset), size, seed)
        official_fingerprint = dataset.dataset._fingerprint
        dataset.dataset = dataset.dataset.select(indices)
        dataset._mattr_split = split; dataset._mattr_official_indices = indices
        rows = []
        for i, source_index in enumerate(indices):
            clean, corrupt, labels = dataset[i]
            raw = dataset.dataset[i]
            names = [raw['metadata']['indirect_object'], raw['metadata']['subject']]
            ids = [dataset.tokenizer(' '+n, add_special_tokens=False).input_ids for n in names]
            if any(len(x)!=1 for x in ids) or [x[0] for x in ids] != list(labels):
                raise ValueError('Official IOI labels are not single tokens; inspect before changing metric.')
            rows.append({'split': split, 'subset_index': i, 'official_filtered_index': source_index,
                         'clean_prompt': clean, 'corrupted_prompt': corrupt,
                         'correct_token_id': labels[0], 'incorrect_token_id': labels[1],
                         'prompt_sha256': hashlib.sha256(clean.encode()).hexdigest()})
        manifest = pd.DataFrame(rows)
        if split in self.manifests:
            pd.testing.assert_frame_equal(manifest, self.manifests[split])
        if manifest.prompt_sha256.duplicated().any():
            raise ValueError(f'Duplicate prompts within {split}.')
        if set(self.full_prompt_sets)=={'train','validation'}:
            if self.full_prompt_sets['train'] & self.full_prompt_sets['validation']:
                raise ValueError('Official train and validation clean prompts overlap.')
        self.manifests[split] = manifest; self.datasets[split] = dataset
        manifest.to_csv(self.out/f'{split}_manifest.csv', index=False)
        write_json(self.out/f'{split}_selection.json', {'split': split, 'seed': seed,
            'requested_size': size, 'selected_size': len(dataset), 'indices': indices,
            'official_fingerprint': official_fingerprint, 'subset_fingerprint': dataset.dataset._fingerprint,
            'manifest_sha256': digest(self.out/f'{split}_manifest.csv')})
        return dataset

    @contextmanager
    def patch_loader(self):
        from MIB_circuit_track.dataset import HFEAPDataset
        original = HFEAPDataset.__init__
        selection = self
        @functools.wraps(original)
        def init(dataset, *args, **kwargs):
            original(dataset, *args, **kwargs)
            split = kwargs.get('split', args[2] if len(args)>2 else 'train')
            selection.select(dataset, split)
        with patch.object(HFEAPDataset, '__init__', init):
            yield


def export_ranking(graph, out, name='ranking'):
    rows = []
    for key, node in graph.nodes.items():
        if key == 'logits':
            continue
        index = graph.forward_index(node, attn_slice=False)
        rows.append({'component': key, 'forward_index': int(index), 'score': float(graph.nodes_scores[index])})
    frame = pd.DataFrame(rows).sort_values(['score','component'], ascending=[False,True])
    frame.insert(0, 'rank', range(1,len(frame)+1))
    frame.to_csv(Path(out)/(name+'.csv'), index=False)
    graph.to_pt(str(Path(out)/(name+'.pt')))
    graph.to_json(str(Path(out)/(name+'.json')))
    return frame
