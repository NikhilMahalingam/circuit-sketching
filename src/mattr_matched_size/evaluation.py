"""Official MIB integrals and patching, with immutable exact-output reuse."""
from contextlib import contextmanager
from functools import partial
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import sha, write_json
from src.mattr_compact_validation.inference import Recorder, assign_scores
from .topology import digest


class ExactCache:
    def __init__(self, out, identity):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.identity = identity
        self.hits = self.misses = 0

    def get(self, mask_hash, compute):
        identity = {**self.identity, 'mask_sha256': mask_hash}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        path = self.out / (key + '.npz')
        meta = path.with_suffix('.json')
        if path.exists():
            assert json.loads(meta.read_text()) == identity
            self.hits += 1
            with np.load(path) as saved:
                return saved['margin'].copy(), saved['top1'].copy()
        margin, top1 = compute()
        margin = np.asarray(margin, dtype=np.float32)
        top1 = np.asarray(top1, dtype=np.float32)
        assert margin.shape == top1.shape and np.isfinite(margin).all() and np.isfinite(top1).all()
        np.savez_compressed(path, margin=margin, top1=top1)
        write_json(meta, identity)
        self.misses += 1
        return margin.copy(), top1.copy()


class OfficialEngine:
    def __init__(self, model, graph, dataset, out, protocol_hash, cohort_hash, batch=20):
        import eap.evaluate as eap
        from MIB_circuit_track.metrics import get_metric
        self.model, self.graph, self.ds, self.out = model, graph, dataset, Path(out)
        self.batch = batch
        self.loader = dataset.to_dataloader(batch)
        self.metric = partial(get_metric('logit_diff','ioi',model.tokenizer,model),mean=False,loss=False)
        self.raw_evaluate = eap.evaluate_graph
        self.cache = ExactCache(self.out / 'exact_outputs', {'protocol_sha256': protocol_hash,
                    'cohort_sha256': cohort_hash, 'batch': batch, 'model': 'pinned GPT2float32',
                    'intervention': 'pinned official patching'})
        self.refs = {}
        self.reference_hits = 0
        self.clean_reference = None
        self.latest = {}

    def current(self):
        before = self.graph.in_graph[self.graph.real_edge_mask.bool()].numpy().copy()
        self.graph.prune()
        after = self.graph.in_graph[self.graph.real_edge_mask.bool()].numpy().copy()
        assert np.array_equal(before, after), 'Query mask not already official-pruned.'
        key = digest(after)
        def compute():
            rec = Recorder(self.model)
            with torch.no_grad():
                value = self.raw_evaluate(self.model, self.graph, self.loader, rec, quiet=True, intervention='patching')
            margin, top1 = rec.arrays(len(self.ds))
            assert margin.shape == (1,len(self.ds))
            np.testing.assert_array_equal(value.numpy(), margin[0])
            assert digest(self.graph.in_graph[self.graph.real_edge_mask.bool()].numpy()) == key
            return margin[0], top1[0]
        margin, top1 = self.cache.get(key, compute)
        self.latest[key] = (margin, top1)
        return key, margin, top1

    @contextmanager
    def installed(self):
        import MIB_circuit_track.evaluation as official
        baseline = official.evaluate_baseline
        def graph_call(model, graph, loader, metric, **kw):
            assert model is self.model and graph is self.graph and loader is self.loader and metric is self.metric
            assert kw.get('intervention','patching') == 'patching'
            _, margin, _ = self.current()
            return torch.from_numpy(margin.copy())
        def baseline_call(model, loader, metric, **kw):
            assert model is self.model and loader is self.loader and metric is self.metric
            rec = Recorder(self.model)
            with torch.no_grad():
                result = baseline(model,loader,rec,quiet=True)
            margin, top1 = rec.arrays(len(self.ds))
            np.testing.assert_array_equal(result.numpy(),margin[0])
            self.clean_reference = margin[0].copy()
            np.savez_compressed(self.out / 'clean_reference.npz',margin=margin[0],top1=top1[0])
            return result
        with patch.object(official,'evaluate_graph',graph_call),patch.object(official,'evaluate_baseline',baseline_call):
            yield

    def area(self, scores, grid):
        from MIB_circuit_track.evaluation import evaluate_area_under_curve
        assign_scores(self.graph, scores)
        if self.refs:
            self.reference_hits += 2
        with self.installed(),torch.no_grad():
            result = evaluate_area_under_curve(self.model,self.graph,self.loader,self.metric,
                      quiet=True,level='edge',absolute=False,intervention='patching',refs=self.refs)
        assert abs(self.refs['baseline'] - self.refs['corrupted']) > 1e-8
        rows, per = [], []
        for j,fraction in enumerate(grid):
            self.graph.apply_topn(int(fraction*len(self.graph.edges)),absolute=False,level='edge',prune=True)
            mask = self.graph.in_graph[self.graph.real_edge_mask.bool()].numpy().copy()
            key = digest(mask)
            margin, top1 = self.latest[key]
            assert np.isclose(float((margin>0).mean()),result[5][j],atol=1e-7)
            actual = int(mask.sum())
            assert np.isclose(actual,result[0][j])
            rows.append({'fraction':fraction,'nominal_n':int(fraction*len(self.graph.edges)), 'actual_edges':actual,
                         'mask_sha256':key,'accuracy':result[5][j], 'faithfulness':result[4][j],
                         'mean_margin':float(margin.mean()),'top1':float(top1.mean())})
            for i in range(len(margin)):
                per.append({'fraction':fraction,'example':i,'margin':float(margin[i]),
                            'correct':int(margin[i]>0),'top1':int(top1[i])})
        if self.clean_reference is not None:
            np.testing.assert_allclose(self.latest[rows[-1]['mask_sha256']][0],self.clean_reference,atol=1e-4,rtol=2e-5)
        return result,pd.DataFrame(rows),pd.DataFrame(per)

    def selected(self, scores, n, expected_hash):
        assign_scores(self.graph,scores)
        self.graph.apply_topn(int(n),absolute=False,level='edge',prune=True)
        key, margin, top1 = self.current()
        assert key == expected_hash, 'Official circuit changed since topology freeze.'
        return {'mask_sha256':key,'accuracy':float((margin>0).mean()),'mean_margin':float(margin.mean()),
                'faithfulness':(float(torch.from_numpy(margin).mean())-self.refs['corrupted'])/(self.refs['baseline']-self.refs['corrupted']),
                'actual_edges':int(self.graph.in_graph[self.graph.real_edge_mask.bool()].sum()),'top1':float(top1.mean())},margin,top1
