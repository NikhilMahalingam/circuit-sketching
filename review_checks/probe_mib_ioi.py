"""Inspect the official local MIB APIs and data; no synthetic IOI fallback."""
import os
from pathlib import Path
import sys
import json

ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache/huggingface')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
REPO = ROOT / 'external/MIB-circuit-track'
sys.path[:0] = [str(REPO), str(REPO / 'EAP-IG/src')]

import torch
from transformer_lens import HookedTransformer
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.utils import MODEL_NAME_TO_FULLNAME, TASKS_TO_HF_NAMES
from MIB_circuit_track.metrics import get_metric
from eap.graph import Graph
from eap.attribute_node import attribute_node
from eap.utils import tokenize_plus
from functools import partial

print('CUDA', torch.__version__, torch.cuda.is_available(), flush=True)
model = HookedTransformer.from_pretrained(MODEL_NAME_TO_FULLNAME['gpt2'], device='cuda')
model.eval()
model.set_use_attn_result(True)
model.set_use_split_qkv_input(True)
model.set_use_hook_mlp_in(True)
report = {}
for split in ('train', 'validation', 'test'):
    ds = HFEAPDataset('mib-bench/' + TASKS_TO_HF_NAMES['ioi'], model.tokenizer,
                      split=split, task='ioi', model_name='gpt2')
    report[split] = {'n_filtered': len(ds), 'columns': ds.dataset.column_names,
                     'first_raw': ds.dataset[0], 'first_loaded': ds[0],
                     'fingerprint': ds.dataset._fingerprint,
                     'features': str(ds.dataset.features)}
    print(split, len(ds), ds.dataset.column_names, ds[0], flush=True)
    if split == 'train':
        train = ds
(ROOT / '.cache/mib_ioi_probe.json').write_text(json.dumps(report, indent=2))
train.head(8)
metric = get_metric('logit_diff', 'ioi', model.tokenizer, model)
for clean, corrupt, labels in train.to_dataloader(batch_size=4):
    tokens, mask, lengths, _ = tokenize_plus(model, clean)
    with torch.no_grad():
        logits = model(tokens, attention_mask=mask)
    print('clean score', metric(logits, None, lengths, labels, mean=False), flush=True)
graph = Graph.from_model(model, node_scores=True)
attribute_node(model, graph, train.to_dataloader(4), partial(metric, mean=True, loss=True),
               'EAP', 'patching', quiet=True)
print('Top heads:', sorted([(n.name, float(n.score)) for n in graph.nodes.values()
                           if hasattr(n, 'head')], key=lambda x: abs(x[1]), reverse=True)[:12])
print('PROBE PASSED', flush=True)
