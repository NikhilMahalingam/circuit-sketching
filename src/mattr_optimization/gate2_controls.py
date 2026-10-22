"""Unchanged MAttr, conditional controls separating subset and training seeds.

Only dataset selection is decoupled from the upstream training RNG. No trainer,
mask, optimizer, donor, metric or step-budget changes are made.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from . import gate2
from .common import ROOT, sha, write_json

PAIRS = ((43, 42), (44, 42), (42, 43), (42, 44))  # training seed, subset seed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--training-seed', type=int, choices=(42,43,44), required=True)
    p.add_argument('--subset-seed', type=int, choices=(42,43,44), required=True)
    a = p.parse_args()
    if (a.training_seed, a.subset_seed) not in PAIRS:
        p.error('Only the four prespecified conditional controls are supported.')
    a.out.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (a.out/'.control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _run(a)


def _run(a):
    proto_args = SimpleNamespace(sizes=[128], seeds=[a.training_seed], validation_size=128,
                                 batch_size=20, gate1_evidence=None)
    proto = gate2.protocol_for(proto_args)
    proto.update(conditional_control=True, training_seed=a.training_seed,
                 subset_seed=a.subset_seed, control_adapter_sha256=sha(Path(__file__)))
    path = a.out/'campaign_protocol.json'
    if path.exists() and json.loads(path.read_text()) != proto:
        raise ValueError('Control protocol changed; refusing resume.')
    write_json(path, proto)
    marker = a.out/'completed.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if not saved['passed'] or any(sha(a.out/n) != h for n,h in saved['sha256'].items()):
            raise ValueError('Completed control artifacts changed.')
        print('Verified completed control: no model forwards.')
        return
    original = gate2.NestedData
    class ControlledData(original):
        def __init__(self, out, pool, seed, validation_size):
            super().__init__(out, pool, a.subset_seed, validation_size)
    with patch.object(gate2, 'NestedData', ControlledData):
        gate2.run_job(SimpleNamespace(out=a.out, pool=128, seed=a.training_seed, protocol=path))


if __name__ == '__main__':
    main()
