"""CPU-only post-freeze repair; original inference source and protocol stay intact.

The frozen structural extractor explicitly supplies actual_edges to dict(), while
the unchanged Gate5F overlap helper supplies that key again through **kwargs.
Remove only the redundant helper key in a scoped adapter after verifying counts.
"""
import argparse
from contextlib import contextmanager
from numbers import Number
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.mattr_optimization.common import ROOT, configure, git, sha, write_json
from . import structure
from .core import DEFAULT, N, digest, protocol, read


@contextmanager
def corrected_overlap_counts():
    """Call the real frozen extractor/helper, removing just their duplicate key."""
    original = structure.overlap

    def without_duplicate(mapping, mask, classes):
        values = original(mapping, mask, classes).copy()
        count = values.pop('actual_edges')
        if count != int(np.asarray(mask, dtype=bool).sum()):
            raise ValueError('Overlap edge count differs from the measured mask')
        return values

    with patch.object(structure, 'overlap', without_duplicate):
        yield


def validate_counts(out, p):
    """The retained explicit query count must equal the hash-verified mask count."""
    from src.mattr_matched_size.topology import digest as mask_digest
    queries = pd.read_csv(out / 'queries.csv')
    keys = {r['key'] for r in p['rankings'] if r['kind'] != 'splice'}
    selected = queries[queries.ranking.isin(keys) & (queries.selection == 'cap') & queries.target.isin([116, 296])]
    assert len(selected) == 2 * len(keys)
    for row in selected.itertuples():
        mask = np.load(out / 'masks' / (row.mask_sha256 + '.npy'), allow_pickle=False)
        if mask.shape != (N,) or mask.dtype != np.bool_:
            raise ValueError('Unexpected canonical mask dimensions/type')
        if mask_digest(mask) != row.mask_sha256 or int(mask.sum()) != row.actual_edges:
            raise ValueError('Frozen query/mask/count mismatch')
    return len(selected)


def assert_same_decision(a, b):
    """Labels are exact; numerical replay permits only float64 roundoff."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_same_decision(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_same_decision(x, y)
    elif isinstance(a, Number) and not isinstance(a, bool):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    else:
        assert a == b


def run(out, dry_run=False):
    from .analysis import analyze, completeness, REPORT, INTERPRETATION
    p = protocol(out)  # Original frozen source, inputs, protocol and ranking hashes.
    completeness(out, p)  # Completed jobs still identify the original inference commit.
    counts = validate_counts(out, p)
    if dry_run:
        print(f'Validated {len(p["jobs"])} completed jobs and {counts} structural queries. CPU-only repair ready.')
        return
    start = time.perf_counter()
    prior = read(out / 'gate5g_decision.json') if (out / 'gate5g_decision.json').exists() else None
    protocol_hash = sha(out / 'gate5g_protocol.json')
    completed = {j['id']: sha(out / 'evaluations' / j['id'] / 'completed.json') for j in p['jobs']}
    record = dict(status='running', frozen_implementation_commit=p['implementation_commit'],
        protocol_sha256=protocol_hash, repair_commit=git(ROOT, 'rev-parse', 'HEAD').decode().strip(),
        repair_source_sha256=sha(__file__), structural_queries_validated=counts,
        repair='Scoped removal of duplicate overlap.actual_edges; keep the validated frozen query count.',
        evaluation_marker_digest=digest(completed), model_forwards=0, backward_calls=0, optimizer_updates=0,
        registered_metrics_thresholds_masks_and_data_unchanged=True)
    write_json(out / 'analysis_repair.json', record)
    try:
        with corrected_overlap_counts():
            analyze(out)
        after = read(out / 'gate5g_decision.json')
        if prior is not None:
            assert_same_decision(prior, after)
        assert sha(out / 'gate5g_protocol.json') == protocol_hash
        assert completed == {j['id']: sha(out / 'evaluations' / j['id'] / 'completed.json') for j in p['jobs']}
        completeness(out, p)  # Re-verify output bytes as well as completion markers.
        record.update(status='passed', seconds=time.perf_counter()-start,
            existing_registered_decisions_preserved=prior is not None)
        write_json(out / 'analysis_repair.json', record)
        meta = read(out / 'analysis_metadata.json')
        meta['offline_analysis_repair'] = record
        write_json(out / 'analysis_metadata.json', meta)
        body = (out / REPORT).read_text(encoding='utf-8')
        note = ('## Post-freeze offline analysis repair\n\n'
            f'The duplicate actual_edges dictionary key was removed through a scoped adapter; all {counts} '
            'structural query counts were checked against frozen masks. No model forwards or evaluation changes occurred. '
            f'Original inference commit: `{p["implementation_commit"]}`. Repair commit: `{record["repair_commit"]}`. '
            'The original protocol/hash, thresholds, shared bootstrap and registered decisions are preserved. '
            'See analysis_metadata.json and GATE5G_ANALYSIS_REPAIR.md.\n\n')
        body = body.replace(INTERPRETATION, note + INTERPRETATION, 1)
        (out / REPORT).write_text(body, encoding='utf-8')
        (ROOT / REPORT).write_text(body.replace('(figures/', f'(results/mattr_fidelity_splice/{out.name}/figures/'), encoding='utf-8')
        print('Offline repair passed; original protocol and completed GPU evaluations preserved.', flush=True)
    except BaseException as exc:
        record.update(status='failed', seconds=time.perf_counter()-start, error=repr(exc))
        write_json(out / 'analysis_repair.json', record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT)
    parser.add_argument('--dry-run', action='store_true', help='Verify frozen/completed inputs and counts without rerunning analysis')
    args = parser.parse_args()
    configure()
    run(args.out.resolve(), args.dry_run)


if __name__ == '__main__':
    main()
