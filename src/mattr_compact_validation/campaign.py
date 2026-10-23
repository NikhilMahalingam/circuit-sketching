"""Explicit prepare/inference/report/audit stages, with no trainer entry point."""
import argparse
import json
from pathlib import Path
import time
import pandas as pd
from src.mattr_optimization.common import ROOT, configure, sha, write_json
from .design import prepare, validate_protocol


def audit(out):
    p = validate_protocol(out)
    protected = json.loads((out / 'protected_previous_hashes.json').read_text())
    for name, expected in protected.items():
        assert sha(ROOT / name) == expected, name
    process = json.loads((out / 'inference_process_summary.json').read_text())
    assert process['success'] and process['counts']['optimizer_updates'] == process['counts']['backward_calls'] == 0
    events = pd.read_csv(out / 'forward_events.csv')
    assert events.success.all() and not events.grad_enabled.any()
    assert len(events) == process['counts']['forward_batches']
    assert int(events.batch_size.sum()) == process['counts']['example_forwards']
    assert json.loads((out / 'model_weight_audit.json').read_text())['passed']
    count = 0
    for path in sorted((out / 'evaluations').glob('*/official_result.json')):
        record = json.loads(path.read_text())
        assert record['identity']['protocol_sha256'] == sha(out / 'gate5d_protocol.json')
        assert record['per_prompt_sha256'] == sha(path.parent / 'per_prompt.csv')
        count += 1
    assert count == 15
    result = {'passed': True, 'historical_files_unchanged': len(protected), 'evaluated_rankings': count,
              'protocol_sha256': sha(out / 'gate5d_protocol.json'), 'cohort_size': p['cohort']['size'],
              'actual_forwards_verified': True, 'weight_updates': 0, 'model_backwards': 0,
              'frozen_sources_and_checkpoints_verified': True, 'additional_model_forwards': 0}
    write_json(out / 'final_validation.json', result)
    for path in (out / 'GATE5D_COMPACT_GENERALIZATION_REPORT.md', ROOT / 'GATE5D_COMPACT_GENERALIZATION_REPORT.md'):
        body = path.read_text().split('\n\n## Final preservation audit')[0]
        path.write_text(body + '\n\n## Final preservation audit\n\n```json\n' + json.dumps(result, indent=2) + '\n```\n')
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/mattr_compact_validation/20261023_gate5d'))
    parser.add_argument('--stage', choices=['prepare','inference','report','audit','all'], default='prepare')
    args = parser.parse_args()
    configure()
    out = args.out.resolve()
    if args.stage in ('prepare','all'):
        prepare(out)
    if args.stage in ('inference','all'):
        from .inference import run
        run(out)
    if args.stage in ('report','all'):
        from .analysis import generate
        generate(out)
    if args.stage in ('audit','all'):
        audit(out)


if __name__ == '__main__':
    main()
