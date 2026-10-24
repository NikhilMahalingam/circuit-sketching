"""Reconcile descriptive stage totals with unique audited process counters.

Frozen training/selection code and raw stage records are never modified.
Run from repository root after campaign analysis and final audit.
"""
import json
from pathlib import Path
import pandas as pd


def run(out):
    out = Path(out)
    csv = out / 'compute_accounting.csv'
    archive = out / 'compute_accounting_stage_aggregate.csv'
    if not archive.exists():
        archive.write_bytes(csv.read_bytes())
    original = pd.read_csv(archive)
    corrected = original.copy()
    differences = []
    for index, row in corrected.iterrows():
        folder = out / 'jobs' / row.job
        if not folder.exists():
            folder = out / row.job
        paths = sorted((folder / 'attempts').glob('*/process_summary.json')) + [folder / 'process_summary.json']
        counts = {key: 0 for key in ('forward_batches', 'example_forwards', 'backward_calls', 'optimizer_updates')}
        for path in paths:
            process = json.loads(path.read_text())
            assert process['success']
            for key in counts:
                counts[key] += process['counts'][key]
        for key, value in counts.items():
            if int(row[key]) != value:
                differences.append({'job': row.job, 'counter': key, 'stage_sum': int(row[key]), 'unique_process_count': value})
            corrected.loc[index, key] = value
    # The pilot's outer magnitude-probe timer includes its inner patching check.
    stages = pd.read_csv(out / 'pilot' / 'compute_accounting.csv')
    overlap = float(stages.loc[stages.stage == 'official_patching_checks', 'seconds'].sum())
    corrected.loc[corrected.job == 'pilot', 'stage_seconds'] -= overlap
    corrected.to_csv(csv, index=False)
    audit = json.loads((out / 'final_validation.json').read_text())
    for key, expected in audit['counts'].items():
        assert int(corrected[key].sum()) == expected
    note = {'scope': 'descriptive accounting only; frozen protocol, training, raw stage events, metrics, and decisions unchanged',
            'reason': 'nested pilot official_patching_checks is included in training_only_magnitude_probe; stage sums double count 13 forwards and overlapping wall time',
            'counter_corrections': differences, 'overlapping_pilot_seconds': overlap, 'counts': audit['counts']}
    (out / 'accounting_reconciliation.json').write_text(json.dumps(note, indent=2) + '\n')
    for report in [out / 'GATE5C_HIERARCHICAL_MATTR_REPORT.md', Path('GATE5C_HIERARCHICAL_MATTR_REPORT.md')]:
        body = report.read_text()
        for key in ('forward_batches', 'example_forwards'):
            old, new = int(original[key].sum()), int(corrected[key].sum())
            body = body.replace(f'**{old:,}**', f'**{new:,}**')
        old, new = original.stage_seconds.sum() / 60, corrected.stage_seconds.sum() / 60
        body = body.replace(f'**{old:.3f}min**', f'**{new:.3f}min**')
        # Replace only the pilot table row with its reconciled values.
        for line in body.splitlines():
            if line.startswith('| pilot |'):
                values = corrected.loc[corrected.job == 'pilot'].iloc[0]
                replacement = '| pilot | ' + ' | '.join(f'{values[k]:.4f}' if k.endswith('seconds') or k.endswith('mib') else str(int(values[k])) for k in corrected.columns if k != 'job') + ' |'
                body = body.replace(line, replacement)
        body = body.split('\n\n## Accounting reconciliation')[0]
        body += '\n\n## Accounting reconciliation\n\nThe nested pilot patching check was included in its parent timer. The descriptive stage sum duplicated 13 root forwards and 13 example-forwards; the reconciled total is **26,713 root forwards**, **134,161 example-forwards**, **18,140 backward calls**, and **18,120 optimizer updates**. The overlap of 1.525 seconds is also removed from aggregate stage time. Original stage sums remain in `compute_accounting_stage_aggregate.csv`; unique process counters match `final_validation.json`. Timed training, official evaluations, candidate comparisons, frozen decisions, and total child-process wall time are unchanged. The pilot outer-stage false flag reflects its intentional completion sentinel; its process succeeded.\n'
        report.write_text(body)
    print(json.dumps(note, indent=2))


if __name__ == '__main__':
    run('results/mattr_hierarchy/20261022_gate5c')
