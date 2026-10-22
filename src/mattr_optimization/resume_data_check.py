"""Before a Gate-1 restart, compare current official rows with frozen manifests."""
import argparse
import csv
import json
from pathlib import Path
import time


def compare(dataset, path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(dataset):
        raise ValueError('Official data length changed since checkpoint.')
    for i, old in enumerate(rows):
        clean, donor, labels = dataset[i]
        if (int(old['official_filtered_index']) != i or old['clean_prompt'] != clean or
            old['corrupted_prompt'] != donor or int(old['correct_token']) != labels[0] or
            int(old['incorrect_token']) != labels[1] or
            json.loads(old['metadata']) != dataset.dataset[i]['metadata']):
            raise ValueError(f'Official row changed since checkpoint: {path}:{i}')


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report-only', action='store_true')
    args, _ = parser.parse_known_args()
    if args.report_only or not (args.out/'experiment_config.json').exists():
        return
    start = time.perf_counter()
    from .common import configure
    configure()
    from transformers import AutoTokenizer
    from MIB_circuit_track.dataset import HFEAPDataset
    tokenizer = AutoTokenizer.from_pretrained('gpt2', local_files_only=True)
    checked = {}
    for split in ('train', 'validation'):
        manifest = args.out/f'{split}_manifest.csv'
        if not manifest.exists():
            continue
        official = HFEAPDataset('mib-bench/ioi', tokenizer, split=split, task='ioi', model_name='gpt2')
        compare(official, manifest)
        checked[split] = len(official)
    record = {'passed': True, 'checked_rows': checked, 'seconds': time.perf_counter()-start,
              'additional_model_forwards': 0}
    with (args.out/'resume_data_checks.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record)+'\n')
    print('Resume data guard:', record, flush=True)


if __name__ == '__main__':
    main()
