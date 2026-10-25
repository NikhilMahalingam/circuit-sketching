"""Add descriptive report supplements after the official07_package stage."""
from pathlib import Path
import hashlib,json,shutil

out=Path(__file__).resolve().parent;dest=out/'package'
path=dest/'package_manifest.json';manifest=json.loads(path.read_text(encoding='utf-8'))
names=['GATE5F_DATA_SCALING_REPORT.generated.md','INTERPRETATION.md','report_interpretation.py',
       'interpretation_provenance.json','full_data_primary_summary.csv','primary_seed_interpretation.csv',
       'factorial_interpretation.csv','actual_training_example_usage.csv','known_head_summary.csv',
       'matched_training_overhead.csv','package_report_supplements.py']
for name in names:
    shutil.copyfile(out/name,dest/name)
    manifest['sha256'][name]=hashlib.sha256((dest/name).read_bytes()).hexdigest()
manifest['post_execution_supplements']='Descriptive report interpretation only; no frozen source, criteria, model output or decision modified.'
path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
for name,h in manifest['sha256'].items():assert hashlib.sha256((dest/name).read_bytes()).hexdigest()==h,name
print('Report package and supplement checksums verified.')
