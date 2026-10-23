from pathlib import Path
import json,pandas as pd,hashlib
r=Path.cwd();o=r/'results/mattr_compact_validation/20261023_gate5d';cohort=pd.read_csv(o/'cohort_manifest.csv',keep_default_na=False);audit=json.loads((o/'cohort_audit.json').read_text());hashes=set();indices=set()
for rec in audit['excluded_manifests']:
 p=r/rec['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256'];f=pd.read_csv(p,keep_default_na=False)
 if rec['role']=='historical_full_validation_exposure_only':continue
 for col in ['clean_prompt','corrupted_prompt']:
  if col in f:hashes.update(hashlib.sha256(x.encode()).hexdigest() for x in f[col])
 if rec['role']=='excluded_small_evaluation' and 'official_filtered_index' in f:indices.update(f.official_filtered_index.astype(int))
for row in cohort.itertuples():assert hashlib.sha256(row.clean_prompt.encode()).hexdigest()==row.clean_sha256
assert not set(cohort.clean_sha256)&hashes;assert not set(cohort.official_filtered_index)&indices
assert cohort.clean_prompt.is_unique and cohort.clean_sha256.is_unique
(o/'independent_cohort_exclusion_audit.json').write_text(json.dumps({'passed':True,'scope':'independent read-only rehash of all historical cohort files and exact strings; no selection/evaluation changes','selected':len(cohort),'input_manifests':len(audit['excluded_manifests']),'hash_overlaps':0,'index_overlaps':0,'duplicate_clean_strings':0},indent=2)+'\n')
print('Independent512-prompt exclusion audit passed')
