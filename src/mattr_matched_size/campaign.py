"""Explicit topology, freeze, inference, report and preservation-audit stages."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from src.mattr_optimization.common import ROOT, configure, sha, write_json
from .design import prepare,protocol


def audit(out):
    started=time.perf_counter();p=protocol(out)
    for name,value in p['topology_sha256'].items():assert sha(out/name)==value,name
    protected=json.loads((out/'protected_previous_hashes.json').read_text())
    for name,value in protected.items():assert sha(ROOT/name)==value,name
    proc=json.loads((out/'inference_process_summary.json').read_text())
    assert proc['success'] and not proc['projection_stop']
    assert proc['counts']['optimizer_updates']==proc['counts']['backward_calls']==0
    events=pd.read_csv(out/'forward_events.csv')
    assert events.success.all() and not events.grad_enabled.any()
    assert len(events)==proc['counts']['forward_batches'] and int(events.batch_size.sum())==proc['counts']['example_forwards']
    assert json.loads((out/'model_weight_audit.json').read_text())['passed']
    assert json.loads((out/'exact_cache_correctness.json').read_text())['passed']
    assert json.loads((out/'correctness_passed.json').read_text())['passed']
    for rec in p['catalog']:
        saved=json.loads((out/'nominal'/rec['key']/'official_result.json').read_text())
        assert saved['protocol_sha256']==sha(out/'gate5e_protocol.json')
        assert saved['per_prompt_sha256']==sha(out/'nominal'/rec['key']/'per_prompt.csv')
    q=pd.read_csv(out/'circuit_queries.csv')
    measured=pd.read_csv(out/'matched_results.csv')
    keys=['ranking','selection','target']
    expected=q.set_index(keys).sort_index();actual=measured.set_index(keys).sort_index()
    assert expected.index.equals(actual.index)
    for col in ('actual_edges','nominal_n','mask_sha256'):
        assert expected[col].tolist()==actual[col].tolist(),col
    assert not any('closure' in x for x in measured.ranking)
    cohort=pd.read_csv(out/'cohort_manifest.csv',keep_default_na=False)
    old=json.loads((out/'cohort_audit.json').read_text())
    hashes,indices=set(),set()
    from src.mattr_compact_validation.design import text_hash
    for entry in old['exclusion_manifests']:
        path=ROOT/entry['path'];assert sha(path)==entry['sha256']
        if entry.get('role')=='historical_full_validation_exposure_only':continue
        frame=pd.read_csv(path,keep_default_na=False)
        for col in ('clean_prompt','corrupted_prompt'):
            if col in frame:hashes.update(text_hash(x) for x in frame[col])
        if entry.get('role')!='excluded_training' and 'official_filtered_index' in frame:
            indices.update(frame.official_filtered_index.astype(int))
    selected=set(cohort.clean_sha256)|set(cohort.corrupted_sha256)
    assert len(selected)==2*len(cohort) and not selected&hashes and not set(cohort.official_filtered_index)&indices
    result={'passed':True,'protocol_sha256':sha(out/'gate5e_protocol.json'),'historical_files_unchanged':len(protected),
            'full_grid_rankings':len(p['catalog']),'capped_upper_queries':len(q),'cohort_prompts':len(cohort),
            'clean_donor_crossrole_overlap':0,'no_matched_closure_queries':True,'original_checkpoint_hashes_verified':True,
            'model_weights_unchanged':True,'actual_forwards_verified':True,'optimizer_updates':0,'model_backwards':0,
            'Gate5D_decision_unchanged':True,'additional_model_forwards':0,'seconds':time.perf_counter()-started}
    write_json(out/'final_validation.json',result)
    for path in (out/'GATE5E_MATCHED_SIZE_REPORT.md',ROOT/'GATE5E_MATCHED_SIZE_REPORT.md'):
        body=path.read_text().split('\n\n## Final preservation audit')[0]
        path.write_text(body+'\n\n## Final preservation audit\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n')
    print(json.dumps(result,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/mattr_matched_size/20261023_gate5e'))
    parser.add_argument('--stage',choices=['topology','prepare','inference','report','audit'],default='prepare')
    a=parser.parse_args();configure();out=a.out.resolve()
    if a.stage=='topology':
        from .topology import run
        run(out)
    elif a.stage=='prepare':prepare(out)
    elif a.stage=='inference':
        from .inference import run
        run(out)
    elif a.stage=='report':
        from .analysis import generate
        generate(out)
    elif a.stage=='audit':audit(out)


if __name__=='__main__':main()
