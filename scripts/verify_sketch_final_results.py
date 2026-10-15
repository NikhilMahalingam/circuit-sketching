"""Validate saved final results without rerunning a model or changing settings."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('result_dir',type=Path)
args=parser.parse_args();out=args.result_dir.resolve()
config=json.loads((out/'experiment_config.json').read_text())
missing_local_only=[]
def unchanged(name,sha,optional=False):
    name=name.replace('\\','/');path=ROOT/name
    if optional and not path.exists() and name.endswith('.failed.ipynb'):
        missing_local_only.append(name)
        return
    content=path.read_bytes()
    raw=hashlib.sha256(content).hexdigest()
    portable=config.get('portable_text_sha256',{}).get(name)
    assert raw==sha or (portable and hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest()==portable), name
for name,sha in config['old_notebooks'].items():
    unchanged(name,sha,optional=True)
for name,sha in config['implementation'].items():
    unchanged(name,sha)
assert json.loads((out/'correctness_passed.json').read_text())['passed']
query_set=set(pd.read_csv(out/'query_definitions.csv').query_id)
decision_pairs=pd.read_csv(out/'decision_pairs.csv')
fixed_validation={};nested_train={};checked=[]
for marker in sorted(out.glob('seed*_n*/complete.json')):
    folder=marker.parent;info=json.loads(marker.read_text());n=info['discovery_examples']
    assert info['validation_examples']==128 and not info['smoke']
    pred=pd.read_csv(folder/'predictions.csv',dtype={'state':str})
    effects=pd.read_csv(folder/'per_example_effects.csv')
    reported=pd.read_csv(folder/'prediction_comparison.csv').set_index(['anchor_strength','method'])
    for (anchor,method),group in pred.groupby(['anchor_strength','method']):
        assert not group.query_id.duplicated().any()
        assert set(group.query_id)==query_set
        row=reported.loc[(anchor,method)]
        np.testing.assert_allclose(row.mae,abs(group.prediction-group.validation_effect).mean(),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(row.spearman,group.prediction.rank().corr(group.validation_effect.rank()),atol=1e-12,rtol=1e-12)
        decisions=group.merge(decision_pairs,on='query_id',validate='one_to_many')
        exact=abs(decisions.validation_effect)<decisions.threshold
        approximate=abs(decisions.prediction)<decisions.threshold
        difficult=decisions.difficult.to_numpy(dtype=bool)
        checks={'decision_accuracy':(exact==approximate).mean(),
                'false_prune_rate':(approximate & ~exact).mean(),
                'false_retain_rate':(~approximate & exact).mean(),
                'difficult_accuracy':(exact[difficult]==approximate[difficult]).mean(),
                'difficult_decision_count':int(difficult.sum())}
        for name,value in checks.items():np.testing.assert_allclose(row[name],value,atol=1e-12,rtol=1e-12)
    assert np.isfinite(pred[['prediction','train_effect','validation_effect']].to_numpy()).all()
    for row in pred.itertuples():
        assert len(row.state)==16 and row.state[row.candidate_index]=='0'
    for (qid,split),group in effects.groupby(['query_id','split']):
        values=group.sort_values('example_index').effect.to_numpy()
        assert len(values)==(n if split=='train' else 128)
        col=split+'_effect'
        np.testing.assert_allclose(pred[pred.query_id==qid][col],values.mean(),atol=1e-12,rtol=1e-12)
        if split=='validation':
            if qid in fixed_validation:np.testing.assert_allclose(values,fixed_validation[qid],atol=1e-12,rtol=1e-12)
            else:fixed_validation[qid]=values
        else:
            key=(info['seed'],qid)
            if key in nested_train:
                previous=nested_train[key];prefix=min(len(previous),len(values))
                np.testing.assert_allclose(values[:prefix],previous[:prefix],atol=1e-10,rtol=1e-10)
            nested_train[key]=values
    sk=pred[pred.method.str.startswith('sketch_rank_')]
    np.testing.assert_allclose(sk.prediction-sk.validation_effect,
        sk.taylor_signed_error+sk.compression_signed_error+sk.generalization_shift,atol=1e-10,rtol=1e-10)
    stages=pd.read_csv(folder/'compute_accounting.csv');calls=pd.read_csv(folder/'forward_log.csv')
    assert int(stages.forward_passes_batches.sum())==len(calls)
    assert int(stages.examples_processed.sum())==int(calls.batch_size.sum())
    assert stages.success.all() and stages.peak_cuda_allocated_mib.max()<12288
    checked.append({'seed':info['seed'],'discovery_examples':n,'queries':len(query_set),
                    'forward_batches':len(calls),'peak_cuda_allocated_mib':float(stages.peak_cuda_allocated_mib.max())})
assert all(any(row['seed']==42 and row['discovery_examples']==n for row in checked) for n in [16,32,64,128])
pending=config.get('pending_seed_extension',[])
assert not pending, f'Pending seed extension: {pending}'
summary={'passed':True,'configurations':checked,'fixed_validation_examples':128,
         'validation_effects_identical_across_configurations':True,'previous_notebooks_unchanged':True,
         'missing_ignored_local_failed_notebooks':missing_local_only,
         'decomposition_identity_checked':True,'forward_logs_reconciled':True,'no_smoke_fallback':True}
summary['nested_train_effect_prefixes_identical']=True
summary['heldout_metrics_independently_recomputed']=True
(out/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
