"""Temporary synthetic fixtures test report plumbing, never benchmark results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization import gate2_final_analysis as analysis
from src.mattr_optimization.common import GRID, sha


class ReportTests(unittest.TestCase):
    def test_complete_analysis_and_immutable_criteria(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            campaign=root/'main';campaign.mkdir()
            controls=root/'controls';controls.mkdir()
            frozen=root/'frozen';frozen.mkdir()
            qs=[];fs=[];us=[];ss=[];cs=[];extras=[]
            val=pd.DataFrame({'official_filtered_index':[5,9],'clean_prompt':['a','b']})
            def quality(seed,n):
                return pd.DataFrame([{'seed':seed,'pool':n,'split':split,'step':step,
                    'cpr':n/128+step/50000,'cmd':.2,'accuracy_auc':.7}
                    for split,step in [('validation',500),('validation',1000),('validation',5000),('train',5000)]])
            def curves(seed,n):
                return pd.DataFrame([{'seed':seed,'pool':n,'step':step,'split':'validation','fraction':f,
                    'weighted_edge_count':int(f*32491),'faithfulness':f,'accuracy':.7}
                    for step in (500,1000,5000) for f in GRID])
            for seed in (42,43,44):
                for n in (16,32,64,128):
                    job=campaign/f'seed{seed}_n{n}';job.mkdir()
                    (job/'checkpoints').mkdir()
                    (job/'experiment_config.json').write_text(json.dumps({'seed':seed,'pool':n,'source_provenance':{}}))
                    (job/'completed.json').write_text(json.dumps({'passed':True,'sha256':{'experiment_config.json':sha(job/'experiment_config.json')}}))
                    val.to_csv(job/'validation_manifest.csv',index=False)
                    pd.DataFrame({'official_filtered_index':np.arange(n)+seed*1000}).to_csv(job/'train_manifest.csv',index=False)
                    pd.DataFrame({'subset_index':np.arange(5000)%n}).to_csv(job/'training_example_draws.csv',index=False)
                    for split in ('train','validation'):
                        evaluation=job/f'evaluation/{split}_step5000';evaluation.mkdir(parents=True)
                        (evaluation/'official_result.json').write_text(json.dumps({'references':{'baseline':3.,'corrupted':-2.}}))
                    for step in (500,1000,5000):
                        torch.save({'training_seconds':step*.3},job/f'checkpoints/step_{step:06d}.pt')
                        if step!=5000:
                            extras.append({'kind':'checkpoint_train_diagnostic','seed':seed,'pool':n,'split':'train',
                                           'step':step,'cpr':1.1,'cmd':.3,'accuracy_auc':.8})
                    qs.append(quality(seed,n));fs.append(curves(seed,n))
                    us.append(pd.DataFrame({'seed':seed,'pool':n,'subset_index':np.arange(n),'actual_presentations':np.bincount(np.arange(5000)%n)}))
                    ss.append({'seed':seed,'pool':n,'training_seconds':1500,'measured_stage_seconds':1600,
                               'forward_batches':5500,'example_forwards':15000,'peak_allocated_mib':4000})
                    cs.append({'seed':seed,'pool':n,'stage':'training','seconds':1500})
            for name,frames in [('quality_by_step',qs),('faithfulness_curves',fs),('training_example_usage',us)]:
                pd.concat(frames).to_csv(campaign/(name+'.csv'),index=False)
            pd.DataFrame(ss).to_csv(campaign/'sample_efficiency.csv',index=False)
            pd.DataFrame(cs).to_csv(campaign/'compute_accounting.csv',index=False)
            for train,subset in ((43,42),(44,42),(42,43),(42,44)):
                job=controls/f'train{train}_subset{subset}';job.mkdir()
                (job/'completed.json').write_text(json.dumps({'passed':True,'sha256':{}}))
                (job/'experiment_config.json').write_text(json.dumps({'source_provenance':{},'campaign_protocol':{'training_seed':train,'subset_seed':subset}}))
                (job/'upstream_command.json').write_text(json.dumps({'unchanged_algorithm':True}))
                val.to_csv(job/'validation_manifest.csv',index=False)
                pd.read_csv(campaign/f'seed{subset}_n128/train_manifest.csv').to_csv(job/'train_manifest.csv',index=False)
                quality(train,128).to_csv(job/'quality.csv',index=False)
                pd.DataFrame({'seconds':[1600]}).to_csv(job/'compute_accounting.csv',index=False)
            extras.append({'kind':'full_data_reference','seed':42,'pool':10000,'split':'validation','step':5000,'cpr':1.2,'cmd':.4,'accuracy_auc':.8})
            pd.DataFrame(extras).to_csv(frozen/'quality.csv',index=False)
            (frozen/'completed.json').write_text(json.dumps({'passed':True,'gate1_unchanged':True,'training_updates':0}))
            ref=frozen/'gate1_on_gate2_validation';ref.mkdir()
            curves(42,128).query('step==5000')[['fraction','weighted_edge_count','faithfulness','accuracy']].to_csv(ref/'faithfulness_curve.csv',index=False)
            pd.DataFrame({'seconds':[60]}).to_csv(frozen/'compute_accounting.csv',index=False)
            out=root/'final'
            argv=['analysis','--campaigns',str(campaign),'--controls',str(controls),'--frozen',str(frozen),'--out',str(out)]
            with patch.object(analysis,'ROOT',root),patch('sys.argv',argv):
                analysis.main()
                criteria=sha(out/'gate3_success_criteria.json')
                analysis.main()
                self.assertEqual(criteria,sha(out/'gate3_success_criteria.json'))
            self.assertEqual(len(pd.read_csv(out/'quality_all_steps.csv')),36)
            self.assertEqual(len(pd.read_csv(out/'conditional_seed_controls.csv')),21)
            self.assertTrue((root/'GATE2_FINAL_ANALYSIS.md').exists())
            frozen_protocol=json.loads((out/'gate3_success_criteria.json').read_text())
            self.assertFalse(frozen_protocol['gate3_executed'])


if __name__=='__main__':
    unittest.main()
