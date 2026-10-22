"""Temporary synthetic report fixture: high CPR must not bypass compact guards."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from src.mattr_optimization import gate3_report as report
from src.mattr_optimization.gate3 import CRITERIA
from src.mattr_optimization.common import ROOT,GRID,sha


class ScreeningReportTests(unittest.TestCase):
    def test_negative_stop_and_single_winner_confirmation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'campaign';out.mkdir()
            shutil.copyfile(CRITERIA,out/'gate3_success_criteria.json')
            (out/'gate3_protocol.json').write_text(json.dumps({'screen_rule':'fixture frozen rule'}))
            (out/'gate3_protocol.sha256').write_text(sha(out/'gate3_protocol.json'))
            (out/'GATE3_SAMPLING_PROTOCOL.md').write_text('Explicitly synthetic CPU fixture only.')
            for lam in (1.,.5,.25):
                for seed in (42,43,44):
                    job=out/f'lambda{lam}_seed{seed}';job.mkdir()
                    (job/'experiment_config.json').write_text(json.dumps({'lambda':lam,'seed':seed}))
                    (job/'completed_1000.json').write_text(json.dumps({'passed':True,'sha256':{}}))
                    (job/'quality_1000.json').write_text(json.dumps({'lambda':lam,'seed':seed,'step':1000,
                        'cpr':10 if lam==1 else 11,'cmd':9,'accuracy_auc':.8 if lam==1 else .5}))
                    (job/'launch_elapsed_1000.json').write_text(json.dumps({'seconds':115}))
                    c=pd.DataFrame([{'stage':'training','seconds':100,'success':True,'peak_allocated_mib':100,
                        'forward_batches':1128,'example_forwards':1128,'backward_calls':1000,'optimizer_updates':1000}])
                    c.to_csv(job/'compute_accounting.csv',index=False)
                    folder=job/'evaluation_step1000';folder.mkdir()
                    pd.DataFrame({'fraction':GRID,'weighted_edge_count':[1]*10,'faithfulness':[1.]*10,
                                  'accuracy':[.8 if lam==1 else .5]*10}).to_csv(folder/'faithfulness_curve.csv',index=False)
            def baseline(seed,step):
                d={'seed':seed,'step':step,'cpr':10.,'cmd':9.,'accuracy_auc':.8,
                   'training_seconds':100. if step==1000 else 500.,'peak_allocated_mib':100.}
                for f in (.005,.01,.02,.05):d[f'accuracy_{f}']=.8
                return d
            with patch.object(report,'ROOT',root),patch.object(report,'baseline_row',baseline):
                result=report.generate(out)
                self.assertIsNone(result['winner'])
                self.assertTrue(all(not x['eligible'] for x in result['screen_decisions'].values()))
                self.assertEqual(result,report.generate(out))
            self.assertEqual(len(pd.read_csv(out/'paired_results.csv')),6)
            self.assertEqual(len(pd.read_csv(out/'faithfulness_curves.csv')),90)
            self.assertIn('Negative result',(out/'GATE3_ADAPTIVE_SAMPLING_REPORT.md').read_text())
            # Separate synthetic campaign: both qualify; only the higher frozen
            # relative-CPR candidate may continue, retaining an identical screen decision.
            positive=root/'positive';shutil.copytree(out,positive)
            (positive/'screen_decision.json').unlink()
            for job in positive.glob('*_seed*'):
                config=json.loads((job/'experiment_config.json').read_text());lam=config['lambda']
                if lam==1.:continue
                path=job/'quality_1000.json';q=json.loads(path.read_text())
                q.update(cpr=10.4 if lam==.5 else 10.3,cmd=9.4 if lam==.5 else 9.3,accuracy_auc=.8)
                path.write_text(json.dumps(q))
                curve=pd.read_csv(job/'evaluation_step1000/faithfulness_curve.csv');curve.accuracy=.8
                curve.to_csv(job/'evaluation_step1000/faithfulness_curve.csv',index=False)
            with patch.object(report,'ROOT',root),patch.object(report,'baseline_row',baseline):
                chosen=report.generate(positive)
                self.assertEqual(chosen['winner'],.5)
            for job in positive.glob('lambda0.5_seed*'):
                archived=job/'attempts/fixture';archived.mkdir(parents=True)
                shutil.copyfile(job/'compute_accounting.csv',archived/'compute_accounting.csv')
                (archived/'process_summary.json').write_text('{}')
                c=pd.read_csv(job/'compute_accounting.csv');c.seconds=400;c.optimizer_updates=4000;c.backward_calls=4000
                c.to_csv(job/'compute_accounting.csv',index=False)
                (job/'completed_5000.json').write_text(json.dumps({'passed':True,'sha256':{}}))
                q=json.loads((job/'quality_1000.json').read_text());q.update(step=5000,cpr=11.,cmd=10.)
                (job/'quality_5000.json').write_text(json.dumps(q))
                (job/'launch_elapsed_5000.json').write_text(json.dumps({'seconds':415}))
                shutil.copytree(job/'evaluation_step1000',job/'evaluation_step5000')
            with patch.object(report,'ROOT',root),patch.object(report,'baseline_row',baseline):
                self.assertEqual(report.generate(positive),chosen)
            assessment=json.loads((positive/'success_assessment.json').read_text())
            self.assertTrue(assessment['confirmation']['passed'])
            self.assertFalse(assessment['confirmation']['general_model_fidelity_improvement'])
            self.assertEqual(len(pd.read_csv(positive/'paired_results.csv').query('step==5000')),3)


if __name__=='__main__':unittest.main()
