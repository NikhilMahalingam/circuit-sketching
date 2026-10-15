"""Add measured H1 setup cost to an already executed Stage C pilot.

This is deterministic postprocessing of saved logs; it never reruns or changes
an intervention result. Original per-method loop times remain in wall_seconds.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('result_dir', type=Path)
out = parser.parse_args().result_dir
stages = pd.read_csv(out/'runtime_breakdown.csv')
quality = pd.read_csv(out/'circuit_quality.csv')
h1 = stages.loc[stages.stage == 'stage_b_h1']
if len(h1) != 1:
    raise ValueError('Expected one measured H1 stage.')
h1 = h1.iloc[0]
hybrid = quality.method != 'exact_acdc'
quality['shared_h1_experiment_seconds'] = np.where(hybrid, h1.wall_seconds, 0.0)
quality['shared_h1_forward_batches'] = np.where(hybrid, h1.forward_passes_batches, 0)
quality['wall_seconds_including_h1_experiment'] = quality.wall_seconds + quality.shared_h1_experiment_seconds
quality['forward_batches_including_h1_experiment'] = quality.forward_batches + quality.shared_h1_forward_batches
quality.to_csv(out/'circuit_quality.csv', index=False)

decisions = pd.read_csv(out/'hybrid_acdc_results.csv')
fig, axes = plt.subplots(1,3,figsize=(15,4))
axes[0].scatter(quality.wall_seconds_including_h1_experiment,quality.validation_score)
axes[1].scatter(quality.forward_batches_including_h1_experiment+quality.backward_calls,quality.validation_score)
fractions = decisions.groupby('method').exact_verification.mean()
hybrid_quality = quality.set_index('method').loc[fractions.index]
axes[2].scatter(fractions,hybrid_quality.validation_score)
axes[0].set(xlabel='wall seconds incl. measured H1 experiment',ylabel='validation IOI score')
axes[1].set(xlabel='forward + backward calls incl. H1 (unweighted)',ylabel='validation IOI score')
axes[2].set(xlabel='fraction exactly verified',ylabel='validation IOI score')
fig.tight_layout(); fig.savefig(out/'h2_quality_cost.png',dpi=150);plt.close(fig)
refresh_quality=quality[quality.method.str.startswith('second_order_refresh_')]
refresh_axis=refresh_quality.refresh_after_removals.map(lambda x:-1 if x=='error' else (0 if pd.isna(x) else x))
fig,ax=plt.subplots();ax.scatter(refresh_axis,refresh_quality.wall_seconds,c=refresh_quality.validation_score)
ax.set(xlabel='refresh interval (-1=train error trigger; 0=none)',ylabel='wall seconds')
fig.tight_layout();fig.savefig(out/'h2_refresh.png',dpi=150);plt.close(fig)
print(quality[['method','wall_seconds','shared_h1_experiment_seconds',
               'wall_seconds_including_h1_experiment','validation_score']].to_string(index=False))
