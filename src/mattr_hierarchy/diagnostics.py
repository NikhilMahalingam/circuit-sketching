"""Supplemental descriptive score decomposition; never used for selection.

The frozen model, sampling, regularization, control and advancement protocols
remain untouched. This module reads existing vectors and emits no model forward.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import ROOT,sha,write_json
from src.mattr_optimization.gate2_final_analysis import table
from .adapter import Hierarchy

def run(out):
    started=time.perf_counter();torch.set_num_threads(4)
    mapping=pd.read_csv(out/'edge_mapping.csv',keep_default_na=False)
    rows=[]
    for folder in sorted((out/'jobs').glob('*')):
        if not (folder/'completed_1000.json').exists():continue
        cfg=json.loads((folder/'experiment_config.json').read_text())
        ids=np.load(out/f'{cfg["family"]}{"_shuffled" if cfg["shuffled"] else ""}_groups.npy') if cfg['family'] in ('H1','H2') else None
        transform=Hierarchy(ids,kind='hierarchy' if ids is not None else cfg['family'])
        for path in sorted(folder.glob('quality_*.json')):
            step=int(path.stem.split('_')[-1])
            state=torch.load(folder/f'checkpoints/step_{step:06d}.pt',map_location='cpu',weights_only=False);raw=state['scores']
            effective,rc=transform.parts(raw)
            ranked=pd.read_csv(folder/f'ranking_{step}.csv').set_index('edge')
            exact=torch.tensor(ranked.loc[mapping.edge,'score'].to_numpy(),dtype=torch.float32)
            torch.testing.assert_close(effective,exact,atol=2e-6,rtol=1e-5)
            h=raw[:transform.group_count] if ids is not None else torch.zeros(1)
            expanded=h[transform.ids] if ids is not None else torch.zeros_like(effective)
            ve=float(effective.double().var(unbiased=False));vh=float(expanded.double().var(unbiased=False));vr=float(rc.double().var(unbiased=False))
            rows.append({'job':folder.name,'family':cfg['family'],'gamma':cfg['gamma'],'shuffled':cfg['shuffled'],'seed':cfg['seed'],'step':step,
               'weighted_component_variance':vh,'centered_residual_variance':vr,'effective_variance':ve,
               'component_variance_fraction':vh/max(ve,1e-30),'residual_variance_fraction':vr/max(ve,1e-30),
               'variance_additivity_error':ve-vh-vr,'between_component_variance_unweighted':float(h.double().var(unbiased=False))})
            for name,value in [('original_parameters',raw),('effective_scores',exact),('raw_residuals',raw[transform.group_count:]),('centered_residuals',rc)]:
                target=folder/f'{name}_{step}.pt'
                if not target.exists():torch.save(value,target)
    frame=pd.DataFrame(rows);frame.to_csv(out/'effective_score_decomposition.csv',index=False)
    gradients=pd.read_csv(out/'gradient_diagnostics.csv')
    summary=gradients.groupby('job').agg(median_abs_task_loss=('mattr_loss',lambda x:abs(x).median()),median_regularization_loss=('regularization_loss','median'),
           median_gradient_ratio=('gradient_ratio','median'),median_Adam_update_contribution=('adam_update_regularizer_relative_contribution','median'),
           max_Adam_update_contribution=('adam_update_regularizer_relative_contribution','max')).reset_index()
    summary.to_csv(out/'regularization_contribution_summary.csv',index=False)
    # Separate methods in the requested stability figure instead of pooling controls.
    stability=pd.read_csv(out/'ranking_stability.csv')
    if not stability.empty:
        import matplotlib.pyplot as plt
        latest=stability.step.max()
        display=stability[(stability.step==latest)&stability.level.isin(['edge','H1','H2'])]
        figure,axis=plt.subplots(figsize=(9,4))
        for (family,gamma,shuffled),g in display.groupby(['family','gamma','shuffled']):
            means=g.groupby('level').spearman.mean().reindex(['edge','H1','H2'])
            label=f'{family}, gamma={gamma:g}'+(' shuffled' if shuffled else '')
            axis.plot(range(3),means.to_numpy(),marker='o',label=label)
        axis.set_xticks(range(3),['Individual edges','H1 source means','H2 receiver means'])
        axis.set_ylabel('Mean cross-seed Spearman (three seed pairs)')
        axis.set_title(f'Matched {latest}-update rankings; descriptive stability')
        axis.legend(fontsize=7);figure.tight_layout()
        figure.savefig(out/'figures/component_vs_edge_stability.png',dpi=150);plt.close(figure)
    state={'source_sha256':sha(Path(__file__)),'scope':'post-run descriptive variance identity and contribution summaries; no protocol/decisions changed',
           'additional_model_forwards':0,'seconds':time.perf_counter()-started}
    write_json(out/'supplemental_diagnostics_provenance.json',state)
    notes='\n\n## Effective-score decomposition and measured penalty contribution\n\n'
    notes+='This supplementary analysis is descriptive and does not change the frozen selection or advancement criteria. Group means are expanded back to edges before computing the component fraction of effective-score variance, so unequal group sizes are accounted for. Centered residuals are orthogonal to group-constant scores up to floating-point error. Original packed, raw residual, centered residual and exact exported effective vectors are saved locally at each evaluated checkpoint.\n\n'
    notes+=table(frame)+'\n\n'+table(summary)
    notes+='\n\nThe prescribed mean-square coefficients have small aggregate gradient norms; same-state Adam contribution is a more direct diagnostic of their update effect. This does not authorize another gamma or scale sweep. Coarse score sharing can still remove edge-specific ranking information, and component stability without corresponding official accuracy/CMD evidence does not establish better causal fidelity.\n'
    for path in (out/'GATE5C_HIERARCHICAL_MATTR_REPORT.md',ROOT/'GATE5C_HIERARCHICAL_MATTR_REPORT.md'):
        body=path.read_text().split('\n\n## Effective-score decomposition')[0]
        path.write_text(body.rstrip()+notes)
    print(json.dumps(state,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args().out.resolve())
