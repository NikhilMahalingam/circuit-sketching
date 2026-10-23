from pathlib import Path
import json,pandas as pd,hashlib
r=Path.cwd();o=r/'results/mattr_compact_validation/20261023_gate5d'
s=pd.read_csv(o/'method_means.csv').set_index('method');b=pd.read_csv(o/'paired_bootstrap_accuracy.csv');v=json.loads((o/'gate5d_decision.json').read_text())
assert v['classification']=='Mixed or inconclusive' and not v['primary']['passed']
h,base=s.loc['hierarchy'],s.loc['baseline']
text='''## Interpretation of the completed experiment

The registered outcome is **mixed/inconclusive**, not a compact-generalization success under all frozen criteria. At **0.5% nominal retention**, mean positive-margin accuracy increased from **16.67% to 53.58%** (+36.91 percentage points; simultaneous95% interval **+33.79 to +40.01 points**), with positive changes in all three seeds. At **1%**, it increased from **70.96% to 88.74%** (+17.77 points; interval **+16.05 to +19.40**), but the changes were **+53.91, 0.00 and -0.59 points** for seeds42/43/44. Thus only one seed improved at1%, failing the prespecified at-least-two-seeds condition. The pooled bootstrap result does not override that condition. Baseline seeds43/44 were already at99.80%/100% at1%; this ceiling is context for the negative consistency decision, not a reason to relax it.

The gamma-zero hierarchy reached **53.52%/89.13%** at0.5%/1%, essentially matching or slightly exceeding the regularized hierarchy. This supports parameterization/optimization geometry as the main explanation rather than a demonstrated benefit from the residual penalty. The real hierarchy exceeded shuffled groups by **15.69 and4.04 points**, respectively, passing the prespecified pooled control comparisons with two-seed consistency. Generic edge regularization stayed near ordinary MAttr (**16.47%/71.03%**) and does not explain the large pooled compact gains; however, the real-versus-edge-L2 comparison also misses two-positive-seed consistency at1%. The complete structural-advantage criterion does not pass because all required control safeguards must pass.

Mean accuracy log-AUC rose **0.710848 to0.782791** (+0.071943; secondary95% paired prompt interval +0.067091 to+0.076649). Mean CPR fell **9.904998 to9.869015**, and mean CMD fell **8.921454 to8.879688**. There is no CPR or leaderboard improvement claim. Full-vocabulary top1, captured separately from the same logits, rose from12.17% to34.83% at0.5% and69.47% to78.26% at1%; these are not the positive-margin endpoints.

All nine exact official template strata had positive equally seed-weighted mean primary differences at both cutoffs. This is a descriptive result within existing template families, conditional on the seed mixture; it does not cure the1% seed inconsistency or establish unseen-template generalization. `official_template_labels.csv` maps each hash to its exact generic benchmark template, with no individual prompt text. Independent exclusion and optimizer-step audits supplement the frozen-protocol checks without changing the cohort, metrics or decisions.

The evidence supports a **narrow research motivation for very compact circuits, particularly the unregularized hierarchical parameterization**, rather than a validated regularization-specific improvement. Keep ordinary MAttr as the CPR reference and preserve the failed Gate5C/5D decisions. No further training, algorithm, ranking refinement, or leaderboard submission was performed.

'''
(o/'INTERPRETATION.md').write_text(text)
for path in [o/'GATE5D_COMPACT_GENERALIZATION_REPORT.md',r/'GATE5D_COMPACT_GENERALIZATION_REPORT.md']:
 body=path.read_text();first,rest=body.split('\n',1)
 if rest.lstrip().startswith('## Interpretation of the completed experiment'):
  rest=rest[rest.index('## Executed decision'):]
 path.write_text(first+'\n\n'+text+rest.lstrip())
print('Added descriptive interpretation; protocol and frozen decisions unchanged')
