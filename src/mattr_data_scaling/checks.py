"""Bounded correctness checks, with no registered experimental training."""
import inspect
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.mattr_optimization.common import UPSTREAM,sha,write_json
from src.mattr_hierarchy.adapter import Hierarchy
from src.mattr_hierarchy.runner import tensor_state_equal
from .training import learn

def schedule_audit(out):
    import matryoshka_attribution.trainer as trainer
    from matryoshka_attribution.masks import build_mask
    source=inspect.getsource(trainer.learn_scores)
    assert 'lr_schedule' not in inspect.signature(trainer.learn_scores).parameters
    assert 'range(steps)' in source
    rng=np.random.default_rng(61);plan=pd.DataFrame(dict(row=[0]*5000,k=rng.uniform(1,63,5000)))
    def loss(mask):return ((mask-torch.linspace(0,1,64))**2).sum()
    options=dict(variant='topk',k_schedule='uniform',T=.5,n_iters=30,lr=.05,optimizer='adam',device='cpu')
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        tmp=Path(tmp);adapter=Hierarchy(kind='bypass')
        a=learn(trainer.learn_scores,64,loss,adapter,plan,target=1000,folder=tmp/'standalone',identity='schedule',**options)
        b=learn(trainer.learn_scores,64,loss,adapter,plan,target=5000,folder=tmp/'long',identity='schedule',**options)
        from .training import checked_checkpoint
        mid=checked_checkpoint(tmp/'long/step_001000.pt','schedule')
        assert torch.equal(a['scores'],mid['scores']) and tensor_state_equal(a['optimizer'],mid['optimizer'])
        # Exact optimizer/score/stream state restoration for both parameterizations.
        for method,ad in [('baseline',adapter),('h2',Hierarchy(np.repeat(np.arange(8),8)))]:
            full=learn(trainer.learn_scores,64,loss,ad,plan,target=30,folder=tmp/(method+'_full'),identity=method,**options)
            learn(trainer.learn_scores,64,loss,ad,plan,target=15,folder=tmp/(method+'_split'),identity=method,**options)
            split=learn(trainer.learn_scores,64,loss,ad,plan,target=30,folder=tmp/(method+'_split'),identity=method,**options)
            assert torch.equal(full['scores'],split['scores']) and tensor_state_equal(full['optimizer'],split['optimizer'])
            assert tensor_state_equal(full['rng'],split['rng'])
        for k in (1.,5.5,32.,63.):
            score=torch.linspace(-1,1,64,requires_grad=True);mask=build_mask(score,k,'topk',T=.5,n_iters=30).mask
            assert abs(float(mask.detach().sum())-k)<1e-3
            assert torch.isfinite(torch.autograd.grad(mask.square().sum(),score)[0]).all()
    path=UPSTREAM/'src/matryoshka_attribution/trainer.py';lines,first=inspect.getsourcelines(trainer.learn_scores)
    data=dict(passed=True,intermediate_1000_equivalent=True,optimizer_state_equal=True,baseline_and_h2_exact_resume=True,
              synthetic_cpu_updates=5000+1000+120,registered_GPU_updates=0,
              reference=dict(path=str(path),first_line=first,last_line=first+len(lines)-1,sha256=sha(path)),
              findings=dict(lr='constant Adam lr=.05; no warmup/decay',temperature='constant .5',
                  planned_duration='only loop bound; optimizer initialization independent of steps',
                  model_construction='may consume global Torch RNG; isolated frozen streams remove coupling',
                  resume='scores, Adam, Python/NumPy/Torch/CUDA RNG, independent stream position; donor cache rebuilt and charged'))
    write_json(out/'schedule_audit.json',data)
    (out/'TRAINING_SCHEDULE_AUDIT.md').write_text('# Training schedule audit\n\n'
        'The pinned trainer has constant learning rate and temperature; planned duration only controls the loop bound. '
        'A real CPU execution of the official trainer verified that update 1,000 of a 5,000-update run equals a standalone 1,000-update run, including Adam state. '
        'This is a synthetic correctness test, not an IOI research measurement.\n\n'
        f'Official source: `{path}`, lines {first}–{first+len(lines)-1}, SHA-256 `{sha(path)}`.\n\n'
        'Thirty-update CPU tests checked both baseline and H2 exact 15+15 restoration. The model loss and activation patching are not reimplemented. '
        'Training-example draws and uniform-k draws are separate frozen streams; hierarchy initialization is deterministic zero. '
        'Historical42–44 k sequences are verified for all5,000 draws;45–46 use independent explicitly seeded CPU generators. '
        'Intermediate checkpoints are reusable only when data, source, scores, optimizer and full draw streams pass the separate inventory.\n')

def gpu_preflight(out):
    from .core import configure_runtime,verified,job_key
    from .training import train
    configure_runtime()
    for method in ('baseline','h2'):train(out,128,method,42,target=30,resume=True,smoke=True)
    from .attribution import preflight
    preflight(out)
    updates=seconds=0.
    for path in (out/'attempts').rglob('rates.json'):
        d=json.loads(path.read_text())
        if d.get('updates'):updates+=d['updates'];seconds+=d['training_seconds']
    eap=json.loads((out/'eap_preflight.json').read_text())
    assert updates>0 and all(verified(out/'smoke'/job_key(m,128,42)) for m in ('baseline','h2'))
    write_json(out/'preflight_gpu.json',dict(passed=True,training_seconds_per_update=seconds/updates,
        eap_seconds_per_example=eap['seconds_per_example'],evaluation_seconds_per_ranking_projection=eap['evaluation_seconds_per_ranking_projection'],
        model_weight_sha256=eap['model_weight_sha256'],bounded_smokes_only=True,full_experiment_started=False))
