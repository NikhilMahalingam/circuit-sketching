"""Local packed-score checkpoint adapter over the unchanged official Adam/topk loop."""
import random
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def checkpointed(original, total, loss_fn, *, folder, identity, interval=10,
                 on_update=None, parameter_count=None, inspect_step=None, **kwargs):
    if kwargs.get('optimizer', 'adam') != 'adam' or kwargs.get('variant', 'topk') != 'topk':
        raise ValueError('Gate-0 checkpoint adapter supports only official Adam/topk.')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    candidates = sorted(folder.glob('step_*.pt'))
    saved = torch.load(candidates[-1], map_location='cpu', weights_only=False) if candidates else None
    if saved and saved['identity'] != identity:
        raise ValueError('Checkpoint identity changed; start a new run directory.')
    target = kwargs.pop('steps')
    offset = saved['completed'] if saved else 0
    if offset > target:
        raise ValueError('Checkpoint exceeds requested budget.')
    prefix = saved['train_log'] if saved else []
    elapsed = saved['training_seconds'] if saved else 0.
    if offset == target:
        from matryoshka_attribution.trainer import TrainResult
        return TrainResult(saved['scores'], [r[3] for r in prefix], [r[1] for r in prefix], prefix, elapsed)
    base_adam = torch.optim.Adam
    holder = {}
    records = []
    external = kwargs.pop('on_step', None)
    start = time.perf_counter()

    class CapturedAdam(base_adam):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            holder['optimizer'] = self
            params = [p for g in self.param_groups for p in g['params']]
            if len(params) != 1 or params[0].numel() != (parameter_count or total):
                raise ValueError('Unexpected upstream optimizer layout.')
            if saved:
                with torch.no_grad():
                    params[0].copy_(saved['scores'].to(params[0]))
                self.load_state_dict(saved['optimizer'])
                restore_rng(saved['rng'])

        def step(self, *args, **kw):
            score = self.param_groups[0]['params'][0]
            if on_update:
                on_update(float(score.grad.detach().double().norm()))
            if inspect_step:
                inspect_step(score, self)
            return super().step(*args, **kw)

    def callback(step, k, loss, scores):
        step += offset
        records.append((step, float(k), float(k)/total, float(loss)))
        if external:
            external(step, k, loss, scores)
        if (step+1) % interval == 0 or step+1 == target:
            if scores.is_cuda:
                torch.cuda.synchronize()
            path = folder/f'step_{step+1:06d}.pt'
            state = {'identity': identity, 'completed': step+1,
                'scores': scores.detach().cpu().clone(), 'optimizer': holder['optimizer'].state_dict(),
                'rng': rng_state(), 'train_log': prefix+records,
                'training_seconds': elapsed+time.perf_counter()-start}
            temp = path.with_suffix('.tmp')
            torch.save(state, temp)
            temp.replace(path)
    with patch.object(torch.optim, 'Adam', CapturedAdam):
        result = original(total, loss_fn, steps=target-offset, on_step=callback, **kwargs)
    result.train_log = prefix+records
    result.loss_log = [r[3] for r in result.train_log]
    result.k_log = [r[1] for r in result.train_log]
    result.train_time_s += elapsed
    return result
