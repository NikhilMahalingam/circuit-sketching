"""Resume the upstream loop using its Adam state and RNGs, without rewriting it."""
from pathlib import Path
import random
import time
from unittest.mock import patch

import numpy as np
import torch


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def learn_checkpointed(original, total, loss_fn, *, checkpoint_dir, identity,
                       interval=50, meter=None, on_checkpoint=None, **kwargs):
    """Call official learn_scores; restore only scores/Adam/RNG and loop offset.

    Default MAttr has no step-dependent LR and its loop index is used only for
    logging. Calling it for the remaining steps therefore preserves its update
    sequence. Tests compare uninterrupted and resumed runs bit-for-bit.
    """
    if kwargs.get('optimizer', 'adam') != 'adam' or kwargs.get('variant', 'topk') != 'topk':
        raise ValueError('Resume adapter is restricted to the requested official Adam/topk recipe.')
    folder = Path(checkpoint_dir); folder.mkdir(parents=True, exist_ok=True)
    candidates = sorted(folder.glob('step_*.pt'))
    saved = torch.load(candidates[-1], map_location='cpu', weights_only=False) if candidates else None
    if saved and saved['identity'] != identity:
        raise ValueError('Checkpoint configuration/source identity differs; use a new run directory.')
    start_step = saved['completed_steps'] if saved else 0
    target_steps = int(kwargs.pop('steps'))
    if start_step > target_steps:
        raise ValueError('Checkpoint is beyond requested optimization budget.')
    if saved and start_step == target_steps:
        from matryoshka_attribution.trainer import TrainResult
        return TrainResult(scores=saved['scores'], loss_log=saved['loss_log'],
                           k_log=saved['k_log'], train_log=saved['train_log'],
                           train_time_s=saved['training_seconds'])
    prefix_log = saved['train_log'] if saved else []
    prior_seconds = saved['training_seconds'] if saved else 0.
    original_adam = torch.optim.Adam
    holder = {}

    class InstrumentedAdam(original_adam):
        def __init__(self, *args, **adam_kwargs):
            super().__init__(*args, **adam_kwargs)
            holder['optimizer'] = self
            if saved:
                parameters = [p for group in self.param_groups for p in group['params']]
                if len(parameters) != 1 or parameters[0].numel() != total:
                    raise ValueError('Unexpected official optimizer parameter layout.')
                with torch.no_grad():
                    parameters[0].copy_(saved['scores'].to(parameters[0]))
                self.load_state_dict(saved['optimizer'])
                restore_rng(saved['rng'])

        def step(self, *args, **step_kwargs):
            result = super().step(*args, **step_kwargs)
            if meter:
                meter.counts['optimizer_steps'] += 1
            return result

    external_callback = kwargs.pop('on_step', None)
    new_log = []
    start_time = time.perf_counter()

    def callback(local_step, k, loss, scores):
        global_step = local_step + start_step
        new_log.append((global_step, float(k), float(k)/total, float(loss)))
        if external_callback:
            external_callback(global_step, k, loss, scores)
        completed = global_step + 1
        if completed % interval == 0 or completed == target_steps:
            if scores.is_cuda:
                torch.cuda.synchronize()
            log = prefix_log + new_log
            checkpoint = {'identity': identity, 'completed_steps': completed,
                'scores': scores.detach().cpu().clone(),
                'optimizer': holder['optimizer'].state_dict(), 'rng': capture_rng(),
                'train_log': list(log), 'loss_log': [r[3] for r in log],
                'k_log': [r[1] for r in log],
                'training_seconds': prior_seconds+time.perf_counter()-start_time}
            # Immutable versioned checkpoint. Ignore an incomplete .tmp after failure.
            path = folder/f'step_{completed:06d}.pt'
            temp = path.with_suffix('.tmp')
            torch.save(checkpoint, temp); temp.replace(path)
            if on_checkpoint:
                on_checkpoint(path, checkpoint)

    with patch.object(torch.optim, 'Adam', InstrumentedAdam):
        result = original(total, loss_fn, steps=target_steps-start_step, on_step=callback, **kwargs)
    result.train_log = prefix_log + new_log
    result.loss_log = [r[3] for r in result.train_log]
    result.k_log = [r[1] for r in result.train_log]
    result.train_time_s += prior_seconds
    return result
