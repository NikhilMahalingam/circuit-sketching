"""Opt-in H1/H2 experiments. Validation prompts are used only for final assessment."""
import time

import numpy as np
import pandas as pd
import torch

from .core import nystrom_sketch


def local_estimator(experiment, state, rank, limit, seed, ledger=None):
    """Local Taylor gradient and randomized HVP Hessian sketch on train data."""
    z = torch.tensor(state, device=experiment.model.cfg.device,
                     dtype=experiment.model.cfg.dtype).requires_grad_(True)
    function = lambda point: experiment.continuous(point, 'train', limit=limit).mean()
    start = time.perf_counter()
    value = function(z)
    gradient = torch.autograd.grad(value, z)[0].detach()
    if ledger is not None:
        ledger.backward_calls += 1
    if rank:
        hessian, _, _ = nystrom_sketch(function, z.detach(), rank, seed)
        if ledger is not None:
            ledger.backward_calls += 2*rank
            ledger.hvp_calls = getattr(ledger, 'hvp_calls', 0) + rank
        curvature = torch.diag(hessian).detach()
    else:
        curvature = torch.zeros_like(gradient)
    if experiment.ledger is not None:
        experiment.ledger.sync()
    return {'value': float(value.detach()), 'gradient': gradient.cpu().numpy(),
            'curvature': curvature.cpu().numpy(), 'rank': rank,
            'estimator_seconds': time.perf_counter()-start, 'hvp_operations':rank}


def collect_h1(experiment, thresholds, ranks, train_limit, seed, ledger=None):
    """Record all intact-candidate effects along exact training trajectories.

    Train examples construct estimators and trajectories. Validation effects
    are separate held-out labels; they never influence pruning or calibration.
    """
    rows = []
    n = len(experiment.edges)
    for threshold in thresholds:
        state = [0]*n
        for step in range(n):
            intact = [i for i, bit in enumerate(state) if not bit]
            estimators = {rank: local_estimator(experiment,state,rank,train_limit,
                                                 seed+step+int(1000*threshold),ledger) for rank in ranks}
            baseline_train = experiment.evaluate_state(state,'train').mean()
            baseline_cal = experiment.evaluate_state(state,'calibration').mean()
            baseline_val = experiment.evaluate_state(state,'validation').mean()
            for i in intact:
                next_state = state.copy(); next_state[i] = 1
                train_effect = experiment.evaluate_state(next_state,'train').mean()-baseline_train
                cal_effect = experiment.evaluate_state(next_state,'calibration').mean()-baseline_cal
                val_effect = experiment.evaluate_state(next_state,'validation').mean()-baseline_val
                for rank, est in estimators.items():
                    first = float(est['gradient'][i])
                    second = first + 0.5*float(est['curvature'][i])
                    rows.append({'threshold':threshold,'step':step,'state':''.join(map(str,state)),
                                 'removed_edges':sum(state),'candidate_index':i,
                                 'edge':experiment.edges[i].name,'rank':rank,
                                 'true_train_effect':train_effect,'true_calibration_effect':cal_effect,
                                 'true_validation_effect':val_effect,
                                 'first_order_prediction':first,'second_order_prediction':second,
                                 'first_order_error_validation':abs(first-val_effect),
                                 'second_order_error_validation':abs(second-val_effect),
                                 'first_order_decision_agrees':(abs(first)<threshold)==(abs(val_effect)<threshold),
                                 'second_order_decision_agrees':(abs(second)<threshold)==(abs(val_effect)<threshold),
                                 'estimator_seconds':est['estimator_seconds'],
                                 'hvp_operations':est['hvp_operations'], 'train_examples':train_limit})
            # Exact ACDC decision from train examples only, in fixed order.
            trial = state.copy(); trial[step] = 1
            effect = experiment.evaluate_state(trial,'train').mean()-baseline_train
            if abs(effect) < threshold:
                state = trial
    return pd.DataFrame(rows)


def rank_summary(predictions):
    from scipy.stats import pearsonr, spearmanr
    rows=[]
    for rank, group in predictions.groupby('rank'):
        true = group.true_validation_effect.to_numpy()
        for label, column in [('first_order','first_order_prediction'),
                              ('second_order','second_order_prediction')]:
            pred = group[column].to_numpy()
            rows.append({'rank':rank,'method':label,'n_decisions':len(group),
                         'mae':float(np.mean(abs(pred-true))),
                         'pearson':float(pearsonr(pred,true).statistic) if np.std(pred)>0 and np.std(true)>0 else np.nan,
                         'spearman':float(spearmanr(pred,true).statistic) if np.std(pred)>0 and np.std(true)>0 else np.nan,
                         'decision_agreement':float(np.mean((abs(pred)<group.threshold.to_numpy())==(abs(true)<group.threshold.to_numpy()))),
                         'false_prune_rate':float(np.mean((abs(pred)<group.threshold.to_numpy()) & (abs(true)>=group.threshold.to_numpy()))),
                         'false_retain_rate':float(np.mean((abs(pred)>=group.threshold.to_numpy()) & (abs(true)<group.threshold.to_numpy())))})
    return pd.DataFrame(rows)


def calibrate_bound(predictions, method='second_order', quantile=0.9):
    """Calibrate on withheld training prompts, never validation residuals."""
    key = method + '_prediction'
    return float(np.quantile(abs(predictions[key]-predictions.true_calibration_effect),quantile))


def hybrid_acdc(experiment, threshold, rank, train_limit, bound, refresh_after,
                verify_high_impact=True, seed=0, ledger=None):
    """Confidence-gated restricted sequential pruning using training prompts."""
    n=len(experiment.edges)
    state=[0]*n
    accepted_since_refresh=0
    estimator=None
    rows=[]
    exact_queries=0
    for i in range(n):
        if estimator is None or (isinstance(refresh_after,int) and accepted_since_refresh>=refresh_after):
            estimator=local_estimator(experiment,state,rank,train_limit,seed+i,ledger)
            accepted_since_refresh=0
        prediction=float(estimator['gradient'][i]+0.5*estimator['curvature'][i])
        # Error bound is empirical, not a guaranteed confidence interval.
        uncertain=abs(abs(prediction)-threshold)<=bound
        high_impact=verify_high_impact and abs(prediction)>=threshold+bound
        verify=uncertain or high_impact
        if verify:
            effect=float(experiment.exact_candidate_effect(state,i,'train').mean())
            exact_queries+=1
            remove=abs(effect)<threshold
        else:
            effect=np.nan
            remove=abs(prediction)<threshold
        if remove:
            state[i]=1
            accepted_since_refresh+=1
        if refresh_after == 'error' and verify and abs(effect-prediction)>bound:
            estimator=None
        rows.append({'step':i,'edge':experiment.edges[i].name,'predicted_effect':prediction,
                     'verified_effect':effect,'exact_verification':verify,'pruned':remove,
                     'state_after':''.join(map(str,state)),'empirical_error_bound':bound,
                     'remaining_candidate_edges':n-sum(state)})
    return state,pd.DataFrame(rows),exact_queries
