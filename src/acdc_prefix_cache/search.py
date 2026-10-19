"""Exact one-edge proposals; checkpoint both retained identity and decision trace."""
import json
from pathlib import Path
import time

from .engine import sync
from .graph import digest


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8'); temp.replace(path)


def exact_acdc(executor, order, threshold, checkpoint=None, stop_after=None, checkpoint_every=1):
    """MIB-compatible signed loss increase, like ACDC's abs_value_threshold=False.

    Fixed real-edge traversal; no group deletion, connectivity shortcuts or
    automatic cascading deletion. A disconnected sender's outgoing edge is
    still tested. Pruned MIB graphs are an independent output-level oracle.
    """
    order = list(order)
    if len(order)!=len(set(order)) or set(order)-executor.graph.edges.keys():
        raise ValueError('Candidate traversal contains duplicates or unknown edges.')
    if threshold<=0 or checkpoint_every<1: raise ValueError('Use a positive threshold and checkpoint interval.')
    identity = digest({'executor':executor.identity,'order':order,'threshold':threshold,
                       'decision_rule':'signed_loss_increase_strict_less','checkpoint_every':checkpoint_every})
    checkpoint = Path(checkpoint) if checkpoint else None
    rows=[]; removed=set(); current=float(executor.baseline.mean()); prior_seconds=0.
    if checkpoint and checkpoint.exists():
        state=json.loads(checkpoint.read_text())
        if state['identity']!=identity: raise ValueError('Checkpoint protocol/data/graph differs.')
        rows=state['trace']; removed=set(state['removed']); current=state['current_loss']
        prior_seconds=state['search_seconds']
        if [r['edge'] for r in rows]!=order[:len(rows)]: raise ValueError('Checkpoint traversal differs.')
        if executor.dependencies.circuit_hash(removed)!=state['circuit_hash']:
            raise ValueError('Checkpoint retained circuit is corrupt.')
        if rows:
            actual = float(executor.evaluate(removed).mean())
            if actual!=current: raise ValueError('Checkpoint loss does not replay bitwise.')
    sync(executor.model); started=time.perf_counter()
    for step in range(len(rows),len(order)):
        if stop_after is not None and step>=stop_after: break
        edge=order[step]; before=executor.dependencies.circuit_hash(removed)
        proposed=float(executor.evaluate(removed|{edge}).mean())
        delta=proposed-current; accepted=delta<threshold
        if accepted: removed.add(edge); current=proposed
        rows.append({'step':step,'edge':edge,'loss':proposed,'loss_change':delta,
            'removed':accepted,'current_loss':current,'state_before_hash':before,
            'state_after_hash':executor.dependencies.circuit_hash(removed),
            'removed_count':len(removed),'earliest_affected_layer':executor.evaluations[-1]['earliest_affected_layer']})
        if checkpoint and ((step+1)%checkpoint_every==0 or len(rows)==len(order) or len(rows)==stop_after):
            sync(executor.model)
            atomic_json(checkpoint,{'identity':identity,'trace':rows,'removed':sorted(removed),
                'current_loss':current,'circuit_hash':executor.dependencies.circuit_hash(removed),
                'search_seconds':prior_seconds+time.perf_counter()-started,
                'complete':len(rows)==len(order)})
    sync(executor.model)
    return {'trace':rows,'removed':sorted(removed),'retained':executor.dependencies.retained(removed),
        'circuit_hash':executor.dependencies.circuit_hash(removed),'current_loss':current,
        'search_seconds':prior_seconds+time.perf_counter()-started,'candidate_checks':len(rows),
        'partial_graph_trajectory':len(order)!=len(executor.graph.edges)}
