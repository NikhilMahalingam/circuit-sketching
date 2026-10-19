"""Conservative dependency proofs on the actual MIB sender/receiver DAG."""
from dataclasses import dataclass
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class PrefixPlan:
    start_layer: int
    removed_count: int
    affected_nodes: frozenset
    reusable_nodes: frozenset


class Dependencies:
    def __init__(self, graph):
        self.graph = graph
        self.layers = graph.cfg['n_layers']
        self.phase = {}
        self.children = {name: set() for name in graph.nodes}
        for name, node in graph.nodes.items():
            if name == 'input': phase = -1
            elif name == 'logits': phase = 2*self.layers
            elif name.startswith('a'): phase = 2*node.layer
            elif name.startswith('m'): phase = 2*node.layer+1
            else: raise ValueError(f'Unsupported MIB node: {name}')
            self.phase[name] = phase
        for edge in graph.edges.values():
            if self.phase[edge.parent.name] >= self.phase[edge.child.name]:
                raise ValueError(f'Non-topological edge: {edge.name}')
            self.children[edge.parent.name].add(edge.child.name)
        # Descendants use the FULL graph, not just surviving paths. This is
        # conservative even when a receiver has become disconnected.
        self.descendants = {}
        for name in sorted(self.phase, key=self.phase.get, reverse=True):
            found = {name}
            for child in self.children[name]: found.update(self.descendants[child])
            self.descendants[name] = frozenset(found)
        self.order = tuple(sorted(graph.edges, key=lambda name: (
            -self.phase[graph.edges[name].child.name],
            -int(graph.edges[name].matrix_index[0]), name)))
        self.identity = digest([(n, graph.edges[n].parent.name, graph.edges[n].child.name,
                                 graph.edges[n].hook, graph.edges[n].matrix_index) for n in self.order])

    def plan(self, removed, requested_start=None):
        removed = frozenset(removed)
        unknown = removed - self.graph.edges.keys()
        if unknown: raise ValueError(f'Unknown edges: {sorted(unknown)}')
        affected = set()
        for name in removed: affected.update(self.descendants[self.graph.edges[name].child.name])
        # The output receiver is hook_resid_post of the final block, so keep
        # that block in the suffix; skipping all blocks would skip its hook.
        start = min((self.graph.edges[n].child.layer for n in removed), default=0)
        if requested_start is not None:
            if not 0 <= requested_start < self.layers: raise ValueError('Invalid prefix boundary.')
            start = requested_start
        prefix = frozenset(n for n, phase in self.phase.items() if phase < 2*start)
        dirty = prefix & affected
        skipped_hooks = [n for n in removed if self.graph.edges[n].child.layer < start]
        if dirty or skipped_hooks:
            raise ValueError(f'Unsafe clean-prefix reuse at block {start}: changed nodes {sorted(dirty)}; '
                             f'skipped receivers {skipped_hooks}')
        return PrefixPlan(start, len(removed), frozenset(affected), prefix)

    def retained(self, removed):
        return sorted(self.graph.edges.keys() - set(removed))

    def circuit_hash(self, removed):
        return digest(self.retained(removed))
