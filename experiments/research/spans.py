"""Unlabeled walk contexts and exact contiguous segmentation; not a correctness score."""
import numpy as np

from experiments.research.segmentation import partition_scores

def contexts(history):
    """Symmetric walk contexts at horizons 1, 2, 4, 8; not causal paths.

    Isolated nodes receive a unit self-loop, matching the archived prototype.
    This fallback has not been validated as a semantic-state estimator.
    """
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 2 or history.shape[0] != history.shape[1] or (not len(history)) or (not np.isfinite(history).all()) or np.any(history < 0):
        raise ValueError('Expected a finite, nonnegative, nonempty square history')
    weights = history + history.T
    degree = weights.sum(1)
    isolated = degree == 0
    weights[np.flatnonzero(isolated), np.flatnonzero(isolated)] = 1.0
    degree = weights.sum(1)
    pi = degree / degree.sum()
    transition = weights / degree[:, None]
    np.testing.assert_allclose(pi @ transition, pi, atol=1e-12, rtol=0)
    power = np.eye(len(history))
    total = np.zeros_like(history)
    result = {}
    for horizon in range(1, 9):
        power = power @ transition
        total += power
        if horizon in [1, 2, 4, 8]:
            context = total / horizon
            flow = pi[:, None] * context - np.outer(pi, pi)
            assert np.max(abs(flow - flow.T)) < 1e-12
            result[horizon] = (context, flow)
    return (pi, result, int(isolated.sum()))

def interval_scores(matrix):
    prefix = np.pad(matrix, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    n = len(matrix)
    scores = np.full((n + 1, n + 1), -np.inf)
    for b in range(1, n + 1):
        a = np.arange(b)
        scores[a, b] = prefix[b, b] - prefix[a, b] - prefix[b, a] + prefix[a, a]
    return scores

def partition(flow):
    """Exact contiguous flow partition, with deterministic fewer-span ties."""
    return partition_scores(interval_scores(flow))


def quality(flow, cuts):
    return float(sum((flow[a:b, a:b].sum() for (a, b) in zip(cuts[:-1], cuts[1:]))))


def constrained_partition(flow, atoms):
    """Optimize the same objective after summing indivisible token blocks."""
    atoms = np.asarray(atoms)
    if atoms.ndim != 1 or len(atoms) < 2 or atoms.dtype.kind not in 'iu' or (atoms[0] != 0) or (atoms[-1] != len(flow)) or np.any(np.diff(atoms) <= 0):
        raise ValueError('Atoms must be increasing integer endpoints from 0 to n')
    blocked = np.add.reduceat(np.add.reduceat(flow, atoms[:-1], axis=0), atoms[:-1], axis=1)
    (cuts, _) = partition(blocked)
    return atoms[cuts]
