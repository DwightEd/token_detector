"""Deterministic directed dependency states and word-external contiguous partitions."""

import re

import numpy as np


def partition_scores(scores, atoms=None, cut_penalty=0.0):
    """Optimize arbitrary interval scores on legal endpoints; charge internal cuts."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or len(scores) < 2:
        raise ValueError('Expected an (n+1)-square interval score matrix')
    n = len(scores) - 1
    atoms = np.arange(n + 1) if atoms is None else np.asarray(atoms)
    if (atoms.ndim != 1 or len(atoms) < 2 or atoms.dtype.kind not in 'iu'
            or atoms[0] != 0 or atoms[-1] != n or np.any(np.diff(atoms) <= 0)
            or not np.isfinite(cut_penalty) or cut_penalty < 0):
        raise ValueError('Expected complete increasing endpoints and nonnegative cut cost')
    best = np.full(n + 1, -np.inf)
    count = np.zeros(n + 1, dtype=int)
    parent = np.zeros(n + 1, dtype=int)
    best[0] = 0.0
    for index, b in enumerate(atoms[1:], 1):
        previous = atoms[:index]
        candidate = best[previous] + scores[previous, b] - cut_penalty * (previous > 0)
        if not np.isfinite(candidate).any():
            raise ValueError('No finite partition reaches an allowed endpoint')
        tied = np.flatnonzero(candidate >= candidate.max() - 1e-12)
        chosen = min(tied, key=lambda k: (count[previous[k]], previous[k]))
        a = previous[chosen]
        (best[b], count[b], parent[b]) = (candidate[chosen], count[a] + 1, a)
    cuts = [n]
    while cuts[-1] > 0:
        cuts.append(int(parent[cuts[-1]]))
    return (np.array(cuts[::-1]), float(best[n]))

def legal_atoms(response):
    """Allow word-external, nonoverlapping cuts on the original token axis."""
    text = response['text']
    offsets = np.array(response['offsets'])
    interior = np.zeros(len(text) + 1, dtype=bool)
    for word in re.finditer("\\w+(?:['’]\\w+)*", text):
        interior[word.start() + 1:word.end()] = True
    allowed = np.ones(len(offsets) + 1, dtype=bool)
    for cut in range(1, len(offsets)):
        position = int(offsets[cut, 0])
        allowed[cut] = not (offsets[cut - 1, 1] > position or interior[position])
    return (np.flatnonzero(allowed), allowed)


def _inputs(profiles, available):
    p = np.asarray(profiles, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    if (p.ndim != 2 or not len(p) or p.shape != mask.shape or not p.shape[1]
            or not np.isfinite(p).all() or np.any(p < 0) or np.any(p[~mask] != 0)
            or not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0)):
        raise ValueError('Expected normalized nonnegative routing on its causal support')
    p = p / p.sum(1, keepdims=True)
    return p, mask

def background(profiles):
    """One total uniform routing event smooths the empirical destination prior."""
    n, d = profiles.shape
    return (profiles.sum(0) + 1 / d) / (n + 1)

def _prediction(counts, base, mask):
    q = np.where(mask, base + counts, 0.)
    return q / q.sum()

def interval_gains(profiles, available):
    """Expected prequential log-score gain over the common restricted background."""
    p, mask = _inputs(profiles, available)
    base = background(p)
    baseline = base[None, :] * mask
    baseline /= baseline.sum(1, keepdims=True)
    scores = np.full((len(p) + 1, len(p) + 1), -np.inf)
    for first in range(len(p)):
        counts, gain = np.zeros(p.shape[1]), 0.
        for last in range(first, len(p)):
            q = _prediction(counts, base, mask[last])
            active = p[last] > 0
            gain += p[last, active] @ (np.log(q[active]) - np.log(baseline[last, active]))
            scores[first, last + 1] = gain
            counts += p[last]
    return scores

def fitted_profiles(profiles, available, cuts):
    """Keep shared counts, token-specific predictions and all individual residuals."""
    p, mask = _inputs(profiles, available)
    cuts = np.asarray(cuts)
    if (cuts.ndim != 1 or cuts.dtype.kind not in 'iu' or cuts[0] != 0
            or cuts[-1] != len(p) or np.any(np.diff(cuts) <= 0)):
        raise ValueError('Cuts must partition the original rows')
    base = background(p)
    predicted = np.zeros_like(p)
    states = []
    for first, last in zip(cuts[:-1], cuts[1:]):
        counts = np.zeros(p.shape[1])
        for row in range(first, last):
            predicted[row] = _prediction(counts, base, mask[row])
            counts += p[row]
        states.append((base + counts) / (1 + last - first))
    return {'predicted': predicted, 'residual': p - predicted,
            'shared': np.asarray(states), 'background': base}

def segment(profiles, available, atoms):
    scores = interval_gains(profiles, available)
    penalty = float(np.log(2 * max(len(atoms) - 2, 1)))
    cuts, objective = partition_scores(scores, atoms, penalty)
    return {'cuts': cuts, 'objective': objective, 'cut_penalty': penalty, 'interval_gains': scores,
            **fitted_profiles(profiles, available, cuts)}

def predictive_gain(fit, hold, available, cuts):
    """Transfer fit-view predictive states; no held-view parameter estimation."""
    fitted = fitted_profiles(fit, available, cuts)
    p, mask = _inputs(hold, available)
    base = fitted['background'][None, :] * mask
    base /= base.sum(1, keepdims=True)
    active = p > 0
    return float(np.sum(p[active] * (np.log(fitted['predicted'][active]) - np.log(base[active]))) / len(p))


def predictive_segment(fit, hold, available, atoms):
    """Choose coarse/zero-cost partitions by routing prediction, without error labels.

    The head views are correlated and the second is used for selection; its gain
    is consequently not an untouched validation estimate or a correctness score.
    """
    scores = interval_gains(fit, available)
    penalties = [float(np.log(2 * max(len(atoms) - 2, 1))), 0.]
    candidates = [partition_scores(scores, atoms, penalty)[0] for penalty in penalties]
    gains = [predictive_gain(fit, hold, available, cuts) for cuts in candidates]
    chosen = int(gains[1] > gains[0])
    return {'cuts': candidates[chosen], 'cut_penalty': penalties[chosen],
            'selection_gains': np.asarray(gains),
            **fitted_profiles(fit, available, candidates[chosen])}
