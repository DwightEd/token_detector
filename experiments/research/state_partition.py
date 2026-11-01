"""Contiguous states selected by symmetric, leave-one-row-out routing prediction."""

import numpy as np

from myproject.segmentation import _inputs, background, partition_scores


def residual_gains(fit, hold, available):
    """Exact masked squared-error gain, using one zero-state pseudo-observation.

    Every row is predicted from the other rows of its candidate interval. The
    paired views remain correlated; this is not an untouched validation score.
    """
    fit, hold = np.asarray(fit, float), np.asarray(hold, float)
    mask = np.asarray(available, bool)
    if (fit.ndim != 2 or not len(fit) or not fit.shape[1]
            or fit.shape != hold.shape or fit.shape != mask.shape
            or not np.isfinite(fit).all() or not np.isfinite(hold).all()
            or np.any(fit[~mask] != 0) or np.any(hold[~mask] != 0)):
        raise ValueError('Expected finite paired routing residuals on their causal support')

    def cumulative(values):
        return np.concatenate([np.zeros_like(values[:1]), values.cumsum(axis=0)])

    f, h, m = map(cumulative, (fit, hold, mask.astype(float)))
    cross = cumulative(np.sum(fit * hold, axis=1))
    norm = cumulative(np.sum(fit**2 + hold**2, axis=1))
    n = len(fit)
    gains = np.full((n + 1, n + 1), -np.inf)
    for first in range(n):
        total_f, total_h = f[first+1:] - f[first], h[first+1:] - h[first]
        support = m[first+1:] - m[first]
        count = np.arange(1, n-first+1)
        interaction = np.sum(total_f * total_h, axis=1) - (cross[first+1:] - cross[first])
        cost = np.sum((support-2) * (total_f**2 + total_h**2), axis=1)
        gains[first, first+1:] = (2 * interaction/count
            - (cost + norm[first+1:] - norm[first])/(2 * count**2))
        gains[first, first+1] = 0.
    return gains


def routing_partition(fit, hold, available, atoms):
    """Fit dependency boundaries; the downstream token observations stay intact."""
    residuals = []
    for profiles in [fit, hold]:
        values, mask = _inputs(profiles, available)
        base = np.where(mask, background(values), 0.)
        base /= base.sum(1, keepdims=True)
        residuals.append(np.sqrt(values) - np.sqrt(base))
    gains = residual_gains(*residuals, available)
    cuts, objective = partition_scores(gains, atoms)
    return {'cuts': cuts, 'objective': objective}
