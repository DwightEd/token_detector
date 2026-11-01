"""History-marginal matching and exact-complexity boundary controls."""
from collections import Counter
import numpy as np
TOL = 1e-07

def maxent(history, cap=20000):
    """IPF match receiver, sender and lag marginals on strictly causal edges."""
    history = np.asarray(history, dtype=np.float64)
    if history.ndim != 2 or history.shape[0] != history.shape[1] or len(history) < 2 or (not np.isfinite(history).all()) or np.any(history < 0) or np.any(np.triu(history)) or (history.sum() <= 0) or (cap < 25):
        raise ValueError('Expected positive-mass strictly lower-triangular history')
    n = len(history)
    (i, j) = np.tril_indices(n, -1)
    d = i - j
    total = float(history.sum())
    observed = history[i, j] / total
    targets = [np.bincount(index, weights=observed, minlength=n) for index in [i, j, d]]
    value = np.ones(len(i)) / len(i)
    errors = [np.inf] * 3
    for step in range(1, cap + 1):
        for (index, target) in zip([i, j, d], targets):
            actual = np.bincount(index, weights=value, minlength=n)
            ratio = np.divide(target, actual, out=np.zeros(n), where=actual > 0)
            value *= ratio[index]
        if step % 25 == 0:
            errors = [float(np.abs(np.bincount(index, weights=value, minlength=n) - target).sum()) for (index, target) in zip([i, j, d], targets)]
            if max(errors) <= TOL:
                break
    result = np.zeros_like(history)
    result[i, j] = value * total
    return (result, {'cycles': step, 'normalized_L1_errors': errors, 'matched': max(errors) <= TOL, 'relative_matrix_L1_change': float(abs(result - history).sum() / total)})

def legal_orders(cuts, allowed):
    """Enumerate distinct legal length orders without factorial duplicates."""
    cuts = np.asarray(cuts)
    allowed = np.asarray(allowed, dtype=bool)
    if cuts.ndim != 1 or len(cuts) < 2 or cuts.dtype.kind not in 'iu' or (cuts[0] != 0) or (cuts[-1] != len(allowed) - 1) or np.any(np.diff(cuts) <= 0):
        raise ValueError('Cuts must cover the allowed token axis exactly')
    remaining = Counter(np.diff(cuts).tolist())

    def visit(prefix):
        if len(prefix) == len(cuts):
            yield np.asarray(prefix)
            return
        for length in sorted(remaining):
            end = prefix[-1] + length
            if remaining[length] and allowed[end]:
                remaining[length] -= 1
                yield from visit(prefix + [end])
                remaining[length] += 1
    if allowed[0]:
        yield from visit([0])
