"""Conditional signed span interactions; no truth labels or trained LM parameters."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, xlogy
from .dependence import maxent

class HistoryView:
    """Signed conditional histories with receiver/sender/lag nuisance offsets."""

    def __init__(self, history, offset=None):
        self._bins_cache = {}
        self.h = np.asarray(history, dtype=np.float64)
        if self.h.ndim != 3 or self.h.shape[0] != 2 or self.h.shape[1] != self.h.shape[2] or (self.h.shape[1] < 2) or (not np.isfinite(self.h).all()) or np.any(self.h < 0) or np.any(np.triu(self.h)):
            raise ValueError('Expected two finite nonnegative causal sign channels')
        self.n = self.h.shape[1]
        if offset is None:
            pairs = [maxent(h) for h in self.h]
            self.e = np.stack([p[0] for p in pairs])
            self.matching = [p[1] for p in pairs]
            if not all((p['matched'] for p in self.matching)):
                raise ValueError('Signed nuisance marginal matching failed')
        else:
            self.e = np.asarray(offset, dtype=np.float64)
            if self.e.shape != self.h.shape or not np.isfinite(self.e).all() or np.any(self.e < 0) or np.any(np.triu(self.e)):
                raise ValueError('Offsets must match the signed causal history')
            self.matching = []
        self.mass = self.h.sum(2)
        total = self.mass.sum(0)
        self.weight = np.divide(self.mass, total[None, :], out=np.zeros_like(self.mass), where=total[None, :] > 0)
        self.hp = np.pad(self.h.cumsum(2), ((0, 0), (0, 0), (1, 0)))
        self.ep = np.pad(self.e.cumsum(2), ((0, 0), (0, 0), (1, 0)))
        row_e = self.e.sum(2)
        q = np.divide(self.e, row_e[:, :, None], out=np.zeros_like(self.e), where=row_e[:, :, None] > 0)
        y = np.divide(self.h, self.mass[:, :, None], out=np.zeros_like(self.h), where=self.mass[:, :, None] > 0)
        if np.any((y > 0) & (q <= 0)):
            raise ValueError('Observed conditional support absent in nuisance model')
        self.baseline_ce = float(-np.sum(self.weight[:, :, None] * xlogy(y, q)) / (self.n - 1))

    def bins(self, a, b):
        key = (a, b)
        if key not in self._bins_cache:
            self._bins_cache[key] = self._compute_bins(a, b)
        return self._bins_cache[key]

    def _compute_bins(self, a, b):
        target = np.arange(a, b)
        hin = self.hp[:, target, target] - self.hp[:, target, a]
        hout = self.hp[:, target, a]
        ein = self.ep[:, target, target] - self.ep[:, target, a]
        eout = self.ep[:, target, a]
        active = (ein > 0) & (eout > 0) & (self.mass[:, target] > 0)
        if a == 0:
            active[:] = False
        active[:, 0] = False
        y = np.divide(hin, hin + hout, out=np.zeros_like(hin), where=hin + hout > 0)
        la = np.zeros_like(y)
        ly = np.zeros_like(y)
        la[active] = np.log(ein[active]) - np.log(eout[active])
        if np.any(active & ((hin <= 0) | (hout <= 0))):
            raise ValueError('Boundary token-level MLE: protocol has no smoothing fallback')
        ly[active] = np.log(hin[active]) - np.log(hout[active])
        return (y, la, ly, self.weight[:, target], active)

    def gain(self, a, b, kappa):
        (y, la, _, weight, active) = self.bins(a, b)
        k = np.broadcast_to(kappa, y.shape)
        contribution = k * y - (np.logaddexp(0, la + k) - np.logaddexp(0, la))
        return float(np.sum((weight * contribution)[active]))

def estimate(y, offset, weight):
    if not len(y) or weight.sum() == 0:
        return (0.0, 0.0)

    def score(k):
        return float(np.sum(weight * (expit(offset + k) - y)))
    (lo, hi) = (-1.0, 1.0)
    while score(lo) > 0:
        lo *= 2
        if lo < -1024:
            raise ValueError('No finite lower MLE bracket')
    while score(hi) < 0:
        hi *= 2
        if hi > 1024:
            raise ValueError('No finite upper MLE bracket')
    k = float(brentq(score, lo, hi, xtol=1e-12, rtol=1e-14))
    return (k, score(k))

class SpanInteraction:
    """Fit shared or individual conditional offsets and transfer between views."""

    def __init__(self, fit, hold):
        (self.fit, self.hold) = (fit, hold)
        self._interval_cache = {}

    def interval(self, a, b, model):
        key = (a, b, model)
        if key not in self._interval_cache:
            self._interval_cache[key] = self._compute_interval(a, b, model)
        return self._interval_cache[key]

    def _compute_interval(self, a, b, model):
        (y, la, ly, weight, active) = self.fit.bins(a, b)
        if model == 'M1':
            (k, error) = estimate(y[active], la[active], weight[active])
            parameters = np.full((2, b - a), k)
            k_record = [k]
            errors = [error]
        elif model == 'M2':
            solutions = [estimate(y[s, active[s]], la[s, active[s]], weight[s, active[s]]) for s in [0, 1]]
            parameters = np.repeat(np.array([p[0] for p in solutions])[:, None], b - a, axis=1)
            k_record = [p[0] for p in solutions]
            errors = [p[1] for p in solutions]
        elif model == 'M3':
            parameters = np.where(active, ly - la, 0.0)
            k_record = parameters.tolist()
            errors = []
        else:
            raise ValueError(model)
        return {'fit_gain_sum': self.fit.gain(a, b, parameters), 'hold_gain_sum': self.hold.gain(a, b, parameters), 'parameters': k_record, 'score_equation_residuals': errors, 'identifiable_rows_by_sign': active.sum(1).tolist()}

    def partition(self, cuts, model):
        cuts = np.asarray(cuts)
        if cuts.ndim != 1 or len(cuts) < 2 or cuts.dtype.kind not in 'iu' or (cuts[0] != 0) or (cuts[-1] != self.fit.n) or (self.fit.n != self.hold.n) or np.any(np.diff(cuts) <= 0):
            raise ValueError('Cuts must cover both token-aligned histories')
        rows = [self.interval(int(a), int(b), model) for (a, b) in zip(cuts[:-1], cuts[1:])]
        fit_delta = sum((r['fit_gain_sum'] for r in rows)) / (self.fit.n - 1)
        hold_delta = sum((r['hold_gain_sum'] for r in rows)) / (self.hold.n - 1)
        return {'fit_delta': fit_delta, 'hold_delta': hold_delta, 'fit_ce': self.fit.baseline_ce - fit_delta, 'hold_ce': self.hold.baseline_ce - hold_delta, 'intervals': rows}
