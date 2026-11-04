"""V3 witness emissions and exact persistent-state inference."""

from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp

from .measure import unit_ranges


def witness_emissions(values):
    """Log likelihood ratio of (p_source + p_absent)/2 against p_source."""
    return np.logaddexp(0., np.asarray(values, float)) - np.log(2.)


def transition_grid(pi, c, pi_weights, c_weights):
    """Form weighted state-transition nodes from prevalence and persistence priors."""
    pi, c = np.meshgrid(pi, c, indexing="ij")
    pi, c = pi.ravel(), c.ravel()
    weights = np.log(np.multiply.outer(pi_weights, c_weights).ravel())
    transition = np.array([(1-pi)+c*pi, (1-c)*pi,
                           (1-c)*(1-pi), c+(1-c)*pi])
    return pi, c, weights, np.log(transition)


@lru_cache(maxsize=128)
def uniform_grid(length):
    """Exact positive quadrature for uniform prevalence and persistence priors."""
    nodes, weights = leggauss(length // 2 + 1)
    nodes, weights = (nodes + 1) / 2, weights / 2
    return transition_grid(nodes, nodes, weights, weights)


def state_marginals(values, pi, weights, transition):
    """Integrate unnormalized two-state joints; return token odds and node evidence.

    Inputs are validated emissions and log transition weights. Experiments reuse
    this kernel with their own observations or priors.
    """
    t00, t01, t10, t11 = transition
    forward = np.empty((len(values), len(pi), 2))
    forward[0, :, 0], forward[0, :, 1] = np.log1p(-pi), np.log(pi) + values[0]
    for t in range(1, len(values)):
        previous = forward[t-1]
        forward[t, :, 0] = np.logaddexp(previous[:, 0]+t00, previous[:, 1]+t10)
        forward[t, :, 1] = values[t]+np.logaddexp(previous[:, 0]+t01, previous[:, 1]+t11)
    node_evidence = np.logaddexp(forward[-1, :, 0], forward[-1, :, 1])
    backward, score = np.zeros((len(pi), 2)), np.empty(len(values))
    for t in range(len(values)-1, -1, -1):
        joints = forward[t] + backward + weights[:, None]
        score[t] = logsumexp(joints[:, 1]) - logsumexp(joints[:, 0])
        next_zero = np.logaddexp(t00+backward[:, 0], t01+values[t]+backward[:, 1])
        next_one = np.logaddexp(t10+backward[:, 0], t11+values[t]+backward[:, 1])
        backward = np.column_stack([next_zero, next_one])
    return score, node_evidence


class WitnessDetector:
    """V3: fixed equal expert mixture, with integrated prevalence and persistence."""

    def score(self, pack: dict) -> np.ndarray:
        return self.score_full_gap(pack["full_gap"], pack["answer_index"], pack["unit_index"])

    def score_full_gap(self, values, answer_index, unit_index) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        answers, units = np.asarray(answer_index), np.asarray(unit_index)
        if (values.ndim != 1 or answers.shape != values.shape or units.shape != values.shape
                or not np.isfinite(values).all()):
            raise ValueError("Matching finite full-gap and original token axes required")
        score = np.empty(len(values))
        if not len(values):
            return score
        starts, stops = unit_ranges(answers, units)
        for start, stop in zip(starts, stops, strict=True):
            emission = witness_emissions(values[start:stop])
            pi, _, weights, transition = uniform_grid(len(emission))
            score[start:stop], _ = state_marginals(emission, pi, weights, transition)
        return score
