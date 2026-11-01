"""Finite CPU checks of model identities, not factual-detection experiments.

See docs/source-reading/THEORY.md for assumptions and proofs. Enumeration is
an independent oracle and is never used to produce detector scores.
"""

import itertools

import numpy as np
import pytest
from scipy.special import betaln, expit, logsumexp

from myproject.pooling import cavity_scores
from myproject.sequence import persistent_scores


@pytest.mark.parametrize("alpha,beta", [(1., 1.), (.2, 3.), (7., .4)])
def test_exchangeable_pooling_preserves_within_span_order_and_ties(alpha, beta):
    rng = np.random.default_rng(20261102)
    for n in (2, 5, 12):
        values = rng.normal(size=n) * 3
        values[-1] = values[0]
        scores = cavity_scores(values, alpha, beta)['score']
        differences = values[:, None] - values[None, :]
        score_differences = scores[:, None] - scores[None, :]
        strict = np.abs(differences) > 1e-10
        np.testing.assert_array_equal(
            np.sign(score_differences[strict]), np.sign(differences[strict]))
        np.testing.assert_allclose(scores[-1], scores[0], rtol=0, atol=2e-13)


def test_score_jacobian_equals_posterior_covariance_and_has_unit_diagonal():
    values = np.array([-1.7, .4, 2.1, -.3])
    alpha, beta = .6, 1.8
    states = np.array(list(itertools.product((0, 1), repeat=len(values))))
    counts = states.sum(axis=1)
    log_mass = (states @ values + betaln(alpha + counts, beta + len(values) - counts)
                - betaln(alpha, beta))
    probability = np.exp(log_mass - logsumexp(log_mass))
    means = probability @ states
    centered = states - means
    covariance = (centered.T * probability) @ centered
    expected = covariance / (means * (1 - means))[:, None]
    step = 1e-5
    jacobian = np.empty_like(expected)
    for j in range(len(values)):
        offset = np.zeros_like(values)
        offset[j] = step
        jacobian[:, j] = (
            cavity_scores(values + offset, alpha, beta)['score']
            - cavity_scores(values - offset, alpha, beta)['score']) / (2 * step)
    np.testing.assert_allclose(jacobian, expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.diag(jacobian), 1., rtol=0, atol=1e-9)
    assert np.min(jacobian) >= -1e-9
    np.testing.assert_allclose(
        expit(cavity_scores(values, alpha, beta)['score']), means, atol=2e-14)


def test_cavity_excludes_own_evidence_and_exchangeable_model_ignores_order():
    values = np.array([-.8, .2, 1.6, -.4, .9])
    base = cavity_scores(values)
    changed = values.copy()
    changed[2] += 4.3
    other = cavity_scores(changed)
    np.testing.assert_allclose(
        other['cavity_log_odds'][2], base['cavity_log_odds'][2], atol=2e-14)
    permutation = np.array([4, 1, 3, 0, 2])
    permuted = cavity_scores(values[permutation])
    np.testing.assert_allclose(permuted['score'], base['score'][permutation], atol=2e-14)
    np.testing.assert_allclose(permuted['log_evidence'], base['log_evidence'], atol=2e-14)


def test_persistence_zero_recovers_beta_sharing_without_witness_transform():
    values = np.array([-2.2, .3, 1.7, -.6, .4, 2.5])
    exchangeable = cavity_scores(values)
    no_copy = persistent_scores(values, copying=False)
    np.testing.assert_allclose(no_copy['score'], exchangeable['score'], atol=3e-14)
    np.testing.assert_allclose(no_copy['log_evidence'], exchangeable['log_evidence'], atol=3e-14)


def test_preserving_local_order_does_not_preserve_cross_span_order():
    # This explicit pair prevents the stronger, false global-rank conclusion.
    low_context = np.array([-5., -5., .1])
    high_context = np.array([5., 5., 0.])
    assert low_context[-1] > high_context[-1]
    assert cavity_scores(low_context)['score'][-1] < cavity_scores(high_context)['score'][-1]


@pytest.mark.parametrize('witness', [False, True])
def test_integrated_persistence_matches_copy_refresh_assignment_enumeration(witness):
    """Integrate Beta counts over states AND copy bits, without the grid/DP."""
    values = np.array([-.9, .6, 1.7, -.2])
    n = len(values)
    emission = np.log((1 + np.exp(values)) / 2) if witness else values
    states, masses, c_means = [], [], []
    for z in itertools.product((0, 1), repeat=n):
        for copied in itertools.product((0, 1), repeat=n - 1):
            if any(copy and z[t] != z[t - 1] for t, copy in enumerate(copied, 1)):
                continue
            fresh = [z[0]] + [z[t] for t, copy in enumerate(copied, 1) if not copy]
            ones, draws, copies = sum(fresh), len(fresh), sum(copied)
            masses.append(np.dot(z, emission) + betaln(1 + ones, 1 + draws - ones)
                          + betaln(1 + copies, n - copies))
            states.append(z)
            c_means.append((1 + copies) / (n + 1))
    states, masses = np.asarray(states), np.asarray(masses)
    evidence = logsumexp(masses)
    expected = np.array([logsumexp(masses[states[:, i] == 1])
                         - logsumexp(masses[states[:, i] == 0]) for i in range(n)])
    actual = persistent_scores(values, witness=witness)
    np.testing.assert_allclose(actual['score'], expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(actual['log_evidence'], evidence, rtol=0, atol=2e-14)
    np.testing.assert_allclose(actual['persistence_mean'], np.exp(masses - evidence) @ c_means,
                               rtol=0, atol=2e-14)
