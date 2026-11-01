"""Shared inference agrees with independent finite latent-state calculations."""

import itertools

import numpy as np
import pytest
from scipy.special import betaln, logsumexp

from myproject.pooling import cavity_scores, grouped_scores
from myproject.shared import SharedDetector


@pytest.mark.parametrize("alpha,beta", [(1., 1.), (.5, 2.), (3., .7)])
def test_shared_log_odds_match_complete_assignment_enumeration(alpha, beta):
    values = np.array([-1.3, .2, 2.4, -.5])
    states = np.array(list(itertools.product((0, 1), repeat=len(values))))
    count = states.sum(axis=1)
    joint = (states @ values + betaln(alpha + count, beta + len(values) - count)
             - betaln(alpha, beta))
    expected = np.array([logsumexp(joint[states[:, i] == 1])
                         - logsumexp(joint[states[:, i] == 0])
                         for i in range(len(values))])
    result = cavity_scores(values, alpha=alpha, beta=beta)
    np.testing.assert_allclose(result['score'], expected, rtol=0, atol=2e-14)
    np.testing.assert_allclose(result['log_evidence'], logsumexp(joint), atol=2e-14)


def test_shared_state_never_crosses_answer_or_span_boundaries():
    values = np.array([-.7, .3, 2., -.8, 1.2, .4])
    answers = np.array([0, 0, 0, 0, 1, 1])
    units = np.array([0, 0, 1, 1, 0, 0])
    expected = np.concatenate([cavity_scores(values[start:start + 2])['score']
                               for start in (0, 2, 4)])
    np.testing.assert_array_equal(grouped_scores(values, answers, units)['score'], expected)


def test_shared_uses_full_source_gap_without_local_or_route_fields():
    values = np.array([-2., .2, 1.7])
    answers, units = np.zeros(3, dtype=int), np.zeros(3, dtype=int)
    detector = SharedDetector()
    expected = cavity_scores(values)['score']
    np.testing.assert_array_equal(
        detector.score_full_gap(values, answers, units), expected)
    pack = {'target': np.arange(3), 'delta': np.column_stack([values * 100, values]),
            'answer_index': answers, 'unit_index': units}
    np.testing.assert_array_equal(detector.score(pack), expected)


def test_shared_configuration_round_trip_and_rejects_nonfinite_gap(tmp_path):
    path = tmp_path / 'shared.npz'
    detector = SharedDetector('Summary', method='witness')
    detector.save(path)
    restored = SharedDetector.load(path)
    assert (restored.task, restored.method, restored.segmentation) == (
        'Summary', 'witness', 'punct')
    with pytest.raises(ValueError, match='finite'):
        restored.score_full_gap([np.nan], [0], [0])
