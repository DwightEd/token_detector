"""Numerical contracts that survive implementation changes."""

import itertools

import numpy as np

from myproject.detector import Detector, target_window_mean
from myproject.graph import chain_log_odds, smooth_scores
from myproject_experiment.data import select_rows


def test_chain_matches_complete_binary_enumeration():
    emission = np.array([-2., 0., .4, 1.5])
    penalties = np.array([.5, .1, .25])
    states = np.array(list(itertools.product((0, 1), repeat=4)))
    energy = states @ emission - ((states[:, 1:] != states[:, :-1]) * penalties).sum(1)
    weight = np.exp(energy)
    expected = np.array([np.log(weight[states[:, i] == 1].sum()
                                / weight[states[:, i] == 0].sum()) for i in range(4)])
    np.testing.assert_allclose(chain_log_odds(emission, penalties), expected, atol=1e-14)


def test_chain_respects_answer_boundaries_and_zero_strength():
    target = np.array([0, 1, 0, 5])
    answer = np.array([0, 0, 1, 1])
    score = np.array([.1, .4, .7, .9])
    together = smooth_scores(score, target, answer)
    np.testing.assert_array_equal(together[:2], smooth_scores(score[:2], target[:2], answer[:2]))
    np.testing.assert_array_equal(together[2:], smooth_scores(score[2:], target[2:], answer[2:]))
    np.testing.assert_array_equal(smooth_scores(score, target, answer, 0), score)


def test_window_keeps_excluded_raw_position_gaps():
    values = np.array([1., 10., 100.])
    actual = target_window_mean(values, np.array([0, 1, 10]), np.array([0]), np.array([3]), 2)
    np.testing.assert_array_equal(actual, [5.5, 5.5, 100.])


def test_dev_values_never_fit_references():
    pack = {
        "delta": np.array([[1., 2.], [3., 4.], [10., 20.], [30., 40.]]),
        "route": np.arange(4.), "target": np.array([0, 1, 0, 1]),
        "answer_index": np.array([0, 0, 1, 1]), "unit_index": np.array([0, 0, 0, 0]),
        "development": np.array([False, False, True, True]),
    }
    first = Detector("QA").fit(select_rows(pack, "fit")).references
    changed = {key: value.copy() for key, value in pack.items()}
    changed["delta"][2:] *= 1000
    changed["route"][2:] *= 1000
    second = Detector("QA").fit(select_rows(changed, "fit")).references
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
