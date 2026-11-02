"""Preregistered algebraic examples only; no files, model, or target truth."""
from copy import deepcopy
import math

import pytest

from experiments.research.bijection_readout import assignment_distribution

EVENTS = ['Cobalt Cup', 'Ivory Cup']
NAMES = ['Mira Tanswick', 'Oren Bellamy']
MATRIX = [[-10, -13], [-12, -11]]


def readout(matrix, **changes):
    options = dict(events=EVENTS, candidates=NAMES, source_is_admitted_bijection=True)
    return assignment_distribution(matrix, **dict(options, **changes))


def test_preregistered_diagonal_matrix_includes_all_four_scores():
    original = deepcopy(MATRIX)
    value = readout(MATRIX)
    assert MATRIX == original and value['log_joint'] == MATRIX
    assert value['status'] == 'ok' and value['assignment_energies'] == [-21., -25.]
    assert value['delta'] == 4. and value['map_assignment'] == [0, 1]
    assert value['map_assignment_index'] == 0 and value['map_candidates'] == NAMES
    assert not value['tie']
    p = 1 / (1 + math.exp(-4))
    assert value['assignment_probabilities'] == pytest.approx([p, 1 - p])
    assert value['marginals'][0] == pytest.approx([p, 1 - p])
    assert value['marginals'][1] == pytest.approx([1 - p, p])
    for row in value['marginals']:
        assert math.fsum(row) == pytest.approx(1.)
    for column in zip(*value['marginals']):
        assert math.fsum(column) == pytest.approx(1.)
    assert value['conditional_on_admitted_bijection'] and not value['calibrated_truth_probability']


def test_preregistered_negative_interaction_can_be_high_confidence_wrong():
    value = readout([[-14, -10], [-11, -15]])
    assert value['assignment_energies'] == [-29., -21.] and value['delta'] == -8.
    assert value['map_assignment'] == [1, 0]  # A hypothetical true diagonal would be wrong.
    assert value['assignment_probabilities'][1] > .999
    assert not value['independent_factual_likelihood_assumed']


def test_exact_tie_has_no_candidate_order_tie_break():
    value = readout([[-2, -3], [-1, -2]])
    assert value['assignment_energies'] == [-4., -4.] and value['delta'] == 0.
    assert value['tie'] and value['map_assignment_index'] is None
    assert value['map_assignment'] is value['map_candidates'] is None
    assert value['assignment_probabilities'] == [.5, .5]
    assert value['marginals'] == [[.5, .5], [.5, .5]]


def test_fixed_additive_nuisance_terms_cancel_while_independent_rows_collapse():
    a, b = [-2, -5], [-1, -9]
    shifted = [[MATRIX[i][j] + a[i] + b[j] for j in range(2)] for i in range(2)]
    assert shifted == [[-13, -24], [-18, -25]]
    first, second = readout(MATRIX), readout(shifted)
    assert second['assignment_energies'] == [-38., -42.]
    assert second['assignment_energies'] == [energy - 17 for energy in first['assignment_energies']]
    assert first['delta'] == second['delta'] == 4.
    assert first['assignment_probabilities'] == second['assignment_probabilities']
    assert [max(range(2), key=lambda j: row[j]) for row in MATRIX] == [0, 1]
    assert [max(range(2), key=lambda j: row[j]) for row in shifted] == [0, 0]
    assert first['map_assignment'] == second['map_assignment'] == [0, 1]


@pytest.mark.parametrize('a,b', [([.1, -.7], [.9, -.4]), ([-20., 13.], [5., -8.]),
                                ([0., 0.], [10., 0.])])
def test_moderate_float_row_column_invariance_uses_only_roundoff_tolerance(a, b):
    shifted = [[MATRIX[i][j] + a[i] + b[j] for j in range(2)] for i in range(2)]
    first, second = readout(MATRIX), readout(shifted)
    assert second['delta'] == pytest.approx(first['delta'], rel=0, abs=1e-12)
    assert second['assignment_probabilities'] == pytest.approx(first['assignment_probabilities'], abs=1e-12)
    assert second['map_assignment'] == first['map_assignment']


@pytest.mark.parametrize('admission', [False, None, 1, 'yes'])
def test_shape_never_establishes_bijection_or_admission(admission):
    value = readout(MATRIX, source_is_admitted_bijection=admission)
    assert value['status'] == 'unknown' and value['reason'] == 'source_bijection_not_explicitly_admitted'
    assert value['assignment_probabilities'] is value['map_assignment'] is None


def test_absent_admission_uses_safe_default():
    assert assignment_distribution(MATRIX, events=EVENTS, candidates=NAMES)['status'] == 'unknown'


def test_absent_names_or_events_are_unknown_even_after_explicit_admission():
    assert assignment_distribution(MATRIX, source_is_admitted_bijection=True)['status'] == 'unknown'


@pytest.mark.parametrize('changes', [dict(candidates=[]), dict(candidates=NAMES[:1]),
    dict(candidates=[NAMES[0], NAMES[0]]), dict(candidates=['', NAMES[1]]),
    dict(events=['same', 'same']), dict(events=[' ', EVENTS[1]])])
def test_missing_or_duplicate_people_or_events_are_unknown(changes):
    assert readout(MATRIX, **changes)['status'] == 'unknown'


@pytest.mark.parametrize('matrix', [[], [[-1, -2]], [[-1], [-2]], [[-1, -2, -3], [-4, -5, -6]],
    [[-1, True], [-2, -3]], [[-1, None], [-2, -3]], [[-1, float('nan')], [-2, -3]],
    [[-1, float('inf')], [-2, -3]], [[-1, 10**1000], [-2, -3]]])
def test_malformed_nonfinite_boolean_or_unrepresentable_scores_are_unknown(matrix):
    assert readout(matrix)['status'] == 'unknown'


@pytest.mark.parametrize('matrix', [[[1e308, 0.], [0., 1e308]],
    [[-1e308, 0.], [0., -1e308]], [[1e308, -1e308], [0., 0.]]])
def test_energy_or_delta_overflow_returns_unknown_without_clipping(matrix):
    value = readout(matrix)
    assert value['status'] == 'unknown' and value['delta'] is None
    assert value['assignment_probabilities'] is None


def test_large_equal_negative_energies_still_normalize_and_log_mass_is_stable():
    value = readout([[-5e307, -5e307], [-5e307, -5e307]])
    assert value['status'] == 'ok' and value['assignment_energies'] == [-1e308, -1e308]
    assert value['assignment_probabilities'] == [.5, .5] and value['tie']
    assert value['assignment_log_probabilities'] == pytest.approx([-math.log(2)] * 2)


def test_large_finite_difference_reports_exp_underflow_without_clipping():
    value = readout([[-1, -1001], [-1001, -1]])
    assert value['status'] == 'ok' and value['delta'] == 2000.
    assert value['assignment_probabilities'] == [1., 0.] and value['normalization_underflow']
    assert value['assignment_log_probabilities'] == [0., -2000.]


def test_rounded_equal_probabilities_do_not_erase_a_computed_energy_map():
    tiny = math.ulp(0.)
    value = readout([[0., 0.], [tiny, 0.]])
    assert value['assignment_energies'] == [0., tiny] and value['delta'] == -tiny
    assert value['assignment_probabilities'] == [.5, .5]
    assert value['map_assignment_index'] == 1 and not value['tie']


def test_order_is_preserved_and_result_does_not_alias_caller_lists():
    events, names, matrix = list(reversed(EVENTS)), list(reversed(NAMES)), deepcopy(MATRIX)
    value = readout(matrix, events=events, candidates=names)
    events[0] = names[0] = 'changed'; matrix[0][0] = 999
    assert value['events'] == list(reversed(EVENTS)) and value['candidates'] == list(reversed(NAMES))
    assert value['log_joint'] == MATRIX
