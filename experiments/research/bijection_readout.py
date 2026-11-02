"""Two-assignment toy energy model; caller must separately admit a bijection.

Rows and columns preserve the caller's frozen event/name order. A 2x2 shape
does not establish source semantics. No model, files, gold, fitted parameters,
length correction, or claim of calibrated joint factuality is involved.
"""
import math

CONTRACT_VERSION = 'two-by-two-admitted-bijection-v1'


def _unknown(reason):
    return dict(status='unknown', reason=reason, contract_version=CONTRACT_VERSION,
        assignment_energies=None, delta=None, assignment_probabilities=None,
        assignment_log_probabilities=None, map_assignment_index=None,
        map_assignment=None, map_candidates=None, tie=None, marginals=None,
        calibrated_truth_probability=False)


def _labels(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
        and all(isinstance(name, str) and name and name == name.strip() for name in value)
        and value[0] != value[1])


def assignment_distribution(log_joint, *, events=None, candidates=None,
                            source_is_admitted_bijection=False):
    """Compare diag/offdiag sums of four complete-name-plus-EOS joint log scores.

Admission asserts two complete distinct people/events and one use of each;
this function cannot prove that assertion from shape or scores. Additive row
and column terms cancel algebraically in Delta, not necessarily exactly after
Python-float rounding. Ties use exact computed energies, without tolerance or
candidate-order tie breaking. Finite arithmetic overflow returns unknown;
ordinary exp underflow is retained explicitly as zero floating-point mass.
"""
    if source_is_admitted_bijection is not True:
        return _unknown('source_bijection_not_explicitly_admitted')
    if not _labels(events) or not _labels(candidates):
        return _unknown('two_distinct_complete_events_and_candidates_required')
    if (not isinstance(log_joint, (list, tuple)) or len(log_joint) != 2
            or any(not isinstance(row, (list, tuple)) or len(row) != 2 for row in log_joint)):
        return _unknown('exact_two_by_two_score_matrix_required')
    try:
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for row in log_joint for value in row):
            return _unknown('finite_nonboolean_python_scores_required')
        matrix = [[float(value) for value in row] for row in log_joint]
        energies = [math.fsum((matrix[0][0], matrix[1][1])),
                    math.fsum((matrix[0][1], matrix[1][0]))]
        delta = math.fsum((energies[0], -energies[1]))
        if not all(math.isfinite(value) for value in [*energies, delta]):
            return _unknown('nonfinite_assignment_arithmetic')
        peak = max(energies)
        shifted = [value - peak for value in energies]
        weights = [math.exp(value) for value in shifted]
        total = math.fsum(weights)
        log_total = math.log(total)
        probabilities = [value / total for value in weights]
        log_probabilities = [value - log_total for value in shifted]
        log_normalizer = peak + log_total
        if not all(math.isfinite(value) for value in
                   [*probabilities, *log_probabilities, log_normalizer]):
            return _unknown('nonfinite_assignment_normalization')
    except (OverflowError, ValueError):
        return _unknown('assignment_arithmetic_overflow')
    assignments = [[0, 1], [1, 0]]
    tie = energies[0] == energies[1]
    winner = None if tie else (0 if energies[0] > energies[1] else 1)
    chosen = None if winner is None else list(assignments[winner])
    return dict(status='ok', reason=None, contract_version=CONTRACT_VERSION,
        events=list(events), candidates=list(candidates), log_joint=matrix,
        source_is_admitted_bijection=True, assignment_candidate_indices=assignments,
        assignment_energies=energies, delta=delta, assignment_probabilities=probabilities,
        assignment_log_probabilities=log_probabilities, log_assignment_normalizer=log_normalizer,
        map_assignment_index=winner, map_assignment=chosen,
        map_candidates=None if chosen is None else [candidates[index] for index in chosen],
        tie=tie, marginals=[list(probabilities), list(reversed(probabilities))],
        normalization_underflow=any(value == 0.0 for value in weights),
        conditional_on_admitted_bijection=True, calibrated_truth_probability=False,
        independent_factual_likelihood_assumed=False, length_normalization=False, temperature=1.0,
        numerical_policy='python_float_fsum_exp_shift_exact_energy_tie_no_clipping')
