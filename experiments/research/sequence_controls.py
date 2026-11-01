"""Fixed falsification controls calling the shared mainline sequence kernel."""

import numpy as np

from myproject.detector import unit_ranges
from myproject.sequence import persistent_scores


def span_score_components(scores, answer_index, unit_index):
    """Delete within-span variation or span means on the original valid axis.

    Every supplied score contributes, regardless of any later label coverage.
    These are diagnostic score transforms, not calibrated belief estimates.
    """
    scores, answers, units = map(np.asarray, (scores, answer_index, unit_index))
    if (scores.ndim != 1 or answers.shape != scores.shape or units.shape != scores.shape
            or not np.isfinite(scores).all()):
        raise ValueError('Finite scores and matching original span axes required')
    means = np.empty(len(scores), dtype=np.float64)
    starts, stops = unit_ranges(answers, units) if len(scores) else ([], [])
    for start, stop in zip(starts, stops, strict=True):
        means[start:stop] = np.mean(scores[start:stop])
    centered = scores-means
    if not np.isfinite(means).all() or not np.isfinite(centered).all():
        raise ValueError('Span mean or centered scores are nonfinite')
    return {'span_mean': means, 'span_centered': centered}


def span_order_checks(scores, centered, answer_index, unit_index):
    """Count actual floating-point order/tie changes without changing scores."""
    scores, centered, answers, units = map(
        np.asarray, (scores, centered, answer_index, unit_index))
    if (scores.ndim != 1 or any(values.shape != scores.shape for values in
                               (centered, answers, units))
            or not np.isfinite(scores).all() or not np.isfinite(centered).all()):
        raise ValueError('Finite original and centered scores with matching span axes required')
    result = {name: 0 for name in ('spans', 'pairs', 'strict_order_reversals',
                                  'strict_pairs_collapsed_to_ties', 'original_ties_split')}
    starts, stops = unit_ranges(answers, units) if len(scores) else ([], [])
    for start, stop in zip(starts, stops, strict=True):
        a, b = np.triu_indices(stop-start, 1)
        before, after = scores[start:stop], centered[start:stop]
        old = (before[a] > before[b]).astype(int)-(before[a] < before[b])
        new = (after[a] > after[b]).astype(int)-(after[a] < after[b])
        result['spans'] += 1
        result['pairs'] += len(a)
        result['strict_order_reversals'] += int(np.sum(old*new < 0))
        result['strict_pairs_collapsed_to_ties'] += int(np.sum((old != 0) & (new == 0)))
        result['original_ties_split'] += int(np.sum((old == 0) & (new != 0)))
    return result


def token_state_controls(pack, witness_score, token_score, candidate='token_state'):
    """Fixed clipping and span-mean deletion controls for token-state inference."""
    starts, stops = unit_ranges(pack['answer_index'], pack['unit_index'])
    result = {'clipped_score': np.empty(len(pack['target'])),
              'witness_mean_score': span_score_components(
                  witness_score, pack['answer_index'], pack['unit_index'])['span_mean'],
              f'{candidate}_mean_score': span_score_components(
                  token_score, pack['answer_index'], pack['unit_index'])['span_mean']}
    if not len(pack['target']):
        return result
    for a, b in zip(starts, stops, strict=True):
        clipped = np.maximum(pack['delta'][a:b, 1], -np.log(2.))
        result['clipped_score'][a:b] = persistent_scores(clipped)['score']
    return result


def grouped_witness_controls(values, answer_index, unit_index):
    """Three fixed falsification controls, preserving every original token gap.

    The permutation is a computational order control, not a new LM observation.
    Its scores are mapped back to original token positions before evaluation.
    Clipping is an energy baseline, not a normalized word-density claim.
    """
    values, answer_index, unit_index = map(np.asarray, (values, answer_index, unit_index))
    if values.ndim != 1 or answer_index.shape != values.shape or unit_index.shape != values.shape:
        raise ValueError('Original token axes must match contrasts')
    starts, stops = unit_ranges(answer_index, unit_index)
    result = {k: np.empty(len(values)) for k in ['no_copy_score', 'clipped_score', 'permuted_score']}
    for a, b in zip(starts, stops, strict=True):
        d = values[a:b]
        result['no_copy_score'][a:b] = persistent_scores(d, copying=False, witness=True)['score']
        result['clipped_score'][a:b] = persistent_scores(np.maximum(d, -np.log(2.)))['score']
        rng = np.random.default_rng(np.random.SeedSequence([17, int(answer_index[a]), int(unit_index[a])]))
        order = rng.permutation(len(d))
        mapped = np.empty(len(d))
        mapped[order] = persistent_scores(d[order], witness=True)['score']
        result['permuted_score'][a:b] = mapped
    return result
