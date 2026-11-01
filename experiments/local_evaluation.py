"""Source-paired uncertainty for discrimination within original spans."""

import numpy as np
from sklearn.metrics import roc_auc_score

from myproject.detector import unit_ranges


def _pair_counts(labels, scores):
    """Count twice-concordant positive/negative pairs, including exact ties."""
    positive, negative = scores[labels], np.sort(scores[~labels])
    left = np.searchsorted(negative, positive, side='left')
    right = np.searchsorted(negative, positive, side='right')
    return len(positive) * len(negative), sum(map(int, left + right))


def auc_pair_decomposition(labels, scores, answer_index, unit_index):
    """Split pooled AUC pairs into same-span and cross-span comparisons.

    Inputs must already contain only known binary labels on the sealed original
    token order. Unknown labels are not coerced to True or silently discarded.
    Answer/unit changes define contiguous spans; repeated identities are not
    merged across intervening spans. Ties contribute one half. This is a
    descriptive pair-weighted identity, not a macro estimand or a confidence
    interval. AP does not admit this decomposition. Runtime is O(n log n).
    """
    y, values, answers, units = map(np.asarray, (labels, scores, answer_index, unit_index))
    if (y.ndim != 1 or any(a.shape != y.shape for a in (values, answers, units))
            or y.dtype.kind not in 'biuf' or not np.isin(y, [0, 1]).all()
            or values.dtype.kind not in 'biuf' or not np.isfinite(values).all()):
        raise ValueError('Known binary labels, finite real scores and matching original axes required')
    y = y.astype(bool)
    total_pairs, total_twice = _pair_counts(y, values)
    starts, stops = unit_ranges(answers, units) if len(y) else ([], [])
    within_pairs = within_twice = mixed = 0
    for start, stop in zip(starts, stops, strict=True):
        pairs, twice = _pair_counts(y[start:stop], values[start:stop])
        within_pairs += pairs
        within_twice += twice
        mixed += bool(pairs)

    def report(pairs, twice):
        return {'positive_negative_pairs': pairs, 'twice_concordant_pairs': twice,
                'auroc': twice / (2 * pairs) if pairs else None}

    return {
        'pooled': report(total_pairs, total_twice),
        'within_span': report(within_pairs, within_twice),
        'between_spans': report(total_pairs - within_pairs, total_twice - within_twice),
        'known_token_spans': len(starts), 'mixed_known_token_spans': mixed,
        'within_pair_fraction': within_pairs / total_pairs if total_pairs else None,
        'weighting': 'all known positive-negative token pairs; exact ties count one half',
    }


def paired_span_differences(labels, baseline, candidate, source_ids, answer_index,
                            unit_index, repetitions=1000, seed=17):
    """Bootstrap source-level sums of within-span pair concordance.

    Repeated source clusters repeat their spans; they do not merge duplicate
    spans into a larger group. Pure-label spans contribute zero pairs.
    """
    y, before, after, sources, answers, units = map(
        np.asarray, (labels, baseline, candidate, source_ids, answer_index, unit_index))
    if (y.ndim != 1 or any(a.shape != y.shape for a in
                          (before, after, sources, answers, units))
            or not np.isfinite(before).all() or not np.isfinite(after).all()
            or repetitions < 1):
        raise ValueError('Matching finite original token arrays and positive repetitions required')
    y = y.astype(bool)
    unique, source_axis = np.unique(sources, return_inverse=True)
    pairs = np.zeros(len(unique))
    concordance = np.zeros((2, len(unique)))
    starts, stops = unit_ranges(answers, units) if len(y) else ([], [])
    mixed = 0
    for a, b in zip(starts, stops, strict=True):
        source = source_axis[a]
        if np.any(source_axis[a:b] != source):
            raise ValueError('An original span crosses source clusters')
        positive = int(y[a:b].sum())
        weight = positive * (b-a-positive)
        if not weight:
            continue
        mixed += 1
        pairs[source] += weight
        for index, scores in enumerate((before, after)):
            concordance[index, source] += weight * roc_auc_score(y[a:b], scores[a:b])
    total = float(pairs.sum())
    point = concordance.sum(axis=1)/total if total else (None, None)
    values = []
    if total:
        rng = np.random.default_rng(seed)
        for _ in range(repetitions):
            counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
            denominator = counts @ pairs
            if denominator:
                values.append(float(counts @ (concordance[1]-concordance[0])/denominator))
    return {
        'baseline': float(point[0]) if total else None,
        'candidate': float(point[1]) if total else None,
        'difference': float(point[1]-point[0]) if total else None,
        'bootstrap': {'unit': 'source_id', 'seed': seed, 'repetitions': repetitions,
                      'defined': len(values), 'degenerate': repetitions-len(values),
                      'interval95': np.quantile(values, [.025, .975]).tolist() if values else None},
        'spans': len(starts), 'mixed_spans': mixed,
        'positive_negative_pairs': int(total), 'sources': len(unique),
        'weighting': 'positive-negative pairs within each original span',
    }
