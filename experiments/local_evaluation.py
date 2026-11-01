"""Source-paired uncertainty for discrimination within original spans."""

import numpy as np
from sklearn.metrics import roc_auc_score

from myproject.detector import unit_ranges


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
