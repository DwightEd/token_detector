"""Compare signed conditional-dependency profiles on an explicit common edge mask."""

import numpy as np


def signed_profile_distance(source, answer):
    """Squared Hellinger on promoting/inhibiting edge masses; no fitted scale.

    Input vectors must contain the same observed sender edges. Zero mass means
    unavailable, not compatibility. Raw mean squared difference is a comparator.
    """
    source, answer = np.asarray(source), np.asarray(answer)
    if source.shape != answer.shape or source.ndim != 1 or not len(source):
        raise ValueError('Expected two nonempty vectors over the same sender axis')
    if not np.isfinite(source).all() or not np.isfinite(answer).all():
        raise ValueError('Unobserved edges cannot enter the profile')
    masses = [np.concatenate([np.maximum(-v, 0), np.maximum(v, 0)])
              for v in [source, answer]]
    totals = [float(v.sum()) for v in masses]
    distance = (float(np.clip(1 - np.sqrt(masses[0] / totals[0] * masses[1] / totals[1]).sum(), 0, 1))
                if all(t > 0 for t in totals) else None)
    return {'hellinger': distance, 'mse': float(np.mean((source - answer) ** 2)),
            'source_mass': totals[0], 'answer_mass': totals[1],
            'source_promoting_mass': float(masses[0][:len(source)].sum()),
            'source_inhibiting_mass': float(masses[0][len(source):].sum()),
            'answer_promoting_mass': float(masses[1][:len(source)].sum()),
            'answer_inhibiting_mass': float(masses[1][len(source):].sum())}


def mask_preserving_permutation(mask):
    """Cycle rows only within identical observation-mask classes, preserving columns."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError('Expected a sender by receiver mask')
    permutation = np.arange(len(mask))
    groups = {}
    for row, available in enumerate(mask):
        groups.setdefault(available.tobytes(), []).append(row)
    for rows in groups.values():
        permutation[rows] = np.roll(rows, 1)
    assert np.array_equal(mask[permutation], mask)
    return permutation


def qualify_dependency_family(rows):
    """Frozen all-orders/targets separation, absolute collateral and identity null."""
    expected = {(so, ao, a, b) for so in [0, 1] for ao in [0, 1]
                for a in [0, 1] for b in [0, 1]}
    if len(rows) != 16 or {(r['source_order'], r['answer_order'], r['a'], r['b'])
                           for r in rows} != expected:
        raise ValueError('Expected every assignment and independent ordering exactly once')
    summary = {}
    targets, collateral = rows[0]['targets'], rows[0]['collateral']
    for readout in ['hellinger', 'mse']:
        def value(row, word, null=False):
            return row['costs'].get(word, {}).get(('null_' if null else '') + readout)

        available = (all(value(r, w) is not None for r in rows for w in targets + collateral)
                     and all(value(r, w, True) is not None for r in rows for w in targets))
        if not available:
            summary[readout] = {'qualified': False, 'reason': 'required column unavailable'}
            continue
        margins = {w: float(min(value(r, w) for r in rows if not r['supported'])
                            - max(value(r, w) for r in rows if r['supported'])) for w in targets}
        null_margin = min(value(r, w, True) - value(r, w)
                          for r in rows if r['supported'] for w in targets)
        contrasts = {}
        for source_order in [0, 1]:
            for answer_order in [0, 1]:
                selected = [r for r in rows if r['source_order'] == source_order
                            and r['answer_order'] == answer_order]

                def delta(words):
                    wrong = [value(r, w) for r in selected if not r['supported'] for w in words]
                    right = [value(r, w) for r in selected if r['supported'] for w in words]
                    return float(np.mean(wrong) - np.mean(right))

                contrasts[f'{source_order}{answer_order}'] = {
                    'target': delta(targets), 'collateral': {w: delta([w]) for w in collateral}}
        valid_contrasts = all(v['target'] > 0
                              and all(abs(d) < v['target'] for d in v['collateral'].values())
                              for v in contrasts.values())
        summary[readout] = {'target_margins': margins, 'supported_min_null_minus_true': float(null_margin),
                            'order_contrasts': contrasts,
                            'qualified': bool(all(v > 0 for v in margins.values())
                                              and null_margin > 0 and valid_contrasts)}
    return summary
