"""Lexically anchored contextual mismatch; an observation, not truth probability."""

from collections import Counter
import re

import numpy as np

WORDS = re.compile(r"\w+(?:['’]\w+)*", re.UNICODE)
STOPWORDS = frozenset('a an the and or but while whereas of to in on at for from '
                     'with by as is are was were be been being has have had it its '
                     'this that these those'.split())


def word_states(text, offsets, hidden):
    """Mean original subword states, then unit-normalize each lexical word."""
    offsets = np.asarray(offsets)
    hidden = np.asarray(hidden, dtype=np.float64)
    if offsets.shape != (len(hidden), 2):
        raise ValueError('Offsets and hidden states must share the original token axis')
    words = []
    for match in WORDS.finditer(text):
        rows = np.flatnonzero((offsets[:, 0] < match.end()) & (offsets[:, 1] > match.start()))
        if not len(rows):
            raise ValueError('A lexical word has no original token overlap')
        state = hidden[rows].mean(0)
        norm = np.linalg.norm(state)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('Invalid contextual state')
        words.append({'word': match.group().casefold(), 'start': match.start(),
                      'end': match.end(), 'tokens': rows.tolist(), 'state': state / norm})
    return words


def anchored_residual(source_words, answer_words, spans, minimum_other_anchors=4):
    """Leave-one-anchor-out shared displacement plus individual residual.

    Exact lexical matches must be unique in both complete texts. Matching never
    uses an evaluator-provided entity, role, target list or correctness label.
    Coverage failures remain unavailable, not negative hallucination scores.
    """
    source_counts = Counter(w['word'] for w in source_words)
    answer_counts = Counter(w['word'] for w in answer_words)
    eligible = {w['word']: w for w in source_words
                if source_counts[w['word']] == answer_counts[w['word']] == 1
                and w['word'] not in STOPWORDS}
    result = []
    for first, last in spans:
        anchors = [w for w in answer_words if first <= w['start'] and w['end'] <= last
                   and w['word'] in eligible]
        available = len(anchors) > minimum_other_anchors
        displacements = np.asarray([w['state'] - eligible[w['word']]['state'] for w in anchors])
        for index, word in enumerate(anchors):
            others = np.delete(displacements, index, axis=0)
            residual = (float(np.sum((displacements[index] - others.mean(0)) ** 2))
                        if available else None)
            result.append({k: word[k] for k in ['word', 'start', 'end', 'tokens']} | {
                'source_start': eligible[word['word']]['start'], 'available': available,
                'anchors_in_span': len(anchors), 'residual': residual,
                'cosine': float(1 - word['state'] @ eligible[word['word']]['state']),
            })
    return result


def qualify_family(cells):
    """Crossed contrasts and per-target separation across every fixed rendering."""
    summary = {}
    for readout in ['residual', 'cosine', 'gap']:
        primary = cells['primary']
        renderings = [r for rendering in ['primary', 'source_reordered', 'answer_reordered']
                      for r in cells[rendering]]
        wrong = [r[readout] for r in renderings if not r['supported']]
        supported = [r[readout] for r in renderings if r['supported']]
        delta = (np.mean([r[readout] for r in primary if not r['supported']])
                 - np.mean([r[readout] for r in primary if r['supported']]))
        collateral = (np.mean([r[f'{readout}_collateral'] for r in primary if not r['supported']])
                      - np.mean([r[f'{readout}_collateral'] for r in primary if r['supported']]))
        per_target = {}
        for target in primary[0]['target_costs']:
            wrong_costs = [r['target_costs'][target][readout] for r in renderings
                           if not r['supported']]
            supported_costs = [r['target_costs'][target][readout] for r in renderings
                               if r['supported']]
            per_target[target] = float(min(wrong_costs) - max(supported_costs))
        summary[readout] = {'crossed_delta': float(delta), 'collateral_delta': float(collateral),
                            'min_wrong_minus_max_supported': float(min(wrong) - max(supported)),
                            'per_target_min_wrong_minus_max_supported': per_target,
                            'qualified': bool(delta > 0 and all(v > 0 for v in per_target.values())
                                              and collateral < delta)}
    return summary
