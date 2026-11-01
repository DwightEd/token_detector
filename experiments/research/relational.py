"""Matched relational source controls and a shared latent null-expert model."""

import json
import re

import numpy as np
from scipy.special import logsumexp

from experiments.research.table_bindings import YEAR, number, table_regions
from myproject.detector import unit_ranges
from myproject.pooling import cavity_scores, grouped_scores

STRING = re.compile(r'"(?:\\.|[^"\\])*"')


def relational_sources(source):
    """Rotate numeric cells, retaining all literal cells and noncell bytes.

    Row labels, year headers and percentage/currency classes never mix.
    These controls destroy some table bindings; they are not certified false worlds.
    """
    replacements = {'row': [], 'column': []}
    for start, stop, table in table_regions(source):
        cells = [cell for row in table for cell in row]
        if not all(isinstance(cell, str) for cell in cells):
            continue
        tokens = list(STRING.finditer(source, start, stop))
        if [json.loads(m[0]) for m in tokens] != cells:
            raise ValueError('JSON table string coordinates changed')
        eligible, index, header_open = [], 0, True
        for r, row in enumerate(table):
            header = (header_open and
                      any(re.fullmatch(YEAR+r'\s*%?', c.strip()) for c in row[1:]))
            literal_years = sum(bool(re.fullmatch(YEAR, c.strip())) for c in row[1:])
            if literal_years >= 1 or (not header and any(number(c) is not None for c in row[1:])):
                header_open = False
            for c, cell in enumerate(row):
                token = tokens[index]; index += 1
                if c and not header and number(cell) is not None:
                    kind = ('%' in cell, tuple(s for s in '$£€' if s in cell))
                    eligible.append((r, c, kind, token))
        for mode, axis in [('row', 1), ('column', 0)]:
            groups = {}
            for entry in eligible:
                groups.setdefault((entry[axis], entry[2]), []).append(entry[3])
            for group in groups.values():
                if len(group) < 2:
                    continue
                for current, donor in zip(group, group[1:]+group[:1], strict=True):
                    replacements[mode].append((current.start(), current.end(), donor[0]))
    worlds, seen = [], {source}
    for mode, changes in replacements.items():
        transformed = source
        for a, b, value in sorted(changes, reverse=True):
            transformed = transformed[:a]+value+transformed[b:]
        if transformed not in seen:
            worlds.append({'name': mode, 'source': transformed})
            seen.add(transformed)
    return worlds


def multiworld_scores(contrasts, answer_index, unit_index):
    """Marginalize a uniform null world and Beta(1,1) token responsibility.

    World posteriors use the full span evidence. Token risks remain distinct;
    expert responsibilities are not factual-error probabilities.
    """
    contrasts = np.asarray(contrasts, float)
    if (contrasts.ndim != 2 or not all(contrasts.shape)
            or not np.isfinite(contrasts).all()):
        raise ValueError('Expected finite token-by-world contrasts')
    if np.shape(answer_index) != (len(contrasts),) or np.shape(unit_index) != (len(contrasts),):
        raise ValueError('Original token axes must match contrasts')
    if contrasts.shape[1] == 1:
        score = grouped_scores(contrasts[:, 0], answer_index, unit_index)['score']
        return {'world_shared': score, 'matched_empty_shared': score.copy()}
    result = np.empty(len(contrasts))
    starts, stops = unit_ranges(answer_index, unit_index)
    for a, b in zip(starts, stops, strict=True):
        worlds = [cavity_scores(d) for d in contrasts[a:b].T]
        evidence = np.array([w['log_evidence'] for w in worlds])[:, None]
        odds = np.array([w['score'] for w in worlds])
        result[a:b] = (logsumexp(evidence-np.logaddexp(0, -odds), axis=0)
                       - logsumexp(evidence-np.logaddexp(0, odds), axis=0))
    return {'world_shared': result,
            'matched_empty_shared': grouped_scores(contrasts[:, 0], answer_index, unit_index)['score']}
