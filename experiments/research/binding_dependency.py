"""A latent numeric antecedent and three-valued source-relation certificates."""

import re

import numpy as np

from experiments.research.table_bindings import ATOMS, YEAR, bindings, literal_links, number


def source_cells(source):
    cells = {}
    for value, facts in bindings(source).items():
        for fact in facts:
            key = (fact['table'], fact['row'])
            cells.setdefault((key, fact['year']), set()).add(value)
    return cells


def relation(cells, value, year):
    """Existential support, explicit contradiction, otherwise unknown.

    Values shared across source rows retain all interpretations. This does not
    certify row semantics, units, arithmetic derivations or an outside-table year.
    """
    rows = {row for (row, _), values in cells.items() if value in values}
    if not rows:
        return 'unknown'
    candidates = [cells.get((row, year)) for row in rows]
    if any(values is not None and value in values for values in candidates):
        return 'support'
    return 'conflict' if all(v is not None for v in candidates) else 'unknown'


def dependency_readout(text, offsets, units, history, source):
    """Use explicit relation syntax, otherwise pre-year attention.

    history[i,j] is original answer-token attention from i to j. Existing text
    units delimit antecedents; no gold span/antecedent is used for inference.
    Unknown mass is retained; non-year tokens have no score, not zero risk.
    """
    offsets, history = np.asarray(offsets), np.asarray(history, float)
    n = len(offsets)
    if history.shape != (n, n) or not np.isfinite(history).all() or np.any(history < 0):
        raise ValueError('Finite nonnegative original answer attention required')
    cells, records = source_cells(source), []
    links = literal_links(text)
    atoms = [m for m in ATOMS.finditer(text) if
             (m.start() == 0 or not (text[m.start()-1].isalnum() or text[m.start()-1] == '_')) and
             (m.end() == len(text) or not (text[m.end()].isalnum() or text[m.end()] == '_'))]
    schemes = ['dependency', 'uniform', 'nearest', 'permuted']
    values = {k: np.full(n, np.nan) for k in schemes}
    for year in atoms:
        if not re.fullmatch(YEAR, year[0]):
            continue
        target = np.flatnonzero((offsets[:, 0] < year.end()) & (offsets[:, 1] > year.start()))
        if not len(target) or target[0] == 0:
            continue
        row = int(target[0]-1)
        unit = next(u for u in units if u['start'] <= target[0] < u['stop'])
        start = unit.get('text_start', unit['start'])
        anchored = [link for link in links if link['year_span'] == [year.start(), year.end()]]
        if len(anchored) > 1:
            continue  # Competing grammatical links are not an attention label.
        mentions = []
        if anchored:
            a, b = anchored[0]['value_span']
            tokens = np.flatnonzero((offsets[:, 0] < b) & (offsets[:, 1] > a))
            mentions.append((number(anchored[0]['value']), tokens, a))
        else:
            if re.match(r'[ \t]*:', text[year.end():]):
                continue  # A year heading does not qualify the previous amount.
            for atom in atoms:
                if atom.end() > year.start() or re.fullmatch(YEAR, atom[0]):
                    continue
                tokens = np.flatnonzero((offsets[:, 0] < atom.end()) & (offsets[:, 1] > atom.start()))
                tokens = tokens[(tokens >= start) & (tokens <= row)]
                if len(tokens):
                    mentions.append((number(atom[0]), tokens, atom.start()))
        if not mentions:
            continue
        candidates = sorted({value for value, _, _ in mentions})
        masses = (np.ones(1) if anchored else np.array([
            sum(history[row, tokens].sum() for v, tokens, _ in mentions if v == value) for value in candidates]))
        if masses.sum() <= 0:
            continue
        weights = masses/masses.sum()
        nearest = max(mentions, key=lambda m: m[2])[0]
        distributions = {'dependency': weights, 'uniform': np.ones(len(weights))/len(weights),
            'nearest': np.array([float(v == nearest) for v in candidates]), 'permuted': weights[::-1]}
        states = [relation(cells, v, year[0]) for v in candidates]
        mention_mass = np.array([1. if anchored else history[row, tokens].sum()
                                 for _, tokens, _ in mentions])
        antecedents = [{'value': v, 'tokens': tokens.tolist(), 'start': int(a),
                        'weight': float(m/mention_mass.sum()), 'relation': relation(cells, v, year[0])}
                       for (v, tokens, a), m in zip(mentions, mention_mass, strict=True)]
        scores = {k: {s: float(w @ np.array([state == s for state in states]))
                      for s in ['support', 'conflict', 'unknown']} for k, w in distributions.items()}
        for key in schemes:
            if scores[key]['unknown'] <= 1e-12:
                values[key][target] = scores[key]['conflict']
        records.append({'start': year.start(), 'end': year.end(), 'year': year[0],
            'attention_row': None if anchored else row, 'unit_start': start, 'values': candidates,
            'routing': 'explicit_syntax' if anchored else 'pre_year_attention',
            'antecedents': antecedents, 'target_tokens': target.tolist(),
            'weights': {k: w.tolist() for k, w in distributions.items()}, 'relation': states,
            'relation_mass': scores, 'numeric_attention_mass': None if anchored else float(masses.sum())})
    return values, records


def dependency_checks(rows):
    """Routing must work for false claims too; evaluate all declared contrasts."""
    selected = {r['name']: r for r in rows if r.get('intended_values')}
    contrasts, routes = [], []
    for order in [0, 1]:
        for placement in ['near', 'far']:
            for slot in [0, 1]:
                scores = {key: {0: [], 1: []} for key in ['dependency', 'uniform', 'nearest', 'permuted']}
                for prior in [0, 1]:
                    for risk in [0, 1]:
                        row = selected.get(f'distractor_o{order}_{placement}_p{prior}_b{risk}')
                        edge = None
                        if row:
                            a, b = row['probe_spans'][slot]
                            edge = next((e for e in row['edges'] if e['start'] == a and e['end'] == b), None)
                        if edge is None or len(edge['values']) != 2 or 'unknown' in edge['relation']:
                            for key in scores:
                                scores[key][risk].append(None)
                            routes.append({'order': order, 'placement': placement, 'slot': slot,
                                           'prior': prior, 'risk': risk, 'intended_probability': None})
                            continue
                        effective_risk = risk ^ int(row.get('source_reassigned', False))
                        wanted = number(row['intended_values'][slot])
                        probability = (edge['weights']['dependency'][edge['values'].index(wanted)]
                                       if wanted in edge['values'] else 0.)
                        routes.append({'order': order, 'placement': placement, 'slot': slot,
                                       'prior': prior ^ int(row.get('source_reassigned', False)),
                                       'risk': effective_risk, 'query_original_risk': risk,
                                       'intended_probability': probability})
                        for key in scores:
                            scores[key][effective_risk].append(edge['relation_mass'][key]['conflict'])
                margins = {k: None if any(v is None for v in s[0]+s[1]) else min(s[1])-max(s[0])
                           for k, s in scores.items()}
                contrasts.append({'order': order, 'placement': placement, 'slot': slot, 'margins': margins})
    passed = (all(r['intended_probability'] is not None and r['intended_probability'] > .5 for r in routes)
              and all(c['margins']['dependency'] is not None and c['margins']['dependency'] > 0 for c in contrasts)
              and all(c['margins']['permuted'] is not None and c['margins']['permuted'] <= 0 for c in contrasts)
              and all(c['margins']['dependency'] > c['margins']['nearest'] and
                      c['margins']['dependency'] > c['margins']['uniform'] for c in contrasts if c['placement'] == 'far'))
    return {'mechanism_passed': passed, 'routing': routes, 'contrasts': contrasts,
        'benchmark_target_met': False, 'scope': 'stage1 known-source dependency qualification; source-reassignment confirmation still required'}
