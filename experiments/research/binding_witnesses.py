"""Source-certified row/value/year bindings, with no response error annotations."""

from decimal import Decimal
from hashlib import sha256
from itertools import combinations
import json

from experiments.research.constraint_interaction import literal
from experiments.research.table_bindings import bindings, number, table_regions


def binding_pairs(source, maximum=16):
    records, ambiguous = {}, set()
    for value, facts in bindings(source).items():
        for fact in facts:
            if not fact['row_label'].strip():
                continue
            key = (fact['table'], fact['row'], fact['year'])
            record = {k: fact[k] for k in ['table', 'row', 'year', 'row_label']}
            record['value'] = literal(Decimal(value))
            if key in records and records[key]['value'] != record['value']:
                ambiguous.add(key)
            records[key] = record
    records = {k: v for k, v in records.items() if k not in ambiguous}
    meanings = {}
    for fact in records.values():
        key = (fact['table'], fact['row_label'].strip().casefold(), fact['year'])
        meanings.setdefault(key, set()).add(fact['value'])
    unique = {}
    for fact in records.values():
        key = (fact['table'], fact['row_label'].strip().casefold(), fact['year'])
        if len(meanings[key]) == 1:
            unique.setdefault(key, fact)
    records = {(f['table'], f['row'], f['year']): f for f in unique.values()}
    pairs = []
    for first, second in combinations(records.values(), 2):
        if first['table'] != second['table'] or first['year'] == second['year'] or first['value'] == second['value']:
            continue
        negative = [records.get((f['table'], f['row'], other['year']))
                    for f, other in [(first, second), (second, first)]]
        if any(n is None or n['value'] == f['value'] for n, f in zip(negative, [first, second])):
            continue  # Missing cells do not certify a contradictory claim.
        key = repr([(f['table'], f['row'], f['year'], f['value']) for f in [first, second]])
        pairs.append({'id': sha256(key.encode()).hexdigest(), 'facts': [first, second],
                      'negative_cell_values': [n['value'] for n in negative]})
    return sorted(pairs, key=lambda p: p['id'])[:maximum]


def binding_text(pair, risk=0, style='bank', reverse=False):
    facts = pair['facts'][::-1] if reverse else pair['facts']
    years = [f['year'] for f in facts]
    if risk:
        years.reverse()
    text, spans = '', []
    for i, (fact, year) in enumerate(zip(facts, years, strict=True)):
        if i:
            text += '; '
        field, value = fact['row_label'], fact['value']
        if style == 'probe':
            text += f'The {field} entry for {value} is dated '
        elif style == 'retell':
            text += f'The {field} entry {value} belongs to reporting year '
        else:
            text += f'In table {fact["table"]+1}, {field} had entry {value} in '
        start = len(text); text += year; spans.append([start, len(text)])
    return {'text': text+'.', 'rhs_spans': spans}


def binding_worlds(pair):
    worlds = []
    for prior_risk in [0, 1]:
        for risk in [0, 1]:
            previous = binding_text(pair, prior_risk)['text']+'\n\n'
            current = binding_text(pair, risk, 'retell')
            worlds.append({'text': previous+current['text'], 'risk': risk,
                'coherence': int(prior_risk == risk), 'prior_risk': prior_risk,
                'rhs_spans': [[a+len(previous), b+len(previous)] for a, b in current['rhs_spans']]})
    return worlds


def binding_probe_pair(sample, pairs):
    if sample.id != 'fred:tatqa:test:50':
        return None
    candidates = [p for p in pairs if p['facts'][0]['row'] == p['facts'][1]['row']
                  and p['facts'][0]['row_label'].casefold() in sample.response.casefold()]
    return candidates[0] if candidates else None


def swap_year_columns(source, pair):
    """Swap complete data columns, including totals; retain the year header."""
    first, second = pair['facts']
    if first['table'] != second['table'] or first['row'] != second['row']:
        raise ValueError('A same-row year pair is required')
    parsed, columns = bindings(source), []
    for fact in [first, second]:
        matches = [f for f in parsed[number(fact['value'])] if
                   all(f[k] == fact[k] for k in ['table', 'row', 'year'])]
        if len(matches) != 1:
            raise ValueError('Source column is ambiguous')
        columns.append(matches[0]['column'])
    start, stop, table = list(table_regions(source))[first['table']]
    a, b = columns
    headers = [i for i, row in enumerate(table[:first['row']]) if
               max(a, b) < len(row) and str(row[a]).strip() == first['year'] and
               str(row[b]).strip() == second['year']]
    if len(headers) != 1 or any(max(a, b) >= len(row) for row in table[headers[0]+1:]):
        raise ValueError('Require a complete table with one explicit year header')
    for row in table[headers[0]+1:]:
        row[a], row[b] = row[b], row[a]
    return source[:start]+json.dumps(table, ensure_ascii=False)+source[stop:]


def binding_probes(sample, pairs, distractors=False):
    """Keep original answers; add explicit source probes only for the declared case."""
    probes = [{'name': 'original', 'text': sample.response, 'original': True}]
    pair = binding_probe_pair(sample, pairs)
    if pair is None:
        return probes
    for reverse in [False, True]:
        for prior_risk in [0, 1]:
            for risk in [0, 1]:
                previous = binding_text(pair, prior_risk)['text']+'\n\n'
                current = binding_text(pair, risk, 'probe', reverse)
                probes.append({'name': f'order{int(reverse)}_p{prior_risk}_b{risk}',
                    'text': previous+current['text'], 'original': False,
                    'binding_risk': risk, 'binding_coherence': int(prior_risk == risk),
                    'probe_spans': [[a+len(previous), b+len(previous)] for a, b in current['rhs_spans']]})
    if distractors:
        for reverse in [False, True]:
            facts = pair['facts'][::-1] if reverse else pair['facts']
            for placement in ['near', 'far']:
                for prior in [0, 1]:
                    for risk in [0, 1]:
                        text = binding_text(pair, prior)['text']+'\n\n'
                        spans = []
                        for i, fact in enumerate(facts):
                            other = facts[1-i]
                            if i:
                                text += '; '
                            comparison = f'(comparison amount {other["value"]})'
                            entry = f'the {fact["row_label"]} entry {fact["value"]}'
                            text += (comparison+' '+entry if placement == 'near' else entry+' '+comparison)
                            text += ' belongs to reporting year '
                            start = len(text)
                            text += other['year'] if risk else fact['year']
                            spans.append([start, len(text)])
                        probes.append({'name': f'distractor_o{int(reverse)}_{placement}_p{prior}_b{risk}',
                            'text': text+'.', 'original': False, 'binding_risk': risk,
                            'binding_coherence': int(prior == risk), 'placement': placement,
                            'probe_spans': spans, 'intended_values': [f['value'] for f in facts]})
    return probes
