"""Observable year/value binding conflicts; a narrow diagnostic, not an entailment model."""

from decimal import Decimal, InvalidOperation
import json
import re

import numpy as np

NUMBER = r'[-+]?\d+(?:,\d{3})*(?:\.\d+)?'
YEAR = r'(?:19|20)\d{2}'
ATOMS = re.compile(NUMBER)
FORWARD = re.compile(
    rf'(?<![\w.])(?P<value>{NUMBER})[ \t]*%?[ \t]*'
    rf'(?:(?:thousand|million|billion)s?[ \t]*)?'
    rf'(?:in|for|during|as of)[ \t]+(?P<year>{YEAR})(?!\d)', re.I)
BACKWARD = re.compile(
    rf'\b(?:in|for|during)\s+(?P<year>{YEAR})(?!\d),?'
    rf'(?:(?![\d.;\n]).){{0,100}}?\b(?:was|were|is|are|of|at|totaled|totalled)\s*'
    rf'[$£€]?\s*(?P<value>{NUMBER})', re.I)


def number(text):
    """Normalize a literal amount only; do not invent units or derived values."""
    text = str(text).strip().replace(',', '').replace('$', '').replace('£', '').replace('€', '')
    text = text.strip().removesuffix('%').strip()
    if text.startswith('(') and text.endswith(')'):
        text = '-'+text[1:-1]
    if re.fullmatch(r'[-+]?\d+(?:\.\d+)?', text) is None:
        return None
    try:
        return str(Decimal(text).normalize())
    except InvalidOperation:
        return None


def table_regions(source):
    """Locate embedded JSON tables without reserializing surrounding evidence."""
    decoder = json.JSONDecoder()
    cursor = 0
    while (start := source.find('[[', cursor)) >= 0:
        try:
            value, length = decoder.raw_decode(source[start:])
        except json.JSONDecodeError:
            cursor = start+2
            continue
        cursor = start+length
        if isinstance(value, list) and all(isinstance(row, list) for row in value):
            yield start, cursor, value


def tables(source):
    for _, _, value in table_regions(source):
        yield value


def bindings(source):
    """Read explicit multi-year table columns, retaining all competing row witnesses."""
    result = {}
    for table_index, table in enumerate(tables(source)):
        header = {}
        for row_index, row in enumerate(table):
            years = {column: str(cell).strip() for column, cell in enumerate(row[1:], 1)
                     if re.fullmatch(YEAR, str(cell).strip())}
            if not header and len(years) >= 2:
                header = years
                continue
            for column, year in header.items():
                if column >= len(row) or (value := number(row[column])) is None:
                    continue
                result.setdefault(value, []).append({'year': year, 'table': table_index,
                    'row': row_index, 'column': column, 'row_label': str(row[0]),
                    'literal_value': str(row[column])})
    return result


def literal_links(response):
    """Parse explicit numeric-year syntax, retaining both argument spans."""
    result, seen = [], set()
    for pattern in [FORWARD, BACKWARD]:
        for match in pattern.finditer(response):
            if pattern is FORWARD and re.match(r'[ \t]*:', response[match.end():]):
                continue  # "$780 For 2018: ..." starts a new heading.
            key = (match.span('value'), match.span('year'))
            if key in seen:
                continue
            seen.add(key)
            result.append({'value': match['value'], 'year': match['year'],
                           'value_span': list(match.span('value')), 'year_span': list(match.span('year'))})
    return result


def conflicts(source, response):
    """Check explicit amount-in-year / in-year-value statements, with no error labels.

    Unknown or derived amounts abstain. A witness means no parsed table binding
    matches; prose evidence, units and row semantics remain unmodeled limitations.
    """
    facts, result = bindings(source), []
    for link in literal_links(response):
        value = number(link['value'])
        if value not in facts:
            continue
        compatible = sorted({fact['year'] for fact in facts[value]})
        result.append(link | {
            'allowed_years': compatible, 'conflict': link['year'] not in compatible,
            'condition_seen_in_table': link['year'] in {
                f['year'] for candidates in facts.values() for f in candidates},
            'source_witnesses': facts[value]})
    return result


def scores(source, response, offsets):
    """Project each conflicting pair to both endpoints, without guessing the guilty one."""
    offsets = np.asarray(offsets)
    output = {key: np.zeros(len(offsets)) for key in ['binding_conflict', 'atom_absence', 'year_absence']}
    source_atoms = {number(m[0]) for m in ATOMS.finditer(source)}
    for match in ATOMS.finditer(response):
        if number(match[0]) not in source_atoms:
            mask = (offsets[:, 0] < match.end()) & (offsets[:, 1] > match.start())
            output['atom_absence'][mask] = 1.
            if re.fullmatch(YEAR, match[0]):
                output['year_absence'][mask] = 1.
    witnesses = conflicts(source, response)
    for pair in witnesses:
        if pair['conflict']:
            for start, stop in [pair['value_span'], pair['year_span']]:
                mask = (offsets[:, 0] < stop) & (offsets[:, 1] > start)
                output['binding_conflict'][mask] = .5
    return output, witnesses
