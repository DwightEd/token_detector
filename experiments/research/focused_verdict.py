"""Pure exact-text focus requests and conservative token-risk projection.

The matched whole-response control uses the same builder with span (0, len(R)).
Only the before/focus/after partition changes; its concatenation always equals R.
This necessary serialization change is not a pure internal-model causal claim.
No model calls, file access, generated spans, truth labels or calibration occur.
"""

from collections.abc import Mapping
import json
import math
from numbers import Integral, Real

from experiments.research.semantic_verdict import (
    LABELS, SEMANTIC_CLASSES, VerdictValidationError, label_assignments, reduce_verdict,
    _SYSTEM as _WHOLE_SYSTEM, build_request as _whole_request,
)


CONTRACT_VERSION = 'focused-verdict-v1'
LIMITATIONS = (
    'Text units are deterministic character partitions, not complete atomic claims.',
    'Pronouns, cross-unit dependencies, abbreviations and quoted speech can impair units.',
    'Full response context resolves references; only the designated source supplies facts.',
    'Conditional-label risks are uncalibrated diagnostics, not hallucination posteriors.',
    'Broadcasting unit risk to tokens cannot locate errors within a multi-claim unit.',
    'Unknown risk 1 is a conservative engineering fallback, not a model judgment.',
)
_CLOSERS = '\"\'\u2019\u201d\u00bb)]}'
_FOCUS_RULE = (
    'The ordered concatenation response_before_focus + focus_text + '
    'response_after_focus is the complete original response. Judge only focus_text '
    'as intended in that original context. Use the surrounding response only to '
    'resolve references, scope and dependencies, never as factual evidence. '
    'Unresolved references or missing necessary premises remain unresolved. '
    'A text unit need not be an independent or complete proposition. '
)
# Reuse exactly the old faithfulness definitions; no extra null-value rule.
_SYSTEM = _WHOLE_SYSTEM.replace('complete response', 'focus_text').replace(
    'A response with no', 'A focus_text with no').replace(
    'Do not extract spans,', _FOCUS_RULE + 'Do not extract spans,')


def response_units(text):
    """Partition every original character, attaching boundary whitespace to the left.

    Newlines and Chinese terminal punctuation end units. ASCII .?!; end units
    before whitespace, closing quotes/brackets or end of text. Decimal-internal
    dots never split. Abbreviations deliberately receive no special repairs.
    """
    if not isinstance(text, str):
        raise VerdictValidationError('text must be an original string')
    units, start, i = [], 0, 0
    while i < len(text):
        char = text[i]
        decimal = (char == '.' and 0 < i < len(text) - 1
                   and text[i - 1] in '0123456789' and text[i + 1] in '0123456789')
        boundary = char in '\r\n\u3002\uff01\uff1f\uff1b' or (
            char in '.?!;' and not decimal and (
                i + 1 == len(text) or text[i + 1].isspace() or text[i + 1] in _CLOSERS))
        i += 1
        if boundary:
            while i < len(text) and text[i] in _CLOSERS:
                i += 1
            while i < len(text) and text[i].isspace():
                i += 1
            units.append({'start': start, 'end': i, 'text': text[start:i]})
            start = i
    if start < len(text):
        units.append({'start': start, 'end': len(text), 'text': text[start:]})
    return units


def _integer(value):
    return isinstance(value, Integral) and not isinstance(value, bool)


def build_request(source, response, precontext, focus_span, label_assignment):
    """Select an exact occurrence without exposing offsets or IDs to the model.

    Before/focus/after retain every response character once, including whitespace.
    The complete-response control has focus_span=(0, len(response)).
    """
    request = _whole_request(source, response, precontext, label_assignment)
    if (not isinstance(focus_span, (tuple, list)) or len(focus_span) != 2
            or not all(_integer(value) for value in focus_span)):
        raise VerdictValidationError('focus_span must be a pair of integer character offsets')
    start, end = focus_span
    if not 0 <= start <= end <= len(response) or (start == end and response):
        raise VerdictValidationError('focus_span must select exact nonempty response text')
    request.update(contract_version=CONTRACT_VERSION, mode='focused_semantic_verdict')
    request['messages'] = [
        {'role': 'system', 'content': _SYSTEM + json.dumps(
            request['label_assignment'], separators=(',', ':'))},
        {'role': 'user', 'content': json.dumps({
            'source': source, 'pre_generation_context': precontext,
            'response_before_focus': response[:start], 'focus_text': response[start:end],
            'response_after_focus': response[end:],
        }, ensure_ascii=False, separators=(',', ':'))},
    ]
    return request


def project_token_risks(units, unitrisks, token_offsets):
    """Max over overlapping character units, preserving every input token position.

    Unit risks are [0,1] or explicit unknown None. Unknown overlapping units and
    invalid, zero-width or out-of-response token offsets receive risk 1 with an
    unknown mask. Duplicate offsets remain distinct tokens. No tokenizer runs.
    """
    units, unitrisks = list(units), list(unitrisks)
    if len(units) != len(unitrisks):
        raise VerdictValidationError('one risk or explicit None is required per unit')
    cursor = 0
    for unit in units:
        if (not isinstance(unit, Mapping) or not _integer(unit.get('start'))
                or not _integer(unit.get('end')) or not isinstance(unit.get('text'), str)
                or unit['start'] != cursor or unit['end'] <= cursor
                or len(unit['text']) != unit['end'] - cursor):
            raise VerdictValidationError('units must be an exact contiguous character partition')
        cursor = unit['end']
    for risk in unitrisks:
        if risk is not None:
            try:
                valid = (isinstance(risk, Real) and not isinstance(risk, bool)
                         and math.isfinite(risk) and 0 <= risk <= 1)
            except (OverflowError, TypeError):
                valid = False
            if not valid:
                raise VerdictValidationError('unit risks must be finite [0,1] values or None')
    risks, unknown_mask, overlaps = [], [], []
    for offset in token_offsets:
        valid = (isinstance(offset, (tuple, list)) and len(offset) == 2
                 and all(_integer(value) for value in offset)
                 and 0 <= offset[0] < offset[1] <= cursor)
        indices = ([i for i, unit in enumerate(units)
                    if offset[0] < unit['end'] and offset[1] > unit['start']]
                   if valid else [])
        unknown = not indices or any(unitrisks[i] is None for i in indices)
        risks.append(1.0 if unknown else float(max(unitrisks[i] for i in indices)))
        unknown_mask.append(unknown)
        overlaps.append(indices)
    return {'risks': risks, 'unknown_mask': unknown_mask, 'overlapping_units': overlaps}
