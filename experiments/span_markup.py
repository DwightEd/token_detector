"""Decode FRED's released edit markup onto the unchanged erroneous input text."""

import re


ERROR_TYPES = {'numerical', 'entity', 'relation', 'temporal', 'contradictory', 'unverifiable'}
TAG = re.compile(r'<(/?)([A-Za-z_]+)>')


def decode_edits(marked):
    """Delete is original text; mark is its correction; unverifiable is removed text.

    The release writes contradictory closing tags as another opening tag. Only
    that documented type is toggled; other malformed nesting is rejected.
    """
    stack, parts, spans, length, end = [], [], [], 0, 0

    def append(text):
        nonlocal length
        if not text or 'mark' in stack:
            return
        start = length
        parts.append(text)
        length += len(text)
        if 'delete' in stack or 'unverifiable' in stack:
            kind = next((x for x in reversed(stack) if x in ERROR_TYPES), 'unspecified')
            if spans and spans[-1]['end'] == start and spans[-1]['error_type'] == kind:
                spans[-1]['end'] = length
            else:
                spans.append({'start': start, 'end': length, 'label': 1, 'error_type': kind})

    for match in TAG.finditer(marked):
        append(marked[end:match.start()])
        closing, name = match.groups()
        if name not in ERROR_TYPES | {'mark', 'delete'}:
            raise ValueError(f'Unknown FRED edit tag: {name}')
        if not closing and name == 'contradictory' and stack and stack[-1] == name:
            stack.pop()
        elif closing:
            if not stack or stack.pop() != name:
                raise ValueError('Unbalanced FRED edit tags')
        else:
            stack.append(name)
        end = match.end()
    append(marked[end:])
    if stack:
        raise ValueError('Unclosed FRED edit tags')
    return ''.join(parts), spans


def fred_text(prompt):
    """Separate evidence from the editing input, retaining exact response characters."""
    prefix = 'Read the following references:\n'
    marker = ('\nPlease identify all the errors in the following text using the information '
              'in the references provided and suggest edits:\nText: ')
    if not prompt.startswith(prefix) or prompt.count(marker) != 1:
        raise ValueError('Unexpected FRED editing prompt')
    source, response = prompt[len(prefix):].split(marker)
    if not source.strip() or not response:
        raise ValueError('Empty FRED evidence or input text')
    return source, response
