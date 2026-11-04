"""Conservative certificates for a finite, explicit table/answer grammar.

This is not a semantic parser. Only embedded JSON row tables with literal year
columns and explicit units are understood. Direct statements use ``ROW in YEAR
was AMOUNT`` (also year-first and amount-before-year). Binary statements use
``ROW: AMOUNT in YEAR +/- AMOUNT in YEAR = AMOUNT``. Unparsed prose is neutral.
All returned token indices address ``offsets[target]``, in the supplied order.
"""

from collections import Counter
from decimal import Decimal, InvalidOperation, localcontext
import json
import re
import unicodedata

import numpy as np


_YEAR = r"(?:19|20)\d{2}"
_NUM = r"(?:\([+-]?\d+(?:,\d{3})*(?:\.\d+)?\)|[+-]?\d+(?:,\d{3})*(?:\.\d+)?)"
_SCALE = r"(?:thousand|million|billion)s?"
_MAX_PROGRAMS = 4096
_MAX_TERMS = 256


def _name(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def _decimal(text):
    text = text.strip().replace(",", "")
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
        if not text.startswith(("-", "+")):
            text = "-" + text
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _unit(text):
    """Parse an entire unit/header literal; an empty signature is unspecified."""
    text = _name(text)
    currency = re.findall(r"[$£€]|\b(?:usd|gbp|eur)\b", text)
    scales = re.findall(rf"\b{_SCALE}\b", text)
    percent = re.findall(r"%|\bpercent\b", text)
    signature = (tuple(sorted(set(currency))),
                 tuple(sorted({s.rstrip("s") for s in scales})), bool(percent))
    stripped = re.sub(rf"[$£€%()]|\b(?:usd|gbp|eur|percent|in|{_SCALE})\b", "", text)
    if stripped.strip() or len(signature[0]) > 1 or len(signature[1]) > 1:
        return None
    if signature[2] and (signature[0] or signature[1]):
        return None
    return signature


def _merge_unit(left, right):
    if left is None or right is None:
        return None
    merged = []
    for a, b in zip(left[:2], right[:2]):
        if a and b and a != b:
            return None
        merged.append(a or b)
    percent = left[2] or right[2]
    if percent and any(merged):
        return None
    return (*merged, percent)


def _has_unit(unit):
    return unit is not None and any(unit)


def _amount_pattern(prefix):
    return (rf"(?P<{prefix}currency>[$£€]|\bUSD\b|\bGBP\b|\bEUR\b)?[ \t]*"
            rf"(?P<{prefix}value>{_NUM})[ \t]*"
            rf"(?P<{prefix}scale>{_SCALE})?[ \t]*(?P<{prefix}percent>%|percent\b)?")


def _amount(match, prefix):
    spans = [match.span(prefix + part) for part in ("currency", "scale", "percent")
             if match.group(prefix + part)]
    unit = _unit(" ".join(match.group(prefix + part) or ""
                          for part in ("currency", "scale", "percent")))
    literal = match.group(prefix + "value")
    start, _ = match.span(prefix + "value")
    magnitude = re.search(r"\d[\d,.]*", literal)
    sign_spans = [(start + i, start + i + 1) for i, char in enumerate(literal) if char in "+-()"]
    return {"value": _decimal(literal), "unit": unit, "literal": literal,
            "value_span": (start + magnitude.start(), start + magnitude.end()),
            "sign_spans": sign_spans, "unit_spans": spans}


def _header_unit(text):
    """Only a complete unit literal or an explicit terminal unit parenthesis."""
    unit = _unit(text)
    if unit is not None:
        return unit
    prefix = r"(?:(?:amounts?|values?|years? ended (?:january|february|march|april|may|june|july|august|september|october|november|december) \d{1,2},?)\s*)?"
    match = re.fullmatch(prefix + r"\(\s*(?P<unit>(?:in\s+)?(?:[$£€]\s*)?(?:thousand|million|billion)s?)\s*\)",
                         _name(text))
    return _unit(match["unit"]) if match else None


def _json_tables(source):
    """Decode JSON cells while retaining each cell's original source span."""
    decoder = json.JSONDecoder()
    cursor = 0
    while (start := source.find("[[", cursor)) >= 0:
        try:
            table, size = decoder.raw_decode(source[start:])
        except json.JSONDecodeError:
            cursor = start + 2
            continue
        cursor = start + size
        if not (isinstance(table, list) and table
                and all(isinstance(row, list) for row in table)):
            continue
        # Recover scalar cell boundaries without searching for repeated values.
        cells, position = [], start + 1
        valid = True
        for row in table:
            while position < cursor and source[position].isspace():
                position += 1
            if source[position] != "[":
                valid = False
                break
            position += 1
            spans = []
            for _ in row:
                while source[position].isspace():
                    position += 1
                _, length = decoder.raw_decode(source[position:])
                spans.append((position, position + length))
                position += length
                while source[position].isspace():
                    position += 1
                if source[position] == ",":
                    position += 1
            while source[position].isspace():
                position += 1
            position += 1  # closing row bracket
            while position < cursor and source[position].isspace():
                position += 1
            if position < cursor and source[position] == ",":
                position += 1
            cells.append(spans)
        if valid:
            yield start, cursor, table, cells


def _source_scopes(source):
    scopes, problems = [], []
    for table_id, (start, stop, rows, spans) in enumerate(_json_tables(source)):
        # Header candidates belong to the structured prefix. A year-looking
        # amount in a named data row must never become a second year header.
        headers = [(i, {j: str(cell).strip() for j, cell in enumerate(row[1:], 1)
                        if re.fullmatch(_YEAR, str(cell).strip())})
                   for i, row in enumerate(rows)
                   if row and (i == 0 or not str(row[0]).strip() or _has_unit(_header_unit(str(row[0]))))]
        headers = [(i, years) for i, years in headers if years]
        if len(headers) != 1 or any(re.fullmatch(_amount_pattern("prior"), str(cell).strip(), re.I)
                                    for row in rows[:headers[0][0]] for cell in row[1:]):
            problems.append("missing_or_ambiguous_year_header")
            scopes.append({"table": table_id, "span": [start, stop], "facts": {},
                           "complete": False, "header_context": [], "units": set(),
                           "header_context_spans": [],
                           "subjects": {_name(row[0]) for row in rows if row and str(row[0]).strip()}})
            continue
        header_index, years = headers[0]
        header_unit = _header_unit(str(rows[header_index][0]))
        if header_unit is None:
            # A field-name header is allowed; it does not declare a unit.
            header_unit = ((), (), False)
        column_units = dict.fromkeys(years, header_unit)
        first_data = header_index + 1
        while first_data < len(rows):
            row = rows[first_data]
            if not all(j < len(row) and _has_unit(_unit(str(row[j]))) for j in years):
                break
            if row and str(row[0]).strip() and not _has_unit(_unit(str(row[0]))):
                break
            for j in years:
                column_units[j] = _merge_unit(column_units[j], _unit(str(row[j])))
            first_data += 1
        facts = {}
        complete = True
        for i in range(first_data, len(rows)):
            row = rows[i]
            if not row or any(isinstance(cell, (dict, list, bool)) for cell in row):
                complete = False
                continue
            label = _name(row[0])
            path = label if label else ("anonymous_row", i)
            for j, year in years.items():
                if j >= len(row):
                    complete = False
                    continue
                literal = str(row[j]).strip()
                match = re.fullmatch(_amount_pattern("cell"), literal, re.I)
                if match is None:
                    complete = False
                    continue
                amount = _amount(match, "cell")
                unit = _merge_unit(column_units[j], amount["unit"])
                if not _has_unit(unit):
                    complete = False
                    continue
                key = (path, year, unit)
                fact = facts.setdefault(key, {"subject": label, "path": path, "year": year,
                                             "unit": unit, "values": {}})
                provenance = {"table": table_id, "table_span": [start, stop], "row": i,
                              "column": j, "row_span": list(spans[i][0]),
                              "value_span": list(spans[i][j]),
                              "header_span": list(spans[header_index][j]),
                              "literal": literal}
                fact["values"].setdefault(amount["value"], []).append(provenance)
        scopes.append({"table": table_id, "span": [start, stop], "facts": facts,
                       "complete": complete,
                       "header_context": rows[:first_data],
                       "header_context_spans": spans[:first_data],
                       "subjects": {_name(row[0]) for row in rows[first_data:] if row and str(row[0]).strip()},
                       "units": {key[2] for key in facts}})
    return scopes, problems


def _anchors(response, scopes):
    labels = {subject for scope in scopes for subject in scope["subjects"]}
    matches = []
    for label in sorted(labels):
        pattern = r"(?<![\w-])" + r"\s+".join(re.escape(s) for s in label.split()) + r"(?![\w-])"
        for match in re.finditer(pattern, response, re.I):
            matches.append((match.start(), match.end(), label))
    return [m for m in matches if not any(a <= m[0] and b >= m[1] and b-a > m[1]-m[0]
                                         for a, b, _ in matches)]


def _claims(response, scopes):
    claims = []
    endpoint = r"(?=[ \t]*(?:\.(?!\d)|;|\n|$))"
    for start, stop, subject in _anchors(response, scopes):
        amount = _amount_pattern("a")
        patterns = [
            ("direct", rf"[ \t]+(?:in|for|during)[ \t]+(?P<year>{_YEAR})[ \t]+(?:was|were|is|are|totaled|totalled)[ \t]+{amount}{endpoint}"),
            ("direct", rf"[ \t]+(?:was|were|is|are|totaled|totalled)[ \t]+{amount}[ \t]+(?:in|for|during)[ \t]+(?P<year>{_YEAR}){endpoint}"),
            ("binary", rf"[ \t]*:[ \t]*{_amount_pattern('left')}[ \t]+in[ \t]+(?P<leftyear>{_YEAR})[ \t]*(?P<operator>[+-])[ \t]*{_amount_pattern('right')}[ \t]+in[ \t]+(?P<rightyear>{_YEAR})[ \t]*=[ \t]*{_amount_pattern('result')}{endpoint}"),
        ]
        for schema, pattern in patterns:
            match = re.compile(pattern, re.I).match(response, stop)
            if match:
                claims.append({"schema": schema, "subject": subject, "subject_span": (start, stop),
                               "span": (start, match.end()), "match": match})
        # The year-first schema is anchored backward, without consuming prose.
        before = re.search(rf"\b(?:In|For|During)[ \t]+(?P<year>{_YEAR}),?[ \t]+(?:the[ \t]+(?:amount[ \t]+of[ \t]+)?)?$",
                           response[:start], re.I)
        after = re.compile(rf"[ \t]+(?:was|were|is|are|totaled|totalled)[ \t]+{amount}{endpoint}", re.I).match(response, stop)
        if before and after:
            claims.append({"schema": "direct", "subject": subject, "subject_span": (start, stop),
                           "span": (before.start(), after.end()), "match": after,
                           "year_match": before})
    unique = {}
    for claim in claims:
        unique[(claim["schema"], claim["subject_span"], claim["span"])] = claim
    return list(unique.values())


def _assertion_scope(response, claims):
    """Require the whole response to consist of finite declarative clauses.

    No arbitrary preamble or follow-up prose is permitted: it could retract,
    quote, hypothesize or otherwise change the scope of an apparent assertion.
    """
    if (re.search(r'["“”«»`?]|(?<!\w)[\'‘’]|[\'‘’](?!\w)', response)
            or re.search(r"\b(?:if|unless|suppose|supposed|assume|assumed|assuming|assumptions?|hypothetically|hypothetical|hypotheses|hypothesis|imagine|pretend|whether|false|untrue|incorrect|erroneous|counterfactual|conditional|reject\w*|den(?:y|ies|ied|ial)|not|never|may|might|would|could|should|allegedly|reportedly)\b", response, re.I)):
        return False
    regions = []
    for claim in claims:
        start, stop = claim["span"]
        boundaries = [m.end() for m in re.finditer(r"[.!?;][ \t]+|\n", response[:start])]
        frame_start = boundaries[-1] if boundaries else 0
        prefix = response[frame_start:start]
        if re.fullmatch(r"[ \t]*(?:(?:[-*•]|\d{1,2}[.)])[ \t]+)?(?:the[ \t]+(?:amount[ \t]+of[ \t]+)?)?",
                        prefix, re.I) is None:
            return False
        regions.append((frame_start, stop))
    position = 0
    for start, stop in sorted(regions) + [(len(response), len(response))]:
        gap = response[position:start]
        gap = re.sub(r"(?m)^[ \t]*\d{1,2}[.)][ \t]+$", "", gap)
        if re.fullmatch(r"\s*(?:[.;]\s*)?", gap) is None:
            return False
        position = max(position, stop)
    return bool(regions)


def _resolved_unit(observed, scope):
    possibilities = [unit for unit in scope["units"] if _merge_unit(observed, unit) == unit]
    return possibilities[0] if len(possibilities) == 1 else None


def _terms(terms):
    canonical = sorted({tuple(sorted(set(term))) for term in terms}, key=lambda t: (len(t), t))
    result = []
    for term in canonical:
        if not any(set(previous).issubset(term) for previous in result):
            result.append(term)
    return [list(term) for term in result]


def _role_tokens(roles, response, offsets, selected):
    result, ownership = [], {}
    for role_index, role in enumerate(roles):
        spans = role["spans"]
        indices = []
        for local, original in enumerate(selected):
            left, right = offsets[original]
            if not any(left < b and right > a for a, b in spans):
                continue
            if any(not response[p].isspace() and not any(a <= p < b for a, b in spans)
                   for p in range(int(left), int(right))):
                return None
            if local in ownership and ownership[local] != role_index:
                return None
            ownership[local] = role_index
            indices.append(local)
        for a, b in spans:
            if any(not response[p].isspace() and not any(offsets[selected[t], 0] <= p < offsets[selected[t], 1]
                                                       for t in indices)
                   for p in range(a, b)):
                return None
        if not indices:
            return None
        result.append(indices)
    return result


def _numeric_roles(name, amount):
    roles = [{"name": name, "spans": [amount["value_span"]]}]
    if amount["sign_spans"]:
        roles.append({"name": name + "_sign", "spans": amount["sign_spans"]})
    return roles


def _numeric_repairs(name, observed, witness, role_indices):
    """Explicit sign and magnitude are separate atoms; absent signs abstain."""
    magnitude_changed = observed["value"].copy_abs() != witness.copy_abs()
    sign_changed = (observed["value"] < 0) != (witness < 0)
    repairs = [role_indices[name]] if magnitude_changed else []
    if sign_changed and observed["sign_spans"]:
        repairs.append(role_indices[name + "_sign"])
    return repairs, sign_changed and not observed["sign_spans"]


def _arithmetic(left, right, operator):
    with localcontext() as context:
        context.prec = (max(left.adjusted(), right.adjusted(), 0)
                        - min(left.as_tuple().exponent, right.as_tuple().exponent, 0) + 3)
        return left + right if operator == "+" else left - right


def _plausibly_rounded(value, observed):
    """Visible lower precision leaves ordinary quantization unresolved."""
    if value == observed:
        return False
    quantum = Decimal(1).scaleb(observed.as_tuple().exponent)
    return (observed.as_tuple().exponent > value.as_tuple().exponent
            and _arithmetic(value, observed, "-").copy_abs() < quantum)


def _evaluate(claim, scopes):
    match = claim["match"]
    subject = claim["subject"]
    roles = [{"name": "subject", "spans": [claim["subject_span"]]}]
    if claim["schema"] == "direct":
        year_match = claim.get("year_match", match)
        year = year_match["year"]
        amount = _amount(match, "a")
        roles += [{"name": "year", "spans": [year_match.span("year")]}]
        roles += _numeric_roles("value", amount)
        units = amount["unit_spans"]
    else:
        left, right, result = [_amount(match, prefix) for prefix in ("left", "right", "result")]
        roles += [{"name": "left_year", "spans": [match.span("leftyear")]}]
        roles += _numeric_roles("left_value", left)
        roles += [{"name": "operator", "spans": [match.span("operator")]},
                  {"name": "right_year", "spans": [match.span("rightyear")]}]
        roles += _numeric_roles("right_value", right) + _numeric_roles("result", result)
        units = left["unit_spans"] + right["unit_spans"] + result["unit_spans"]
        # Unsigned parentheses could mean accounting negation or math grouping.
        if any(re.fullmatch(r"\(\d[\d,.]*\)", match[prefix + "value"])
               for prefix in ("left", "right", "result")):
            return roles, [], [], "unknown", "ambiguous_parenthesized_operand"
    if units:
        roles.append({"name": "unit", "spans": units})
    if not claim["assertion_scope"]:
        return roles, [], [], "unknown", "unresolved_assertion_scope"
    role_indices = {role["name"]: i for i, role in enumerate(roles)}
    interpretations, all_terms, witness_traces = [], [], []
    unprojectable_sign = False
    candidates = [scope for scope in scopes if subject in scope["subjects"]]
    for scope in candidates:
        if not scope["complete"]:
            return roles, [], [], "unknown", "incomplete_table_scope"
        observed_units = [amount["unit"]] if claim["schema"] == "direct" else [a["unit"] for a in (left, right, result)]
        resolved = [_resolved_unit(unit, scope) for unit in observed_units]
        if any(unit is None for unit in resolved) or len(set(resolved)) != 1:
            return roles, [], [], "unknown", "missing_or_ambiguous_unit"
        unit = resolved[0]
        observed_years = [year] if claim["schema"] == "direct" else [match["leftyear"], match["rightyear"]]
        # This gate precedes enumeration. Alternative rows never rescue missing keys.
        for condition in observed_years:
            fact = scope["facts"].get((subject, condition, unit))
            if fact is None:
                return roles, [], [], "unknown", "missing_observed_key"
            if len(fact["values"]) != 1:
                return roles, [], [], "unknown", "contradictory_observed_key"
        # Anonymous rows remain parsed source data, but have no literal subject
        # in the declared answer grammar and therefore are not repair witnesses.
        facts = [f for f in scope["facts"].values() if f["unit"] == unit and f["subject"]]
        terms, traces, compatible_seen = [], [], False
        if claim["schema"] == "direct":
            for fact in facts:
                for value, provenance in fact["values"].items():
                    term = [role_indices[name] for name, mismatch in (
                        ("subject", fact["path"] != subject), ("year", fact["year"] != year)) if mismatch]
                    numeric, unmapped = _numeric_repairs("value", amount, value, role_indices)
                    term += numeric
                    unprojectable_sign |= unmapped
                    compatible_seen |= not term and not unmapped
                    terms.append(term)
                    traces.append({"path": fact["path"], "year": fact["year"], "value": str(value),
                                   "unit": unit, "repair": term, "source": provenance,
                                   "unprojectable_sign": unmapped})
        else:
            by_path = {}
            for fact in facts:
                by_path.setdefault(fact["path"], []).append(fact)
            programs = sum(2 * sum(len(f["values"]) for f in group) ** 2
                           for group in by_path.values())
            if programs > _MAX_PROGRAMS:
                return roles, [], [], "unknown", "witness_enumeration_budget"
            for a in facts:
                for b in by_path[a["path"]]:
                    for av, ap in a["values"].items():
                        for bv, bp in b["values"].items():
                            for operator in ("+", "-"):
                                value = _arithmetic(av, bv, operator)
                                term = [role_indices[name] for name, mismatch in (
                                    ("subject", a["path"] != subject),
                                    ("left_year", a["year"] != match["leftyear"]),
                                    ("operator", operator != match["operator"]),
                                    ("right_year", b["year"] != match["rightyear"])) if mismatch]
                                program_unmapped = False
                                for name, observed, expected in (("left_value", left, av),
                                                                 ("right_value", right, bv),
                                                                 ("result", result, value)):
                                    numeric, unmapped = _numeric_repairs(name, observed, expected, role_indices)
                                    term += numeric
                                    unprojectable_sign |= unmapped
                                    program_unmapped |= unmapped
                                if (not term or term == [role_indices["result"]]) and _plausibly_rounded(value, result["value"]):
                                    return roles, [], [], "unknown", "unresolved_rounding_precision"
                                terms.append(term)
                                compatible_seen |= not term and not program_unmapped
                                traces.append({"path": a["path"], "left_year": a["year"],
                                               "right_year": b["year"], "left_value": str(av),
                                               "right_value": str(bv), "operator": operator,
                                               "result": str(value), "unit": unit, "repair": term,
                                               "source": [ap, bp], "unprojectable_sign": program_unmapped})
        if not terms:
            return roles, [], [], "unknown", "empty_witness_family"
        interpretations.append("supported" if compatible_seen else "incompatible")
        all_terms.extend(terms)
        witness_traces.append({"table": scope["table"], "table_span": scope["span"],
                               "header_context": scope["header_context"],
                               "header_context_spans": scope["header_context_spans"],
                               "witnesses": traces})
    if not interpretations:
        return roles, [], [], "unknown", "no_source_anchor"
    if "supported" in interpretations:
        status = "supported" if set(interpretations) == {"supported"} else "unknown"
        return roles, [], witness_traces, status, "compatible_witness" if status == "supported" else "mixed_interpretations"
    if unprojectable_sign:
        return roles, [], witness_traces, "unknown", "unprojectable_implicit_sign"
    terms = _terms(all_terms)
    if len(terms) > _MAX_TERMS:
        return roles, [], [], "unknown", "repair_term_budget"
    return roles, terms, witness_traces, "incompatible", "closed_incompatible"


def compile_proofs(source, response, offsets, target, question=None):
    """Return incompatible monotone-DNF factors and neutral coverage diagnostics.

    ``target`` is a Boolean mask or distinct integer indices into the original
    Nx2 character offsets. No input is mutated or removed. ``question`` is
    retained as optional context; this finite grammar never guesses its intent.
    Missing/ambiguous structures, keys, units or token projections are unknown.
    Supported claims produce no factor and do not label any token as clean.
    """
    if not isinstance(source, str) or not isinstance(response, str):
        raise TypeError("source and response must be strings")
    offsets = np.asarray(offsets)
    target = np.asarray(target)
    if offsets.ndim != 2 or offsets.shape[1] != 2 or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError("offsets must be an integer Nx2 array")
    if target.ndim != 1:
        raise ValueError("target must be a one-dimensional mask or index array")
    if target.dtype == bool:
        if len(target) != len(offsets):
            raise ValueError("target mask must match offsets")
        selected = np.flatnonzero(target)
    elif np.issubdtype(target.dtype, np.integer):
        selected = target.astype(np.int64, copy=False)
    elif target.size == 0:
        selected = np.empty(0, dtype=np.int64)
    else:
        raise ValueError("target indices must be integers")
    if (np.any(selected < 0) or np.any(selected >= len(offsets))
            or len(np.unique(selected)) != len(selected)):
        raise ValueError("target indices must be distinct and in range")
    if (np.any(offsets < 0) or np.any(offsets[:, 1] < offsets[:, 0])
            or np.any(offsets > len(response))):
        raise ValueError("offsets must be valid response character intervals")
    scopes, source_problems = _source_scopes(source)
    components, diagnostics = [], []
    claims = _claims(response, scopes)
    assertion_scope = _assertion_scope(response, claims)
    for claim in claims:
        claim["assertion_scope"] = assertion_scope
        roles, terms, witnesses, status, reason = _evaluate(claim, scopes)
        tokens = _role_tokens(roles, response, offsets, selected)
        if tokens is None:
            status, reason = "unknown", "ambiguous_token_projection"
        record = {"schema": claim["schema"], "span": list(claim["span"]),
                  "status": status, "reason": reason, "roles": roles,
                  "source_interpretations": witnesses}
        diagnostics.append(record)
        if status == "incompatible":
            components.append({"role_tokens": tokens, "repair_terms": terms,
                               "status": status, "provenance": record})
    if not diagnostics:
        diagnostics.append({"schema": None, "status": "unknown", "reason": "no_supported_claim_schema"})
    counts = Counter(record["status"] for record in diagnostics)
    reasons = Counter(record["reason"] for record in diagnostics)
    return {"components": components, "diagnostics": {
        "schema": "explicit_json_year_table_v2", "claims": diagnostics,
        "counts": {status: counts[status] for status in ("supported", "incompatible", "unknown")},
        "reasons": dict(reasons), "source_problems": source_problems,
        "source_tables": len(scopes), "source_facts": sum(len(s["facts"]) for s in scopes),
        "original_token_count": len(offsets), "scored_token_count": len(selected),
        "question_present": question is not None,
    }}
