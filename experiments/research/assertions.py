"""Bounded dependency-to-numeric-role adapter, with original character offsets.

This module consumes a frozen parser's output; it loads no model and checks no
source facts. ``candidate`` means a supported syntactic frame, not truth or a
proof of semantic scope. Unsupported roles remain explicit unknown records.
"""

from decimal import Decimal
import re


_UNSIGNED = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMBER = rf"(?:[+−-]?{_UNSIGNED}|\({_UNSIGNED}\))"
_NUMBERS = re.compile(rf"(?<![\w.,]){_NUMBER}(?!\w|[.,]\d)")
_YEAR = re.compile(r"(?:19|20)\d{2}\Z")
_RANGE = re.compile(r"(?:^|\|)TokenRange=(\d+):(\d+)(?:\||$)")
_TEMPORAL_CASE = frozenset({"in", "for", "during", "of", "since", "until", "from", "to", "by"})
_CLAUSES = frozenset({"conj", "ccomp", "xcomp", "advcl", "acl", "parataxis"})
_NOUN_PARTS = frozenset({"det", "amod", "compound", "flat", "fixed", "nmod", "case", "nummod"})
_NEGATION = frozenset({"not", "n't", "no", "never", "neither", "nor"})
_HYPOTHESIS = frozenset({"if", "unless", "would", "could", "might", "may", "suppose", "assume"})
_RETRACTION = frozenset({"lie", "false", "incorrect", "wrong", "retract", "withdraw", "correction"})
_VALUE_VERBS = frozenset({"be", "total", "equal", "reach", "amount"})


def parse_conllu(text, conllu):
    """Read UD words, requiring literal Python-character TokenRange agreement.

    Multiword surface rows and empty nodes do not enter the basic dependency
    graph. Surface ranges, when supplied, are checked; empty-node ranges are
    rejected because empty nodes have no original surface token. Their presence
    is recorded as ``nonword_rows`` on the first word of that sentence. Every
    ordinary word must have its own exact range, even inside a multiword token:
    missing ranges and normalized contractions cannot silently invent offsets.
    Malformed graphs/ranges raise ValueError, allowing the caller to retain the
    original parse and an explicit parse failure for the whole input.
    """
    result, rows, sentence_id, ordinal, seen_sentences = [], [], None, 0, set()

    def finish():
        nonlocal rows, sentence_id, ordinal
        if not rows:
            sentence_id = None
            return
        ordinal += 1
        sid = sentence_id if sentence_id is not None else str(ordinal)
        if sid in seen_sentences:
            raise ValueError(f"duplicate sentence ID: {sid}")
        seen_sentences.add(sid)
        words, nonwords = [], []
        for fields in rows:
            identifier, form, lemma, upos, _, feats, head, deprel, _, misc = fields
            match = _RANGE.search(misc)
            if "-" in identifier or "." in identifier:
                kind = "multiword" if "-" in identifier else "empty_node"
                if kind == "empty_node" and match is not None:
                    raise ValueError(f"empty node has a surface range: {sid}:{identifier}")
                if match is not None:
                    start, end = map(int, match.groups())
                    if not 0 <= start < end <= len(text) or text[start:end] != form:
                        raise ValueError(f"nonliteral TokenRange: {sid}:{identifier}")
                nonwords.append({"id": identifier, "kind": kind, "handling": "excluded_from_basic_graph"})
                continue
            if not identifier.isdigit() or not head.isdigit() or match is None:
                raise ValueError(f"missing basic ID, head or TokenRange: {sid}:{identifier}")
            start, end = map(int, match.groups())
            if not 0 <= start < end <= len(text) or text[start:end] != form:
                raise ValueError(f"nonliteral TokenRange: {sid}:{identifier}")
            words.append({"sent_id": sid, "id": int(identifier), "form": form,
                          "lemma": lemma, "upos": upos, "head": int(head),
                          "deprel": deprel, "start": start, "end": end, "feats": feats})
        identifiers = {word["id"] for word in words}
        if len(identifiers) != len(words):
            raise ValueError(f"duplicate word ID in sentence {sid}")
        by_id = {item["id"]: item for item in words}
        for word in words:
            if word["head"] and word["head"] not in identifiers:
                raise ValueError(f"missing head in sentence {sid}")
            visited, current = set(), word
            while current["head"]:
                if current["id"] in visited:
                    raise ValueError(f"dependency cycle in sentence {sid}")
                visited.add(current["id"])
                current = by_id[current["head"]]
        if words and nonwords:
            words[0]["nonword_rows"] = nonwords
        result.extend(words)
        rows, sentence_id = [], None

    for line in conllu.splitlines():
        if not line.strip():
            finish()
        elif line.startswith("# sent_id ="):
            sentence_id = line.partition("=")[2].strip()
        elif not line.startswith("#"):
            fields = line.split("\t")
            if len(fields) != 10:
                raise ValueError("CoNLL-U word row must have ten columns")
            rows.append(fields)
    finish()
    spans = sorted((word["start"], word["end"]) for word in result)
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("overlapping original word ranges")
    return result


def numeric_mentions(text):
    """Return literal signed/comma/decimal/parenthesized numbers, without units.

    ``lexical_year`` is an inventory flag only. A year is temporal in the adapter
    only when the dependency tree provides a temporal case attachment.
    """
    mentions = []
    for match in _NUMBERS.finditer(text):
        literal = match.group()
        normalized = literal.replace(",", "").replace("−", "-")
        if normalized.startswith("("):
            normalized = "-" + normalized[1:-1]
        mentions.append({"span": list(match.span()), "text": literal,
                         "value": str(Decimal(normalized)),
                         "lexical_year": bool(_YEAR.fullmatch(literal))})
    return mentions


def _base(token):
    return token["deprel"].split(":", 1)[0]


def _lemma(token):
    return (token["lemma"] if token["lemma"] != "_" else token["form"]).lower()


def _role(text, items):
    if not items:
        return None
    start, end = min(item["start"] for item in items), max(item["end"] for item in items)
    return {"span": [start, end], "text": text[start:end]}


def _ancestors(token, by_id):
    yield token
    while token["head"]:
        token = by_id[token["head"]]
        yield token


def _descendants(token, children, allowed=None):
    result = [token]
    for child in children.get(token["id"], []):
        if allowed is None or _base(child) in allowed:
            result.extend(_descendants(child, children, allowed))
    return result


def _frame(token, by_id, children):
    """Find a local predicate without crossing a separate clause boundary."""
    for current in _ancestors(token, by_id):
        attached = children.get(current["id"], [])
        if any(_base(child) in {"nsubj", "csubj", "cop"} for child in attached):
            return current
        if current["upos"] in {"VERB", "AUX"}:
            return current
        if _base(current) in _CLAUSES:
            return current
    return current


def _path(token, frame, by_id):
    result = []
    for item in _ancestors(token, by_id):
        result.append({key: item[key] for key in ("id", "form", "head", "deprel", "start", "end")})
        if item["id"] == frame["id"]:
            break
    return result


def _temporal_root(token, by_id, children):
    if not _YEAR.fullmatch(token["form"]):
        return None
    for current in _ancestors(token, by_id):
        cases = [child for child in children.get(current["id"], []) if _base(child) == "case"]
        if _base(current) in {"obl", "nmod"} and any(_lemma(case) in _TEMPORAL_CASE for case in cases):
            return current
        if current["upos"] in {"VERB", "AUX"} or _base(current) in _CLAUSES:
            break
    return None


def _record(mention, frame, token, text, by_id, reason):
    return {"amount": {"span": mention["span"], "text": mention["text"]} if mention else None,
            "predicate": _role(text, [frame]) if frame else None, "subject": None,
            "condition_year": None, "condition_years": [], "condition": None,
            "kind": "unknown", "status": "unknown", "reason": reason,
            "dependency_path": _path(token, frame, by_id) if token and frame else []}


def _math_roles(text, mentions):
    """Identify explicit infix arithmetic roles; no evaluation or value search."""
    roles = {}
    # Require an operator between adjacent literal numbers. Prose and dates do
    # not qualify; a bare hyphen between two years is conservatively ambiguous.
    groups, group = [], []
    for mention in mentions:
        if group:
            previous = group[-1]
            gap = text[previous["span"][1]:mention["span"][0]]
            if re.fullmatch(r"[\s()%$£€]*[+*/×÷=−-][\s()%$£€]*", gap):
                group.append(mention)
                continue
            if not gap and mention["text"].startswith(("-", "+", "−")):
                group.append(mention)
                continue
            if len(group) > 1:
                groups.append(group)
        group = [mention]
    if len(group) > 1:
        groups.append(group)
    for group in groups:
        start, end = group[0]["span"][0], group[-1]["span"][1]
        expression = text[start:end]
        equal = expression.count("=")
        ambiguous = all(item["lexical_year"] for item in group) or equal > 1
        equality = start + expression.index("=") if equal else None
        if equality is not None and sum(item["span"][0] > equality for item in group) != 1:
            ambiguous = True
        # In a compact "100-50", the literal matcher includes the minus in
        # the second mention. Do not silently call that a positive operand.
        if any(left["span"][1] == right["span"][0] for left, right in zip(group, group[1:])):
            ambiguous = True
        for mention in group:
            kind = "result" if equality is not None and mention["span"][0] > equality else "operand"
            roles[tuple(mention["span"])] = {
                "kind": kind, "expression": {"span": [start, end], "text": expression},
                "reason": "ambiguous_arithmetic_scope" if ambiguous else "explicit_infix_role",
                "status": "unknown" if ambiguous else "candidate"}
    return roles


def numeric_assertions(text, tokens):
    """Extract local quantity/temporal roles from positive declarative frames.

    Copular predicates and a small explicit value-verb family are supported.
    The subject and year must belong to the same local predicate as the amount;
    no antecedents or coordinated arguments are inherited. Syntactic candidates
    have unproved scope/unit/coreference obligations in ``limitations``. Arithmetic
    records describe literal operand/result roles only, never a direct cell claim.
    Every numeric mention yields a record or an explicit temporal role on a record.
    """
    mentions = numeric_mentions(text)
    math = _math_roles(text, mentions)
    records, sentences = [], {}
    for token in tokens:
        if text[token["start"]:token["end"]] != token["form"]:
            raise ValueError("token text does not match original character range")
        sentences.setdefault(token["sent_id"], []).append(token)
    global_retraction = any(_lemma(token) in _RETRACTION for token in tokens)
    consumed = set()
    for sentence in sentences.values():
        by_id = {token["id"]: token for token in sentence}
        children = {}
        for token in sentence:
            children.setdefault(token["head"], []).append(token)
        start, end = min(token["start"] for token in sentence), max(token["end"] for token in sentence)
        local = [mention for mention in mentions if start <= mention["span"][0] < end]
        years, amounts = [], []
        for mention in local:
            overlap = [token for token in sentence if token["start"] < mention["span"][1] and token["end"] > mention["span"][0]]
            numeric = [token for token in overlap if any(character.isdigit() for character in token["form"])]
            if len(numeric) != 1:
                record = _record(mention, None, None, text, by_id, "ambiguous_numeric_tokenization")
                records.append(record)
                consumed.add(tuple(mention["span"]))
                continue
            token = numeric[0]
            outside = text[token["start"]:mention["span"][0]] + text[mention["span"][1]:token["end"]]
            if any(character.isdigit() for character in outside):
                records.append(_record(mention, None, None, text, by_id, "partial_numeric_parser_token"))
                consumed.add(tuple(mention["span"]))
                continue
            frame = _frame(token, by_id, children)
            temporal = _temporal_root(token, by_id, children)
            item = (mention, token, frame, temporal)
            (years if temporal else amounts).append(item)
        scope_reason = None
        lemmas = {_lemma(token) for token in sentence}
        if global_retraction:
            scope_reason = "retraction_or_nonliteral_scope_cue"
        elif lemmas & _NEGATION:
            scope_reason = "negation_scope_unresolved"
        elif lemmas & _HYPOTHESIS or any("Mood=Imp" in token.get("feats", "") for token in sentence):
            scope_reason = "hypothetical_or_directive_scope_unresolved"
        elif "?" in text[start:end]:
            scope_reason = "interrogative_scope_unresolved"
        elif any(character in text[start:end] for character in ('"', '“', '”', '‘', '’')) or re.search(r"(?<!\w)'|'(?!\w)", text[start:end]):
            scope_reason = "quotation_scope_unresolved"
        for mention, token, frame, _ in amounts:
            consumed.add(tuple(mention["span"]))
            record = _record(mention, frame, token, text, by_id, "unsupported_quantity_frame")
            record["limitations"] = ["syntax_candidate_only", "unit_scope_unverified", "scope_probe_not_exhaustive"]
            local_years = [item for item in years if item[2]["id"] == frame["id"]]
            record["condition_years"] = [{"span": item[0]["span"], "text": item[0]["text"]} for item in local_years]
            if len(local_years) == 1:
                record["condition_year"] = record["condition_years"][0]
                record["condition"] = _role(text, _descendants(local_years[0][3], children, _NOUN_PARTS))
                record["condition_dependency_path"] = _path(local_years[0][1], frame, by_id)
            attached = children.get(frame["id"], [])
            subjects = [child for child in attached if _base(child) == "nsubj"]
            subject_tokens = _descendants(subjects[0], children, _NOUN_PARTS) if len(subjects) == 1 else []
            # Temporal modifiers are separate roles, never part of the subject.
            temporal_ids = {part["id"] for item in local_years for part in _descendants(item[3], children)}
            subject_tokens = [part for part in subject_tokens if part["id"] not in temporal_ids]
            record["subject"] = _role(text, subject_tokens)
            ordered_subject = sorted(subject_tokens, key=lambda part: part["start"])
            discontinuous_subject = any(text[left["end"]:right["start"]].strip()
                                        for left, right in zip(ordered_subject, ordered_subject[1:]))
            peers = [item for item in amounts if item[2]["id"] == frame["id"]]
            copular = any(_base(child) == "cop" for child in attached)
            value_verb = frame["upos"] in {"VERB", "AUX"} and _lemma(frame) in _VALUE_VERBS
            embedded = any(_base(part) in {"ccomp", "xcomp", "advcl", "acl"}
                           for part in _ancestors(frame, by_id))
            unresolved_quantity_conj = any(_base(child) == "conj" and not any(
                _base(part) in {"nsubj", "csubj"} for part in children.get(child["id"], []))
                for child in attached)
            if scope_reason:
                record["reason"] = scope_reason
            elif embedded:
                record["reason"] = "embedded_proposition_scope_unresolved"
            elif tuple(mention["span"]) in math:
                record.update(math[tuple(mention["span"])])
            elif any(part["upos"] == "PRON" for part in subject_tokens):
                record["reason"] = "anaphoric_subject_unresolved"
            elif any(_base(part) == "conj" for subject in subjects for part in _descendants(subject, children)):
                record["reason"] = "coordinated_subject_unresolved"
            elif not subjects:
                record["reason"] = "missing_local_subject"
            elif len(subjects) != 1:
                record["reason"] = "ambiguous_local_subject"
            elif discontinuous_subject:
                record["reason"] = "discontinuous_subject_span_unresolved"
            elif len(local_years) > 1:
                record["reason"] = "multiple_temporal_conditions_unresolved"
            elif unresolved_quantity_conj:
                record["reason"] = "coordinated_quantity_scope_unresolved"
            elif len(peers) != 1:
                record["reason"] = "multiple_quantities_in_predicate_unresolved"
            elif any(part["id"] == token["id"] for part in subject_tokens):
                record["reason"] = "quantity_in_subject_not_value"
            elif any(_base(child) in {"case", "advmod"} for child in attached):
                record["reason"] = "quantity_modifier_scope_unresolved"
            elif copular or value_verb:
                record.update(kind="assertion", status="candidate", reason="local_positive_quantity_frame")
            records.append(record)
        for mention, token, frame, _ in years:
            consumed.add(tuple(mention["span"]))
            if any(item[2]["id"] == frame["id"] for item in amounts):
                continue
            record = _record(None, frame, token, text, by_id, scope_reason or "temporal_predicate_without_amount")
            record["condition_year"] = {"span": mention["span"], "text": mention["text"]}
            record["condition_years"] = [record["condition_year"]]
            records.append(record)
    for mention in mentions:
        if tuple(mention["span"]) not in consumed:
            records.append(_record(mention, None, None, text, {}, "numeric_mention_without_parser_token"))
    return sorted(records, key=lambda record: (record["amount"] or record["condition_year"])["span"])
