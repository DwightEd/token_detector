"""Source-only occurrence identities, with no semantic binding or risk inference.

Serialized field paths license a structural relation only. Prose relationships,
string contents, entity aliases and computed values remain explicitly unknown.
The inventory never accepts a response or an annotation.
"""

import ast
from collections import defaultdict
import hashlib
import json
import re

from .table_bindings import ATOMS
from .typed_source import MAX_SOURCE_CHARS, prepare_source


CONTRACT = "relation-source-occurrence-inventory-v1"
OFFSET_UNIT = "Python Unicode code points; half-open"
_CONTAINERS = {"object", "list", "tuple"}


def _hash(text):
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _identity(value):
    return _hash(json.dumps(value, ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), allow_nan=False))


def _python_locations(source, tree):
    """Map validated AST nodes; Python's columns count UTF-8 bytes, not chars."""
    text = source.strip()
    shift = len(source) - len(source.lstrip())
    # Unicode line separators inside a literal do not increment AST line numbers.
    lines = re.split(r"(?<=\n)|(?<=\r)(?!\n)", text)
    starts, cursor = [], 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line)

    def span(node):
        def coordinate(line, byte_column):
            prefix = lines[line - 1].encode("utf-8")[:byte_column].decode("utf-8")
            return shift + starts[line - 1] + len(prefix)
        return [coordinate(node.lineno, node.col_offset),
                coordinate(node.end_lineno, node.end_col_offset)]

    locations = {}

    def walk(node, typed, path, key_span=None):
        locations[path] = {"span": span(node), "key_span": key_span}
        if typed["type"] == "object":
            for key_node, child_node, (key, child) in zip(
                    node.keys, node.values, typed["items"], strict=True):
                walk(child_node, child, path + (key,), span(key_node))
        elif typed["type"] in {"list", "tuple"}:
            for index, (child_node, child) in enumerate(zip(node.elts, typed["items"], strict=True)):
                walk(child_node, child, path + (index,))

    walk(ast.parse(text, mode="eval").body, tree, ())
    return locations


def _json_locations(source, tree):
    """Use the already validated tree to locate original JSON tokens verbatim."""
    decoder, locations = json.JSONDecoder(), {}

    def whitespace(position):
        while position < len(source) and source[position] in " \t\r\n":
            position += 1
        return position

    def punctuation(position, literal):
        position = whitespace(position)
        if source[position:position + 1] != literal:
            raise ValueError("Validated JSON location mismatch")
        return position + 1

    def walk(node, path, position, key_span=None):
        start = whitespace(position)
        kind = node["type"]
        if kind == "object":
            end = punctuation(start, "{")
            for index, (key, child) in enumerate(node["items"]):
                if index:
                    end = punctuation(end, ",")
                key_start = whitespace(end)
                parsed_key, key_end = decoder.raw_decode(source, key_start)
                if parsed_key != key:
                    raise ValueError("Validated JSON key mismatch")
                end = walk(child, path + (key,), punctuation(key_end, ":"),
                           [key_start, key_end])
            end = punctuation(end, "}")
        elif kind == "list":
            end = punctuation(start, "[")
            for index, child in enumerate(node["items"]):
                if index:
                    end = punctuation(end, ",")
                end = walk(child, path + (index,), end)
            end = punctuation(end, "]")
        else:
            _, end = decoder.raw_decode(source, start)
            if kind in {"integer", "decimal"} and source[start:end] != node["value"]:
                raise ValueError("Validated JSON numeric spelling mismatch")
        locations[path] = {"span": [start, end], "key_span": key_span}
        return end

    if whitespace(walk(tree, (), 0)) != len(source):
        raise ValueError("Validated JSON trailing content mismatch")
    return locations


def source_inventory(source, source_id):
    """Preserve exact source fields and lexical numerals, allowing semantic unknown.

    ``fields`` are serialized scalar occurrences, not inferred natural-language
    facts. A full path (including every array index) is part of their identity.
    Parent scalar context is preserved without interpreting it as time or owner.
    ``lexical_occurrences`` are raw numeral-shaped substrings only: dates, passage
    numbers and escape digits remain possible. There is no numeric normalization.
    """
    if not isinstance(source, str) or not isinstance(source_id, str) or not source_id:
        raise ValueError("A source string and nonempty original source ID are required")
    source_sha = _hash(source)
    result = dict(schema=CONTRACT, source_id=source_id, source_sha256=source_sha,
                  characters=len(source), offset_unit=OFFSET_UNIT, syntax=None,
                  status="semantic_unknown", reason=None, fields=[], scopes=[],
                  lexical_occurrences=[], unknown_regions=[],
                  semantic_status="not_assessed", annotation_reads=0, model_calls=0)
    if len(source) > MAX_SOURCE_CHARS:
        result.update(reason="limit_exceeded", lexical_status="not_processed_limit",
                      unknown_regions=[dict(span=[0, len(source)], reason="source_limit_exceeded")])
        return result
    result["lexical_status"] = "raw_lexemes_only"
    for match in ATOMS.finditer(source):
        coordinate = list(match.span())
        result["lexical_occurrences"].append(dict(
            occurrence_id=_identity([CONTRACT, source_id, source_sha, "lexical", coordinate]),
            span=coordinate, raw_text=match[0], kind="raw_numeral_lexeme",
            relation_license="unknown", semantic_status="not_assessed"))
    prepared = prepare_source(source)
    if prepared["status"] != "transformed":
        result["reason"] = prepared["reason"]
        result["unknown_regions"] = [dict(span=[0, len(source)],
                                          reason="unlicensed_source_" + prepared["reason"])]
        return result
    tree, syntax = prepared["parsed_tree"], prepared["syntax"]
    result["syntax"] = syntax
    if tree["type"] not in _CONTAINERS:
        result["reason"] = "scalar_root_has_no_serialized_relation"
        result["unknown_regions"] = [dict(span=[0, len(source)], reason=result["reason"])]
        return result
    locations = (_json_locations(source, tree) if syntax == "json"
                 else _python_locations(source, tree))
    nodes, field_by_path = {}, {}

    def walk(node, path):
        nodes[path] = node
        position = locations[path]
        if node["type"] in _CONTAINERS:
            result["scopes"].append(dict(path=list(path), kind=node["type"], span=position["span"]))
            children = (node["items"] if node["type"] == "object"
                        else enumerate(node["items"]))
            for key, child in children:
                walk(child, path + (key,))
            return
        start, stop = position["span"]
        fact_id = _identity([CONTRACT, source_id, source_sha, "field", list(path), [start, stop]])
        field = dict(occurrence_id=fact_id, path=list(path), owner_path=list(path[:-1]),
                     predicate_segment=path[-1], scalar_type=node["type"], value=node["value"],
                     span=[start, stop], raw_text=source[start:stop], key_span=position["key_span"],
                     relation_license="explicit_serialized_field" if isinstance(path[-1], str)
                     else "explicit_serialized_array_item", semantic_status="not_assessed",
                     entity=None, predicate=None, time=None, unit=None, polarity=None,
                     ancestor_context=[])
        if position["key_span"] is not None:
            left, right = position["key_span"]
            field["raw_key"] = source[left:right]
        result["fields"].append(field)
        field_by_path[path] = field
        if node["type"] == "string":
            result["unknown_regions"].append(dict(span=[start, stop],
                reason="string_content_relations_not_parsed", occurrence_id=fact_id))

    walk(tree, ())
    for path, field in field_by_path.items():
        for length in range(len(path)):
            ancestor = path[:length]
            node = nodes[ancestor]
            if node["type"] != "object":
                continue
            siblings = [field_by_path[ancestor + (key,)]["occurrence_id"]
                        for key, child in node["items"]
                        if child["type"] not in _CONTAINERS and ancestor + (key,) != path]
            field["ancestor_context"].append(dict(scope_path=list(ancestor),
                                                  scalar_field_ids=siblings))
    result.update(status="structural_fields_only", reason="semantic_relations_not_inferred")
    return result


def structural_cohorts(inventory):
    """Group potential controls; no group certifies semantic exchangeability.

    Array indices are generalized only inside a separate cohort key; the original
    full paths and distinct occurrence IDs are preserved in every member.
    """
    if inventory.get("schema") != CONTRACT:
        raise ValueError("Unsupported source inventory contract")
    groups = defaultdict(list)
    for field in inventory["fields"]:
        path = field["path"]
        groups[("scalar_siblings", _identity(path[:-1]), field["scalar_type"])].append(field)
        if any(type(part) is int for part in path):
            template = [{"array_index": "any"} if type(part) is int else {"key": part} for part in path]
            groups[("repeated_array_field", _identity(template), field["scalar_type"])].append(field)
        groups[("repeated_typed_literal", _identity([field["scalar_type"], field["value"]]))].append(field)
    result = []
    for key, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        result.append(dict(cohort_id=_identity([CONTRACT, inventory["source_id"], inventory["source_sha256"], key]),
                           basis=key[0], members=[dict(occurrence_id=f["occurrence_id"], path=f["path"])
                                                for f in members],
                           semantic_exchangeability="unknown", response_support="not_assessed"))
    return result


def align_source_spans(inventory, source, model_text, source_start, offsets,
                       token_ids, expected_token_ids, source_mask):
    """Map source occurrences to a supplied *original-token-matched* model axis.

    The caller must reproduce actual tokenizer/chat offsets; cached IDs or masks
    alone do not establish character offsets. Matching IDs, exact raw substring
    and source mask are all required. Boundary-crossing tokens stay explicit;
    this function does not allocate them to one fact or manufacture a partition.
    """
    if (inventory.get("schema") != CONTRACT or not isinstance(source, str) or not source
            or _hash(source) != inventory.get("source_sha256")
            or not isinstance(model_text, str) or type(source_start) is not int
            or source_start < 0 or model_text[source_start:source_start + len(source)] != source):
        raise ValueError("Source/model character identity changed")
    token_ids, expected_token_ids, offsets, source_mask = (
        list(value) for value in (token_ids, expected_token_ids, offsets, source_mask))
    if (any(type(value) is not int or value < 0 for value in token_ids + expected_token_ids)
            or token_ids != expected_token_ids or not len(offsets) == len(token_ids) == len(source_mask)):
        raise ValueError("Original token ID, offset or mask axis changed")
    checked = []
    for coordinate in offsets:
        if (len(coordinate) != 2 or any(type(part) is not int for part in coordinate)
                or not 0 <= coordinate[0] <= coordinate[1] <= len(model_text)):
            raise ValueError("Invalid model character offset")
        checked.append(tuple(coordinate))
    expected_mask = [left < source_start + len(source) and right > source_start and right > left
                     for left, right in checked]
    if any(type(value) is not bool for value in source_mask) or source_mask != expected_mask:
        raise ValueError("Original source mask and character offsets disagree")
    mapped = []
    for kind in ("fields", "lexical_occurrences"):
        for item in inventory[kind]:
            coordinate = item["span"]
            if (len(coordinate) != 2 or any(type(part) is not int for part in coordinate)
                    or not 0 <= coordinate[0] < coordinate[1] <= len(source)
                    or source[coordinate[0]:coordinate[1]] != item["raw_text"]):
                raise ValueError("Inventory occurrence character identity changed")
            left, right = (source_start + part for part in item["span"])
            overlap = [index for index, (start, stop) in enumerate(checked)
                       if start < right and stop > left and stop > start]
            crossing = [index for index in overlap if checked[index][0] < left or checked[index][1] > right]
            exclusive = [index for index in overlap if index not in crossing]
            mapped.append(dict(occurrence_id=item["occurrence_id"], kind=kind,
                               source_span=item["span"], model_span=[left, right],
                               overlap_tokens=overlap, contained_tokens=exclusive,
                               boundary_crossing_tokens=crossing,
                               status="unmapped" if not overlap else
                               ("boundary_overlap" if crossing else "contained")))
    return dict(schema=CONTRACT, source_sha256=inventory["source_sha256"],
                token_ids_sha256=_identity(token_ids), offsets_sha256=_identity(offsets),
                source_start=source_start, tokens=len(token_ids), occurrences=mapped,
                status="character_overlap_and_token_ids_matched", semantic_status="not_assessed",
                limitation="Caller must bind supplied tokenizer offsets to the sealed original cache.")
