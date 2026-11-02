"""Pure, bounded, lossless typed-source representation (no model or file access).

The accepted language is strict JSON, or Python literal dictionaries with string
keys, lists, tuples, strings, booleans, None and decimal numeric tokens. Python
expressions, sets, bytes, nondecimal numbers, and duplicate keys are rejected.
Numeric token spelling is retained; no number passes through binary float.
"""

import ast
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re


SOURCE_TRANSFORM_ID = "typed-source-v1"
MAX_SOURCE_CHARS = 200_000
MAX_RENDERED_CHARS = 8_000_000
MAX_DEPTH = 64
MAX_NODES = 20_000
MAX_STRING_CHARS = 100_000
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z")


class SourceRejected(ValueError):
    """An explicit, deterministic reason to retain the complete raw source."""

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class _Number(str):
    pass


class _Pairs(list):
    pass


def _hash(text):
    # surrogatepass defines an identity even for rejected invalid Unicode input.
    return hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _bounded_json(value):
    encoder = json.JSONEncoder(ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    chunks, length = [], 0
    for chunk in encoder.iterencode(value):
        length += len(chunk)
        if length > MAX_RENDERED_CHARS:
            raise SourceRejected("limit_exceeded")
        chunks.append(chunk)
    return "".join(chunks)


def _string(value):
    if type(value) is not str:
        raise SourceRejected("unsupported_key_type")
    if len(value) > MAX_STRING_CHARS:
        raise SourceRejected("limit_exceeded")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise SourceRejected("invalid_unicode") from None
    return value


def _number(token):
    token = str(token)
    if not _NUMBER.fullmatch(token):
        raise SourceRejected("unsupported_number")
    try:
        finite = Decimal(token).is_finite()
    except InvalidOperation:
        finite = False
    if not finite:
        raise SourceRejected("nonfinite_number")
    return {"type": "decimal" if any(c in token for c in ".eE") else "integer",
            "value": token}


def _pairs(items):
    seen = set()
    for key, _ in items:
        if key in seen:
            raise SourceRejected("duplicate_key")
        seen.add(key)
    return _Pairs(items)


def _nonfinite(_):
    raise SourceRejected("nonfinite_number")


def _budget():
    count = 0

    def check(depth):
        nonlocal count
        count += 1
        if depth > MAX_DEPTH or count > MAX_NODES:
            raise SourceRejected("limit_exceeded")

    return check


def _parse(source):
    if type(source) is not str:
        raise TypeError("source must be a string")
    if len(source) > MAX_SOURCE_CHARS:
        raise SourceRejected("limit_exceeded")
    check = _budget()

    def from_json(value, depth=0):
        check(depth)
        if isinstance(value, _Number):
            return _number(value)
        if isinstance(value, _Pairs):
            return {"type": "object", "items": [
                [_string(key), from_json(child, depth + 1)] for key, child in value]}
        if type(value) is list:
            return {"type": "list", "items": [from_json(v, depth + 1) for v in value]}
        kind = {str: "string", bool: "boolean", type(None): "null"}[type(value)]
        return {"type": kind, "value": _string(value) if kind == "string" else value}

    try:
        raw = json.loads(source, parse_int=_Number, parse_float=_Number,
                         parse_constant=_nonfinite, object_pairs_hook=_pairs)
    except json.JSONDecodeError:
        pass
    except RecursionError:
        raise SourceRejected("limit_exceeded") from None
    else:
        return from_json(raw), "json"

    try:
        expression = ast.parse(source.strip(), mode="eval")
    except (SyntaxError, ValueError):
        raise SourceRejected("unparseable") from None
    except (RecursionError, MemoryError):
        raise SourceRejected("limit_exceeded") from None
    literal_source = source.strip()

    def from_python(node, depth=0):
        check(depth)
        if isinstance(node, ast.Dict):
            items, seen = [], set()
            for key, value in zip(node.keys, node.values):
                if not isinstance(key, ast.Constant) or type(key.value) is not str:
                    raise SourceRejected("unsupported_key_type")
                name = _string(key.value)
                if name in seen:
                    raise SourceRejected("duplicate_key")
                seen.add(name)
                items.append([name, from_python(value, depth + 1)])
            return {"type": "object", "items": items}
        if isinstance(node, (ast.List, ast.Tuple)):
            return {"type": "list" if isinstance(node, ast.List) else "tuple",
                    "items": [from_python(v, depth + 1) for v in node.elts]}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float):
                return _number(ast.get_source_segment(literal_source, node))
        if isinstance(node, ast.Constant):
            value = node.value
            if type(value) in (int, float):
                return _number(ast.get_source_segment(literal_source, node))
            if type(value) in (str, bool, type(None)):
                kind = {str: "string", bool: "boolean", type(None): "null"}[type(value)]
                return {"type": kind, "value": _string(value) if kind == "string" else value}
        raise SourceRejected("unsupported_syntax")

    return from_python(expression.body), "python_literal"


def parse_source(source):
    """Return the canonical typed tree, or raise SourceRejected without execution."""
    return _parse(source)[0]


def _validated_tree(tree, with_paths=False):
    check = _budget()

    def walk(node, path):
        check(len(path))
        if type(node) is not dict or type(node.get("type")) is not str:
            raise SourceRejected("invalid_typed_tree")
        kind = node["type"]
        container = kind in ("object", "list", "tuple")
        expected = {"type", "items" if container else "value"}
        if with_paths:
            expected.add("path")
            if type(node.get("path")) is not list or _json(node["path"]) != _json(path):
                raise SourceRejected("invalid_path")
        if set(node) != expected:
            raise SourceRejected("invalid_typed_tree")
        result = {"type": kind}
        if container:
            if type(node["items"]) is not list:
                raise SourceRejected("invalid_typed_tree")
            items, seen = [], set()
            for index, item in enumerate(node["items"]):
                if kind == "object":
                    if type(item) is not list or len(item) != 2:
                        raise SourceRejected("invalid_typed_tree")
                    name = _string(item[0])
                    if name in seen:
                        raise SourceRejected("duplicate_key")
                    seen.add(name)
                    items.append([name, walk(item[1], path + [name])])
                else:
                    items.append(walk(item, path + [index]))
            result["items"] = items
        else:
            value = node["value"]
            valid = ((kind == "string" and type(value) is str)
                     or (kind == "boolean" and type(value) is bool)
                     or (kind == "null" and value is None))
            if kind in ("integer", "decimal") and type(value) is str:
                valid = _number(value)["type"] == kind
            if not valid:
                raise SourceRejected("invalid_typed_tree")
            result["value"] = _string(value) if kind == "string" else value
        return result

    return walk(tree, [])


def render(tree):
    """Render a validated typed tree with explicit paths and no semantic rules."""
    tree = _validated_tree(tree)

    def paths(node, path):
        node = dict(node, path=path)
        if node["type"] == "object":
            node["items"] = [[key, paths(child, path + [key])] for key, child in node["items"]]
        elif node["type"] in ("list", "tuple"):
            node["items"] = [paths(child, path + [i]) for i, child in enumerate(node["items"])]
        return node

    return _bounded_json({"format": SOURCE_TRANSFORM_ID, "tree": paths(tree, [])})


def decode(rendered):
    """Validate the representation and recover the complete canonical typed tree."""
    if type(rendered) is not str:
        raise TypeError("rendered must be a string")
    if len(rendered) > MAX_RENDERED_CHARS:
        raise SourceRejected("limit_exceeded")
    try:
        envelope = json.loads(rendered, object_pairs_hook=_pairs, parse_constant=_nonfinite)
        # Convert pairs while retaining duplicate detection performed above.
        def objects(value):
            if isinstance(value, _Pairs):
                return {key: objects(child) for key, child in value}
            if type(value) is list:
                return [objects(child) for child in value]
            return value
        envelope = objects(envelope)
    except (json.JSONDecodeError, ValueError):
        raise SourceRejected("invalid_rendering") from None
    except RecursionError:
        raise SourceRejected("limit_exceeded") from None
    if (type(envelope) is not dict or set(envelope) != {"format", "tree"}
            or envelope["format"] != SOURCE_TRANSFORM_ID):
        raise SourceRejected("invalid_rendering")
    return _validated_tree(envelope["tree"], with_paths=True)


def prepare_source(source):
    """Transform accepted sources; retain rejected sources verbatim with identity."""
    if type(source) is not str:
        raise TypeError("source must be a string")
    result = {"status": "transformed", "source_text": source, "parsed_tree": None,
              "syntax": None, "transformation_id": SOURCE_TRANSFORM_ID,
              "raw_sha256": _hash(source), "tree_sha256": None,
              "rendered_sha256": _hash(source), "reason": None}
    try:
        tree, syntax = _parse(source)
        rendered = render(tree)
        if decode(rendered) != tree:
            raise RuntimeError("typed source round trip violated")
    except (SourceRejected, RecursionError) as exc:
        reason = exc.reason if isinstance(exc, SourceRejected) else "limit_exceeded"
        result.update(status="unchanged_" + reason, reason=reason)
    else:
        result.update(source_text=rendered, parsed_tree=tree, syntax=syntax,
                      tree_sha256=_hash(_json(tree)), rendered_sha256=_hash(rendered))
    return result
