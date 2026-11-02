"""Model-free preservation and rejection checks for H004 source representation."""

from copy import deepcopy
import json

import pytest

from experiments.research.typed_source import (
    MAX_SOURCE_CHARS, SourceRejected, decode, parse_source, prepare_source, render,
)
from experiments.research import typed_source


@pytest.mark.parametrize("source", [
    '{"a":[null,false,true,"null",{"a/b~":[]}],"empty":{}}',
    "{'a': (None, False, True), 'b': ['é', -2, 1.234567890123456789]}",
    'null', 'false', '[]', '{}', '"a string"', '(1,)',
    '{"large":123456789012345678901234567890,"decimal":1.0000000000000000001,"exp":1e999}',
])
def test_complete_roundtrip_preserves_accepted_tree(source):
    result = prepare_source(source)
    assert result["status"] == "transformed"
    tree = parse_source(source)
    assert result["parsed_tree"] == tree == decode(render(tree))
    assert result["raw_sha256"] != result["rendered_sha256"]
    assert prepare_source(source) == result


def test_types_paths_order_and_exact_number_tokens():
    source = '{"z":null,"a":false,"n":[-0,0.000000000000000000123,1E+99]}'
    result = prepare_source(source)
    items = result["parsed_tree"]["items"]
    assert [key for key, _ in items] == ["z", "a", "n"]
    assert items[0][1] == {"type": "null", "value": None}
    assert items[1][1] == {"type": "boolean", "value": False}
    assert [node["value"] for node in items[2][1]["items"]] == [
        "-0", "0.000000000000000000123", "1E+99"]
    rendered = json.loads(result["source_text"])
    assert rendered["tree"]["items"][2][1]["items"][1]["path"] == ["n", 1]
    assert "unknown" not in result["source_text"]


@pytest.mark.parametrize("source,reason", [
    ('{"x":1,"x":2}', "duplicate_key"),
    ("{'x': 1, 'x': 2}", "duplicate_key"),
    ("{1: 'integer', True: 'bool'}", "unsupported_key_type"),
    ("{False: 0, 0: 1}", "unsupported_key_type"),
    ('{"n":NaN}', "nonfinite_number"),
    ('{"n":Infinity}', "nonfinite_number"),
    ("{'n': float('nan')}", "unsupported_syntax"),
    ("{'n': 0x10}", "unsupported_number"),
    ("{'n': 1_000}", "unsupported_number"),
    ("{'s': {1, 2}}", "unsupported_syntax"),
    ("{'b': b'bytes'}", "unsupported_syntax"),
    ('plain unstructured source', "unparseable"),
    ('{"x":"\\ud800"}', "invalid_unicode"),
])
def test_rejections_keep_every_original_character(source, reason):
    result = prepare_source(source)
    assert result["status"] == "unchanged_" + reason
    assert result["source_text"] == source
    assert result["parsed_tree"] is None
    assert result["tree_sha256"] is None
    with pytest.raises(SourceRejected):
        parse_source(source)


def test_python_payload_is_never_executed(tmp_path):
    marker = tmp_path / "must_not_exist"
    source = "{'payload': __import__('pathlib').Path(" + repr(str(marker)) + ").write_text('bad')}"
    assert prepare_source(source)["status"] == "unchanged_unsupported_syntax"
    assert not marker.exists()


@pytest.mark.parametrize("source", [
    'x' * (MAX_SOURCE_CHARS + 1), '[' * 70 + '0' + ']' * 70,
    '[' + ','.join('0' for _ in range(20_001)) + ']', '"' + 'x' * 100_001 + '"',
], ids=["source_length", "depth", "nodes", "string_length"])
def test_stress_limits_retain_raw_source(source):
    result = prepare_source(source)
    assert result["status"] == "unchanged_limit_exceeded"
    assert result["source_text"] == source


def test_decoder_rejects_changed_paths_types_and_duplicates():
    document = json.loads(render(parse_source('{"a":[false]}')))
    altered = deepcopy(document)
    altered["tree"]["items"][0][1]["items"][0]["path"] = ["a", False]
    with pytest.raises(SourceRejected):
        decode(json.dumps(altered))
    altered = deepcopy(document)
    altered["tree"]["items"][0][1]["items"][0]["value"] = 0
    with pytest.raises(SourceRejected):
        decode(json.dumps(altered))
    with pytest.raises(SourceRejected):
        decode('{"format":"typed-source-v1","format":"typed-source-v1","tree":{}}')


def test_whitespace_changes_raw_identity_but_not_typed_identity():
    first = prepare_source('{"a": false}')
    second = prepare_source('  { "a" : false }\n')
    assert first["raw_sha256"] != second["raw_sha256"]
    assert first["tree_sha256"] == second["tree_sha256"]
    assert first["source_text"] == second["source_text"]


def test_render_size_guard_stops_serialization_before_path_expansion(monkeypatch):
    source = json.dumps({"k" * 100: [0] * 100}, separators=(",", ":"))
    tree = parse_source(source)
    full_size = len(render(tree))
    original_iterencode = json.JSONEncoder.iterencode
    emitted_sizes = []

    def watched_iterencode(self, value, _one_shot=False):
        for chunk in original_iterencode(self, value, _one_shot):
            emitted_sizes.append(len(chunk))
            yield chunk

    monkeypatch.setattr(typed_source, "MAX_RENDERED_CHARS", 500)
    monkeypatch.setattr(json.JSONEncoder, "iterencode", watched_iterencode)
    with pytest.raises(SourceRejected, match="limit_exceeded"):
        render(tree)
    # One chunk may cross the budget, but the remaining paths are never encoded.
    assert 500 < sum(emitted_sizes) <= 500 + max(emitted_sizes)
    assert sum(emitted_sizes) < full_size / 10
    emitted_sizes.clear()
    result = prepare_source(source)
    assert result["status"] == "unchanged_limit_exceeded"
    assert result["source_text"] == source
    assert sum(emitted_sizes) < full_size / 10
