"""Lossless source occurrence/axis contracts, without natural-data truth labels."""

from copy import deepcopy
import json

import pytest

from experiments.research.relation_inventory import (
    align_source_spans, source_inventory, structural_cohorts,
)


def _fields(inventory):
    return {tuple(field["path"]): field for field in inventory["fields"]}


@pytest.mark.parametrize("source", [
    '{"name":"甲","records":[{"year":2024,"value":100},{"year":2025,"value":100}]}',
    "  {'name': '甲', 'records': ({'year': 2024, 'value': 100}, {'year': 2025, 'value': 100})}  ",
])
def test_occurrences_keep_duplicate_values_complete_paths_and_context(source):
    inventory = source_inventory(source, "source-A")
    fields = _fields(inventory)
    first, second = (fields[("records", index, "value")] for index in (0, 1))
    assert first["value"] == second["value"] == "100"
    assert first["occurrence_id"] != second["occurrence_id"]
    assert first["owner_path"] == ["records", 0]
    assert second["owner_path"] == ["records", 1]
    for index, field in enumerate((first, second)):
        context = {tuple(c["scope_path"]): c["scalar_field_ids"] for c in field["ancestor_context"]}
        assert fields[("records", index, "year")]["occurrence_id"] in context[("records", index)]
        assert fields[("name",)]["occurrence_id"] in context[()]
        assert field["entity"] is field["predicate"] is field["time"] is field["unit"] is None
    assert inventory["semantic_status"] == "not_assessed"
    assert inventory["annotation_reads"] == inventory["model_calls"] == 0


def test_python_utf8_columns_escaped_strings_and_multiline_literals():
    source = "\n {'甲': '公司\\\'甲', '值': -1.50,\n '故事': '''甲\n乙''', '续': 'a' 'b'} \n"
    fields = _fields(source_inventory(source, "utf8"))
    assert fields[("甲",)]["raw_text"] == "'公司\\\'甲'"
    assert fields[("甲",)]["value"] == "公司'甲"
    assert fields[("值",)]["raw_text"] == "-1.50"
    assert fields[("故事",)]["raw_text"] == "'''甲\n乙'''"
    assert fields[("故事",)]["value"] == "甲\n乙"
    assert fields[("续",)]["raw_text"] == "'a' 'b'"
    assert fields[("续",)]["value"] == "ab"
    for field in fields.values():
        assert source[slice(*field["span"])] == field["raw_text"]
        assert source[slice(*field["key_span"])] == field["raw_key"]


def test_json_types_escapes_precision_and_null_stay_distinct():
    source = r'{"a":"\u7532","false":false,"missing":null,"x":1E+999,"y":1.000000000000000001}'
    inventory = source_inventory(source, "typed")
    fields = _fields(inventory)
    assert fields[("a",)]["value"] == "甲"
    assert fields[("a",)]["raw_text"] == r'"\u7532"'
    assert fields[("false",)]["value"] is False
    assert fields[("false",)]["scalar_type"] == "boolean"
    assert fields[("missing",)]["value"] is None
    assert fields[("missing",)]["scalar_type"] == "null"
    assert fields[("x",)]["value"] == "1E+999"
    assert fields[("y",)]["value"] == "1.000000000000000001"
    assert json.loads(json.dumps(inventory)) == inventory


@pytest.mark.parametrize("separator", ["\u2028", "\x85", "\v", "\r\n", "\r"])
def test_python_location_rows_use_only_python_newlines(separator):
    source = "{'甲': '''one" + separator + "two''', '乙': 200}"
    fields = _fields(source_inventory(source, "line-separators"))
    assert fields[("乙",)]["raw_text"] == "200"
    for field in fields.values():
        assert source[slice(*field["span"])] == field["raw_text"]


@pytest.mark.parametrize("source,reason", [
    ('{"value":100,"value":200}', "duplicate_key"),
    ("{'value': 100, 'value': 200}", "duplicate_key"),
    ("A earned 100; B earned 200.", "unparseable"),
    ("100", "scalar_root_has_no_serialized_relation"),
    ("{'payload': __import__('os').getcwd()}", "unsupported_syntax"),
])
def test_unlicensed_sources_keep_full_unknown_without_nearest_owner(source, reason):
    inventory = source_inventory(source, "unknown")
    assert inventory["fields"] == []
    assert inventory["status"] == "semantic_unknown"
    assert inventory["reason"] == reason
    assert inventory["unknown_regions"][0]["span"] == [0, len(source)]
    assert all(item["relation_license"] == "unknown" for item in inventory["lexical_occurrences"])


def test_string_contents_and_lexical_digits_do_not_become_facts():
    source = '{"review":"A earned 100; B earned 200.","code":"\\u0031"}'
    inventory = source_inventory(source, "strings")
    assert len(inventory["fields"]) == 2
    assert len(inventory["unknown_regions"]) == 2
    assert [item["raw_text"] for item in inventory["lexical_occurrences"]] == ["100", "200", "0031"]
    assert all(item["semantic_status"] == "not_assessed" for item in inventory["lexical_occurrences"])


def test_potential_cohorts_retain_same_value_members_without_certifying_a_swap():
    inventory = source_inventory('{"reviews":[{"stars":5,"date":2024},{"stars":5,"date":2025}]}', "cohorts")
    cohorts = structural_cohorts(inventory)
    stars = [c for c in cohorts if c["basis"] == "repeated_array_field"
             and c["members"][0]["path"][-1] == "stars"]
    assert len(stars) == 1
    assert [m["path"] for m in stars[0]["members"]] == [["reviews", 0, "stars"], ["reviews", 1, "stars"]]
    assert len({m["occurrence_id"] for m in stars[0]["members"]}) == 2
    assert all(c["semantic_exchangeability"] == "unknown" for c in cohorts)
    assert all(c["response_support"] == "not_assessed" for c in cohorts)


def _alignment_fixture():
    source = "A 100 B 200"
    model_text = "<" + source + ">"
    # Duplicate offsets are valid BPE observations. The first numeric token also
    # contains the preceding space and therefore cannot be assigned exclusively.
    offsets = [(0, 0), (0, 1), (1, 2), (2, 6), (2, 6), (6, 7), (7, 8), (8, 9), (9, 12), (12, 13)]
    ids = list(range(len(offsets)))
    mask = [False, False, True, True, True, True, True, True, True, False]
    return dict(inventory=source_inventory(source, "axis"), source=source, model_text=model_text,
                source_start=1, offsets=offsets, token_ids=ids, expected_token_ids=ids, source_mask=mask)


def test_token_alignment_matches_original_axis_and_preserves_boundary_ambiguity():
    args = _alignment_fixture()
    mapping = align_source_spans(**args)
    first, second = mapping["occurrences"]
    assert first["source_span"] == [2, 5]
    assert first["overlap_tokens"] == [3, 4]
    assert first["contained_tokens"] == []
    assert first["boundary_crossing_tokens"] == [3, 4]
    assert first["status"] == "boundary_overlap"
    assert second["overlap_tokens"] == second["contained_tokens"] == [8]
    assert second["boundary_crossing_tokens"] == []
    assert mapping["semantic_status"] == "not_assessed"


@pytest.mark.parametrize("change", ["source", "offset", "token", "mask", "occurrence", "bool_id"])
def test_alignment_rejects_identity_and_axis_errors(change):
    args = deepcopy(_alignment_fixture())
    if change == "source":
        args["source"] = "A 200 B 100"
    elif change == "offset":
        args["offsets"][0] = (-1, 0)
    elif change == "token":
        args["expected_token_ids"][0] = 99
        args["token_ids"] = list(range(len(args["offsets"])))
    elif change == "mask":
        args["source_mask"][0] = True
    elif change == "bool_id":
        args["token_ids"][0] = False
    else:
        args["inventory"]["lexical_occurrences"][0]["raw_text"] = "200"
    with pytest.raises(ValueError):
        align_source_spans(**args)
