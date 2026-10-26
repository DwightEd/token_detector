"""Explicit cohorts preserve native rows and reject incomplete ID selections."""

import json

import pytest

from myproject_experiment import run
from myproject_experiment.datasets import prepare_dataset, read_jsonl
from .test_datasets import hallurag_row, rag_response, ragtruth_files, write_rows


@pytest.fixture(params=["ragtruth", "hallurag", "claimdb"])
def native_dataset(tmp_path, request):
    name, raw = request.param, tmp_path / "raw"
    if name == "ragtruth":
        split = "test"
        ragtruth_files(raw, [rag_response(key, labels=[] if key == "b" else
                                         [{"start": 2, "end": 3}]) for key in "abc"])
    elif name == "hallurag":
        split, rows = "val", []
        for index, key in enumerate("abc"):
            row = hallurag_row("template_1")
            row.update(id=key, original_split=split)
            row["prompt"]["useful_art_i"] = index
            rows.append(row)
        write_rows(raw / "val.jsonl", rows)
    else:
        split = "public"
        write_rows(raw / "test-public.jsonl", [
            {"claim_id": key, "db_name": "finance", "claim": f"Claim {key}.",
             "label": label} for key, label in zip("abc", [
                 "ENTAILED", "CONTRADICTED", "NOT ENOUGH INFO"], strict=True)])
    return name, raw, split


def test_selected_rows_keep_native_order_labels_and_provenance(tmp_path, native_dataset):
    name, raw, split = native_dataset
    full, selected = tmp_path / "full", tmp_path / "selected"
    original = prepare_dataset(name, raw, full, split=split)
    result = prepare_dataset(name, raw, selected, split=split, ids=["c", "a"])
    for filename in ("inputs.jsonl", "annotations.jsonl"):
        lines = (full / filename).read_bytes().splitlines(keepends=True)
        expected = b"".join(line for line in lines if json.loads(line)["id"] in {"a", "c"})
        assert (selected / filename).read_bytes() == expected
    assert [row["id"] for row in read_jsonl(selected / "inputs.jsonl")] == ["a", "c"]
    assert result["samples"] == 2
    assert result["selection"] == {
        "task": None, "limit": None, "order": "original file order", "ids": ["c", "a"],
    }
    assert result["raw_files"] == original["raw_files"]
    assert result["provenance_files"] == original["provenance_files"]


def test_default_and_explicit_none_have_identical_files(tmp_path, native_dataset):
    name, raw, split = native_dataset
    prepare_dataset(name, raw, tmp_path / "default", split=split)
    prepare_dataset(name, raw, tmp_path / "none", split=split, ids=None)
    for filename in ("inputs.jsonl", "annotations.jsonl", "manifest.json"):
        assert (tmp_path / "default" / filename).read_bytes() == (tmp_path / "none" / filename).read_bytes()


@pytest.mark.parametrize("ids, message", [
    ([], "nonempty list"), ([""], "nonempty list"), ([1], "nonempty list"),
    ([None], "nonempty list"), ("a", "nonempty list"),
    (["a", "a"], "Duplicate requested"), (["a", "missing"], "not found"),
])
def test_invalid_or_incomplete_selection_publishes_nothing(tmp_path, native_dataset, ids, message):
    name, raw, split = native_dataset
    output = tmp_path / "selected"
    with pytest.raises(ValueError, match=message):
        prepare_dataset(name, raw, output, split=split, ids=ids)
    assert not output.exists()


def test_ids_and_limit_are_not_silently_combined(tmp_path, native_dataset):
    name, raw, split = native_dataset
    with pytest.raises(ValueError, match="cannot be combined"):
        prepare_dataset(name, raw, tmp_path / "selected", split=split, ids=["a"], limit=1)
    assert not (tmp_path / "selected").exists()


def test_unselected_duplicate_native_id_still_fails(tmp_path, native_dataset):
    name, raw, split = native_dataset
    path = raw / ({"ragtruth": "response.jsonl", "hallurag": "val.jsonl",
                   "claimdb": "test-public.jsonl"}[name])
    rows = list(read_jsonl(path))
    write_rows(path, [*rows, rows[1]])
    with pytest.raises(ValueError, match="Duplicate sample"):
        prepare_dataset(name, raw, tmp_path / "selected", split=split, ids=["a"])
    assert not (tmp_path / "selected").exists()


def test_id_from_another_split_is_missing(tmp_path):
    raw = tmp_path / "raw"
    ragtruth_files(raw, [rag_response("train-only", split="train"), rag_response("test")])
    with pytest.raises(ValueError, match="not found"):
        prepare_dataset("ragtruth", raw, tmp_path / "selected", ids=["train-only"])
    assert not (tmp_path / "selected").exists()


@pytest.mark.parametrize("structured", [False, True])
def test_cli_reads_json_array_or_roster_object(tmp_path, monkeypatch, structured):
    raw, output, ids_path = tmp_path / "raw", tmp_path / "selected", tmp_path / "ids.json"
    ragtruth_files(raw, [rag_response(key) for key in "abc"])
    ids = ["c", "a"]
    ids_path.write_text(json.dumps({"ids": ids, "samples": 2} if structured else ids))
    monkeypatch.setattr("sys.argv", ["run", "prepare-dataset", "--name", "ragtruth",
                        "--data", str(raw), "--ids", str(ids_path), "--output", str(output)])
    run.main()
    assert [row["id"] for row in read_jsonl(output / "inputs.jsonl")] == ["a", "c"]
    assert json.loads((output / "manifest.json").read_text())["selection"]["ids"] == ids


@pytest.mark.parametrize("payload", [None, {}, {"ids": None}, {"ids": "a"}])
def test_cli_invalid_roster_never_defaults_to_all_samples(tmp_path, monkeypatch, payload):
    ids_path = tmp_path / "ids.json"
    ids_path.write_text(json.dumps(payload))
    monkeypatch.setattr("sys.argv", ["run", "prepare-dataset", "--name", "ragtruth",
                        "--data", str(tmp_path / "raw"), "--ids", str(ids_path),
                        "--output", str(tmp_path / "selected")])
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2
    assert not (tmp_path / "selected").exists()


def test_cli_ids_and_limit_are_mutually_exclusive(tmp_path):
    with pytest.raises(SystemExit) as error:
        run.parser().parse_args(["prepare-dataset", "--name", "ragtruth", "--data", str(tmp_path),
                                 "--ids", "ids.json", "--limit", "1", "--output", "unused"])
    assert error.value.code == 2
