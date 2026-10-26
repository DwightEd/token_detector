"""The official pickle-to-JSONL step preserves native data before adaptation."""

import pickle

import pytest

from myproject_experiment.datasets import prepare_dataset, read_jsonl, read_samples
from myproject_experiment.export_hallurag import export_hallurag
from myproject_experiment.data import file_hash

from .test_datasets import hallurag_row


def test_export_then_prepare_preserves_original_response_and_native_labels(tmp_path):
    original = hallurag_row("template_1")
    original["sentence_data"][0]["internal_states"] = {"large_observation": [1., 2.]}
    member = tmp_path / "test_generator.pickle"
    member.write_bytes(pickle.dumps([original]))
    output = tmp_path / "raw/test.jsonl"
    report = export_hallurag([member], output)
    row, = read_jsonl(output)
    assert row["id"] == "hallurag:test_generator.pickle:0"
    assert row["llm_response"] == original["llm_response"]
    assert row["prompt"] == original["prompt"]
    assert [sentence["target"] for sentence in row["sentence_data"]] == ["non-hallucinated", "invalid"]
    assert all("internal_states" not in sentence for sentence in row["sentence_data"])
    assert report["responses"] == 1
    prepared = tmp_path / "prepared"
    manifest = prepare_dataset("hallurag", output.parent, prepared)
    receipt = output.with_suffix(".receipt.json")
    assert manifest["provenance_files"][str(receipt)] == file_hash(receipt)
    sample, = read_samples(prepared / "inputs.jsonl")
    assert "ORACLE" not in sample.prompt
    assert sample.response == original["llm_response"]
    saved = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_hallurag([member], output)
    assert output.read_bytes() == saved


@pytest.mark.parametrize("names", [["test_a.pickle", "val_a.pickle"],
                                 ["test_a.pickle", "test_a.pickle"], []])
def test_export_rejects_ambiguous_members_before_reading_files(tmp_path, names):
    with pytest.raises(ValueError):
        export_hallurag([tmp_path / name for name in names], tmp_path / "export.jsonl")


def test_pickle_globals_are_rejected_without_publishing_partial_export(tmp_path):
    member = tmp_path / "test_not_plain.pickle"
    member.write_bytes(pickle.dumps(ValueError("not a released text row")))
    output = tmp_path / "raw/test.jsonl"
    with pytest.raises(ValueError, match="pickle global"):
        export_hallurag([member], output)
    assert list(output.parent.iterdir()) == []
