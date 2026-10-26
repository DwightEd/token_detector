"""Native annotation granularity, prompt evidence and split joins stay intact."""

import json
from dataclasses import asdict

import pytest

from myproject_experiment.datasets import Sample, prepare_dataset, read_jsonl, read_samples
from myproject_experiment.data import file_hash


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def ragtruth_files(path, responses):
    write_rows(path / "source_info.jsonl", [{
        "source_id": "s", "task_type": "QA", "source_info": {"passages": "Paris."},
        "prompt": "Use passages:\n巴黎🙂 is a city.\noutput:",
    }])
    write_rows(path / "response.jsonl", responses)


def rag_response(identity="r", split="test", **updates):
    return {"id": identity, "source_id": "s", "response": "巴黎🙂很美。", "split": split,
            "model": "original-generator", "labels": [{"start": 2, "end": 3}], **updates}


def test_ragtruth_unicode_offsets_and_labels_are_separate(tmp_path):
    raw, prepared = tmp_path / "raw", tmp_path / "prepared"
    ragtruth_files(raw, [rag_response()])
    manifest = prepare_dataset("ragtruth", raw, prepared)
    sample, = read_samples(prepared / "inputs.jsonl")
    annotation, = read_jsonl(prepared / "annotations.jsonl")
    assert sample.inputs == {"prompt": "Use passages:\n巴黎🙂 is a city.\noutput:",
                             "source": "巴黎🙂 is a city.", "response": "巴黎🙂很美。"}
    assert annotation["granularity"] == "token"
    assert annotation["spans"] == [{"start": 2, "end": 3, "label": 1}]
    assert sample.response[2:3] == "🙂"
    assert set(asdict(sample)).isdisjoint({"labels", "spans", "native_label", "reference"})
    assert manifest["samples"] == manifest["sources"] == 1
    assert set(manifest["raw_files"]) == {
        str(raw / "source_info.jsonl"), str(raw / "response.jsonl"),
    }


def test_ragtruth_keeps_official_splits_and_all_source_siblings(tmp_path):
    raw = tmp_path / "raw"
    ragtruth_files(raw, [rag_response("train", "train"), rag_response("a"),
                        rag_response("b", model="another-generator", labels=[])])
    output = tmp_path / "test"
    prepare_dataset("ragtruth", raw, output)
    samples = list(read_samples(output / "inputs.jsonl"))
    assert [sample.id for sample in samples] == ["a", "b"]
    assert {sample.source_id for sample in samples} == {"s"}
    assert {sample.split for sample in samples} == {"test"}
    prepare_dataset("ragtruth", raw, tmp_path / "train", split="train")
    assert [sample.id for sample in read_samples(tmp_path / "train/inputs.jsonl")] == ["train"]


def test_duplicate_ids_and_invalid_offsets_fail_before_creating_outputs(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "prepared"
    ragtruth_files(raw, [rag_response(), rag_response()])
    with pytest.raises(ValueError, match="Duplicate sample"):
        prepare_dataset("ragtruth", raw, output)
    assert not output.exists()
    ragtruth_files(raw, [rag_response(labels=[{"start": 2, "end": 100}])])
    with pytest.raises(ValueError, match="character span"):
        prepare_dataset("ragtruth", raw, output)
    assert not output.exists()


def hallurag_row(template):
    source = "### Chunk 1: Retrieved article\nThe price is £5.\n\n"
    if template == "template_langchain_hub":
        messages = [{"role": "user", "content": "Question: Cost?\nContext: " + source + " \nAnswer:"}]
    else:
        prefix = ("Your knowledge is limited to only this information:\n" if template == "template_1"
                  else "Only use the information included in these chunks to answer the question:\n")
        messages = [{"role": "system", "content": "Use retrieved text."},
                    {"role": "user", "content": prefix + source + "\nQUESTION: Cost?\nBRIEF ANSWER:"}]
    return {
        "id": "hallurag:original.pickle:0", "original_split": "test",
        "model": "original-generator", "quantization": "int4",
        "prompt": {"useful_art_i": 17, "qna_id": "17_1", "rag_prompt": messages,
                   "prompt_template_name": template,
                   "passage": {"context": "ORACLE SECRET", "answer_quote": "SECRET"}},
        "llm_response": "£5. Maybe🙂. Unannotated tail.",
        "sentence_data": [{"cum_sentence": "£5.", "target": "non-hallucinated"},
                          {"cum_sentence": "£5. Maybe🙂.", "target": "invalid"}],
    }


@pytest.mark.parametrize("template", ["template_langchain_hub", "template_1", "template_2"])
def test_hallurag_uses_retrieved_context_and_retains_unknown_sentences(tmp_path, template):
    raw, output = tmp_path / "raw", tmp_path / "prepared"
    row = hallurag_row(template)
    write_rows(raw / "test.jsonl", [row])
    manifest = prepare_dataset("hallurag", raw, output)
    sample, = read_samples(output / "inputs.jsonl")
    annotation, = read_jsonl(output / "annotations.jsonl")
    assert sample.source == "### Chunk 1: Retrieved article\nThe price is £5.\n\n"
    assert "ORACLE" not in sample.prompt and "SECRET" not in sample.prompt
    assert sample.source_id == "17"
    assert sample.response == row["llm_response"]
    assert annotation["granularity"] == "sentence"
    assert annotation["spans"] == [
        {"start": 0, "end": 3, "label": 0, "native_label": "non-hallucinated"},
        {"start": 3, "end": 11, "label": None, "native_label": "invalid"},
    ]
    assert manifest["bootstrap_type"] == "article"
    assert annotation["spans"][-1]["end"] < len(sample.response)


def test_hallurag_misaligned_labels_are_retained_without_inventing_spans(tmp_path):
    raw = tmp_path / "raw"
    row = hallurag_row("template_1")
    row["sentence_data"][1]["cum_sentence"] = "Different response."
    write_rows(raw / "test.jsonl", [row])
    manifest = prepare_dataset("hallurag", raw, tmp_path / "bad-prefix")
    annotation, = read_jsonl(tmp_path / "bad-prefix/annotations.jsonl")
    assert annotation["coverage"] == "unalignable" and annotation["spans"] == []
    assert annotation["native_sentences"] == row["sentence_data"]
    assert manifest["coverage"] == {"unalignable": 1}
    row = hallurag_row("template_1")
    row["original_split"] = "train"
    write_rows(raw / "test.jsonl", [row])
    with pytest.raises(ValueError, match="different official split"):
        prepare_dataset("hallurag", raw, tmp_path / "bad-split")


@pytest.mark.parametrize("target", ["undefined", None])
def test_hallurag_unknown_native_labels_remain_unknown(tmp_path, target):
    raw = tmp_path / "raw"
    row = hallurag_row("template_1")
    row["sentence_data"][0]["target"] = target
    write_rows(raw / "test.jsonl", [row])
    prepare_dataset("hallurag", raw, tmp_path / "prepared")
    annotation, = read_jsonl(tmp_path / "prepared/annotations.jsonl")
    assert annotation["spans"][0]["label"] is None
    assert annotation["spans"][0]["native_label"] == target


def test_claimdb_preserves_native_labels_and_rejects_text_inference(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "prepared"
    row = {"claim_id": 42, "bird_id": 999, "db_name": "finance", "claim": "The total is 3.",
           "extra_info": "total means SUM(amount)", "label": "NOT ENOUGH INFO",
           "category": "OUT-OF-SCHEMA"}
    write_rows(raw / "test-public.jsonl", [row])
    prepare_dataset("claimdb", raw, output, split="public")
    sample, = read_samples(output / "inputs.jsonl")
    annotation, = read_jsonl(output / "annotations.jsonl")
    assert annotation["native_label"] == "NOT ENOUGH INFO"
    assert annotation["granularity"] == "claim"
    assert sample.split == "public" and sample.source_id == "finance"
    assert sample.source == sample.evidence_path == ""
    assert "999" not in sample.prompt and "OUT-OF-SCHEMA" not in sample.prompt
    assert "NOT ENOUGH INFO" not in sample.prompt
    with pytest.raises(ValueError, match="structured evidence"):
        sample.inputs
    with pytest.raises(ValueError, match="public or private"):
        prepare_dataset("claimdb", raw, tmp_path / "test-alias")


def test_private_claims_stay_unlabelled_and_prepared_ids_cannot_collide(tmp_path):
    raw = tmp_path / "raw"
    write_rows(raw / "test-private.jsonl", [{"claim_id": 3, "db_name": "finance",
                                            "claim": "A private claim.", "extra_info": ""}])
    prepare_dataset("claimdb", raw, tmp_path / "private", split="private")
    annotation, = read_jsonl(tmp_path / "private/annotations.jsonl")
    assert annotation["native_label"] is None and annotation["coverage"] == "hidden"
    sample = Sample("a", "s", "test", "QA", "Source: fact.", "fact.", "Answer.")
    write_rows(tmp_path / "duplicates.jsonl", [asdict(sample), asdict(sample)])
    with pytest.raises(ValueError, match="Duplicate or empty"):
        list(read_samples(tmp_path / "duplicates.jsonl"))


def test_claimdb_uses_official_flat_database_paths_and_binds_revision_receipt(tmp_path):
    raw, output = tmp_path / "raw", tmp_path / "prepared"
    database = raw / "test_dbs/finance.sqlite"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"fixture evidence bytes")
    receipt = raw / "ACQUISITION_RECEIPT.json"
    receipt.write_text(json.dumps({"hf_revision": "pinned-release"}))
    write_rows(raw / "test-public.jsonl", [{"claim_id": 42, "db_name": "finance",
                                           "claim": "The total is 3.", "extra_info": "",
                                           "label": "ENTAILED"}])
    manifest = prepare_dataset("claimdb", raw, output, split="public")
    sample, = read_samples(output / "inputs.jsonl")
    assert sample.evidence_path == str(database.resolve())
    assert manifest["evidence_files"] == {str(database.resolve()): file_hash(database)}
    assert manifest["provenance_files"] == {str(receipt.resolve()): file_hash(receipt)}


@pytest.mark.parametrize("prompt,source,response", [
    ("A", ["A"], "A"), ("A A", "A", "A"), ("A", "B", "A"),
])
def test_inference_inputs_validate_strings_and_unique_source(prompt, source, response):
    with pytest.raises(ValueError, match="strings|exactly once"):
        Sample("id", "source", "test", "QA", prompt, source, response).inputs


def test_hallurag_cli_accepts_native_val_and_validation_alias(tmp_path, monkeypatch):
    from myproject_experiment.run import main

    raw = tmp_path / "raw"
    row = hallurag_row("template_langchain_hub")
    row["id"] = "hallurag:val_original.pickle:0"
    row["original_split"] = "val"
    write_rows(raw / "val.jsonl", [row])
    for split in ("val", "validation"):
        output = tmp_path / split
        monkeypatch.setattr("sys.argv", [
            "myproject_experiment.run", "prepare-dataset", "--name", "hallurag",
            "--data", str(raw), "--split", split, "--output", str(output),
        ])
        main()
        sample, = read_samples(output / "inputs.jsonl")
        assert sample.split == "val"
        assert json.loads((output / "manifest.json").read_text())["split"] == "val"
    for filename in ("inputs.jsonl", "annotations.jsonl"):
        assert (tmp_path / "val" / filename).read_bytes() == (tmp_path / "validation" / filename).read_bytes()


@pytest.mark.parametrize("split", ["test", "val", "train"])
def test_hallurag_binds_only_the_requested_split_receipt(tmp_path, split):
    raw = tmp_path / "raw"
    row = hallurag_row("template_1")
    row["original_split"] = split
    write_rows(raw / f"{split}.jsonl", [row])
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    for name in ("ACQUISITION_RECEIPT.json", "VAL_ACQUISITION_RECEIPT.json",
                 "TRAIN_ACQUISITION_RECEIPT.json"):
        (provenance / name).write_text(json.dumps({"member_receipt": name}))
    name = "ACQUISITION_RECEIPT.json" if split == "test" else f"{split.upper()}_ACQUISITION_RECEIPT.json"
    receipt = provenance / name
    result = prepare_dataset("hallurag", raw, tmp_path / "prepared", split=split)
    assert result["provenance_files"] == {str(receipt.resolve()): file_hash(receipt)}


def test_hallurag_val_alias_preserves_frozen_test_inputs_and_annotations(tmp_path):
    from pathlib import Path

    project = Path(__file__).resolve().parents[2]
    raw = project / "data/HalluRAG/raw"
    frozen = project / "experiment_results/cross_dataset_20261018/hallurag_prepared_v1"
    if not (raw / "test.jsonl").exists() or not (frozen / "manifest.json").exists():
        pytest.skip("The optional frozen HalluRAG TEST artifacts are not installed")
    manifest = json.loads((frozen / "manifest.json").read_text())
    result = prepare_dataset("hallurag", raw, tmp_path / "test", split="test")
    assert result["samples"] == manifest["samples"]
    for filename, key in (("inputs.jsonl", "inputs_sha256"),
                          ("annotations.jsonl", "annotations_sha256")):
        assert file_hash(frozen / filename) == manifest[key]
        assert file_hash(tmp_path / "test" / filename) == manifest[key]
