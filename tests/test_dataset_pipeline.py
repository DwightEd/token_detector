"""Dataset orchestration preserves evidence isolation and native evaluation units."""

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from myproject.api import HallucinationDetector
from myproject_experiment import data, pipeline, run
from myproject_experiment.datasets import Sample, read_jsonl
from myproject_experiment.evaluate import evaluate_dataset


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def make_sample(identity="a", source_id="article-a", response="ABCDEFGH", **changes):
    fields = {"id": identity, "source_id": source_id, "split": "test", "task": "QA",
              "prompt": "Facts: one fact. Answer the question.", "source": "one fact.",
              "response": response, "generator": "original-generator"}
    return Sample(**{**fields, **changes})


def prepare_fixture(path, samples, annotations=None):
    path.mkdir()
    write_rows(path / "inputs.jsonl", [asdict(sample) for sample in samples])
    if annotations is None:
        annotations = [{"id": sample.id, "response_sha256": data.text_hash(sample.response),
                        "granularity": "token", "coverage": "complete", "spans": []}
                       for sample in samples]
    write_rows(path / "annotations.jsonl", annotations)
    data.write_json(path / "manifest.json", {
        "dataset": "fixture", "split": "test", "bootstrap_type": "article",
        "samples": len(samples), "inputs_sha256": data.file_hash(path / "inputs.jsonl"),
        "annotations_sha256": data.file_hash(path / "annotations.jsonl"),
    })
    return path


@pytest.fixture
def model_files(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "tokenizer.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.weight": "model-00001-of-00001.safetensors"},
    }))
    (model / "model-00001-of-00001.safetensors").write_bytes(b"frozen fixture weights")
    references = tmp_path / "references.npz"
    references.write_bytes(b"frozen fixture references")
    return model, references


class FakeDetector:
    """Deterministic scores replace model computation, while orchestration stays real."""

    def __init__(self, fail_on=None, tokens=None):
        self.fail_on = fail_on
        self.tokens = tokens
        self.responses = []

    def predict(self, prompt, source, response):
        self.responses.append(response)
        if response == self.fail_on:
            raise RuntimeError("injected observation failure")
        if self.tokens is not None:
            return self.tokens
        return [{"token_id": ord(character), "text": character, "start": index,
                 "end": index + 1, "score": (index + 1) / (len(response) + 1)}
                for index, character in enumerate(response)]


def install_detector(monkeypatch, detector):
    loader = Mock(return_value=detector)
    monkeypatch.setattr(HallucinationDetector, "from_pretrained", loader)
    return loader


def predict(prepared, output, model_files):
    model, references = model_files
    pipeline.predict_dataset(prepared, model, "QA", output, references, device="cpu")


def test_failure_leaves_checkpoint_and_resume_scores_only_missing_response(
        tmp_path, model_files, monkeypatch):
    prepared = prepare_fixture(tmp_path / "prepared", [make_sample("a", response="first"),
                                                        make_sample("b", response="second")])
    output = tmp_path / "scores"
    first_detector = FakeDetector(fail_on="second")
    install_detector(monkeypatch, first_detector)
    with pytest.raises(RuntimeError, match="observation failure"):
        predict(prepared, output, model_files)
    checkpoint = data.prediction_path(output / "predictions", "a")
    checksum = data.file_hash(checkpoint)
    assert not (output / "complete.json").exists()
    assert not (output / "predictions/seal.json").exists()
    second_detector = FakeDetector()
    install_detector(monkeypatch, second_detector)
    predict(prepared, output, model_files)
    assert second_detector.responses == ["second"]
    assert data.file_hash(checkpoint) == checksum
    assert data.read_json(output / "complete.json")["samples"] == 2
    loader = install_detector(monkeypatch, FakeDetector())
    predict(prepared, output, model_files)
    loader.assert_not_called()


def test_prediction_never_opens_annotations(tmp_path, model_files, monkeypatch):
    prepared = prepare_fixture(tmp_path / "prepared", [make_sample()])
    annotation_path = prepared / "annotations.jsonl"
    annotation_path.write_text("This file is deliberately not valid JSON.")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == annotation_path.resolve():
            raise AssertionError("Prediction opened annotations")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    install_detector(monkeypatch, FakeDetector())
    predict(prepared, tmp_path / "scores", model_files)
    assert data.read_json(tmp_path / "scores/complete.json")["annotation_reads"] == 0


def test_structured_evidence_is_rejected_before_loading_model(tmp_path, model_files, monkeypatch):
    prepared = prepare_fixture(tmp_path / "prepared", [make_sample(source="",
                                task="ClaimVerification", evidence_path="database.sqlite")])
    loader = install_detector(monkeypatch, FakeDetector())
    with pytest.raises(ValueError, match="structured evidence"):
        predict(prepared, tmp_path / "scores", model_files)
    loader.assert_not_called()
    assert not (tmp_path / "scores").exists()


@pytest.mark.parametrize("mutation", ["score", "extra-file"])
def test_sealed_prediction_tampering_is_rejected_before_annotation_reads(
        tmp_path, model_files, monkeypatch, mutation):
    prepared = prepare_fixture(tmp_path / "prepared", [make_sample()])
    output = tmp_path / "scores"
    install_detector(monkeypatch, FakeDetector())
    predict(prepared, output, model_files)
    if mutation == "score":
        path = data.prediction_path(output / "predictions", "a")
        row = data.read_json(path)
        row["tokens"][0]["score"] = .99
        path.write_text(json.dumps(row))
    else:
        (output / "predictions/unexpected.json").write_text("{}")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == (prepared / "annotations.jsonl").resolve():
            raise AssertionError("Evaluation read labels before verifying predictions")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(ValueError, match="[Ss]ealed prediction"):
        evaluate_dataset(output, None, tmp_path / "evaluation", repetitions=16)
    assert not (tmp_path / "evaluation").exists()


def test_sentence_evaluation_weights_overlap_and_bootstraps_whole_articles(
        tmp_path, model_files, monkeypatch):
    samples = [make_sample("a", "article-0", generator="generator-1"),
               make_sample("b", "article-0", generator="generator-2"),
               make_sample("c", "article-1", generator="generator-1"),
               make_sample("d", "article-1", generator="generator-2"),
               make_sample("e", "unalignable-article")]
    annotations = []
    for index, sample in enumerate(samples):
        if index < 2:
            spans = [{"start": 0, "end": 4, "label": 0},
                     {"start": 4, "end": 8, "label": None, "native_label": "undefined"}]
        elif index < 4:
            spans = [{"start": 4, "end": 8, "label": 1}]
        else:
            spans = []
        annotations.append({"id": sample.id, "response_sha256": data.text_hash(sample.response),
                            "granularity": "sentence", "spans": spans,
                            "coverage": "unalignable" if index == 4 else "annotated_sentences"})
    prepared = prepare_fixture(tmp_path / "prepared", samples, annotations)
    detector = FakeDetector(tokens=[
        {"token_id": 1, "text": "ABC", "start": 0, "end": 3, "score": 0.},
        {"token_id": 2, "text": "DEFGH", "start": 3, "end": 8, "score": 1.},
    ])
    install_detector(monkeypatch, detector)
    output = tmp_path / "scores"
    predict(prepared, output, model_files)
    report = evaluate_dataset(output, None, tmp_path / "evaluation", repetitions=128, seed=13)
    units = list(read_jsonl(tmp_path / "evaluation/units.jsonl"))
    assert [row["score"] for row in units] == [.25, .25, 1., 1.]
    assert report["granularity"] == "sentence"
    assert report["samples_scored"] == 5 and report["samples_evaluated"] == 4
    assert report["metrics"]["units"] == 4 and report["metrics"]["errors"] == 2
    assert report["metrics"]["auroc"] == report["metrics"]["ap"] == 1.
    assert report["unknown_units_excluded"] == 2
    assert report["unalignable_responses_excluded"] == 1
    assert report["groups_evaluated"] == 2 and report["group_type"] == "article"
    draws = np.random.default_rng(13).integers(0, 2, size=(128, 2))
    mixed_articles = int((draws[:, 0] != draws[:, 1]).sum())
    any_positive_article = int((draws == 1).any(axis=1).sum())
    assert report["bootstrap"]["auroc"] == {
        "defined": mixed_articles, "degenerate": 128 - mixed_articles, "interval95": [1., 1.],
    }
    assert report["bootstrap"]["ap"]["defined"] == any_positive_article
    assert set(report["by_generator"]) == {"generator-1", "generator-2"}


def test_modified_prepared_input_cannot_be_scored_under_original_manifest(
        tmp_path, model_files, monkeypatch):
    prepared = prepare_fixture(tmp_path / "prepared", [make_sample()])
    changed = asdict(make_sample(response="Changed response"))
    write_rows(prepared / "inputs.jsonl", [changed])
    loader = install_detector(monkeypatch, FakeDetector())
    with pytest.raises(ValueError, match="[Mm]anifest|[Pp]repared|[Ii]nput"):
        predict(prepared, tmp_path / "scores", model_files)
    loader.assert_not_called()


def test_weight_mutation_prevents_failed_run_resume(tmp_path, model_files, monkeypatch):
    prepared = prepare_fixture(tmp_path / "prepared", [make_sample("a", response="first"),
                                                        make_sample("b", response="second")])
    output = tmp_path / "scores"
    install_detector(monkeypatch, FakeDetector(fail_on="second"))
    with pytest.raises(RuntimeError, match="observation failure"):
        predict(prepared, output, model_files)
    checkpoint = data.prediction_path(output / "predictions", "a")
    checksum = data.file_hash(checkpoint)
    model, _ = model_files
    (model / "model-00001-of-00001.safetensors").write_bytes(b"different fixture weights")
    loader = install_detector(monkeypatch, FakeDetector())
    with pytest.raises(ValueError, match="[Rr]esume|[Mm]odel|[Ww]eight"):
        predict(prepared, output, model_files)
    loader.assert_not_called()
    assert data.file_hash(checkpoint) == checksum
    assert not (output / "complete.json").exists()


def test_cli_prepares_predicts_and_evaluates_without_gpu(tmp_path, model_files, monkeypatch, capsys):
    raw = tmp_path / "raw"
    raw.mkdir()
    write_rows(raw / "source_info.jsonl", [{"source_id": "s", "task_type": "QA",
               "source_info": {}, "prompt": "Use passages:\none fact.\noutput:"}])
    write_rows(raw / "response.jsonl", [{"id": "r", "source_id": "s", "split": "test",
               "model": "original-generator", "response": "AB", "labels": [{"start": 1, "end": 2}]}])
    prepared, scores, evaluation = tmp_path / "prepared", tmp_path / "scores", tmp_path / "evaluation"
    install_detector(monkeypatch, FakeDetector())
    model, references = model_files
    commands = [
        ["prepare-dataset", "--name", "ragtruth", "--data", str(raw), "--task", "QA",
         "--output", str(prepared)],
        ["predict-dataset", "--input", str(prepared), "--model", str(model), "--task", "QA",
         "--references", str(references), "--device", "cpu", "--output", str(scores)],
        ["evaluate-dataset", "--scores", str(scores), "--bootstrap", "16", "--output", str(evaluation)],
    ]
    for arguments in commands:
        monkeypatch.setattr("sys.argv", ["myproject_experiment.run", *arguments])
        run.main()
    report = data.read_json(evaluation / "evaluation.json")
    assert report["granularity"] == "token" and report["metrics"]["units"] == 2
    assert report["metrics"]["auroc"] == report["metrics"]["ap"] == 1.
    assert "token AUROC=1.0 AP=1.0" in capsys.readouterr().out


def test_atomic_json_failure_publishes_nothing_and_existing_result_is_immutable(tmp_path, monkeypatch):
    destination = tmp_path / "result.json"

    def interrupted_dump(value, stream, **kwargs):
        stream.write('{"incomplete":')
        stream.flush()
        raise RuntimeError("interrupted JSON serialization")

    with monkeypatch.context() as patcher:
        patcher.setattr(data.json, "dump", interrupted_dump)
        with pytest.raises(RuntimeError, match="interrupted JSON"):
            data.write_json(destination, {"value": 1})
    assert not destination.exists()
    assert not (tmp_path / ".result.json.tmp").exists()
    data.write_json(destination, {"value": 1})
    original = destination.read_bytes()
    with pytest.raises(FileExistsError):
        data.write_json(destination, {"value": 2})
    assert destination.read_bytes() == original
    assert not (tmp_path / ".result.json.tmp").exists()


def test_resume_cleans_only_owned_hard_kill_temporaries(tmp_path, model_files, monkeypatch):
    prepared = prepare_fixture(tmp_path / "prepared", [make_sample()])
    output = tmp_path / "scores"
    install_detector(monkeypatch, FakeDetector())
    predict(prepared, output, model_files)
    checkpoint = data.prediction_path(output / "predictions", "a")
    original = data.file_hash(checkpoint)
    owned = [output / ".run.json.tmp", output / ".complete.json.tmp",
             output / "predictions/.seal.json.tmp"]
    for path in owned:
        path.write_text('{"unfinished":')
    unrelated = output / ".user-note.tmp"
    unrelated.write_text("Preserve unrelated files.")
    loader = install_detector(monkeypatch, FakeDetector())
    predict(prepared, output, model_files)
    loader.assert_not_called()
    assert all(not path.exists() for path in owned)
    assert unrelated.read_text() == "Preserve unrelated files."
    assert data.file_hash(checkpoint) == original
