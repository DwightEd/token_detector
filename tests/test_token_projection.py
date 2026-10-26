"""Sentence annotations project onto the unchanged token axis without using scores."""

import copy
from pathlib import Path

import pytest

from myproject_experiment import data, datasets, run
from myproject_experiment.evaluate import evaluate_dataset

from .test_dataset_pipeline import (
    FakeDetector,
    install_detector,
    make_sample,
    model_files,
    predict,
    prepare_fixture,
)


def annotation(spans, coverage="annotated_sentences"):
    return {"granularity": "sentence", "coverage": coverage, "spans": spans}


def token(start, end, score=.5):
    return {"start": start, "end": end, "score": score}


def test_unicode_duplicate_bpe_offsets_keep_distinct_token_labels():
    response = "甲🙂乙。"
    spans = [{"start": 0, "end": 1, "label": 0},
             {"start": 1, "end": 2, "label": 1},
             {"start": 2, "end": len(response), "label": 0}]
    # Two BPE pieces can share the emoji's original character interval.
    tokens = [token(0, 1), token(1, 2, .1), token(1, 2, .9), token(2, 4)]
    projected = datasets.sentence_token_labels(annotation(spans), tokens)
    assert projected == [
        {"label": False, "sentence_indices": [0], "exclusion": None},
        {"label": True, "sentence_indices": [1], "exclusion": None},
        {"label": True, "sentence_indices": [1], "exclusion": None},
        {"label": False, "sentence_indices": [2], "exclusion": None},
    ]


@pytest.mark.parametrize("spans,expected_indices", [
    ([{"start": 0, "end": 2, "label": 0}, {"start": 2, "end": 4, "label": 1}], [0, 1]),
    ([{"start": 0, "end": 2, "label": None}, {"start": 2, "end": 4, "label": 1}], [0, 1]),
    ([{"start": 2, "end": 3, "label": 1}], [0]),
])
def test_positive_overlap_has_priority_over_negative_unknown_and_uncovered(spans, expected_indices):
    result, = datasets.sentence_token_labels(annotation(spans), [token(0, 4)])
    assert result == {"label": True, "sentence_indices": expected_indices, "exclusion": None}


@pytest.mark.parametrize("spans,tokens,expected", [
    ([{"start": 0, "end": 1, "label": 0}, {"start": 1, "end": 2, "label": None}],
     [token(0, 4)], {"label": None, "sentence_indices": [0, 1], "exclusion": "undefined"}),
    ([{"start": 0, "end": 2, "label": 0}],
     [token(0, 4)], {"label": None, "sentence_indices": [0], "exclusion": "uncovered"}),
    ([{"start": 0, "end": 1, "label": 1}],
     [token(1, 2)], {"label": None, "sentence_indices": [], "exclusion": "uncovered"}),
    ([], [token(0, 1)], {"label": None, "sentence_indices": [], "exclusion": "uncovered"}),
])
def test_unknown_and_uncovered_are_excluded_with_half_open_boundaries(spans, tokens, expected):
    assert datasets.sentence_token_labels(annotation(spans), tokens) == [expected]


def test_negative_requires_complete_coverage_and_can_cross_adjacent_sentences():
    spans = [{"start": 0, "end": 2, "label": 0}, {"start": 2, "end": 4, "label": 0}]
    result, = datasets.sentence_token_labels(annotation(spans), [token(0, 4)])
    assert result == {"label": False, "sentence_indices": [0, 1], "exclusion": None}


def test_unalignable_response_excludes_every_token():
    tokens = [token(0, 1), token(1, 2), token(1, 2), token(2, 4)]
    projected = datasets.sentence_token_labels(annotation([], coverage="unalignable"), tokens)
    assert projected == [
        {"label": None, "sentence_indices": [], "exclusion": "unalignable"},
    ] * len(tokens)


@pytest.mark.parametrize("spans", [
    [{"start": 0, "end": 1, "label": 2}],
    [{"start": 0, "end": 1, "label": "0"}],
    [{"start": -1, "end": 1, "label": 0}],
    [{"start": 1, "end": 1, "label": 0}],
    [{"start": .5, "end": 1, "label": 0}],
    [{"start": 0, "end": 3, "label": 0}, {"start": 2, "end": 4, "label": 1}],
    [{"start": 2, "end": 4, "label": 0}, {"start": 0, "end": 2, "label": 1}],
])
def test_invalid_native_labels_offsets_and_sentence_order_are_rejected(spans):
    with pytest.raises(ValueError):
        datasets.sentence_token_labels(annotation(spans), [token(0, 4)])


def test_projection_is_score_independent_and_does_not_modify_inputs():
    gold = annotation([{"start": 0, "end": 1, "label": 0},
                       {"start": 1, "end": 2, "label": 1},
                       {"start": 2, "end": 3, "label": None}])
    tokens = [token(0, 1, .01), token(1, 2, .99), token(2, 3, .4)]
    original_gold, original_tokens = copy.deepcopy(gold), copy.deepcopy(tokens)
    initial = datasets.sentence_token_labels(gold, tokens)
    assert gold == original_gold and tokens == original_tokens
    reversed_scores = [{**row, "score": 1 - row["score"]} for row in tokens]
    without_scores = [{key: value for key, value in row.items() if key != "score"} for row in tokens]
    assert datasets.sentence_token_labels(gold, reversed_scores) == initial
    assert datasets.sentence_token_labels(gold, without_scores) == initial


def test_sealed_token_evaluation_preserves_scores_exports_all_labels_and_keeps_native_results(
        tmp_path, model_files, monkeypatch):
    samples = [make_sample("a", "article-a", response="ABCDEF"),
               make_sample("b", "article-b", response="UVWXYZ")]
    annotations = [
        {"id": "a", "response_sha256": data.text_hash("ABCDEF"), "granularity": "sentence",
         "coverage": "annotated_sentences", "spans": [
             {"start": 0, "end": 2, "label": 0},
             {"start": 2, "end": 4, "label": 1},
             {"start": 4, "end": 5, "label": None, "native_label": "undefined"},
         ]},
        {"id": "b", "response_sha256": data.text_hash("UVWXYZ"), "granularity": "sentence",
         "coverage": "unalignable", "spans": []},
    ]
    prepared = prepare_fixture(tmp_path / "prepared", samples, annotations)
    detector = FakeDetector()
    install_detector(monkeypatch, detector)
    output = tmp_path / "scores"
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == (prepared / "annotations.jsonl").resolve():
            raise AssertionError("Inference read ground truth")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(Path, "open", guarded_open)
        predict(prepared, output, model_files)
    assert detector.responses == ["ABCDEF", "UVWXYZ"]
    immutable = {path: data.file_hash(path) for path in output.rglob("*") if path.is_file()}
    native_before = evaluate_dataset(output, None, tmp_path / "native-before", repetitions=16)
    report = evaluate_dataset(output, None, tmp_path / "token", repetitions=16,
                              label_granularity="token")
    projected = list(datasets.read_jsonl(tmp_path / "token/token_labels.jsonl"))
    units = list(datasets.read_jsonl(tmp_path / "token/units.jsonl"))
    assert len(projected) == 12 and all("score" not in row for row in projected)
    assert [row["label"] for row in projected] == [False, False, True, True, None, None] + [None] * 6
    assert [row["exclusion"] for row in projected] == [None] * 4 + ["undefined", "uncovered"] + ["unalignable"] * 6
    assert [row["start"] for row in projected] == list(range(6)) * 2
    assert [row["token_index"] for row in projected] == list(range(6)) * 2
    assert [row["native_sentence_indices"] for row in projected] == [[0], [0], [1], [1], [2], []] + [[]] * 6
    assert [row["score"] for row in units] == [1 / 7, 2 / 7, 3 / 7, 4 / 7]
    assert report["granularity"] == "token"
    assert report["metrics"]["units"] == 4 and report["metrics"]["errors"] == 2
    assert report["metrics"]["auroc"] == report["metrics"]["ap"] == 1.
    assert report["tokens_scored"] == 12 and report["tokens_evaluated"] == 4
    assert report["tokens_excluded"] == {"undefined": 1, "uncovered": 1, "unalignable": 6}
    assert native_before["granularity"] == "sentence" and native_before["metrics"]["units"] == 2
    native_after = evaluate_dataset(output, None, tmp_path / "native-after", repetitions=16)
    assert native_before == native_after
    assert (tmp_path / "native-before/units.jsonl").read_bytes() == (tmp_path / "native-after/units.jsonl").read_bytes()
    assert {path: data.file_hash(path) for path in immutable} == immutable
    monkeypatch.setattr("sys.argv", [
        "myproject_experiment.run", "evaluate-dataset", "--scores", str(output),
        "--label-granularity", "token", "--bootstrap", "16", "--output", str(tmp_path / "cli-token"),
    ])
    run.main()
    assert data.read_json(tmp_path / "cli-token/evaluation.json")["metrics"] == report["metrics"]


def test_token_evaluation_rejects_annotations_beyond_original_response(
        tmp_path, model_files, monkeypatch):
    sample = make_sample(response="AB")
    gold = {"id": sample.id, "response_sha256": data.text_hash(sample.response),
            "granularity": "sentence", "coverage": "annotated_sentences",
            "spans": [{"start": 0, "end": 3, "label": 1}]}
    prepared = prepare_fixture(tmp_path / "prepared", [sample], [gold])
    install_detector(monkeypatch, FakeDetector())
    output = tmp_path / "scores"
    predict(prepared, output, model_files)
    with pytest.raises(ValueError, match="character span"):
        evaluate_dataset(output, None, tmp_path / "invalid", repetitions=8,
                         label_granularity="token")
    assert not (tmp_path / "invalid").exists()
