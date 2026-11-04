"""The production path extracts once and persists reusable token observations."""

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from myproject import HallucinationDetector, Observations
from experiments.inference import HallucinationDetector as ResearchDetector
from .test_api import CharacterTokenizer, text_detector


def test_extract_save_load_and_score_preserve_original_tokens(
        text_detector, tmp_path, capsys):
    detector = HallucinationDetector(text_detector.model, CharacterTokenizer())
    example = {"prompt": "Facts: B. Answer:", "source": "B.", "response": "A\x00B. C."}
    observed = detector.extract(**example, answer_index=7)
    assert set(observed.pack) == {"full_gap", "target", "token_id", "answer_index", "unit_index"}
    assert observed.pack["target"].tolist() == [0, 2, 3, 4, 5, 6]
    assert observed.pack["answer_index"].tolist() == [7] * 6
    assert list(tmp_path.iterdir()) == []

    path = tmp_path / "observations.npz"
    observed.save(path)
    restored = Observations.load(path)
    for key in observed.pack:
        np.testing.assert_array_equal(restored.pack[key], observed.pack[key])
    for key in ("text", "answer_ids", "units", "special_ids"):
        assert restored.response[key] == observed.response[key]
    np.testing.assert_array_equal(restored.response["offsets"], observed.response["offsets"])
    assert detector.score(restored) == detector.predict(**example)
    assert capsys.readouterr() == ("", "")


def test_cli_can_score_saved_observations_without_loading_a_model(
        text_detector, tmp_path, monkeypatch, capsys):
    from myproject.run import main

    detector = HallucinationDetector(text_detector.model, CharacterTokenizer())
    observed = detector.extract("Facts: B.", "A.", "B.")
    path = tmp_path / "observations.npz"
    observed.save(path)
    monkeypatch.setattr("sys.argv", ["token-detect", "score", "--input", str(path)])
    main()
    assert json.loads(capsys.readouterr().out) == detector.score(observed)


@pytest.mark.parametrize("response,forward_count", [("A. C. D.", 4), ("A. " * 8, 6)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_core_extraction_matches_research_measurement_without_local_replays(
        text_detector, response, forward_count, dtype):
    text_detector.model.to(dtype=dtype)
    detector = HallucinationDetector(text_detector.model, CharacterTokenizer())
    example = {"prompt": "Facts: B. Answer:", "source": "B.", "response": response}
    expected, _ = text_detector.measure(**example)
    forwards = []

    def record_forward(_module, _args, kwargs):
        forwards.append(kwargs["input_ids"].shape[1])

    handle = detector.model.model.register_forward_pre_hook(record_forward, with_kwargs=True)
    try:
        observed = detector.extract(**example)
    finally:
        handle.remove()
    np.testing.assert_array_equal(observed.pack["full_gap"], expected["delta"][:, 1])
    for key in ("target", "token_id", "answer_index", "unit_index"):
        np.testing.assert_array_equal(observed.pack[key], expected[key])
    from experiments.research.readouts import SharedDetector

    expected_scores = SharedDetector(method="witness").score(expected)
    np.testing.assert_array_equal(
        [row["score"] for row in detector.score(observed)], expected_scores)
    # Only the two full-history worlds, including a continuation chunk boundary.
    assert len(forwards) == forward_count


def test_empty_and_invalid_input_contract(text_detector):
    detector = HallucinationDetector(text_detector.model, CharacterTokenizer())
    assert detector.predict("Facts: B.", "", "B.") == []
    assert detector.predict("Facts: B.", "\x00", "B.") == []
    with pytest.raises(ValueError, match="at least one token"):
        detector.extract("Facts: B.", "", "B.")
    with pytest.raises(ValueError, match="occurring once"):
        detector.extract("Facts: B.", "A.", "C.")


def test_importing_core_does_not_load_research_code():
    subprocess.run([sys.executable, "-c",
                    "import sys, myproject; "
                    "assert not any(n == 'experiments' or n.startswith('experiments.') "
                    "for n in sys.modules)"], check=True)


@pytest.mark.parametrize("targets,offsets", [
    ([1, 0], [[0, 1], [1, 2]]),
    ([0, 1], [[-1, 1], [1, 99]]),
    ([0, 1], [[[0, 1], [1, 2]]]),
])
def test_load_rejects_files_that_change_the_original_response_axis(tmp_path, targets, offsets):
    response = {"text": "AB", "answer_ids": [65, 66], "offsets": offsets,
                "special_ids": [], "units": [{"start": 0, "stop": 2}]}
    pack = {"full_gap": np.zeros(2), "target": np.asarray(targets),
            "token_id": np.asarray([65, 66])[targets],
            "answer_index": np.zeros(2, dtype=int), "unit_index": np.zeros(2, dtype=int)}
    path = tmp_path / "changed-axis.npz"
    Observations(pack, response).save(path)
    with pytest.raises(ValueError, match="Observation"):
        Observations.load(path)


@pytest.mark.parametrize("method", ["shared", "persistent", "witness"])
def test_research_prediction_keeps_its_explicit_readout(text_detector, method):
    detector = ResearchDetector(text_detector.model, CharacterTokenizer(), method=method)
    example = {"prompt": "Facts: B. Answer:", "source": "B.", "response": "A. C. D."}
    pack, _ = detector.measure(**example)
    expected = detector.readout.score(pack)
    np.testing.assert_array_equal(
        [row["score"] for row in detector.predict(**example)], expected)
