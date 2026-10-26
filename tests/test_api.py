"""The public text API runs end to end without diagnostic output or files."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from huggingface_hub.utils.tqdm import progress_bar_states
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.utils import logging

from myproject import HallucinationDetector
from myproject.detector import Detector


class CharacterTokenizer:
    """A deterministic tokenizer with real response offsets for a tiny Llama."""

    all_special_ids = [0]
    added_tokens_decoder = {}

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and add_generation_prompt
        return "<user>" + messages[0]["content"] + "</user><assistant>"

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens and return_offsets_mapping
        return {"input_ids": list(map(ord, text)),
                "offset_mapping": [(index, index + 1) for index in range(len(text))]}

    def decode(self, tokens):
        return "".join(map(chr, tokens))


@pytest.fixture
def text_detector():
    torch.manual_seed(11)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=128, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.,
    )).eval().requires_grad_(False)
    model.set_attn_implementation("sdpa")
    return HallucinationDetector(model, CharacterTokenizer())


def test_fit_predict_is_silent_keeps_original_offsets_and_writes_only_on_save(
        text_detector, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    example = {"prompt": "Facts: B. Answer:", "source": "B.", "response": "A\x00B."}
    assert text_detector.fit([example]) is text_detector
    result = text_detector.predict(**example)
    assert [row["token_id"] for row in result] == [65, 66, 46]
    assert [(row["text"], row["start"], row["end"]) for row in result] == [
        ("A", 0, 1), ("B", 2, 3), (".", 3, 4),
    ]
    assert all(set(row) == {"token_id", "text", "start", "end", "score"} for row in result)
    assert all(np.isfinite(row["score"]) for row in result)
    assert list(tmp_path.iterdir()) == []
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""
    path = tmp_path / "references.npz"
    text_detector.save(path)
    assert list(tmp_path.iterdir()) == [path]
    restored = Detector.load(path)
    for name, values in text_detector.readout.references.items():
        np.testing.assert_array_equal(restored.references[name], values)


@pytest.mark.parametrize("task", ["QA", "Summary", "Data2txt"])
def test_from_pretrained_uses_bundled_task_references(task):
    model, tokenizer = object(), CharacterTokenizer()
    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer), \
            patch("myproject.model.load_model", return_value=model) as load_model:
        detector = HallucinationDetector.from_pretrained("local-model", task=task, device="cpu")
    load_model.assert_called_once_with("local-model", "cpu")
    assert detector.model is model and detector.tokenizer is tokenizer
    assert detector.readout.task == task
    assert all(values.shape == (10001,) for values in detector.readout.references.values())


def test_custom_reference_task_is_checked_before_loading_model(tmp_path):
    path = tmp_path / "summary.npz"
    Detector("Summary", {key: np.array([0., 1.]) for key in ("prior", "window", "route")}).save(path)
    with patch("myproject.model.load_model") as load_model:
        with pytest.raises(ValueError, match="Reference task"):
            HallucinationDetector.from_pretrained("local-model", task="QA", references=path)
    load_model.assert_not_called()


@pytest.mark.parametrize("progress_enabled", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_model_loading_is_quiet_and_restores_caller_logging_settings(
        progress_enabled, fail, capsys):
    original_verbosity = logging.get_verbosity()
    original_progress = logging.is_progress_bar_enabled()
    original_hub_progress = dict(progress_bar_states)
    saved_hub_progress = {"_global": False, "caller.group": True}

    def noisy_loader(*args, **kwargs):
        logging.get_logger("transformers.modeling_utils").warning("normal loader warning")
        for _ in logging.tqdm(range(2), desc="Loading checkpoint shards"):
            pass
        if fail:
            raise RuntimeError("model cannot load")
        return object()

    try:
        logging.set_verbosity_info()
        if progress_enabled:
            logging.enable_progress_bar()
        else:
            logging.disable_progress_bar()
        progress_bar_states.clear()
        progress_bar_states.update(saved_hub_progress)
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=CharacterTokenizer()), \
                patch("myproject.model.load_model", side_effect=noisy_loader):
            if fail:
                with pytest.raises(RuntimeError, match="model cannot load"):
                    HallucinationDetector.from_pretrained("local-model", device="cpu")
            else:
                HallucinationDetector.from_pretrained("local-model", device="cpu")
        assert logging.get_verbosity() == logging.INFO
        assert logging.is_progress_bar_enabled() is progress_enabled
        assert progress_bar_states == saved_hub_progress
        captured = capsys.readouterr()
        assert captured.out == captured.err == ""
    finally:
        logging.set_verbosity(original_verbosity)
        if original_progress:
            logging.enable_progress_bar()
        else:
            logging.disable_progress_bar()
        progress_bar_states.clear()
        progress_bar_states.update(original_hub_progress)


def test_unfitted_empty_and_invalid_source_inputs_fail_before_measurement():
    detector = HallucinationDetector(object(), CharacterTokenizer())
    with pytest.raises(ValueError, match="Fit reference"):
        detector.predict("Facts: B.", "A.", "B.")
    with pytest.raises(ValueError, match="at least one example"):
        detector.fit([])
    with pytest.raises(ValueError, match="at least one token"):
        detector.fit([{"prompt": "Facts: B.", "source": "B.", "response": ""}])
    detector.readout = Detector("QA", {key: np.array([0., 1.])
                                      for key in ("prior", "window", "route")})
    assert detector.predict("Facts: B.", "", "B.") == []
    for prompt, source in [("Facts: B.", ""), ("Facts: B.", "C."), ("B. B.", "B.")]:
        with pytest.raises(ValueError, match="occurring once"):
            detector.predict(prompt, "A.", source)
    assert detector._observer is None
