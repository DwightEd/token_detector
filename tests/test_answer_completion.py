"""Literal full-name/EOS alignment checks; fake model and CPU tensors only."""
from contextlib import nullcontext
import math
import sys
from types import SimpleNamespace

import pytest

from experiments import answer_completion as module
from experiments.semantic_qualification import HFGenerator, HF_RUNTIME

CANDIDATE = 'Mira Tanswick'
MESSAGES = [{'role': 'user', 'content': 'Synthetic source-only question.'}]
IDENTITY = dict(model_id='Qwen/Qwen2.5-3B-Instruct', model_files_sha256={'fake': 'a' * 64},
                runtime=dict(HF_RUNTIME))


class Tokenizer:
    eos_token_id = 7

    def __init__(self, prefix_ids=None, suffix=None, broken_prefix=False, broken_roundtrip=False):
        self.prefix_ids = [10, 11] if prefix_ids is None else prefix_ids
        self.suffix = [2, 3, 4] if suffix is None else suffix
        self.broken_prefix, self.broken_roundtrip = broken_prefix, broken_roundtrip
        self.standalone_candidate_calls = 0

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages == MESSAGES and add_generation_prompt is True
        return list(self.prefix_ids) if tokenize else 'CHAT:'

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == 'CHAT:':
            return list(self.prefix_ids)
        if text == 'CHAT:' + CANDIDATE:
            prefix = [99, *self.prefix_ids[1:]] if self.broken_prefix else self.prefix_ids
            return [*prefix, *self.suffix]
        if text == CANDIDATE:
            self.standalone_candidate_calls += 1
            return [8, 9]  # Deliberately differs: standalone encoding is irrelevant.
        raise AssertionError('Unexpected or silently repaired literal candidate')

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert ids == self.suffix and skip_special_tokens is False and clean_up_tokenization_spaces is False
        return 'changed literal' if self.broken_roundtrip else CANDIDATE


class Backend(module.HFAnswerCompletion):
    def __init__(self, tokenizer=None, model=None, admitted=True):
        super().__init__('never-loaded', IDENTITY, admit_resources=admitted)
        self.fake_tokenizer, self.fake_model = tokenizer or Tokenizer(), model
        self.prepare_calls = self.ensure_calls = 0

    def _prepare(self):
        self.prepare_calls += 1
        self.tokenizer = self.fake_tokenizer

    def _ensure_model(self):
        self.ensure_calls += 1
        self.model = self.fake_model
        self.admission = {'device': 'cuda:0', 'cuda_allocator_limit_bytes': 3 * 1024**3,
                          'test_only_no_actual_cuda': True}


def test_metadata_retains_every_context_dependent_name_token_then_fixed_eos():
    tokenizer = Tokenizer()
    value = module.completion_metadata(tokenizer, MESSAGES, CANDIDATE)
    assert value['prefix_ids'] == [10, 11]
    assert value['candidate_token_ids'] == [2, 3, 4] and value['candidate_tokens'] == 3
    assert value['input_ids'] == [10, 11, 2, 3, 4] and value['input_tokens'] == 5
    assert value['target_token_ids'] == [2, 3, 4, 7] and value['target_tokens'] == 4
    assert value['target_prediction_positions'] == [1, 2, 3, 4] and value['logits_to_keep'] == 4
    assert value['raw_candidate'] == CANDIDATE and value['completion_eos_appended_to_input'] is False
    assert value['prefix_alignment_verified'] and value['candidate_roundtrip_verified'] and value['causal_shift_verified']
    assert tokenizer.standalone_candidate_calls == 0


def test_chat_prefix_can_contain_eos_without_an_appended_completion_eos():
    value = module.completion_metadata(Tokenizer(prefix_ids=[10, 7, 11]), MESSAGES, CANDIDATE)
    assert value['input_ids'] == [10, 7, 11, 2, 3, 4]
    assert value['completion_eos_appended_to_input'] is False
    assert value['target_token_ids'] == [2, 3, 4, 7]
    assert value['target_prediction_positions'] == [2, 3, 4, 5]


@pytest.mark.parametrize('candidate', ['', ' Mira Tanswick', 'Mira Tanswick ', '\tMira Tanswick'])
def test_literal_edge_whitespace_is_rejected_without_strip(candidate):
    with pytest.raises(ValueError, match='never strip'):
        module.completion_metadata(Tokenizer(), MESSAGES, candidate)


@pytest.mark.parametrize('tokenizer,match', [
    (Tokenizer(broken_prefix=True), 'strict prefix'),
    (Tokenizer(broken_roundtrip=True), 'complete literal'),
    (Tokenizer(prefix_ids=[]), 'nonempty chat prefix'),
    (Tokenizer(suffix=[]), 'nonempty suffix'),
    (Tokenizer(suffix=[2, 7]), 'fixed tokenizer EOS'),
])
def test_unstable_or_incomplete_completion_fails_in_tokenizer_only_stage(tokenizer, match):
    backend = Backend(tokenizer)
    with pytest.raises(ValueError, match=match):
        backend.validate_completion(MESSAGES, CANDIDATE)
    assert backend.model is None and backend.ensure_calls == 0


def test_chat_template_prefix_must_match_its_literal_encoding():
    class Misaligned(Tokenizer):
        def encode(self, text, **kwargs):
            return [9, 11] if text == 'CHAT:' else super().encode(text, **kwargs)
    with pytest.raises(ValueError, match='prefix tokenization'):
        module.completion_metadata(Misaligned(), MESSAGES, CANDIDATE)


@pytest.mark.parametrize('eos', [None, [7], -1, True])
def test_one_integer_eos_is_mandatory(eos):
    tokenizer = Tokenizer(); tokenizer.eos_token_id = eos
    with pytest.raises(ValueError, match='fixed tokenizer EOS'):
        module.completion_metadata(tokenizer, MESSAGES, CANDIDATE)


def test_constructor_and_unadmitted_score_do_not_load_or_import_model(monkeypatch):
    assert module.HFAnswerCompletion._prepare is HFGenerator._prepare
    assert module.HFAnswerCompletion._ensure_model is HFGenerator._ensure_model
    assert module.HFAnswerCompletion.verify_frozen_identity is HFGenerator.verify_frozen_identity
    backend = module.HFAnswerCompletion('never-loaded', IDENTITY)
    assert backend.model is backend.tokenizer is None
    monkeypatch.setattr(backend, '_ensure_model', lambda: pytest.fail('No weights without admission'))
    with pytest.raises(RuntimeError, match='admission'):
        backend.validate_completion(MESSAGES, CANDIDATE)
    with pytest.raises(RuntimeError, match='admission'):
        backend.score_completion(MESSAGES, CANDIDATE)
    assert backend.model is backend.tokenizer is None


def test_validate_completion_is_tokenizer_only_and_length_guard_precedes_model_load():
    backend = Backend()
    assert backend.validate_completion(MESSAGES, CANDIDATE)['target_tokens'] == 4
    assert backend.ensure_calls == 0 and backend.model is None
    backend = Backend(Tokenizer(prefix_ids=[10] * 4095))
    with pytest.raises(RuntimeError, match='truncation is forbidden'):
        backend.score_completion(MESSAGES, CANDIDATE)
    assert backend.ensure_calls == 0


@pytest.fixture
def fake_torch(monkeypatch):
    import torch as cpu
    calls = []
    def tensor(*args, **kwargs):
        assert kwargs.pop('device') == 'cuda:0'
        return cpu.tensor(*args, **kwargs)  # CPU only; no .to(cuda) or real CUDA calls.
    class TorchProxy:
        # Preserve Torch's internal symbol imports while replacing all CUDA entry points.
        def __getattr__(self, name):
            return getattr(cpu, name)
    proxy = TorchProxy()
    proxy.tensor, proxy.inference_mode = tensor, nullcontext
    proxy.cuda = SimpleNamespace(synchronize=lambda device: calls.append(('sync', device)),
        reset_peak_memory_stats=lambda device: calls.append(('reset', device)),
        max_memory_allocated=lambda device: 1024, max_memory_reserved=lambda device: 2048)
    monkeypatch.setitem(sys.modules, 'torch', proxy)
    return cpu, calls


class Model:
    config = SimpleNamespace(vocab_size=12)

    def __init__(self, cpu, bad=None):
        self.calls, self.bad = [], bad
        self.full_logits = cpu.arange(60, dtype=cpu.float64).reshape(1, 5, 12) / 13
        for position, target in zip([1, 2, 3, 4], [2, 3, 4, 7]):
            self.full_logits[0, position, target] += position * .7
        if bad == 'nonfinite':
            self.full_logits[0, 3, 0] = float('nan')  # Unselected vocab entry must still reject.

    def __call__(self, encoded, **kwargs):
        assert encoded.device.type == 'cpu'
        assert encoded.tolist() == [[10, 11, 2, 3, 4]]  # EOS is scored, never fed.
        assert kwargs['attention_mask'].tolist() == [[1] * 5]
        assert kwargs['use_cache'] is False and kwargs['logits_to_keep'] == 4
        self.calls.append(kwargs)
        tail = self.full_logits[:, -kwargs['logits_to_keep']:, :]
        if self.bad == 'shape':
            tail = tail[..., :-1]
        return SimpleNamespace(logits=tail)


def test_single_forward_uses_correct_causal_rows_full_vocab_first_token_and_eos(fake_torch):
    cpu, cuda_calls = fake_torch
    model = Model(cpu); backend = Backend(model=model)
    raw = backend.score_completion(MESSAGES, CANDIDATE)
    expected_logits, expected_lse = [], []
    for position, token in zip([1, 2, 3, 4], [2, 3, 4, 7]):
        row = model.full_logits[0, position].tolist()
        expected_logits.append(row[token])
        peak = max(row); expected_lse.append(peak + math.log(math.fsum(math.exp(x - peak) for x in row)))
    logp = [a - b for a, b in zip(expected_logits, expected_lse)]
    assert raw['target_token_logits'] == pytest.approx(expected_logits)
    assert raw['full_vocab_logsumexp'] == pytest.approx(expected_lse, abs=1e-12)
    assert raw['target_token_log_probabilities'] == pytest.approx(logp, abs=1e-12)
    assert raw['joint_log_probability'] == pytest.approx(math.fsum(logp), abs=1e-12)
    assert raw['joint_log_probability'] < logp[0] and raw['joint_log_probability'] < logp[-1]
    assert raw['full_vocab_normalizer_dtype'] == 'float64' and raw['logits_shape'] == [1, 4, 12]
    assert raw['actual_forward_passes'] == len(model.calls) == 1 and raw['output_tokens'] == 0
    assert raw['peak_cuda_allocated_bytes'] == 1024 and raw['peak_cuda_reserved_bytes'] == 2048
    assert backend.ensure_calls == 1 and ('reset', 0) in cuda_calls


@pytest.mark.parametrize('bad', ['shape', 'nonfinite'])
def test_full_vocab_shape_and_all_entries_must_be_valid(fake_torch, bad):
    cpu, _ = fake_torch
    model = Model(cpu, bad); backend = Backend(model=model)
    with pytest.raises(ValueError, match='finite full-vocabulary logits'):
        backend.score_completion(MESSAGES, CANDIDATE)
    assert len(model.calls) == 1


def test_out_of_vocabulary_eos_rejects_before_forward(fake_torch):
    cpu, _ = fake_torch
    tokenizer = Tokenizer(); tokenizer.eos_token_id = 12
    model = Model(cpu); backend = Backend(tokenizer, model)
    with pytest.raises(ValueError, match='complete model vocabulary'):
        backend.score_completion(MESSAGES, CANDIDATE)
    assert model.calls == []
