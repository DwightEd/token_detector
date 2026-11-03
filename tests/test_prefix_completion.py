"""CPU tensors/fake model only; the prefix backend never enters real CUDA."""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import data, prefix_completion as module
from experiments.semantic_qualification import HFGenerator, HF_RUNTIME
from tests.test_answer_completion import fake_torch
from tests.test_completion_trie import Tokenizer

MESSAGES = [dict(role='user', content='Synthetic source-only question.')]


@pytest.fixture
def identity(tmp_path):
    root = tmp_path / 'fake-model'; root.mkdir()
    payloads = {'config.json': dict(vocab_size=128), 'tokenizer_config.json': {}, 'tokenizer.json': {},
        'model.safetensors.index.json': dict(weight_map={'a': 'model.safetensors'}),
        'added_tokens.json': {}, 'special_tokens_map.json': {}, 'vocab.json': {}}
    for name, value in payloads.items(): (root / name).write_text(json.dumps(value), encoding='utf-8')
    (root / 'model.safetensors').write_bytes(b'fake-never-loaded-weights')
    (root / 'merges.txt').write_text('fake', encoding='utf-8')
    return root, dict(model_id='Qwen/Qwen2.5-3B-Instruct', runtime=deepcopy(HF_RUNTIME),
                     model_files_sha256={p.name: data.file_hash(p) for p in root.iterdir()})


class Model:
    training = False

    def __init__(self, cpu, bad=None, mutate=None):
        self.cpu, self.bad, self.mutate, self.calls = cpu, bad, mutate, []
        self.config = SimpleNamespace(vocab_size=128)

    def __call__(self, encoded, **kwargs):
        assert encoded.device.type == 'cpu'
        assert kwargs['logits_to_keep'] == 1 and kwargs['use_cache'] is False
        assert kwargs['attention_mask'].tolist() == [[1] * encoded.shape[1]]
        context = encoded.tolist()[0]
        self.calls.append(context)
        if self.bad == 'raise_second' and len(self.calls) == 2: raise RuntimeError('synthetic forward failure')
        dtype = self.cpu.bfloat16 if self.bad == 'bfloat16' else self.cpu.float32 if self.bad == 'dtype' else self.cpu.float16
        logits = self.cpu.zeros((1, 2 if self.bad == 'shape' else 1, 128), dtype=dtype)
        if self.bad == 'nonfinite': logits[0, 0, 127] = float('nan')
        if self.mutate is not None: self.mutate()
        return SimpleNamespace(logits=logits)


class Backend(module.HFPrefixCompletion):
    def __init__(self, identity, model=None, tokenizer=None, admitted=True):
        super().__init__(*identity, admit_resources=admitted)
        self.fake_model, self.fake_tokenizer = model, tokenizer or Tokenizer()
        self.prepare_calls = self.ensure_calls = 0

    def _prepare(self):
        self.prepare_calls += 1
        self.tokenizer = self.fake_tokenizer
        self.admission = dict(device='cuda:0', available_ram_bytes=3 * 1024**3,
            free_cuda_bytes=4 * 1024**3, cuda_allocator_limit_bytes=3 * 1024**3, test_only_no_actual_cuda=True)

    def _ensure_model(self):
        self.ensure_calls += 1
        self.model = self.fake_model


def test_lazy_constructor_and_inherited_resource_runtime_loading_policy(identity):
    assert module.HFPrefixCompletion._prepare is HFGenerator._prepare
    assert module.HFPrefixCompletion._ensure_model is HFGenerator._ensure_model
    assert module.HFPrefixCompletion.verify_frozen_identity is HFGenerator.verify_frozen_identity
    backend = module.HFPrefixCompletion(*identity)
    assert backend.model is backend.tokenizer is None
    with pytest.raises(RuntimeError, match='admission'): backend.validate_trie(MESSAGES, ['A', 'B'])
    result = backend.score_trie({})
    assert result['status'] == 'engineering_failed' and result['model_calls'] == 0
    assert backend.model is backend.tokenizer is None


def test_tokenizer_only_validation_before_any_model_load(identity):
    backend = Backend(identity)
    messages, candidates = deepcopy(MESSAGES), ['A', 'AB', 'B']
    before = deepcopy((messages, candidates))
    trie = backend.validate_trie(messages, candidates)
    assert trie['status'] == 'ok' and backend.ensure_calls == 0 and backend.model is None
    assert (messages, candidates) == before


def test_unique_prefix_only_forwards_full_fp16_retention_and_independent_cpu_oracle(identity, fake_torch):
    cpu, cuda_calls = fake_torch
    model = Model(cpu); backend = Backend(identity, model)
    messages, candidates = deepcopy(MESSAGES), ['A', 'AB', 'B']
    trie = backend.validate_trie(messages, candidates)
    frozen = deepcopy(trie)
    result = backend.score_trie(trie)
    assert result['status'] == 'scored', {key: result.get(key) for key in ('reason', 'error_stage', 'model_calls')}
    assert trie == frozen
    assert result['model_calls'] == result['actual_forward_passes'] == len(model.calls) == len(trie['nodes'])
    assert result['pure_kernel_model_calls'] == result['tree_result']['model_calls'] == 0
    assert len({tuple(c) for c in model.calls}) == len(model.calls)
    assert {tuple(c) for c in model.calls} == {tuple(n['context_ids']) for n in trie['nodes'].values()}
    assert result['input_tokens'] == sum(len(context) for context in model.calls)
    assert result['seconds_total'] >= result['seconds_forward'] >= 0 and result['output_tokens'] == 0
    assert backend.ensure_calls == 1 and ('reset', 0) in cuda_calls
    oracle = []
    for raw in result['raw_nodes']:
        vector = raw['full_fp16_vector']
        assert isinstance(vector, np.ndarray) and vector.dtype == np.float16 and vector.shape == (128,)
        assert vector.flags.writeable is False
        assert raw['context_sha256'] == module.kernel._digest(raw['context_ids'])
        assert raw['full_vector_sha256'] == hashlib.sha256(vector.tobytes()).hexdigest()
        assert raw['logits_shape'] == [1, 1, 128]
        assert raw['peak_cuda_allocated_bytes'] == 1024 and raw['peak_cuda_reserved_bytes'] == 2048
        oracle.append(dict(node_id=raw['node_id'], context_ids=raw['context_ids'], vocab_size=128,
                           full_vocab_logits=vector.tolist()))
    assert result['tree_result'] == module.kernel.score_trie(trie, oracle)
    archive = BytesIO()
    np.savez_compressed(archive, row=result['raw_nodes'][0]['full_fp16_vector'])
    archive.seek(0)
    with np.load(archive, allow_pickle=False) as packed:
        assert hashlib.sha256(packed['row'].tobytes()).hexdigest() == result['raw_nodes'][0]['full_vector_sha256']
    assert all('full_vocab_logits' not in raw for raw in result['raw_nodes'])


@pytest.mark.parametrize('kind', ['too_many_nodes', 'too_long', 'eos_outside', 'temporary_budget'])
def test_all_node_and_vocabulary_budgets_precede_weight_loading(identity, kind):
    tokenizer = Tokenizer()
    candidates = ['A' * 17, 'B' * 17] if kind == 'too_many_nodes' else ['A']
    if kind == 'too_long':
        class LongTokenizer(Tokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return [ord('Q')] * 4096 if tokenize else 'Q' * 4096
        tokenizer = LongTokenizer()
    elif kind == 'eos_outside': tokenizer.eos_token_id = 200
    elif kind == 'temporary_budget':
        (identity[0] / 'config.json').write_text(json.dumps(dict(vocab_size=5_000_000)), encoding='utf-8')
        identity[1]['model_files_sha256']['config.json'] = data.file_hash(identity[0] / 'config.json')
    backend = Backend(identity, tokenizer=tokenizer)
    with pytest.raises(ValueError): backend.validate_trie(deepcopy(MESSAGES), candidates)
    assert backend.ensure_calls == 0 and backend.model is None


@pytest.mark.parametrize('mutation', ['messages', 'candidates', 'trie', 'tokenizer', 'asset', 'identity', 'resource'])
def test_changed_validation_inputs_or_instrument_reject_before_weights(identity, mutation):
    backend = Backend(identity)
    messages, candidates = deepcopy(MESSAGES), ['A', 'B']
    trie = backend.validate_trie(messages, candidates)
    if mutation == 'messages': messages[0]['content'] = 'changed'
    elif mutation == 'candidates': candidates.append('C')
    elif mutation == 'trie': trie['completion_metadata']['A']['prefix_ids'][0] += 1
    elif mutation == 'tokenizer': backend.tokenizer.eos_token_id = 1
    elif mutation == 'asset': (identity[0] / 'model.safetensors').write_bytes(b'changed-weights')
    elif mutation == 'identity': backend.model_identity['runtime']['torch'] = 'changed'
    else: backend.admission['cuda_allocator_limit_bytes'] = 4 * 1024**3
    result = backend.score_trie(trie)
    assert result['status'] == 'engineering_failed' and result['error_stage'] == 'preflight'
    assert result['model_calls'] == 0 and result['raw_nodes'] == [] and backend.ensure_calls == 0


@pytest.mark.parametrize('bad', ['nonfinite', 'shape', 'dtype', 'bfloat16', 'raise_second'])
def test_failed_forward_capture_retains_partial_raw_and_cost_without_retry(identity, fake_torch, bad):
    cpu, _ = fake_torch
    model = Model(cpu, bad); backend = Backend(identity, model)
    trie = backend.validate_trie(deepcopy(MESSAGES), ['A', 'AB', 'B'])
    result = backend.score_trie(trie)
    assert result['status'] == 'engineering_failed' and result['automatic_retry'] is False
    assert result['model_calls'] == len(model.calls) == (2 if bad == 'raise_second' else 1)
    assert result['actual_forward_passes'] == 1 and len(result['raw_nodes']) == len(model.calls)
    assert all(raw['seconds'] >= 0 and raw['context_sha256'] for raw in result['raw_nodes'])
    if bad == 'nonfinite': assert np.isnan(result['raw_nodes'][0]['full_fp16_vector'][127])
    elif bad in ('shape', 'dtype'):
        assert result['raw_nodes'][0]['full_fp16_vector'] is None
        assert 'invalid_raw_logits' in result['raw_nodes'][0]
    elif bad == 'bfloat16':
        assert result['raw_nodes'][0]['full_fp16_vector'] is None
        assert result['raw_nodes'][0]['invalid_raw_bytes'].dtype == np.uint8
    else:
        assert result['raw_nodes'][0]['full_fp16_vector'] is not None
        assert result['raw_nodes'][1]['actual_forward_passes'] == 0


def test_mid_forward_metadata_mutation_stops_after_retaining_first_row(identity, fake_torch):
    cpu, _ = fake_torch
    messages = deepcopy(MESSAGES)
    model = Model(cpu, mutate=lambda: messages[0].update(content='changed during forward'))
    backend = Backend(identity, model)
    trie = backend.validate_trie(messages, ['A', 'AB', 'B'])
    result = backend.score_trie(trie)
    assert result['status'] == 'engineering_failed' and result['error_stage'] == 'immutability_guard'
    assert result['model_calls'] == len(result['raw_nodes']) == 1
    assert result['raw_nodes'][0]['full_fp16_vector'] is not None


def test_probability_rejection_retains_all_observations_and_true_pipeline_cost(identity, fake_torch, monkeypatch):
    cpu, _ = fake_torch
    model = Model(cpu); backend = Backend(identity, model)
    trie = backend.validate_trie(deepcopy(MESSAGES), ['A', 'AB', 'B'])
    monkeypatch.setattr(module.kernel, 'score_trie', lambda *args: dict(status='unknown', reason='synthetic_strict_gate'))
    result = backend.score_trie(trie)
    assert result['status'] == 'engineering_failed' and result['error_stage'] == 'pure_probability_gate'
    assert result['tree_result']['reason'] == 'synthetic_strict_gate'
    assert result['model_calls'] == result['actual_forward_passes'] == len(trie['nodes'])
    assert len(result['raw_nodes']) == len(trie['nodes']) and result['pure_kernel_model_calls'] == 0


@pytest.mark.parametrize('damage', ['measurement_error', 'over_limit'])
def test_required_resource_measurements_cannot_fail_open(identity, fake_torch, monkeypatch, damage):
    cpu, _ = fake_torch
    import torch
    if damage == 'measurement_error':
        def failed(device): raise RuntimeError('synthetic peak measurement failure')
        monkeypatch.setattr(torch.cuda, 'max_memory_allocated', failed)
    else:
        monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda device: 4 * 1024**3)
    model = Model(cpu); backend = Backend(identity, model)
    trie = backend.validate_trie(deepcopy(MESSAGES), ['A', 'AB', 'B'])
    result = backend.score_trie(trie)
    assert result['status'] == 'engineering_failed' and result['error_stage'] == 'resource_measurement'
    assert result['model_calls'] == result['actual_forward_passes'] == len(model.calls) == 1
    assert len(result['raw_nodes']) == 1 and result['raw_nodes'][0]['full_fp16_vector'] is not None
    assert result['automatic_retry'] is False
