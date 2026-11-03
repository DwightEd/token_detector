"""Tiny official HF models test engineering identity, never original weights."""

import json
import threading
from types import MethodType

import pytest
import torch

from experiments.data import file_hash
from experiments.research.streamed_observer import ResourceLimits, SafeTensorStore, StreamedObserver


@pytest.fixture
def tiny(tmp_path):
    from safetensors.torch import save_file
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaRotaryEmbedding
    torch.manual_seed(137)
    config = LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        tie_word_embeddings=False, attention_dropout=0.,
                        attention_bias=False, mlp_bias=False,
                        max_position_embeddings=131072,
                        rope_theta=500000., rope_scaling={'rope_type': 'llama3',
                        'factor': 8., 'low_freq_factor': 1., 'high_freq_factor': 4.,
                        'original_max_position_embeddings': 8192})
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    model.model.rotary_emb = LlamaRotaryEmbedding(config, device='cpu')
    state = model.state_dict()
    keys = list(state)
    files, weight_map = {}, {}
    for index, names in enumerate((keys[:len(keys)//2], keys[len(keys)//2:])):
        name = f'model-{index}.safetensors'
        path = tmp_path / name
        save_file({key: state[key].contiguous() for key in names}, str(path))
        files[name] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
        weight_map.update({key: name for key in names})
    index = tmp_path / 'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': weight_map}), encoding='utf-8')
    config_data = json.loads(json.dumps(config.to_dict()))
    (tmp_path / 'config.json').write_text(json.dumps(config_data), encoding='utf-8')
    config_hash = file_hash(tmp_path / 'config.json')
    files['config.json'] = {'sha256': config_hash, 'bytes': (tmp_path/'config.json').stat().st_size}
    def original_linear(module, inputs):
        return torch.nn.functional.linear(inputs.float(), module.weight.float())
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = MethodType(original_linear, module)
    model.model.embed_tokens.register_forward_hook(lambda module, args, out: out.float())
    return {'root': tmp_path, 'files': files, 'index_hash': file_hash(index),
            'config': config_data, 'config_hash': config_hash, 'reference': model}


def _store(tiny):
    return SafeTensorStore(tiny['root'], index_sha256=tiny['index_hash'], expected_files=tiny['files'])


def _observer(tiny, **kwargs):
    return StreamedObserver(_store(tiny), tiny['config'], config_sha256=tiny['config_hash'],
                            device='cpu', original_contract=False, **kwargs)


def test_full_native_hidden_logits_and_residency(tiny):
    ids = [1, 5, 7, 8, 9, 10]
    with torch.inference_mode():
        expected = tiny['reference'].model(torch.tensor([ids])).last_hidden_state
        expected_logits = tiny['reference'].lm_head(expected)
    with _observer(tiny) as observer:
        actual = observer.hidden(ids).last_hidden_state
        actual_logits = observer.logits(actual)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(actual_logits, expected_logits, atol=1e-6, rtol=1e-6)
        assert actual.dtype == torch.float32
        report = observer.resource_report()
        assert report['all_parameters_meta']
        assert report['active_module'] is None
        assert report['module_loads'] == 5  # embedding + two blocks + norm + head
        assert not report['original_capture_qualified']


def test_official_prefix_cache_and_noop_hooks(tiny):
    ids = [1, 2, 4, 8, 16, 20]
    with _observer(tiny) as observer:
        full = observer.hidden(ids).last_hidden_state
        prefix = observer.hidden(ids[:4], use_cache=True)
        seen = []
        handle = observer.layers[1].self_attn.q_proj.register_forward_hook(
            lambda module, args, out: seen.append(out.detach().clone()))
        suffix = observer.hidden(ids[4:], past_key_values=prefix.past_key_values,
                                 use_cache=True).last_hidden_state
        handle.remove()
        torch.testing.assert_close(suffix, full[:, 4:], atol=2e-6, rtol=2e-6)
        assert len(seen) == 1 and seen[0].dtype == torch.float32
        noop = observer.layers[0].register_forward_hook(lambda module, args, out: out)
        repeated = observer.hidden(ids).last_hidden_state
        noop.remove()
        torch.testing.assert_close(repeated, full, atol=0, rtol=0)
        assert observer.resource_report()['all_parameters_meta']


def test_direct_child_requires_explicit_bounded_context(tiny):
    with _observer(tiny) as observer:
        path = 'model.layers.0.self_attn.o_proj'
        value = torch.randn(1, 3, 32)
        with pytest.raises(RuntimeError, match='not resident'):
            with torch.inference_mode():
                observer.native.get_submodule(path)(value)
        with observer.materialize(path) as module:
            actual = module(value)
            expected = tiny['reference'].model.layers[0].self_attn.o_proj(value)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            assert not module.weight.is_meta
        assert module.weight.is_meta
        assert observer.resource_report()['active_module'] is None


@pytest.mark.parametrize('failure', ['partial_preload', 'nested_capture', 'forward'])
def test_exception_cleanup(tiny, failure, monkeypatch):
    with _observer(tiny) as observer:
        if failure == 'partial_preload':
            original = observer.store.tensor
            calls = []
            def broken(name, device):
                calls.append(name)
                if len(calls) == 3:  # embedding + first block param succeeded
                    raise RuntimeError('injected partial preload')
                return original(name, device)
            monkeypatch.setattr(observer.store, 'tensor', broken)
        elif failure == 'nested_capture':
            def broken(module, args, out):
                raise RuntimeError('injected nested capture')
            observer.layers[0].self_attn.q_proj.register_forward_hook(broken)
        else:
            def broken(*args, **kwargs):
                raise RuntimeError('injected forward')
            monkeypatch.setattr(observer.layers[0].mlp, 'forward', broken)
        with pytest.raises(RuntimeError, match='injected'):
            observer.hidden([1, 2, 3])
        assert observer.resource_report()['all_parameters_meta']
        assert observer.resource_report()['active_module'] is None


def test_weight_index_config_identity_and_dtype_gates(tiny):
    with pytest.raises(ValueError, match='index SHA'):
        SafeTensorStore(tiny['root'], index_sha256='0'*64, expected_files=tiny['files'])
    store = _store(tiny)
    wrong = dict(tiny['config'], hidden_size=64)
    with pytest.raises(ValueError, match='config SHA'):
        StreamedObserver(store, wrong, config_sha256=tiny['config_hash'],
                         device='cpu', original_contract=False)
    name = next(iter(store.headers))
    store.headers[name]['shape'] = (999,)
    with pytest.raises(ValueError, match='parameter shape'):
        StreamedObserver(store, tiny['config'], config_sha256=tiny['config_hash'],
                         device='cpu', original_contract=False)


def test_resource_gates_and_explicit_close(tiny):
    with _observer(tiny, limits=ResourceLimits(max_sequence_tokens=2)) as observer:
        with pytest.raises(MemoryError, match='Sequence'):
            observer.hidden([1, 2, 3])
    with _observer(tiny, limits=ResourceLimits(max_planned_gpu_bytes=1)) as observer:
        with pytest.raises(MemoryError, match='GPU allocation budget'):
            observer.hidden([1, 2])
        assert observer.resource_report()['all_parameters_meta']
    observer = _observer(tiny)
    observer.close()
    with pytest.raises(RuntimeError, match='open no-grad'):
        observer.hidden([1])


@pytest.mark.parametrize('entry', ['embedding', 'head', 'child'])
def test_direct_entry_rows_and_autocast_are_rejected(tiny, entry):
    with _observer(tiny, limits=ResourceLimits(max_sequence_tokens=2)) as observer:
        def call(rows):
            if entry == 'embedding':
                return observer.native.model.embed_tokens(torch.ones(1, rows, dtype=torch.long))
            if entry == 'head':
                return observer.native.lm_head(torch.zeros(rows, 32))
            with observer.materialize('model.layers.0.self_attn.o_proj', rows=2) as module:
                return module(torch.zeros(rows, 32))
        with torch.inference_mode(), pytest.raises(MemoryError, match='rows'):
            call(3)
        assert observer.resource_report()['all_parameters_meta']
        with torch.inference_mode(), torch.autocast('cpu'), pytest.raises(RuntimeError, match='Autocast'):
            call(1)
        assert observer.resource_report()['all_parameters_meta']


def test_owner_thread_is_explicit_and_safe(tiny):
    with _observer(tiny) as observer:
        failures = []
        def other_thread():
            try:
                observer.hidden([1, 2])
            except RuntimeError as error:
                failures.append(str(error))
        task = threading.Thread(target=other_thread)
        task.start()
        task.join(timeout=10)
        assert failures and 'owner thread' in failures[0]
        assert observer.resource_report()['all_parameters_meta']
        with observer.materialize('model.layers.0.self_attn.o_proj') as resident:
            def rejected_parent():
                try:
                    with torch.inference_mode():
                        observer.layers[0](torch.zeros(1, 1, 32))
                except RuntimeError as error:
                    failures.append(str(error))
            task = threading.Thread(target=rejected_parent)
            task.start()
            task.join(timeout=10)
            assert not resident.weight.is_meta  # failed always_call cannot evict owner


def test_complete_header_keys_and_bf16_dtype(tiny):
    from safetensors import safe_open
    from safetensors.torch import save_file
    shard = next(n for n in tiny['files'] if n.endswith('.safetensors'))
    path = tiny['root'] / shard
    with safe_open(path, framework='pt') as archive:
        tensors = {key: archive.get_tensor(key).clone() for key in archive.keys()}
    first = next(iter(tensors))
    tensors[first] = tensors[first].float()
    save_file(tensors, str(path))
    tiny['files'][shard] = {'sha256': file_hash(path), 'bytes': path.stat().st_size}
    with pytest.raises(ValueError, match='unquantized BF16'):
        _store(tiny)
