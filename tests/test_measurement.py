"""A tiny CPU Llama verifies the four source worlds and temporary resources."""

from unittest.mock import patch

import numpy as np
import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from myproject.model import attention_backend, prefill
from myproject.observer import Observer
from myproject.source import source_full


def dense_logp(model, prompt, answer):
    ids = torch.tensor([[*prompt, *answer[:-1]]])
    with torch.no_grad(), attention_backend(model, "eager"):
        hidden = model.model(input_ids=ids, use_cache=False).last_hidden_state[0]
        logp = model.lm_head(hidden[len(prompt) - 1:]).float().log_softmax(-1)
    return logp.gather(1, torch.tensor(answer)[:, None])[:, 0].numpy()


@pytest.fixture
def observer():
    torch.manual_seed(7)
    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=128, attention_dropout=0.,
    )).eval().requires_grad_(False)
    model.set_attn_implementation("sdpa")
    return Observer(model)


def test_four_source_worlds_match_independent_dense_forward(observer):
    source = {"prompt_with_source": [1, 3, 4, 5], "prompt_without_source": [1, 4],
              "source_mask": [False, True, False, True]}
    response = {"answer_ids": [6, 7, 8, 9], "units": [{"start": 0, "stop": 2}, {"start": 2, "stop": 4}]}
    result = observer.measure(source, response)
    full = (dense_logp(observer.model, [1, 4], response["answer_ids"])
            - dense_logp(observer.model, [1, 3, 4, 5], response["answer_ids"]))
    local = np.concatenate([
        dense_logp(observer.model, [1, 4], answer)
        - dense_logp(observer.model, [1, 3, 4, 5], answer)
        for answer in ([6, 7], [8, 9])
    ])
    np.testing.assert_allclose(result["delta"], np.column_stack((local, full)), atol=2e-6, rtol=0)
    np.testing.assert_array_equal(result["token_id"], response["answer_ids"])


def test_route_matches_explicit_physical_head_write_vectors(observer):
    model = observer.model
    source = {"prompt_with_source": [1, 3, 4, 5], "prompt_without_source": [1, 4],
              "source_mask": [False, True, False, True]}
    response = {"answer_ids": [6, 7, 8, 9], "units": [{"start": 0, "stop": 4}]}
    actual = observer.measure(source, response)["route"]
    values, weights, handles = {}, {}, []

    def save_value(index):
        def capture(module, args, output):
            values[index] = output[0].detach()
        return capture

    def save_weight(index):
        def capture(module, args, output):
            weights[index] = output[1][0].detach()
        return capture

    for index, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.v_proj.register_forward_hook(save_value(index)))
        handles.append(layer.self_attn.register_forward_hook(save_weight(index)))
    try:
        with torch.no_grad(), attention_backend(model, "eager"):
            model.model(input_ids=torch.tensor([[1, 3, 4, 5, 6, 7, 8, 9]]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    source_mask = torch.tensor([False, True, False, True, False, False, False, False])
    history_mask = torch.arange(8) >= 4
    layers = []
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        head_dim = attention.head_dim
        heads = attention.q_proj.out_features // head_dim
        kv_heads = attention.k_proj.out_features // head_dim
        value = values[index].reshape(-1, kv_heads, head_dim).repeat_interleave(heads // kv_heads, dim=1)
        projection = attention.o_proj.weight.float().reshape(-1, heads, head_dim).permute(1, 2, 0)
        writes = torch.einsum("shd,hdo->sho", value.float(), projection)
        magnitude = weights[index][:, 3:7].transpose(0, 1).float() * writes.norm(dim=-1).T
        source_total = magnitude[:, :, source_mask].sum(-1)
        source_total[0] -= magnitude[0, :, 3]
        difference = magnitude[:, :, history_mask].sum(-1) - source_total
        layers.append(difference.sum(-1) / magnitude.sum((1, 2)))
    np.testing.assert_allclose(actual, torch.stack(layers).mean(0).numpy(), atol=2e-6, rtol=0)


def test_forward_failure_restores_backend_and_removes_hooks(observer):
    model = observer.model
    cache = prefill(model, [1, 3, 4])
    with patch.object(model.model, "forward", side_effect=RuntimeError("native failure")):
        with pytest.raises(RuntimeError, match="native failure"):
            source_full(model, cache, [1, 3, 4, 5], [6, 7],
                        [False, True, False, True], observer.grams)
    assert model.config._attn_implementation == "sdpa"
    assert all(not layer.self_attn._forward_hooks for layer in model.model.layers)
