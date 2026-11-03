"""Independent tiny-official-model audit of same-forward postRoPE diagnostics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from .test_streamed_observer import tiny
from experiments.research.relation_intervention import (
    Branch, attention_diagnostic, capture_rows, q_patch,
)


@pytest.fixture
def native_facade(tiny):
    from transformers.models.llama import modeling_llama
    model = tiny['reference']
    # Synthetic instrument controls need informative attention. This affects
    # only a freshly randomized tiny test model and never original weights.
    with torch.no_grad():
        model.model.layers[0].self_attn.q_proj.weight.mul_(12)
        model.model.layers[0].self_attn.k_proj.weight.mul_(12)
    config = model.config
    observer = SimpleNamespace(native=model, layers=model.model.layers,
        device=torch.device('cpu'), implementation=modeling_llama,
        head_layout=lambda index: (config.num_attention_heads,
            config.num_key_value_heads, config.hidden_size//config.num_attention_heads))
    return observer, [1, 2, 3, 5, 8, 13, 21, 31, 17, 10]


def _prefix(observer, ids, row):
    with torch.inference_mode(), capture_rows(observer, [row], layer=0) as record:
        output = observer.native.model(torch.tensor([ids]), use_cache=True)
    return record, output.past_key_values


def test_condition_prefill_requires_future_mask(native_facade):
    observer, ids = native_facade
    row = 3
    record, cache = _prefix(observer, ids, row)
    diagnostic = attention_diagnostic(observer, record, cache, [row], layer=0,
                                     absolute=1e-6, relative=1e-6)
    assert diagnostic['attention'].shape == (4, 1, len(ids))
    assert not diagnostic['attention'][..., row+1:].any()
    assert diagnostic['av_max_error'] < 1e-6
    with pytest.raises(RuntimeError, match='headAV'):
        attention_diagnostic(observer, record, cache, [row], layer=0,
                             variant='omit_causal', absolute=1e-6, relative=1e-6)


@pytest.mark.parametrize('variant', ['omit_rope', 'interleave_gqa'])
def test_rejects_wrong_rope_and_kv_head_ownership(native_facade, variant):
    observer, ids = native_facade
    record, cache = _prefix(observer, ids, 3)
    with pytest.raises(RuntimeError, match='headAV'):
        attention_diagnostic(observer, record, cache, [3], layer=0,
                             variant=variant, absolute=1e-6, relative=1e-6)


def test_cached_last_query_uses_absolute_position_and_appended_self(native_facade):
    observer, ids = native_facade
    prefix_ids = ids[:-1]
    with torch.inference_mode():
        prefix = observer.native.model(torch.tensor([prefix_ids]), use_cache=True)
        with capture_rows(observer, [0], layer=0) as record:
            output = observer.native.model(torch.tensor([ids[-1:]]),
                                           past_key_values=prefix.past_key_values, use_cache=True)
    position = len(prefix_ids)
    diagnostic = attention_diagnostic(observer, record, output.past_key_values,
                                     [position], layer=0, absolute=1e-6, relative=1e-6)
    assert diagnostic['attention'].shape[-1] == position+1
    assert (diagnostic['attention'][..., -1] > 0).all()  # current self key is causal
    np.testing.assert_allclose(diagnostic['attention'].sum(-1), 1., atol=2e-6)
    with pytest.raises(RuntimeError, match='headAV'):
        attention_diagnostic(observer, record, output.past_key_values, [0], layer=0,
                             absolute=1e-6, relative=1e-6)


def test_capture_hook_sees_actual_postpatch_query(native_facade):
    observer, ids = native_facade
    donor = torch.full((1, len(ids), observer.native.config.hidden_size), .75)
    before = [(len(m._forward_hooks), len(m._forward_pre_hooks))
              for m in observer.native.modules()]
    with torch.inference_mode(), q_patch(observer, Branch('q_role'), donor, layer=0):
        with capture_rows(observer, [3], layer=0) as record:
            output = observer.native.model(torch.tensor([ids]), use_cache=True)
    assert torch.equal(record['query'], donor[:, 3:4])
    diagnostic = attention_diagnostic(observer, record, output.past_key_values,
                                     [3], layer=0, absolute=1e-6, relative=1e-6)
    assert diagnostic['av_max_error'] < 1e-6
    after = [(len(m._forward_hooks), len(m._forward_pre_hooks))
             for m in observer.native.modules()]
    assert before == after
