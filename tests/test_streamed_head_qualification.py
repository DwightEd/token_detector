"""Tiny official model audits; never execute original observer weights."""

import numpy as np
import pytest
import torch

from .test_streamed_observer import tiny, _observer
from experiments.streamed_head_qualification import (
    POLICY, _prefix_prerequisite, allocator_cleanup, cache_digest, clone_cache,
    compare_outputs, qualify_sequence,
)


def test_official_four_call_full_head_cache_noop(tiny):
    saved = []
    with _observer(tiny) as observer:
        result = qualify_sequence(observer, [1, 2, 3, 5, 8, 10], 11,
                                  np.array([False, True, True, True, False, False]),
                                  layer_index=1,
                                  observe=lambda name, output, counts, resources: saved.append((name, output)))
        assert result['qualified']
        assert result['counts'] == {'model_calls': 4, 'head_calls': 3}
        assert result['bulk_vs_cached']['within_budget']
        assert result['cached_vs_noop']['bitwise_equal']
        assert result['pristine_cache_unchanged']
        assert result['all_layer_clone_storage_independent']
        assert result['hook_and_residency_cleanup']
        assert [name for name, _ in saved] == ['bulk', 'cached', 'noop']
        assert all(value['logits'].shape == (48,) for _, value in saved)


def test_cache_clones_all_layers_and_preserves_original(tiny):
    with _observer(tiny) as observer:
        prefix = observer.hidden([1, 2, 3], use_cache=True).past_key_values
        before = cache_digest(prefix)
        cloned = clone_cache(prefix, observer.native.config, 3)
        original_pairs, clone_pairs = prefix.to_legacy_cache(), cloned.to_legacy_cache()
        assert len(clone_pairs) == 2
        for originals, clones in zip(original_pairs, clone_pairs, strict=True):
            for a, b in zip(originals, clones, strict=True):
                assert torch.equal(a, b)
                assert a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
        with torch.inference_mode():
            clone_pairs[0][0].add_(1)
        assert cache_digest(prefix) == before
        assert cache_digest(cloned) != before


def test_noop_requires_bitwise_not_merely_tolerance():
    original = {'hidden': np.ones(4, dtype=np.float32),
                'logits': np.arange(8, dtype=np.float32), 'logp': -2.}
    changed = {'hidden': original['hidden'].copy(), 'logits': original['logits'].copy(), 'logp': -2.}
    changed['logits'][3] += 1e-5
    comparison = compare_outputs(original, changed)
    assert comparison['within_budget']
    assert not comparison['bitwise_equal']
    changed['hidden'][0] += .1
    changed['logp'] += .01
    comparison = compare_outputs(original, changed)
    assert not comparison['within_budget']
    assert POLICY['original_token_logp_absolute'] == .002


def test_allocator_cleanup_refuses_resident_weights(tiny):
    with _observer(tiny) as observer:
        with observer.materialize('model.layers.0.self_attn.o_proj'):
            with pytest.raises(RuntimeError, match='no resident'):
                allocator_cleanup(observer)
        assert allocator_cleanup(observer)['live_caches_preserved']


def test_failure_keeps_exact_attempt_counts_and_does_not_retry(tiny, monkeypatch):
    counts = {'model_calls': 0, 'head_calls': 0}
    with _observer(tiny) as observer:
        def broken(hidden):
            raise RuntimeError('injected head failure')
        monkeypatch.setattr(observer, 'logits', broken)
        with pytest.raises(RuntimeError, match='injected'):
            qualify_sequence(observer, [1, 2, 3], 4, [False, True, False],
                             layer_index=1, counts=counts)
        assert counts == {'model_calls': 1, 'head_calls': 1}
        assert observer.resource_report()['all_parameters_meta']


@pytest.mark.parametrize('changed', [False, True])
def test_prefix_seal_requires_unchanged_shared_adapter(tmp_path, changed):
    from pathlib import Path
    from experiments import streamed_head_qualification as stage
    from experiments.data import file_hash, write_json
    base = Path(stage.__file__).parent
    assets, observer = {'synthetic': 'assets'}, {'synthetic': 'capture'}
    write_json(tmp_path/'plan.json', {'id': '16465', 'assets': assets,
        'observer_capture_identity': observer, 'complete_original_prompt_tokens': 252,
        'capture_sha256': {}, 'source_runtime': {'source_sha256': {
            'experiments.research/streamed_observer.py': '0'*64 if changed else file_hash(base/'research'/'streamed_observer.py'),
            'experiments/streamed_qualification.py': file_hash(base/'streamed_qualification.py')}}})
    write_json(tmp_path/'summary.json', {'all_qualified': True, 'model_calls': 2})
    for name in ('native', 'blocked'):
        write_json(tmp_path/f'{name}.json', {'qualified': True})
    write_json(tmp_path/'seal.json', {p.name: file_hash(p) for p in tmp_path.glob('*.json')})
    if changed:
        with pytest.raises(ValueError, match='shared execution dependency'):
            _prefix_prerequisite(tmp_path, '16465', assets, observer)
    else:
        assert _prefix_prerequisite(tmp_path, '16465', assets, observer)['complete_original_prompt_tokens'] == 252
