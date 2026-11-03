"""Tiny official CPU engineering tests; no original weights or target labels."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy

import numpy as np
import pytest
import torch

from experiments import natural_relation_observation as stage
from experiments.data import read_json, verify_seal
from experiments.research.relation_intervention import capture_rows
from .test_streamed_observer import tiny, _observer


@contextmanager
def _synthetic_archive_capture(model, layer, start, stop_after_layer=False):
    """A test-only archiver oracle; every payload comes from official modules."""
    assert stop_after_layer is False
    block, records, handles = model.model.layers[layer], {}, []
    handles.append(block.register_forward_pre_hook(
        lambda m, args: records.update(residual=args[0][0, start:].detach().cpu().clone())))
    for name, module, full in [('query', block.self_attn.q_proj, False),
        ('key', block.self_attn.k_proj, True), ('value', block.self_attn.v_proj, True),
        ('attention_write', block.self_attn.o_proj, False), ('mlp_write', block.mlp, False)]:
        def save(module, args, output, name=name, full=full):
            records[name] = output[0, 0 if full else start:].detach().cpu().clone()
        handles.append(module.register_forward_hook(save))
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


@pytest.fixture
def original(tiny):
    ids, p = [1, 2, 3, 5, 8, 13, 21, 31, 17, 10], 6
    rows = list(range(p - 1, len(ids)))
    axes = dict(token_ids=ids, answer_ids=ids[p:], query_positions=rows,
                source_mask=[False, True, True, True, True, False, False, False, False, False])
    with torch.inference_mode(), _synthetic_archive_capture(tiny['reference'], 1, p - 1) as captured:
        expected = tiny['reference'].model(torch.tensor([ids]), use_cache=False).last_hidden_state
        expected_logits = tiny['reference'].lm_head(expected[:, rows[:-1], :])[0]
    values = dict(nodes=np.stack([captured[k].numpy() for k in
        ('residual', 'attention_write', 'mlp_write')], axis=1),
        query=captured['query'].numpy().reshape(len(rows), 4, 8),
        key=captured['key'].numpy().reshape(len(ids), 2, 8),
        value=captured['value'].numpy().reshape(len(ids), 2, 8))
    # The unused archived blocked world is deliberately nonfinite: the gate
    # must inspect only the explicitly requested native world.
    reference = {name: np.stack([value.astype(np.float16),
                 np.full_like(value, np.nan, dtype=np.float16)]) for name, value in values.items()}
    mapping = dict(answer_offsets=[[i, i + 1] for i in range(len(ids) - p)],
        era_frontier_status='known_token_order', era_first_token=3, era_complete_token=3,
        source_anchors={name: {'overlap_tokens': positions} for name, positions in
            [('number300', [2]), ('eraBC', [3]), ('G03', [1, 2, 3, 4]), ('G03_closure', [4]), ('G06', [4])]})
    return axes, reference, mapping, expected_logits


def _observe(tiny, original, output, *, counts=None):
    axes, reference, mapping, _ = original
    with _observer(tiny) as observer:
        result = stage.observe_original(observer, axes, reference, mapping,
            _synthetic_archive_capture, output, counts=counts, archival_layer=1,
            dtype_guard=lambda model: nullcontext({}))
        assert observer.resource_report()['all_parameters_meta']
    return result


def test_single_full_call_single_untiled_head_retains_all_layers_and_shifted_axes(tiny, original, tmp_path):
    output = tmp_path / 'observed'
    output.mkdir()
    result = _observe(tiny, original, output)
    assert result['actual_counts'] == {'model_calls': 1, 'head_calls': 1}
    assert result['layers'] == 2 and result['physical_heads_per_layer'] == [4, 4]
    assert result['query_rows'] == 5 and result['original_target_rows'] == 4
    assert result['new_detector_scores'] == 0 and result['auroc'] is None
    verify_seal(output / 'raw')
    verify_seal(output / 'attention')
    raw_axis = read_json(output / 'raw/query-axis.json')['rows']
    assert raw_axis[0]['receiver_kind'] == 'prompt_last'
    assert raw_axis[0]['target_answer_index'] == 0
    assert raw_axis[1]['receiver_answer_index'] == 0 and raw_axis[1]['target_answer_index'] == 1
    assert raw_axis[-1]['receiver_answer_index'] == 3 and raw_axis[-1]['target_answer_index'] is None
    assert not raw_axis[-1]['observed_next_target']
    with np.load(output / 'output.npz') as saved:
        np.testing.assert_allclose(saved['logits'], original[3].numpy(), atol=2e-6, rtol=2e-6)
        assert saved['logits'].shape == (4, 48)
    endpoints = read_json(output / 'original-token-endpoints.json')['tokens']
    assert [row['token_id'] for row in endpoints] == original[0]['answer_ids']
    assert len(endpoints) == 4
    assert endpoints[-1]['receiver_target_axis']['absolute_query_position'] == 8
    for index, row in enumerate(endpoints):
        expected = original[3][index].log_softmax(-1)[original[0]['answer_ids'][index]]
        assert row['logp'] == pytest.approx(float(expected), abs=2e-6)
    for layer in range(2):
        with np.load(output / f'attention/layer-{layer:02d}.npz') as arrays:
            attention = arrays['attention']
            assert attention.shape == (4, 5, 10)
            assert np.count_nonzero(attention[:, 0, 6:]) == 0
            assert np.count_nonzero(attention[:, 1, 7:]) == 0
            with np.load(output / f'attention/profile-{layer:02d}.npz') as profile:
                np.testing.assert_array_equal(profile['source_at_or_after_era_complete'],
                                              attention[..., [3, 4]].sum(-1))


@pytest.mark.parametrize('fault', ['archive', 'phase', 'head', 'head_shape', 'head_nonfinite', 'endpoint'])
def test_failed_gate_retains_native_data_and_never_retries(tiny, original, tmp_path, monkeypatch, fault):
    output = tmp_path / fault
    output.mkdir()
    counts = {'model_calls': 0, 'head_calls': 0}
    if fault == 'archive':
        original[1]['query'][0, 1, 2, 3] += np.float16(5)
    if fault == 'phase':
        real = stage.attention_diagnostic
        def broken(*args, **kwargs):
            if kwargs['layer'] == 1:
                raise RuntimeError('injected phase failure')
            return real(*args, **kwargs)
        monkeypatch.setattr(stage, 'attention_diagnostic', broken)
    if fault == 'endpoint':
        monkeypatch.setattr(stage, 'token_endpoints', lambda *args: (_ for _ in ()).throw(
            RuntimeError('injected endpoint failure')))
    with _observer(tiny) as observer:
        if fault == 'head':
            monkeypatch.setattr(observer, 'logits', lambda hidden: (_ for _ in ()).throw(
                RuntimeError('injected head failure')))
        if fault == 'head_shape':
            monkeypatch.setattr(observer, 'logits', lambda hidden: torch.zeros(1, hidden.shape[1], 47))
        if fault == 'head_nonfinite':
            monkeypatch.setattr(observer, 'logits', lambda hidden: torch.full((1, hidden.shape[1], 48), float('nan')))
        with pytest.raises((RuntimeError, ValueError)):
            stage.observe_original(observer, *original[:3], _synthetic_archive_capture,
                output, counts=counts, archival_layer=1, dtype_guard=lambda model: nullcontext({}))
        assert observer.resource_report()['all_parameters_meta']
    assert counts == {'model_calls': 1, 'head_calls': int(fault in {'head', 'head_shape', 'head_nonfinite', 'endpoint'})}
    verify_seal(output / 'raw')
    assert (output / 'raw/all-layer-KV.npz').is_file()
    assert (output / 'raw/layer-00.npz').is_file() and (output / 'raw/layer-01.npz').is_file()
    assert (output / 'raw/final-hidden.npz').is_file()
    if fault == 'phase':
        assert (output / 'attention/layer-00.npz').is_file()
    if fault in {'head_shape', 'head_nonfinite', 'endpoint'}:
        assert (output / 'output.npz').is_file()
    before = dict(counts)
    with _observer(tiny) as observer, pytest.raises(FileExistsError):
        stage.observe_original(observer, *original[:3], _synthetic_archive_capture,
            output, counts=counts, archival_layer=1, dtype_guard=lambda model: nullcontext({}))
    assert counts == before


def test_mainline_alignment_offsets_manual_anchors_and_unknown_boundary(monkeypatch):
    from experiments.datasets import Sample
    from experiments.relation_alignment import reconstruct_alignment
    from myproject.inputs import encode_source, encode_response
    from .test_relation_intervention import WordTokenizer
    source = 'Alpha 300 BC. 30'
    anchors = dict(G03=(0, 13), time_scope=(6, 12), number300=(6, 9),
                   eraBC=(10, 12), G03_closure=(12, 13), G06=(14, 16), angle30=(14, 16))
    monkeypatch.setattr(stage, 'ANCHORS', anchors)
    monkeypatch.setattr(stage, 'LITERALS', dict(number300='300', eraBC='BC', time_scope='300 BC'))
    sample = Sample(id='synthetic', source_id='source', split='test', task='QA',
                    prompt='Source:' + source, source=source, response='300 years ago.')
    tokenizer = WordTokenizer()
    prompt = encode_source(tokenizer, sample.prompt, sample.source)
    answer = encode_response(tokenizer, sample.response)
    ids = prompt['prompt_with_source'] + answer['answer_ids']
    axes = dict(token_ids=ids, answer_ids=answer['answer_ids'],
                source_mask=prompt['source_mask'] + [False] * len(answer['answer_ids']),
                query_positions=list(range(len(prompt['prompt_with_source']) - 1, len(ids))))
    inventory = dict(schema=stage.CONTRACT, source_sha256=stage.text_hash(source),
                     fields=[], lexical_occurrences=[])
    previous = reconstruct_alignment(sample, tokenizer, axes, inventory)
    previous.update(id=sample.id, source_id=sample.source_id)
    prepared = stage.prepare_axes(sample, tokenizer, axes, previous)
    assert prepared['era_frontier_status'] == 'known_token_order'
    assert prepared['source_anchors']['number300']['literal'] == '300'
    assert prepared['source_anchors']['eraBC']['literal'] == 'BC'
    assert prepared['response_anchors']['years'][0]['character_span'] == [4, 9]
    assert len(prepared['answer_offsets']) == len(axes['answer_ids'])
    broken = deepcopy(previous)
    broken['alignment']['offsets_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='character axis'):
        stage.prepare_axes(sample, tokenizer, axes, broken)
    # A deliberately wider token makes the lexical boundaries explicit rather
    # than repairing them to an apparently precise semantic occurrence.
    monkeypatch.setattr(stage, 'ANCHORS', dict(anchors, eraBC=(11, 12)))
    monkeypatch.setattr(stage, 'LITERALS', dict(number300='300', eraBC='C'))
    crossed = stage.prepare_axes(sample, tokenizer, axes, previous)
    assert crossed['source_anchors']['eraBC']['status'] == 'boundary_overlap'
    assert crossed['source_anchors']['eraBC']['boundary_crossing_tokens']


@pytest.mark.parametrize('kind', ['last_query_removed', 'shifted_target', 'nonboolean_source', 'future_source'])
def test_changed_original_axes_rejected_before_any_model_attempt(tiny, original, tmp_path, kind):
    axes = deepcopy(original[0])
    if kind == 'last_query_removed':
        axes['query_positions'].pop()
    elif kind == 'shifted_target':
        axes['answer_ids'][0] = 4
    elif kind == 'nonboolean_source':
        axes['source_mask'][0] = 0
    else:
        axes['source_mask'][-1] = True
    counts = {'model_calls': 0, 'head_calls': 0}
    with pytest.raises(ValueError, match='axes changed'):
        stage.query_axis(axes, original[2]['answer_offsets'])
    assert counts == {'model_calls': 0, 'head_calls': 0}


def test_full_vocabulary_probability_is_not_two_candidate_share():
    logits = torch.tensor([[0., 1., 9., -2.], [6., 0., -1., 1.]], dtype=torch.float32)
    rows = stage.token_endpoints(logits, [1, 0], [[0, 1], [1, 2]])
    assert rows[0]['probability'] < .001 and rows[0]['rank'] == 2
    assert not rows[0]['probability_above_half']
    assert rows[1]['probability'] > .9 and rows[1]['probability_above_half']
    with pytest.raises(ValueError):
        stage.token_endpoints(logits, [1], [[0, 1]])


def test_profiles_keep_all_era_frontier_keys_and_unknown_boundary():
    axes = dict(token_ids=[1, 2, 3, 4, 5, 6], answer_ids=[5, 6],
                query_positions=[3, 4, 5], source_mask=[False, True, True, True, False, False])
    attention = np.arange(2 * 3 * 6, dtype=np.float32).reshape(2, 3, 6)
    mapping = dict(source_anchors={'eraBC': {'overlap_tokens': [2]}},
                   era_frontier_status='known_token_order', era_first_token=2, era_complete_token=2)
    result = stage.profiles({'attention': attention}, axes, mapping)
    np.testing.assert_array_equal(result['source_at_or_after_era_complete'], attention[..., [2, 3]].sum(-1))
    assert result['source_at_or_after_era_complete'].shape == (2, 3)
    mapping['era_frontier_status'] = 'unknown'
    assert 'source_at_or_after_era_complete' not in stage.profiles({'attention': attention}, axes, mapping)
