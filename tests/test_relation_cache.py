"""Independent small numerical oracles for the archived-attention instrument."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from experiments.relation_cache_qualification import select_records
from experiments.research.relation_cache import (
    DEFAULT_POLICY, FIELDS, archive_headers, iter_attention, load_capture,
    qualify_arrays, rope_tables, validate_axes,
)


CONFIG = {'model_type': 'llama', 'hidden_size': 4096, 'num_attention_heads': 32,
          'num_key_value_heads': 8, 'rope_theta': 500000.,
          'max_position_embeddings': 131072,
          'rope_scaling': {'rope_type': 'llama3', 'factor': 8.,
                           'low_freq_factor': 1., 'high_freq_factor': 4.,
                           'original_max_position_embeddings': 8192}}


def _oracle(query, key, source, positions, prompt, *, blocked):
    """Full, float64 NumPy test oracle, never used by experiment production."""
    cosine, sine = rope_tables(CONFIG, np.arange(len(key)))
    c, s = cosine.numpy().astype(np.float64), sine.numpy().astype(np.float64)
    # Explicit per-head loops independently exercise contiguous GQA indexing.
    attention = np.zeros((len(query), 32, len(key)))
    for row, position in enumerate(positions):
        allowed = np.arange(len(key)) <= position
        if blocked and not source[position]:
            allowed &= ~source
        for head in range(32):
            q = query[row, head].astype(np.float64)
            k = key[:, head // 4].astype(np.float64)
            qrotate = np.r_[-q[64:], q[:64]]
            krotate = np.c_[-k[:, 64:], k[:, :64]]
            qr = q * c[position] + qrotate * s[position]
            kr = k * c + krotate * s
            score = kr @ qr / np.sqrt(128.)
            score[~allowed] = -np.inf
            probability = np.exp(score - score.max())
            attention[row, head] = probability / probability.sum()
    local = np.zeros((len(query), 32, 8), dtype=np.float16)
    groups = np.zeros((len(query), 5, 32), dtype=np.float32)
    for row, position in enumerate(positions):
        for sender in range(position + 1):
            if sender == position:
                group = 3
            elif source[sender]:
                group = 0
            elif sender < prompt:
                group = 4
            elif position - sender <= 8:
                group = 1
            else:
                group = 2
            groups[row, group] += attention[row, :, sender]
        for lag in range(1, 9):
            if position - lag >= prompt:
                local[row, :, lag - 1] = attention[row, :, position - lag]
    return attention, local, groups


@pytest.fixture(scope='module')
def synthetic():
    rng = np.random.default_rng(137)
    n, prompt = 24, 12
    t, rows = n - prompt, n - prompt + 1
    # Canary derived from pre-archival FP32, replay sees only rounded FP16.
    queries = rng.normal(0, 1.8, (2, rows, 32, 128)).astype(np.float32)
    keys = rng.normal(0, 1.8, (2, n, 8, 128)).astype(np.float32)
    positions = np.arange(prompt - 1, n, dtype=np.int64)
    source = np.zeros(n, dtype=bool)
    source[2:9] = True
    local, groups, full = [], [], []
    for world in range(2):
        a, l, g = _oracle(queries[world], keys[world], source, positions,
                          prompt, blocked=bool(world))
        full.append(a)
        local.append(l)
        groups.append(g)
    arrays = {'query': queries.astype(np.float16), 'key': keys.astype(np.float16),
              'local_attention': np.stack(local), 'group_mass': np.stack(groups),
              'token_ids': np.arange(n, dtype=np.int64),
              'answer_ids': np.arange(prompt, n, dtype=np.int64), 'source_mask': source,
              'query_positions': positions,
              'local_positions': positions[:, None] - np.arange(1, 9),
              'local_valid': positions[:, None] - np.arange(1, 9) >= prompt}
    record = {'audit': {'layer': 15, 'local_width': 8, 'input_tokens': n,
                       'prompt_length': prompt, 'answer_tokens': t, 'node_rows': rows,
                       'stop_after_layer': True, 'full_model_forwards': 0,
                       'groups': ['source', 'local', 'remote', 'self', 'other_prompt'],
                       'worlds': [{'blocked': False, 'executed_layers': 16,
                                   'full_model_completed': False},
                                  {'blocked': True, 'executed_layers': 16,
                                   'full_model_completed': False}]}}
    return arrays, record, np.stack(full)


def test_rounding_bound_and_negative_controls(synthetic):
    arrays, record, _ = synthetic
    result = qualify_arrays(arrays, CONFIG, record, query_chunk=4)
    assert result['status'] == 'qualified'
    assert set(result['controls'].values()) == {'rejected'}
    for metric in result['metrics']['faithful'].values():
        assert metric['violations'] == 0
        assert metric['precision_unknown_elements'] == 0
        assert metric['max_error_over_tolerance'] < 1


def test_all_heads_positions_and_world_mask(synthetic):
    arrays, record, oracle = synthetic
    for world in range(2):
        chunks = list(iter_attention(arrays['query'][world], arrays['key'][world],
            arrays['query_positions'], arrays['source_mask'], CONFIG,
            blocked=bool(world), query_chunk=5))
        actual = np.concatenate([part[2].numpy() for part in chunks])
        bound = np.concatenate([part[4] for part in chunks])
        assert actual.shape == oracle[world].shape
        assert np.all(np.abs(actual - oracle[world]) <= bound[:, :, None])
        assert np.allclose(actual.sum(-1), 1., atol=2e-6)
        for row, position in enumerate(arrays['query_positions']):
            assert not actual[row, :, position + 1:].any()
        if world:
            assert not actual[:, :, arrays['source_mask']].any()
    assert validate_axes(arrays, record)['prechoice_rows'] == [0, 12]


@pytest.mark.parametrize('name', ['answer_ids', 'query_positions', 'local_positions', 'local_valid'])
def test_reject_wrong_original_axis(synthetic, name):
    arrays, record, _ = synthetic
    modified = {k: v.copy() for k, v in arrays.items()}
    if modified[name].dtype == bool:
        modified[name].flat[0] = ~modified[name].flat[0]
    else:
        modified[name].flat[0] += 1
    with pytest.raises(ValueError, match='axis changed'):
        validate_axes(modified, record)


def test_reject_wrong_world_and_nonfinite(synthetic):
    arrays, record, _ = synthetic
    changed = copy.deepcopy(record)
    changed['audit']['worlds'][1]['executed_layers'] = 1
    with pytest.raises(ValueError, match='World'):
        validate_axes(arrays, changed)
    changed = {k: v.copy() for k, v in arrays.items()}
    changed['query'].flat[0] = np.inf
    with pytest.raises(ValueError, match='Non-finite'):
        validate_axes(changed, record)


def test_headers_and_memory_admission_before_decode(synthetic, tmp_path, monkeypatch):
    arrays, _, _ = synthetic
    archive = dict(arrays)
    for name in FIELDS - archive.keys():
        archive[name] = np.zeros((1,), dtype=np.float16)
    path = tmp_path / 'arrays.npz'
    np.savez_compressed(path, **archive)
    headers = archive_headers(path)
    assert headers['query']['shape'] == list(arrays['query'].shape)
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('decoded before admission'))
    with pytest.raises(MemoryError, match='budget'):
        load_capture(path, max_memory_bytes=1)


def test_canaries_choose_distinct_sources_without_labels():
    records = [{'id': str(i), 'source_id': str(i // 3), 'task': 'QA',
                'split': 'test', 'partition': 'test'} for i in range(12)]
    records.append({'id': 'train', 'source_id': 'x', 'task': 'QA',
                    'split': 'train', 'partition': 'fit'})
    selected = select_records({'records': records}, limit=3)
    assert len(selected) == len({r['source_id'] for r in selected}) == 3
    assert selected == select_records({'records': records[::-1]}, limit=3)
    assert len(select_records({'records': records}, limit=None)) == 12
    with pytest.raises(ValueError, match='outside QA TEST'):
        select_records({'records': records}, ids=['train'])


def test_policy_is_predeclared():
    assert DEFAULT_POLICY.fp32_logit_absolute == 2e-4
    assert DEFAULT_POLICY.fp32_dot_factor == 1024.
    assert DEFAULT_POLICY.precision_ceiling == .05


def test_loose_intervals_are_unknown_and_bad_canary_fails(synthetic):
    arrays, record, _ = synthetic
    unknown = qualify_arrays(arrays, CONFIG, record, include_controls=False,
                             policy=replace(DEFAULT_POLICY, precision_ceiling=1e-8))
    assert unknown['status'] == 'precision_unknown'
    assert unknown['metrics']['faithful']['group_mass']['precision_unknown_elements'] > 0
    changed = {k: v.copy() for k, v in arrays.items()}
    changed['group_mass'][0, 0, :, 0] = [0, 0, 0, 0, 1]
    failed = qualify_arrays(changed, CONFIG, record, include_controls=False)
    assert failed['status'] == 'numeric_contract_failed'
    assert failed['metrics']['faithful']['group_mass']['physical_heads'][0]['violations'] > 0
