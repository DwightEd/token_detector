"""Synthetic policy/source/asset gates; no original model or cache values."""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.data import file_hash, write_json
from experiments.streamed_qualification import COORDINATE_POLICY, coordinate_summary, verify_assets


def test_fixed_coordinate_policy_all_heads_and_failures():
    saved = np.ones((4, 8, 128), dtype=np.float16)
    actual = saved.astype(np.float32)
    actual[0, 3, 5] += .001
    passed = coordinate_summary(actual, saved, head_axis=1)
    assert passed['violations'] == 0 and len(passed['physical_heads']) == 8
    actual[0, 3, 5] += .01
    failed = coordinate_summary(actual, saved, head_axis=1)
    assert failed['violations'] == 1
    assert failed['physical_heads'][3]['violations'] == 1
    assert COORDINATE_POLICY['fp32_backend_absolute'] == .002
    assert COORDINATE_POLICY['fp32_backend_relative'] == .0001


@pytest.mark.parametrize('kind', ['dtype', 'shape', 'nonfinite'])
def test_coordinate_contract_rejects_invalid(kind):
    saved = np.ones((3, 4), dtype=np.float16)
    actual = saved.astype(np.float32)
    if kind == 'dtype':
        actual = actual.astype(np.float16)
    elif kind == 'shape':
        actual = actual[:2]
    else:
        actual[0, 0] = np.nan
    with pytest.raises(ValueError):
        coordinate_summary(actual, saved)


def _fake_assets(tmp_path, *, armed):
    model = tmp_path / 'model'
    model.mkdir()
    rows, verified = [], []
    names = [f'model-{i}.safetensors' for i in range(4)] + [
        'config.json', 'tokenizer_config.json', 'model.safetensors.index.json',
        'tokenizer.json', 'special_tokens_map.json']
    for name in names:
        path = model / name
        path.write_bytes(b'synthetic asset')
        row = {'relative_path': f'model/{name}', 'size': path.stat().st_size,
               'sha256': file_hash(path)}
        rows.append(row)
        verified.append({'name': name, 'sha256': row['sha256'], 'bytes': row['size']})
    manifest = tmp_path / 'manifest-v1.json'
    write_json(manifest, {'data_root': str(tmp_path), 'files': rows})
    digest = file_hash(manifest)
    write_json(tmp_path / 'mirror-state-v1.json', {'status': 'complete', 'armed': armed,
        'active': 0, 'failed': 0, 'manifest_sha256': digest})
    write_json(tmp_path / 'independent-verification-v1.json', {
        'schema': 'h017-original-hf-assets-independent-verification-v1',
        'status': 'all9_assets_SHA_verified_291_BF16_headers_index_matched',
        'manifest_sha256': digest, 'weights_match_original_capture_MODEL_ENVIRONMENT': True,
        'mirror_successfully_disarmed': True, 'root_lock_absent': True,
        'tensor_count': 291, 'files': verified})
    return manifest, model


def test_asset_gate_requires_completed_disarmed_receipt(tmp_path):
    manifest, model = _fake_assets(tmp_path, armed=True)
    with pytest.raises(ValueError, match='complete/disarmed'):
        verify_assets(manifest, model)


def test_asset_gate_binds_exact_metadata_and_directory(tmp_path):
    manifest, model = _fake_assets(tmp_path, armed=False)
    result = verify_assets(manifest, model)
    assert len(result['files']) == 9
    assert not result['index_historical_identity']
    (model / 'config.json').write_bytes(b'changed asset!!')
    with pytest.raises(ValueError, match='metadata bytes'):
        verify_assets(manifest, model)


def test_ast_extraction_executes_only_pinned_selected_definitions(tmp_path, monkeypatch):
    from experiments import streamed_qualification as stage
    root = tmp_path / 'capture_code'
    root.mkdir()
    path = root / 'synthetic.py'
    path.write_text("raise RuntimeError('unselected top level')\n"
                    "class LayerCaptured(Exception): pass\n"
                    "def capture_selected_layer(): return 'selected'\n", encoding='utf-8')
    monkeypatch.setattr(stage, 'PRODUCER', {'synthetic.py': (file_hash(path),
                        ('LayerCaptured', 'capture_selected_layer'))})
    functions = stage.frozen_capture_functions(tmp_path)
    assert functions['capture_selected_layer']() == 'selected'
    path.write_text('modified', encoding='utf-8')
    with pytest.raises(ValueError, match='source changed'):
        stage.frozen_capture_functions(tmp_path)


def test_native_dtype_guard_precedes_historical_float_cast_and_cleans_up():
    import torch
    from experiments.streamed_qualification import _native_dtype_guards
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
        setattr(layer.self_attn, name, torch.nn.Identity())
    layer.mlp = torch.nn.Identity()
    model = SimpleNamespace(model=SimpleNamespace(layers=[None]*15+[layer]))
    with pytest.raises(RuntimeError, match='before capture conversion'):
        with _native_dtype_guards(model):
            layer.self_attn.q_proj(torch.zeros(1, dtype=torch.bfloat16))
    assert not layer._forward_pre_hooks
    assert not layer.self_attn.q_proj._forward_hooks
    with _native_dtype_guards(model) as observed:
        layer.self_attn.q_proj(torch.zeros(1, dtype=torch.float32))
    assert observed == {'query': 'torch.float32'}


@pytest.mark.parametrize('name', ['token_ids', 'answer_ids', 'query_positions', 'source_mask'])
def test_reference_axes_reject_silent_casts(tmp_path, name):
    from experiments.streamed_qualification import _reference_payload
    arrays = {'token_ids': np.arange(4, dtype=np.int64),
              'answer_ids': np.arange(2, 4, dtype=np.int64),
              'query_positions': np.arange(1, 4, dtype=np.int64),
              'source_mask': np.array([False, True, False, False]),
              **{key: np.zeros(1, dtype=np.float16) for key in ('nodes', 'query', 'key', 'value')}}
    arrays[name] = arrays[name].astype(np.float32)
    path = tmp_path / 'synthetic.npz'
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match='dtype|boolean'):
        _reference_payload(path, 2, 2, 4)
