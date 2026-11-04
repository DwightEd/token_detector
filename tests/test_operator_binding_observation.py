"""Storage and complete first-gate execution on an official tiny observer."""

import numpy as np
import pytest
import torch

from experiments import operator_binding_observation as stage
from experiments.data import read_json, verify_seal
from experiments.datasets import Sample
from myproject.measure import answer_axis
from .test_streamed_observer import tiny, _observer


def test_streamed_npz_retains_dtype_layout_and_full_array(tmp_path, monkeypatch):
    # The chunk boundary crosses rows. Storage must reconstruct the whole tensor
    # without changing its coordinates, signed zero or dtype.
    monkeypatch.setattr(stage, 'MAX_STAGING_BYTES', 24)
    value = torch.arange(77, dtype=torch.float32).reshape(7, 11).t()
    value[0, 0] = -0.
    counts = {}
    output = tmp_path/'raw.npz'
    stage.stream_arrays(output, {'value': value, 'flags': np.array([True, False])}, counts=counts)
    with np.load(output, allow_pickle=False) as saved:
        assert saved['value'].shape == (11, 7)
        assert saved['value'].tobytes() == value.contiguous().numpy().tobytes()
        assert saved['flags'].dtype == np.bool_
    assert counts['max_host_staging_bytes'] <= 24
    with pytest.raises(FileExistsError):
        stage.stream_arrays(output, {'value': value}, counts=counts)


@pytest.fixture
def case():
    ids, p = [1, 2, 3, 5, 8, 13, 21, 31, 17, 10], 6
    response = dict(text='abcd', answer_ids=ids[p:], special_ids=[],
                    offsets=[[i, i+1] for i in range(4)], units=[dict(start=0, stop=4)])
    return dict(P=p, T=4, N=10,
        sample=Sample('tiny', 'owner', 'test', 'Data2txt', 'source', 'source', 'abcd'),
        response=response, axis=answer_axis(response), metadata={},
        axes=dict(token_ids=np.array(ids, dtype=np.int64),
            answer_ids=np.array(ids[p:], dtype=np.int64),
            source_mask=np.array([False, True, True, True, True, False] + [False]*4),
            prechoice_positions=np.arange(p-1, len(ids)-1, dtype=np.int64)))


def test_whole_original_axis_noop_four_heads_and_frozen_endpoint(tiny, case, tmp_path):
    counts = {}
    with _observer(tiny) as observer:
        result = stage.observe_case(observer, case, tmp_path/'complete', counts)
    assert result['completed'] and result['auroc'] is None
    assert result['annotation_reads'] == 0 and result['phase2_admitted'] is False
    assert counts['model_calls'] == 1 and counts['head_calls'] == 4
    assert counts['extra_mlp_calls'] == 8 and counts['extra_norm_calls'] == 8
    assert counts['final_norm_replays'] == 4 and counts['partition_o_calls'] == 6
    prediction = read_json(tmp_path/'complete/prediction.json')
    assert [row['token_id'] for row in prediction['tokens']] == case['response']['answer_ids']
    with np.load(tmp_path/'complete/raw/endpoints-native.npz') as native, np.load(
            tmp_path/'complete/raw/endpoints-source.npz') as source, np.load(
            tmp_path/'complete/readouts.npz') as readouts:
        expected_e = source['logp'] - native['logp']
        assert np.array_equal(readouts['source_e'], expected_e)
        expected_risk = torch.sigmoid(torch.from_numpy(expected_e)).numpy()
        np.testing.assert_array_equal([row['score'] for row in prediction['tokens']], expected_risk)
    verify_seal(tmp_path/'complete/raw')
    verify_seal(tmp_path/'complete')


@pytest.mark.parametrize('kind', ['exception', 'nonfinite'])
def test_first_head_failure_preserves_native_and_never_restarts(tiny, case, tmp_path, monkeypatch, kind):
    counts = {}
    output = tmp_path/'failed'
    with _observer(tiny) as observer:
        def fail(*args, **kwargs):
            if kind == 'exception':
                raise RuntimeError('injected first head failure')
            return torch.full((1, 4, 48), float('nan'))
        monkeypatch.setattr(observer, 'logits', fail)
        with pytest.raises((RuntimeError, ValueError)):
            stage.observe_case(observer, case, output, counts)
        assert observer.resource_report()['all_parameters_meta']
    assert counts['model_calls'] == 1 and counts['head_calls'] == 1
    assert (output/'raw/layer-00.npz').is_file()
    assert (output/'raw/kv-01.npz').is_file()
    if kind == 'nonfinite':
        with np.load(output/'raw/logits-native.npz') as saved:
            assert np.isnan(saved['logits']).all()
    assert read_json(output/'failure.json')['automatic_retry'] is False
    verify_seal(output)
    with _observer(tiny) as observer, pytest.raises(FileExistsError):
        stage.observe_case(observer, case, output, counts)
    assert counts['model_calls'] == 1


def test_invalid_finalnorm_return_is_retained_before_rejection(tiny, tmp_path):
    counts = {}
    with _observer(tiny) as observer, torch.inference_mode():
        raw = torch.arange(320, dtype=torch.float32).reshape(1, 10, 32)
        rows = torch.tensor([5, 6, 7, 8])
        hook = observer.native.model.norm.register_forward_hook(
            lambda _module, _args, value: torch.full_like(value, float('nan')))
        try:
            with pytest.raises(RuntimeError, match='finite full-N'):
                stage._norm_replay(observer, raw, rows, None,
                    name='failed', output=tmp_path, counts=counts)
        finally:
            hook.remove()
        assert observer.resource_report()['all_parameters_meta']
    with np.load(tmp_path/'finalnorm-failed.npz') as saved:
        assert np.isnan(saved['output']).all()
        np.testing.assert_array_equal(saved['input'], raw.numpy())
    assert counts['final_norm_replays'] == 1
