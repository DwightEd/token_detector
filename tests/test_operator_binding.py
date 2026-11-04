"""Synthetic CPU oracles for H018; no pretrained checkpoints or tokenizers."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')

from experiments.research import operator_binding as kernel


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


class Square(torch.nn.Module):
    def forward(self, value):
        return value.square()


def square_fixture():
    layer = SimpleNamespace(post_attention_layernorm=torch.nn.Identity(), mlp=Square())
    x = torch.tensor([[[2., -3., 4.], [1., 5., -2.]]])
    source = torch.tensor([[[.5, -.25, 1.], [.25, 1., -.5]]])
    return layer, x, source


def test_nonlinear_oracle_counts_and_noop_identity():
    layer, x, source = square_fixture()
    before, original_source = x.clone(), source.clone()
    counts = {}
    with torch.inference_mode():
        result = kernel.local_effect(layer, x, source, x.clone(), x.square(), 0, counts)
    expected = 2 * x * source - source.square()
    assert torch.equal(result['delta_source'], expected)
    assert result['gates']['passed']
    assert list(result['norms']) == ['reference', 'source', 'sham', 'm0']
    assert kernel._same_bits(result['mlps']['reference'], result['mlps']['m0'])
    assert counts == dict(extra_norm_calls=4, extra_norm_rows=8,
                          extra_mlp_calls=4, extra_mlp_rows=8)
    assert torch.equal(x, before) and torch.equal(source, original_source)


def test_official_tiny_rmsnorm_mlp_native_rows_and_seed():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

    torch.manual_seed(123)
    config = LlamaConfig(hidden_size=8, intermediate_size=12, num_attention_heads=2,
                         num_key_value_heads=1, num_hidden_layers=1)
    layer = SimpleNamespace(post_attention_layernorm=LlamaRMSNorm(8), mlp=LlamaMLP(config))
    full = torch.randn(1, 7, 8)
    positions = [2, 5, 6]
    x, source = full[:, positions].clone(), torch.randn(1, 3, 8) * .1
    with torch.inference_mode():
        norm = layer.post_attention_layernorm(full)
        native = layer.mlp(norm)
        result = kernel.local_effect(layer, x, source, norm[:, positions],
                                     native[:, positions], 1, {})
        again = kernel.local_effect(layer, x, source, norm[:, positions],
                                    native[:, positions], 1, {})
    assert result['gates']['passed']
    assert torch.equal(result['sham_write'], again['sham_write'])
    assert torch.allclose(source.norm(dim=-1), result['sham_write'].norm(dim=-1),
                          atol=1e-6, rtol=1e-5)


def test_native_coordinate_mismatch_returns_all_raw_without_rescuing_gate():
    layer, x, source = square_fixture()
    with torch.inference_mode():
        result = kernel.local_effect(layer, x, source, x + .01, x.square(), 0, {})
    assert not result['gates']['passed']
    assert not result['gates']['reference_norm']['passed']
    assert result['gates']['reference_mlp']['passed']
    assert set(result['mlps']) == {'reference', 'source', 'sham', 'm0'}
    assert 'delta_source' in result


def test_zero_source_is_finite_identity_and_bad_zero_noise_refused(monkeypatch):
    layer, x, source = square_fixture()
    with torch.inference_mode():
        result = kernel.local_effect(layer, x, source * 0, x, x.square(), 0, {})
    assert result['gates']['passed']
    assert torch.count_nonzero(result['delta_source']) == 0
    assert torch.count_nonzero(result['delta_sham']) == 0
    monkeypatch.setattr(kernel, 'matched_noise', lambda value, seed: torch.zeros_like(value))
    counts = {}
    with torch.inference_mode(), pytest.raises(ValueError, match='zero sham'):
        kernel.local_effect(layer, x, source, x, x.square(), 0, counts)
    assert counts == {}


@pytest.mark.parametrize('bad', ['dtype', 'shape', 'nan', 'grad', 'amp', 'layer', 'layer_range'])
def test_invalid_interfaces_call_no_modules(bad):
    layer, x, source = square_fixture()
    counts = {}
    if bad == 'grad':
        with pytest.raises(RuntimeError, match='no-grad'):
            kernel.local_effect(layer, x, source, x, x.square(), 0, counts)
    else:
        context = torch.autocast('cpu') if bad == 'amp' else torch.inference_mode()
        with torch.inference_mode(), context:
            if bad == 'dtype':
                source = source.half()
            elif bad == 'shape':
                source = source[:, :1]
            elif bad == 'nan':
                source[0, 0, 0] = float('nan')
            with pytest.raises((ValueError, TypeError, RuntimeError)):
                kernel.local_effect(layer, x, source, x, x.square(),
                                    True if bad == 'layer' else 32 if bad == 'layer_range' else 0,
                                    counts)
    assert counts == {}


def test_exception_preserves_completed_calls_and_attempt_counts():
    layer, x, source = square_fixture()

    class FailSecond(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('synthetic module failure')
            return value.square()

    layer.mlp = FailSecond()
    counts = {}
    with torch.inference_mode(), pytest.raises(kernel.ReplayFailure) as caught:
        kernel.local_effect(layer, x, source, x, x.square(), 0, counts)
    assert set(caught.value.partial['norms']) == {'reference', 'source'}
    assert set(caught.value.partial['mlps']) == {'reference'}
    assert counts['extra_mlp_calls'] == 2 and counts['extra_norm_calls'] == 2


def test_signed_zero_is_a_real_bitwise_negative_control():
    positive, negative = torch.tensor([0.]), torch.tensor([-0.])
    assert torch.equal(positive, negative)
    assert not kernel._same_bits(positive, negative)
    assert not kernel._same_bits(positive.double(), negative.double())


def test_noop_failure_returns_raw_and_all_replays_share_contiguous_layout():
    layer, x, source = square_fixture()

    class ChangedFourth(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            assert value.is_contiguous()
            return value + (.001 if self.calls == 4 else 0.)

    layer.post_attention_layernorm = ChangedFourth()
    x, source = x.transpose(1, 2), source.transpose(1, 2)
    assert not x.is_contiguous()
    with torch.inference_mode():
        result = kernel.local_effect(layer, x, source, x.clone(), x.square(), 0, {})
    assert not result['gates']['passed']
    assert not result['gates']['m0_norm']['passed']
    assert not result['gates']['m0_mlp']['passed']
    assert set(result['mlps']) == {'reference', 'source', 'sham', 'm0'}


def test_ordered_double_accumulation_differs_from_p006_float32():
    accumulator = kernel.OrderedAccumulator(expected_layers=3)
    with torch.inference_mode():
        for index, scalar in enumerate([1e8, 1., -1e8]):
            value = torch.full((1, 2, 3), scalar)
            accumulator.add(index, value, -value, value)
        result = accumulator.finish()
    assert result['gates']['passed']
    assert torch.equal(result['source64'], torch.ones(1, 2, 3, dtype=torch.float64))
    assert torch.equal(result['source32'], torch.ones(1, 2, 3))
    assert torch.count_nonzero(result['p006_source32']) == 0
    assert torch.equal(result['sham64'], -result['source64'])
    assert result['cpu_transfer_bytes'] == 0
    assert not hasattr(accumulator, 'layers')


def test_accumulator_rejects_missing_reordered_duplicate_and_axis_drift():
    value = torch.ones(1, 2, 3)
    accumulator = kernel.OrderedAccumulator(expected_layers=2)
    with torch.inference_mode():
        with pytest.raises(ValueError, match='ascending'):
            accumulator.add(1, value, value, value)
        accumulator.add(0, value, value, value)
        with pytest.raises(ValueError, match='incomplete'):
            accumulator.finish()
        with pytest.raises(ValueError, match='ascending'):
            accumulator.add(0, value, value, value)
        with pytest.raises(ValueError, match='shape'):
            accumulator.add(1, value[:, :1], value[:, :1], value[:, :1])
        accumulator.add(1, value, value, value)
        assert accumulator.finish()['gates']['passed']
        with pytest.raises(ValueError, match='finished'):
            accumulator.finish()


def test_all_32_layers_required_and_cpu_reference_tampering_is_retained():
    value = torch.ones(1, 1, 2)
    accumulator = kernel.OrderedAccumulator()
    with torch.inference_mode():
        for index in range(32):
            accumulator.add(index, value, value, value)
        accumulator._cpu['source'][0, 0, 0] += 1
        result = accumulator.finish()
    assert result['layers'] == 32
    assert not result['gates']['passed']
    assert 'source64' in result


def test_full_vocab_denominator_and_signed_risk_are_not_binary_renormalization():
    native_logits = torch.tensor([[0., 0., 0.], [2., -1., 4.]])
    counter_logits = torch.tensor([[0., 0., 10.], [3., -1., 4.]])
    targets = torch.tensor([0, 0])
    with torch.inference_mode():
        native = kernel.full_vocab_endpoints(native_logits, targets)
        counter = kernel.full_vocab_endpoints(counter_logits, targets)
        score = kernel.score_endpoints(native, counter)
    assert torch.equal(native['logp'], native['target_logit'] - torch.logsumexp(native_logits, -1))
    assert score['logit_difference'][0] == 0
    assert score['lse_difference'][0] > 8
    assert score['e'][0] < -8 and score['risk'][0] < .001
    assert score['e'][1] > 0 and score['risk'][1] > .5
    assert score['gates']['passed']
    assert score['risk'].dtype == torch.float32


def test_decomposition_is_operational_tolerance_and_tampering_fails():
    targets = torch.tensor([0, 1])
    with torch.inference_mode():
        native = kernel.full_vocab_endpoints(torch.tensor([[1e4, 1e4], [3., 4.]]), targets)
        counter = kernel.full_vocab_endpoints(torch.tensor([[1e4 + 1, 1e4], [4., 4.]]), targets)
        score = kernel.score_endpoints(native, counter)
        expected = 32 * torch.finfo(torch.float32).eps * (
            native['target_logit'].abs() + counter['target_logit'].abs()
            + native['logsumexp'].abs() + counter['logsumexp'].abs() + 1)
        assert torch.equal(score['decomposition_tolerance'], expected)
        counter['logp'] = counter['logp'] + 5
        bad = kernel.score_endpoints(native, counter)
    assert score['gates']['passed']
    assert not bad['gates']['passed']
    assert 'risk' in bad and 'e' in bad


@pytest.mark.parametrize('bad', ['dtype', 'target', 'length', 'nan'])
def test_endpoint_axes_and_precision_are_fail_closed(bad):
    logits, targets = torch.zeros(2, 4), torch.tensor([0, 1])
    if bad == 'dtype':
        logits = logits.half()
    elif bad == 'target':
        targets[0] = 4
    elif bad == 'length':
        targets = targets[:1]
    else:
        logits[0, 0] = float('nan')
    with torch.inference_mode(), pytest.raises((ValueError, TypeError)):
        kernel.full_vocab_endpoints(logits, targets)
