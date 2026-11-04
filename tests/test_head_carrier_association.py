"""Tiny independent checks of the marginal-erasure operator, not real data."""

from itertools import combinations

import numpy as np
import pytest

from experiments.research.head_carrier_association import erase_head_carrier_association


def erase(a, v, **kwargs):
    return erase_head_carrier_association(a, v, all_visible=kwargs.pop('all_visible', True),
                                          **kwargs)


def deltas(result):
    return (result['delta_coefficients'][..., :, None]
            * result['value_difference'][..., None, :])


def test_random_batch_matches_direct_attention_and_preserves_both_marginals():
    rng = np.random.default_rng(17)
    a = rng.uniform(0, 0.4, (5, 7, 2, 2)).astype(np.float32)
    v = rng.normal(size=(1, 7, 2, 13)).astype(np.float32)
    original_a, original_v = a.copy(), v.copy()
    result = erase(a, v)
    b = result['erased_attention']
    native_av = np.einsum('...hs,...sd->...hd', a.astype(np.float64), v.astype(np.float64))
    erased_av = np.einsum('...hs,...sd->...hd', b, v.astype(np.float64))
    np.testing.assert_allclose(deltas(result), erased_av-native_av, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(b.sum(-1), a.astype(np.float64).sum(-1), rtol=2e-15)
    np.testing.assert_allclose(b.sum(-2), a.astype(np.float64).sum(-2), rtol=2e-15)
    np.testing.assert_allclose(b[..., 0, 0]*b[..., 1, 1],
                               b[..., 0, 1]*b[..., 1, 0], rtol=2e-15)
    assert (b >= 0).all() and result['passed']
    assert result['tuple_passed'].shape == (5, 7)
    assert all(type(value) is bool for value in result['gates'].values())
    np.testing.assert_array_equal(a, original_a)
    np.testing.assert_array_equal(v, original_v)


def test_factor_norm_is_two_head_concat_not_zero_sum():
    a = np.array([[0.7, 0.1], [0.2, 0.6]], dtype=np.float32)
    v = np.array([[1, 2, -3], [-2, 5, 1]], dtype=np.float32)
    result = erase(a, v)
    message = deltas(result)
    assert result['concat_norm'] > 0
    np.testing.assert_allclose(result['head_norms'], np.linalg.norm(message, axis=-1))
    np.testing.assert_allclose(result['concat_norm'], np.linalg.norm(message.reshape(-1)))
    np.testing.assert_array_equal(message[0]+message[1], np.zeros(3))


def test_zero_mass_is_exact_noop_including_signed_zero():
    a = np.array([[0., -0.], [-0., 0.]], dtype=np.float32)
    result = erase(a, np.array([[4, -1], [-5, 8]], dtype=np.float32))
    np.testing.assert_array_equal(result['erased_attention'].view(np.uint64),
                                  a.astype(np.float64).view(np.uint64))
    assert result['zero_mass'] and not result['mi_defined'] and not result['chi_defined']
    assert result['mi'] == result['chi'] == result['kappa'] == result['concat_norm'] == 0
    assert result['passed']


def test_empty_row_and_zero_cells_are_valid_without_log_smoothing():
    a = np.array([[0, 0], [0.3, 0.7]], dtype=np.float32)
    result = erase(a, np.eye(2, dtype=np.float32))
    np.testing.assert_allclose(result['erased_attention'], a, rtol=2e-15, atol=0)
    assert result['mi_defined'] and not result['chi_defined']
    assert result['chi'] == result['mi'] == 0 and result['passed']


def test_diagonal_coupling_has_known_mutual_information_and_censored_chi():
    a = np.eye(2, dtype=np.float32)
    result = erase(a, np.eye(2, dtype=np.float32))
    np.testing.assert_array_equal(result['erased_attention'], np.full((2, 2), 0.5))
    assert result['mi'] == pytest.approx(np.log(2), abs=1e-15)
    assert not result['chi_defined'] and result['chi'] == 0
    assert result['kappa'] == 0.5 and result['passed']


def test_rank_one_table_is_numerical_noop_not_required_bit_identity():
    a = np.outer(np.array([0.125, 0.375], np.float32),
                 np.array([0.25, 0.75], np.float32))
    result = erase(a, np.array([[1, -4], [3, 8]], dtype=np.float32))
    assert result['passed']
    np.testing.assert_allclose(result['erased_attention'], a, rtol=2e-15, atol=0)
    assert abs(result['mi']) < 1e-15
    assert abs(result['chi']) < 1e-15
    assert result['concat_norm'] == 0


def test_same_value_and_common_translation_invariance():
    rng = np.random.default_rng(20)
    a = np.array([[0.4, 0.01], [0.02, 0.3]], dtype=np.float32)
    one = rng.integers(-8, 8, size=(1, 11)).astype(np.float32)
    same = erase(a, np.repeat(one, 2, axis=0))
    np.testing.assert_array_equal(deltas(same), np.zeros((2, 11)))
    assert same['concat_norm'] == 0 and same['passed']
    v = rng.integers(-8, 8, size=(2, 11)).astype(np.float32)
    shift = rng.integers(-8, 8, size=(1, 11)).astype(np.float32)
    np.testing.assert_array_equal(deltas(erase(a, v)), deltas(erase(a, v+shift)))


def test_shared_output_projection_annihilates_erasure_but_distinct_heads_need_not():
    a = np.array([[0.6, 0.1], [0.1, 0.4]], dtype=np.float32)
    v = np.array([[1, 2, 3], [4, 0, -1]], dtype=np.float32)
    result = erase(a, v)
    message = deltas(result)
    o1 = np.arange(15, dtype=np.float64).reshape(3, 5)
    np.testing.assert_array_equal(message[0]@o1+message[1]@o1, np.zeros(5))
    o2 = o1[::-1].copy()
    expected = result['kappa']*(v[0].astype(float)-v[1].astype(float))@(o2-o1)
    np.testing.assert_allclose(message[0]@o1+message[1]@o2, expected, rtol=2e-15)


def test_full_contiguous_gqa_block_and_each_of_48_pairs_are_independent():
    rng = np.random.default_rng(5)
    attention = rng.uniform(0.01, 0.4, (32, 4)).astype(np.float32)
    values = rng.normal(size=(8, 4, 9)).astype(np.float32)
    outputs = rng.normal(size=(32, 9, 6))
    expanded = np.repeat(values, 4, axis=0)
    base = np.einsum('hs,hsd->hd', attention.astype(float), expanded.astype(float))
    pairs = [(g, h1, h2) for g in range(8)
             for h1, h2 in combinations(range(g*4, g*4+4), 2)]
    assert len(pairs) == 48
    for g, h1, h2 in pairs:
        selected = attention[[h1, h2]][:, [0, 3]]
        result = erase(selected, values[g, [0, 3]])
        changed = attention.astype(np.float64).copy()
        changed[np.ix_([h1, h2], [0, 3])] = result['erased_attention']
        actual = np.einsum('hs,hsd->hd', changed, expanded.astype(float))-base
        observed = np.einsum('hd,hdo->o', actual, outputs)
        predicted = (deltas(result)[0]@outputs[h1]+deltas(result)[1]@outputs[h2])
        np.testing.assert_allclose(observed, predicted, rtol=1e-12, atol=2e-15)
        np.testing.assert_allclose(changed.sum(0), attention.astype(float).sum(0),
                                   rtol=2e-15, atol=1e-15)
        np.testing.assert_allclose(changed.sum(1), attention.astype(float).sum(1), rtol=2e-15)
        assert result['passed']


def test_head_and_carrier_permutations_covary_without_semantic_sign():
    a = np.array([[0.5, 0.1], [0.2, 0.6]], dtype=np.float32)
    v = np.array([[1, 2], [-2, 4]], dtype=np.float32)
    native = erase(a, v)
    heads = erase(a[::-1], v)
    carriers = erase(a[:, ::-1], v[::-1])
    np.testing.assert_allclose(deltas(heads), deltas(native)[::-1])
    np.testing.assert_allclose(deltas(carriers), deltas(native))
    assert heads['chi'] == pytest.approx(-float(native['chi']))
    assert carriers['chi'] == pytest.approx(-float(native['chi']))
    assert heads['mi'] == pytest.approx(float(native['mi']))
    assert carriers['mi'] == pytest.approx(float(native['mi']))


def test_compact_broadcast_values_do_not_expand_in_factorized_result():
    a = np.full((6, 4, 2, 2), [[0.4, 0.1], [0.1, 0.3]], dtype=np.float32)
    core = np.arange(4*2*7, dtype=np.float32).reshape(1, 4, 2, 7)
    result = erase(a, np.broadcast_to(core, (6, 4, 2, 7)))
    assert result['value_difference'].shape == (1, 4, 7)
    assert result['delta_coefficients'].shape == (6, 4, 2)
    assert deltas(result).shape == (6, 4, 2, 7)
    assert result['passed']


def test_tiny_float32_subnormals_and_unbalanced_mass_remain_defined():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    a = np.array([[[tiny, tiny], [tiny, tiny]],
                  [[1., tiny], [tiny, tiny]]], dtype=np.float32)
    result = erase(a, np.array([[1, -1], [-1, 2]], dtype=np.float32))
    assert np.isfinite(result['mi']).all() and np.isfinite(result['chi']).all()
    assert result['mi_defined'].all() and result['chi_defined'].all()
    assert result['passed']


@pytest.mark.parametrize('a,v', [
    (np.zeros((2, 3), np.float32), np.zeros((2, 4), np.float32)),
    (np.zeros((2, 2), np.float32), np.zeros((3, 4), np.float32)),
    (np.zeros((2, 2), np.float64), np.zeros((2, 4), np.float32)),
    (np.zeros((2, 2), np.float32), np.zeros((2, 4), np.int32)),
    (np.zeros((0, 2, 2), np.float32), np.zeros((2, 4), np.float32)),
    (np.zeros((2, 2), np.float32), np.zeros((2, 0), np.float32)),
    (np.zeros((3, 2, 2), np.float32), np.zeros((4, 2, 5), np.float32)),
    (np.full((2, 2), -1, np.float32), np.zeros((2, 4), np.float32)),
    (np.full((2, 2), np.inf, np.float32), np.zeros((2, 4), np.float32)),
    (np.zeros((2, 2), np.float32), np.full((2, 4), np.nan, np.float32)),
])
def test_invalid_inputs_fail_before_numeric_work(a, v):
    with pytest.raises(ValueError):
        erase(a, v)


@pytest.mark.parametrize('visible', [False, 1, 'True',
    np.array([[True, False], [True, True]]), np.ones((2,), dtype=bool)])
def test_visibility_is_explicit_and_false_is_rejected_even_for_zero_mass(visible):
    with pytest.raises(ValueError):
        erase(np.zeros((2, 2), np.float32), np.zeros((2, 4), np.float32),
              all_visible=visible)


def test_attention_shaped_bool_visibility_is_accepted():
    a = np.ones((2, 2), dtype=np.float32)
    result = erase(a, np.eye(2, dtype=np.float32), all_visible=np.ones_like(a, dtype=bool))
    assert result['passed']


def test_diagnostic_result_has_no_risk_or_semantic_verdict():
    result = erase(np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32))
    assert not ({'risk', 'score', 'truth', 'prediction', 'source_compatible'} & set(result))
