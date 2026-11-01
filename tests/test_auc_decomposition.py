"""Compare the scalable diagnostic with exhaustive positive-negative pairs."""

import numpy as np
import pytest

from experiments.local_evaluation import auc_pair_decomposition


@pytest.mark.parametrize('seed', [17, 84, 20261102])
def test_counts_match_independent_pair_enumeration(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, 40)
    scores = rng.integers(-2, 3, len(y))  # Deliberate ties.
    answers = np.repeat(np.arange(4), 10)
    units = np.tile(np.repeat([0, 1], 5), 4)
    actual = auc_pair_decomposition(y, scores, answers, units)
    expected = {key: [0, 0] for key in ('pooled', 'within_span', 'between_spans')}
    for i in np.flatnonzero(y):
        for j in np.flatnonzero(~y.astype(bool)):
            twice = 2 * int(scores[i] > scores[j]) + int(scores[i] == scores[j])
            key = ('within_span' if answers[i] == answers[j] and units[i] == units[j]
                   else 'between_spans')
            for name in ('pooled', key):
                expected[name][0] += 1
                expected[name][1] += twice
    for key, (pairs, twice) in expected.items():
        assert actual[key]['positive_negative_pairs'] == pairs
        assert actual[key]['twice_concordant_pairs'] == twice
        assert actual[key]['auroc'] == twice / (2 * pairs)
    weight = actual['within_pair_fraction']
    np.testing.assert_allclose(actual['pooled']['auroc'],
                               weight * actual['within_span']['auroc']
                               + (1 - weight) * actual['between_spans']['auroc'], atol=1e-15)


def test_repeated_span_ids_reset_at_answer_boundaries_and_disjoint_runs():
    result = auc_pair_decomposition([0, 1, 0, 1, 0, 1], [1, 2, 3, 4, 5, 6],
                                    [0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 0])
    assert result['known_token_spans'] == 4
    assert result['within_span']['positive_negative_pairs'] == 2
    assert result['pooled']['positive_negative_pairs'] == 9


@pytest.mark.parametrize('labels', [[], [0], [1, 1], [0, 0]])
def test_undefined_auc_is_reported_as_none(labels):
    n = len(labels)
    result = auc_pair_decomposition(labels, np.zeros(n), np.zeros(n), np.zeros(n))
    assert result['pooled']['auroc'] is None
    assert result['between_spans']['auroc'] is None


@pytest.mark.parametrize('scores', [
    np.array([np.iinfo(np.int64).min, 0], dtype=np.int64),
    np.array([2**63, 2**63 + 1], dtype=np.uint64),
    np.array([False, True]),
])
def test_score_order_does_not_round_large_integers_to_float(scores):
    result = auc_pair_decomposition([0, 1], scores, [0, 0], [0, 0])
    assert result['pooled']['auroc'] == 1.


@pytest.mark.parametrize('labels,scores', [([0, None], [0, 1]), ([-1, 1], [0, 1]),
                                          ([0, 2], [0, 1]), ([0, 1], [0, np.nan])])
def test_unknown_labels_and_nonfinite_scores_are_not_silently_coerced(labels, scores):
    with pytest.raises(ValueError, match='Known binary'):
        auc_pair_decomposition(labels, scores, [0, 0], [0, 0])


def test_existing_evaluation_route_reports_decomposition_after_unknown_exclusion():
    from experiments.fullgap_dataset import _paired_views

    keys = [('a', index) for index in range(3)]
    labels = {key: {'label': label, 'source_id': 's',
                    'exclusion': 'unknown' if label is None else None}
              for key, label in zip(keys, [False, None, True], strict=True)}
    methods = {'frozen': {keys[0]: {'score': 1.}, keys[2]: {'score': 2.}}}
    native = {key: (0, 0) for key in keys}
    report, strata = _paired_views(
        keys, labels, methods, [{'id': 'a', 'source_id': 's'}], native, contrasts={})
    assert report['counts']['unknown_tokens'] == 1
    assert report['counts']['tokens_evaluable'] == 2
    assert report['methods']['frozen']['auc_pair_decomposition']['pooled']['auroc'] == 1.
    assert strata == {}


@pytest.mark.parametrize('known', [True, False])
def test_fullgap_evaluate_serializes_pooled_decomposition(tmp_path, monkeypatch, known):
    """Exercise real final serialization; frozen-data/model boundaries are fixtures."""
    from types import SimpleNamespace

    from experiments import data, evaluate as evaluation, fullgap_dataset as fullgap

    native, scores, prepared, output = [tmp_path / name for name in
                                        ('native', 'scores', 'prepared', 'evaluation')]
    for directory in (native, scores, prepared):
        directory.mkdir()
    for path in (native / 'seal.json', scores / 'seal.json', prepared / 'annotations.jsonl'):
        path.write_text('{}', encoding='utf-8')
    contract = {'native_granularity': 'token', 'label_provenance': 'fixture', 'label_scope': 'source support'}
    data.write_json(prepared / 'manifest.json', {
        'dataset': 'fixture', 'annotations_sha256': data.file_hash(prepared / 'annotations.jsonl'),
        'bootstrap_type': 'source'})
    data.write_json(scores / 'plan.json', {'protocol_sha256': 'fixture', 'source': {}})
    sample = SimpleNamespace(id='a', task='QA')
    plan = {'prepared': str(prepared), 'input_sha256': 'fixture', 'source': {}}
    keys = [('a', 0), ('a', 1)]
    labels = {key: {**dict.fromkeys(fullgap.TOKEN_IDENTITY), 'id': 'a', 'source_id': 's',
                    'label': bool(index) if known else None,
                    'exclusion': None if known else 'unknown'}
              for index, key in enumerate(keys)}
    rows = {key: {'score': float(index)} for index, key in enumerate(keys)} if known else {}
    monkeypatch.setattr(fullgap, 'research_identity', lambda: {})
    monkeypatch.setattr(fullgap, 'completed_identity', lambda value: value)
    monkeypatch.setattr(fullgap, '_verify_scores', lambda *_: (plan, [sample], [], ['witness']))
    monkeypatch.setattr(fullgap, '_response_metadata', lambda *_: ([{'id': 'a', 'source_id': 's'}], {}))
    monkeypatch.setattr(fullgap, '_label_contract', lambda *_: contract)
    monkeypatch.setattr(fullgap, '_native_token_identities', lambda *_: labels)
    monkeypatch.setattr(fullgap, '_native_token_units', lambda *_: {key: (0, 0) for key in keys})
    monkeypatch.setattr(fullgap, '_verify_method_evaluation', lambda *_: None)
    monkeypatch.setattr(fullgap, '_verify_paths', lambda *_: None)
    monkeypatch.setattr(fullgap, '_evaluation_rows', lambda *_: (labels, rows))

    def frozen_evaluation(_scores, _annotations, destination, **_kwargs):
        destination.mkdir()
        data.write_json(destination / 'seal.json', {})
        return {**contract, 'samples_scored': 1, 'unalignable_responses_excluded': 0,
                'coverage': None, 'bootstrap': {'seed': fullgap.BOOTSTRAP_SEED,
                                               'repetitions': fullgap.BOOTSTRAP_REPETITIONS},
                'metrics': {'units': 2 if known else 0, 'auroc': 1. if known else None,
                            'ap': 1. if known else None}}

    monkeypatch.setattr(evaluation, 'evaluate_dataset', frozen_evaluation)
    fullgap.evaluate(native, scores, output)
    saved = data.read_json(output / 'evaluation.json')
    assert saved['methods']['witness']['units'] == (2 if known else 0)
    decomposition = saved['methods']['witness']['auc_pair_decomposition']
    if known:
        assert decomposition['pooled']['auroc'] == 1.
        assert decomposition['within_span']['positive_negative_pairs'] == 1
    else:
        # Real evaluate_dataset currently rejects no-evaluable populations;
        # this fixture also checks the report boundary remains None-safe.
        assert decomposition is None
    assert data.read_json(output / 'seal.json')['evaluation.json'] == data.file_hash(output / 'evaluation.json')
