"""Full-vocabulary synthetic oracles for the pure H015 tree contract."""
import builtins
from copy import deepcopy
import math

import pytest

from experiments.answer_completion import completion_metadata
from experiments.reconstruction_qualification import _META
from experiments.research import completion_trie as kernel


class Tokenizer:
    eos_token_id = 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert add_generation_prompt is True
        return [ord(c) for c in 'Q:'] if tokenize else 'Q:'

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        return ''.join(chr(i) for i in ids)


def metadata(names=('A', 'AB', 'B')):
    return {name: completion_metadata(Tokenizer(), [dict(role='user', content='Synthetic source-only question')], name)
            for name in names}


def observations(trie, value=0.):
    return [dict(node_id=key, context_ids=list(node['context_ids']), vocab_size=128,
                 full_vocab_logits=[value] * 128) for key, node in trie['nodes'].items()]


def test_existing_metadata_contract_and_full_vocab_uniform_oracle():
    framed = metadata()
    assert kernel.META_FIELDS == set(_META) == set(framed['A'])
    frozen = deepcopy(framed)
    trie = kernel.build_trie(framed)
    assert framed == frozen and trie['status'] == 'ok'
    rows = observations(trie)
    original_rows = deepcopy(rows)
    scores = kernel.score_trie(trie, rows)
    assert scores['status'] == 'ok' and rows == original_rows
    oracle = {name: 128. ** -len(path['steps']) for name, path in trie['paths'].items()}
    assert scores['candidate_pool_mass'] == pytest.approx(math.fsum(oracle.values()))
    assert 0 <= scores['candidate_pool_mass'] <= 1
    for name, path in scores['paths'].items():
        assert path['raw_joint_log_probability'] == pytest.approx(-len(path['steps']) * math.log(128.))
        assert path['joint_probability'] == pytest.approx(oracle[name])
        assert path['conditional_mass_given_candidate_pool'] == pytest.approx(oracle[name] / math.fsum(oracle.values()))
    assert math.fsum(p['conditional_mass_given_candidate_pool'] for p in scores['paths'].values()) == pytest.approx(1.)
    assert scores['conditional_mass_is_full_vocabulary_probability'] is False
    assert scores['calibrated_truth_probability'] is False and scores['model_calls'] == 0
    assert all('full_vocab_logits' not in node and node['full_vector_retained'] is False for node in scores['nodes'].values())


def test_shared_tokens_unique_contexts_and_prefix_name_eos_share_one_row():
    trie = kernel.build_trie(metadata(('Ana Li', 'Ana Li Jr', 'Bob Ray')))
    assert trie['counts']['unique_nodes'] < trie['counts']['logical_target_positions']
    assert len({tuple(node['context_ids']) for node in trie['nodes'].values()}) == len(trie['nodes'])
    short, long = trie['paths']['Ana Li'], trie['paths']['Ana Li Jr']
    short_eos = short['steps'][-1]
    long_next = long['steps'][len(short['candidate_token_ids'])]
    assert short_eos['node_id'] == long_next['node_id'] and short_eos['token_id'] == 0
    assert long_next['token_id'] == ord(' ')
    branch = trie['nodes'][short_eos['node_id']]
    assert {edge['token_id'] for edge in branch['children']} == {0, ord(' ')}
    assert next(edge for edge in branch['children'] if edge['token_id'] == 0)['next_node_id'] is None
    result = kernel.score_trie(trie, observations(trie))
    row = result['nodes'][short_eos['node_id']]
    assert len(row['edges']) == 2 and row['full_vocab_logsumexp'] == pytest.approx(math.log(128))
    assert result['counts']['node_observations'] == len(trie['nodes'])


@pytest.mark.parametrize('damage', ['empty', 'count', 'prefix', 'eos', 'internal_eos', 'bool_id', 'causal', 'alias'])
def test_malformed_framing_or_token_alias_is_explicit_unknown(damage):
    value = metadata(('A', 'B'))
    if damage == 'empty': value = {}
    elif damage == 'count': value['A']['target_tokens'] += 1
    elif damage == 'prefix':
        other = metadata(('B',))['B']; other['prefix_ids'][0] += 1
        other['input_ids'][0] += 1; value['B'] = other
    elif damage == 'eos': value['B']['eos_token_id'] = 1; value['B']['target_token_ids'][-1] = 1
    elif damage == 'internal_eos': value['A']['candidate_token_ids'][0] = 0
    elif damage == 'bool_id': value['A']['prefix_ids'][0] = True
    elif damage == 'causal': value['A']['target_prediction_positions'][0] += 1
    else:
        value['B'] = deepcopy(value['A']); value['B']['raw_candidate'] = 'B'
    result = kernel.build_trie(value)
    assert result['status'] == 'unknown' and result['reason']
    if damage == 'alias': assert result['reason'] == 'tokenized_candidate_alias'


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'extra', 'context', 'rank2', 'length',
                                    'vocab', 'nan', 'inf', 'bool', 'declared_logz'])
def test_observation_population_vectors_and_normalizer_source_are_strict(damage):
    trie = kernel.build_trie(metadata())
    rows = observations(trie)
    if damage == 'missing': rows.pop()
    elif damage == 'duplicate': rows.append(deepcopy(rows[0]))
    elif damage == 'extra': rows[0]['node_id'] = 'f' * 64
    elif damage == 'context': rows[0]['context_ids'][-1] += 1
    elif damage == 'rank2': rows[0]['full_vocab_logits'] = [rows[0]['full_vocab_logits']]
    elif damage == 'length': rows[0]['full_vocab_logits'].pop()
    elif damage == 'vocab':
        for row in rows: row['vocab_size'] = 4; row['full_vocab_logits'] = [0.] * 4
    elif damage == 'declared_logz': rows[0]['full_vocab_logsumexp'] = 0.
    else: rows[0]['full_vocab_logits'][0] = {'nan': float('nan'), 'inf': float('inf'), 'bool': True}[damage]
    result = kernel.score_trie(trie, rows)
    assert result['status'] == 'unknown' and result['reason']


def test_duplicate_or_changed_edge_cannot_be_accepted_as_a_new_trie():
    trie = kernel.build_trie(metadata())
    trie['nodes'][trie['root_node_id']]['children'].append(deepcopy(trie['nodes'][trie['root_node_id']]['children'][0]))
    assert kernel.score_trie(trie, observations(trie))['reason'] == 'trie_structure_or_identity_changed'


def test_extremely_negative_common_shift_uses_retained_stable_decomposition():
    trie = kernel.build_trie(metadata(('A', 'B')))
    result = kernel.score_trie(trie, observations(trie, -1e308))
    assert result['status'] == 'ok'
    for node in result['nodes'].values():
        assert node['full_vocab_logsumexp'] == -1e308
        assert node['logits_max'] == -1e308 and node['shifted_logsumexp'] == pytest.approx(math.log(128))
        assert all(edge['log_probability'] == pytest.approx(-math.log(128)) for edge in node['edges'])
    assert result['candidate_pool_mass'] == pytest.approx(2 / 128.**2)


def test_probability_underflow_preserves_log_mass_and_conditional_mass():
    trie = kernel.build_trie(metadata(('A', 'B')))
    rows = observations(trie)
    for row in rows:
        for edge in trie['nodes'][row['node_id']]['children']:
            row['full_vocab_logits'][edge['token_id']] = -1000.
    result = kernel.score_trie(trie, rows)
    assert result['status'] == 'ok' and result['candidate_pool_mass'] == 0.
    assert result['pool_mass_underflow'] is True and math.isfinite(result['log_candidate_pool_mass'])
    assert all(p['joint_probability_underflow'] for p in result['paths'].values())
    assert [p['conditional_mass_given_candidate_pool'] for p in result['paths'].values()] == [.5, .5]


def test_strict_float_boundary_is_unknown_without_epsilon_or_clamping():
    trie = kernel.build_trie(metadata(('A', 'B')))
    rows = observations(trie, -1000.)
    root = next(row for row in rows if row['node_id'] == trie['root_node_id'])
    root['full_vocab_logits'][ord('A')] = 0.
    root['full_vocab_logits'][ord('B')] = -.0192
    result = kernel.score_trie(trie, rows)
    assert result['status'] == 'unknown' and result['reason'] == 'node_child_mass_exceeds_one'


def test_nonfinite_shift_arithmetic_is_not_replaced_by_probability_zero():
    trie = kernel.build_trie(metadata(('A',)))
    rows = observations(trie)
    rows[0]['full_vocab_logits'][0] = -1e308
    rows[0]['full_vocab_logits'][1] = 1e308
    result = kernel.score_trie(trie, rows)
    assert result['status'] == 'unknown' and result['reason'] == 'nonfinite_shifted_logits'


def test_kernel_calls_do_not_open_files_or_import_model_runtimes(monkeypatch):
    value = metadata(('A', 'B'))
    def forbidden(*args, **kwargs): raise AssertionError('Pure tree attempted file IO')
    monkeypatch.setattr(builtins, 'open', forbidden)
    trie = kernel.build_trie(value)
    result = kernel.score_trie(trie, observations(trie))
    assert result['status'] == 'ok'
    assert not set(kernel.__dict__) & {'torch', 'transformers', 'HFGenerator', 'AutoTokenizer'}
