"""A pure, observer-defined probability tree of complete candidate+EOS codes.

One full vocabulary row belongs to each unique causal context. This contract
does not establish LM replay, factual correctness, or backend qualification.
"""
from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math

CONTRACT_VERSION = 'shared-context-completion-trie-v1'
COMPLETION_CONTRACT = 'full-candidate-plus-eos-v1'
WHITESPACE_POLICY = 'reject_leading_or_trailing_whitespace'
META_FIELDS = frozenset(('completion_contract_version', 'raw_candidate', 'whitespace_policy',
    'prefix_ids', 'prefix_tokens', 'candidate_token_ids', 'candidate_tokens', 'eos_token_id',
    'input_ids', 'input_tokens', 'completion_eos_appended_to_input', 'target_token_ids',
    'target_tokens', 'target_prediction_positions', 'logits_to_keep', 'prefix_alignment_verified',
    'candidate_roundtrip_verified', 'causal_shift_verified'))


class _Invalid(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise _Invalid(reason)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode('utf-8')).hexdigest()


def _node_key(context):
    return _digest(dict(contract=CONTRACT_VERSION, context_ids=context))


def _unknown(reason, **details):
    return dict(schema=CONTRACT_VERSION, status='unknown', reason=reason,
                model_calls=0, language_model_exact_replay_established=False, **details)


def _ids(value):
    return isinstance(value, list) and bool(value) and all(type(i) is int and i >= 0 for i in value)


def _metadata(candidate, value):
    _require(isinstance(candidate, str) and candidate and candidate == candidate.strip(), 'invalid_candidate_literal')
    _require(isinstance(value, Mapping) and set(value) == META_FIELDS, 'invalid_completion_metadata_fields')
    _require(value['raw_candidate'] == candidate and value['completion_contract_version'] == COMPLETION_CONTRACT
        and value['whitespace_policy'] == WHITESPACE_POLICY
        and value['completion_eos_appended_to_input'] is False
        and all(value[k] is True for k in ('prefix_alignment_verified', 'candidate_roundtrip_verified', 'causal_shift_verified')),
        'completion_metadata_contract_mismatch')
    _require(all(_ids(value[k]) for k in ('prefix_ids', 'candidate_token_ids', 'input_ids',
                                        'target_token_ids', 'target_prediction_positions')), 'invalid_completion_token_ids')
    prefix, tokens, eos = value['prefix_ids'], value['candidate_token_ids'], value['eos_token_id']
    _require(type(eos) is int and eos >= 0 and eos not in tokens, 'invalid_or_internal_eos')
    _require(value['input_ids'] == prefix + tokens and value['target_token_ids'] == tokens + [eos]
        and value['target_prediction_positions'] == list(range(len(prefix) - 1, len(prefix) + len(tokens))),
        'completion_causal_alignment_mismatch')
    counts = dict(prefix_tokens=len(prefix), candidate_tokens=len(tokens), input_tokens=len(prefix) + len(tokens),
                  target_tokens=len(tokens) + 1, logits_to_keep=len(tokens) + 1)
    _require(all(type(value[k]) is int and value[k] == count for k, count in counts.items()),
             'completion_token_count_mismatch')


def _build(candidate_metadata):
    _require(isinstance(candidate_metadata, Mapping) and candidate_metadata, 'empty_or_invalid_candidate_metadata')
    _require(all(isinstance(c, str) for c in candidate_metadata), 'invalid_candidate_literal')
    candidates = sorted(candidate_metadata)
    for candidate in candidates:
        _metadata(candidate, candidate_metadata[candidate])
    first = candidate_metadata[candidates[0]]
    prefix, eos = list(first['prefix_ids']), first['eos_token_id']
    _require(all(value['prefix_ids'] == prefix and value['eos_token_id'] == eos for value in candidate_metadata.values()),
             'inconsistent_prefix_or_eos')
    codes = [tuple(candidate_metadata[c]['target_token_ids']) for c in candidates]
    _require(len(set(codes)) == len(codes), 'tokenized_candidate_alias')
    nodes, paths = {}, {}
    for candidate in candidates:
        tokens = candidate_metadata[candidate]['candidate_token_ids']
        steps = []
        for i, token in enumerate([*tokens, eos]):
            context = prefix + tokens[:i]
            key = _node_key(context)
            node = nodes.setdefault(key, dict(node_id=key, context_ids=context, candidate_depth=i, children={}))
            _require(node['context_ids'] == context, 'context_identity_collision')
            edge = dict(token_id=token, next_node_id=_node_key(context + [token]) if i < len(tokens) else None,
                        terminal_candidate=candidate if i == len(tokens) else None)
            _require(token not in node['children'] or node['children'][token] == edge, 'duplicate_or_aliased_edge')
            node['children'][token] = edge
            steps.append(dict(node_id=key, token_id=token))
        paths[candidate] = dict(candidate=candidate, candidate_token_ids=list(tokens), target_token_ids=[*tokens, eos], steps=steps)
    nodes = {key: dict(value, children=[value['children'][token] for token in sorted(value['children'])])
             for key, value in sorted(nodes.items())}
    return dict(schema=CONTRACT_VERSION, status='ok', reason=None, prefix_ids=prefix, eos_token_id=eos,
        candidate_order=candidates, completion_metadata={c: deepcopy(dict(candidate_metadata[c])) for c in candidates},
        root_node_id=_node_key(prefix), nodes=nodes, paths=paths,
        counts=dict(candidates=len(candidates), unique_nodes=len(nodes),
            unique_edges=sum(len(n['children']) for n in nodes.values()),
            logical_target_positions=sum(len(path['steps']) for path in paths.values())),
        one_distribution_per_context=True, language_model_exact_replay_established=False)


def build_trie(candidate_metadata):
    """Validate existing v1 framing; deduplicate full context IDs, never candidates."""
    try:
        return _build(candidate_metadata)
    except (_Invalid, TypeError, ValueError, OverflowError) as error:
        return _unknown(str(error) if isinstance(error, _Invalid) else 'malformed_completion_metadata')


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _lse(values):
    peak = max(values)
    total = math.fsum(math.exp(value - peak) for value in values)
    result = peak + math.log(total)
    _require(math.isfinite(result), 'nonfinite_logsumexp')
    return result, peak, total


def _score(trie, observations):
    _require(isinstance(trie, dict) and trie.get('status') == 'ok', 'invalid_or_unknown_trie')
    expected = _build(trie.get('completion_metadata'))
    _require(_canonical(trie) == _canonical(expected), 'trie_structure_or_identity_changed')
    _require(isinstance(observations, list), 'node_observations_must_be_a_list')
    by_node, contexts, vocabulary = {}, set(), None
    for row in observations:
        _require(isinstance(row, dict) and set(row) == {'node_id', 'context_ids', 'vocab_size', 'full_vocab_logits'},
                 'invalid_node_observation_fields')
        key = row['node_id']
        _require(isinstance(key, str) and key in trie['nodes'] and key not in by_node, 'extra_or_duplicate_node_observation')
        _require(_ids(row['context_ids']) and row['context_ids'] == trie['nodes'][key]['context_ids'],
                 'node_context_mismatch')
        context = tuple(row['context_ids'])
        _require(context not in contexts, 'aliased_node_context')
        contexts.add(context)
        vocab, vector = row['vocab_size'], row['full_vocab_logits']
        _require(type(vocab) is int and vocab > 0 and isinstance(vector, list) and len(vector) == vocab
            and all(_finite(value) for value in vector), 'invalid_full_vocabulary_vector')
        _require(vocabulary is None or vocabulary == vocab, 'inconsistent_vocabulary_size')
        vocabulary = vocab
        _require(max(row['context_ids'] + [e['token_id'] for e in trie['nodes'][key]['children']]) < vocab,
                 'token_outside_complete_vocabulary')
        by_node[key] = row
    _require(set(by_node) == set(trie['nodes']), 'missing_node_observation')
    node_results = {}
    for key, node in trie['nodes'].items():
        row = by_node[key]
        vector = [float(value) for value in row['full_vocab_logits']]
        peak = max(vector)
        shifted = [value - peak for value in vector]
        _require(all(math.isfinite(value) for value in shifted), 'nonfinite_shifted_logits')
        total = math.fsum(math.exp(value) for value in shifted)
        shifted_z, full_z = math.log(total), peak + math.log(total)
        _require(math.isfinite(full_z), 'nonfinite_full_vocab_normalizer')
        edges = []
        for edge in node['children']:
            token = edge['token_id']; logp = shifted[token] - shifted_z
            _require(math.isfinite(logp) and logp <= 0, 'invalid_edge_log_probability')
            edges.append(dict(edge, raw_logit=vector[token], log_probability=logp,
                              probability=math.exp(logp), probability_underflow=math.exp(logp) == 0))
        child_log_mass, _, _ = _lse([edge['log_probability'] for edge in edges])
        child_mass = math.fsum(edge['probability'] for edge in edges)
        _require(child_log_mass <= 0 and child_mass <= 1, 'node_child_mass_exceeds_one')
        node_results[key] = dict(node_id=key, context_ids=list(node['context_ids']), vocab_size=vocabulary,
            normalizer_dtype='float64', logits_max=peak, shifted_logsumexp=shifted_z,
            full_vocab_logsumexp=full_z, selected_child_log_mass=child_log_mass,
            selected_child_mass=child_mass, edges=edges, full_vector_retained=False)
    paths = {}
    for candidate in trie['candidate_order']:
        path = trie['paths'][candidate]
        logp = [next(edge['log_probability'] for edge in node_results[step['node_id']]['edges']
                     if edge['token_id'] == step['token_id']) for step in path['steps']]
        joint = math.fsum(logp)
        _require(math.isfinite(joint) and joint <= 0, 'nonfinite_candidate_joint')
        probability = math.exp(joint)
        paths[candidate] = dict(path, edge_log_probabilities=logp, raw_joint_log_probability=joint,
            joint_probability=probability, joint_probability_underflow=probability == 0)
    values = [paths[c]['raw_joint_log_probability'] for c in trie['candidate_order']]
    log_mass, peak, scaled_total = _lse(values)
    pool_mass = math.fsum(path['joint_probability'] for path in paths.values())
    _require(log_mass <= 0 and pool_mass <= 1 and math.exp(log_mass) <= 1, 'candidate_pool_mass_exceeds_one')
    for path in paths.values():
        path['conditional_mass_given_candidate_pool'] = math.exp(path['raw_joint_log_probability'] - peak) / scaled_total
    return dict(schema=CONTRACT_VERSION, status='ok', reason=None, trie=deepcopy(trie), nodes=node_results, paths=paths,
        log_candidate_pool_mass=log_mass, candidate_pool_mass=pool_mass, pool_mass_underflow=pool_mass == 0,
        counts=dict(trie['counts'], node_observations=len(observations)), strict_probability_gate=True,
        probability_clamping_performed=False, epsilon_admission_performed=False,
        conditional_mass_is_full_vocabulary_probability=False, calibrated_truth_probability=False,
        factual_correctness='unknown_not_assessed', language_model_exact_replay_established=False, model_calls=0)


def score_trie(trie, node_observations):
    """Consume one full row/node; retain edges and LSE decomposition, not vectors."""
    try:
        return _score(trie, node_observations)
    except (_Invalid, TypeError, ValueError, OverflowError, KeyError) as error:
        return _unknown(str(error) if isinstance(error, _Invalid) else 'malformed_or_nonfinite_trie_observations')
