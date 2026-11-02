"""Synthetic finite-label collection, no truth access and no model loading."""

from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments import data, semantic_qualification as legacy, verdict_qualification as module


IDENTITY = {'model_id': 'fake-finite-label-instrument',
            'model_files_sha256': {'weights.bin': 'a' * 64}}
SOURCE = 'Evidence X.'
RESPONSE = 'A has X.'
PROMPT = 'Question: Is X assigned to A?\nRead ' + SOURCE + '\nAnswer completely.'


class FakeScorer:
    model_identity = IDENTITY

    def __init__(self, count=20):
        self.count, self.calls, self.counts, self.validations = count, [], [], []

    def count_input_tokens(self, messages):
        self.counts.append(deepcopy(messages))
        return self.count(messages) if callable(self.count) else self.count

    def validate_labels(self, messages, labels):
        self.validations.append(deepcopy(messages))
        count = self.count(messages) if callable(self.count) else self.count
        return {'input_tokens': count, 'label_token_ids': dict(zip(labels, [10, 11, 12])),
                'label_decodes': dict(zip(labels, labels)), 'prefix_alignment_verified': True}

    def score_labels(self, messages, labels):
        self.calls.append((deepcopy(messages), list(labels)))
        validation = self.validate_labels(messages, labels)
        return {**validation, 'label_logits': dict(zip(labels, [2.0, 1.0, -1.0])),
                'full_vocab_logsumexp': 4.0, 'output_tokens': 0,
                'actual_forward_passes': 1, 'seconds': 0.25}


@pytest.fixture
def panel(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'research_identity', lambda: {'source': 'fixed'})
    monkeypatch.setattr(module, 'completed_identity', lambda old: old)
    source = tmp_path / 'panel'
    source.mkdir()
    rows = [{'id': 'META_ONLY_' + str(i), 'source_id': 'S' + str(i), 'split': 'test',
             'task': 'QA', 'prompt': PROMPT, 'source': SOURCE, 'response': RESPONSE,
             'generator': 'original-generator'} for i in range(18)]
    inputs = source / 'inputs.jsonl'
    inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    for name in ['evaluation_only.jsonl', 'annotations.jsonl', 'manifest.json']:
        (source / name).write_text('GOLD MUST NOT BE OPENED', encoding='utf-8')
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(module.DEFAULT_PROTOCOL), encoding='utf-8')
    return inputs, protocol, tmp_path / 'outputs' / 'qualification'


def run(panel, scorer=None, identity=IDENTITY):
    return module.qualify(*panel, scorer or FakeScorer(), identity)


def first_case(panel):
    return data.read_json(panel[2] / 'cases' / data.text_hash('META_ONLY_0') / 'case.json')


def test_prespecified_sixteen_control_population_keeps_full_s3_and_gold_isolation(panel, monkeypatch):
    inputs, protocol, output = panel
    inputs.write_text(''.join(inputs.read_text(encoding='utf-8').splitlines(keepends=True)[:16]), encoding='utf-8')
    frozen = dict(module.DEFAULT_PROTOCOL, expected_cases=16, max_forward_passes=96)
    protocol.write_text(json.dumps(frozen), encoding='utf-8')
    original=Path.open
    def guarded(path,*a,**k):
        assert path.name not in {'evaluation_only.jsonl','annotations.jsonl','manifest.json'}
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'open',guarded)
    complete=run(panel)
    assert complete['coverage']['cases']==complete['coverage']['scored_cases']==16
    assert complete['costs']['logical_requests']==96
    assert complete['costs']['actual_forward_passes']==6
    assert complete['evaluation_material_accessed'] is False


@pytest.mark.parametrize('cases,budget',[(16,108),(18,96),(17,102),(True,6)])
def test_population_budget_pairs_are_frozen_before_any_forward(panel,cases,budget):
    frozen=dict(module.DEFAULT_PROTOCOL,expected_cases=cases,max_forward_passes=budget)
    panel[1].write_text(json.dumps(frozen),encoding='utf-8')
    scorer=FakeScorer()
    with pytest.raises(ValueError):run(panel,scorer)
    assert scorer.calls==scorer.validations==[]


def test_full_roster_freezes_before_forward_and_never_opens_gold(panel, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name not in {'evaluation_only.jsonl', 'annotations.jsonl', 'manifest.json'}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    scorer = FakeScorer()
    score = scorer.score_labels
    def frozen_score(messages, labels):
        plan = data.read_json(panel[2] / 'plan.json')
        assert [r['id'] for r in plan['records']] == ['META_ONLY_' + str(i) for i in range(18)]
        assert len(scorer.validations) >= 6
        return score(messages, labels)
    scorer.score_labels = frozen_score
    complete = run(panel, scorer)
    assert complete['state'] == 'observations_sealed_not_evaluated'
    assert complete['evaluation_material_accessed'] is False and complete['auroc_computed'] is False
    assert complete['costs']['actual_forward_passes'] == len(scorer.calls) == 6
    assert complete['costs']['logical_requests'] == 108
    assert complete['costs']['exact_request_reuses'] == 102
    assert complete['costs']['output_tokens'] == 0
    assert complete['coverage']['cases'] == complete['coverage']['scored_cases'] == 18
    for messages, labels in scorer.calls:
        assert labels == ['A', 'B', 'C'] and 'META_ONLY' not in json.dumps(messages)
        payload = json.loads(messages[-1]['content'])
        assert payload == {'source': SOURCE, 'response': RESPONSE,
                           'pre_generation_context': legacy.probe.pre_generation_context(PROMPT, SOURCE)}
    assert len(first_case(panel)['observations']) == 6
    data.verify_seal(panel[2], required={'plan.json', 'complete.json', 'source_identity.json'})


def test_completed_resume_never_counts_or_scores(panel):
    complete = run(panel)
    scorer = FakeScorer()
    scorer.count_input_tokens = lambda _: pytest.fail('Completed cache must not be counted')
    scorer.validate_labels = lambda *_: pytest.fail('Completed cache must not be tokenized')
    assert run(panel, scorer) == complete and scorer.calls == []


def test_input_limit_retains_all_cases_and_every_unknown(panel):
    scorer = FakeScorer(4097)
    complete = run(panel, scorer)
    assert scorer.calls == [] and complete['costs']['actual_forward_passes'] == 0
    assert complete['coverage']['unknown_cases'] == 18
    assert complete['costs']['input_rejections'] == 6
    case = first_case(panel)
    assert case['verdict'] == {'status': 'unknown', 'reason': 'input_token_limit', 'class_scores': None}
    assert all(o['execution']['status'] == 'input_rejected' for o in case['observations'])


def test_late_invalid_token_boundary_stops_before_any_forward(panel):
    scorer = FakeScorer()
    validate = scorer.validate_labels
    def fail_late(messages, labels):
        if len(scorer.validations) == 5:
            raise ValueError('label prefix boundary invalid')
        return validate(messages, labels)
    scorer.validate_labels = fail_late
    with pytest.raises(ValueError, match='prefix boundary'):
        run(panel, scorer)
    assert scorer.calls == []
    assert data.read_json(panel[2] / 'plan.json')['records'][-1]['id'] == 'META_ONLY_17'


@pytest.mark.parametrize('failure', ['missing_validation', 'duplicate_ids', 'missing_ids', 'missing_decode'])
def test_invalid_preflight_metadata_never_reaches_any_forward(panel, failure):
    scorer = FakeScorer()
    validate = scorer.validate_labels
    if failure == 'missing_validation':
        scorer.validate_labels = None
    else:
        def malformed(messages, labels):
            result = validate(messages, labels)
            if failure == 'duplicate_ids':
                result['label_token_ids']['C'] = result['label_token_ids']['B']
            elif failure == 'missing_ids':
                del result['label_token_ids']['C']
            else:
                del result['label_decodes']['C']
            return result
        scorer.validate_labels = malformed
    with pytest.raises(ValueError, match='validate_labels|tokenizer preflight'):
        run(panel, scorer)
    assert scorer.calls == []


@pytest.mark.parametrize('change', ['inputs', 'order', 'protocol', 'model', 'source'])
def test_resume_rejects_all_frozen_identity_changes_before_calls(panel, monkeypatch, change):
    run(panel)
    identity = IDENTITY
    if change in {'inputs', 'order'}:
        lines = panel[0].read_text(encoding='utf-8').splitlines()
        lines = list(reversed(lines)) if change == 'order' else [s.replace(RESPONSE, 'A has Y.') for s in lines]
        panel[0].write_text('\n'.join(lines) + '\n', encoding='utf-8')
    elif change == 'protocol':
        panel[1].write_text(json.dumps({**module.DEFAULT_PROTOCOL, 'notes': 'changed'}), encoding='utf-8')
    elif change == 'model':
        identity = {**IDENTITY, 'model_files_sha256': {'weights.bin': 'b' * 64}}
    else:
        monkeypatch.setattr(module, 'research_identity', lambda: {'source': 'changed'})
    scorer = FakeScorer()
    scorer.model_identity = identity
    with pytest.raises(ValueError, match='Resume plan differs'):
        run(panel, scorer, identity)
    assert scorer.calls == scorer.counts == []


def test_partial_case_reuses_sealed_cache_without_repeating_forward(panel, monkeypatch):
    execute = module._execute
    calls = []
    def interrupt_between_requests(*args):
        if len(calls) == 2:
            raise RuntimeError('interrupted before third request')
        result = execute(*args)
        calls.append(result)
        return result
    monkeypatch.setattr(module, '_execute', interrupt_between_requests)
    initial = FakeScorer()
    with pytest.raises(RuntimeError, match='interrupted'):
        run(panel, initial)
    assert len(initial.calls) == 2
    monkeypatch.setattr(module, '_execute', execute)
    resumed = FakeScorer()
    complete = run(panel, resumed)
    assert len(resumed.calls) == 4 and complete['costs']['actual_forward_passes'] == 6


def test_bad_backend_is_retained_and_unsealed_request_never_reissued(panel):
    scorer = FakeScorer()
    score = scorer.score_labels
    scorer.score_labels = lambda *args: {**score(*args), 'actual_forward_passes': 2}
    with pytest.raises(ValueError, match='forward metadata'):
        run(panel, scorer)
    raw = list((panel[2] / 'requests').glob('*/backend_raw.json'))
    assert len(raw) == 1 and data.read_json(raw[0])['value']['actual_forward_passes'] == 2
    resumed = FakeScorer()
    with pytest.raises(RuntimeError, match='explicit artifact recovery'):
        run(panel, resumed)
    assert resumed.calls == resumed.counts == []


def test_impossible_joint_label_mass_rejected_before_second_forward(panel):
    scorer = FakeScorer()
    score = scorer.score_labels
    scorer.score_labels = lambda *args: {**score(*args), 'label_logits': dict.fromkeys('ABC', 0.0),
                                         'full_vocab_logsumexp': 0.0}
    with pytest.raises(ValueError, match='joint label logsumexp'):
        run(panel, scorer)
    assert len(scorer.calls) == 1
    assert len(list((panel[2] / 'requests').glob('*/backend_raw.json'))) == 1
    assert list((panel[2] / 'requests').glob('*/seal.json')) == []


def test_impossible_sealed_partial_cache_is_checked_before_any_new_forward(panel, monkeypatch):
    execute = module._execute
    def interrupt_after_first(*args):
        if list((panel[2] / 'requests').glob('*/seal.json')):
            raise RuntimeError('interrupt before second request')
        return execute(*args)
    monkeypatch.setattr(module, '_execute', interrupt_after_first)
    with pytest.raises(RuntimeError, match='interrupt'):
        run(panel)
    root = next((panel[2] / 'requests').iterdir())
    response = data.read_json(root / 'response.json')
    response.update(label_logits=dict.fromkeys('ABC', 0.0), full_vocab_logsumexp=0.0)
    (root / 'response.json').write_text(json.dumps(response), encoding='utf-8')
    (root / 'backend_raw.json').write_text(json.dumps({'value': {
        k: v for k, v in response.items() if k != 'status'}}), encoding='utf-8')
    seal = {name: data.file_hash(root / name) for name in
            ['request.json', 'response.json', 'backend_raw.json']}
    (root / 'seal.json').write_text(json.dumps(seal), encoding='utf-8')
    monkeypatch.setattr(module, '_execute', execute)
    scorer = FakeScorer()
    with pytest.raises(ValueError, match='joint label logsumexp'):
        run(panel, scorer)
    assert scorer.calls == scorer.counts == []


def test_sealed_cache_tamper_stops_before_any_forward(panel):
    run(panel)
    response = next((panel[2] / 'requests').glob('*/response.json'))
    response.write_text('{}', encoding='utf-8')
    scorer = FakeScorer()
    with pytest.raises(ValueError, match='Sealed artifact changed'):
        run(panel, scorer)
    assert scorer.calls == scorer.counts == []


def test_original_roster_and_protocol_cannot_be_shortened(panel):
    lines = panel[0].read_text(encoding='utf-8').splitlines()
    panel[0].write_text('\n'.join(lines[:-1]) + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='18-case'):
        run(panel)
    assert not panel[2].exists()


def test_annotation_alias_rejected_before_open(panel, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name != 'evaluation_only.jsonl'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    with pytest.raises(ValueError, match='original inputs.jsonl'):
        module.qualify(panel[0].parent / 'evaluation_only.jsonl', *panel[1:], FakeScorer(), IDENTITY)


class FakeTokenizer:
    prefix = '<original-chat-prefix>\n'
    prefix_ids = [1, 2, 3, 4]
    labels = {'A': [5], 'B': [6], 'C': [7]}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_tensors=None):
        assert add_generation_prompt is True
        if not tokenize:
            return self.prefix
        if return_tensors is not None:
            import torch
            return torch.tensor([self.prefix_ids])
        return list(self.prefix_ids)

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        if text == self.prefix:
            return list(self.prefix_ids)
        if text in self.labels:
            return list(self.labels[text])
        if text.startswith(self.prefix) and text[len(self.prefix):] in self.labels:
            return list(self.prefix_ids) + list(self.labels[text[len(self.prefix):]])
        raise AssertionError('Unexpected tokenizer input')

    def decode(self, ids, **kwargs):
        return next(label for label, values in self.labels.items() if values == ids)


def fake_backend(monkeypatch):
    torch = pytest.importorskip('torch')
    backend = object.__new__(legacy.HFGenerator)
    backend.tokenizer = FakeTokenizer()
    backend.admission = {'fake_cpu': True}
    backend._prepare = lambda: None
    backend._ensure_model = lambda: None
    calls = []
    vocabulary_logits = torch.tensor([0., -1., -2., -3., -4., 3., 2., 1., 5.], dtype=torch.float32)
    def forward(encoded, **kwargs):
        calls.append((encoded.clone(), kwargs, torch.is_grad_enabled()))
        return SimpleNamespace(logits=vocabulary_logits.reshape(1, 1, -1))
    backend.model = forward
    tensor_to = torch.Tensor.to
    def cpu_to(tensor, *args, **kwargs):
        if args == ('cuda:0',) and not kwargs:
            return tensor
        return tensor_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    return backend, calls, vocabulary_logits


def test_hf_label_scoring_is_one_dense_forward_without_generation_or_cache(monkeypatch):
    backend, calls, vocabulary_logits = fake_backend(monkeypatch)
    result = backend.score_labels([{'role': 'user', 'content': 'whole input'}], ['A', 'B', 'C'])
    assert len(calls) == 1
    encoded, options, gradient_enabled = calls[0]
    assert encoded.tolist() == [FakeTokenizer.prefix_ids]
    assert options['use_cache'] is False and options['logits_to_keep'] == 1
    assert options['attention_mask'].tolist() == [[1, 1, 1, 1]]
    assert gradient_enabled is False
    assert result['label_logits'] == {'A': 3., 'B': 2., 'C': 1.}
    assert result['full_vocab_logsumexp'] == pytest.approx(
        math.log(sum(math.exp(x) for x in vocabulary_logits.tolist())), abs=1e-12)
    assert result['label_token_ids'] == {'A': 5, 'B': 6, 'C': 7}
    assert result['label_decodes'] == {'A': 'A', 'B': 'B', 'C': 'C'}
    assert result['input_tokens'] == 4 and result['output_tokens'] == 0
    assert result['actual_forward_passes'] == 1 and result['prefix_alignment_verified'] is True
    assert 'text' not in result and result['seconds'] >= 0


@pytest.mark.parametrize('failure', ['multitoken', 'duplicate', 'boundary', 'prefix', 'labels', 'limit'])
def test_hf_label_engineering_failure_stops_before_model_forward(monkeypatch, failure):
    backend, calls, _ = fake_backend(monkeypatch)
    labels = ['A', 'B', 'C']
    if failure in {'multitoken', 'duplicate'}:
        backend.tokenizer.labels = {**FakeTokenizer.labels, 'C': [7, 8] if failure == 'multitoken' else [6]}
    elif failure in {'boundary', 'prefix'}:
        encode = backend.tokenizer.encode
        def changed(text, **kwargs):
            if (failure == 'boundary' and text == backend.tokenizer.prefix + 'C'
                    or failure == 'prefix' and text == backend.tokenizer.prefix):
                return [99]
            return encode(text, **kwargs)
        backend.tokenizer.encode = changed
    elif failure == 'labels':
        labels = ['D', 'E', 'F']
    else:
        backend.tokenizer.prefix_ids = [1] * 4097
    backend._ensure_model = lambda: pytest.fail('Invalid tokenizer/input must fail before model load')
    with pytest.raises((ValueError, RuntimeError)):
        backend.score_labels([], labels)
    assert calls == []
