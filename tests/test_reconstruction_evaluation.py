"""Independent synthetic source-role archive/release tests; no production data."""
from copy import deepcopy
import importlib
from itertools import product
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from experiments import answer_completion, data, reconstruction_evaluation as module
from experiments import reconstruction_qualification as qualifier
from experiments.datasets import Sample
from experiments.research import source_reconstruction as kernel


MODEL = dict(model_id='synthetic-reader-completion', model_files_sha256={'weights': 'a' * 64})


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def seal(root):
    write(root / 'seal.json', {str(p.relative_to(root)): data.file_hash(p)
        for p in root.rglob('*') if p.is_file() and p != root / 'seal.json'})


class Tokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert add_generation_prompt is True
        text = json.dumps(messages, ensure_ascii=False, sort_keys=True) + '\nASSISTANT\n'
        return self.encode(text, add_special_tokens=False) if tokenize else text

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [ord(c) + 32 for c in text]

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        return ''.join(chr(i - 32) for i in ids)


def synthetic48():
    """Separate test facts/labels; never imports or reads the H007 generator."""
    inputs, design, truth, answers = [], [], [], {}
    for family, vocabulary, sd, rd, burden in product(
            ('winner', 'sole_signer'), ('lex1', 'lex2'), (0, 1), (0, 1), (0, 4, 12)):
        people = ('Ava Lorn', 'Bex Pell') if vocabulary == 'lex1' else ('Cara Wynn', 'Dax Ford')
        events = ('Amber Cup', 'Cobalt Cup') if family == 'winner' else ('Amber Accord', 'Cobalt Accord')
        source_people = people if sd == 0 else people[::-1]
        facts = []
        for event, person in zip(events, source_people, strict=True):
            if family == 'winner':
                facts.extend([f'The {event} final was a match between {people[0]} and {people[1]}.',
                              f'{person} was the sole winner of the {event} final.'])
            else:
                facts.append(f'{person} was the sole signer of the {event}.')
        source = ' '.join(facts + list(kernel.FILLERS))
        for event, person in zip(events, source_people, strict=True):
            answers[source, event] = person
        a, b = people if rd == 0 else people[::-1]
        x, y = events
        target = (f'{a} won the {x} final, and {b} won the {y} final. ' if family == 'winner' else
                  f'{a} was the sole signer of the {x}, and {b} was the sole signer of the {y}. ')
        response = target + ' '.join(kernel.FILLERS[:burden])
        sid = f'{family}-{vocabulary}-s{sd}'
        row = dict(id=f'{sid}-r{rd}-b{burden}', source_id=sid, task='QA', split='test',
                   source=source, prompt='Test source: ' + source, response=response, generator='synthetic')
        inputs.append(row)
        span = [0, len(target)]
        design.append(dict(id=row['id'], source_id=sid, target_span=span, target_text=target,
            whole_span=[0, len(response)],
            input_text_sha256={k: data.text_hash(row[k]) for k in ('prompt', 'source', 'response')},
            factor_cell=dict(relation_family=family, vocabulary=vocabulary, source_direction=sd,
                             response_direction=rd, distractor_count=burden)))
        outcome = 'supported' if sd == rd else 'contradicted'
        truth.append(dict(id=row['id'], source_id=sid, target_span=span,
            source_sha256=data.text_hash(source), response_sha256=data.text_hash(response),
            target_outcome=outcome, whole_outcome=outcome, filler_outcomes=['supported'] * burden,
            target_benchmark_labels_used=False,
            construction_rule='entity_event_bijections_equal_iff_source_direction_equals_response_direction'))
    return inputs, design, truth, answers


class Backend:
    model_identity = MODEL

    def __init__(self, answers, preference='correct'):
        self.answers, self.preference, self.calls = answers, preference, 0

    def validate_completion(self, messages, candidate):
        return answer_completion.completion_metadata(Tokenizer(), messages, candidate)

    def score_completion(self, messages, candidate):
        self.calls += 1
        metadata = self.validate_completion(messages, candidate)
        body = json.loads(messages[-1]['content'])
        query = next(q for q in kernel.source_queries(body['source'])['queries']
                     if q['question'] == body['question'])
        reference = self.answers[body['source'], query['event']]
        preferred = reference if self.preference == 'correct' else query['candidates'][0]
        joint = -1. if candidate == preferred else -5.
        if self.preference == 'tie': joint = -2.
        count = metadata['target_tokens']
        # Exact binary-representable sums keep synthetic ties independent of name length.
        logp = [joint] + [0.] * (count - 1)
        normalizers = [10.] * count
        selected = [10. + value for value in logp]
        logp = [value - 10. for value in selected]
        return dict(metadata, target_token_logits=selected, full_vocab_logsumexp=normalizers,
            target_token_log_probabilities=logp, joint_log_probability=math.fsum(logp),
            full_vocab_normalizer_dtype='float64', vocab_size=65536, logits_shape=[1, count, 65536],
            actual_forward_passes=1, output_tokens=0, seconds=.125,
            resource_admission=dict(device='cuda:0', available_ram_bytes=3 * 1024**3,
                free_cuda_bytes=4 * 1024**3, cuda_allocator_limit_bytes=3 * 1024**3),
            peak_cuda_allocated_bytes=1024, peak_cuda_reserved_bytes=2048)


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'write_json', write)
    source = dict(source_sha256={name: data.file_hash(Path(importlib.import_module(
        name[:-3].replace('/', '.')).__file__)) for name in module._HELPERS})
    for target in (module, qualifier):
        monkeypatch.setattr(target, 'research_identity', lambda: deepcopy(source))
        monkeypatch.setattr(target, 'completed_identity', lambda start: start)
    tokenizer_calls = []

    def tokenizer(model, identity):
        tokenizer_calls.append((model, identity))
        assert identity == MODEL
        return Tokenizer(), lambda: None
    monkeypatch.setattr(module, '_cpu_tokenizer', tokenizer)

    def build(preference='correct'):
        rows, design_rows, labels, answers = synthetic48()
        inference = tmp_path / 'inference'; inference.mkdir()
        inputs = inference / 'inputs.jsonl'
        inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        design = inference / 'plan.json'
        write(design, dict(schema='h007-burden-controls-plan-v1', records=design_rows)); seal(inference)
        construction = tmp_path / 'construction'; construction.mkdir()
        truth = construction / 'construction_truth.jsonl'
        truth.write_text(''.join(json.dumps(row) + '\n' for row in labels), encoding='utf-8'); seal(construction)
        protocol = tmp_path / 'observation-protocol.json'
        write(protocol, dict(qualifier.DEFAULT_PROTOCOL, original_inputs_sha256=data.file_hash(inputs)))
        root = tmp_path / 'outputs' / 'observations'
        backend = Backend(answers, preference)
        # Construct a sealed fake observation archive directly: collector gates
        # have their own suite; this suite tests only the independent CPU reader.
        frozen = data.read_json(protocol)
        prepared = qualifier._prepare([Sample(**row) for row in rows])
        protocol_hash = data.file_hash(protocol)
        executions = []
        for row in prepared['candidates']:
            metadata = backend.validate_completion(row['messages'], row['candidate'])
            execution = qualifier._execution(row, metadata, protocol_hash, MODEL, source)
            executions.append(dict(query_key=row['query_key'], candidate=row['candidate'],
                                   key=qualifier._digest(execution), execution=execution))
        root.mkdir(parents=True)
        plan = qualifier._plan(inputs, data.file_hash(inputs), protocol_hash, frozen,
                               MODEL, source, prepared, executions)
        for name, value in [('plan.json', plan), ('protocol.json', frozen),
                            ('model_identity.json', MODEL), ('source_identity.json', source)]:
            write(root / name, value)
        for row in executions:
            child = root / 'requests' / row['key']; child.mkdir(parents=True)
            write(child / 'request.json', row['execution'])
        names = ['plan.json', 'protocol.json', 'model_identity.json', 'source_identity.json'] + [
            str(Path('requests') / row['key'] / 'request.json') for row in executions]
        write(root / 'freeze.json', {name: data.file_hash(root / name) for name in names})
        responses = {}
        for row in executions:
            child = root / 'requests' / row['key']
            raw = backend.score_completion(row['execution']['messages'], row['candidate'])
            response = qualifier._response(raw, row['execution']['preflight'])
            write(child / 'backend_raw.json', {'value': raw}); write(child / 'response.json', response)
            seal(child); responses[row['key']] = response
        write(root / 'complete.json', qualifier._completion(prepared, executions, responses, frozen))
        seal(root)
        assert backend.calls == 32
        model = tmp_path / 'model'; model.mkdir()
        evaluation_protocol = tmp_path / 'evaluation-protocol.json'
        write(evaluation_protocol, module.DEFAULT_PROTOCOL)
        return inputs, design, truth, root, model, tmp_path / 'outputs' / 'review', evaluation_protocol
    build.tokenizer_calls = tokenizer_calls
    return build


def guard_references(args, monkeypatch, allow=False):
    original = Path.open
    accesses = []
    design, construction, output = args[1], args[2].parent, args[5]

    def guarded(path, *rest, **kwargs):
        if path == design or path == design.parent / 'seal.json' or path.is_relative_to(construction):
            accesses.append(path)
            assert allow, 'Reference material opened before validation/freeze'
            assert (output / 'freeze.json').is_file()
            assert data.read_json(output / 'plan.json')['state'] == 'frozen_before_reference_release'
        return original(path, *rest, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    return accesses


def test_complete32_rebuild_then_reference_release_and_all_flip_directions(make_run, monkeypatch):
    args = make_run()
    before = {str(p.relative_to(args[3])): p.read_bytes() for p in args[3].rglob('*') if p.is_file()}
    accesses = guard_references(args, monkeypatch, allow=True)
    summary = module.evaluate(*args)
    assert accesses and len(make_run.tokenizer_calls) == 1
    assert summary['source_qa'] == dict(unique_sources=8, unique_queries=16, correct=16, ties=0, wrong=0)
    for name in ('sourceflip', 'eventflip'):
        assert summary[name]['pairs'] == summary[name]['correctly_reversed_pairs'] == 8
        assert all(r['oriented_raw_interaction'] == 4. for r in summary[name]['records'])
    assert summary['costs']['actual_forward_passes'] == 32 and summary['costs']['seconds_forward'] == 4.
    assert summary['model_calls'] == 0 and summary['completion_metadata_rebuilt'] == 32
    assert summary['secondary_case_metrics']['auroc'] == 1.
    assert summary['method_promoted'] is False and summary['candidate_length_bias_removed'] is False
    assert before == {str(p.relative_to(args[3])): p.read_bytes() for p in args[3].rglob('*') if p.is_file()}
    candidates = data.read_json(args[5] / 'candidate-observations.json')
    assert len(candidates) == 32
    assert all(len(row['response']['target_token_log_probabilities']) ==
               row['response']['candidate_tokens'] + 1 for row in candidates)
    assert len(data.read_json(args[5] / 'cases.json')) == 48
    data.verify_seal(args[5])


@pytest.mark.parametrize('damage', ['root_extra', 'child_extra', 'seal_escape', 'child_seal_escape'])
def test_entire_inventory_is_rejected_before_any_payload_hash_or_source(make_run, monkeypatch, damage):
    args = make_run()
    if damage == 'root_extra': (args[3] / 'extra').write_text('no read', encoding='utf-8')
    elif damage == 'child_extra':
        (next((args[3] / 'requests').iterdir()) / 'extra').write_text('no read', encoding='utf-8')
    else:
        child = next((args[3] / 'requests').iterdir()) if damage == 'child_seal_escape' else args[3]
        path = child / 'seal.json'; value = data.read_json(path); value['../outside'] = 'f' * 64; write(path, value)
        if damage == 'child_seal_escape': seal(args[3])
    def forbidden(*rest, **kwargs): raise AssertionError('Hash/source reached before path admission')
    monkeypatch.setattr(data, 'file_hash', forbidden)
    monkeypatch.setattr(module, 'research_identity', forbidden)
    accesses = guard_references(args, monkeypatch)
    with pytest.raises(ValueError): module.evaluate(*args)
    assert accesses == [] and not args[5].exists()


@pytest.mark.parametrize('damage', ['cost', 'risk', 'helper', 'token_first', 'token_eos', 'raw_joint'])
def test_resealed_observation_tampering_rejected_before_reference_access(make_run, monkeypatch, damage):
    args = make_run()
    if damage in ('cost', 'risk'):
        path = args[3] / 'complete.json'; value = data.read_json(path)
        if damage == 'cost': value['costs']['actual_forward_passes'] += 1
        else: value['records'][0]['case_max_slot_risk'] = .01
        write(path, value)
    elif damage == 'helper':
        path = args[3] / 'source_identity.json'; value = data.read_json(path)
        value['source_sha256']['experiments.research/source_reconstruction.py'] = 'f' * 64; write(path, value)
    else:
        child = next((args[3] / 'requests').iterdir())
        path = child / 'backend_raw.json'; value = data.read_json(path)
        if damage == 'raw_joint': value['value']['joint_log_probability'] += .25
        else: value['value']['target_token_ids'][0 if damage == 'token_first' else -1] += 1
        write(path, value); seal(child)
    seal(args[3]); accesses = guard_references(args, monkeypatch)
    with pytest.raises(ValueError): module.evaluate(*args)
    assert accesses == [] and not args[5].exists()


def test_protocol_alias_into_truth_tree_rejected_without_any_read(make_run, monkeypatch):
    args = list(make_run()); args[6] = args[2].parent / 'protocol.json'
    accesses = guard_references(args, monkeypatch)
    def forbidden(*rest, **kwargs): raise AssertionError('Unsafe protocol reached content')
    monkeypatch.setattr(data, 'file_hash', forbidden)
    with pytest.raises(ValueError, match='Independent'): module.evaluate(*args)
    assert accesses == []


@pytest.mark.parametrize('preference,correct,ties', [('first', 8, 0), ('tie', 0, 16)])
def test_name_shortcut_or_ties_are_not_correct_role_reconstruction(make_run, preference, correct, ties):
    args = make_run(preference)
    summary = module.evaluate(*args)
    assert summary['source_qa']['correct'] == correct and summary['source_qa']['ties'] == ties
    assert summary['sourceflip']['correctly_reversed_pairs'] == 0
    assert summary['eventflip']['correctly_reversed_pairs'] == 0
    if preference == 'tie':
        assert all(q['greedy_candidate'] is None for q in data.read_json(args[5] / 'queries.json'))


def test_positive_raw_interaction_alone_does_not_mean_correct_pair():
    first = dict(candidates=['Ava Lorn', 'Bex Pell'], reference_candidate='Ava Lorn',
                 raw_canonical_log_odds=8., query_key='a', greedy_correct=True, tie=False)
    second = dict(first, reference_candidate='Bex Pell', raw_canonical_log_odds=2., query_key='b', greedy_correct=False)
    record = module._flip_pair(first, second, 'sourceflip', {})
    assert record['oriented_raw_interaction'] == 3. and record['positive_source_or_event_response']
    assert not record['correctly_reversed'] and not record['both_queries_correct']


@pytest.mark.parametrize('mutation', ['seal', 'extra_member'])
def test_final_observation_mutation_revokes_cpu_completion(make_run, monkeypatch, mutation):
    args = make_run(); original = module._seal_cpu
    def sealing(root, names, hashes):
        original(root, names, hashes)
        if root == args[5]:
            path = args[3] / ('seal.json' if mutation == 'seal' else 'unexpected.json')
            path.write_text((path.read_text(encoding='utf-8') if path.exists() else '{}') + '\n', encoding='utf-8')
    monkeypatch.setattr(module, '_seal_cpu', sealing)
    with pytest.raises(ValueError): module.evaluate(*args)
    assert (args[5] / 'freeze.json').is_file() and (args[5] / 'queries.json').is_file()
    assert not (args[5] / 'complete.json').exists() and not (args[5] / 'seal.json').exists()


@pytest.mark.parametrize('mutation', ['replace_result', 'extra_member'])
def test_final_cpu_output_mutation_is_not_blessed_by_sealing(make_run, monkeypatch, mutation):
    args = make_run(); original = module._seal_cpu
    extra = args[5] / 'unexpected.json'; hashes = []; original_hash = data.file_hash
    def hashing(path):
        if Path(path) == extra: hashes.append(path); raise AssertionError('Unsafe extra was hashed')
        return original_hash(path)
    monkeypatch.setattr(data, 'file_hash', hashing)
    def sealing(root, names, expected):
        write(root / ('summary.json' if mutation == 'replace_result' else 'unexpected.json'), {'changed': True})
        original(root, names, expected)
    monkeypatch.setattr(module, '_seal_cpu', sealing)
    with pytest.raises(ValueError): module.evaluate(*args)
    assert hashes == [] and not (args[5] / 'complete.json').exists() and not (args[5] / 'seal.json').exists()


def test_changed_completion_readback_revokes_marker_before_seal(make_run, monkeypatch):
    args = make_run()
    def changed(path, value):
        write(path, {'changed': True} if Path(path) == args[5] / 'complete.json' else value)
    monkeypatch.setattr(data, 'write_json', changed)
    with pytest.raises(ValueError, match='Written CPU result differs'): module.evaluate(*args)
    assert (args[5] / 'freeze.json').is_file()
    assert not (args[5] / 'complete.json').exists() and not (args[5] / 'seal.json').exists()


@pytest.mark.parametrize('mutation', ['join', 'factor', 'filler', 'outcome'])
def test_reference_identity_factor_and_generator_contract_are_verified(make_run, mutation):
    args = make_run()
    if mutation == 'factor':
        value = data.read_json(args[1]); value['records'][0]['factor_cell']['source_direction'] = 1
        write(args[1], value); seal(args[1].parent)
    else:
        rows = [json.loads(line) for line in args[2].read_text(encoding='utf-8').splitlines()]
        if mutation == 'join': rows[0]['source_sha256'] = 'f' * 64
        elif mutation == 'filler': rows[0]['filler_outcomes'] = ['contradicted']
        else: rows[0].update(target_outcome='contradicted', whole_outcome='contradicted')
        args[2].write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8'); seal(args[2].parent)
    with pytest.raises(ValueError): module.evaluate(*args)
    assert (args[5] / 'freeze.json').is_file() and not (args[5] / 'complete.json').exists()


def test_cpu_runtime_hashes_weights_once_and_never_prepares_or_loads_model(tmp_path, monkeypatch):
    from experiments import semantic_qualification as semantic
    model = tmp_path / 'model'; model.mkdir()
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        write(model / name, {})
    write(model / 'model.safetensors.index.json', dict(weight_map={'a': 'model.safetensors'}))
    (model / 'model.safetensors').write_bytes(b'fake-weights-never-loaded')
    identity = dict(model_id='Qwen/Qwen2.5-3B-Instruct', runtime=deepcopy(semantic.HF_RUNTIME),
        model_files_sha256={p.name: data.file_hash(p) for p in model.iterdir()})
    original_hash = data.file_hash; weight_hashes = []
    def hashing(path):
        if Path(path).suffix == '.safetensors': weight_hashes.append(Path(path))
        return original_hash(path)
    monkeypatch.setattr(data, 'file_hash', hashing)
    def forbidden(*args, **kwargs): raise AssertionError('CPU reader entered a model/GPU runtime')
    monkeypatch.setattr(semantic.HFGenerator, '_prepare', forbidden)
    monkeypatch.setattr(semantic.HFGenerator, '_ensure_model', forbidden)
    monkeypatch.setattr(module, 'version', lambda package: identity['runtime'][package])
    calls = []
    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs)); return SimpleNamespace(is_fast=True)
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=from_pretrained)))
    tokenizer, guard = module._cpu_tokenizer(model, identity)
    guard(); guard()
    assert tokenizer.is_fast and weight_hashes == [model / 'model.safetensors']
    assert calls == [(model, dict(local_files_only=True, trust_remote_code=False, use_fast=True))]
    (model / 'model.safetensors').write_bytes(b'changed')
    with pytest.raises(ValueError, match='assets changed'): guard()
    assert len(weight_hashes) == 1


def test_cpu_freeze_rejects_changed_metadata_before_reference_release(tmp_path, monkeypatch):
    output = tmp_path / 'outputs' / 'review'
    def changed(path, value):
        write(path, {'changed': True} if Path(path).name == 'plan.json' else value)
    monkeypatch.setattr(data, 'write_json', changed)
    with pytest.raises(ValueError, match='metadata differs'):
        module._freeze_cpu(output, {'state': 'expected'}, {}, {})
    assert not (output / 'freeze.json').exists()


def test_cpu_freeze_check_admits_inventory_before_any_rehash(tmp_path, monkeypatch):
    output = tmp_path / 'outputs' / 'review'
    monkeypatch.setattr(data, 'write_json', write)
    frozen = module._freeze_cpu(output, {'state': 'expected'}, {}, {})
    (output / 'extra.json').write_text('no hash', encoding='utf-8')
    def forbidden(*args, **kwargs): raise AssertionError('Unexpected CPU member reached hashing')
    monkeypatch.setattr(data, 'file_hash', forbidden)
    with pytest.raises(ValueError, match='CPU reconstruction review members'):
        module._check_cpu_freeze(output, frozen)
