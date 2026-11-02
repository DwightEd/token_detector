"""Synthetic independent CPU release, identity-chain and complete-burden checks."""
from copy import deepcopy
import importlib
import json
import math
from pathlib import Path

import pytest

from experiments import answer_completion, data
from experiments import masked_reconstruction_evaluation as module
from experiments import masked_reconstruction_qualification as observer
from experiments.datasets import Sample
from tests import test_reconstruction_evaluation as old
from tests.test_reconstruction_evaluation import make_run


@pytest.fixture
def make_bridge(make_run, tmp_path, monkeypatch):
    old_model = dict(model_id='synthetic-reader-completion',
        model_files_sha256={f'asset-{i}': str(i) * 64 for i in range(9)},
        execution=dict(installed_observer_wheel_sha256='a' * 64, CPU_module_target='old-target'),
        readout=dict(prefix=module.DEFAULT_PROTOCOL['h008_readout_prefix']), runtime=dict(fake='unchanged'))
    new_model = deepcopy(old_model)
    new_model['execution'] = dict(installed_observer_wheel_sha256='b' * 64, CPU_module_target='new-target')
    new_model['readout']['prefix'] = module.DEFAULT_PROTOCOL['h011_readout_prefix']
    monkeypatch.setattr(old, 'MODEL', old_model)
    source = dict(source_sha256={name: data.file_hash(Path(importlib.import_module(
        name[:-3].replace('/', '.')).__file__)) for name in module._HELPERS})
    for target in (module, observer.collector):
        monkeypatch.setattr(target, 'research_identity', lambda: deepcopy(source))
        monkeypatch.setattr(target, 'completed_identity', lambda value: value)
    def tokenizer(path, identity):
        assert identity in (old_model, new_model)
        return old.Tokenizer(), lambda: None
    monkeypatch.setattr(module.shared, '_cpu_tokenizer', tokenizer)

    def build(mode='correct'):
        base = make_run()
        inputs, design, truth, prior_observation, model, control, prior_protocol = base
        control_protocol = dict(data.read_json(prior_protocol),
            notes='Legitimate frozen old protocol annotation', hypothesis_id='SYNTHETIC_H008')
        old.write(prior_protocol, control_protocol)
        module.shared.evaluate(*base)
        rows, _, _, answers = old.synthetic48()
        prepared = observer._prepare([Sample(**row) for row in rows])
        protocol = tmp_path / 'masked-observation-protocol.json'
        old.write(protocol, dict(observer.DEFAULT_PROTOCOL, original_inputs_sha256=data.file_hash(inputs)))
        frozen = data.read_json(protocol)
        protocol_hash = data.file_hash(protocol)
        root = tmp_path / 'outputs' / 'masked-observations'
        root.mkdir(parents=True)
        executions = []
        for row in prepared['candidates']:
            metadata = observer.collector._metadata(
                answer_completion.completion_metadata(old.Tokenizer(), row['messages'], row['candidate']),
                row['candidate'])
            execution = observer.collector._execution(row, metadata, protocol_hash, new_model, source)
            executions.append(dict(query_key=row['query_key'], candidate=row['candidate'],
                key=observer.collector._digest(execution), execution=execution))
        plan = observer._plan(inputs, data.file_hash(inputs), protocol_hash, frozen,
                              new_model, source, prepared, executions)
        for name, value in [('plan.json', plan), ('protocol.json', frozen), ('model_identity.json', new_model),
                             ('source_identity.json', source)]:
            old.write(root / name, value)
        for row in executions:
            child = root / 'requests' / row['key']
            child.mkdir(parents=True)
            old.write(child / 'request.json', row['execution'])
        freeze_names = ['plan.json', 'protocol.json', 'model_identity.json', 'source_identity.json'] + [
            str(Path('requests') / row['key'] / 'request.json') for row in executions]
        old.write(root / 'freeze.json', {name: data.file_hash(root / name) for name in freeze_names})
        responses = {}
        for row in executions:
            query = prepared['queries'][row['query_key']]
            reference = answers[query['source'], query['event']]
            burden = sum(filler in query['masked_response'] for filler in observer.collector.kernel.FILLERS)
            preferred = reference
            if mode == 'reverse12' and burden == 12:
                preferred = next(name for name in query['candidates'] if name != reference)
            joint = -1. if row['candidate'] == preferred else -5.
            if mode == 'tie4' and burden == 4:
                joint = -2.
            metadata = row['execution']['preflight']
            count = metadata['target_tokens']
            logp = [joint] + [0.] * (count - 1)
            raw = dict(metadata, target_token_logits=[10. + value for value in logp],
                full_vocab_logsumexp=[10.] * count, target_token_log_probabilities=logp,
                joint_log_probability=math.fsum(logp), full_vocab_normalizer_dtype='float64',
                vocab_size=65536, logits_shape=[1, count, 65536], actual_forward_passes=1,
                output_tokens=0, seconds=.125,
                resource_admission=dict(device='cuda:0', available_ram_bytes=3 * 1024**3,
                    free_cuda_bytes=4 * 1024**3, cuda_allocator_limit_bytes=3 * 1024**3),
                peak_cuda_allocated_bytes=1024, peak_cuda_reserved_bytes=2048)
            response = observer.collector._response(raw, metadata)
            child = root / 'requests' / row['key']
            old.write(child / 'backend_raw.json', dict(value=raw))
            old.write(child / 'response.json', response)
            old.seal(child)
            responses[row['key']] = response
        old.write(root / 'complete.json', observer._completion(prepared, executions, responses, frozen))
        old.seal(root)
        evaluation_protocol = tmp_path / 'masked-evaluation-protocol.json'
        old.write(evaluation_protocol, dict(module.DEFAULT_PROTOCOL,
            h008_control_cpu_seal_sha256=data.file_hash(control / 'seal.json'),
            h008_control_observation_seal_sha256=data.file_hash(prior_observation / 'seal.json'),
            h008_observer_wheel_sha256='a' * 64, h011_observer_wheel_sha256='b' * 64,
            h008_cpu_module_target='old-target', h011_cpu_module_target='new-target'))
        return inputs, design, truth, root, model, control, tmp_path / 'outputs' / 'masked-review', evaluation_protocol
    return build


def guard_references(args, monkeypatch):
    original = Path.open
    reads = []
    design, construction, control, output = args[1], args[2].parent, args[5], args[6]
    def guarded(path, *rest, **kwargs):
        if path == design or path == design.parent / 'seal.json' or path.is_relative_to(construction) or path.is_relative_to(control):
            reads.append(path)
            assert (output / 'freeze.json').is_file(), 'References touched before independent evaluator freeze'
            assert data.read_json(output / 'plan.json')['state'] == 'frozen_before_reference_release'
        return original(path, *rest, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    return reads


@pytest.mark.parametrize('mode', ['correct', 'reverse12', 'tie4'])
def test_full96_rebuilt_before_reference_release_and_all_burdens_retained(make_bridge, monkeypatch, mode):
    args = make_bridge(mode)
    reads = guard_references(args, monkeypatch)
    result = module.evaluate(*args)
    assert reads and result['completion_metadata_rebuilt'] == 96
    assert result['model_calls'] == 0 and result['all_results_retained'] is True
    assert result['unique_queries'] == 48 and result['response_slots'] == 96
    assert set(result['by_burden']) == {'0', '4', '12'}
    for burden, cohort in result['by_burden'].items():
        assert cohort['queries'] == 16
        expected_correct = 0 if (mode == 'reverse12' and burden == '12') or (mode == 'tie4' and burden == '4') else 16
        assert cohort['correct'] == expected_correct
        assert cohort['ties'] == (16 if mode == 'tie4' and burden == '4' else 0)
        assert cohort['sourceflip']['pairs'] == cohort['eventflip']['pairs'] == 8
        assert cohort['sourceflip']['correctly_reversed_pairs'] == (8 if expected_correct == 16 else 0)
        assert cohort['eventflip']['correctly_reversed_pairs'] == (8 if expected_correct == 16 else 0)
    assert len(data.read_json(args[6] / 'queries.json')) == 48
    assert len(data.read_json(args[6] / 'cases.json')) == 48
    assert len(data.read_json(args[6] / 'candidate-observations.json')) == 96
    data.verify_seal(args[6])


def test_corrupt_observation_rejects_before_any_reference_read(make_bridge, monkeypatch):
    args = make_bridge()
    child = next((args[3] / 'requests').iterdir())
    value = data.read_json(child / 'backend_raw.json')
    value['value']['target_token_log_probabilities'][0] -= .5
    old.write(child / 'backend_raw.json', value)
    old.seal(child)
    old.seal(args[3])
    reads = guard_references(args, monkeypatch)
    with pytest.raises(ValueError):
        module.evaluate(*args)
    assert reads == [] and not args[6].exists()


def test_wrong_control_seal_pin_rejects_after_freeze_without_completing(make_bridge, monkeypatch):
    args = make_bridge()
    value = data.read_json(args[-1])
    value['h008_control_cpu_seal_sha256'] = 'f' * 64
    old.write(args[-1], value)
    guard_references(args, monkeypatch)
    with pytest.raises(ValueError, match='Pinned H008 CPU seal'):
        module.evaluate(*args)
    assert (args[6] / 'freeze.json').is_file()
    assert not (args[6] / 'complete.json').exists() and not (args[6] / 'seal.json').exists()


def test_late_control_resealing_revokes_completed_review(make_bridge, monkeypatch):
    args = make_bridge()
    verify = data.verify_seal
    changed = []
    def mutate(root, *rest, **kwargs):
        result = verify(root, *rest, **kwargs)
        if Path(root) == args[6] and not changed:
            path = args[5] / 'summary.json'
            value = data.read_json(path)
            value['injected_late_change'] = True
            old.write(path, value)
            old.seal(args[5])
            changed.append(path)
        return result
    monkeypatch.setattr(data, 'verify_seal', mutate)
    with pytest.raises(ValueError):
        module.evaluate(*args)
    assert changed
    assert not (args[6] / 'complete.json').exists() and not (args[6] / 'seal.json').exists()


def test_identity_whitelist_requires_exact_pins_and_rejects_changed_numeric_instrument():
    protocol = dict(module.DEFAULT_PROTOCOL, h008_observer_wheel_sha256='a' * 64,
        h011_observer_wheel_sha256='b' * 64, h008_cpu_module_target='old', h011_cpu_module_target='new')
    original = dict(model_files_sha256={str(i): str(i) * 64 for i in range(9)},
        execution=dict(installed_observer_wheel_sha256='a' * 64, CPU_module_target='old'),
        readout=dict(prefix=protocol['h008_readout_prefix']), numeric=dict(dtype='float16'))
    changed = deepcopy(original)
    changed['execution'] = dict(installed_observer_wheel_sha256='b' * 64, CPU_module_target='new')
    changed['readout']['prefix'] = protocol['h011_readout_prefix']
    assert len(module._identity_pair(original, changed, protocol)) == 3
    changed['numeric']['dtype'] = 'bfloat16'
    with pytest.raises(ValueError, match='Unapproved'):
        module._identity_pair(original, changed, protocol)
    changed['numeric']['dtype'] = 'float16'
    changed['execution']['CPU_module_target'] = 'unreviewed-target'
    with pytest.raises(ValueError, match='external protocol pin'):
        module._identity_pair(original, changed, protocol)
