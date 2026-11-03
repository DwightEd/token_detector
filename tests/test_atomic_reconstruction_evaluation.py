"""Fake complete atomic observations through the common independent CPU reader."""
from copy import deepcopy
import importlib
import math
from pathlib import Path
import pytest

from experiments import answer_completion, atomic_reconstruction_evaluation as module
from experiments import atomic_reconstruction_qualification as observer
from experiments import data
from experiments.datasets import Sample
from tests import test_reconstruction_evaluation as old
from tests.test_reconstruction_evaluation import make_run


@pytest.mark.parametrize('tie', [False, True])
def test_atomic_profile_reconstructs32_before_reference_release_and_retains48_originals(make_run, monkeypatch, tmp_path, tie):
    base = make_run()
    inputs, design, truth, _, model_path, output, _ = base
    rows, _, _, answers = old.synthetic48()
    source = dict(source_sha256={name: data.file_hash(Path(importlib.import_module(
        name[:-3].replace('/', '.')).__file__)) for name in module._HELPERS})
    monkeypatch.setattr(module.shared, 'research_identity', lambda: deepcopy(source))
    monkeypatch.setattr(module.shared, 'completed_identity', lambda value: value)
    prepared = observer._prepare([Sample(**row) for row in rows])
    root = tmp_path / 'outputs' / 'atomic'
    root.mkdir(parents=True)
    observation_protocol = tmp_path / 'atomic-observation-protocol.json'
    old.write(observation_protocol, dict(observer.DEFAULT_PROTOCOL, original_inputs_sha256=data.file_hash(inputs)))
    frozen = data.read_json(observation_protocol)
    protocol_hash = data.file_hash(observation_protocol)
    executions = []
    for row in prepared['candidates']:
        metadata = answer_completion.completion_metadata(old.Tokenizer(), row['messages'], row['candidate'])
        execution = observer.collector._execution(row, metadata, protocol_hash, old.MODEL, source)
        executions.append(dict(query_key=row['query_key'], candidate=row['candidate'],
            key=observer.collector._digest(execution), execution=execution))
    plan = observer._plan(inputs, data.file_hash(inputs), protocol_hash, frozen, old.MODEL, source, prepared, executions)
    for name, value in [('plan.json', plan), ('protocol.json', frozen),
                         ('model_identity.json', old.MODEL), ('source_identity.json', source)]:
        old.write(root / name, value)
    for row in executions:
        child = root / 'requests' / row['key']
        child.mkdir(parents=True)
        old.write(child / 'request.json', row['execution'])
    names = ['plan.json', 'protocol.json', 'model_identity.json', 'source_identity.json'] + [
        str(Path('requests') / row['key'] / 'request.json') for row in executions]
    old.write(root / 'freeze.json', {name: data.file_hash(root / name) for name in names})
    responses = {}
    for row in executions:
        query = prepared['queries'][row['query_key']]
        reference = answers[query['source'], query['event']]
        joint = -2. if tie else -1. if row['candidate'] == reference else -5.
        metadata = row['execution']['preflight']
        count = metadata['target_tokens']
        logp = [joint] + [0.] * (count - 1)
        raw = dict(metadata, target_token_logits=[10. + value for value in logp],
            full_vocab_logsumexp=[10.] * count, target_token_log_probabilities=logp,
            joint_log_probability=math.fsum(logp), full_vocab_normalizer_dtype='float64',
            vocab_size=65536, logits_shape=[1, count, 65536], actual_forward_passes=1,
            output_tokens=0, seconds=.125, resource_admission=dict(device='cuda:0',
                available_ram_bytes=3 * 1024**3, free_cuda_bytes=4 * 1024**3,
                cuda_allocator_limit_bytes=3 * 1024**3),
            peak_cuda_allocated_bytes=1024, peak_cuda_reserved_bytes=2048)
        response = observer.collector._response(raw, metadata)
        child = root / 'requests' / row['key']
        old.write(child / 'backend_raw.json', dict(value=raw))
        old.write(child / 'response.json', response)
        old.seal(child)
        responses[row['key']] = response
    old.write(root / 'complete.json', observer._completion(prepared, executions, responses, frozen))
    old.seal(root)
    protocol = tmp_path / 'atomic-evaluation-protocol.json'
    old.write(protocol, module.DEFAULT_PROTOCOL)
    args = inputs, design, truth, root, model_path, output, protocol
    accesses = old.guard_references(args, monkeypatch, allow=True)
    result = module.evaluate(*args)
    assert accesses
    assert result['source_qa']['correct'] == (0 if tie else 16)
    assert result['source_qa']['ties'] == (16 if tie else 0)
    assert result['coverage']['response_slots'] == 96
    assert result['coverage']['unique_queries'] == 16
    assert result['secondary_case_metrics']['evaluation_units'] == 48
    assert result['secondary_case_metrics']['auroc'] == (.5 if tie else 1.)
    assert result['model_calls'] == 0
    assert data.read_json(output / 'plan.json')['protocol']['primary_control'] == 'sealed_H011_burden_zero_only'
    data.verify_seal(output)
