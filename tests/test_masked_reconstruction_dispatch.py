import sys
from pathlib import Path
import pytest
from experiments import run, answer_completion
from experiments import masked_reconstruction_qualification as qualifier


def args():
    return ['r4-experiment', 'transfer', 'qualify-masked-reconstruction',
        '--inputs', 'inputs.jsonl', '--protocol', 'protocol.json', '--model', 'model',
        '--model-identity', 'identity.json', '--output', 'outputs/masked']


def test_masked_dispatch_uses_shared_complete_sequence_backend(monkeypatch):
    identity, backend, calls = {'synthetic': True}, object(), []
    monkeypatch.setattr(run.data, 'read_json', lambda path: identity)
    monkeypatch.setattr(answer_completion, 'HFAnswerCompletion', lambda path, identity, admit_resources: backend)
    monkeypatch.setattr(qualifier, 'qualify', lambda *values: calls.append(values))
    monkeypatch.setattr(sys, 'argv', args() + ['--admit-resources'])
    run.main()
    assert calls == [(Path('inputs.jsonl'), Path('protocol.json'), Path('outputs/masked'), backend, identity)]


def test_masked_resource_gate_precedes_metadata_and_backend(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No metadata/backend before explicit resource admission')
    monkeypatch.setattr(run.data, 'read_json', forbidden)
    monkeypatch.setattr(answer_completion, 'HFAnswerCompletion', forbidden)
    monkeypatch.setattr(sys, 'argv', args())
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2


def test_masked_cpu_dispatch_cannot_construct_observation_backend(monkeypatch):
    from experiments import masked_reconstruction_evaluation as evaluator
    calls = []
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU dispatch cannot construct or prepare a neural observer')
    monkeypatch.setattr(run.data, 'read_json', forbidden)
    monkeypatch.setattr(answer_completion, 'HFAnswerCompletion', forbidden)
    monkeypatch.setattr(evaluator, 'evaluate', lambda *values: calls.append(values))
    values = ['inputs.jsonl', 'plan.json', 'construction_truth.jsonl', 'observations',
              'model', 'h008-review', 'outputs/review', 'protocol.json']
    options = ['inputs', 'design', 'construction', 'observations', 'model', 'h008-review', 'output', 'protocol']
    argv = ['r4-experiment', 'transfer', 'evaluate-masked-reconstruction']
    for name, value in zip(options, values, strict=True):
        argv.extend(['--' + name, value])
    monkeypatch.setattr(sys, 'argv', argv)
    run.main()
    assert calls == [tuple(Path(value) for value in values)]
