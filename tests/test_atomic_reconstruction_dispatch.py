import sys
from pathlib import Path
import pytest
from experiments import run, answer_completion
from experiments import atomic_reconstruction_qualification as observer
from experiments import atomic_reconstruction_evaluation as evaluator


def args():
    return ['r4-experiment', 'transfer', 'qualify-atomic-reconstruction',
        '--inputs', 'inputs.jsonl', '--protocol', 'protocol.json', '--model', 'model',
        '--model-identity', 'identity.json', '--output', 'outputs/atomic']


def test_atomic_dispatch_reuses_complete_sequence_backend(monkeypatch):
    identity, backend, calls = {'synthetic': True}, object(), []
    monkeypatch.setattr(run.data, 'read_json', lambda path: identity)
    monkeypatch.setattr(answer_completion, 'HFAnswerCompletion', lambda path, identity, admit_resources: backend)
    monkeypatch.setattr(observer, 'qualify', lambda *values: calls.append(values))
    monkeypatch.setattr(sys, 'argv', args() + ['--admit-resources'])
    run.main()
    assert calls == [(Path('inputs.jsonl'), Path('protocol.json'), Path('outputs/atomic'), backend, identity)]


def test_atomic_resource_gate_precedes_metadata_and_backend(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('No backend before resource gate')
    monkeypatch.setattr(run.data, 'read_json', forbidden)
    monkeypatch.setattr(answer_completion, 'HFAnswerCompletion', forbidden)
    monkeypatch.setattr(sys, 'argv', args())
    with pytest.raises(SystemExit) as error: run.main()
    assert error.value.code == 2


def test_atomic_CPU_dispatch_passes_no_model_instance(monkeypatch):
    calls = []
    def forbidden(*args, **kwargs): raise AssertionError('CPU stage cannot construct a model')
    monkeypatch.setattr(run.data, 'read_json', forbidden)
    monkeypatch.setattr(answer_completion, 'HFAnswerCompletion', forbidden)
    monkeypatch.setattr(evaluator, 'evaluate', lambda *values: calls.append(values))
    options = ['inputs', 'design', 'construction', 'observations', 'model', 'output', 'protocol']
    values = ['inputs.jsonl', 'plan.json', 'construction_truth.jsonl', 'observations', 'model', 'outputs/review', 'protocol.json']
    argv = ['r4-experiment', 'transfer', 'evaluate-atomic-reconstruction']
    for name, value in zip(options, values, strict=True): argv += ['--' + name, value]
    monkeypatch.setattr(sys, 'argv', argv)
    run.main()
    assert calls == [tuple(Path(value) for value in values)]
