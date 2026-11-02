import sys
from pathlib import Path
import pytest
from experiments import run, nli_qualification, nli_evaluation


def args():
    return ['r4-experiment', 'transfer', 'qualify-nli', '--inputs', 'inputs.jsonl',
            '--protocol', 'protocol.json', '--model', 'model', '--model-identity',
            'identity.json', '--output', 'outputs/nli']


def test_nli_dispatch_uses_classifier_head(monkeypatch):
    identity, backend, calls = {'synthetic': True}, object(), []
    monkeypatch.setattr(run.data, 'read_json', lambda p: identity)
    monkeypatch.setattr(nli_qualification, 'HFEntailment', lambda p, i, admit_resources: backend)
    monkeypatch.setattr(nli_qualification, 'qualify', lambda *a: calls.append(a))
    monkeypatch.setattr(sys, 'argv', args() + ['--admit-resources'])
    run.main()
    assert calls == [(Path('inputs.jsonl'), Path('protocol.json'), Path('outputs/nli'), backend, identity)]


def test_nli_gate_precedes_identity_and_backend(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError('Resource gate must precede metadata and backend')
    monkeypatch.setattr(run.data, 'read_json', forbidden)
    monkeypatch.setattr(nli_qualification, 'HFEntailment', forbidden)
    monkeypatch.setattr(sys, 'argv', args())
    with pytest.raises(SystemExit) as error:
        run.main()
    assert error.value.code == 2


def test_nli_cpu_dispatch_passes_tokenizer_path_without_constructing_backend(monkeypatch):
    calls = []
    def forbidden(*a, **k):
        raise AssertionError('CLI reader may only delegate to tokenizer-only audited reader')
    monkeypatch.setattr(nli_qualification, 'HFEntailment', forbidden)
    monkeypatch.setattr(nli_evaluation, 'evaluate', lambda *a: calls.append(a))
    monkeypatch.setattr(sys, 'argv', ['r4-experiment', 'transfer', 'evaluate-nli',
        '--inputs', 'inputs.jsonl', '--observations', 'observations', '--natural-panel',
        'natural', '--controls-panel', 'controls', '--model', 'model', '--protocol',
        'protocol.json', '--output', 'outputs/review'])
    run.main()
    assert calls == [(Path('inputs.jsonl'), Path('observations'), Path('natural'),
                      Path('controls'), Path('model'), Path('outputs/review'), Path('protocol.json'))]
