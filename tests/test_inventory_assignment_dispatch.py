from pathlib import Path
import sys
from experiments import run, answer_completion
from experiments import natural_masked_inventory as inventory
from experiments import masked_assignment as assignment


def forbidden(*args, **kwargs):
    raise AssertionError('CPU metadata/cache dispatch cannot construct or read neural backend metadata')


def test_natural_artifact_dispatch_never_prepares_neural_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(run.data, 'read_json', forbidden)
    monkeypatch.setattr(answer_completion, 'HFAnswerCompletion', forbidden)
    monkeypatch.setattr(inventory, 'prepare', lambda *values: calls.append(values))
    monkeypatch.setattr(sys, 'argv', ['r4-experiment', 'transfer', 'prepare-natural-masked-inventory',
        '--inputs', 'inputs.jsonl', '--h010-root', 'h010', '--output', 'outputs/inventory', '--protocol', 'protocol.json'])
    run.main()
    assert calls == [(Path('inputs.jsonl'), Path('h010'), Path('outputs/inventory'), Path('protocol.json'))]


def test_cached_assignment_dispatch_never_prepares_neural_backend(monkeypatch):
    calls = []
    monkeypatch.setattr(run.data, 'read_json', forbidden)
    monkeypatch.setattr(answer_completion, 'HFAnswerCompletion', forbidden)
    monkeypatch.setattr(assignment, 'analyze', lambda *values: calls.append(values))
    monkeypatch.setattr(sys, 'argv', ['r4-experiment', 'transfer', 'analyze-masked-assignments',
        '--inputs', 'inputs.jsonl', '--observations', 'observations', '--model', 'model',
        '--reconstruction-review', 'review', '--output', 'outputs/assignment', '--protocol', 'protocol.json'])
    run.main()
    assert calls == [(Path('inputs.jsonl'), Path('observations'), Path('model'), Path('review'),
                      Path('outputs/assignment'), Path('protocol.json'))]
