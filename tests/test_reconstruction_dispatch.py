import sys
from pathlib import Path
import pytest
from experiments import run, answer_completion, reconstruction_qualification as qualifier
from experiments import reconstruction_evaluation as evaluator


def args():
    return ['r4-experiment','transfer','qualify-reconstruction','--inputs','inputs.jsonl',
        '--protocol','protocol.json','--model','model','--model-identity','identity.json','--output','outputs/reconstruction']


def test_reconstruction_dispatches_complete_sequence_backend(monkeypatch):
    identity,backend,calls={'synthetic':True},object(),[]
    monkeypatch.setattr(run.data,'read_json',lambda p:identity)
    monkeypatch.setattr(answer_completion,'HFAnswerCompletion',lambda p,i,admit_resources:backend)
    monkeypatch.setattr(qualifier,'qualify',lambda *a:calls.append(a))
    monkeypatch.setattr(sys,'argv',args()+['--admit-resources']);run.main()
    assert calls==[(Path('inputs.jsonl'),Path('protocol.json'),Path('outputs/reconstruction'),backend,identity)]


def test_reconstruction_gate_precedes_metadata(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Resource gate before metadata/backend')
    monkeypatch.setattr(run.data,'read_json',forbidden)
    monkeypatch.setattr(answer_completion,'HFAnswerCompletion',forbidden)
    monkeypatch.setattr(sys,'argv',args())
    with pytest.raises(SystemExit) as e:run.main()
    assert e.value.code==2


def test_reconstruction_cpu_dispatch_passes_tokenizer_path_only(monkeypatch):
    calls=[]
    def forbidden(*a,**k):raise AssertionError('CPU CLI cannot construct completion backend')
    monkeypatch.setattr(answer_completion,'HFAnswerCompletion',forbidden)
    monkeypatch.setattr(run.data,'read_json',forbidden)
    monkeypatch.setattr(evaluator,'evaluate',lambda *a:calls.append(a))
    monkeypatch.setattr(sys,'argv',['r4-experiment','transfer','evaluate-reconstruction',
        '--inputs','inputs.jsonl','--design','plan.json','--construction','construction_truth.jsonl',
        '--observations','observations','--model','model','--output','outputs/review','--protocol','protocol.json'])
    run.main()
    assert calls==[(Path('inputs.jsonl'),Path('plan.json'),Path('construction_truth.jsonl'),
        Path('observations'),Path('model'),Path('outputs/review'),Path('protocol.json'))]
