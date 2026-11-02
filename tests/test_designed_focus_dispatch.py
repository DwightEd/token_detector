import sys
from pathlib import Path
import pytest
from experiments import run, designed_focus_qualification as qualifier, designed_focus_evaluation as evaluator
from experiments import semantic_qualification


def args():
    return ['r4-experiment','transfer','qualify-designed-focus','--inputs','inputs.jsonl',
        '--protocol','protocol.json','--model','model','--model-identity','identity.json','--output','outputs/burden']


def test_designed_dispatches_shared_model_and_fixed_qualifier(monkeypatch):
    identity,backend,calls={'synthetic':True},object(),[]
    monkeypatch.setattr(run.data,'read_json',lambda p:identity)
    monkeypatch.setattr(semantic_qualification,'HFGenerator',lambda p,i,admit_resources:backend)
    monkeypatch.setattr(qualifier,'qualify',lambda *a:calls.append(a))
    monkeypatch.setattr(sys,'argv',args()+['--admit-resources']);run.main()
    assert calls==[(Path('inputs.jsonl'),Path('protocol.json'),Path('outputs/burden'),backend,identity)]


def test_designed_gate_precedes_metadata(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Gate before metadata/model')
    monkeypatch.setattr(run.data,'read_json',forbidden)
    monkeypatch.setattr(sys,'argv',args())
    with pytest.raises(SystemExit) as e:run.main()
    assert e.value.code==2


def test_designed_cpu_dispatch_never_constructs_model(monkeypatch):
    calls=[]
    def forbidden(*a,**k):raise AssertionError('No model in CPU reader')
    monkeypatch.setattr(semantic_qualification,'HFGenerator',forbidden)
    monkeypatch.setattr(run.data,'read_json',forbidden)
    monkeypatch.setattr(evaluator,'evaluate',lambda *a:calls.append(a))
    monkeypatch.setattr(sys,'argv',['r4-experiment','transfer','evaluate-designed-focus',
        '--inputs','inputs.jsonl','--design','plan.json','--construction','construction_truth.jsonl',
        '--observations','observations','--output','outputs/review','--protocol','protocol.json'])
    run.main()
    assert calls==[(Path('inputs.jsonl'),Path('plan.json'),Path('construction_truth.jsonl'),
        Path('observations'),Path('outputs/review'),Path('protocol.json'))]
