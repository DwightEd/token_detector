import sys
from pathlib import Path
import pytest
from experiments import run,focused_verdict_qualification,semantic_qualification

def args():
    return ['r4-experiment','transfer','qualify-focused-verdict','--inputs','inputs.jsonl',
      '--protocol','protocol.json','--model','model','--model-identity','identity.json','--output','outputs/focused']

def test_focused_cli_uses_shared_focused_stage(monkeypatch):
    identity={'synthetic':True};generator=object();calls=[]
    monkeypatch.setattr(run.data,'read_json',lambda p:identity)
    monkeypatch.setattr(semantic_qualification,'HFGenerator',lambda p,i,admit_resources:generator)
    monkeypatch.setattr(focused_verdict_qualification,'qualify',lambda *a:calls.append(a))
    monkeypatch.setattr(sys,'argv',args()+['--admit-resources'])
    run.main()
    assert calls==[(Path('inputs.jsonl'),Path('protocol.json'),Path('outputs/focused'),generator,identity)]

def test_focused_cli_resource_gate_precedes_model_metadata(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Resource gate must precede model/metadata')
    monkeypatch.setattr(run.data,'read_json',forbidden)
    monkeypatch.setattr(semantic_qualification,'HFGenerator',forbidden)
    monkeypatch.setattr(sys,'argv',args())
    with pytest.raises(SystemExit) as e:run.main()
    assert e.value.code==2
