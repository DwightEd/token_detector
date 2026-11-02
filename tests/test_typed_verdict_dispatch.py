import sys
from pathlib import Path
import pytest
from experiments import run,typed_verdict_qualification,semantic_qualification

def arguments():
    return ['r4-experiment','transfer','qualify-typed-verdict','--inputs','inputs.jsonl',
       '--protocol','protocol.json','--model','model','--model-identity','identity.json',
       '--donors','natural','controls','--output','outputs/h004']

def test_typed_cli_preserves_shared_collector_and_donor_order(monkeypatch):
    identity={'synthetic':True};generator=object();calls=[]
    monkeypatch.setattr(run.data,'read_json',lambda p:identity)
    monkeypatch.setattr(semantic_qualification,'HFGenerator',lambda p,i,admit_resources:generator)
    monkeypatch.setattr(typed_verdict_qualification,'qualify',lambda *a:calls.append(a))
    monkeypatch.setattr(sys,'argv',arguments()+['--admit-resources'])
    run.main()
    assert calls==[(Path('inputs.jsonl'),Path('protocol.json'),Path('outputs/h004'),generator,
                    identity,[Path('natural'),Path('controls')])]

def test_typed_cli_resource_gate_precedes_identity_or_model_access(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('Explicit admission must run first')
    monkeypatch.setattr(run.data,'read_json',forbidden)
    monkeypatch.setattr(semantic_qualification,'HFGenerator',forbidden)
    monkeypatch.setattr(sys,'argv',arguments())
    with pytest.raises(SystemExit) as e:run.main()
    assert e.value.code==2
