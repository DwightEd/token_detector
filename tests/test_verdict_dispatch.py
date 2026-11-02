"""Central dispatch must preserve the resource gate and shared collector API."""
import sys
from pathlib import Path
import pytest
from experiments import run, semantic_qualification, verdict_qualification

def arguments():
    return ['r4-experiment','transfer','qualify-verdict','--inputs','inputs.jsonl',
            '--protocol','protocol.json','--model','model','--model-identity','identity.json',
            '--output','outputs/review']

def test_central_verdict_dispatch_uses_frozen_shared_collector(monkeypatch):
    identity={'model_id':'synthetic'};calls=[];generator=object()
    monkeypatch.setattr(run.data,'read_json',lambda p: identity)
    monkeypatch.setattr(semantic_qualification,'HFGenerator',
                        lambda p,i,admit_resources: generator if admit_resources and i is identity else None)
    monkeypatch.setattr(verdict_qualification,'qualify',lambda *a:calls.append(a))
    monkeypatch.setattr(sys,'argv',arguments()+['--admit-resources'])
    run.main()
    assert calls==[(Path('inputs.jsonl'),Path('protocol.json'),Path('outputs/review'),generator,identity)]

def test_central_verdict_refuses_before_identity_or_model_access(monkeypatch):
    def denied(*a,**k):raise AssertionError('Resource gate must run first')
    monkeypatch.setattr(run.data,'read_json',denied)
    monkeypatch.setattr(semantic_qualification,'HFGenerator',denied)
    monkeypatch.setattr(sys,'argv',arguments())
    with pytest.raises(SystemExit) as error:run.main()
    assert error.value.code==2
