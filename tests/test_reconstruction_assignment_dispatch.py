import sys
from pathlib import Path
from experiments import run, reconstruction_assignment


def test_cached_assignment_stage_delegates_without_instrument_or_reference_reads(monkeypatch):
    calls=[]
    def forbidden(*a,**k):raise AssertionError('CLI may only delegate CPU cache analysis')
    monkeypatch.setattr(run.data,'read_json',forbidden)
    monkeypatch.setattr(reconstruction_assignment,'analyze',lambda *a:calls.append(a))
    monkeypatch.setattr(sys,'argv',['r4-experiment','transfer','analyze-reconstruction-assignments',
        '--inputs','inputs.jsonl','--observations','observations','--model','model',
        '--reconstruction-review','review','--output','outputs/assignment','--protocol','protocol.json'])
    run.main()
    assert calls==[(Path('inputs.jsonl'),Path('observations'),Path('model'),Path('review'),
        Path('outputs/assignment'),Path('protocol.json'))]
