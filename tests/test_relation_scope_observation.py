"""Tiny native CPU checks of the sequential observation/retention contract."""

import weakref
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import relation_scope_observation as stage
from experiments.data import file_hash, read_json, verify_seal, write_json
from .test_streamed_observer import tiny, _observer


@pytest.fixture
def world():
    ids, p = [1,2,3,5,8,13,21,31,17,10], 6
    return dict(token_ids=ids,prompt_tokens=p,answer_ids=ids[p:],
        answer_offsets=[[i,i+1] for i in range(len(ids)-p)],
        source_mask=[False,True,True,True,True,False,False,False,False,False],
        query_positions=list(range(p-1,len(ids))),capture_positions=[2,4,6,8],
        capture_receivers=[dict(query_position=i) for i in (2,4,6,8)],
        source_anchors={'Monday:payload':{'overlap_tokens':[2]},
                        'Tuesday:payload':{'overlap_tokens':[3,4]}})


def test_full_head_exact_original_axis_all_layers_and_cache_dies_before_head(tiny,world,tmp_path,monkeypatch):
    output = tmp_path/'native'
    counts = dict(model_calls=0,head_calls=0)
    with torch.inference_mode():
        hidden = tiny['reference'].model(torch.tensor([world['token_ids']]),use_cache=False).last_hidden_state
        expected = tiny['reference'].lm_head(hidden[:,5:9,:])[0].numpy()
    with _observer(tiny) as observer:
        refs = {}
        real_hidden, real_logits = observer.hidden, observer.logits
        def hidden_once(*args,**kwargs):
            value = real_hidden(*args,**kwargs)
            refs['cache'] = weakref.ref(value.past_key_values)
            return value
        def head_without_live_cache(*args,**kwargs):
            assert refs['cache']() is None
            return real_logits(*args,**kwargs)
        monkeypatch.setattr(observer,'hidden',hidden_once)
        monkeypatch.setattr(observer,'logits',head_without_live_cache)
        result = stage.observe_world(observer,world,{'00':17,'30':18},output,counts=counts,target_index=2)
    assert counts == {'model_calls':1,'head_calls':1}
    assert result['all_layers_AV_qualified']
    assert len(result['phase']) == 2
    assert result['hook_cleanup'] and result['new_detector_scores'] == 0 and result['auroc'] is None
    with np.load(output/'output.npz') as saved:
        np.testing.assert_allclose(saved['logits'],expected,atol=2e-6,rtol=2e-6)
        logits = torch.from_numpy(saved['logits'].copy())
    assert [r['token_id'] for r in result['original_token_endpoints']] == world['answer_ids']
    assert result['candidates']['00']['logit'] == float(logits[2,17])
    endpoints = result['original_token_endpoints']
    assert endpoints[0]['receiver_target_axis']['absolute_query_position'] == 5
    assert endpoints[2]['receiver_target_axis']['absolute_query_position'] == 7
    for layer in range(2):
        with np.load(output/f'raw/kv-{layer:02d}.npz') as cache:
            assert cache['key'].shape == (1,2,10,8)
        with np.load(output/f'attention/layer-{layer:02d}.npz') as trace:
            attention = trace['attention']
            assert attention.shape == (4,4,10)
            for row,position in enumerate(world['capture_positions']):
                assert np.count_nonzero(attention[:,row,position+1:]) == 0
    verify_seal(output/'raw')
    verify_seal(output/'attention')
    verify_seal(output)


@pytest.mark.parametrize('fault',['phase','head','nonfinite_head'])
def test_rejected_attempt_keeps_raw_and_never_retries(tiny,world,tmp_path,monkeypatch,fault):
    output = tmp_path/fault
    counts = dict(model_calls=0,head_calls=0)
    with _observer(tiny) as observer:
        if fault=='phase':
            actual = stage.attention_diagnostic
            def fail(*args,**kwargs):
                if kwargs['layer']==1:
                    raise RuntimeError('injected native AV failure')
                return actual(*args,**kwargs)
            monkeypatch.setattr(stage,'attention_diagnostic',fail)
        elif fault=='head':
            def fail(hidden):
                raise RuntimeError('injected head failure')
            monkeypatch.setattr(observer,'logits',fail)
        else:
            monkeypatch.setattr(observer,'logits',lambda hidden:torch.full((1,4,48),float('nan')))
        with pytest.raises(RuntimeError):
            stage.observe_world(observer,world,{'00':17,'30':18},output,counts=counts,target_index=2)
        assert counts == {'model_calls':1,'head_calls':0 if fault=='phase' else 1}
        assert read_json(output/'failure.json')['automatic_retry'] is False
        assert (output/'raw/kv-01.npz').is_file()
        assert (output/'raw/trace-01.npz').is_file()
        verify_seal(output/'raw')
        verify_seal(output)
        with pytest.raises(FileExistsError):
            stage.observe_world(observer,world,{'00':17,'30':18},output,counts=counts,target_index=2)


def test_invalid_prechoice_axis_never_calls_model(tiny,world,tmp_path):
    world['query_positions'][0] += 1
    counts = dict(model_calls=0,head_calls=0)
    with _observer(tiny) as observer, pytest.raises(ValueError):
        stage.observe_world(observer,world,{'00':17,'30':18},tmp_path/'bad',counts=counts,target_index=2)
    assert counts == {'model_calls':0,'head_calls':0}


def test_five_world_progress_is_append_only_and_reused_prefix_never_forwarded(tmp_path,monkeypatch):
    order = list(stage.WORLD_ORDER)
    prepared = dict(worlds={name:{'id':name} for name in order},candidate_ids={'00':17,'30':18},
                    target_answer_index=2,world_order=order)
    calls = []
    def observe(observer,world,candidates,output,*,counts,target_index):
        calls.append(world['id'])
        counts['model_calls']+=1
        counts['head_calls']+=1
        return {'world':world['id']}
    monkeypatch.setattr(stage,'observe_world',observe)
    first = tmp_path/'fresh'
    first.mkdir()
    result,counts = stage.observe_worlds(None,prepared,first)
    assert list(result) == order and calls == order
    assert counts == {'model_calls':5,'head_calls':5}
    assert len(list(first.glob('progress-*.json'))) == 5
    assert not (first/'progress.json').exists()
    assert read_json(first/'progress-Saturday_exception.json')['actual_counts'] == counts
    second = tmp_path/'remaining'
    second.mkdir()
    calls.clear()
    reused = {name:result[name] for name in order[:2]}
    all_results,new_counts = stage.observe_worlds(None,prepared,second,reused=reused)
    assert calls == order[2:] and len(all_results) == 5
    assert new_counts == {'model_calls':3,'head_calls':3}
    assert len(list(second.glob('progress-*.json'))) == 5


def test_failed_later_world_preserves_the_actual_new_attempt_counters(tmp_path,monkeypatch):
    counts = {'model_calls':0,'head_calls':0}
    order = list(stage.WORLD_ORDER)
    def observe(observer,world,candidates,output,*,counts,target_index):
        counts['model_calls']+=1
        if world['id']==order[1]:
            raise RuntimeError('injected second model failure')
        counts['head_calls']+=1
        return {'world':world['id']}
    monkeypatch.setattr(stage,'observe_world',observe)
    prepared = dict(worlds={n:{'id':n} for n in order},candidate_ids={},target_answer_index=55,world_order=order)
    with pytest.raises(RuntimeError):
        stage.observe_worlds(None,prepared,tmp_path,counts=counts)
    assert counts == {'model_calls':2,'head_calls':1}
    assert read_json(tmp_path/'progress-native.json')['actual_counts'] == {'model_calls':1,'head_calls':1}


@pytest.fixture
def retained_prefix(tmp_path):
    previous_stage = tmp_path/'previous.py'
    previous_stage.write_bytes(Path(stage.__file__).read_bytes())
    code = {'experiments/relation_scope_observation.py':file_hash(previous_stage),
            'experiments/research_kernel.py':'a'*64,'experiments/run.py':'b'*64}
    identity = {'source_sha256':dict(code)}
    assets,prepared = {'identity':'frozen-assets'}, {'answer_ids':[17,18]}
    root = tmp_path/'retained'
    root.mkdir()
    write_json(root/'plan.json',{'assets':assets,'source_runtime':{'source_sha256':code}})
    write_json(root/'eligibility.json',prepared)
    write_json(root/'failure.json',{'type':'FileExistsError','message':'immutable progress.json',
                                  'actual_counts':{'model_calls':2,'head_calls':2}})
    for name in stage.WORLD_ORDER[:2]:
        unit = root/name
        unit.mkdir()
        for sub in ('raw','attention'):
            (unit/sub).mkdir()
            write_json(unit/sub/'receipt.json',{'preserved':True})
            stage._seal(unit/sub)
        np.savez_compressed(unit/'output.npz',logits=np.zeros((2,48),dtype=np.float32))
        write_json(unit/'result.json',{'all_layers_AV_qualified':True,'hook_cleanup':True,
            'original_token_endpoints':[{'token_id':17},{'token_id':18}]})
        stage._seal(unit)
    stage._seal(root)
    return root,previous_stage,prepared,assets,identity


def test_reuse_requires_all_nested_seals_and_unchanged_computation(retained_prefix):
    root,previous,prepared,assets,identity = retained_prefix
    result,receipt = stage._reuse_units(root,previous,prepared,assets,identity)
    assert list(result) == ['native','S_minus']
    assert all(not value['model_recomputed'] for value in receipt.values())
    identity['source_sha256']['experiments/research_kernel.py'] = 'c'*64
    with pytest.raises(ValueError,match='dependency changed'):
        stage._reuse_units(root,previous,prepared,assets,identity)


def test_reuse_rejects_tampered_raw_even_when_endpoint_json_is_untouched(retained_prefix):
    root,previous,prepared,assets,identity = retained_prefix
    (root/'native/raw/receipt.json').write_text('{"changed":true}',encoding='utf-8')
    with pytest.raises(ValueError,match='Sealed artifact changed'):
        stage._reuse_units(root,previous,prepared,assets,identity)
