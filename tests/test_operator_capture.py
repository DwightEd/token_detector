"""Tiny official-model engineering checks; no original weights or dataset."""
from copy import deepcopy

import pytest
import torch

from .test_streamed_observer import tiny, _observer
from myproject.model import attention_backend
from experiments.research import operator_capture as capture


@pytest.fixture(autouse=True)
def two_threads():
    previous=torch.get_num_threads()
    torch.set_num_threads(2)
    try:yield
    finally:torch.set_num_threads(previous)


def _axes():
    ids=[1,2,4,8,16,20,21,22]
    return ids,[False,True,True,False,False,False,False,False],list(range(3,7))


def _hooks(observer):
    return [(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in observer.native.modules()]


def test_streamed_native_unchanged_replays_before_release_and_streams_after(tiny,monkeypatch):
    ids,mask,positions=_axes()
    original=capture.local_effect
    residents=[]
    def resident(layer,*args,**kwargs):
        residents.append(all(not p.is_meta and p.dtype==torch.bfloat16 for p in layer.parameters()))
        return original(layer,*args,**kwargs)
    monkeypatch.setattr(capture,'local_effect',resident)
    with _observer(tiny) as observer, attention_backend(observer.native,'eager'):
        expected=observer.hidden(ids,use_cache=True).last_hidden_state.clone()
        before=_hooks(observer);loads=observer.loads;stored={};counts={}
        def save(index,arrays):
            assert all(p.is_meta for p in observer.layers[index].parameters())
            stored[index]={k:v.detach().cpu().clone() for k,v in arrays.items()}
        with capture.capture_local_effects(observer,ids,mask,positions,counts,save) as state:
            actual=observer.hidden(ids,use_cache=True).last_hidden_state
        assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))
        assert observer.loads-loads==4  # embed +2blocks+finalnorm; replay loads nothing.
        assert residents==[True,True] and _hooks(observer)==before
        assert state['gates']['passed'] and state['completed_layers']==[0,1]
        assert torch.equal(state['native_postnorm'].view(torch.int32),actual.view(torch.int32))
        assert state['H_raw'].shape==actual.shape==(1,8,32)
        assert counts['extra_norm_calls']==counts['extra_mlp_calls']==8
        assert counts['native_mlp_calls']==counts['native_postattention_norm_calls']==2
        assert counts['partition_o_calls']==6 and counts['av_replays']==2
        assert counts['host_transfer_bytes']==0  # This is a CPU-only fixture.
        assert state['accumulation']['layers']==2
        assert state['accumulation']['source64'].dtype==torch.float64
        assert state['accumulation']['source32'].dtype==torch.float32
        for data in stored.values():
            assert data['native_v'].shape==(2,8,8)
            assert data['native_attention'].shape==(4,4,8)
            assert data['pre_o_av'].shape==(1,4,32)
            assert data['key_groups'].sum(0).eq(1).all()
            assert data['key_groups'][1,positions[-1]]  # R history includes self.
            future=torch.arange(8)[None]>torch.tensor(positions)[:,None]
            assert not data['native_attention'].masked_select(future[None]).any()
            assert torch.equal(data['x_mid'],data['layer_input']+data['native_attention_output'])
            assert torch.equal(data['layer_output'],data['x_mid']+data['native_mlp_write'])
            assert torch.equal(data['replay_mlps_reference'],data['replay_mlps_m0'])
        assert observer.resource_report()['all_parameters_meta']


def test_local_numerical_failure_retains_raw_once_and_stops_before_next_layer(tiny,monkeypatch):
    ids,mask,positions=_axes();old=capture.local_effect
    def fail(*args,**kwargs):
        result=old(*args,**kwargs)
        result['gates']['reference_norm']['passed']=False
        result['gates']['passed']=False
        return result
    monkeypatch.setattr(capture,'local_effect',fail)
    with _observer(tiny) as observer,attention_backend(observer.native,'eager'):
        before=_hooks(observer);saved=[];counts={}
        with pytest.raises(capture.CaptureFailure,match='after raw retention'):
            with capture.capture_local_effects(observer,ids,mask,positions,counts,
                    lambda i,a:saved.append((i,set(a)))) as state:
                observer.hidden(ids,use_cache=True)
        assert len(saved)==1 and saved[0][0]==0
        assert {'layer_output','native_mlp_write','replay_mlps_source','delta_source'}<=saved[0][1]
        assert state['completed_layers']==[] and counts['extra_mlp_calls']==4
        assert not state['gates']['passed'] and _hooks(observer)==before
        assert observer.resource_report()['all_parameters_meta']


def test_exceptional_local_module_retains_partial_before_parent_release(tiny):
    ids,mask,positions=_axes()
    with _observer(tiny) as observer,attention_backend(observer.native,'eager'):
        entries=0
        def fail_source(module,args):
            nonlocal entries
            entries+=1
            if entries==3:raise RuntimeError('synthetic local source module failure')
        external=observer.layers[0].mlp.register_forward_pre_hook(fail_source)
        before=_hooks(observer);saved=[];counts={}
        def save(i,a):
            assert not observer.layers[i].mlp.gate_proj.weight.is_meta
            saved.append((i,set(a)))
        try:
            with pytest.raises(capture.CaptureFailure,match='synthetic local source'):
                with capture.capture_local_effects(observer,ids,mask,positions,counts,save):
                    observer.hidden(ids,use_cache=True)
            assert _hooks(observer)==before
        finally:external.remove()
        assert len(saved)==1 and {'replay_norms_source','replay_mlps_reference'}<=saved[0][1]
        assert 'replay_mlps_source' not in saved[0][1]
        assert counts['extra_mlp_calls']==counts['extra_norm_calls']==2
        assert observer.resource_report()['all_parameters_meta']


def test_callback_failure_not_repeated_and_cleanup(tiny):
    ids,mask,positions=_axes()
    with _observer(tiny) as observer,attention_backend(observer.native,'eager'):
        before=_hooks(observer);calls=[]
        def fail(i,a):
            calls.append(i);raise OSError('synthetic writer failed')
        with pytest.raises(capture.CaptureFailure,match='synthetic writer'):
            with capture.capture_local_effects(observer,ids,mask,positions,{},fail):
                observer.hidden(ids,use_cache=True)
        assert calls==[0] and _hooks(observer)==before
        assert observer.resource_report()['all_parameters_meta']


def test_wrong_ids_and_source_axis_rejected_without_native_materialization(tiny):
    ids,mask,positions=_axes()
    with _observer(tiny) as observer:
        bad=deepcopy(mask);bad[-1]=True
        with pytest.raises(ValueError,match='prompt-only'):
            with capture.capture_local_effects(observer,ids,bad,positions,{},lambda *x:None):pass
        changed=list(ids);changed[0]+=1
        with pytest.raises(ValueError,match='original token IDs'):
            with capture.capture_local_effects(observer,ids,mask,positions,{},lambda *x:None):
                observer.hidden(changed,use_cache=True)
        assert observer.loads==0 and not observer._operator_capture_active


def test_no_sdpa_attention_fallback(tiny):
    ids,mask,positions=_axes()
    with _observer(tiny) as observer:
        before=_hooks(observer);saved=[];counts={}
        with pytest.raises(capture.CaptureFailure,match='actual softmax'):
            with capture.capture_local_effects(observer,ids,mask,positions,counts,
                    lambda i,a:saved.append((i,set(a)))):
                observer.hidden(ids,use_cache=True)
        assert len(saved)==1 and counts.get('extra_mlp_calls',0)==0
        assert _hooks(observer)==before and observer.resource_report()['all_parameters_meta']


def test_returned_invalid_norm_is_retained_before_gate(tiny):
    ids,mask,positions=_axes()
    with _observer(tiny) as observer,attention_backend(observer.native,'eager'):
        def invalid(module,args,output):return torch.full_like(output,float('nan'))
        extra=observer.layers[0].post_attention_layernorm.register_forward_hook(invalid)
        saved={};before=_hooks(observer)
        try:
            with pytest.raises(capture.CaptureFailure,match='nonfinite'):
                with capture.capture_local_effects(observer,ids,mask,positions,{},
                        lambda i,a:saved.update({k:v.clone() for k,v in a.items()})):
                    observer.hidden(ids,use_cache=True)
            assert _hooks(observer)==before
        finally:extra.remove()
        assert saved['native_norm_out'].isnan().all()
        assert observer.resource_report()['all_parameters_meta']


def test_returned_invalid_finalnorm_is_available_in_partial_state(tiny):
    ids,mask,positions=_axes()
    with _observer(tiny) as observer,attention_backend(observer.native,'eager'):
        extra=observer.native.model.norm.register_forward_hook(
            lambda m,a,o:torch.full_like(o,float('nan')))
        before=_hooks(observer)
        try:
            with pytest.raises(ValueError,match='nonfinite'):
                with capture.capture_local_effects(observer,ids,mask,positions,{},lambda *x:None) as state:
                    observer.hidden(ids,use_cache=True)
            assert state['native_postnorm'].isnan().all() and state['H_raw'] is not None
            assert _hooks(observer)==before
        finally:extra.remove()
