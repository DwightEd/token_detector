"""Synthetic checks of the explicit offline FP32 dataset loader contract."""
from types import SimpleNamespace

import pytest
import torch

from experiments import nli_qualification as stage
from .test_nli_qualification import MODEL


class _Padded(dict):
    def to(self, device):
        assert device=='cuda:0'
        return self


class _Tokenizer:
    def pad(self,pairs,*,padding,return_tensors):
        assert padding and return_tensors=='pt'
        return _Padded(input_ids=torch.tensor([pairs[0]['input_ids']]),
                       attention_mask=torch.tensor([pairs[0]['attention_mask']]))


@pytest.fixture
def mocked_cuda(monkeypatch):
    calls = {'loads':[],'fractions':[]}
    monkeypatch.setattr(stage,'version',lambda name:stage.HF_RUNTIME[name])
    monkeypatch.setattr(stage,'_available_ram',lambda:1024**3)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'get_device_capability',lambda device:(12,0))
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(7*1024**3,8*1024**3))
    monkeypatch.setattr(torch.cuda,'get_per_process_memory_fraction',lambda device:.875)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda fraction,device:calls['fractions'].append(fraction))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device:None)
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda device:None)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda device:1024**2)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    from experiments import streamed_qualification
    monkeypatch.setattr(streamed_qualification,'_resources',lambda device:{'bounded_synthetic':True})
    class Model:
        training=False
        def eval(self):return self
        def requires_grad_(self,value):
            assert not value
            return self
        def parameters(self):
            return iter([SimpleNamespace(dtype=torch.float32,is_meta=False,device=torch.device('cuda:0'))])
        def to(self,*args):
            raise AssertionError('Dataset loader must not build a CPU model then move it')
        def __call__(self,**encoded):
            assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
            return SimpleNamespace(logits=torch.tensor([[0.,1.,2.]],dtype=torch.float32))
    def load(path,**options):
        calls['loads'].append(options)
        return Model()
    from transformers import AutoModelForSequenceClassification
    monkeypatch.setattr(AutoModelForSequenceClassification,'from_pretrained',load)
    return calls


def test_dataset_dispatch_streams_directly_to_gpu_fp32_and_restores_precision(tmp_path,monkeypatch,mocked_cuda):
    backend=stage.HFEntailment(tmp_path,MODEL,True,execution_mode='dataset-fp32-batch1-v1')
    monkeypatch.setattr(backend,'verify_frozen_identity',lambda:None)
    backend.tokenizer=_Tokenizer()
    pair={'input_ids':[1,3,4,2,5,2],'attention_mask':[1]*6}
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        first=backend.classify([pair])
        second=backend.classify([pair])
        assert first['logits']==second['logits']==[[0.,1.,2.]]
        assert len(mocked_cuda['loads'])==1
        options=mocked_cuda['loads'][0]
        assert options['device_map']=={'':'cuda:0'} and options['torch_dtype']==torch.float32
        assert options['use_safetensors'] and options['local_files_only'] and options['low_cpu_mem_usage']
        assert options['attn_implementation']=='eager'
        assert first['resource_admission']['cuda_allocator_limit_bytes']==6*1024**3
        assert first['resources_before']=={'bounded_synthetic':True}
        assert mocked_cuda['fractions'][0]==.75
    finally:
        backend.close()
    assert mocked_cuda['fractions'][-1]==.875
    assert (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)==flags
    assert backend.model is None and not backend.admitted


def test_dataset_batch_and_ram_rejection_precede_any_load(tmp_path,monkeypatch,mocked_cuda):
    backend=stage.HFEntailment(tmp_path,MODEL,True,execution_mode='dataset-fp32-batch1-v1')
    pair={'input_ids':[1,3,4,2,5,2],'attention_mask':[1]*6}
    with pytest.raises(ValueError,match='batch1'):
        backend.classify([pair,pair])
    monkeypatch.setattr(stage,'_available_ram',lambda:511*1024**2)
    with pytest.raises(RuntimeError,match='floors'):
        backend.classify([pair])
    assert not mocked_cuda['loads']


def test_legacy_mode_keeps_original_two_gib_ram_gate(tmp_path,monkeypatch,mocked_cuda):
    backend=stage.HFEntailment(tmp_path,MODEL,True)
    monkeypatch.setattr(backend,'verify_frozen_identity',lambda:None)
    backend.tokenizer=_Tokenizer()
    pair={'input_ids':[1,3,4,2,5,2],'attention_mask':[1]*6}
    with pytest.raises(RuntimeError,match='2 GiB RAM'):
        backend.classify([pair])
    assert backend.execution_mode=='legacy-pilot' and not mocked_cuda['loads']


def test_unknown_execution_mode_is_not_an_implicit_resource_fallback(tmp_path):
    with pytest.raises(ValueError,match='execution mode'):
        stage.HFEntailment(tmp_path,MODEL,True,execution_mode='try-lower-precision')


def test_failed_native_loader_still_allows_precision_and_allocator_cleanup(tmp_path,monkeypatch,mocked_cuda):
    backend=stage.HFEntailment(tmp_path,MODEL,True,execution_mode='dataset-fp32-batch1-v1')
    monkeypatch.setattr(backend,'verify_frozen_identity',lambda:None)
    backend.tokenizer=_Tokenizer()
    from transformers import AutoModelForSequenceClassification
    def fail(*args,**kwargs):
        raise RuntimeError('injected native loader failure')
    monkeypatch.setattr(AutoModelForSequenceClassification,'from_pretrained',fail)
    flags=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    try:
        with pytest.raises(RuntimeError,match='loader failure'):
            backend.classify([{'input_ids':[1,3,4,2,5,2],'attention_mask':[1]*6}])
    finally:
        backend.close()
    assert backend.model is None and not backend.admitted
    assert mocked_cuda['fractions'][-1]==.875
    assert (torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)==flags
