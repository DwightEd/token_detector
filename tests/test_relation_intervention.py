"""Constructed-stimulus and exact-path audits; no original weights or labels."""

from copy import deepcopy
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.relation_intervention import run_assay
from experiments.research.relation_intervention import (
    BRANCHES, Branch, EligibilityError, WORLD_ORDER, apply_branch, matched_noise,
    prepare_quartet, scientific_summary,
)
from experiments.streamed_head_qualification import cache_digest, clone_cache
from .test_streamed_observer import tiny, _observer


class WordTokenizer:
    is_fast=True
    all_special_ids=[]
    added_tokens_decoder={}

    def __init__(self,split_numerals=False):
        self.vocabulary={}
        self.split_numerals=split_numerals

    def apply_chat_template(self,messages,*,tokenize,add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return '<user>'+messages[0]['content'].strip()+'</user><assistant>'

    def __call__(self,text,*,add_special_tokens,return_offsets_mapping):
        assert add_special_tokens is False and return_offsets_mapping is True
        pattern=r'[A-Za-z_]+|\d|[^A-Za-z_\d]' if self.split_numerals else r'[A-Za-z_]+|\d+|[^A-Za-z_\d]'
        ids,offsets=[],[]
        for match in re.finditer(pattern,text):
            self.vocabulary.setdefault(match[0],len(self.vocabulary))
            ids.append(self.vocabulary[match[0]])
            offsets.append(match.span())  # Actual Fast tokenizers commonly use tuples.
        return {'input_ids':ids,'offset_mapping':offsets}

    def decode(self,ids):
        inverse={v:k for k,v in self.vocabulary.items()}
        return ''.join(inverse[value] for value in ids)


def test_quartet_has_exact_role_value_schema_prefix_positions_and_no_rescue():
    tokenizer=WordTokenizer()
    quartet=prepare_quartet(tokenizer)
    worlds=quartet['worlds']
    assert list(worlds)==list(WORLD_ORDER)
    assert [worlds[name]['alpha_value'] for name in WORLD_ORDER]==[100,200,200,100]
    assert quartet['budget']=={'model_calls':48,'head_calls':44}
    assert len(BRANCHES)==10
    assert len({world['query_position'] for world in worlds.values()})==1
    assert len({tuple(r['position'] for r in world['occurrences']) for world in worlds.values()})==1
    for world in worlds.values():
        assert world['condition_position']<world['query_position']
        assert not any(world['source_mask'][p] for p in world['off_target_positions'])
        assert all(world['token_ids'][r['position']]==quartet['candidate_ids'][str(r['value'])]
                   for r in world['occurrences'])
    with pytest.raises(EligibilityError,match='one complete'):
        prepare_quartet(WordTokenizer(split_numerals=True))
    with pytest.raises(EligibilityError,match='length'):
        prepare_quartet(WordTokenizer(),maximum=2)


def _random_cache(seed):
    from transformers.cache_utils import DynamicCache
    generator=torch.Generator().manual_seed(seed)
    return DynamicCache.from_legacy_cache(tuple(tuple(torch.randn(1,2,6,8,generator=generator)
                                       for _ in range(2)) for _ in range(2)))


@pytest.mark.parametrize('kind',['q_role','k_role','v_role','qk_role','v_value','noop','random_k','off_target_random_k'])
def test_patch_exact_positions_channels_upper_cache_and_donor_immutability(kind):
    config=SimpleNamespace(num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,hidden_size=32)
    pristine,role,value=_random_cache(1),_random_cache(2),_random_cache(3)
    before={id(c):cache_digest(c) for c in (pristine,role,value)}
    branch_cache=clone_cache(pristine,config,6)
    world={'occurrences':[{'position':2},{'position':4}],'off_target_positions':[0,1]}
    branch=Branch(kind,0 if 'random' in kind else None)
    apply_branch(branch_cache,role,value,world,branch,layer=1)
    assert all(cache_digest(c)==before[id(c)] for c in (pristine,role,value))
    original=pristine.to_legacy_cache()
    changed=branch_cache.to_legacy_cache()
    assert all(torch.equal(a,b) for a,b in zip(original[0],changed[0],strict=True))
    if kind in {'noop','q_role'}:
        assert cache_digest(pristine)==cache_digest(branch_cache)
    elif kind in {'v_role','v_value'}:
        assert torch.equal(original[1][0],changed[1][0])
        donor=role if kind=='v_role' else value
        assert torch.equal(changed[1][1][:,:,[2,4]],donor.to_legacy_cache()[1][1][:,:,[2,4]])
        assert torch.equal(original[1][1][:,:,[0,1,3,5]],changed[1][1][:,:,[0,1,3,5]])
    else:
        assert torch.equal(original[1][1],changed[1][1])
        positions=[0,1] if kind=='off_target_random_k' else [2,4]
        untouched=[i for i in range(6) if i not in positions]
        assert torch.equal(original[1][0][:,:,untouched],changed[1][0][:,:,untouched])
        delta=role.to_legacy_cache()[1][0][:,:,[2,4]]-original[1][0][:,:,[2,4]]
        actual=changed[1][0][:,:,positions]-original[1][0][:,:,positions]
        torch.testing.assert_close(actual.norm(dim=-1),delta.norm(dim=-1),atol=1e-6,rtol=1e-6)


def test_matched_random_noise_is_scoped_seeded_per_head_per_position():
    delta=torch.arange(32,dtype=torch.float32).reshape(1,2,2,8)
    delta[0,0,0]=0
    state=torch.random.get_rng_state().clone()
    first,repeat,other=matched_noise(delta,0),matched_noise(delta,0),matched_noise(delta,1)
    assert torch.equal(first,repeat) and not torch.equal(first,other)
    assert torch.equal(torch.random.get_rng_state(),state)
    torch.testing.assert_close(first.norm(dim=-1),delta.norm(dim=-1))
    assert torch.count_nonzero(first[0,0,0])==0


def test_tiny_official_engine_keeps_exact48_44_all_units_and_cache_reuse(tiny,tmp_path):
    quartet=prepare_quartet(WordTokenizer())
    assert max(max(world['token_ids']) for world in quartet['worlds'].values())<48
    with _observer(tiny) as observer:
        result=run_assay(observer,quartet,tmp_path,layer=1)
        assert result['actual_counts']=={'model_calls':48,'head_calls':44}
        assert result['cache_identity_unchanged']
        assert result['all_physical_heads_retained']
        assert not result['natural_detection_evaluated'] and result['auroc'] is None
        assert len(result['rows'])==4
        assert len(list((tmp_path/'units').glob('*/seal.json')))==44
        assert len(list((tmp_path/'prefills').glob('*/all-layer-KV.npz')))==4
        assert observer.resource_report()['all_parameters_meta']


def test_instrument_failure_retains_completed_prefill_without_another_call(tiny,tmp_path,monkeypatch):
    import experiments.relation_intervention as stage
    def broken(*args,**kwargs):raise RuntimeError('injected phase rejection')
    monkeypatch.setattr(stage,'attention_diagnostic',broken)
    counts={'model_calls':0,'head_calls':0}
    with _observer(tiny) as observer:
        with pytest.raises(RuntimeError,match='phase rejection'):
            run_assay(observer,prepare_quartet(WordTokenizer()),tmp_path,layer=1,counts=counts)
        assert counts=={'model_calls':1,'head_calls':0}
        assert (tmp_path/'prefills/S00/native-trace.npz').is_file()
        assert (tmp_path/'prefills/S00/all-layer-KV.npz').is_file()
        assert observer.resource_report()['all_parameters_meta']


def _endpoint(margin,top=0,address=0.):
    return {'margin':margin,'top1':top,'candidates':{'100':{'logit':-margin/2},'200':{'logit':margin/2}},
            'address':{'address':address}}


def test_scientific_gates_keep_single_quartet_negative_and_low_legibility():
    quartet=prepare_quartet(WordTokenizer())
    observations={}
    for name,world in quartet['worlds'].items():
        margin=2. if world['alpha_value']==200 else -2.
        branch={b.name+('' if b.seed is None else '-'+str(b.seed)):_endpoint(margin) for b in BRANCHES}
        branch['k_role']=_endpoint(-margin,address=.25)
        branch['v_value']=_endpoint(-margin)
        observations[name]={'baseline':_endpoint(margin,top=quartet['candidate_ids']['200']), 'branches':branch}
    passed=scientific_summary(quartet,observations)
    assert passed['scoped_mechanism_gate']
    observations['S00']['branches']['k_role']['margin']=-2.
    assert not scientific_summary(quartet,observations)['scoped_mechanism_gate']
    observations['S00']['branches']['k_role']['margin']=2.
    observations['S00']['baseline']['top1']=47
    failed=scientific_summary(quartet,observations)
    assert not failed['task_legible'] and not failed['scoped_mechanism_gate']
    assert len(failed['rows'])==4 and failed['auroc'] is None
