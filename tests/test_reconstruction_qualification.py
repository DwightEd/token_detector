"""New source-only completion gates with synthetic facts and a fake scorer."""
from copy import deepcopy
from itertools import product
import json
import math
from pathlib import Path
import pytest
from experiments import answer_completion, data, reconstruction_qualification as module
from experiments.research import source_reconstruction as kernel

MODEL=dict(model_id='synthetic-completion',model_files_sha256={'weights':'a'*64})


class Tokenizer:
    eos_token_id=1
    def apply_chat_template(self,messages,tokenize,add_generation_prompt):
        assert add_generation_prompt is True
        text=json.dumps(messages,ensure_ascii=False,sort_keys=True)+'\nASSISTANT\n'
        return self.encode(text,add_special_tokens=False) if tokenize else text
    def encode(self,text,add_special_tokens):
        assert add_special_tokens is False
        return [ord(c)+32 for c in text]
    def decode(self,ids,skip_special_tokens,clean_up_tokenization_spaces):
        return ''.join(chr(i-32) for i in ids)


class Backend:
    model_identity=MODEL
    def __init__(self,check=None,bad=None):
        self.check,self.bad,self.calls,self.preflights=check,bad,0,0
    def validate_completion(self,messages,candidate):
        self.preflights+=1
        return answer_completion.completion_metadata(Tokenizer(),messages,candidate)
    def score_completion(self,messages,candidate):
        if self.check:self.check(messages,candidate)
        self.calls+=1
        m=answer_completion.completion_metadata(Tokenizer(),messages,candidate)
        n=m['target_tokens'];logits=[-.2*i for i in range(n)];norms=[1.+.1*i for i in range(n)]
        logp=[a-b for a,b in zip(logits,norms)]
        result=dict(m,target_token_logits=logits,full_vocab_logsumexp=norms,
            target_token_log_probabilities=logp,joint_log_probability=math.fsum(logp),
            full_vocab_normalizer_dtype='float64',vocab_size=65536,logits_shape=[1,n,65536],
            actual_forward_passes=1,output_tokens=0,seconds=.125,
            resource_admission=dict(device='cuda:0',available_ram_bytes=3*1024**3,
                free_cuda_bytes=4*1024**3,cuda_allocator_limit_bytes=3*1024**3),
            peak_cuda_allocated_bytes=1024,peak_cuda_reserved_bytes=2048)
        if self.bad=='first_token':result['target_token_log_probabilities']=logp[1:]
        if self.bad=='joint':result['joint_log_probability']+=1.
        if self.bad=='mutation':messages[0]['content']='changed'
        return result


def fake48():
    rows=[]
    for family,v,s,r,b in product(('winner','sole_signer'),(0,1),(0,1),(0,1),(0,4,12)):
        people=[('Ava Lorn','Bex Pell'),('Cara Wynn','Dax Ford')][v]
        events=('Amber Cup','Cobalt Cup') if family=='winner' else ('Amber Accord','Cobalt Accord')
        assigned=people if s==0 else people[::-1]
        if family=='winner':
            facts=[]
            for event,person in zip(events,assigned):
                facts.extend([f'The {event} final was a match between {people[0]} and {people[1]}.',
                    f'{person} was the sole winner of the {event} final.'])
        else:facts=[f'{p} was the sole signer of the {e}.' for p,e in zip(assigned,events)]
        source=' '.join(facts+list(kernel.FILLERS))
        answer=people if r==0 else people[::-1]
        if family=='winner':
            target=f'{answer[0]} won the {events[0]} final, and {answer[1]} won the {events[1]} final. '
        else:
            target=f'{answer[0]} was the sole signer of the {events[0]}, and {answer[1]} was the sole signer of the {events[1]}. '
        response=target+' '.join(kernel.FILLERS[:b])
        rows.append(dict(id=f'{family}-{v}-{s}-{r}-{b}',source_id=f'{family}-{v}-{s}',task='QA',
            split='test',prompt='Original source: '+source,source=source,response=response,generator='synthetic'))
    return rows


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    def write(path,value):
        Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    monkeypatch.setattr(data,'write_json',write)
    monkeypatch.setattr(module,'research_identity',lambda:{'source_sha256':{'fixed':'b'*64}})
    monkeypatch.setattr(module,'completed_identity',lambda source:source)
    def build(count=48):
        root=tmp_path/'inference';root.mkdir()
        inputs=root/'inputs.jsonl'
        inputs.write_text(''.join(json.dumps(r)+'\n' for r in fake48()[:count]),encoding='utf-8')
        protocol=tmp_path/'protocol.json'
        write(protocol,dict(module.DEFAULT_PROTOCOL,original_inputs_sha256=data.file_hash(inputs)))
        return inputs,protocol,tmp_path/'outputs'/'observe'
    return build


def test_all32_full_completions_frozen_before_first_forward_without_response_queries(prepared,monkeypatch):
    inputs,protocol,output=prepared()
    original_open=Path.open
    def guarded(path,*a,**k):
        assert path.name not in {'construction_truth.jsonl','evaluation_only.jsonl','manifest.json'}
        return original_open(path,*a,**k)
    monkeypatch.setattr(Path,'open',guarded)
    def frozen(messages,candidate):
        plan=data.read_json(output/'plan.json');bindings=data.read_json(output/'freeze.json')
        assert len(plan['prepared']['originals'])==48 and len(plan['executions'])==32
        assert len([n for n in bindings if n.endswith('request.json')])==32
        assert backend.preflights==32
        body=json.loads(messages[-1]['content'])
        assert set(body)=={'source','question'} and candidate in kernel.source_queries(body['source'])['people']
        assert all(name not in body['question'] for name in kernel.source_queries(body['source'])['people'])
    backend=Backend(frozen)
    result=module.qualify(inputs,protocol,output,backend,MODEL)
    assert backend.calls==32
    assert result['coverage']==dict(original_cases=48,unique_sources=8,unique_queries=16,
        unique_completions=32,response_slots=96,unknown_cases=0,unknown_slots=0)
    assert result['costs']['actual_forward_passes']==32 and result['costs']['output_tokens']==0
    assert len(result['records'])==48 and all(len(r['slots'])==2 for r in result['records'])
    data.verify_seal(output)


@pytest.mark.parametrize('bad',['first_token','joint','mutation'])
def test_bad_observation_is_retained_and_never_marked_complete(prepared,bad):
    args=prepared();backend=Backend(bad=bad)
    with pytest.raises(ValueError):module.qualify(*args,backend,MODEL)
    assert backend.calls==1 and not (args[2]/'complete.json').exists() and not (args[2]/'seal.json').exists()
    assert len(list((args[2]/'requests').glob('*/backend_raw.json')))==1


def test_missing_population_rejects_before_tokenizer_or_model(prepared):
    args=prepared(count=47);backend=Backend()
    with pytest.raises(ValueError,match='All48'):module.qualify(*args,backend,MODEL)
    assert backend.calls==backend.preflights==0 and not args[2].exists()


def test_truth_protocol_alias_rejects_before_any_file_read(prepared,monkeypatch):
    inputs,protocol,output=prepared()
    def forbidden(*a,**k):raise AssertionError('May not read a label/protocol alias')
    monkeypatch.setattr(data,'read_json',forbidden)
    with pytest.raises(ValueError,match='Independent'):
        module.qualify(inputs,protocol.with_name('construction_truth.jsonl'),output,Backend(),MODEL)


def test_final_input_mutation_revokes_complete_and_seal(prepared,monkeypatch):
    args=prepared();original=module._seal
    def sealing(root,names):
        original(root,names)
        if root==args[2]:args[0].write_text('changed',encoding='utf-8')
    monkeypatch.setattr(module,'_seal',sealing)
    with pytest.raises(ValueError,match='Frozen'):module.qualify(*args,Backend(),MODEL)
    assert not (args[2]/'complete.json').exists() and not (args[2]/'seal.json').exists()
    assert (args[2]/'freeze.json').exists()


def test_later_backend_cannot_reseal_an_earlier_response(prepared):
    args=prepared()
    def tamper(messages,candidate):
        if backend.calls==1:
            path=next((args[2]/'requests').glob('*/response.json'))
            value=data.read_json(path);value['joint_log_probability']+=.25
            path.write_text(json.dumps(value),encoding='utf-8')
    backend=Backend(tamper)
    with pytest.raises(ValueError,match='Previously sealed'):
        module.qualify(*args,backend,MODEL)
    assert backend.calls==2 and not (args[2]/'complete.json').exists()


def test_unexpected_member_rejects_before_its_payload_hash(prepared,monkeypatch):
    args=prepared();unexpected=args[2]/'unexpected.json';original=data.file_hash;accesses=[]
    def hashing(path):
        if Path(path)==unexpected:
            accesses.append(path);raise AssertionError('Unsafe member was hashed')
        return original(path)
    monkeypatch.setattr(data,'file_hash',hashing)
    def extra(messages,candidate):unexpected.write_text('NEVER READ',encoding='utf-8')
    with pytest.raises(ValueError,match='Unexpected reconstruction root'):
        module.qualify(*args,Backend(extra),MODEL)
    assert accesses==[] and not (args[2]/'complete.json').exists()


@pytest.mark.parametrize('changed',['prefix','eos'])
def test_candidate_events_must_share_one_conditional_prefix_and_eos(prepared,changed):
    args=prepared()
    class Changed(Backend):
        def validate_completion(self,messages,candidate):
            m=super().validate_completion(messages,candidate)
            if self.preflights==2:
                if changed=='prefix':
                    m['prefix_ids']=[999]+m['prefix_ids'];m['prefix_tokens']+=1
                    m['input_ids']=[999]+m['input_ids'];m['input_tokens']+=1
                    m['target_prediction_positions']=[i+1 for i in m['target_prediction_positions']]
                else:m['eos_token_id']=2;m['target_token_ids'][-1]=2
            return m
    backend=Changed()
    with pytest.raises(ValueError,match='prefix|EOS'):module.qualify(*args,backend,MODEL)
    assert backend.calls==0 and not args[2].exists()
