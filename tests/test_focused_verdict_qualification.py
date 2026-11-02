"""Matched whole/unit collection on synthetic inputs; no gold or model loading."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import data, focused_verdict_qualification as module
from experiments import verdict_qualification as shared
from experiments.datasets import Sample, read_samples


MODEL=dict(model_id='synthetic-focus',model_files_sha256={'weights':'a'*64})
SOURCE='Original evidence.'
RESPONSE='Repeated. Repeated. Tail 3.14.'


class Backend:
    model_identity=MODEL

    def __init__(self,count=20,check=None):self.count,self.check=count,check;self.calls=[];self.validations=[]
    def count_input_tokens(self,messages):return self.count
    def validate_labels(self,messages,labels):
        self.validations.append(deepcopy(messages))
        return dict(input_tokens=self.count,label_token_ids=dict(zip(labels,[10,11,12])),
            label_decodes=dict(zip(labels,labels)),prefix_alignment_verified=True)
    def score_labels(self,messages,labels):
        if self.check:self.check(messages)
        self.calls.append(deepcopy(messages))
        return dict(self.validate_labels(messages,labels),label_logits=dict(zip(labels,[2.,1.,0.])),
            full_vocab_logsumexp=4.,actual_forward_passes=1,output_tokens=0,seconds=.125)


@pytest.fixture
def prepared(tmp_path,monkeypatch):
    def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    monkeypatch.setattr(data,'write_json',write)
    for mod in (module,shared):
        monkeypatch.setattr(mod,'research_identity',lambda:{'source_sha256':{'fixed':'a'*64}})
        monkeypatch.setattr(mod,'completed_identity',lambda start:start)
    def build(response=RESPONSE):
        panel=tmp_path/'panel';panel.mkdir();inputs=panel/'inputs.jsonl'
        rows=[dict(id='ORIGINAL_METADATA_ONLY_'+str(i),source_id='source-'+str(i),split='test',task='QA',
            prompt='Read '+SOURCE+' Answer completely.',source=SOURCE,response=response,generator='original-generator') for i in range(34)]
        inputs.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
        for name in ('manifest.json','evaluation_only.jsonl','annotations.jsonl'):
            (panel/name).write_text('MUST NOT OPEN',encoding='utf-8')
        protocol=tmp_path/'protocol.json';write(protocol,dict(module.DEFAULT_PROTOCOL,original_inputs_sha256=data.file_hash(inputs)))
        return inputs,protocol,tmp_path/'outputs'/'focused'
    return build


def test_all_original_characters_and_metadata_frozen_before_first_forward(prepared,monkeypatch):
    inputs,protocol,output=prepared()
    original=Path.open
    def guard(path,*a,**k):
        assert path.name not in {'manifest.json','evaluation_only.jsonl','annotations.jsonl'}
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'open',guard)
    def frozen(messages):
        plan=data.read_json(output/'plan.json')
        assert plan['prepared_cases']==136 and len(plan['flat_records'])==136
        assert all((output/name).is_file() for name in ('protocol.json','source_identity.json','model_identity.json','shared_protocol.json'))
        data.verify_seal(output/'prepared_flat',expected={'inputs.jsonl'})
        assert (output/'observations'/'plan.json').is_file()
        payload=json.loads(messages[-1]['content'])
        assert payload['source']==SOURCE
        assert payload['response_before_focus']+payload['focus_text']+payload['response_after_focus']==RESPONSE
        assert 'ORIGINAL_METADATA_ONLY_' not in json.dumps(messages)
        assert not any(row['flat_id'] in json.dumps(messages) for row in plan['flat_records'])
    backend=Backend(check=frozen)
    result=module.qualify(inputs,protocol,output,backend,MODEL)
    assert len(backend.calls)==24 and result['costs']['actual_forward_passes']==24
    assert result['coverage']==dict(original_cases=34,whole_focuses=34,unit_focuses=102,
        flat_cases=136,scored_focuses=136,unknown_focuses=0,unknown_units=0,empty_responses=0)
    assert result['h003_observations_imported'] is False
    flat=list(read_samples(output/'prepared_flat'/'inputs.jsonl'))
    assert all(sample.response==RESPONSE and sample.source==SOURCE and sample.generator=='original-generator' for sample in flat)
    plan=data.read_json(output/'plan.json')
    for record in plan['records']:
        assert ''.join(unit['text'] for unit in record['units'])==RESPONSE
    duplicate_units=[json.loads(m[-1]['content']) for m in backend.calls if json.loads(m[-1]['content'])['focus_text']=='Repeated. ']
    assert {row['response_before_focus'] for row in duplicate_units}=={'','Repeated. '}
    data.verify_seal(output)


def test_single_unit_and_matched_whole_share_exact_shared_cache(prepared):
    inputs,protocol,output=prepared('One whole response.')
    backend=Backend();result=module.qualify(inputs,protocol,output,backend,MODEL)
    assert result['coverage']['flat_cases']==68 and len(backend.calls)==6
    assert result['costs']['logical_requests']==408
    assert result['costs']['exact_request_reuses']==402
    for row in result['records']:
        assert row['whole']['flat_id']!=row['units'][0]['flat_id']
        assert row['whole']['risk']==row['units'][0]['risk']


def test_overlong_focuses_retain_every_unit_with_conservative_unknown(prepared):
    inputs,protocol,output=prepared();backend=Backend(count=4097)
    result=module.qualify(inputs,protocol,output,backend,MODEL)
    assert not backend.calls
    assert result['coverage']['unknown_focuses']==136 and result['coverage']['unknown_units']==102
    assert all(row['whole']['risk']==1.0 and row['whole']['unknown_fallback'] for row in result['records'])
    assert all(unit['risk']==1.0 and unit['unknown_fallback'] for row in result['records'] for unit in row['units'])


def test_wrong_original_input_hash_rejects_before_any_backend_access(prepared):
    inputs,protocol,output=prepared();value=data.read_json(protocol)
    value['original_inputs_sha256']='f'*64;data.write_json(protocol,value)
    backend=Backend()
    with pytest.raises(ValueError,match='input hash'):
        module.qualify(inputs,protocol,output,backend,MODEL)
    assert backend.calls==backend.validations==[] and not output.exists()


def test_no_unit_cap_duplicate_occurrences_and_empty_response_are_retained():
    response='Same. '*300
    samples=[Sample('many','s','test','QA','Read '+SOURCE,SOURCE,response),
             Sample('empty','e','test','QA','Read '+SOURCE,SOURCE,''),
             Sample('whitespace','w','test','QA','Read '+SOURCE,SOURCE,'  \n  ')]
    originals,flat,metadata=module._prepare(samples)
    assert len(originals[0]['units'])==300
    assert len({unit['flat_id'] for unit in originals[0]['units']})==300
    assert ''.join(unit['text'] for unit in originals[0]['units'])==response
    assert originals[1]['units']==[]
    assert len(originals[2]['units'])==1
    assert len(flat)==304 and len(metadata)==304
    assert all(row['response']==response for row in flat if row['source_id']=='s')


def test_protocol_cannot_introduce_a_unit_cap_or_retry_existing_output(prepared):
    inputs,protocol,output=prepared();value=data.read_json(protocol)
    value['max_units']=2;data.write_json(protocol,value)
    backend=Backend()
    with pytest.raises(ValueError,match='Prespecified'):
        module.qualify(inputs,protocol,output,backend,MODEL)
    assert not backend.calls
    output.mkdir(parents=True)
    with pytest.raises(FileExistsError,match='automatic retry'):
        module.qualify(inputs,protocol,output,backend,MODEL)
