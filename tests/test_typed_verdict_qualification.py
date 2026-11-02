"""H004 factor isolation and request reuse, using synthetic originals only."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from experiments import data, typed_verdict_qualification as module, verdict_qualification as shared
from experiments.observation_reuse import equivalence_key
from experiments.research import semantic_verdict as verdict

IDENTITY=dict(model_id='synthetic',model_files_sha256={'weights':'a'*64})
SOURCE="{'reservations': None, 'other': False}"

class Backend:
    model_identity=IDENTITY
    def __init__(self):self.calls=[]
    def count_input_tokens(self,messages):return 20
    def validate_labels(self,messages,labels):
        return dict(input_tokens=20,label_token_ids=dict(zip(labels,[10,11,12])),
                    label_decodes=dict(zip(labels,labels)),prefix_alignment_verified=True)
    def score_labels(self,messages,labels):
        self.calls.append(deepcopy(messages))
        return dict(self.validate_labels(messages,labels),label_logits=dict(zip(labels,[2.,1.,0.])),
                    full_vocab_logsumexp=4.,actual_forward_passes=1,output_tokens=0,seconds=.25)

def execution(messages):
    return dict(messages=messages,labels=list(verdict.LABELS),model_identity=IDENTITY,
       scoring=dict(use_cache=False,logits_to_keep=1,full_prefill=True,output_tokens=0),
       admission=dict(max_input_tokens=4096))

def test_factorial_A_is_byte_identical_B_preserves_full_typed_source_C_only_adds_rule():
    assignment=verdict.label_assignments()[0]
    old=verdict.build_request(SOURCE,'No reservations.','Question context',assignment)
    a=module.build_request(SOURCE,'No reservations.','Question context',assignment,'A')
    assert a==old
    b=module.build_request(SOURCE,'No reservations.','Question context',assignment,'B')
    c=module.build_request(SOURCE,'No reservations.','Question context',assignment,'C')
    d=module.build_request(SOURCE,'No reservations.','Question context',assignment,'D')
    assert b['messages'][0]==a['messages'][0]
    assert c['messages'][1]==a['messages'][1]
    assert d['messages'][1]==b['messages'][1]
    assert c['messages'][0]['content']==a['messages'][0]['content']+module.NULL_RULE
    payload=json.loads(b['messages'][1]['content'])
    assert payload['response']=='No reservations.' and payload['pre_generation_context']=='Question context'
    assert '"type":"null"' in payload['source'] and '"type":"boolean"' in payload['source']

def test_unparseable_B_equals_A_and_D_equals_C_exactly():
    for assignment in verdict.label_assignments():
        assert module.build_request('Raw prose.','Answer','Context',assignment,'A')==module.build_request('Raw prose.','Answer','Context',assignment,'B')
        assert module.build_request('Raw prose.','Answer','Context',assignment,'C')==module.build_request('Raw prose.','Answer','Context',assignment,'D')

def test_reuse_never_calls_backend_or_changes_donor_response():
    messages=verdict.build_request('Evidence.','Answer.','Context',verdict.label_assignments()[0])['messages']
    backend=Backend();raw=backend.score_labels(messages,verdict.LABELS);backend.calls=[]
    old=dict(raw,status='scored');donors={equivalence_key(execution(messages)):dict(response=old,provenance={'old':'frozen'})}
    original=deepcopy(donors)
    scorer=module.ReusingScorer(backend,IDENTITY,donors)
    assert scorer.count_input_tokens(messages)==20
    scorer.validate_labels(messages,verdict.LABELS)
    value=scorer.score_labels(messages,verdict.LABELS)
    assert not backend.calls and donors==original
    assert value['observation_origin']['mode']=='donor_reuse'
    assert value['label_logits']==raw['label_logits']
    assert value['actual_forward_passes']==1  # historical cost, not a new call

def test_new_equivalent_request_runs_once_and_records_in_run_reuse():
    backend=Backend();scorer=module.ReusingScorer(backend,IDENTITY,{})
    messages=[dict(role='user',content='new')]
    a=scorer.score_labels(messages,verdict.LABELS);b=scorer.score_labels(messages,verdict.LABELS)
    assert len(backend.calls)==1
    assert a['observation_origin']['mode']=='observed'
    assert b['observation_origin']['mode']=='in_run_reuse'
    assert a['label_logits']==b['label_logits']

def test_backend_identity_mismatch_rejected_before_any_tokenizer_access():
    with pytest.raises(ValueError,match='identities differ'):
        module.ReusingScorer(Backend(),dict(IDENTITY,model_id='other'),{})

@pytest.fixture
def synthetic(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'research_identity',lambda:{'source':'fixed'})
    monkeypatch.setattr(module,'completed_identity',lambda x:x)
    monkeypatch.setattr(shared,'research_identity',lambda:{'source':'fixed'})
    monkeypatch.setattr(shared,'completed_identity',lambda x:x)
    parents=[];originals=[]
    for index,count in enumerate([18,16]):
        root=tmp_path/f'donor{index}';root.mkdir();source=root/'inputs.jsonl'
        rows=[dict(id=f'original{index}_{i}',source_id=f's{index}_{i}',task='QA',split='test',
              prompt='Read '+SOURCE+' then answer.',source=SOURCE,response='No reservations.',generator='original') for i in range(count)]
        originals+=rows
        source.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        data.write_json(root/'plan.json',dict(inputs_path=str(source),inputs_sha256=module.BASE_PROTOCOL['donor_inputs_sha256'][index]))
        data.write_json(root/'seal.json',{'plan.json':data.file_hash(root/'plan.json')})
        parents.append(root)
    inputdir=tmp_path/'combined';inputdir.mkdir();inputs=inputdir/'inputs.jsonl'
    inputs.write_text(''.join(json.dumps(r)+'\n' for r in originals),encoding='utf-8')
    for name in ['evaluation_only.jsonl','manifest.json']:(inputdir/name).write_text('MUST NOT OPEN',encoding='utf-8')
    protocol=tmp_path/'protocol.json';data.write_json(protocol,module.BASE_PROTOCOL)
    old_context=shared._context({k:originals[0][k] for k in ['source','response','prompt']})[0]
    fake=Backend();donors={}
    for assignment in verdict.label_assignments():
        messages=verdict.build_request(SOURCE,'No reservations.',old_context,assignment)['messages']
        donors[equivalence_key(execution(messages))]=dict(response=dict(fake.score_labels(messages,verdict.LABELS),status='scored'),provenance={'old':'frozen'})
    monkeypatch.setattr(module,'load_donor',lambda root,identity,expected:deepcopy(donors))
    return inputs,protocol,tmp_path/'outputs'/'h004',parents

def test_all34_originals_four_arms_freeze_and_reuse_without_gold(synthetic,monkeypatch):
    inputs,protocol,output,parents=synthetic
    original=Path.open
    def guard(path,*a,**k):
        assert path.name not in {'evaluation_only.jsonl','manifest.json'}
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'open',guard)
    backend=Backend();result=module.qualify(inputs,protocol,output,backend,IDENTITY,parents)
    assert result['cases']==34 and result['costs']['logical_requests']==816
    assert result['costs']['new_forward_passes']==len(backend.calls)==18
    assert result['costs']['borrowed_unique_observations']==6
    assert result['evaluation_material_accessed'] is False
    for arm in module.CONDITIONS:
        child=data.read_json(output/arm/'complete.json')
        assert len(child['records'])==34 and child['coverage']['scored_cases']==34
    data.verify_seal(output)

def test_same_messages_cannot_replace_missing_original_cases(synthetic):
    inputs,protocol,output,parents=synthetic
    rows=[json.loads(line) for line in inputs.read_text(encoding='utf-8').splitlines()]
    rows[-1]=dict(rows[0],id='renamed_unique',source_id='renamed_source')
    inputs.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    backend=Backend()
    with pytest.raises(ValueError,match='exact ordered18'):
        module.qualify(inputs,protocol,output,backend,IDENTITY,parents)
    assert not backend.calls and not output.exists()

def test_protocol_budget_cannot_be_tuned(synthetic):
    inputs,protocol,output,parents=synthetic
    altered=dict(module.BASE_PROTOCOL,max_input_tokens=4097)
    protocol.write_text(json.dumps(altered),encoding='utf-8')
    with pytest.raises(ValueError,match='prespecified'):
        module.qualify(inputs,protocol,output,Backend(),IDENTITY,parents)
