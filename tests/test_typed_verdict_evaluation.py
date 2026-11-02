"""Synthetic factorial review gates; neither real gold nor model weights are used."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import data, observation_reuse as reuse, typed_verdict_evaluation as module
from experiments import typed_verdict_qualification as typed, verdict_qualification as shared
from experiments.research import semantic_verdict as verdict


MODEL = dict(model_id='synthetic',model_files_sha256={'weights':'a'*64})
SOURCE = "{'x': None, 'y': False}"


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def seal(root):
    write(root/'seal.json',{str(p.relative_to(root)):data.file_hash(p)
        for p in root.rglob('*') if p.is_file() and p != root/'seal.json'})


class Scorer:
    model_identity = MODEL

    def __init__(self,rejected=False):self.rejected = rejected
    def count_input_tokens(self,messages):
        source=json.loads(messages[-1]['content'])['source']
        return 4097 if self.rejected and 'typed-source-v1' in source else 20
    def validate_labels(self,messages,labels):
        return dict(input_tokens=self.count_input_tokens(messages),label_token_ids=dict(zip(labels,[10,11,12])),
            label_decodes=dict(zip(labels,labels)),prefix_alignment_verified=True)
    def score_labels(self,messages,labels):
        # A fixed synthetic semantic preference, independently of assignment.
        text=messages[0]['content'];assignment=json.loads(text[text.find('{'):text.find('}')+1])
        return dict(self.validate_labels(messages,labels),
            label_logits={label:(3. if assignment[label]=='supported' else 1.) for label in labels},
            full_vocab_logsumexp=5.,output_tokens=0,actual_forward_passes=1,seconds=.125)


@pytest.fixture
def make_run(tmp_path,monkeypatch):
    monkeypatch.setattr(data,'write_json',write)  # Atomic fsync is tested separately.
    modules = {'experiments/typed_verdict_qualification.py':typed,
        'experiments.research/typed_source.py':typed.typed_source,
        'experiments.research/semantic_verdict.py':verdict,
        'experiments/semantic_qualification.py':reuse.backend,
        'experiments/verdict_qualification.py':shared,
        'experiments/observation_reuse.py':reuse}
    source=dict(source_sha256={name:data.file_hash(Path(mod.__file__)) for name,mod in modules.items()})
    for mod in (typed,shared,module):
        monkeypatch.setattr(mod,'research_identity',lambda:deepcopy(source))
        monkeypatch.setattr(mod,'completed_identity',lambda old:old)

    def build(rejected=False):
        panels,all_rows,donors,hashes = [],[],[],[]
        for group,count in enumerate((18,16)):
            panel=tmp_path/f'panel{group}';panel.mkdir();panels.append(panel)
            rows=[dict(id=f'g{group}-{i}',source_id=f's{group}-{i}',task='QA',split='test',
                source=SOURCE,response='X is unreported.',prompt='Read '+SOURCE+' Answer.',generator='original') for i in range(count)]
            all_rows.extend(rows)
            inputs=panel/'inputs.jsonl'
            inputs.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
            hashes.append(data.file_hash(inputs))
            protocol=tmp_path/f'donor-protocol{group}.json'
            write(protocol,dict(shared.DEFAULT_PROTOCOL,expected_cases=count,max_forward_passes=count*6))
            donor=tmp_path/'outputs'/f'old{group}';donors.append(donor)
            shared.qualify(inputs,protocol,donor,Scorer(),MODEL)
            if group == 0:
                released=[dict(id=row['id'],source_id=row['source_id'],cohort='natural',response_sha256=data.text_hash(row['response']),
                    human_mechanism_expectation=dict(mechanism='synthetic',expectation='Review only.'),
                    released_annotation_record=dict(id=row['id'],response_sha256=data.text_hash(row['response']),
                        granularity='token',coverage='complete',spans=[dict(start=0,end=len(row['response']),label=i%2)]))
                    for i,row in enumerate(rows)]
                manifest=dict(schema='qualification-case-panel-v1',counts=dict(total_inputs=18,cohorts={'natural':18}),
                    selection=dict(selected_id_order_sha256=shared._digest([r['id'] for r in rows])),
                    input_rows=[dict(id=r['id'],source_id=r['source_id'],original_task=r['task'],generator=r['generator'],cohort='natural',
                        **{k+'_sha256':data.text_hash(r[k]) for k in ('prompt','source','response')}) for r in rows])
            else:
                released=[dict(id=r['id'],source_id=r['source_id'],pair_id='p'+str(i//2),
                    construction_logical_outcome='supported' if i%2==0 else 'conflict',
                    source_sha256=data.text_hash(r['source']),response_sha256=data.text_hash(r['response'])) for i,r in enumerate(rows)]
                manifest=dict(schema='constructed-observation-controls-v1',counts=dict(inputs=16,pairs=8,exact_word_bag_pairs=6),
                    ids=[r['id'] for r in rows],pairs=[dict(pair_id='p'+str(i),ids=[rows[2*i]['id'],rows[2*i+1]['id']],bag_equality_required=i<6) for i in range(8)])
            (panel/'evaluation_only.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in released),encoding='utf-8')
            manifest.update(inputs_sha256=hashes[-1],evaluation_only_sha256=data.file_hash(panel/'evaluation_only.jsonl'))
            write(panel/'manifest.json',manifest);seal(panel)
        monkeypatch.setattr(typed,'BASE_PROTOCOL',dict(typed.BASE_PROTOCOL,donor_inputs_sha256=hashes))
        combined=tmp_path/'combined';combined.mkdir();inputs=combined/'inputs.jsonl'
        inputs.write_text(''.join(json.dumps(r)+'\n' for r in all_rows),encoding='utf-8')
        protocol=tmp_path/'observe-protocol.json';write(protocol,typed.BASE_PROTOCOL)
        root=tmp_path/'outputs'/'factorial'
        typed.qualify(inputs,protocol,root,Scorer(rejected),MODEL,donors)
        evaluation_protocol=tmp_path/'evaluation-protocol.json';write(evaluation_protocol,module.DEFAULT_PROTOCOL)
        return inputs,root,*panels,tmp_path/'outputs'/'review',evaluation_protocol
    return build


def guard_release(args,monkeypatch,require_frozen=False):
    original=Path.open
    forbidden={p/name for p in args[2:4] for name in ('manifest.json','evaluation_only.jsonl')}
    def guarded(path,*a,**k):
        if path in forbidden:
            assert require_frozen, 'Evaluation data reached before validation'
            assert all((args[4]/name).is_file() for name in ('plan.json','protocol.json','source_identity.json'))
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'open',guarded)


def test_whole_factorial_review_freezes_before_labels_and_preserves_archive(make_run,monkeypatch):
    args=make_run();before={str(p.relative_to(args[1])):p.read_bytes() for p in args[1].rglob('*') if p.is_file()}
    guard_release(args,monkeypatch,require_frozen=True)
    result=module.evaluate(*args)
    assert result['cases']==34 and result['packets']==136 and result['model_calls']==0
    summary=data.read_json(args[4]/'summary.json')
    assert summary['automatic_natural_accuracy_created'] is False
    assert summary['target_label_selection_performed'] is False
    for arm in typed.CONDITIONS:
        assert summary['natural'][arm]['whole_case_denominator']==18
        assert summary['controls'][arm]['denominator']==16
        assert sum(sum(r.values()) for r in summary['controls'][arm]['confusion'].values())==16
        assert len(summary['controls'][arm]['pairs'])==8
    assert len(summary['factorial_effects']['natural']['rows'])==18
    assert before=={str(p.relative_to(args[1])):p.read_bytes() for p in args[1].rglob('*') if p.is_file()}
    data.verify_seal(args[4])


def test_all_rejected_cases_remain_in_control_denominators_and_effects(make_run):
    args=make_run(rejected=True);module.evaluate(*args)
    summary=data.read_json(args[4]/'summary.json')
    assert summary['controls']['B']['unknown_predictions']==16
    assert summary['natural']['D']['unscored_unknown_cases']==18
    assert all(row['differences']['B_minus_A'] is None for row in summary['factorial_effects']['controls']['rows'])


@pytest.mark.parametrize('failure',['missing','cost','helper','donor_origin','root_roster'])
def test_observation_failures_never_release_labels(make_run,monkeypatch,failure):
    args=make_run();root=args[1]
    if failure=='missing':next((root/'D'/'cases').glob('*/case.json')).unlink()
    else:
        path=root/'plan.json' if failure in {'helper','root_roster'} else root/'C'/'complete.json'
        if failure=='donor_origin':path=next((root/'A'/'requests').glob('*/response.json'))
        value=data.read_json(path)
        if failure=='helper':value['source_identity']['source_sha256'][module._HELPERS[0]]='f'*64
        elif failure=='root_roster':value['records'][0]['source_id']='wrong'
        elif failure=='cost':value['costs']['actual_forward_passes']+=1
        else:
            value['observation_origin']['donor']['old_request_key']='f'*64
            write(path.parent/'backend_raw.json',{'value':{k:v for k,v in value.items() if k!='status'}})
        write(path,value)
        if failure=='donor_origin':seal(path.parent);seal(root/'A')
        if failure=='cost':seal(root/'C')
        seal(root)
    guard_release(args,monkeypatch)
    with pytest.raises(ValueError):module.evaluate(*args)
    assert not args[4].exists()


@pytest.mark.parametrize('failure',['seal','metadata_alias'])
def test_unsafe_seal_and_metadata_alias_reject_before_any_hash(make_run,monkeypatch,failure):
    args=make_run()
    if failure == 'seal':
        value=data.read_json(args[1]/'seal.json')
        value['..\\panel0\\evaluation_only.jsonl']='f'*64;write(args[1]/'seal.json',value)
    else:
        path=args[1]/'plan.json';path.unlink()
        try:path.symlink_to(args[2]/'evaluation_only.jsonl')
        except OSError:pytest.skip('Symbolic links unavailable')
    monkeypatch.setattr(data,'file_hash',lambda p:pytest.fail('Unsafe inventory reached file hashing'))
    with pytest.raises(ValueError):module.evaluate(*args)


def test_protocol_cannot_enable_label_selection(make_run,monkeypatch):
    args=make_run();value=dict(module.DEFAULT_PROTOCOL,target_label_selection_allowed=True);write(args[-1],value)
    guard_release(args,monkeypatch)
    with pytest.raises(ValueError,match='Prespecified'):module.evaluate(*args)
