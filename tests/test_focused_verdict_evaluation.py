"""Synthetic whole/unit review gates; no real annotations or models are used."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import data, focused_verdict_evaluation as module
from experiments import focused_verdict_qualification as focused, verdict_qualification as shared
from experiments import semantic_qualification as semantic
from experiments.datasets import Sample
from experiments.research import focused_verdict as focus, semantic_verdict as verdict


MODEL = dict(model_id='synthetic-focus-reader', model_files_sha256={'weights': 'a'*64})
SOURCE = 'Original evidence.'
RESPONSE = 'First. Second.'


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def seal(root):
    write(root/'seal.json', {str(p.relative_to(root)):data.file_hash(p)
        for p in root.rglob('*') if p.is_file() and p != root/'seal.json'})


class Backend:
    model_identity = MODEL

    def __init__(self, rejected=False): self.rejected = rejected
    def count_input_tokens(self, messages): return 4097 if self.rejected else 20
    def validate_labels(self, messages, labels):
        return dict(input_tokens=self.count_input_tokens(messages),
            label_token_ids=dict(zip(labels, [10,11,12])), label_decodes=dict(zip(labels, labels)),
            prefix_alignment_verified=True)
    def score_labels(self, messages, labels):
        text = messages[0]['content']
        assignment = json.loads(text[text.find('{'):text.find('}')+1])
        return dict(self.validate_labels(messages, labels),
            label_logits={label:(3. if assignment[label] == 'supported' else 1.) for label in labels},
            full_vocab_logsumexp=5., output_tokens=0, actual_forward_passes=1, seconds=.125)


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'write_json', write)
    modules = {'experiments/focused_verdict_qualification.py':focused,
        'experiments.research/focused_verdict.py':focus,
        'experiments/verdict_qualification.py':shared,
        'experiments.research/semantic_verdict.py':verdict,
        'experiments/semantic_qualification.py':semantic}
    source = dict(source_sha256={name:data.file_hash(Path(mod.__file__)) for name,mod in modules.items()})
    for mod in (focused, shared, module):
        monkeypatch.setattr(mod, 'research_identity', lambda:deepcopy(source))
        monkeypatch.setattr(mod, 'completed_identity', lambda old:old)

    def build(rejected=False):
        panels, combined_rows = [], []
        for group, count in enumerate((18,16)):
            panel = tmp_path/f'panel{group}'; panel.mkdir(); panels.append(panel)
            rows = [dict(id=f'g{group}-{i}', source_id=f's{group}-{i}', task='QA', split='test',
                source=SOURCE, response=RESPONSE, prompt='Read '+SOURCE+' Answer.', generator='original')
                for i in range(count)]
            combined_rows.extend(rows)
            inputs = panel/'inputs.jsonl'
            inputs.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
            if group == 0:
                labels = []
                for i,row in enumerate(rows):
                    sentence = i >= 12
                    spans = ([dict(start=0,end=7,label=0), dict(start=7,end=len(RESPONSE),label=None if i==17 else 1)]
                        if sentence else [dict(start=0,end=5,label=i%2)])
                    labels.append(dict(id=row['id'], source_id=row['source_id'],
                        cohort='sentence_fixture' if sentence else 'token_fixture',
                        response_sha256=data.text_hash(RESPONSE),
                        human_mechanism_expectation=dict(mechanism='synthetic', expectation='Review only.'),
                        released_annotation_record=dict(id=row['id'],response_sha256=data.text_hash(RESPONSE),
                            granularity='sentence' if sentence else 'token',
                            coverage='annotated_sentences' if sentence else 'complete',spans=spans)))
                manifest = dict(schema='qualification-case-panel-v1',
                    counts=dict(total_inputs=18,cohorts={'token_fixture':12,'sentence_fixture':6}),
                    selection=dict(selected_id_order_sha256=shared._digest([r['id'] for r in rows])),
                    input_rows=[dict(id=r['id'],source_id=r['source_id'],original_task=r['task'],
                        generator=r['generator'],cohort=labels[i]['cohort'],
                        **{k+'_sha256':data.text_hash(r[k]) for k in ('prompt','source','response')})
                        for i,r in enumerate(rows)])
            else:
                labels = [dict(id=r['id'],source_id=r['source_id'],pair_id='p'+str(i//2),
                    construction_logical_outcome='supported' if i%2==0 else ('conflict' if i<14 else 'not_determined'),
                    source_sha256=data.text_hash(SOURCE),response_sha256=data.text_hash(RESPONSE))
                    for i,r in enumerate(rows)]
                manifest = dict(schema='constructed-observation-controls-v1',
                    counts=dict(inputs=16,pairs=8,exact_word_bag_pairs=6),ids=[r['id'] for r in rows],
                    pairs=[dict(pair_id='p'+str(i),ids=[rows[2*i]['id'],rows[2*i+1]['id']],
                        bag_equality_required=i<6) for i in range(8)])
            (panel/'evaluation_only.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels), encoding='utf-8')
            manifest.update(inputs_sha256=data.file_hash(inputs),
                evaluation_only_sha256=data.file_hash(panel/'evaluation_only.jsonl'))
            write(panel/'manifest.json',manifest); seal(panel)
        combined = tmp_path/'combined'; combined.mkdir(); inputs = combined/'inputs.jsonl'
        inputs.write_text(''.join(json.dumps(r)+'\n' for r in combined_rows), encoding='utf-8')
        protocol = tmp_path/'observe-protocol.json'
        write(protocol,dict(focused.DEFAULT_PROTOCOL,original_inputs_sha256=data.file_hash(inputs)))
        root = tmp_path/'outputs'/'focused'
        focused.qualify(inputs,protocol,root,Backend(rejected),MODEL)
        evaluation_protocol = tmp_path/'evaluation-protocol.json'; write(evaluation_protocol,module.DEFAULT_PROTOCOL)
        return inputs,root,*panels,tmp_path/'outputs'/'review',evaluation_protocol
    return build


def guard_release(args, monkeypatch, require_frozen=False):
    original = Path.open
    forbidden = {p/name for p in args[2:4] for name in ('manifest.json','evaluation_only.jsonl')}
    def guarded(path, *a, **k):
        if path in forbidden:
            assert require_frozen, 'Evaluation data reached before observation validation'
            assert all((args[4]/name).is_file() for name in ('plan.json','protocol.json','source_identity.json'))
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'open',guarded)


def test_matched_reader_freezes_before_labels_and_preserves_archive(make_run,monkeypatch):
    args = make_run()
    before = {str(p.relative_to(args[1])):p.read_bytes() for p in args[1].rglob('*') if p.is_file()}
    guard_release(args,monkeypatch,require_frozen=True)
    result = module.evaluate(*args)
    assert result['cases'] == result['packets'] == 34 and result['model_calls'] == 0
    assert len(result['records']) == 34
    packets = [data.read_json(args[4]/row['file']) for row in result['records']]
    assert len({row['id'] for row in packets}) == 34
    assert before == {str(p.relative_to(args[1])):p.read_bytes() for p in args[1].rglob('*') if p.is_file()}
    summary = data.read_json(args[4]/'summary.json')
    assert summary['model_calls'] == 0 and summary['target_label_selection_performed'] is False
    assert summary['natural']['pooled_mixed_granularity_metric_created'] is False
    cohorts = summary['natural']['cohorts']
    token, sentence = cohorts['token_fixture::token'], cohorts['sentence_fixture::sentence']
    assert token['denominator_granularity'] == 'character'
    assert sentence['denominator_granularity'] == 'sentence'
    for mode in ('whole','unit'):
        assert token[mode]['known_positive'] == 30 and token[mode]['known_negative'] == 138
        assert token[mode]['character_auroc'] == .5
        assert sentence[mode]['known_positive'] == 5 and sentence[mode]['known_negative'] == 6
        assert sentence[mode]['missing_label_units'] == 1
        assert summary['controls'][mode]['known_units'] == 16
    assert len(summary['controls']['pairs']) == 8
    data.verify_seal(args[4])


def test_rejected_observations_are_retained_with_conservative_unit_risk(make_run):
    args = make_run(rejected=True)
    result = module.evaluate(*args)
    assert result['cases'] == result['packets'] == 34
    complete = data.read_json(args[1]/'complete.json')
    assert complete['coverage']['unknown_focuses'] == 102
    assert all(r['whole']['risk'] == 1 and all(u['risk'] == 1 for u in r['units']) for r in complete['records'])
    summary = data.read_json(args[4]/'summary.json')
    assert summary['controls']['denominator'] == 16
    assert summary['controls']['whole']['unknown_score_units'] == 16
    assert summary['controls']['unit']['unknown_score_units'] == 16


@pytest.mark.parametrize('failure',['missing','cost','helper','source','root_roster','flat_roster','coverage','input_hash'])
def test_tampering_never_releases_evaluation_material(make_run,monkeypatch,failure):
    args = make_run(); root = args[1]
    if failure == 'missing':
        next((root/'observations'/'cases').glob('*/case.json')).unlink()
    elif failure == 'input_hash':
        args[0].write_text(args[0].read_text(encoding='utf-8')+'\n', encoding='utf-8')
    else:
        path = root/('complete.json' if failure in {'cost','coverage'} else 'plan.json')
        value = data.read_json(path)
        if failure == 'cost': value['costs']['actual_forward_passes'] += 1
        elif failure == 'coverage': value['coverage']['unit_focuses'] -= 1
        elif failure == 'helper': value['source_identity']['source_sha256'][module._HELPERS[0]] = 'f'*64
        elif failure == 'source': value['records'][0]['input_text_sha256']['source'] = 'f'*64
        elif failure == 'root_roster': value['records'][0]['source_id'] = 'wrong'
        else: value['flat_records'][0]['focus_span'] = [0,1]
        write(path,value); seal(root)
    guard_release(args,monkeypatch)
    with pytest.raises(ValueError): module.evaluate(*args)
    assert not args[4].exists()


@pytest.mark.parametrize('failure',['seal','parent_alias','child_alias','child_seal'])
def test_unsafe_inventory_rejects_before_any_hash(make_run,monkeypatch,failure):
    args = make_run(); root = args[1]
    if failure in {'seal','child_seal'}:
        path = root/'seal.json' if failure == 'seal' else root/'observations'/'seal.json'
        value = data.read_json(path); value['..\\panel0\\evaluation_only.jsonl'] = 'f'*64; write(path,value)
    else:
        path = root/'plan.json' if failure == 'parent_alias' else root/'observations'/'plan.json'
        path.unlink()
        try: path.symlink_to(args[2]/'evaluation_only.jsonl')
        except OSError: pytest.skip('Symbolic links unavailable')
    monkeypatch.setattr(data,'file_hash',lambda p:pytest.fail('Unsafe inventory reached file hashing'))
    with pytest.raises(ValueError): module.evaluate(*args)


def test_protocol_cannot_enable_target_selection(make_run,monkeypatch):
    args = make_run(); value = dict(module.DEFAULT_PROTOCOL,target_label_selection_allowed=True); write(args[-1],value)
    guard_release(args,monkeypatch)
    with pytest.raises(ValueError,match='[Pp]respecified'): module.evaluate(*args)


def test_native_character_and_sentence_axes_preserve_missing_truth_and_unknown_risk():
    sample = Sample('s','source','test','QA','Read '+SOURCE,SOURCE,'A. B.')
    readout = dict(whole=dict(risk=.4,unknown_fallback=False),units=[
        dict(start=0,end=3,text='A. ',risk=.2,unknown_fallback=False),
        dict(start=3,end=5,text='B.',risk=1.,unknown_fallback=True)])
    spans = [dict(start=0,end=1,label=1),dict(start=3,end=4,label=None)]
    annotation = dict(granularity='token',coverage='complete',spans=spans)
    released = dict(released_annotation_record=annotation)
    projected = module._natural(sample,readout,released)
    assert projected['native_units'] == spans
    assert projected['labels'] == [1,0,0,None,0]
    assert projected['unit']['risks'] == [.2,.2,.2,1.,1.]
    assert projected['unit']['unknown_mask'] == [False,False,False,True,True]
    assert projected['whole']['risks'] == [.4]*5
    assert projected['denominator_granularity'] == 'character'
    annotation['coverage'] = 'partial'
    assert module._natural(sample,readout,released)['labels'] == [1,None,None,None,None]
    annotation.update(granularity='sentence',coverage='annotated_sentences',
        spans=[dict(start=0,end=3,label=0),dict(start=3,end=5,label=None)])
    projected = module._natural(sample,readout,released)
    assert projected['native_units'] == annotation['spans']
    assert projected['labels'] == [0,None]
    assert projected['unit']['risks'] == [.2,1.]
    assert projected['denominator_granularity'] == 'sentence'
    annotation['coverage'] = 'unalignable'
    assert module._natural(sample,readout,released)['labels'] == [None,None]


def test_metric_ties_single_class_and_unknown_denominators_are_explicit():
    metric = module._metrics([0,1,None],[.5,.5,1.],[False,True,True])
    assert metric['auroc'] == metric['ap'] == .5
    assert (metric['known_positive'],metric['known_negative'],metric['missing_label_units']) == (1,1,1)
    assert metric['unknown_score_units'] == 2 and metric['unknown_score_known_label_units'] == 1
    metric = module._metrics([1,None],[.2,1.],[False,True])
    assert metric['auroc'] is None and metric['ap'] == 1.
    metric = module._metrics([None],[1.],[True])
    assert metric['auroc'] is metric['ap'] is None and metric['known_units'] == 0
