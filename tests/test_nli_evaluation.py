"""Synthetic annotation-release gates; no real gold, assets, or models."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import data, nli_evaluation as module, nli_qualification as observer
from experiments import verdict_qualification as shared


MODEL = dict(model_id=observer.REPO, revision=observer.REVISION,
             model_files_sha256=dict(observer.PINNED_SHA256), runtime=dict(observer.HF_RUNTIME))
SOURCE = 'Synthetic evidence.'


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def seal(root):
    write(root / 'seal.json', {str(path.relative_to(root)): data.file_hash(path)
        for path in root.rglob('*') if path.is_file() and path != root / 'seal.json'})


class Tokenizer:
    def encode(self, text, *, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False and truncation is False
        return [] if text == 'EMPTY-TOKENIZED' else [ord(char) + 10 for char in text]

    def num_special_tokens_to_add(self, pair=False):
        return 3 if pair else 2

    def build_inputs_with_special_tokens(self, first, second):
        return [1, *first, 2, *second, 2]


class Backend:
    model_identity = MODEL

    def prepare_pairs(self, source, hypothesis):
        return observer._prepare_pairs(Tokenizer(), source, hypothesis)

    def classify(self, pairs):
        return dict(logits=[[0., 1., 2.] for _ in pairs], actual_forward_passes=1,
            input_tokens=sum(len(pair['input_ids']) for pair in pairs),
            padded_input_tokens=len(pairs) * max(len(pair['input_ids']) for pair in pairs), seconds=.125)


class TokenizerOnly(Backend):
    def __init__(self, model, identity, admit_resources=False):
        assert admit_resources is False
        self.model_identity, self.model = identity, None

    def verify_frozen_identity(self):
        return None

    def classify(self, pairs):
        pytest.fail('Evaluator must never classify or load a real model')


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'write_json', write)
    repo = Path(module.__file__).resolve().parents[1]
    source = dict(source_sha256={name: data.file_hash(repo / name.replace('experiments.research/', 'experiments/research/'))
                               for name in module._HELPERS})
    for mod in (observer, module):
        monkeypatch.setattr(mod, 'research_identity', lambda: deepcopy(source))
        monkeypatch.setattr(mod, 'completed_identity', lambda old: old)
    monkeypatch.setattr(module, 'HFEntailment', TokenizerOnly)

    def build(unknown=False):
        evidence = 'EMPTY-TOKENIZED' if unknown else SOURCE
        panels, combined = [], []
        for group, count in enumerate((18, 16)):
            panel = tmp_path / f'panel{group}'
            panel.mkdir()
            panels.append(panel)
            rows = []
            for index in range(count):
                response = 'First. Second. Third.' + (' Fourth.' if group == 0 and index < 2 else '')
                rows.append(dict(id=f'g{group}-{index}', source_id=f's{group}-{index}', split='test',
                    task='QA', source=evidence, response=response, prompt='Read ' + evidence + ' Answer.', generator='original'))
            combined.extend(rows)
            inputs = panel / 'inputs.jsonl'
            inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            if group == 0:
                labels = []
                for index, row in enumerate(rows):
                    sentence = index >= 12
                    spans = ([dict(start=0, end=7, label=0),
                              dict(start=7, end=len(row['response']), label=None if index == 17 else 1)]
                             if sentence else [dict(start=0, end=5, label=index % 2)])
                    labels.append(dict(id=row['id'], source_id=row['source_id'],
                        cohort='sentence_fixture' if sentence else 'token_fixture',
                        response_sha256=data.text_hash(row['response']),
                        human_mechanism_expectation=dict(mechanism='synthetic', expectation='Review only.'),
                        released_annotation_record=dict(id=row['id'], response_sha256=data.text_hash(row['response']),
                            granularity='sentence' if sentence else 'token',
                            coverage='annotated_sentences' if sentence else 'complete', spans=spans)))
                manifest = dict(schema='qualification-case-panel-v1',
                    counts=dict(total_inputs=18, cohorts={'token_fixture': 12, 'sentence_fixture': 6}),
                    selection=dict(selected_id_order_sha256=shared._digest([row['id'] for row in rows])),
                    input_rows=[dict(id=row['id'], source_id=row['source_id'], original_task=row['task'],
                        generator=row['generator'], cohort=labels[index]['cohort'],
                        **{name + '_sha256': data.text_hash(row[name]) for name in ('prompt', 'source', 'response')})
                        for index, row in enumerate(rows)])
            else:
                labels = [dict(id=row['id'], source_id=row['source_id'], pair_id='p' + str(index // 2),
                    construction_logical_outcome='supported' if index % 2 == 0 else
                        ('conflict' if index < 14 else 'not_determined'),
                    source_sha256=data.text_hash(evidence), response_sha256=data.text_hash(row['response']))
                    for index, row in enumerate(rows)]
                manifest = dict(schema='constructed-observation-controls-v1',
                    counts=dict(inputs=16, pairs=8, exact_word_bag_pairs=6), ids=[row['id'] for row in rows],
                    pairs=[dict(pair_id='p' + str(index), ids=[rows[2 * index]['id'], rows[2 * index + 1]['id']],
                                bag_equality_required=index < 6) for index in range(8)])
            labels_path = panel / 'evaluation_only.jsonl'
            labels_path.write_text(''.join(json.dumps(row) + '\n' for row in labels), encoding='utf-8')
            manifest.update(inputs_sha256=data.file_hash(inputs), evaluation_only_sha256=data.file_hash(labels_path))
            write(panel / 'manifest.json', manifest)
            seal(panel)
        combined_root = tmp_path / 'combined'
        combined_root.mkdir()
        inputs = combined_root / 'inputs.jsonl'
        inputs.write_text(''.join(json.dumps(row) + '\n' for row in combined), encoding='utf-8')
        protocol = tmp_path / 'observe-protocol.json'
        write(protocol, dict(observer.DEFAULT_PROTOCOL, original_inputs_sha256=data.file_hash(inputs)))
        root = tmp_path / 'outputs' / 'nli'
        observer.qualify(inputs, protocol, root, Backend(), MODEL)
        model_path = tmp_path / 'synthetic-model-path'
        model_path.mkdir()
        evaluation_protocol = tmp_path / 'evaluation-protocol.json'
        write(evaluation_protocol, module.DEFAULT_PROTOCOL)
        return inputs, root, *panels, model_path, tmp_path / 'outputs' / 'review', evaluation_protocol
    return build


def guard_release(args, monkeypatch, allow=False, on_release=None):
    original = Path.open
    forbidden = {panel / name for panel in args[2:4] for name in ('manifest.json', 'evaluation_only.jsonl')}
    def guarded(path, *rest, **kwargs):
        if path in forbidden:
            assert allow, 'Evaluation material accessed before archive validation'
            assert all((args[5] / name).is_file() for name in ('plan.json', 'protocol.json', 'source_identity.json'))
            if on_release:
                on_release()
        return original(path, *rest, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)


def reseal_observation(root):
    for child in (root / 'batches').iterdir():
        seal(child)
    freeze = data.read_json(root / 'freeze.json')
    write(root / 'freeze.json', {name: data.file_hash(root / name) for name in freeze})
    seal(root)


def test_valid_review_releases_only_after_freezing_and_preserves_every_archive_byte(make_run, monkeypatch):
    args = make_run()
    before = {str(path.relative_to(args[1])): path.read_bytes() for path in args[1].rglob('*') if path.is_file()}
    guard_release(args, monkeypatch, allow=True)
    result = module.evaluate(*args)
    assert result['cases'] == result['packets'] == 34 and result['model_calls'] == 0
    assert result['tokenizer_only_validation'] is True
    assert len(result['records']) == 34
    packets = [data.read_json(args[5] / record['file']) for record in result['records']]
    assert len({packet['id'] for packet in packets}) == 34
    assert before == {str(path.relative_to(args[1])): path.read_bytes() for path in args[1].rglob('*') if path.is_file()}
    summary = data.read_json(args[5] / 'summary.json')
    assert summary['model_calls'] == 0 and summary['target_label_selection_performed'] is False
    assert summary['natural']['pooled_mixed_granularity_metric_created'] is False
    cohorts = summary['natural']['cohorts']
    token, sentence = cohorts['token_fixture::token'], cohorts['sentence_fixture::sentence']
    assert token['denominator_granularity'] == 'character' and sentence['denominator_granularity'] == 'sentence'
    for cohort in (token, sentence):
        assert 'whole' not in cohort and 'document_max_unit_broadcast' in cohort and 'unit' in cohort
    for mode in ('document_max_unit_broadcast', 'unit'):
        assert token[mode]['known_positive'] == 30 and token[mode]['character_auroc'] == .5
        assert sentence[mode]['known_positive'] == 5 and sentence[mode]['known_negative'] == 6
        assert sentence[mode]['missing_label_units'] == 1
    assert len(summary['controls']['pairs']) == 8
    assert 'whole' not in summary['controls'] and summary['controls']['unit']['known_units'] == 16
    assert all(pair['positive_minus_supported_risk'] == {'unit': 0.0}
               for pair in summary['controls']['pairs'])
    for packet in packets[18:]:
        assert packet['construction_risks'] == {'unit': max(unit['risk'] for unit in packet['readouts']['units'])}
    assert all('whole' not in packet['native_evaluation'] and
               'document_max_unit_broadcast' in packet['native_evaluation'] for packet in packets[:18])
    data.verify_seal(args[5])


def test_unknown_unit_risks_keep_all_34_cases_and_104_units(make_run):
    args = make_run(unknown=True)
    result = module.evaluate(*args)
    assert result['cases'] == result['packets'] == 34
    complete = data.read_json(args[1] / 'complete.json')
    assert complete['coverage']['unknown_units'] == 104
    assert all(unit['risk'] == 1 and unit['unknown_fallback'] for row in complete['records'] for unit in row['units'])
    assert data.read_json(args[5] / 'summary.json')['controls']['denominator'] == 16


@pytest.mark.parametrize('failure', ['cost', 'raw', 'raw_cost', 'readout', 'plan', 'helper', 'ids', 'input_hash'])
def test_resealed_tampering_never_releases_annotation_material(make_run, monkeypatch, failure):
    args = make_run()
    root = args[1]
    if failure == 'input_hash':
        args[0].write_text(args[0].read_text(encoding='utf-8') + '\n', encoding='utf-8')
    elif failure in {'raw', 'raw_cost'}:
        path = next((root / 'batches').glob('*/backend_raw.json'))
        raw = data.read_json(path)
        if failure == 'raw': raw['value']['logits'][0][0] += 4
        else: raw['value']['input_tokens'] += 1
        write(path, raw)
    else:
        path = root / ('complete.json' if failure in {'cost', 'readout'} else 'plan.json')
        value = data.read_json(path)
        if failure == 'cost': value['costs']['actual_forward_passes'] += 1
        elif failure == 'readout': value['records'][0]['units'][0]['risk'] = .01
        elif failure == 'plan': value['records'][0]['units'][0]['text'] = 'Wrong partition.'
        elif failure == 'helper': value['source_identity']['source_sha256'][module._HELPERS[0]] = 'f' * 64
        else: value['records'][0]['id'] = 'not-original'
        write(path, value)
    if failure != 'input_hash': reseal_observation(root)
    guard_release(args, monkeypatch)
    with pytest.raises(ValueError): module.evaluate(*args)
    assert not args[5].exists()


@pytest.mark.parametrize('failure', ['root_seal', 'batch_seal', 'root_alias', 'batch_alias'])
def test_unsafe_nested_inventory_fails_before_hashing_or_source_identity(make_run, monkeypatch, failure):
    args = make_run()
    root = args[1]
    batch = next((root / 'batches').iterdir())
    if failure.endswith('seal'):
        path = (root if failure == 'root_seal' else batch) / 'seal.json'
        value = data.read_json(path)
        value['..\\panel0\\evaluation_only.jsonl'] = 'f' * 64
        write(path, value)
    else:
        path = root / 'plan.json' if failure == 'root_alias' else batch / 'request.json'
        path.unlink()
        try: path.symlink_to(args[2] / 'evaluation_only.jsonl')
        except OSError: pytest.skip('Symbolic links unavailable')
    monkeypatch.setattr(data, 'file_hash', lambda path: pytest.fail('Unsafe inventory reached hashing'))
    monkeypatch.setattr(module, 'research_identity', lambda: pytest.fail('Unsafe inventory reached source identity'))
    guard_release(args, monkeypatch)
    with pytest.raises(ValueError): module.evaluate(*args)


def test_protocol_cannot_enable_target_label_selection(make_run, monkeypatch):
    args = make_run()
    write(args[-1], dict(module.DEFAULT_PROTOCOL, target_label_selection_allowed=True))
    guard_release(args, monkeypatch)
    with pytest.raises(ValueError, match='[Pp]respecified'): module.evaluate(*args)
    assert not args[5].exists()


def test_resealed_observation_root_during_release_is_rejected(make_run, monkeypatch):
    args = make_run()
    changed = []
    def replace_seal_bytes():
        if not changed:
            path = args[1] / 'seal.json'
            path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
            changed.append(True)
    guard_release(args, monkeypatch, allow=True, on_release=replace_seal_bytes)
    with pytest.raises(ValueError): module.evaluate(*args)
    assert changed and not (args[5] / 'complete.json').exists()
