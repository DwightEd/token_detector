"""Independent fake48 construction/release checks; no production truth is read."""
from copy import deepcopy
import importlib
from itertools import product
import json
from pathlib import Path

import pytest

from experiments import data, designed_focus_evaluation as module
from experiments import designed_focus_qualification as qualifier, verdict_qualification as shared
from experiments.datasets import Sample, read_samples
from experiments.research import focused_verdict as focus


MODEL = dict(model_id='synthetic-designed-reader', model_files_sha256={'weights': 'a' * 64})


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def seal(root):
    write(root / 'seal.json', {str(path.relative_to(root)): data.file_hash(path)
        for path in root.rglob('*') if path.is_file() and path != root / 'seal.json'})


def fake48():
    """Self-contained facts and labels; never imports the production generator."""
    rows, records, truth = [], [], []
    fillers = [f'The {name} Store held a blue bowl.' for name in
               ('Ash', 'Birch', 'Cedar', 'Dogwood', 'Elm', 'Fir', 'Gum', 'Holly', 'Ivy', 'Juniper', 'Kauri', 'Larch')]
    for family, vocabulary, source_direction, response_direction, burden in product(
            ('winner', 'sole_signer'), ('lex1', 'lex2'), (0, 1), (0, 1), (0, 4, 12)):
        people = ('Ava Lorn', 'Bex Pell') if vocabulary == 'lex1' else ('Cara Wynn', 'Dax Ford')
        events = ('Amber Cup', 'Cobalt Cup') if family == 'winner' else ('Amber Pact', 'Cobalt Pact')
        source_people = people if source_direction == 0 else people[::-1]
        answer_people = people if response_direction == 0 else people[::-1]
        noun = 'winner' if family == 'winner' else 'signer'
        source = ' '.join([f'{person} was the sole {noun} of the {event}.'
                           for person, event in zip(source_people, events)] + fillers)
        a, b = answer_people
        x, y = events
        target = f'{a} was the sole {noun} of the {x}, and {b} was the sole {noun} of the {y}. '
        response = target + ' '.join(fillers[:burden])
        source_id = f'{family}-{vocabulary}-s{source_direction}'
        sample_id = f'{source_id}-r{response_direction}-b{burden}'
        prompt = 'Read ' + source + ' Answer factually.'
        row = dict(id=sample_id, source_id=source_id, task='QA', split='test',
                   source=source, response=response, prompt=prompt, generator='synthetic-only')
        rows.append(row)
        record = dict(id=sample_id, source_id=source_id,
            factor_cell=dict(relation_family=family, vocabulary=vocabulary, source_direction=source_direction,
                             response_direction=response_direction, distractor_count=burden),
            target_span=[0, len(target)], target_text=target, whole_span=[0, len(response)],
            input_text_sha256={name: data.text_hash(row[name]) for name in ('prompt', 'source', 'response')})
        records.append(record)
        outcome = 'supported' if source_direction == response_direction else 'contradicted'
        truth.append(dict(id=sample_id, source_id=source_id, target_outcome=outcome, whole_outcome=outcome,
            source_sha256=data.text_hash(source), response_sha256=data.text_hash(response), target_span=record['target_span'],
            filler_outcomes=['supported'] * burden, target_benchmark_labels_used=False,
            construction_rule='entity_event_bijections_equal_iff_source_direction_equals_response_direction'))
    return rows, records, truth


class Backend:
    model_identity = MODEL

    def __init__(self, rejected=False): self.rejected = rejected
    def count_input_tokens(self, messages): return 4097 if self.rejected else 20
    def validate_labels(self, messages, labels):
        return dict(input_tokens=self.count_input_tokens(messages), label_token_ids=dict(zip(labels, [10, 11, 12])),
            label_decodes=dict(zip(labels, labels)), prefix_alignment_verified=True)
    def score_labels(self, messages, labels):
        assert not self.rejected
        text = messages[0]['content']
        assignment = json.loads(text[text.find('{'):text.find('}') + 1])
        return dict(self.validate_labels(messages, labels),
            label_logits={label: 3. if assignment[label] == 'supported' else 1. for label in labels},
            full_vocab_logsumexp=5., output_tokens=0, actual_forward_passes=1, seconds=.125)


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'write_json', write)
    # Module locations, not test/repository paths, also work for installed-wheel checks.
    source = dict(source_sha256={name: data.file_hash(Path(importlib.import_module(name[:-3].replace('/', '.')).__file__))
                               for name in module._HELPERS})
    for mod in (qualifier, shared, module):
        monkeypatch.setattr(mod, 'research_identity', lambda: deepcopy(source))
        monkeypatch.setattr(mod, 'completed_identity', lambda old: old)

    def build(rejected=False):
        rows, records, labels = fake48()
        inference = tmp_path / 'inference'; inference.mkdir()
        inputs = inference / 'inputs.jsonl'
        inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        design = inference / 'plan.json'
        write(design, dict(schema='h007-burden-controls-plan-v1', records=records))
        construction = tmp_path / 'construction'; construction.mkdir()
        truth = construction / 'construction_truth.jsonl'
        truth.write_text(''.join(json.dumps(row) + '\n' for row in labels), encoding='utf-8')
        seal(construction)
        protocol = tmp_path / 'observe-protocol.json'
        write(protocol, dict(qualifier.DEFAULT_PROTOCOL, original_inputs_sha256=data.file_hash(inputs)))
        observations = tmp_path / 'outputs' / 'observe'
        qualifier.qualify(inputs, protocol, observations, Backend(rejected), MODEL)
        evaluation_protocol = tmp_path / 'evaluation-protocol.json'
        write(evaluation_protocol, module.DEFAULT_PROTOCOL)
        return inputs, design, truth, observations, tmp_path / 'outputs' / 'evaluate', evaluation_protocol
    return build


def guard_truth(args, monkeypatch, allow=False):
    original = Path.open
    truth_tree, output = args[2].parent, args[4]
    accesses = []
    def guarded(path, *rest, **kwargs):
        if path.is_relative_to(truth_tree):
            accesses.append(path.name)
            assert allow, 'Construction tree reached before archive/design validation'
            assert (output / 'plan.json').is_file() and (output / 'freeze.json').is_file()
            assert data.read_json(output / 'plan.json')['state'] == 'frozen_before_construction_truth'
        return original(path, *rest, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    return accesses


def test_complete_fake48_release_occurs_after_freeze_and_preserves_archive(make_run, monkeypatch):
    args = make_run()
    before = {str(p.relative_to(args[3])): p.read_bytes() for p in args[3].rglob('*') if p.is_file()}
    accesses = guard_truth(args, monkeypatch, allow=True)
    summary = module.evaluate(*args)
    assert accesses and summary['cases'] == 48 and summary['paired_cells'] == 24
    assert summary['model_calls'] == 0 and summary['primary_interaction'] == pytest.approx(0.)
    assert len(data.read_json(args[4] / 'cases.json')) == 48
    assert before == {str(p.relative_to(args[3])): p.read_bytes() for p in args[3].rglob('*') if p.is_file()}
    assert summary['target_benchmark_labels_accessed'] is False and summary['method_promoted'] is False
    data.verify_seal(args[4])


@pytest.mark.parametrize('field', ['cost', 'readout'])
def test_resealed_completion_tampering_is_rejected_before_any_truth_read(make_run, monkeypatch, field):
    args = make_run()
    path = args[3] / 'complete.json'
    value = data.read_json(path)
    if field == 'cost': value['costs']['actual_forward_passes'] += 1
    else: value['records'][0]['units'][0]['risk'] = .12345
    write(path, value); seal(args[3])
    accesses = guard_truth(args, monkeypatch)
    with pytest.raises(ValueError): module.evaluate(*args)
    assert accesses == [] and not args[4].exists()


def test_all_unknown48_cases_remain_in_every_burden_denominator(make_run):
    args = make_run(rejected=True)
    summary = module.evaluate(*args)
    assert summary['cases'] == 48 and summary['coverage']['unknown_focuses'] == 96
    packets = data.read_json(args[4] / 'cases.json')
    assert len(packets) == 48 and all(all(score['risk'] == 1. and score['unknown_fallback']
        for score in packet['risks'].values()) for packet in packets)
    for mode in module.DEFAULT_PROTOCOL['modes']:
        for burden in ('0', '4', '12'):
            metric = summary['by_mode_and_burden'][mode][burden]
            assert metric['known_units'] == metric['unknown_score_units'] == 16
            assert metric['paired_cells'] == metric['tied_pairs'] == 8


def test_primary_interaction_uses_paired_risk_changes_and_has_the_prespecified_sign():
    _, design, truth = fake48()
    table = {'whole': {0: (.1, .8), 4: (.2, .7), 12: (.4, .5)},
             'target_first_unit': {0: (.2, .9), 4: (.1, .8), 12: (.1, .6)}}
    readouts = []
    for row, label in zip(design, truth, strict=True):
        positive = int(label['target_outcome'] == 'contradicted')
        burden = row['factor_cell']['distractor_count']
        scores = {mode: dict(risk=table[mode][burden][positive], unknown_fallback=False)
                  for mode in table}
        readouts.append(dict(id=row['id'], whole=scores['whole'], units=[scores['target_first_unit']]))
    summary, packets = module._analysis(design, readouts, truth)
    assert len(packets) == 48 and summary['paired_cells'] == 24
    assert summary['primary_interaction'] == pytest.approx(.4)
    for readout in readouts:
        readout['whole'], readout['units'][0] = readout['units'][0], readout['whole']
    assert module._analysis(design, readouts, truth)[0]['primary_interaction'] == pytest.approx(-.4)


def test_wrong_factor_cell_fails_before_truth_release(make_run, monkeypatch):
    args = make_run()
    design = data.read_json(args[1])
    design['records'][0]['factor_cell']['distractor_count'] = 4
    write(args[1], design)
    accesses = guard_truth(args, monkeypatch)
    with pytest.raises(ValueError, match='[Ff]actor|[Dd]uplicate'): module.evaluate(*args)
    assert accesses == [] and not args[4].exists()


@pytest.mark.parametrize('mismatch', ['source_join', 'logical_outcome'])
def test_analysis_rejects_wrong_label_joins_or_generator_rule(mismatch):
    _, design, truth = fake48()
    readouts = [dict(id=row['id'], whole=dict(risk=.3, unknown_fallback=False),
                    units=[dict(risk=.3, unknown_fallback=False)]) for row in design]
    if mismatch == 'source_join': truth[0]['source_sha256'] = 'f' * 64
    else: truth[0].update(target_outcome='contradicted', whole_outcome='contradicted')
    with pytest.raises(ValueError, match='join|rule'): module._analysis(design, readouts, truth)


@pytest.mark.parametrize('aliased', ['protocol', 'design'])
def test_design_or_protocol_inside_truth_tree_is_rejected_before_open(make_run, monkeypatch, aliased):
    args = list(make_run())
    if aliased == 'protocol':
        path = args[2].parent / 'evaluation-protocol.json'
        write(path, module.DEFAULT_PROTOCOL); args[5] = path
    else:
        path = args[2].parent / 'plan.json'
        write(path, data.read_json(args[1])); args[1] = path
    accesses = guard_truth(args, monkeypatch)
    with pytest.raises(ValueError): module.evaluate(*args)
    assert accesses == [] and not args[4].exists()


def test_changed_observation_after_final_evaluator_seal_revokes_completion(make_run, monkeypatch):
    args = make_run()
    original_seal = shared._seal
    changed = []
    def seal_then_change(root, names):
        original_seal(root, names)
        if Path(root) == args[4]:
            path = args[3] / 'seal.json'
            path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
            changed.append(True)
    monkeypatch.setattr(shared, '_seal', seal_then_change)
    with pytest.raises(ValueError): module.evaluate(*args)
    assert changed and (args[4] / 'plan.json').is_file() and (args[4] / 'cases.json').is_file()
    assert not (args[4] / 'complete.json').exists() and not (args[4] / 'seal.json').exists()
