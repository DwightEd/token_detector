"""H012 artifact-stage tests using complete synthetic 18/719/136 inventories."""
import json
from pathlib import Path

import pytest

from experiments import data, natural_masked_inventory as module


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False), encoding='utf-8')


def field(text, words):
    return dict(status='processed', text_sha256=data.text_hash(text), characters=len(text),
        offset_unit=module.kernel.OFFSET_UNIT, entities=[dict(occurrence_id=i, text=word, type='PERSON',
        start_char=text.index(word), end_char=text.index(word) + len(word)) for i, word in enumerate(words)])


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'write_json', write)
    def build():
        inference = tmp_path / 'inference'; inference.mkdir()
        inputs = inference / 'inputs.jsonl'
        h010 = tmp_path / 'h010'; h010.mkdir()
        samples, inventories = [], []
        for index in range(18):
            source_words = [f'Value{i:04d}' for i in range(40 if index < 17 else 39)]
            response_words = source_words[:8 if index < 10 else 7]
            source, response = ' '.join(source_words), ' '.join(response_words)
            samples.append(dict(id=f'case-{index}', source=source, response=response, ignored_metadata='not read'))
            inventories.append(dict(id=f'case-{index}', source=field(source, source_words), response=field(response, response_words)))
        inputs.write_text(''.join(json.dumps(row) + '\n' for row in samples), encoding='utf-8')
        inventory_path = h010 / 'pilot-inventory.jsonl'
        inventory_path.write_text(''.join(json.dumps(row) + '\n' for row in inventories), encoding='utf-8')
        identity = dict(spacy='3.8.18', model='en_core_web_sm', model_version='3.8.0',
                        environment='MUST_NOT_FOLLOW', model_files_sha256={'MUST_NOT_OPEN_WEIGHTS': 'b' * 64})
        write(h010 / 'pilot-freeze.json', dict(schema='h010-span-pilot-freeze-v1', inputs=str(inputs),
            inputs_sha256=data.file_hash(inputs), cases=18, fields=['source', 'response'],
            read_whitelist=['id', 'source', 'response'], readiness_sha256='a' * 64, model_identity=identity))
        write(h010 / 'synthetic-probe.json', dict(schema='h010-synthetic-offset-engineering-v1',
              semantic_NER_accuracy_assessed=False, checks=dict(exact_offsets=True)))
        summary = dict(schema='h010-span-inventory-summary-v1', status='inventory_complete',
            cases_retained=18, fields_retained=36, unknown_fields=0, processed_fields=36,
            task_gold_opened=False, gpu_calls=0, factuality_scores_computed=0, model_identity=identity,
            model_spans_by_type={'source': {'PERSON': 719}, 'response': {'PERSON': 136}},
            evidence_sha256={'readiness.json': 'a' * 64, **{name: data.file_hash(h010 / name)
                for name in module.PAYLOADS - {'pilot-summary.json'}}})
        write(h010 / 'pilot-summary.json', summary)
        write(h010 / 'pilot-seal.json', dict(schema=module.DEFAULT_PROTOCOL['h010_seal_schema'],
            files_sha256={name: data.file_hash(h010 / name) for name in module.PAYLOADS}))
        for name in ('readiness.json', 'evaluation_only.jsonl', 'model.safetensors', 'engineering-notes.md'):
            (h010 / name).write_text('FORBIDDEN AUXILIARY DATA', encoding='utf-8')
        protocol = tmp_path / 'protocol.json'
        write(protocol, dict(module.DEFAULT_PROTOCOL, original_inputs_sha256=data.file_hash(inputs)))
        output = tmp_path / 'outputs' / 'natural-artifacts'
        return inputs, h010, output, protocol
    return build


def reseal(h010, *, relink=True):
    if relink:
        summary = data.read_json(h010 / 'pilot-summary.json')
        for name in module.PAYLOADS - {'pilot-summary.json'}:
            summary['evidence_sha256'][name] = data.file_hash(h010 / name)
        write(h010 / 'pilot-summary.json', summary)
    write(h010 / 'pilot-seal.json', dict(schema=module.DEFAULT_PROTOCOL['h010_seal_schema'],
        files_sha256={name: data.file_hash(h010 / name) for name in module.PAYLOADS}))


def test_complete_population_freezes_before_compiling_and_never_reads_auxiliary_h010(make_run, monkeypatch):
    args = make_run()
    original_open, compiler = Path.open, module.kernel.prepare_natural_questions
    reads = []
    def guarded(path, *rest, **kwargs):
        if path.is_relative_to(args[1]):
            assert path.name in module.PAYLOADS | {'pilot-seal.json'}
            reads.append(path.name)
        return original_open(path, *rest, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    def compile(samples, inventories):
        assert (args[2] / 'freeze.json').is_file()
        plan = data.read_json(args[2] / 'plan.json')
        assert plan['state'] == 'frozen_before_cloze_compilation'
        assert set(plan['h010_snapshot']['files_sha256']) == module.PAYLOADS
        assert plan['frontend_assets_read'] is False
        return compiler(samples, inventories)
    monkeypatch.setattr(module.kernel, 'prepare_natural_questions', compile)
    summary = module.prepare(*args)
    assert set(reads) == module.PAYLOADS | {'pilot-seal.json'}
    assert summary['counts']['originals'] == 18
    assert summary['counts']['source_occurrences'] == 719 and summary['counts']['response_occurrences'] == 136
    assert summary['qa_admitted'] is False and summary['risk'] is None
    assert summary['method_promoted'] is False and summary['model_calls'] == summary['nlp_calls'] == 0
    artifact = data.read_json(args[2] / 'artifact.json')
    assert len(artifact['proposals']) == 136
    assert all(p['qa_admitted'] is False and p['risk'] is None for p in artifact['proposals'])
    assert [row['id'] for row in artifact['originals']] == [f'case-{i}' for i in range(18)]
    data.verify_seal(args[2])


@pytest.mark.parametrize('damage', ['seal_escape', 'seal_missing', 'oversize'])
def test_entire_named_namespace_is_admitted_before_any_hash_or_source(make_run, monkeypatch, damage):
    args = make_run()
    if damage == 'oversize':
        with (args[1] / 'synthetic-probe.json').open('wb') as stream:
            stream.truncate(module.DEFAULT_PROTOCOL['max_payload_bytes'] + 1)
    else:
        seal = data.read_json(args[1] / 'pilot-seal.json')
        if damage == 'seal_escape': seal['files_sha256']['../forbidden'] = 'f' * 64
        else: del seal['files_sha256']['pilot-inventory.jsonl']
        write(args[1] / 'pilot-seal.json', seal)
    def forbidden(*args, **kwargs): raise AssertionError('Hash/source/compile reached before namespace admission')
    monkeypatch.setattr(data, 'file_hash', forbidden)
    monkeypatch.setattr(module, '_source_identity', forbidden)
    monkeypatch.setattr(module.kernel, 'prepare_natural_questions', forbidden)
    with pytest.raises(ValueError): module.prepare(*args)
    assert not args[2].exists()


@pytest.mark.parametrize('damage', ['order', 'bounds', 'population', 'input_hash'])
def test_resealed_metadata_or_offsets_cannot_change_the_fixed_population(make_run, damage):
    args = make_run()
    if damage == 'input_hash':
        protocol = data.read_json(args[3]); protocol['original_inputs_sha256'] = 'f' * 64; write(args[3], protocol)
    else:
        path = args[1] / 'pilot-inventory.jsonl'
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        if damage == 'order': rows[0], rows[1] = rows[1], rows[0]
        elif damage == 'bounds': rows[0]['response']['entities'][0]['end_char'] += 1
        else: rows[-1]['response']['entities'].pop()
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8'); reseal(args[1])
    with pytest.raises(ValueError): module.prepare(*args)
    assert not args[2].exists()


@pytest.mark.parametrize('mutation', ['h010_reseal', 'input', 'protocol', 'extra_output', 'producer_code'])
def test_last_guard_revokes_completion_and_seal_after_late_change(make_run, monkeypatch, mutation):
    args = make_run()
    verify, source_identity = data.verify_seal, module._source_identity
    changed = []
    def source():
        identity = source_identity()
        if changed and mutation == 'producer_code': identity['changed'] = True
        return identity
    monkeypatch.setattr(module, '_source_identity', source)
    def mutate(root, *rest, **kwargs):
        result = verify(root, *rest, **kwargs)
        if Path(root) == args[2] and not changed:
            if mutation == 'h010_reseal':
                path = args[1] / 'pilot-seal.json'; path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
            elif mutation == 'input':
                args[0].write_text(args[0].read_text(encoding='utf-8') + '\n', encoding='utf-8')
            elif mutation == 'protocol':
                args[3].write_text(args[3].read_text(encoding='utf-8') + '\n', encoding='utf-8')
            elif mutation == 'extra_output':
                write(args[2] / 'injected.json', {})
            changed.append(True)
        return result
    monkeypatch.setattr(data, 'verify_seal', mutate)
    with pytest.raises(ValueError): module.prepare(*args)
    assert changed and (args[2] / 'artifact.json').is_file()
    assert not (args[2] / 'complete.json').exists() and not (args[2] / 'seal.json').exists()


def test_changed_complete_readback_cannot_leave_success_marker(make_run, monkeypatch):
    args = make_run()
    def changed(path, value):
        write(path, {'changed': True} if Path(path).name == 'complete.json' else value)
    monkeypatch.setattr(data, 'write_json', changed)
    with pytest.raises(ValueError, match='artifact differs'): module.prepare(*args)
    assert (args[2] / 'freeze.json').is_file() and not (args[2] / 'complete.json').exists()
    assert not (args[2] / 'seal.json').exists()


@pytest.mark.parametrize('mutation', ['qa', 'zero_risk', 'target_identity'])
def test_artifact_cannot_claim_qa_zero_risk_or_replace_an_occurrence(make_run, monkeypatch, mutation):
    args = make_run()
    compiler = module.kernel.prepare_natural_questions
    def changed(samples, inventories):
        value = compiler(samples, inventories)
        first = value['proposals'][0]
        if mutation == 'qa': first['cloze_artifact']['question'] = 'Forbidden executable instruction'
        elif mutation == 'zero_risk': first['risk'] = 0.
        else: first['target_readout']['occurrence_id'] = 99
        return value
    monkeypatch.setattr(module.kernel, 'prepare_natural_questions', changed)
    with pytest.raises(ValueError, match='unscored'): module.prepare(*args)
    assert (args[2] / 'freeze.json').is_file() and not (args[2] / 'artifact.json').exists()
    assert not (args[2] / 'complete.json').exists() and not (args[2] / 'seal.json').exists()


def test_existing_output_is_preserved_and_protocol_cannot_alias_h010(make_run):
    args = make_run()
    args[2].mkdir(parents=True); marker = args[2] / 'keep'; marker.write_text('preserve', encoding='utf-8')
    with pytest.raises(FileExistsError): module.prepare(*args)
    assert marker.read_text(encoding='utf-8') == 'preserve'
    values = list(args); values[2] = args[2].parent / 'fresh'; values[3] = args[1] / 'protocol.json'
    with pytest.raises(ValueError, match='Independent'): module.prepare(*values)
