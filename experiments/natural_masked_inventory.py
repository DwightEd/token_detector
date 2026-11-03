"""Freeze and retain H012 string artifacts from the exact sealed H010 inventory.

Only four named H010 payloads and its custom pilot seal are read. Adjacent
engineering files, environment/model paths, gold and predictions are not opened.
"""
from collections import Counter
import json
from pathlib import Path
import sys

from . import data, verdict_evaluation as reader
from .research import natural_masked_questions as kernel
from .verdict_evaluation import _equal, _member, _sha

PAYLOADS = {'pilot-freeze.json', 'synthetic-probe.json', 'pilot-inventory.jsonl', 'pilot-summary.json'}
_FROZEN = {'plan.json', 'protocol.json', 'source_identity.json', 'freeze.json'}
_MAX_PAYLOAD = 16 * 1024**2
_MAX_SEAL = 64 * 1024
DEFAULT_PROTOCOL = dict(schema='natural-masked-inventory-v1', expected_original_cases=18,
    expected_source_occurrences=719, expected_response_occurrences=136, original_inputs_sha256=None,
    h010_seal_schema='h010-engineering-inventory-seal-v1',
    compiler_contract=kernel.CONTRACT_VERSION, max_payload_bytes=_MAX_PAYLOAD, max_seal_bytes=_MAX_SEAL,
    inference_fields=['id', 'source', 'response'], artifact_only=True, model_calls_allowed=False,
    tokenizer_calls_allowed=False, nlp_calls_allowed=False, qa_admitted=False, scoring_performed=False,
    target_selection_allowed=False, unproposed_regions='unknown_unscored_no_default_zero')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _source_identity():
    """Hash only executed producer/helper code, never unrelated resources."""
    return dict(scope='named_H012_production_code_only', python=sys.version, source_sha256={
        'experiments/natural_masked_inventory.py': data.file_hash(Path(__file__)),
        'experiments.research/natural_masked_questions.py': data.file_hash(Path(kernel.__file__)),
        'experiments/data.py': data.file_hash(Path(data.__file__)),
        'experiments/verdict_evaluation.py': data.file_hash(Path(reader.__file__))})


def _paths(inputs, h010_root, output, protocol):
    i, h, o, p = (Path(path).absolute() for path in (inputs, h010_root, output, protocol))
    trees = (i.parent, h, o)
    _require(all(path.resolve() == path for path in (i, h, o, p)) and i.name == 'inputs.jsonl'
        and 'outputs' in o.parts and all(not a.is_relative_to(b) and not b.is_relative_to(a)
            for index, a in enumerate(trees) for b in trees[index + 1:])
        and all(not p.is_relative_to(tree) for tree in trees)
        and not any(part.casefold() in {'annotations', 'evaluation_only', 'evaluation_only.jsonl',
            'annotations.jsonl', 'construction_truth.jsonl'} for part in p.parts),
        'Independent unaliased inference/H010/output trees and external protocol required')
    if o.exists():
        raise FileExistsError('Retained natural span artifacts cannot be overwritten or automatically retried')
    for path in (i, p):
        _require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size <= _MAX_PAYLOAD,
                 'Bounded regular input/protocol files required')
    return i, h, o, p


def _inventory(root):
    """Named members only: auxiliary H010 files are neither read nor blessed."""
    _require(root.is_dir() and root.resolve() == root, 'Unaliased H010 root required')
    for name in PAYLOADS | {'pilot-seal.json'}:
        path = _member(root, name)
        maximum = _MAX_SEAL if name == 'pilot-seal.json' else _MAX_PAYLOAD
        _require(0 < path.stat().st_size <= maximum, 'H010 member exceeds fixed size admission')
    seal = data.read_json(root / 'pilot-seal.json')
    _require(isinstance(seal, dict) and set(seal) == {'schema', 'files_sha256'}
        and seal['schema'] == DEFAULT_PROTOCOL['h010_seal_schema']
        and isinstance(seal['files_sha256'], dict) and set(seal['files_sha256']) == PAYLOADS
        and all(_sha(value) for value in seal['files_sha256'].values()),
        'Exact custom H010 four-member seal required')
    return seal


def _snapshot(root, expected=None):
    seal = _inventory(root)  # All names/aliases/sizes/keys precede every payload hash.
    current = dict(seal_sha256=data.file_hash(root / 'pilot-seal.json'),
                   files_sha256={name: data.file_hash(root / name) for name in sorted(PAYLOADS)})
    _require(current['files_sha256'] == seal['files_sha256'], 'H010 sealed payload hash changed')
    if expected is not None:
        _equal(current, expected, 'Initial H010 seal or allowed payload changed')
    return current


def _protocol(path):
    value = data.read_json(path)
    _require(isinstance(value, dict) and not set(value) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
        and all(type(value.get(key)) is type(expected) and value[key] == expected
            for key, expected in DEFAULT_PROTOCOL.items() if key != 'original_inputs_sha256')
        and _sha(value.get('original_inputs_sha256')), 'Fixed complete H012 artifact-only protocol required')
    return value


def _rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _validate(inputs, h010, snapshot, protocol):
    raw_inputs = _rows(inputs)
    _require(all(isinstance(row, dict) and all(isinstance(row.get(key), str)
        for key in ('id', 'source', 'response')) for row in raw_inputs), 'Original inference fields required')
    samples = [{key: row[key] for key in ('id', 'source', 'response')} for row in raw_inputs]
    inventories = _rows(h010 / 'pilot-inventory.jsonl')
    freeze, summary, probe = (data.read_json(h010 / name) for name in
                              ('pilot-freeze.json', 'pilot-summary.json', 'synthetic-probe.json'))
    count = protocol['expected_original_cases']
    _require(len(samples) == count and len({row['id'] for row in samples}) == count
        and len(inventories) == count and [row.get('id') for row in inventories] == [row['id'] for row in samples],
        'Complete ordered original18/inventory population required')
    input_hash = data.file_hash(inputs)
    _require(input_hash == protocol['original_inputs_sha256'] == freeze.get('inputs_sha256')
        and freeze.get('schema') == 'h010-span-pilot-freeze-v1' and freeze.get('cases') == count
        and Path(freeze.get('inputs', '')).absolute() == inputs
        and freeze.get('fields') == ['source', 'response'] and freeze.get('read_whitelist') == ['id', 'source', 'response']
        and summary.get('schema') == 'h010-span-inventory-summary-v1' and summary.get('status') == 'inventory_complete'
        and summary.get('cases_retained') == count and summary.get('fields_retained') == 2 * count
        and summary.get('unknown_fields') == 0 and summary.get('processed_fields') == 2 * count
        and summary.get('model_identity') == freeze.get('model_identity')
        and summary.get('task_gold_opened') is False and summary.get('gpu_calls') == 0
        and summary.get('factuality_scores_computed') == 0, 'Frozen H010/input/summary contract changed')
    identity = freeze.get('model_identity', {})
    _require(identity.get('spacy') == '3.8.18' and identity.get('model') == 'en_core_web_sm'
        and identity.get('model_version') == '3.8.0', 'Frozen externally supervised H010 language frontend required')
    evidence = summary.get('evidence_sha256')
    _require(isinstance(evidence, dict) and set(evidence) == {'readiness.json', 'pilot-freeze.json',
        'synthetic-probe.json', 'pilot-inventory.jsonl'} and all(_sha(value) for value in evidence.values())
        and all(evidence[name] == snapshot['files_sha256'][name] for name in PAYLOADS - {'pilot-summary.json'})
        and freeze.get('readiness_sha256') == evidence['readiness.json'], 'H010 frozen evidence hash links changed')
    _require(probe.get('schema') == 'h010-synthetic-offset-engineering-v1'
        and probe.get('semantic_NER_accuracy_assessed') is False
        and isinstance(probe.get('checks'), dict) and probe['checks']
        and all(value is True for value in probe['checks'].values()), 'H010 engineering probe contract changed')
    labels = {key: Counter() for key in ('source', 'response')}
    for sample, inventory in zip(samples, inventories, strict=True):
        for key in ('source', 'response'):
            field = inventory.get(key)
            _require(kernel._field_reason(field, sample[key]) is None
                and all(kernel._entity_reason(entity, sample[key]) is None for entity in field['entities']),
                'H010 source/response identity, occurrence IDs or exact Unicode offsets changed')
            labels[key].update(entity['type'] for entity in field['entities'])
    _require(sum(labels['source'].values()) == protocol['expected_source_occurrences']
        and sum(labels['response'].values()) == protocol['expected_response_occurrences']
        and summary.get('model_spans_by_type') == {key: dict(value) for key, value in labels.items()},
        'All719 source/136 response emitted occurrences required without selection')
    return samples, inventories, identity


def _output_inventory(output, names, sealed=False):
    expected = set(names) | ({'seal.json'} if sealed else set())
    _require(output.resolve() == output and output.is_dir()
        and {path.name for path in output.iterdir()} == expected, 'Exact H012 output inventory required')
    for name in expected:
        _member(output, name)


def _admit_artifact(artifact, samples, inventories, protocol):
    _require(isinstance(artifact, dict) and artifact.get('schema') == kernel.CONTRACT_VERSION
        and set(artifact) == {'schema', 'originals', 'source_pools', 'proposals', 'counts', 'limitations'}
        and artifact['counts']['originals'] == protocol['expected_original_cases']
        and artifact['counts']['source_occurrences'] == protocol['expected_source_occurrences']
        and artifact['counts']['response_occurrences'] == protocol['expected_response_occurrences']
        and [row['id'] for row in artifact['originals']] == [row['id'] for row in samples],
        'Compiler must retain every original and frozen occurrence')
    expected = [(sample['id'], index, entity) for sample, row in zip(samples, inventories, strict=True)
                for index, entity in enumerate(row['response']['entities'])]
    _require(len(artifact['proposals']) == len(expected), 'Every emitted target needs its own proposal')
    for proposal, (owner, index, entity) in zip(artifact['proposals'], expected, strict=True):
        readout = proposal.get('target_readout', {})
        _require(proposal.get('owner_id') == owner and type(proposal.get('response_entity_index')) is int
            and proposal['response_entity_index'] == index
            and all(type(readout.get(key)) is type(entity[key]) and readout[key] == entity[key]
                    for key in ('occurrence_id', 'text', 'type', 'start_char', 'end_char'))
            and proposal.get('qa_admitted') is False and proposal.get('risk') is None
            and proposal.get('semantic_status') == kernel._SEMANTIC
            and (proposal.get('cloze_artifact') is None or set(proposal['cloze_artifact']) ==
                 {'masked_response', 'target_placeholder', 'native_label'}),
            'Every exact target must remain unscored with no executable QA fields')
    source_expected = [kernel._provenance(entity, sample['id'], index)
        for sample, row in zip(samples, inventories, strict=True) for index, entity in enumerate(row['source']['entities'])]
    source_actual = [entity for pool in artifact['source_pools'].values() for entity in pool['occurrences']]
    order = lambda entity: (entity['owner_id'], entity['entity_index'])
    _equal(sorted(source_actual, key=order), sorted(source_expected, key=order),
           'Every source occurrence provenance must be retained without deduplication')
    _require(all(row.get('qa_admitted') is False and row.get('risk') is None
        and row.get('unproposed_response_regions') == kernel.UNKNOWN for row in artifact['originals'])
        and all(pool.get('qa_admitted') is False and pool.get('risk') is None
            and pool.get('semantic_status') == kernel.UNKNOWN for pool in artifact['source_pools'].values())
        and all(artifact['counts'].get(key) == 0 for key in ('qa_admitted', 'model_calls', 'tokenizer_calls', 'nlp_calls')),
        'Source pools and unproposed regions cannot imply answerability or zero risk')


def prepare(inputs, h010_root, output, protocol):
    inputs, h010, output, protocol_path = _paths(inputs, h010_root, output, protocol)
    snapshot = _snapshot(h010)
    frozen = _protocol(protocol_path)
    protocol_hash, source = data.file_hash(protocol_path), _source_identity()
    samples, inventories, frontend_identity = _validate(inputs, h010, snapshot, frozen)
    input_hash = data.file_hash(inputs)
    own_code = dict(outer=data.file_hash(Path(__file__)), kernel=data.file_hash(Path(kernel.__file__)))
    plan = dict(schema='natural-masked-inventory-plan-v1', state='frozen_before_cloze_compilation',
        inputs_path=str(inputs), inputs_sha256=input_hash, h010_root=str(h010), h010_snapshot=snapshot,
        protocol_sha256=protocol_hash, source_identity_sha256=kernel._key(source), code_sha256=own_code,
        compiler_contract=kernel.CONTRACT_VERSION, original_ids=[row['id'] for row in samples],
        frontend_identity=frontend_identity, frontend_assets_read=False, evaluation_material_accessed=False,
        nlp_calls=0, tokenizer_calls=0, model_calls=0, scoring_performed=False)
    output.mkdir(parents=True)
    names, hashes, seal_hash = set(), {}, None

    def retain(name, value):
        data.write_json(output / name, value); names.add(name)
        _output_inventory(output, names)
        _equal(data.read_json(output / name), value, 'Written H012 artifact differs from prepared value')
        hashes[name] = data.file_hash(output / name)

    for name, value in [('plan.json', plan), ('protocol.json', frozen), ('source_identity.json', source)]:
        retain(name, value)
    retain('freeze.json', dict(hashes))
    freeze_hashes = dict(hashes)

    def unchanged():
        _output_inventory(output, names, sealed=seal_hash is not None)
        _require(all(data.file_hash(output / name) == value for name, value in hashes.items()), 'H012 retained output changed')
        _require(set(freeze_hashes) == _FROZEN and all(hashes[name] == value for name, value in freeze_hashes.items()),
                 'H012 pre-compilation freeze changed')
        if seal_hash is not None:
            _require(data.file_hash(output / 'seal.json') == seal_hash, 'Initial H012 output seal changed')
        _snapshot(h010, snapshot)
        _require(data.file_hash(inputs) == input_hash and data.file_hash(protocol_path) == protocol_hash,
                 'Original input/protocol changed')
        _require(own_code == dict(outer=data.file_hash(Path(__file__)), kernel=data.file_hash(Path(kernel.__file__))),
                 'H012 production code changed')
        _equal(_source_identity(), source, 'H012 producer/helper source identity changed')

    unchanged()
    artifact = kernel.prepare_natural_questions(samples, inventories)
    _admit_artifact(artifact, samples, inventories, frozen)
    summary = dict(schema='natural-masked-inventory-summary-v1', state='artifact_inventory_only',
        counts=artifact['counts'], semantic_status=kernel.UNKNOWN, qa_admitted=False, risk=None,
        scores_computed=0, nlp_calls=0, tokenizer_calls=0, model_calls=0,
        external_supervision='Frozen H010 spaCy NER/POS/parser trained externally, including OntoNotes; no task gold used.',
        natural_qa_coverage_established=False, method_promoted=False, h011_masked_qa_failure_unresolved=True,
        limitations=artifact['limitations'], frontend_assets_read=False,
        h010_only_named_payloads_hashed=sorted(PAYLOADS), auxiliary_h010_files_blessed=False)
    for name, value in [('artifact.json', artifact), ('counts.json', artifact['counts']), ('summary.json', summary)]:
        retain(name, value)
    unchanged()
    try:
        retain('complete.json', dict(schema='natural-masked-inventory-complete-v1', state='sealed_unscored_cloze_inventory',
            originals=len(samples), source_occurrences=719, response_occurrences=136,
            qa_admitted=False, risk=None, nlp_calls=0, tokenizer_calls=0, model_calls=0,
            scoring_performed=False, method_promoted=False, h010_initial_seal_sha256=snapshot['seal_sha256']))
        unchanged()
        data.write_json(output / 'seal.json', dict(hashes))
        _output_inventory(output, names, sealed=True)
        _equal(data.read_json(output / 'seal.json'), hashes, 'Final H012 seal differs from original artifact hashes')
        seal_hash = data.file_hash(output / 'seal.json')
        data.verify_seal(output, expected=names); unchanged()
    except BaseException:
        (output / 'complete.json').unlink(missing_ok=True); (output / 'seal.json').unlink(missing_ok=True)
        raise
    return summary
