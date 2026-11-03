"""Synthetic CPU-only execution, frozen-score release order and corruption checks."""
from copy import deepcopy
from dataclasses import asdict
from itertools import product
import hashlib
import json
import sys

import numpy as np
import pytest

from experiments import data, paired_scope_qualification as collect
from experiments import paired_scope_evaluation as evaluate, reconstruction_qualification as source
from experiments import prefix_completion, run
from experiments.datasets import Sample
from tests.test_completion_trie import Tokenizer
from tests.test_paired_scope import samples


def population():
    rows = []
    for original in samples():
        for burden in (0, 4, 12):
            row = dict(original, id=original['id'] + ':' + str(burden))
            row['response'] += ' '.join(collect.masked.FILLERS[:burden])
            rows.append(Sample(**row, task='toy', split='diagnostic', prompt=row['source']))
    return rows


class Framing(Tokenizer):
    is_fast = True

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = 'Q:' + collect._digest(messages)
        return [ord(c) for c in text] if tokenize else text


def result(trie, identity):
    rows, records = [], []
    for key, node in trie['nodes'].items():
        vector = np.zeros(128, dtype=np.float16)
        rows.append(dict(node_id=key, context_ids=node['context_ids'], vocab_size=128, full_vocab_logits=vector.tolist()))
        records.append(dict(node_id=key, context_ids=node['context_ids'], context_sha256=collect._digest(node['context_ids']),
            full_fp16_vector=vector, actual_forward_passes=1, input_tokens=len(node['context_ids']), output_tokens=0,
            logits_shape=[1, 1, 128], observed_dtype='torch.float16', full_vector_dtype=vector.dtype.str,
            full_vector_shape=[128], full_vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
            seconds=0., peak_cuda_allocated_bytes=100, peak_cuda_reserved_bytes=200,
            resource_admission=dict(device='cuda:0', available_ram_bytes=3 * 1024**3,
                free_cuda_bytes=4 * 1024**3, cuda_allocator_limit_bytes=3 * 1024**3)))
    return dict(schema='prefix-only-full-vector-observation-v1', status='scored', raw_nodes=records,
        trie_sha256=collect._digest(trie), model_identity=identity, model_identity_sha256=collect._digest(identity),
        model_calls=len(records), attempted_model_calls=len(records), actual_forward_passes=len(records),
        input_tokens=sum(len(n['context_ids']) for n in trie['nodes'].values()), output_tokens=0,
        seconds_forward=0., tree_result=collect.trie_kernel.score_trie(trie, rows))


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    # Provenance's OS/runtime scan has its own tests; exercise artifact guards here.
    stable_source = dict(source_sha256={'synthetic_fixture_only': 'a' * 64})
    monkeypatch.setattr(collect, 'research_identity', lambda: deepcopy(stable_source))
    monkeypatch.setattr(collect, 'completed_identity', lambda value: value == stable_source or pytest.fail('source changed'))
    monkeypatch.setattr(evaluate, 'research_identity', lambda: deepcopy(stable_source))
    monkeypatch.setattr(evaluate, 'completed_identity', lambda value: value == stable_source or pytest.fail('source changed'))
    inputs = tmp_path / 'inference' / 'inputs.jsonl'; inputs.parent.mkdir()
    rows = population()
    inputs.write_text(''.join(json.dumps(asdict(s)) + '\n' for s in rows), encoding='utf-8')
    model = tmp_path / 'model'; model.mkdir(); data.write_json(model / 'config.json', dict(vocab_size=128))
    identity = dict(model_id='synthetic-never-loaded', model_files_sha256={'config.json': data.file_hash(model / 'config.json')})
    donor = dict(root=str(tmp_path / 'donor'), nodes={}, snapshot=dict(seal_sha256='d' * 64))
    protocol = tmp_path / 'protocol.json'
    data.write_json(protocol, dict(collect.DEFAULT_PROTOCOL, original_inputs_sha256=data.file_hash(inputs), engineering_seal_sha256='d' * 64))
    monkeypatch.setattr(collect, '_engineering', lambda *args: deepcopy(donor))
    monkeypatch.setattr(collect, '_tokenizer', lambda *args: Framing())
    monkeypatch.setattr(collect, '_model_guard', lambda *args: lambda: None)
    preparation = tmp_path / 'outputs' / 'preparation'
    manifest = collect.prepare(inputs, protocol, model, identity, donor['root'], preparation)
    class Backend:
        admitted = True
        model_identity = identity
        calls = 0
        registered = 0
        mutate = None
        def validate_trie(self, messages, candidates):
            self.registered += 1
            return collect.trie_kernel.build_trie({c: collect.completion_metadata(Framing(), messages, c) for c in candidates})
        def score_trie(self, trie):
            assert self.registered == 32
            self.calls += 1
            out = result(trie, identity)
            if self.mutate: self.mutate()
            return out
    return dict(tmp=tmp_path, inputs=inputs, rows=rows, protocol=protocol, model=model, identity=identity,
        preparation=preparation, manifest=manifest, backend=Backend(), output=tmp_path / 'outputs' / 'observation')


def observe(f):
    return collect.qualify(f['preparation'], f['output'], f['backend'], f['identity'],
        measure_resources=lambda: dict(allocated_bytes=100, reserved_bytes=200))


def reference_bundle(f):
    prepared = source._prepare(f['rows'])
    design, truth = [], []
    for original in prepared['originals']:
        sample, alignment = original['sample'], original['alignment']
        family, vocabulary, sd, rd, burden = map(int, sample['id'].split(':'))
        cell = dict(relation_family=('winner', 'sole_signer')[family], vocabulary=('lex1', 'lex2')[vocabulary],
                    source_direction=sd, response_direction=rd, distractor_count=burden)
        outcome = 'supported' if sd == rd else 'contradicted'
        design.append(dict(id=sample['id'], source_id=sample['source_id'], input_text_sha256=original['input_text_sha256'],
            target_span=alignment['target_span'], target_text=alignment['target_text'], whole_span=[0, len(sample['response'])], factor_cell=cell))
        truth.append(dict(id=sample['id'], source_id=sample['source_id'], source_sha256=original['input_text_sha256']['source'],
            response_sha256=original['input_text_sha256']['response'], target_span=alignment['target_span'], target_outcome=outcome,
            whole_outcome=outcome, filler_outcomes=['supported'] * burden, target_benchmark_labels_used=False,
            construction_rule='entity_event_bijections_equal_iff_source_direction_equals_response_direction'))
    design_path = f['inputs'].parent / 'plan.json'
    data.write_json(design_path, dict(schema='h007-burden-controls-plan-v1', records=design))
    data.write_json(f['inputs'].parent / 'seal.json', {n: data.file_hash(f['inputs'].parent / n) for n in ('inputs.jsonl', 'plan.json')})
    truth_path = f['tmp'] / 'construction' / 'construction_truth.jsonl'; truth_path.parent.mkdir()
    truth_path.write_text(''.join(json.dumps(r) + '\n' for r in truth), encoding='utf-8')
    data.write_json(truth_path.parent / 'seal.json', {truth_path.name: data.file_hash(truth_path)})
    return design_path, truth_path


def test_complete_roundtrip_and_gold_follows_full_four_arm_freeze(frozen, monkeypatch):
    f = frozen
    assert f['manifest']['model_calls_during_preparation'] == 0
    assert len(f['manifest']['population']['excluded_ids']) == 32
    completed = observe(f)
    assert completed['costs']['actual_forward_passes'] == f['manifest']['budget']['new_forward_passes']
    assert set(completed['readouts']['arms']) == set(collect.kernel.ARMS)
    assert f['backend'].calls == 32
    assert evaluate.validate(f['output'])[1] == completed
    design, truth = reference_bundle(f)
    output = f['tmp'] / 'outputs' / 'evaluation'
    read = data.read_json
    accesses = []
    def checked(path):
        if path in (design, truth.parent / 'seal.json', design.parent / 'seal.json'):
            assert (output / 'freeze.json').is_file()
            assert set(read(output / 'scores.json')['arms']) == set(collect.kernel.ARMS)
            accesses.append(path)
        return read(path)
    monkeypatch.setattr(data, 'read_json', checked)
    summary = evaluate.evaluate(f['output'], design, truth, output)
    assert accesses and summary['primary_cases'] == 16
    assert len(summary['arms']) == 4 and summary['new_model_calls'] == 0
    for arm in summary['arms'].values():
        assert arm['source_assignment']['total'] == 8 and arm['query_argmax']['total'] == 16
        assert arm['secondary_case_metrics']['known_units'] == 16
    assert (output / 'seal.json').is_file() and f['backend'].calls == 32


@pytest.mark.parametrize('kind', ['extra', 'raw', 'npz', 'seal'])
def test_corruption_fails_without_reference_access(frozen, kind):
    f = frozen; observe(f)
    child = next((f['output'] / 'queries').iterdir())
    target = {'extra': child / 'extra.json', 'raw': child / 'raw.json', 'npz': child / 'vectors.npz', 'seal': f['output'] / 'seal.json'}[kind]
    target.write_bytes(b'corruption')
    with pytest.raises((ValueError, KeyError)):
        evaluate.validate(f['output'])
    assert f['backend'].calls == 32


def test_midrun_input_mutation_retains_first_raw_and_stops_without_retry(frozen):
    f = frozen
    f['backend'].mutate = lambda: f['inputs'].write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        observe(f)
    assert f['backend'].calls == 1
    assert not (f['output'] / 'complete.json').exists() and not (f['output'] / 'seal.json').exists()
    assert data.read_json(f['output'] / 'failure.json')['automatic_retry'] is False
    assert len(list((f['output'] / 'queries').glob('*/vectors.npz'))) == 1


def test_failed_postcapture_telemetry_retains_vectors_without_success(frozen):
    f = frozen; calls = []
    def measure():
        calls.append(1)
        if len(calls) == 2: raise RuntimeError('synthetic telemetry failure')
        return dict(allocated_bytes=100, reserved_bytes=200)
    with pytest.raises(ValueError, match='allocator'):
        collect.qualify(f['preparation'], f['output'], f['backend'], f['identity'], measure_resources=measure)
    assert f['backend'].calls == 1 and len(list((f['output'] / 'queries').glob('*/vectors.npz'))) == 1
    assert not (f['output'] / 'seal.json').exists()


def test_partial_cached_tree_rejected_before_any_weight_or_output(frozen, monkeypatch):
    f = frozen
    manifest = deepcopy(f['manifest'])
    query = next(iter(manifest['queries'].values()))
    query['reused_node_ids'] = [next(iter(query['trie']['nodes']))]
    # Even if a tampered preparation is resealed consistently, cached nodes cannot be rerun.
    (f['preparation'] / 'manifest.json').unlink()
    data.write_json(f['preparation'] / 'manifest.json', manifest)
    for name in ('freeze.json', 'seal.json'):
        (f['preparation'] / name).unlink()
        if name == 'freeze.json': names = collect.PREPARATION_FILES - {'freeze.json'}
        else: names = collect.PREPARATION_FILES
        data.write_json(f['preparation'] / name, {n: data.file_hash(f['preparation'] / n) for n in names})
    monkeypatch.setattr(collect, '_build', lambda *args: manifest)
    with pytest.raises(ValueError, match='Cached tree'):
        observe(f)
    assert f['backend'].calls == 0 and not f['output'].exists()


def test_resource_cli_gate_precedes_identity_read_and_backend(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('No backend before admission')
    monkeypatch.setattr(data, 'read_json', forbidden)
    monkeypatch.setattr(prefix_completion, 'HFPrefixCompletion', forbidden)
    monkeypatch.setattr(sys, 'argv', ['r4-experiment', 'transfer', 'qualify-paired-scope',
        '--preparation', 'preparation', '--model', 'model', '--model-identity', 'identity.json', '--output', 'outputs/result'])
    with pytest.raises(SystemExit) as error: run.main()
    assert error.value.code == 2
