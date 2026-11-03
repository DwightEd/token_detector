"""Frozen H016 paired-scope plans and complete shared-prefix observations.

Preparation is offline tokenizer-only. Observation never opens references; old
selected-logit completions are not qualified full-vocabulary node caches.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import re
import time
import traceback

import numpy as np

from . import data, verdict_evaluation as paths
from .answer_completion import completion_metadata
from .datasets import read_samples
from .prefix_completion import HFPrefixCompletion, MAX_TEMPORARY_VECTOR_BYTES
from .provenance import completed_identity, research_identity
from .research import completion_trie as trie_kernel, masked_reconstruction as masked
from .research import paired_scope as kernel
from .semantic_qualification import ANNOTATION_NAMES, _model_identity

DEFAULT_PROTOCOL = dict(schema='paired-scope-prefix-qualification-v1',
    original_inputs_sha256=None, engineering_seal_sha256=None,
    expected_input_cases=48, expected_primary_cases=16, expected_sources=8,
    expected_queries_per_scope=16, scopes=list(kernel.SCOPES), arms=list(kernel.ARMS),
    population='all_exact_burden_zero_originals_from_frozen_h007_inputs',
    question_template=masked.QUESTION_TEMPLATE, candidate_pool_depends_only_on_source=True,
    completion_contract='full-candidate-plus-eos-v1', context_contract=trie_kernel.CONTRACT_VERSION,
    max_context_tokens=512, max_nodes_per_query=32, temperature=1.0,
    length_normalization=False, attention_implementation='sdpa', output_tokens=0,
    use_cache=False, full_vector_dtype='float16', full_vocab_normalizer_dtype='float64',
    cache_identity_ignores_execution_provenance_only=True,
    partial_cached_tree_policy='stop_before_any_model_call',
    population_previously_exposed=True, model_selection_performed=False,
    natural_task_performance_claim=False, evaluation_material_accessed=False)
PREPARATION_FILES = {'manifest.json', 'protocol.json', 'model_identity.json', 'source_identity.json', 'freeze.json'}
ROOT_FILES = {'manifest.json', 'protocol.json', 'model_identity.json', 'source_identity.json', 'preparation.json', 'freeze.json'}
PROBES = ('short_shared_branch', 'bounded_512_shared_branch')
ENGINEERING_FILES = {'complete.json', 'freeze.json', 'git-package-binding.json', 'manifest.json',
    'model-identity.json', 'package-admission.json', 'plan.json', 'run-spec.json', 'summary.json'} | {
    probe + suffix for probe in PROBES for suffix in
    ('-current-before.json', '-raw.json', '-replay.json', '-vectors.npz')}
INSTRUMENT_HELPERS = ('experiments/prefix_completion.py', 'experiments/research/completion_trie.py',
    'experiments/answer_completion.py', 'experiments/semantic_qualification.py')
_require = trie_kernel._require
_digest = trie_kernel._digest


def _path(value, *, inference=False):
    value = Path(value).absolute()
    _require(value.resolve() == value and not value.is_symlink(), 'Unaliased path required')
    forbidden = ANNOTATION_NAMES | {'construction_truth.jsonl', 'evaluation_only', 'annotations'}
    _require(not any(part.casefold() in forbidden for part in value.parts), 'Annotation paths forbidden before score freeze')
    if inference:
        _require(value.name == 'inputs.jsonl', 'Inference-only inputs.jsonl required')
    return value


def _protocol(path):
    value = data.read_json(path)
    _require(isinstance(value, dict) and not set(value) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
        and all(type(value.get(k)) is type(v) and value[k] == v for k, v in DEFAULT_PROTOCOL.items()
                if k not in ('original_inputs_sha256', 'engineering_seal_sha256'))
        and all(isinstance(value.get(k), str) and re.fullmatch('[0-9a-f]{64}', value[k])
                for k in ('original_inputs_sha256', 'engineering_seal_sha256')), 'Exact paired-scope protocol required')
    return value


def _flat_snapshot(root, names):
    root = _path(root)
    _require(root.is_dir() and {p.name for p in root.iterdir()} == set(names) | {'seal.json'},
             'Exact sealed flat inventory required')
    for name in set(names) | {'seal.json'}:
        paths._member(root, name)
    seal = data.read_json(root / 'seal.json')
    _require(isinstance(seal, dict) and set(seal) == set(names)
        and all(isinstance(v, str) and re.fullmatch('[0-9a-f]{64}', v) for v in seal.values()),
        'Exact seal member map required before payload hashing')
    _require(all(data.file_hash(root / name) == digest for name, digest in seal.items()), 'Initial sealed payload changed')
    return dict(seal_sha256=data.file_hash(root / 'seal.json'), file_sha256=seal)


def _base_identity(identity):
    """Execution provenance is retained, but does not change the numerical instrument."""
    return {k: deepcopy(v) for k, v in identity.items() if k != 'execution'}


def _engineering(root, identity):
    snapshot = _flat_snapshot(root, ENGINEERING_FILES)
    root = _path(root)
    manifest = data.read_json(root / 'manifest.json')
    donor_identity = data.read_json(root / 'model-identity.json')
    receipt = data.read_json(root / 'package-admission.json')
    summary, complete = (data.read_json(root / n) for n in ('summary.json', 'complete.json'))
    _require(_base_identity(donor_identity) == _base_identity(identity)
        and summary.get('status') == 'engineering_passed' and summary.get('strict_probability_gates_passed') is True
        and complete.get('full_vectors_retained') is complete.get('same_kernel_replay') is True
        and complete.get('maximum_context_tokens') == 512, 'Qualified same-instrument full-vector donor required')
    package = Path(__file__).resolve().parents[1]
    for name in INSTRUMENT_HELPERS:
        _require(receipt['source_files_sha256'].get(name) == receipt['installed_files_sha256'].get(name)
            == data.file_hash(package / name), 'Engineering observer helper differs from current instrument')
    nodes = {}
    _require([p['probe_id'] for p in manifest['probes']] == list(PROBES), 'Exact two engineering probes required')
    for probe in manifest['probes']:
        trie = probe['trie']
        _require(trie == trie_kernel.build_trie(trie['completion_metadata']) and trie['status'] == 'ok',
                 'Donor node contexts differ from canonical trie')
        for key, node in trie['nodes'].items():
            _require(key not in nodes, 'Donor context aliases two distinct probes')
            nodes[key] = dict(context_ids=deepcopy(node['context_ids']), probe_id=probe['probe_id'])
    return dict(root=str(root), snapshot=snapshot, nodes=nodes,
        numerical_identity=_base_identity(donor_identity), execution_provenance=donor_identity.get('execution'),
        only_qualified_full_vector_cache=True, legacy_selected_logit_caches_eligible=False)


def _population(samples):
    """Declare b0 from exact suffix bytes; retain the full48 selection inventory."""
    _require(len(samples) == 48 and len({s.id for s in samples}) == 48, 'All48 original inference rows required')
    compiled = masked.prepare_masked_queries(samples)
    selected, rows, counts = [], [], {}
    by_id = {s.id: s for s in samples}
    for row in compiled['originals']:
        sample = by_id[row['id']]
        _require(row['status'] == 'ok', 'Unknown input compiler occurrence retained; complete paired admission blocked')
        suffix = sample.response[row['target_span'][1]:]
        possible = [b for b in (0, 4, 12) if suffix == ' '.join(masked.FILLERS[:b])]
        _require(len(possible) == 1, 'Exact declared burden suffix required')
        burden = possible[0]
        counts[row['source_sha256'], burden] = counts.get((row['source_sha256'], burden), 0) + 1
        rows.append(dict(sample=asdict(sample), source_key=row['source_sha256'], burden=burden,
            target_span=row['target_span'], primary=burden == 0,
            response_sha256=data.text_hash(sample.response)))
        if burden == 0:
            selected.append(sample)
    _require(len(compiled['sources']) == 8 and len(counts) == 24 and set(counts.values()) == {2}
        and len(selected) == 16, 'Complete8-source/two-direction/three-burden grid required')
    prepared = kernel.prepare(selected)
    expected = dict(originals=16, sources=8, logical_queries=32, unique_queries=16,
        logical_candidates=64, unique_candidates=32, unknown_slots=0)
    _require(all(c == expected for c in prepared['counts']['by_scope'].values())
        and prepared['counts']['unique_queries'] == 32 and prepared['counts']['unique_candidates'] == 64,
        'Complete paired population required; unknown slots cannot be silently filtered')
    return prepared, dict(originals=rows, selected_ids=[s.id for s in selected],
        excluded_ids=[r['sample']['id'] for r in rows if not r['primary']],
        exclusion_rule='prespecified_nonzero_exact_suffix_burden_only', unknown_cases=0)


def _tokenizer(model_path, identity):
    # No HFGenerator._prepare, _ensure_model, CUDA or weight construction.
    HFPrefixCompletion(model_path, identity).verify_frozen_identity()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True,
        trust_remote_code=False, use_fast=True)
    _require(tokenizer.is_fast is True, 'Pinned local fast tokenizer required')
    return tokenizer


def _model_guard(model_path, identity):
    model_path = _path(model_path)
    files = identity['model_files_sha256']
    _require(len(files) == 9 and {str(p.relative_to(model_path)) for p in model_path.rglob('*') if p.is_file()} == set(files),
             'Exact nine model assets required')
    stamps = {}
    for name in files:
        p = paths._member(model_path, name)
        stamps[name] = (p.stat().st_size, p.stat().st_mtime_ns)
    def guard():
        _require({str(p.relative_to(model_path)) for p in model_path.rglob('*') if p.is_file()} == set(files),
                 'Frozen model inventory changed')
        for name, stamp in stamps.items():
            p = paths._member(model_path, name)
            _require((p.stat().st_size, p.stat().st_mtime_ns) == stamp
                and (name.endswith('.safetensors') or data.file_hash(p) == files[name]), 'Model/tokenizer asset changed')
    return guard


def _build(inputs, protocol_path, frozen, model_path, identity, engineering, tokenizer):
    prepared, population = _population(list(read_samples(inputs)))
    vocab = data.read_json(model_path / 'config.json')['vocab_size']
    _require(type(vocab) is int and vocab > 0, 'Frozen vocabulary size required')
    queries, nodes = {}, {}
    for key, query in prepared['queries'].items():
        metadata = {c: completion_metadata(tokenizer, deepcopy(query['messages']), c) for c in query['candidates']}
        trie = trie_kernel.build_trie(metadata)
        _require(trie['status'] == 'ok' and len(trie['nodes']) <= frozen['max_nodes_per_query']
            and len(trie['nodes']) * vocab * 32 <= MAX_TEMPORARY_VECTOR_BYTES, 'Complete trie/vector budget rejected')
        for node_id, node in trie['nodes'].items():
            _require(len(node['context_ids']) <= frozen['max_context_tokens']
                and all(type(i) is int and 0 <= i < vocab for i in node['context_ids'])
                and all(0 <= e['token_id'] < vocab for e in node['children']), 'Untruncated node exceeds qualified512/vocabulary')
            # Current fixed requests must be disjoint after complete-Q deduplication.
            _require(node_id not in nodes, 'Partial cross-query cache overlap requires explicit reuse; no duplicated forwards')
            nodes[node_id] = dict(query_key=key, context_ids=deepcopy(node['context_ids']))
        reused = sorted(set(trie['nodes']) & set(engineering['nodes']))
        _require(not reused or set(reused) == set(trie['nodes']),
                 'Partial donor tree reuse is not supported; stop before any model call')
        for node_id in reused:
            _require(trie['nodes'][node_id]['context_ids'] == engineering['nodes'][node_id]['context_ids'],
                     'Cache node/context identity collision')
        queries[key] = dict(query_key=key, scope=query['scope'], messages=deepcopy(query['messages']),
            candidates=deepcopy(query['candidates']), trie=trie, reused_node_ids=reused)
    new_nodes = {k: v for k, v in nodes.items() if k not in engineering['nodes']}
    by_scope = {scope: dict(unique_queries=sum(q['scope'] == scope for q in queries.values()),
        unique_nodes=sum(q['trie']['counts']['unique_nodes'] for q in queries.values() if q['scope'] == scope),
        new_forward_passes=sum(len(q['trie']['nodes']) - len(q['reused_node_ids']) for q in queries.values()
                              if q['scope'] == scope),
        input_tokens=sum(len(n['context_ids']) for k, n in new_nodes.items() if queries[n['query_key']]['scope'] == scope))
        for scope in kernel.SCOPES}
    budget = dict(unique_queries=len(queries), complete_candidate_paths=64, unique_context_nodes=len(nodes),
        new_forward_passes=len(new_nodes), reused_nodes=len(nodes) - len(new_nodes),
        planned_input_tokens=sum(len(n['context_ids']) for n in new_nodes.values()),
        maximum_context_tokens=max(len(n['context_ids']) for n in nodes.values()),
        retained_full_fp16_vector_bytes=len(nodes) * vocab * 2,
        maximum_temporary_python_vector_bytes=max(len(q['trie']['nodes']) for q in queries.values()) * vocab * 32,
        generated_tokens=0, extra_language_warmups=0, by_scope=by_scope)
    return dict(schema='paired-scope-tokenizer-manifest-v1', state='frozen_before_model_load',
        inputs_path=str(inputs), original_inputs_sha256=data.file_hash(inputs),
        protocol_path=str(protocol_path), protocol_sha256=data.file_hash(protocol_path), protocol=deepcopy(frozen),
        model_path=str(model_path), model_identity=deepcopy(identity), full_vocabulary_size=vocab,
        population=population, prepared=prepared, queries=queries, nodes=nodes, budget=budget,
        engineering=engineering, model_calls_during_preparation=0, cuda_calls_during_preparation=0,
        evaluation_material_accessed=False, h013_failed_record_modified=False)


def prepare(inputs, protocol, model_path, model_identity, engineering, output):
    """Offline fast-tokenizer preparation; no admission, weights, CUDA or references."""
    inputs, protocol, model_path, engineering, output = map(_path, (inputs, protocol, model_path, engineering, output))
    _path(inputs, inference=True)
    _require(not output.exists() and 'outputs' in output.parts and not output.is_relative_to(inputs.parent)
        and not protocol.is_relative_to(inputs.parent) and not protocol.is_relative_to(output)
        and not engineering.is_relative_to(output) and not model_path.is_relative_to(output), 'Fresh independent preparation required')
    frozen, identity = _protocol(protocol), _model_identity(model_identity)
    _require(data.file_hash(inputs) == frozen['original_inputs_sha256'], 'Original input SHA changed')
    source = research_identity()
    donor = _engineering(engineering, identity)
    _require(donor['snapshot']['seal_sha256'] == frozen['engineering_seal_sha256'], 'Pinned engineering seal changed')
    tokenizer = _tokenizer(model_path, identity)
    guard = _model_guard(model_path, identity)
    manifest = _build(inputs, protocol, frozen, model_path, identity, donor, tokenizer)
    _require(manifest['original_inputs_sha256'] == frozen['original_inputs_sha256'], 'Inputs changed during token preparation')
    guard(); completed_identity(source)
    _require(_engineering(engineering, identity) == donor, 'Initial engineering qualification changed')
    output.mkdir(parents=True)
    for name, value in [('manifest.json', manifest), ('protocol.json', frozen),
                        ('model_identity.json', identity), ('source_identity.json', source)]:
        data.write_json(output / name, value)
    freeze = {name: data.file_hash(output / name) for name in PREPARATION_FILES - {'freeze.json'}}
    data.write_json(output / 'freeze.json', freeze)
    data.write_json(output / 'seal.json', {name: data.file_hash(output / name) for name in PREPARATION_FILES})
    _flat_snapshot(output, PREPARATION_FILES)
    return manifest


def _inventory(root, keys, observed=(), pending=None, *, complete=False, sealed=False, frozen=True):
    root = _path(root)
    expected = (ROOT_FILES if frozen else ROOT_FILES - {'freeze.json'}) | {'queries'} | ({'complete.json'} if complete else set()) | ({'seal.json'} if sealed else set())
    _require(root.is_dir() and {p.name for p in root.iterdir()} == expected, 'Exact paired observation root inventory required')
    for name in expected - {'queries'}:
        paths._member(root, name)
    children = paths._member(root, 'queries', directory=True)
    _require({p.name for p in children.iterdir()} == set(keys), 'Exact planned query child inventory required')
    names = set(expected) - {'queries', 'seal.json'}
    _require(set(observed) <= set(keys) and (pending is None or pending in set(keys) - set(observed)), 'Invalid child progress')
    for key in keys:
        _require(re.fullmatch('[0-9a-f]{64}', key), 'Canonical query key required')
        child = paths._member(children, key, directory=True)
        members = {'request.json'} | ({'raw.json', 'vectors.npz', 'replay.json', 'seal.json'} if key in observed else set())
        if key == pending:
            members = {p.name for p in child.iterdir()}
            _require({'request.json'} <= members <= {'request.json', 'raw.json', 'vectors.npz', 'replay.json'}, 'Unsafe partial child inventory')
        _require({p.name for p in child.iterdir()} == members, 'Unexpected query child member')
        for name in members:
            paths._member(child, name)
            names.add(f'queries/{key}/{name}')
    return names


def _allocator():
    import torch
    torch.cuda.synchronize(0)
    allocated, reserved = torch.cuda.memory_allocated(0), torch.cuda.memory_reserved(0)
    _require(type(allocated) is int and type(reserved) is int and 0 <= allocated <= reserved <= 3 * 1024**3,
             'Valid current CUDA allocator counters required')
    return dict(allocated_bytes=allocated, reserved_bytes=reserved)


def _persist(child, result):
    arrays, serial = {}, {k: deepcopy(v) for k, v in result.items() if k != 'raw_nodes'}
    serial['raw_nodes'] = []
    for raw in result['raw_nodes']:
        row = {}
        for key, value in raw.items():
            if isinstance(value, np.ndarray):
                name = raw['node_id'] + '__' + key
                arrays[name] = value
                row[key] = dict(npz_member=name, dtype=value.dtype.str, shape=list(value.shape),
                    raw_bytes_sha256=hashlib.sha256(value.tobytes()).hexdigest())
            else:
                row[key] = deepcopy(value)
        serial['raw_nodes'].append(row)
    with (child / 'vectors.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    data.write_json(child / 'raw.json', serial)
    return serial


def _replay(child, trie, raw, vocab):
    _require(raw.get('status') == 'scored' and raw.get('schema') == 'prefix-only-full-vector-observation-v1'
        and raw.get('trie_sha256') == _digest(trie) and raw.get('model_calls') == raw.get('attempted_model_calls')
        == raw.get('actual_forward_passes') == len(trie['nodes'])
        and raw.get('input_tokens') == sum(len(n['context_ids']) for n in trie['nodes'].values())
        and raw.get('output_tokens') == 0, 'Complete one-forward-per-context observation required')
    records = raw.get('raw_nodes')
    _require(isinstance(records, list) and len(records) == len(trie['nodes'])
        and {r['node_id'] for r in records} == set(trie['nodes']), 'Exact raw node population required')
    rows = []
    with np.load(child / 'vectors.npz', allow_pickle=False) as saved:
        expected = {r['node_id'] + '__full_fp16_vector' for r in records}
        _require(set(saved.files) == expected, 'Exact full-vector NPZ members required')
        for row in records:
            node = trie['nodes'][row['node_id']]
            ref = row['full_fp16_vector']
            vector = saved[row['node_id'] + '__full_fp16_vector']
            _require(row['context_ids'] == node['context_ids'] and row['context_sha256'] == _digest(node['context_ids'])
                and row.get('actual_forward_passes') == 1 and row.get('input_tokens') == len(node['context_ids'])
                and row.get('output_tokens') == 0 and row.get('logits_shape') == [1, 1, vocab]
                and row.get('observed_dtype') == 'torch.float16' and row.get('full_vector_dtype') == vector.dtype.str
                and row.get('full_vector_shape') == [vocab] and ref == dict(npz_member=row['node_id'] + '__full_fp16_vector',
                    dtype=vector.dtype.str, shape=[vocab], raw_bytes_sha256=hashlib.sha256(vector.tobytes()).hexdigest())
                and vector.dtype == np.dtype('float16') and list(vector.shape) == [vocab]
                and bool(np.isfinite(vector).all()) and hashlib.sha256(vector.tobytes()).hexdigest() == row['full_vector_sha256'],
                'Original full FP16 row bytes/context/shape changed')
            admission = row.get('resource_admission', {})
            _require(admission.get('device') == 'cuda:0'
                and type(admission.get('available_ram_bytes')) is int and admission['available_ram_bytes'] >= 2 * 1024**3
                and type(admission.get('free_cuda_bytes')) is int and admission['free_cuda_bytes'] >= 3 * 1024**3
                and admission.get('cuda_allocator_limit_bytes') == 3 * 1024**3, 'Frozen resource admission required')
            a, r = row.get('peak_cuda_allocated_bytes'), row.get('peak_cuda_reserved_bytes')
            _require(type(a) is int and type(r) is int and 0 <= a <= r <= 3 * 1024**3
                and type(row.get('seconds')) in (int, float) and math.isfinite(row['seconds']) and row['seconds'] >= 0,
                'Finite node duration and allocator peaks required')
            rows.append(dict(node_id=row['node_id'], context_ids=row['context_ids'], vocab_size=vocab, full_vocab_logits=vector.tolist()))
    replay = trie_kernel.score_trie(trie, rows)
    _require(replay['status'] == 'ok' and replay == raw['tree_result'], 'Same-kernel full-vector persistence replay changed')
    _require(raw.get('seconds_forward') == math.fsum(r['seconds'] for r in records)
        or raw.get('seconds_forward') == sum(r['seconds'] for r in records), 'Captured forward durations changed')
    current = raw.get('collector_current_allocator')
    _require(isinstance(current, dict) and set(current) == {'before', 'after'}, 'Current allocator measurements missing')
    for value in current.values():
        _require(set(value) == {'allocated_bytes', 'reserved_bytes'}
            and type(value['allocated_bytes']) is int and type(value['reserved_bytes']) is int
            and 0 <= value['allocated_bytes'] <= value['reserved_bytes'] <= 3 * 1024**3, 'Current allocator measurement invalid')
    return replay


def _complete(manifest, results):
    readouts = kernel.derive(manifest['prepared'], {k: r['tree_result'] for k, r in results.items()})
    raw_nodes = [node for result in results.values() for node in result['raw_nodes']]
    return dict(schema='paired-scope-prefix-complete-v1', state='scores_frozen_not_evaluated',
        readouts=readouts, budget=manifest['budget'],
        costs=dict(actual_forward_passes=sum(r['actual_forward_passes'] for r in results.values()),
            attempted_model_calls=sum(r['attempted_model_calls'] for r in results.values()),
            input_tokens=sum(r['input_tokens'] for r in results.values()), output_tokens=0,
            seconds_forward_and_capture=math.fsum(r['seconds_forward'] for r in results.values()),
            peak_cuda_allocated_bytes=max((n['peak_cuda_allocated_bytes'] for n in raw_nodes), default=0),
            peak_cuda_reserved_bytes=max((n['peak_cuda_reserved_bytes'] for n in raw_nodes), default=0)),
        all_four_arms_retained=True, calibrated_truth_probability=False, method_promoted=False,
        evaluation_material_accessed=False, automatic_retry=False, h013_failed_record_modified=False)


def qualify(preparation, output, backend, model_identity, *, measure_resources=None):
    """Consume a sealed token-only plan; one immutable attempt, no references."""
    preparation, output = _path(preparation), _path(output)
    _require(not output.exists() and 'outputs' in output.parts and not output.is_relative_to(preparation)
        and not preparation.is_relative_to(output), 'Fresh independent observation namespace required')
    initial = _flat_snapshot(preparation, PREPARATION_FILES)
    manifest = data.read_json(preparation / 'manifest.json')
    identity = _model_identity(model_identity)
    _require(manifest['model_identity'] == identity == getattr(backend, 'model_identity', None), 'Backend/manifest identity mismatch')
    _require(getattr(backend, 'admitted', False) is True, 'Explicit resource admission required')
    inputs, protocol_path, model_path = map(_path, (manifest['inputs_path'], manifest['protocol_path'], manifest['model_path']))
    frozen = _protocol(protocol_path)
    _require(data.file_hash(inputs) == frozen['original_inputs_sha256']
        and frozen == manifest['protocol'] and data.file_hash(protocol_path) == manifest['protocol_sha256'], 'Original input/protocol changed')
    donor = _engineering(manifest['engineering']['root'], identity)
    _require(donor == manifest['engineering'] and donor['snapshot']['seal_sha256'] == frozen['engineering_seal_sha256'],
             'Initial qualified cache seal changed')
    tokenizer = _tokenizer(model_path, identity)
    _require(_build(inputs, protocol_path, frozen, model_path, identity, donor, tokenizer) == manifest,
             'Complete tokenizer-only manifest reconstruction changed')
    source = data.read_json(preparation / 'source_identity.json')
    completed_identity(source)
    model_guard = _model_guard(model_path, identity)
    _require(not any(q['reused_node_ids'] for q in manifest['queries'].values()),
             'Cached tree needs explicit zero-forward import; no cached node may be rerun')
    output.mkdir(parents=True)
    source_files = [('manifest.json', manifest), ('protocol.json', frozen), ('model_identity.json', identity),
        ('source_identity.json', source), ('preparation.json', dict(path=str(preparation), snapshot=initial))]
    for name, value in source_files:
        data.write_json(output / name, value)
    for key, query in manifest['queries'].items():
        child = output / 'queries' / key
        child.mkdir(parents=True)
        data.write_json(child / 'request.json', query)
    planned = _inventory(output, manifest['queries'], frozen=False)
    hashes = {name: data.file_hash(output / name) for name in planned - {'freeze.json'}}
    data.write_json(output / 'freeze.json', dict(hashes))
    hashes['freeze.json'] = data.file_hash(output / 'freeze.json')
    observed, results, seal_hash = set(), {}, None
    measure = measure_resources or _allocator
    def unchanged(pending=None, complete=False):
        _inventory(output, manifest['queries'], observed, pending, complete=complete, sealed=seal_hash is not None)
        _require(all(data.file_hash(output / name) == digest for name, digest in hashes.items()), 'Initially retained observation file changed')
        _require(_flat_snapshot(preparation, PREPARATION_FILES) == initial
            and _engineering(donor['root'], identity) == donor
            and data.file_hash(inputs) == frozen['original_inputs_sha256']
            and data.file_hash(protocol_path) == manifest['protocol_sha256']
            and getattr(backend, 'model_identity', None) == identity, 'Initial inputs/protocol/model/cache/preparation changed')
        if seal_hash is not None:
            _require(data.file_hash(output / 'seal.json') == seal_hash, 'Initial root seal changed')
        model_guard(); completed_identity(source)
    try:
        # All32 tries register before _ensure_model; no subset admission.
        for key, query in manifest['queries'].items():
            checked = backend.validate_trie(deepcopy(query['messages']), list(query['candidates']))
            _require(checked == query['trie'], 'Admitted backend tokenizer changed a frozen query')
        unchanged()
        for key, query in manifest['queries'].items():
            unchanged()
            child = output / 'queries' / key
            before = measure()
            result = backend.score_trie(deepcopy(query['trie']))
            try:
                after = measure()
            except Exception as error:
                after = dict(measurement_failed=True, error_type=type(error).__name__, reason=str(error))
            result['collector_current_allocator'] = dict(before=before, after=after)
            serial = _persist(child, result)
            unchanged(pending=key)
            replay = _replay(child, query['trie'], serial, manifest['full_vocabulary_size'])
            data.write_json(child / 'replay.json', dict(same_kernel_replay=True,
                independent_mathematical_implementation=False, numpy_allow_pickle=False, result=replay))
            _require(data.read_json(child / 'raw.json') == serial
                and data.read_json(child / 'request.json') == query, 'Disk raw/request differs from captured memory')
            data.write_json(child / 'seal.json', {name: data.file_hash(child / name)
                for name in ('request.json', 'raw.json', 'vectors.npz', 'replay.json')})
            observed.add(key); results[key] = serial
            for name in ('request.json', 'raw.json', 'vectors.npz', 'replay.json', 'seal.json'):
                hashes[f'queries/{key}/{name}'] = data.file_hash(child / name)
            unchanged()
        complete = _complete(manifest, results)
        _require(complete['costs']['actual_forward_passes'] == complete['costs']['attempted_model_calls']
            == manifest['budget']['new_forward_passes']
            and complete['costs']['input_tokens'] == manifest['budget']['planned_input_tokens'], 'Frozen actual cost population changed')
        data.write_json(output / 'complete.json', complete); hashes['complete.json'] = data.file_hash(output / 'complete.json')
        unchanged(complete=True)
        data.write_json(output / 'seal.json', hashes); seal_hash = data.file_hash(output / 'seal.json')
        _require(data.read_json(output / 'seal.json') == hashes, 'Final seal differs from initially retained payloads')
        unchanged(complete=True)
        return complete
    except BaseException as error:
        (output / 'complete.json').unlink(missing_ok=True)
        (output / 'seal.json').unlink(missing_ok=True)
        data.write_json(output / 'failure.json', dict(schema='paired-scope-failure-v1',
            status='failed_retained_no_retry', error_type=type(error).__name__, reason=str(error),
            traceback=traceback.format_exc(), sealed_query_receipts=len(observed), automatic_retry=False,
            actual_counts_only_in_retained_raw=True, evaluation_material_accessed=False))
        raise
