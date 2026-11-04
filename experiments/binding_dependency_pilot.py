"""Qualification of numeric dependency routing, before whole-detector integration."""

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from experiments import data
from experiments.constraint_pilot import CASES, check_inputs
from experiments.fit_ablation import verify_seal
from experiments.native_dataset import read_native
from experiments.provenance import completed_identity, research_identity
from experiments.research.binding_dependency import dependency_checks, dependency_readout
from experiments.research.binding_witnesses import (
    binding_pairs, binding_probe_pair, binding_probes, swap_year_columns,
)


def run(native, output, protocol, reference=None):
    import torch
    from transformers import AutoTokenizer

    from experiments.research.attention import attention_profiles
    from myproject.inputs import encode_response, encode_source
    from myproject.model import load_model

    plan, samples, observations = read_native(native)
    selected = [(s, r) for (_, s), r in zip(samples, observations, strict=True) if s.id in CASES]
    if {s.id for s, _ in selected} != set(CASES):
        raise ValueError('All declared original cases are required')
    output.mkdir(parents=True, exist_ok=False)
    identity = research_identity()
    model_hashes = {p: data.file_hash(Path(p)) for p in plan['model_files_sha256']}
    if model_hashes != plan['model_files_sha256']:
        raise ValueError('Frozen model changed')
    metadata = {'source': identity, 'native': str(native.resolve()),
        'native_seal_sha256': data.file_hash(native/'seal.json'),
        'protocol': str(protocol.resolve()), 'protocol_sha256': data.file_hash(protocol),
        'model_files_sha256': model_hashes, 'annotation_reads': 0,
        'scope': 'stage1 source-certified dependency routing; no natural token benchmark'}
    if reference is not None:
        verify_seal(reference); verify_seal(reference/'scores')
        reference_plan = data.read_json(reference/'plan.json')
        if reference_plan.get('reference') or reference_plan['native_seal_sha256'] != metadata['native_seal_sha256']:
            raise ValueError('Require the unchanged stage1 observation population')
        metadata.update(reference=str(reference.resolve()), reference_seal_sha256=data.file_hash(reference/'seal.json'),
                        reference_score_seal_sha256=data.file_hash(reference/'scores/seal.json'),
                        scope='stage2 coherent source reassignment of fixed queries; no internal source-use claim')
    data.write_json(output/'plan.json', metadata)
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], use_fast=True, local_files_only=True)
    cases = []
    for sample, original in selected:
        pairs = binding_pairs(sample.source)
        queries = binding_probes(sample, pairs, distractors=True)
        pair = binding_probe_pair(sample, pairs)
        reassigned = reference is not None and pair is not None
        source = swap_year_columns(sample.source, pair) if reassigned else sample.source
        if reassigned:
            changed = encode_source(tokenizer, sample.prompt.replace(sample.source, source), source)
            original_prompt = encode_source(tokenizer, sample.prompt, sample.source)
            if Counter(changed['prompt_with_source']) != Counter(original_prompt['prompt_with_source']):
                raise ValueError('Source column reassignment changed the prompt tokenizer bag')
            for query in queries:
                query['source_context_changed'] = True
                if 'binding_risk' in query:
                    old_risk = query['binding_risk']
                    query['binding_prior_risk'] = (old_risk if query['binding_coherence'] else 1-old_risk) ^ 1
                    query['binding_risk'] = old_risk ^ 1
        encoded = [encode_response(tokenizer, q['text']) for q in queries]
        for placement in ['near', 'far']:
            controls = [e['answer_ids'] for q, e in zip(queries, encoded) if q.get('placement') == placement]
            if controls and any(Counter(e) != Counter(controls[0]) for e in controls[1:]):
                raise ValueError('Distractor controls changed tokenizer multiset')
        cases.append({'id': sample.id, 'source': source, 'source_reassigned': reassigned,
            'available': len(pairs) >= 3, 'pairs': pairs,
            'queries': [q | {'encoded': e} for q, e in zip(queries, encoded, strict=True)]})
        with np.load(native/original['file']) as raw:
            ids = np.asarray(encoded[0]['answer_ids'])
            offsets = np.asarray(encoded[0]['offsets'])
            target = np.flatnonzero((offsets[:, 1] > offsets[:, 0]) & ~np.isin(ids, encoded[0]['special_ids']))
            unit_index = np.repeat(np.arange(len(encoded[0]['units'])),
                                  [u['stop']-u['start'] for u in encoded[0]['units']])[target]
            if (not np.array_equal(encoded[0]['answer_ids'], raw['full_token_id']) or
                    not np.array_equal(offsets, raw['offsets']) or
                    not np.array_equal(target, raw['target']) or
                    not np.array_equal(unit_index, raw['unit_index'])):
                raise ValueError('Original token identity changed')
    data.write_json(output/'inputs.json', {'cases': cases})
    if reference is not None:
        check_fixed_queries(reference, cases, metadata)
    torch.manual_seed(42)
    model = load_model(Path(plan['model']), 'cuda:0')
    counts = {'backbone_calls': 0, 'processed_tokens': 0}

    def count(_module, _args, kwargs):
        counts['backbone_calls'] += 1
        counts['processed_tokens'] += kwargs['input_ids'].numel()

    handle = model.model.register_forward_pre_hook(count, with_kwargs=True)
    root = output/'observations'; root.mkdir()
    records, start = [], time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    try:
        for case, (sample, _) in zip(cases, selected, strict=True):
            entry = {'id': case['id'], 'available': case['available'], 'queries': []}
            if case['available']:
                text = sample.prompt.replace(sample.source, case['source'])
                prompt = encode_source(tokenizer, text, case['source'])['prompt_with_source']
                for index, query in enumerate(case['queries']):
                    encoded = query['encoded']
                    result = attention_profiles(model, prompt, encoded['answer_ids'])
                    p, n = result['prompt_tokens'], len(encoded['answer_ids'])
                    file = root/f'{len(records)}_{index}.npz'
                    with file.open('xb') as out:
                        np.savez_compressed(out, fit=result['fit'][:, p:p+n],
                            hold=result['hold'][:, p:p+n], token_id=encoded['answer_ids'], offsets=encoded['offsets'])
                    entry['queries'].append({'name': query['name'], 'file': str(file.relative_to(output)),
                                             'sha256': data.file_hash(file)})
                print('COLLECTED', case['id'], len(entry['queries']), flush=True)
            records.append(entry)
    finally:
        handle.remove()
    torch.cuda.synchronize()
    check_inputs(native, protocol, metadata)
    if any(data.file_hash(Path(p)) != h for p, h in model_hashes.items()):
        raise ValueError('Model changed during collection')
    data.write_json(output/'collection.json', {'records': records, 'counts': counts,
        'seconds': time.monotonic()-start, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'source': completed_identity(identity), 'annotation_reads': 0})
    data.write_json(output/'seal.json', {n: data.file_hash(output/n) for n in ['plan.json', 'inputs.json', 'collection.json']})
    score(output)


def score(root):
    verify_seal(root)
    plan = data.read_json(root/'plan.json')
    identity = research_identity()
    if identity != plan['source']:
        raise ValueError('Source changed before dependency scoring')
    cases = {c['id']: c for c in data.read_json(root/'inputs.json')['cases']}
    destination = root/'scores'; destination.mkdir(exist_ok=False)
    records = []
    for entry in data.read_json(root/'collection.json')['records']:
        case = cases[entry['id']]
        by_name = {q['name']: q for q in case['queries']}
        for query in entry['queries']:
            if data.file_hash(root/query['file']) != query['sha256']:
                raise ValueError('Raw attention changed')
            meta = by_name[query['name']]
            with np.load(root/query['file']) as raw:
                history = (raw['fit']+raw['hold'])/2
                values, edges = dependency_readout(meta['text'], raw['offsets'],
                    meta['encoded']['units'], history, case['source'])
                file = destination/f'{len(records)}.npz'
                with file.open('xb') as out:
                    np.savez_compressed(out, **values, token_id=raw['token_id'], offsets=raw['offsets'])
            records.append({'id': entry['id'], 'name': query['name'], 'edges': edges,
                'file': str(file.relative_to(root)), 'sha256': data.file_hash(file)})
    data.write_json(destination/'complete.json', {'records': records, 'source': completed_identity(identity),
                                                 'annotation_reads': 0})
    data.write_json(destination/'seal.json', {'complete.json': data.file_hash(destination/'complete.json')})
    verify_seal(root); verify_seal(destination)
    # Programmatic diagnostic labels/antecedents enter only after all scores are sealed.
    evaluated = scored_rows(root)
    check_inputs(Path(plan['native']), Path(plan['protocol']), plan)
    data.write_json(root/'evaluation.json', {'checks': dependency_checks(evaluated), 'cases': evaluated,
        'unavailable': [c['id'] for c in cases.values() if not c['available']],
        'source': completed_identity(identity), 'human_annotation_reads': 0})
    if plan.get('reference'):
        reference = Path(plan['reference'])
        check_fixed_queries(reference, list(cases.values()), plan)
        cross_sources(reference, root)


def replay(observations, output, protocol):
    """Reuse sealed physical states without attributing them to new source code."""
    verify_seal(observations)
    original_plan = data.read_json(observations/'plan.json')
    collection = data.read_json(observations/'collection.json')
    check_inputs(Path(original_plan['native']), Path(original_plan['protocol']), original_plan)
    output.mkdir(parents=True, exist_ok=False)
    identity = research_identity()
    metadata = original_plan | {'source': identity, 'observation_source': collection['source'],
        'observations': str(observations.resolve()), 'observation_seal_sha256': data.file_hash(observations/'seal.json'),
        'protocol': str(protocol.resolve()), 'protocol_sha256': data.file_hash(protocol),
        'scope': 'new scorer on sealed physical attention; no new backbone calls'}
    data.write_json(output/'plan.json', metadata)
    data.write_json(output/'inputs.json', data.read_json(observations/'inputs.json'))
    for case in collection['records']:
        for query in case['queries']:
            file = (observations/query['file']).resolve()
            if data.file_hash(file) != query['sha256']:
                raise ValueError('Cached physical attention changed')
            query['file'] = str(file)
    data.write_json(output/'collection.json', {'records': collection['records'],
        'source': collection['source'], 'replay_source': completed_identity(identity),
        'counts': {'backbone_calls': 0, 'processed_tokens': 0},
        'reused_counts': collection.get('reused_counts', collection['counts'])})
    data.write_json(output/'seal.json', {n: data.file_hash(output/n) for n in ['plan.json', 'inputs.json', 'collection.json']})
    score(output)
    verify_seal(observations)
    if data.file_hash(observations/'seal.json') != metadata['observation_seal_sha256']:
        raise ValueError('Original observation seal changed during replay')


def check_fixed_queries(reference, cases, metadata):
    verify_seal(reference); verify_seal(reference/'scores')
    if (data.file_hash(reference/'seal.json') != metadata['reference_seal_sha256'] or
            data.file_hash(reference/'scores/seal.json') != metadata['reference_score_seal_sha256']):
        raise ValueError('Reference observation/score seal changed')
    previous = data.read_json(reference/'inputs.json')['cases']
    for old, new in zip(previous, cases, strict=True):
        if old['id'] != new['id'] or old['available'] != new['available']:
            raise ValueError('Fixed query population changed')
        for before, after in zip(old['queries'], new['queries'], strict=True):
            if any(json.dumps(before.get(k), sort_keys=True) != json.dumps(after.get(k), sort_keys=True) for k in
                   ['name', 'text', 'encoded', 'probe_spans', 'intended_values']):
                raise ValueError('Fixed query text, axes or diagnostic targets changed')


def scored_rows(root):
    """Join programmatic targets to sealed scores; derived reports are not inputs."""
    verify_seal(root); verify_seal(root/'scores')
    cases = {c['id']: c for c in data.read_json(root/'inputs.json')['cases']}
    rows = data.read_json(root/'scores/complete.json')['records']
    expected = {(r['id'], q['name']) for r in data.read_json(root/'collection.json')['records'] for q in r['queries']}
    if len(rows) != len(expected) or {(r['id'], r['name']) for r in rows} != expected:
        raise ValueError('Require exactly the sealed observation population')
    joined = []
    for row in rows:
        if data.file_hash(root/row['file']) != row['sha256']:
            raise ValueError('Dependency score file changed')
        case = cases[row['id']]
        query = next(q for q in case['queries'] if q['name'] == row['name'])
        joined.append(row | {'source_reassigned': case.get('source_reassigned', False)} |
                      {k: query[k] for k in ['probe_spans', 'intended_values'] if k in query})
    return joined


def cross_sources(reference, root):
    """Cross physical old/new attention with old/new external certificates."""
    for path in [reference, root, reference/'scores', root/'scores']:
        verify_seal(path)
    inputs = [data.read_json(p/'inputs.json')['cases'] for p in [reference, root]]
    collections = [data.read_json(p/'collection.json')['records'] for p in [reference, root]]
    records, query_count = [], 0
    for before, after, old_raw, new_raw in zip(*inputs, *collections, strict=True):
        if (len({before['id'], after['id'], old_raw['id'], new_raw['id']}) != 1 or
                before['available'] != after['available']):
            raise ValueError('Source contrast changed query coverage')
        for old, new in zip(before['queries'], after['queries'], strict=True):
            if old['name'] != new['name'] or old['text'] != new['text'] or old['encoded'] != new['encoded']:
                raise ValueError('Source contrast changed the fixed answer axes')
        for old, new in zip(old_raw['queries'], new_raw['queries'], strict=True):
            if old['name'] != new['name']:
                raise ValueError('Source contrast changed query ordering')
            query_count += 1
            meta = next(q for q in before['queries'] if q['name'] == old['name'])
            quadrants = {}
            for i, (directory, query) in enumerate([(reference, old), (root, new)]):
                if data.file_hash(directory/query['file']) != query['sha256']:
                    raise ValueError('Source contrast raw features changed')
                with np.load(directory/query['file']) as raw:
                    if not np.array_equal(raw['token_id'], meta['encoded']['answer_ids']) or not np.array_equal(
                            raw['offsets'], meta['encoded']['offsets']):
                        raise ValueError('Source contrast raw token axes changed')
                    if not meta.get('intended_values'):
                        continue
                    for j, case in enumerate([before, after]):
                        _, edges = dependency_readout(meta['text'], raw['offsets'], meta['encoded']['units'],
                            (raw['fit']+raw['hold'])/2, case['source'])
                        quadrants[f'attention{i}_source{j}'] = edges
            if not quadrants:
                continue
            for slot, (a, b) in enumerate(meta['probe_spans']):
                observations = {}
                for key, edges in quadrants.items():
                    edge = next((e for e in edges if e['start'] == a and e['end'] == b), None)
                    observations[key] = edge['relation_mass']['dependency'] if edge else None
                old_mass, new_mass = observations['attention0_source0'], observations['attention1_source1']
                delta = None if old_mass is None or new_mass is None else new_mass['conflict']-old_mass['conflict']
                signed = None if delta is None else delta*(1-2*meta['binding_risk'])
                records.append({'name': meta['name'], 'slot': slot, 'original_risk': meta['binding_risk'],
                    'cross': observations, 'signed_source_flip': signed})
    paired_passed = len(records) == 32 and all(r['signed_source_flip'] is not None and
        r['signed_source_flip'] > 0 for r in records)
    stage_passed = (paired_passed and query_count == 27 and all(dependency_checks(scored_rows(p))[
        'mechanism_passed'] for p in [reference, root]))
    data.write_json(root/'source_cross.json', {'records': records, 'fixed_collected_queries': query_count,
        'paired_flip_passed': paired_passed, 'stage2_mechanism_passed': stage_passed,
        'internal_source_use_claim': False,
        'scope': 'certificate-only flips are expected; the cross does not prove attention internally verifies evidence',
        'source': completed_identity(data.read_json(root/'plan.json')['source'])})
