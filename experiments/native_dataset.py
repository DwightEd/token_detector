"""Cached observations for the shared mainline and explicit legacy R4 comparisons."""

from pathlib import Path
from collections import Counter
import time

import numpy as np

from experiments import data
from experiments.datasets import _check_spans, read_jsonl, read_samples
from experiments.fit_ablation import verify_seal
from experiments.provenance import completed_identity, research_identity
from experiments.research.r4 import Detector
from experiments.research.distribution import DISTRIBUTION_FIELDS
from experiments.research.readouts import METHODS, SEQUENCE_METHODS, SHARED_METHODS, WITNESS_METHODS, SharedDetector

FIELDS = tuple(k for k in data.FIELDS if k != 'development')
STAGES = ['prior', 'source', 'base', 'unary', 'score']
CONTRASTS = {'adaptation': ('frozen_score', 'adapted_score'),
             'shared_stage': ('point', 'frozen_prior'),
             'later_stages': ('frozen_prior', 'frozen_score')}


def prepared_inputs(groups):
    samples = []
    for index, group in enumerate(groups):
        root = Path(group['path'])
        manifest = data.read_json(root/'manifest.json')
        if (data.file_hash(root/'inputs.jsonl') != group['input_sha256']
                or data.file_hash(root/'manifest.json') != group['manifest_sha256']
                or group['input_sha256'] != manifest['inputs_sha256']):
            raise ValueError('Prepared inference input or manifest changed')
        for file, expected in manifest['raw_files'].items():
            if data.file_hash(Path(file)) != expected:
                raise ValueError('Dataset acquisition changed')
        rows = list(read_samples(root/'inputs.jsonl'))
        if not rows or len(rows) != manifest['samples']:
            raise ValueError('Prepared sample count changed')
        for sample in rows:
            _ = sample.inputs
        if group['role'] == 'test' and (set(manifest['coverage']) != {'complete'}
                or manifest['selection']['limit'] is not None or manifest['selection'].get('ids')):
            raise ValueError('Use complete, unfiltered released test annotations')
        samples.extend((index, sample) for sample in rows)
    identities = [sample.id for _, sample in samples]
    if len(set(identities)) != len(identities):
        raise ValueError('Prepared splits share sample identities')
    sources = {role: {s.source_id for g, s in samples if groups[g]['role'] == role}
               for role in ['fit', 'test']}
    if sources['fit'] & sources['test']:
        raise ValueError('FIT and test share reference sources')
    if len(sources['fit']) != sum(groups[g]['role'] == 'fit' for g, _ in samples):
        raise ValueError('Use one FIT response per reference source')
    return samples


def collect(output, fit, test, model_path, task, protocol, method='shared'):
    import torch
    from experiments.inference import HallucinationDetector

    groups = []
    for role, roots in [('fit', fit), ('test', test)]:
        for root in roots:
            root = root.resolve()
            groups.append({'path': str(root), 'role': role,
                'input_sha256': data.file_hash(root/'inputs.jsonl'),
                'manifest_sha256': data.file_hash(root/'manifest.json')})
    samples = prepared_inputs(groups)
    if not test or any(s.task != task for _, s in samples):
        raise ValueError('Provide nonempty test inputs using one fixed task')
    output.mkdir(parents=True, exist_ok=False)
    identity = research_identity()
    model_files = sorted(p for p in model_path.iterdir() if p.is_file()
                         and (p.suffix in {'.json', '.safetensors', '.bin'} or p.name == 'tokenizer.model'))
    model_hashes = {str(p.resolve()): data.file_hash(p) for p in model_files}
    data.write_json(output/'plan.json', {'groups': groups, 'task': task,
        'model': str(model_path.resolve()), 'model_files_sha256': model_hashes,
        'reference': None, 'reference_sha256': None,
        'protocol': str(protocol.resolve()), 'protocol_sha256': data.file_hash(protocol),
        'source': identity, 'measurement_method': method, 'annotation_reads': 0})
    torch.manual_seed(42)
    detector = HallucinationDetector.from_pretrained(model_path, task=task, method=method)
    counts = {'backbone_calls': 0, 'processed_tokens': 0}

    def count(_module, _args, kwargs):
        counts['backbone_calls'] += 1
        counts['processed_tokens'] += kwargs['input_ids'].numel()

    handle = detector.model.model.register_forward_pre_hook(count, with_kwargs=True)
    destination = output/'observations'; destination.mkdir()
    torch.cuda.reset_peak_memory_stats()
    start, records = time.monotonic(), []
    try:
        for index, (group, sample) in enumerate(samples):
            pack, encoded = detector.measure(**sample.inputs, answer_index=index)
            if not all(np.isfinite(v).all() for v in pack.values()):
                raise ValueError('Nonfinite native observation')
            file = data.prediction_path(destination, sample.id).with_suffix('.npz')
            with file.open('xb') as stream:
                np.savez_compressed(stream, **pack, offsets=np.asarray(encoded['offsets']),
                    full_token_id=np.asarray(encoded['answer_ids']), special_ids=np.asarray(encoded['special_ids']))
            records.append({'id': sample.id, 'group': group, 'file': str(file.relative_to(output)),
                            'sha256': data.file_hash(file), 'response_sha256': data.text_hash(sample.response)})
            print('COLLECTED', index+1, len(samples), sample.id, flush=True)
    finally:
        handle.remove()
    torch.cuda.synchronize()
    prepared_inputs(groups)
    if any(data.file_hash(Path(p)) != h for p, h in model_hashes.items()):
        raise ValueError('Model bytes changed during native measurement')
    data.write_json(output/'collection.json', {'records': records, 'source': completed_identity(identity),
        'counts': counts, 'seconds': time.monotonic()-start,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'annotation_reads': 0})
    data.write_json(output/'seal.json', {n: data.file_hash(output/n) for n in ['plan.json', 'collection.json']})


def read_native(native):
    """Inspect sealed original observations, retaining their historical source identity."""
    verify_seal(native)
    plan = data.read_json(native/'plan.json')
    if (data.file_hash(Path(plan['protocol'])) != plan['protocol_sha256']
            or (plan.get('reference') is not None and
                data.file_hash(Path(plan['reference'])) != plan['reference_sha256'])):
        raise ValueError('Frozen protocol or references changed')
    samples = prepared_inputs(plan['groups'])
    records = data.read_json(native/'collection.json')['records']
    if len(samples) != len(records):
        raise ValueError('Native sample coverage changed')
    for index, (record, (group, sample)) in enumerate(zip(records, samples, strict=True)):
        if (record['id'] != sample.id or record['group'] != group
                or record['response_sha256'] != data.text_hash(sample.response)
                or data.file_hash(native/record['file']) != record['sha256']):
            raise ValueError('Native sample identity changed')
        with np.load(native/record['file']) as raw:
            offsets, ids, target = raw['offsets'], raw['full_token_id'], raw['target']
            if (offsets.shape != (len(ids), 2) or not np.issubdtype(offsets.dtype, np.integer)
                    or np.any(offsets < 0) or np.any(offsets > len(sample.response))
                    or np.any(offsets[:, 1] < offsets[:, 0])):
                raise ValueError('Invalid original character axis')
            wanted = np.flatnonzero((offsets[:, 1] > offsets[:, 0]) & ~np.isin(ids, raw['special_ids']))
            if (not np.array_equal(target, wanted) or not np.array_equal(raw['token_id'], ids[target])
                    or not np.all(raw['answer_index'] == index)):
                raise ValueError('Original valid token axis changed')
    return plan, samples, records


def collect_cached(output, prepared, cache, task, protocol):
    """Repackage verified primitive probabilities, never archived detector scores.

    Historical GPU weights/code are not authenticated by a current tokenizer
    match. This is explicitly a new readout evaluation on cached observations.
    """
    from transformers import AutoTokenizer

    from experiments.measure import cached_measurement
    from myproject.inputs import encode_response, encode_source
    from experiments.research.measurement import answer_rows

    prepared, cache = prepared.resolve(), cache.resolve()
    groups = [{'path': str(prepared), 'role': 'test',
        'input_sha256': data.file_hash(prepared/'inputs.jsonl'),
        'manifest_sha256': data.file_hash(prepared/'manifest.json')}]
    samples = prepared_inputs(groups)
    prepared_manifest = data.read_json(prepared/'manifest.json')
    if prepared_manifest.get('dataset') == 'ragtruth' and (
            len(samples) != 900 or len({s.source_id for _, s in samples}) != 150 or
            sorted(Counter(s.generator for _, s in samples).values()) != [150]*6):
        raise ValueError('This RAGTruth TEST protocol requires 900 answers, 150 sources and all six generators')
    manifest = data.read_json(cache/'manifest.json')
    rows = [r for r in manifest['records'] if r['task'] == task and r['split'] == 'test']
    by_id = {r['id']: r for r in rows}
    if (len(rows) != len(by_id) or set(by_id) != {s.id for _, s in samples}
            or any(s.task != task or s.split != 'test' for _, s in samples)):
        raise ValueError('Require the exact complete prepared test population')
    output.mkdir(parents=True, exist_ok=False)
    identity = research_identity()
    model = Path(manifest['model'])
    mapping_files = [p for p in model.iterdir() if p.is_file() and p.suffix == '.json']
    mapping_hashes = {str(p.resolve()): data.file_hash(p) for p in mapping_files}
    metadata = {'groups': groups, 'task': task, 'model': str(model),
        'model_files_sha256': mapping_hashes, 'reference': None, 'reference_sha256': None,
        'protocol': str(protocol.resolve()), 'protocol_sha256': data.file_hash(protocol),
        'cache': str(cache), 'cache_manifest_sha256': data.file_hash(cache/'manifest.json'),
        'source': identity, 'annotation_reads': 0, 'new_model_calls': 0,
        'observation_identity_scope': 'historical primitives; original GPU source/weight hashes unavailable; current tokenizer/config mapping checked'}
    data.write_json(output/'plan.json', metadata)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, use_fast=True)
    destination = output/'observations'; destination.mkdir()
    source_inputs, records, bindings = {}, [], {}
    start = time.monotonic()
    for index, (_, sample) in enumerate(samples):
        record = by_id[sample.id]
        if record['source_id'] != sample.source_id or record['generator'] != sample.generator:
            raise ValueError('Cache source or generator identity changed')
        response_file, source_file = cache/record['directory']/'response.json', cache/record['source_file']
        files = [response_file, source_file]+[cache/record['directory']/name for name in
            ['observations.npz', 'with_source.npz', 'without_source.npz']]
        for path in files:
            fingerprint = data.file_hash(path)
            if bindings.setdefault(str(path), fingerprint) != fingerprint:
                raise ValueError('A primitive input changed during collection')
        response = data.read_json(response_file)
        current = encode_response(tokenizer, sample.response)
        if (response['text'] != sample.response or any(
                not np.array_equal(current[k], response[k]) for k in ['answer_ids', 'offsets'])
                or current['units'] != response['units']
                or set(current['special_ids']) != set(response['special_ids'])):
            raise ValueError(f'Original response token, span or text changed: {sample.id}')
        if sample.source_id not in source_inputs:
            source_inputs[sample.source_id] = encode_source(tokenizer, sample.prompt, sample.source)
        if data.read_json(source_file) != source_inputs[sample.source_id]:
            raise ValueError('Original source prompt token IDs/mask differ')
        observed = cached_measurement(cache, record)
        with np.load(files[-2]) as present, np.load(files[-1]) as absent:
            n = len(current['answer_ids'])
            if (observed['delta'].shape != (n, 2) or observed['route'].shape != (n,) or
                    any(present[k].shape != (n,) or absent[k].shape != (n,) for k in ['local', 'full'])):
                raise ValueError('Primitive observations must match the full answer axis')
            if any(not np.array_equal(z['token_id'], current['answer_ids']) for z in [present, absent]):
                raise ValueError('Primitive answer token identity differs')
            delta = np.column_stack([absent[k].astype(float)-present[k] for k in ['local', 'full']])
            if (not np.array_equal(observed['delta'], delta) or
                    not np.array_equal(observed['route'], present['raw_route'])):
                raise ValueError('Cached observations differ from original primitive probabilities')
        pack = answer_rows(observed, response, index)
        if not all(np.isfinite(v).all() for v in pack.values()):
            raise ValueError('Nonfinite primitive measurement')
        file = destination/f'{index}.npz'
        with file.open('xb') as stream:
            np.savez_compressed(stream, **pack, offsets=response['offsets'],
                full_token_id=response['answer_ids'], special_ids=response['special_ids'])
        records.append({'id': sample.id, 'group': 0, 'file': str(file.relative_to(output)),
            'sha256': data.file_hash(file), 'response_sha256': data.text_hash(sample.response)})
        if (index+1) % 100 == 0 or index+1 == len(samples):
            print('CACHED', task, index+1, len(samples), flush=True)
    prepared_inputs(groups)
    if (any(data.file_hash(Path(p)) != h for p, h in (bindings | mapping_hashes).items()) or
            data.file_hash(cache/'manifest.json') != metadata['cache_manifest_sha256'] or
            data.file_hash(protocol) != metadata['protocol_sha256']):
        raise ValueError('A cached collection input changed')
    data.write_json(output/'collection.json', {'records': records, 'source': completed_identity(identity),
        'primitive_files_sha256': bindings, 'seconds': time.monotonic()-start,
        'counts': {'backbone_calls': 0, 'processed_tokens': 0}, 'annotation_reads': 0})
    data.write_json(output/'seal.json', {p: data.file_hash(output/p) for p in ['plan.json', 'collection.json']})


def validate(native):
    """Replaying the frozen scorer additionally requires the exact original runtime/code."""
    if research_identity() != data.read_json(native/'plan.json')['source']:
        raise ValueError('Frozen source/runtime, protocol or references changed')
    return read_native(native)


def qualify_proofs(native, output, protocol):
    """Source-only compiler coverage gate, before reading response annotations."""
    from experiments.research.proofs import compile_proofs

    plan, samples, observed = read_native(native)
    identity = research_identity()
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output/'plan.json', {'native': str(native.resolve()),
        'native_seal_sha256': data.file_hash(native/'seal.json'), 'source': identity,
        'protocol': str(protocol.resolve()), 'protocol_sha256': data.file_hash(protocol),
        'annotation_reads': 0})
    records, counts = [], Counter()
    for record, (group, sample) in zip(observed, samples, strict=True):
        if plan['groups'][group]['role'] != 'test':
            continue
        with np.load(native/record['file']) as raw:
            proof = compile_proofs(sample.source, sample.response, raw['offsets'], raw['target'])
        records.append({'id': sample.id, 'group': group, 'proofs': proof})
        counts.update(proof['diagnostics']['counts'])
    data.write_json(output/'coverage.json', {'records': records, 'claims': dict(counts),
        'responses': len(records), 'incompatible_responses': sum(bool(r['proofs']['components']) for r in records),
        'annotation_reads': 0, 'source': completed_identity(identity)})
    data.write_json(output/'seal.json', {name: data.file_hash(output/name) for name in ['plan.json', 'coverage.json']})
    print('PROOF_COVERAGE', len(records), dict(counts), flush=True)


def score(native, output, method='shared', compare_r4=False, protocol=None, ablate_witness=False):
    reader = read_native if method in SEQUENCE_METHODS else validate
    plan, samples, records = reader(native)
    if method not in METHODS:
        raise ValueError('Unknown native readout method')
    if compare_r4 and method == 'r4':
        raise ValueError('compare_r4 is only for the no-FIT shared comparison')
    if method in SEQUENCE_METHODS and protocol is None:
        raise ValueError('Persistent replay requires its own frozen scoring protocol')
    if ablate_witness and method != 'witness':
        raise ValueError('Witness controls require the fixed witness readout')
    identity = research_identity()
    fit, reference = [], None
    if method == 'r4':
        for record in records:
            if plan['groups'][record['group']]['role'] == 'fit':
                with np.load(native/record['file']) as raw:
                    fit.append({key: raw[key] for key in FIELDS})
        if not fit:
            raise ValueError('The legacy R4 FIT comparison requires FIT observations')
        combined = {key: np.concatenate([p[key] for p in fit]) for key in FIELDS}
        reference = Path(plan.get('reference') or
                         Path(__file__).parents[1]/'myproject/references'/f"{plan['task']}.npz")
        detectors = {'frozen': Detector.load(reference),
                     'adapted': Detector(plan['task']).fit(combined)}
        systems = ['point', 'local_gap', 'full_gap']+[f'{m}_{s}' for m in detectors for s in STAGES]
        contrasts = CONTRASTS
    else:
        detectors = {'shared': SharedDetector(plan['task'])}
        systems = ['full_gap', 'shared_score']
        contrasts = {'sharing': ('full_gap', 'shared_score')}
        if method in SEQUENCE_METHODS:
            detectors[method] = SharedDetector(plan['task'], method=method)
            systems.append(f'{method}_score')
            contrasts[method] = ('shared_score', f'{method}_score')
            if method in WITNESS_METHODS:
                detectors['persistent'] = SharedDetector(plan['task'], method='persistent')
                systems.append('persistent_score')
                if method == 'witness':
                    contrasts['witness_vs_persistent'] = ('persistent_score', 'witness_score')
                else:
                    systems.append('witness_score')
                    contrasts['responsibility_vs_witness'] = ('witness_score', 'responsibility_score')
            if method in {'matched_local', 'symmetric_local', 'fixed_local', 'fixed_persistence'}:
                controls = ['witness'] + (['symmetric_local'] if method == 'matched_local' else [])
                for control in controls:
                    detectors[control] = SharedDetector(plan['task'], method=control)
                    systems.append(f'{control}_score')
                    contrasts[f'{method}_vs_{control}'] = (f'{control}_score', f'{method}_score')
                contrasts[f'{method}_vs_raw'] = ('full_gap', f'{method}_score')
            if method == 'conditional':
                detectors['witness'] = SharedDetector(plan['task'], method='witness')
                systems.extend(['witness_score', *DISTRIBUTION_FIELDS])
                systems.append('conditional_rank_score')
                contrasts['conditional_vs_witness'] = ('witness_score', 'conditional_score')
                contrasts['conditional_vs_raw'] = ('calibrated_gap', 'conditional_score')
                contrasts['stabilization'] = ('conditional_rank_score', 'conditional_score')
            if method == 'onset':
                detectors['witness'] = SharedDetector(plan['task'], method='witness')
                systems.extend(['witness_score', 'onset_entry_score'])
                contrasts['onset_vs_witness'] = ('witness_score', 'onset_score')
            if method == 'token_state':
                detectors['witness'] = SharedDetector(plan['task'], method='witness')
                systems.extend(['witness_score', 'clipped_score', 'witness_mean_score',
                                'token_state_mean_score'])
                contrasts = {f'token_state_vs_{key}': (key, 'token_state_score') for key in
                             ['witness_score', 'clipped_score', 'full_gap', 'token_state_mean_score']}
                contrasts['witness_vs_mean'] = ('witness_mean_score', 'witness_score')
            if method == 'adaptive_token':
                systems.extend(['witness_score', 'token_state_score', 'uniform_model_score',
                                'clipped_score', 'witness_mean_score', 'adaptive_token_mean_score'])
                contrasts = {f'adaptive_vs_{key}': (key, 'adaptive_token_score') for key in
                             ['witness_score', 'token_state_score', 'uniform_model_score',
                              'clipped_score', 'adaptive_token_mean_score']}
            if method == 'proof_revision':
                systems.extend(['witness_score', 'proof_hard_score'])
                contrasts = {'bounded_vs_witness': ('witness_score', 'proof_revision_score'),
                             'bounded_vs_hard': ('proof_hard_score', 'proof_revision_score'),
                             'hard_vs_witness': ('witness_score', 'proof_hard_score')}
        if compare_r4:
            reference = Path(__file__).parents[1]/'myproject/references'/f"{plan['task']}.npz"
            frozen_r4 = Detector.load(reference)
            systems.append('frozen_score')
            contrasts['versus_r4'] = ('frozen_score', 'shared_score')
            if method in SEQUENCE_METHODS:
                systems.extend(f'frozen_{s}' for s in ['prior', 'source', 'base', 'unary'])
                contrasts['candidate_versus_r4'] = ('frozen_score', f'{method}_score')
        if ablate_witness:
            controls = ['no_copy_score', 'clipped_score', 'permuted_score']
            systems.extend(controls)
            contrasts = {k: (k, 'witness_score') for k in controls}
    output.mkdir(parents=True, exist_ok=False)
    if method == 'r4':
        detectors['adapted'].save(output/'adapted_references.npz')
    data.write_json(output/'plan.json', {'native': str(native.resolve()),
        'native_seal_sha256': data.file_hash(native/'seal.json'), 'source': identity,
        'method': method, 'primary': 'frozen_score' if method == 'r4' else
        f'{method}_score' if method in SEQUENCE_METHODS else 'shared_score',
        'observation_source': plan['source'],
        'witness_controls': ablate_witness,
        'localization_systems': (['full_gap', 'shared_score', 'witness_score', f'{method}_score', 'frozen_score']
            if method in {'fixed_local', 'fixed_persistence'} else
            ['full_gap', 'witness_score', 'matched_local_score',
             'symmetric_local_score', 'frozen_score']
            if method in {'matched_local', 'symmetric_local'} else
            ['witness_score', *controls] if ablate_witness else
            ['full_gap', 'shared_score', 'witness_score', 'responsibility_score', 'frozen_score']
            if method == 'responsibility' else
            ['full_gap', 'calibrated_gap', 'conditional_score', 'witness_score', 'frozen_score']
            if method == 'conditional' else
            ['full_gap', 'witness_score', 'onset_score', 'onset_entry_score', 'frozen_score']
            if method == 'onset' else
            ['full_gap', 'witness_score', 'token_state_score', 'clipped_score',
             'witness_mean_score', 'token_state_mean_score', 'frozen_score']
            if method == 'token_state' else
            ['full_gap', 'witness_score', 'token_state_score', 'adaptive_token_score',
             'uniform_model_score', 'clipped_score', 'witness_mean_score',
             'adaptive_token_mean_score', 'frozen_score']
            if method == 'adaptive_token' else
            ['full_gap', 'witness_score', 'proof_hard_score', 'proof_revision_score', 'frozen_score']
            if method == 'proof_revision' else None),
        'scoring_protocol': str(protocol.resolve()) if protocol else None,
        'scoring_protocol_sha256': data.file_hash(protocol) if protocol else None,
        'reference': str(reference.resolve()) if reference is not None else None,
        'reference_sha256': data.file_hash(reference) if reference is not None else None,
        'systems': systems, 'contrasts': contrasts, 'annotation_reads': 0})
    predictions = []
    for record, (group, sample) in zip(records, samples, strict=True):
        if record['id'] != sample.id or record['group'] != group:
            raise ValueError('Sample order changed')
        if plan['groups'][group]['role'] != 'test':
            continue
        with np.load(native/record['file']) as raw:
            pack = {key: raw[key] for key in (*FIELDS, *DISTRIBUTION_FIELDS) if key in raw}
            proof_diagnostics = None
            if method == 'proof_revision':
                from experiments.research.revision import proof_pack

                pack = proof_pack(pack, sample.source, sample.response, raw['offsets'])
            values = {'full_gap': pack['delta'][:, 1]}
            if method == 'conditional':
                values.update({key: pack[key] for key in DISTRIBUTION_FIELDS})
                values['conditional_rank_score'] = SharedDetector(plan['task'], method='conditional').score(
                    {**pack, 'calibrated_gap': pack['rank_gap']})
            if method == 'r4':
                values['local_gap'] = pack['delta'][:, 0]
            for name, detector in detectors.items():
                if method != 'r4':
                    parts = detector.score_components(pack)
                    values[f'{name}_score'] = parts['score']
                    if name == 'responsibility':
                        values['witness_score'] = parts['witness_score']
                    if name == 'onset':
                        values['onset_entry_score'] = parts['entry_score']
                    if name == 'adaptive_token':
                        values.update({key: parts[key] for key in ['witness_score', 'token_state_score',
                                       'uniform_model_score', 'deviation_probability']})
                    if name == 'proof_revision':
                        values.update(witness_score=parts['witness_score'], proof_hard_score=parts['hard_score'])
                        proof_diagnostics = parts['proof_diagnostics']
                    continue
                stages = detector.score_components(pack)
                values['point'] = stages['point']
                values.update({f'{name}_{key}': stages['final' if key == 'score' else key] for key in STAGES})
            if compare_r4:
                stages = frozen_r4.score_components(pack)
                values['frozen_score'] = stages['final']
                if method in SEQUENCE_METHODS:
                    values.update({f'frozen_{s}': stages[s] for s in ['prior', 'source', 'base', 'unary']})
            if ablate_witness:
                from experiments.research.sequence_controls import grouped_witness_controls

                values.update(grouped_witness_controls(pack['delta'][:, 1], pack['answer_index'], pack['unit_index']))
            if method == 'token_state':
                from experiments.research.sequence_controls import token_state_controls

                values.update(token_state_controls(pack, values['witness_score'], values['token_state_score']))
            if method == 'adaptive_token':
                from experiments.research.sequence_controls import token_state_controls

                values.update(token_state_controls(pack, values['witness_score'],
                                                   values['adaptive_token_score'], candidate='adaptive_token'))
            file = output/Path(record['file']).name
            with file.open('xb') as stream:
                np.savez_compressed(stream, **values,
                    **{k: raw[k] for k in ['target', 'token_id', 'answer_index', 'unit_index', 'offsets']})
        predictions.append({**record, 'file': file.name, 'sha256': data.file_hash(file),
                            **({'proof_diagnostics': proof_diagnostics} if proof_diagnostics is not None else {})})
    reader(native)
    if protocol and data.file_hash(protocol) != data.read_json(output/'plan.json')['scoring_protocol_sha256']:
        raise ValueError('Scoring protocol changed during inference')
    data.write_json(output/'complete.json', {'records': predictions, 'source': completed_identity(identity),
                                            'fit_answers': len(fit), 'annotation_reads': 0})
    data.write_json(output/'seal.json', {p.name: data.file_hash(p) for p in output.iterdir()})


def evaluate(native, scores, output):
    from experiments.evaluate import paired_source_differences, ranking, token_labels

    identity = research_identity()
    plan, samples, native_records = read_native(native)
    verify_seal(scores)
    score_plan = data.read_json(scores/'plan.json')
    if (score_plan.get('scoring_protocol') and
            data.file_hash(Path(score_plan['scoring_protocol'])) != score_plan['scoring_protocol_sha256']):
        raise ValueError('Frozen candidate scoring protocol changed')
    if data.file_hash(native/'seal.json') != score_plan['native_seal_sha256']:
        raise ValueError('Score/native binding changed')
    records = data.read_json(scores/'complete.json')['records']
    expected = {s.id for g, s in samples if plan['groups'][g]['role'] == 'test'}
    if len(records) != len(expected) or {r['id'] for r in records} != expected:
        raise ValueError('Score sample coverage changed')
    native_by_id = {r['id']: r for r in native_records}
    by_id = {sample.id: sample for _, sample in samples}
    results = {}
    for group, specification in enumerate(plan['groups']):
        if specification['role'] != 'test':
            continue
        prepared = Path(specification['path'])
        manifest = data.read_json(prepared/'manifest.json')
        if data.file_hash(prepared/'annotations.jsonl') != manifest['annotations_sha256']:
            raise ValueError('Annotation provenance changed')
        annotation_rows = list(read_jsonl(prepared/'annotations.jsonl'))
        annotations = {a['id']: a for a in annotation_rows}
        if (len(annotations) != len(annotation_rows)
                or set(annotations) != {s.id for g, s in samples if g == group}):
            raise ValueError('Annotations require an exact test sample join')
        labels, source_ids, generators, arrays = [], [], [], {key: [] for key in score_plan['systems']}
        answer_axes, unit_axes = [], []
        typed_labels, positive_answers = [], 0
        for record in records:
            if record['group'] != group:
                continue
            sample, annotation = by_id[record['id']], annotations[record['id']]
            if (annotation['granularity'] != 'token' or annotation['coverage'] != 'complete'
                    or annotation['response_sha256'] != data.text_hash(sample.response)):
                raise ValueError('Complete character-span labels on original responses required')
            _check_spans(sample.response, annotation['spans'])
            if any(s['label'] != 1 for s in annotation['spans']):
                raise ValueError('This token format requires positive error spans and complete coverage')
            original = native_by_id[record['id']]
            if record['group'] != original['group']:
                raise ValueError('Score/native dataset group changed')
            with np.load(scores/record['file']) as raw, np.load(native/original['file']) as observed:
                for axis in ['target', 'token_id', 'answer_index', 'unit_index', 'offsets']:
                    if not np.array_equal(raw[axis], observed[axis]):
                        raise ValueError('Score/native original token axis changed')
                projected = token_labels({'labels': annotation['spans']}, {'offsets': raw['offsets']})[raw['target']]
                labels.append(projected)
                positive_answers += bool(projected.any())
                typed_labels.append({kind: token_labels({'labels': [s for s in annotation['spans']
                    if s.get('error_type', 'unspecified') == kind]}, {'offsets': raw['offsets']})[raw['target']]
                    for kind in {s.get('error_type', 'unspecified') for s in annotation['spans']}})
                source_ids.extend([sample.source_id]*len(projected))
                answer_axes.append(raw['answer_index']); unit_axes.append(raw['unit_index'])
                generators.extend([sample.generator]*len(projected))
                for key in arrays:
                    arrays[key].append(raw[key])
        label_parts = labels
        labels = np.concatenate(label_parts)
        arrays = {key: np.concatenate(parts) for key, parts in arrays.items()}
        categories = {}
        for kind in sorted({kind for row in typed_labels for kind in row}):
            mask = np.concatenate([row.get(kind, np.zeros_like(y))
                                   for row, y in zip(typed_labels, label_parts, strict=True)])
            selected = mask | ~labels
            categories[kind] = {'positives': int(mask.sum()), 'clean_tokens': int((~labels).sum()),
                'metrics': {m: ranking(mask[selected], arrays[f'{m}_score'][selected])
                            for m in (['shared', score_plan['method']] if score_plan.get('method') in SEQUENCE_METHODS else
                                      ['shared'] if score_plan.get('method') == 'shared' else ['frozen', 'adapted'])}}
        results[prepared.name] = {'tokens': len(labels), 'positives': int(labels.sum()),
            'responses': len(typed_labels), 'positive_responses': positive_answers,
            'error_types': categories,
            'sources': len(set(source_ids)), 'metrics': {key: ranking(labels, v) for key, v in arrays.items()},
            'by_generator': {g: {key: ranking(labels[np.asarray(generators) == g], v[np.asarray(generators) == g])
                                for key, v in arrays.items()} for g in sorted(set(generators))},
            'differences': {key: paired_source_differences(labels, arrays[a], arrays[b], np.asarray(source_ids))
                            for key, (a, b) in score_plan['contrasts'].items()},
            'annotation_sha256': data.file_hash(prepared/'annotations.jsonl'),
            'prompt_policy': manifest['prompt_policy'], 'label_provenance': sorted({a.get('label_provenance', '') for a in annotations.values()}),
            'calibration_saturation': {m: {edge: float(np.mean(arrays[f'{m}_source'] == value))
                for edge, value in [('zero', 0), ('one', 1)]}
                for m in ([] if score_plan.get('method') in SHARED_METHODS else ['frozen', 'adapted'])}}
        if score_plan.get('method') in SHARED_METHODS:
            results[prepared.name].pop('calibration_saturation')
        if score_plan.get('method') in SEQUENCE_METHODS:
            from experiments.evaluate import localization_diagnostics

            results[prepared.name]['localization'] = {key: localization_diagnostics(
                labels, arrays[key], np.concatenate(answer_axes), np.concatenate(unit_axes))
                for key in (score_plan.get('localization_systems') or
                            ['full_gap', 'shared_score', f"{score_plan['method']}_score", 'frozen_score']) if key in arrays}
        print(prepared.name, {k: (results[prepared.name]['metrics'][k]['auroc'],
                                  results[prepared.name]['metrics'][k]['ap'])
                             for k in (['full_gap', 'shared_score', f"{score_plan['method']}_score"] if score_plan.get('method') in SEQUENCE_METHODS
                                       else ['full_gap', 'shared_score'] if score_plan.get('method') == 'shared'
                                       else ['point', 'frozen_score', 'adapted_score'])}, flush=True)
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output/'evaluation.json', {'groups': results, 'score_seal_sha256': data.file_hash(scores/'seal.json'),
        'evaluation_source': completed_identity(identity), 'scorer_source': score_plan['source'],
        'observation_source': plan['source'],
        'label_scope': 'released annotation granularity/provenance, no new judge or label generation'})
